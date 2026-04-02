//! # Unmanaged Drive Component
//!
//! Raw block device with sector-based and byte-level I/O.
//! Mirrors `li.cil.oc.server.component.Drive` from OC Scala source.
//!
//! Unlike managed filesystems, unmanaged drives expose raw sectors
//! and do not provide file/directory abstraction. The guest OS
//! (or a user program) is responsible for implementing a filesystem
//! on top of the raw blocks.

use crate::components::Address;
use std::io::{Read, Write};

/// Sector size in bytes. Fixed at 512 matching OC.
pub const SECTOR_SIZE: usize = 512;

/// Maximum number of speed tiers (0..5).
pub const MAX_SPEED_TIERS: usize = 6;

/// Per-speed-tier call budget costs, from `Drive.scala`.
#[derive(Debug, Clone, Copy)]
pub struct DriveCosts {
    pub read_sector:  [f64; MAX_SPEED_TIERS],
    pub write_sector: [f64; MAX_SPEED_TIERS],
    pub read_byte:    [f64; MAX_SPEED_TIERS],
    pub write_byte:   [f64; MAX_SPEED_TIERS],
}

impl Default for DriveCosts {
    fn default() -> Self {
        Self {
            read_sector:  [1.0/10.0, 1.0/20.0, 1.0/30.0, 1.0/40.0, 1.0/50.0, 1.0/60.0],
            write_sector: [1.0/5.0,  1.0/10.0, 1.0/15.0, 1.0/20.0, 1.0/25.0, 1.0/30.0],
            read_byte:    [1.0/48.0, 1.0/64.0, 1.0/80.0, 1.0/96.0, 1.0/112.0,1.0/128.0],
            write_byte:   [1.0/24.0, 1.0/32.0, 1.0/40.0, 1.0/48.0, 1.0/56.0, 1.0/64.0],
        }
    }
}

/// Unmanaged drive (raw block device).
pub struct Drive {
    pub address: Address,

    /// Total capacity in bytes.
    capacity: usize,

    /// Number of platters (affects seek simulation).
    platter_count: usize,

    /// Raw data backing the drive.
    data: Vec<u8>,

    /// Number of sectors (`capacity / SECTOR_SIZE`).
    sector_count: usize,

    /// Sectors per platter (`sector_count / platter_count`).
    sectors_per_platter: usize,

    /// Current head position (in platter-local sector units).
    head_pos: usize,

    /// Optional human-readable label.
    label: Option<String>,

    /// Whether the drive is locked (read-only).
    locked: bool,

    /// Speed tier (0..5). Higher = faster, cheaper I/O.
    speed: usize,

    /// Per-tier I/O costs.
    costs: DriveCosts,

    /// Seek threshold in sectors. Seek penalty if delta > this.
    seek_threshold: usize,

    /// Seek penalty in seconds.
    seek_time: f64,

    /// Path to the persistence file (if any).
    save_path: Option<String>,

    /// Whether data has been modified since last save.
    dirty: bool,
}

impl Drive {
    /// Create a new drive with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Total size in bytes (must be a multiple of 512).
    /// * `platter_count` - Number of platters (1..8 typical).
    /// * `speed` - Speed tier (0..5). Clamped to valid range.
    /// * `locked` - Whether the drive is read-only.
    pub fn new(capacity: usize, platter_count: usize, speed: usize, locked: bool) -> Self {
        let capacity = (capacity / SECTOR_SIZE) * SECTOR_SIZE; // align
        let platter_count = platter_count.max(1);
        let sector_count = capacity / SECTOR_SIZE;
        let sectors_per_platter = if platter_count > 0 {
            sector_count / platter_count
        } else {
            sector_count
        }.max(1);

        Self {
            address: crate::components::new_address(),
            capacity,
            platter_count,
            data: vec![0u8; capacity],
            sector_count,
            sectors_per_platter,
            head_pos: 0,
            label: None,
            locked,
            speed: speed.min(MAX_SPEED_TIERS - 1),
            costs: DriveCosts::default(),
            seek_threshold: 16,
            seek_time: 0.05,  // 50ms seek penalty
            save_path: None,
            dirty: false,
        }
    }

    /// Create a standard HDD at a given tier.
    ///
    /// Tier 0 (T1): 1 MiB,  1 platter, speed 0
    /// Tier 1 (T2): 2 MiB,  2 platters, speed 1
    /// Tier 2 (T3): 4 MiB,  4 platters, speed 2
    pub fn new_hdd(tier: usize) -> Self {
        match tier.min(2) {
            0 => Self::new(1 * 1024 * 1024, 1, 0, false),
            1 => Self::new(2 * 1024 * 1024, 2, 1, false),
            _ => Self::new(4 * 1024 * 1024, 4, 2, false),
        }
    }

    pub const fn component_name() -> &'static str { "drive" }

    // -- Getters --------------------------------------------------------

    pub fn capacity(&self) -> usize { self.capacity }
    pub fn sector_size(&self) -> usize { SECTOR_SIZE }
    pub fn sector_count(&self) -> usize { self.sector_count }
    pub fn platter_count(&self) -> usize { self.platter_count }
    pub fn speed(&self) -> usize { self.speed }
    pub fn is_locked(&self) -> bool { self.locked }

    pub fn get_label(&self) -> Option<&str> { self.label.as_deref() }
    pub fn set_label(&mut self, label: Option<&str>) -> Result<(), &'static str> {
        if self.locked { return Err("drive is read only"); }
        self.label = label.map(|s| s.chars().take(16).collect());
        Ok(())
    }

    /// Cost to read a sector at the current speed tier.
    pub fn read_sector_cost(&self) -> f64 { self.costs.read_sector[self.speed] }
    /// Cost to write a sector.
    pub fn write_sector_cost(&self) -> f64 { self.costs.write_sector[self.speed] }
    /// Cost to read a byte.
    pub fn read_byte_cost(&self) -> f64 { self.costs.read_byte[self.speed] }
    /// Cost to write a byte.
    pub fn write_byte_cost(&self) -> f64 { self.costs.write_byte[self.speed] }

    // -- Sector I/O -----------------------------------------------------

    /// Read a full sector (512 bytes). `sector` is 0-based.
    ///
    /// Returns the sector data or an error string.
    /// Also returns `true` if a seek penalty should be applied.
    pub fn read_sector(&mut self, sector: usize) -> Result<(Vec<u8>, bool), &'static str> {
        self.validate_sector(sector)?;
        let seek = self.move_head(sector);
        let offset = sector * SECTOR_SIZE;
        let mut buf = vec![0u8; SECTOR_SIZE];
        buf.copy_from_slice(&self.data[offset..offset + SECTOR_SIZE]);
        Ok((buf, seek))
    }

    /// Write data to a sector. `sector` is 0-based.
    /// Data is truncated or zero-padded to `SECTOR_SIZE`.
    pub fn write_sector(&mut self, sector: usize, sector_data: &[u8]) -> Result<bool, &'static str> {
        if self.locked { return Err("drive is read only"); }
        self.validate_sector(sector)?;
        let seek = self.move_head(sector);
        let offset = sector * SECTOR_SIZE;
        let copy_len = sector_data.len().min(SECTOR_SIZE);
        self.data[offset..offset + copy_len].copy_from_slice(&sector_data[..copy_len]);
        // Zero-pad remainder if data is shorter than sector
        if copy_len < SECTOR_SIZE {
            self.data[offset + copy_len..offset + SECTOR_SIZE].fill(0);
        }
        self.dirty = true;
        Ok(seek)
    }

    // -- Byte I/O -------------------------------------------------------

    /// Read a single byte at the given offset (0-based internally).
    ///
    /// In Lua, offsets are 1-based; the dispatch layer subtracts 1.
    pub fn read_byte(&mut self, offset: usize) -> Result<(u8, bool), &'static str> {
        if offset >= self.capacity {
            return Err("invalid offset, not in a usable sector");
        }
        let sector = offset / SECTOR_SIZE;
        self.validate_sector(sector)?;
        let seek = self.move_head(sector);
        Ok((self.data[offset], seek))
    }

    /// Write a single byte at the given offset (0-based internally).
    pub fn write_byte(&mut self, offset: usize, value: u8) -> Result<bool, &'static str> {
        if self.locked { return Err("drive is read only"); }
        if offset >= self.capacity {
            return Err("invalid offset, not in a usable sector");
        }
        let sector = offset / SECTOR_SIZE;
        self.validate_sector(sector)?;
        let seek = self.move_head(sector);
        self.data[offset] = value;
        self.dirty = true;
        Ok(seek)
    }

    // -- Head movement / seek simulation --------------------------------

    fn validate_sector(&self, sector: usize) -> Result<(), &'static str> {
        if sector >= self.sector_count {
            Err("invalid offset, not in a usable sector")
        } else {
            Ok(())
        }
    }

    /// Move the head to the given sector. Returns `true` if a seek
    /// penalty should be applied (head moved beyond threshold).
    fn move_head(&mut self, sector: usize) -> bool {
        let new_pos = sector % self.sectors_per_platter;
        if self.head_pos == new_pos {
            return false;
        }
        let delta = (self.head_pos as isize - new_pos as isize).unsigned_abs();
        self.head_pos = new_pos;
        delta > self.seek_threshold
    }

    // -- Persistence ----------------------------------------------------

    /// Set the save path for disk persistence.
    pub fn set_save_path(&mut self, path: String) {
        self.save_path = Some(path);
    }

    /// Save drive contents to disk (gzip compressed).
    pub fn save_to_disk(&mut self) -> std::io::Result<()> {
        let _prof = crate::profiler::scope(crate::profiler::Cat::RealDisk, "save");
        let path = match &self.save_path {
            Some(p) => p.clone(),
            None => return Ok(()),
        };
        if !self.dirty { return Ok(()); }

        if let Some(parent) = std::path::Path::new(&path).parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = std::fs::File::create(&path)?;
        let mut encoder = flate2_or_raw_write(file);
        encoder.write_all(&self.data)?;
        self.dirty = false;
        Ok(())
    }

    /// Load drive contents from disk.
    pub fn load_from_disk(&mut self) -> std::io::Result<()> {
        let _prof = crate::profiler::scope(crate::profiler::Cat::RealDisk, "load");
        let path = match &self.save_path {
            Some(p) => p.clone(),
            None => return Ok(()),
        };
        if !std::path::Path::new(&path).exists() {
            return Ok(());
        }

        let compressed = std::fs::read(&path)?;
        // Try gzip decompression first, fall back to raw
        match decompress_gzip(&compressed) {
            Ok(raw) => {
                let copy_len = raw.len().min(self.data.len());
                self.data[..copy_len].copy_from_slice(&raw[..copy_len]);
            }
            Err(_) => {
                let copy_len = compressed.len().min(self.data.len());
                self.data[..copy_len].copy_from_slice(&compressed[..copy_len]);
            }
        }
        self.dirty = false;
        Ok(())
    }

    /// Load raw data into the drive (for pre-populating disk images).
    pub fn load_image(&mut self, data: &[u8]) {
        let copy_len = data.len().min(self.data.len());
        self.data[..copy_len].copy_from_slice(&data[..copy_len]);
        self.dirty = true;
    }

    /// Get a reference to the raw data (for debugging/inspection).
    pub fn raw_data(&self) -> &[u8] { &self.data }
}

// -- Compression helpers ------------------------------------------------

/// Minimal gzip decompression without the flate2 crate.
/// Uses a simple raw deflate implementation or passes through.
fn decompress_gzip(data: &[u8]) -> std::io::Result<Vec<u8>> {
    use std::io::Read;
    // Check gzip magic bytes
    if data.len() < 2 || data[0] != 0x1f || data[1] != 0x8b {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "not gzip"));
    }
    let mut decoder = flate2::read::GzDecoder::new(std::io::Cursor::new(data));
    let mut out = Vec::new();
    decoder.read_to_end(&mut out)?;
    Ok(out)
}

fn flate2_or_raw_write(file: std::fs::File) -> flate2::write::GzEncoder<std::fs::File> {
    flate2::write::GzEncoder::new(file, flate2::Compression::default())
}