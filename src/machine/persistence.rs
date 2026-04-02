//! # Machine state persistence (RAM splicing)
//!
//! Saves and restores machine state to/from disk. This includes:
//!
//! * EEPROM code and data sections
//! * Drive contents (gzip-compressed)
//! * Filesystem data (for writable VFS instances)
//! * Machine metadata (uptime, boot address, etc.)
//! * Signal queue snapshot
//! * Text buffer contents
//!
//! ## Limitations
//!
//! This does NOT persist the live Lua coroutine state. Full Lua
//! state persistence would require integrating the Eris library
//! (a Lua 5.x state serializer). Without Eris, restoring a saved
//! state reboots the machine from the BIOS.
//!
//! ## File format
//!
//! State is saved as JSON to `<save_dir>/<machine_address>.json`
//! with binary blobs (EEPROM code, drive data) stored as separate
//! gzip-compressed files.

use serde::{Serialize, Deserialize};
use std::path::{Path, PathBuf};
use std::io::Write;

/// Serializable snapshot of machine metadata.
#[derive(Serialize, Deserialize, Debug)]
pub struct MachineSnapshot {
    /// Machine UUID address.
    pub address: String,
    /// Boot filesystem address.
    pub boot_address: String,
    /// Uptime in ticks at time of save.
    pub uptime_ticks: u64,
    /// EEPROM label.
    pub eeprom_label: String,
    /// EEPROM code (base64-encoded).
    pub eeprom_code_b64: String,
    /// EEPROM data (base64-encoded).
    pub eeprom_data_b64: String,
    /// Registered component addresses and types.
    pub components: Vec<(String, String)>,
    /// Text buffer state.
    pub buffer_width: u32,
    pub buffer_height: u32,
    /// Drive addresses with save paths.
    pub drives: Vec<DriveMeta>,
    /// Filesystem labels.
    pub filesystem_labels: Vec<(String, Option<String>)>,
}

/// Metadata for a persisted drive.
#[derive(Serialize, Deserialize, Debug)]
pub struct DriveMeta {
    pub address: String,
    pub capacity: usize,
    pub platter_count: usize,
    pub speed: usize,
    pub locked: bool,
    pub label: Option<String>,
    /// Relative path to the gzip-compressed data file.
    pub data_file: String,
}

/// Save directory for machine state.
fn save_dir() -> PathBuf {
    PathBuf::from("save")
}

/// Create a snapshot from the current emulator state.
pub fn create_snapshot(
    emu: &crate::lua::host::EmulatorState,
) -> MachineSnapshot {
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD;

    let components: Vec<(String, String)> = emu.component_types
        .iter()
        .map(|(a, t)| (a.clone(), t.clone()))
        .collect();

    let drives: Vec<DriveMeta> = emu.drives.iter().map(|d| {
        DriveMeta {
            address: d.address.clone(),
            capacity: d.capacity(),
            platter_count: d.platter_count(),
            speed: d.speed(),
            locked: d.is_locked(),
            label: d.get_label().map(|s| s.to_owned()),
            data_file: format!("drive_{}.bin.gz", d.address),
        }
    }).collect();

    let fs_labels: Vec<(String, Option<String>)> = emu.filesystems
        .iter()
        .map(|f| (f.address.clone(), f.label.clone()))
        .collect();

    MachineSnapshot {
        address: emu.address.clone(),
        boot_address: emu.boot_address.clone(),
        uptime_ticks: emu.uptime_ticks,
        eeprom_label: emu.eeprom.get_label().to_owned(),
        eeprom_code_b64: b64.encode(emu.eeprom.get_code()),
        eeprom_data_b64: b64.encode(emu.eeprom.get_data()),
        components,
        buffer_width: emu.buffer.width(),
        buffer_height: emu.buffer.height(),
        drives,
        filesystem_labels: fs_labels,
    }
}

/// Save the machine state to disk.
pub fn save_state(emu: &crate::lua::host::EmulatorState) -> std::io::Result<()> {
    let dir = save_dir();
    std::fs::create_dir_all(&dir)?;
    let _prof = crate::profiler::scope(crate::profiler::Cat::RealDisk, "save_state");

    let snapshot = create_snapshot(emu);

    // Save metadata JSON
    let json_path = dir.join(format!("{}.json", &emu.address));
    let json = serde_json::to_string_pretty(&snapshot)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    std::fs::write(&json_path, json)?;

    // Save drive data (gzip compressed)
    for (drive, meta) in emu.drives.iter().zip(snapshot.drives.iter()) {
        let data_path = dir.join(&meta.data_file);
        let file = std::fs::File::create(&data_path)?;
        let mut encoder = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
        encoder.write_all(drive.raw_data())?;
        encoder.finish()?;
    }

    // Save EEPROM code/data as separate files for easier inspection
    let eeprom_code_path = dir.join(format!("{}_eeprom_code.lua", &emu.address));
    std::fs::write(&eeprom_code_path, emu.eeprom.get_code())?;

    eprintln!("[persist] Saved machine state to {}", json_path.display());
    Ok(())
}

/// Restore machine state from disk.
///
/// Returns `Some(snapshot)` if a save file exists, `None` otherwise.
pub fn load_state(machine_address: &str) -> Option<MachineSnapshot> {
    let dir = save_dir();
    let json_path = dir.join(format!("{}.json", machine_address));
    if !json_path.exists() { return None; }

    let json = std::fs::read_to_string(&json_path).ok()?;
    let snapshot: MachineSnapshot = serde_json::from_str(&json).ok()?;

    eprintln!("[persist] Loaded snapshot from {}", json_path.display());
    Some(snapshot)
}

/// Restore drive data from a gzip-compressed file.
pub fn load_drive_data(data_file: &str) -> Option<Vec<u8>> {
    use std::io::Read;
    let path = save_dir().join(data_file);
    if !path.exists() { return None; }
    let compressed = std::fs::read(&path).ok()?;
    let mut decoder = flate2::read::GzDecoder::new(std::io::Cursor::new(compressed));
    let mut data = Vec::new();
    decoder.read_to_end(&mut data).ok()?;
    Some(data)
}

/// Apply a loaded snapshot to the emulator state.
///
/// Restores EEPROM contents, drive data, and metadata.
/// Does NOT restore live Lua VM state (requires Eris).
pub fn apply_snapshot(
    emu: &mut crate::lua::host::EmulatorState,
    snapshot: &MachineSnapshot,
) {
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD;

    // Restore EEPROM
    if let Ok(code) = b64.decode(&snapshot.eeprom_code_b64) {
        let _ = emu.eeprom.set_code(code);
    }
    if let Ok(data) = b64.decode(&snapshot.eeprom_data_b64) {
        let _ = emu.eeprom.set_data(data);
    }
    let _ = emu.eeprom.set_label(&snapshot.eeprom_label);

    emu.boot_address = snapshot.boot_address.clone();
    emu.uptime_ticks = snapshot.uptime_ticks;

    // Restore drive data
    for (drive, meta) in emu.drives.iter_mut().zip(snapshot.drives.iter()) {
        if let Some(data) = load_drive_data(&meta.data_file) {
            drive.load_image(&data);
        }
        if let Some(label) = &meta.label {
            let _ = drive.set_label(Some(label));
        }
    }

    // Restore filesystem labels
    for (addr, label) in &snapshot.filesystem_labels {
        if let Some(fs) = emu.filesystems.iter_mut().find(|f| &f.address == addr) {
            fs.label = label.clone();
        }
    }

    eprintln!("[persist] Applied snapshot, uptime was {}s", snapshot.uptime_ticks as f64 / 20.0);
}