//! # EEPROM Component
//!
//! Stores the BIOS code and a small volatile data section.
//! Mirrors `EEPROM.scala`.

use crate::config::OcConfig;
use crate::components::Address;

/// EEPROM (BIOS) component.
pub struct Eeprom {
    pub address: Address,
    /// The stored code (Lua source or bytecode).
    code: Vec<u8>,
    /// Volatile data section (persisted with the EEPROM item, not the world).
    data: Vec<u8>,
    /// Human-readable label (max 24 chars).
    label: String,
    /// Once set to `true`, code cannot be overwritten.
    read_only: bool,
    /// Limits from config.
    max_code_size: usize,
    max_data_size: usize,
}

impl Eeprom {
    pub fn new(config: &OcConfig) -> Self {
        Self {
            address: crate::components::new_address(),
            code: Vec::new(),
            data: Vec::new(),
            label: "EEPROM".into(),
            read_only: false,
            max_code_size: config.eeprom_size,
            max_data_size: config.eeprom_data_size,
        }
    }

    pub const fn component_name() -> &'static str { "eeprom" }

    /// `eeprom.get()` / read the code section.
    pub fn get_code(&self) -> &[u8] { &self.code }

    /// `eeprom.set(data)` / overwrite the code section.
    pub fn set_code(&mut self, code: Vec<u8>) -> Result<(), &'static str> {
        if self.read_only { return Err("storage is readonly"); }
        if code.len() > self.max_code_size { return Err("not enough space"); }
        self.code = code;
        Ok(())
    }

    /// `eeprom.getData()`.
    pub fn get_data(&self) -> &[u8] { &self.data }

    /// `eeprom.setData(data)`.
    pub fn set_data(&mut self, data: Vec<u8>) -> Result<(), &'static str> {
        if data.len() > self.max_data_size { return Err("not enough space"); }
        self.data = data;
        Ok(())
    }

    /// `eeprom.getLabel()`.
    pub fn get_label(&self) -> &str { &self.label }

    /// `eeprom.setLabel(label)`.
    pub fn set_label(&mut self, label: &str) -> Result<(), &'static str> {
        if self.read_only { return Err("storage is readonly"); }
        let trimmed: String = label.chars().take(24).collect();
        self.label = if trimmed.is_empty() { "EEPROM".into() } else { trimmed };
        Ok(())
    }

    /// `eeprom.makeReadonly(checksum)`.
    pub fn make_readonly(&mut self, checksum: &str) -> Result<(), &'static str> {
        if checksum == self.checksum() {
            self.read_only = true;
            Ok(())
        } else {
            Err("incorrect checksum")
        }
    }

    /// CRC32 checksum of the code section (hex string).
    pub fn checksum(&self) -> String {
        let crc = crc32_simple(&self.code);
        format!("{:08x}", crc)
    }

    /// `eeprom.getSize()`.
    pub fn max_code_size(&self) -> usize { self.max_code_size }

    /// `eeprom.getDataSize()`.
    pub fn max_data_size(&self) -> usize { self.max_data_size }

    /// Load BIOS from a Lua source string.
    pub fn flash(&mut self, source: &str) {
        self.code = source.as_bytes().to_vec();
    }
}

/// Minimal CRC32 (IEEE) / no external crate needed.
fn crc32_simple(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}