//! # EEPROM Component
//!
//! This module implements the EEPROM (Electrically Erasable Programmable
//! Read-Only Memory) component as it exists in OpenComputers. The EEPROM
//! is the first thing a computer reads on boot: it holds the BIOS code
//! (typically a Lua script that bootstraps the operating system) and a
//! small volatile data section used for persisting tiny values like
//! the boot filesystem address.
//!
//! ## Relationship to OpenComputers
//!
//! This module mirrors `li.cil.oc.server.component.EEPROM` from the
//! OpenComputers Scala source. In the original mod:
//!
//! * The EEPROM is an item (crafted from redstone, gold, paper, etc.).
//! * When placed inside a computer case, it provides the `eeprom`
//!   component, exposing methods like `get()`, `set()`, `getData()`,
//!   `setData()`, `getLabel()`, `setLabel()`, `getSize()`,
//!   `getDataSize()`, `getChecksum()`, and `makeReadonly()`.
//! * The code section stores Lua source or precompiled bytecode (up to
//!   `eepromSize` bytes, default 4096).
//! * The data section stores arbitrary bytes (up to `eepromDataSize`
//!   bytes, default 256).
//! * The label is a human-readable string, truncated to 24 characters.
//! * Once `makeReadonly()` is called with the correct CRC32 checksum
//!   of the current code, the EEPROM becomes permanently read-only.
//!
//! ## Data flow
//!
//! ```text
//! Lua guest
//!   |
//!   | component.invoke(eeprom_addr, "get")
//!   v
//! host.rs dispatch_invoke()
//!   |
//!   | calls Eeprom::get_code()
//!   v
//! Returns &[u8] pushed onto Lua stack as a string
//! ```
//!
//! ## Checksum algorithm
//!
//! The checksum is computed using the IEEE CRC32 polynomial
//! (`0xEDB88320` reflected). This is the same algorithm used by zlib,
//! gzip, PNG, and many other formats. The result is formatted as an
//! 8-character lowercase hexadecimal string (e.g. `"a1b2c3d4"`).
//!
//! A minimal implementation is included in this module to avoid pulling
//! in an external crate just for one hash.
//!
//! ## Thread safety
//!
//! `Eeprom` is not `Sync`. It is owned exclusively by the
//! [`EmulatorState`](crate::lua::host::EmulatorState) and accessed
//! only from the main emulation thread.

use crate::config::OcConfig;
use crate::components::Address;

/// EEPROM (BIOS) component.
///
/// Represents a single EEPROM chip installed in the emulated computer.
/// Holds two independent byte buffers (code and data), a label, and
/// a read-only flag.
///
/// # Capacity limits
///
/// Both the code and data sections have maximum sizes configured via
/// [`OcConfig`]. Attempts to write data exceeding these limits will
/// return an error string matching what OpenComputers would produce
/// in-game (`"not enough space"`).
///
/// # Read-only mode
///
/// Once [`make_readonly`](Eeprom::make_readonly) succeeds, the code
/// section and label become immutable for the lifetime of this struct.
/// The data section is *not* affected by read-only mode (matching OC
/// behaviour where volatile data can always be written).
///
/// # Examples (conceptual)
///
/// ```text
/// let mut eeprom = Eeprom::new(&config);
/// eeprom.flash("print('hello')");
/// assert_eq!(eeprom.get_code(), b"print('hello')");
/// assert_eq!(eeprom.get_label(), "EEPROM");
/// ```
pub struct Eeprom {
    /// The unique UUID address of this component, generated at
    /// construction time via [`new_address`](crate::components::new_address).
    ///
    /// This address is used by Lua code to identify the component in
    /// calls like `component.invoke(address, method, ...)`.
    pub address: Address,

    /// The stored BIOS code.
    ///
    /// This can be either:
    /// * Raw Lua source text (UTF-8 encoded).
    /// * Precompiled Lua bytecode (if `allow_bytecode` is enabled in
    ///   the config). Bytecode starts with the `\x1bLua` magic bytes.
    ///
    /// Maximum length is governed by [`max_code_size`](Eeprom::max_code_size).
    /// Default: empty (no BIOS loaded).
    code: Vec<u8>,

    /// Volatile data section.
    ///
    /// In OpenComputers, this section is persisted with the EEPROM item
    /// (not with the world save). Typical use cases include storing the
    /// boot filesystem address so the BIOS knows which disk to load
    /// the OS from.
    ///
    /// Maximum length is governed by [`max_data_size`](Eeprom::max_data_size).
    /// Default: empty.
    data: Vec<u8>,

    /// Human-readable label displayed in the tooltip and returned by
    /// `eeprom.getLabel()`.
    ///
    /// Truncated to 24 characters on write. If set to an empty string,
    /// it reverts to the default `"EEPROM"`.
    label: String,

    /// When `true`, the code section and label are immutable.
    ///
    /// Set via [`make_readonly`](Eeprom::make_readonly). Cannot be
    /// reversed (matching OC behaviour where making an EEPROM read-only
    /// is a one-way operation meant to protect crafted BIOSes).
    read_only: bool,

    /// Maximum allowed byte length of the code section.
    ///
    /// Sourced from [`OcConfig::eeprom_size`]. Default: 4096 bytes.
    max_code_size: usize,

    /// Maximum allowed byte length of the data section.
    ///
    /// Sourced from [`OcConfig::eeprom_data_size`]. Default: 256 bytes.
    max_data_size: usize,
}

impl Eeprom {
    /// Create a new, empty EEPROM with limits drawn from the given config.
    ///
    /// # What happens
    ///
    /// * A fresh UUID address is generated.
    /// * Code and data buffers start empty.
    /// * Label defaults to `"EEPROM"`.
    /// * Read-only flag is `false`.
    /// * Size limits are copied from `config.eeprom_size` and
    ///   `config.eeprom_data_size`.
    ///
    /// # Arguments
    ///
    /// * `config` - The global emulator configuration. Only the
    ///   `eeprom_size` and `eeprom_data_size` fields are read.
    ///
    /// # Returns
    ///
    /// A freshly initialised `Eeprom` ready to have code flashed into it.
    pub fn new(config: &OcConfig) -> Self {
        let e = Self {
            address: crate::components::new_address(),
            code: Vec::new(),
            data: Vec::new(),
            label: "EEPROM".into(),
            read_only: false,
            max_code_size: config.eeprom_size,
            max_data_size: config.eeprom_data_size,
        };
        log::info!("EEPROM created: addr={}, code_max={}, data_max={}",
            e.address, e.max_code_size, e.max_data_size);
        e
    }

    /// Returns the OC component type name for this component.
    ///
    /// This is the string `"eeprom"`, which appears in:
    /// * `component.list()` results
    /// * `component.type(address)` return values
    /// * Signal payloads for `component_added` / `component_removed`
    ///
    /// Defined as a `const fn` so it can be evaluated at compile time
    /// and used in match arms without allocation.
    pub const fn component_name() -> &'static str { "eeprom" }

    /// Read the code section (corresponds to `eeprom.get()` in Lua).
    ///
    /// Returns a byte slice referencing the internal code buffer.
    /// The slice may be empty if no code has been flashed.
    ///
    /// # Lua equivalent
    ///
    /// ```lua
    /// local code = component.invoke(eeprom_addr, "get")
    /// -- `code` is a string containing the BIOS source/bytecode
    /// ```
    ///
    /// # Performance
    ///
    /// This is a zero-copy operation; it returns a reference into the
    /// existing allocation.
    pub fn get_code(&self) -> &[u8] { &self.code }

    /// Overwrite the code section (corresponds to `eeprom.set(data)` in Lua).
    ///
    /// # Arguments
    ///
    /// * `code` - The new code bytes. Ownership is transferred into the
    ///   EEPROM. Can be Lua source or bytecode.
    ///
    /// # Errors
    ///
    /// Returns `Err("storage is readonly")` if the EEPROM has been made
    /// read-only via [`make_readonly`](Eeprom::make_readonly).
    ///
    /// Returns `Err("not enough space")` if `code.len()` exceeds
    /// [`max_code_size`](Eeprom::max_code_size).
    ///
    /// # Lua equivalent
    ///
    /// ```lua
    /// local ok, err = component.invoke(eeprom_addr, "set", new_code)
    /// ```
    ///
    /// # Side effects
    ///
    /// In OpenComputers, writing to the EEPROM deducts energy from the
    /// network. This emulator currently does not enforce energy costs
    /// for EEPROM writes (configurable via `eeprom_write_cost` in the
    /// config, but not yet wired up).
    pub fn set_code(&mut self, code: Vec<u8>) -> Result<(), &'static str> {
        if self.read_only {
            log::debug!("EEPROM set_code rejected: read-only");
            return Err("storage is readonly");
        }
        if code.len() > self.max_code_size {
            log::warn!("EEPROM set_code: {} bytes exceeds max {}",
                code.len(), self.max_code_size);
            return Err("not enough space");
        }
        log::debug!("EEPROM code written: {} bytes", code.len());
        self.code = code;
        Ok(())
    }

    /// Read the volatile data section (corresponds to `eeprom.getData()` in Lua).
    ///
    /// Returns a byte slice referencing the internal data buffer.
    /// The slice may be empty if no data has been written.
    ///
    /// # Lua equivalent
    ///
    /// ```lua
    /// local data = component.invoke(eeprom_addr, "getData")
    /// ```
    pub fn get_data(&self) -> &[u8] { &self.data }

    /// Overwrite the volatile data section (corresponds to `eeprom.setData(data)` in Lua).
    ///
    /// Note: the data section is *not* protected by read-only mode.
    /// Even if `make_readonly` has been called, `set_data` will still
    /// succeed (matching OC behaviour).
    ///
    /// # Arguments
    ///
    /// * `data` - The new data bytes. Ownership is transferred.
    ///
    /// # Errors
    ///
    /// Returns `Err("not enough space")` if `data.len()` exceeds
    /// [`max_data_size`](Eeprom::max_data_size).
    ///
    /// # Lua equivalent
    ///
    /// ```lua
    /// component.invoke(eeprom_addr, "setData", some_bytes)
    /// ```
    pub fn set_data(&mut self, data: Vec<u8>) -> Result<(), &'static str> {
        if data.len() > self.max_data_size { return Err("not enough space"); }
        self.data = data;
        Ok(())
    }

    /// Read the label (corresponds to `eeprom.getLabel()` in Lua).
    ///
    /// # Returns
    ///
    /// A string slice containing the current label. Never empty (defaults
    /// to `"EEPROM"`).
    pub fn get_label(&self) -> &str { &self.label }

    /// Set the label (corresponds to `eeprom.setLabel(label)` in Lua).
    ///
    /// The label is truncated to 24 characters. If the resulting string
    /// is empty, it reverts to the default `"EEPROM"`.
    ///
    /// # Arguments
    ///
    /// * `label` - The desired label string. May contain any Unicode
    ///   characters; only the first 24 chars are kept.
    ///
    /// # Errors
    ///
    /// Returns `Err("storage is readonly")` if the EEPROM is read-only.
    ///
    /// # Lua equivalent
    ///
    /// ```lua
    /// component.invoke(eeprom_addr, "setLabel", "My Custom BIOS")
    /// ```
    pub fn set_label(&mut self, label: &str) -> Result<(), &'static str> {
        if self.read_only { return Err("storage is readonly"); }
        let trimmed: String = label.chars().take(24).collect();
        self.label = if trimmed.is_empty() { "EEPROM".into() } else { trimmed };
        Ok(())
    }

    /// Lock the EEPROM into read-only mode (corresponds to
    /// `eeprom.makeReadonly(checksum)` in Lua).
    ///
    /// This is a one-way operation. Once successful, the code section
    /// and label can never be modified again (for this instance).
    ///
    /// # Arguments
    ///
    /// * `checksum` - Must exactly match the hex string returned by
    ///   [`checksum()`](Eeprom::checksum). This prevents accidental
    ///   locking (the user must prove they know the current contents).
    ///
    /// # Errors
    ///
    /// Returns `Err("incorrect checksum")` if the provided checksum
    /// does not match.
    ///
    /// # Lua equivalent
    ///
    /// ```lua
    /// local cs = component.invoke(eeprom_addr, "getChecksum")
    /// component.invoke(eeprom_addr, "makeReadonly", cs)
    /// ```
    ///
    /// # Design note
    ///
    /// In OpenComputers, making a BIOS read-only is typically done for
    /// distribution purposes (e.g. the default Lua BIOS shipped with OC
    /// is read-only). The checksum requirement exists so users don't
    /// accidentally lock their custom BIOSes.
    pub fn make_readonly(&mut self, checksum: &str) -> Result<(), &'static str> {
        if checksum == self.checksum() {
            self.read_only = true;
            log::info!("EEPROM locked read-only (checksum={checksum})");
            Ok(())
        } else {
            log::warn!("EEPROM make_readonly: incorrect checksum");
            Err("incorrect checksum")
        }
    }

    /// Compute the CRC32 checksum of the code section.
    ///
    /// Returns an 8-character lowercase hexadecimal string representing
    /// the IEEE CRC32 of the code bytes.
    ///
    /// # Examples (conceptual)
    ///
    /// ```text
    /// eeprom.flash("print('hi')");
    /// let cs = eeprom.checksum(); // e.g. "3d4e5f6a"
    /// ```
    ///
    /// # Lua equivalent
    ///
    /// ```lua
    /// local cs = component.invoke(eeprom_addr, "getChecksum")
    /// -- cs is a string like "3d4e5f6a"
    /// ```
    pub fn checksum(&self) -> String {
        let crc = crc32_simple(&self.code);
        format!("{:08x}", crc)
    }

    /// Maximum allowed code size in bytes (corresponds to `eeprom.getSize()` in Lua).
    ///
    /// This value is set at construction time from the config and does
    /// not change.
    ///
    /// # Lua equivalent
    ///
    /// ```lua
    /// local max = component.invoke(eeprom_addr, "getSize")
    /// -- max is typically 4096
    /// ```
    pub fn max_code_size(&self) -> usize { self.max_code_size }

    /// Maximum allowed data size in bytes (corresponds to `eeprom.getDataSize()` in Lua).
    ///
    /// # Lua equivalent
    ///
    /// ```lua
    /// local max = component.invoke(eeprom_addr, "getDataSize")
    /// -- max is typically 256
    /// ```
    pub fn max_data_size(&self) -> usize { self.max_data_size }

    /// Load BIOS code from a Lua source string.
    ///
    /// This is a convenience method used during emulator initialisation
    /// to flash the BIOS without going through the normal `set_code`
    /// path (which checks read-only status and size limits).
    ///
    /// # Arguments
    ///
    /// * `source` - A Lua source string. Converted to bytes via
    ///   `as_bytes()` (UTF-8 encoding preserved).
    ///
    /// # Panics
    ///
    /// Does not panic. If the source exceeds the maximum code size,
    /// it is stored anyway (this method bypasses size checks). This is
    /// intentional: the host should only call this with known-good data.
    ///
    /// # When to use
    ///
    /// Call this during setup (e.g. `App::new()`) to load the BIOS from
    /// an embedded or file-system source. Do *not* use this to implement
    /// the Lua-facing `eeprom.set()` method (use [`set_code`] instead).
    pub fn flash(&mut self, source: &str) {
        self.code = source.as_bytes().to_vec();
        log::info!("EEPROM flashed: {} bytes, checksum={}", self.code.len(), self.checksum());
    }
}

/// Compute the IEEE CRC32 checksum of a byte slice.
///
/// This is a minimal, dependency-free implementation of the CRC32
/// algorithm using the reflected polynomial `0xEDB88320`.
///
/// # Algorithm
///
/// ```text
/// 1. Initialise CRC to 0xFFFFFFFF.
/// 2. For each byte in the input:
///    a. XOR the byte into the low 8 bits of CRC.
///    b. For each of the 8 bits:
///       - If the lowest bit is set, shift right and XOR with 0xEDB88320.
///       - Otherwise, just shift right.
/// 3. Invert all bits of the final CRC.
/// ```
///
/// This produces results identical to `java.util.zip.CRC32` (which is
/// what OpenComputers uses on the JVM side) and to the `crc32` crate.
///
/// # Arguments
///
/// * `data` - The input bytes to hash.
///
/// # Returns
///
/// The 32-bit CRC value.
///
/// # Performance
///
/// This is a bit-at-a-time implementation (no lookup table). For the
/// small data sizes involved (max 4096 bytes for EEPROM code, 256 for
/// data), performance is not a concern. A table-driven implementation
/// would be ~8x faster but adds 1 KiB of static data.
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