//! # User settings persistence
//!
//! Saves and loads the user-configurable subset of [`OcConfig`] and
//! GUI state to `save/settings.json`. Only the settings exposed in
//! the F8 overlay are persisted — hardware-level constants (screen
//! resolutions, GPU cost tables, etc.) always come from the defaults.
//!
//! ## When saved
//!
//! * On F8 "Apply" (closing the settings GUI with changes).
//! * On application exit (via window close).
//!
//! ## When loaded
//!
//! * During `App::new()`, before components are created.
//!
//! ## File format
//!
//! Plain JSON via `serde_json`. All fields have `#[serde(default)]`
//! so that adding new settings in future versions does not break
//! loading old config files.

use serde::{Serialize, Deserialize};

/// Path to the settings file on the host filesystem.
const SETTINGS_PATH: &str = "save/settings.json";

/// Serializable subset of user-configurable settings.
///
/// This struct captures everything the user can change through the
/// F8 overlay. It does NOT include hardware constants, component
/// addresses, or runtime state.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SavedSettings {
    // -- Machine tab --

    /// Maximum execution time (seconds) before a forced yield.
    pub timeout: f64,

    /// Maximum number of signals in the queue.
    pub max_signal_queue_size: usize,

    /// Whether to ignore power requirements (creative mode).
    pub ignore_power: bool,

    /// Whether Lua bytecode loading is permitted.
    pub allow_bytecode: bool,

    /// Whether `__gc` metamethods are permitted in the sandbox.
    pub allow_gc: bool,

    /// Whether persistence (Eris) is allowed.
    pub allow_persistence: bool,

    // -- Display tab --

    /// Active rendering backend name.
    ///
    /// One of `"Software"`, `"VulkanIndirect"`, `"VulkanDirect"`.
    pub render_mode: String,

    /// Whether vertical sync is enabled.
    pub vsync: bool,

    /// FPS cap when vsync is off. `None` means unlimited.
    pub fps_limit: Option<u32>,

    /// Master volume multiplier (0.0–1.0).
    pub master_volume: f32,

    /// Effect/ambient volume (0.0–1.0).
    pub effect_volume: f32,

    /// Beep/UI sound volume (0.0–1.0).
    pub beep_volume: f32,

    // -- Storage tab --

    /// EEPROM code section size in bytes.
    pub eeprom_size: usize,

    /// EEPROM data section size in bytes.
    pub eeprom_data_size: usize,

    /// Tmpfs size in KiB. 0 disables tmpfs.
    pub tmp_size_kib: usize,

    /// Maximum open file handles per filesystem.
    pub max_handles: usize,

    /// Maximum bytes per `read()` call.
    pub max_read_buffer: usize,

    /// Whether tmpfs is erased on reboot.
    pub erase_tmp_on_reboot: bool,

    // -- Fun tab --

    /// Whether Emacs-style keybindings are enabled in the GUI.
    pub pinky_mode: bool,
}

impl Default for SavedSettings {
    /// Defaults matching `OcConfig::default()` and the overlay defaults.
    fn default() -> Self {
        Self {
            timeout: 5.0,
            max_signal_queue_size: 256,
            ignore_power: false,
            allow_bytecode: false,
            allow_gc: false,
            allow_persistence: true,
            render_mode: "Software".into(),
            vsync: false,
            fps_limit: None,
            master_volume: 1.0,
            effect_volume: 0.4,
            beep_volume: 0.3,
            eeprom_size: 4096,
            eeprom_data_size: 256,
            tmp_size_kib: 64,
            max_handles: 12,
            max_read_buffer: 2048,
            erase_tmp_on_reboot: true,
            pinky_mode: false,
        }
    }
}

impl SavedSettings {
    /// Apply saved settings to the emulator configuration.
    ///
    /// Copies all user-configurable values from this struct into
    /// the corresponding fields of `OcConfig`.
    pub fn apply_to_config(&self, cfg: &mut crate::config::OcConfig) {
        cfg.timeout = self.timeout;
        cfg.max_signal_queue_size = self.max_signal_queue_size;
        cfg.ignore_power = self.ignore_power;
        cfg.allow_bytecode = self.allow_bytecode;
        cfg.allow_gc = self.allow_gc;
        cfg.allow_persistence = self.allow_persistence;
        cfg.master_volume = self.master_volume;
        cfg.effect_volume = self.effect_volume;
        cfg.beep_volume = self.beep_volume;
        cfg.eeprom_size = self.eeprom_size;
        cfg.eeprom_data_size = self.eeprom_data_size;
        cfg.tmp_size_kib = self.tmp_size_kib;
        cfg.max_handles = self.max_handles;
        cfg.max_read_buffer = self.max_read_buffer;
        cfg.erase_tmp_on_reboot = self.erase_tmp_on_reboot;
    }

    /// Capture current settings from the config and runtime state.
    pub fn from_current(
        cfg: &crate::config::OcConfig,
        mode: crate::render::RenderMode,
        vsync: bool,
        fps_limit: Option<u32>,
        pinky: bool,
    ) -> Self {
        Self {
            timeout: cfg.timeout,
            max_signal_queue_size: cfg.max_signal_queue_size,
            ignore_power: cfg.ignore_power,
            allow_bytecode: cfg.allow_bytecode,
            allow_gc: cfg.allow_gc,
            allow_persistence: cfg.allow_persistence,
            render_mode: match mode {
                crate::render::RenderMode::Software => "Software",
                crate::render::RenderMode::VulkanIndirect => "VulkanIndirect",
                crate::render::RenderMode::VulkanDirect => "VulkanDirect",
            }.into(),
            vsync,
            fps_limit,
            master_volume: cfg.master_volume,
            effect_volume: cfg.effect_volume,
            beep_volume: cfg.beep_volume,
            eeprom_size: cfg.eeprom_size,
            eeprom_data_size: cfg.eeprom_data_size,
            tmp_size_kib: cfg.tmp_size_kib,
            max_handles: cfg.max_handles,
            max_read_buffer: cfg.max_read_buffer,
            erase_tmp_on_reboot: cfg.erase_tmp_on_reboot,
            pinky_mode: pinky,
        }
    }

    /// Get the render mode as enum.
    pub fn render_mode(&self) -> crate::render::RenderMode {
        match self.render_mode.as_str() {
            "VulkanIndirect" => crate::render::RenderMode::VulkanIndirect,
            "VulkanDirect" => crate::render::RenderMode::VulkanDirect,
            _ => crate::render::RenderMode::Software,
        }
    }
}

/// Load settings from `save/settings.json`.
///
/// Returns `Default` if the file does not exist or cannot be parsed.
/// Never fails — always returns a valid `SavedSettings`.
pub fn load_settings() -> SavedSettings {
    let path = std::path::Path::new(SETTINGS_PATH);
    if !path.exists() {
        eprintln!("[settings] No saved settings, using defaults");
        return SavedSettings::default();
    }
    match std::fs::read_to_string(path) {
        Ok(json) => match serde_json::from_str::<SavedSettings>(&json) {
            Ok(s) => {
                eprintln!("[settings] Loaded from {}", SETTINGS_PATH);
                s
            }
            Err(e) => {
                eprintln!("[settings] Parse error: {e}, using defaults");
                SavedSettings::default()
            }
        }
        Err(e) => {
            eprintln!("[settings] Read error: {e}, using defaults");
            SavedSettings::default()
        }
    }
}

/// Save settings to `save/settings.json`.
///
/// Creates the `save/` directory if needed.
pub fn save_settings(settings: &SavedSettings) {
    let path = std::path::Path::new(SETTINGS_PATH);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match serde_json::to_string_pretty(settings) {
        Ok(json) => match std::fs::write(path, json) {
            Ok(()) => eprintln!("[settings] Saved to {}", SETTINGS_PATH),
            Err(e) => eprintln!("[settings] Write error: {e}"),
        }
        Err(e) => eprintln!("[settings] Serialize error: {e}"),
    }
}