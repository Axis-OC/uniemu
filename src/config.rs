//! # OpenComputers Configuration
//!
//! All magic numbers extracted from `application.conf` and `Settings.scala`
//! in the OpenComputers source code. Every field is documented with its
//! original Scala source path where applicable.
//!
//! This module replaces the Typesafe/HOCON config system used by the
//! original OC mod with plain Rust structs. The default values match
//! vanilla OC 1.8.x out of the box.
//!
//! ## Loading
//!
//! Currently, the config is always constructed via `OcConfig::default()`.
//! A future enhancement could load from a TOML or JSON file, but the
//! defaults are sufficient for most use cases.
//!
//! ## Tier system
//!
//! OpenComputers uses a zero-based tier system internally:
//! * Tier 0 = T1 in-game (the cheapest)
//! * Tier 1 = T2
//! * Tier 2 = T3 (the most powerful)
//!
//! Arrays indexed by tier always have `MAX_TIERS` (3) elements.

use serde::{Serialize, Deserialize};

/// Tier index type. Zero-based: T1 = 0, T2 = 1, T3 = 2.
pub type Tier = usize;

/// Maximum number of hardware tiers.
pub const MAX_TIERS: usize = 3;

/// Number of GPU cost tiers (matches `GraphicsCard.scala` array lengths).
pub const GPU_COST_TIERS: usize = 3;

/// Number of filesystem speed tiers (matches `FileSystem.scala` array lengths).
pub const FS_SPEED_TIERS: usize = 6;

// -----------------------------------------------------------------------
// Per-tier cost structures
// -----------------------------------------------------------------------

/// GPU operation costs per tier.
///
/// Each array is `[tier0, tier1, tier2]`. The value is the call-budget
/// cost deducted per invocation. Lower values mean the operation is
/// cheaper and more operations can fit in a single tick's budget.
///
/// Source: `GraphicsCard.scala` cost arrays.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GpuCosts {
    /// Budget cost for `gpu.setBackground()`.
    pub set_background: [f64; GPU_COST_TIERS],

    /// Budget cost for `gpu.setForeground()`.
    pub set_foreground: [f64; GPU_COST_TIERS],

    /// Budget cost for `gpu.setPaletteColor()`.
    pub set_palette_color: [f64; GPU_COST_TIERS],

    /// Budget cost for `gpu.set(x, y, text)`.
    pub set: [f64; GPU_COST_TIERS],

    /// Budget cost for `gpu.copy(x, y, w, h, tx, ty)`.
    pub copy: [f64; GPU_COST_TIERS],

    /// Budget cost for `gpu.fill(x, y, w, h, char)`.
    pub fill: [f64; GPU_COST_TIERS],
}

impl Default for GpuCosts {
    /// Exact values from `GraphicsCard.scala`:
    ///
    /// ```text
    /// val setCosts  = Array(1.0/64,  1.0/128, 1.0/256)
    /// val fillCosts = Array(1.0/32,  1.0/64,  1.0/128)
    /// val copyCosts = Array(1.0/16,  1.0/32,  1.0/64)
    /// ```
    fn default() -> Self {
        Self {
            set_background:   [1.0/32.0, 1.0/64.0,  1.0/128.0],
            set_foreground:   [1.0/32.0, 1.0/64.0,  1.0/128.0],
            set_palette_color:[1.0/2.0,  1.0/8.0,   1.0/16.0 ],
            set:              [1.0/64.0, 1.0/128.0,  1.0/256.0],
            copy:             [1.0/16.0, 1.0/32.0,   1.0/64.0 ],
            fill:             [1.0/32.0, 1.0/64.0,   1.0/128.0],
        }
    }
}

/// Energy costs for GPU pixel operations.
///
/// These are per-character energy costs deducted from the network's
/// energy buffer. Currently not fully enforced by this emulator.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GpuEnergyCosts {
    /// Energy per character written via `gpu.set`.
    pub set_cost: f64,
    /// Energy per character copied via `gpu.copy`.
    pub copy_cost: f64,
    /// Energy per character filled via `gpu.fill`.
    pub fill_cost: f64,
    /// Energy per character cleared (space fill).
    pub clear_cost: f64,
}

impl Default for GpuEnergyCosts {
    fn default() -> Self {
        Self {
            set_cost: 1.0 / 64.0,
            copy_cost: 1.0 / 16.0,
            fill_cost: 1.0 / 32.0,
            clear_cost: 1.0 / 128.0,
        }
    }
}

/// VRAM size multipliers per tier.
///
/// The actual VRAM available is
/// `max_resolution.width * max_resolution.height * multiplier`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct VramSizes {
    /// Per-tier multipliers.
    pub multipliers: [f64; GPU_COST_TIERS],
}

impl Default for VramSizes {
    fn default() -> Self {
        Self { multipliers: [1.0, 2.0, 4.0] }
    }
}

/// Complete emulator configuration.
///
/// All defaults match `application.conf` shipped with OpenComputers 1.8.x.
/// Field names follow OC conventions where possible.

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OcConfig {
    // -- Screen / GPU --

    /// Maximum resolution per screen tier: `(width, height)`.
    ///
    /// Source: `Settings.screenResolutionsByTier`
    ///
    /// * T1 (index 0): 50x16
    /// * T2 (index 1): 80x25
    /// * T3 (index 2): 160x50
    pub screen_resolutions: [(u32, u32); MAX_TIERS],

    /// Maximum color depth per screen tier.
    ///
    /// Source: `Settings.screenDepthsByTier`
    pub screen_depths: [crate::display::ColorDepth; MAX_TIERS],

    /// Per-tier call-budget costs for GPU operations.
    pub gpu_costs: GpuCosts,

    /// Per-pixel energy costs for GPU operations.
    pub gpu_energy_costs: GpuEnergyCosts,

    /// VRAM size multipliers per tier.
    pub vram_sizes: VramSizes,

    /// Cost multiplier for `bitblt` operations.
    pub bitblt_cost: f64,

    // -- Machine --

    /// Call budgets per CPU tier.
    ///
    /// Determines how many direct calls a computer can perform per tick.
    /// Default: `[0.5, 1.0, 1.5]`.
    pub call_budgets: [f64; MAX_TIERS],

    /// Maximum execution time (seconds) before a forced yield.
    ///
    /// Source: `Settings.get.timeout` / `system.timeout()` in machine.lua.
    pub timeout: f64,

    /// Maximum number of signals in the queue.
    pub max_signal_queue_size: usize,

    /// Server tick frequency (ticks per energy event).
    pub tick_frequency: u32,

    /// Energy cost per tick for a running computer.
    pub computer_cost: f64,

    /// Energy multiplier while sleeping with no pending signals.
    pub sleep_cost_factor: f64,

    /// Maximum total RAM in bytes.
    pub max_total_ram: usize,

    /// Startup delay in seconds.
    pub startup_delay: f64,

    // -- EEPROM --

    /// Maximum code size in bytes.
    pub eeprom_size: usize,

    /// Maximum data size in bytes.
    pub eeprom_data_size: usize,

    /// Energy cost per EEPROM write.
    pub eeprom_write_cost: f64,

    // -- Filesystem --

    /// Size of `/tmp` in KiB. 0 disables it.
    pub tmp_size_kib: usize,

    /// Maximum open handles per filesystem per machine.
    pub max_handles: usize,

    /// Maximum bytes per `read()` call.
    pub max_read_buffer: usize,

    /// Per-file overhead for capacity accounting.
    pub file_cost: u64,

    /// Whether `/tmp` is erased on reboot.
    pub erase_tmp_on_reboot: bool,

    // -- Network --

    /// Maximum network packet size.
    pub max_network_packet_size: usize,

    /// Maximum data parts per network packet.
    pub max_network_packet_parts: usize,

    /// Maximum registered users per machine.
    pub max_users: usize,

    /// Maximum username length.
    pub max_username_length: usize,

    // -- Power --

    /// Whether to ignore power entirely (creative mode).
    pub ignore_power: bool,

    /// Energy buffer capacity for a computer.
    pub buffer_computer: f64,

    // -- Sound --

    /// Master volume multiplier applied to all audio output.
    ///
    /// Range: 0.0 (mute) to 1.0 (full). Default: 1.0.
    /// This is multiplied with both `effect_volume` and `beep_volume`
    /// before playing any sound, acting as a global volume knob.
    pub master_volume: f32,

    /// Volume for disk access and ambient sound effects.
    ///
    /// Range: 0.0 (mute) to 1.0 (full). Default: 0.4.
    /// Controls HDD access clicks, floppy insert/eject sounds, and the
    /// `computer_running` ambient loop. The effective volume heard is
    /// `master_volume * effect_volume`.
    pub effect_volume: f32,

    /// Volume for `computer.beep()` square-wave tones.
    ///
    /// Range: 0.0 (mute) to 1.0 (full). Default: 0.3.
    /// Controls both `beep(frequency, duration)` and `beep(pattern)`.
    /// The effective volume heard is `master_volume * beep_volume`.
    pub beep_volume: f32,

    // -- Misc --

    /// Whether Lua bytecode loading is permitted.
    pub allow_bytecode: bool,

    /// Whether `__gc` metamethods are permitted in the sandbox.
    pub allow_gc: bool,

    /// Whether persistence (Eris) is allowed.
    pub allow_persistence: bool,

    /// Whether item stack inspection is allowed.
    pub allow_item_stack_inspection: bool,

    /// Whether player usernames are appended to keyboard signals.
    pub input_username: bool,

    /// Selected CPU tier (0 = T1, 1 = T2, 2 = T3). Affects call budget.
    pub cpu_tier: usize,

    /// Selected RAM tier (0 = T1 192K, 1 = T2 384K, 2 = T3 768K,
    /// 3 = T4 unlimited). Affects `computer.totalMemory()`.
    pub ram_tier: usize,

    /// Selected screen/GPU tier (0 = T1, 1 = T2, 2 = T3).
    /// Affects max resolution and colour depth.
    pub screen_tier: usize,

    /// Multiplier for the per-tick call budget.
    /// 1.0 = OC default. `f64::INFINITY` = unlimited (no budget).
    pub call_budget_scale: f64,

    /// Multiplier for GPU call throughput.
    /// Higher values mean each GPU call costs less budget.
    /// 1.0 = OC default. `f64::INFINITY` = zero-cost GPU calls.
    pub gpu_budget_scale: f64,

}

impl Default for OcConfig {
    /// Construct the default configuration matching vanilla OC 1.8.x.
    fn default() -> Self {
        use crate::display::ColorDepth::*;
        Self {
            screen_resolutions: [(50, 16), (80, 25), (160, 50)],
            screen_depths: [OneBit, FourBit, EightBit],
            gpu_costs: GpuCosts::default(),
            gpu_energy_costs: GpuEnergyCosts::default(),
            vram_sizes: VramSizes::default(),
            bitblt_cost: 0.5,
            call_budgets: [0.5, 1.0, 1.5],
            timeout: 5.0,
            max_signal_queue_size: 256,
            tick_frequency: 1,
            computer_cost: 0.5,
            sleep_cost_factor: 0.1,
            max_total_ram: 8 * 1024 * 1024,
            startup_delay: 0.25,
            eeprom_size: 4096,
            eeprom_data_size: 256,
            eeprom_write_cost: 50.0,
            tmp_size_kib: 64,
            max_handles: 12,
            max_read_buffer: 2048,
            file_cost: 512,
            erase_tmp_on_reboot: true,
            max_network_packet_size: 8192,
            max_network_packet_parts: 8,
            max_users: 16,
            max_username_length: 32,
            ignore_power: false,
            buffer_computer: 500.0,
            master_volume: 1.0,
            effect_volume: 0.4,
            beep_volume: 0.3,
            allow_bytecode: false,
            allow_gc: false,
            allow_persistence: true,
            allow_item_stack_inspection: true,
            input_username: false,
            cpu_tier: 2,
            ram_tier: 2,
            screen_tier: 2,
            call_budget_scale: 1.0,
            gpu_budget_scale: 1.0,
        }
    }
}