//! # OpenComputers Configuration
//!
//! All magic numbers extracted from `application.conf` and `Settings.scala`.
//! Every field is documented with its original Scala source path.
//!
//! This module replaces the Typesafe/HOCON config system used by OC with
//! plain Rust structs that can be loaded from a TOML or JSON file, or
//! constructed with [`OcConfig::default()`] for vanilla OC defaults.

/// Tier index. OC uses zero-based tiers internally (T1 = 0, T2 = 1, T3 = 2).
pub type Tier = usize;

/// Maximum number of hardware tiers supported.
pub const MAX_TIERS: usize = 3;

/// Number of GPU cost tiers (matches `GraphicsCard.scala` array lengths).
pub const GPU_COST_TIERS: usize = 3;

/// Number of filesystem speed tiers (matches `FileSystem.scala` array lengths).
pub const FS_SPEED_TIERS: usize = 6;

// ─── Per-tier cost arrays ──────────────────────────────────────────────────

/// GPU operation costs per tier, extracted from `GraphicsCard.scala`.
///
/// Each array is `[tier0, tier1, tier2]` / the call-budget cost deducted
/// per invocation.  Lower values = cheaper = faster.
#[derive(Debug, Clone, Copy)]
pub struct GpuCosts {
    /// `setBackground` / `setForeground` budget cost.
    pub set_background: [f64; GPU_COST_TIERS],
    pub set_foreground: [f64; GPU_COST_TIERS],
    /// `setPaletteColor` budget cost.
    pub set_palette_color: [f64; GPU_COST_TIERS],
    /// `gpu.set(x,y,s)` budget cost.
    pub set: [f64; GPU_COST_TIERS],
    /// `gpu.copy(…)` budget cost.
    pub copy: [f64; GPU_COST_TIERS],
    /// `gpu.fill(…)` budget cost.
    pub fill: [f64; GPU_COST_TIERS],
}

impl Default for GpuCosts {
    /// Exact values from `GraphicsCard.scala`:
    /// ```scala
    /// final val setCosts = Array(1.0/64, 1.0/128, 1.0/256)
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

/// Energy costs for GPU pixel operations (per-pixel, from `Settings.get`).
#[derive(Debug, Clone, Copy)]
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

/// VRAM multipliers per tier, from `Settings.get.vramSizes`.
#[derive(Debug, Clone, Copy)]
pub struct VramSizes {
    /// Multiplier applied to `maxResolution.w * maxResolution.h` for VRAM.
    pub multipliers: [f64; GPU_COST_TIERS],
}

impl Default for VramSizes {
    fn default() -> Self {
        Self { multipliers: [1.0, 2.0, 4.0] }
    }
}

/// Complete emulator configuration.
///
/// Field names follow OpenComputers conventions where possible.
/// All defaults match `application.conf` shipped with OC 1.8.x.
#[derive(Debug, Clone)]
pub struct OcConfig {
    // ── Screen / GPU ────────────────────────────────────────────────────

    /// Maximum resolution per screen tier: `(width, height)`.
    ///
    /// Source: `Settings.screenResolutionsByTier`
    /// ```text
    /// T1 (index 0): 50×16
    /// T2 (index 1): 80×25
    /// T3 (index 2): 160×50
    /// ```
    pub screen_resolutions: [(u32, u32); MAX_TIERS],

    /// Maximum color depth per screen tier.
    ///
    /// Source: `Settings.screenDepthsByTier`
    pub screen_depths: [crate::display::ColorDepth; MAX_TIERS],

    /// Per-tier call-budget costs for GPU operations.
    pub gpu_costs: GpuCosts,

    /// Per-pixel energy costs for GPU operations.
    pub gpu_energy_costs: GpuEnergyCosts,

    /// VRAM size multipliers.
    pub vram_sizes: VramSizes,

    /// Cost multiplier for `bitblt`, from `Settings.get.bitbltCost`.
    pub bitblt_cost: f64,

    // ── Machine ─────────────────────────────────────────────────────────

    /// Call budgets per CPU tier. Determines how many direct calls a
    /// computer can perform per tick.
    ///
    /// Source: `Settings.get.callBudgets`  
    /// Default: `[0.5, 1.0, 1.5]`
    pub call_budgets: [f64; MAX_TIERS],

    /// Maximum execution time (seconds) before a forced yield.
    ///
    /// Source: `Settings.get.timeout` / `system.timeout()` in machine.lua
    pub timeout: f64,

    /// Maximum number of signals in the queue before overflow.
    ///
    /// Source: `Settings.get.maxSignalQueueSize`
    pub max_signal_queue_size: usize,

    /// Tick frequency (server ticks between energy consumption events).
    ///
    /// Source: `Settings.get.tickFrequency`
    pub tick_frequency: u32,

    /// Energy cost per tick for a running computer.
    ///
    /// Source: `Settings.get.computerCost`
    pub computer_cost: f64,

    /// Sleep cost factor (energy multiplier while sleeping with no signals).
    pub sleep_cost_factor: f64,

    /// Maximum total RAM across all memory modules, in bytes.
    ///
    /// Source: `Settings.get.maxTotalRam`
    pub max_total_ram: usize,

    /// Startup delay in seconds after loading a machine.
    pub startup_delay: f64,

    // ── EEPROM ──────────────────────────────────────────────────────────

    /// Maximum code size of an EEPROM, in bytes.
    ///
    /// Source: `Settings.get.eepromSize`
    pub eeprom_size: usize,

    /// Maximum volatile data size of an EEPROM, in bytes.
    ///
    /// Source: `Settings.get.eepromDataSize`
    pub eeprom_data_size: usize,

    /// Energy cost per EEPROM write operation.
    pub eeprom_write_cost: f64,

    // ── Filesystem ──────────────────────────────────────────────────────

    /// Size of the `/tmp` filesystem in KiB.  0 disables it.
    ///
    /// Source: `Settings.get.tmpSize`
    pub tmp_size_kib: usize,

    /// Maximum number of open file handles per filesystem per machine.
    pub max_handles: usize,

    /// Maximum bytes a single `read()` call may return.
    pub max_read_buffer: usize,

    /// Per-file overhead cost for filesystem capacity accounting.
    pub file_cost: u64,

    /// Whether to erase `/tmp` on reboot.
    pub erase_tmp_on_reboot: bool,

    // ── Network / Signals ───────────────────────────────────────────────

    /// Maximum network packet size.
    pub max_network_packet_size: usize,

    /// Maximum number of data parts in a network packet.
    pub max_network_packet_parts: usize,

    /// Maximum number of users that can be registered on a machine.
    pub max_users: usize,

    /// Maximum username length.
    pub max_username_length: usize,

    // ── Power ───────────────────────────────────────────────────────────

    /// Whether to ignore power entirely (creative mode).
    pub ignore_power: bool,

    /// Buffer size for a computer node (in energy units).
    pub buffer_computer: f64,

    // ── Misc ────────────────────────────────────────────────────────────

    /// Whether to allow Lua bytecode loading.
    pub allow_bytecode: bool,

    /// Whether to allow `__gc` metamethods in the sandbox.
    pub allow_gc: bool,

    /// Whether to allow persistence (Eris).
    pub allow_persistence: bool,

    /// Whether `component.list` / `getStackInSlot` etc. expose item info.
    pub allow_item_stack_inspection: bool,

    /// Whether player usernames are appended to keyboard signals.
    pub input_username: bool,
}

impl Default for OcConfig {
    fn default() -> Self {
        use crate::display::ColorDepth::*;
        Self {
            // Screen / GPU
            screen_resolutions: [(50, 16), (80, 25), (160, 50)],
            screen_depths: [OneBit, FourBit, EightBit],
            gpu_costs: GpuCosts::default(),
            gpu_energy_costs: GpuEnergyCosts::default(),
            vram_sizes: VramSizes::default(),
            bitblt_cost: 0.5,

            // Machine
            call_budgets: [0.5, 1.0, 1.5],
            timeout: 5.0,
            max_signal_queue_size: 256,
            tick_frequency: 1,
            computer_cost: 0.5,
            sleep_cost_factor: 0.1,
            max_total_ram: 8 * 1024 * 1024, // 8 MiB
            startup_delay: 0.25,

            // EEPROM
            eeprom_size: 4096,
            eeprom_data_size: 256,
            eeprom_write_cost: 50.0,

            // Filesystem
            tmp_size_kib: 64,
            max_handles: 12,
            max_read_buffer: 2048,
            file_cost: 512,
            erase_tmp_on_reboot: true,

            // Network
            max_network_packet_size: 8192,
            max_network_packet_parts: 8,
            max_users: 16,
            max_username_length: 32,

            // Power
            ignore_power: false,
            buffer_computer: 500.0,

            // Misc
            allow_bytecode: false,
            allow_gc: false,
            allow_persistence: true,
            allow_item_stack_inspection: true,
            input_username: false,
        }
    }
}