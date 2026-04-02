//! # GPU Component
//!
//! Implements the `gpu` component API from OpenComputers, mirroring
//! `li.cil.oc.server.component.GraphicsCard` from the Scala source.
//!
//! ## Overview
//!
//! The GPU is the primary interface between Lua programs and the text
//! display. It provides methods to set resolution, change colors, write
//! text, fill rectangles, copy regions, and manage the color palette.
//!
//! ## Two rendering backends
//!
//! The GPU component itself is backend-agnostic; it operates on a
//! [`TextBuffer`] regardless of how that buffer is eventually rendered.
//! The distinction between INDIRECT and DIRECT mode affects whether
//! call budgets are enforced:
//!
//! ```text
//! +------------------+----------------------------+-----------------------------+
//! | Aspect           | INDIRECT                   | DIRECT                      |
//! +------------------+----------------------------+-----------------------------+
//! | Call budget      | Enforced per-tick          | Bypassed (infinite budget)  |
//! | Energy costs     | Deducted per operation     | Ignored                     |
//! | Render latency   | 1 tick (50 ms)             | Immediate (vsync)           |
//! | Dirty tracking   | Full buffer re-upload      | Per-row / per-cell SSBO     |
//! +------------------+----------------------------+-----------------------------+
//! ```
//!
//! ## Tier system
//!
//! GPUs come in three tiers (0, 1, 2), each with different:
//! * Maximum resolution (e.g. 50x16, 80x25, 160x50)
//! * Maximum color depth (1-bit, 4-bit, 8-bit)
//! * Call budget costs (higher tiers are cheaper per operation)
//!
//! The tier is set at construction time and cannot change.
//!
//! ## Color model
//!
//! Colors are either:
//! * Direct RGB: 24-bit `0x00RRGGBB` values
//! * Palette indices: 0-15 referencing the configurable palette
//!
//! The [`PackedColor`] type encodes both representations.
//!
//! ## Coordinate system
//!
//! OpenComputers uses 1-based coordinates in Lua (like Fortran!).
//! The GPU methods in this module accept the same 1-based coordinates
//! that Lua passes in. The translation to 0-based [`TextBuffer`]
//! coordinates happens in the dispatch layer (`host.rs`), where
//! `x.saturating_sub(1)` is applied.
//!
//! ## Screen binding
//!
//! A GPU must be bound to a screen before it can render. The `bind()`
//! method stores the screen's address. Multiple GPUs can bind to the
//! same screen, but only the most recently bound one "wins" (in OC,
//! this causes visual flickering; here, they share the same buffer).

use crate::config::{OcConfig, Tier, GpuCosts};
use crate::display::{TextBuffer, ColorDepth, PackedColor};
use crate::machine::{Machine, EmulationMode};
use crate::components::Address;

/// GPU component instance.
///
/// This struct holds the GPU's configuration (tier, costs, limits) and
/// the address of the screen it is bound to. It does NOT own the
/// [`TextBuffer`]; that is owned by [`EmulatorState`](crate::lua::host::EmulatorState)
/// and passed into GPU methods as a mutable reference.
///
/// # Lifetime
///
/// Created once during emulator setup. Lives for the entire session.
/// The screen binding can change at runtime via `bind()`.
///
/// # Thread safety
///
/// Not `Sync`. Accessed only from the main emulation thread.
pub struct Gpu {
    /// The hardware tier of this GPU (0, 1, or 2).
    ///
    /// Determines:
    /// * Maximum resolution (from `config.screen_resolutions[tier]`)
    /// * Maximum color depth (from `config.screen_depths[tier]`)
    /// * Per-operation call budget costs (from `config.gpu_costs`)
    ///
    /// Higher tiers have higher limits and lower costs (meaning more
    /// operations can be performed per tick).
    tier: Tier,

    /// Address of the screen this GPU is currently bound to.
    ///
    /// `None` if no screen is bound. Set by `bind()`.
    /// In OpenComputers, a GPU must be bound to a screen before any
    /// rendering methods work. In this emulator, we are more lenient
    /// (operations write to the buffer regardless), but `getScreen()`
    /// returns `nil` if unbound.
    screen_address: Option<Address>,

    /// The unique UUID address of this GPU component.
    ///
    /// Used in `component.list("gpu")` and `component.invoke(addr, ...)`.
    pub address: Address,

    /// Per-tier call budget cost tables.
    ///
    /// Each GPU operation (set, fill, copy, etc.) has a cost that is
    /// deducted from the machine's per-tick call budget. If the budget
    /// is exhausted, the operation fails with `BudgetExhausted`.
    ///
    /// In DIRECT mode, these costs are ignored (budget is infinite).
    costs: GpuCosts,

    /// Maximum resolution this GPU supports, from config.
    ///
    /// The actual resolution is `min(gpu_max, screen_max)`, but since
    /// we create both at the same tier, this is typically just the
    /// screen's max resolution.
    max_resolution: (u32, u32),

    /// Maximum color depth this GPU supports, from config.
    ///
    /// One of `OneBit`, `FourBit`, or `EightBit`.
    max_depth: ColorDepth,
}

/// Result type for GPU operations that may fail.
///
/// Most GPU methods return `GpuResult<T>` where `T` is the success
/// payload (often `()` or `bool`).
pub type GpuResult<T = ()> = Result<T, GpuError>;

/// Errors that a GPU method can produce.
///
/// These are translated into Lua-side errors in the dispatch layer.
/// `BudgetExhausted` is special: in OC, it causes the machine to yield
/// (similar to `LimitReachedException`).
#[derive(Debug)]
pub enum GpuError {
    /// The per-tick call budget has been exhausted (INDIRECT mode only).
    ///
    /// Equivalent to `LimitReachedException` in OpenComputers.
    /// The Lua program should yield and wait for the next tick.
    BudgetExhausted,

    /// Not enough energy in the computer's power network.
    ///
    /// Currently not enforced by this emulator (energy is always
    /// infinite unless configured otherwise).
    NotEnoughEnergy,

    /// No screen is bound to this GPU.
    ///
    /// Returned by operations that require a screen (currently not
    /// enforced; we always have a buffer).
    NoScreen,

    /// An argument was invalid.
    ///
    /// The static string describes what was wrong, e.g.:
    /// * `"unsupported resolution"` - Width/height out of range
    /// * `"unsupported depth"` - Depth not available at this tier
    InvalidArgument(&'static str),
}

impl Gpu {
    /// Create a new GPU of the specified tier.
    ///
    /// # Arguments
    ///
    /// * `tier` - Hardware tier (0, 1, or 2). Values above 2 are
    ///   clamped to 2.
    /// * `config` - Global configuration providing resolution limits,
    ///   depth limits, and cost tables.
    ///
    /// # What happens
    ///
    /// * A fresh UUID address is generated.
    /// * Resolution and depth limits are read from the config for the
    ///   given tier.
    /// * Cost tables are copied from `config.gpu_costs`.
    /// * No screen is bound initially (`screen_address = None`).
    pub fn new(tier: Tier, config: &OcConfig) -> Self {
        let tier = tier.min(2);
        let g = Self {
            tier,
            screen_address: None,
            address: crate::components::new_address(),
            costs: config.gpu_costs,
            max_resolution: config.screen_resolutions[tier],
            max_depth: config.screen_depths[tier],
        };
        log::info!("GPU created: tier={tier}, max_res={}x{}, max_depth={:?}",
            g.max_resolution.0, g.max_resolution.1, g.max_depth);
        g
    }

    /// Returns the OC component type name: `"gpu"`.
    pub const fn component_name() -> &'static str { "gpu" }

    /// Maximum resolution for this GPU tier.
    ///
    /// Returns `(max_width, max_height)`. For tier 2: `(160, 50)`.
    ///
    /// Note: the effective max resolution is `min(gpu_max, screen_max)`.
    /// Since both are created at the same tier in this emulator, they
    /// are typically equal.
    #[inline]
    pub fn max_resolution(&self) -> (u32, u32) { self.max_resolution }

    /// Maximum color depth for this GPU tier.
    ///
    /// * Tier 0: `OneBit` (1-bit, monochrome)
    /// * Tier 1: `FourBit` (4-bit, 16 colors)
    /// * Tier 2: `EightBit` (8-bit, 256 colors)
    #[inline]
    pub fn max_depth(&self) -> ColorDepth { self.max_depth }

    // -------------------------------------------------------------------
    // Core API (mirrors GraphicsCard.scala @Callback methods)
    // -------------------------------------------------------------------

    /// Bind this GPU to a screen (`gpu.bind(address, reset?)` in Lua).
    ///
    /// # Arguments
    ///
    /// * `buffer` - The text buffer to reset (if `reset` is true).
    /// * `address` - The address of the screen component to bind to.
    /// * `reset` - If `true`, resets resolution to max, depth to max,
    ///   foreground to white, and background to black.
    ///
    /// # Returns
    ///
    /// Always `Ok(())`. In OC, binding can fail if the screen doesn't
    /// exist, but we check that in the dispatch layer.
    pub fn bind(
        &mut self,
        buffer: &mut TextBuffer,
        address: Address,
        reset: bool,
    ) -> GpuResult {
        self.screen_address = Some(address);
        if reset {
            let (mw, mh) = self.max_resolution;
            buffer.set_resolution(mw, mh);
            buffer.set_color_depth(self.max_depth);
            buffer.set_foreground(PackedColor::rgb(0xFF_FF_FF));
            buffer.set_background(PackedColor::rgb(0x00_00_00));
        }
        Ok(())
    }

    /// Change the active resolution (`gpu.setResolution(w, h)` in Lua).
    ///
    /// # Arguments
    ///
    /// * `machine` - The machine (currently unused but reserved for
    ///   future budget/energy enforcement).
    /// * `buffer` - The text buffer to resize.
    /// * `w`, `h` - Desired width and height in characters.
    ///
    /// # Returns
    ///
    /// * `Ok(true)` - Resolution was changed.
    /// * `Ok(false)` - Resolution was already `(w, h)`.
    /// * `Err(InvalidArgument)` - Width or height is 0, exceeds the
    ///   per-tier maximum, or `w * h` exceeds `max_w * max_h`.
    ///
    /// # Side effects
    ///
    /// Resizing clears the entire buffer (all cells reset to default).
    pub fn set_resolution(
        &self,
        machine: &mut Machine,
        buffer: &mut TextBuffer,
        w: u32, h: u32,
    ) -> GpuResult<bool> {
        let (mw, mh) = self.max_resolution;
        if w < 1 || h < 1 || w > mw || h > mh || w * h > mw * mh {
            return Err(GpuError::InvalidArgument("unsupported resolution"));
        }
        Ok(buffer.set_resolution(w, h))
    }

    /// Get the current resolution (`gpu.getResolution()` in Lua).
    ///
    /// Returns `(width, height)` in characters.
    pub fn get_resolution(&self, buffer: &TextBuffer) -> (u32, u32) {
        (buffer.width(), buffer.height())
    }

    /// Set the background color (`gpu.setBackground(color, fromPalette?)` in Lua).
    ///
    /// # Arguments
    ///
    /// * `machine` - Used to deduct call budget (INDIRECT mode).
    /// * `buffer` - The text buffer whose background color is changed.
    /// * `color` - Either a 24-bit RGB value or a palette index.
    /// * `from_palette` - If `true`, `color` is interpreted as a palette
    ///   index (0-15); otherwise, as `0x00RRGGBB`.
    ///
    /// # Returns
    ///
    /// On success: `Ok((old_color_rgb, old_was_palette))`, returning the
    /// previous background color's resolved RGB value and whether it was
    /// a palette reference.
    ///
    /// On failure: `Err(BudgetExhausted)` if the call budget is depleted.
    pub fn set_background(
        &self,
        machine: &mut Machine,
        buffer: &mut TextBuffer,
        color: u32,
        from_palette: bool,
    ) -> GpuResult<(u32, bool)> {
        machine.consume_call_budget(self.costs.set_background[self.tier])
            .map_err(|_| GpuError::BudgetExhausted)?;
        let old = buffer.background();
        let new_color = if from_palette {
            PackedColor::palette(color as u8)
        } else {
            PackedColor::rgb(color)
        };
        buffer.set_background(new_color);
        Ok((old.value(), old.is_from_palette()))
    }

    /// Set the foreground color (`gpu.setForeground(color, fromPalette?)` in Lua).
    ///
    /// Analogous to [`set_background`](Gpu::set_background) but for the
    /// text (foreground) color. See that method's documentation for
    /// argument and return value details.
    pub fn set_foreground(
        &self,
        machine: &mut Machine,
        buffer: &mut TextBuffer,
        color: u32,
        from_palette: bool,
    ) -> GpuResult<(u32, bool)> {
        machine.consume_call_budget(self.costs.set_foreground[self.tier])
            .map_err(|_| GpuError::BudgetExhausted)?;
        let old = buffer.foreground();
        let new_color = if from_palette {
            PackedColor::palette(color as u8)
        } else {
            PackedColor::rgb(color)
        };
        buffer.set_foreground(new_color);
        Ok((old.value(), old.is_from_palette()))
    }

    /// Write a string at a position (`gpu.set(x, y, value, vertical?)` in Lua).
    ///
    /// # Arguments
    ///
    /// * `machine` - Used to deduct call budget.
    /// * `buffer` - The text buffer to write into.
    /// * `x`, `y` - 1-based column and row (translated to 0-based in
    ///   the dispatch layer).
    /// * `value` - The string to write. Each character occupies one cell.
    /// * `vertical` - If `true`, characters are laid out top-to-bottom
    ///   instead of left-to-right.
    ///
    /// # Errors
    ///
    /// Returns `Err(BudgetExhausted)` if the call budget is depleted.
    ///
    /// # Note
    ///
    /// Characters that fall outside the buffer bounds are silently
    /// discarded (no wrapping).
    pub fn set(
        &self,
        machine: &mut Machine,
        buffer: &mut TextBuffer,
        x: u32, y: u32,
        value: &str,
        vertical: bool,
    ) -> GpuResult {
        let char_count = value.chars().count() as f64;
        machine.consume_call_budget(self.costs.set[self.tier])
            .map_err(|_| GpuError::BudgetExhausted)?;
        buffer.set(x, y, value, vertical);
        Ok(())
    }

    /// Fill a rectangle with a character (`gpu.fill(x, y, w, h, char)` in Lua).
    ///
    /// # Arguments
    ///
    /// * `machine` - Used to deduct call budget.
    /// * `buffer` - The text buffer.
    /// * `x`, `y` - Top-left corner (1-based in Lua, translated in dispatch).
    /// * `w`, `h` - Width and height of the rectangle in cells.
    /// * `ch` - The character to fill with. Typically `' '` for clearing.
    ///
    /// # Errors
    ///
    /// Returns `Err(BudgetExhausted)` if the call budget is depleted.
    ///
    /// # Side effects
    ///
    /// All cells in the rectangle are set to `ch` with the current
    /// foreground and background colors.
    pub fn fill(
        &self,
        machine: &mut Machine,
        buffer: &mut TextBuffer,
        x: u32, y: u32, w: u32, h: u32,
        ch: char,
    ) -> GpuResult {
        machine.consume_call_budget(self.costs.fill[self.tier])
            .map_err(|_| GpuError::BudgetExhausted)?;
        buffer.fill(x, y, w, h, ch as u32);
        Ok(())
    }

    /// Copy a rectangular region (`gpu.copy(x, y, w, h, tx, ty)` in Lua).
    ///
    /// # Arguments
    ///
    /// * `machine` - Used to deduct call budget.
    /// * `buffer` - The text buffer.
    /// * `x`, `y` - Top-left corner of the source rectangle.
    /// * `w`, `h` - Size of the rectangle.
    /// * `tx`, `ty` - Translation offset (positive = right/down).
    ///
    /// # Errors
    ///
    /// Returns `Err(BudgetExhausted)` if the call budget is depleted.
    ///
    /// # Overlap handling
    ///
    /// Overlapping source and destination regions are handled correctly
    /// (the buffer internally chooses the right iteration order, similar
    /// to `memmove`).
    pub fn copy(
        &self,
        machine: &mut Machine,
        buffer: &mut TextBuffer,
        x: u32, y: u32, w: u32, h: u32,
        tx: i32, ty: i32,
    ) -> GpuResult {
        machine.consume_call_budget(self.costs.copy[self.tier])
            .map_err(|_| GpuError::BudgetExhausted)?;
        buffer.copy(x, y, w, h, tx, ty);
        Ok(())
    }

    /// Set the color depth (`gpu.setDepth(depth)` in Lua).
    ///
    /// # Arguments
    ///
    /// * `buffer` - The text buffer (palette is reset on depth change).
    /// * `bits` - Number of color bits: 1, 4, or 8.
    ///
    /// # Returns
    ///
    /// * `Ok(true)` - Depth was changed.
    /// * `Ok(false)` - Depth was already set to the requested value.
    /// * `Err(InvalidArgument("unsupported depth"))` - The requested
    ///   depth exceeds this GPU's tier capability.
    pub fn set_depth(
        &self,
        buffer: &mut TextBuffer,
        bits: u8,
    ) -> GpuResult<bool> {
        let depth = match bits {
            1 => ColorDepth::OneBit,
            4 if self.max_depth >= ColorDepth::FourBit  => ColorDepth::FourBit,
            8 if self.max_depth >= ColorDepth::EightBit => ColorDepth::EightBit,
            _ => return Err(GpuError::InvalidArgument("unsupported depth")),
        };
        Ok(buffer.set_color_depth(depth))
    }

    /// Get the current color depth in bits (`gpu.getDepth()` in Lua).
    ///
    /// Returns 1, 4, or 8.
    pub fn get_depth(&self, buffer: &TextBuffer) -> u8 {
        buffer.palette().depth().bits()
    }

    /// Read a single cell (`gpu.get(x, y)` in Lua).
    ///
    /// # Arguments
    ///
    /// * `buffer` - The text buffer.
    /// * `x`, `y` - Cell coordinates (0-based internally).
    ///
    /// # Returns
    ///
    /// `Some((character, fg_rgb, bg_rgb, fg_palette_idx, bg_palette_idx))`
    /// where:
    /// * `character` - The Unicode character in the cell.
    /// * `fg_rgb` - Resolved 24-bit foreground color.
    /// * `bg_rgb` - Resolved 24-bit background color.
    /// * `fg_palette_idx` - `Some(index)` if the foreground is a palette
    ///   reference, `None` if direct RGB.
    /// * `bg_palette_idx` - Same for background.
    ///
    /// Returns `None` if coordinates are out of bounds.
    pub fn get(&self, buffer: &TextBuffer, x: u32, y: u32)
        -> Option<(char, u32, u32, Option<u32>, Option<u32>)>
    {
        buffer.get(x, y).map(|cell| {
            let fg_resolved = cell.foreground.resolve(buffer.palette());
            let bg_resolved = cell.background.resolve(buffer.palette());
            let fg_idx = if cell.foreground.is_from_palette() {
                Some(cell.foreground.value())
            } else { None };
            let bg_idx = if cell.background.is_from_palette() {
                Some(cell.background.value())
            } else { None };
            (
                char::from_u32(cell.codepoint).unwrap_or('?'),
                fg_resolved,
                bg_resolved,
                fg_idx,
                bg_idx,
            )
        })
    }
}