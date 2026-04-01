//! # GPU Component
//!
//! Implements the `gpu` component API from `GraphicsCard.scala` with
//! two rendering backends:
//!
//! | Aspect          | INDIRECT                  | DIRECT                     |
//! |-----------------|---------------------------|----------------------------|
//! | Call budget     | Enforced per-tick         | **Bypassed** (∞ budget)    |
//! | Energy costs    | Deducted per operation    | **Ignored**                |
//! | Render latency  | 1 tick (50 ms)            | **Immediate** (vsync)      |
//! | Dirty tracking  | Full buffer re-upload     | Per-row / per-cell SSBO    |

use crate::config::{OcConfig, Tier, GpuCosts};
use crate::display::{TextBuffer, ColorDepth, PackedColor};
use crate::machine::{Machine, EmulationMode};
use crate::components::Address;

/// GPU component instance.
///
/// Wraps a [`TextBuffer`] and provides the full `gpu.*` API surface.
/// The emulation mode is read from the parent [`Machine`] on each call.
pub struct Gpu {
    /// The tier of this GPU (0, 1, or 2).
    tier: Tier,
    /// Address of the screen this GPU is bound to (if any).
    screen_address: Option<Address>,
    /// Own component address.
    pub address: Address,
    /// Cost tables for this tier.
    costs: GpuCosts,
    /// Per-tier resolution limits.
    max_resolution: (u32, u32),
    /// Per-tier depth limit.
    max_depth: ColorDepth,
}

/// Result type for GPU operations that may fail due to budget exhaustion.
pub type GpuResult<T = ()> = Result<T, GpuError>;

/// Errors that a GPU call can produce.
#[derive(Debug)]
pub enum GpuError {
    /// Call budget exhausted (INDIRECT mode only).  Equivalent to
    /// `LimitReachedException` in OC.
    BudgetExhausted,
    /// Not enough energy in the network.
    NotEnoughEnergy,
    /// No screen bound.
    NoScreen,
    /// Invalid argument (bad resolution, unsupported depth, etc.).
    InvalidArgument(&'static str),
}

impl Gpu {
    /// Create a new GPU of the specified tier.
    pub fn new(tier: Tier, config: &OcConfig) -> Self {
        let tier = tier.min(2);
        Self {
            tier,
            screen_address: None,
            address: crate::components::new_address(),
            costs: config.gpu_costs,
            max_resolution: config.screen_resolutions[tier],
            max_depth: config.screen_depths[tier],
        }
    }

    /// The component type name.
    pub const fn component_name() -> &'static str { "gpu" }

    /// Maximum resolution for this GPU tier.
    #[inline]
    pub fn max_resolution(&self) -> (u32, u32) { self.max_resolution }

    /// Maximum color depth for this GPU tier.
    #[inline]
    pub fn max_depth(&self) -> ColorDepth { self.max_depth }

    // ── Core API (mirrors GraphicsCard.scala @Callback methods) ─────────

    /// `gpu.bind(address, reset?)` / bind to a screen.
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

    /// `gpu.setResolution(w, h)` / change active resolution.
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

    /// `gpu.getResolution()`.
    pub fn get_resolution(&self, buffer: &TextBuffer) -> (u32, u32) {
        (buffer.width(), buffer.height())
    }

    /// `gpu.setBackground(color, fromPalette?)`.
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

    /// `gpu.setForeground(color, fromPalette?)`.
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

    /// `gpu.set(x, y, value, vertical?)`.
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

    /// `gpu.fill(x, y, w, h, char)`.
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

    /// `gpu.copy(x, y, w, h, tx, ty)`.
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

    /// `gpu.setDepth(depth)`.
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

    /// `gpu.getDepth()` / returns bit count (1, 4, or 8).
    pub fn get_depth(&self, buffer: &TextBuffer) -> u8 {
        buffer.palette().depth().bits()
    }

    /// `gpu.get(x, y)` / returns `(char, fg, bg, fgIdx?, bgIdx?)`.
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