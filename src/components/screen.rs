//! # Screen Component
//!
//! Represents a physical screen block in the OpenComputers world.
//! The screen is a passive component: it provides metadata (tier,
//! touch mode, aspect ratio) but does not perform any rendering
//! itself. Rendering is done by the GPU component writing into
//! a shared [`TextBuffer`](crate::display::TextBuffer).
//!
//! ## Relationship to OpenComputers
//!
//! Mirrors `li.cil.oc.common.tileentity.Screen` and
//! `li.cil.oc.server.component.Screen` from the Scala source.
//!
//! In-game, a screen is a block (or multi-block structure) that:
//! * Has a tier determining max resolution and color depth.
//! * Can be in "character" mode (clicks report cell coordinates) or
//!   "precise" mode (clicks report exact pixel coordinates).
//! * Has an aspect ratio based on how many blocks wide and tall
//!   the multi-block screen is (e.g. a 2x1 screen has aspect 2:1).
//! * Can be turned on/off (when off, it shows a black screen).
//! * Has keyboard(s) attached (each screen block can have one).
//!
//! ## Exposed Lua methods
//!
//! * `screen.isOn()` / `screen.turnOn()` / `screen.turnOff()`
//! * `screen.getAspectRatio()` -> `(width_ratio, height_ratio)`
//! * `screen.getKeyboards()` -> table of keyboard addresses
//! * `screen.isPrecise()` / `screen.setPrecise(bool)`
//! * `screen.isTouchModeInverted()` / `screen.setTouchModeInverted(bool)`
//!
//! ## Multi-block screens
//!
//! In the real mod, screens can be combined into multi-block structures
//! up to 8x8 blocks. The total resolution is the sum of individual
//! block resolutions. This emulator does not simulate multi-block
//! screens; there is always a single 1x1 screen.

use crate::config::Tier;
use crate::display::ColorDepth;
use crate::components::Address;

/// Screen component.
///
/// Holds tier info, touch mode, and aspect ratio. Does NOT own the
/// text buffer (that is in [`EmulatorState`](crate::lua::host::EmulatorState)).
///
/// # Relationship to GPU
///
/// The GPU binds to a screen's address. The screen's tier limits the
/// maximum resolution and depth that the GPU can set (the effective
/// limit is `min(gpu_tier_limit, screen_tier_limit)`). In this
/// emulator, both are created at the same tier, so the limits match.
pub struct Screen {
    /// Unique UUID address of this screen component.
    pub address: Address,

    /// Hardware tier (0, 1, or 2).
    ///
    /// Determines max resolution and color depth:
    /// * Tier 0: 50x16, 1-bit
    /// * Tier 1: 80x25, 4-bit
    /// * Tier 2: 160x50, 8-bit
    tier: Tier,

    /// Whether the screen is in "precise" (sub-character) touch mode.
    ///
    /// * `false` (default): Touch/click events report cell coordinates
    ///   (integer column and row).
    /// * `true`: Touch/click events report exact pixel coordinates
    ///   within the screen area (floating-point).
    ///
    /// This emulator does not currently generate touch events, so this
    /// flag has no practical effect yet.
    touch_mode: bool,

    /// Physical aspect ratio of the screen.
    ///
    /// For single-block screens: `(1.0, 1.0)`.
    /// For multi-block screens: `(blocks_wide, blocks_tall)`.
    ///
    /// Used by `screen.getAspectRatio()` in Lua, which some programs
    /// use to calculate pixel aspect ratios for proper rendering.
    aspect: (f64, f64),
}

impl Screen {
    /// Create a new screen of the specified tier.
    ///
    /// # Arguments
    ///
    /// * `tier` - Hardware tier (0, 1, or 2). Values above 2 are
    ///   clamped to 2.
    ///
    /// # Returns
    ///
    /// A screen with a fresh UUID, default aspect ratio (1:1),
    /// and touch mode disabled.
    pub fn new(tier: Tier) -> Self {
        Self {
            address: crate::components::new_address(),
            tier: tier.min(2),
            touch_mode: false,
            aspect: (1.0, 1.0),
        }
    }

    /// Returns the OC component type name: `"screen"`.
    pub const fn component_name() -> &'static str { "screen" }

    /// The hardware tier of this screen.
    #[inline] pub fn tier(&self) -> Tier { self.tier }

    /// Maximum resolution this screen supports, looked up from config.
    ///
    /// Convenience method; the GPU also checks its own tier limit.
    /// The effective maximum is `min(gpu_max, screen_max)`.
    pub fn max_resolution(&self, config: &crate::config::OcConfig) -> (u32, u32) {
        config.screen_resolutions[self.tier]
    }

    /// Maximum color depth this screen supports, looked up from config.
    pub fn max_depth(&self, config: &crate::config::OcConfig) -> ColorDepth {
        config.screen_depths[self.tier]
    }

    /// Whether the screen is in precise (sub-character) touch mode.
    pub fn is_touch_mode(&self) -> bool { self.touch_mode }

    /// Set the touch mode.
    ///
    /// * `true` -> precise mode (floating-point coordinates)
    /// * `false` -> character mode (integer cell coordinates)
    pub fn set_touch_mode(&mut self, v: bool) { self.touch_mode = v; }

    /// Get the aspect ratio `(width, height)`.
    ///
    /// For a single-block screen, this is `(1.0, 1.0)`.
    pub fn aspect(&self) -> (f64, f64) { self.aspect }

    /// Set the aspect ratio.
    ///
    /// Used when simulating multi-block screens.
    pub fn set_aspect(&mut self, w: f64, h: f64) { self.aspect = (w, h); }
}