//! # Screen Component
//!
//! Represents a physical screen block. Holds tier information,
//! touch mode, and links to its text buffer.
//! Mirrors `Screen.scala` / the screen tile entity.

use crate::config::Tier;
use crate::display::ColorDepth;
use crate::components::Address;

/// Screen component.
pub struct Screen {
    pub address: Address,
    tier: Tier,
    /// Whether the screen is in "precise" (touch) mode vs character mode.
    touch_mode: bool,
    /// Physical aspect ratio (for multi-block screens, this is blocks wide/tall).
    aspect: (f64, f64),
}

impl Screen {
    pub fn new(tier: Tier) -> Self {
        Self {
            address: crate::components::new_address(),
            tier: tier.min(2),
            touch_mode: false,
            aspect: (1.0, 1.0),
        }
    }

    pub const fn component_name() -> &'static str { "screen" }

    #[inline] pub fn tier(&self) -> Tier { self.tier }

    /// Maximum resolution this screen supports (from config, but repeated here
    /// for convenience / the GPU clamps to `min(gpu_max, screen_max)`).
    pub fn max_resolution(&self, config: &crate::config::OcConfig) -> (u32, u32) {
        config.screen_resolutions[self.tier]
    }

    pub fn max_depth(&self, config: &crate::config::OcConfig) -> ColorDepth {
        config.screen_depths[self.tier]
    }

    pub fn is_touch_mode(&self) -> bool { self.touch_mode }
    pub fn set_touch_mode(&mut self, v: bool) { self.touch_mode = v; }

    pub fn aspect(&self) -> (f64, f64) { self.aspect }
    pub fn set_aspect(&mut self, w: f64, h: f64) { self.aspect = (w, h); }
}