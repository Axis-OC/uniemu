//! # Rendering subsystem
//!
//! Two pipelines share a common [`Renderer`] trait:
//! - [`IndirectRenderer`] / CPU uploads cell buffer each tick
//! - [`DirectRenderer`] / GPU-resident SSBOs, persistently mapped

pub mod vulkan_ctx;
pub mod pipeline;
pub mod indirect;
pub mod direct;

use crate::display::TextBuffer;

/// Common renderer interface.
pub trait Renderer {
    /// Render one frame.  Called per-vsync (DIRECT) or per-tick (INDIRECT).
    fn render_frame(&mut self, buffer: &TextBuffer) -> Result<(), RenderError>;

    /// Resize the swapchain to match new window dimensions.
    fn resize(&mut self, width: u32, height: u32) -> Result<(), RenderError>;
}

/// Rendering errors.
#[derive(Debug)]
pub enum RenderError {
    /// The swapchain is out of date and must be recreated.
    SwapchainOutOfDate,
    /// A Vulkan API call failed.
    Vulkan(ash::vk::Result),
    /// Missing compiled shader files.
    ShaderNotFound(&'static str),
    /// Other.
    Other(String),
}

impl From<ash::vk::Result> for RenderError {
    fn from(e: ash::vk::Result) -> Self { Self::Vulkan(e) }
}