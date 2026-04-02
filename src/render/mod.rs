//! # Rendering subsystem
//!
//! Provides three rendering backends for displaying the emulated
//! OpenComputers screen:
//!
//! ```text
//! +------------------+-----------------------------+-----------+--------+
//! | Backend          | Path                        | Budget    | Hotkey |
//! +------------------+-----------------------------+-----------+--------+
//! | Software         | CPU raster -> softbuffer     | N/A       | F5    |
//! | VulkanIndirect   | CPU -> staging -> GPU SSBO   | per-tick  | F5    |
//! | VulkanDirect     | CPU -> persistent-map SSBO   | none      | F5    |
//! +------------------+-----------------------------+-----------+--------+
//! ```
//!
//! ## Software backend
//!
//! The software backend uses the `softbuffer` crate to blit a pixel
//! buffer directly to the window surface. It reads from the `TextBuffer`
//! and the glyph atlas on every frame. No GPU hardware is required.
//!
//! ## Vulkan INDIRECT backend
//!
//! Each frame (when the buffer is dirty), the cell data is copied
//! from the `TextBuffer` into a host-visible, persistently mapped
//! Vulkan SSBO. A fullscreen fragment shader reads the SSBO and the
//! glyph atlas texture to produce the final image. Call budgets are
//! enforced.
//!
//! ## Vulkan DIRECT backend
//!
//! Double-buffered persistent-mapped SSBOs with a `compose.comp`
//! compute shader dispatch that copies only dirty rows from the write
//! buffer to the read buffer. The fragment shader reads from the read
//! buffer. Call budgets are bypassed (infinite budget).
//!
//! ## Debug bar
//!
//! An optional overlay (F9) that shows real-time performance metrics
//! at the top of the screen. Rendered as a second draw call in the
//! same render pass (Vulkan) or composited in software.
//!
//! ## Sub-modules
//!
//! * [`vulkan_ctx`] - Vulkan instance, device, swapchain setup
//! * [`pipeline`] - Graphics pipeline and shader loading
//! * [`indirect`] - INDIRECT renderer implementation
//! * [`direct`] - DIRECT renderer implementation
//! * [`software`] - CPU rasteriser
//! * [`debug_bar`] - Debug overlay resources

pub mod vulkan_ctx;
pub mod pipeline;
pub mod indirect;
pub mod direct;
pub mod software;
pub mod debug_bar;

/// Active rendering backend.
///
/// Cycled by pressing F5 (or via the settings GUI).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderMode {
    /// CPU rasteriser -> `softbuffer`. Always available.
    Software,
    /// Vulkan with CPU-uploaded SSBO each dirty frame. Call budget enforced.
    VulkanIndirect,
    /// Vulkan with persistent-mapped SSBO. Call budget bypassed.
    VulkanDirect,
}

impl RenderMode {
    /// Cycle to the next mode in the sequence:
    /// Software -> VulkanIndirect -> VulkanDirect -> Software.
    pub fn next(self) -> Self {
        match self {
            Self::Software       => Self::VulkanIndirect,
            Self::VulkanIndirect => Self::VulkanDirect,
            Self::VulkanDirect   => Self::Software,
        }
    }

    /// Short human-readable label for the title bar / debug overlay.
    pub fn label(self) -> &'static str {
        match self {
            Self::Software       => "Software",
            Self::VulkanIndirect => "Vulkan INDIRECT",
            Self::VulkanDirect   => "Vulkan DIRECT",
        }
    }
}

/// Errors from the Vulkan rendering backends.
#[derive(Debug)]
pub enum RenderError {
    /// The swapchain is out of date (window resized). Must recreate.
    SwapchainOutOfDate,
    /// A Vulkan API call returned an error.
    Vulkan(ash::vk::Result),
    /// A required SPIR-V shader file was not found.
    ShaderNotFound(&'static str),
    /// No GPU memory type matched the requirements.
    NoSuitableMemory,
    /// Any other error.
    Other(String),
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SwapchainOutOfDate => write!(f, "swapchain out of date"),
            Self::Vulkan(e)         => write!(f, "Vulkan error: {e:?}"),
            Self::ShaderNotFound(p) => write!(f, "shader not found: {p}"),
            Self::NoSuitableMemory  => write!(f, "no suitable GPU memory type"),
            Self::Other(s)          => write!(f, "{s}"),
        }
    }
}

impl From<ash::vk::Result> for RenderError {
    fn from(e: ash::vk::Result) -> Self { Self::Vulkan(e) }
}
