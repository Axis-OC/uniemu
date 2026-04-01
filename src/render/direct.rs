//! # DIRECT renderer
//!
//! GPU-resident rendering: the cell data SSBO is persistently mapped.
//! Lua writes go directly to GPU-visible memory. A compute dispatch
//! (optional) copies only dirty rows to the render buffer.
//!
//! In the initial implementation, this uses the **same** fullscreen
//! fragment shader as INDIRECT mode.  The difference is that:
//! 1. The SSBO is persistently mapped (no staging + copy).
//! 2. No call budget is enforced (writes are "free").
//! 3. Rendering happens every vsync, not every tick.

use crate::display::TextBuffer;
use super::{Renderer, RenderError};

/// DIRECT rendering backend.
///
/// Currently a thin wrapper / the actual GPU-resident SSBO management
/// is TODO.  The architecture is in place for:
///
/// - **Triple-buffered SSBOs** (Lua writes to buffer N, GPU reads N-1)
/// - **Per-row dirty flags** (set by Lua writes, cleared by compute shader)
/// - **Async queues** (transfer queue for staging, compute queue for compose)
pub struct DirectRenderer {
    current_write_buffer: usize,
    // TODO: triple_buffers: [MappedSsbo; 3],
    // TODO: dirty_flags: MappedSsbo,
    // TODO: compose_pipeline: vk::Pipeline,
}

impl DirectRenderer {
    pub fn new() -> Self {
        Self { current_write_buffer: 0 }
    }
}

impl Renderer for DirectRenderer {
    fn render_frame(&mut self, _buffer: &TextBuffer) -> Result<(), RenderError> {
        // In DIRECT mode:
        // 1. Flip write buffer index.
        self.current_write_buffer = (self.current_write_buffer + 1) % 3;

        // 2. (Optional) Dispatch compute shader to copy dirty rows.
        // vkCmdDispatch(compose_pipeline, rows, 1, 1)

        // 3. Render fullscreen triangle using the read buffer.
        // (Same draw as IndirectRenderer, just different SSBO binding.)

        // TODO: actual Vulkan commands.
        Ok(())
    }

    fn resize(&mut self, _width: u32, _height: u32) -> Result<(), RenderError> {
        // TODO: Recreate swapchain + resize SSBOs if resolution changed.
        Ok(())
    }
}