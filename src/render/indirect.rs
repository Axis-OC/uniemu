//! # INDIRECT renderer
//!
//! CPU-side: each tick, the [`TextBuffer`] is uploaded to a Vulkan
//! storage buffer.  The fullscreen fragment shader reads from it.
//!
//! This mode faithfully reproduces OpenComputers' tick-rate rendering.

use ash::vk;
use crate::display::TextBuffer;
use super::{Renderer, RenderError};
use super::vulkan_ctx::VulkanContext;
use super::pipeline::{TextPipeline, TextPushConstants};

/// INDIRECT rendering backend.
pub struct IndirectRenderer {
    pipeline: TextPipeline,
    // TODO: The following are placeholders for the real Vulkan resources.
    // In a complete implementation, these would be:
    //   - cell_buffer: vk::Buffer  (SSBO with cell data)
    //   - cell_memory: vk::DeviceMemory
    //   - atlas_image: vk::Image  (R8 glyph atlas texture)
    //   - atlas_view: vk::ImageView
    //   - atlas_sampler: vk::Sampler
    //   - descriptor_pool: vk::DescriptorPool
    //   - descriptor_set: vk::DescriptorSet
    resolution: (u32, u32),
}

impl IndirectRenderer {
    /// Create the INDIRECT renderer.
    pub fn new(ctx: &VulkanContext, atlas_pixels: &[u8]) -> Result<Self, RenderError> {
        let pipeline = TextPipeline::new(ctx)?;

        // TODO: Create Vulkan buffer for cell data.
        // TODO: Create Vulkan image for glyph atlas, upload atlas_pixels.
        // TODO: Create descriptor pool and set.

        Ok(Self {
            pipeline,
            resolution: (80, 25), // default T2
        })
    }

    /// Upload the current text buffer state to the GPU SSBO.
    fn upload_cells(&mut self, _ctx: &VulkanContext, buffer: &TextBuffer) {
        self.resolution = (buffer.width(), buffer.height());

        // Build the SSBO data: 3 × u32 per cell.
        let _cell_data: Vec<u32> = buffer.cells().iter().flat_map(|c| {
            [
                c.codepoint,
                c.foreground.resolve(buffer.palette()),
                c.background.resolve(buffer.palette()),
            ]
        }).collect();

        // TODO: memcpy into mapped staging buffer, then vkCmdCopyBuffer
        // to the device-local SSBO.
    }

    /// Record and submit a frame.
    fn draw_frame(&self, ctx: &mut VulkanContext) -> Result<(), RenderError> {
        let frame = ctx.current_frame;
        let device = &ctx.device;

        unsafe {
            // Wait for previous frame to finish.
            device.wait_for_fences(&[ctx.in_flight[frame]], true, u64::MAX)
                .map_err(RenderError::Vulkan)?;

            // Acquire swapchain image.
            let (image_index, _suboptimal) = ctx.swapchain_loader
                .acquire_next_image(ctx.swapchain, u64::MAX, ctx.image_available[frame], vk::Fence::null())
                .map_err(|e| match e {
                    vk::Result::ERROR_OUT_OF_DATE_KHR => RenderError::SwapchainOutOfDate,
                    other => RenderError::Vulkan(other),
                })?;

            device.reset_fences(&[ctx.in_flight[frame]]).unwrap();

            let cmd = ctx.command_buffers[frame];
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();

            let begin_info = vk::CommandBufferBeginInfo::default();
            device.begin_command_buffer(cmd, &begin_info).unwrap();

            let clear = vk::ClearValue {
                color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] },
            };
            let clears = [clear];
            let rp_begin = vk::RenderPassBeginInfo::default()
                .render_pass(ctx.render_pass)
                .framebuffer(ctx.framebuffers[image_index as usize])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: ctx.swapchain_extent,
                })
                .clear_values(&clears);

            device.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline.pipeline);

            // Dynamic viewport/scissor.
            let viewport = vk::Viewport {
                x: 0.0, y: 0.0,
                width: ctx.swapchain_extent.width as f32,
                height: ctx.swapchain_extent.height as f32,
                min_depth: 0.0, max_depth: 1.0,
            };
            device.cmd_set_viewport(cmd, 0, &[viewport]);
            device.cmd_set_scissor(cmd, 0, &[vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: ctx.swapchain_extent,
            }]);

            // Push constants.
            let pc = TextPushConstants {
                resolution: [self.resolution.0, self.resolution.1],
                cell_pixels: [8, 16],
                atlas_cells: [256, 256],
            };
            let pc_bytes: &[u8] = std::slice::from_raw_parts(
                &pc as *const _ as *const u8,
                std::mem::size_of::<TextPushConstants>(),
            );
            device.cmd_push_constants(
                cmd, self.pipeline.pipeline_layout,
                vk::ShaderStageFlags::FRAGMENT,
                0, pc_bytes,
            );

            // TODO: Bind descriptor set with SSBO + atlas texture.
            // device.cmd_bind_descriptor_sets(...)

            // Draw fullscreen triangle (3 vertices, no buffer).
            device.cmd_draw(cmd, 3, 1, 0, 0);

            device.cmd_end_render_pass(cmd);
            device.end_command_buffer(cmd).unwrap();

            // Submit.
            let wait_semaphores = [ctx.image_available[frame]];
            let signal_semaphores = [ctx.render_finished[frame]];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let cmd_bufs = [cmd];
            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&cmd_bufs)
                .signal_semaphores(&signal_semaphores);

            device.queue_submit(ctx.graphics_queue, &[submit_info], ctx.in_flight[frame])
                .map_err(RenderError::Vulkan)?;

            // Present.
            let swapchains = [ctx.swapchain];
            let image_indices = [image_index];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            ctx.swapchain_loader.queue_present(ctx.present_queue, &present_info)
                .map_err(|e| match e {
                    vk::Result::ERROR_OUT_OF_DATE_KHR => RenderError::SwapchainOutOfDate,
                    other => RenderError::Vulkan(other),
                })?;

            ctx.current_frame = (frame + 1) % ctx.max_frames_in_flight;
        }

        Ok(())
    }
}

impl Renderer for IndirectRenderer {
    fn render_frame(&mut self, buffer: &TextBuffer) -> Result<(), RenderError> {
        if buffer.is_dirty() {
            // self.upload_cells(&ctx, buffer);
            // buffer.clear_dirty() should be called by the caller
        }
        // self.draw_frame(&mut ctx)
        // TODO: Need ctx reference / restructure to pass it in or store it.
        Ok(())
    }

    fn resize(&mut self, _width: u32, _height: u32) -> Result<(), RenderError> {
        // TODO: Recreate swapchain.
        Ok(())
    }
}