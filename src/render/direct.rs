//! # DIRECT Vulkan renderer -- zero-budget, compute-accelerated
//!
//! The DIRECT renderer is designed for maximum throughput: it bypasses
//! the per-tick call budget so Lua can issue unlimited `gpu.*` calls,
//! and it uses a compute shader to minimize CPU->GPU data transfer.
//!
//! ## Architecture: double-buffered SSBOs with dirty-row compose
//!
//! ```text
//! Frame N (current_frame = 0)             Frame N+1 (current_frame = 1)
//!
//! +------------------+                    +------------------+
//! | write_ssbo[0]    |  <-- CPU writes    | write_ssbo[1]    |  <-- CPU writes
//! | (HOST_VISIBLE)   |     dirty rows     | (HOST_VISIBLE)   |     dirty rows
//! +------------------+                    +------------------+
//!         |                                       |
//!         | compose.comp                          | compose.comp
//!         | (copies dirty rows only)              | (copies dirty rows only)
//!         v                                       v
//! +------------------+                    +------------------+
//! | read_ssbo[0]     |  <-- frag reads    | read_ssbo[1]     |  <-- frag reads
//! | (DEVICE_LOCAL)   |                    | (DEVICE_LOCAL)   |
//! +------------------+                    +------------------+
//! ```
//!
//! ## Per-frame pipeline
//!
//! ```text
//! 1. Shadow-diff:
//!    - Compare each cell against the shadow buffer (CPU-side copy).
//!    - Find which rows have changed.
//!
//! 2. Upload dirty rows:
//!    - memcpy changed cells into write_ssbo[frame_index].
//!    - Set dirty_flags[row] = 1 for each changed row.
//!
//! 3. Inherit clean data:
//!    - cmd_copy_buffer: read_ssbo[1-frame] -> read_ssbo[frame]
//!    - This brings the previous frame's clean rows into the current
//!      read buffer.
//!
//! 4. Barrier: TRANSFER -> COMPUTE
//!
//! 5. Dispatch compose.comp:
//!    - For each row: if dirty_flags[row] == 1, copy from write_ssbo
//!      to read_ssbo, then clear the flag.
//!    - Workgroup size: 256. Dispatched as ceil(rows / 256).
//!
//! 6. Barrier: COMPUTE -> FRAGMENT
//!
//! 7. Render pass:
//!    - text.frag reads from read_ssbo[frame].
//!    - Fullscreen triangle + debug bar overlay.
//!
//! 8. Present to swapchain.
//! ```
//!
//! ## Budget bypass
//!
//! In [`EmulationMode::Direct`](crate::machine::EmulationMode::Direct),
//! [`Machine::consume_call_budget`](crate::machine::Machine::consume_call_budget)
//! is a no-op, so Lua can issue unlimited `gpu.*` calls per tick.
//! The shadow-diff mechanism ensures that only actually changed rows
//! are uploaded, keeping GPU bandwidth usage minimal even under heavy
//! Lua activity.
//!
//! ## Shadow buffer
//!
//! A CPU-side `Vec<u32>` mirrors the contents of the last uploaded
//! frame. On each frame, cells are compared against the shadow; only
//! rows where at least one cell differs are marked dirty and uploaded.
//! This avoids re-uploading the entire SSBO when only a few characters
//! change (which is the common case for terminal-like applications).
//!
//! ## Resource ownership
//!
//! `DirectRenderer` owns:
//! * 2x write SSBOs (HOST_VISIBLE, persistently mapped)
//! * 2x dirty-flags SSBOs (HOST_VISIBLE, persistently mapped)
//! * 2x read SSBOs (DEVICE_LOCAL)
//! * 2x compute descriptor sets
//! * 2x graphics descriptor sets
//! * Compose compute pipeline + layout + DSL
//! * Text graphics pipeline
//! * Atlas image/view/sampler
//! * Descriptor pool
//! * Shadow buffer (CPU-side Vec)
//! * Debug bar overlay

use ash::vk;
use std::os::raw::c_int;

use crate::display::TextBuffer;
use crate::display::font::ATLAS_SIZE;
use super::RenderError;
use super::vulkan_ctx::VulkanContext;
use super::pipeline::{self, TextPipeline, TextPushConstants};
use super::indirect::{upload_atlas, find_memory_type};

/// Maximum cell count we pre-allocate for.
///
/// 160 columns * 50 rows = 8000 cells.
const MAX_CELLS: usize = 160 * 50;

/// Maximum rows (for dirty-flags SSBO).
///
/// 64 rows is generous; T3 screens are only 50 rows.
const MAX_ROWS: usize = 64;

/// SSBO byte size for cell data: `MAX_CELLS * 3 * sizeof(u32)`.
///
/// 8000 * 12 = 96,000 bytes.
const SSBO_SIZE: u64 = (MAX_CELLS * 3 * 4) as u64;

/// Dirty-flags SSBO byte size: one `u32` per row.
///
/// 64 * 4 = 256 bytes.
const DIRTY_SIZE: u64 = (MAX_ROWS * 4) as u64;

// -----------------------------------------------------------------------
// Push constants for compose.comp
// -----------------------------------------------------------------------

/// Push constants for the compose compute shader.
///
/// Must match `compose.comp`'s `layout(push_constant)` block exactly.
#[repr(C)]
#[derive(Clone, Copy)]
struct ComposePush {
    /// Grid width in columns.
    cols: u32,
    /// Grid height in rows.
    rows: u32,
}

// -----------------------------------------------------------------------
// Per-frame-in-flight resources
// -----------------------------------------------------------------------

/// Resources that are duplicated per frame-in-flight (double-buffered).
///
/// Each frame has its own set of SSBOs and descriptor sets to avoid
/// synchronisation between frames.
struct FrameData {
    /// CPU -> GPU staging SSBO.
    ///
    /// HOST_VISIBLE | HOST_COHERENT, persistently mapped.
    /// Written by the CPU with dirty cell data.
    write_buf: vk::Buffer,
    write_mem: vk::DeviceMemory,
    write_ptr: *mut u32,

    /// Per-row dirty flags SSBO.
    ///
    /// HOST_VISIBLE | HOST_COHERENT, persistently mapped.
    /// Set to 1 by the CPU for each dirty row.
    /// Cleared to 0 by compose.comp after copying.
    dirty_buf: vk::Buffer,
    dirty_mem: vk::DeviceMemory,
    dirty_ptr: *mut u32,

    /// Fragment-shader source-of-truth SSBO.
    ///
    /// DEVICE_LOCAL (fast GPU access).
    /// Written by compose.comp, read by text.frag.
    read_buf: vk::Buffer,
    read_mem: vk::DeviceMemory,

    /// Descriptor set for compose.comp: {write, dirty, read}.
    compute_ds: vk::DescriptorSet,

    /// Descriptor set for text.frag: {read, atlas}.
    graphics_ds: vk::DescriptorSet,
}

/// DIRECT rendering backend.
///
/// See module-level documentation for the full architecture description.
pub struct DirectRenderer {
    // -- Pipelines --

    /// Text rendering graphics pipeline (shared with INDIRECT).
    text_pipeline: TextPipeline,

    /// Compose compute pipeline (copies dirty rows from write to read SSBO).
    compose_pipeline: vk::Pipeline,

    /// Pipeline layout for compose.comp (push constants + 3 SSBOs).
    compose_layout: vk::PipelineLayout,

    /// Descriptor set layout for compose.comp (3 storage buffers).
    compose_dsl: vk::DescriptorSetLayout,

    // -- Per-frame resources (double-buffered) --

    /// Two sets of frame resources, indexed by `ctx.current_frame`.
    frames: [FrameData; 2],

    // -- Atlas --

    /// Glyph atlas image (same as INDIRECT; uploaded once).
    atlas_image: vk::Image,
    atlas_memory: vk::DeviceMemory,
    atlas_view: vk::ImageView,
    atlas_sampler: vk::Sampler,

    // -- Descriptors --

    /// Descriptor pool for all sets (compute + graphics + debug bar).
    pool: vk::DescriptorPool,

    // -- Shadow buffer for dirty detection --

    /// CPU-side copy of the last uploaded cell data.
    ///
    /// Layout: `[codepoint, fg, bg, codepoint, fg, bg, ...]`
    /// = 3 u32 values per cell, same as the SSBO layout.
    ///
    /// Used to detect which rows have changed since the last frame.
    shadow: Vec<u32>,

    // -- Tracked resolution --

    /// Cached resolution from the last render call.
    resolution: (u32, u32),

    /// Optional debug bar overlay.
    debug_bar: Option<super::debug_bar::VkDebugBar>,
}

impl DirectRenderer {
    /// Create all Vulkan resources for the DIRECT renderer.
    ///
    /// # What happens
    ///
    /// 1. Create the text graphics pipeline.
    /// 2. Create the compose compute pipeline.
    /// 3. Upload the glyph atlas.
    /// 4. Create the descriptor pool.
    /// 5. Create two sets of frame resources (write/dirty/read SSBOs +
    ///    compute/graphics descriptor sets).
    /// 6. Create the debug bar overlay.
    ///
    /// # Safety
    ///
    /// `ctx` must be fully initialised. `atlas_pixels` must be >=
    /// `ATLAS_SIZE * ATLAS_SIZE` bytes.
    pub unsafe fn new(
        ctx: &mut VulkanContext,
        atlas_pixels: &[u8],
    ) -> Result<Self, RenderError> {
        log::info!("Creating DIRECT renderer (double-buffered, compute compose)...");
        let text_pipeline = TextPipeline::new(ctx)?;

        log::debug!("Creating compose compute pipeline...");
        let (compose_pipeline, compose_layout, compose_dsl) = create_compose_pipeline(ctx)?;
        log::debug!("Compose pipeline created (3 SSBOs, 8-byte push constants)");

        let (atlas_image, atlas_memory, atlas_view, atlas_sampler) =
            upload_atlas(ctx, atlas_pixels)?;

        log::debug!("Creating descriptor pool (9 SSBOs + 3 samplers, 5 sets)");
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 9,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 3,
            },
        ];
        let pool_ci = vk::DescriptorPoolCreateInfo::default()
            .max_sets(5)
            .pool_sizes(&pool_sizes);
        let pool = ctx.device.create_descriptor_pool(&pool_ci, None)
            .map_err(RenderError::Vulkan)?;

        log::debug!("Creating frame resources (2 frames in flight)...");
        let f0 = create_frame(ctx, &pool, &compose_dsl,
            &text_pipeline.descriptor_set_layout, atlas_view, atlas_sampler)?;
        log::debug!("  Frame 0: write={} dirty={} read={} bytes",
            SSBO_SIZE, DIRTY_SIZE, SSBO_SIZE);
        let f1 = create_frame(ctx, &pool, &compose_dsl,
            &text_pipeline.descriptor_set_layout, atlas_view, atlas_sampler)?;
        log::debug!("  Frame 1: created");

        let debug_bar = super::debug_bar::VkDebugBar::new(
            ctx, pool,
            text_pipeline.descriptor_set_layout,
            atlas_view, atlas_sampler,
        ).ok();

        log::info!("DIRECT renderer created (shadow buffer={} bytes)",
            MAX_CELLS * 3 * 4);
        Ok(Self {
            text_pipeline,
            compose_pipeline, compose_layout, compose_dsl,
            frames: [f0, f1],
            atlas_image, atlas_memory, atlas_view, atlas_sampler,
            pool,
            shadow: vec![0u32; MAX_CELLS * 3],
            resolution: (80, 25), debug_bar,
        })
    }

/// Render one frame using the DIRECT pipeline.
    ///
    /// Performs shadow-diff to detect which rows changed since the last
    /// frame, uploads only dirty rows into the write SSBO, dispatches
    /// the compose compute shader to merge dirty rows into the read SSBO,
    /// renders the text via the fullscreen fragment shader, and presents.
    ///
    /// ## Per-frame pipeline
    ///
    /// ```text
    /// 1. Shadow diff (CPU): compare cells against shadow buffer, find dirty rows.
    /// 2. Upload dirty rows into write_ssbo[frame].
    /// 3. cmd_copy_buffer: read_ssbo[other_frame] -> read_ssbo[frame]  (inherit clean rows).
    /// 4. Barrier: TRANSFER -> COMPUTE.
    /// 5. Dispatch compose.comp (copies dirty rows from write to read, clears flags).
    /// 6. Barrier: COMPUTE -> FRAGMENT.
    /// 7. Render pass: text.frag reads from read_ssbo[frame].
    /// 8. Present to swapchain.
    /// ```
    ///
    /// ## Semaphore strategy
    ///
    /// Same as INDIRECT: `render_finished` is indexed by swapchain image
    /// (not frame-in-flight) to prevent semaphore reuse before the
    /// presentation engine releases it.
    ///
    /// # Arguments
    ///
    /// * `ctx` — Mutable Vulkan context.
    /// * `buffer` — The text buffer to render.
    /// * `debug_cells` — Optional debug bar overlay data.
    ///
    /// # Errors
    ///
    /// * `RenderError::SwapchainOutOfDate` — Swapchain needs recreation.
    /// * `RenderError::Vulkan(e)` — Any other Vulkan error.
    pub fn render_frame(
        &mut self,
        ctx: &mut VulkanContext,
        buffer: &TextBuffer,
        debug_cells: Option<&[[u32; 3]]>,
    ) -> Result<(), RenderError> {
        let device = &ctx.device;
        let fi = ctx.current_frame;
        let fr = &self.frames[fi];

        let (cols, rows) = (buffer.width(), buffer.height());
        let total = (cols * rows) as usize;

        // Resolution change invalidates the entire shadow buffer.
        let full_dirty = self.resolution != (cols, rows);
        if full_dirty {
            self.resolution = (cols, rows);
            self.shadow.resize(total * 3, 0);
            self.shadow.fill(0);
        }

        let cells = buffer.cells();
        let count = total.min(MAX_CELLS);
        let mut any_dirty = false;

        unsafe {
            std::ptr::write_bytes(fr.dirty_ptr, 0, MAX_ROWS);

            let src = fr.write_ptr;
            for row in 0..(rows as usize).min(MAX_ROWS) {
                let row_start = row * cols as usize;
                let row_end = (row_start + cols as usize).min(count);
                let mut row_dirty = full_dirty;

                if !row_dirty {
                    let shadow_base = row_start * 3;
                    let shadow_end = row_end * 3;
                    let shadow_slice = &self.shadow[shadow_base..shadow_end];

                    'cmp: for ci in row_start..row_end {
                        let base = (ci - row_start) * 3;
                        let cell = &cells[ci];
                        let cp = cell.codepoint;
                        let fg = cell.foreground.resolve(buffer.palette());
                        let bg = cell.background.resolve(buffer.palette());
                        if shadow_slice[base] != cp
                            || shadow_slice[base + 1] != fg
                            || shadow_slice[base + 2] != bg
                        {
                            row_dirty = true;
                            break 'cmp;
                        }
                    }
                }

                if row_dirty {
                    any_dirty = true;
                    for ci in row_start..row_end {
                        let base = ci * 3;
                        let cell = &cells[ci];
                        let cp = cell.codepoint;
                        let fg = cell.foreground.resolve(buffer.palette());
                        let bg = cell.background.resolve(buffer.palette());
                        *src.add(base)     = cp;
                        *src.add(base + 1) = fg;
                        *src.add(base + 2) = bg;
                        self.shadow[base]     = cp;
                        self.shadow[base + 1] = fg;
                        self.shadow[base + 2] = bg;
                    }
                    *fr.dirty_ptr.add(row) = 1;
                }
            }
        }

        // Second early-out: shadow diff found nothing actually changed.
        if !any_dirty {
            return Ok(());
        }

        // -- Record command buffer and submit --
        unsafe {
            // Wait for this frame-in-flight slot to be free.
            {
                let _prof = crate::profiler::scope(
                    crate::profiler::Cat::VkFence, "wait_fence");
                device.wait_for_fences(&[ctx.in_flight[fi]], true, u64::MAX)
                    .map_err(RenderError::Vulkan)?;
            }

            // Acquire the next swapchain image.
            let image_index;
            {
                let _prof = crate::profiler::scope(
                    crate::profiler::Cat::VkAcquire, "acquire_image");
                let (idx, _) = ctx.swapchain_loader
                    .acquire_next_image(ctx.swapchain, u64::MAX,
                        ctx.image_available[fi], vk::Fence::null())
                    .map_err(|e| match e {
                        vk::Result::ERROR_OUT_OF_DATE_KHR => RenderError::SwapchainOutOfDate,
                        o => RenderError::Vulkan(o),
                    })?;
                image_index = idx;
            }

            let img = image_index as usize;

            // If another frame-in-flight is still rendering to this swapchain
            // image, wait for it before we overwrite the framebuffer.
            if ctx.images_in_flight[img] != vk::Fence::null() {
                device.wait_for_fences(&[ctx.images_in_flight[img]], true, u64::MAX)
                    .map_err(RenderError::Vulkan)?;
            }
            ctx.images_in_flight[img] = ctx.in_flight[fi];

            device.reset_fences(&[ctx.in_flight[fi]]).unwrap();
            let cmd = ctx.command_buffers[fi];
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();

            // Inherit clean rows from the other frame's read SSBO.
            // This avoids re-uploading unchanged rows every frame.
            let other = &self.frames[1 - fi];
            let copy = [vk::BufferCopy { src_offset: 0, dst_offset: 0, size: SSBO_SIZE }];
            device.cmd_copy_buffer(cmd, other.read_buf, fr.read_buf, &copy);

            // Barrier: TRANSFER writes must complete before compute reads.
            let transfer_to_compute = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
            device.cmd_pipeline_barrier(cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(), &transfer_to_compute, &[], &[]);

            // Dispatch compose.comp: for each dirty row, copies data from
            // write_ssbo to read_ssbo and clears the dirty flag.
            device.cmd_bind_pipeline(
                cmd, vk::PipelineBindPoint::COMPUTE, self.compose_pipeline);
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE,
                self.compose_layout, 0, &[fr.compute_ds], &[]);
            let cpc = ComposePush { cols, rows };
            let cpc_bytes: &[u8] = std::slice::from_raw_parts(
                &cpc as *const _ as *const u8, 8);
            device.cmd_push_constants(cmd, self.compose_layout,
                vk::ShaderStageFlags::COMPUTE, 0, cpc_bytes);
            device.cmd_dispatch(cmd, (rows + 255) / 256, 1, 1);

            // Barrier: compute writes must complete before fragment reads.
            let compute_to_fragment = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)];
            device.cmd_pipeline_barrier(cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(), &compute_to_fragment, &[], &[]);

            // -- Render pass (uses framebuffer[img], not framebuffer[fi]) --
            let clear = vk::ClearValue {
                color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] },
            };
            let clears = [clear];
            let rp = vk::RenderPassBeginInfo::default()
                .render_pass(ctx.render_pass)
                .framebuffer(ctx.framebuffers[img])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent: ctx.swapchain_extent,
                })
                .clear_values(&clears);
            device.cmd_begin_render_pass(cmd, &rp, vk::SubpassContents::INLINE);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS,
                self.text_pipeline.pipeline);

            // Dynamic viewport and scissor.
            let vp = vk::Viewport {
                x: 0.0, y: 0.0,
                width: ctx.swapchain_extent.width as f32,
                height: ctx.swapchain_extent.height as f32,
                min_depth: 0.0, max_depth: 1.0,
            };
            device.cmd_set_viewport(cmd, 0, &[vp]);
            device.cmd_set_scissor(cmd, 0, &[vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: ctx.swapchain_extent,
            }]);

            // Push constants for the text fragment shader.
            let pc = TextPushConstants {
                resolution: [cols, rows],
                cell_pixels: [8, 16],
                atlas_cells: [256, 256],
                screen_pixels: [ctx.swapchain_extent.width, ctx.swapchain_extent.height],
            };
            let pc_bytes: &[u8] = std::slice::from_raw_parts(
                &pc as *const _ as *const u8,
                std::mem::size_of::<TextPushConstants>());
            device.cmd_push_constants(cmd, self.text_pipeline.pipeline_layout,
                vk::ShaderStageFlags::FRAGMENT, 0, pc_bytes);

            // Bind the graphics descriptor set (read SSBO + atlas sampler).
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS,
                self.text_pipeline.pipeline_layout, 0, &[fr.graphics_ds], &[]);

            // Fullscreen triangle draw.
            device.cmd_draw(cmd, 3, 1, 0, 0);

            // Optional debug bar overlay (second draw call, same render pass).
            if let (Some(bar), Some(cells)) = (&self.debug_bar, debug_cells) {
                bar.draw(
                    device, cmd,
                    self.text_pipeline.pipeline_layout,
                    cells,
                    ctx.swapchain_extent.width,
                );
            }

            device.cmd_end_render_pass(cmd);
            device.end_command_buffer(cmd).unwrap();

            // -- Submit --
            // Signal render_finished[img] (indexed by swapchain image).
            {
                let _prof = crate::profiler::scope(
                    crate::profiler::Cat::VkSubmit, "queue_submit");
                let wait = [ctx.image_available[fi]];
                let signal = [ctx.render_finished[img]];
                let stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
                let cmds = [cmd];
                let submit = vk::SubmitInfo::default()
                    .wait_semaphores(&wait)
                    .wait_dst_stage_mask(&stages)
                    .command_buffers(&cmds)
                    .signal_semaphores(&signal);
                device.queue_submit(ctx.graphics_queue, &[submit], ctx.in_flight[fi])
                    .map_err(RenderError::Vulkan)?;
            }

            // -- Present --
            // Wait on render_finished[img] before displaying.
            {
                let _prof = crate::profiler::scope(
                    crate::profiler::Cat::VkPresent, "queue_present");
                let swapchains = [ctx.swapchain];
                let indices = [image_index];
                let signal = [ctx.render_finished[img]];
                let present = vk::PresentInfoKHR::default()
                    .wait_semaphores(&signal)
                    .swapchains(&swapchains)
                    .image_indices(&indices);
                ctx.swapchain_loader.queue_present(ctx.present_queue, &present)
                    .map_err(|e| match e {
                        vk::Result::ERROR_OUT_OF_DATE_KHR => RenderError::SwapchainOutOfDate,
                        o => RenderError::Vulkan(o),
                    })?;
            }

            ctx.current_frame = (fi + 1) % ctx.max_frames_in_flight;
        }
        Ok(())
    }

    /// Destroy all owned Vulkan resources.
    pub fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            let _ = device.device_wait_idle();
            for f in &self.frames {
                device.unmap_memory(f.write_mem);
                device.destroy_buffer(f.write_buf, None);
                device.free_memory(f.write_mem, None);
                device.unmap_memory(f.dirty_mem);
                device.destroy_buffer(f.dirty_buf, None);
                device.free_memory(f.dirty_mem, None);
                device.destroy_buffer(f.read_buf, None);
                device.free_memory(f.read_mem, None);
            }
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_sampler(self.atlas_sampler, None);
            device.destroy_image_view(self.atlas_view, None);
            device.destroy_image(self.atlas_image, None);
            device.free_memory(self.atlas_memory, None);
            device.destroy_pipeline(self.compose_pipeline, None);
            device.destroy_pipeline_layout(self.compose_layout, None);
            device.destroy_descriptor_set_layout(self.compose_dsl, None);
            if let Some(bar) = &self.debug_bar { bar.destroy(device); }
            self.text_pipeline.destroy(device);
        }
    }
}

// -----------------------------------------------------------------------
// Resource creation helpers
// -----------------------------------------------------------------------

/// Create a HOST_VISIBLE | HOST_COHERENT buffer, persistently mapped.
///
/// Returns `(buffer, memory, mapped_pointer)`.
unsafe fn host_buf(
    ctx: &VulkanContext, size: u64, usage: vk::BufferUsageFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory, *mut u8), RenderError> {
    let ci = vk::BufferCreateInfo::default()
        .size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buf = ctx.device.create_buffer(&ci, None).map_err(RenderError::Vulkan)?;
    let req = ctx.device.get_buffer_memory_requirements(buf);
    let mi = find_memory_type(&ctx.instance, ctx.physical_device, req.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)?;
    let ai = vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(mi);
    let mem = ctx.device.allocate_memory(&ai, None).map_err(RenderError::Vulkan)?;
    ctx.device.bind_buffer_memory(buf, mem, 0).map_err(RenderError::Vulkan)?;
    let ptr = ctx.device.map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        .map_err(RenderError::Vulkan)? as *mut u8;
    std::ptr::write_bytes(ptr, 0, size as usize);
    Ok((buf, mem, ptr))
}

/// Create a DEVICE_LOCAL buffer (not mapped).
///
/// Returns `(buffer, memory)`.
unsafe fn device_buf(
    ctx: &VulkanContext, size: u64, usage: vk::BufferUsageFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory), RenderError> {
    let ci = vk::BufferCreateInfo::default()
        .size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buf = ctx.device.create_buffer(&ci, None).map_err(RenderError::Vulkan)?;
    let req = ctx.device.get_buffer_memory_requirements(buf);
    let mi = find_memory_type(&ctx.instance, ctx.physical_device, req.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
    let ai = vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(mi);
    let mem = ctx.device.allocate_memory(&ai, None).map_err(RenderError::Vulkan)?;
    ctx.device.bind_buffer_memory(buf, mem, 0).map_err(RenderError::Vulkan)?;
    Ok((buf, mem))
}

/// Build one frame's worth of buffers + descriptor sets.
///
/// Creates:
/// * write_buf (HOST_VISIBLE, for CPU -> GPU dirty row data)
/// * dirty_buf (HOST_VISIBLE, for per-row dirty flags)
/// * read_buf (DEVICE_LOCAL, for fragment shader reads)
/// * compute_ds (bindings: write, dirty, read)
/// * graphics_ds (bindings: read, atlas)
unsafe fn create_frame(
    ctx: &VulkanContext,
    pool: &vk::DescriptorPool,
    compose_dsl: &vk::DescriptorSetLayout,
    graphics_dsl: &vk::DescriptorSetLayout,
    atlas_view: vk::ImageView,
    atlas_sampler: vk::Sampler,
) -> Result<FrameData, RenderError> {
    let (write_buf, write_mem, write_ptr) =
        host_buf(ctx, SSBO_SIZE, vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let (dirty_buf, dirty_mem, dirty_ptr_raw) =
        host_buf(ctx, DIRTY_SIZE, vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let dirty_ptr = dirty_ptr_raw as *mut u32;
    let (read_buf, read_mem) =
        device_buf(ctx, SSBO_SIZE,
            vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::TRANSFER_SRC)?;

    // Allocate descriptor sets
    let layouts = [*compose_dsl, *graphics_dsl];
    let ai = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(*pool)
        .set_layouts(&layouts);
    let sets = ctx.device.allocate_descriptor_sets(&ai).map_err(RenderError::Vulkan)?;
    let compute_ds = sets[0];
    let graphics_ds = sets[1];

    // Write compute DS: binding 0=write, 1=dirty, 2=read
    let wb = [vk::DescriptorBufferInfo::default().buffer(write_buf).range(vk::WHOLE_SIZE)];
    let db = [vk::DescriptorBufferInfo::default().buffer(dirty_buf).range(vk::WHOLE_SIZE)];
    let rb = [vk::DescriptorBufferInfo::default().buffer(read_buf).range(vk::WHOLE_SIZE)];
    let cw = [
        vk::WriteDescriptorSet::default().dst_set(compute_ds).dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&wb),
        vk::WriteDescriptorSet::default().dst_set(compute_ds).dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&db),
        vk::WriteDescriptorSet::default().dst_set(compute_ds).dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&rb),
    ];
    ctx.device.update_descriptor_sets(&cw, &[]);

    // Write graphics DS: binding 0=read, 1=atlas
    let rb2 = [vk::DescriptorBufferInfo::default().buffer(read_buf).range(vk::WHOLE_SIZE)];
    let ai2 = [vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(atlas_view).sampler(atlas_sampler)];
    let gw = [
        vk::WriteDescriptorSet::default().dst_set(graphics_ds).dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&rb2),
        vk::WriteDescriptorSet::default().dst_set(graphics_ds).dst_binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&ai2),
    ];
    ctx.device.update_descriptor_sets(&gw, &[]);

    Ok(FrameData {
        write_buf, write_mem, write_ptr: write_ptr as *mut u32,
        dirty_buf, dirty_mem, dirty_ptr,
        read_buf, read_mem,
        compute_ds, graphics_ds,
    })
}

/// Create the compose compute pipeline.
///
/// The compose shader has:
/// * 3 SSBO bindings (write, dirty, read)
/// * Push constants: (cols: u32, rows: u32)
/// * Workgroup size: 256
/// * Each invocation handles one row
unsafe fn create_compose_pipeline(ctx: &VulkanContext)
    -> Result<(vk::Pipeline, vk::PipelineLayout, vk::DescriptorSetLayout), RenderError>
{
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];
    let dsl_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    let dsl = ctx.device.create_descriptor_set_layout(&dsl_ci, None)
        .map_err(RenderError::Vulkan)?;

    let pc = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0).size(8);
    let pc_ranges = [pc];
    let dsls = [dsl];
    let pl_ci = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&dsls).push_constant_ranges(&pc_ranges);
    let layout = ctx.device.create_pipeline_layout(&pl_ci, None)
        .map_err(RenderError::Vulkan)?;

    let code = pipeline::load_spirv("shaders/compiled/compose_comp.spv")?;
    let module = pipeline::create_shader_module(&ctx.device, &code)?;

    let entry = c"main";
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module).name(entry);

    let ci = vk::ComputePipelineCreateInfo::default()
        .stage(stage).layout(layout);
    let pipe = ctx.device.create_compute_pipelines(
        vk::PipelineCache::null(), &[ci], None,
    ).map_err(|(_p, e)| RenderError::Vulkan(e))?[0];

    ctx.device.destroy_shader_module(module, None);

    Ok((pipe, layout, dsl))
}