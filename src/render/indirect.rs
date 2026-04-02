//! # INDIRECT Vulkan renderer
//!
//! Each tick the [`TextBuffer`] cell data is copied into a host-visible
//! Vulkan SSBO. A fullscreen fragment shader reads the SSBO + glyph
//! atlas texture to produce the final image.
//!
//! ## Pipeline overview
//!
//! ```text
//! Per frame (when buffer is dirty):
//!
//! 1. upload_cells()
//!    |
//!    | For each cell in the TextBuffer:
//!    |   resolve foreground/background through palette
//!    |   write (codepoint, fg_rgb, bg_rgb) as 3x u32
//!    |   into the persistently mapped SSBO
//!    v
//! 2. render_frame()
//!    |
//!    | a. Wait for previous frame's fence
//!    | b. Acquire swapchain image
//!    | c. Reset + begin command buffer
//!    | d. Begin render pass (clear to black)
//!    | e. Bind graphics pipeline (text.vert + text.frag)
//!    | f. Set dynamic viewport + scissor
//!    | g. Push constants (resolution, cell size, atlas grid, screen size)
//!    | h. Bind descriptor set (SSBO + atlas sampler)
//!    | i. cmd_draw(3, 1, 0, 0)  -- fullscreen triangle
//!    | j. (optional) Debug bar overlay draw
//!    | k. End render pass
//!    | l. End command buffer
//!    | m. Submit to graphics queue
//!    | n. Present to swapchain
//!    v
//! 3. Advance frame index (double-buffered: 0 -> 1 -> 0 -> ...)
//! ```
//!
//! ## Resource ownership
//!
//! `IndirectRenderer` owns **all** Vulkan resources it creates:
//!
//! * Cell SSBO (buffer + memory, persistently mapped)
//! * Atlas image (device-local, uploaded once via staging buffer)
//! * Atlas image view
//! * Atlas sampler (nearest-neighbour)
//! * Descriptor pool (containing the main set + debug bar set)
//! * Descriptor set (SSBO + atlas)
//! * Text pipeline (vertex/fragment shaders, pipeline layout, DSL)
//! * Debug bar overlay resources
//!
//! The shared [`VulkanContext`] is borrowed mutably only during
//! [`render_frame`] (for command buffer recording and queue submission).
//!
//! ## Memory strategy
//!
//! * **Cell SSBO**: `HOST_VISIBLE | HOST_COHERENT`, persistently mapped.
//!   Updated via pointer writes every dirty frame -- no staging buffer
//!   needed. Size: `MAX_CELLS * 12` bytes = 96 KiB for 160x50.
//!
//! * **Atlas image**: `DEVICE_LOCAL`, uploaded once via a transient
//!   staging buffer that is destroyed immediately after the copy.
//!   Size: 16 MiB (4096x4096 R8).
//!
//! ## Double buffering
//!
//! The INDIRECT renderer uses a single SSBO (not double-buffered)
//! because the upload is synchronised by the frame fence: we wait
//! for the previous frame to complete before writing new data.
//! This is simpler than the DIRECT renderer's double-buffered approach
//! but adds a frame of latency.

use ash::vk;
use std::ffi::c_void;

use crate::display::TextBuffer;
use crate::display::font::ATLAS_SIZE;
use super::RenderError;
use super::vulkan_ctx::VulkanContext;
use super::pipeline::{TextPipeline, TextPushConstants};

/// Maximum number of cells we pre-allocate the SSBO for.
///
/// 160 columns * 50 rows = 8,000 cells.
/// 8,000 * 12 bytes per cell = 96 KiB -- trivially small for any GPU.
///
/// If the actual text grid is smaller, the excess SSBO space is
/// simply unused.
const MAX_CELLS: usize = 160 * 50;

/// Bytes per cell in the SSBO.
///
/// Layout: `[codepoint: u32, foreground_rgb: u32, background_rgb: u32]`
/// = 3 * 4 = 12 bytes.
///
/// This matches the `cells[]` array in `text.frag`:
/// ```glsl
/// layout(std430, set = 0, binding = 0) readonly buffer CellData {
///     uint cells[];
/// };
/// // cells[idx * 3 + 0] = codepoint
/// // cells[idx * 3 + 1] = fg_packed
/// // cells[idx * 3 + 2] = bg_packed
/// ```
const CELL_STRIDE: usize = 3 * std::mem::size_of::<u32>();

/// INDIRECT rendering backend.
///
/// Owns all Vulkan resources for the INDIRECT rendering path.
/// Created once during initialisation and destroyed when the renderer
/// is dropped or the render mode is changed.
///
/// # Usage
///
/// ```text
/// let mut renderer = IndirectRenderer::new(&mut ctx, &atlas.pixels)?;
///
/// // Each frame:
/// renderer.render_frame(&mut ctx, &buffer, debug_cells)?;
///
/// // Cleanup:
/// renderer.destroy(&ctx.device);
/// ```
pub struct IndirectRenderer {
    // -- Pipeline --

    /// The shared text rendering pipeline (vertex + fragment shaders,
    /// pipeline layout, descriptor set layout).
    pipeline: TextPipeline,

    // -- Cell SSBO (host-visible, persistently mapped) --

    /// Vulkan buffer handle for the cell data SSBO.
    cell_buffer: vk::Buffer,

    /// Device memory backing the cell SSBO.
    cell_memory: vk::DeviceMemory,

    /// Persistently mapped pointer to the SSBO data.
    ///
    /// Points to `MAX_CELLS * 3` u32 values (= `MAX_CELLS * 12` bytes).
    /// Written to by [`upload_cells`] every dirty frame.
    ///
    /// Because the memory is `HOST_COHERENT`, no explicit flush is needed
    /// after writes -- the GPU sees the data immediately.
    cell_ptr: *mut u8,

    // -- Glyph atlas (device-local image) --

    /// The glyph atlas image (4096x4096, R8_UNORM, DEVICE_LOCAL).
    atlas_image: vk::Image,

    /// Device memory backing the atlas image.
    atlas_memory: vk::DeviceMemory,

    /// Image view for the atlas (TYPE_2D, R8_UNORM).
    atlas_view: vk::ImageView,

    /// Sampler for the atlas (nearest-neighbour, for pixel-perfect glyphs).
    atlas_sampler: vk::Sampler,

    // -- Descriptors --

    /// Descriptor pool from which the main set and debug bar set are
    /// allocated.
    descriptor_pool: vk::DescriptorPool,

    /// Descriptor set binding the cell SSBO (binding 0) and the atlas
    /// sampler (binding 1).
    descriptor_set: vk::DescriptorSet,

    // -- Current text grid size --

    /// Cached resolution from the last `upload_cells` call.
    ///
    /// Used to generate correct push constants. Updated every time
    /// `upload_cells` or `render_frame` is called.
    resolution: (u32, u32),

    /// Optional debug bar overlay (F9).
    debug_bar: Option<super::debug_bar::VkDebugBar>,
}

impl IndirectRenderer {
    /// Create the INDIRECT renderer, uploading the glyph atlas.
    ///
    /// # What happens
    ///
    /// 1. Create the text rendering pipeline.
    /// 2. Allocate + map the cell SSBO (HOST_VISIBLE | HOST_COHERENT).
    /// 3. Upload the glyph atlas to a DEVICE_LOCAL image via staging.
    /// 4. Create the descriptor pool (2 SSBOs + 2 samplers for main + overlay).
    /// 5. Allocate and write the main descriptor set.
    /// 6. Create the debug bar overlay resources (if possible).
    ///
    /// # Safety
    ///
    /// `ctx` must be a valid, fully initialised [`VulkanContext`].
    /// `atlas_pixels` must contain at least `ATLAS_SIZE * ATLAS_SIZE` bytes.
    ///
    /// # Errors
    ///
    /// Returns `RenderError` on any Vulkan API failure.
    pub unsafe fn new(
        ctx: &mut VulkanContext,
        atlas_pixels: &[u8],
    ) -> Result<Self, RenderError> {
        log::info!("Creating INDIRECT renderer...");
        let pipeline = TextPipeline::new(ctx)?;

        // -- Cell SSBO ------------------------------------------------
        let ssbo_size = (MAX_CELLS * CELL_STRIDE) as u64;
        log::debug!("Cell SSBO: {} bytes (max {} cells)", ssbo_size, MAX_CELLS);

        let buf_ci = vk::BufferCreateInfo::default()
            .size(ssbo_size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let cell_buffer = ctx.device.create_buffer(&buf_ci, None)
            .map_err(RenderError::Vulkan)?;

        let mem_req = ctx.device.get_buffer_memory_requirements(cell_buffer);
        let mem_idx = find_memory_type(
            &ctx.instance, ctx.physical_device,
            mem_req.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let alloc_ci = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_req.size)
            .memory_type_index(mem_idx);
        let cell_memory = ctx.device.allocate_memory(&alloc_ci, None)
            .map_err(RenderError::Vulkan)?;
        ctx.device.bind_buffer_memory(cell_buffer, cell_memory, 0)
            .map_err(RenderError::Vulkan)?;

        let cell_ptr = ctx.device.map_memory(
            cell_memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty(),
        ).map_err(RenderError::Vulkan)? as *mut u8;
        log::debug!("Cell SSBO: allocated, HOST_VISIBLE|HOST_COHERENT, mem_type={}",
            mem_idx);
        // Zero-initialise (all cells = codepoint 0, fg 0, bg 0).
        std::ptr::write_bytes(cell_ptr, 0, ssbo_size as usize);

        // -- Atlas image ----------------------------------------------
        log::debug!("Uploading glyph atlas ({}x{}, {} bytes)...",
            ATLAS_SIZE, ATLAS_SIZE, atlas_pixels.len());
        let (atlas_image, atlas_memory, atlas_view, atlas_sampler) =
            upload_atlas(ctx, atlas_pixels)?;
        log::debug!("Glyph atlas uploaded to DEVICE_LOCAL image");

        log::debug!("Creating descriptor pool (2 SSBO + 2 sampler, 2 sets max)");
        // -- Descriptor pool + set ------------------------------------
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 2,  // main + overlay
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 2,  // main + overlay
            },
        ];
        let pool_ci = vk::DescriptorPoolCreateInfo::default()
            .max_sets(2)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = ctx.device.create_descriptor_pool(&pool_ci, None)
            .map_err(RenderError::Vulkan)?;

        let layouts = [pipeline.descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);
        let descriptor_set = ctx.device.allocate_descriptor_sets(&alloc_info)
            .map_err(RenderError::Vulkan)?[0];

        let debug_bar = super::debug_bar::VkDebugBar::new(
            ctx, descriptor_pool,
            pipeline.descriptor_set_layout,
            atlas_view, atlas_sampler,
        ).ok();

        // Write descriptors (SSBO + atlas sampler).
        let buf_info = [vk::DescriptorBufferInfo::default()
            .buffer(cell_buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];

        let img_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(atlas_view)
            .sampler(atlas_sampler)];

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buf_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&img_info),
        ];
        ctx.device.update_descriptor_sets(&writes, &[]);
        log::info!("INDIRECT renderer created successfully");
        Ok(Self {
            pipeline,
            cell_buffer, cell_memory, cell_ptr,
            atlas_image, atlas_memory, atlas_view, atlas_sampler,
            descriptor_pool, descriptor_set,
            resolution: (80, 25), debug_bar,
        })
    }

    /// Upload cell data from the [`TextBuffer`] into the mapped SSBO.
    ///
    /// For each cell, resolves the foreground and background colors
    /// through the palette and writes three u32 values (codepoint,
    /// fg_rgb, bg_rgb) into the mapped memory.
    ///
    /// # Performance
    ///
    /// At maximum resolution (160x50 = 8000 cells), this writes
    /// 96 KiB of data. On modern hardware with write-combining memory,
    /// this takes < 0.1 ms.
    ///
    /// # Side effects
    ///
    /// Updates `self.resolution` to match the buffer dimensions.
    pub(crate) fn upload_cells(&mut self, buffer: &TextBuffer) {
        self.resolution = (buffer.width(), buffer.height());
        let cells = buffer.cells();
        let count = cells.len().min(MAX_CELLS);

        // Build packed u32 triples directly into mapped memory.
        let dst = self.cell_ptr as *mut u32;
        for (i, cell) in cells[..count].iter().enumerate() {
            unsafe {
                *dst.add(i * 3)     = cell.codepoint;
                *dst.add(i * 3 + 1) = cell.foreground.resolve(buffer.palette());
                *dst.add(i * 3 + 2) = cell.background.resolve(buffer.palette());
            }
        }
    }

/// Record commands and submit a frame.
    ///
    /// If the buffer is dirty, calls [`upload_cells`] first to update
    /// the SSBO. Then records a command buffer with:
    ///
    /// 1. Render pass (clear to black)
    /// 2. Pipeline bind + dynamic state (viewport, scissor)
    /// 3. Push constants + descriptor set bind
    /// 4. Fullscreen triangle draw (3 vertices, no VBO)
    /// 5. Optional debug bar overlay draw
    /// 6. Render pass end
    ///
    /// Then submits the command buffer and presents the swapchain image.
    ///
    /// ## Semaphore strategy
    ///
    /// `render_finished` semaphores are indexed by **swapchain image**
    /// (not by frame-in-flight), preventing the validation error where
    /// a semaphore is reused before the presentation engine releases it.
    /// An `images_in_flight` fence table ensures we never write to a
    /// swapchain image that another frame-in-flight is still rendering to.
    ///
    /// # Arguments
    ///
    /// * `ctx` — Mutable reference to the Vulkan context.
    /// * `buffer` — The text buffer to render.
    /// * `debug_cells` — Optional debug bar cell data (from F9 overlay).
    ///
    /// # Errors
    ///
    /// * `RenderError::SwapchainOutOfDate` — Window was resized; caller
    ///   should recreate the swapchain and retry.
    /// * `RenderError::Vulkan(e)` — Any other Vulkan error.
    pub fn render_frame(
        &mut self,
        ctx: &mut VulkanContext,
        buffer: &TextBuffer,
        debug_cells: Option<&[[u32; 3]]>,
    ) -> Result<(), RenderError> {
        if buffer.is_dirty() {
            self.upload_cells(buffer);
        }
        self.upload_cells(buffer);

        let device = &ctx.device;
        let frame = ctx.current_frame;

        unsafe {
            // Wait for this frame-in-flight slot to finish its previous work.
            {
                let _prof = crate::profiler::scope(
                    crate::profiler::Cat::VkFence, "wait_fence");
                device.wait_for_fences(&[ctx.in_flight[frame]], true, u64::MAX)
                    .map_err(RenderError::Vulkan)?;
            }

            // Acquire the next swapchain image.
            let image_index;
            {
                let _prof = crate::profiler::scope(
                    crate::profiler::Cat::VkAcquire, "acquire_image");
                let (idx, _) = ctx.swapchain_loader
                    .acquire_next_image(
                        ctx.swapchain, u64::MAX,
                        ctx.image_available[frame], vk::Fence::null(),
                    )
                    .map_err(|e| match e {
                        vk::Result::ERROR_OUT_OF_DATE_KHR => RenderError::SwapchainOutOfDate,
                        other => RenderError::Vulkan(other),
                    })?;
                image_index = idx;
            }

            let img = image_index as usize;

            // If another frame-in-flight is still using this swapchain
            // image, wait for it to complete before we overwrite it.
            if ctx.images_in_flight[img] != vk::Fence::null() {
                device.wait_for_fences(&[ctx.images_in_flight[img]], true, u64::MAX)
                    .map_err(RenderError::Vulkan)?;
            }
            // Mark this swapchain image as owned by our frame's fence.
            ctx.images_in_flight[img] = ctx.in_flight[frame];

            device.reset_fences(&[ctx.in_flight[frame]]).unwrap();

            let cmd = ctx.command_buffers[frame];
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();

            // -- Render pass --
            let clear = vk::ClearValue {
                color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] },
            };
            let clears = [clear];
            let rp_info = vk::RenderPassBeginInfo::default()
                .render_pass(ctx.render_pass)
                .framebuffer(ctx.framebuffers[img])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent: ctx.swapchain_extent,
                })
                .clear_values(&clears);

            device.cmd_begin_render_pass(cmd, &rp_info, vk::SubpassContents::INLINE);
            device.cmd_bind_pipeline(
                cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline.pipeline);

            // Dynamic viewport + scissor (pipeline was created with these as dynamic state).
            let viewport = vk::Viewport {
                x: 0.0, y: 0.0,
                width: ctx.swapchain_extent.width as f32,
                height: ctx.swapchain_extent.height as f32,
                min_depth: 0.0, max_depth: 1.0,
            };
            device.cmd_set_viewport(cmd, 0, &[viewport]);
            device.cmd_set_scissor(cmd, 0, &[vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: ctx.swapchain_extent,
            }]);

            // Push constants tell the fragment shader the grid dimensions,
            // cell pixel size, atlas layout, and actual screen size.
            let pc = TextPushConstants {
                resolution: [self.resolution.0, self.resolution.1],
                cell_pixels: [8, 16],
                atlas_cells: [256, 256],
                screen_pixels: [ctx.swapchain_extent.width, ctx.swapchain_extent.height],
            };
            let pc_bytes: &[u8] = std::slice::from_raw_parts(
                &pc as *const _ as *const u8,
                std::mem::size_of::<TextPushConstants>(),
            );
            device.cmd_push_constants(
                cmd, self.pipeline.pipeline_layout,
                vk::ShaderStageFlags::FRAGMENT, 0, pc_bytes,
            );

            // Bind the descriptor set containing the cell SSBO and atlas sampler.
            device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline_layout,
                0, &[self.descriptor_set], &[],
            );

            // Fullscreen triangle: 3 vertices, no vertex buffer.
            // The vertex shader generates positions from gl_VertexIndex.
            device.cmd_draw(cmd, 3, 1, 0, 0);

            // Optional debug bar overlay (second draw call in the same render pass).
            if let (Some(bar), Some(cells)) = (&self.debug_bar, debug_cells) {
                bar.draw(
                    device, cmd,
                    self.pipeline.pipeline_layout,
                    cells,
                    ctx.swapchain_extent.width,
                );
            }

            device.cmd_end_render_pass(cmd);
            device.end_command_buffer(cmd).unwrap();

            // -- Submit --
            // Wait on image_available[frame] (signalled by acquire).
            // Signal render_finished[img] (indexed by swapchain image, not frame).
            {
                let _prof = crate::profiler::scope(
                    crate::profiler::Cat::VkSubmit, "queue_submit");
                let wait_sems = [ctx.image_available[frame]];
                let signal_sems = [ctx.render_finished[img]];
                let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
                let cmd_bufs = [cmd];
                let submit = vk::SubmitInfo::default()
                    .wait_semaphores(&wait_sems)
                    .wait_dst_stage_mask(&wait_stages)
                    .command_buffers(&cmd_bufs)
                    .signal_semaphores(&signal_sems);

                device.queue_submit(
                    ctx.graphics_queue, &[submit], ctx.in_flight[frame])
                    .map_err(RenderError::Vulkan)?;
            }

            // -- Present --
            // Wait on render_finished[img] so the image is fully rendered
            // before the presentation engine displays it.
            {
                let _prof = crate::profiler::scope(
                    crate::profiler::Cat::VkPresent, "queue_present");
                let swapchains = [ctx.swapchain];
                let indices = [image_index];
                let signal_sems = [ctx.render_finished[img]];
                let present = vk::PresentInfoKHR::default()
                    .wait_semaphores(&signal_sems)
                    .swapchains(&swapchains)
                    .image_indices(&indices);

                ctx.swapchain_loader.queue_present(ctx.present_queue, &present)
                    .map_err(|e| match e {
                        vk::Result::ERROR_OUT_OF_DATE_KHR => RenderError::SwapchainOutOfDate,
                        other => RenderError::Vulkan(other),
                    })?;
            }

            ctx.current_frame = (frame + 1) % ctx.max_frames_in_flight;
        }
        Ok(())
    }
    
    /// Destroy all owned Vulkan resources.
    ///
    /// Waits for the device to be idle before destroying anything.
    /// Must be called before the `VulkanContext` is dropped.
    pub fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            let _ = device.device_wait_idle();
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_sampler(self.atlas_sampler, None);
            device.destroy_image_view(self.atlas_view, None);
            device.destroy_image(self.atlas_image, None);
            device.free_memory(self.atlas_memory, None);
            device.unmap_memory(self.cell_memory);
            device.destroy_buffer(self.cell_buffer, None);
            device.free_memory(self.cell_memory, None);
            if let Some(bar) = &self.debug_bar { bar.destroy(device); }
            self.pipeline.destroy(device);
        }
    }
}

// -----------------------------------------------------------------------
// Atlas upload (staging buffer -> device-local image)
// -----------------------------------------------------------------------

/// Create the glyph atlas image, upload pixels, and return the
/// `(image, memory, view, sampler)` tuple.
///
/// ## Upload process
///
/// ```text
/// 1. Create a HOST_VISIBLE staging buffer (16 MiB).
/// 2. memcpy atlas pixels into the staging buffer.
/// 3. Create a DEVICE_LOCAL image (4096x4096, R8_UNORM).
/// 4. Record a one-shot command buffer:
///    a. Transition image: UNDEFINED -> TRANSFER_DST_OPTIMAL
///    b. Copy staging buffer -> image
///    c. Transition image: TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
/// 5. Submit and wait.
/// 6. Destroy the staging buffer (no longer needed).
/// 7. Create an image view (TYPE_2D, R8_UNORM).
/// 8. Create a sampler (NEAREST filter, CLAMP_TO_EDGE).
/// ```
///
/// # Safety
///
/// `ctx` must be valid. `pixels` must contain at least
/// `ATLAS_SIZE * ATLAS_SIZE` bytes.
pub(crate) unsafe fn upload_atlas(
    ctx: &mut VulkanContext,
    pixels: &[u8],
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView, vk::Sampler), RenderError> {
    let device = &ctx.device;
    let size = (ATLAS_SIZE as u64) * (ATLAS_SIZE as u64);
    log::debug!("Atlas upload: staging buffer {} bytes", size);
    assert!(pixels.len() as u64 >= size, "atlas pixel data too small");

    // -- Staging buffer -----------------------------------------------
    let stg_ci = vk::BufferCreateInfo::default()
        .size(size)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let stg_buf = device.create_buffer(&stg_ci, None).map_err(RenderError::Vulkan)?;

    let stg_req = device.get_buffer_memory_requirements(stg_buf);
    let stg_mem_idx = find_memory_type(
        &ctx.instance, ctx.physical_device, stg_req.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let stg_alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(stg_req.size)
        .memory_type_index(stg_mem_idx);
    let stg_mem = device.allocate_memory(&stg_alloc, None).map_err(RenderError::Vulkan)?;
    device.bind_buffer_memory(stg_buf, stg_mem, 0).map_err(RenderError::Vulkan)?;

    let ptr = device.map_memory(stg_mem, 0, size, vk::MemoryMapFlags::empty())
        .map_err(RenderError::Vulkan)? as *mut u8;
    std::ptr::copy_nonoverlapping(pixels.as_ptr(), ptr, size as usize);
    device.unmap_memory(stg_mem);
    log::debug!("Atlas upload: staging buffer allocated (mem_type={})", stg_mem_idx);

    // -- Image --------------------------------------------------------
    let img_ci = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::R8_UNORM)
        .extent(vk::Extent3D { width: ATLAS_SIZE, height: ATLAS_SIZE, depth: 1 })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = device.create_image(&img_ci, None).map_err(RenderError::Vulkan)?;

    let img_req = device.get_image_memory_requirements(image);
    let img_mem_idx = find_memory_type(
        &ctx.instance, ctx.physical_device, img_req.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let img_alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(img_req.size)
        .memory_type_index(img_mem_idx);
    let img_mem = device.allocate_memory(&img_alloc, None).map_err(RenderError::Vulkan)?;
    device.bind_image_memory(image, img_mem, 0).map_err(RenderError::Vulkan)?;
    log::debug!("Atlas upload: image created {}x{} R8_UNORM (mem_type={})",
        ATLAS_SIZE, ATLAS_SIZE, img_mem_idx);

    // -- One-shot command buffer for layout transitions + copy ---------
    let alloc = vk::CommandBufferAllocateInfo::default()
        .command_pool(ctx.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cmd = device.allocate_command_buffers(&alloc).unwrap()[0];
    device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();
    log::debug!("Atlas upload: copy + transitions done, staging destroyed");

    // Transition UNDEFINED -> TRANSFER_DST_OPTIMAL.
    let barrier_to_dst = vk::ImageMemoryBarrier::default()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0, level_count: 1,
            base_array_layer: 0, layer_count: 1,
        })
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[], &[], &[barrier_to_dst],
    );

    // Copy staging buffer -> image.
    let region = vk::BufferImageCopy::default()
        .image_subresource(vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0, base_array_layer: 0, layer_count: 1,
        })
        .image_extent(vk::Extent3D { width: ATLAS_SIZE, height: ATLAS_SIZE, depth: 1 });
    device.cmd_copy_buffer_to_image(
        cmd, stg_buf, image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region],
    );

    // Transition TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL.
    let barrier_to_read = vk::ImageMemoryBarrier::default()
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0, level_count: 1,
            base_array_layer: 0, layer_count: 1,
        })
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::FRAGMENT_SHADER,
        vk::DependencyFlags::empty(),
        &[], &[], &[barrier_to_read],
    );

    device.end_command_buffer(cmd).unwrap();

    let cmds = [cmd];
    let submit = vk::SubmitInfo::default().command_buffers(&cmds);
    device.queue_submit(ctx.graphics_queue, &[submit], vk::Fence::null()).unwrap();
    device.queue_wait_idle(ctx.graphics_queue).unwrap();
    device.free_command_buffers(ctx.command_pool, &cmds);

    // Destroy staging buffer.
    device.destroy_buffer(stg_buf, None);
    device.free_memory(stg_mem, None);

    // -- Image view ---------------------------------------------------
    let view_ci = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(vk::Format::R8_UNORM)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0, level_count: 1,
            base_array_layer: 0, layer_count: 1,
        });
    let view = device.create_image_view(&view_ci, None).map_err(RenderError::Vulkan)?;

    // -- Sampler (nearest-neighbour for pixel-perfect glyphs) ---------
    let sampler_ci = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::NEAREST)
        .min_filter(vk::Filter::NEAREST)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE);
    let sampler = device.create_sampler(&sampler_ci, None).map_err(RenderError::Vulkan)?;
    log::debug!("Atlas: image view (R8_UNORM) + sampler (NEAREST) created");
    Ok((image, img_mem, view, sampler))
}

// -----------------------------------------------------------------------
// Memory type selection
// -----------------------------------------------------------------------

/// Find a Vulkan memory type matching the filter and required properties.
///
/// Iterates over all memory types reported by the physical device and
/// returns the index of the first one that:
/// 1. Is set in `type_filter` (a bitmask from `memoryRequirements.memoryTypeBits`).
/// 2. Has all bits in `properties` set in its `propertyFlags`.
///
/// # Arguments
///
/// * `instance` - The Vulkan instance (for querying physical device properties).
/// * `phys` - The physical device handle.
/// * `type_filter` - Bitmask of acceptable memory type indices.
/// * `properties` - Required memory property flags (e.g. `HOST_VISIBLE | HOST_COHERENT`).
///
/// # Errors
///
/// Returns `RenderError::NoSuitableMemory` if no memory type matches.
pub(crate) fn find_memory_type(
    instance: &ash::Instance, phys: vk::PhysicalDevice,
    type_filter: u32, properties: vk::MemoryPropertyFlags,
) -> Result<u32, RenderError> {
    let mem_props = unsafe { instance.get_physical_device_memory_properties(phys) };
    for i in 0..mem_props.memory_type_count {
        if (type_filter & (1 << i)) != 0
            && mem_props.memory_types[i as usize].property_flags.contains(properties)
        {
            log::trace!("Memory type {} selected for {:?}", i, properties);
            return Ok(i);
        }
    }
    log::error!("No suitable memory type for filter=0x{:X} props={:?}",
        type_filter, properties);
    Err(RenderError::NoSuitableMemory)
}