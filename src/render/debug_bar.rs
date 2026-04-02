//! # Vulkan debug bar overlay
//!
//! A tiny SSBO + descriptor set for rendering one row of debug text
//! as a second draw call inside the existing render pass. Uses the
//! same [`TextPipeline`](super::pipeline::TextPipeline) and glyph
//! atlas as the main draw.
//!
//! ## Visual layout
//!
//! The debug bar occupies the top 16 pixels of the screen (one cell
//! row of 8x16 pixel characters). A scissor rect restricts fragments
//! to that strip; the fullscreen triangle plus the text fragment
//! shader handle the rest.
//!
//! ```text
//! +------------------------------------------+
//! | Software | 120 fps | 0.83ms | GPU: ...   |  <- debug bar (16px tall)
//! +------------------------------------------+
//! | Normal screen content below              |
//! |                                          |
//! | ...                                      |
//! +------------------------------------------+
//! ```
//!
//! ## Implementation
//!
//! The debug bar has its own:
//! * HOST_VISIBLE SSBO (persistently mapped, max 512 columns * 12 bytes)
//! * Descriptor set (SSBO + atlas sampler, same layout as the main set)
//!
//! On each frame (when F9 is active), the overlay:
//! 1. Writes cell data (codepoint, fg, bg) into the mapped SSBO.
//! 2. Sets a scissor rect to the top 16 pixels.
//! 3. Pushes modified push constants (1 row, cols = screen_w / 8).
//! 4. Binds its own descriptor set.
//! 5. Issues `cmd_draw(3, 1, 0, 0)` (fullscreen triangle, clipped).
//!
//! ## Colour scheme
//!
//! The debug bar uses Catppuccin Mocha-inspired colours:
//! * Background: `0x11111B` (near-black)
//! * Foreground: `0xA6E3A1` (green)

use ash::vk;
use super::RenderError;
use super::vulkan_ctx::VulkanContext;
use super::pipeline::TextPushConstants;
use super::indirect::find_memory_type;

/// Maximum debug bar width in columns.
///
/// 512 columns * 8 pixels = 4096 pixels, enough for 4K displays.
pub const OVERLAY_MAX_COLS: u32 = 512;

/// SSBO byte size: `MAX_COLS * 3 * sizeof(u32)`.
///
/// Each cell is 3 u32s (codepoint, fg, bg) = 12 bytes.
/// 512 * 12 = 6144 bytes.
const OVERLAY_SSBO_SIZE: u64 = (OVERLAY_MAX_COLS as u64) * 12;

/// Height of the debug bar in pixels (one cell row).
pub const BAR_HEIGHT_PX: u32 = 16;

/// Vulkan resources for the debug bar overlay.
///
/// Created once during renderer initialisation. Updated every frame
/// when F9 is active.
pub struct VkDebugBar {
    /// The SSBO holding cell data for the debug bar.
    ssbo: vk::Buffer,

    /// Device memory backing the SSBO.
    memory: vk::DeviceMemory,

    /// Persistently mapped pointer to the SSBO data.
    ///
    /// Points to `OVERLAY_MAX_COLS * 3` u32 values.
    /// Written to directly every frame (HOST_COHERENT, no flush needed).
    ptr: *mut u32,

    /// Descriptor set for this overlay's draw call.
    ///
    /// Binds the overlay SSBO at binding 0 and the shared atlas
    /// sampler at binding 1.
    pub descriptor_set: vk::DescriptorSet,
}

impl VkDebugBar {
    /// Allocate the SSBO and descriptor set.
    ///
    /// # Safety
    ///
    /// * `pool` must have room for one more descriptor set.
    /// * `dsl` must be the same layout used by the main text pipeline.
    /// * `atlas_view` and `atlas_sampler` must be valid.
    /// * `ctx` must be fully initialised.
    pub unsafe fn new(
        ctx: &VulkanContext,
        pool: vk::DescriptorPool,
        dsl: vk::DescriptorSetLayout,
        atlas_view: vk::ImageView,
        atlas_sampler: vk::Sampler,
    ) -> Result<Self, RenderError> {
        let device = &ctx.device;
        log::debug!("Creating debug bar overlay (SSBO={} bytes, max {} cols)",
            OVERLAY_SSBO_SIZE, OVERLAY_MAX_COLS);
        // -- HOST_VISIBLE SSBO, persistently mapped --
        let buf_ci = vk::BufferCreateInfo::default()
            .size(OVERLAY_SSBO_SIZE)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let ssbo = device.create_buffer(&buf_ci, None).map_err(RenderError::Vulkan)?;

        let req = device.get_buffer_memory_requirements(ssbo);
        let mi = find_memory_type(
            &ctx.instance, ctx.physical_device, req.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let ai = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(mi);
        let memory = device.allocate_memory(&ai, None).map_err(RenderError::Vulkan)?;
        device.bind_buffer_memory(ssbo, memory, 0).map_err(RenderError::Vulkan)?;

        let ptr = device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            .map_err(RenderError::Vulkan)? as *mut u32;
        std::ptr::write_bytes(ptr as *mut u8, 0, OVERLAY_SSBO_SIZE as usize);

        // -- Descriptor set --
        let layouts = [dsl];
        let ds_ai = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        let descriptor_set = device.allocate_descriptor_sets(&ds_ai)
            .map_err(RenderError::Vulkan)?[0];

        let buf_info = [vk::DescriptorBufferInfo::default()
            .buffer(ssbo).offset(0).range(vk::WHOLE_SIZE)];
        let img_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(atlas_view)
            .sampler(atlas_sampler)];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set).dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buf_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set).dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&img_info),
        ];
        device.update_descriptor_sets(&writes, &[]);
        
        log::debug!("Debug bar overlay created");
        Ok(Self { ssbo, memory, ptr, descriptor_set })
    }

    /// Upload cells and issue a second draw call for the debug bar.
    ///
    /// # Arguments
    ///
    /// * `device` - The Vulkan device.
    /// * `cmd` - The command buffer (must be between begin/end render pass).
    /// * `pipeline_layout` - The text pipeline layout (for push constants).
    /// * `cells` - Array of `[codepoint, fg_rgb, bg_rgb]` triples.
    ///   Length should be <= `OVERLAY_MAX_COLS`.
    /// * `screen_w` - Current window width in pixels.
    ///
    /// # Safety
    ///
    /// Must be called between `cmd_begin_render_pass` and
    /// `cmd_end_render_pass`. The text pipeline must already be bound.
    pub unsafe fn draw(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        pipeline_layout: vk::PipelineLayout,
        cells: &[[u32; 3]],
        screen_w: u32,
    ) {
        let cols = (screen_w / 8).min(OVERLAY_MAX_COLS);
        let count = cells.len().min(cols as usize);

        // Upload cell data into the persistently mapped SSBO
        for (i, cell) in cells[..count].iter().enumerate() {
            *self.ptr.add(i * 3)     = cell[0]; // codepoint
            *self.ptr.add(i * 3 + 1) = cell[1]; // fg
            *self.ptr.add(i * 3 + 2) = cell[2]; // bg
        }
        // Pad remainder with background-colored spaces
        let pad_bg = if count > 0 { cells[0][2] } else { 0x0C0C0C };
        for i in count..(cols as usize) {
            *self.ptr.add(i * 3)     = ' ' as u32;
            *self.ptr.add(i * 3 + 1) = 0x00CC66;
            *self.ptr.add(i * 3 + 2) = pad_bg;
        }

        // Scissor: restrict to the top strip only
        device.cmd_set_scissor(cmd, 0, &[vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width: screen_w, height: BAR_HEIGHT_PX },
        }]);

        // Push constants for a 1-row grid
        let pc = TextPushConstants {
            resolution: [cols, 1],
            cell_pixels: [8, BAR_HEIGHT_PX],
            atlas_cells: [256, 256],
            screen_pixels: [screen_w, BAR_HEIGHT_PX],
        };
        let pc_bytes: &[u8] = std::slice::from_raw_parts(
            &pc as *const _ as *const u8,
            std::mem::size_of::<TextPushConstants>(),
        );
        device.cmd_push_constants(
            cmd, pipeline_layout,
            vk::ShaderStageFlags::FRAGMENT, 0, pc_bytes,
        );

        // Bind the overlay's descriptor set
        device.cmd_bind_descriptor_sets(
            cmd, vk::PipelineBindPoint::GRAPHICS,
            pipeline_layout, 0, &[self.descriptor_set], &[],
        );

        // Draw fullscreen triangle (scissor clips to the bar area)
        device.cmd_draw(cmd, 3, 1, 0, 0);
    }

    /// Destroy all owned Vulkan resources.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        device.unmap_memory(self.memory);
        device.destroy_buffer(self.ssbo, None);
        device.free_memory(self.memory, None);
    }
}