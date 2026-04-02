//! # Vulkan pipeline creation helpers
//!
//! This module provides:
//!
//! * [`TextPipeline`] -- The graphics pipeline used by both INDIRECT and
//!   DIRECT renderers. It consists of a fullscreen triangle vertex shader
//!   (`text.vert`) and a fragment shader (`text.frag`) that reads cell
//!   data from an SSBO and glyph bitmaps from a sampled texture.
//!
//! * [`TextPushConstants`] -- The push constant layout shared between the
//!   fragment shader and the Rust code. Must match the GLSL `layout`
//!   exactly.
//!
//! * [`load_spirv`] / [`create_shader_module`] -- Utilities for loading
//!   pre-compiled SPIR-V shader binaries from disk.
//!
//! ## Pipeline topology
//!
//! The pipeline uses a fullscreen triangle (3 vertices, no vertex buffer):
//!
//! ```text
//!     (-1, 3)
//!       *
//!      /|
//!     / |
//!    /  |
//!   /   |
//!  *----*
//! (-1,-1) (3,-1)
//! ```
//!
//! The vertex shader generates clip-space positions and UV coordinates
//! from `gl_VertexIndex` alone. The fragment shader does all the work:
//! it computes which text cell the current pixel belongs to, fetches
//! the cell data from the SSBO, looks up the glyph in the atlas, and
//! outputs the foreground or background color.
//!
//! ## Descriptor set layout
//!
//! ```text
//! Set 0:
//!   Binding 0: STORAGE_BUFFER  (cell data SSBO, read-only)
//!   Binding 1: COMBINED_IMAGE_SAMPLER (glyph atlas, nearest filter)
//! ```
//!
//! ## Push constants
//!
//! ```text
//! Offset  Size  Name            Description
//! 0       8     resolution      Grid size in cells [cols, rows]
//! 8       8     cell_pixels     Cell dimensions [8, 16]
//! 16      8     atlas_cells     Atlas grid [256, 256]
//! 24      8     screen_pixels   Window size in pixels [w, h]
//! ```
//!
//! Total: 32 bytes (well within the 128-byte minimum guarantee).

use ash::vk;
use std::path::Path;
use super::RenderError;
use super::vulkan_ctx::VulkanContext;

/// Push constants for the text rendering fragment shader.
///
/// Must match the GLSL `layout(push_constant)` block in `text.frag`
/// EXACTLY -- same field order, same sizes, same alignment.
///
/// `#[repr(C)]` ensures C-compatible layout with no Rust-specific
/// padding.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TextPushConstants {
    /// Text grid size in cells: `[columns, rows]`.
    ///
    /// For a standard T3 screen: `[160, 50]`.
    pub resolution: [u32; 2],

    /// Pixel size of each cell: `[width, height]`.
    ///
    /// Always `[8, 16]` for Unifont glyphs.
    pub cell_pixels: [u32; 2],

    /// Glyph atlas grid dimensions: `[columns, rows]`.
    ///
    /// Always `[256, 256]` (256 cells per axis in a 4096x4096 atlas).
    pub atlas_cells: [u32; 2],

    /// Actual window size in pixels: `[width, height]`.
    ///
    /// Used by the shader to determine which pixels are within the
    /// text area and which should be rendered as black border.
    pub screen_pixels: [u32; 2],
}

/// Graphics pipeline for text rendering.
///
/// Holds the descriptor set layout, pipeline layout, and pipeline handle.
/// Created once and shared by both INDIRECT and DIRECT renderers.
///
/// # Shaders
///
/// * `text_vert.spv` -- Fullscreen triangle vertex shader.
/// * `text_frag.spv` -- Fragment shader that reads SSBO + atlas.
///
/// Both are pre-compiled from GLSL to SPIR-V and loaded from
/// `shaders/compiled/`.
pub struct TextPipeline {
    /// Descriptor set layout defining the SSBO and sampler bindings.
    pub descriptor_set_layout: vk::DescriptorSetLayout,

    /// Pipeline layout with push constants and the descriptor set layout.
    pub pipeline_layout: vk::PipelineLayout,

    /// The compiled graphics pipeline handle.
    pub pipeline: vk::Pipeline,
}

impl TextPipeline {
    /// Create the text rendering pipeline.
    ///
    /// # What happens
    ///
    /// 1. Load vertex and fragment SPIR-V from disk.
    /// 2. Create shader modules.
    /// 3. Define descriptor set layout (1 SSBO + 1 sampler).
    /// 4. Define pipeline layout (descriptor set + push constants).
    /// 5. Configure all fixed-function state (no vertex input, triangle
    ///    list topology, no blending, dynamic viewport/scissor).
    /// 6. Create the graphics pipeline.
    /// 7. Destroy the shader modules (no longer needed).
    ///
    /// # Errors
    ///
    /// Returns `RenderError::ShaderNotFound` if the SPIR-V files are
    /// missing, or `RenderError::Vulkan` for any Vulkan API failure.
    pub fn new(ctx: &VulkanContext) -> Result<Self, RenderError> {
        let vert_code = load_spirv("shaders/compiled/text_vert.spv")?;
        let frag_code = load_spirv("shaders/compiled/text_frag.spv")?;

        let vert_module = create_shader_module(&ctx.device, &vert_code)?;
        let frag_module = create_shader_module(&ctx.device, &frag_code)?;

        let entry = c"main";
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(entry),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(entry),
        ];

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];

        let dsl_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout = unsafe {
            ctx.device.create_descriptor_set_layout(&dsl_ci, None)
                .map_err(RenderError::Vulkan)?
        };

        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<TextPushConstants>() as u32);

        let push_ranges = [push_range];
        let set_layouts = [descriptor_set_layout];
        let layout_ci = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_ranges);

        let pipeline_layout = unsafe {
            ctx.device.create_pipeline_layout(&layout_ci, None)
                .map_err(RenderError::Vulkan)?
        };

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewport = vk::Viewport {
            x: 0.0, y: 0.0,
            width: ctx.swapchain_extent.width as f32,
            height: ctx.swapchain_extent.height as f32,
            min_depth: 0.0, max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: ctx.swapchain_extent,
        };
        let viewports = [viewport];
        let scissors = [scissor];
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(&viewports)
            .scissors(&scissors);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::CLOCKWISE);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false);
        let blend_attachments = [blend_attachment];
        let blending = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(&blend_attachments);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&dynamic_states);

        let pipeline_ci = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&blending)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .render_pass(ctx.render_pass)
            .subpass(0);

        let pipeline_cis = [pipeline_ci];
        let pipeline = unsafe {
            ctx.device.create_graphics_pipelines(
                vk::PipelineCache::null(), &pipeline_cis, None
            ).map_err(|(_pipelines, err)| RenderError::Vulkan(err))?[0]
        };

        unsafe {
            ctx.device.destroy_shader_module(vert_module, None);
            ctx.device.destroy_shader_module(frag_module, None);
        }
        log::info!("Text pipeline created (DSL, layout, pipeline)");
        log::debug!("  Push constants: {} bytes", std::mem::size_of::<TextPushConstants>());
        log::debug!("  Bindings: SSBO(0) + Sampler(1)");

        Ok(Self { descriptor_set_layout, pipeline_layout, pipeline })
    }

    /// Destroy all owned Vulkan resources.
    ///
    /// Must be called before the `VulkanContext` is dropped.
    pub fn destroy(&self, device: &ash::Device) {
        log::debug!("Destroying text pipeline resources");
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

/// Load a pre-compiled SPIR-V shader from disk.
///
/// # Arguments
///
/// * `path` - Path to the `.spv` file (relative to the working directory).
///
/// # Returns
///
/// A `Vec<u32>` containing the SPIR-V words (little-endian).
///
/// # Errors
///
/// * `RenderError::ShaderNotFound` - File does not exist or cannot be read.
/// * `RenderError::Other` - File size is not a multiple of 4 bytes
///   (invalid SPIR-V).
pub(crate) fn load_spirv(path: &str) -> Result<Vec<u32>, RenderError> {
    log::debug!("Loading SPIR-V: {path}");
    let bytes = std::fs::read(Path::new(path))
        .map_err(|_| {
            log::error!("Shader not found: {path}");
            RenderError::ShaderNotFound(Box::leak(path.to_string().into_boxed_str()))
        })?;
    if bytes.len() % 4 != 0 {
        log::error!("SPIR-V not 4-byte aligned: {path} ({} bytes)", bytes.len());
        return Err(RenderError::Other("SPIR-V not aligned".into()));
    }
    log::debug!("  {path}: {} bytes ({} words)", bytes.len(), bytes.len() / 4);
    Ok(bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}


/// Create a Vulkan shader module from SPIR-V words.
///
/// # Arguments
///
/// * `device` - The Vulkan logical device.
/// * `code` - SPIR-V words (from [`load_spirv`]).
///
/// # Returns
///
/// A `vk::ShaderModule` handle. The caller is responsible for
/// destroying it after pipeline creation.
pub(crate) fn create_shader_module(device: &ash::Device, code: &[u32]) -> Result<vk::ShaderModule, RenderError> {
    let ci = vk::ShaderModuleCreateInfo::default().code(code);
    unsafe { device.create_shader_module(&ci, None).map_err(RenderError::Vulkan) }
}