//! # Vulkan pipeline creation helpers
//!
//! Loads SPIR-V shaders and creates the graphics pipeline for text rendering.

use ash::vk;
use std::path::Path;

use super::RenderError;
use super::vulkan_ctx::VulkanContext;

/// Push constant layout matching the fragment shader.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TextPushConstants {
    /// Text grid size in cells (e.g. 160, 50).
    pub resolution: [u32; 2],
    /// Pixel size of one cell (e.g. 8, 16).
    pub cell_pixels: [u32; 2],
    /// Glyph atlas grid dimensions (e.g. 256, 256).
    pub atlas_cells: [u32; 2],
}

/// All pipeline objects for the text renderer.
pub struct TextPipeline {
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
}

impl TextPipeline {
    /// Create the text rendering pipeline.
    ///
    /// Expects compiled SPIR-V at `shaders/compiled/text_vert.spv` and
    /// `shaders/compiled/text_frag.spv`.
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

        // ── Descriptor set layout ───────────────────────────────────
        // Binding 0: SSBO (cell data)
        // Binding 1: Combined image sampler (glyph atlas)
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

        let dsl_ci = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings);
        let descriptor_set_layout = unsafe {
            ctx.device.create_descriptor_set_layout(&dsl_ci, None)
                .map_err(RenderError::Vulkan)?
        };

        // ── Pipeline layout ─────────────────────────────────────────
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

        // ── Pipeline ────────────────────────────────────────────────
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

        // Clean up shader modules (no longer needed after pipeline creation).
        unsafe {
            ctx.device.destroy_shader_module(vert_module, None);
            ctx.device.destroy_shader_module(frag_module, None);
        }

        Ok(Self { descriptor_set_layout, pipeline_layout, pipeline })
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

fn load_spirv(path: &str) -> Result<Vec<u32>, RenderError> {
    let bytes = std::fs::read(Path::new(path))
        .map_err(|_| RenderError::ShaderNotFound(
            // Leak a &'static str for the error.  This is fine for error paths.
            Box::leak(path.to_string().into_boxed_str())
        ))?;

    // SPIR-V must be u32-aligned.
    if bytes.len() % 4 != 0 {
        return Err(RenderError::Other("SPIR-V size not aligned to 4 bytes".into()));
    }

    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(words)
}

fn create_shader_module(device: &ash::Device, code: &[u32]) -> Result<vk::ShaderModule, RenderError> {
    let ci = vk::ShaderModuleCreateInfo::default().code(code);
    unsafe { device.create_shader_module(&ci, None).map_err(RenderError::Vulkan) }
}