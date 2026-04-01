//! # Vulkan context
//!
//! Instance, physical device, logical device, queues, swapchain.
//! Uses `ash` 0.38 exclusively / no helper crates.

use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use std::ffi::{CStr, CString};

use super::RenderError;

/// Core Vulkan objects needed by all renderers.
pub struct VulkanContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub surface_loader: ash::khr::surface::Instance,
    pub surface: vk::SurfaceKHR,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
    pub graphics_family: u32,
    pub present_family: u32,
    pub swapchain_loader: ash::khr::swapchain::Device,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_views: Vec<vk::ImageView>,
    pub swapchain_format: vk::Format,
    pub swapchain_extent: vk::Extent2D,
    pub command_pool: vk::CommandPool,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub render_pass: vk::RenderPass,
    pub framebuffers: Vec<vk::Framebuffer>,
    // Sync primitives (per frame-in-flight).
    pub image_available: Vec<vk::Semaphore>,
    pub render_finished: Vec<vk::Semaphore>,
    pub in_flight: Vec<vk::Fence>,
    pub current_frame: usize,
    pub max_frames_in_flight: usize,
}

impl VulkanContext {
    /// Initialise the full Vulkan stack for the given window.
    ///
    /// # Safety
    /// The window handles must be valid and outlive this context.
pub unsafe fn new(
        window: &impl HasWindowHandle,
        display: &impl HasDisplayHandle,
        width: u32,
        height: u32,
    ) -> Result<Self, RenderError> {
        let entry = ash::Entry::load()
            .map_err(|e| RenderError::Other(format!("failed to load Vulkan loader: {e}")))?;


        // ── Instance ────────────────────────────────────────────────
        let app_info = vk::ApplicationInfo::default()
            .application_name(CStr::from_bytes_with_nul_unchecked(b"OC-Emulator\0"))
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(CStr::from_bytes_with_nul_unchecked(b"NoEngine\0"))
            .api_version(vk::API_VERSION_1_1);

        let mut extensions = vec![
            ash::khr::surface::NAME.as_ptr(),
        ];

        // Platform surface extension.
        let raw_display = display.display_handle()
            .map_err(|e| RenderError::Other(format!("{e}")))?
            .as_raw();
        match raw_display {
            #[cfg(target_os = "windows")]
            RawDisplayHandle::Windows(_) => {
                extensions.push(ash::khr::win32_surface::NAME.as_ptr());
            }
            #[cfg(target_os = "linux")]
            RawDisplayHandle::Xlib(_) => {
                extensions.push(ash::khr::xlib_surface::NAME.as_ptr());
            }
            #[cfg(target_os = "linux")]
            RawDisplayHandle::Wayland(_) => {
                extensions.push(ash::khr::wayland_surface::NAME.as_ptr());
            }
            _ => return Err(RenderError::Other("unsupported display handle".into())),
        }

        let instance_ci = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extensions);

        let instance = entry.create_instance(&instance_ci, None)
            .map_err(RenderError::Vulkan)?;

        // ── Surface ─────────────────────────────────────────────────
        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);
        let surface = create_surface(&entry, &instance, window, display)?;

        // ── Physical device ─────────────────────────────────────────
        let physical_devices = instance.enumerate_physical_devices()
            .map_err(RenderError::Vulkan)?;

        let (physical_device, graphics_family, present_family) = physical_devices.iter()
            .find_map(|&pd| {
                let props = instance.get_physical_device_queue_family_properties(pd);
                let gfx = props.iter().position(|qf|
                    qf.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                )?;
                let present = props.iter().enumerate().position(|(i, _)|
                    surface_loader.get_physical_device_surface_support(pd, i as u32, surface).unwrap_or(false)
                )?;
                Some((pd, gfx as u32, present as u32))
            })
            .ok_or(RenderError::Other("no suitable GPU found".into()))?;

        // ── Logical device ──────────────────────────────────────────
        let unique_families: Vec<u32> = {
            let mut v = vec![graphics_family, present_family];
            v.sort_unstable();
            v.dedup();
            v
        };

        let queue_priorities = [1.0f32];
        let queue_cis: Vec<_> = unique_families.iter().map(|&family|
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family)
                .queue_priorities(&queue_priorities)
        ).collect();

        let device_extensions = [ash::khr::swapchain::NAME.as_ptr()];
        let device_ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_cis)
            .enabled_extension_names(&device_extensions);

        let device = instance.create_device(physical_device, &device_ci, None)
            .map_err(RenderError::Vulkan)?;

        let graphics_queue = device.get_device_queue(graphics_family, 0);
        let present_queue = device.get_device_queue(present_family, 0);

        // ── Swapchain ───────────────────────────────────────────────
        let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);

        let caps = surface_loader
            .get_physical_device_surface_capabilities(physical_device, surface)
            .map_err(RenderError::Vulkan)?;

        let formats = surface_loader
            .get_physical_device_surface_formats(physical_device, surface)
            .map_err(RenderError::Vulkan)?;

        let format = formats.iter()
            .find(|f| f.format == vk::Format::B8G8R8A8_SRGB && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .unwrap_or(&formats[0]);

        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D {
                width: width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        };

        let image_count = (caps.min_image_count + 1).min(
            if caps.max_image_count > 0 { caps.max_image_count } else { u32::MAX }
        );

        let sharing_mode = if graphics_family != present_family {
            vk::SharingMode::CONCURRENT
        } else {
            vk::SharingMode::EXCLUSIVE
        };

        let families = [graphics_family, present_family];
        let mut swapchain_ci = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(sharing_mode)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true);

        if sharing_mode == vk::SharingMode::CONCURRENT {
            swapchain_ci = swapchain_ci.queue_family_indices(&families);
        }

        let swapchain = swapchain_loader.create_swapchain(&swapchain_ci, None)
            .map_err(RenderError::Vulkan)?;

        let swapchain_images = swapchain_loader.get_swapchain_images(swapchain)
            .map_err(RenderError::Vulkan)?;

        let swapchain_views: Vec<_> = swapchain_images.iter().map(|&img| {
            let ci = vk::ImageViewCreateInfo::default()
                .image(img)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format.format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            device.create_image_view(&ci, None).unwrap()
        }).collect();

        // ── Render pass ─────────────────────────────────────────────
        let attachment = vk::AttachmentDescription::default()
            .format(format.format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let color_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let color_refs = [color_ref];
        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_refs);

        let subpasses = [subpass];
        let attachments = [attachment];
        let rp_ci = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpasses);

        let render_pass = device.create_render_pass(&rp_ci, None)
            .map_err(RenderError::Vulkan)?;

        // ── Framebuffers ────────────────────────────────────────────
        let framebuffers: Vec<_> = swapchain_views.iter().map(|&view| {
            let attachments = [view];
            let fb_ci = vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(extent.width)
                .height(extent.height)
                .layers(1);
            device.create_framebuffer(&fb_ci, None).unwrap()
        }).collect();

        // ── Command pool + buffers ──────────────────────────────────
        let pool_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(graphics_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = device.create_command_pool(&pool_ci, None)
            .map_err(RenderError::Vulkan)?;

        let max_frames = 2;
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(max_frames as u32);

        let command_buffers = device.allocate_command_buffers(&alloc_info)
            .map_err(RenderError::Vulkan)?;

        // ── Sync objects ────────────────────────────────────────────
        let sem_ci = vk::SemaphoreCreateInfo::default();
        let fence_ci = vk::FenceCreateInfo::default()
            .flags(vk::FenceCreateFlags::SIGNALED);

        let mut image_available = Vec::with_capacity(max_frames);
        let mut render_finished = Vec::with_capacity(max_frames);
        let mut in_flight = Vec::with_capacity(max_frames);

        for _ in 0..max_frames {
            image_available.push(device.create_semaphore(&sem_ci, None).unwrap());
            render_finished.push(device.create_semaphore(&sem_ci, None).unwrap());
            in_flight.push(device.create_fence(&fence_ci, None).unwrap());
        }

        Ok(Self {
            entry, instance, surface_loader, surface,
            physical_device, device,
            graphics_queue, present_queue,
            graphics_family, present_family,
            swapchain_loader, swapchain, swapchain_images, swapchain_views,
            swapchain_format: format.format, swapchain_extent: extent,
            command_pool, command_buffers,
            render_pass, framebuffers,
            image_available, render_finished, in_flight,
            current_frame: 0, max_frames_in_flight: max_frames,
        })
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            for &f in &self.in_flight { self.device.destroy_fence(f, None); }
            for &s in &self.render_finished { self.device.destroy_semaphore(s, None); }
            for &s in &self.image_available { self.device.destroy_semaphore(s, None); }
            self.device.destroy_command_pool(self.command_pool, None);
            for &fb in &self.framebuffers { self.device.destroy_framebuffer(fb, None); }
            self.device.destroy_render_pass(self.render_pass, None);
            for &v in &self.swapchain_views { self.device.destroy_image_view(v, None); }
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

// ── Platform surface creation ───────────────────────────────────────────

unsafe fn create_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    window: &impl HasWindowHandle,
    display: &impl HasDisplayHandle,
) -> Result<vk::SurfaceKHR, RenderError> {
    let raw_window = window.window_handle()
        .map_err(|e| RenderError::Other(format!("{e}")))?
        .as_raw();

    match raw_window {
        #[cfg(target_os = "windows")]
        RawWindowHandle::Win32(handle) => {
            let loader = ash::khr::win32_surface::Instance::new(entry, instance);
            let ci = vk::Win32SurfaceCreateInfoKHR::default()
                .hinstance(handle.hinstance.unwrap().get() as _)
                .hwnd(handle.hwnd.get() as _);
            loader.create_win32_surface(&ci, None).map_err(RenderError::Vulkan)
        }
        #[cfg(target_os = "linux")]
        RawWindowHandle::Xlib(handle) => {
            let raw_display = display.display_handle().unwrap().as_raw();
            if let RawDisplayHandle::Xlib(dh) = raw_display {
                let loader = ash::khr::xlib_surface::Instance::new(entry, instance);
                let ci = vk::XlibSurfaceCreateInfoKHR::default()
                    .dpy(dh.display.unwrap().as_ptr() as *mut _)
                    .window(handle.window);
                loader.create_xlib_surface(&ci, None).map_err(RenderError::Vulkan)
            } else {
                Err(RenderError::Other("display handle mismatch".into()))
            }
        }
        #[cfg(target_os = "linux")]
        RawWindowHandle::Wayland(handle) => {
            let raw_display = display.display_handle().unwrap().as_raw();
            if let RawDisplayHandle::Wayland(dh) = raw_display {
                let loader = ash::khr::wayland_surface::Instance::new(entry, instance);
                let ci = vk::WaylandSurfaceCreateInfoKHR::default()
                    .display(dh.display.as_ptr() as *mut _)
                    .surface(handle.surface.as_ptr() as *mut _);
                loader.create_wayland_surface(&ci, None).map_err(RenderError::Vulkan)
            } else {
                Err(RenderError::Other("display handle mismatch".into()))
            }
        }
        _ => Err(RenderError::Other("unsupported window handle type".into())),
    }
}