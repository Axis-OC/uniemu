//! # Vulkan context
//!
//! This module initialises the entire Vulkan stack from scratch:
//! instance, physical device selection, logical device, queues,
//! surface, swapchain, render pass, framebuffers, command pool,
//! command buffers, and synchronisation primitives.
//!
//! Uses `ash` 0.38 exclusively -- no helper crates like `vk-mem`,
//! `vulkano`, or `wgpu`.
//!
//! ## Initialisation sequence
//!
//! ```text
//! 1.  Load Vulkan loader (ash::Entry::load)
//! 2.  Create VkInstance with platform surface extensions
//! 3.  Create VkSurfaceKHR from the window handle
//! 4.  Enumerate physical devices, select one with:
//!     - Graphics queue family
//!     - Present queue family (may be the same)
//! 5.  Create VkDevice with VK_KHR_swapchain extension
//! 6.  Get graphics and present queues
//! 7.  Select present mode (MAILBOX > IMMEDIATE > FIFO)
//! 8.  Create VkSwapchainKHR
//! 9.  Get swapchain images, create image views
//! 10. Create render pass (single color attachment, CLEAR -> STORE)
//! 11. Create framebuffers (one per swapchain image)
//! 12. Create command pool + 2 command buffers (double-buffered)
//! 13. Create sync primitives:
//!     - 2x image_available semaphores
//!     - 2x render_finished semaphores
//!     - 2x in_flight fences (signaled initially)
//! ```
//!
//! ## Swapchain recreation
//!
//! When the window is resized, [`recreate_swapchain`](VulkanContext::recreate_swapchain)
//! destroys old framebuffers and image views, creates a new swapchain
//! (passing the old one for recycling), and rebuilds everything.
//!
//! ## Platform surface creation
//!
//! Supports:
//! * Windows: `VK_KHR_win32_surface`
//! * Linux/X11: `VK_KHR_xlib_surface`
//! * Linux/Wayland: `VK_KHR_wayland_surface`
//!
//! ## Present mode selection
//!
//! Preference order:
//! 1. **MAILBOX** - No vsync, no tearing, lowest latency.
//! 2. **IMMEDIATE** - No vsync, may tear, lowest latency.
//! 3. **FIFO** - Vsync (fallback, always available).
//!
//! Can be overridden via the settings GUI (F8).
//!
//! ## Resource lifetime
//!
//! `VulkanContext` implements `Drop` which destroys everything in
//! reverse creation order. The renderers (`IndirectRenderer`,
//! `DirectRenderer`) MUST be destroyed before the context, because
//! they hold references to the device and swapchain.

use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use std::ffi::{CStr, CString};

use log;

use super::RenderError;

/// Core Vulkan objects needed by all renderers.
///
/// This struct is the "Vulkan root" of the emulator. It owns the
/// instance, device, swapchain, and all associated objects. Renderers
/// borrow it mutably during frame recording and submission.
///
/// # Drop order
///
/// The `Drop` implementation destroys objects in reverse creation order:
/// fences, semaphores, command pool, framebuffers, render pass, image
/// views, swapchain, device, surface, instance.
///
/// Renderers that hold Vulkan handles (buffers, images, pipelines)
/// MUST call their own `destroy()` method before this struct is dropped.
pub struct VulkanContext {
    /// The Vulkan entry point (loader).
    pub entry: ash::Entry,

    /// The Vulkan instance.
    pub instance: ash::Instance,

    /// Surface extension loader.
    pub surface_loader: ash::khr::surface::Instance,

    /// The window surface.
    pub surface: vk::SurfaceKHR,

    /// The selected physical device (GPU).
    pub physical_device: vk::PhysicalDevice,

    /// The logical device.
    pub device: ash::Device,

    /// The graphics queue (for command buffer submission).
    pub graphics_queue: vk::Queue,

    /// The present queue (for swapchain presentation).
    ///
    /// May be the same queue as `graphics_queue` if both families
    /// are the same.
    pub present_queue: vk::Queue,

    /// Graphics queue family index.
    pub graphics_family: u32,

    /// Present queue family index.
    pub present_family: u32,

    /// Swapchain extension loader.
    pub swapchain_loader: ash::khr::swapchain::Device,

    /// The swapchain handle.
    pub swapchain: vk::SwapchainKHR,

    /// Swapchain images (owned by the swapchain, not by us).
    pub swapchain_images: Vec<vk::Image>,

    /// Image views for the swapchain images (owned by us).
    pub swapchain_views: Vec<vk::ImageView>,

    /// Swapchain image format (e.g. `B8G8R8A8_UNORM`).
    pub swapchain_format: vk::Format,

    /// Swapchain image extent (width x height in pixels).
    pub swapchain_extent: vk::Extent2D,

    /// Command pool for the graphics family.
    pub command_pool: vk::CommandPool,

    /// Pre-allocated command buffers (one per frame-in-flight).
    pub command_buffers: Vec<vk::CommandBuffer>,

    /// The render pass (single color attachment, CLEAR load op).
    pub render_pass: vk::RenderPass,

    /// Framebuffers (one per swapchain image).
    pub framebuffers: Vec<vk::Framebuffer>,

    /// Per-frame semaphores: signaled when swapchain image is acquired.
    pub image_available: Vec<vk::Semaphore>,

    /// Per-frame semaphores: signaled when rendering is finished.
    pub render_finished: Vec<vk::Semaphore>,

    /// Per-frame fences: signaled when the GPU finishes processing
    /// the command buffer for this frame. Used to prevent the CPU
    /// from overwriting data still in use by the GPU.
    pub in_flight: Vec<vk::Fence>,

    /// Current frame index (0 or 1 for double buffering).
    pub current_frame: usize,

    /// Per-swapchain-image fence tracking.
    /// `images_in_flight[image_index]` holds the fence of the frame
    /// currently using that swapchain image, or `vk::Fence::null()`.
    pub images_in_flight: Vec<vk::Fence>,

    /// Maximum frames in flight (always 2).
    pub max_frames_in_flight: usize,

    /// Human-readable GPU name (e.g. "NVIDIA GeForce RTX 4090").
    pub gpu_name: String,

    /// Active present mode.
    pub present_mode: vk::PresentModeKHR,

    /// Debug utils messenger (only present if validation layers are active).
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    /// Debug utils extension loader.
    debug_utils_loader: Option<ash::ext::debug_utils::Instance>,
}

/// Vulkan debug messenger callback — routes validation layer messages
/// through the `log` crate, matching game-engine conventions.
///
/// Message severity mapping:
/// - VERBOSE → trace
/// - INFO    → debug  
/// - WARNING → warn
/// - ERROR   → error
unsafe extern "system" fn vulkan_debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    msg_type: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let msg = if callback_data.is_null() {
        "(null callback data)".to_string()
    } else {
        let raw = (*callback_data).p_message;
        if raw.is_null() {
            "(null message)".to_string()
        } else {
            CStr::from_ptr(raw).to_string_lossy().into_owned()
        }
    };

    let kind = if msg_type.contains(vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION) {
        "VALIDATION"
    } else if msg_type.contains(vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE) {
        "PERFORMANCE"
    } else {
        "GENERAL"
    };

    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        log::error!("[Vulkan/{kind}] {msg}");
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        log::warn!("[Vulkan/{kind}] {msg}");
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::INFO) {
        log::debug!("[Vulkan/{kind}] {msg}");
    } else {
        log::trace!("[Vulkan/{kind}] {msg}");
    }

    vk::FALSE
}

impl VulkanContext {
    /// Initialise the full Vulkan stack for the given window.
    ///
    /// # Safety
    ///
    /// The window and display handles must be valid and must outlive
    /// this context. The caller must ensure the window is not destroyed
    /// while this context exists.
    ///
    /// # Arguments
    ///
    /// * `window` - Window handle provider (for surface creation).
    /// * `display` - Display handle provider (for determining the
    ///   platform surface extension).
    /// * `width` - Initial window width.
    /// * `height` - Initial window height.
    ///
    /// # Errors
    ///
    /// Returns `RenderError::Other` if the Vulkan loader cannot be
    /// loaded, no suitable GPU is found, or the display handle type
    /// is unsupported.
    ///
    /// Returns `RenderError::Vulkan` for any Vulkan API failure.
    pub unsafe fn new(
        window: &impl HasWindowHandle,
        display: &impl HasDisplayHandle,
        width: u32,
        height: u32,
    ) -> Result<Self, RenderError> {
        log::info!("=== Vulkan initialisation ===");

        let entry = ash::Entry::load()
            .map_err(|e| {
                log::error!("Failed to load Vulkan loader: {e}");
                RenderError::Other(format!("failed to load Vulkan loader: {e}"))
            })?;
        log::debug!("Vulkan loader loaded");

        // Instance
        let app_info = vk::ApplicationInfo::default()
            .application_name(CStr::from_bytes_with_nul_unchecked(b"OC-Emulator\0"))
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(CStr::from_bytes_with_nul_unchecked(b"NoEngine\0"))
            .api_version(vk::API_VERSION_1_1);

        let mut extensions = vec![
            ash::khr::surface::NAME.as_ptr(),
        ];

        // Platform surface extension
        let raw_display = display.display_handle()
            .map_err(|e| RenderError::Other(format!("{e}")))?
            .as_raw();
        match raw_display {
            #[cfg(target_os = "windows")]
            RawDisplayHandle::Windows(_) => {
                extensions.push(ash::khr::win32_surface::NAME.as_ptr());
                log::debug!("Platform: Windows (VK_KHR_win32_surface)");
            }
            #[cfg(target_os = "linux")]
            RawDisplayHandle::Xlib(_) => {
                extensions.push(ash::khr::xlib_surface::NAME.as_ptr());
                log::debug!("Platform: Linux/X11 (VK_KHR_xlib_surface)");
            }
            #[cfg(target_os = "linux")]
            RawDisplayHandle::Wayland(_) => {
                extensions.push(ash::khr::wayland_surface::NAME.as_ptr());
                log::debug!("Platform: Linux/Wayland (VK_KHR_wayland_surface)");
            }
            _ => return Err(RenderError::Other("unsupported display handle".into())),
        }

        // Try to enable debug utils if validation layers are available
        let mut enable_debug = false;
        let available_layers = entry.enumerate_instance_layer_properties().unwrap_or_default();
        let has_validation = available_layers.iter().any(|l| {
            let name = CStr::from_ptr(l.layer_name.as_ptr());
            name.to_bytes() == b"VK_LAYER_KHRONOS_validation"
        });
        let mut layer_names: Vec<*const i8> = Vec::new();
        if has_validation {
            layer_names.push(c"VK_LAYER_KHRONOS_validation".as_ptr());
            extensions.push(ash::ext::debug_utils::NAME.as_ptr());
            enable_debug = true;
            log::info!("Vulkan validation layers ENABLED");
        } else {
            log::debug!("Vulkan validation layers not available");
        }

        log::debug!("Instance extensions: {}", extensions.len());

        let instance_ci = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extensions)
            .enabled_layer_names(&layer_names);

        let instance = entry.create_instance(&instance_ci, None)
            .map_err(|e| {
                log::error!("vkCreateInstance failed: {e:?}");
                RenderError::Vulkan(e)
            })?;
        log::info!("VkInstance created (API 1.1)");

        // Debug messenger
        let (debug_utils_loader, debug_messenger) = if enable_debug {
            let loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let ci = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                )
                .pfn_user_callback(Some(vulkan_debug_callback));
            match loader.create_debug_utils_messenger(&ci, None) {
                Ok(messenger) => {
                    log::debug!("Vulkan debug messenger created");
                    (Some(loader), Some(messenger))
                }
                Err(e) => {
                    log::warn!("Failed to create debug messenger: {e:?}");
                    (None, None)
                }
            }
        } else {
            (None, None)
        };

        // Surface
        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);
        let surface = create_surface(&entry, &instance, window, display)?;
        log::debug!("VkSurfaceKHR created");

        // Physical device
        let physical_devices = instance.enumerate_physical_devices()
            .map_err(RenderError::Vulkan)?;
        log::info!("Found {} physical device(s)", physical_devices.len());

        for (i, &pd) in physical_devices.iter().enumerate() {
            let props = instance.get_physical_device_properties(pd);
            let name = CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy();
            let dev_type = match props.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU   => "Discrete",
                vk::PhysicalDeviceType::INTEGRATED_GPU => "Integrated",
                vk::PhysicalDeviceType::VIRTUAL_GPU    => "Virtual",
                vk::PhysicalDeviceType::CPU            => "CPU",
                _ => "Other",
            };
            let driver = vk::api_version_major(props.driver_version);
            log::info!("  [{}] {} ({}, driver v{}.{}.{})", i, name, dev_type,
                vk::api_version_major(props.driver_version),
                vk::api_version_minor(props.driver_version),
                vk::api_version_patch(props.driver_version));
        }

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
            .ok_or_else(|| {
                log::error!("No suitable GPU found with graphics+present queues");
                RenderError::Other("no suitable GPU found".into())
            })?;

        // Log selected device memory
        let mem_props = instance.get_physical_device_memory_properties(physical_device);
        log::info!("Device memory heaps:");
        for i in 0..mem_props.memory_heap_count as usize {
            let heap = &mem_props.memory_heaps[i];
            let mib = heap.size / (1024 * 1024);
            let flags = if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                "DEVICE_LOCAL"
            } else {
                "HOST"
            };
            log::info!("  Heap {}: {} MiB ({})", i, mib, flags);
        }

        // Present mode
        let present_modes = surface_loader
            .get_physical_device_surface_present_modes(physical_device, surface)
            .map_err(RenderError::Vulkan)?;
        log::debug!("Available present modes: {:?}", present_modes);

        let present_mode = if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
            log::info!("Present mode: MAILBOX (no vsync, no tearing)");
            vk::PresentModeKHR::MAILBOX
        } else if present_modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
            log::info!("Present mode: IMMEDIATE (no vsync, may tear)");
            vk::PresentModeKHR::IMMEDIATE
        } else {
            log::info!("Present mode: FIFO (vsync fallback)");
            vk::PresentModeKHR::FIFO
        };

        // Logical device 
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
        log::info!("VkDevice created (graphics family={}, present family={})",
            graphics_family, present_family);

        // Swapchain 
        let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);

        let caps = surface_loader
            .get_physical_device_surface_capabilities(physical_device, surface)
            .map_err(RenderError::Vulkan)?;

        let formats = surface_loader
            .get_physical_device_surface_formats(physical_device, surface)
            .map_err(RenderError::Vulkan)?;

        let format = formats.iter()
            .find(|f| f.format == vk::Format::B8G8R8A8_UNORM
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .or_else(|| formats.iter().find(|f| f.format == vk::Format::B8G8R8A8_UNORM))
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
            //.present_mode(vk::PresentModeKHR::FIFO)
            .present_mode(present_mode)
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
        log::info!("Swapchain created: {}x{}, format={:?}, images={}, present={:?}",
            extent.width, extent.height, format.format, image_count, present_mode);

        // Render pass 
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
        log::debug!("Render pass created (1 color attachment, CLEAR->STORE)");

        // Framebuffers 
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
        log::debug!("Framebuffers created: {}", swapchain_views.len());

        // Command pool + buffers 
        let pool_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(graphics_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = device.create_command_pool(&pool_ci, None)
            .map_err(RenderError::Vulkan)?;
        log::debug!("Command pool created (family={}, RESET_COMMAND_BUFFER)", graphics_family);

        let max_frames = 2;
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(max_frames as u32);

        let command_buffers = device.allocate_command_buffers(&alloc_info)
            .map_err(RenderError::Vulkan)?;

        // Sync objects 
        let sem_ci = vk::SemaphoreCreateInfo::default();
        let fence_ci = vk::FenceCreateInfo::default()
            .flags(vk::FenceCreateFlags::SIGNALED);

        let mut image_available = Vec::with_capacity(max_frames);
        let mut in_flight = Vec::with_capacity(max_frames);

        for _ in 0..max_frames {
            image_available.push(device.create_semaphore(&sem_ci, None).unwrap());
            in_flight.push(device.create_fence(&fence_ci, None).unwrap());
        }

        // render_finished: one per swapchain image (not per frame-in-flight!)
        let mut render_finished = Vec::with_capacity(swapchain_images.len());
        for _ in 0..swapchain_images.len() {
            render_finished.push(device.create_semaphore(&sem_ci, None).unwrap());
        }

        // Track which fence is associated with each swapchain image
        let images_in_flight = vec![vk::Fence::null(); swapchain_images.len()];

        for _ in 0..max_frames {
            image_available.push(device.create_semaphore(&sem_ci, None).unwrap());
            render_finished.push(device.create_semaphore(&sem_ci, None).unwrap());
            in_flight.push(device.create_fence(&fence_ci, None).unwrap());
        }

        log::debug!("Sync objects: {} semaphore pairs + {} fences", max_frames, max_frames);
        
        let gpu_name = {
            let props = instance.get_physical_device_properties(physical_device);
            let name = CStr::from_ptr(props.device_name.as_ptr())
                .to_string_lossy().into_owned();
            log::info!("Vulkan ready: {} ({}x{})", name, extent.width, extent.height);
            name
        };

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
            current_frame: 0, max_frames_in_flight: max_frames, gpu_name, present_mode,
            debug_messenger, debug_utils_loader, images_in_flight
        })

    }
    /// Recreate the swapchain after a window resize.
    ///
    /// # What happens
    ///
    /// 1. Wait for device idle (drain all pending work).
    /// 2. Destroy old framebuffers and image views.
    /// 3. Create a new swapchain (passing old swapchain for recycling).
    /// 4. Destroy the old swapchain.
    /// 5. Get new swapchain images and create image views.
    /// 6. Recreate framebuffers.
    ///
    /// # Arguments
    ///
    /// * `width` - New window width.
    /// * `height` - New window height.
    ///
    /// # Errors
    ///
    /// Returns `RenderError::Vulkan` on any API failure.
    pub unsafe fn recreate_swapchain(
        &mut self, width: u32, height: u32,
    ) -> Result<(), super::RenderError> {
        log::info!("Recreating swapchain: {}x{} -> {}x{}",
            self.swapchain_extent.width, self.swapchain_extent.height,
            width, height);

        self.device.device_wait_idle().map_err(RenderError::Vulkan)?;
        log::debug!("Device idle, destroying old resources ({} framebuffers, {} views)",
            self.framebuffers.len(), self.swapchain_views.len());

        for &fb in &self.framebuffers { self.device.destroy_framebuffer(fb, None); }
        for &v in &self.swapchain_views { self.device.destroy_image_view(v, None); }
        let old_swapchain = self.swapchain;

        let caps = self.surface_loader
            .get_physical_device_surface_capabilities(self.physical_device, self.surface)
            .map_err(RenderError::Vulkan)?;
        let formats = self.surface_loader
            .get_physical_device_surface_formats(self.physical_device, self.surface)
            .map_err(RenderError::Vulkan)?;
        let format = formats.iter()
            .find(|f| f.format == vk::Format::B8G8R8A8_UNORM
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .or_else(|| formats.iter().find(|f| f.format == vk::Format::B8G8R8A8_UNORM))
            .unwrap_or(&formats[0]);

        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D {
                width:  width.clamp(caps.min_image_extent.width,  caps.max_image_extent.width),
                height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        };

        let image_count = (caps.min_image_count + 1).min(
            if caps.max_image_count > 0 { caps.max_image_count } else { u32::MAX }
        );

        let sharing_mode = if self.graphics_family != self.present_family {
            vk::SharingMode::CONCURRENT
        } else {
            vk::SharingMode::EXCLUSIVE
        };
        let families = [self.graphics_family, self.present_family];

        let mut sc_ci = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(sharing_mode)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            //.present_mode(vk::PresentModeKHR::FIFO)
            .present_mode(self.present_mode)
            .clipped(true)
            .old_swapchain(old_swapchain);

        if sharing_mode == vk::SharingMode::CONCURRENT {
            sc_ci = sc_ci.queue_family_indices(&families);
        }

        self.swapchain = self.swapchain_loader.create_swapchain(&sc_ci, None)
            .map_err(RenderError::Vulkan)?;
        self.swapchain_loader.destroy_swapchain(old_swapchain, None);

        self.swapchain_images = self.swapchain_loader.get_swapchain_images(self.swapchain)
            .map_err(RenderError::Vulkan)?;

        self.swapchain_views = self.swapchain_images.iter().map(|&img| {
            let ci = vk::ImageViewCreateInfo::default()
                .image(img)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format.format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0, level_count: 1,
                    base_array_layer: 0, layer_count: 1,
                });
            self.device.create_image_view(&ci, None).unwrap()
        }).collect();

        self.swapchain_format = format.format;
        self.swapchain_extent = extent;

        self.framebuffers = self.swapchain_views.iter().map(|&view| {
            let attachments = [view];
            let fb_ci = vk::FramebufferCreateInfo::default()
                .render_pass(self.render_pass)
                .attachments(&attachments)
                .width(extent.width)
                .height(extent.height)
                .layers(1);
            self.device.create_framebuffer(&fb_ci, None).unwrap()
        }).collect();
        
        for &s in &self.render_finished {
            self.device.destroy_semaphore(s, None);
        }
        let sem_ci = vk::SemaphoreCreateInfo::default();
        self.render_finished = (0..self.swapchain_images.len())
            .map(|_| self.device.create_semaphore(&sem_ci, None).unwrap())
            .collect();

        self.images_in_flight = vec![vk::Fence::null(); self.swapchain_images.len()];

        log::info!("Swapchain recreated: {}x{} format={:?} images={} present={:?}",
            extent.width, extent.height, format.format,
            self.swapchain_images.len(), self.present_mode);
        Ok(())
    }
}



impl Drop for VulkanContext {
    fn drop(&mut self) {
        log::info!("Destroying Vulkan context...");
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

/// Create a platform-specific Vulkan surface from window handles.
///
/// # Supported platforms
///
/// * Windows: `RawWindowHandle::Win32` -> `VK_KHR_win32_surface`
/// * Linux/X11: `RawWindowHandle::Xlib` -> `VK_KHR_xlib_surface`
/// * Linux/Wayland: `RawWindowHandle::Wayland` -> `VK_KHR_wayland_surface`
///
/// # Safety
///
/// The window and display handles must be valid.
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