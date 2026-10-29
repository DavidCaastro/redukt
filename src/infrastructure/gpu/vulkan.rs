//! Vulkan FFI types and dynamic loader for compute-only GPU offload.
//! Loaded at runtime from vulkan-1.dll — zero compile-time dependencies.
//! Only used on Windows x86_64 with Vulkan 1.1+ runtime.

#![allow(non_snake_case, non_camel_case_types, dead_code)]

use std::ffi::c_void;
use std::ptr::{null, null_mut};

// ---- Handle types ----

pub type VkInstance = *mut c_void;
pub type VkPhysicalDevice = *mut c_void;
pub type VkDevice = *mut c_void;
pub type VkQueue = *mut c_void;
pub type VkCommandBuffer = *mut c_void;
pub type VkBuffer = u64;
pub type VkDeviceMemory = u64;
pub type VkShaderModule = u64;
pub type VkPipeline = u64;
pub type VkPipelineLayout = u64;
pub type VkDescriptorSetLayout = u64;
pub type VkDescriptorPool = u64;
pub type VkDescriptorSet = u64;
pub type VkCommandPool = u64;
pub type VkFence = u64;
pub type VkPipelineCache = u64;
pub type VkDeviceSize = u64;

// ---- Constants ----

const VK_STRUCTURE_TYPE_APPLICATION_INFO: u32 = 0;
const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: u32 = 1;
const VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO: u32 = 2;
const VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO: u32 = 3;
const VK_STRUCTURE_TYPE_SUBMIT_INFO: u32 = 4;
const VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO: u32 = 5;
const VK_STRUCTURE_TYPE_FENCE_CREATE_INFO: u32 = 8;
const VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO: u32 = 12;
const VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO: u32 = 16;
const VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO: u32 = 18;
const VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO: u32 = 29;
const VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO: u32 = 30;
const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO: u32 = 32;
const VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO: u32 = 33;
const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO: u32 = 34;
const VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET: u32 = 35;
const VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO: u32 = 39;
const VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO: u32 = 40;
const VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO: u32 = 42;

const VK_SUCCESS: i32 = 0;
const VK_QUEUE_COMPUTE_BIT: u32 = 2;
const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: u32 = 0x00000020;
const VK_SHARING_MODE_EXCLUSIVE: u32 = 0;
const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: u32 = 0x02;
const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: u32 = 0x04;
const VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: u32 = 7;
const VK_SHADER_STAGE_COMPUTE_BIT: u32 = 0x00000020;
const VK_PIPELINE_BIND_POINT_COMPUTE: u32 = 1;
const VK_COMMAND_BUFFER_LEVEL_PRIMARY: u32 = 0;
const VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: u32 = 0x02;
const VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: u32 = 0x01;
const VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: u32 = 1;
const VK_NULL_HANDLE: u64 = 0;

// Vulkan 1.1 = 0x00401000 (variant=0, major=1, minor=1, patch=0)
const VK_API_VERSION_1_1: u32 = (1 << 22) | (1 << 12);

const VK_MAX_MEMORY_TYPES: usize = 32;
const VK_MAX_MEMORY_HEAPS: usize = 16;
const VK_MAX_PHYSICAL_DEVICE_NAME_SIZE: usize = 256;

// ---- Structs ----

#[repr(C)]
struct VkApplicationInfo {
    s_type: u32,
    p_next: *const c_void,
    p_application_name: *const u8,
    application_version: u32,
    p_engine_name: *const u8,
    engine_version: u32,
    api_version: u32,
}

#[repr(C)]
struct VkInstanceCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    p_application_info: *const VkApplicationInfo,
    enabled_layer_count: u32,
    pp_enabled_layer_names: *const *const u8,
    enabled_extension_count: u32,
    pp_enabled_extension_names: *const *const u8,
}

#[repr(C)]
struct VkDeviceQueueCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    queue_family_index: u32,
    queue_count: u32,
    p_queue_priorities: *const f32,
}

#[repr(C)]
struct VkDeviceCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    queue_create_info_count: u32,
    p_queue_create_infos: *const VkDeviceQueueCreateInfo,
    enabled_layer_count: u32,
    pp_enabled_layer_names: *const *const u8,
    enabled_extension_count: u32,
    pp_enabled_extension_names: *const *const u8,
    p_enabled_features: *const c_void,
}

#[repr(C)]
struct VkPhysicalDeviceProperties {
    api_version: u32,
    driver_version: u32,
    vendor_id: u32,
    device_id: u32,
    device_type: u32,
    device_name: [u8; VK_MAX_PHYSICAL_DEVICE_NAME_SIZE],
    pipeline_cache_uuid: [u8; 16],
    limits: [u8; 504],  // VkPhysicalDeviceLimits (we skip parsing)
    sparse_properties: [u8; 20],
}

#[repr(C)]
#[derive(Clone)]
struct VkQueueFamilyProperties {
    queue_flags: u32,
    queue_count: u32,
    timestamp_valid_bits: u32,
    min_image_transfer_granularity: [u32; 3],
}

#[repr(C)]
#[derive(Copy, Clone)]
struct VkMemoryType {
    property_flags: u32,
    heap_index: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct VkMemoryHeap {
    size: VkDeviceSize,
    flags: u32,
    _pad: u32,
}

#[repr(C)]
struct VkPhysicalDeviceMemoryProperties {
    memory_type_count: u32,
    memory_types: [VkMemoryType; VK_MAX_MEMORY_TYPES],
    memory_heap_count: u32,
    memory_heaps: [VkMemoryHeap; VK_MAX_MEMORY_HEAPS],
}

#[repr(C)]
struct VkBufferCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    size: VkDeviceSize,
    usage: u32,
    sharing_mode: u32,
    queue_family_index_count: u32,
    p_queue_family_indices: *const u32,
}

#[repr(C)]
struct VkMemoryRequirements {
    size: VkDeviceSize,
    alignment: VkDeviceSize,
    memory_type_bits: u32,
}

#[repr(C)]
struct VkMemoryAllocateInfo {
    s_type: u32,
    p_next: *const c_void,
    allocation_size: VkDeviceSize,
    memory_type_index: u32,
}

#[repr(C)]
struct VkShaderModuleCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    code_size: usize,
    p_code: *const u32,
}

#[repr(C)]
struct VkDescriptorSetLayoutBinding {
    binding: u32,
    descriptor_type: u32,
    descriptor_count: u32,
    stage_flags: u32,
    p_immutable_samplers: *const c_void,
}

#[repr(C)]
struct VkDescriptorSetLayoutCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    binding_count: u32,
    p_bindings: *const VkDescriptorSetLayoutBinding,
}

#[repr(C)]
struct VkPushConstantRange {
    stage_flags: u32,
    offset: u32,
    size: u32,
}

#[repr(C)]
struct VkPipelineLayoutCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    set_layout_count: u32,
    p_set_layouts: *const VkDescriptorSetLayout,
    push_constant_range_count: u32,
    p_push_constant_ranges: *const VkPushConstantRange,
}

#[repr(C)]
struct VkPipelineShaderStageCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    stage: u32,
    module: VkShaderModule,
    p_name: *const u8,
    p_specialization_info: *const c_void,
}

#[repr(C)]
struct VkComputePipelineCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    stage: VkPipelineShaderStageCreateInfo,
    layout: VkPipelineLayout,
    base_pipeline_handle: VkPipeline,
    base_pipeline_index: i32,
}

#[repr(C)]
struct VkDescriptorPoolSize {
    descriptor_type: u32,
    descriptor_count: u32,
}

#[repr(C)]
struct VkDescriptorPoolCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    max_sets: u32,
    pool_size_count: u32,
    p_pool_sizes: *const VkDescriptorPoolSize,
}

#[repr(C)]
struct VkDescriptorSetAllocateInfo {
    s_type: u32,
    p_next: *const c_void,
    descriptor_pool: VkDescriptorPool,
    descriptor_set_count: u32,
    p_set_layouts: *const VkDescriptorSetLayout,
}

#[repr(C)]
struct VkDescriptorBufferInfo {
    buffer: VkBuffer,
    offset: VkDeviceSize,
    range: VkDeviceSize,
}

#[repr(C)]
struct VkWriteDescriptorSet {
    s_type: u32,
    p_next: *const c_void,
    dst_set: VkDescriptorSet,
    dst_binding: u32,
    dst_array_element: u32,
    descriptor_count: u32,
    descriptor_type: u32,
    p_image_info: *const c_void,
    p_buffer_info: *const VkDescriptorBufferInfo,
    p_texel_buffer_view: *const c_void,
}

#[repr(C)]
struct VkCommandPoolCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    queue_family_index: u32,
}

#[repr(C)]
struct VkCommandBufferAllocateInfo {
    s_type: u32,
    p_next: *const c_void,
    command_pool: VkCommandPool,
    level: u32,
    command_buffer_count: u32,
}

#[repr(C)]
struct VkCommandBufferBeginInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    p_inheritance_info: *const c_void,
}

#[repr(C)]
struct VkSubmitInfo {
    s_type: u32,
    p_next: *const c_void,
    wait_semaphore_count: u32,
    p_wait_semaphores: *const u64,
    p_wait_dst_stage_mask: *const u32,
    command_buffer_count: u32,
    p_command_buffers: *const VkCommandBuffer,
    signal_semaphore_count: u32,
    p_signal_semaphores: *const u64,
}

#[repr(C)]
struct VkFenceCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
}

// ---- Function pointer types ----

type PFN_vkVoidFunction = *mut c_void;
type PFN_vkGetInstanceProcAddr =
    unsafe extern "system" fn(VkInstance, *const u8) -> PFN_vkVoidFunction;

// ---- Dynamic function table ----

struct VkFns {
    DestroyInstance: unsafe extern "system" fn(VkInstance, *const c_void),
    EnumeratePhysicalDevices:
        unsafe extern "system" fn(VkInstance, *mut u32, *mut VkPhysicalDevice) -> i32,
    GetPhysicalDeviceProperties:
        unsafe extern "system" fn(VkPhysicalDevice, *mut VkPhysicalDeviceProperties),
    GetPhysicalDeviceMemoryProperties:
        unsafe extern "system" fn(VkPhysicalDevice, *mut VkPhysicalDeviceMemoryProperties),
    GetPhysicalDeviceQueueFamilyProperties:
        unsafe extern "system" fn(VkPhysicalDevice, *mut u32, *mut VkQueueFamilyProperties),
    CreateDevice:
        unsafe extern "system" fn(VkPhysicalDevice, *const VkDeviceCreateInfo, *const c_void, *mut VkDevice) -> i32,
    DestroyDevice: unsafe extern "system" fn(VkDevice, *const c_void),
    GetDeviceQueue: unsafe extern "system" fn(VkDevice, u32, u32, *mut VkQueue),
    CreateBuffer:
        unsafe extern "system" fn(VkDevice, *const VkBufferCreateInfo, *const c_void, *mut VkBuffer) -> i32,
    DestroyBuffer: unsafe extern "system" fn(VkDevice, VkBuffer, *const c_void),
    GetBufferMemoryRequirements:
        unsafe extern "system" fn(VkDevice, VkBuffer, *mut VkMemoryRequirements),
    AllocateMemory:
        unsafe extern "system" fn(VkDevice, *const VkMemoryAllocateInfo, *const c_void, *mut VkDeviceMemory) -> i32,
    FreeMemory: unsafe extern "system" fn(VkDevice, VkDeviceMemory, *const c_void),
    BindBufferMemory:
        unsafe extern "system" fn(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) -> i32,
    MapMemory:
        unsafe extern "system" fn(VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, u32, *mut *mut c_void) -> i32,
    UnmapMemory: unsafe extern "system" fn(VkDevice, VkDeviceMemory),
    CreateShaderModule:
        unsafe extern "system" fn(VkDevice, *const VkShaderModuleCreateInfo, *const c_void, *mut VkShaderModule) -> i32,
    DestroyShaderModule: unsafe extern "system" fn(VkDevice, VkShaderModule, *const c_void),
    CreateDescriptorSetLayout:
        unsafe extern "system" fn(VkDevice, *const VkDescriptorSetLayoutCreateInfo, *const c_void, *mut VkDescriptorSetLayout) -> i32,
    DestroyDescriptorSetLayout:
        unsafe extern "system" fn(VkDevice, VkDescriptorSetLayout, *const c_void),
    CreatePipelineLayout:
        unsafe extern "system" fn(VkDevice, *const VkPipelineLayoutCreateInfo, *const c_void, *mut VkPipelineLayout) -> i32,
    DestroyPipelineLayout:
        unsafe extern "system" fn(VkDevice, VkPipelineLayout, *const c_void),
    CreateComputePipelines:
        unsafe extern "system" fn(VkDevice, VkPipelineCache, u32, *const VkComputePipelineCreateInfo, *const c_void, *mut VkPipeline) -> i32,
    DestroyPipeline: unsafe extern "system" fn(VkDevice, VkPipeline, *const c_void),
    CreateDescriptorPool:
        unsafe extern "system" fn(VkDevice, *const VkDescriptorPoolCreateInfo, *const c_void, *mut VkDescriptorPool) -> i32,
    DestroyDescriptorPool:
        unsafe extern "system" fn(VkDevice, VkDescriptorPool, *const c_void),
    AllocateDescriptorSets:
        unsafe extern "system" fn(VkDevice, *const VkDescriptorSetAllocateInfo, *mut VkDescriptorSet) -> i32,
    UpdateDescriptorSets:
        unsafe extern "system" fn(VkDevice, u32, *const VkWriteDescriptorSet, u32, *const c_void),
    CreateCommandPool:
        unsafe extern "system" fn(VkDevice, *const VkCommandPoolCreateInfo, *const c_void, *mut VkCommandPool) -> i32,
    DestroyCommandPool: unsafe extern "system" fn(VkDevice, VkCommandPool, *const c_void),
    AllocateCommandBuffers:
        unsafe extern "system" fn(VkDevice, *const VkCommandBufferAllocateInfo, *mut VkCommandBuffer) -> i32,
    ResetCommandBuffer: unsafe extern "system" fn(VkCommandBuffer, u32) -> i32,
    BeginCommandBuffer:
        unsafe extern "system" fn(VkCommandBuffer, *const VkCommandBufferBeginInfo) -> i32,
    EndCommandBuffer: unsafe extern "system" fn(VkCommandBuffer) -> i32,
    CmdBindPipeline: unsafe extern "system" fn(VkCommandBuffer, u32, VkPipeline),
    CmdBindDescriptorSets:
        unsafe extern "system" fn(VkCommandBuffer, u32, VkPipelineLayout, u32, u32, *const VkDescriptorSet, u32, *const u32),
    CmdPushConstants:
        unsafe extern "system" fn(VkCommandBuffer, VkPipelineLayout, u32, u32, u32, *const c_void),
    CmdDispatch: unsafe extern "system" fn(VkCommandBuffer, u32, u32, u32),
    QueueSubmit: unsafe extern "system" fn(VkQueue, u32, *const VkSubmitInfo, VkFence) -> i32,
    CreateFence:
        unsafe extern "system" fn(VkDevice, *const VkFenceCreateInfo, *const c_void, *mut VkFence) -> i32,
    DestroyFence: unsafe extern "system" fn(VkDevice, VkFence, *const c_void),
    WaitForFences:
        unsafe extern "system" fn(VkDevice, u32, *const VkFence, u32, u64) -> i32,
    ResetFences: unsafe extern "system" fn(VkDevice, u32, *const VkFence) -> i32,
}

// ---- Dynamic loader ----

extern "system" {
    fn LoadLibraryA(name: *const u8) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
}

macro_rules! vk_load {
    ($gpa:expr, $inst:expr, $name:literal) => {{
        let ptr = ($gpa)($inst, concat!("vk", $name, "\0").as_ptr());
        assert!(!ptr.is_null(), concat!("vk", $name, " not found"));
        std::mem::transmute(ptr)
    }};
}

unsafe fn load_vulkan_fns(instance: VkInstance, gpa: PFN_vkGetInstanceProcAddr) -> VkFns {
    VkFns {
        DestroyInstance: vk_load!(gpa, instance, "DestroyInstance"),
        EnumeratePhysicalDevices: vk_load!(gpa, instance, "EnumeratePhysicalDevices"),
        GetPhysicalDeviceProperties: vk_load!(gpa, instance, "GetPhysicalDeviceProperties"),
        GetPhysicalDeviceMemoryProperties: vk_load!(gpa, instance, "GetPhysicalDeviceMemoryProperties"),
        GetPhysicalDeviceQueueFamilyProperties: vk_load!(gpa, instance, "GetPhysicalDeviceQueueFamilyProperties"),
        CreateDevice: vk_load!(gpa, instance, "CreateDevice"),
        DestroyDevice: vk_load!(gpa, instance, "DestroyDevice"),
        GetDeviceQueue: vk_load!(gpa, instance, "GetDeviceQueue"),
        CreateBuffer: vk_load!(gpa, instance, "CreateBuffer"),
        DestroyBuffer: vk_load!(gpa, instance, "DestroyBuffer"),
        GetBufferMemoryRequirements: vk_load!(gpa, instance, "GetBufferMemoryRequirements"),
        AllocateMemory: vk_load!(gpa, instance, "AllocateMemory"),
        FreeMemory: vk_load!(gpa, instance, "FreeMemory"),
        BindBufferMemory: vk_load!(gpa, instance, "BindBufferMemory"),
        MapMemory: vk_load!(gpa, instance, "MapMemory"),
        UnmapMemory: vk_load!(gpa, instance, "UnmapMemory"),
        CreateShaderModule: vk_load!(gpa, instance, "CreateShaderModule"),
        DestroyShaderModule: vk_load!(gpa, instance, "DestroyShaderModule"),
        CreateDescriptorSetLayout: vk_load!(gpa, instance, "CreateDescriptorSetLayout"),
        DestroyDescriptorSetLayout: vk_load!(gpa, instance, "DestroyDescriptorSetLayout"),
        CreatePipelineLayout: vk_load!(gpa, instance, "CreatePipelineLayout"),
        DestroyPipelineLayout: vk_load!(gpa, instance, "DestroyPipelineLayout"),
        CreateComputePipelines: vk_load!(gpa, instance, "CreateComputePipelines"),
        DestroyPipeline: vk_load!(gpa, instance, "DestroyPipeline"),
        CreateDescriptorPool: vk_load!(gpa, instance, "CreateDescriptorPool"),
        DestroyDescriptorPool: vk_load!(gpa, instance, "DestroyDescriptorPool"),
        AllocateDescriptorSets: vk_load!(gpa, instance, "AllocateDescriptorSets"),
        UpdateDescriptorSets: vk_load!(gpa, instance, "UpdateDescriptorSets"),
        CreateCommandPool: vk_load!(gpa, instance, "CreateCommandPool"),
        DestroyCommandPool: vk_load!(gpa, instance, "DestroyCommandPool"),
        AllocateCommandBuffers: vk_load!(gpa, instance, "AllocateCommandBuffers"),
        ResetCommandBuffer: vk_load!(gpa, instance, "ResetCommandBuffer"),
        BeginCommandBuffer: vk_load!(gpa, instance, "BeginCommandBuffer"),
        EndCommandBuffer: vk_load!(gpa, instance, "EndCommandBuffer"),
        CmdBindPipeline: vk_load!(gpa, instance, "CmdBindPipeline"),
        CmdBindDescriptorSets: vk_load!(gpa, instance, "CmdBindDescriptorSets"),
        CmdPushConstants: vk_load!(gpa, instance, "CmdPushConstants"),
        CmdDispatch: vk_load!(gpa, instance, "CmdDispatch"),
        QueueSubmit: vk_load!(gpa, instance, "QueueSubmit"),
        CreateFence: vk_load!(gpa, instance, "CreateFence"),
        DestroyFence: vk_load!(gpa, instance, "DestroyFence"),
        WaitForFences: vk_load!(gpa, instance, "WaitForFences"),
        ResetFences: vk_load!(gpa, instance, "ResetFences"),
    }
}

// ---- SPIR-V bytecode for F32 mat-vec compute shader ----
// Equivalent GLSL:
//   #version 450
//   layout(local_size_x = 256) in;
//   layout(set=0, binding=0) readonly buffer M { float d[]; } mat;
//   layout(set=0, binding=1) readonly buffer V { float d[]; } vec;
//   layout(set=0, binding=2) writeonly buffer O { float d[]; } out_buf;
//   layout(push_constant) uniform PC { uint rows; uint cols; };
//   void main() {
//       uint row = gl_GlobalInvocationID.x;
//       if (row >= rows) return;
//       float sum = 0.0;
//       for (uint c = 0; c < cols; c++)
//           sum += mat.d[row * cols + c] * vec.d[c];
//       out_buf.d[row] = sum;
//   }

fn mat_vec_spirv() -> Vec<u32> {
    // IDs:  1=void 2=bool 3=u32 4=f32 5=v3u 6/7/8=rtarrays 9/10/11=buf structs
    //       12=PC struct 13-15=ptr SB structs 16=ptr SB f32 17=ptr Input v3u
    //       18=ptr PC struct 19=ptr PC u32 20=fn_void
    //       21=gid_var 22=mat_var 23=vec_var 24=out_var 25=pc_var
    //       26=const_0u 27=const_1u 28=const_0f
    //       29=main 30=entry_lbl 31..55=SSA values
    vec![
        // Header
        0x07230203, 0x00010300, 0, 56, 0,
        // OpCapability Shader
        0x00020011, 1,
        // OpMemoryModel Logical GLSL450
        0x0003000E, 0, 1,
        // OpEntryPoint GLCompute %29 "main" %21
        0x0006000F, 5, 29, 0x6E69616D, 0x00000000, 21,
        // OpExecutionMode %29 LocalSize 256 1 1
        0x00060010, 29, 17, 256, 1, 1,

        // ---- Decorations ----
        // %21 BuiltIn GlobalInvocationId
        0x00040047, 21, 11, 28,
        // ArrayStride 4 on runtime arrays
        0x00040047, 6, 6, 4,
        0x00040047, 7, 6, 4,
        0x00040047, 8, 6, 4,
        // Member 0 Offset 0 on buffer structs
        0x00050048, 9, 0, 35, 0,
        0x00050048, 10, 0, 35, 0,
        0x00050048, 11, 0, 35, 0,
        // Block decoration on buffer structs
        0x00030047, 9, 2,
        0x00030047, 10, 2,
        0x00030047, 11, 2,
        // Descriptor set + binding for buffer variables
        0x00040047, 22, 34, 0,  // mat: set=0
        0x00040047, 22, 33, 0,  // mat: binding=0
        0x00040047, 23, 34, 0,  // vec: set=0
        0x00040047, 23, 33, 1,  // vec: binding=1
        0x00040047, 24, 34, 0,  // out: set=0
        0x00040047, 24, 33, 2,  // out: binding=2
        // Push constant struct
        0x00050048, 12, 0, 35, 0,  // member 0 offset 0
        0x00050048, 12, 1, 35, 4,  // member 1 offset 4
        0x00030047, 12, 2,          // Block

        // ---- Types ----
        // %1 = void
        0x00020013, 1,
        // %2 = bool
        0x00020014, 2,
        // %3 = uint32
        0x00040015, 3, 32, 0,
        // %4 = float32
        0x00030016, 4, 32,
        // %5 = vec3<uint>
        0x00040017, 5, 3, 3,
        // %6,%7,%8 = RuntimeArray<float>
        0x0003001D, 6, 4,
        0x0003001D, 7, 4,
        0x0003001D, 8, 4,
        // %9,%10,%11 = struct { RuntimeArray<float> }
        0x0003001E, 9, 6,
        0x0003001E, 10, 7,
        0x0003001E, 11, 8,
        // %12 = struct { uint, uint }  (PushConst)
        0x0004001E, 12, 3, 3,
        // Pointer types
        // %13 = ptr StorageBuffer %9
        0x00040020, 13, 12, 9,
        // %14 = ptr StorageBuffer %10
        0x00040020, 14, 12, 10,
        // %15 = ptr StorageBuffer %11
        0x00040020, 15, 12, 11,
        // %16 = ptr StorageBuffer float
        0x00040020, 16, 12, 4,
        // %17 = ptr Input vec3<uint>
        0x00040020, 17, 1, 5,
        // %18 = ptr PushConstant %12
        0x00040020, 18, 9, 12,
        // %19 = ptr PushConstant uint
        0x00040020, 19, 9, 3,
        // %20 = fn() -> void
        0x00030021, 20, 1,

        // ---- Variables ----
        // %21 = gl_GlobalInvocationID (Input)
        0x0004003B, 17, 21, 1,
        // %22 = mat (StorageBuffer)
        0x0004003B, 13, 22, 12,
        // %23 = vec (StorageBuffer)
        0x0004003B, 14, 23, 12,
        // %24 = out (StorageBuffer)
        0x0004003B, 15, 24, 12,
        // %25 = push_constant (PushConstant)
        0x0004003B, 18, 25, 9,

        // ---- Constants ----
        // %26 = 0u
        0x0004002B, 3, 26, 0,
        // %27 = 1u
        0x0004002B, 3, 27, 1,
        // %28 = 0.0f
        0x0004002B, 4, 28, 0,

        // ---- Function ----
        // %29 = OpFunction void None %20
        0x00050036, 1, 29, 0, 20,
        // %30 = OpLabel (entry)
        0x000200F8, 30,

        // Load gl_GlobalInvocationID
        // %31 = OpLoad vec3<uint> %21
        0x0004003D, 5, 31, 21,
        // %32 = OpCompositeExtract uint %31 0  (row = gid.x)
        0x00050051, 3, 32, 31, 0,

        // Load push constants
        // %33 = OpAccessChain ptr_PC_uint %25 %26  (rows)
        0x00050041, 19, 33, 25, 26,
        // %34 = OpLoad uint %33
        0x0004003D, 3, 34, 33,
        // %35 = OpAccessChain ptr_PC_uint %25 %27  (cols)
        0x00050041, 19, 35, 25, 27,
        // %36 = OpLoad uint %35
        0x0004003D, 3, 36, 35,

        // Bounds check: if (row < rows) goto body else goto merge
        // %37 = OpULessThan bool %32 %34
        0x000500B0, 2, 37, 32, 34,
        // OpSelectionMerge %38 None
        0x000300F7, 38, 0,
        // OpBranchConditional %37 %39 %38
        0x000400FA, 37, 39, 38,

        // %39 = OpLabel (body: compute base)
        0x000200F8, 39,
        // %40 = OpIMul uint %32 %36  (base = row * cols)
        0x00050084, 3, 40, 32, 36,
        // OpBranch %41
        0x000200F9, 41,

        // %41 = OpLabel (loop header)
        0x000200F8, 41,
        // %42 = OpPhi uint [%26, %39] [%54, %46]   (c)
        0x000700F5, 3, 42, 26, 39, 54, 46,
        // %43 = OpPhi float [%28, %39] [%53, %46]  (sum)
        0x000700F5, 4, 43, 28, 39, 53, 46,
        // %44 = OpULessThan bool %42 %36  (c < cols)
        0x000500B0, 2, 44, 42, 36,
        // OpLoopMerge %45 %46 None
        0x000400F6, 45, 46, 0,
        // OpBranchConditional %44 %46 %45
        0x000400FA, 44, 46, 45,

        // %46 = OpLabel (loop body / continue)
        0x000200F8, 46,
        // %47 = OpIAdd uint %40 %42  (idx = base + c)
        0x00050080, 3, 47, 40, 42,
        // %48 = OpAccessChain ptr_SB_f %22 %26 %47  (mat.d[idx])
        0x00060041, 16, 48, 22, 26, 47,
        // %49 = OpLoad float %48
        0x0004003D, 4, 49, 48,
        // %50 = OpAccessChain ptr_SB_f %23 %26 %42  (vec.d[c])
        0x00060041, 16, 50, 23, 26, 42,
        // %51 = OpLoad float %50
        0x0004003D, 4, 51, 50,
        // %52 = OpFMul float %49 %51
        0x00050085, 4, 52, 49, 51,
        // %53 = OpFAdd float %43 %52  (sum += product)
        0x00050081, 4, 53, 43, 52,
        // %54 = OpIAdd uint %42 %27  (c++)
        0x00050080, 3, 54, 42, 27,
        // OpBranch %41
        0x000200F9, 41,

        // %45 = OpLabel (loop merge: store result)
        0x000200F8, 45,
        // %55 = OpAccessChain ptr_SB_f %24 %26 %32  (out.d[row])
        0x00060041, 16, 55, 24, 26, 32,
        // OpStore %55 %43
        0x0003003E, 55, 43,
        // OpBranch %38
        0x000200F9, 38,

        // %38 = OpLabel (selection merge: return)
        0x000200F8, 38,
        // OpReturn
        0x000100FD,
        // OpFunctionEnd
        0x00010038,
    ]
}

// ---- GPU Context ----

pub struct GpuContext {
    fns: VkFns,
    instance: VkInstance,
    device: VkDevice,
    queue: VkQueue,
    pipeline: VkPipeline,
    pipeline_layout: VkPipelineLayout,
    desc_set_layout: VkDescriptorSetLayout,
    desc_pool: VkDescriptorPool,
    desc_set: VkDescriptorSet,
    mat_buffer: VkBuffer,
    mat_memory: VkDeviceMemory,
    vec_buffer: VkBuffer,
    vec_memory: VkDeviceMemory,
    vec_ptr: *mut f32,
    out_buffer: VkBuffer,
    out_memory: VkDeviceMemory,
    out_ptr: *mut f32,
    cmd_pool: VkCommandPool,
    cmd_buf: VkCommandBuffer,
    fence: VkFence,
    rows: u32,
    cols: u32,
    pub device_name: String,
}

// GpuContext is only accessed from one thread at a time.
// The raw pointers (vec_ptr, out_ptr) are mapped GPU memory used only during mat_vec_mul_into.
// Rwkv7Model is shared via Arc for batch mode, but GPU is None in batch mode (single-file only).
unsafe impl Send for GpuContext {}
unsafe impl Sync for GpuContext {}

impl GpuContext {
    /// Try to initialize Vulkan and upload a weight matrix to the GPU.
    /// Returns None if Vulkan is unavailable or initialization fails.
    pub fn try_init(mat_data: &[f32], rows: usize, cols: usize) -> Option<Self> {
        unsafe { Self::init_inner(mat_data, rows, cols) }
    }

    unsafe fn init_inner(mat_data: &[f32], rows: usize, cols: usize) -> Option<Self> {
        assert_eq!(mat_data.len(), rows * cols);

        // Step 1: Load vulkan-1.dll
        let lib = LoadLibraryA(b"vulkan-1.dll\0".as_ptr());
        if lib.is_null() { return None; }
        let gpa_raw = GetProcAddress(lib, b"vkGetInstanceProcAddr\0".as_ptr());
        if gpa_raw.is_null() { return None; }
        let gpa: PFN_vkGetInstanceProcAddr = std::mem::transmute(gpa_raw);

        // Step 2: Create instance
        let create_instance: unsafe extern "system" fn(
            *const VkInstanceCreateInfo, *const c_void, *mut VkInstance,
        ) -> i32 = std::mem::transmute(gpa(null_mut(), b"vkCreateInstance\0".as_ptr()));

        let app_info = VkApplicationInfo {
            s_type: VK_STRUCTURE_TYPE_APPLICATION_INFO,
            p_next: null(),
            p_application_name: b"redukt\0".as_ptr(),
            application_version: 1,
            p_engine_name: b"redukt\0".as_ptr(),
            engine_version: 1,
            api_version: VK_API_VERSION_1_1,
        };
        let create_info = VkInstanceCreateInfo {
            s_type: VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            p_next: null(),
            flags: 0,
            p_application_info: &app_info,
            enabled_layer_count: 0,
            pp_enabled_layer_names: null(),
            enabled_extension_count: 0,
            pp_enabled_extension_names: null(),
        };
        let mut instance: VkInstance = null_mut();
        if create_instance(&create_info, null(), &mut instance) != VK_SUCCESS {
            return None;
        }

        // Step 3: Load all function pointers
        let f = load_vulkan_fns(instance, gpa);

        // Step 4: Enumerate physical devices — prefer integrated GPU
        let mut count = 0u32;
        (f.EnumeratePhysicalDevices)(instance, &mut count, null_mut());
        if count == 0 {
            (f.DestroyInstance)(instance, null());
            return None;
        }
        let mut devices = vec![null_mut(); count as usize];
        (f.EnumeratePhysicalDevices)(instance, &mut count, devices.as_mut_ptr());

        let mut chosen: VkPhysicalDevice = null_mut();
        let mut chosen_name = String::new();
        let mut mem_props: VkPhysicalDeviceMemoryProperties = std::mem::zeroed();

        for &dev in &devices {
            let mut props: VkPhysicalDeviceProperties = std::mem::zeroed();
            (f.GetPhysicalDeviceProperties)(dev, &mut props);
            let name = std::ffi::CStr::from_ptr(props.device_name.as_ptr() as *const i8)
                .to_string_lossy().to_string();
            if props.device_type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU {
                chosen = dev;
                chosen_name = name;
                break;
            }
            if chosen.is_null() {
                chosen = dev;
                chosen_name = name;
            }
        }
        if chosen.is_null() {
            (f.DestroyInstance)(instance, null());
            return None;
        }
        (f.GetPhysicalDeviceMemoryProperties)(chosen, &mut mem_props);

        // Step 5: Find compute queue family
        let mut qf_count = 0u32;
        (f.GetPhysicalDeviceQueueFamilyProperties)(chosen, &mut qf_count, null_mut());
        let mut qf_props = vec![
            VkQueueFamilyProperties {
                queue_flags: 0, queue_count: 0,
                timestamp_valid_bits: 0, min_image_transfer_granularity: [0; 3],
            };
            qf_count as usize
        ];
        (f.GetPhysicalDeviceQueueFamilyProperties)(
            chosen, &mut qf_count, qf_props.as_mut_ptr(),
        );
        let compute_family = qf_props.iter().position(|q| q.queue_flags & VK_QUEUE_COMPUTE_BIT != 0);
        let compute_family = match compute_family {
            Some(idx) => idx as u32,
            None => {
                (f.DestroyInstance)(instance, null());
                return None;
            }
        };

        // Step 6: Create logical device + queue
        let priority = 1.0f32;
        let queue_ci = VkDeviceQueueCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            p_next: null(),
            flags: 0,
            queue_family_index: compute_family,
            queue_count: 1,
            p_queue_priorities: &priority,
        };
        let device_ci = VkDeviceCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            p_next: null(),
            flags: 0,
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_ci,
            enabled_layer_count: 0,
            pp_enabled_layer_names: null(),
            enabled_extension_count: 0,
            pp_enabled_extension_names: null(),
            p_enabled_features: null(),
        };
        let mut device: VkDevice = null_mut();
        if (f.CreateDevice)(chosen, &device_ci, null(), &mut device) != VK_SUCCESS {
            (f.DestroyInstance)(instance, null());
            return None;
        }
        let mut queue: VkQueue = null_mut();
        (f.GetDeviceQueue)(device, compute_family, 0, &mut queue);

        // Helper: find memory type with required properties
        let find_mem_type = |type_bits: u32, required: u32| -> Option<u32> {
            for i in 0..mem_props.memory_type_count {
                if (type_bits & (1 << i)) != 0
                    && (mem_props.memory_types[i as usize].property_flags & required) == required
                {
                    return Some(i);
                }
            }
            None
        };

        // Helper: create buffer + allocate + bind + optionally map
        let create_buffer = |size: VkDeviceSize, map: bool| -> Option<(VkBuffer, VkDeviceMemory, *mut f32)> {
            let buf_ci = VkBufferCreateInfo {
                s_type: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                p_next: null(),
                flags: 0,
                size,
                usage: VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                sharing_mode: VK_SHARING_MODE_EXCLUSIVE,
                queue_family_index_count: 0,
                p_queue_family_indices: null(),
            };
            let mut buf: VkBuffer = VK_NULL_HANDLE;
            if (f.CreateBuffer)(device, &buf_ci, null(), &mut buf) != VK_SUCCESS {
                return None;
            }
            let mut req: VkMemoryRequirements = std::mem::zeroed();
            (f.GetBufferMemoryRequirements)(device, buf, &mut req);
            let mem_type = find_mem_type(
                req.memory_type_bits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            )?;
            let alloc_info = VkMemoryAllocateInfo {
                s_type: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                p_next: null(),
                allocation_size: req.size,
                memory_type_index: mem_type,
            };
            let mut memory: VkDeviceMemory = VK_NULL_HANDLE;
            if (f.AllocateMemory)(device, &alloc_info, null(), &mut memory) != VK_SUCCESS {
                (f.DestroyBuffer)(device, buf, null());
                return None;
            }
            if (f.BindBufferMemory)(device, buf, memory, 0) != VK_SUCCESS {
                (f.FreeMemory)(device, memory, null());
                (f.DestroyBuffer)(device, buf, null());
                return None;
            }
            let mut ptr: *mut c_void = null_mut();
            if map {
                if (f.MapMemory)(device, memory, 0, size, 0, &mut ptr) != VK_SUCCESS {
                    (f.FreeMemory)(device, memory, null());
                    (f.DestroyBuffer)(device, buf, null());
                    return None;
                }
            }
            Some((buf, memory, ptr as *mut f32))
        };

        // Step 7: Create buffers
        let mat_size = (rows * cols * 4) as VkDeviceSize;
        let vec_size = (cols * 4) as VkDeviceSize;
        let out_size = (rows * 4) as VkDeviceSize;

        let (mat_buffer, mat_memory, mat_ptr) = create_buffer(mat_size, true)?;
        let (vec_buffer, vec_memory, vec_ptr) = create_buffer(vec_size, true)?;
        let (out_buffer, out_memory, out_ptr) = create_buffer(out_size, true)?;

        // Upload matrix data
        std::ptr::copy_nonoverlapping(mat_data.as_ptr(), mat_ptr, rows * cols);
        // Unmap matrix (no longer needed from CPU)
        (f.UnmapMemory)(device, mat_memory);

        // Step 8: Create compute pipeline
        let spirv = mat_vec_spirv();
        let shader_ci = VkShaderModuleCreateInfo {
            s_type: VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            p_next: null(),
            flags: 0,
            code_size: spirv.len() * 4,
            p_code: spirv.as_ptr(),
        };
        let mut shader: VkShaderModule = VK_NULL_HANDLE;
        if (f.CreateShaderModule)(device, &shader_ci, null(), &mut shader) != VK_SUCCESS {
            return None;
        }

        // Descriptor set layout: 3 storage buffers
        let bindings = [
            VkDescriptorSetLayoutBinding {
                binding: 0, descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptor_count: 1, stage_flags: VK_SHADER_STAGE_COMPUTE_BIT,
                p_immutable_samplers: null(),
            },
            VkDescriptorSetLayoutBinding {
                binding: 1, descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptor_count: 1, stage_flags: VK_SHADER_STAGE_COMPUTE_BIT,
                p_immutable_samplers: null(),
            },
            VkDescriptorSetLayoutBinding {
                binding: 2, descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptor_count: 1, stage_flags: VK_SHADER_STAGE_COMPUTE_BIT,
                p_immutable_samplers: null(),
            },
        ];
        let desc_layout_ci = VkDescriptorSetLayoutCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: null(),
            flags: 0,
            binding_count: 3,
            p_bindings: bindings.as_ptr(),
        };
        let mut desc_set_layout: VkDescriptorSetLayout = VK_NULL_HANDLE;
        if (f.CreateDescriptorSetLayout)(device, &desc_layout_ci, null(), &mut desc_set_layout) != VK_SUCCESS {
            return None;
        }

        // Pipeline layout: descriptor set + push constants (rows, cols)
        let pc_range = VkPushConstantRange {
            stage_flags: VK_SHADER_STAGE_COMPUTE_BIT,
            offset: 0,
            size: 8, // 2 × u32
        };
        let pl_ci = VkPipelineLayoutCreateInfo {
            s_type: VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            p_next: null(),
            flags: 0,
            set_layout_count: 1,
            p_set_layouts: &desc_set_layout,
            push_constant_range_count: 1,
            p_push_constant_ranges: &pc_range,
        };
        let mut pipeline_layout: VkPipelineLayout = VK_NULL_HANDLE;
        if (f.CreatePipelineLayout)(device, &pl_ci, null(), &mut pipeline_layout) != VK_SUCCESS {
            return None;
        }

        // Compute pipeline
        let stage = VkPipelineShaderStageCreateInfo {
            s_type: VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: null(),
            flags: 0,
            stage: VK_SHADER_STAGE_COMPUTE_BIT,
            module: shader,
            p_name: b"main\0".as_ptr(),
            p_specialization_info: null(),
        };
        let pipe_ci = VkComputePipelineCreateInfo {
            s_type: VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            p_next: null(),
            flags: 0,
            stage,
            layout: pipeline_layout,
            base_pipeline_handle: VK_NULL_HANDLE,
            base_pipeline_index: -1,
        };
        let mut pipeline: VkPipeline = VK_NULL_HANDLE;
        if (f.CreateComputePipelines)(device, VK_NULL_HANDLE, 1, &pipe_ci, null(), &mut pipeline) != VK_SUCCESS {
            return None;
        }
        // Shader module no longer needed
        (f.DestroyShaderModule)(device, shader, null());

        // Descriptor pool + set
        let pool_size = VkDescriptorPoolSize {
            descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptor_count: 3,
        };
        let dp_ci = VkDescriptorPoolCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            p_next: null(),
            flags: 0,
            max_sets: 1,
            pool_size_count: 1,
            p_pool_sizes: &pool_size,
        };
        let mut desc_pool: VkDescriptorPool = VK_NULL_HANDLE;
        if (f.CreateDescriptorPool)(device, &dp_ci, null(), &mut desc_pool) != VK_SUCCESS {
            return None;
        }
        let ds_ai = VkDescriptorSetAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: null(),
            descriptor_pool: desc_pool,
            descriptor_set_count: 1,
            p_set_layouts: &desc_set_layout,
        };
        let mut desc_set: VkDescriptorSet = VK_NULL_HANDLE;
        if (f.AllocateDescriptorSets)(device, &ds_ai, &mut desc_set) != VK_SUCCESS {
            return None;
        }

        // Write descriptors
        let buf_infos = [
            VkDescriptorBufferInfo { buffer: mat_buffer, offset: 0, range: mat_size },
            VkDescriptorBufferInfo { buffer: vec_buffer, offset: 0, range: vec_size },
            VkDescriptorBufferInfo { buffer: out_buffer, offset: 0, range: out_size },
        ];
        let writes = [
            VkWriteDescriptorSet {
                s_type: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                p_next: null(),
                dst_set: desc_set, dst_binding: 0, dst_array_element: 0,
                descriptor_count: 1, descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                p_image_info: null(), p_buffer_info: &buf_infos[0], p_texel_buffer_view: null(),
            },
            VkWriteDescriptorSet {
                s_type: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                p_next: null(),
                dst_set: desc_set, dst_binding: 1, dst_array_element: 0,
                descriptor_count: 1, descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                p_image_info: null(), p_buffer_info: &buf_infos[1], p_texel_buffer_view: null(),
            },
            VkWriteDescriptorSet {
                s_type: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                p_next: null(),
                dst_set: desc_set, dst_binding: 2, dst_array_element: 0,
                descriptor_count: 1, descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                p_image_info: null(), p_buffer_info: &buf_infos[2], p_texel_buffer_view: null(),
            },
        ];
        (f.UpdateDescriptorSets)(device, 3, writes.as_ptr(), 0, null());

        // Step 9: Command pool + buffer + fence
        let cp_ci = VkCommandPoolCreateInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            p_next: null(),
            flags: VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queue_family_index: compute_family,
        };
        let mut cmd_pool: VkCommandPool = VK_NULL_HANDLE;
        if (f.CreateCommandPool)(device, &cp_ci, null(), &mut cmd_pool) != VK_SUCCESS {
            return None;
        }
        let cb_ai = VkCommandBufferAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: null(),
            command_pool: cmd_pool,
            level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            command_buffer_count: 1,
        };
        let mut cmd_buf: VkCommandBuffer = null_mut();
        if (f.AllocateCommandBuffers)(device, &cb_ai, &mut cmd_buf) != VK_SUCCESS {
            return None;
        }
        let fence_ci = VkFenceCreateInfo {
            s_type: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            p_next: null(),
            flags: 0,
        };
        let mut fence: VkFence = VK_NULL_HANDLE;
        if (f.CreateFence)(device, &fence_ci, null(), &mut fence) != VK_SUCCESS {
            return None;
        }

        Some(GpuContext {
            fns: f,
            instance,
            device,
            queue,
            pipeline,
            pipeline_layout,
            desc_set_layout,
            desc_pool,
            desc_set,
            mat_buffer,
            mat_memory,
            vec_buffer,
            vec_memory,
            vec_ptr,
            out_buffer,
            out_memory,
            out_ptr,
            cmd_pool,
            cmd_buf,
            fence,
            rows: rows as u32,
            cols: cols as u32,
            device_name: chosen_name,
        })
    }

    /// Execute GPU mat-vec: out = matrix @ vec.
    /// Writes input vec to GPU, dispatches compute shader, reads result.
    pub fn mat_vec_mul_into(&self, out: &mut [f32], input_vec: &[f32]) {
        assert_eq!(input_vec.len(), self.cols as usize);
        assert!(out.len() >= self.rows as usize);

        unsafe {
            let f = &self.fns;
            let cols = self.cols as usize;
            let rows = self.rows as usize;

            // Upload input vector
            std::ptr::copy_nonoverlapping(input_vec.as_ptr(), self.vec_ptr, cols);

            // Record command buffer
            (f.ResetCommandBuffer)(self.cmd_buf, 0);
            let begin_info = VkCommandBufferBeginInfo {
                s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                p_next: null(),
                flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                p_inheritance_info: null(),
            };
            (f.BeginCommandBuffer)(self.cmd_buf, &begin_info);
            (f.CmdBindPipeline)(self.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
            (f.CmdBindDescriptorSets)(
                self.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                self.pipeline_layout, 0, 1, &self.desc_set, 0, null(),
            );
            let pc = [self.rows, self.cols];
            (f.CmdPushConstants)(
                self.cmd_buf, self.pipeline_layout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0, 8,
                pc.as_ptr() as *const c_void,
            );
            let groups = (rows + 255) / 256;
            (f.CmdDispatch)(self.cmd_buf, groups as u32, 1, 1);
            (f.EndCommandBuffer)(self.cmd_buf);

            // Submit and wait
            let submit = VkSubmitInfo {
                s_type: VK_STRUCTURE_TYPE_SUBMIT_INFO,
                p_next: null(),
                wait_semaphore_count: 0,
                p_wait_semaphores: null(),
                p_wait_dst_stage_mask: null(),
                command_buffer_count: 1,
                p_command_buffers: &self.cmd_buf,
                signal_semaphore_count: 0,
                p_signal_semaphores: null(),
            };
            (f.ResetFences)(self.device, 1, &self.fence);
            (f.QueueSubmit)(self.queue, 1, &submit, self.fence);
            (f.WaitForFences)(self.device, 1, &self.fence, 1, u64::MAX);

            // Read result
            std::ptr::copy_nonoverlapping(self.out_ptr, out.as_mut_ptr(), rows);
        }
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        unsafe {
            let f = &self.fns;
            (f.DestroyFence)(self.device, self.fence, null());
            (f.DestroyCommandPool)(self.device, self.cmd_pool, null());
            (f.DestroyDescriptorPool)(self.device, self.desc_pool, null());
            (f.DestroyPipeline)(self.device, self.pipeline, null());
            (f.DestroyPipelineLayout)(self.device, self.pipeline_layout, null());
            (f.DestroyDescriptorSetLayout)(self.device, self.desc_set_layout, null());
            // Unmap and free memory
            (f.UnmapMemory)(self.device, self.vec_memory);
            (f.UnmapMemory)(self.device, self.out_memory);
            (f.FreeMemory)(self.device, self.mat_memory, null());
            (f.FreeMemory)(self.device, self.vec_memory, null());
            (f.FreeMemory)(self.device, self.out_memory, null());
            (f.DestroyBuffer)(self.device, self.mat_buffer, null());
            (f.DestroyBuffer)(self.device, self.vec_buffer, null());
            (f.DestroyBuffer)(self.device, self.out_buffer, null());
            (f.DestroyDevice)(self.device, null());
            (f.DestroyInstance)(self.instance, null());
        }
    }
}
