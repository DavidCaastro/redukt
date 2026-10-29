//! Optional GPU offload for compute-bound matmuls via Vulkan compute shaders.
//! Activated with `--gpu` flag. Falls back to CPU if Vulkan is unavailable.
//! Zero compile-time dependencies: vulkan-1.dll loaded dynamically at runtime.

#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
mod vulkan;

#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
pub use vulkan::GpuContext;

#[cfg(not(all(target_os = "windows", target_arch = "x86_64")))]
pub struct GpuContext;

#[cfg(not(all(target_os = "windows", target_arch = "x86_64")))]
impl GpuContext {
    pub fn try_init(_data: &[f32], _rows: usize, _cols: usize) -> Option<Self> {
        None
    }
    pub fn mat_vec_mul_into(&self, _out: &mut [f32], _vec: &[f32]) {
        unreachable!()
    }
}
