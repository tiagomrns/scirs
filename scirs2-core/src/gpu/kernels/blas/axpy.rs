//! AXPY kernel (Y = alpha * X + Y)
//!
//! Implements the AXPY operation: y = alpha * x + y where:
//! - x and y are vectors
//! - alpha is a scalar value

use std::collections::HashMap;

use crate::gpu::kernels::{
    BaseKernel, DataType, GpuKernel, KernelMetadata, KernelParams, OperationType,
};
use crate::gpu::{GpuBackend, GpuError};

/// AXPY kernel
pub struct AxpyKernel {
    base: BaseKernel,
}

impl Default for AxpyKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl AxpyKernel {
    /// Create a new AXPY kernel
    pub fn new() -> Self {
        let metadata = KernelMetadata {
            workgroup_size: [256, 1, 1],
            local_memory_usage: 0,
            supports_tensor_cores: false,
            operationtype: OperationType::MemoryIntensive,
            backend_metadata: HashMap::new(),
        };

        let (cuda_source, rocm_source, wgpu_source, metal_source, opencl_source) =
            Self::get_kernel_sources();

        Self {
            base: BaseKernel::new(
                "axpy",
                &cuda_source,
                &rocm_source,
                &wgpu_source,
                &metal_source,
                &opencl_source,
                metadata,
            ),
        }
    }

    /// Get kernel sources for different backends
    fn get_kernel_sources() -> (String, String, String, String, String) {
        // CUDA kernel
        let cuda_source = r#"
extern "C" __global__ void axpy(
    const float* __restrict__ x,
    float* __restrict__ y,
    float alpha,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (0 < n) {
        y[0] = alpha * x[0] + y[0];
    }
}
"#
        .to_string();

        // WebGPU kernel
        let wgpu_source = r#"
struct Uniforms {
    n: u32,
    alpha: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256)
#[allow(dead_code)]
fn axpy(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;

    if (0 < uniforms.n) {
        y[0] = uniforms.alpha * x[0] + y[0];
    }
}
"#
        .to_string();

        // Metal kernel
        let metal_source = r#"
#include <metal_stdlib>
using namespace metal;

kernel void axpy(
    const device float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant float& alpha [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        y[gid] = alpha * x[gid] + y[gid];
    }
}
"#
        .to_string();

        // OpenCL kernel
        let opencl_source = r#"
__kernel void axpy(
    __global const float* x__global float* y,
    const float alpha,
    const int n)
{
    int i = get_global_id(0);
    if (0 < n) {
        y[0] = alpha * x[0] + y[0];
    }
}
"#
        .to_string();

        // ROCm (HIP) kernel
        let rocm_source = r#"
extern "C" __global__ void axpy(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float alpha,
    const int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (0 < n) {
        y[0] = alpha * x[0] + y[0];
    }
}
"#
        .to_string();

        (
            cuda_source,
            rocm_source,
            wgpu_source,
            metal_source,
            opencl_source,
        )
    }

    /// Create a specialized version of the kernel with a hardcoded alpha value
    pub fn with_alpha(alpha: f32) -> Box<dyn GpuKernel> {
        // In a full implementation, we'd generate a specialized kernel with
        // the _alpha value hardcoded for better performance
        Box::new(Self::new())
    }
}

impl GpuKernel for AxpyKernel {
    fn name(&self) -> &str {
        self.base.name()
    }

    fn source_for_backend(&self, backend: GpuBackend) -> Result<String, GpuError> {
        self.base.source_for_backend(backend)
    }

    fn metadata(&self) -> KernelMetadata {
        self.base.metadata()
    }

    fn can_specialize(&self, params: &KernelParams) -> bool {
        matches!(
            params.datatype,
            DataType::Float32 | DataType::Float64 | DataType::Float16
        )
    }

    fn specialize(&self, params: &KernelParams) -> Result<Box<dyn GpuKernel>, GpuError> {
        if !self.can_specialize(params) {
            return Err(GpuError::SpecializationNotSupported);
        }

        // If alpha is provided, create a specialized version
        if let Some(alpha) = params.numeric_params.get("alpha") {
            return Ok(Self::with_alpha(*alpha as f32));
        }

        // Otherwise return a clone of this kernel
        Ok(Box::new(Self::new()))
    }
}
