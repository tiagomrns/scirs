# GPU Kernel Registration Example

This document shows how modules should register their GPU kernels with the centralized registry in scirs2-core.

## Example: FFT Module Registering GPU Kernels

```rust
// In scirs2-fft/src/gpu_kernels.rs

use scirs2_core::gpu_registry::{register_module_kernel, KernelId, KernelSource};
use scirs2_core::gpu::GpuBackend;

// CUDA kernel source for 2D FFT
const FFT2D_CUDA_SOURCE: &str = r#"
extern "C" __global__ void fft2d_c32(
    const float2* __restrict__ input,
    float2* __restrict__ output,
    const int width,
    const int height
) {
    // FFT implementation...
}
"#;

// Register kernels during module initialization
pub fn register_fft_kernels() {
    // Register CUDA version
    register_module_kernel(
        KernelId::new("fft", "fft2d", "c32"),
        KernelSource {
            source: FFT2D_CUDA_SOURCE.to_string(),
            backend: GpuBackend::Cuda,
            entry_point: "fft2d_c32".to_string(),
            workgroup_size: (32, 8, 1),
            shared_memory: 16384,
            uses_tensor_cores: false,
        },
    );
    
    // Register batched variant
    register_module_kernel(
        KernelId::with_variant("fft", "fft2d", "c32", "batched"),
        KernelSource {
            source: FFT2D_BATCHED_CUDA_SOURCE.to_string(),
            backend: GpuBackend::Cuda,
            entry_point: "fft2d_batched_c32".to_string(),
            workgroup_size: (32, 8, 1),
            shared_memory: 16384,
            uses_tensor_cores: false,
        },
    );
    
    // Register Metal version for macOS
    #[cfg(target_os = "macos")]
    register_module_kernel(
        KernelId::new("fft", "fft2d", "c32"),
        KernelSource {
            source: FFT2D_METAL_SOURCE.to_string(),
            backend: GpuBackend::Metal,
            entry_point: "fft2d_c32".to_string(),
            workgroup_size: (32, 8, 1),
            shared_memory: 16384,
            uses_tensor_cores: false,
        },
    );
}

// Call registration function in module init
#[ctor::ctor]
fn init() {
    register_fft_kernels();
}
```

## Using Registered Kernels

```rust
use scirs2_core::gpu_registry::{get_kernel, KernelId};
use scirs2_core::gpu::GpuDevice;

pub fn perform_fft_2d(device: &GpuDevice, data: &Array2<Complex32>) -> Result<Array2<Complex32>, FftError> {
    // Get the appropriate kernel for this device
    let kernel_id = KernelId::new("fft", "fft2d", "c32");
    let kernel = get_kernel(&kernel_id, device)?;
    
    // Launch the kernel
    // ... kernel execution code ...
    
    Ok(result)
}
```

## Best Practices

1. **Register kernels early**: Use `#[ctor::ctor]` or lazy_static to register kernels during module initialization
2. **Support multiple backends**: Register kernels for different GPU backends when possible
3. **Use meaningful IDs**: Follow the naming convention: `module_operation_dtype[__variant]`
4. **Document requirements**: Specify workgroup sizes, shared memory needs, and special features
5. **Version compatibility**: Use variants for different kernel versions or optimizations

## Kernel ID Naming Convention

- **Module**: The owning module (e.g., "linalg", "fft", "neural", "spatial")
- **Operation**: The specific operation (e.g., "gemm", "conv2d", "reduce_sum")
- **Data type**: The data type (e.g., "f32", "f64", "c32", "c64", "i32")
- **Variant** (optional): Specific variant (e.g., "transposed", "batched", "strided")

Examples:
- `linalg_gemm_f32` - Standard f32 GEMM
- `fft_fft2d_c64__batched` - Batched 2D FFT for complex64
- `neural_conv2d_f16__depthwise` - Depthwise convolution with f16

## Module-Specific Kernels

Each module is responsible for:

1. **Defining kernel sources**: Keep kernel source code in the module
2. **Registering with core**: Use the registry API to register kernels
3. **Handling fallbacks**: Provide CPU fallbacks when GPU is not available
4. **Testing kernels**: Test both CPU and GPU paths

## Migration Guide

If you have existing GPU kernels in your module:

1. **Identify all GPU kernels** in your module
2. **Create KernelId** for each kernel following the naming convention
3. **Extract kernel source** into constants or files
4. **Register kernels** using `register_module_kernel`
5. **Update kernel usage** to use `get_kernel` from registry
6. **Remove direct GPU API calls** and use core abstractions
7. **Test the migration** on different GPU backends

## Error Handling

```rust
use scirs2_core::gpu_registry::{get_kernel, has_kernel, KernelId};

// Check if kernel exists before using
let kernel_id = KernelId::new("mymodule", "myop", "f32");
if !has_kernel(&kernel_id) {
    // Fall back to CPU implementation
    return cpu_fallback(data);
}

// Get kernel with proper error handling
match get_kernel(&kernel_id, device) {
    Ok(kernel) => {
        // Use the kernel
    }
    Err(GpuError::KernelNotFound(_)) => {
        // Kernel not registered for this configuration
        return cpu_fallback(data);
    }
    Err(GpuError::BackendNotSupported(_)) => {
        // This GPU backend doesn't have this kernel
        return cpu_fallback(data);
    }
    Err(e) => {
        // Other GPU errors
        return Err(e.into());
    }
}
```

## Benefits

1. **Centralized management**: All kernels in one place
2. **Automatic caching**: Compiled kernels are cached
3. **Backend flexibility**: Easy to support multiple GPU backends
4. **Consistent API**: Same interface for all GPU operations
5. **Better debugging**: Central place to monitor GPU usage
6. **Resource sharing**: Avoid duplicate kernel compilations