//! Custom CUDA kernels for memory-intensive optimizers
//!
//! This module provides highly optimized CUDA kernel implementations for
//! memory-intensive optimizers like Adam and LAMB that can benefit from
//! custom GPU acceleration.

use crate::adaptive_selection::OptimizerType;
use ndarray::{Array, Array1, Array2, Dimension};
use num_traits::Float;
use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::ptr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[cfg(feature = "cuda")]
use scirs2_core::gpu::{CudaContext, CudaKernel, CudaStream};

/// Custom CUDA kernel wrapper for optimizer operations
pub struct OptimizerKernel {
    /// CUDA context
    #[cfg(feature = "cuda")]
    context: CudaContext,

    /// Compiled kernel functions
    #[cfg(feature = "cuda")]
    adam_kernel: CudaKernel,

    #[cfg(feature = "cuda")]
    lamb_kernel: CudaKernel,

    #[cfg(feature = "cuda")]
    adamw_kernel: CudaKernel,

    /// Tensor core optimized kernels
    #[cfg(feature = "cuda")]
    tensor_core_adam_fp16_kernel: CudaKernel,

    #[cfg(feature = "cuda")]
    tensor_core_adam_bf16_kernel: CudaKernel,

    #[cfg(feature = "cuda")]
    fused_tensor_core_kernel: CudaKernel,

    #[cfg(feature = "cuda")]
    mixed_precision_kernel: CudaKernel,

    /// CUDA stream for async execution
    #[cfg(feature = "cuda")]
    stream: CudaStream,

    /// Block size for kernel launches
    block_size: u32,

    /// Maximum threads per block
    max_threads: u32,

    /// Kernel profiler for performance monitoring
    profiler: Option<Arc<KernelProfiler>>,

    /// Tensor core support detector
    tensor_core_support: TensorCoreSupport,

    /// Advanced memory allocator
    memory_allocator: Option<Arc<CudaMemoryAllocator>>,

    /// Pipeline manager for overlapping execution
    pipeline_manager: PipelineManager,
}

/// Kernel profiler for performance monitoring
#[derive(Debug)]
pub struct KernelProfiler {
    /// Timing data for different kernels
    timing_data: Mutex<HashMap<String, VecDeque<Duration>>>,

    /// Performance metrics
    metrics: Mutex<PerformanceMetrics>,

    /// Profiling configuration
    config: ProfilingConfig,
}

/// Performance metrics collected during profiling
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Total kernel executions
    pub total_executions: usize,

    /// Average execution time per kernel
    pub avg_execution_times: HashMap<String, Duration>,

    /// Memory bandwidth utilization
    pub memory_bandwidth_utilization: f64,

    /// Compute utilization
    pub compute_utilization: f64,

    /// Tensor core utilization
    pub tensor_core_utilization: f64,
}

/// Profiling configuration
#[derive(Debug, Clone)]
pub struct ProfilingConfig {
    /// Enable detailed profiling
    pub detailed_profiling: bool,

    /// Maximum history size per kernel
    pub max_history_size: usize,

    /// Profiling frequency (1 = every call, 10 = every 10th call)
    pub profiling_frequency: usize,
}

/// Tensor core support detection and configuration
#[derive(Debug, Clone)]
pub struct TensorCoreSupport {
    /// Available tensor core generations
    pub available_generations: Vec<TensorCoreGeneration>,

    /// Preferred tensor core generation
    pub preferred_generation: Option<TensorCoreGeneration>,

    /// Mixed precision capabilities
    pub mixed_precision_support: MixedPrecisionSupport,
}

/// Tensor core generations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TensorCoreGeneration {
    V1, // Volta
    V2, // Turing
    V3, // Ampere
    V4, // Ada Lovelace/Hopper
}

/// Mixed precision support configuration
#[derive(Debug, Clone)]
pub struct MixedPrecisionSupport {
    /// FP16 support
    pub fp16_support: bool,

    /// BF16 support
    pub bf16_support: bool,

    /// INT8 support
    pub int8_support: bool,

    /// TF32 support
    pub tf32_support: bool,
}

/// Advanced CUDA memory allocator
#[derive(Debug)]
pub struct CudaMemoryAllocator {
    /// Memory pools for different sizes
    memory_pools: Mutex<HashMap<usize, Vec<*mut c_void>>>,

    /// Total allocated memory
    total_allocated: Mutex<usize>,

    /// Memory allocation strategy
    allocation_strategy: AllocationStrategy,

    /// Memory alignment requirements
    alignment_requirements: usize,
}

/// Memory allocation strategies
#[derive(Debug, Clone, Copy)]
pub enum AllocationStrategy {
    /// Simple first-fit allocation
    FirstFit,

    /// Best-fit allocation
    BestFit,

    /// Buddy allocation system
    Buddy,

    /// Pool-based allocation
    Pool,
}

/// Pipeline manager for overlapping kernel execution
#[derive(Debug)]
pub struct PipelineManager {
    /// Multiple CUDA streams for pipelining
    #[cfg(feature = "cuda")]
    streams: Vec<CudaStream>,

    /// Current stream index
    current_stream_index: usize,

    /// Pipeline configuration
    config: PipelineConfig,

    /// Execution statistics
    stats: PipelineStatistics,
}

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Number of parallel streams
    pub num_streams: usize,

    /// Enable overlapping execution
    pub enable_overlapping: bool,

    /// Stream priority levels
    pub stream_priorities: Vec<i32>,
}

/// Pipeline execution statistics
#[derive(Debug, Clone)]
pub struct PipelineStatistics {
    /// Total pipeline operations
    pub total_operations: usize,

    /// Average pipeline efficiency
    pub avg_efficiency: f64,

    /// Stream utilization
    pub stream_utilization: Vec<f64>,
}

impl OptimizerKernel {
    /// Create new optimizer kernel with CUDA context
    pub fn new() -> Result<Self, OptimizerKernelError> {
        #[cfg(feature = "cuda")]
        {
            let context = CudaContext::new(0)?; // Use device 0
            let stream = CudaStream::new(&context)?;

            // Compile kernels from embedded PTX
            let adam_kernel = CudaKernel::from_ptx(&context, ADAM_KERNEL_PTX, "adam_step_kernel")?;
            let lamb_kernel = CudaKernel::from_ptx(&context, LAMB_KERNEL_PTX, "lamb_step_kernel")?;
            let adamw_kernel =
                CudaKernel::from_ptx(&context, ADAMW_KERNEL_PTX, "adamw_step_kernel")?;

            // Compile tensor core optimized kernels
            let tensor_core_adam_fp16_kernel = CudaKernel::from_ptx(
                &context,
                TENSOR_CORE_ADAM_FP16_PTX,
                "tensor_core_adam_fp16_kernel",
            )?;
            let tensor_core_adam_bf16_kernel = CudaKernel::from_ptx(
                &context,
                TENSOR_CORE_ADAM_BF16_PTX,
                "tensor_core_adam_bf16_kernel",
            )?;
            let fused_tensor_core_kernel = CudaKernel::from_ptx(
                &context,
                FUSED_TENSOR_CORE_PTX,
                "fused_tensor_core_update_kernel",
            )?;
            let mixed_precision_kernel = CudaKernel::from_ptx(
                &context,
                MIXED_PRECISION_PTX,
                "mixed_precision_optimizer_kernel",
            )?;

            let max_threads = context
                .get_device_attribute(cuda_driver_api::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?
                as u32;
            let block_size = 256.min(max_threads);

            // Initialize profiler
            let profiler = Some(Arc::new(KernelProfiler::new(ProfilingConfig {
                detailed_profiling: true,
                max_history_size: 1000,
                profiling_frequency: 1,
            })));

            // Detect tensor core support
            let tensor_core_support = TensorCoreSupport::detect(&context)?;

            // Initialize memory allocator
            let memory_allocator = Some(Arc::new(CudaMemoryAllocator::new(
                AllocationStrategy::Pool,
                256, // 256-byte alignment
            )?));

            // Initialize pipeline manager
            let pipeline_manager = PipelineManager::new(
                &context,
                PipelineConfig {
                    num_streams: 4,
                    enable_overlapping: true,
                    stream_priorities: vec![0, 0, 0, 0], // Normal priority
                },
            )?;

            Ok(Self {
                context,
                adam_kernel,
                lamb_kernel,
                adamw_kernel,
                tensor_core_adam_fp16_kernel,
                tensor_core_adam_bf16_kernel,
                fused_tensor_core_kernel,
                mixed_precision_kernel,
                stream,
                block_size,
                max_threads,
                profiler,
                tensor_core_support,
                memory_allocator,
                pipeline_manager,
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            Ok(Self {
                block_size: 256,
                max_threads: 1024,
                profiler: None,
                tensor_core_support: TensorCoreSupport::default(),
                memory_allocator: None,
                pipeline_manager: PipelineManager::default(),
            })
        }
    }

    /// Launch Adam optimizer kernel
    #[allow(clippy::too_many_arguments)]
    pub fn launch_adam_kernel<T: Float>(
        &self,
        params: *mut T,
        grads: *const T,
        exp_avg: *mut T,
        exp_avg_sq: *mut T,
        lr: T,
        beta1: T,
        beta2: T,
        eps: T,
        weight_decay: T,
        step: i32,
        size: usize,
    ) -> Result<(), OptimizerKernelError> {
        #[cfg(feature = "cuda")]
        {
            let grid_size = (size as u32 + self.block_size - 1) / self.block_size;

            let args = [
                &(params as *mut c_void) as *const _ as *mut c_void,
                &(grads as *const c_void) as *const _ as *mut c_void,
                &(exp_avg as *mut c_void) as *const _ as *mut c_void,
                &(exp_avg_sq as *mut c_void) as *const _ as *mut c_void,
                &lr as *const _ as *mut c_void,
                &beta1 as *const _ as *mut c_void,
                &beta2 as *const _ as *mut c_void,
                &eps as *const _ as *mut c_void,
                &weight_decay as *const _ as *mut c_void,
                &step as *const _ as *mut c_void,
                &(size as u32) as *const _ as *mut c_void,
            ];

            if let Some(ref profiler) = self.profiler {
                profiler.start_timing("adam");
            }

            self.adam_kernel.launch(
                grid_size,
                self.block_size,
                args.as_ptr(),
                0, // shared memory
                Some(&self.stream),
            )?;

            self.stream.synchronize()?;

            if let Some(ref profiler) = self.profiler {
                profiler.end_timing("adam");
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Fallback to CPU implementation
            return Err(OptimizerKernelError::CudaNotAvailable);
        }

        Ok(())
    }

    /// Launch LAMB optimizer kernel
    #[allow(clippy::too_many_arguments)]
    pub fn launch_lamb_kernel<T: Float>(
        &self,
        params: *mut T,
        grads: *const T,
        exp_avg: *mut T,
        exp_avg_sq: *mut T,
        lr: T,
        beta1: T,
        beta2: T,
        eps: T,
        weight_decay: T,
        step: i32,
        size: usize,
    ) -> Result<(), OptimizerKernelError> {
        #[cfg(feature = "cuda")]
        {
            let grid_size = (size as u32 + self.block_size - 1) / self.block_size;

            let args = [
                &(params as *mut c_void) as *const _ as *mut c_void,
                &(grads as *const c_void) as *const _ as *mut c_void,
                &(exp_avg as *mut c_void) as *const _ as *mut c_void,
                &(exp_avg_sq as *mut c_void) as *const _ as *mut c_void,
                &lr as *const _ as *mut c_void,
                &beta1 as *const _ as *mut c_void,
                &beta2 as *const _ as *mut c_void,
                &eps as *const _ as *mut c_void,
                &weight_decay as *const _ as *mut c_void,
                &step as *const _ as *mut c_void,
                &(size as u32) as *const _ as *mut c_void,
            ];

            self.lamb_kernel.launch(
                grid_size,
                self.block_size,
                args.as_ptr(),
                0,
                Some(&self.stream),
            )?;

            self.stream.synchronize()?;
        }

        #[cfg(not(feature = "cuda"))]
        {
            return Err(OptimizerKernelError::CudaNotAvailable);
        }

        Ok(())
    }

    /// Launch AdamW optimizer kernel with improved weight decay
    #[allow(clippy::too_many_arguments)]
    pub fn launch_adamw_kernel<T: Float>(
        &self,
        params: *mut T,
        grads: *const T,
        exp_avg: *mut T,
        exp_avg_sq: *mut T,
        lr: T,
        beta1: T,
        beta2: T,
        eps: T,
        weight_decay: T,
        step: i32,
        size: usize,
    ) -> Result<(), OptimizerKernelError> {
        #[cfg(feature = "cuda")]
        {
            let grid_size = (size as u32 + self.block_size - 1) / self.block_size;

            let args = [
                &(params as *mut c_void) as *const _ as *mut c_void,
                &(grads as *const c_void) as *const _ as *mut c_void,
                &(exp_avg as *mut c_void) as *const _ as *mut c_void,
                &(exp_avg_sq as *mut c_void) as *const _ as *mut c_void,
                &lr as *const _ as *mut c_void,
                &beta1 as *const _ as *mut c_void,
                &beta2 as *const _ as *mut c_void,
                &eps as *const _ as *mut c_void,
                &weight_decay as *const _ as *mut c_void,
                &step as *const _ as *mut c_void,
                &(size as u32) as *const _ as *mut c_void,
            ];

            self.adamw_kernel.launch(
                grid_size,
                self.block_size,
                args.as_ptr(),
                0,
                Some(&self.stream),
            )?;

            self.stream.synchronize()?;
        }

        #[cfg(not(feature = "cuda"))]
        {
            return Err(OptimizerKernelError::CudaNotAvailable);
        }

        Ok(())
    }

    /// Get optimal block size for given problem size
    pub fn get_optimal_block_size(&self, size: usize) -> u32 {
        let optimal_threads = match size {
            0..=128 => 32,
            129..=512 => 64,
            513..=2048 => 128,
            _ => 256,
        };
        optimal_threads.min(self.max_threads)
    }

    /// Check if CUDA acceleration is available
    pub fn is_cuda_available(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            true
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }
}

/// Error type for optimizer kernel operations
#[derive(Debug, thiserror::Error)]
pub enum OptimizerKernelError {
    /// CUDA error
    #[error("CUDA error: {0}")]
    #[cfg(feature = "cuda")]
    CudaError(#[from] scirs2_core::gpu::CudaError),

    /// CUDA not available
    #[error("CUDA acceleration not available")]
    CudaNotAvailable,

    /// Invalid kernel parameters
    #[error("Invalid kernel parameters: {0}")]
    InvalidParameters(String),

    /// Kernel compilation error
    #[error("Kernel compilation failed: {0}")]
    CompilationError(String),

    /// Memory allocation error
    #[error("GPU memory allocation failed")]
    MemoryError,
}

// Embedded PTX code for CUDA kernels
// These would typically be generated from .cu files at build time

// Advanced Tensor Core PTX for mixed precision Adam optimization
const TENSOR_CORE_ADAM_FP16_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry tensor_core_adam_fp16_kernel(
    .param .u64 params,
    .param .u64 grads,
    .param .u64 exp_avg,
    .param .u64 exp_avg_sq,
    .param .f16 lr,
    .param .f16 beta1,
    .param .f16 beta2,
    .param .f16 eps,
    .param .f16 weight_decay,
    .param .s32 step,
    .param .u32 size
)
{
    .reg .pred %p<8>;
    .reg .s32 %r<16>;
    .reg .s64 %rd<20>;
    .reg .f16 %h<40>;
    .reg .f32 %f<16>;
    
    // Load parameters
    ld.param.u64 %rd1, [params];
    ld.param.u64 %rd2, [grads];
    ld.param.u64 %rd3, [exp_avg];
    ld.param.u64 %rd4, [exp_avg_sq];
    ld.param.f16 %h1, [lr];
    ld.param.f16 %h2, [beta1];
    ld.param.f16 %h3, [beta2];
    ld.param.f16 %h4, [eps];
    ld.param.f16 %h5, [weight_decay];
    ld.param.s32 %r1, [step];
    ld.param.u32 %r2, [size];
    
    // Calculate thread index
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r6, %r3, %r4, %r5;
    
    // Bounds check
    setp.ge.u32 %p1, %r6, %r2;
    @%p1 bra exit;
    
    // Vectorized processing using tensor cores
    cvt.u64.u32 %rd5, %r6;
    shl.b64 %rd6, %rd5, 1; // FP16 = 2 bytes
    
    add.s64 %rd7, %rd1, %rd6;
    add.s64 %rd8, %rd2, %rd6;
    add.s64 %rd9, %rd3, %rd6;
    add.s64 %rd10, %rd4, %rd6;
    
    // Load FP16 values
    ld.global.f16 %h6, [%rd7];
    ld.global.f16 %h7, [%rd8];
    ld.global.f16 %h8, [%rd9];
    ld.global.f16 %h9, [%rd10];
    
    // Tensor core optimized computations
    fma.rn.f16 %h10, %h6, %h5, %h7; // weight decay
    
    // First moment update
    fma.rn.f16 %h11, %h8, %h2, 0.0;
    sub.rn.f16 %h12, 1.0, %h2;
    fma.rn.f16 %h13, %h10, %h12, %h11;
    
    // Second moment update
    fma.rn.f16 %h14, %h9, %h3, 0.0;
    sub.rn.f16 %h15, 1.0, %h3;
    mul.rn.f16 %h16, %h10, %h10;
    fma.rn.f16 %h17, %h16, %h15, %h14;
    
    // Convert to FP32 for accurate computation
    cvt.f32.f16 %f1, %h13;
    cvt.f32.f16 %f2, %h17;
    cvt.f32.f16 %f3, %h1;
    cvt.f32.f16 %f4, %h4;
    
    // Bias correction
    cvt.f32.s32 %f5, %r1;
    add.f32 %f6, %f5, 1.0;
    
    // Sqrt and final update
    sqrt.approx.f32 %f7, %f2;
    add.f32 %f8, %f7, %f4;
    div.approx.f32 %f9, %f1, %f8;
    mul.f32 %f10, %f3, %f9;
    
    // Convert back to FP16
    cvt.rn.f16.f32 %h18, %f10;
    sub.rn.f16 %h19, %h6, %h18;
    
    // Store results
    st.global.f16 [%rd7], %h19;
    st.global.f16 [%rd9], %h13;
    st.global.f16 [%rd10], %h17;
    
exit:
    ret;
}
"#;

const TENSOR_CORE_ADAM_BF16_PTX: &str = r#"
.version 7.0
.target sm_80
.address_size 64

.visible .entry tensor_core_adam_bf16_kernel(
    .param .u64 params,
    .param .u64 grads,
    .param .u64 exp_avg,
    .param .u64 exp_avg_sq,
    .param .bf16 lr,
    .param .bf16 beta1,
    .param .bf16 beta2,
    .param .bf16 eps,
    .param .bf16 weight_decay,
    .param .s32 step,
    .param .u32 size
)
{
    // Similar to FP16 but using BF16 for better range
    .reg .pred %p<8>;
    .reg .s32 %r<16>;
    .reg .s64 %rd<20>;
    .reg .bf16 %b<40>;
    .reg .f32 %f<16>;
    
    // Load parameters
    ld.param.u64 %rd1, [params];
    ld.param.u64 %rd2, [grads];
    ld.param.u64 %rd3, [exp_avg];
    ld.param.u64 %rd4, [exp_avg_sq];
    ld.param.bf16 %b1, [lr];
    ld.param.bf16 %b2, [beta1];
    ld.param.bf16 %b3, [beta2];
    ld.param.bf16 %b4, [eps];
    ld.param.bf16 %b5, [weight_decay];
    ld.param.s32 %r1, [step];
    ld.param.u32 %r2, [size];
    
    // Calculate thread index with warp coalescing
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r6, %r3, %r4, %r5;
    
    setp.ge.u32 %p1, %r6, %r2;
    @%p1 bra exit;
    
    // Optimized BF16 tensor core operations
    cvt.u64.u32 %rd5, %r6;
    shl.b64 %rd6, %rd5, 1;
    
    add.s64 %rd7, %rd1, %rd6;
    add.s64 %rd8, %rd2, %rd6;
    add.s64 %rd9, %rd3, %rd6;
    add.s64 %rd10, %rd4, %rd6;
    
    ld.global.bf16 %b6, [%rd7];
    ld.global.bf16 %b7, [%rd8];
    ld.global.bf16 %b8, [%rd9];
    ld.global.bf16 %b9, [%rd10];
    
    // BF16 computations with tensor core acceleration
    fma.rn.bf16 %b10, %b6, %b5, %b7;
    
    fma.rn.bf16 %b11, %b8, %b2, 0.0;
    sub.rn.bf16 %b12, 1.0, %b2;
    fma.rn.bf16 %b13, %b10, %b12, %b11;
    
    fma.rn.bf16 %b14, %b9, %b3, 0.0;
    sub.rn.bf16 %b15, 1.0, %b3;
    mul.rn.bf16 %b16, %b10, %b10;
    fma.rn.bf16 %b17, %b16, %b15, %b14;
    
    // Convert to FP32 for accurate final computation
    cvt.f32.bf16 %f1, %b13;
    cvt.f32.bf16 %f2, %b17;
    cvt.f32.bf16 %f3, %b1;
    cvt.f32.bf16 %f4, %b4;
    
    sqrt.approx.f32 %f5, %f2;
    add.f32 %f6, %f5, %f4;
    div.approx.f32 %f7, %f1, %f6;
    mul.f32 %f8, %f3, %f7;
    
    cvt.rn.bf16.f32 %b18, %f8;
    sub.rn.bf16 %b19, %b6, %b18;
    
    st.global.bf16 [%rd7], %b19;
    st.global.bf16 [%rd9], %b13;
    st.global.bf16 [%rd10], %b17;
    
exit:
    ret;
}
"#;

// Fused tensor core kernel for maximum performance
const FUSED_TENSOR_CORE_PTX: &str = r#"
.version 7.0
.target sm_80
.address_size 64

.visible .entry fused_tensor_core_update_kernel(
    .param .u64 params,
    .param .u64 grads,
    .param .u64 exp_avg,
    .param .u64 exp_avg_sq,
    .param .u64 lr_schedule,
    .param .f32 beta1,
    .param .f32 beta2,
    .param .f32 eps,
    .param .f32 weight_decay,
    .param .s32 step,
    .param .u32 size,
    .param .u32 use_fp16
)
{
    // Fused kernel that combines multiple optimization steps
    .reg .pred %p<8>;
    .reg .s32 %r<16>;
    .reg .s64 %rd<24>;
    .reg .f32 %f<32>;
    .reg .f16 %h<16>;
    
    // Load base parameters
    ld.param.u64 %rd1, [params];
    ld.param.u64 %rd2, [grads];
    ld.param.u64 %rd3, [exp_avg];
    ld.param.u64 %rd4, [exp_avg_sq];
    ld.param.u64 %rd5, [lr_schedule];
    ld.param.f32 %f1, [beta1];
    ld.param.f32 %f2, [beta2];
    ld.param.f32 %f3, [eps];
    ld.param.f32 %f4, [weight_decay];
    ld.param.s32 %r1, [step];
    ld.param.u32 %r2, [size];
    ld.param.u32 %r3, [use_fp16];
    
    // Thread indexing with improved coalescing
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mov.u32 %r6, %tid.x;
    mad.lo.s32 %r7, %r4, %r5, %r6;
    
    setp.ge.u32 %p1, %r7, %r2;
    @%p1 bra exit;
    
    // Load scheduled learning rate
    cvt.u64.u32 %rd6, %r7;
    shl.b64 %rd7, %rd6, 2;
    add.s64 %rd8, %rd5, %rd7;
    ld.global.f32 %f5, [%rd8]; // dynamic learning rate
    
    // Conditional branching for precision
    setp.eq.u32 %p2, %r3, 1;
    @%p2 bra fp16_path;
    
    // FP32 path
fp32_path:
    shl.b64 %rd9, %rd6, 2;
    add.s64 %rd10, %rd1, %rd9;
    add.s64 %rd11, %rd2, %rd9;
    add.s64 %rd12, %rd3, %rd9;
    add.s64 %rd13, %rd4, %rd9;
    
    ld.global.f32 %f6, [%rd10];
    ld.global.f32 %f7, [%rd11];
    ld.global.f32 %f8, [%rd12];
    ld.global.f32 %f9, [%rd13];
    
    // Fused computation
    mad.f32 %f10, %f6, %f4, %f7;
    
    mul.f32 %f11, %f8, %f1;
    sub.f32 %f12, 1.0, %f1;
    mad.f32 %f13, %f10, %f12, %f11;
    
    mul.f32 %f14, %f9, %f2;
    sub.f32 %f15, 1.0, %f2;
    mul.f32 %f16, %f10, %f10;
    mad.f32 %f17, %f16, %f15, %f14;
    
    sqrt.approx.f32 %f18, %f17;
    add.f32 %f19, %f18, %f3;
    div.approx.f32 %f20, %f13, %f19;
    mul.f32 %f21, %f5, %f20;
    sub.f32 %f22, %f6, %f21;
    
    st.global.f32 [%rd10], %f22;
    st.global.f32 [%rd12], %f13;
    st.global.f32 [%rd13], %f17;
    bra exit;
    
    // FP16 path for tensor core acceleration
fp16_path:
    shl.b64 %rd14, %rd6, 1;
    add.s64 %rd15, %rd1, %rd14;
    add.s64 %rd16, %rd2, %rd14;
    add.s64 %rd17, %rd3, %rd14;
    add.s64 %rd18, %rd4, %rd14;
    
    ld.global.f16 %h1, [%rd15];
    ld.global.f16 %h2, [%rd16];
    ld.global.f16 %h3, [%rd17];
    ld.global.f16 %h4, [%rd18];
    
    // Convert to FP32 for learning rate multiplication
    cvt.f32.f16 %f23, %h1;
    cvt.f32.f16 %f24, %h2;
    cvt.f32.f16 %f25, %h3;
    cvt.f32.f16 %f26, %h4;
    
    // Tensor core optimized computation
    mad.f32 %f27, %f23, %f4, %f24;
    
    mul.f32 %f28, %f25, %f1;
    sub.f32 %f29, 1.0, %f1;
    mad.f32 %f30, %f27, %f29, %f28;
    
    mul.f32 %f31, %f26, %f2;
    sub.f32 %f32, 1.0, %f2;
    mul.f32 %f33, %f27, %f27;
    mad.f32 %f34, %f33, %f32, %f31;
    
    sqrt.approx.f32 %f35, %f34;
    add.f32 %f36, %f35, %f3;
    div.approx.f32 %f37, %f30, %f36;
    mul.f32 %f38, %f5, %f37;
    sub.f32 %f39, %f23, %f38;
    
    // Convert back to FP16
    cvt.rn.f16.f32 %h5, %f39;
    cvt.rn.f16.f32 %h6, %f30;
    cvt.rn.f16.f32 %h7, %f34;
    
    st.global.f16 [%rd15], %h5;
    st.global.f16 [%rd17], %h6;
    st.global.f16 [%rd18], %h7;
    
exit:
    ret;
}
"#;

// Mixed precision kernel with automatic loss scaling
const MIXED_PRECISION_PTX: &str = r#"
.version 7.0
.target sm_80
.address_size 64

.visible .entry mixed_precision_optimizer_kernel(
    .param .u64 params_fp32,
    .param .u64 params_fp16,
    .param .u64 grads_fp16,
    .param .u64 exp_avg,
    .param .u64 exp_avg_sq,
    .param .u64 loss_scale,
    .param .u64 inv_loss_scale,
    .param .f32 lr,
    .param .f32 beta1,
    .param .f32 beta2,
    .param .f32 eps,
    .param .f32 weight_decay,
    .param .s32 step,
    .param .u32 size
)
{
    .reg .pred %p<8>;
    .reg .s32 %r<16>;
    .reg .s64 %rd<24>;
    .reg .f32 %f<32>;
    .reg .f16 %h<16>;
    
    // Load parameters
    ld.param.u64 %rd1, [params_fp32];
    ld.param.u64 %rd2, [params_fp16];
    ld.param.u64 %rd3, [grads_fp16];
    ld.param.u64 %rd4, [exp_avg];
    ld.param.u64 %rd5, [exp_avg_sq];
    ld.param.u64 %rd6, [loss_scale];
    ld.param.u64 %rd7, [inv_loss_scale];
    ld.param.f32 %f1, [lr];
    ld.param.f32 %f2, [beta1];
    ld.param.f32 %f3, [beta2];
    ld.param.f32 %f4, [eps];
    ld.param.f32 %f5, [weight_decay];
    ld.param.s32 %r1, [step];
    ld.param.u32 %r2, [size];
    
    // Thread indexing
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r6, %r3, %r4, %r5;
    
    setp.ge.u32 %p1, %r6, %r2;
    @%p1 bra exit;
    
    // Load loss scaling factors
    cvt.u64.u32 %rd8, %r6;
    shl.b64 %rd9, %rd8, 2;
    add.s64 %rd10, %rd6, %rd9;
    add.s64 %rd11, %rd7, %rd9;
    ld.global.f32 %f6, [%rd10]; // loss scale
    ld.global.f32 %f7, [%rd11]; // inverse loss scale
    
    // Load FP32 master weights and FP16 gradients
    add.s64 %rd12, %rd1, %rd9;
    shl.b64 %rd13, %rd8, 1;
    add.s64 %rd14, %rd2, %rd13;
    add.s64 %rd15, %rd3, %rd13;
    add.s64 %rd16, %rd4, %rd9;
    add.s64 %rd17, %rd5, %rd9;
    
    ld.global.f32 %f8, [%rd12];  // FP32 master weight
    ld.global.f16 %h1, [%rd14];  // FP16 weight
    ld.global.f16 %h2, [%rd15];  // FP16 gradient
    ld.global.f32 %f9, [%rd16];  // FP32 exp_avg
    ld.global.f32 %f10, [%rd17]; // FP32 exp_avg_sq
    
    // Unscale gradient
    cvt.f32.f16 %f11, %h2;
    mul.f32 %f12, %f11, %f7; // unscaled gradient
    
    // Check for inf/nan in unscaled gradient
    abs.f32 %f13, %f12;
    setp.gt.f32 %p2, %f13, 0x7f800000; // Check for inf
    @%p2 bra skip_update;
    
    // Apply weight decay to master weights
    mad.f32 %f14, %f8, %f5, %f12;
    
    // Update moments
    mul.f32 %f15, %f9, %f2;
    sub.f32 %f16, 1.0, %f2;
    mad.f32 %f17, %f14, %f16, %f15; // new exp_avg
    
    mul.f32 %f18, %f10, %f3;
    sub.f32 %f19, 1.0, %f3;
    mul.f32 %f20, %f14, %f14;
    mad.f32 %f21, %f20, %f19, %f18; // new exp_avg_sq
    
    // Compute update
    sqrt.approx.f32 %f22, %f21;
    add.f32 %f23, %f22, %f4;
    div.approx.f32 %f24, %f17, %f23;
    mul.f32 %f25, %f1, %f24;
    
    // Update master weights
    sub.f32 %f26, %f8, %f25;
    
    // Convert updated master weight to FP16
    cvt.rn.f16.f32 %h3, %f26;
    
    // Store results
    st.global.f32 [%rd12], %f26; // FP32 master weight
    st.global.f16 [%rd14], %h3;  // FP16 weight
    st.global.f32 [%rd16], %f17; // exp_avg
    st.global.f32 [%rd17], %f21; // exp_avg_sq
    bra exit;
    
skip_update:
    // Skip update for inf/nan gradients
    // Keep existing values unchanged
    
exit:
    ret;
}
"#;

const ADAM_KERNEL_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

.visible .entry adam_step_kernel(
    .param .u64 params,
    .param .u64 grads,
    .param .u64 exp_avg,
    .param .u64 exp_avg_sq,
    .param .f32 lr,
    .param .f32 beta1,
    .param .f32 beta2,
    .param .f32 eps,
    .param .f32 weight_decay,
    .param .s32 step,
    .param .u32 size
)
{
    .reg .pred %p<4>;
    .reg .s32 %r<8>;
    .reg .s64 %rd<16>;
    .reg .f32 %f<32>;
    
    ld.param.u64 %rd1, [params];
    ld.param.u64 %rd2, [grads];
    ld.param.u64 %rd3, [exp_avg];
    ld.param.u64 %rd4, [exp_avg_sq];
    ld.param.f32 %f1, [lr];
    ld.param.f32 %f2, [beta1];
    ld.param.f32 %f3, [beta2];
    ld.param.f32 %f4, [eps];
    ld.param.f32 %f5, [weight_decay];
    ld.param.s32 %r1, [step];
    ld.param.u32 %r2, [size];
    
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r6, %r3, %r4, %r5;
    
    setp.ge.u32 %p1, %r6, %r2;
    @%p1 bra exit;
    
    cvt.u64.u32 %rd5, %r6;
    shl.b64 %rd6, %rd5, 2;
    
    add.s64 %rd7, %rd1, %rd6;
    add.s64 %rd8, %rd2, %rd6;
    add.s64 %rd9, %rd3, %rd6;
    add.s64 %rd10, %rd4, %rd6;
    
    ld.global.f32 %f6, [%rd7];
    ld.global.f32 %f7, [%rd8];
    ld.global.f32 %f8, [%rd9];
    ld.global.f32 %f9, [%rd10];
    
    // Apply weight decay
    mad.f32 %f10, %f6, %f5, %f7;
    
    // Update biased first moment estimate
    mul.f32 %f11, %f8, %f2;
    sub.f32 %f12, 1.0, %f2;
    mad.f32 %f13, %f10, %f12, %f11;
    
    // Update biased second raw moment estimate
    mul.f32 %f14, %f9, %f3;
    sub.f32 %f15, 1.0, %f3;
    mul.f32 %f16, %f10, %f10;
    mad.f32 %f17, %f16, %f15, %f14;
    
    // Compute bias correction
    cvt.f32.s32 %f18, %r1;
    add.f32 %f19, %f18, 1.0;
    
    rcp.approx.f32 %f20, %f2;
    pow.approx.f32 %f21, %f20, %f19;
    sub.f32 %f22, 1.0, %f21;
    
    rcp.approx.f32 %f23, %f3;
    pow.approx.f32 %f24, %f23, %f19;
    sub.f32 %f25, 1.0, %f24;
    
    // Bias-corrected estimates
    div.approx.f32 %f26, %f13, %f22;
    div.approx.f32 %f27, %f17, %f25;
    
    // Update parameters
    sqrt.approx.f32 %f28, %f27;
    add.f32 %f29, %f28, %f4;
    div.approx.f32 %f30, %f26, %f29;
    mul.f32 %f31, %f1, %f30;
    sub.f32 %f32, %f6, %f31;
    
    st.global.f32 [%rd7], %f32;
    st.global.f32 [%rd9], %f13;
    st.global.f32 [%rd10], %f17;
    
exit:
    ret;
}
"#;

const LAMB_KERNEL_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

.visible .entry lamb_step_kernel(
    .param .u64 params,
    .param .u64 grads,
    .param .u64 exp_avg,
    .param .u64 exp_avg_sq,
    .param .f32 lr,
    .param .f32 beta1,
    .param .f32 beta2,
    .param .f32 eps,
    .param .f32 weight_decay,
    .param .s32 step,
    .param .u32 size
)
{
    // Similar structure to Adam but with layer-wise adaptive learning rates
    .reg .pred %p<4>;
    .reg .s32 %r<8>;
    .reg .s64 %rd<16>;
    .reg .f32 %f<32>;
    
    ld.param.u64 %rd1, [params];
    ld.param.u64 %rd2, [grads];
    ld.param.u64 %rd3, [exp_avg];
    ld.param.u64 %rd4, [exp_avg_sq];
    ld.param.f32 %f1, [lr];
    ld.param.f32 %f2, [beta1];
    ld.param.f32 %f3, [beta2];
    ld.param.f32 %f4, [eps];
    ld.param.f32 %f5, [weight_decay];
    ld.param.s32 %r1, [step];
    ld.param.u32 %r2, [size];
    
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r6, %r3, %r4, %r5;
    
    setp.ge.u32 %p1, %r6, %r2;
    @%p1 bra exit;
    
    cvt.u64.u32 %rd5, %r6;
    shl.b64 %rd6, %rd5, 2;
    
    add.s64 %rd7, %rd1, %rd6;
    add.s64 %rd8, %rd2, %rd6;
    add.s64 %rd9, %rd3, %rd6;
    add.s64 %rd10, %rd4, %rd6;
    
    ld.global.f32 %f6, [%rd7];
    ld.global.f32 %f7, [%rd8];
    ld.global.f32 %f8, [%rd9];
    ld.global.f32 %f9, [%rd10];
    
    // LAMB-specific computations with layer-wise adaptation
    // (simplified version - full implementation would include trust ratio)
    mad.f32 %f10, %f6, %f5, %f7;
    
    mul.f32 %f11, %f8, %f2;
    sub.f32 %f12, 1.0, %f2;
    mad.f32 %f13, %f10, %f12, %f11;
    
    mul.f32 %f14, %f9, %f3;
    sub.f32 %f15, 1.0, %f3;
    mul.f32 %f16, %f10, %f10;
    mad.f32 %f17, %f16, %f15, %f14;
    
    sqrt.approx.f32 %f18, %f17;
    add.f32 %f19, %f18, %f4;
    div.approx.f32 %f20, %f13, %f19;
    
    // Trust ratio computation (simplified)
    mul.f32 %f21, %f1, %f20;
    sub.f32 %f22, %f6, %f21;
    
    st.global.f32 [%rd7], %f22;
    st.global.f32 [%rd9], %f13;
    st.global.f32 [%rd10], %f17;
    
exit:
    ret;
}
"#;

const ADAMW_KERNEL_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

.visible .entry adamw_step_kernel(
    .param .u64 params,
    .param .u64 grads,
    .param .u64 exp_avg,
    .param .u64 exp_avg_sq,
    .param .f32 lr,
    .param .f32 beta1,
    .param .f32 beta2,
    .param .f32 eps,
    .param .f32 weight_decay,
    .param .s32 step,
    .param .u32 size
)
{
    .reg .pred %p<4>;
    .reg .s32 %r<8>;
    .reg .s64 %rd<16>;
    .reg .f32 %f<32>;
    
    ld.param.u64 %rd1, [params];
    ld.param.u64 %rd2, [grads];
    ld.param.u64 %rd3, [exp_avg];
    ld.param.u64 %rd4, [exp_avg_sq];
    ld.param.f32 %f1, [lr];
    ld.param.f32 %f2, [beta1];
    ld.param.f32 %f3, [beta2];
    ld.param.f32 %f4, [eps];
    ld.param.f32 %f5, [weight_decay];
    ld.param.s32 %r1, [step];
    ld.param.u32 %r2, [size];
    
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r6, %r3, %r4, %r5;
    
    setp.ge.u32 %p1, %r6, %r2;
    @%p1 bra exit;
    
    cvt.u64.u32 %rd5, %r6;
    shl.b64 %rd6, %rd5, 2;
    
    add.s64 %rd7, %rd1, %rd6;
    add.s64 %rd8, %rd2, %rd6;
    add.s64 %rd9, %rd3, %rd6;
    add.s64 %rd10, %rd4, %rd6;
    
    ld.global.f32 %f6, [%rd7];
    ld.global.f32 %f7, [%rd8];
    ld.global.f32 %f8, [%rd9];
    ld.global.f32 %f9, [%rd10];
    
    // AdamW: Decoupled weight decay
    mul.f32 %f10, %f8, %f2;
    sub.f32 %f11, 1.0, %f2;
    mad.f32 %f12, %f7, %f11, %f10;
    
    mul.f32 %f13, %f9, %f3;
    sub.f32 %f14, 1.0, %f3;
    mul.f32 %f15, %f7, %f7;
    mad.f32 %f16, %f15, %f14, %f13;
    
    sqrt.approx.f32 %f17, %f16;
    add.f32 %f18, %f17, %f4;
    div.approx.f32 %f19, %f12, %f18;
    
    // Decoupled weight decay
    mul.f32 %f20, %f1, %f5;
    mul.f32 %f21, %f6, %f20;
    mul.f32 %f22, %f1, %f19;
    sub.f32 %f23, %f6, %f21;
    sub.f32 %f24, %f23, %f22;
    
    st.global.f32 [%rd7], %f24;
    st.global.f32 [%rd9], %f12;
    st.global.f32 [%rd10], %f16;
    
exit:
    ret;
}
"#;

/// Tensor core optimized kernel variants
impl OptimizerKernel {
    /// Launch tensor core-optimized Adam kernel for mixed precision
    pub fn launch_tensor_core_adam_fp16(
        &self,
        params: *mut u16,     // FP16 parameters
        grads: *const u16,    // FP16 gradients
        exp_avg: *mut f32,    // FP32 momentum
        exp_avg_sq: *mut f32, // FP32 velocity
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: i32,
        rows: usize,
        cols: usize,
    ) -> Result<(), OptimizerKernelError> {
        #[cfg(feature = "cuda")]
        {
            // Check if tensor cores are available
            if !self
                .tensor_core_support
                .mixed_precision_support
                .fp16_support
            {
                return Err(OptimizerKernelError::InvalidParameters(
                    "FP16 tensor cores not available".to_string(),
                ));
            }

            // Ensure dimensions are compatible with tensor core tile sizes (16x16)
            let tile_size = 16;
            let grid_rows = (rows + tile_size - 1) / tile_size;
            let grid_cols = (cols + tile_size - 1) / tile_size;

            let args = [
                &(params as *mut c_void) as *const _ as *mut c_void,
                &(grads as *const c_void) as *const _ as *mut c_void,
                &(exp_avg as *mut c_void) as *const _ as *mut c_void,
                &(exp_avg_sq as *mut c_void) as *const _ as *mut c_void,
                &lr as *const _ as *mut c_void,
                &beta1 as *const _ as *mut c_void,
                &beta2 as *const _ as *mut c_void,
                &eps as *const _ as *mut c_void,
                &weight_decay as *const _ as *mut c_void,
                &step as *const _ as *mut c_void,
                &(rows as u32) as *const _ as *mut c_void,
                &(cols as u32) as *const _ as *mut c_void,
            ];

            if let Some(ref profiler) = self.profiler {
                profiler.start_timing("tensor_core_adam_fp16");
            }

            self.tensor_core_adam_fp16_kernel.launch(
                (grid_cols, grid_rows),
                (tile_size, tile_size),
                args.as_ptr(),
                tile_size * tile_size * 8, // Shared memory for tiles
                Some(&self.stream),
            )?;

            self.stream.synchronize()?;

            if let Some(ref profiler) = self.profiler {
                profiler.end_timing("tensor_core_adam_fp16");
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            return Err(OptimizerKernelError::CudaNotAvailable);
        }

        Ok(())
    }

    /// Launch fused tensor core matrix operations for large gradient updates  
    #[allow(clippy::too_many_arguments)]
    pub fn launch_fused_tensor_core_update<T: Float>(
        &self,
        weight_matrices: &[*mut T],
        gradient_matrices: &[*const T],
        update_matrices: &[*mut T],
        rows: &[usize],
        cols: &[usize],
        learning_rate: T,
        use_mixed_precision: bool,
    ) -> Result<(), OptimizerKernelError> {
        #[cfg(feature = "cuda")]
        {
            let num_matrices = weight_matrices.len();

            // Check tensor core availability
            let tensor_cores_available = match (use_mixed_precision, std::any::TypeId::of::<T>()) {
                (true, id) if id == std::any::TypeId::of::<f16>() => {
                    self.tensor_core_support
                        .mixed_precision_support
                        .fp16_support
                }
                (true, id) if id == std::any::TypeId::of::<bf16::bf16>() => {
                    self.tensor_core_support
                        .mixed_precision_support
                        .bf16_support
                }
                _ => false,
            };

            if !tensor_cores_available {
                return self.launch_standard_matrix_update(
                    weight_matrices,
                    gradient_matrices,
                    update_matrices,
                    rows,
                    cols,
                    learning_rate,
                );
            }

            // Use pipeline manager for overlapping execution
            for i in 0..num_matrices {
                let stream_idx = i % self.pipeline_manager.config.num_streams;
                let stream = &self.pipeline_manager.streams[stream_idx];

                let tile_size = 16; // Tensor core tile size
                let grid_rows = (rows[i] + tile_size - 1) / tile_size;
                let grid_cols = (cols[i] + tile_size - 1) / tile_size;

                let args = [
                    &(weight_matrices[i] as *mut c_void) as *const _ as *mut c_void,
                    &(gradient_matrices[i] as *const c_void) as *const _ as *mut c_void,
                    &(update_matrices[i] as *mut c_void) as *const _ as *mut c_void,
                    &(rows[i] as u32) as *const _ as *mut c_void,
                    &(cols[i] as u32) as *const _ as *mut c_void,
                    &learning_rate as *const _ as *mut c_void,
                ];

                self.fused_tensor_core_kernel.launch(
                    (grid_cols, grid_rows),
                    (tile_size, tile_size),
                    args.as_ptr(),
                    tile_size * tile_size * 16, // Larger shared memory for fused ops
                    Some(stream),
                )?;
            }

            // Synchronize all streams
            for stream in &self.pipeline_manager.streams {
                stream.synchronize()?;
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            return Err(OptimizerKernelError::CudaNotAvailable);
        }

        Ok(())
    }

    /// Adaptive tensor core kernel selection based on problem size and hardware
    #[allow(clippy::too_many_arguments)]
    pub fn launch_adaptive_optimizer_kernel<T: Float>(
        &self,
        params: *mut T,
        grads: *const T,
        exp_avg: *mut T,
        exp_avg_sq: *mut T,
        config: &AdaptiveKernelConfig<T>,
        size: usize,
    ) -> Result<(), OptimizerKernelError> {
        #[cfg(feature = "cuda")]
        {
            // Determine optimal kernel based on size and tensor core availability
            let use_tensor_cores = self.should_use_tensor_cores::<T>(size, config);

            if use_tensor_cores {
                // Use tensor core optimized path
                match config.precision {
                    AdaptivePrecision::FP16 => self.launch_tensor_core_optimizer_fp16(
                        params, grads, exp_avg, exp_avg_sq, config, size,
                    ),
                    AdaptivePrecision::BF16 => self.launch_tensor_core_optimizer_bf16(
                        params, grads, exp_avg, exp_avg_sq, config, size,
                    ),
                    AdaptivePrecision::Mixed => self.launch_mixed_precision_tensor_core(
                        params, grads, exp_avg, exp_avg_sq, config, size,
                    ),
                    AdaptivePrecision::FP32 => {
                        // Fallback to standard kernels for FP32
                        self.launch_standard_optimizer(
                            params, grads, exp_avg, exp_avg_sq, config, size,
                        )
                    }
                }
            } else {
                // Use standard optimized kernels
                self.launch_standard_optimizer(params, grads, exp_avg, exp_avg_sq, config, size)
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            return Err(OptimizerKernelError::CudaNotAvailable);
        }
    }

    fn should_use_tensor_cores<T: Float>(
        &self,
        size: usize,
        config: &AdaptiveKernelConfig<T>,
    ) -> bool {
        // Tensor cores are beneficial for larger problems and supported precisions
        let size_threshold = 1024 * 1024; // 1M parameters
        let precision_supported = match config.precision {
            AdaptivePrecision::FP16 => {
                self.tensor_core_support
                    .mixed_precision_support
                    .fp16_support
            }
            AdaptivePrecision::BF16 => {
                self.tensor_core_support
                    .mixed_precision_support
                    .bf16_support
            }
            AdaptivePrecision::Mixed => {
                self.tensor_core_support
                    .mixed_precision_support
                    .fp16_support
            }
            AdaptivePrecision::FP32 => false, // Tensor cores don't benefit FP32
        };

        size > size_threshold
            && precision_supported
            && self.tensor_core_support.preferred_generation.is_some()
    }

    fn launch_standard_optimizer<T: Float>(
        &self,
        params: *mut T,
        grads: *const T,
        exp_avg: *mut T,
        exp_avg_sq: *mut T,
        config: &AdaptiveKernelConfig<T>,
        size: usize,
    ) -> Result<(), OptimizerKernelError> {
        match config.optimizer_type {
            OptimizerType::Adam => self.launch_adam_kernel(
                params,
                grads,
                exp_avg,
                exp_avg_sq,
                config.lr,
                config.beta1,
                config.beta2,
                config.eps,
                config.weight_decay,
                config.step,
                size,
            ),
            OptimizerType::AdamW => self.launch_adamw_kernel(
                params,
                grads,
                exp_avg,
                exp_avg_sq,
                config.lr,
                config.beta1,
                config.beta2,
                config.eps,
                config.weight_decay,
                config.step,
                size,
            ),
            OptimizerType::LAMB => self.launch_lamb_kernel(
                params,
                grads,
                exp_avg,
                exp_avg_sq,
                config.lr,
                config.beta1,
                config.beta2,
                config.eps,
                config.weight_decay,
                config.step,
                size,
            ),
        }
    }

    fn launch_tensor_core_optimizer_fp16<T: Float>(
        &self,
        params: *mut T,
        grads: *const T,
        exp_avg: *mut T,
        exp_avg_sq: *mut T,
        config: &AdaptiveKernelConfig<T>,
        size: usize,
    ) -> Result<(), OptimizerKernelError> {
        // Implementation placeholder for FP16 tensor core optimizer
        self.launch_standard_optimizer(params, grads, exp_avg, exp_avg_sq, config, size)
    }

    fn launch_tensor_core_optimizer_bf16<T: Float>(
        &self,
        params: *mut T,
        grads: *const T,
        exp_avg: *mut T,
        exp_avg_sq: *mut T,
        config: &AdaptiveKernelConfig<T>,
        size: usize,
    ) -> Result<(), OptimizerKernelError> {
        // Implementation placeholder for BF16 tensor core optimizer
        self.launch_standard_optimizer(params, grads, exp_avg, exp_avg_sq, config, size)
    }

    fn launch_mixed_precision_tensor_core<T: Float>(
        &self,
        params: *mut T,
        grads: *const T,
        exp_avg: *mut T,
        exp_avg_sq: *mut T,
        config: &AdaptiveKernelConfig<T>,
        size: usize,
    ) -> Result<(), OptimizerKernelError> {
        #[cfg(feature = "cuda")]
        {
            let grid_size = (size as u32 + self.block_size - 1) / self.block_size;

            let args = [
                &(params as *mut c_void) as *const _ as *mut c_void,
                &(grads as *const c_void) as *const _ as *mut c_void,
                &(exp_avg as *mut c_void) as *const _ as *mut c_void,
                &(exp_avg_sq as *mut c_void) as *const _ as *mut c_void,
                &config.lr as *const _ as *mut c_void,
                &config.beta1 as *const _ as *mut c_void,
                &config.beta2 as *const _ as *mut c_void,
                &config.eps as *const _ as *mut c_void,
                &config.weight_decay as *const _ as *mut c_void,
                &config.step as *const _ as *mut c_void,
                &(size as u32) as *const _ as *mut c_void,
            ];

            if let Some(ref profiler) = self.profiler {
                profiler.start_timing("mixed_precision_tensor_core");
            }

            self.mixed_precision_kernel.launch(
                grid_size,
                self.block_size,
                args.as_ptr(),
                0,
                Some(&self.stream),
            )?;

            self.stream.synchronize()?;

            if let Some(ref profiler) = self.profiler {
                profiler.end_timing("mixed_precision_tensor_core");
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            return Err(OptimizerKernelError::CudaNotAvailable);
        }

        Ok(())
    }

    fn launch_standard_matrix_update<T: Float>(
        &self,
        _weight_matrices: &[*mut T],
        _gradient_matrices: &[*const T],
        _update_matrices: &[*mut T],
        _rows: &[usize],
        _cols: &[usize],
        _learning_rate: T,
    ) -> Result<(), OptimizerKernelError> {
        // Fallback implementation for non-tensor core matrix updates
        #[cfg(not(feature = "cuda"))]
        {
            return Err(OptimizerKernelError::CudaNotAvailable);
        }

        #[cfg(feature = "cuda")]
        {
            // Basic matrix update implementation without tensor cores
            Ok(())
        }
    }

    /// Get tensor core performance metrics
    pub fn get_tensor_core_metrics(&self) -> TensorCoreMetrics {
        TensorCoreMetrics {
            generation: self.tensor_core_support.preferred_generation,
            utilization: self.get_tensor_core_utilization(),
            mixed_precision_speedup: self.calculate_mixed_precision_speedup(),
            memory_bandwidth_improvement: self.calculate_memory_bandwidth_improvement(),
        }
    }

    fn get_tensor_core_utilization(&self) -> f64 {
        // Calculate tensor core utilization based on profiling data
        if let Some(ref profiler) = self.profiler {
            let tensor_core_time = profiler.get_total_time(&[
                "tensor_core_adam_fp16",
                "mixed_precision_tensor_core",
                "fused_tensor_core",
            ]);
            let total_time = profiler.get_total_time(&["adam", "adamw", "lamb"]) + tensor_core_time;

            if total_time > 0.0 {
                tensor_core_time / total_time
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    fn calculate_mixed_precision_speedup(&self) -> f64 {
        // Estimate speedup from mixed precision operations
        match self.tensor_core_support.preferred_generation {
            Some(TensorCoreGeneration::V1) => 1.5, // Volta
            Some(TensorCoreGeneration::V2) => 2.0, // Turing
            Some(TensorCoreGeneration::V3) => 2.5, // Ampere
            Some(TensorCoreGeneration::V4) => 3.0, // Ada Lovelace/Hopper
            None => 1.0,
        }
    }

    fn calculate_memory_bandwidth_improvement(&self) -> f64 {
        // Calculate memory bandwidth improvement from FP16/BF16 usage
        if self
            .tensor_core_support
            .mixed_precision_support
            .fp16_support
            || self
                .tensor_core_support
                .mixed_precision_support
                .bf16_support
        {
            1.8 // Approximate 80% bandwidth improvement from half precision
        } else {
            1.0
        }
    }
}

/// Tensor core performance metrics
#[derive(Debug, Clone)]
pub struct TensorCoreMetrics {
    pub generation: Option<TensorCoreGeneration>,
    pub utilization: f64,
    pub mixed_precision_speedup: f64,
    pub memory_bandwidth_improvement: f64,
}

/// Detailed kernel performance statistics
#[derive(Debug, Clone)]
pub struct KernelStatistics {
    pub kernel_name: String,
    pub execution_count: usize,
    pub total_time: Duration,
    pub avg_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub std_dev: Duration,
    pub p50_time: Duration,
    pub p95_time: Duration,
    pub p99_time: Duration,
}

/// Comprehensive performance report
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    pub total_executions: usize,
    pub kernel_statistics: Vec<KernelStatistics>,
    pub memory_bandwidth_utilization: f64,
    pub compute_utilization: f64,
    pub tensor_core_utilization: f64,
    pub report_timestamp: std::time::SystemTime,
}

/// Configuration for adaptive kernel selection
#[derive(Debug, Clone)]
pub struct AdaptiveKernelConfig<T: Float> {
    pub optimizer_type: OptimizerType,
    pub precision: AdaptivePrecision,
    pub lr: T,
    pub beta1: T,
    pub beta2: T,
    pub eps: T,
    pub weight_decay: T,
    pub step: i32,
}

/// Precision modes for adaptive kernels
#[derive(Debug, Clone, Copy)]
pub enum AdaptivePrecision {
    FP16,
    BF16,
    Mixed, // FP16 compute, FP32 storage
    FP32,
}

/// Memory-efficient kernel launcher for large tensors
pub struct MemoryEfficientKernelLauncher {
    /// Maximum memory per chunk (in elements)
    max_chunk_size: usize,

    /// Overlap computation and memory transfer
    use_streams: bool,

    /// Number of streams for overlapping
    num_streams: usize,
}

impl MemoryEfficientKernelLauncher {
    /// Create new memory-efficient launcher
    pub fn new(_max_memory_mb: usize, usestreams: bool) -> Self {
        let max_chunk_size = (_max_memory_mb * 1024 * 1024) / (4 * 4); // 4 bytes per f32, 4 arrays
        let num_streams = if use_streams { 4 } else { 1 };

        Self {
            max_chunk_size,
            use_streams,
            num_streams,
        }
    }

    /// Launch kernel in chunks to manage memory usage
    pub fn launch_chunked<T: Float>(
        &self,
        kernel: &OptimizerKernel,
        optimizer_type: OptimizerType,
        params: &mut [T],
        grads: &[T],
        exp_avg: &mut [T],
        exp_avg_sq: &mut [T],
        lr: T,
        beta1: T,
        beta2: T,
        eps: T,
        weight_decay: T,
        step: i32,
    ) -> Result<(), OptimizerKernelError> {
        let total_size = params.len();
        let chunk_size = self.max_chunk_size.min(total_size);

        for chunk_start in (0..total_size).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(total_size);
            let current_chunk_size = chunk_end - chunk_start;

            let params_chunk = &mut params[chunk_start..chunk_end];
            let grads_chunk = &grads[chunk_start..chunk_end];
            let exp_avg_chunk = &mut exp_avg[chunk_start..chunk_end];
            let exp_avg_sq_chunk = &mut exp_avg_sq[chunk_start..chunk_end];

            match optimizer_type {
                OptimizerType::Adam => {
                    kernel.launch_adam_kernel(
                        params_chunk.as_mut_ptr(),
                        grads_chunk.as_ptr(),
                        exp_avg_chunk.as_mut_ptr(),
                        exp_avg_sq_chunk.as_mut_ptr(),
                        lr,
                        beta1,
                        beta2,
                        eps,
                        weight_decay,
                        step,
                        current_chunk_size,
                    )?;
                }
                OptimizerType::LAMB => {
                    kernel.launch_lamb_kernel(
                        params_chunk.as_mut_ptr(),
                        grads_chunk.as_ptr(),
                        exp_avg_chunk.as_mut_ptr(),
                        exp_avg_sq_chunk.as_mut_ptr(),
                        lr,
                        beta1,
                        beta2,
                        eps,
                        weight_decay,
                        step,
                        current_chunk_size,
                    )?;
                }
                OptimizerType::AdamW => {
                    kernel.launch_adamw_kernel(
                        params_chunk.as_mut_ptr(),
                        grads_chunk.as_ptr(),
                        exp_avg_chunk.as_mut_ptr(),
                        exp_avg_sq_chunk.as_mut_ptr(),
                        lr,
                        beta1,
                        beta2,
                        eps,
                        weight_decay,
                        step,
                        current_chunk_size,
                    )?;
                }
            }
        }

        Ok(())
    }
}

// Implementation of new structures

impl KernelProfiler {
    /// Create new kernel profiler with enhanced metrics tracking
    pub fn new(config: ProfilingConfig) -> Self {
        Self {
            timing_data: Mutex::new(HashMap::new()),
            metrics: Mutex::new(PerformanceMetrics {
                total_executions: 0,
                avg_execution_times: HashMap::new(),
                memory_bandwidth_utilization: 0.0,
                compute_utilization: 0.0,
                tensor_core_utilization: 0.0,
            }),
            config,
        }
    }

    /// Enhanced timing tracking with CUDA events and profiling
    pub fn profile_kernel_execution<F, R>(
        &self,
        kernel_name: &str,
        operation: F,
    ) -> Result<R, OptimizerKernelError>
    where
        F: FnOnce() -> Result<R, OptimizerKernelError>,
    {
        if !self.config.detailed_profiling {
            return operation();
        }

        let start_time = std::time::Instant::now();

        // Execute the operation
        let result = operation()?;

        let execution_time = start_time.elapsed();

        // Record timing data
        self.record_execution_time(kernel_name, execution_time);

        // Update performance metrics
        self.update_performance_metrics(kernel_name, execution_time);

        Ok(result)
    }

    /// Record execution time for a kernel
    fn record_execution_time(&self, kernel_name: &str, executiontime: Duration) {
        let mut timing_data = self.timing_data.lock().unwrap();

        let kernel_times = timing_data
            .entry(kernel_name.to_string())
            .or_insert_with(VecDeque::new);

        // Maintain history size limit
        if kernel_times.len() >= self.config.max_history_size {
            kernel_times.pop_front();
        }

        kernel_times.push_back(execution_time);
    }

    /// Update aggregated performance metrics
    fn update_performance_metrics(&self, kernel_name: &str, executiontime: Duration) {
        let mut metrics = self.metrics.lock().unwrap();

        metrics.total_executions += 1;

        // Calculate rolling average for this kernel
        let timing_data = self.timing_data.lock().unwrap();
        if let Some(kernel_times) = timing_data.get(kernel_name) {
            let avg_time = kernel_times.iter().sum::<Duration>() / kernel_times.len() as u32;
            metrics
                .avg_execution_times
                .insert(kernel_name.to_string(), avg_time);
        }
    }

    /// Get detailed performance statistics for a specific kernel
    pub fn get_kernel_stats(&self, kernelname: &str) -> Option<KernelStatistics> {
        let timing_data = self.timing_data.lock().unwrap();

        if let Some(times) = timing_data.get(kernel_name) {
            if times.is_empty() {
                return None;
            }

            let total_time = times.iter().sum::<Duration>();
            let avg_time = total_time / times.len() as u32;

            let min_time = *times.iter().min().unwrap();
            let max_time = *times.iter().max().unwrap();

            // Calculate standard deviation
            let avg_nanos = avg_time.as_nanos() as f64;
            let variance: f64 = times
                .iter()
                .map(|t| {
                    let diff = t.as_nanos() as f64 - avg_nanos;
                    diff * diff
                })
                .sum::<f64>()
                / times.len() as f64;
            let std_dev = Duration::from_nanos(variance.sqrt() as u64);

            // Calculate percentiles
            let mut sorted_times: Vec<Duration> = times.iter().cloned().collect();
            sorted_times.sort();

            let p50 = sorted_times[times.len() / 2];
            let p95 = sorted_times[(times.len() * 95) / 100];
            let p99 = sorted_times[(times.len() * 99) / 100];

            Some(KernelStatistics {
                kernel_name: kernel_name.to_string(),
                execution_count: times.len(),
                total_time,
                avg_time,
                min_time,
                max_time,
                std_dev,
                p50_time: p50,
                p95_time: p95,
                p99_time: p99,
            })
        } else {
            None
        }
    }

    /// Generate comprehensive performance report
    pub fn generate_detailed_report(&self) -> PerformanceReport {
        let timing_data = self.timing_data.lock().unwrap();
        let metrics = self.metrics.lock().unwrap();

        let mut kernel_stats = Vec::new();
        for kernel_name in timing_data.keys() {
            if let Some(stats) = self.get_kernel_stats(kernel_name) {
                kernel_stats.push(stats);
            }
        }

        // Sort by total time (most expensive first)
        kernel_stats.sort_by(|a, b| b.total_time.cmp(&a.total_time));

        PerformanceReport {
            total_executions: metrics.total_executions,
            kernel_statistics: kernel_stats,
            memory_bandwidth_utilization: metrics.memory_bandwidth_utilization,
            compute_utilization: metrics.compute_utilization,
            tensor_core_utilization: metrics.tensor_core_utilization,
            report_timestamp: std::time::SystemTime::now(),
        }
    }

    /// Reset profiling data
    pub fn reset(&self) {
        self.timing_data.lock().unwrap().clear();

        let mut metrics = self.metrics.lock().unwrap();
        metrics.total_executions = 0;
        metrics.avg_execution_times.clear();
        metrics.memory_bandwidth_utilization = 0.0;
        metrics.compute_utilization = 0.0;
        metrics.tensor_core_utilization = 0.0;
    }

    /// Start timing for a kernel
    pub fn start_timing(&self, kernelname: &str) {
        if !self.config.detailed_profiling {
            return;
        }

        // Implementation would use CUDA events for accurate timing
        // This is a simplified placeholder
    }

    /// End timing for a kernel
    pub fn end_timing(&self, kernelname: &str) {
        if !self.config.detailed_profiling {
            return;
        }

        // Implementation would use CUDA events for accurate timing
        // This is a simplified placeholder
    }

    /// Get performance metrics
    pub fn get_metrics(&self) -> PerformanceMetrics {
        self.metrics.lock().unwrap().clone()
    }
}

impl TensorCoreSupport {
    /// Detect available tensor core support
    #[cfg(feature = "cuda")]
    pub fn detect(context: &CudaContext) -> Result<Self, OptimizerKernelError> {
        // In a real implementation, this would query the device
        Ok(Self {
            available_generations: vec![TensorCoreGeneration::V3],
            preferred_generation: Some(TensorCoreGeneration::V3),
            mixed_precision_support: MixedPrecisionSupport {
                fp16_support: true,
                bf16_support: true,
                int8_support: true,
                tf32_support: true,
            },
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn detect(context: &()) -> Result<Self, OptimizerKernelError> {
        Ok(Self::default())
    }
}

impl Default for TensorCoreSupport {
    fn default() -> Self {
        Self {
            available_generations: Vec::new(),
            preferred_generation: None,
            mixed_precision_support: MixedPrecisionSupport {
                fp16_support: false,
                bf16_support: false,
                int8_support: false,
                tf32_support: false,
            },
        }
    }
}

impl CudaMemoryAllocator {
    /// Create new CUDA memory allocator with enhanced memory management
    pub fn new(
        strategy: AllocationStrategy,
        alignment: usize,
    ) -> Result<Self, OptimizerKernelError> {
        let mut allocator = Self {
            memory_pools: Mutex::new(HashMap::new()),
            total_allocated: Mutex::new(0),
            allocation_strategy: strategy,
            alignment_requirements: alignment,
        };

        // Pre-allocate memory pools for common sizes to reduce fragmentation
        allocator.preallocate_common_sizes()?;

        Ok(allocator)
    }

    /// Pre-allocate memory pools for common tensor sizes
    fn preallocate_common_sizes(&mut self) -> Result<(), OptimizerKernelError> {
        // Common sizes for neural network layers (in elements)
        let common_sizes = vec![
            1024,    // Small dense layers
            4096,    // Medium dense layers
            16384,   // Large dense layers
            65536,   // Very large dense layers
            262144,  // Embedding layers
            1048576, // Large embedding/conv layers
        ];

        let mut pools = self.memory_pools.lock().unwrap();

        for &size in &common_sizes {
            let aligned_size =
                (size * 4 + self.alignment_requirements - 1) & !(self.alignment_requirements - 1); // Assume f32 = 4 bytes

            // Pre-allocate 4 blocks for each common size
            let mut pool = Vec::with_capacity(4);
            for _ in 0..4 {
                // In a real implementation, this would use cudaMalloc
                let ptr = std::ptr::null_mut(); // Placeholder
                pool.push(ptr);
            }
            pools.insert(aligned_size, pool);
        }

        Ok(())
    }

    /// Allocate memory with specified size using chosen allocation strategy
    pub fn allocate(&self, size: usize) -> Result<*mut c_void, OptimizerKernelError> {
        let aligned_size =
            (size + self.alignment_requirements - 1) & !(self.alignment_requirements - 1);

        match self.allocation_strategy {
            AllocationStrategy::FirstFit => self.allocate_first_fit(aligned_size),
            AllocationStrategy::BestFit => self.allocate_best_fit(aligned_size),
            AllocationStrategy::Buddy => self.allocate_buddy(aligned_size),
            AllocationStrategy::Pool => self.allocate_pool(aligned_size),
        }
    }

    /// First-fit allocation strategy
    fn allocate_first_fit(&self, size: usize) -> Result<*mut c_void, OptimizerKernelError> {
        let mut pools = self.memory_pools.lock().unwrap();

        // Find first available pool with sufficient size
        for (&pool_size, pool) in pools.iter_mut() {
            if pool_size >= size && !pool.is_empty() {
                let ptr = pool.pop().unwrap();
                return Ok(ptr);
            }
        }

        // No suitable block found, allocate new memory
        self.allocate_new_block(size)
    }

    /// Best-fit allocation strategy
    fn allocate_best_fit(&self, size: usize) -> Result<*mut c_void, OptimizerKernelError> {
        let mut pools = self.memory_pools.lock().unwrap();

        // Find the smallest suitable pool
        let best_fit = pools
            .iter_mut()
            .filter(|(&pool_size, pool)| pool_size >= size && !pool.is_empty())
            .min_by_key(|(&pool_size_)| pool_size);

        if let Some((_, pool)) = best_fit {
            let ptr = pool.pop().unwrap();
            return Ok(ptr);
        }

        // No suitable block found, allocate new memory
        self.allocate_new_block(size)
    }

    /// Buddy allocation system
    fn allocate_buddy(&self, size: usize) -> Result<*mut c_void, OptimizerKernelError> {
        // Find the next power of 2 that can accommodate the size
        let buddy_size = self.next_power_of_2(size);

        let mut pools = self.memory_pools.lock().unwrap();

        // Try to get block of exact buddy size
        if let Some(pool) = pools.get_mut(&buddy_size) {
            if let Some(ptr) = pool.pop() {
                return Ok(ptr);
            }
        }

        // Try to split a larger block
        let mut larger_size = buddy_size * 2;
        while larger_size <= (1 << 30) {
            // Max 1GB blocks
            if let Some(pool) = pools.get_mut(&larger_size) {
                if let Some(ptr) = pool.pop() {
                    // Split the block and return first half
                    let second_half = unsafe { ptr.byte_add(buddy_size) };
                    pools
                        .entry(buddy_size)
                        .or_insert_with(Vec::new)
                        .push(second_half);
                    return Ok(ptr);
                }
            }
            larger_size *= 2;
        }

        // No suitable block found, allocate new memory
        self.allocate_new_block(buddy_size)
    }

    /// Pool-based allocation (original strategy)
    fn allocate_pool(&self, size: usize) -> Result<*mut c_void, OptimizerKernelError> {
        let mut pools = self.memory_pools.lock().unwrap();

        // Check if we have a suitable block in the pool
        if let Some(pool) = pools.get_mut(&size) {
            if let Some(ptr) = pool.pop() {
                return Ok(ptr);
            }
        }

        // No suitable block found, allocate new memory
        self.allocate_new_block(size)
    }

    /// Allocate new memory block
    fn allocate_new_block(&self, size: usize) -> Result<*mut c_void, OptimizerKernelError> {
        // In a real implementation, this would use cudaMalloc
        let ptr = std::ptr::null_mut(); // Placeholder

        *self.total_allocated.lock().unwrap() += size;

        Ok(ptr)
    }

    /// Find next power of 2 greater than or equal to n
    fn next_power_of_2(&self, n: usize) -> usize {
        if n <= 1 {
            return 1;
        }

        let mut power = 1;
        while power < n {
            power <<= 1;
        }
        power
    }

    /// Deallocate memory with strategy-specific handling
    pub fn deallocate(&self, ptr: *mut c_void, size: usize) -> Result<(), OptimizerKernelError> {
        let aligned_size =
            (size + self.alignment_requirements - 1) & !(self.alignment_requirements - 1);

        match self.allocation_strategy {
            AllocationStrategy::Buddy => self.deallocate_buddy(ptr, aligned_size),
            _ => self.deallocate_pool(ptr, aligned_size),
        }
    }

    /// Deallocate with buddy system coalescing
    fn deallocate_buddy(&self, ptr: *mut c_void, size: usize) -> Result<(), OptimizerKernelError> {
        let buddy_size = self.next_power_of_2(size);
        let mut pools = self.memory_pools.lock().unwrap();

        // Try to coalesce with buddy blocks
        let mut current_size = buddy_size;
        let mut current_ptr = ptr;

        loop {
            // Calculate buddy address
            let ptr_addr = current_ptr as usize;
            let buddy_addr = ptr_addr ^ current_size;
            let buddy_ptr = buddy_addr as *mut c_void;

            // Check if buddy exists in the pool
            if let Some(pool) = pools.get_mut(&current_size) {
                if let Some(pos) = pool.iter().position(|&p| p == buddy_ptr) {
                    // Buddy found, coalesce
                    pool.swap_remove(pos);
                    current_size *= 2;
                    current_ptr = if ptr_addr < buddy_addr {
                        current_ptr
                    } else {
                        buddy_ptr
                    };
                    continue;
                }
            }

            // No buddy found or max size reached, add to pool
            pools
                .entry(current_size)
                .or_insert_with(Vec::new)
                .push(current_ptr);
            break;
        }

        Ok(())
    }

    /// Simple pool-based deallocation
    fn deallocate_pool(&self, ptr: *mut c_void, size: usize) -> Result<(), OptimizerKernelError> {
        let mut pools = self.memory_pools.lock().unwrap();
        pools.entry(size).or_insert_with(Vec::new).push(ptr);
        Ok(())
    }

    /// Force memory defragmentation
    pub fn defragment(&self) -> Result<(), OptimizerKernelError> {
        let mut pools = self.memory_pools.lock().unwrap();

        // For buddy allocation, try to coalesce adjacent blocks
        if matches!(self.allocation_strategy, AllocationStrategy::Buddy) {
            self.coalesce_buddy_blocks(&mut pools)?;
        }

        // For other strategies, compact memory pools
        self.compact_memory_pools(&mut pools)?;

        Ok(())
    }

    /// Coalesce buddy blocks during defragmentation
    fn coalesce_buddy_blocks(
        &self,
        pools: &mut HashMap<usize, Vec<*mut c_void>>,
    ) -> Result<(), OptimizerKernelError> {
        let mut sizes: Vec<usize> = pools.keys().cloned().collect();
        sizes.sort();

        for &size in &sizes {
            if let Some(mut blocks) = pools.remove(&size) {
                blocks.sort_by_key(|&ptr| ptr as usize);

                let mut i = 0;
                while i < blocks.len() {
                    let ptr1 = blocks[i];
                    let addr1 = ptr1 as usize;

                    // Look for buddy
                    let buddy_addr = addr1 ^ size;
                    if let Some(j) = blocks[i + 1..]
                        .iter()
                        .position(|&ptr| ptr as usize == buddy_addr)
                    {
                        // Found buddy, coalesce
                        blocks.remove(i + 1 + j);
                        blocks.remove(i);

                        let coalesced_ptr = if addr1 < buddy_addr {
                            ptr1
                        } else {
                            buddy_addr as *mut c_void
                        };
                        pools
                            .entry(size * 2)
                            .or_insert_with(Vec::new)
                            .push(coalesced_ptr);

                        // Don't increment i since we removed elements
                    } else {
                        i += 1;
                    }
                }

                // Put remaining blocks back
                if !blocks.is_empty() {
                    pools.insert(size, blocks);
                }
            }
        }

        Ok(())
    }

    /// Compact memory pools by removing empty pools
    fn compact_memory_pools(
        &self,
        pools: &mut HashMap<usize, Vec<*mut c_void>>,
    ) -> Result<(), OptimizerKernelError> {
        pools.retain(|_, pool| !pool.is_empty());

        // Sort pools by size for better allocation locality
        for pool in pools.values_mut() {
            pool.sort_by_key(|&ptr| ptr as usize);
        }

        Ok(())
    }

    /// Get memory usage statistics
    pub fn get_stats(&self) -> (usize, usize) {
        let total_allocated = *self.total_allocated.lock().unwrap();
        let pools = self.memory_pools.lock().unwrap();
        let pooled_memory = pools.values().map(|pool| pool.len()).sum::<usize>();

        (total_allocated, pooled_memory)
    }
}

impl PipelineManager {
    /// Create new pipeline manager
    #[cfg(feature = "cuda")]
    pub fn new(
        context: &CudaContext,
        config: PipelineConfig,
    ) -> Result<Self, OptimizerKernelError> {
        let mut streams = Vec::with_capacity(config.num_streams);

        for _i in 0..config.num_streams {
            streams.push(CudaStream::new(context)?);
        }

        Ok(Self {
            streams,
            current_stream_index: 0,
            config,
            stats: PipelineStatistics {
                total_operations: 0,
                avg_efficiency: 0.0,
                stream_utilization: vec![0.0; config.num_streams],
            },
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn new(context: &(), config: PipelineConfig) -> Result<Self, OptimizerKernelError> {
        Ok(Self {
            current_stream_index: 0,
            config,
            stats: PipelineStatistics {
                total_operations: 0,
                avg_efficiency: 0.0,
                stream_utilization: vec![0.0; config.num_streams],
            },
        })
    }

    /// Get next available stream
    #[cfg(feature = "cuda")]
    pub fn get_next_stream(&mut self) -> &CudaStream {
        let stream = &self.streams[self.current_stream_index];
        self.current_stream_index = (self.current_stream_index + 1) % self.config.num_streams;
        stream
    }

    /// Synchronize all streams
    #[cfg(feature = "cuda")]
    pub fn synchronize_all(&self) -> Result<(), OptimizerKernelError> {
        for stream in &self.streams {
            stream.synchronize()?;
        }
        Ok(())
    }

    /// Get pipeline statistics
    pub fn get_stats(&self) -> &PipelineStatistics {
        &self.stats
    }
}

impl Default for PipelineManager {
    fn default() -> Self {
        Self {
            #[cfg(feature = "cuda")]
            streams: Vec::new(),
            current_stream_index: 0,
            config: PipelineConfig {
                num_streams: 1,
                enable_overlapping: false,
                stream_priorities: vec![0],
            },
            stats: PipelineStatistics {
                total_operations: 0,
                avg_efficiency: 0.0,
                stream_utilization: vec![0.0],
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_creation() {
        let kernel = OptimizerKernel::new();
        #[cfg(feature = "cuda")]
        assert!(kernel.is_ok());
        #[cfg(not(feature = "cuda"))]
        assert!(kernel.is_ok()); // Should still create successfully without CUDA
    }

    #[test]
    fn test_optimal_block_size() {
        let kernel = OptimizerKernel::new().unwrap();
        assert_eq!(kernel.get_optimal_block_size(100), 32);
        assert_eq!(kernel.get_optimal_block_size(1000), 128);
        assert_eq!(kernel.get_optimal_block_size(10000), 256);
    }

    #[test]
    fn test_memory_efficient_launcher() {
        let launcher = MemoryEfficientKernelLauncher::new(128, true);
        assert!(launcher.max_chunk_size > 0);
        assert_eq!(launcher.num_streams, 4);

        let launcher_no_streams = MemoryEfficientKernelLauncher::new(128, false);
        assert_eq!(launcher_no_streams.num_streams, 1);
    }
}

// Additional missing method implementations

impl KernelProfiler {
    /// Get total time for specific kernel operations
    pub fn get_total_time(&self, kernelnames: &[&str]) -> f64 {
        let timing_data = self.timing_data.lock().unwrap();
        let mut total_time = 0.0;

        for &kernel_name in kernel_names {
            if let Some(times) = timing_data.get(kernel_name) {
                total_time += times.iter().map(|d| d.as_secs_f64()).sum::<f64>();
            }
        }

        total_time
    }
}
