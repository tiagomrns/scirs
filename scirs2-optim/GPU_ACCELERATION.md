# GPU Acceleration for scirs2-optim

This document describes the GPU acceleration features implemented in the scirs2-optim module.

## Overview

The scirs2-optim module now includes comprehensive GPU acceleration support for machine learning optimizers, enabling significant performance improvements for large-scale training workloads.

## Features Implemented

### 1. GPU-Accelerated Optimizers

The following optimizers have GPU-accelerated implementations:

- **SGD** (Stochastic Gradient Descent)
  - Basic SGD with weight decay
  - SGD with momentum (standard and Nesterov)
  - Heavy-ball method
  - Accelerated SGD with adaptive restart

- **Adam** and **AdamW**
  - Standard Adam with bias correction
  - AdamW with decoupled weight decay
  - Mixed precision variants (FP16/BF16)
  - Fused kernels with gradient clipping

- **LAMB** (Layer-wise Adaptive Moments)
  - Layer-wise adaptation for large batch training
  - Fused norm computation
  - Multi-GPU support with efficient synchronization

- **RMSprop**
  - Standard and centered variants
  - Nesterov momentum support
  - Async memory operations for AMD GPUs

- **AdaGrad**
  - Standard AdaGrad with accumulation
  - Diagonal preconditioning
  - Window-based variant for memory efficiency
  - AdaDelta extension

### 2. Multi-GPU Support

- **Synchronization Strategies**:
  - Ring all-reduce for large tensors
  - Tree all-reduce for small tensors
  - Hierarchical all-reduce for multi-node setups
  - Pipeline parallel gradient synchronization

- **Gradient Compression**:
  - Top-k sparsification
  - Error feedback mechanism
  - Bandwidth optimization

- **Zero Redundancy Optimizer** (ZeRO-style):
  - Optimizer state partitioning
  - Reduced memory footprint
  - Efficient parameter gathering

### 3. Mixed Precision Training

- **Automatic Mixed Precision (AMP)**:
  - Dynamic loss scaling
  - Overflow detection and recovery
  - FP16/BF16 support

- **Tensor Core Optimization**:
  - Automatic dimension padding
  - GEMM configuration optimization
  - Support for Volta, Turing, and Ampere architectures

### 4. Platform Support

- **CUDA** (NVIDIA GPUs):
  - Full optimizer kernel suite
  - Tensor core acceleration
  - CUDA memory pooling

- **ROCm** (AMD GPUs):
  - HIP kernel implementations
  - Wavefront-optimized operations
  - Local Data Share (LDS) utilization

### 5. Memory Management

- **CUDA Memory Pool**:
  - Reduces allocation overhead
  - Memory reuse and caching
  - Automatic defragmentation
  - Thread-safe operations

- **Statistics and Monitoring**:
  - Memory usage tracking
  - Cache hit rate monitoring
  - Peak usage analysis

## Usage Examples

### Basic GPU Optimizer Usage

```rust
use scirs2_optim::{SGDGpu, GpuOptimizerConfig};
use ndarray::Array1;

// Create GPU-accelerated SGD optimizer
let mut optimizer = SGDGpu::new(0.01);

// Initialize GPU resources
let config = GpuOptimizerConfig::default();
optimizer.initialize_gpu(1000, config)?;
optimizer.to_gpu()?;

// Use optimizer as normal
let params = Array1::zeros(1000);
let gradients = Array1::ones(1000);
let updated = optimizer.step(&params, &gradients)?;
```

### Mixed Precision Training

```rust
use scirs2_optim::{AdamGpu, MixedPrecisionConfig, MixedPrecisionOptimizer};

// Configure mixed precision
let mp_config = MixedPrecisionConfig {
    init_scale: 65536.0,
    use_tensor_cores: true,
    ..Default::default()
};

// Wrap optimizer with mixed precision
let adam = AdamGpu::new(0.001);
let mut mp_optimizer = MixedPrecisionOptimizer::new(adam, mp_config);

// Scale loss before backward
let scaled_loss = mp_optimizer.scale_loss(loss);

// Unscale gradients and check for overflow
let has_overflow = mp_optimizer.unscale_and_check_overflow(&mut gradients)?;
if !has_overflow {
    optimizer.step(&params, &gradients)?;
}
```

### Multi-GPU Training

```rust
use scirs2_optim::{MultiGpuSetup, MultiGpuConfig, SyncStrategy};

// Setup multi-GPU configuration
let config = MultiGpuConfig {
    num_gpus: 4,
    sync_strategy: SyncStrategy::RingAllReduce,
    gradient_compression: true,
    ..Default::default()
};

// Initialize multi-GPU setup
let setup = MultiGpuSetup::new(num_gpus, max_param_size)?;

// Synchronize gradients across GPUs
for (rank, sync_manager) in setup.sync_managers.iter_mut().enumerate() {
    sync_manager.sync_gradients(&mut gradients)?;
}
```

### Memory Pool Usage

```rust
use scirs2_optim::{ThreadSafeMemoryPool, MemoryPoolConfig};

// Configure memory pool
let pool = ThreadSafeMemoryPool::new(4 * 1024 * 1024 * 1024); // 4GB

// Allocate memory from pool
let memory = pool.allocate(1024 * 1024)?; // 1MB

// Memory is automatically returned to pool when dropped
// Check statistics
let stats = pool.get_stats();
println!("Cache hit rate: {:.2}%", stats.cache_hits as f64 / 
         (stats.cache_hits + stats.cache_misses) as f64 * 100.0);
```

## Performance Considerations

1. **Batch Size**: Larger batch sizes benefit more from GPU acceleration
2. **Parameter Count**: GPU overhead is amortized better with larger models
3. **Memory Alignment**: Tensor dimensions should be multiples of 16 for tensor cores
4. **Multi-GPU Scaling**: Use appropriate synchronization strategy based on tensor size
5. **Mixed Precision**: Can provide 2-3x speedup with minimal accuracy impact

## Building with GPU Support

Enable the GPU feature in your `Cargo.toml`:

```toml
[dependencies]
scirs2-optim = { version = "0.1.0", features = ["gpu"] }
```

Ensure you have the appropriate GPU drivers and SDK installed:
- NVIDIA GPUs: CUDA Toolkit 11.0+
- AMD GPUs: ROCm 5.0+

## Future Enhancements

While comprehensive GPU support has been implemented, future improvements could include:

1. **Additional Platforms**:
   - Intel GPU support (oneAPI/SYCL)
   - Apple Metal support
   - WebGPU for browser-based training

2. **Advanced Features**:
   - Graph-based optimization
   - Kernel fusion opportunities
   - Custom kernel generation

3. **Integration**:
   - Direct integration with deep learning frameworks
   - ONNX runtime support
   - TensorRT optimization

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use gradient accumulation
2. **Kernel Launch Failures**: Check GPU compute capability
3. **Mixed Precision Overflow**: Adjust loss scaling parameters
4. **Multi-GPU Deadlock**: Ensure consistent operations across ranks

### Performance Profiling

Use the built-in statistics to identify bottlenecks:

```rust
let stats = optimizer.get_memory_stats();
println!("{}", stats); // Displays detailed memory usage

let overflow_stats = mp_optimizer.get_overflow_stats();
println!("Overflow rate: {:.2}%", overflow_stats.overflow_rate * 100.0);
```

## Contributing

Contributions to improve GPU acceleration are welcome! Areas of interest:
- Additional optimizer implementations
- Performance optimizations
- Platform-specific improvements
- Documentation and examples