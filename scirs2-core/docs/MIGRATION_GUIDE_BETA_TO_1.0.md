# SciRS2-Core Migration Guide: Beta → 1.0

This guide helps users migrate from SciRS2-Core Beta releases to the stable 1.0 version.

## Overview

SciRS2-Core 1.0 represents our commitment to API stability and production readiness. This guide covers all breaking changes and provides clear migration paths for existing code.

## Breaking Changes

### 1. Error Handling

**Beta API:**
```rust
use scirs2_core::error::CoreError;

fn process_data() -> Result<Vec<f64>, CoreError> {
    // Beta error handling
    Err(CoreError::InvalidInput("Bad data".to_string()))
}
```

**1.0 API:**
```rust
use scirs2_core::error::{CoreError, ErrorContext};

fn process_data() -> Result<Vec<f64>, CoreError> {
    // 1.0 error handling with context
    Err(CoreError::invalid_input("Bad data")
        .with_context("process_data")
        .with_field("expected", "positive values"))
}
```

**Migration Steps:**
1. Replace string-based error constructors with builder methods
2. Add context information for better error diagnostics
3. Use the new `ErrorContext` trait for custom error types

### 2. Array Protocol

**Beta API:**
```rust
use scirs2_core::array_protocol::ArrayProtocol;

impl ArrayProtocol for MyArray {
    fn shape(&self) -> &[usize] { &self.shape }
    fn strides(&self) -> &[isize] { &self.strides }
    // ... other methods
}
```

**1.0 API:**
```rust
use scirs2_core::array_protocol::{ArrayProtocol, ArrayMetadata};

impl ArrayProtocol for MyArray {
    fn metadata(&self) -> ArrayMetadata {
        ArrayMetadata::new(self.shape.clone(), self.strides.clone())
            .with_device(self.device)
            .with_layout(self.layout)
    }
    // ... simplified interface
}
```

**Migration Steps:**
1. Implement the new `metadata()` method
2. Remove individual getter methods
3. Use `ArrayMetadata` builder for rich metadata

### 3. Memory Management

**Beta API:**
```rust
use scirs2_core::memory::MemoryPool;

let pool = MemoryPool::new(1024 * 1024 * 100); // 100MB
let buffer = pool.allocate(1024)?;
```

**1.0 API:**
```rust
use scirs2_core::memory::{MemoryPool, PoolConfig};

let pool = MemoryPool::builder()
    .initial_size(100 * 1024 * 1024)
    .growth_factor(1.5)
    .enable_metrics()
    .build()?;
let buffer = pool.allocate_aligned(1024, 64)?; // 64-byte alignment
```

**Migration Steps:**
1. Use the builder pattern for pool configuration
2. Specify alignment requirements explicitly
3. Enable metrics for production monitoring

### 4. GPU Operations

**Beta API:**
```rust
use scirs2_core::gpu::{GpuArray, GpuDevice};

let device = GpuDevice::new(0)?;
let gpu_array = GpuArray::from_slice(&data, &device)?;
```

**1.0 API:**
```rust
use scirs2_core::gpu::{GpuArray, GpuContext, DeviceSelector};

let context = GpuContext::new()
    .device_selector(DeviceSelector::BestAvailable)
    .enable_profiling()
    .build()?;
let gpu_array = context.array_from_slice(&data)?;
```

**Migration Steps:**
1. Replace direct device creation with `GpuContext`
2. Use `DeviceSelector` for automatic device selection
3. Manage GPU resources through context

### 5. Validation Framework

**Beta API:**
```rust
use scirs2_core::validation::Validator;

let validator = Validator::new()
    .add_rule(|x| x > 0.0)
    .add_rule(|x| x < 100.0);
```

**1.0 API:**
```rust
use scirs2_core::validation::{Validator, ValidationRule, Constraint};

let validator = Validator::builder()
    .add_constraint(Constraint::range(0.0..100.0))
    .add_constraint(Constraint::not_nan())
    .on_failure(ValidationFailureAction::Correct)
    .build();
```

**Migration Steps:**
1. Use predefined constraints instead of closures
2. Specify failure handling strategies
3. Leverage the builder pattern for complex validators

## Feature Changes

### Removed Features
- `experimental_jit`: Moved to separate `scirs2-jit` crate
- `legacy_api`: Removed deprecated APIs
- `unsafe_optimizations`: Replaced with safe alternatives

### New Features
- `production`: Comprehensive production features bundle
- `observability`: OpenTelemetry integration
- `cloud_storage`: S3/GCS/Azure backend support

### Renamed Features
- `serialize` → `serialization`
- `gpu_cuda` → `cuda`
- `gpu_opencl` → `opencl`

## Performance Considerations

### Memory Allocation
1.0 introduces aligned allocations by default, which may increase memory usage slightly but improves performance:

```rust
// Beta: Unaligned allocation
let arr = Array::zeros(1000);

// 1.0: Aligned allocation for SIMD
let arr = Array::zeros_aligned(1000, Alignment::SIMD);
```

### Parallel Processing
The parallel scheduler has been optimized in 1.0:

```rust
// Beta: Manual thread pool
let pool = ThreadPool::new(num_cpus::get());

// 1.0: Automatic work-stealing scheduler
use scirs2_core::parallel::par_apply;
par_apply(&mut array, |x| x * 2.0);
```

## Migration Checklist

- [ ] Update `Cargo.toml` to use `scirs2-core = "1.0"`
- [ ] Run `cargo fix --edition-idioms` to update code style
- [ ] Replace deprecated error handling patterns
- [ ] Update array protocol implementations
- [ ] Migrate memory pool configurations
- [ ] Update GPU initialization code
- [ ] Review and update validation rules
- [ ] Update feature flags in `Cargo.toml`
- [ ] Run test suite with `--all-features`
- [ ] Profile performance-critical sections
- [ ] Update documentation references

## Common Issues and Solutions

### Issue: Compilation errors after upgrade
**Solution:** Check for removed features and update `Cargo.toml` accordingly.

### Issue: Performance regression
**Solution:** Enable the `production` feature flag and review memory alignment.

### Issue: GPU operations fail
**Solution:** Update to use `GpuContext` and check device compatibility.

### Issue: Validation behaves differently
**Solution:** Review constraint definitions and failure handling strategies.

## Getting Help

- **Documentation**: [https://docs.rs/scirs2-core/1.0](https://docs.rs/scirs2-core/1.0)
- **Migration examples**: See `examples/migration/` directory
- **Community support**: GitHub Discussions
- **Commercial support**: Available for enterprise users

## Version Support Timeline

- Beta versions: Security updates until 2026-01-01
- 1.0.x: Long-term support (LTS) until 2028-01-01
- Future versions: Follow semantic versioning

---

*This guide is part of the SciRS2-Core documentation. For the latest version, visit our [GitHub repository](https://github.com/scirs/scirs2-core).*