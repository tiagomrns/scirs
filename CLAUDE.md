# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SciRS2 is a comprehensive scientific computing and AI/ML infrastructure in Rust, providing SciPy-compatible APIs while leveraging Rust's performance, safety, and concurrency features. The project contains over 1.5 million lines of code across 24 modular crates.

## Development Commands

### Build Commands
```bash
cargo build                    # Build all workspace members
cargo build --release         # Release build with optimizations
```

### Test Commands
```bash
cargo test
cargo test -p scirs2-{module}  # Test specific module
```

### Development Workflow
1. Format code: `cargo fmt`
2. Lint: `cargo clippy` (MUST have zero warnings)
3. Build: `cargo build`
4. Test: `cargo test | tail`
5. Fix any issues and return to step 1
6. Only proceed when ALL steps pass cleanly
7. Commit & Push only when everything passes

### Reference Source Paths

Read your memory

When implementing SciPy-compatible functionality, directly read these source files for reference.

## Critical Development Principles

### Zero Warnings Policy
- **MANDATORY**: Fix ALL build errors AND warnings
- Applies to: samples, unit tests, DOC tests
- Use `#[allow(clippy::too_many_arguments)]` for unavoidable "too many arguments"
- Use `#[allow(dead_code)]` for "function is never used" warnings
- Run `cargo clippy` before every commit

### Testing Requirements
- **ALWAYS** use `cargo nextest run` instead of `cargo test`
- Test all code paths including edge cases
- Numerical comparison tests against SciPy reference
- Performance benchmarks for critical operations
- DOC tests for all public APIs

### API Updates
- rand 0.9.x: Update API calls (gen_range → random_range, thread_rng → rng)
- Maintain SciPy API compatibility where reasonable
- Document any deviations from SciPy's API

## Workspace Architecture

### Crate Structure
```
scirs2/                  # Main integration crate (re-exports all modules)
├── scirs2-core/        # Core utilities (MUST be used by all modules)
├── scirs2-linalg/      # Linear algebra
├── scirs2-stats/       # Statistics and distributions
├── scirs2-optimize/    # Optimization algorithms
├── scirs2-integrate/   # Integration and ODEs
├── scirs2-interpolate/ # Interpolation
├── scirs2-fft/        # Fast Fourier Transform
├── scirs2-special/    # Special functions
├── scirs2-signal/     # Signal processing
├── scirs2-sparse/     # Sparse matrices
├── scirs2-spatial/    # Spatial algorithms
├── scirs2-cluster/    # Clustering
├── scirs2-ndimage/    # N-dimensional images
├── scirs2-io/         # Input/output
├── scirs2-datasets/   # Sample datasets
├── scirs2-autograd/   # Automatic differentiation
├── scirs2-neural/     # Neural networks
├── scirs2-optim/      # ML optimizers
├── scirs2-graph/      # Graph processing
├── scirs2-transform/  # Data transformation
├── scirs2-metrics/    # ML metrics
├── scirs2-text/       # Text processing
├── scirs2-vision/     # Computer vision
└── scirs2-series/     # Time series
```

### Dependency Rules
- scirs2-core: No dependencies on other project crates
- scirs2: Depends on all crates via feature flags
- Use workspace inheritance: `dependency = { workspace = true }`

## Core Module Usage Policy

### MANDATORY Core Utilities
Always use scirs2-core modules instead of implementing your own:
- `scirs2-core::validation` - Parameter validation (check_positive, check_shape, check_finite)
- `scirs2-core::error` - Base error types
- `scirs2-core::numeric` - Generic numerical operations
- `scirs2-core::cache` - Caching mechanisms
- `scirs2-core::constants` - Mathematical/physical constants
- `scirs2-core::utils` - Common utilities

### Strict Acceleration Policy

#### SIMD Operations
- **MANDATORY**: Use `scirs2-core::simd_ops::SimdUnifiedOps` trait
- **FORBIDDEN**: Direct use of `wide`, `packed_simd`, or platform intrinsics
- **FORBIDDEN**: Custom SIMD implementations in modules

```rust
// GOOD
use scirs2_core::simd_ops::SimdUnifiedOps;
let result = f32::simd_add(&a.view(), &b.view());

// BAD - NEVER do this
// let result = custom_simd_add(a, b);
```

#### Parallel Processing
- **MANDATORY**: Use `scirs2-core::parallel_ops` for all parallelism
- **FORBIDDEN**: Direct dependency on `rayon` in modules
- **REQUIRED**: Import via `use scirs2_core::parallel_ops::*`

```rust
// GOOD
use scirs2_core::parallel_ops::*;

// BAD - NEVER do this
// use rayon::prelude::*;
```

#### GPU Operations
- **MANDATORY**: Use `scirs2-core::gpu` module
- **FORBIDDEN**: Direct CUDA/OpenCL/Metal API calls
- **FORBIDDEN**: Custom kernel implementations

#### Platform Detection
- **MANDATORY**: Use `scirs2-core::simd_ops::PlatformCapabilities::detect()`
- **FORBIDDEN**: Custom CPU feature detection

## Performance Guidelines

### Optimization Priority
1. Use core-provided optimizations first
2. Provide scalar fallbacks for SIMD/parallel code
3. Benchmark against SciPy for comparison
4. Profile before optimizing

### Memory Efficiency
- Use chunked operations for large data
- Leverage ndarray's zero-copy views
- Minimize allocations in hot paths
- Use `ArrayViewMut` instead of cloning

## Code Style and Organization

### File Structure
- Keep files under 500 lines when possible
- Organize by functionality, not by type
- Use subdirectories for related functionality
- Separate public API from implementation

### Naming Conventions
- `mod.rs`: Public interface and re-exports
- `implementation.rs`: Core implementation
- `utils.rs`: Module-specific utilities
- Follow Rust naming conventions strictly

### Documentation Requirements
- Document all public APIs
- Include usage examples in doc comments
- Add references to papers/algorithms
- Document performance characteristics
- Note thread-safety guarantees

## Common Patterns

### Error Handling
```rust
use scirs2_core::error::Result;

pub fn my_function() -> Result<T> {
    // Implementation
}
```

### Parameter Validation
```rust
use scirs2_core::validation::{check_positive, check_shape};

pub fn process(data: &Array2<f64>, k: usize) -> Result<()> {
    check_positive(k, "k")?;
    check_shape(data, (None, Some(3)), "data")?;
    // Implementation
}
```

### Feature Flags
```toml
[dependencies]
scirs2-core = { workspace = true, features = ["simd", "parallel", "gpu"] }

[features]
default = ["parallel"]
simd = ["scirs2-core/simd"]
parallel = ["scirs2-core/parallel"]
gpu = ["scirs2-core/gpu"]
```

## Continuous Integration

The project uses GitHub Actions with:
- Rust stable toolchain
- cargo-nextest for testing
- System dependencies: OpenBLAS, LAPACK, etc.
- Zero warnings enforcement
- Comprehensive test coverage

## Version Information
- Current version: 0.1.0-beta.1
- Repository: https://github.com/cool-japan/scirs
- Main branch: master