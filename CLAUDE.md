# CLAUDE.md - Instructions for AI Assistant

## Development Principles

- Fix all build errors AND warnings - zero warnings policy for samples, unit tests, and DOC tests
- Update rand API when using 0.9.1 (gen_range → random_range, thread_rng → rng)
- Always follow this workflow:
  1. Format code with `cargo fmt`
  2. Check for linting issues with `cargo clippy`
  3. Build (`cargo build`)
  4. Test (unit tests, sample code tests, DOC tests with `cargo test`)
  5. Fix issues
  6. Return to step 1 if ANY fixes were made - NEVER proceed until all steps pass cleanly
  7. Commit & Push (ONLY if all builds and tests pass without errors or warnings)
- CRITICAL: Continue the build-test-fix cycle until ALL errors AND warnings are resolved
- Never consider work complete until the entire workflow passes completely
- Always run builds and tests again after making changes, unless the user explicitly requests "commit & push only"
- Prefer OptimizedDataFrame over old DataFrame implementation
- Remove any DataFrame tests/samples that cause errors

## Project-Specific Guidelines

- Maintain API compatibility with SciPy where possible
- Follow Rust idioms and best practices
- Ensure comprehensive test coverage
- Document all public APIs
- Benchmark against SciPy for performance comparison
- Focus on memory safety and efficiency
- Leverage Rust's parallelism via Rayon where appropriate
- Use repository URL in package metadata: https://github.com/cool-japan/scirs

## Modular Architecture

- Follow a modular crate structure:
  - **scirs2-core**: Core utilities and common functionality
  - **scirs2-{module}**: Domain-specific functionality (linalg, stats, etc.)
  - **scirs2**: Main crate that re-exports from other crates

- Dependency Structure:
  - scirs2-core: No dependencies on other project crates
  - scirs2-{module}: Depends only on scirs2-core, not on other modules
  - scirs2: Depends on all other crates, enables them via feature flags

- Error Handling:
  - Each module defines its own error types
  - Provide conversions between module errors and core errors
  - No circular dependencies in error handling

## Dependency Management

- Use workspace inheritance for consistent dependency versioning:
  - Define all shared dependencies with their versions in the root `Cargo.toml`
  - Use `workspace = true` in module Cargo.toml files instead of specifying versions directly
  
- When adding a new dependency:
  1. First check if the dependency is already defined in root `Cargo.toml`
  2. If already defined, reference it with `dependency.workspace = true` in the module
  3. If not defined but will be used by multiple modules:
     - Add it to the root `Cargo.toml` first
     - Then reference it with `workspace = true` in the module
  4. If it's a module-specific dependency unlikely to be shared:
     - Define it directly in the module's `Cargo.toml`

- How to handle feature-gated dependencies with workspace inheritance:
  
  ```toml
  # In workspace root Cargo.toml - define dependencies normally
  reqwest = { version = "0.11", features = ["blocking", "json"] }
  cached = "0.48.1"
  
  # In module Cargo.toml - for feature-gated dependencies, use workspace = true WITH optional = true
  reqwest = { workspace = true, optional = true }
  cached = { workspace = true, optional = true }
  
  # Then in your features section:
  [features]
  online = ["reqwest"]  # This enables the optional reqwest dependency
  caching = ["cached"]  # This enables the optional cached dependency
  ```
  
  - ALWAYS use `workspace = true` for dependencies defined in the root Cargo.toml
  - Add `optional = true` to dependencies used in feature flags
  - If a dependency needs different features in different modules, you can still use workspace inheritance:
    ```toml
    # In module A
    serde = { workspace = true, features = ["derive"] }
    
    # In module B
    serde = { workspace = true, features = ["derive", "rc"] }
    ```
     
- Categorize dependencies in root `Cargo.toml` for clarity:
  ```toml
  # Core dependencies
  ndarray = { version = "0.16.1", features = ["serde", "rayon"] }
  
  # Math-specific dependencies
  special = "0.10.2"
  
  # IO dependencies
  serde = { version = "1.0", features = ["derive"] }
  ```
  
- Periodically review direct version specifications in module `Cargo.toml` files to identify candidates for workspace inheritance

## Core Module Usage Policy

- ALWAYS use scirs2-core modules for common functionality to avoid reinventing the wheel:
  - Use `scirs2-core::validation` for parameter checking (e.g., check_positive, check_shape, check_finite)
  - Use `scirs2-core::error` as base for module-specific error types
  - Use `scirs2-core::numeric` for generic numerical operations
  - Use `scirs2-core::cache` when implementing caching mechanisms
  - Use `scirs2-core::config` for configuration settings
  - Use `scirs2-core::constants` for mathematical and physical constants
  - Use `scirs2-core::parallel` for parallel processing (when feature-enabled)
  - Use `scirs2-core::simd_ops` for ALL SIMD operations - NEVER implement custom SIMD
  - Use `scirs2-core::gpu` for ALL GPU operations - NEVER implement custom GPU kernels
  - Use `scirs2-core::utils` for common utility functions
  - Use `scirs2-core::simd_ops::PlatformCapabilities` for platform detection
  - Use `scirs2-core::simd_ops::AutoOptimizer` for automatic optimization selection
- Before implementing new utility functions in module-specific crates, always check if similar functionality already exists in scirs2-core
- When you find duplicate functionality that should be in scirs2-core, create a task to refactor it

## Strict Acceleration Policy

### SIMD Operations
- **MANDATORY**: Use `scirs2-core::simd_ops::SimdUnifiedOps` trait for all SIMD operations
- **FORBIDDEN**: Direct use of `wide`, `packed_simd`, or platform-specific SIMD intrinsics in modules
- **FORBIDDEN**: Custom SIMD implementations in individual modules
- All SIMD operations MUST go through the unified abstraction layer
- Example usage:
  ```rust
  use scirs2_core::simd_ops::SimdUnifiedOps;
  
  // Good - uses core SIMD operations
  let result = f32::simd_add(&a.view(), &b.view());
  
  // Bad - direct SIMD implementation
  // let result = custom_simd_add(a, b);  // FORBIDDEN
  ```

### GPU Operations
- **MANDATORY**: Use `scirs2-core::gpu` module for all GPU operations
- **FORBIDDEN**: Direct CUDA/OpenCL/Metal API calls in individual modules
- **FORBIDDEN**: Custom kernel implementations outside of core
- GPU kernels must be registered in the core GPU kernel registry
- Use feature flags to conditionally enable GPU support

### BLAS Operations
- **MANDATORY**: All BLAS operations go through `scirs2-core`
- **CONFIGURED**: BLAS backend selection is handled by core's platform-specific configuration
- **FORBIDDEN**: Direct dependency on BLAS libraries in individual modules
- Modules should only depend on `scirs2-core` with appropriate features enabled

### Platform Detection
- **MANDATORY**: Use `scirs2-core::simd_ops::PlatformCapabilities::detect()` for capability detection
- **FORBIDDEN**: Custom CPU feature detection in modules
- **FORBIDDEN**: Duplicate platform detection code
- Example:
  ```rust
  use scirs2_core::simd_ops::PlatformCapabilities;
  
  let caps = PlatformCapabilities::detect();
  if caps.simd_available {
      // Use SIMD path
  }
  ```

### Optimization Selection
- **MANDATORY**: Use `scirs2-core::simd_ops::AutoOptimizer` for automatic optimization selection
- The optimizer decides whether to use GPU, SIMD, or scalar implementations based on:
  - Problem size
  - Available hardware capabilities
  - Performance heuristics
- Example:
  ```rust
  use scirs2_core::simd_ops::AutoOptimizer;
  
  let optimizer = AutoOptimizer::new();
  if optimizer.should_use_gpu(problem_size) {
      // Use GPU implementation from core
  } else if optimizer.should_use_simd(problem_size) {
      // Use SIMD implementation from core
  } else {
      // Use scalar implementation
  }

## Performance Optimization Policy

- For performance-critical code, ALWAYS use core-provided optimizations:
  - SIMD: use `scirs2-core::simd_ops` trait methods - NEVER implement custom SIMD
  - Parallelism: use `scirs2-core::parallel_ops` instead of direct Rayon usage
  - Memory efficiency: use `chunk_wise_op` and other core-provided memory-efficient algorithms
  - Caching: use `TTLSizedCache`, `CacheBuilder`, or `#[cached]` from core rather than custom caching solutions
  - GPU: use `scirs2-core::gpu` module for all GPU operations
- Each module should enable relevant core features in its Cargo.toml:
  ```toml
  [dependencies]
  scirs2-core = { workspace = true, features = ["simd", "parallel", "gpu"] }
  ```
- Provide scalar fallbacks for operations that use SIMD or parallel processing
- Never reimplement optimization code that exists in core modules

### Parallel Processing Policy

- **MANDATORY**: Use `scirs2-core::parallel_ops` for all parallel operations
- **FORBIDDEN**: Direct dependency on `rayon` in module Cargo.toml files
- **REQUIRED**: Import parallel functionality via `use scirs2_core::parallel_ops::*`
- The parallel_ops module provides:
  - Full Rayon functionality when `parallel` feature is enabled
  - Sequential fallbacks when `parallel` feature is disabled
  - Helper functions: `par_range()`, `par_chunks()`, `par_scope()`, `par_join()`
  - Runtime checks: `is_parallel_enabled()`, `num_threads()`
- Example migration:
  ```rust
  // Old - direct Rayon usage
  use rayon::prelude::*;
  
  // New - use core abstractions
  use scirs2_core::parallel_ops::*;
  ```

## Refactoring Priority

When you encounter code that violates these policies, prioritize refactoring in this order:
1. **SIMD implementations** - Replace all custom SIMD with `scirs2-core::simd_ops`
2. **GPU implementations** - Centralize all GPU kernels in `scirs2-core::gpu`
3. **Platform detection** - Replace with `PlatformCapabilities::detect()`
4. **BLAS operations** - Ensure all go through core
5. **Parallel operations** - Replace direct Rayon usage with core abstractions

## Implementation Notes

- Use ndarray for multidimensional arrays
- Prefer pure Rust implementations where possible
- Use FFI bindings to established libraries where necessary for performance
- Use const generics for dimension-aware computations where applicable
- Ensure proper error handling with Result types
- Use traits for algorithm abstractions
- Leverage trait-based design for flexible, extensible interfaces
- Prefer composition over inheritance when designing components
- Utilize Rust's type system to ensure correctness at compile time

## Clippy Compliance Rules

- Follow Rust's idiomatic patterns to avoid common Clippy warnings:
  - Use iterators with `enumerate()` instead of needless range loops
  - Create type aliases for complex return types to avoid `type_complexity` warnings
  - Keep function argument count under 7 when possible, or create builder patterns
  - Use `.contains()` method for range checks instead of manual comparisons
  - Avoid redundant `return` statements at the end of functions/blocks
  - Prefix unused parameters with underscore (`_param_name`)
  - Use `&mut` iterators instead of index-based loops for modifying elements
  - Avoid field reassignment with default by using struct initialization syntax
  - Use string methods like `.to_string()` instead of `format!()` for simple strings
  - Use `match` statements rather than `if/else if` chains for comparisons
- Run `cargo clippy` before every commit and address all warnings
- When fixing Clippy warnings, look for patterns and fix similar issues across the codebase
- For unavoidable Clippy warnings, use targeted `#[allow(clippy::specific_lint)]` annotations with comments explaining why

## Code Organization Strategy

- Split large implementation files into smaller, focused modules
  - Organize modules by functionality rather than by type
  - Create separate files for complex algorithms
  - Use subdirectories for related functionality groups
- Adopt consistent file naming conventions:
  - `module/mod.rs`: Public module interface and re-exports
  - `module/implementation.rs`: Core implementation details
  - `module/utils.rs`: Shared utilities for the module
- Aim for files under 500 lines of code where possible
- For complex implementations:
  - Create a directory structure matching the logical components
  - Separate public API from internal implementation details
  - Use private modules for implementation specifics
- Structure modules to optimize for AI code assistance:
  - Maintain clear boundaries between different concerns
  - Document module relationships explicitly
  - Keep related code together, even if it means some duplication

## Testing Approach

- Unit tests for all functionality
- Property-based testing for mathematical properties
- Numerical comparison tests against SciPy reference results
- Performance benchmarks
- DOC tests for all public APIs
- When substantial architectural changes are made that break DOC tests:
  - Temporarily mark broken DOC tests with `ignore` attribute (````ignore`)
  - Add `# FIXME:` comments explaining the issue
  - Create an issue to track and fix these tests later
  - Ensure library code still compiles and passes unit tests
- Build incremental test coverage:
  - Start with basic functionality tests
  - Add edge cases as implementation matures
  - Create separate test modules for complex functionality
- Isolate test dependencies:
  - Mock external systems when possible
  - Avoid tests that depend on specific environments
  - Use feature flags to conditionally compile integration tests

## AI Collaboration Best Practices

- Provide clear context for each development task:
  - Current state of implementation
  - Design constraints and requirements
  - References to related modules or documentation
- Break complex tasks into smaller, well-defined steps:
  - Define interfaces before implementations
  - Implement core functionality before edge cases
  - Review and refine incrementally
- Structure code for easier AI assistance:
  - Add detailed comments for complex algorithms
  - Use consistent naming patterns 
  - Provide examples of similar implementations elsewhere in the codebase
- Document AI collaboration workflow:
  - Keep a record of architectural decisions
  - Document reasoning behind design choices
  - Track issues that required multiple iterations to resolve

## Documentation Standards

- All public APIs must be documented
- Include examples in doc comments
- Provide references to papers/algorithms when applicable
- Add notes about performance characteristics
- Document any deviations from SciPy's API
- Document module relationships and dependencies
- Include high-level architectural overview in each crate's README
- Add inline documentation for complex algorithms
- Clearly mark internal-only APIs with appropriate visibility
- Use consistent documentation patterns across the codebase:
  - Function signature and return type explanations
  - Parameter descriptions with valid ranges/units
  - Error condition documentation
  - Performance characteristics and big-O complexity
  - Thread-safety and concurrency considerations