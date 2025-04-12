# SciRS2 Design Principles

## Modular Architecture

SciRS2 adopts a modular structure, providing independent crates for each functional area of SciPy. This design offers several advantages:

1. **Flexible Dependency Management**: Users can select only the features they need
2. **Parallel Development**: Each module can be developed, tested, and released independently
3. **Clear Responsibility**: Each module focuses on a specific functional area
4. **Code Organization**: Logical separation of related code improves maintainability
5. **Prevention of Circular Dependencies**: Clear dependency relationships between modules

## Crate Structure

SciRS2 adopts the following crate structure:

```
/
├── Cargo.toml (Workspace configuration)
├── scirs2-core/ (Core utilities and common functionality)
│   ├── Cargo.toml
│   └── src/
│       ├── constants.rs
│       ├── error.rs
│       ├── lib.rs
│       └── utils.rs
├── scirs2-linalg/ (Linear algebra module)
│   ├── Cargo.toml
│   └── src/
│       ├── basic.rs
│       ├── blas.rs
│       ├── decomposition.rs
│       ├── eigen.rs
│       ├── error.rs
│       ├── lapack.rs
│       ├── lib.rs
│       ├── norm.rs
│       ├── solve.rs
│       └── special.rs
├── scirs2-integrate/
├── scirs2-interpolate/
├── scirs2-optimize/
├── scirs2-fft/
├── scirs2-stats/
├── scirs2-special/
├── scirs2-signal/
├── scirs2-sparse/
├── scirs2-spatial/
└── scirs2/ (Main integration crate)
    ├── Cargo.toml
    └── src/
        └── lib.rs (Re-exports from all other crates)
```

This modular structure is designed as follows:
- **scirs2-core**: Core utilities and common functionality used by other modules
- **scirs2-linalg**: Linear algebra functionality (BLAS/LAPACK wrappers, matrix operations, decompositions)
- **scirs2-integrate**: Numerical integration algorithms
- **scirs2-interpolate**: Interpolation functionality
- **scirs2-optimize**: Optimization algorithms
- **scirs2-fft**: Fast Fourier Transform
- **scirs2-stats**: Statistical functions
- **scirs2-special**: Special functions
- **scirs2-signal**: Signal processing
- **scirs2-sparse**: Sparse matrix operations
- **scirs2-spatial**: Spatial algorithms
- **scirs2**: Main integration crate that re-exports functionality from all other crates

## Dependency Structure

The modular design helps avoid circular dependencies:

1. **scirs2-core**: Contains shared utilities and has no dependencies on other project crates
2. **scirs2-{module}**: Each module depends only on scirs2-core, not on other modules
3. **scirs2**: The main crate that depends on and re-exports all other modules

### Core Module Usage Guidelines

To avoid code duplication and improve consistency across modules:

1. **Use scirs2-core validation functions**: Always use validation utilities from `scirs2-core::validation` for parameter checking, shape validation, numerical bounds, etc.
2. **Use scirs2-core error handling**: Leverage the error system from `scirs2-core::error` and extend it with module-specific error types when necessary
3. **Use scirs2-core numeric traits**: Apply numeric traits from `scirs2-core::numeric` for generic numerical operations
4. **Use scirs2-core caching mechanisms**: Employ `scirs2-core::cache` for optimizing performance-critical operations
5. **Use scirs2-core configuration system**: Utilize `scirs2-core::config` for module configuration
6. **Use scirs2-core constants**: Reference mathematical and physical constants from `scirs2-core::constants`
7. **Use scirs2-core parallel processing**: Leverage `scirs2-core::parallel` for multi-threaded operations (when feature-enabled)
8. **Use scirs2-core SIMD operations**: Apply `scirs2-core::simd` for vectorized computations (when feature-enabled)
9. **Use scirs2-core utilities**: Employ common utility functions from `scirs2-core::utils` for general operations

### Prioritizing Consistent Performance Optimization

For optimal performance and code consistency:

1. **SIMD Acceleration**: When implementing numerical algorithms that process arrays or vectors:
   - Always use `scirs2-core::simd` operations rather than implementing custom SIMD code
   - Enable the `simd` feature flag in module Cargo.toml files to access this functionality
   - Consider providing both scalar and SIMD implementations with feature flags

2. **Parallel Processing**: When processing large datasets or performing computation-heavy tasks:
   - Use `scirs2-core::parallel` instead of direct Rayon usage 
   - Enable the `parallel` feature flag in module Cargo.toml files
   - Use the `par_` prefixed functions for parallel equivalents of common operations

3. **Caching**: For computations with repeated inputs or expensive calculations:
   - Use `scirs2-core::cache::TTLSizedCache` for data that should expire
   - Use `scirs2-core::cache::CacheBuilder` to configure custom caches
   - Use the `#[cached]` macro from scirs2-core for function-level memoization

4. **Memory Efficiency**: For operations on large datasets:
   - Use `scirs2-core::parallel::chunk_wise_op` for processing data in manageable chunks
   - Use `scirs2-core::parallel::memory_efficient_cumsum` and similar operations

This approach ensures all modules benefit from the same optimizations and performance characteristics.

## Feature Selection

The main `scirs2` crate provides feature flags to selectively enable modules:

```toml
# Using only linear algebra and statistics
scirs2 = { version = "0.1.0", features = ["linalg", "stats"] }

# Using all features
scirs2 = { version = "0.1.0", features = ["all"] }
```

And each module can depend on scirs2-core:

```toml
[dependencies]
scirs2-core = { version = "0.1.0" }
```

## Error Handling

Each sub-crate defines its own error types to avoid circular dependencies:

1. **scirs2-core**: Defines base error traits and common error handling utilities
2. **scirs2-{module}**: Each module defines its own error types specific to its domain
3. **Error Conversion**: Each module provides conversions to/from core error types
4. **Result Types**: All functions use the `Result` type with appropriate error types
5. **Detailed Messages**: Error variants include detailed information for debugging

## Core Design Principles

1. **API Compatibility**: Maintain API similarity with SciPy where appropriate for Rust
2. **Rust Idioms**: Leverage Rust's type system, ownership model, and performance features
3. **Type Safety**: Use Rust's type system to prevent common numerical errors
4. **Performance**: Prioritize computational efficiency without sacrificing safety
5. **Zero-cost Abstractions**: Ensure high-level interfaces don't compromise performance

## Data Representation

SciRS2 primarily uses `ndarray` for multi-dimensional array operations:

- N-dimensional arrays with static or dynamic dimensionality
- Broadcasting capabilities similar to NumPy
- Efficient indexing and slicing
- Integration with BLAS/LAPACK via optional features

Additionally, it uses:
- `nalgebra` for specialized linear algebra operations
- `num-complex` for complex number support
- Custom data structures for specific algorithms

## Implementation Patterns

1. **Generic Programming**:
   - Use traits to define algorithm requirements
   - Support multiple numeric types (f32, f64, complex)
   - Use trait bounds to enforce constraints

2. **Function Design**:
   - Match SciPy's function signatures where appropriate
   - Use builder patterns for complex configurations
   - Provide sensible defaults that match SciPy behavior

3. **Performance Optimizations**:
   - SIMD instructions where applicable
   - Parallelization via Rayon
   - Efficient memory usage patterns
   - FFI bindings to established C/Fortran libraries for critical paths

## Module Responsibilities

### Linear Algebra (scirs2-linalg)

- Basic matrix operations: determinants, inverses, etc.
- Matrix decompositions: LU, QR, SVD, Cholesky
- Eigenvalue/eigenvector computations
- Matrix norms and condition numbers
- Linear equation solvers
- Special matrix functions
- BLAS and LAPACK wrappers

### Integration (scirs2-integrate)

- Numerical integration algorithms
- ODE solvers
- Quadrature methods

### Interpolation (scirs2-interpolate)

- 1D interpolation methods (linear, nearest, cubic)
- Spline interpolation (cubic splines, Akima splines)
- Multi-dimensional interpolation (regular grid and scattered data)
- Advanced interpolation methods:
  - Radial Basis Function (RBF) interpolation
  - Kriging interpolation with uncertainty quantification
  - Barycentric interpolation
- Grid transformation and resampling utilities
- Tensor product interpolation for high-dimensional data
- Utility functions for error estimation, differentiation, and integration

### Optimization (scirs2-optimize)

- Local and global optimization algorithms
- Constrained and unconstrained optimization
- Linear and nonlinear programming

### FFT (scirs2-fft)

- Fast Fourier Transform algorithms
- Real and complex FFT
- Multi-dimensional FFT

### Statistics (scirs2-stats)

- Descriptive statistics
- Probability distributions
- Statistical tests
- Random number generation

### Special Functions (scirs2-special)

- Special mathematical functions
- Bessel functions
- Gamma and beta functions
- Orthogonal polynomials

### Signal Processing (scirs2-signal)

- Filter design and application
- Signal analysis
- Window functions

### Sparse Matrices (scirs2-sparse)

- Various sparse matrix formats
- Sparse matrix operations
- Sparse linear solvers

### Spatial Algorithms (scirs2-spatial)

- Distance computations
- Spatial transformations
- Spatial data structures

### Machine Learning and AI Modules

#### Automatic Differentiation (scirs2-autograd)
- Reverse-mode automatic differentiation
- Tensor-based computation with graph tracking
- Gradient computation and propagation
- Neural network operations:
  - Activation functions
  - Cross-entropy loss functions
  - Convolution operations
  - Pooling operations
- Optimizers for machine learning (SGD, Adam, Momentum SGD, AdaGrad)
- Higher-order derivatives
- BLAS acceleration for linear algebra operations

#### Neural Networks (scirs2-neural)
- Neural network building blocks (layers, activations, loss functions)
- Backpropagation infrastructure
- Model architecture implementations

#### ML Optimization (scirs2-optim)
- Stochastic gradient descent and variants
- Learning rate scheduling
- Regularization techniques

#### Graph Processing (scirs2-graph)
- Graph operations and algorithms
- Support for graph neural networks
- Centrality measures and community detection

#### Data Transformation (scirs2-transform)
- Data normalization and standardization
- Feature engineering utilities
- Dimensionality reduction

#### Metrics and Evaluation (scirs2-metrics)
- Classification metrics (accuracy, F1, ROC)
- Regression metrics (MSE, MAE)
- Model evaluation utilities

#### Text Processing (scirs2-text)
- Tokenization utilities
- Embedding operations
- Text preprocessing

#### Dataset Management (scirs2-datasets)
- Standard dataset loaders and interfaces
- Data splitting and validation utilities
- Batch processing

#### Computer Vision (scirs2-vision)
- Image processing operations
- Feature extraction
- Image transforms and augmentation

#### Time Series Analysis (scirs2-series)
- Time series decomposition
- Forecasting algorithms
- Temporal feature extraction

#### Clustering (scirs2-cluster)
- Vector quantization algorithms
- Hierarchical clustering
- Density-based clustering

#### N-dimensional Image Processing (scirs2-ndimage)
- Filtering operations
- Morphological operations
- Image measurements and analysis

## Testing Strategy

1. **Unit Tests**: Test individual functions and algorithms
2. **Property Tests**: Verify mathematical properties and invariants
3. **Numerical Tests**: Compare against reference implementations
4. **Benchmark Tests**: Monitor performance characteristics
5. **Integration Tests**: Test cross-module functionality

## Documentation Strategy

1. **API Documentation**: Comprehensive docs for all public APIs
2. **Tutorials**: Step-by-step guides for common tasks
3. **Theory**: Mathematical background for implemented algorithms
4. **Examples**: Practical usage examples with code
5. **Performance Notes**: Guidance on algorithm selection

## Development Process

1. **Phase 1**: Implement core infrastructure and linear algebra (scirs2-linalg)
2. **Phase 2**: Develop basic statistical and optimization functionality
3. **Phase 3**: Add integration, interpolation, and FFT modules
4. **Phase 4**: Implement remaining modules based on priority
5. **Ongoing**: Continuous integration, testing, and performance optimization