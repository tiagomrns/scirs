# scirs2-linalg TODO

This module provides linear algebra functionality comparable to NumPy/SciPy's linalg module, serving as a fundamental building block for scientific computing in Rust.

## Current Status

- [x] Set up module structure
- [x] Error handling
- [x] Basic matrix operations (det, inv, solve)
- [x] Matrix decompositions (LU, QR, SVD, etc.)
- [x] Eigenvalue problems (interface)
- [x] BLAS interface
- [x] LAPACK interface
- [x] Core functionality implemented
- [x] Fix all warnings and doctests in the implementation
- [x] Advanced functionality and edge cases
- [x] SciPy-compatible API wrappers (compat module)
- [x] Create comprehensive test suite against SciPy
- [x] Add comprehensive documentation and tutorials

## Known Issues

- Matrix functions' implementation may have numerical stability issues

## Matrix Operations

- [x] Basic operations (add, subtract, multiply, divide)
- [x] Determinant calculation
- [x] Matrix inversion
- [x] Matrix multiplication
- [x] Matrix power
- [x] Matrix exponential
- [x] Matrix norms (Frobenius, nuclear, spectral)
- [x] Condition number calculation
- [x] Matrix rank computation
- [x] Matrix logarithm
- [x] Matrix square root
- [x] Matrix sign function

## Matrix Decompositions

- [x] LU decomposition
- [x] QR decomposition
- [x] SVD (Singular Value Decomposition)
- [x] Cholesky decomposition
- [x] Eigendecomposition
- [x] Schur decomposition
- [x] Polar decomposition
- [x] QZ decomposition
- [x] Complete orthogonal decomposition

## Linear System Solvers

- [x] Direct solvers for general matrices
- [x] Direct solvers for triangular matrices
- [x] Direct solvers for symmetric matrices
- [x] Direct solvers for positive definite matrices
- [x] Least squares solvers
- [x] Conjugate gradient method
- [x] GMRES (Generalized Minimal Residual Method)
- [x] Jacobi method
- [x] Gauss-Seidel method
- [x] Successive over-relaxation (SOR)
- [x] Multigrid methods
- [x] Krylov subspace methods (expanded)

## Specialized Matrix Operations

- [x] Banded matrices
- [x] Symmetric matrices
- [x] Tridiagonal matrices
- [x] Structured matrices (Toeplitz, Hankel, Circulant)
- [x] Block diagonal matrices
- [x] Low-rank approximation
- [x] Block tridiagonal matrices
- [x] Sparse direct solvers

## Tensor Operations

- [x] Basic tensor contraction
- [x] Einstein summation (einsum)
- [x] Batch matrix multiplication
- [x] Higher-Order SVD (HOSVD)
- [x] Mode-n product
- [x] Tensor train decomposition
- [x] Tucker decomposition
- [x] Canonical Polyadic decomposition
- [x] Tensor networks

## AI/ML Support Features

- [x] Batch matrix operations (optimized for mini-batch processing)
- [x] Gradient calculation utilities for neural networks
- [x] Efficient matrix multiplication for large parameter matrices
- [x] Low-rank approximation techniques for dimensionality reduction
- [x] Kronecker product optimization for neural network layers
- [x] Specialized operations for convolutional layers (im2col, etc.)
- [x] Fast random projections for large-scale ML
- [x] Matrix-free operations for iterative solvers in large models
- [x] Tensor contraction operations for deep learning
- [x] Structured matrices support (Toeplitz, circulant) for efficient representations
- [x] Attention mechanism optimizations
  - [x] Standard attention implementations (scaled dot-product, multi-head)
  - [x] Memory-efficient attention (flash attention, linear attention)
  - [x] Position-aware attention variants (RoPE, ALiBi, relative positional)
  - [x] Batched attention operations for high-throughput training
- [x] Quantization-aware linear algebra
  - [x] Basic integer quantization (8-bit, 4-bit)
  - [x] Symmetric and asymmetric quantization
  - [x] Per-channel quantization
  - [x] Quantized matrix-free operations
  - [x] Numerical stability analysis for quantization
  - [x] Fusion operations for quantized matrices
  - [x] Specialized solvers for quantized matrices
- [x] Mixed-precision operations
  - [x] Basic mixed-precision linear algebra (matmul, matvec, solve)
  - [x] Iterative refinement for improved accuracy
  - [x] Mixed-precision QR and SVD
  - [x] Cholesky decomposition with mixed precision
  - [x] Enhanced dot product with Kahan summation for numerical stability
  - [x] SIMD-accelerated mixed-precision operations
  - [x] Mixed-precision least squares solver
  - [x] Mixed-precision matrix inversion
  - [x] Mixed-precision determinant calculation
- [x] Sparse-dense matrix operations

## NumPy/SciPy Compatibility Improvements

- [x] Consistent API with NumPy's linalg
  - [x] Standardize function naming and parameter ordering (via compat module)
  - [x] Ensure equivalent functionality for all NumPy linalg functions (most core functions)
  - [x] Document differences from NumPy where they exist for good reasons
- [x] Type-generic linear algebra operations
  - [x] Unified wrappers for operations on different numeric types
  - [x] Consistent error handling across numeric types
  - [x] Automatic precision selection based on input requirements
- [x] Higher-dimensional array support
  - [x] Convert key operations to handle arrays with multiple batch dimensions
  - [x] Implement broadcasting behavior consistent with NumPy
  - [x] Support for vectorized application of operations to batched arrays

## Optimization Tasks

- [x] Comprehensive tests and benchmarks
  - [x] Test suite that verifies numerical accuracy against SciPy results
  - [ ] Performance benchmarks for all key operations
  - [x] Correctness validation for edge cases
- [x] Performance optimizations for large matrices
  - [x] Cache-friendly algorithms (implemented in perf_opt module)
  - [x] SIMD optimizations
  - [x] Loop tiling and blocking (implemented in SIMD-accelerated matrix multiplication)
  - [x] Memory layout optimizations (blocked and in-place operations)
  - [x] Fusion of consecutive operations for quantized matrices
  - [x] Memory-efficient operations with matrix-free approach
- [x] Improve error messages and handling
  - [x] More detailed error diagnostics for singular matrices
  - [ ] Suggestions for regularization approaches when decompositions fail
  - [ ] Improved numerical stability checks
- [x] Add more examples and documentation
  - [x] Practical tutorials for common scientific and engineering applications
  - [x] Conversion guides for SciPy/NumPy users (via examples and compat module)
  - [ ] Performance optimization guidelines
- [ ] Support for sparse matrices
  - [ ] Integration with scirs2-sparse for all relevant operations
  - [ ] Specialized algorithms for sparse linear algebra
  - [ ] Support for mixed sparse-dense operations
- [ ] Parallel computation support
  - [x] Initial Rayon integration
  - [ ] Algorithm-specific parallel implementations
  - [ ] Work-stealing scheduler optimizations
  - [ ] Thread pool configurations
  - [ ] Standard `workers` parameter across parallelizable functions

## Feature Enhancements

- [x] Autodiff for matrix operations
  - [x] Basic integration with scirs2-autograd
  - [ ] Forward and reverse mode automatic differentiation (limited by scirs2-autograd)
  - [ ] Matrix calculus operations with gradient tracking (pending scirs2-autograd features)
  - [x] Framework for integration with optimization (ready for when features are available)
- [x] Complex number support
  - [x] Complete implementation for all decompositions
  - [x] Specialized algorithms for common complex matrix operations
  - [x] Handling of Hermitian matrices and operations
- [x] Extended precision operations
  - [x] Support for higher precision operations using a type-generic approach
  - [x] Specialized algorithms with extended precision for key operations
  - [x] Error bounds calculations for ill-conditioned matrices
- [x] Random matrix generation
  - [x] Standard distributions (uniform, normal, etc.)
  - [x] Specialized matrices (orthogonal, correlation, etc.)
  - [x] Structured random matrices for testing
- [ ] Matrix calculus utilities
  - [ ] Derivatives of matrix operations
  - [ ] Matrix differential operators
  - [ ] Support for matrix-valued functions
- [ ] Statistical functions on matrices
  - [ ] Matrix-variate distributions
  - [ ] Statistical tests for matrices
  - [ ] Random sampling from matrix distributions
- [ ] Eigenvalue solvers for specific matrix types
  - [ ] Specialized fast algorithms for structured matrices
  - [ ] Sparse eigensolvers (Arnoldi, Lanczos methods)
  - [ ] Partial eigenvalue computation for large matrices

## Integration Tasks

- [ ] Integration with GPU libraries
  - [ ] CUDA support
  - [ ] OpenCL support
  - [ ] Vulkan compute support
  - [ ] ROCm support for AMD GPUs
- [ ] Support for distributed linear algebra
  - [ ] MPI integration
  - [ ] Distributed matrix operations
  - [ ] Collective operations
  - [ ] Scalable algorithms for large clusters
- [ ] Integration with other scientific computing ecosystems
  - [ ] Python interoperability
  - [ ] Julia interoperability
  - [ ] C/C++ interoperability
  - [ ] WebAssembly support
- [ ] Hardware-specific optimizations
  - [ ] AVX/AVX2/AVX-512 optimizations
  - [ ] ARM Neon optimizations
  - [ ] GPU offloading
  - [ ] TPU/IPU support for AI workloads

## Documentation and Examples

- [ ] Comprehensive API documentation
- [ ] Tutorials for common use cases
- [ ] Performance comparison with NumPy/SciPy
- [ ] Jupyter notebook examples
- [ ] Interactive examples
- [ ] Domain-specific guides (engineering, finance, ML, etc.)
- [ ] Algorithm selection guidelines based on problem characteristics

## Long-term Goals

- [ ] Performance comparable to or better than NumPy/SciPy
- [ ] Support for specialized hardware (TPUs, FPGAs)
- [ ] Domain-specific optimizations
- [ ] Seamless integration with AI/ML frameworks
- [ ] Automatic algorithm selection based on problem characteristics
- [ ] Self-tuning performance based on hardware and problem size

## Advanced Matrix Decompositions

- [ ] Generalized eigenvalue decompositions
- [ ] Randomized SVD for large matrices
- [ ] Hierarchical matrix factorizations
- [ ] Kronecker-factored approximate curvature
- [ ] CUR decomposition for feature selection
- [ ] Tensor-Train decomposition for high-dimensional problems
- [ ] Scalable algorithms for tall-and-skinny or short-and-fat matrices

## Special Matrix Types and Operations

- [ ] Sparse factorizations (sparse Cholesky, sparse LU)
- [ ] Circulant and Toeplitz solvers using FFT
- [ ] Preconditioners for iterative methods
- [ ] Fast transforms (DCT, DST, Hadamard)
- [ ] Doubly stochastic matrix approximation
- [ ] Low-rank updates to factorizations
- [ ] Structured matrix approximations
- [ ] Matrix differential equations solvers