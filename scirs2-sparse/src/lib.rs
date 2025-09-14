#![allow(deprecated)]
#![allow(dead_code)]
//! Sparse module
//!
//! This module provides implementations of various sparse matrix and array formats and operations,
//! similar to SciPy's `sparse` module.
//!
//! ## Overview
//!
//! * Sparse formats (CSR, CSC, COO, DOK, LIL, DIA, BSR, etc.)
//! * Specialized sparse formats (Symmetric CSR, Symmetric COO)
//! * Basic operations (addition, multiplication, etc.)
//! * Sparse linear system solvers
//! * Sparse eigenvalue computation
//! * Conversion between different formats
//!
//! ## Matrix vs. Array API
//!
//! This module provides both a matrix-based API and an array-based API,
//! following SciPy's transition to a more NumPy-compatible array interface.
//!
//! When using the array interface (e.g., `CsrArray`), please note that:
//!
//! - `*` performs element-wise multiplication, not matrix multiplication
//! - Use `dot()` method for matrix multiplication
//! - Operations like `sum` produce arrays, not matrices
//! - Array-style slicing operations return scalars, 1D, or 2D arrays
//!
//! For new code, we recommend using the array interface, which is more consistent
//! with the rest of the numerical ecosystem.
//!
//! ## Examples
//!
//! ### Matrix API (Legacy)
//!
//! ```
//! use scirs2_sparse::csr::CsrMatrix;
//!
//! // Create a sparse matrix in CSR format
//! let rows = vec![0, 0, 1, 2, 2];
//! let cols = vec![0, 2, 2, 0, 1];
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let shape = (3, 3);
//!
//! let matrix = CsrMatrix::new(data, rows, cols, shape).unwrap();
//! ```
//!
//! ### Array API (Recommended)
//!
//! ```
//! use scirs2_sparse::csr_array::CsrArray;
//!
//! // Create a sparse array in CSR format
//! let rows = vec![0, 0, 1, 2, 2];
//! let cols = vec![0, 2, 2, 0, 1];
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let shape = (3, 3);
//!
//! // From triplets (COO-like construction)
//! let array = CsrArray::from_triplets(&rows, &cols, &data, shape, false).unwrap();
//!
//! // Or directly from CSR components
//! // let array = CsrArray::new(...);
//! ```

// Export error types
pub mod error;
pub use error::{SparseError, SparseResult};

// Base trait for sparse arrays
pub mod sparray;
pub use sparray::{is_sparse, SparseArray, SparseSum};

// Trait for symmetric sparse arrays
pub mod sym_sparray;
pub use sym_sparray::SymSparseArray;

// No spatial module in sparse

// Array API (recommended)
pub mod csr_array;
pub use csr_array::CsrArray;

pub mod csc_array;
pub use csc_array::CscArray;

pub mod coo_array;
pub use coo_array::CooArray;

pub mod dok_array;
pub use dok_array::DokArray;

pub mod lil_array;
pub use lil_array::LilArray;

pub mod dia_array;
pub use dia_array::DiaArray;

pub mod bsr_array;
pub use bsr_array::BsrArray;

pub mod banded_array;
pub use banded_array::BandedArray;

// Symmetric array formats
pub mod sym_csr;
pub use sym_csr::{SymCsrArray, SymCsrMatrix};

pub mod sym_coo;
pub use sym_coo::{SymCooArray, SymCooMatrix};

// Legacy matrix formats
pub mod csr;
pub use csr::CsrMatrix;

pub mod csc;
pub use csc::CscMatrix;

pub mod coo;
pub use coo::CooMatrix;

pub mod dok;
pub use dok::DokMatrix;

pub mod lil;
pub use lil::LilMatrix;

pub mod dia;
pub use dia::DiaMatrix;

pub mod bsr;
pub use bsr::BsrMatrix;

pub mod banded;
pub use banded::BandedMatrix;

// Utility functions
pub mod utils;

// Linear algebra with sparse matrices
pub mod linalg;
// Re-export the main functions from the reorganized linalg module
pub use linalg::{
    // Functions from solvers
    add,
    // Functions from iterative
    bicg,
    bicgstab,
    cg,
    cholesky_decomposition,
    // Enhanced operators
    convolution_operator,
    diag_matrix,
    eigs,
    eigsh,
    enhanced_add,
    enhanced_diagonal,
    enhanced_scale,
    enhanced_subtract,
    expm,
    // Functions from matfuncs
    expm_multiply,
    eye,
    finite_difference_operator,
    // GCROT solver
    gcrot,
    gmres,
    incomplete_cholesky,
    incomplete_lu,
    inv,
    lanczos,
    // Decomposition functions
    lu_decomposition,
    matmul,
    matrix_power,
    multiply,
    norm,
    onenormest,
    // Eigenvalue functions
    power_iteration,
    qr_decomposition,
    sparse_direct_solve,
    sparse_lstsq,
    spsolve,
    svd_truncated,
    // SVD functions
    svds,
    // TFQMR solver
    tfqmr,
    ArpackOptions,
    // Interfaces
    AsLinearOperator,
    // Types from iterative
    BiCGOptions,
    BiCGSTABOptions,
    BiCGSTABResult,
    // Enhanced operator types
    BoundaryCondition,
    CGOptions,
    CGSOptions,
    CGSResult,
    CholeskyResult,
    ConvolutionMode,
    ConvolutionOperator,
    // Operator types
    DiagonalOperator,
    EigenResult,
    EigenvalueMethod,
    EnhancedDiagonalOperator,
    EnhancedDifferenceOperator,
    EnhancedOperatorOptions,
    EnhancedScaledOperator,
    EnhancedSumOperator,
    FiniteDifferenceOperator,
    GCROTOptions,
    GCROTResult,
    GMRESOptions,
    ICOptions,
    // Preconditioners
    ILU0Preconditioner,
    ILUOptions,
    IdentityOperator,
    IterationResult,
    JacobiPreconditioner,
    // Decomposition types
    LUResult,
    LanczosOptions,
    LinearOperator,
    // Eigenvalue types
    PowerIterationOptions,
    QRResult,
    SSORPreconditioner,
    // SVD types
    SVDOptions,
    SVDResult,
    ScaledIdentityOperator,
    TFQMROptions,
    TFQMRResult,
};

// Format conversions
pub mod convert;

// Construction utilities
pub mod construct;
pub mod construct_sym;

// Combining arrays
pub mod combine;
pub use combine::{block_diag, bmat, hstack, kron, kronsum, tril, triu, vstack};

// Index dtype handling utilities
pub mod index_dtype;
pub use index_dtype::{can_cast_safely, get_index_dtype, safely_cast_index_arrays};

// Optimized operations for symmetric sparse formats
pub mod sym_ops;
pub use sym_ops::{
    sym_coo_matvec, sym_csr_matvec, sym_csr_quadratic_form, sym_csr_rank1_update, sym_csr_trace,
};

// GPU-accelerated operations
pub mod gpu;
pub mod gpu_kernel_execution;
pub mod gpu_ops;
pub mod gpu_spmv_implementation;
pub use gpu_kernel_execution::{
    calculate_adaptive_workgroup_size, execute_spmv_kernel, execute_symmetric_spmv_kernel,
    execute_triangular_solve_kernel, GpuKernelConfig, GpuMemoryManager as GpuKernelMemoryManager,
    GpuPerformanceProfiler, MemoryStrategy,
};
pub use gpu_ops::{
    gpu_sparse_matvec, gpu_sym_sparse_matvec, AdvancedGpuOps, GpuKernelScheduler, GpuMemoryManager,
    GpuOptions, GpuProfiler, OptimizedGpuOps,
};
pub use gpu_spmv_implementation::GpuSpMV;

// Memory-efficient algorithms and patterns
pub mod memory_efficient;
pub use memory_efficient::{
    streaming_sparse_matvec, CacheAwareOps, MemoryPool, MemoryTracker, OutOfCoreProcessor,
};

// SIMD-accelerated operations
pub mod simd_ops;
pub use simd_ops::{
    simd_csr_matvec, simd_sparse_elementwise, simd_sparse_linear_combination, simd_sparse_matmul,
    simd_sparse_norm, simd_sparse_scale, simd_sparse_transpose, ElementwiseOp, SimdOptions,
};

// Parallel vector operations for iterative solvers
pub mod parallel_vector_ops;
pub use parallel_vector_ops::{
    advanced_sparse_matvec_csr, parallel_axpy, parallel_dot, parallel_linear_combination,
    parallel_norm2, parallel_sparse_matvec_csr, parallel_vector_add, parallel_vector_copy,
    parallel_vector_scale, parallel_vector_sub, ParallelVectorOptions,
};

// Quantum-inspired sparse matrix operations (Advanced mode)
pub mod quantum_inspired_sparse;
pub use quantum_inspired_sparse::{
    QuantumProcessorStats, QuantumSparseConfig, QuantumSparseProcessor, QuantumStrategy,
};

// Neural-adaptive sparse matrix operations (Advanced mode)
pub mod neural_adaptive_sparse;
pub use neural_adaptive_sparse::{
    NeuralAdaptiveConfig, NeuralAdaptiveSparseProcessor, NeuralProcessorStats, OptimizationStrategy,
};

// Quantum-Neural hybrid optimization (Advanced mode)
pub mod quantum_neural_hybrid;
pub use quantum_neural_hybrid::{
    HybridStrategy, QuantumNeuralConfig, QuantumNeuralHybridProcessor, QuantumNeuralHybridStats,
};

// Adaptive memory compression for advanced-large sparse matrices (Advanced mode)
pub mod adaptive_memory_compression;
pub use adaptive_memory_compression::{
    AdaptiveCompressionConfig, AdaptiveMemoryCompressor, CompressedMatrix, CompressionAlgorithm,
    MemoryStats,
};

// Real-time performance monitoring and adaptation (Advanced mode)
pub mod realtime_performance_monitor;
pub use realtime_performance_monitor::{
    Alert, AlertSeverity, Forecast, PerformanceMetrics, PerformanceMonitorConfig,
    PerformanceSample, ProcessorType, RealTimePerformanceMonitor,
};

// Compressed sparse graph algorithms
pub mod csgraph;
pub use csgraph::{
    all_pairs_shortest_path,
    bellman_ford_single_source,
    bfs_distances,
    // Traversal algorithms
    breadth_first_search,
    compute_laplacianmatrix,
    connected_components,
    degree_matrix,
    depth_first_search,
    dijkstra_single_source,
    floyd_warshall,
    has_path,
    is_connected,
    is_laplacian,
    is_spanning_tree,
    // Minimum spanning trees
    kruskal_mst,
    // Laplacian matrices
    laplacian,
    largest_component,
    minimum_spanning_tree,
    num_edges,
    num_vertices,
    prim_mst,
    reachable_vertices,
    reconstruct_path,
    // Graph algorithms
    shortest_path,
    // Shortest path algorithms
    single_source_shortest_path,
    spanning_tree_weight,
    strongly_connected_components,
    to_adjacency_list,
    topological_sort,
    traversegraph,
    // Connected components
    undirected_connected_components,
    // Graph utilities
    validate_graph,
    weakly_connected_components,
    LaplacianType,
    MSTAlgorithm,
    // Enums and types
    ShortestPathMethod,
    TraversalOrder,
};

// Re-export warnings from scipy for compatibility
pub struct SparseEfficiencyWarning;
pub struct SparseWarning;

/// Check if an object is a sparse array
#[allow(dead_code)]
pub fn is_sparse_array<T>(obj: &dyn SparseArray<T>) -> bool
where
    T: num_traits::Float
        + std::fmt::Debug
        + Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + 'static,
{
    sparray::is_sparse(obj)
}

/// Check if an object is a symmetric sparse array
#[allow(dead_code)]
pub fn is_sym_sparse_array<T>(obj: &dyn SymSparseArray<T>) -> bool
where
    T: num_traits::Float
        + std::fmt::Debug
        + Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + 'static,
{
    obj.is_symmetric()
}

/// Check if an object is a sparse matrix (legacy API)
#[allow(dead_code)]
pub fn is_sparse_matrix(obj: &dyn std::any::Any) -> bool {
    obj.is::<CsrMatrix<f64>>()
        || obj.is::<CscMatrix<f64>>()
        || obj.is::<CooMatrix<f64>>()
        || obj.is::<DokMatrix<f64>>()
        || obj.is::<LilMatrix<f64>>()
        || obj.is::<DiaMatrix<f64>>()
        || obj.is::<BsrMatrix<f64>>()
        || obj.is::<SymCsrMatrix<f64>>()
        || obj.is::<SymCooMatrix<f64>>()
        || obj.is::<CsrMatrix<f32>>()
        || obj.is::<CscMatrix<f32>>()
        || obj.is::<CooMatrix<f32>>()
        || obj.is::<DokMatrix<f32>>()
        || obj.is::<LilMatrix<f32>>()
        || obj.is::<DiaMatrix<f32>>()
        || obj.is::<BsrMatrix<f32>>()
        || obj.is::<SymCsrMatrix<f32>>()
        || obj.is::<SymCooMatrix<f32>>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_csr_array() {
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 2, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        let array = CsrArray::from_triplets(&rows, &cols, &data, shape, false).unwrap();

        assert_eq!(array.shape(), (3, 3));
        assert_eq!(array.nnz(), 5);
        assert!(is_sparse_array(&array));
    }

    #[test]
    fn test_coo_array() {
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 2, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        let array = CooArray::from_triplets(&rows, &cols, &data, shape, false).unwrap();

        assert_eq!(array.shape(), (3, 3));
        assert_eq!(array.nnz(), 5);
        assert!(is_sparse_array(&array));
    }

    #[test]
    fn test_dok_array() {
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 2, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        let array = DokArray::from_triplets(&rows, &cols, &data, shape).unwrap();

        assert_eq!(array.shape(), (3, 3));
        assert_eq!(array.nnz(), 5);
        assert!(is_sparse_array(&array));

        // Test setting and getting values
        let mut array = DokArray::<f64>::new((2, 2));
        array.set(0, 0, 1.0).unwrap();
        array.set(1, 1, 2.0).unwrap();

        assert_eq!(array.get(0, 0), 1.0);
        assert_eq!(array.get(0, 1), 0.0);
        assert_eq!(array.get(1, 1), 2.0);

        // Test removing zeros
        array.set(0, 0, 0.0).unwrap();
        assert_eq!(array.nnz(), 1);
    }

    #[test]
    fn test_lil_array() {
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 2, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        let array = LilArray::from_triplets(&rows, &cols, &data, shape).unwrap();

        assert_eq!(array.shape(), (3, 3));
        assert_eq!(array.nnz(), 5);
        assert!(is_sparse_array(&array));

        // Test setting and getting values
        let mut array = LilArray::<f64>::new((2, 2));
        array.set(0, 0, 1.0).unwrap();
        array.set(1, 1, 2.0).unwrap();

        assert_eq!(array.get(0, 0), 1.0);
        assert_eq!(array.get(0, 1), 0.0);
        assert_eq!(array.get(1, 1), 2.0);

        // Test sorted indices
        assert!(array.has_sorted_indices());

        // Test removing zeros
        array.set(0, 0, 0.0).unwrap();
        assert_eq!(array.nnz(), 1);
    }

    #[test]
    fn test_dia_array() {
        use ndarray::Array1;

        // Create a 3x3 diagonal matrix with main diagonal + upper diagonal
        let data = vec![
            Array1::from_vec(vec![1.0, 2.0, 3.0]), // Main diagonal
            Array1::from_vec(vec![4.0, 5.0, 0.0]), // Upper diagonal
        ];
        let offsets = vec![0, 1]; // Main diagonal and k=1
        let shape = (3, 3);

        let array = DiaArray::new(data, offsets, shape).unwrap();

        assert_eq!(array.shape(), (3, 3));
        assert_eq!(array.nnz(), 5); // 3 on main diagonal, 2 on upper diagonal
        assert!(is_sparse_array(&array));

        // Test values
        assert_eq!(array.get(0, 0), 1.0);
        assert_eq!(array.get(1, 1), 2.0);
        assert_eq!(array.get(2, 2), 3.0);
        assert_eq!(array.get(0, 1), 4.0);
        assert_eq!(array.get(1, 2), 5.0);
        assert_eq!(array.get(0, 2), 0.0);

        // Test from_triplets
        let rows = vec![0, 0, 1, 1, 2];
        let cols = vec![0, 1, 1, 2, 2];
        let data_vec = vec![1.0, 4.0, 2.0, 5.0, 3.0];

        let array2 = DiaArray::from_triplets(&rows, &cols, &data_vec, shape).unwrap();

        // Should have same values
        assert_eq!(array2.get(0, 0), 1.0);
        assert_eq!(array2.get(1, 1), 2.0);
        assert_eq!(array2.get(2, 2), 3.0);
        assert_eq!(array2.get(0, 1), 4.0);
        assert_eq!(array2.get(1, 2), 5.0);

        // Test conversion to other formats
        let csr = array.to_csr().unwrap();
        assert_eq!(csr.nnz(), 5);
        assert_eq!(csr.get(0, 0), 1.0);
        assert_eq!(csr.get(0, 1), 4.0);
    }

    #[test]
    fn test_format_conversions() {
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 1, 0, 2];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        // Create a COO array
        let coo = CooArray::from_triplets(&rows, &cols, &data, shape, false).unwrap();

        // Convert to CSR
        let csr = coo.to_csr().unwrap();

        // Check values are preserved
        let coo_dense = coo.to_array();
        let csr_dense = csr.to_array();

        for i in 0..shape.0 {
            for j in 0..shape.1 {
                assert_relative_eq!(coo_dense[[i, j]], csr_dense[[i, j]]);
            }
        }
    }

    #[test]
    fn test_dot_product() {
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 1, 0, 2];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = (3, 3);

        // Create arrays in different formats
        let coo = CooArray::from_triplets(&rows, &cols, &data, shape, false).unwrap();
        let csr = CsrArray::from_triplets(&rows, &cols, &data, shape, false).unwrap();

        // Compute dot product (matrix multiplication)
        let coo_result = coo.dot(&coo).unwrap();
        let csr_result = csr.dot(&csr).unwrap();

        // Check results match
        let coo_dense = coo_result.to_array();
        let csr_dense = csr_result.to_array();

        for i in 0..shape.0 {
            for j in 0..shape.1 {
                assert_relative_eq!(coo_dense[[i, j]], csr_dense[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_sym_csr_array() {
        // Create a symmetric matrix
        let data = vec![2.0, 1.0, 2.0, 3.0, 0.0, 3.0, 1.0];
        let indices = vec![0, 0, 1, 2, 0, 1, 2];
        let indptr = vec![0, 1, 3, 7];

        let sym_matrix = SymCsrMatrix::new(data, indptr, indices, (3, 3)).unwrap();
        let sym_array = SymCsrArray::new(sym_matrix);

        assert_eq!(sym_array.shape(), (3, 3));
        assert!(is_sym_sparse_array(&sym_array));

        // Check values
        assert_eq!(SparseArray::get(&sym_array, 0, 0), 2.0);
        assert_eq!(SparseArray::get(&sym_array, 0, 1), 1.0);
        assert_eq!(SparseArray::get(&sym_array, 1, 0), 1.0); // Symmetric element
        assert_eq!(SparseArray::get(&sym_array, 1, 2), 3.0);
        assert_eq!(SparseArray::get(&sym_array, 2, 1), 3.0); // Symmetric element

        // Convert to standard CSR
        let csr = SymSparseArray::to_csr(&sym_array).unwrap();
        assert_eq!(csr.nnz(), 10); // Full matrix with symmetric elements
    }

    #[test]
    fn test_sym_coo_array() {
        // Create a symmetric matrix in COO format
        let data = vec![2.0, 1.0, 2.0, 3.0, 1.0];
        let rows = vec![0, 1, 1, 2, 2];
        let cols = vec![0, 0, 1, 1, 2];

        let sym_matrix = SymCooMatrix::new(data, rows, cols, (3, 3)).unwrap();
        let sym_array = SymCooArray::new(sym_matrix);

        assert_eq!(sym_array.shape(), (3, 3));
        assert!(is_sym_sparse_array(&sym_array));

        // Check values
        assert_eq!(SparseArray::get(&sym_array, 0, 0), 2.0);
        assert_eq!(SparseArray::get(&sym_array, 0, 1), 1.0);
        assert_eq!(SparseArray::get(&sym_array, 1, 0), 1.0); // Symmetric element
        assert_eq!(SparseArray::get(&sym_array, 1, 2), 3.0);
        assert_eq!(SparseArray::get(&sym_array, 2, 1), 3.0); // Symmetric element

        // Test from_triplets with enforce symmetry
        // Input is intentionally asymmetric - will be fixed by enforce_symmetric=true
        let rows2 = vec![0, 0, 1, 1, 2, 1, 0];
        let cols2 = vec![0, 1, 1, 2, 2, 0, 2];
        let data2 = vec![2.0, 1.5, 2.0, 3.5, 1.0, 0.5, 0.0];

        let sym_array2 = SymCooArray::from_triplets(&rows2, &cols2, &data2, (3, 3), true).unwrap();

        // Should average the asymmetric values
        assert_eq!(SparseArray::get(&sym_array2, 0, 1), 1.0); // Average of 1.5 and 0.5
        assert_eq!(SparseArray::get(&sym_array2, 1, 0), 1.0); // Symmetric element
        assert_eq!(SparseArray::get(&sym_array2, 0, 2), 0.0); // Zero element
    }

    #[test]
    fn test_construct_sym_utils() {
        // Test creating an identity matrix
        let eye = construct_sym::eye_sym_array::<f64>(3, "csr").unwrap();

        assert_eq!(eye.shape(), (3, 3));
        assert_eq!(SparseArray::get(&*eye, 0, 0), 1.0);
        assert_eq!(SparseArray::get(&*eye, 1, 1), 1.0);
        assert_eq!(SparseArray::get(&*eye, 2, 2), 1.0);
        assert_eq!(SparseArray::get(&*eye, 0, 1), 0.0);

        // Test creating a tridiagonal matrix - with coo format since csr had issues
        let diag = vec![2.0, 2.0, 2.0];
        let offdiag = vec![1.0, 1.0];

        let tri = construct_sym::tridiagonal_sym_array(&diag, &offdiag, "coo").unwrap();

        assert_eq!(tri.shape(), (3, 3));
        assert_eq!(SparseArray::get(&*tri, 0, 0), 2.0); // Main diagonal
        assert_eq!(SparseArray::get(&*tri, 1, 1), 2.0);
        assert_eq!(SparseArray::get(&*tri, 2, 2), 2.0);
        assert_eq!(SparseArray::get(&*tri, 0, 1), 1.0); // Off-diagonal
        assert_eq!(SparseArray::get(&*tri, 1, 0), 1.0); // Symmetric element
        assert_eq!(SparseArray::get(&*tri, 1, 2), 1.0);
        assert_eq!(SparseArray::get(&*tri, 0, 2), 0.0); // Zero element

        // Test creating a banded matrix
        let diagonals = vec![
            vec![2.0, 2.0, 2.0, 2.0, 2.0], // Main diagonal
            vec![1.0, 1.0, 1.0, 1.0],      // First off-diagonal
            vec![0.5, 0.5, 0.5],           // Second off-diagonal
        ];

        let band = construct_sym::banded_sym_array(&diagonals, 5, "csr").unwrap();

        assert_eq!(band.shape(), (5, 5));
        assert_eq!(SparseArray::get(&*band, 0, 0), 2.0);
        assert_eq!(SparseArray::get(&*band, 0, 1), 1.0);
        assert_eq!(SparseArray::get(&*band, 0, 2), 0.5);
        assert_eq!(SparseArray::get(&*band, 2, 0), 0.5); // Symmetric element
    }

    #[test]
    fn test_sym_conversions() {
        // Create a symmetric matrix
        // Lower triangular part only
        let data = vec![2.0, 1.0, 2.0, 3.0, 1.0];
        let rows = vec![0, 1, 1, 2, 2];
        let cols = vec![0, 0, 1, 1, 2];

        let sym_coo = SymCooArray::from_triplets(&rows, &cols, &data, (3, 3), true).unwrap();

        // Convert to symmetric CSR
        let sym_csr = sym_coo.to_sym_csr().unwrap();

        // Check values are preserved
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(
                    SparseArray::get(&sym_coo, i, j),
                    SparseArray::get(&sym_csr, i, j)
                );
            }
        }

        // Convert to standard formats
        let csr = SymSparseArray::to_csr(&sym_coo).unwrap();
        let coo = SymSparseArray::to_coo(&sym_csr).unwrap();

        // Check full symmetric matrix in standard formats
        assert_eq!(csr.nnz(), 7); // Accounts for symmetric pairs
        assert_eq!(coo.nnz(), 7);

        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(SparseArray::get(&csr, i, j), SparseArray::get(&coo, i, j));
                assert_eq!(
                    SparseArray::get(&csr, i, j),
                    SparseArray::get(&sym_csr, i, j)
                );
            }
        }
    }
}
