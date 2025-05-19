#![recursion_limit = "512"]

//! # SciRS2 Core
//!
//! Core utilities and common functionality for the SciRS2 library.
//!
//! This crate provides shared utilities, error types, and common traits
//! used across the SciRS2 ecosystem of crates.
//!
//! ## Overview
//!
//! * Common error types and traits
//! * High-performance numerical operations
//!   * SIMD-accelerated computations
//!   * Parallel processing for multi-core systems
//!   * Memory-efficient algorithms
//!   * GPU acceleration abstractions
//! * Caching and memoization for optimized performance
//! * Type definitions and conversions
//! * Physical and mathematical constants
//! * Configuration system
//! * Input/output utilities
//! * Validation utilities
//! * Numeric traits and conversions
//! * Memory management utilities
//! * Logging and diagnostics
//! * Profiling tools
//! * Random number generation
//!
//! ## Performance Optimizations
//!
//! The library provides several performance optimization features:
//!
//! * **SIMD Operations**: Uses CPU vector instructions for faster array operations
//! * **Parallel Processing**: Leverages multi-core systems for improved performance
//! * **GPU Acceleration**: Provides abstractions for GPU computation (CUDA, WebGPU, Metal)
//! * **Memory-Efficient Algorithms**: Optimizes memory usage for large-scale computations
//! * **Caching and Memoization**: Avoids redundant computations
//! * **Profiling and Instrumentation**: Identifies performance bottlenecks
//! * **Memory Management**: Efficient memory utilization and pooling
//!
//! ## Additional Utilities
//!
//! * **Logging**: Structured logging for scientific applications
//! * **Random Number Generation**: Consistent interface for random sampling
//! * **Type Conversions**: Safe numeric and complex number conversions
//!
//! ## Feature Flags
//!
//! These features can be controlled via feature flags:
//!
//! * `simd`: Enable SIMD acceleration
//! * `parallel`: Enable parallel processing
//! * `cache`: Enable caching and memoization functionality
//! * `validation`: Enable validation utilities
//! * `logging`: Enable structured logging and diagnostics
//! * `gpu`: Enable GPU acceleration abstractions
//! * `memory_management`: Enable advanced memory management
//! * `memory_efficient`: Enable memory-efficient array operations and views
//! * `array`: Enable scientific array types (MaskedArray, RecordArray)
//! * `profiling`: Enable performance profiling tools
//! * `random`: Enable random number generation utilities
//! * `types`: Enable type conversion utilities
//! * `linalg`: Enable linear algebra with BLAS/LAPACK bindings
//! * `all`: Enable all features except backend-specific ones

// Re-export modules
#[cfg(feature = "array")]
pub mod array;
pub mod array_protocol;
#[cfg(feature = "cache")]
pub mod cache;
pub mod config;
pub mod constants;
pub mod error;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod io;
#[cfg(feature = "logging")]
pub mod logging;
#[cfg(feature = "memory_management")]
pub mod memory;
#[cfg(feature = "memory_efficient")]
pub mod memory_efficient;
pub mod ndarray_ext;
pub mod numeric;
#[cfg(feature = "parallel")]
pub mod parallel;
#[cfg(feature = "profiling")]
pub mod profiling;
#[cfg(feature = "random")]
pub mod random;
#[cfg(feature = "simd")]
pub mod simd;
// Universal Functions (ufuncs) module
#[cfg(feature = "types")]
pub mod types;
#[cfg(feature = "ufuncs")]
pub mod ufuncs;
pub mod utils;
pub mod validation;

// Re-exports
#[cfg(feature = "cache")]
pub use crate::cache::*;
pub use crate::config::*;
pub use crate::constants::*;
pub use crate::error::*;
#[cfg(feature = "gpu")]
pub use crate::gpu::*;
pub use crate::io::*;
#[cfg(feature = "logging")]
pub use crate::logging::*;
#[cfg(feature = "memory_management")]
pub use crate::memory::{
    format_memory_report, generate_memory_report, global_buffer_pool, track_allocation,
    track_deallocation, track_resize, BufferPool, ChunkProcessor, ChunkProcessor2D,
    GlobalBufferPool, ZeroCopyView,
};

#[cfg(feature = "memory_efficient")]
pub use crate::memory_efficient::{
    chunk_wise_binary_op, chunk_wise_op, chunk_wise_reduce, create_disk_array, create_mmap,
    create_temp_mmap, diagonal_view, evaluate, load_chunks, open_mmap, register_fusion,
    transpose_view, view_as, view_mut_as, AccessMode, AdaptiveChunking, AdaptiveChunkingBuilder,
    AdaptiveChunkingParams, AdaptiveChunkingResult, ArithmeticOps, ArrayView, BroadcastOps,
    ChunkIter, ChunkedArray, ChunkingStrategy, CompressedMemMapBuilder, CompressedMemMappedArray,
    CompressionAlgorithm, DiskBackedArray, FusedOp, LazyArray, LazyOp, LazyOpKind,
    MemoryMappedArray, MemoryMappedChunkIter, MemoryMappedChunks, MemoryMappedSlice,
    MemoryMappedSlicing, OpFusion, OutOfCoreArray, ViewMut, ZeroCopyOps,
};

// Re-export the parallel memory-mapped array capabilities
#[cfg(all(feature = "memory_efficient", feature = "parallel"))]
pub use crate::memory_efficient::MemoryMappedChunksParallel;

#[cfg(feature = "array")]
pub use crate::array::{
    is_masked, mask_array, masked_equal, masked_greater, masked_inside, masked_invalid,
    masked_less, masked_outside, masked_where, record_array_from_arrays, record_array_from_records,
    record_array_from_typed_arrays, ArrayError, FieldValue, MaskedArray, Record, RecordArray,
    NOMASK,
};

#[cfg(feature = "memory_metrics")]
pub use crate::memory::metrics::{
    clear_snapshots,
    compare_snapshots,
    // Utility functions
    format_bytes,
    format_duration,
    load_snapshots,
    save_snapshots,
    take_snapshot,
    MemoryEvent,
    MemoryEventType,
    // Core metrics types
    MemoryMetricsCollector,
    MemoryMetricsConfig,
    // Memory snapshots and leak detection
    MemorySnapshot,
    SnapshotDiff,
    // Tracked memory components
    TrackedBufferPool,
    TrackedChunkProcessor,
    TrackedChunkProcessor2D,
};

#[cfg(all(feature = "memory_metrics", feature = "gpu"))]
pub use crate::memory::metrics::{setup_gpu_memory_tracking, TrackedGpuBuffer, TrackedGpuContext};
pub use crate::numeric::*;
#[cfg(feature = "parallel")]
pub use crate::parallel::*;
#[cfg(feature = "profiling")]
pub use crate::profiling::{profiling_memory_tracker, Profiler, Timer};
#[cfg(feature = "random")]
pub use crate::random::*;
#[cfg(feature = "simd")]
pub use crate::simd::*;
#[cfg(feature = "types")]
pub use crate::types::{convert, ComplexConversionError, ComplexExt, ComplexOps};
pub use crate::utils::*;
pub use crate::validation::*;

/// SciRS2 core version information
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
