//! Performance optimization utilities for critical paths
//!
//! This module provides tools and utilities for optimizing performance-critical
//! sections of scirs2-core based on profiling data.

use std::hint;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Performance hints for critical code paths
pub struct PerformanceHints;

impl PerformanceHints {
    /// Hint that a branch is likely to be taken
    #[inline(always)]
    pub fn likely(cond: bool) -> bool {
        // Note: std::intrinsics::likely/unlikely are not stable yet
        // For now, just return the condition as-is
        cond
    }

    /// Hint that a branch is unlikely to be taken
    #[inline(always)]
    pub fn unlikely(cond: bool) -> bool {
        // Note: std::intrinsics::unlikely is not stable yet
        // For now, just return the condition as-is
        cond
    }

    /// Prefetch data for read access
    #[inline(always)]
    pub fn prefetch_read<T>(data: &T) {
        hint::black_box(data);
        // Prefetch instructions are architecture-specific
        // For now, just use black_box to prevent optimization
    }

    /// Prefetch data for write access
    #[inline(always)]
    pub fn prefetch_write<T>(data: &mut T) {
        hint::black_box(data);
        // Prefetch instructions are architecture-specific
        // For now, just use black_box to prevent optimization
    }
}

/// Adaptive optimization based on runtime characteristics
pub struct AdaptiveOptimizer {
    /// Threshold for switching to parallel execution
    parallel_threshold: AtomicUsize,
    /// Threshold for using SIMD operations
    simd_threshold: AtomicUsize,
    /// Cache line size for the current architecture
    cache_line_size: usize,
}

impl AdaptiveOptimizer {
    /// Create a new adaptive optimizer
    pub fn new() -> Self {
        Self {
            parallel_threshold: AtomicUsize::new(10_000),
            simd_threshold: AtomicUsize::new(1_000),
            cache_line_size: Self::detect_cache_line_size(),
        }
    }

    /// Detect the cache line size for the current architecture
    fn detect_cache_line_size() -> usize {
        #[cfg(target_arch = "x86_64")]
        {
            64 // Common for x86_64
        }
        #[cfg(target_arch = "aarch64")]
        {
            128 // Common for ARM64
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            64 // Default fallback
        }
    }

    /// Check if parallel execution should be used for given size
    #[inline]
    #[allow(unused_variables)]
    pub fn should_use_parallel(&self, size: usize) -> bool {
        #[cfg(feature = "parallel")]
        {
            size >= self.parallel_threshold.load(Ordering::Relaxed)
        }
        #[cfg(not(feature = "parallel"))]
        {
            false
        }
    }

    /// Check if SIMD should be used for given size
    #[inline]
    #[allow(unused_variables)]
    pub fn should_use_simd(&self, size: usize) -> bool {
        #[cfg(feature = "simd")]
        {
            size >= self.simd_threshold.load(Ordering::Relaxed)
        }
        #[cfg(not(feature = "simd"))]
        {
            false
        }
    }

    /// Update thresholds based on performance measurements
    pub fn update_thresholds(&self, operation: &str, size: usize, duration_ns: u64) {
        // Simple heuristic: adjust thresholds based on operation efficiency
        let ops_per_ns = size as f64 / duration_ns as f64;

        if operation.contains("parallel") && ops_per_ns < 0.1 {
            // Parallel overhead too high, increase threshold
            self.parallel_threshold
                .fetch_add(size / 10, Ordering::Relaxed);
        } else if operation.contains("simd") && ops_per_ns < 1.0 {
            // SIMD not efficient enough, increase threshold
            self.simd_threshold.fetch_add(size / 10, Ordering::Relaxed);
        }
    }

    /// Get optimal chunk size for cache-friendly operations
    #[inline]
    pub fn optimal_chunk_size<T>(&self) -> usize {
        // Calculate chunk size based on cache line size and element size
        let element_size = std::mem::size_of::<T>();
        let elements_per_cache_line = self.cache_line_size / element_size.max(1);

        // Use multiple cache lines for better performance
        elements_per_cache_line * 16
    }
}

impl Default for AdaptiveOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Fast path optimizations for common operations
pub mod fast_paths {
    use super::*;

    /// Optimized array addition for f64
    #[inline]
    #[allow(unused_variables)]
    pub fn add_f64_arrays(a: &[f64], b: &[f64], result: &mut [f64]) -> Result<(), &'static str> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err("Array lengths must match");
        }

        let len = a.len();
        let optimizer = AdaptiveOptimizer::new();

        #[cfg(feature = "simd")]
        if optimizer.should_use_simd(len) {
            // For now, just use scalar implementation
            // TODO: Use SIMD operations when available
            for i in 0..len {
                result[i] = a[i] + b[i];
            }
            return Ok(());
        }

        #[cfg(feature = "parallel")]
        if optimizer.should_use_parallel(len) {
            use rayon::prelude::*;
            result
                .par_chunks_mut(optimizer.optimal_chunk_size::<f64>())
                .zip(a.par_chunks(optimizer.optimal_chunk_size::<f64>()))
                .zip(b.par_chunks(optimizer.optimal_chunk_size::<f64>()))
                .for_each(|((r_chunk, a_chunk), b_chunk)| {
                    for i in 0..r_chunk.len() {
                        r_chunk[i] = a_chunk[i] + b_chunk[i];
                    }
                });
            return Ok(());
        }

        // Scalar fallback with loop unrolling
        let chunks = len / 8;

        for i in 0..chunks {
            let idx = i * 8;
            result[idx] = a[idx] + b[idx];
            result[idx + 1] = a[idx + 1] + b[idx + 1];
            result[idx + 2] = a[idx + 2] + b[idx + 2];
            result[idx + 3] = a[idx + 3] + b[idx + 3];
            result[idx + 4] = a[idx + 4] + b[idx + 4];
            result[idx + 5] = a[idx + 5] + b[idx + 5];
            result[idx + 6] = a[idx + 6] + b[idx + 6];
            result[idx + 7] = a[idx + 7] + b[idx + 7];
        }

        for i in (chunks * 8)..len {
            result[i] = a[i] + b[i];
        }

        Ok(())
    }

    /// Optimized matrix multiplication kernel
    #[inline]
    pub fn matmul_kernel(
        a: &[f64],
        b: &[f64],
        c: &mut [f64],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(), &'static str> {
        if a.len() != m * k || b.len() != k * n || c.len() != m * n {
            return Err("Invalid matrix dimensions");
        }

        // Tile sizes for cache optimization
        const TILE_M: usize = 64;
        const TILE_N: usize = 64;
        const TILE_K: usize = 64;

        // Clear result matrix
        c.fill(0.0);

        // TODO: Fix parallel implementation to properly handle mutable borrowing
        // #[cfg(feature = "parallel")]
        // if optimizer.should_use_parallel(m * n) {
        //     ...
        // }

        // Serial tiled implementation
        for i0 in (0..m).step_by(TILE_M) {
            for j0 in (0..n).step_by(TILE_N) {
                for k0 in (0..k).step_by(TILE_K) {
                    let i_max = (i0 + TILE_M).min(m);
                    let j_max = (j0 + TILE_N).min(n);
                    let k_max = (k0 + TILE_K).min(k);

                    for i in i0..i_max {
                        for j in j0..j_max {
                            let mut sum = c[i * n + j];
                            for k_idx in k0..k_max {
                                sum += a[i * k + k_idx] * b[k_idx * n + j];
                            }
                            c[i * n + j] = sum;
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Memory access pattern optimizer
pub struct MemoryAccessOptimizer {
    /// Stride detection for array access
    _stride_detector: StrideDetector,
}

#[derive(Default)]
struct StrideDetector {
    _last_address: Option<usize>,
    _detected_stride: Option<isize>,
    _confidence: f32,
}

impl MemoryAccessOptimizer {
    pub fn new() -> Self {
        Self {
            _stride_detector: StrideDetector::default(),
        }
    }

    /// Analyze memory access pattern and suggest optimizations
    pub fn analyze_access_pattern<T>(&mut self, addresses: &[*const T]) -> AccessPattern {
        if addresses.is_empty() {
            return AccessPattern::Unknown;
        }

        // Simple stride detection
        let mut strides = Vec::new();
        for window in addresses.windows(2) {
            let stride = (window[1] as isize) - (window[0] as isize);
            strides.push(stride / std::mem::size_of::<T>() as isize);
        }

        // Check if all strides are equal (sequential access)
        if strides.windows(2).all(|w| w[0] == w[1]) {
            match strides[0] {
                1 => AccessPattern::Sequential,
                -1 => AccessPattern::ReverseSequential,
                s if s > 1 => AccessPattern::Strided(s as usize),
                _ => AccessPattern::Random,
            }
        } else {
            AccessPattern::Random
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AccessPattern {
    Sequential,
    ReverseSequential,
    Strided(usize),
    Random,
    Unknown,
}

impl Default for MemoryAccessOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_optimizer() {
        let optimizer = AdaptiveOptimizer::new();

        // Test threshold detection
        assert!(!optimizer.should_use_parallel(100));

        // Only test parallel execution if the feature is enabled
        #[cfg(feature = "parallel")]
        assert!(optimizer.should_use_parallel(100_000));

        // Test chunk size calculation
        let chunk_size = optimizer.optimal_chunk_size::<f64>();
        assert!(chunk_size > 0);
        assert_eq!(chunk_size % 16, 0); // Should be multiple of 16
    }

    #[test]
    fn test_fast_path_addition() {
        let a = vec![1.0; 1000];
        let b = vec![2.0; 1000];
        let mut result = vec![0.0; 1000];

        fast_paths::add_f64_arrays(&a, &b, &mut result).unwrap();

        for val in result {
            assert_eq!(val, 3.0);
        }
    }

    #[test]
    fn test_memory_access_pattern() {
        let mut optimizer = MemoryAccessOptimizer::new();

        // Sequential access
        let addresses: Vec<*const f64> = (0..10)
            .map(|i| (i * std::mem::size_of::<f64>()) as *const f64)
            .collect();
        assert_eq!(
            optimizer.analyze_access_pattern(&addresses),
            AccessPattern::Sequential
        );

        // Strided access
        let addresses: Vec<*const f64> = (0..10)
            .map(|i| (i * 3 * std::mem::size_of::<f64>()) as *const f64)
            .collect();
        assert_eq!(
            optimizer.analyze_access_pattern(&addresses),
            AccessPattern::Strided(3)
        );
    }
}
