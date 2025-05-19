//! Adaptive chunking strategies for memory-efficient operations.
//!
//! This module provides algorithms that dynamically determine optimal
//! chunk sizes based on workload characteristics, memory constraints,
//! and data distribution patterns. Adaptive chunking can significantly
//! improve performance by balancing memory usage with processing efficiency.

use super::chunked::ChunkingStrategy;
use super::memmap::MemoryMappedArray;
use super::memmap_chunks::MemoryMappedChunks;
use crate::error::{CoreError, CoreResult};
use ndarray::Dimension;
use std::time::Duration;

/// Parameters for configuring adaptive chunking behavior.
#[derive(Debug, Clone)]
pub struct AdaptiveChunkingParams {
    /// Target memory usage per chunk (in bytes)
    pub target_memory_usage: usize,

    /// Maximum chunk size (in elements)
    pub max_chunk_size: usize,

    /// Minimum chunk size (in elements)
    pub min_chunk_size: usize,

    /// Target processing time per chunk (for time-based adaptation)
    pub target_chunk_duration: Option<Duration>,

    /// Whether to consider data distribution (can be expensive to calculate)
    pub consider_distribution: bool,

    /// Whether to adjust for parallel processing
    pub optimize_for_parallel: bool,

    /// Number of worker threads to optimize for (when parallel is enabled)
    pub num_workers: Option<usize>,
}

impl Default for AdaptiveChunkingParams {
    fn default() -> Self {
        // Default to 64MB target memory usage per chunk
        const DEFAULT_TARGET_MEMORY: usize = 64 * 1024 * 1024;

        Self {
            target_memory_usage: DEFAULT_TARGET_MEMORY,
            max_chunk_size: usize::MAX,
            min_chunk_size: 1024,
            target_chunk_duration: None,
            consider_distribution: false,
            optimize_for_parallel: false,
            num_workers: None,
        }
    }
}

/// Result of adaptive chunking analysis.
#[derive(Debug, Clone)]
pub struct AdaptiveChunkingResult {
    /// Recommended chunking strategy
    pub strategy: ChunkingStrategy,

    /// Estimated memory usage per chunk (in bytes)
    pub estimated_memory_per_chunk: usize,

    /// Factors that influenced the chunking decision
    pub decision_factors: Vec<String>,
}

/// Trait for adaptive chunking capabilities.
pub trait AdaptiveChunking<A: Clone + Copy + 'static> {
    /// Calculate an optimal chunking strategy based on array characteristics.
    ///
    /// # Arguments
    ///
    /// * `params` - Parameters to guide the adaptive chunking process
    ///
    /// # Returns
    ///
    /// A result containing the recommended chunking strategy and metadata
    fn adaptive_chunking(
        &self,
        params: AdaptiveChunkingParams,
    ) -> CoreResult<AdaptiveChunkingResult>;

    /// Process chunks using an automatically determined optimal chunking strategy.
    ///
    /// # Arguments
    ///
    /// * `params` - Parameters to guide the adaptive chunking process
    /// * `f` - Function to process each chunk
    ///
    /// # Returns
    ///
    /// A vector of results, one for each chunk
    fn process_chunks_adaptive<F, R>(
        &self,
        params: AdaptiveChunkingParams,
        f: F,
    ) -> CoreResult<Vec<R>>
    where
        F: Fn(&[A], usize) -> R;

    /// Process chunks mutably using an automatically determined optimal chunking strategy.
    ///
    /// # Arguments
    ///
    /// * `params` - Parameters to guide the adaptive chunking process
    /// * `f` - Function to process each chunk
    fn process_chunks_mut_adaptive<F>(
        &mut self,
        params: AdaptiveChunkingParams,
        f: F,
    ) -> CoreResult<()>
    where
        F: Fn(&mut [A], usize);

    /// Process chunks in parallel using an automatically determined optimal chunking strategy.
    ///
    /// # Arguments
    ///
    /// * `params` - Parameters to guide the adaptive chunking process
    /// * `f` - Function to process each chunk
    ///
    /// # Returns
    ///
    /// A vector of results, one for each chunk
    #[cfg(feature = "parallel")]
    fn process_chunks_parallel_adaptive<F, R>(
        &self,
        params: AdaptiveChunkingParams,
        f: F,
    ) -> CoreResult<Vec<R>>
    where
        F: Fn(&[A], usize) -> R + Send + Sync,
        R: Send,
        A: Send + Sync;
}

impl<A: Clone + Copy + 'static> AdaptiveChunking<A> for MemoryMappedArray<A> {
    fn adaptive_chunking(
        &self,
        params: AdaptiveChunkingParams,
    ) -> CoreResult<AdaptiveChunkingResult> {
        // Get total number of elements in the array
        let total_elements = self.size;

        // Calculate element size
        let element_size = std::mem::size_of::<A>();

        // Calculate initial chunk size based on target memory usage
        let mut chunk_size = params.target_memory_usage / element_size;

        // Apply min/max constraints
        chunk_size = chunk_size.clamp(params.min_chunk_size, params.max_chunk_size);

        // Ensure we don't exceed total elements
        chunk_size = chunk_size.min(total_elements);

        // Consider dimensionality-specific adjustments
        let decision_factors = self.optimize_for_dimensionality(chunk_size, &params)?;

        // Factor in parallel processing if requested
        let (chunk_size, decision_factors) = if params.optimize_for_parallel {
            self.optimize_for_parallel_processing(chunk_size, decision_factors, &params)
        } else {
            (chunk_size, decision_factors)
        };

        // Create final chunking strategy
        let strategy = ChunkingStrategy::Fixed(chunk_size);

        // Calculate estimated memory per chunk
        let estimated_memory = chunk_size * element_size;

        Ok(AdaptiveChunkingResult {
            strategy,
            estimated_memory_per_chunk: estimated_memory,
            decision_factors,
        })
    }

    fn process_chunks_adaptive<F, R>(
        &self,
        params: AdaptiveChunkingParams,
        f: F,
    ) -> CoreResult<Vec<R>>
    where
        F: Fn(&[A], usize) -> R,
    {
        // Determine optimal chunking strategy
        let adaptive_result = self.adaptive_chunking(params)?;

        // Use determined strategy to process chunks - wrap with Ok
        Ok(self.process_chunks(adaptive_result.strategy, f))
    }

    fn process_chunks_mut_adaptive<F>(
        &mut self,
        params: AdaptiveChunkingParams,
        f: F,
    ) -> CoreResult<()>
    where
        F: Fn(&mut [A], usize),
    {
        // Determine optimal chunking strategy
        let adaptive_result = self.adaptive_chunking(params)?;

        // Use determined strategy to process chunks - wrap with Ok
        self.process_chunks_mut(adaptive_result.strategy, f);
        Ok(())
    }

    #[cfg(feature = "parallel")]
    fn process_chunks_parallel_adaptive<F, R>(
        &self,
        params: AdaptiveChunkingParams,
        f: F,
    ) -> CoreResult<Vec<R>>
    where
        F: Fn(&[A], usize) -> R + Send + Sync,
        R: Send,
        A: Send + Sync,
    {
        // Make sure parameters are optimized for parallel processing
        let mut parallel_params = params;
        parallel_params.optimize_for_parallel = true;

        // Set default number of workers if not specified
        if parallel_params.num_workers.is_none() {
            parallel_params.num_workers = Some(rayon::current_num_threads());
        }

        // Determine optimal chunking strategy for parallel processing
        let adaptive_result = self.adaptive_chunking(parallel_params)?;

        // Use determined strategy to process chunks in parallel
        use super::memmap_chunks::MemoryMappedChunksParallel;
        Ok(self.process_chunks_parallel(adaptive_result.strategy, f))
    }
}

impl<A: Clone + Copy + 'static> MemoryMappedArray<A> {
    /// Optimize chunking based on array dimensionality.
    fn optimize_for_dimensionality(
        &self,
        initial_chunk_size: usize,
        params: &AdaptiveChunkingParams,
    ) -> CoreResult<Vec<String>> {
        let mut decision_factors = Vec::new();
        let mut chunk_size = initial_chunk_size;

        match self.shape.len() {
            1 => {
                // For 1D arrays, we can use the initial chunk size directly
                decision_factors.push("1D array: Using direct chunking".to_string());
            }
            2 => {
                // For 2D arrays, try to align with rows when possible
                let row_length = self.shape[1];

                if chunk_size >= row_length && chunk_size % row_length != 0 {
                    // Adjust to a multiple of row length for better cache behavior
                    let new_size = (chunk_size / row_length) * row_length;
                    if new_size >= params.min_chunk_size {
                        chunk_size = new_size;
                        decision_factors.push(format!(
                            "2D array: Adjusted chunk size to {} (multiple of row length {})",
                            chunk_size, row_length
                        ));
                    }
                }
            }
            3 => {
                // For 3D arrays, try to align with planes or rows
                let plane_size = self.shape[1] * self.shape[2];
                let row_length = self.shape[2];

                if chunk_size >= plane_size && chunk_size % plane_size != 0 {
                    // Adjust to a multiple of plane size for better cache behavior
                    let new_size = (chunk_size / plane_size) * plane_size;
                    if new_size >= params.min_chunk_size {
                        chunk_size = new_size;
                        decision_factors.push(format!(
                            "3D array: Adjusted chunk size to {} (multiple of plane size {})",
                            chunk_size, plane_size
                        ));
                    }
                } else if chunk_size >= row_length && chunk_size % row_length != 0 {
                    // Adjust to a multiple of row length
                    let new_size = (chunk_size / row_length) * row_length;
                    if new_size >= params.min_chunk_size {
                        chunk_size = new_size;
                        decision_factors.push(format!(
                            "3D array: Adjusted chunk size to {} (multiple of row length {})",
                            chunk_size, row_length
                        ));
                    }
                }
            }
            n => {
                decision_factors.push(format!("{}D array: Using default chunking strategy", n));
            }
        }

        Ok(decision_factors)
    }

    /// Optimize chunking for parallel processing.
    fn optimize_for_parallel_processing(
        &self,
        initial_chunk_size: usize,
        mut decision_factors: Vec<String>,
        params: &AdaptiveChunkingParams,
    ) -> (usize, Vec<String>) {
        let mut chunk_size = initial_chunk_size;

        if let Some(num_workers) = params.num_workers {
            let total_elements = self.size;

            // Ideally, we want at least num_workers * 2 chunks for good load balancing
            let target_num_chunks = num_workers * 2;
            let ideal_chunk_size = total_elements / target_num_chunks;

            if ideal_chunk_size >= params.min_chunk_size
                && ideal_chunk_size <= params.max_chunk_size
            {
                // Use the ideal chunk size for parallel processing
                chunk_size = ideal_chunk_size;
                decision_factors.push(format!(
                    "Parallel optimization: Adjusted chunk size to {} for {} workers",
                    chunk_size, num_workers
                ));
            } else if ideal_chunk_size < params.min_chunk_size {
                // If ideal size is too small, use minimum size
                chunk_size = params.min_chunk_size;
                let actual_chunks = total_elements / chunk_size
                    + if total_elements % chunk_size != 0 {
                        1
                    } else {
                        0
                    };
                decision_factors.push(format!(
                    "Parallel optimization: Using minimum chunk size {}, resulting in {} chunks for {} workers",
                    chunk_size, actual_chunks, num_workers
                ));
            }
        } else {
            decision_factors.push(
                "Parallel optimization requested but no worker count specified, using default chunking".to_string()
            );
        }

        (chunk_size, decision_factors)
    }
}

/// Builder for creating adaptive chunking parameters with a fluent API.
#[derive(Debug, Clone)]
pub struct AdaptiveChunkingBuilder {
    params: AdaptiveChunkingParams,
}

impl AdaptiveChunkingBuilder {
    /// Create a new builder with default parameters.
    pub fn new() -> Self {
        Self {
            params: AdaptiveChunkingParams::default(),
        }
    }

    /// Set the target memory usage per chunk.
    pub fn with_target_memory(mut self, bytes: usize) -> Self {
        self.params.target_memory_usage = bytes;
        self
    }

    /// Set the maximum chunk size.
    pub fn with_max_chunk_size(mut self, size: usize) -> Self {
        self.params.max_chunk_size = size;
        self
    }

    /// Set the minimum chunk size.
    pub fn with_min_chunk_size(mut self, size: usize) -> Self {
        self.params.min_chunk_size = size;
        self
    }

    /// Set the target chunk processing duration.
    pub fn with_target_duration(mut self, duration: Duration) -> Self {
        self.params.target_chunk_duration = Some(duration);
        self
    }

    /// Enable consideration of data distribution.
    pub fn consider_distribution(mut self, enable: bool) -> Self {
        self.params.consider_distribution = enable;
        self
    }

    /// Enable optimization for parallel processing.
    pub fn optimize_for_parallel(mut self, enable: bool) -> Self {
        self.params.optimize_for_parallel = enable;
        self
    }

    /// Set the number of worker threads to optimize for.
    pub fn with_num_workers(mut self, workers: usize) -> Self {
        self.params.num_workers = Some(workers);
        self
    }

    /// Build the parameters.
    pub fn build(self) -> AdaptiveChunkingParams {
        self.params
    }
}

impl Default for AdaptiveChunkingBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_adaptive_chunking_1d() {
        // Create a temporary directory for our test files
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_adaptive_1d.bin");

        // Create a test array and save it to a file
        let data: Vec<f64> = (0..100_000).map(|i| i as f64).collect();
        let mut file = File::create(&file_path).unwrap();
        for val in &data {
            file.write_all(&val.to_ne_bytes()).unwrap();
        }
        drop(file);

        // Create a memory-mapped array
        let mmap = MemoryMappedArray::<f64>::open(&file_path, &[100_000]).unwrap();

        // Create adaptive chunking parameters
        let params = AdaptiveChunkingBuilder::new()
            .with_target_memory(1 * 1024 * 1024) // 1MB chunks
            .with_min_chunk_size(1000)
            .with_max_chunk_size(50000)
            .build();

        // Calculate adaptive chunking
        let result = mmap.adaptive_chunking(params).unwrap();

        // Verify results
        match result.strategy {
            ChunkingStrategy::Fixed(chunk_size) => {
                // The chunk size should be close to 1MB / 8 bytes = 131072 elements,
                // but capped at our max of 50000
                assert_eq!(chunk_size, 50000);
            }
            _ => panic!("Expected fixed chunking strategy"),
        }

        // Verify that the estimated memory per chunk is reasonable
        assert!(result.estimated_memory_per_chunk > 0);

        // The decision factors should mention that it's a 1D array
        assert!(result
            .decision_factors
            .iter()
            .any(|s| s.contains("1D array")));
    }

    #[test]
    fn test_adaptive_chunking_2d() {
        // Create a temporary directory for our test files
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_adaptive_2d.bin");

        // Create dimensions that will test row alignment
        let rows = 1000;
        let cols = 120;

        // Create a test 2D array and save it to a file
        let data = Array2::<f64>::from_shape_fn((rows, cols), |(i, j)| (i * cols + j) as f64);
        let mut file = File::create(&file_path).unwrap();
        for val in data.iter() {
            file.write_all(&val.to_ne_bytes()).unwrap();
        }
        drop(file);

        // Create a memory-mapped array
        let mmap = MemoryMappedArray::<f64>::open(&file_path, &[rows, cols]).unwrap();

        // Create adaptive chunking parameters
        let params = AdaptiveChunkingBuilder::new()
            .with_target_memory(100 * 1024) // 100KB chunks
            .with_min_chunk_size(1000)
            .with_max_chunk_size(50000)
            .build();

        // Calculate adaptive chunking
        let result = mmap.adaptive_chunking(params).unwrap();

        // Verify results
        match result.strategy {
            ChunkingStrategy::Fixed(chunk_size) => {
                // The chunk size should be adjusted to be a multiple of the row length (120)
                assert_eq!(
                    chunk_size % cols,
                    0,
                    "Chunk size should be a multiple of row length"
                );
            }
            _ => panic!("Expected fixed chunking strategy"),
        }

        // The decision factors should mention that it's a 2D array
        assert!(result
            .decision_factors
            .iter()
            .any(|s| s.contains("2D array")));
    }

    #[test]
    fn test_adaptive_chunking_parallel() {
        // Create a temporary directory for our test files
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_adaptive_parallel.bin");

        // Create a large test array
        let data: Vec<f64> = (0..1_000_000).map(|i| i as f64).collect();
        let mut file = File::create(&file_path).unwrap();
        for val in &data {
            file.write_all(&val.to_ne_bytes()).unwrap();
        }
        drop(file);

        // Create a memory-mapped array
        let mmap = MemoryMappedArray::<f64>::open(&file_path, &[1_000_000]).unwrap();

        // Create adaptive chunking parameters optimized for parallel processing
        let params = AdaptiveChunkingBuilder::new()
            .with_target_memory(10 * 1024 * 1024) // 10MB chunks
            .optimize_for_parallel(true)
            .with_num_workers(4)
            .build();

        // Calculate adaptive chunking
        let result = mmap.adaptive_chunking(params).unwrap();

        // Verify results
        match result.strategy {
            ChunkingStrategy::Fixed(chunk_size) => {
                // With 4 workers and desiring 8 chunks (2*workers), each chunk should handle ~125,000 elements
                // But it might be adjusted based on other factors
                assert!(chunk_size > 0, "Chunk size should be positive");
            }
            _ => panic!("Expected fixed chunking strategy"),
        }

        // The decision factors should mention parallel optimization
        assert!(result
            .decision_factors
            .iter()
            .any(|s| s.contains("Parallel optimization")));
    }
}
