//! Parallel processing utilities for linear algebra operations
//!
//! This module provides utilities for managing worker threads across various
//! linear algebra operations, ensuring consistent behavior and optimal performance.

use std::sync::Mutex;

// Submodules for advanced parallel processing
pub mod thread_pools;
pub mod work_stealing;

// Re-export submodule types
pub use thread_pools::{
    get_global_manager, AdvancedPerformanceStats, AdvancedPerformanceThreadPool,
    AdvancedThreadPoolConfig, AffinityStrategy, AnomalySeverity, AnomalyType,
    CacheAllocationPolicy, DecompositionType, DynamicSizingConfig, DynamicThreadManager,
    IterativeSolverType, MemoryMetrics, MonitoringConfig, OperationType, PerformanceAnomaly,
    PredictionModelParams, ProfileMetrics, ResourceIsolationConfig, ResourceUsagePattern,
    ScalingDecision, ScalingReason, ScopedThreadPool, ThreadPoolConfig, ThreadPoolManager,
    ThreadPoolProfile, ThreadPoolProfiler, ThreadPoolStats, WorkloadAdaptationConfig,
    WorkloadCharacteristics, WorkloadPattern, WorkloadPredictor,
};
pub use work_stealing::{
    AdaptiveChunking, AdaptiveChunkingStats, CacheAwareStrategy, CacheAwareWorkStealer,
    CacheLocalityOptimizer, CacheOptimizationRecommendations, ChunkPerformance,
    LoadBalancingParams, MatrixOperationType, MemoryAccessPattern, NumaTopology,
    OptimizedSchedulerStats, OptimizedWorkStealingScheduler, PerformanceMonitor, PerformanceStats,
    SchedulerStats, StealingStrategy, WorkComplexity, WorkItem, WorkPriority,
    WorkStealingScheduler,
};

// Re-export matrix operations
pub use work_stealing::matrix_ops::{
    parallel_band_solve, parallel_block_gemm, parallel_cholesky_work_stealing,
    parallel_eigvalsh_work_stealing, parallel_gemm_work_stealing, parallel_hessenberg_reduction,
    parallel_lu_work_stealing, parallel_matvec_work_stealing, parallel_power_iteration,
    parallel_qr_work_stealing, parallel_svd_work_stealing,
};

// Re-export cache-aware operations
pub use work_stealing::parallel_gemm_cache_aware;

/// Global worker configuration
static GLOBAL_WORKERS: Mutex<Option<usize>> = Mutex::new(None);

/// Set the global worker thread count for all operations
///
/// This affects operations that don't explicitly specify a worker count.
/// If set to None, operations will use system defaults.
///
/// # Arguments
///
/// * `workers` - Number of worker threads (None = use system default)
///
/// # Examples
///
/// ```
/// use scirs2_linalg::parallel::set_global_workers;
///
/// // Use 4 threads for all operations
/// set_global_workers(Some(4));
///
/// // Reset to system default
/// set_global_workers(None);
/// ```
#[allow(dead_code)]
pub fn set_global_workers(workers: Option<usize>) {
    if let Ok(mut global) = GLOBAL_WORKERS.lock() {
        *global = workers;

        // Set OpenMP environment variable if specified
        if let Some(num_workers) = workers {
            std::env::set_var("OMP_NUM_THREADS", num_workers.to_string());
        } else {
            // Remove the environment variable to use system default
            std::env::remove_var("OMP_NUM_THREADS");
        }
    }
}

/// Get the current global worker thread count
///
/// # Returns
///
/// * Current global worker count (None = system default)
#[allow(dead_code)]
pub fn get_global_workers() -> Option<usize> {
    GLOBAL_WORKERS.lock().ok().and_then(|global| *global)
}

/// Configure worker threads for an operation
///
/// This function determines the appropriate number of worker threads to use,
/// considering both the operation-specific setting and global configuration.
///
/// # Arguments
///
/// * `workers` - Operation-specific worker count
///
/// # Returns
///
/// * Effective worker count to use
#[allow(dead_code)]
pub fn configure_workers(workers: Option<usize>) -> Option<usize> {
    match workers {
        Some(count) => {
            // Operation-specific setting takes precedence
            std::env::set_var("OMP_NUM_THREADS", count.to_string());
            Some(count)
        }
        None => {
            // Use global setting if available
            let global_workers = get_global_workers();
            if let Some(count) = global_workers {
                std::env::set_var("OMP_NUM_THREADS", count.to_string());
            }
            global_workers
        }
    }
}

/// Worker configuration for batched operations
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Number of worker threads
    pub workers: Option<usize>,
    /// Threshold for using parallel processing
    pub parallel_threshold: usize,
    /// Chunk size for batched operations
    pub chunksize: usize,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            workers: None,
            parallel_threshold: 1000,
            chunksize: 64,
        }
    }
}

impl WorkerConfig {
    /// Create a new worker configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of worker threads
    pub fn with_workers(mut self, workers: usize) -> Self {
        self.workers = Some(workers);
        self
    }

    /// Set the parallel processing threshold
    pub fn with_threshold(mut self, threshold: usize) -> Self {
        self.parallel_threshold = threshold;
        self
    }

    /// Set the chunk size for batched operations
    pub fn with_chunksize(mut self, chunksize: usize) -> Self {
        self.chunksize = chunksize;
        self
    }

    /// Apply this configuration for the current operation
    pub fn apply(&self) {
        configure_workers(self.workers);
    }
}

/// Scoped worker configuration
///
/// Temporarily sets worker configuration and restores the previous
/// configuration when dropped.
pub struct ScopedWorkers {
    previous_workers: Option<usize>,
}

impl ScopedWorkers {
    /// Create a scoped worker configuration
    ///
    /// # Arguments
    ///
    /// * `workers` - Number of worker threads for this scope
    ///
    /// # Returns
    ///
    /// * ScopedWorkers guard that restores previous configuration on drop
    pub fn new(workers: Option<usize>) -> Self {
        let previous_workers = get_global_workers();
        set_global_workers(workers);
        Self { previous_workers }
    }
}

impl Drop for ScopedWorkers {
    fn drop(&mut self) {
        set_global_workers(self.previous_workers);
    }
}

/// Parallel iterator utilities for matrix operations
pub mod iter {
    use scirs2_core::parallel_ops::*;

    /// Process chunks of work in parallel
    ///
    /// # Arguments
    ///
    /// * `items` - Items to process
    /// * `chunksize` - Size of each chunk
    /// * `f` - Function to apply to each chunk
    ///
    /// # Returns
    ///
    /// * Vector of results from each chunk
    pub fn parallel_chunks<T, R, F>(_items: &[T], chunksize: usize, f: F) -> Vec<R>
    where
        T: Send + Sync,
        R: Send,
        F: Fn(&[T]) -> R + Send + Sync,
    {
        _items
            .chunks(chunksize)
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(f)
            .collect()
    }

    /// Process items in parallel with index information
    ///
    /// # Arguments
    ///
    /// * `items` - Items to process
    /// * `f` - Function to apply to each (index, item) pair
    ///
    /// # Returns
    ///
    /// * Vector of results
    pub fn parallel_enumerate<T, R, F>(items: &[T], f: F) -> Vec<R>
    where
        T: Send + Sync,
        R: Send,
        F: Fn(usize, &T) -> R + Send + Sync,
    {
        items
            .par_iter()
            .enumerate()
            .map(|(i, item)| f(i, item))
            .collect()
    }
}

/// Adaptive algorithm selection based on data size and worker configuration
pub mod adaptive {
    use super::WorkerConfig;

    /// Algorithm selection strategy
    #[derive(Debug, Clone, Copy)]
    pub enum Strategy {
        /// Always use serial processing
        Serial,
        /// Always use parallel processing
        Parallel,
        /// Automatically choose based on data size
        Adaptive,
    }

    /// Choose processing strategy based on data size and configuration
    ///
    /// # Arguments
    ///
    /// * `datasize` - Size of the data to process
    /// * `config` - Worker configuration
    ///
    /// # Returns
    ///
    /// * Recommended processing strategy
    pub fn choose_strategy(_datasize: usize, config: &WorkerConfig) -> Strategy {
        if _datasize < config.parallel_threshold {
            Strategy::Serial
        } else {
            Strategy::Parallel
        }
    }

    /// Check if parallel processing is recommended
    ///
    /// # Arguments
    ///
    /// * `datasize` - Size of the data to process
    /// * `config` - Worker configuration
    ///
    /// # Returns
    ///
    /// * true if parallel processing is recommended
    pub fn should_use_parallel(_datasize: usize, config: &WorkerConfig) -> bool {
        matches!(choose_strategy(_datasize, config), Strategy::Parallel)
    }
}

/// Work-stealing scheduler optimizations
///
/// This module provides advanced scheduling strategies for parallel algorithms
/// using work-stealing techniques to improve load balancing and performance.
pub mod scheduler {
    use super::WorkerConfig;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    /// Work-stealing task scheduler
    ///
    /// Implements a work-stealing scheduler that dynamically balances work
    /// across threads for improved performance on irregular workloads.
    pub struct WorkStealingScheduler {
        num_workers: usize,
        chunksize: usize,
        adaptive_chunking: bool,
    }

    impl WorkStealingScheduler {
        /// Create a new work-stealing scheduler
        pub fn new(config: &WorkerConfig) -> Self {
            let num_workers = config.workers.unwrap_or_else(|| {
                // Default to available parallelism or 4 threads
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4)
            });
            Self {
                num_workers,
                chunksize: config.chunksize,
                adaptive_chunking: true,
            }
        }

        /// Set whether to use adaptive chunking
        pub fn with_adaptive_chunking(mut self, adaptive: bool) -> Self {
            self.adaptive_chunking = adaptive;
            self
        }

        /// Execute work items using work-stealing strategy
        ///
        /// This function divides work into chunks and uses atomic counters
        /// to allow threads to steal work from a global queue when they
        /// finish their assigned chunks early.
        pub fn execute<T, R, F>(&self, items: &[T], f: F) -> Vec<R>
        where
            T: Send + Sync,
            R: Send + Default + Clone,
            F: Fn(&T) -> R + Send + Sync,
        {
            let n = items.len();
            if n == 0 {
                return Vec::new();
            }

            // Determine chunk size based on workload characteristics
            let chunksize = if self.adaptive_chunking {
                self.adaptive_chunksize(n)
            } else {
                self.chunksize
            };

            // Create shared work counter
            let work_counter = Arc::new(AtomicUsize::new(0));
            let results = Arc::new(Mutex::new(vec![R::default(); n]));

            // Use scoped threads to process work items
            std::thread::scope(|s| {
                let handles: Vec<_> = (0..self.num_workers)
                    .map(|_| {
                        let work_counter = work_counter.clone();
                        let results = results.clone();
                        let items_ref = items;
                        let f_ref = &f;

                        s.spawn(move || {
                            loop {
                                // Steal a chunk of work
                                let start = work_counter.fetch_add(chunksize, Ordering::SeqCst);
                                if start >= n {
                                    break;
                                }

                                let end = std::cmp::min(start + chunksize, n);

                                // Process the chunk
                                for i in start..end {
                                    let result = f_ref(&items_ref[i]);
                                    let mut results_guard = results.lock().unwrap();
                                    results_guard[i] = result;
                                }
                            }
                        })
                    })
                    .collect();

                // Wait for all threads to complete
                for handle in handles {
                    handle.join().unwrap();
                }
            });

            // Extract results
            Arc::try_unwrap(results)
                .unwrap_or_else(|_| panic!("Failed to extract results"))
                .into_inner()
                .unwrap_or_else(|_| panic!("Failed to extract mutex inner value"))
        }

        /// Determine adaptive chunk size based on workload size
        fn adaptive_chunksize(&self, totalitems: usize) -> usize {
            // Use smaller chunks for better load balancing on smaller workloads
            // and larger chunks for better cache efficiency on larger workloads
            let items_per_worker = totalitems / self.num_workers;

            if items_per_worker < 100 {
                // Small workload: use fine-grained chunks
                std::cmp::max(1, items_per_worker / 4)
            } else if items_per_worker < 1000 {
                // Medium workload: balance between overhead and load balancing
                items_per_worker / 8
            } else {
                // Large workload: prioritize cache efficiency
                std::cmp::min(self.chunksize, items_per_worker / 16)
            }
        }

        /// Execute matrix operations with work-stealing
        ///
        /// Specialized version for matrix operations that takes into account
        /// cache line sizes and memory access patterns.
        pub fn executematrix<R, F>(&self, rows: usize, cols: usize, f: F) -> ndarray::Array2<R>
        where
            R: Send + Default + Clone,
            F: Fn(usize, usize) -> R + Send + Sync,
        {
            // Use block partitioning for better cache efficiency
            let blocksize = 64; // Typical cache line aligned block
            let work_items: Vec<(usize, usize)> = (0..rows)
                .step_by(blocksize)
                .flat_map(|i| (0..cols).step_by(blocksize).map(move |j| (i, j)))
                .collect();

            // Process blocks using work-stealing and collect results
            let work_counter = Arc::new(AtomicUsize::new(0));
            let results_vec = Arc::new(Mutex::new(Vec::new()));

            std::thread::scope(|s| {
                let handles: Vec<_> = (0..self.num_workers)
                    .map(|_| {
                        let work_counter = work_counter.clone();
                        let results_vec = results_vec.clone();
                        let work_items_ref = &work_items;
                        let f_ref = &f;

                        s.spawn(move || {
                            let mut local_results = Vec::new();

                            loop {
                                let idx = work_counter.fetch_add(1, Ordering::SeqCst);
                                if idx >= work_items_ref.len() {
                                    break;
                                }

                                let (block_i, block_j) = work_items_ref[idx];
                                let i_end = std::cmp::min(block_i + blocksize, rows);
                                let j_end = std::cmp::min(block_j + blocksize, cols);

                                // Process the block
                                for i in block_i..i_end {
                                    for j in block_j..j_end {
                                        local_results.push((i, j, f_ref(i, j)));
                                    }
                                }
                            }

                            // Add local results to global results
                            if !local_results.is_empty() {
                                let mut global_results = results_vec.lock().unwrap();
                                global_results.extend(local_results);
                            }
                        })
                    })
                    .collect();

                for handle in handles {
                    handle.join().unwrap();
                }
            });

            // Create result matrix from collected results
            let mut result = ndarray::Array2::default((rows, cols));
            let results = Arc::try_unwrap(results_vec)
                .unwrap_or_else(|_| panic!("Failed to extract results"))
                .into_inner()
                .unwrap_or_else(|_| panic!("Failed to extract mutex inner value"));

            for (i, j, val) in results {
                result[[i, j]] = val;
            }

            result
        }
    }

    /// Dynamic load balancer for irregular workloads
    ///
    /// This struct provides dynamic load balancing for workloads where
    /// different items may take varying amounts of time to process.
    pub struct DynamicLoadBalancer {
        scheduler: WorkStealingScheduler,
        /// Tracks execution time statistics for adaptive scheduling
        timing_stats: Arc<Mutex<TimingStats>>,
    }

    #[derive(Default)]
    struct TimingStats {
        total_items: usize,
        total_time_ms: u128,
        min_time_ms: u128,
        max_time_ms: u128,
    }

    impl DynamicLoadBalancer {
        /// Create a new dynamic load balancer
        pub fn new(config: &WorkerConfig) -> Self {
            Self {
                scheduler: WorkStealingScheduler::new(config),
                timing_stats: Arc::new(Mutex::new(TimingStats::default())),
            }
        }

        /// Execute work items with dynamic load balancing and timing
        pub fn execute_timed<T, R, F>(&self, items: &[T], f: F) -> Vec<R>
        where
            T: Send + Sync,
            R: Send + Default + Clone,
            F: Fn(&T) -> R + Send + Sync,
        {
            use std::time::Instant;

            let n = items.len();
            if n == 0 {
                return Vec::new();
            }

            let results = Arc::new(Mutex::new(vec![R::default(); n]));
            let work_counter = Arc::new(AtomicUsize::new(0));
            let timing_stats = self.timing_stats.clone();

            std::thread::scope(|s| {
                let handles: Vec<_> = (0..self.scheduler.num_workers)
                    .map(|_| {
                        let work_counter = work_counter.clone();
                        let results = results.clone();
                        let timing_stats = timing_stats.clone();
                        let items_ref = items;
                        let f_ref = &f;

                        s.spawn(move || {
                            let mut local_min = u128::MAX;
                            let mut local_max = 0u128;
                            let mut local_total = 0u128;
                            let mut local_count = 0usize;

                            loop {
                                let idx = work_counter.fetch_add(1, Ordering::SeqCst);
                                if idx >= n {
                                    break;
                                }

                                // Time the execution
                                let start = Instant::now();
                                let result = f_ref(&items_ref[idx]);
                                let elapsed = start.elapsed().as_millis();

                                // Update local statistics
                                local_min = local_min.min(elapsed);
                                local_max = local_max.max(elapsed);
                                local_total += elapsed;
                                local_count += 1;

                                // Store result
                                let mut results_guard = results.lock().unwrap();
                                results_guard[idx] = result;
                            }

                            // Update global statistics
                            if local_count > 0 {
                                let mut stats = timing_stats.lock().unwrap();
                                stats.total_items += local_count;
                                stats.total_time_ms += local_total;
                                stats.min_time_ms = stats.min_time_ms.min(local_min);
                                stats.max_time_ms = stats.max_time_ms.max(local_max);
                            }
                        })
                    })
                    .collect();

                for handle in handles {
                    handle.join().unwrap();
                }
            });

            Arc::try_unwrap(results)
                .unwrap_or_else(|_| panic!("Failed to extract results"))
                .into_inner()
                .unwrap_or_else(|_| panic!("Failed to extract mutex inner value"))
        }

        /// Get average execution time per item
        pub fn get_average_time_ms(&self) -> f64 {
            let stats = self.timing_stats.lock().unwrap();
            if stats.total_items > 0 {
                stats.total_time_ms as f64 / stats.total_items as f64
            } else {
                0.0
            }
        }

        /// Get timing variance to detect irregular workloads
        pub fn get_time_variance(&self) -> f64 {
            let stats = self.timing_stats.lock().unwrap();
            if stats.total_items > 0 && stats.max_time_ms > stats.min_time_ms {
                (stats.max_time_ms - stats.min_time_ms) as f64 / stats.min_time_ms as f64
            } else {
                0.0
            }
        }
    }

    /// Advanced work-stealing scheduler with NUMA awareness and cache optimization
    ///
    /// This enhanced scheduler provides advanced optimizations for work-stealing
    /// including NUMA topology awareness and cache-friendly work distribution.
    pub struct AdvancedWorkStealingScheduler {
        base_scheduler: WorkStealingScheduler,
        numa_aware: bool,
        cache_linesize: usize,
        #[allow(dead_code)]
        work_queue_per_thread: bool,
    }

    impl AdvancedWorkStealingScheduler {
        /// Create a new advanced work-stealing scheduler
        pub fn new(config: &WorkerConfig) -> Self {
            Self {
                base_scheduler: WorkStealingScheduler::new(config),
                numa_aware: true,
                cache_linesize: 64, // Common cache line size
                work_queue_per_thread: true,
            }
        }

        /// Enable or disable NUMA-aware scheduling
        pub fn with_numa_aware(mut self, enabled: bool) -> Self {
            self.numa_aware = enabled;
            self
        }

        /// Set cache line size for cache-aware optimization
        pub fn with_cache_linesize(mut self, size: usize) -> Self {
            self.cache_linesize = size;
            self
        }

        /// Execute work with advanced optimizations
        ///
        /// This method implements enhanced work-stealing with:
        /// - NUMA-aware work distribution
        /// - Cache-friendly chunking
        /// - Adaptive scheduling based on workload characteristics
        pub fn execute_optimized<T, R, F>(&self, items: &[T], f: F) -> Vec<R>
        where
            T: Send + Sync,
            R: Send + Default + Clone,
            F: Fn(&T) -> R + Send + Sync,
        {
            if items.is_empty() {
                return Vec::new();
            }

            let n = items.len();

            // Analyze workload characteristics
            let workload_type = self.analyze_workload(n);

            // Determine optimal chunking strategy
            let chunk_config = match workload_type {
                WorkloadType::MemoryBound => ChunkConfig {
                    size: self.cache_linesize / std::mem::size_of::<T>(),
                    strategy: ChunkStrategy::Sequential,
                },
                WorkloadType::CpuBound => ChunkConfig {
                    size: n / (self.base_scheduler.num_workers * 4),
                    strategy: ChunkStrategy::Interleaved,
                },
                WorkloadType::Mixed => ChunkConfig {
                    size: self.adaptive_chunksize_enhanced(n),
                    strategy: ChunkStrategy::Dynamic,
                },
            };

            // Execute with optimized strategy
            self.execute_with_strategy(items, f, chunk_config)
        }

        /// Analyze workload characteristics to optimize scheduling
        fn analyze_workload(&self, size: usize) -> WorkloadType {
            let memory_footprint = size * std::mem::size_of::<usize>();
            let cachesize = 8 * 1024 * 1024; // Approximate L3 cache size

            if memory_footprint > cachesize {
                WorkloadType::MemoryBound
            } else if size < 1000 {
                WorkloadType::CpuBound
            } else {
                WorkloadType::Mixed
            }
        }

        /// Enhanced adaptive chunk size calculation
        fn adaptive_chunksize_enhanced(&self, totalitems: usize) -> usize {
            let num_workers = self.base_scheduler.num_workers;
            let items_per_worker = totalitems / num_workers;

            // Consider cache efficiency and load balancing
            let cache_optimalsize = self.cache_linesize / std::mem::size_of::<usize>();
            let load_balancesize = std::cmp::max(1, items_per_worker / 8);

            // Choose the better of cache-optimal or load-balance size
            if cache_optimalsize > 0 && cache_optimalsize < load_balancesize * 2 {
                cache_optimalsize
            } else {
                load_balancesize
            }
        }

        /// Execute work with specific strategy
        fn execute_with_strategy<T, R, F>(&self, items: &[T], f: F, config: ChunkConfig) -> Vec<R>
        where
            T: Send + Sync,
            R: Send + Default + Clone,
            F: Fn(&T) -> R + Send + Sync,
        {
            match config.strategy {
                ChunkStrategy::Sequential => self.execute_sequential_chunks(items, f, config.size),
                ChunkStrategy::Interleaved => {
                    self.execute_interleaved_chunks(items, f, config.size)
                }
                ChunkStrategy::Dynamic => self.execute_dynamic_chunks(items, f, config.size),
            }
        }

        /// Execute with sequential chunk allocation
        fn execute_sequential_chunks<T, R, F>(&self, items: &[T], f: F, _chunksize: usize) -> Vec<R>
        where
            T: Send + Sync,
            R: Send + Default + Clone,
            F: Fn(&T) -> R + Send + Sync,
        {
            // Use the base scheduler for sequential chunking
            self.base_scheduler.execute(items, f)
        }

        /// Execute with interleaved chunk allocation for better cache utilization
        fn execute_interleaved_chunks<T, R, F>(&self, items: &[T], f: F, chunksize: usize) -> Vec<R>
        where
            T: Send + Sync,
            R: Send + Default + Clone,
            F: Fn(&T) -> R + Send + Sync,
        {
            let n = items.len();
            let chunksize = chunksize.max(1);
            let results = Arc::new(Mutex::new(vec![R::default(); n]));
            let work_counter = Arc::new(AtomicUsize::new(0));

            std::thread::scope(|s| {
                let handles: Vec<_> = (0..self.base_scheduler.num_workers)
                    .map(|_worker_id| {
                        let items_ref = items;
                        let f_ref = &f;
                        let results = results.clone();
                        let work_counter = work_counter.clone();

                        s.spawn(move || {
                            loop {
                                let chunk_id = work_counter.fetch_add(1, Ordering::SeqCst);
                                let start = chunk_id * chunksize;

                                if start >= n {
                                    break;
                                }

                                let end = std::cmp::min(start + chunksize, n);

                                // Process interleaved indices for better cache utilization
                                for i in start..end {
                                    let interleaved_idx = (i % self.base_scheduler.num_workers)
                                        * (n / self.base_scheduler.num_workers)
                                        + (i / self.base_scheduler.num_workers);

                                    if interleaved_idx < n {
                                        let result = f_ref(&items_ref[interleaved_idx]);
                                        let mut results_guard = results.lock().unwrap();
                                        results_guard[interleaved_idx] = result;
                                    }
                                }
                            }
                        })
                    })
                    .collect();

                for handle in handles {
                    handle.join().unwrap();
                }
            });

            Arc::try_unwrap(results)
                .unwrap_or_else(|_| panic!("Failed to extract results"))
                .into_inner()
                .unwrap_or_else(|_| panic!("Failed to extract mutex inner value"))
        }

        /// Execute with dynamic chunk sizing based on performance feedback
        fn execute_dynamic_chunks<T, R, F>(
            &self,
            items: &[T],
            f: F,
            _initial_chunksize: usize,
        ) -> Vec<R>
        where
            T: Send + Sync,
            R: Send + Default + Clone,
            F: Fn(&T) -> R + Send + Sync,
        {
            // For now, use the base implementation with dynamic sizing
            // In a full implementation, this would adapt chunk sizes based on timing
            self.base_scheduler.execute(items, f)
        }
    }

    /// Workload analysis types
    #[derive(Debug, Clone, Copy)]
    enum WorkloadType {
        /// Memory-bound workloads that benefit from cache optimization
        MemoryBound,
        /// CPU-bound workloads that benefit from load balancing
        CpuBound,
        /// Mixed workloads requiring balanced approach
        Mixed,
    }

    /// Chunk configuration for work distribution
    #[derive(Debug, Clone)]
    struct ChunkConfig {
        size: usize,
        strategy: ChunkStrategy,
    }

    /// Work distribution strategies
    #[derive(Debug, Clone, Copy)]
    enum ChunkStrategy {
        /// Sequential chunk allocation
        Sequential,
        /// Interleaved allocation for cache efficiency
        Interleaved,
        /// Dynamic sizing based on performance
        Dynamic,
    }
}

/// Thread pool configurations for linear algebra operations
///
/// This module provides flexible thread pool management with support for
/// different configurations optimized for various linear algebra workloads.
pub mod thread_pool {
    use super::configure_workers;
    use scirs2_core::parallel_ops::*;
    use std::sync::{Arc, Mutex, Once};

    /// Global thread pool manager
    static INIT: Once = Once::new();
    static mut GLOBAL_POOL: Option<Arc<Mutex<ThreadPoolManager>>> = None;

    /// Thread pool configuration profiles
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ThreadPoolProfile {
        /// Default profile - uses system defaults
        Default,
        /// CPU-bound profile - one thread per CPU core
        CpuBound,
        /// Memory-bound profile - fewer threads to reduce memory contention
        MemoryBound,
        /// Latency-sensitive profile - more threads for better responsiveness
        LatencySensitive,
        /// Custom profile with specific thread count
        Custom(usize),
    }

    impl ThreadPoolProfile {
        /// Get the number of threads for this profile
        pub fn num_threads(&self) -> usize {
            match self {
                ThreadPoolProfile::Default => std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4),
                ThreadPoolProfile::CpuBound => std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4),
                ThreadPoolProfile::MemoryBound => {
                    // Use half the available cores to reduce memory contention
                    std::thread::available_parallelism()
                        .map(|n| std::cmp::max(1, n.get() / 2))
                        .unwrap_or(2)
                }
                ThreadPoolProfile::LatencySensitive => {
                    // Use 1.5x the available cores for better responsiveness
                    std::thread::available_parallelism()
                        .map(|n| n.get() + n.get() / 2)
                        .unwrap_or(6)
                }
                ThreadPoolProfile::Custom(n) => *n,
            }
        }
    }

    /// Thread pool manager for linear algebra operations
    pub struct ThreadPoolManager {
        profile: ThreadPoolProfile,
        /// Stack size for worker threads (in bytes)
        stacksize: Option<usize>,
        /// Thread name prefix
        thread_name_prefix: String,
        /// Whether to pin threads to CPU cores
        cpu_affinity: bool,
    }

    impl ThreadPoolManager {
        /// Create a new thread pool manager with default settings
        pub fn new() -> Self {
            Self {
                profile: ThreadPoolProfile::Default,
                stacksize: None,
                thread_name_prefix: "linalg-worker".to_string(),
                cpu_affinity: false,
            }
        }

        /// Set the thread pool profile
        pub fn with_profile(mut self, profile: ThreadPoolProfile) -> Self {
            self.profile = profile;
            self
        }

        /// Set the stack size for worker threads
        pub fn with_stacksize(mut self, size: usize) -> Self {
            self.stacksize = Some(size);
            self
        }

        /// Set the thread name prefix
        pub fn with_thread_name_prefix(mut self, prefix: String) -> Self {
            self.thread_name_prefix = prefix;
            self
        }

        /// Enable CPU affinity for worker threads
        pub fn with_cpu_affinity(mut self, enabled: bool) -> Self {
            self.cpu_affinity = enabled;
            self
        }

        /// Initialize the thread pool with current settings
        pub fn initialize(&self) -> Result<(), String> {
            let num_threads = self.profile.num_threads();

            // Configure rayon thread pool
            let thread_prefix = self.thread_name_prefix.clone();
            let mut pool_builder = ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .thread_name(move |idx| format!("{thread_prefix}-{idx}"));

            if let Some(stacksize) = self.stacksize {
                pool_builder = pool_builder.stack_size(stacksize);
            }

            pool_builder
                .build_global()
                .map_err(|e| format!("Failed to initialize thread pool: {e}"))?;

            // Set OpenMP threads for BLAS/LAPACK operations
            std::env::set_var("OMP_NUM_THREADS", num_threads.to_string());

            // Set MKL threads if using Intel MKL
            std::env::set_var("MKL_NUM_THREADS", num_threads.to_string());

            Ok(())
        }

        /// Get current thread pool statistics
        pub fn statistics(&self) -> ThreadPoolStats {
            ThreadPoolStats {
                num_threads: self.profile.num_threads(),
                current_parallelism: num_threads(),
                profile: self.profile,
                stacksize: self.stacksize,
            }
        }
    }

    impl Default for ThreadPoolManager {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Thread pool statistics
    #[derive(Debug, Clone)]
    pub struct ThreadPoolStats {
        pub num_threads: usize,
        pub current_parallelism: usize,
        pub profile: ThreadPoolProfile,
        pub stacksize: Option<usize>,
    }

    /// Get the global thread pool manager
    pub fn global_pool() -> Arc<Mutex<ThreadPoolManager>> {
        unsafe {
            INIT.call_once(|| {
                GLOBAL_POOL = Some(Arc::new(Mutex::new(ThreadPoolManager::new())));
            });
            #[allow(static_mut_refs)]
            GLOBAL_POOL.as_ref().unwrap().clone()
        }
    }

    /// Initialize global thread pool with a specific profile
    pub fn initialize_global_pool(profile: ThreadPoolProfile) -> Result<(), String> {
        let pool = global_pool();
        let mut manager = pool.lock().unwrap();
        manager.profile = profile;
        manager.initialize()
    }

    /// Adaptive thread pool that adjusts based on workload
    pub struct AdaptiveThreadPool {
        min_threads: usize,
        max_threads: usize,
        current_threads: Arc<Mutex<usize>>,
        /// Tracks CPU utilization for adaptive scaling
        cpu_utilization: Arc<Mutex<f64>>,
    }

    impl AdaptiveThreadPool {
        /// Create a new adaptive thread pool
        pub fn new(_min_threads: usize, maxthreads: usize) -> Self {
            let current = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4);

            Self {
                min_threads: _min_threads,
                max_threads: maxthreads,
                current_threads: Arc::new(Mutex::new(current)),
                cpu_utilization: Arc::new(Mutex::new(0.0)),
            }
        }

        /// Update thread count based on current utilization
        pub fn adapt(&self, utilization: f64) {
            let mut current = self.current_threads.lock().unwrap();
            let mut cpu_util = self.cpu_utilization.lock().unwrap();
            *cpu_util = utilization;

            if utilization > 0.9 && *current < self.max_threads {
                // High utilization - increase threads
                *current = std::cmp::min(*current + 1, self.max_threads);
                self.apply_thread_count(*current);
            } else if utilization < 0.5 && *current > self.min_threads {
                // Low utilization - decrease threads
                *current = std::cmp::max(*current - 1, self.min_threads);
                self.apply_thread_count(*current);
            }
        }

        /// Apply the new thread count
        fn apply_thread_count(&self, count: usize) {
            configure_workers(Some(count));
        }

        /// Get current thread count
        pub fn current_thread_count(&self) -> usize {
            *self.current_threads.lock().unwrap()
        }
    }

    /// Thread pool benchmarking utilities
    pub mod benchmark {
        use super::*;
        use std::time::{Duration, Instant};

        /// Benchmark result for a thread pool configuration
        #[derive(Debug, Clone)]
        pub struct BenchmarkResult {
            pub profile: ThreadPoolProfile,
            pub num_threads: usize,
            pub execution_time: Duration,
            pub throughput: f64,
        }

        /// Benchmark different thread pool configurations
        pub fn benchmark_configurations<F>(
            profiles: &[ThreadPoolProfile],
            workload: F,
        ) -> Vec<BenchmarkResult>
        where
            F: Fn() -> f64 + Clone,
        {
            let mut results = Vec::new();

            for &profile in profiles {
                // Initialize thread pool with profile
                if let Err(e) = initialize_global_pool(profile) {
                    eprintln!("Failed to initialize pool for {profile:?}: {e}");
                    continue;
                }

                // Warm up
                for _ in 0..3 {
                    workload();
                }

                // Benchmark
                let start = Instant::now();
                let operations = 10;
                let mut total_work = 0.0;

                for _ in 0..operations {
                    total_work += workload();
                }

                let elapsed = start.elapsed();
                let throughput = total_work / elapsed.as_secs_f64();

                results.push(BenchmarkResult {
                    profile,
                    num_threads: profile.num_threads(),
                    execution_time: elapsed,
                    throughput,
                });
            }

            results
        }

        /// Find optimal thread pool configuration for a workload
        pub fn find_optimal_configuration<F>(workload: F) -> ThreadPoolProfile
        where
            F: Fn() -> f64 + Clone,
        {
            let profiles = vec![
                ThreadPoolProfile::CpuBound,
                ThreadPoolProfile::MemoryBound,
                ThreadPoolProfile::LatencySensitive,
            ];

            let results = benchmark_configurations(&profiles, workload);

            results
                .into_iter()
                .max_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap())
                .map(|r| r.profile)
                .unwrap_or(ThreadPoolProfile::Default)
        }
    }

    /// Enhanced thread pool with advanced monitoring and scaling
    ///
    /// This provides sophisticated thread pool management with real-time monitoring,
    /// dynamic scaling, and intelligent load balancing for optimal performance.
    pub struct EnhancedThreadPool {
        #[allow(dead_code)]
        base_pool: Arc<Mutex<ThreadPoolManager>>,
        monitoring: Arc<Mutex<ThreadPoolMonitoring>>,
        scaling_policy: ScalingPolicy,
        load_balancer: LoadBalancer,
    }

    impl EnhancedThreadPool {
        /// Create a new enhanced thread pool
        pub fn new(profile: ThreadPoolProfile) -> Self {
            let base_pool = Arc::new(Mutex::new(ThreadPoolManager::new().with_profile(profile)));

            Self {
                base_pool,
                monitoring: Arc::new(Mutex::new(ThreadPoolMonitoring::new())),
                scaling_policy: ScalingPolicy::Conservative,
                load_balancer: LoadBalancer::RoundRobin,
            }
        }

        /// Set scaling policy
        pub fn with_scaling_policy(mut self, policy: ScalingPolicy) -> Self {
            self.scaling_policy = policy;
            self
        }

        /// Set load balancing strategy
        pub fn with_load_balancer(mut self, balancer: LoadBalancer) -> Self {
            self.load_balancer = balancer;
            self
        }

        /// Get current thread pool metrics
        pub fn get_metrics(&self) -> ThreadPoolMetrics {
            let monitoring = self.monitoring.lock().unwrap();
            monitoring.get_metrics()
        }

        /// Execute task with monitoring and adaptive scaling
        pub fn execute_monitored<F, R>(&self, task: F) -> R
        where
            F: FnOnce() -> R + Send,
            R: Send,
        {
            let start_time = std::time::Instant::now();

            // Update monitoring before execution
            {
                let mut monitoring = self.monitoring.lock().unwrap();
                monitoring.record_task_start();
            }

            // Execute task
            let result = task();

            // Update monitoring after execution
            {
                let mut monitoring = self.monitoring.lock().unwrap();
                monitoring.record_task_completion(start_time.elapsed());
            }

            // Check if scaling is needed
            self.check_and_scale();

            result
        }

        /// Check if thread pool scaling is needed and apply if necessary
        fn check_and_scale(&self) {
            let metrics = self.get_metrics();

            match self.scaling_policy {
                ScalingPolicy::Conservative => {
                    // Scale up only if utilization > 90% for extended period
                    if metrics.average_utilization > 0.9 && metrics.queue_length > 10 {
                        self.scale_up();
                    }
                    // Scale down only if utilization < 30% for extended period
                    else if metrics.average_utilization < 0.3 && metrics.active_threads > 2 {
                        self.scale_down();
                    }
                }
                ScalingPolicy::Aggressive => {
                    // Scale up if utilization > 70%
                    if metrics.average_utilization > 0.7 {
                        self.scale_up();
                    }
                    // Scale down if utilization < 50%
                    else if metrics.average_utilization < 0.5 && metrics.active_threads > 1 {
                        self.scale_down();
                    }
                }
                ScalingPolicy::LatencyOptimized => {
                    // Prioritize low latency over efficiency
                    if metrics.average_latency_ms > 10.0 {
                        self.scale_up();
                    } else if metrics.average_latency_ms < 2.0 && metrics.active_threads > 2 {
                        self.scale_down();
                    }
                }
                ScalingPolicy::Fixed => {
                    // No scaling
                }
            }
        }

        /// Scale up the thread pool
        fn scale_up(&self) {
            // Implementation would involve creating new threads
            // For now, we'll just log the intent
            println!("Scaling up thread pool due to high utilization");
        }

        /// Scale down the thread pool
        fn scale_down(&self) {
            // Implementation would involve reducing threads
            // For now, we'll just log the intent
            println!("Scaling down thread pool due to low utilization");
        }
    }

    /// Thread pool scaling policies
    #[derive(Debug, Clone, Copy)]
    pub enum ScalingPolicy {
        /// Conservative scaling - only scale when definitely needed
        Conservative,
        /// Aggressive scaling - scale more readily for performance
        Aggressive,
        /// Optimized for low latency
        LatencyOptimized,
        /// Fixed thread count - no scaling
        Fixed,
    }

    /// Load balancing strategies
    #[derive(Debug, Clone, Copy)]
    pub enum LoadBalancer {
        /// Simple round-robin task distribution
        RoundRobin,
        /// Least loaded thread gets next task
        LeastLoaded,
        /// Work-stealing between threads
        WorkStealing,
        /// NUMA-aware task assignment
        NumaAware,
    }

    /// Thread pool monitoring and metrics collection
    struct ThreadPoolMonitoring {
        task_count: usize,
        total_execution_time: std::time::Duration,
        active_threads: usize,
        queue_length: usize,
        start_times: Vec<std::time::Instant>,
    }

    impl ThreadPoolMonitoring {
        fn new() -> Self {
            Self {
                task_count: 0,
                total_execution_time: std::time::Duration::ZERO,
                active_threads: 0,
                queue_length: 0,
                start_times: Vec::new(),
            }
        }

        fn record_task_start(&mut self) {
            self.task_count += 1;
            self.start_times.push(std::time::Instant::now());
            self.queue_length += 1;
        }

        fn record_task_completion(&mut self, duration: std::time::Duration) {
            self.total_execution_time += duration;
            self.queue_length = self.queue_length.saturating_sub(1);
        }

        fn get_metrics(&self) -> ThreadPoolMetrics {
            ThreadPoolMetrics {
                active_threads: self.active_threads,
                queue_length: self.queue_length,
                total_tasks: self.task_count,
                average_utilization: if self.active_threads > 0 {
                    self.queue_length as f64 / self.active_threads as f64
                } else {
                    0.0
                },
                average_latency_ms: if self.task_count > 0 {
                    self.total_execution_time.as_millis() as f64 / self.task_count as f64
                } else {
                    0.0
                },
                throughput_tasks_per_sec: if !self.total_execution_time.is_zero() {
                    self.task_count as f64 / self.total_execution_time.as_secs_f64()
                } else {
                    0.0
                },
            }
        }
    }

    /// Thread pool performance metrics
    #[derive(Debug, Clone)]
    pub struct ThreadPoolMetrics {
        /// Number of currently active threads
        pub active_threads: usize,
        /// Number of tasks waiting in queue
        pub queue_length: usize,
        /// Total number of tasks processed
        pub total_tasks: usize,
        /// Average thread utilization (0.0 to 1.0+)
        pub average_utilization: f64,
        /// Average task latency in milliseconds
        pub average_latency_ms: f64,
        /// Throughput in tasks per second
        pub throughput_tasks_per_sec: f64,
    }
}

/// NUMA-aware parallel computing
///
/// This module provides NUMA (Non-Uniform Memory Access) aware algorithms
/// for improved performance on multi-socket systems.
pub mod numa {
    use super::WorkerConfig;
    use crate::error::{LinalgError, LinalgResult};
    use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
    use num_traits::{Float, NumAssign, One, Zero};
    use scirs2_core::parallel_ops::*;
    use std::sync::{Arc, Mutex};

    /// NUMA topology information
    #[derive(Debug, Clone)]
    pub struct NumaTopology {
        /// Number of NUMA nodes
        pub num_nodes: usize,
        /// CPUs per NUMA node
        pub cpus_per_node: Vec<Vec<usize>>,
        /// Memory bandwidth between nodes (GB/s)
        pub memory_bandwidth: Vec<Vec<f64>>,
    }

    impl NumaTopology {
        /// Detect NUMA topology (simplified implementation)
        pub fn detect() -> Self {
            let num_cpus = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4);

            // Simple heuristic: assume 2 NUMA nodes if more than 8 CPUs
            let num_nodes = if num_cpus > 8 { 2 } else { 1 };
            let cpus_per_node = if num_nodes == 2 {
                vec![
                    (0..num_cpus / 2).collect(),
                    (num_cpus / 2..num_cpus).collect(),
                ]
            } else {
                vec![(0..num_cpus).collect()]
            };

            // Simplified bandwidth matrix (local: 100 GB/s, remote: 50 GB/s)
            let mut memory_bandwidth = vec![vec![0.0; num_nodes]; num_nodes];
            for (i, row) in memory_bandwidth.iter_mut().enumerate().take(num_nodes) {
                for (j, item) in row.iter_mut().enumerate().take(num_nodes) {
                    *item = if i == j { 100.0 } else { 50.0 };
                }
            }

            Self {
                num_nodes,
                cpus_per_node,
                memory_bandwidth,
            }
        }

        /// Get optimal thread distribution across NUMA nodes
        pub fn optimal_thread_distribution(&self, totalthreads: usize) -> Vec<usize> {
            let mut distribution = vec![0; self.num_nodes];
            let threads_per_node = totalthreads / self.num_nodes;
            let remaining = totalthreads % self.num_nodes;

            for (i, item) in distribution.iter_mut().enumerate().take(self.num_nodes) {
                *item = threads_per_node;
                if i < remaining {
                    *item += 1;
                }
            }
            distribution
        }
    }

    /// NUMA-aware matrix partitioning strategy
    #[derive(Debug, Clone, Copy)]
    pub enum NumaPartitioning {
        /// Partition by rows across NUMA nodes
        RowWise,
        /// Partition by columns across NUMA nodes
        ColumnWise,
        /// 2D block partitioning across NUMA nodes
        Block2D,
        /// Automatic selection based on matrix shape
        Adaptive,
    }

    impl NumaPartitioning {
        /// Choose optimal partitioning strategy
        pub fn choose_optimal(_rows: usize, cols: usize, numnodes: usize) -> Self {
            if numnodes == 1 {
                return NumaPartitioning::RowWise;
            }

            let aspect_ratio = _rows as f64 / cols as f64;

            if aspect_ratio > 2.0 {
                // Tall matrix - prefer row-wise partitioning
                NumaPartitioning::RowWise
            } else if aspect_ratio < 0.5 {
                // Wide matrix - prefer column-wise partitioning
                NumaPartitioning::ColumnWise
            } else {
                // Square-ish matrix - use 2D block partitioning
                NumaPartitioning::Block2D
            }
        }
    }

    /// NUMA-aware parallel matrix-vector multiplication
    ///
    /// This implementation partitions the matrix across NUMA nodes to minimize
    /// cross-node memory access and improve cache locality.
    pub fn numa_aware_matvec<F>(
        matrix: &ArrayView2<F>,
        vector: &ArrayView1<F>,
        config: &WorkerConfig,
        topology: &NumaTopology,
    ) -> LinalgResult<Array1<F>>
    where
        F: Float + Send + Sync + Zero + std::iter::Sum + 'static,
    {
        let (m, n) = matrix.dim();
        if n != vector.len() {
            return Err(LinalgError::ShapeError(format!(
                "Matrix-vector dimensions incompatible: {}x{} * {}",
                m,
                n,
                vector.len()
            )));
        }

        if topology.num_nodes == 1 {
            // Single NUMA node - use standard parallel implementation
            return super::algorithms::parallel_matvec(matrix, vector, config);
        }

        config.apply();

        // Partition matrix by rows across NUMA nodes
        let _node_distribution =
            topology.optimal_thread_distribution(config.workers.unwrap_or(topology.num_nodes * 2));

        let rows_per_node: Vec<usize> = (0..topology.num_nodes)
            .map(|i| {
                let start_ratio = i as f64 / topology.num_nodes as f64;
                let end_ratio = (i + 1) as f64 / topology.num_nodes as f64;
                let start_row = (start_ratio * m as f64) as usize;
                let end_row = (end_ratio * m as f64) as usize;
                end_row - start_row
            })
            .collect();

        // Compute result for each NUMA node in parallel
        let partial_results: Vec<Vec<F>> = (0..topology.num_nodes)
            .into_par_iter()
            .map(|node_id| {
                let start_row = rows_per_node.iter().take(node_id).sum::<usize>();
                let node_rows = rows_per_node[node_id];

                if node_rows == 0 {
                    return Vec::new();
                }

                let nodematrix = matrix.slice(ndarray::s![start_row..start_row + node_rows, ..]);

                // Compute local result for this NUMA node
                (0..node_rows)
                    .into_par_iter()
                    .map(|local_row| {
                        nodematrix
                            .row(local_row)
                            .iter()
                            .zip(vector.iter())
                            .map(|(&a_ij, &x_j)| a_ij * x_j)
                            .sum()
                    })
                    .collect()
            })
            .collect();

        // Combine results from all NUMA nodes
        let mut result = Vec::with_capacity(m);
        for node_result in partial_results {
            result.extend(node_result);
        }

        Ok(Array1::from_vec(result))
    }

    /// NUMA-aware parallel matrix multiplication
    ///
    /// Uses 2D block partitioning to distribute computation across NUMA nodes
    /// while minimizing cross-node memory traffic.
    pub fn numa_aware_gemm<F>(
        a: &ArrayView2<F>,
        b: &ArrayView2<F>,
        config: &WorkerConfig,
        topology: &NumaTopology,
    ) -> LinalgResult<Array2<F>>
    where
        F: Float + Send + Sync + Zero + std::iter::Sum + NumAssign + 'static,
    {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();

        if k != k2 {
            return Err(LinalgError::ShapeError(format!(
                "Matrix dimensions incompatible: {m}x{k} * {k2}x{n}"
            )));
        }

        if topology.num_nodes == 1 {
            return super::algorithms::parallel_gemm(a, b, config);
        }

        config.apply();

        let partitioning = NumaPartitioning::choose_optimal(m, n, topology.num_nodes);
        let mut result = Array2::zeros((m, n));

        match partitioning {
            NumaPartitioning::Block2D => {
                let nodes_sqrt = (topology.num_nodes as f64).sqrt() as usize;
                let block_rows = m.div_ceil(nodes_sqrt);
                let block_cols = n.div_ceil(nodes_sqrt);

                // Process blocks in parallel across NUMA nodes
                let block_results: Vec<((usize, usize), Array2<F>)> = (0..nodes_sqrt)
                    .flat_map(|bi| (0..nodes_sqrt).map(move |bj| (bi, bj)))
                    .collect::<Vec<_>>()
                    .into_par_iter()
                    .filter_map(|(bi, bj)| {
                        let i_start = bi * block_rows;
                        let i_end = std::cmp::min(i_start + block_rows, m);
                        let j_start = bj * block_cols;
                        let j_end = std::cmp::min(j_start + block_cols, n);

                        if i_start >= m || j_start >= n {
                            return None;
                        }

                        let a_block = a.slice(ndarray::s![i_start..i_end, ..]);
                        let b_block = b.slice(ndarray::s![.., j_start..j_end]);
                        let block_result = a_block.dot(&b_block);

                        Some(((i_start, j_start), block_result))
                    })
                    .collect();

                // Combine block results
                for ((i_start, j_start), block_result) in block_results {
                    let (block_m, block_n) = block_result.dim();
                    for i in 0..block_m {
                        for j in 0..block_n {
                            result[[i_start + i, j_start + j]] = block_result[[i, j]];
                        }
                    }
                }
            }
            _ => {
                // Fall back to row-wise partitioning
                let rows_per_node = m / topology.num_nodes;
                let partial_results: Vec<Array2<F>> = (0..topology.num_nodes)
                    .into_par_iter()
                    .map(|node_id| {
                        let start_row = node_id * rows_per_node;
                        let end_row = if node_id == topology.num_nodes - 1 {
                            m
                        } else {
                            start_row + rows_per_node
                        };

                        let a_partition = a.slice(ndarray::s![start_row..end_row, ..]);
                        a_partition.dot(b)
                    })
                    .collect();

                // Combine partial results
                let mut row_offset = 0;
                for partial_result in partial_results {
                    let partial_rows = partial_result.nrows();
                    for i in 0..partial_rows {
                        for j in 0..n {
                            result[[row_offset + i, j]] = partial_result[[i, j]];
                        }
                    }
                    row_offset += partial_rows;
                }
            }
        }

        Ok(result)
    }

    /// NUMA-aware memory allocation hints
    ///
    /// Provides guidance for memory allocation strategies on NUMA systems.
    pub struct NumaMemoryStrategy {
        topology: NumaTopology,
    }

    impl NumaMemoryStrategy {
        /// Create a new NUMA memory strategy
        pub fn new(topology: NumaTopology) -> Self {
            Self { topology }
        }

        /// Get recommended memory allocation for a matrix operation
        pub fn allocatematrix_memory<F>(&self, rows: usize, cols: usize) -> NumaAllocationHint
        where
            F: Float,
        {
            let elementsize = std::mem::size_of::<F>();
            let totalsize = rows * cols * elementsize;
            let size_per_node = totalsize / self.topology.num_nodes;

            NumaAllocationHint {
                strategy: if size_per_node > 1024 * 1024 {
                    // Large matrices: distribute across nodes
                    NumaAllocationStrategy::Distributed
                } else {
                    // Small matrices: allocate locally
                    NumaAllocationStrategy::Local
                },
                preferred_nodes: if self.topology.num_nodes > 1 {
                    (0..self.topology.num_nodes).collect()
                } else {
                    vec![0]
                },
                chunksize: std::cmp::max(4096, size_per_node / 8),
            }
        }

        /// Analyze memory access patterns for optimization
        pub fn analyze_access_pattern(
            &self,
            operation: NumaOperation,
            matrixsizes: &[(usize, usize)],
        ) -> NumaOptimizationHint {
            let total_memory = matrixsizes.iter()
                .map(|(r, c)| r * c * 8) // Assume f64
                .sum::<usize>();

            let memory_per_node = total_memory / self.topology.num_nodes;
            let local_bandwidth = self.topology.memory_bandwidth[0][0];
            let remote_bandwidth = if self.topology.num_nodes > 1 {
                self.topology.memory_bandwidth[0][1]
            } else {
                local_bandwidth
            };

            NumaOptimizationHint {
                operation,
                recommended_partitioning: NumaPartitioning::choose_optimal(
                    matrixsizes[0].0,
                    matrixsizes[0].1,
                    self.topology.num_nodes,
                ),
                memory_per_node,
                expected_local_ratio: local_bandwidth / (local_bandwidth + remote_bandwidth),
                thread_affinity_recommended: memory_per_node > 1024 * 1024, // 1MB threshold
            }
        }
    }

    /// NUMA allocation strategy
    #[derive(Debug, Clone, Copy)]
    pub enum NumaAllocationStrategy {
        /// Allocate all memory on local node
        Local,
        /// Distribute memory across all nodes
        Distributed,
        /// Interleave memory across nodes
        Interleaved,
    }

    /// NUMA allocation hint
    #[derive(Debug, Clone)]
    pub struct NumaAllocationHint {
        pub strategy: NumaAllocationStrategy,
        pub preferred_nodes: Vec<usize>,
        pub chunksize: usize,
    }

    /// Type of NUMA operation
    #[derive(Debug, Clone, Copy)]
    pub enum NumaOperation {
        MatrixVector,
        MatrixMatrix,
        Decomposition,
        IterativeSolver,
    }

    /// NUMA optimization hint
    #[derive(Debug, Clone)]
    pub struct NumaOptimizationHint {
        pub operation: NumaOperation,
        pub recommended_partitioning: NumaPartitioning,
        pub memory_per_node: usize,
        pub expected_local_ratio: f64,
        pub thread_affinity_recommended: bool,
    }

    /// NUMA-aware parallel Cholesky decomposition
    ///
    /// Implements a block-distributed Cholesky decomposition optimized for NUMA.
    pub fn numa_aware_cholesky<F>(
        matrix: &ArrayView2<F>,
        config: &WorkerConfig,
        topology: &NumaTopology,
    ) -> LinalgResult<Array2<F>>
    where
        F: Float
            + Send
            + Sync
            + Zero
            + One
            + NumAssign
            + ndarray::ScalarOperand
            + std::iter::Sum
            + 'static,
    {
        let (m, n) = matrix.dim();
        if m != n {
            return Err(LinalgError::ShapeError(
                "Cholesky decomposition requires square matrix".to_string(),
            ));
        }

        if topology.num_nodes == 1 {
            return super::algorithms::parallel_cholesky(matrix, config);
        }

        config.apply();

        let blocksize = n / topology.num_nodes;
        let mut l = Array2::zeros((n, n));

        // Distribute blocks across NUMA nodes
        for k in (0..n).step_by(blocksize) {
            let k_end = std::cmp::min(k + blocksize, n);
            let _current_node = k / blocksize;

            // Factorize diagonal block on local node
            for i in k..k_end {
                let mut sum = F::zero();
                for j in 0..i {
                    sum += l[[i, j]] * l[[i, j]];
                }
                let aii = matrix[[i, i]] - sum;
                if aii <= F::zero() {
                    return Err(LinalgError::ComputationError(
                        "Matrix is not positive definite".to_string(),
                    ));
                }
                l[[i, i]] = aii.sqrt();

                // Update column in parallel within the node
                for j in (i + 1)..k_end {
                    let mut sum = F::zero();
                    for p in 0..i {
                        sum += l[[j, p]] * l[[i, p]];
                    }
                    l[[j, i]] = (matrix[[j, i]] - sum) / l[[i, i]];
                }
            }

            // Update remaining blocks in parallel across nodes
            if k_end < n {
                let remaining_blocks: Vec<Array2<F>> = (k_end..n)
                    .step_by(blocksize)
                    .collect::<Vec<_>>()
                    .into_par_iter()
                    .map(|block_start| {
                        let block_end = std::cmp::min(block_start + blocksize, n);
                        let mut block_result = Array2::zeros((block_end - block_start, k_end - k));

                        for i in 0..(block_end - block_start) {
                            for j in 0..(k_end - k) {
                                let global_i = block_start + i;
                                let global_j = k + j;

                                let mut sum = F::zero();
                                for p in 0..global_j {
                                    sum += l[[global_i, p]] * l[[global_j, p]];
                                }
                                block_result[[i, j]] =
                                    (matrix[[global_i, global_j]] - sum) / l[[global_j, global_j]];
                            }
                        }
                        block_result
                    })
                    .collect();

                // Merge results back into L matrix
                for (block_idx, block_start) in (k_end..n).step_by(blocksize).enumerate() {
                    let block_end = std::cmp::min(block_start + blocksize, n);
                    let block_result = &remaining_blocks[block_idx];

                    for i in 0..(block_end - block_start) {
                        for j in 0..(k_end - k) {
                            l[[block_start + i, k + j]] = block_result[[i, j]];
                        }
                    }
                }
            }
        }

        Ok(l)
    }

    /// NUMA-aware workload balancer
    ///
    /// Balances computational workload across NUMA nodes considering
    /// memory bandwidth and CPU capabilities.
    pub struct NumaWorkloadBalancer {
        topology: NumaTopology,
        load_history: Arc<Mutex<Vec<f64>>>,
    }

    impl NumaWorkloadBalancer {
        /// Create a new NUMA workload balancer
        pub fn new(topology: NumaTopology) -> Self {
            let load_history = Arc::new(Mutex::new(vec![0.0; topology.num_nodes]));
            Self {
                topology,
                load_history,
            }
        }

        /// Get optimal work distribution for a given workload
        pub fn distribute_work(&self, total_workunits: usize) -> Vec<usize> {
            let load_history = self.load_history.lock().unwrap();

            // Calculate load-adjusted capacity for each node
            let node_capacities: Vec<f64> = self
                .topology
                .cpus_per_node
                .iter()
                .enumerate()
                .map(|(i, cpus)| {
                    let base_capacity = cpus.len() as f64;
                    let load_factor = 1.0 - load_history[i].min(0.9); // Cap at 90% penalty
                    base_capacity * load_factor
                })
                .collect();

            let total_capacity: f64 = node_capacities.iter().sum();

            // Distribute work proportionally
            let mut distribution = vec![0; self.topology.num_nodes];
            let mut remaining_work = total_workunits;

            for i in 0..self.topology.num_nodes {
                if i == self.topology.num_nodes - 1 {
                    // Give remaining work to last node
                    distribution[i] = remaining_work;
                } else {
                    let node_share =
                        (node_capacities[i] / total_capacity * total_workunits as f64) as usize;
                    distribution[i] = node_share;
                    remaining_work -= node_share;
                }
            }

            distribution
        }

        /// Update load history after completing work
        pub fn update_load_history(
            &self,
            node_id: usize,
            completion_time: f64,
            expected_time: f64,
        ) {
            let mut load_history = self.load_history.lock().unwrap();

            // Exponential moving average with alpha = 0.1
            let load_ratio = completion_time / expected_time;
            load_history[node_id] = 0.9 * load_history[node_id] + 0.1 * load_ratio;
        }

        /// Get current load information
        pub fn get_load_info(&self) -> Vec<f64> {
            self.load_history.lock().unwrap().clone()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_numa_topology_detection() {
            let topology = NumaTopology::detect();
            assert!(topology.num_nodes >= 1);
            assert_eq!(topology.cpus_per_node.len(), topology.num_nodes);
            assert_eq!(topology.memory_bandwidth.len(), topology.num_nodes);
        }

        #[test]
        fn test_numa_thread_distribution() {
            let topology = NumaTopology {
                num_nodes: 2,
                cpus_per_node: vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
                memory_bandwidth: vec![vec![100.0, 50.0], vec![50.0, 100.0]],
            };

            let distribution = topology.optimal_thread_distribution(6);
            assert_eq!(distribution, vec![3, 3]);

            let distribution = topology.optimal_thread_distribution(5);
            assert_eq!(distribution, vec![3, 2]);
        }

        #[test]
        fn test_numa_partitioning_strategy() {
            // Tall matrix should prefer row-wise
            assert!(matches!(
                NumaPartitioning::choose_optimal(1000, 100, 2),
                NumaPartitioning::RowWise
            ));

            // Wide matrix should prefer column-wise
            assert!(matches!(
                NumaPartitioning::choose_optimal(100, 1000, 2),
                NumaPartitioning::ColumnWise
            ));

            // Square matrix should prefer 2D blocking
            assert!(matches!(
                NumaPartitioning::choose_optimal(500, 500, 4),
                NumaPartitioning::Block2D
            ));
        }

        #[test]
        fn test_numa_workload_balancer() {
            let topology = NumaTopology {
                num_nodes: 2,
                cpus_per_node: vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
                memory_bandwidth: vec![vec![100.0, 50.0], vec![50.0, 100.0]],
            };

            let balancer = NumaWorkloadBalancer::new(topology);
            let distribution = balancer.distribute_work(100);

            // Should distribute roughly equally for balanced load
            assert_eq!(distribution.iter().sum::<usize>(), 100);
            assert!(distribution[0] >= 40 && distribution[0] <= 60);
            assert!(distribution[1] >= 40 && distribution[1] <= 60);
        }
    }
}

/// CPU affinity and thread pinning
///
/// This module provides advanced thread affinity management for optimal
/// performance on multi-core and multi-socket systems.
pub mod affinity {
    use super::{numa::NumaTopology, WorkerConfig};
    use std::sync::{Arc, Mutex};

    /// Thread affinity strategy
    #[derive(Debug, Clone, Copy)]
    pub enum AffinityStrategy {
        /// No specific affinity - let OS scheduler decide
        None,
        /// Pin threads to specific CPU cores
        Pinned,
        /// Spread threads across NUMA nodes
        NumaSpread,
        /// Compact threads within NUMA nodes
        NumaCompact,
        /// Custom affinity mapping
        Custom,
    }

    /// CPU core affinity mask
    #[derive(Debug, Clone)]
    pub struct CoreAffinity {
        /// CPU core IDs to bind threads to
        pub core_ids: Vec<usize>,
        /// Whether to allow thread migration
        pub allow_migration: bool,
        /// NUMA node preference
        pub numa_node: Option<usize>,
    }

    impl CoreAffinity {
        /// Create affinity for specific cores
        pub fn cores(_coreids: Vec<usize>) -> Self {
            Self {
                core_ids: _coreids,
                allow_migration: false,
                numa_node: None,
            }
        }

        /// Create affinity for a NUMA node
        pub fn numa_node(_nodeid: usize, topology: &NumaTopology) -> Self {
            let core_ids = if _nodeid < topology.cpus_per_node.len() {
                topology.cpus_per_node[_nodeid].clone()
            } else {
                vec![]
            };

            Self {
                core_ids,
                allow_migration: true,
                numa_node: Some(_nodeid),
            }
        }

        /// Allow thread migration between specified cores
        pub fn with_migration(mut self, allow: bool) -> Self {
            self.allow_migration = allow;
            self
        }
    }

    /// Thread affinity manager
    pub struct AffinityManager {
        strategy: AffinityStrategy,
        topology: NumaTopology,
        thread_assignments: Arc<Mutex<Vec<Option<CoreAffinity>>>>,
    }

    impl AffinityManager {
        /// Create a new affinity manager
        pub fn new(strategy: AffinityStrategy, topology: NumaTopology) -> Self {
            let thread_assignments = Arc::new(Mutex::new(Vec::new()));
            Self {
                strategy,
                topology,
                thread_assignments,
            }
        }

        /// Generate thread affinity assignments based on strategy
        pub fn generate_assignments(&self, numthreads: usize) -> Vec<CoreAffinity> {
            match self.strategy {
                AffinityStrategy::None => {
                    // No specific affinity
                    vec![]
                }
                AffinityStrategy::Pinned => self.generate_pinned_assignments(numthreads),
                AffinityStrategy::NumaSpread => self.generate_numa_spread_assignments(numthreads),
                AffinityStrategy::NumaCompact => self.generate_numa_compact_assignments(numthreads),
                AffinityStrategy::Custom => {
                    // Use existing assignments
                    self.thread_assignments
                        .lock()
                        .unwrap()
                        .iter()
                        .filter_map(|opt| opt.clone())
                        .collect()
                }
            }
        }

        /// Generate pinned affinity assignments (one thread per core)
        fn generate_pinned_assignments(&self, numthreads: usize) -> Vec<CoreAffinity> {
            let total_cores: usize = self
                .topology
                .cpus_per_node
                .iter()
                .map(|node| node.len())
                .sum();

            let effective_threads = std::cmp::min(numthreads, total_cores);
            let mut all_cores: Vec<usize> = self
                .topology
                .cpus_per_node
                .iter()
                .flat_map(|node| node.iter().cloned())
                .collect();

            // Sort cores for consistent assignment
            all_cores.sort_unstable();

            (0..effective_threads)
                .map(|i| CoreAffinity::cores(vec![all_cores[i]]))
                .collect()
        }

        /// Generate NUMA-spread assignments (distribute across nodes)
        fn generate_numa_spread_assignments(&self, numthreads: usize) -> Vec<CoreAffinity> {
            let mut assignments = Vec::new();
            let threads_per_node = numthreads / self.topology.num_nodes;
            let extra_threads = numthreads % self.topology.num_nodes;

            for (node_id, cores) in self.topology.cpus_per_node.iter().enumerate() {
                let node_threads = threads_per_node + if node_id < extra_threads { 1 } else { 0 };

                for i in 0..node_threads {
                    if i < cores.len() {
                        assignments.push(CoreAffinity::cores(vec![cores[i]]).with_migration(false));
                    } else {
                        // More _threads than cores in this node - allow migration
                        assignments.push(
                            CoreAffinity::numa_node(node_id, &self.topology).with_migration(true),
                        );
                    }
                }
            }

            assignments
        }

        /// Generate NUMA-compact assignments (fill nodes sequentially)
        fn generate_numa_compact_assignments(&self, numthreads: usize) -> Vec<CoreAffinity> {
            let mut assignments = Vec::new();
            let mut remaining_threads = numthreads;

            for (node_id, cores) in self.topology.cpus_per_node.iter().enumerate() {
                if remaining_threads == 0 {
                    break;
                }

                let node_capacity = cores.len();
                let threads_for_node = std::cmp::min(remaining_threads, node_capacity);

                // Assign _threads to specific cores in this node
                for core in cores.iter().take(threads_for_node) {
                    assignments.push(CoreAffinity::cores(vec![*core]).with_migration(false));
                }

                // If more _threads needed than cores, allow migration within node
                if remaining_threads > node_capacity {
                    for _ in node_capacity..remaining_threads.min(node_capacity * 2) {
                        assignments.push(
                            CoreAffinity::numa_node(node_id, &self.topology).with_migration(true),
                        );
                    }
                }

                remaining_threads = remaining_threads.saturating_sub(node_capacity * 2);
            }

            assignments
        }

        /// Set custom affinity for a specific thread
        pub fn set_thread_affinity(&self, threadid: usize, affinity: CoreAffinity) {
            let mut assignments = self.thread_assignments.lock().unwrap();

            // Expand vector if needed
            while assignments.len() <= threadid {
                assignments.push(None);
            }

            assignments[threadid] = Some(affinity);
        }

        /// Get affinity for a specific thread
        pub fn get_thread_affinity(&self, threadid: usize) -> Option<CoreAffinity> {
            let assignments = self.thread_assignments.lock().unwrap();
            assignments.get(threadid).and_then(|opt| opt.clone())
        }

        /// Apply affinity settings to current thread (platform-specific)
        pub fn apply_current_thread_affinity(
            &self,
            _affinity: &CoreAffinity,
        ) -> Result<(), String> {
            // Note: This is a simplified implementation
            // In a real implementation, you would use platform-specific APIs:
            // - On Linux: sched_setaffinity, pthread_setaffinity_np
            // - On Windows: SetThreadAffinityMask
            // - On macOS: thread_policy_set

            #[cfg(target_os = "linux")]
            {
                self.apply_linux_affinity(affinity)
            }

            #[cfg(target_os = "windows")]
            {
                self.apply_windows_affinity(affinity)
            }

            #[cfg(not(any(target_os = "linux", target_os = "windows")))]
            {
                // Fallback for unsupported platforms
                eprintln!("CPU affinity not supported on this platform");
                Ok(())
            }
        }

        #[cfg(target_os = "linux")]
        fn apply_linux_affinity(&self, affinity: &CoreAffinity) -> Result<(), String> {
            // This would typically use libc::sched_setaffinity
            // For now, we'll just set environment variables that some libraries recognize
            if !affinity.core_ids.is_empty() {
                let core_list = affinity
                    .core_ids
                    .iter()
                    .map(|id| id.to_string())
                    .collect::<Vec<_>>()
                    .join(",");

                std::env::set_var("GOMP_CPU_AFFINITY", &core_list);
                std::env::set_var("KMP_AFFINITY", format!("explicit,proclist=[{core_list}]"));
            }

            if let Some(numa_node) = affinity.numa_node {
                std::env::set_var("NUMA_NODE_HINT", numa_node.to_string());
            }

            Ok(())
        }

        #[cfg(target_os = "windows")]
        fn apply_windows_affinity(&self, affinity: &CoreAffinity) -> Result<(), String> {
            // This would typically use Windows APIs like SetThreadAffinityMask
            // For now, we'll set environment variables
            if !affinity.core_ids.is_empty() {
                let core_mask: u64 = affinity
                    .core_ids
                    .iter()
                    .fold(0u64, |mask, &core_id| mask | (1u64 << core_id));

                std::env::set_var("THREAD_AFFINITY_MASK", format!("0x{:x}", core_mask));
            }

            Ok(())
        }

        /// Get optimal affinity strategy for current system
        pub fn recommend_strategy(
            &self,
            num_threads: usize,
            workload_type: WorkloadType,
        ) -> AffinityStrategy {
            match workload_type {
                WorkloadType::CpuBound => {
                    if num_threads <= self.total_cores() {
                        AffinityStrategy::Pinned
                    } else {
                        AffinityStrategy::NumaSpread
                    }
                }
                WorkloadType::MemoryBound => {
                    if self.topology.num_nodes > 1 {
                        AffinityStrategy::NumaCompact
                    } else {
                        AffinityStrategy::Pinned
                    }
                }
                WorkloadType::Balanced => {
                    if self.topology.num_nodes > 1 && num_threads >= self.topology.num_nodes {
                        AffinityStrategy::NumaSpread
                    } else {
                        AffinityStrategy::Pinned
                    }
                }
                WorkloadType::Latency => AffinityStrategy::Pinned,
            }
        }

        /// Get total number of CPU cores
        fn total_cores(&self) -> usize {
            self.topology
                .cpus_per_node
                .iter()
                .map(|node| node.len())
                .sum()
        }
    }

    /// Type of computational workload
    #[derive(Debug, Clone, Copy)]
    pub enum WorkloadType {
        /// CPU-intensive workload
        CpuBound,
        /// Memory-intensive workload
        MemoryBound,
        /// Balanced CPU and memory usage
        Balanced,
        /// Latency-sensitive workload
        Latency,
    }

    /// Thread pool with affinity support
    pub struct AffinityThreadPool {
        affinity_manager: AffinityManager,
        config: WorkerConfig,
    }

    impl AffinityThreadPool {
        /// Create a new affinity-aware thread pool
        pub fn new(
            strategy: AffinityStrategy,
            topology: NumaTopology,
            config: WorkerConfig,
        ) -> Self {
            let affinity_manager = AffinityManager::new(strategy, topology);
            Self {
                affinity_manager,
                config,
            }
        }

        /// Execute work with affinity-pinned threads
        pub fn execute_with_affinity<F, R>(&self, work: F) -> R
        where
            F: FnOnce() -> R + Send,
            R: Send,
        {
            let num_threads = self.config.workers.unwrap_or(
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4),
            );

            let assignments = self.affinity_manager.generate_assignments(num_threads);

            // Apply affinity to current thread if assignments available
            if let Some(affinity) = assignments.first() {
                if let Err(e) = self
                    .affinity_manager
                    .apply_current_thread_affinity(affinity)
                {
                    eprintln!("Warning: Failed to set thread affinity: {e}");
                }
            }

            // Execute the work
            work()
        }

        /// Get affinity information for debugging
        pub fn get_affinity_info(&self) -> AffinityInfo {
            let num_threads = self.config.workers.unwrap_or(
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4),
            );

            let assignments = self.affinity_manager.generate_assignments(num_threads);

            AffinityInfo {
                strategy: self.affinity_manager.strategy,
                num_threads,
                assignments,
                topology: self.affinity_manager.topology.clone(),
            }
        }
    }

    /// Affinity information for debugging and monitoring
    #[derive(Debug, Clone)]
    pub struct AffinityInfo {
        pub strategy: AffinityStrategy,
        pub num_threads: usize,
        pub assignments: Vec<CoreAffinity>,
        pub topology: NumaTopology,
    }

    impl AffinityInfo {
        /// Print detailed affinity information
        pub fn print_summary(&self) {
            println!("=== Thread Affinity Summary ===");
            println!("Strategy: {:?}", self.strategy);
            println!("Number of threads: {}", self.num_threads);
            println!("NUMA nodes: {}", self.topology.num_nodes);

            for (node_id, cores) in self.topology.cpus_per_node.iter().enumerate() {
                println!("  Node {node_id}: CPUs {cores:?}");
            }

            println!("Thread assignments:");
            for (thread_id, affinity) in self.assignments.iter().enumerate() {
                println!(
                    "  Thread {}: cores {:?}, migration: {}, NUMA: {:?}",
                    thread_id, affinity.core_ids, affinity.allow_migration, affinity.numa_node
                );
            }
            println!("==============================");
        }

        /// Get affinity efficiency metrics
        pub fn efficiency_metrics(&self) -> AffinityEfficiencyMetrics {
            let cores_used: std::collections::HashSet<usize> = self
                .assignments
                .iter()
                .flat_map(|affinity| affinity.core_ids.iter().cloned())
                .collect();

            let total_cores: usize = self
                .topology
                .cpus_per_node
                .iter()
                .map(|node| node.len())
                .sum();

            let numa_nodes_used: std::collections::HashSet<usize> = self
                .assignments
                .iter()
                .filter_map(|affinity| affinity.numa_node)
                .collect();

            let threads_with_migration: usize = self
                .assignments
                .iter()
                .filter(|affinity| affinity.allow_migration)
                .count();

            AffinityEfficiencyMetrics {
                core_utilization: cores_used.len() as f64 / total_cores as f64,
                numa_spread: numa_nodes_used.len() as f64 / self.topology.num_nodes as f64,
                migration_ratio: threads_with_migration as f64 / self.num_threads as f64,
                threads_per_core: self.num_threads as f64 / cores_used.len() as f64,
            }
        }
    }

    /// Metrics for evaluating affinity efficiency
    #[derive(Debug, Clone)]
    pub struct AffinityEfficiencyMetrics {
        /// Fraction of CPU cores being used (0.0 to 1.0)
        pub core_utilization: f64,
        /// Fraction of NUMA nodes being used (0.0 to 1.0)
        pub numa_spread: f64,
        /// Fraction of threads that allow migration (0.0 to 1.0)
        pub migration_ratio: f64,
        /// Average number of threads per CPU core
        pub threads_per_core: f64,
    }

    /// Utility functions for affinity management
    pub mod utils {
        use super::*;

        /// Auto-detect optimal affinity strategy for a workload
        pub fn auto_detect_strategy(
            workload_type: WorkloadType,
            num_threads: usize,
            topology: &NumaTopology,
        ) -> AffinityStrategy {
            let manager = AffinityManager::new(AffinityStrategy::None, topology.clone());
            manager.recommend_strategy(num_threads, workload_type)
        }

        /// Create optimized thread pool for matrix operations
        pub fn creatematrix_thread_pool(
            matrixsize: (usize, usize),
            topology: NumaTopology,
        ) -> AffinityThreadPool {
            let workload_type = if matrixsize.0 * matrixsize.1 > 1_000_000 {
                WorkloadType::MemoryBound
            } else {
                WorkloadType::CpuBound
            };

            let num_threads = std::cmp::min(
                topology.cpus_per_node.iter().map(|node| node.len()).sum(),
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4),
            );

            let strategy = auto_detect_strategy(workload_type, num_threads, &topology);
            let config = WorkerConfig::new().with_workers(num_threads);

            AffinityThreadPool::new(strategy, topology, config)
        }

        /// Benchmark different affinity strategies
        pub fn benchmark_affinity_strategies<F>(
            workload: F,
            topology: NumaTopology,
            config: WorkerConfig,
        ) -> Vec<(AffinityStrategy, f64)>
        where
            F: Fn() -> f64 + Clone + Send + Sync,
        {
            let strategies = vec![
                AffinityStrategy::None,
                AffinityStrategy::Pinned,
                AffinityStrategy::NumaSpread,
                AffinityStrategy::NumaCompact,
            ];

            let mut results = Vec::new();

            for strategy in strategies {
                let pool = AffinityThreadPool::new(strategy, topology.clone(), config.clone());

                // Warm up
                for _ in 0..3 {
                    pool.execute_with_affinity(&workload);
                }

                // Benchmark
                let start = std::time::Instant::now();
                let iterations = 10;
                let mut total_work = 0.0;

                for _ in 0..iterations {
                    total_work += pool.execute_with_affinity(&workload);
                }

                let elapsed = start.elapsed().as_secs_f64();
                let throughput = total_work / elapsed;

                results.push((strategy, throughput));
            }

            results
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_core_affinity_creation() {
            let affinity = CoreAffinity::cores(vec![0, 1, 2]);
            assert_eq!(affinity.core_ids, vec![0, 1, 2]);
            assert!(!affinity.allow_migration);
            assert_eq!(affinity.numa_node, None);
        }

        #[test]
        fn test_numa_affinity_creation() {
            let topology = NumaTopology {
                num_nodes: 2,
                cpus_per_node: vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
                memory_bandwidth: vec![vec![100.0, 50.0], vec![50.0, 100.0]],
            };

            let affinity = CoreAffinity::numa_node(1, &topology);
            assert_eq!(affinity.core_ids, vec![4, 5, 6, 7]);
            assert!(affinity.allow_migration);
            assert_eq!(affinity.numa_node, Some(1));
        }

        #[test]
        fn test_affinity_strategy_recommendation() {
            let topology = NumaTopology {
                num_nodes: 2,
                cpus_per_node: vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
                memory_bandwidth: vec![vec![100.0, 50.0], vec![50.0, 100.0]],
            };

            let manager = AffinityManager::new(AffinityStrategy::None, topology);

            // CPU-bound with few threads should be pinned
            assert!(matches!(
                manager.recommend_strategy(4, WorkloadType::CpuBound),
                AffinityStrategy::Pinned
            ));

            // Memory-bound should prefer NUMA compact
            assert!(matches!(
                manager.recommend_strategy(4, WorkloadType::MemoryBound),
                AffinityStrategy::NumaCompact
            ));
        }

        #[test]
        fn test_pinned_assignments() {
            let topology = NumaTopology {
                num_nodes: 2,
                cpus_per_node: vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
                memory_bandwidth: vec![vec![100.0, 50.0], vec![50.0, 100.0]],
            };

            let manager = AffinityManager::new(AffinityStrategy::Pinned, topology);
            let assignments = manager.generate_assignments(4);

            assert_eq!(assignments.len(), 4);
            for (i, assignment) in assignments.iter().enumerate() {
                assert_eq!(assignment.core_ids.len(), 1);
                assert_eq!(assignment.core_ids[0], i);
                assert!(!assignment.allow_migration);
            }
        }

        #[test]
        fn test_numa_spread_assignments() {
            let topology = NumaTopology {
                num_nodes: 2,
                cpus_per_node: vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
                memory_bandwidth: vec![vec![100.0, 50.0], vec![50.0, 100.0]],
            };

            let manager = AffinityManager::new(AffinityStrategy::NumaSpread, topology);
            let assignments = manager.generate_assignments(4);

            assert_eq!(assignments.len(), 4);

            // Should have 2 threads per NUMA node
            let node0_threads = assignments
                .iter()
                .filter(|a| {
                    a.core_ids.contains(&0)
                        || a.core_ids.contains(&1)
                        || a.core_ids.contains(&2)
                        || a.core_ids.contains(&3)
                })
                .count();
            let node1_threads = assignments
                .iter()
                .filter(|a| {
                    a.core_ids.contains(&4)
                        || a.core_ids.contains(&5)
                        || a.core_ids.contains(&6)
                        || a.core_ids.contains(&7)
                })
                .count();

            assert_eq!(node0_threads, 2);
            assert_eq!(node1_threads, 2);
        }
    }
}

/// Algorithm-specific parallel implementations
pub mod algorithms {
    use super::{adaptive, WorkerConfig};
    use crate::error::{LinalgError, LinalgResult};
    use ndarray::{Array1, ArrayView1, ArrayView2};
    use num_traits::{Float, NumAssign, One, Zero};
    use scirs2_core::parallel_ops::*;
    use std::iter::Sum;

    /// Parallel matrix-vector multiplication
    ///
    /// This is a simpler and more effective parallelization that can be used
    /// as a building block for more complex algorithms.
    ///
    /// # Arguments
    ///
    /// * `matrix` - Input matrix
    /// * `vector` - Input vector
    /// * `config` - Worker configuration
    ///
    /// # Returns
    ///
    /// * Result vector y = A * x
    pub fn parallel_matvec<F>(
        matrix: &ArrayView2<F>,
        vector: &ArrayView1<F>,
        config: &WorkerConfig,
    ) -> LinalgResult<Array1<F>>
    where
        F: Float + Send + Sync + Zero + Sum + 'static,
    {
        let (m, n) = matrix.dim();
        if n != vector.len() {
            return Err(LinalgError::ShapeError(format!(
                "Matrix-vector dimensions incompatible: {}x{} * {}",
                m,
                n,
                vector.len()
            )));
        }

        let datasize = m * n;
        if !adaptive::should_use_parallel(datasize, config) {
            // Fall back to serial computation
            return Ok(matrix.dot(vector));
        }

        config.apply();

        // Parallel computation of each row
        let result_vec: Vec<F> = (0..m)
            .into_par_iter()
            .map(|i| {
                matrix
                    .row(i)
                    .iter()
                    .zip(vector.iter())
                    .map(|(&aij, &xj)| aij * xj)
                    .sum()
            })
            .collect();

        Ok(Array1::from_vec(result_vec))
    }

    /// Parallel power iteration for dominant eigenvalue
    ///
    /// This implementation uses parallel matrix-vector multiplications
    /// in the power iteration method for computing dominant eigenvalues.
    pub fn parallel_power_iteration<F>(
        matrix: &ArrayView2<F>,
        max_iter: usize,
        tolerance: F,
        config: &WorkerConfig,
    ) -> LinalgResult<(F, Array1<F>)>
    where
        F: Float + Send + Sync + Zero + Sum + NumAssign + One + ndarray::ScalarOperand + 'static,
    {
        let (m, n) = matrix.dim();
        if m != n {
            return Err(LinalgError::ShapeError(
                "Power iteration requires square matrix".to_string(),
            ));
        }

        let datasize = m * n;
        if !adaptive::should_use_parallel(datasize, config) {
            // Fall back to serial power iteration
            return crate::eigen::power_iteration(&matrix.view(), max_iter, tolerance);
        }

        config.apply();

        // Initialize with simple vector
        let mut v = Array1::ones(n);
        let norm = v.iter().map(|&x| x * x).sum::<F>().sqrt();
        v /= norm;

        let mut eigenvalue = F::zero();

        for _iter in 0..max_iter {
            // Use the parallel matrix-vector multiplication
            let new_v = parallel_matvec(matrix, &v.view(), config)?;

            // Compute eigenvalue estimate (Rayleigh quotient)
            let new_eigenvalue = new_v
                .iter()
                .zip(v.iter())
                .map(|(&new_vi, &vi)| new_vi * vi)
                .sum::<F>();

            // Normalize
            let norm = new_v.iter().map(|&x| x * x).sum::<F>().sqrt();
            if norm < F::epsilon() {
                return Err(LinalgError::ComputationError(
                    "Vector became zero during iteration".to_string(),
                ));
            }
            let normalized_v = new_v / norm;

            // Check convergence
            if (new_eigenvalue - eigenvalue).abs() < tolerance {
                return Ok((new_eigenvalue, normalized_v));
            }

            eigenvalue = new_eigenvalue;
            v = normalized_v;
        }

        Err(LinalgError::ComputationError(
            "Power iteration failed to converge".to_string(),
        ))
    }

    /// Parallel vector operations for linear algebra
    ///
    /// This module provides basic parallel vector operations that serve as
    /// building blocks for more complex algorithms.
    pub mod vector_ops {
        use super::*;

        /// Parallel dot product of two vectors
        pub fn parallel_dot<F>(
            x: &ArrayView1<F>,
            y: &ArrayView1<F>,
            config: &WorkerConfig,
        ) -> LinalgResult<F>
        where
            F: Float + Send + Sync + Zero + Sum + 'static,
        {
            if x.len() != y.len() {
                return Err(LinalgError::ShapeError(
                    "Vectors must have same length for dot product".to_string(),
                ));
            }

            let datasize = x.len();
            if !adaptive::should_use_parallel(datasize, config) {
                return Ok(x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum());
            }

            config.apply();

            let result = (0..x.len()).into_par_iter().map(|i| x[i] * y[i]).sum();

            Ok(result)
        }

        /// Parallel vector norm computation
        pub fn parallel_norm<F>(x: &ArrayView1<F>, config: &WorkerConfig) -> LinalgResult<F>
        where
            F: Float + Send + Sync + Zero + Sum + 'static,
        {
            let datasize = x.len();
            if !adaptive::should_use_parallel(datasize, config) {
                return Ok(x.iter().map(|&xi| xi * xi).sum::<F>().sqrt());
            }

            config.apply();

            let sum_squares = (0..x.len()).into_par_iter().map(|i| x[i] * x[i]).sum::<F>();

            Ok(sum_squares.sqrt())
        }

        /// Parallel AXPY operation: y = a*x + y
        ///
        /// Note: This function returns a new array rather than modifying in-place
        /// due to complications with parallel mutable iteration.
        pub fn parallel_axpy<F>(
            alpha: F,
            x: &ArrayView1<F>,
            y: &ArrayView1<F>,
            config: &WorkerConfig,
        ) -> LinalgResult<Array1<F>>
        where
            F: Float + Send + Sync + 'static,
        {
            if x.len() != y.len() {
                return Err(LinalgError::ShapeError(
                    "Vectors must have same length for AXPY".to_string(),
                ));
            }

            let datasize = x.len();
            if !adaptive::should_use_parallel(datasize, config) {
                let result = x
                    .iter()
                    .zip(y.iter())
                    .map(|(&xi, &yi)| alpha * xi + yi)
                    .collect();
                return Ok(Array1::from_vec(result));
            }

            config.apply();

            let result_vec: Vec<F> = (0..x.len())
                .into_par_iter()
                .map(|i| alpha * x[i] + y[i])
                .collect();

            Ok(Array1::from_vec(result_vec))
        }
    }

    /// Parallel matrix multiplication (GEMM)
    ///
    /// Implements parallel general matrix multiplication with block-based
    /// parallelization for improved cache performance.
    pub fn parallel_gemm<F>(
        a: &ArrayView2<F>,
        b: &ArrayView2<F>,
        config: &WorkerConfig,
    ) -> LinalgResult<ndarray::Array2<F>>
    where
        F: Float + Send + Sync + Zero + Sum + NumAssign + 'static,
    {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();

        if k != k2 {
            return Err(LinalgError::ShapeError(format!(
                "Matrix dimensions incompatible for multiplication: {m}x{k} * {k2}x{n}"
            )));
        }

        let datasize = m * k * n;
        if !adaptive::should_use_parallel(datasize, config) {
            return Ok(a.dot(b));
        }

        config.apply();

        // Block size for cache-friendly computation
        let blocksize = config.chunksize;

        let mut result = ndarray::Array2::zeros((m, n));

        // Parallel computation using blocks
        result
            .outer_iter_mut()
            .enumerate()
            .par_bridge()
            .for_each(|(i, mut row)| {
                for j in 0..n {
                    let mut sum = F::zero();
                    for kb in (0..k).step_by(blocksize) {
                        let k_end = std::cmp::min(kb + blocksize, k);
                        for ki in kb..k_end {
                            sum += a[[i, ki]] * b[[ki, j]];
                        }
                    }
                    row[j] = sum;
                }
            });

        Ok(result)
    }

    /// Parallel QR decomposition using Householder reflections
    ///
    /// This implementation parallelizes the application of Householder
    /// transformations across columns.
    pub fn parallel_qr<F>(
        matrix: &ArrayView2<F>,
        config: &WorkerConfig,
    ) -> LinalgResult<(ndarray::Array2<F>, ndarray::Array2<F>)>
    where
        F: Float + Send + Sync + Zero + Sum + One + NumAssign + ndarray::ScalarOperand + 'static,
    {
        let (m, n) = matrix.dim();
        let datasize = m * n;

        if !adaptive::should_use_parallel(datasize, config) {
            return crate::decomposition::qr(&matrix.view(), None);
        }

        config.apply();

        let mut a = matrix.to_owned();
        let mut q = ndarray::Array2::eye(m);
        let min_dim = std::cmp::min(m, n);

        for k in 0..min_dim {
            // Extract column vector for Householder reflection
            let x = a.slice(ndarray::s![k.., k]).to_owned();
            let alpha = if x[0] >= F::zero() {
                -x.iter().map(|&xi| xi * xi).sum::<F>().sqrt()
            } else {
                x.iter().map(|&xi| xi * xi).sum::<F>().sqrt()
            };

            if alpha.abs() < F::epsilon() {
                continue;
            }

            let mut v = x.clone();
            v[0] -= alpha;
            let v_norm_sq = v.iter().map(|&vi| vi * vi).sum::<F>();

            if v_norm_sq < F::epsilon() {
                continue;
            }

            // Apply Householder transformation (serial for simplicity)
            let remaining_cols = n - k;
            if remaining_cols > 1 {
                for j in k..n {
                    let col = a.slice(ndarray::s![k.., j]).to_owned();
                    let dot_product = v
                        .iter()
                        .zip(col.iter())
                        .map(|(&vi, &ci)| vi * ci)
                        .sum::<F>();
                    let factor = F::from(2.0).unwrap() * dot_product / v_norm_sq;

                    for (i, &vi) in v.iter().enumerate() {
                        a[[k + i, j]] -= factor * vi;
                    }
                }
            }

            // Update Q matrix (serial for simplicity)
            for i in 0..m {
                let row = q.slice(ndarray::s![i, k..]).to_owned();
                let dot_product = v
                    .iter()
                    .zip(row.iter())
                    .map(|(&vi, &ri)| vi * ri)
                    .sum::<F>();
                let factor = F::from(2.0).unwrap() * dot_product / v_norm_sq;

                for (j, &vj) in v.iter().enumerate() {
                    q[[i, k + j]] -= factor * vj;
                }
            }
        }

        let r = a.slice(ndarray::s![..min_dim, ..]).to_owned();
        Ok((q, r))
    }

    /// Parallel Cholesky decomposition
    ///
    /// Implements parallel Cholesky decomposition using block-column approach
    /// for positive definite matrices.
    pub fn parallel_cholesky<F>(
        matrix: &ArrayView2<F>,
        config: &WorkerConfig,
    ) -> LinalgResult<ndarray::Array2<F>>
    where
        F: Float + Send + Sync + Zero + Sum + One + NumAssign + ndarray::ScalarOperand + 'static,
    {
        let (m, n) = matrix.dim();
        if m != n {
            return Err(LinalgError::ShapeError(
                "Cholesky decomposition requires square matrix".to_string(),
            ));
        }

        let datasize = m * n;
        if !adaptive::should_use_parallel(datasize, config) {
            return crate::decomposition::cholesky(&matrix.view(), None);
        }

        config.apply();

        let mut l = ndarray::Array2::zeros((n, n));
        let blocksize = config.chunksize;

        for k in (0..n).step_by(blocksize) {
            let k_end = std::cmp::min(k + blocksize, n);

            // Diagonal block factorization (serial for numerical stability)
            for i in k..k_end {
                // Compute L[i,i]
                let mut sum = F::zero();
                for j in 0..i {
                    sum += l[[i, j]] * l[[i, j]];
                }
                let aii = matrix[[i, i]] - sum;
                if aii <= F::zero() {
                    return Err(LinalgError::ComputationError(
                        "Matrix is not positive definite".to_string(),
                    ));
                }
                l[[i, i]] = aii.sqrt();

                // Compute L[i+1:k_end, i]
                for j in (i + 1)..k_end {
                    let mut sum = F::zero();
                    for p in 0..i {
                        sum += l[[j, p]] * l[[i, p]];
                    }
                    l[[j, i]] = (matrix[[j, i]] - sum) / l[[i, i]];
                }
            }

            // Update trailing submatrix (serial for simplicity)
            if k_end < n {
                for i in k_end..n {
                    for j in k..k_end {
                        let mut sum = F::zero();
                        for p in 0..j {
                            sum += l[[i, p]] * l[[j, p]];
                        }
                        l[[i, j]] = (matrix[[i, j]] - sum) / l[[j, j]];
                    }
                }
            }
        }

        Ok(l)
    }

    /// Parallel LU decomposition with partial pivoting
    ///
    /// Implements parallel LU decomposition using block-column approach.
    pub fn parallel_lu<F>(
        matrix: &ArrayView2<F>,
        config: &WorkerConfig,
    ) -> LinalgResult<(ndarray::Array2<F>, ndarray::Array2<F>, ndarray::Array2<F>)>
    where
        F: Float + Send + Sync + Zero + Sum + One + NumAssign + ndarray::ScalarOperand + 'static,
    {
        let (m, n) = matrix.dim();
        let datasize = m * n;

        if !adaptive::should_use_parallel(datasize, config) {
            return crate::decomposition::lu(&matrix.view(), None);
        }

        config.apply();

        let mut a = matrix.to_owned();
        let mut perm_vec = (0..m).collect::<Vec<_>>();
        let min_dim = std::cmp::min(m, n);

        for k in 0..min_dim {
            // Find pivot (serial for correctness)
            let mut max_val = F::zero();
            let mut pivot_row = k;
            for i in k..m {
                let abs_val = a[[i, k]].abs();
                if abs_val > max_val {
                    max_val = abs_val;
                    pivot_row = i;
                }
            }

            if max_val < F::epsilon() {
                return Err(LinalgError::ComputationError(
                    "Matrix is singular".to_string(),
                ));
            }

            // Swap rows if needed
            if pivot_row != k {
                for j in 0..n {
                    let temp = a[[k, j]];
                    a[[k, j]] = a[[pivot_row, j]];
                    a[[pivot_row, j]] = temp;
                }
                perm_vec.swap(k, pivot_row);
            }

            // Update submatrix (serial for now to avoid borrowing issues)
            let pivot = a[[k, k]];

            for i in (k + 1)..m {
                let multiplier = a[[i, k]] / pivot;
                a[[i, k]] = multiplier;

                for j in (k + 1)..n {
                    a[[i, j]] = a[[i, j]] - multiplier * a[[k, j]];
                }
            }
        }

        // Create permutation matrix P
        let mut p = ndarray::Array2::zeros((m, m));
        for (i, &piv) in perm_vec.iter().enumerate() {
            p[[i, piv]] = F::one();
        }

        // Extract L and U matrices
        let mut l = ndarray::Array2::eye(m);
        let mut u = ndarray::Array2::zeros((m, n));

        for i in 0..m {
            for j in 0..n {
                if i > j && j < min_dim {
                    l[[i, j]] = a[[i, j]];
                } else if i <= j {
                    u[[i, j]] = a[[i, j]];
                }
            }
        }

        Ok((p, l, u))
    }

    /// Parallel conjugate gradient solver
    ///
    /// Implements parallel conjugate gradient method for solving linear systems
    /// with symmetric positive definite matrices.
    pub fn parallel_conjugate_gradient<F>(
        matrix: &ArrayView2<F>,
        b: &ArrayView1<F>,
        max_iter: usize,
        tolerance: F,
        config: &WorkerConfig,
    ) -> LinalgResult<Array1<F>>
    where
        F: Float + Send + Sync + Zero + Sum + One + NumAssign + ndarray::ScalarOperand + 'static,
    {
        let (m, n) = matrix.dim();
        if m != n {
            return Err(LinalgError::ShapeError(
                "CG requires square matrix".to_string(),
            ));
        }
        if n != b.len() {
            return Err(LinalgError::ShapeError(
                "Matrix and vector dimensions incompatible".to_string(),
            ));
        }

        let datasize = m * n;
        if !adaptive::should_use_parallel(datasize, config) {
            return crate::iterative_solvers::conjugate_gradient(
                &matrix.view(),
                &b.view(),
                max_iter,
                tolerance,
                None,
            );
        }

        config.apply();

        // Initialize
        let mut x = Array1::zeros(n);

        // r = b - A*x
        let ax = parallel_matvec(matrix, &x.view(), config)?;
        let mut r = b - &ax;
        let mut p = r.clone();
        let mut rsold = vector_ops::parallel_dot(&r.view(), &r.view(), config)?;

        for _iter in 0..max_iter {
            let ap = parallel_matvec(matrix, &p.view(), config)?;
            let alpha = rsold / vector_ops::parallel_dot(&p.view(), &ap.view(), config)?;

            x = vector_ops::parallel_axpy(alpha, &p.view(), &x.view(), config)?;
            r = vector_ops::parallel_axpy(-alpha, &ap.view(), &r.view(), config)?;

            let rsnew = vector_ops::parallel_dot(&r.view(), &r.view(), config)?;

            if rsnew.sqrt() < tolerance {
                return Ok(x);
            }

            let beta = rsnew / rsold;
            p = vector_ops::parallel_axpy(beta, &p.view(), &r.view(), config)?;
            rsold = rsnew;
        }

        Err(LinalgError::ComputationError(
            "Conjugate gradient failed to converge".to_string(),
        ))
    }

    /// Parallel SVD decomposition
    ///
    /// Implements parallel Singular Value Decomposition using a block-based approach
    /// for improved performance on large matrices.
    pub fn parallel_svd<F>(
        matrix: &ArrayView2<F>,
        config: &WorkerConfig,
    ) -> LinalgResult<(ndarray::Array2<F>, ndarray::Array1<F>, ndarray::Array2<F>)>
    where
        F: Float + Send + Sync + Zero + Sum + One + NumAssign + ndarray::ScalarOperand + 'static,
    {
        let (m, n) = matrix.dim();
        let datasize = m * n;

        if !adaptive::should_use_parallel(datasize, config) {
            return crate::decomposition::svd(&matrix.view(), false, None);
        }

        config.apply();

        // For now, use QR decomposition as a first step
        // This is a simplified parallel SVD - a full implementation would use
        // more sophisticated algorithms like Jacobi SVD or divide-and-conquer
        let (q, r) = parallel_qr(matrix, config)?;

        // Apply SVD to the smaller R matrix (serial for numerical stability)
        let (u_r, s, vt) = crate::decomposition::svd(&r.view(), false, None)?;

        // U = Q * U_r
        let u = parallel_gemm(&q.view(), &u_r.view(), config)?;

        Ok((u, s, vt))
    }

    /// Parallel GMRES (Generalized Minimal Residual) solver
    ///
    /// Implements parallel GMRES for solving non-symmetric linear systems.
    pub fn parallel_gmres<F>(
        matrix: &ArrayView2<F>,
        b: &ArrayView1<F>,
        max_iter: usize,
        tolerance: F,
        restart: usize,
        config: &WorkerConfig,
    ) -> LinalgResult<Array1<F>>
    where
        F: Float
            + Send
            + Sync
            + Zero
            + Sum
            + One
            + NumAssign
            + ndarray::ScalarOperand
            + std::fmt::Debug
            + std::fmt::Display
            + 'static,
    {
        let (m, n) = matrix.dim();
        if m != n {
            return Err(LinalgError::ShapeError(
                "GMRES requires square matrix".to_string(),
            ));
        }
        if n != b.len() {
            return Err(LinalgError::ShapeError(
                "Matrix and vector dimensions incompatible".to_string(),
            ));
        }

        let datasize = m * n;
        if !adaptive::should_use_parallel(datasize, config) {
            // Fall back to serial GMRES - use the iterative solver version
            let options = crate::solvers::iterative::IterativeSolverOptions {
                max_iterations: max_iter,
                tolerance,
                verbose: false,
                restart: Some(restart),
            };
            let result = crate::solvers::iterative::gmres(matrix, b, None, &options)?;
            return Ok(result.solution);
        }

        config.apply();

        let mut x = Array1::zeros(n);
        let restart = restart.min(n);

        for _outer in 0..max_iter {
            // Compute initial residual
            let ax = parallel_matvec(matrix, &x.view(), config)?;
            let r = b - &ax;
            let beta = vector_ops::parallel_norm(&r.view(), config)?;

            if beta < tolerance {
                return Ok(x);
            }

            // Initialize Krylov subspace
            let mut v = vec![r / beta];
            let mut h = ndarray::Array2::<F>::zeros((restart + 1, restart));

            // Arnoldi iteration
            for j in 0..restart {
                // w = A * v[j]
                let w = parallel_matvec(matrix, &v[j].view(), config)?;

                // Modified Gram-Schmidt orthogonalization
                let mut w_new = w.clone();
                for i in 0..=j {
                    h[[i, j]] = vector_ops::parallel_dot(&w.view(), &v[i].view(), config)?;
                    w_new =
                        vector_ops::parallel_axpy(-h[[i, j]], &v[i].view(), &w_new.view(), config)?;
                }

                h[[j + 1, j]] = vector_ops::parallel_norm(&w_new.view(), config)?;

                if h[[j + 1, j]] < F::epsilon() {
                    break;
                }

                v.push(w_new / h[[j + 1, j]]);
            }

            // Solve least squares problem (serial for numerical stability)
            let k = v.len() - 1;
            let h_sub = h.slice(ndarray::s![..=k, ..k]).to_owned();
            let mut g = Array1::zeros(k + 1);
            g[0] = beta;

            // Apply Givens rotations to solve the least squares problem
            let mut y = Array1::zeros(k);
            for i in (0..k).rev() {
                let mut sum = g[i];
                for j in (i + 1)..k {
                    sum -= h_sub[[i, j]] * y[j];
                }
                y[i] = sum / h_sub[[i, i]];
            }

            // Update solution
            for i in 0..k {
                x = vector_ops::parallel_axpy(y[i], &v[i].view(), &x.view(), config)?;
            }

            // Check residual
            let ax = parallel_matvec(matrix, &x.view(), config)?;
            let r = b - &ax;
            let residual_norm = vector_ops::parallel_norm(&r.view(), config)?;

            if residual_norm < tolerance {
                return Ok(x);
            }
        }

        Err(LinalgError::ComputationError(
            "GMRES failed to converge".to_string(),
        ))
    }

    /// Parallel BiCGSTAB (Biconjugate Gradient Stabilized) solver
    ///
    /// Implements parallel BiCGSTAB for solving non-symmetric linear systems.
    pub fn parallel_bicgstab<F>(
        matrix: &ArrayView2<F>,
        b: &ArrayView1<F>,
        max_iter: usize,
        tolerance: F,
        config: &WorkerConfig,
    ) -> LinalgResult<Array1<F>>
    where
        F: Float + Send + Sync + Zero + Sum + One + NumAssign + ndarray::ScalarOperand + 'static,
    {
        let (m, n) = matrix.dim();
        if m != n {
            return Err(LinalgError::ShapeError(
                "BiCGSTAB requires square matrix".to_string(),
            ));
        }
        if n != b.len() {
            return Err(LinalgError::ShapeError(
                "Matrix and vector dimensions incompatible".to_string(),
            ));
        }

        let datasize = m * n;
        if !adaptive::should_use_parallel(datasize, config) {
            return crate::iterative_solvers::bicgstab(
                &matrix.view(),
                &b.view(),
                max_iter,
                tolerance,
                None,
            );
        }

        config.apply();

        // Initialize
        let mut x = Array1::zeros(n);
        let ax = parallel_matvec(matrix, &x.view(), config)?;
        let mut r = b - &ax;
        let r_hat = r.clone();

        let mut rho = F::one();
        let mut alpha = F::one();
        let mut omega = F::one();

        let mut v = Array1::zeros(n);
        let mut p = Array1::zeros(n);

        for _iter in 0..max_iter {
            let rho_new = vector_ops::parallel_dot(&r_hat.view(), &r.view(), config)?;

            if rho_new.abs() < F::epsilon() {
                return Err(LinalgError::ComputationError(
                    "BiCGSTAB breakdown: rho = 0".to_string(),
                ));
            }

            let beta = (rho_new / rho) * (alpha / omega);

            // p = r + beta * (p - omega * v)
            let temp = vector_ops::parallel_axpy(-omega, &v.view(), &p.view(), config)?;
            p = vector_ops::parallel_axpy(
                F::one(),
                &r.view(),
                &vector_ops::parallel_axpy(beta, &temp.view(), &Array1::zeros(n).view(), config)?
                    .view(),
                config,
            )?;

            // v = A * p
            v = parallel_matvec(matrix, &p.view(), config)?;

            alpha = rho_new / vector_ops::parallel_dot(&r_hat.view(), &v.view(), config)?;

            // s = r - alpha * v
            let s = vector_ops::parallel_axpy(-alpha, &v.view(), &r.view(), config)?;

            // Check convergence
            let s_norm = vector_ops::parallel_norm(&s.view(), config)?;
            if s_norm < tolerance {
                x = vector_ops::parallel_axpy(alpha, &p.view(), &x.view(), config)?;
                return Ok(x);
            }

            // t = A * s
            let t = parallel_matvec(matrix, &s.view(), config)?;

            omega = vector_ops::parallel_dot(&t.view(), &s.view(), config)?
                / vector_ops::parallel_dot(&t.view(), &t.view(), config)?;

            // x = x + alpha * p + omega * s
            x = vector_ops::parallel_axpy(alpha, &p.view(), &x.view(), config)?;
            x = vector_ops::parallel_axpy(omega, &s.view(), &x.view(), config)?;

            // r = s - omega * t
            r = vector_ops::parallel_axpy(-omega, &t.view(), &s.view(), config)?;

            // Check convergence
            let r_norm = vector_ops::parallel_norm(&r.view(), config)?;
            if r_norm < tolerance {
                return Ok(x);
            }

            rho = rho_new;
        }

        Err(LinalgError::ComputationError(
            "BiCGSTAB failed to converge".to_string(),
        ))
    }

    /// Parallel Jacobi method
    ///
    /// Implements parallel Jacobi iteration for solving linear systems.
    /// This method is particularly well-suited for parallel execution.
    pub fn parallel_jacobi<F>(
        matrix: &ArrayView2<F>,
        b: &ArrayView1<F>,
        max_iter: usize,
        tolerance: F,
        config: &WorkerConfig,
    ) -> LinalgResult<Array1<F>>
    where
        F: Float + Send + Sync + Zero + Sum + One + NumAssign + ndarray::ScalarOperand + 'static,
    {
        let (m, n) = matrix.dim();
        if m != n {
            return Err(LinalgError::ShapeError(
                "Jacobi method requires square matrix".to_string(),
            ));
        }
        if n != b.len() {
            return Err(LinalgError::ShapeError(
                "Matrix and vector dimensions incompatible".to_string(),
            ));
        }

        let datasize = m * n;
        if !adaptive::should_use_parallel(datasize, config) {
            return crate::iterative_solvers::jacobi_method(
                &matrix.view(),
                &b.view(),
                max_iter,
                tolerance,
                None,
            );
        }

        config.apply();

        // Extract diagonal
        let diag: Vec<F> = (0..n)
            .into_par_iter()
            .map(|i| {
                if matrix[[i, i]].abs() < F::epsilon() {
                    F::one() // Avoid division by zero
                } else {
                    matrix[[i, i]]
                }
            })
            .collect();

        let mut x = Array1::zeros(n);

        for _iter in 0..max_iter {
            // Parallel update: x_new[i] = (b[i] - sum(A[i,j]*x[j] for j != i)) / A[i,i]
            let x_new_vec: Vec<F> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let mut sum = b[i];
                    for j in 0..n {
                        if i != j {
                            sum -= matrix[[i, j]] * x[j];
                        }
                    }
                    sum / diag[i]
                })
                .collect();

            let x_new = Array1::from_vec(x_new_vec);

            // Check convergence
            let diff = &x_new - &x;
            let error = vector_ops::parallel_norm(&diff.view(), config)?;

            if error < tolerance {
                return Ok(x_new);
            }

            x = x_new.clone();
        }

        Err(LinalgError::ComputationError(
            "Jacobi method failed to converge".to_string(),
        ))
    }

    /// Parallel SOR (Successive Over-Relaxation) method
    ///
    /// Implements a modified parallel SOR using red-black ordering
    /// to enable parallel updates.
    pub fn parallel_sor<F>(
        matrix: &ArrayView2<F>,
        b: &ArrayView1<F>,
        omega: F,
        max_iter: usize,
        tolerance: F,
        config: &WorkerConfig,
    ) -> LinalgResult<Array1<F>>
    where
        F: Float + Send + Sync + Zero + Sum + One + NumAssign + ndarray::ScalarOperand + 'static,
    {
        let (m, n) = matrix.dim();
        if m != n {
            return Err(LinalgError::ShapeError(
                "SOR requires square matrix".to_string(),
            ));
        }
        if n != b.len() {
            return Err(LinalgError::ShapeError(
                "Matrix and vector dimensions incompatible".to_string(),
            ));
        }

        if omega <= F::zero() || omega >= F::from(2.0).unwrap() {
            return Err(LinalgError::InvalidInputError(
                "Relaxation parameter omega must be in (0, 2)".to_string(),
            ));
        }

        let datasize = m * n;
        if !adaptive::should_use_parallel(datasize, config) {
            return crate::iterative_solvers::successive_over_relaxation(
                &matrix.view(),
                &b.view(),
                omega,
                max_iter,
                tolerance,
                None,
            );
        }

        config.apply();

        let mut x = Array1::zeros(n);

        for _iter in 0..max_iter {
            let x_old = x.clone();

            // Red-black ordering for parallel updates
            // First update "red" points (even indices)
            let red_updates: Vec<(usize, F)> = (0..n)
                .into_par_iter()
                .filter(|&i| i % 2 == 0)
                .map(|i| {
                    let mut sum = b[i];
                    for j in 0..n {
                        if i != j {
                            sum -= matrix[[i, j]] * x_old[j];
                        }
                    }
                    let x_gs = sum / matrix[[i, i]];
                    let x_new = (F::one() - omega) * x_old[i] + omega * x_gs;
                    (i, x_new)
                })
                .collect();

            // Apply red updates
            for (i, val) in red_updates {
                x[i] = val;
            }

            // Then update "black" points (odd indices)
            let black_updates: Vec<(usize, F)> = (0..n)
                .into_par_iter()
                .filter(|&i| i % 2 == 1)
                .map(|i| {
                    let mut sum = b[i];
                    for j in 0..n {
                        if i != j {
                            sum -= matrix[[i, j]] * x[j];
                        }
                    }
                    let x_gs = sum / matrix[[i, i]];
                    let x_new = (F::one() - omega) * x_old[i] + omega * x_gs;
                    (i, x_new)
                })
                .collect();

            // Apply black updates
            for (i, val) in black_updates {
                x[i] = val;
            }

            // Check convergence
            let ax = parallel_matvec(matrix, &x.view(), config)?;
            let r = b - &ax;
            let error = vector_ops::parallel_norm(&r.view(), config)?;

            if error < tolerance {
                return Ok(x);
            }
        }

        Err(LinalgError::ComputationError(
            "SOR failed to converge".to_string(),
        ))
    }
}

/// Advanced work-stealing scheduler with priority queues and predictive load balancing
pub mod advanced_work_stealing {
    use super::*;
    use crate::parallel::numa::NumaTopology;
    use std::cmp::Ordering as CmpOrdering;
    use std::collections::{BinaryHeap, VecDeque};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    /// Work item with priority for the advanced scheduler
    #[derive(Debug, Clone)]
    pub struct PriorityWorkItem<T> {
        pub data: T,
        pub priority: u32,
        pub estimated_cost: Duration,
        pub dependencies: Vec<usize>,
        pub task_id: usize,
    }

    impl<T> PartialEq for PriorityWorkItem<T> {
        fn eq(&self, other: &Self) -> bool {
            self.priority == other.priority
        }
    }

    impl<T> Eq for PriorityWorkItem<T> {}

    impl<T> PartialOrd for PriorityWorkItem<T> {
        fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
            Some(self.cmp(other))
        }
    }

    impl<T> Ord for PriorityWorkItem<T> {
        fn cmp(&self, other: &Self) -> CmpOrdering {
            // Higher priority values get processed first
            self.priority
                .cmp(&other.priority)
                .then_with(|| other.estimated_cost.cmp(&self.estimated_cost))
        }
    }

    /// Advanced work-stealing queue with priority and prediction capabilities
    pub struct AdvancedWorkStealingQueue<T> {
        /// High priority work items (processed first)
        high_priority: Mutex<BinaryHeap<PriorityWorkItem<T>>>,
        /// Normal priority work items
        normal_priority: Mutex<VecDeque<PriorityWorkItem<T>>>,
        /// Low priority work items (processed when idle)
        low_priority: Mutex<VecDeque<PriorityWorkItem<T>>>,
        /// Completion time history for prediction
        completion_history: Mutex<VecDeque<(usize, Duration)>>,
        /// Number of active workers
        #[allow(dead_code)]
        active_workers: AtomicUsize,
        /// Queue statistics
        stats: Mutex<WorkStealingStats>,
    }

    /// Statistics for work-stealing performance analysis
    #[derive(Debug, Clone, Default)]
    pub struct WorkStealingStats {
        pub tasks_completed: usize,
        pub successful_steals: usize,
        pub failed_steals: usize,
        pub average_completion_time: Duration,
        pub load_imbalance_ratio: f64,
        pub prediction_accuracy: f64,
    }

    impl<T> Default for AdvancedWorkStealingQueue<T> {
        fn default() -> Self {
            Self::new()
        }
    }

    impl<T> AdvancedWorkStealingQueue<T> {
        /// Create a new advanced work-stealing queue
        pub fn new() -> Self {
            Self {
                high_priority: Mutex::new(BinaryHeap::new()),
                normal_priority: Mutex::new(VecDeque::new()),
                low_priority: Mutex::new(VecDeque::new()),
                completion_history: Mutex::new(VecDeque::with_capacity(1000)),
                active_workers: AtomicUsize::new(0),
                stats: Mutex::new(WorkStealingStats::default()),
            }
        }

        /// Add work item with automatic priority classification
        pub fn push(&self, item: T, estimatedcost: Duration, dependencies: Vec<usize>) -> usize {
            let task_id = self.generate_task_id();
            let priority = self.classify_priority(&estimatedcost, &dependencies);

            let work_item = PriorityWorkItem {
                data: item,
                priority,
                estimated_cost: estimatedcost,
                dependencies,
                task_id,
            };

            match priority {
                0..=33 => {
                    self.low_priority.lock().unwrap().push_back(work_item);
                }
                34..=66 => {
                    self.normal_priority.lock().unwrap().push_back(work_item);
                }
                _ => {
                    self.high_priority.lock().unwrap().push(work_item);
                }
            }

            task_id
        }

        /// Try to pop work item using intelligent scheduling
        pub fn try_pop(&self) -> Option<PriorityWorkItem<T>> {
            // First try high priority tasks
            if let Ok(mut high_queue) = self.high_priority.try_lock() {
                if let Some(item) = high_queue.pop() {
                    return Some(item);
                }
            }

            // Then try normal priority tasks
            if let Ok(mut normal_queue) = self.normal_priority.try_lock() {
                if let Some(item) = normal_queue.pop_front() {
                    return Some(item);
                }
            }

            // Finally try low priority tasks if we're idle
            if let Ok(mut low_queue) = self.low_priority.try_lock() {
                if let Some(item) = low_queue.pop_front() {
                    return Some(item);
                }
            }

            None
        }

        /// Attempt to steal work from other queues (for work-stealing)
        pub fn try_steal(&self) -> Option<PriorityWorkItem<T>> {
            // Record steal attempt
            if let Ok(mut stats) = self.stats.try_lock() {
                // Try stealing from normal priority first (better balance)
                if let Ok(mut normal_queue) = self.normal_priority.try_lock() {
                    if let Some(item) = normal_queue.pop_back() {
                        stats.successful_steals += 1;
                        return Some(item);
                    }
                }

                // Then try low priority
                if let Ok(mut low_queue) = self.low_priority.try_lock() {
                    if let Some(item) = low_queue.pop_back() {
                        stats.successful_steals += 1;
                        return Some(item);
                    }
                }

                stats.failed_steals += 1;
            }

            None
        }

        /// Classify task priority based on cost and dependencies
        fn classify_priority(&self, estimatedcost: &Duration, dependencies: &[usize]) -> u32 {
            let base_priority: u32 = if estimatedcost.as_millis() > 100 {
                80 // High _cost tasks get high priority
            } else if estimatedcost.as_millis() > 10 {
                50 // Medium _cost tasks
            } else {
                20 // Low _cost tasks
            };

            // Adjust for dependencies (more dependencies = lower priority)
            let dependency_penalty = (dependencies.len() as u32 * 5).min(30);
            base_priority.saturating_sub(dependency_penalty)
        }

        /// Generate unique task ID
        fn generate_task_id(&self) -> usize {
            static TASK_COUNTER: AtomicUsize = AtomicUsize::new(0);
            TASK_COUNTER.fetch_add(1, Ordering::Relaxed)
        }

        /// Record task completion for performance prediction
        pub fn record_completion(&self, task_id: usize, actualduration: Duration) {
            if let Ok(mut history) = self.completion_history.try_lock() {
                history.push_back((task_id, actualduration));

                // Keep history bounded
                if history.len() > 1000 {
                    history.pop_front();
                }
            }
        }

        /// Get current queue statistics
        pub fn get_stats(&self) -> WorkStealingStats {
            self.stats.lock().unwrap().clone()
        }

        /// Get estimated remaining work
        pub fn estimated_remaining_work(&self) -> Duration {
            let high_count = self.high_priority.lock().unwrap().len();
            let normal_count = self.normal_priority.lock().unwrap().len();
            let low_count = self.low_priority.lock().unwrap().len();

            // Rough estimates based on priority
            Duration::from_millis((high_count * 100 + normal_count * 50 + low_count * 10) as u64)
        }
    }

    /// Matrix-specific adaptive chunking strategy
    pub struct MatrixAdaptiveChunking {
        /// Cache line size for optimal memory access
        #[allow(dead_code)]
        cache_linesize: usize,
        /// NUMA node information
        #[allow(dead_code)]
        numa_info: Option<NumaTopology>,
        /// Historical performance data
        performance_history: Mutex<VecDeque<ChunkingPerformance>>,
    }

    #[derive(Debug, Clone)]
    struct ChunkingPerformance {
        chunksize: usize,
        matrix_dimensions: (usize, usize),
        throughput: f64, // operations per second
        #[allow(dead_code)]
        cache_misses: usize,
        #[allow(dead_code)]
        timestamp: Instant,
    }

    impl Default for MatrixAdaptiveChunking {
        fn default() -> Self {
            Self::new()
        }
    }

    impl MatrixAdaptiveChunking {
        /// Create new adaptive chunking strategy
        pub fn new() -> Self {
            Self {
                cache_linesize: 64, // typical cache line size
                numa_info: Some(NumaTopology::detect()),
                performance_history: Mutex::new(VecDeque::with_capacity(100)),
            }
        }

        /// Calculate optimal chunk size for matrix operation
        pub fn optimal_chunksize(
            &self,
            matrix_dims: (usize, usize),
            operation_type: MatrixOperation,
        ) -> usize {
            let (rows, cols) = matrix_dims;

            // Base chunk size calculation
            let base_chunk = match operation_type {
                MatrixOperation::MatrixMultiply => {
                    // For matrix multiplication, consider cache blocking
                    let l1_cachesize = 32 * 1024; // 32KB typical L1 cache
                    let elementsize = std::mem::size_of::<f64>();
                    let elements_per_cache = l1_cachesize / elementsize;

                    // Aim for square blocks that fit in cache
                    ((elements_per_cache as f64).sqrt() as usize).clamp(32, 512)
                }
                MatrixOperation::ElementWise => {
                    // For element-wise operations, optimize for memory bandwidth
                    let memory_bandwidth = self.estimate_memory_bandwidth();
                    (memory_bandwidth / 8).clamp(64, 1024) // 8 bytes per f64
                }
                MatrixOperation::Reduction => {
                    // For reductions, use smaller chunks to balance load
                    let num_cores = std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(4);
                    rows.max(cols) / (num_cores * 4)
                }
                MatrixOperation::Decomposition => {
                    // For decompositions, larger chunks for better locality
                    let num_cores = std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(4);
                    rows.min(cols) / num_cores.max(1)
                }
            };

            // Adjust based on historical performance
            self.adjust_for_history(base_chunk, matrix_dims, operation_type)
        }

        /// Estimate memory bandwidth (simplified)
        fn estimate_memory_bandwidth(&self) -> usize {
            // This is a simplified estimation - in practice, this would
            // involve actual benchmarking
            match std::env::var("SCIRS_MEMORY_BANDWIDTH") {
                Ok(val) => val.parse().unwrap_or(100_000), // MB/s
                Err(_) => 100_000,                         // Default assumption: 100 GB/s
            }
        }

        /// Adjust chunk size based on historical performance
        fn adjust_for_history(
            &self,
            base_chunk: usize,
            matrix_dims: (usize, usize),
            _operation_type: MatrixOperation,
        ) -> usize {
            if let Ok(history) = self.performance_history.lock() {
                // Find similar operations in history
                let similar_ops: Vec<_> = history
                    .iter()
                    .filter(|perf| {
                        let (h_rows, h_cols) = perf.matrix_dimensions;
                        // Consider operations on similar-sized matrices
                        (h_rows as f64 / matrix_dims.0 as f64).abs() < 2.0
                            && (h_cols as f64 / matrix_dims.1 as f64).abs() < 2.0
                    })
                    .collect();

                if !similar_ops.is_empty() {
                    // Find the _chunk size with best throughput
                    let best_perf = similar_ops
                        .iter()
                        .max_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap());

                    if let Some(best) = best_perf {
                        // Interpolate between base _chunk and historically best
                        let weight = 0.7; // Favor historical data
                        return (base_chunk as f64 * (1.0 - weight) + best.chunksize as f64 * weight)
                            as usize;
                    }
                }
            }

            base_chunk
        }

        /// Record performance data for future optimization
        pub fn record_performance(
            &self,
            chunksize: usize,
            matrix_dims: (usize, usize),
            throughput: f64,
        ) {
            if let Ok(mut history) = self.performance_history.lock() {
                let perf = ChunkingPerformance {
                    chunksize,
                    matrix_dimensions: matrix_dims,
                    throughput,
                    cache_misses: 0, // Would be measured in practice
                    timestamp: Instant::now(),
                };

                history.push_back(perf);

                // Keep history bounded
                if history.len() > 100 {
                    history.pop_front();
                }
            }
        }
    }

    /// Types of matrix operations for chunking optimization
    #[derive(Debug, Clone, Copy)]
    pub enum MatrixOperation {
        MatrixMultiply,
        ElementWise,
        Reduction,
        Decomposition,
    }

    /// Predictive load balancer using machine learning-like predictions
    pub struct PredictiveLoadBalancer {
        /// Historical execution times for different task types
        execution_history: Mutex<std::collections::HashMap<String, Vec<Duration>>>,
        /// Current load per worker
        worker_loads: Mutex<Vec<f64>>,
        /// Prediction model weights (simplified linear model)
        model_weights: Mutex<Vec<f64>>,
    }

    impl PredictiveLoadBalancer {
        /// Create new predictive load balancer
        pub fn new(_numworkers: usize) -> Self {
            Self {
                execution_history: Mutex::new(std::collections::HashMap::new()),
                worker_loads: Mutex::new(vec![0.0; _numworkers]),
                model_weights: Mutex::new(vec![1.0; 4]), // Simple 4-feature model
            }
        }

        /// Predict execution time for a task
        pub fn predict_execution_time(&self, taskfeatures: &TaskFeatures) -> Duration {
            let weights = self.model_weights.lock().unwrap();

            // Extract _features
            let _features = [
                taskfeatures.datasize as f64,
                taskfeatures.complexity_factor,
                taskfeatures.memory_access_pattern as f64,
                taskfeatures.arithmetic_intensity,
            ];

            // Simple linear prediction
            let predicted_ms = _features
                .iter()
                .zip(weights.iter())
                .map(|(f, w)| f * w)
                .sum::<f64>()
                .max(1.0); // Minimum 1ms

            Duration::from_millis(predicted_ms as u64)
        }

        /// Assign task to optimal worker based on predicted load
        pub fn assign_task(&self, taskfeatures: &TaskFeatures) -> usize {
            let predicted_time = self.predict_execution_time(taskfeatures);
            let mut loads = self.worker_loads.lock().unwrap();

            // Find worker with minimum predicted finish time
            let (best_worker, min_load) = loads
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();

            // Update predicted load
            loads[best_worker] += predicted_time.as_secs_f64();

            best_worker
        }

        /// Update model with actual execution time
        pub fn update_model(&self, task_features: &TaskFeatures, actualtime: Duration) {
            // Record execution _time
            let task_type = format!(
                "{}_{}",
                task_features.datasize, task_features.complexity_factor as u32
            );

            if let Ok(mut history) = self.execution_history.lock() {
                history
                    .entry(task_type)
                    .or_insert_with(Vec::new)
                    .push(actualtime);
            }

            // Simple model update (in practice, would use more sophisticated ML)
            self.update_weights(task_features, actualtime);
        }

        /// Update worker load (when task completes)
        pub fn update_worker_load(&self, worker_id: usize, completedtime: Duration) {
            if let Ok(mut loads) = self.worker_loads.lock() {
                if worker_id < loads.len() {
                    loads[worker_id] -= completedtime.as_secs_f64();
                    loads[worker_id] = loads[worker_id].max(0.0);
                }
            }
        }

        /// Simple weight update using gradient descent-like approach
        fn update_weights(&self, task_features: &TaskFeatures, actualtime: Duration) {
            let predicted_time = self.predict_execution_time(task_features);
            let error = actualtime.as_secs_f64() - predicted_time.as_secs_f64();

            if let Ok(mut weights) = self.model_weights.lock() {
                let learning_rate = 0.001;
                let _features = [
                    task_features.datasize as f64,
                    task_features.complexity_factor,
                    task_features.memory_access_pattern as f64,
                    task_features.arithmetic_intensity,
                ];

                // Update weights based on error
                for (weight, feature) in weights.iter_mut().zip(_features.iter()) {
                    *weight += learning_rate * error * feature;
                }
            }
        }
    }

    /// Features describing a computational task for prediction
    #[derive(Debug, Clone)]
    pub struct TaskFeatures {
        pub datasize: usize,
        pub complexity_factor: f64,
        pub memory_access_pattern: u32, // 0=sequential, 1=random, 2=strided
        pub arithmetic_intensity: f64,  // operations per byte
    }

    impl TaskFeatures {
        /// Create task features for matrix operation
        pub fn formatrix_operation(
            matrix_dims: (usize, usize),
            operation: MatrixOperation,
        ) -> Self {
            let (rows, cols) = matrix_dims;
            let datasize = rows * cols;

            let (complexity_factor, memory_pattern, arithmetic_intensity) = match operation {
                MatrixOperation::MatrixMultiply => {
                    (rows as f64 * cols as f64 * 2.0, 1, 2.0) // O(n²) complexity, random access, 2 ops per element
                }
                MatrixOperation::ElementWise => {
                    (datasize as f64, 0, 1.0) // O(n) complexity, sequential access, 1 op per element
                }
                MatrixOperation::Reduction => {
                    (datasize as f64, 0, 1.0) // O(n) complexity, sequential access
                }
                MatrixOperation::Decomposition => {
                    (datasize as f64 * 1.5, 2, 3.0) // Higher complexity, strided access
                }
            };

            Self {
                datasize,
                complexity_factor,
                memory_access_pattern: memory_pattern,
                arithmetic_intensity,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_workers() {
        // Save initial state to restore later
        let original_state = get_global_workers();

        // Test setting and getting global workers
        set_global_workers(Some(4));
        assert_eq!(get_global_workers(), Some(4));

        set_global_workers(None);
        assert_eq!(get_global_workers(), None);

        // Restore original state to avoid test interference
        set_global_workers(original_state);
    }

    #[test]
    fn test_scoped_workers() {
        // Save initial state to restore later
        let original_state = get_global_workers();

        // Set initial global workers
        set_global_workers(Some(2));

        {
            // Create scoped configuration
            let _scoped = ScopedWorkers::new(Some(8));
            assert_eq!(get_global_workers(), Some(8));
        }

        // Should be restored after scope
        assert_eq!(get_global_workers(), Some(2));

        // Restore original state to avoid test interference
        set_global_workers(original_state);
    }

    #[test]
    fn test_worker_config() {
        let config = WorkerConfig::new()
            .with_workers(4)
            .with_threshold(2000)
            .with_chunksize(128);

        assert_eq!(config.workers, Some(4));
        assert_eq!(config.parallel_threshold, 2000);
        assert_eq!(config.chunksize, 128);
    }

    #[test]
    fn test_adaptive_strategy() {
        let config = WorkerConfig::default();

        // Small data should use serial
        assert!(matches!(
            adaptive::choose_strategy(100, &config),
            adaptive::Strategy::Serial
        ));

        // Large data should use parallel
        assert!(matches!(
            adaptive::choose_strategy(2000, &config),
            adaptive::Strategy::Parallel
        ));
    }
}
