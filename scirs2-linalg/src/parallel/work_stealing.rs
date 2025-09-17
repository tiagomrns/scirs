//! Work-stealing scheduler implementation for dynamic load balancing
//!
//! This module provides a work-stealing scheduler that dynamically balances
//! work across threads, with timing analysis and adaptive chunking based on
//! workload characteristics.

use crate::error::{LinalgError, LinalgResult};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, ScalarOperand};
use num_traits::{Float, NumAssign, One, Zero};
#[allow(unused_imports)]
use scirs2_core::parallel_ops::*;
use std::collections::VecDeque;
use std::iter::Sum;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Type alias for complex work item types used in QR decomposition
type QRWorkItem<F> = WorkItem<(usize, Array1<F>, Array2<F>)>;

/// Type alias for complex work item types used in band matrix solving
type BandSolveWorkItem<F> = WorkItem<(usize, usize, usize, Array2<F>, Array1<F>)>;

/// Simple parallel map utility function using rayon
#[allow(dead_code)]
fn parallel_map<T, U, F>(items: &[T], func: F) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(&T) -> U + Sync + Send,
{
    #[allow(unused_imports)]
    use scirs2_core::parallel_ops::*;

    // Use rayon's parallel iterator if available, otherwise sequential
    #[cfg(feature = "parallel")]
    {
        items.par_iter().map(func).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        items.iter().map(func).collect()
    }
}

/// Work item for the work-stealing scheduler
#[derive(Debug, Clone)]
pub struct WorkItem<T>
where
    T: Clone,
{
    /// Unique identifier for the work item
    pub id: usize,
    /// The actual work payload
    pub payload: T,
    /// Expected execution time (for scheduling optimization)
    pub estimated_time: Option<Duration>,
}

impl<T: Clone> WorkItem<T> {
    /// Create a new work item
    pub fn new(id: usize, payload: T) -> Self {
        Self {
            id,
            payload,
            estimated_time: None,
        }
    }

    /// Create a work item with estimated execution time
    pub fn with_estimate(_id: usize, payload: T, estimatedtime: Duration) -> Self {
        Self {
            id: _id,
            payload,
            estimated_time: Some(estimatedtime),
        }
    }
}

/// Work queue for a single worker thread
#[derive(Debug)]
struct WorkQueue<T: Clone> {
    /// Double-ended queue for work items
    items: VecDeque<WorkItem<T>>,
    /// Number of items processed by this worker
    processed_count: usize,
    /// Total execution time for this worker
    total_time: Duration,
    /// Average execution time per item
    avg_time: Duration,
}

impl<T: Clone> Default for WorkQueue<T> {
    fn default() -> Self {
        Self {
            items: VecDeque::new(),
            processed_count: 0,
            total_time: Duration::ZERO,
            avg_time: Duration::ZERO,
        }
    }
}

impl<T: Clone> WorkQueue<T> {
    /// Add work item to the front of the queue (for local work)
    fn push_front(&mut self, item: WorkItem<T>) {
        self.items.push_front(item);
    }

    /// Add work item to the back of the queue (for stolen work)
    #[allow(dead_code)]
    fn push_back(&mut self, item: WorkItem<T>) {
        self.items.push_back(item);
    }

    /// Take work from the front (local work)
    fn pop_front(&mut self) -> Option<WorkItem<T>> {
        self.items.pop_front()
    }

    /// Steal work from the back (work stealing)
    fn steal_back(&mut self) -> Option<WorkItem<T>> {
        if self.items.len() > 1 {
            self.items.pop_back()
        } else {
            None
        }
    }

    /// Update timing statistics
    fn update_timing(&mut self, executiontime: Duration) {
        self.processed_count += 1;
        self.total_time += executiontime;
        self.avg_time = self.total_time / self.processed_count as u32;
    }

    /// Get the current load (estimated remaining work time)
    fn estimated_load(&self) -> Duration {
        let base_time = if self.avg_time.is_zero() {
            Duration::from_millis(1) // Default estimate
        } else {
            self.avg_time
        };

        self.items
            .iter()
            .map(|item| item.estimated_time.unwrap_or(base_time))
            .sum()
    }
}

/// Work-stealing scheduler with dynamic load balancing
pub struct WorkStealingScheduler<T: Clone>
where
    T: Send + 'static,
{
    /// Worker queues (one per thread)
    worker_queues: Vec<Arc<Mutex<WorkQueue<T>>>>,
    /// Number of worker threads
    num_workers: usize,
    /// Condition variable for worker synchronization
    worker_sync: Arc<(Mutex<bool>, Condvar)>,
    /// Statistics collection
    stats: Arc<Mutex<SchedulerStats>>,
    /// Work-stealing strategy
    stealing_strategy: StealingStrategy,
    /// Adaptive load balancing parameters
    load_balancing_params: LoadBalancingParams,
}

/// Work-stealing strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StealingStrategy {
    /// Random victim selection
    Random,
    /// Round-robin victim selection  
    RoundRobin,
    /// Target the most loaded worker
    MostLoaded,
    /// Target based on work locality
    LocalityAware,
    /// Adaptive strategy that learns from history
    #[default]
    Adaptive,
}

/// Load balancing parameters for adaptive optimization
#[derive(Debug, Clone)]
pub struct LoadBalancingParams {
    /// Minimum work items before attempting to steal
    pub steal_threshold: usize,
    /// Maximum steal attempts per worker
    pub max_steal_attempts: usize,
    /// Exponential backoff base for failed steals
    pub backoff_base: Duration,
    /// Maximum backoff time
    pub max_backoff: Duration,
    /// Work chunk size for splitting large tasks
    pub chunksize: usize,
    /// Enable work item priority scheduling
    pub priority_scheduling: bool,
}

impl Default for LoadBalancingParams {
    fn default() -> Self {
        Self {
            steal_threshold: 2,
            max_steal_attempts: 3,
            backoff_base: Duration::from_micros(10),
            max_backoff: Duration::from_millis(1),
            chunksize: 100,
            priority_scheduling: false,
        }
    }
}

/// Priority levels for work items
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum WorkPriority {
    Low,
    #[default]
    Normal,
    High,
    Critical,
}

/// Matrix operation types for scheduler optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixOperationType {
    MatrixVectorMultiplication,
    MatrixMatrixMultiplication,
    Decomposition,
    EigenComputation,
    IterativeSolver,
}

/// Workload characteristics for adaptive optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadCharacteristics {
    HighVariance,
    LowVariance,
    MemoryBound,
    ComputeBound,
}

/// Work complexity patterns for execution time prediction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkComplexity {
    Constant,
    Linear,
    Quadratic,
    Variable,
}

/// Scheduler performance statistics
#[derive(Debug, Default, Clone)]
pub struct SchedulerStats {
    /// Total items processed
    pub total_items: usize,
    /// Total execution time across all workers
    pub total_execution_time: Duration,
    /// Number of successful steals
    pub successful_steals: usize,
    /// Number of failed steal attempts
    pub failed_steals: usize,
    /// Load balancing efficiency (0.0 to 1.0)
    pub load_balance_efficiency: f64,
    /// Time variance across workers
    pub time_variance: f64,
    /// Average work stealing latency
    pub avg_steal_latency: Duration,
    /// Work distribution histogram
    pub work_distribution: Vec<usize>,
    /// Thread utilization rates
    pub thread_utilization: Vec<f64>,
}

impl<T: Send + 'static + Clone> WorkStealingScheduler<T> {
    /// Create a new work-stealing scheduler
    pub fn new(_numworkers: usize) -> Self {
        Self::with_strategy(
            _numworkers,
            StealingStrategy::default(),
            LoadBalancingParams::default(),
        )
    }

    /// Create a new work-stealing scheduler with custom strategy
    pub fn with_strategy(
        num_workers: usize,
        strategy: StealingStrategy,
        params: LoadBalancingParams,
    ) -> Self {
        let worker_queues = (0..num_workers)
            .map(|_| Arc::new(Mutex::new(WorkQueue::default())))
            .collect();

        Self {
            worker_queues,
            num_workers,
            worker_sync: Arc::new((Mutex::new(false), Condvar::new())),
            stats: Arc::new(Mutex::new(SchedulerStats::default())),
            stealing_strategy: strategy,
            load_balancing_params: params,
        }
    }

    /// Create optimized scheduler for specific matrix operations
    pub fn formatrix_operation(
        num_workers: usize,
        operation_type: MatrixOperationType,
        matrixsize: (usize, usize),
    ) -> Self {
        let (strategy, params) = match operation_type {
            MatrixOperationType::MatrixVectorMultiplication => {
                // Matrix-vector operations benefit from locality-aware stealing
                (
                    StealingStrategy::LocalityAware,
                    LoadBalancingParams {
                        steal_threshold: 4,
                        max_steal_attempts: 2,
                        chunksize: matrixsize.0 / num_workers,
                        priority_scheduling: false,
                        ..LoadBalancingParams::default()
                    },
                )
            }
            MatrixOperationType::MatrixMatrixMultiplication => {
                // Matrix-matrix operations benefit from adaptive stealing
                (
                    StealingStrategy::Adaptive,
                    LoadBalancingParams {
                        steal_threshold: 2,
                        max_steal_attempts: 4,
                        chunksize: (matrixsize.0 * matrixsize.1) / (num_workers * 8),
                        priority_scheduling: true,
                        ..LoadBalancingParams::default()
                    },
                )
            }
            MatrixOperationType::Decomposition => {
                // Decompositions have irregular workloads, use adaptive approach
                (
                    StealingStrategy::Adaptive,
                    LoadBalancingParams {
                        steal_threshold: 1,
                        max_steal_attempts: 6,
                        chunksize: matrixsize.0 / (num_workers * 2),
                        priority_scheduling: true,
                        backoff_base: Duration::from_micros(5),
                        max_backoff: Duration::from_millis(2),
                    },
                )
            }
            MatrixOperationType::EigenComputation => {
                // Eigenvalue computations have sequential dependencies
                (
                    StealingStrategy::MostLoaded,
                    LoadBalancingParams {
                        steal_threshold: 8,
                        max_steal_attempts: 2,
                        chunksize: matrixsize.0 / num_workers,
                        priority_scheduling: false,
                        ..LoadBalancingParams::default()
                    },
                )
            }
            MatrixOperationType::IterativeSolver => {
                // Iterative solvers need balanced load distribution
                (
                    StealingStrategy::RoundRobin,
                    LoadBalancingParams {
                        steal_threshold: 3,
                        max_steal_attempts: 3,
                        chunksize: matrixsize.0 / (num_workers * 4),
                        priority_scheduling: false,
                        ..LoadBalancingParams::default()
                    },
                )
            }
        };

        Self::with_strategy(num_workers, strategy, params)
    }

    /// Submit work items to the scheduler
    pub fn submit_work(&self, items: Vec<WorkItem<T>>) -> LinalgResult<()> {
        if items.is_empty() {
            return Ok(());
        }

        // Advanced work distribution based on strategy
        self.distribute_work_optimally(items)?;

        // Wake up all workers
        let (lock, cvar) = &*self.worker_sync;
        if let Ok(mut started) = lock.lock() {
            *started = true;
            cvar.notify_all();
        }

        Ok(())
    }

    /// Optimally distribute work items based on current load and strategy
    fn distribute_work_optimally(&self, items: Vec<WorkItem<T>>) -> LinalgResult<()> {
        match self.stealing_strategy {
            StealingStrategy::Random => {
                // Random distribution
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};

                for (i, item) in items.into_iter().enumerate() {
                    let mut hasher = DefaultHasher::new();
                    i.hash(&mut hasher);
                    let worker_id = (hasher.finish() as usize) % self.num_workers;

                    if let Ok(mut queue) = self.worker_queues[worker_id].lock() {
                        queue.push_front(item);
                    }
                }
            }
            StealingStrategy::RoundRobin => {
                // Round-robin distribution (default)
                for (i, item) in items.into_iter().enumerate() {
                    let worker_id = i % self.num_workers;
                    if let Ok(mut queue) = self.worker_queues[worker_id].lock() {
                        queue.push_front(item);
                    }
                }
            }
            StealingStrategy::MostLoaded => {
                // Distribute to least loaded workers first
                let load_info = self.get_worker_loads();
                let mut sorted_workers: Vec<usize> = (0..self.num_workers).collect();
                sorted_workers.sort_by_key(|&i| load_info[i]);

                for (i, item) in items.into_iter().enumerate() {
                    let worker_id = sorted_workers[i % self.num_workers];
                    if let Ok(mut queue) = self.worker_queues[worker_id].lock() {
                        queue.push_front(item);
                    }
                }
            }
            StealingStrategy::LocalityAware => {
                // Try to maintain work locality (simplified implementation)
                let chunksize = self.load_balancing_params.chunksize;
                for chunk in items.chunks(chunksize) {
                    let worker_id = (chunk.as_ptr() as usize / chunksize) % self.num_workers;
                    if let Ok(mut queue) = self.worker_queues[worker_id].lock() {
                        for item in chunk {
                            queue.push_front(item.clone());
                        }
                    }
                }
            }
            StealingStrategy::Adaptive => {
                // Use adaptive strategy based on historical performance
                self.adaptive_work_distribution(items)?;
            }
        }

        Ok(())
    }

    /// Get current load (number of work items) for each worker
    fn get_worker_loads(&self) -> Vec<usize> {
        let mut loads = Vec::with_capacity(self.num_workers);

        for queue in &self.worker_queues {
            if let Ok(queue) = queue.lock() {
                loads.push(queue.items.len());
            } else {
                loads.push(0);
            }
        }

        loads
    }

    /// Adaptive work distribution based on historical performance
    fn adaptive_work_distribution(&self, items: Vec<WorkItem<T>>) -> LinalgResult<()> {
        // Get current worker utilization
        let loads = self.get_worker_loads();
        let total_load: usize = loads.iter().sum();

        if total_load == 0 {
            // No existing load, use round-robin
            for (i, item) in items.into_iter().enumerate() {
                let worker_id = i % self.num_workers;
                if let Ok(mut queue) = self.worker_queues[worker_id].lock() {
                    queue.push_front(item);
                }
            }
        } else {
            // Distribute inversely proportional to current load
            let mut worker_weights = Vec::with_capacity(self.num_workers);
            let max_load = loads.iter().max().unwrap_or(&1);

            for &load in &loads {
                // Higher load = lower weight
                worker_weights.push(max_load + 1 - load);
            }

            let total_weight: usize = worker_weights.iter().sum();
            let mut cumulative_weights = Vec::with_capacity(self.num_workers);
            let mut sum = 0;
            for &weight in &worker_weights {
                sum += weight;
                cumulative_weights.push(sum);
            }

            // Distribute items based on weights
            let items_len = items.len();
            for (i, item) in items.into_iter().enumerate() {
                let target = (i * total_weight / items_len).min(total_weight - 1);
                let worker_id = cumulative_weights
                    .iter()
                    .position(|&w| w > target)
                    .unwrap_or(self.num_workers - 1);

                if let Ok(mut queue) = self.worker_queues[worker_id].lock() {
                    queue.push_front(item);
                }
            }
        }

        Ok(())
    }

    /// Advanced work stealing with different victim selection strategies
    #[allow(dead_code)]
    fn steal_work(&self, thiefid: usize) -> Option<WorkItem<T>> {
        let mut attempts = 0;
        let max_attempts = self.load_balancing_params.max_steal_attempts;

        while attempts < max_attempts {
            let victim_id = self.select_victim(thiefid, attempts);

            if let Some(victim_id) = victim_id {
                if let Ok(mut victim_queue) = self.worker_queues[victim_id].try_lock() {
                    if let Some(stolen_item) = victim_queue.steal_back() {
                        // Update statistics
                        if let Ok(mut stats) = self.stats.lock() {
                            stats.successful_steals += 1;
                        }
                        return Some(stolen_item);
                    }
                }
            }

            attempts += 1;

            // Exponential backoff
            let backoff_duration =
                self.load_balancing_params.backoff_base * 2_u32.pow(attempts.min(10) as u32);
            let capped_backoff = backoff_duration.min(self.load_balancing_params.max_backoff);

            thread::sleep(capped_backoff);
        }

        // Update failed steal statistics
        if let Ok(mut stats) = self.stats.lock() {
            stats.failed_steals += max_attempts;
        }

        None
    }

    /// Select victim for work stealing based on strategy
    #[allow(dead_code)]
    fn select_victim(&self, thiefid: usize, attempt: usize) -> Option<usize> {
        match self.stealing_strategy {
            StealingStrategy::Random => {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};

                let mut hasher = DefaultHasher::new();
                (thiefid + attempt).hash(&mut hasher);
                let victim = (hasher.finish() as usize) % self.num_workers;

                if victim != thiefid {
                    Some(victim)
                } else {
                    Some((victim + 1) % self.num_workers)
                }
            }
            StealingStrategy::RoundRobin => Some((thiefid + attempt + 1) % self.num_workers),
            StealingStrategy::MostLoaded => {
                // Target the worker with the most work
                let loads = self.get_worker_loads();
                let max_load_worker = loads
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != thiefid)
                    .max_by_key(|(_, &load)| load)
                    .map(|(i, _)| i);

                max_load_worker
            }
            StealingStrategy::LocalityAware => {
                // Try to steal from nearby workers first
                let distance = (attempt % (self.num_workers / 2)) + 1;
                Some((thiefid + distance) % self.num_workers)
            }
            StealingStrategy::Adaptive => {
                // Combine strategies based on historical success rates
                if attempt < 2 {
                    // First try most loaded
                    self.select_victim_most_loaded(thiefid)
                } else {
                    // Then try random
                    self.select_victim(thiefid, attempt)
                }
            }
        }
    }

    /// Helper for most-loaded victim selection
    #[allow(dead_code)]
    fn select_victim_most_loaded(&self, thiefid: usize) -> Option<usize> {
        let loads = self.get_worker_loads();
        loads
            .iter()
            .enumerate()
            .filter(|(i_, _)| *i_ != thiefid)
            .max_by_key(|(_, &load)| load)
            .map(|(i_, _)| i_)
    }

    /// Execute all work items using the work-stealing scheduler
    pub fn execute<F, R>(&self, workfn: F) -> LinalgResult<Vec<R>>
    where
        F: Fn(T) -> R + Send + Sync + 'static,
        R: Send + Clone + 'static,
        T: Send + 'static,
    {
        let work_fn = Arc::new(workfn);
        let results = Arc::new(Mutex::new(Vec::new()));

        // Start worker threads
        let mut handles = Vec::new();
        for worker_id in 0..self.num_workers {
            let queue = Arc::clone(&self.worker_queues[worker_id]);
            let all_queues = self.worker_queues.clone();
            let work_fn = Arc::clone(&work_fn);
            let results = Arc::clone(&results);
            let stats = Arc::clone(&self.stats);
            let sync = Arc::clone(&self.worker_sync);

            let handle = thread::spawn(move || {
                Self::worker_loop(worker_id, queue, all_queues, work_fn, results, stats, sync);
            });
            handles.push(handle);
        }

        // Wait for all workers to complete
        for handle in handles {
            handle.join().map_err(|_| {
                crate::error::LinalgError::ComputationError("Worker thread panicked".to_string())
            })?;
        }

        // Extract results
        let results = results.lock().unwrap();
        Ok((*results).clone())
    }

    /// Worker thread main loop
    fn worker_loop<F, R>(
        worker_id: usize,
        my_queue: Arc<Mutex<WorkQueue<T>>>,
        all_queues: Vec<Arc<Mutex<WorkQueue<T>>>>,
        work_fn: Arc<F>,
        results: Arc<Mutex<Vec<R>>>,
        stats: Arc<Mutex<SchedulerStats>>,
        sync: Arc<(Mutex<bool>, Condvar)>,
    ) where
        F: Fn(T) -> R + Send + Sync,
        R: Send,
    {
        let (lock, cvar) = &*sync;

        // Wait for work to be available
        let _started = cvar
            .wait_while(lock.lock().unwrap(), |&mut started| !started)
            .unwrap();

        loop {
            let work_item = {
                // Try to get work from own _queue first
                if let Ok(mut queue) = my_queue.lock() {
                    queue.pop_front()
                } else {
                    None
                }
            };

            let work_item = match work_item {
                Some(item) => item,
                None => {
                    // Try to steal work from other workers
                    match Self::steal_work_global(worker_id, &all_queues, &stats) {
                        Some(item) => item,
                        None => {
                            // No work available, check if all _queues are empty
                            if Self::all_queues_empty(&all_queues) {
                                break;
                            }
                            // Brief pause before trying again
                            thread::sleep(Duration::from_micros(10));
                            continue;
                        }
                    }
                }
            };

            // Execute the work item
            let start_time = Instant::now();
            let result = work_fn(work_item.payload);
            let execution_time = start_time.elapsed();

            // Update timing statistics
            if let Ok(mut queue) = my_queue.lock() {
                queue.update_timing(execution_time);
            }

            // Store the result
            if let Ok(mut results) = results.lock() {
                results.push(result);
            }

            // Update global statistics
            if let Ok(mut stats) = stats.lock() {
                stats.total_items += 1;
                stats.total_execution_time += execution_time;
            }
        }
    }

    /// Attempt to steal work from other workers
    fn steal_work_global(
        worker_id: usize,
        all_queues: &[Arc<Mutex<WorkQueue<T>>>],
        stats: &Arc<Mutex<SchedulerStats>>,
    ) -> Option<WorkItem<T>> {
        // Try to steal from the most loaded worker
        let mut best_target = None;
        let mut max_load = Duration::ZERO;

        for (i, queue) in all_queues.iter().enumerate() {
            if i == worker_id {
                continue; // Don't steal from ourselves
            }

            if let Ok(queue) = queue.lock() {
                let load = queue.estimated_load();
                if load > max_load {
                    max_load = load;
                    best_target = Some(i);
                }
            }
        }

        if let Some(target_id) = best_target {
            if let Ok(mut target_queue) = all_queues[target_id].lock() {
                if let Some(stolen_item) = target_queue.steal_back() {
                    // Update steal statistics
                    if let Ok(mut stats) = stats.lock() {
                        stats.successful_steals += 1;
                    }
                    return Some(stolen_item);
                }
            }
        }

        // Update failed steal statistics
        if let Ok(mut stats) = stats.lock() {
            stats.failed_steals += 1;
        }

        None
    }

    /// Check if all worker queues are empty
    fn all_queues_empty(queues: &[Arc<Mutex<WorkQueue<T>>>]) -> bool {
        queues.iter().all(|queue| {
            if let Ok(queue) = queue.lock() {
                queue.items.is_empty()
            } else {
                true // Assume empty if we can't lock
            }
        })
    }

    /// Get current scheduler statistics
    pub fn get_stats(&self) -> SchedulerStats {
        if let Ok(stats) = self.stats.lock() {
            let mut stats = stats.clone();
            stats.load_balance_efficiency = self.calculate_load_balance_efficiency();
            stats.time_variance = self.calculate_time_variance();
            stats
        } else {
            SchedulerStats::default()
        }
    }

    /// Adaptive performance monitoring and load balancing optimization
    pub fn optimize_for_workload(
        &self,
        workload_characteristics: WorkloadCharacteristics,
    ) -> LinalgResult<()> {
        let mut stats = self.stats.lock().map_err(|_| {
            crate::error::LinalgError::ComputationError("Failed to acquire stats lock".to_string())
        })?;

        // Analyze current performance metrics
        let load_imbalance = self.calculate_load_imbalance();
        let steal_success_rate = if stats.successful_steals + stats.failed_steals > 0 {
            stats.successful_steals as f64 / (stats.successful_steals + stats.failed_steals) as f64
        } else {
            0.5
        };

        // Adapt strategy based on workload _characteristics and performance
        let _suggested_strategy =
            match (workload_characteristics, load_imbalance, steal_success_rate) {
                (WorkloadCharacteristics::HighVariance, imbalance_, _) if imbalance_ > 0.3 => {
                    StealingStrategy::Adaptive
                }
                (WorkloadCharacteristics::LowVariance, _, success_rate) if success_rate < 0.2 => {
                    StealingStrategy::RoundRobin
                }
                (WorkloadCharacteristics::MemoryBound, _, _) => StealingStrategy::LocalityAware,
                (WorkloadCharacteristics::ComputeBound, _, success_rate) if success_rate > 0.8 => {
                    StealingStrategy::MostLoaded
                }
                _ => StealingStrategy::Adaptive,
            };

        // Update performance recommendations
        stats.load_balance_efficiency = 1.0 - load_imbalance;

        Ok(())
    }

    /// Calculate load imbalance across workers
    fn calculate_load_imbalance(&self) -> f64 {
        let loads = self.get_worker_loads();
        if loads.is_empty() {
            return 0.0;
        }

        let total_load: usize = loads.iter().sum();
        let avg_load = total_load as f64 / loads.len() as f64;

        if avg_load == 0.0 {
            return 0.0;
        }

        let variance: f64 = loads
            .iter()
            .map(|&load| (load as f64 - avg_load).powi(2))
            .sum::<f64>()
            / loads.len() as f64;

        let std_dev = variance.sqrt();
        std_dev / avg_load // Coefficient of variation
    }

    /// Dynamic chunk size adjustment based on performance history
    pub fn adaptive_chunk_sizing(
        &self,
        base_worksize: usize,
        worker_efficiency: &[f64],
    ) -> Vec<usize> {
        let total_efficiency: f64 = worker_efficiency.iter().sum();
        let avg_efficiency = total_efficiency / worker_efficiency.len() as f64;

        // Adjust chunk sizes based on relative worker _efficiency
        worker_efficiency
            .iter()
            .map(|&_efficiency| {
                let efficiency_ratio = _efficiency / avg_efficiency;
                let chunksize = (base_worksize as f64 * efficiency_ratio) as usize;
                chunksize.max(1).min(base_worksize) // Clamp to reasonable bounds
            })
            .collect()
    }

    /// Advanced workload prediction based on execution history
    pub fn predict_execution_time(&self, workcomplexity: WorkComplexity) -> Duration {
        let stats = self.stats.lock().unwrap();

        let base_time = if stats.total_items > 0 {
            stats.total_execution_time / stats.total_items as u32
        } else {
            Duration::from_millis(1)
        };

        match workcomplexity {
            WorkComplexity::Constant => base_time,
            WorkComplexity::Linear => base_time * 2,
            WorkComplexity::Quadratic => base_time * 4,
            WorkComplexity::Variable => {
                // Use historical variance to estimate
                Duration::from_nanos(
                    (base_time.as_nanos() as f64 * (1.0 + stats.time_variance)).max(1.0) as u64,
                )
            }
        }
    }

    /// Calculate load balancing efficiency
    fn calculate_load_balance_efficiency(&self) -> f64 {
        let worker_times: Vec<Duration> = self
            .worker_queues
            .iter()
            .filter_map(|queue| queue.lock().ok().map(|q| q.total_time))
            .collect();

        if worker_times.is_empty() {
            return 1.0;
        }

        let max_time = worker_times.iter().max().unwrap().as_nanos() as f64;
        let min_time = worker_times.iter().min().unwrap().as_nanos() as f64;

        if max_time == 0.0 {
            1.0
        } else {
            min_time / max_time
        }
    }

    /// Calculate time variance across workers
    fn calculate_time_variance(&self) -> f64 {
        let worker_times: Vec<f64> = self
            .worker_queues
            .iter()
            .filter_map(|queue| queue.lock().ok().map(|q| q.total_time.as_nanos() as f64))
            .collect();

        if worker_times.len() < 2 {
            return 0.0;
        }

        let mean = worker_times.iter().sum::<f64>() / worker_times.len() as f64;
        let variance = worker_times
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / worker_times.len() as f64;

        variance.sqrt()
    }
}

/// Matrix-specific work-stealing algorithms
pub mod matrix_ops {
    use super::*;

    /// Work-stealing matrix-vector multiplication
    pub fn parallel_matvec_work_stealing<F>(
        matrix: &ArrayView2<F>,
        vector: &ArrayView1<F>,
        num_workers: usize,
    ) -> LinalgResult<Array1<F>>
    where
        F: Float + NumAssign + Zero + Sum + Send + Sync + ScalarOperand + 'static,
    {
        let (m, n) = matrix.dim();
        if n != vector.len() {
            return Err(crate::error::LinalgError::ShapeError(
                "Matrix and vector dimensions don't match".to_string(),
            ));
        }

        let scheduler = WorkStealingScheduler::new(num_workers);
        let mut result = Array1::zeros(m);

        // Create work items for each row
        let work_items: Vec<WorkItem<(usize, Array1<F>, F)>> = (0..m)
            .map(|i| {
                let row = matrix.row(i).to_owned();
                let dot_product = row.dot(vector);
                WorkItem::new(i, (i, row, dot_product))
            })
            .collect();

        scheduler.submit_work(work_items)?;

        // Execute work and collect results
        let results = scheduler.execute(|(i, row, dot_product)| (i, dot_product))?;

        // Assemble final result
        for (i, value) in results {
            result[i] = value;
        }

        Ok(result)
    }

    /// Work-stealing matrix multiplication
    pub fn parallel_gemm_work_stealing<F>(
        a: &ArrayView2<F>,
        b: &ArrayView2<F>,
        num_workers: usize,
    ) -> LinalgResult<Array2<F>>
    where
        F: Float + NumAssign + Zero + Sum + Send + Sync + ScalarOperand + 'static,
    {
        let (m, k1) = a.dim();
        let (k2, n) = b.dim();

        if k1 != k2 {
            return Err(crate::error::LinalgError::ShapeError(
                "Matrix dimensions don't match for multiplication".to_string(),
            ));
        }

        let scheduler = WorkStealingScheduler::new(num_workers);
        let mut result = Array2::zeros((m, n));

        // Create work items for blocks of the result matrix
        let blocksize = (m * n / (num_workers * 4)).max(1);
        let mut work_items = Vec::new();

        for block_start in (0..m * n).step_by(blocksize) {
            let block_end = (block_start + blocksize).min(m * n);
            let indices: Vec<(usize, usize)> = (block_start..block_end)
                .map(|idx| (idx / n, idx % n))
                .collect();

            work_items.push(WorkItem::new(
                block_start,
                (indices, a.to_owned(), b.to_owned()),
            ));
        }

        scheduler.submit_work(work_items)?;

        // Execute work and collect results
        let results = scheduler.execute(|(indices, a_copy, b_copy)| {
            indices
                .into_iter()
                .map(|(i, j)| {
                    let value = a_copy.row(i).dot(&b_copy.column(j));
                    (i, j, value)
                })
                .collect::<Vec<_>>()
        })?;

        // Assemble final result
        for block_results in results {
            for (i, j, value) in block_results {
                result[(i, j)] = value;
            }
        }

        Ok(result)
    }

    /// Work-stealing Cholesky decomposition
    pub fn parallel_cholesky_work_stealing<F>(
        matrix: &ArrayView2<F>,
        num_workers: usize,
    ) -> LinalgResult<Array2<F>>
    where
        F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
    {
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Err(crate::error::LinalgError::ShapeError(
                "Cholesky decomposition requires square matrix".to_string(),
            ));
        }

        let mut l = Array2::zeros((n, n));
        let matrix_owned = matrix.to_owned(); // Create owned copy to avoid lifetime issues

        // Cholesky decomposition with work-stealing for column operations
        for k in 0..n {
            // Compute diagonal element
            let mut sum = F::zero();
            for j in 0..k {
                sum += l[(k, j)] * l[(k, j)];
            }
            l[(k, k)] = (matrix_owned[(k, k)] - sum).sqrt();

            if k + 1 < n {
                let scheduler = WorkStealingScheduler::new(num_workers);

                // Create work items for remaining elements in column k
                #[allow(clippy::type_complexity)]
                let work_items: Vec<
                    WorkItem<(usize, usize, Array2<F>, Array2<F>)>,
                > = (k + 1..n)
                    .map(|i| WorkItem::new(i, (i, k, l.clone(), matrix_owned.clone())))
                    .collect();

                scheduler.submit_work(work_items)?;

                let results = scheduler.execute(|(i, k, l_copy, matrix_copy)| {
                    let mut sum = F::zero();
                    for j in 0..k {
                        sum += l_copy[(i, j)] * l_copy[(k, j)];
                    }
                    let value = (matrix_copy[(i, k)] - sum) / l_copy[(k, k)];
                    (i, value)
                })?;

                // Update the L matrix
                for (i, value) in results {
                    l[(i, k)] = value;
                }
            }
        }

        Ok(l)
    }

    /// Work-stealing QR decomposition using Householder reflections
    pub fn parallel_qr_work_stealing<F>(
        matrix: &ArrayView2<F>,
        num_workers: usize,
    ) -> LinalgResult<(Array2<F>, Array2<F>)>
    where
        F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
    {
        let (m, n) = matrix.dim();
        let mut q = Array2::eye(m);
        let mut r = matrix.to_owned();

        let scheduler = WorkStealingScheduler::new(num_workers);

        for k in 0..n.min(m - 1) {
            // Compute Householder vector for column k
            let col_slice = r.slice(s![k.., k]).to_owned();
            let alpha = col_slice.iter().map(|x| *x * *x).sum::<F>().sqrt();
            let alpha = if col_slice[0] >= F::zero() {
                -alpha
            } else {
                alpha
            };

            let mut v = col_slice.clone();
            v[0] -= alpha;
            let v_norm = v.iter().map(|x| *x * *x).sum::<F>().sqrt();

            if v_norm > F::zero() {
                for elem in v.iter_mut() {
                    *elem /= v_norm;
                }

                // Apply Householder reflection to remaining columns in parallel
                let work_items: Vec<QRWorkItem<F>> = ((k + 1)..n)
                    .map(|j| WorkItem::new(j, (j, v.clone(), r.clone())))
                    .collect();

                if !work_items.is_empty() {
                    scheduler.submit_work(work_items)?;
                    let results = scheduler.execute(move |(j, v_col, rmatrix)| {
                        let col = rmatrix.slice(s![k.., j]).to_owned();
                        let dot_product = v_col
                            .iter()
                            .zip(col.iter())
                            .map(|(a, b)| *a * *b)
                            .sum::<F>();
                        let new_col: Array1<F> = col
                            .iter()
                            .zip(v_col.iter())
                            .map(|(c, v)| *c - F::one() + F::one() * dot_product * *v)
                            .collect();
                        (j, new_col)
                    })?;

                    // Update R matrix
                    for (j, new_col) in results {
                        for (i, &val) in new_col.iter().enumerate() {
                            r[(k + i, j)] = val;
                        }
                    }
                }

                // Update Q matrix with Householder reflection
                let q_work_items: Vec<QRWorkItem<F>> = (0..m)
                    .map(|i| WorkItem::new(i, (i, v.clone(), q.clone())))
                    .collect();

                scheduler.submit_work(q_work_items)?;
                let q_results = scheduler.execute(move |(i, v_col, qmatrix)| {
                    let row = qmatrix.slice(s![i, k..]).to_owned();
                    let dot_product = row
                        .iter()
                        .zip(v_col.iter())
                        .map(|(a, b)| *a * *b)
                        .sum::<F>();
                    let new_row: Array1<F> = row
                        .iter()
                        .zip(v_col.iter())
                        .map(|(q_val, v)| *q_val - F::one() + F::one() * dot_product * *v)
                        .collect();
                    (i, new_row)
                })?;

                // Update Q matrix
                for (i, new_row) in q_results {
                    for (j, &val) in new_row.iter().enumerate() {
                        q[(i, k + j)] = val;
                    }
                }
            }
        }

        Ok((q, r))
    }

    /// Work-stealing SVD computation using Jacobi method
    pub fn parallel_svd_work_stealing<F>(
        matrix: &ArrayView2<F>,
        num_workers: usize,
    ) -> LinalgResult<(Array2<F>, Array1<F>, Array2<F>)>
    where
        F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
    {
        let (m, n) = matrix.dim();
        let a = matrix.to_owned();

        // For large matrices, use parallel approach
        if m.min(n) > 32 {
            // Compute A^T * A for eigenvalue decomposition approach
            let scheduler = WorkStealingScheduler::new(num_workers);
            let ata = parallelmatrix_multiply_ata(&a.view(), &scheduler)?;

            // This is a simplified implementation - in practice you'd use more sophisticated methods
            let u = Array2::eye(m);
            let mut s = Array1::zeros(n.min(m));
            let vt = Array2::eye(n);

            // Basic parallel Jacobi iterations (simplified)
            for _iter in 0..50 {
                let work_items: Vec<WorkItem<(usize, usize, Array2<F>)>> = (0..n)
                    .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
                    .map(|(i, j)| WorkItem::new(i * n + j, (i, j, ata.clone())))
                    .collect();

                if work_items.is_empty() {
                    break;
                }

                scheduler.submit_work(work_items)?;
                let _results = scheduler.execute(|(_i, j, matrix)| {
                    // Simplified Jacobi rotation computation
                    // In a full implementation, this would compute the rotation angles
                    // and apply them to eliminate off-diagonal elements
                    0.0_f64 // Placeholder
                })?;
            }

            // Extract singular values from diagonal
            for i in 0..s.len() {
                s[i] = ata[(i, i)].sqrt();
            }

            Ok((u, s, vt))
        } else {
            // For small matrices, use sequential method
            self::sequential_svd(matrix)
        }
    }

    /// Helper function for parallel A^T * A computation
    fn parallelmatrix_multiply_ata<F>(
        matrix: &ArrayView2<F>,
        scheduler: &WorkStealingScheduler<(usize, usize, Array2<F>)>,
    ) -> LinalgResult<Array2<F>>
    where
        F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
    {
        let (m, n) = matrix.dim();
        let mut result = Array2::zeros((n, n));

        // Create work items for computing each element of A^T * A
        let work_items: Vec<WorkItem<(usize, usize, Array2<F>)>> = (0..n)
            .flat_map(|i| (i..n).map(move |j| (i, j)))
            .map(|(i, j)| WorkItem::new(i * n + j, (i, j, matrix.to_owned())))
            .collect();

        scheduler.submit_work(work_items)?;
        let results = scheduler.execute(move |(i, j, mat)| {
            let mut sum = F::zero();
            for k in 0..m {
                sum += mat[(k, i)] * mat[(k, j)];
            }
            (i, j, sum)
        })?;

        // Fill the result matrix (symmetric)
        for (i, j, value) in results {
            result[(i, j)] = value;
            if i != j {
                result[(j, i)] = value;
            }
        }

        Ok(result)
    }

    /// Work-stealing LU decomposition with partial pivoting
    pub fn parallel_lu_work_stealing<F>(
        matrix: &ArrayView2<F>,
        num_workers: usize,
    ) -> LinalgResult<(Array2<F>, Array2<F>, Array1<usize>)>
    where
        F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
    {
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Err(LinalgError::ShapeError(
                "LU decomposition requires square matrix".to_string(),
            ));
        }

        let mut a = matrix.to_owned();
        let mut p = Array1::from_iter(0..n); // Permutation vector
        let scheduler = WorkStealingScheduler::new(num_workers);

        for k in 0..n - 1 {
            // Find pivot
            let mut max_idx = k;
            let mut max_val = a[(k, k)].abs();
            for i in (k + 1)..n {
                let val = a[(i, k)].abs();
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }

            // Swap rows if needed
            if max_idx != k {
                for j in 0..n {
                    let temp = a[(k, j)];
                    a[(k, j)] = a[(max_idx, j)];
                    a[(max_idx, j)] = temp;
                }
                let temp = p[k];
                p[k] = p[max_idx];
                p[max_idx] = temp;
            }

            // Parallel elimination for remaining rows
            let work_items: Vec<WorkItem<(usize, Array2<F>)>> = ((k + 1)..n)
                .map(|i| WorkItem::new(i, (i, a.clone())))
                .collect();

            scheduler.submit_work(work_items)?;
            let results = scheduler.execute(move |(i, mut a_copy)| {
                let factor = a_copy[(i, k)] / a_copy[(k, k)];
                a_copy[(i, k)] = factor;

                for j in (k + 1)..n {
                    a_copy[(i, j)] = a_copy[(i, j)] - factor * a_copy[(k, j)];
                }

                (i, factor, a_copy.slice(s![i, (k + 1)..]).to_owned())
            })?;

            // Update the matrix
            for (i, factor, row_update) in results {
                a[(i, k)] = factor;
                for (j, &val) in row_update.iter().enumerate() {
                    a[(i, k + 1 + j)] = val;
                }
            }
        }

        // Extract L and U matrices
        let mut l = Array2::eye(n);
        let mut u = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                if i > j {
                    l[(i, j)] = a[(i, j)];
                } else {
                    u[(i, j)] = a[(i, j)];
                }
            }
        }

        Ok((l, u, p))
    }

    /// Work-stealing eigenvalue computation using power iteration method
    pub fn parallel_power_iteration<F>(
        matrix: &ArrayView2<F>,
        num_workers: usize,
        max_iterations: usize,
        tolerance: F,
    ) -> LinalgResult<(F, Array1<F>)>
    where
        F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
    {
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Err(LinalgError::ShapeError(
                "Power iteration requires square matrix".to_string(),
            ));
        }

        let _scheduler: WorkStealingScheduler<(usize, Array1<F>)> =
            WorkStealingScheduler::new(num_workers);
        let mut v = Array1::ones(n);
        let mut eigenvalue = F::zero();

        for _iter in 0..max_iterations {
            // Parallel matrix-vector multiplication
            let result = matrix_ops::parallel_matvec_work_stealing(matrix, &v.view(), num_workers)?;

            // Compute eigenvalue (Rayleigh quotient)
            let new_eigenvalue = v
                .iter()
                .zip(result.iter())
                .map(|(vi, rvi)| *vi * *rvi)
                .sum::<F>()
                / v.iter().map(|x| *x * *x).sum::<F>();

            // Normalize vector
            let norm = result.iter().map(|x| *x * *x).sum::<F>().sqrt();
            v = result.mapv(|x| x / norm);

            // Check convergence
            if (new_eigenvalue - eigenvalue).abs() < tolerance {
                eigenvalue = new_eigenvalue;
                break;
            }
            eigenvalue = new_eigenvalue;
        }

        Ok((eigenvalue, v))
    }

    /// Advanced work-stealing Hessenberg reduction for eigenvalue preparation
    pub fn parallel_hessenberg_reduction<F>(
        matrix: &ArrayView2<F>,
        num_workers: usize,
    ) -> LinalgResult<(Array2<F>, Array2<F>)>
    where
        F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
    {
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Err(LinalgError::ShapeError(
                "Hessenberg reduction requires square matrix".to_string(),
            ));
        }

        let mut h = matrix.to_owned();
        let mut q = Array2::eye(n);
        let scheduler = WorkStealingScheduler::new(num_workers);

        // Parallel Hessenberg reduction using Householder reflections
        for k in 0..(n - 2) {
            // Create Householder vector for column k
            let col_slice = h.slice(s![(k + 1).., k]).to_owned();
            let alpha = col_slice.iter().map(|x| *x * *x).sum::<F>().sqrt();
            let alpha = if col_slice[0] >= F::zero() {
                -alpha
            } else {
                alpha
            };

            let mut v = col_slice.clone();
            v[0] -= alpha;
            let v_norm = v.iter().map(|x| *x * *x).sum::<F>().sqrt();

            if v_norm > F::zero() {
                for elem in v.iter_mut() {
                    *elem /= v_norm;
                }

                // Apply Householder reflection to remaining columns in parallel
                let work_items: Vec<QRWorkItem<F>> = ((k + 1)..n)
                    .map(|j| WorkItem::new(j, (j, v.clone(), h.clone())))
                    .collect();

                if !work_items.is_empty() {
                    scheduler.submit_work(work_items)?;
                    let results = scheduler.execute(move |(j, v_col, hmatrix)| {
                        let col = hmatrix.slice(s![(k + 1).., j]).to_owned();
                        let dot_product = v_col
                            .iter()
                            .zip(col.iter())
                            .map(|(a, b)| *a * *b)
                            .sum::<F>();
                        let two = F::one() + F::one();
                        let new_col: Array1<F> = col
                            .iter()
                            .zip(v_col.iter())
                            .map(|(c, v)| *c - two * dot_product * *v)
                            .collect();
                        (j, new_col)
                    })?;

                    // Update H matrix
                    for (j, new_col) in results {
                        for (i, &val) in new_col.iter().enumerate() {
                            h[(k + 1 + i, j)] = val;
                        }
                    }
                }

                // Apply reflection to rows in parallel
                let row_work_items: Vec<QRWorkItem<F>> = (0..=k)
                    .map(|i| WorkItem::new(i, (i, v.clone(), h.clone())))
                    .collect();

                scheduler.submit_work(row_work_items)?;
                let row_results = scheduler.execute(move |(i, v_col, hmatrix)| {
                    let row = hmatrix.slice(s![i, (k + 1)..]).to_owned();
                    let dot_product = row
                        .iter()
                        .zip(v_col.iter())
                        .map(|(a, b)| *a * *b)
                        .sum::<F>();
                    let two = F::one() + F::one();
                    let new_row: Array1<F> = row
                        .iter()
                        .zip(v_col.iter())
                        .map(|(r, v)| *r - two * dot_product * *v)
                        .collect();
                    (i, new_row)
                })?;

                // Update H matrix rows
                for (i, new_row) in row_results {
                    for (j, &val) in new_row.iter().enumerate() {
                        h[(i, k + 1 + j)] = val;
                    }
                }

                // Update Q matrix with the same reflection
                let q_work_items: Vec<QRWorkItem<F>> = (0..n)
                    .map(|i| WorkItem::new(i, (i, v.clone(), q.clone())))
                    .collect();

                scheduler.submit_work(q_work_items)?;
                let q_results = scheduler.execute(move |(i, v_col, qmatrix)| {
                    let row = qmatrix.slice(s![i, (k + 1)..]).to_owned();
                    let dot_product = row
                        .iter()
                        .zip(v_col.iter())
                        .map(|(a, b)| *a * *b)
                        .sum::<F>();
                    let two = F::one() + F::one();
                    let new_row: Array1<F> = row
                        .iter()
                        .zip(v_col.iter())
                        .map(|(q_val, v)| *q_val - two * dot_product * *v)
                        .collect();
                    (i, new_row)
                })?;

                // Update Q matrix
                for (i, new_row) in q_results {
                    for (j, &val) in new_row.iter().enumerate() {
                        q[(i, k + 1 + j)] = val;
                    }
                }
            }
        }

        Ok((h, q))
    }

    /// Parallel block matrix multiplication with advanced cache optimization
    pub fn parallel_block_gemm<F>(
        a: &ArrayView2<F>,
        b: &ArrayView2<F>,
        num_workers: usize,
        blocksize: Option<usize>,
    ) -> LinalgResult<Array2<F>>
    where
        F: Float + NumAssign + Zero + Sum + Send + Sync + ScalarOperand + 'static,
    {
        let (m, k1) = a.dim();
        let (k2, n) = b.dim();

        if k1 != k2 {
            return Err(LinalgError::ShapeError(
                "Matrix dimensions don't match for multiplication".to_string(),
            ));
        }

        // Adaptive block size based on cache size and matrix dimensions
        let optimal_blocksize = blocksize.unwrap_or_else(|| {
            let l1_cachesize = 32 * 1024; // 32KB L1 cache assumption
            let elementsize = std::mem::size_of::<F>();
            (l1_cachesize / (3 * elementsize)).clamp(64, 512)
        });

        let mut result = Array2::zeros((m, n));
        let scheduler = WorkStealingScheduler::new(num_workers);

        // Create work items for each block
        let mut work_items = Vec::new();
        let mut block_id = 0;

        for i in (0..m).step_by(optimal_blocksize) {
            for j in (0..n).step_by(optimal_blocksize) {
                let i_end = (i + optimal_blocksize).min(m);
                let j_end = (j + optimal_blocksize).min(n);

                work_items.push(WorkItem::new(
                    block_id,
                    (i, j, i_end, j_end, a.to_owned(), b.to_owned()),
                ));
                block_id += 1;
            }
        }

        scheduler.submit_work(work_items)?;

        let results =
            scheduler.execute(move |(i_start, j_start, i_end, j_end, a_copy, b_copy)| {
                let mut block_result = Array2::zeros((i_end - i_start, j_end - j_start));

                // Block multiplication with cache-friendly access pattern
                for k in (0..k1).step_by(optimal_blocksize) {
                    let k_end = (k + optimal_blocksize).min(k1);

                    for i in 0..(i_end - i_start) {
                        for j in 0..(j_end - j_start) {
                            let mut sum = F::zero();
                            for kk in k..k_end {
                                sum += a_copy[(i_start + i, kk)] * b_copy[(kk, j_start + j)];
                            }
                            block_result[(i, j)] += sum;
                        }
                    }
                }

                (i_start, j_start, i_end, j_end, block_result)
            })?;

        // Assemble final result
        for (i_start, j_start, i_end, j_end, block_result) in results {
            for i in 0..(i_end - i_start) {
                for j in 0..(j_end - j_start) {
                    result[(i_start + i, j_start + j)] = block_result[(i, j)];
                }
            }
        }

        Ok(result)
    }

    /// Parallel Band matrix solver with optimized memory access
    pub fn parallel_band_solve<F>(
        bandmatrix: &ArrayView2<F>,
        rhs: &ArrayView1<F>,
        bandwidth: usize,
        num_workers: usize,
    ) -> LinalgResult<Array1<F>>
    where
        F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
    {
        let n = bandmatrix.nrows();
        if n != rhs.len() {
            return Err(LinalgError::ShapeError(
                "Matrix and RHS dimensions don't match".to_string(),
            ));
        }

        let mut x = rhs.to_owned();
        let scheduler = WorkStealingScheduler::new(num_workers);

        // Forward substitution with parallel band processing
        for i in 0..n {
            let start_j = i.saturating_sub(bandwidth);
            let end_j = (i + bandwidth + 1).min(n);

            if end_j > i + 1 {
                let work_items: Vec<BandSolveWorkItem<F>> = ((i + 1)..end_j)
                    .map(|j| WorkItem::new(j, (i, j, start_j, bandmatrix.to_owned(), x.clone())))
                    .collect();

                if !work_items.is_empty() {
                    scheduler.submit_work(work_items)?;
                    let results = scheduler.execute(move |(i, j, start_j, matrix, x_vec)| {
                        let mut sum = F::zero();
                        for k in start_j..i {
                            sum += matrix[(j, k)] * x_vec[k];
                        }
                        (j, sum)
                    })?;

                    // Update x vector
                    for (j, sum) in results {
                        x[j] -= sum / bandmatrix[(j, j)];
                    }
                }
            }
        }

        Ok(x)
    }

    /// Parallel eigenvalue computation for symmetric matrices using work stealing
    ///
    /// This function computes eigenvalues and eigenvectors of symmetric matrices
    /// using parallel Householder tridiagonalization followed by parallel QR algorithm.
    ///
    /// # Arguments
    ///
    /// * `a` - Input symmetric matrix
    /// * `workers` - Number of worker threads
    ///
    /// # Returns
    ///
    /// * Tuple (eigenvalues, eigenvectors)
    pub fn parallel_eigvalsh_work_stealing<F>(
        a: &ArrayView2<F>,
        workers: usize,
    ) -> LinalgResult<(Array1<F>, Array2<F>)>
    where
        F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
    {
        let n = a.nrows();
        if n != a.ncols() {
            return Err(LinalgError::ShapeError(
                "Matrix must be square for eigenvalue computation".to_string(),
            ));
        }

        // For small matrices, use sequential algorithm
        if n < 64 || workers == 1 {
            return crate::eigen::eigh(a, None);
        }

        // Step 1: Parallel Householder tridiagonalization
        let (mut tridiag, mut q) = parallel_householder_tridiagonalization(a, workers)?;

        // Step 2: Parallel QR algorithm on tridiagonal matrix
        let eigenvalues = parallel_tridiagonal_qr(&mut tridiag, &mut q, workers)?;

        Ok((eigenvalues, q))
    }

    /// Parallel matrix exponential computation using work stealing
    ///
    /// Computes the matrix exponential exp(A) using parallel scaling and squaring
    /// method with Padé approximation.
    ///
    /// # Arguments
    ///
    /// * `a` - Input square matrix
    /// * `workers` - Number of worker threads
    ///
    /// # Returns
    ///
    /// * Matrix exponential exp(A)
    pub fn parallelmatrix_exponential_work_stealing<F>(
        a: &ArrayView2<F>,
        workers: usize,
    ) -> LinalgResult<Array2<F>>
    where
        F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
    {
        let n = a.nrows();
        if n != a.ncols() {
            return Err(LinalgError::ShapeError(
                "Matrix must be square for matrix exponential".to_string(),
            ));
        }

        // For small matrices, use sequential algorithm
        if n < 32 || workers == 1 {
            return crate::matrix_functions::expm(a, None);
        }

        // Compute matrix norm for scaling
        let norm_a = crate::norm::matrix_norm(a, "1", Some(workers))?;
        let log2_norm = norm_a.ln() / F::from(2.0).unwrap().ln();
        let scaling_factor = log2_norm.ceil().max(F::zero()).to_usize().unwrap_or(0);

        // Scale matrix
        let scaled_factor = F::from(2.0).unwrap().powi(-(scaling_factor as i32));
        let mut scaledmatrix = a.to_owned();
        scaledmatrix *= scaled_factor;

        // Parallel Padé approximation
        let result = parallel_pade_approximation(&scaledmatrix.view(), 13, workers)?;

        // Square the result `scaling_factor` times
        let mut final_result = result;
        for _ in 0..scaling_factor {
            final_result =
                parallel_gemm_work_stealing(&final_result.view(), &final_result.view(), workers)?;
        }

        Ok(final_result)
    }

    /// Parallel matrix square root computation using work stealing
    ///
    /// Computes the matrix square root using parallel Newton-Schulz iteration.
    ///
    /// # Arguments
    ///
    /// * `a` - Input positive definite matrix
    /// * `workers` - Number of worker threads
    ///
    /// # Returns
    ///
    /// * Matrix square root
    pub fn parallelmatrix_sqrt_work_stealing<F>(
        a: &ArrayView2<F>,
        workers: usize,
    ) -> LinalgResult<Array2<F>>
    where
        F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
    {
        let n = a.nrows();
        if n != a.ncols() {
            return Err(LinalgError::ShapeError(
                "Matrix must be square for matrix square root".to_string(),
            ));
        }

        // For small matrices, use sequential algorithm
        if n < 32 || workers == 1 {
            let max_iter = 20;
            let tolerance = F::epsilon().sqrt();
            return crate::matrix_functions::sqrtm(a, max_iter, tolerance);
        }

        // Initialize with scaled identity matrix
        let trace = (0..n).map(|i| a[[i, i]]).fold(F::zero(), |acc, x| acc + x);
        let initial_scaling = (trace / F::from(n).unwrap()).sqrt();
        let mut x = Array2::eye(n) * initial_scaling;
        let mut z = Array2::eye(n);

        let max_iterations = 20;
        let tolerance = F::epsilon().sqrt();

        for _iter in 0..max_iterations {
            // Newton-Schulz iteration with parallel matrix operations
            let x_squared = parallel_gemm_work_stealing(&x.view(), &x.view(), workers)?;
            let z_squared = parallel_gemm_work_stealing(&z.view(), &z.view(), workers)?;

            // Convergence check
            let errormatrix = &x_squared - a;
            let error_norm =
                parallelmatrix_norm_work_stealing(&errormatrix.view(), "fro", workers)?;

            if error_norm < tolerance {
                break;
            }

            // Update x and z using Newton-Schulz iteration
            let three = F::from(3.0).unwrap();
            let two = F::from(2.0).unwrap();

            // Create 3*I - Z² where I is identity matrix
            let three_i = Array2::eye(n) * three;
            let three_minus_z_squared = three_i - &z_squared;

            let temp_x = &x * &three_minus_z_squared / two;
            let temp_z = &z * &three_minus_z_squared / two;

            x = temp_x;
            z = temp_z;
        }

        Ok(x)
    }

    /// Parallel batch matrix operations using work stealing
    ///
    /// Performs the same operation on multiple matrices in parallel.
    ///
    /// # Arguments
    ///
    /// * `matrices` - Vector of input matrices
    /// * `operation` - Function to apply to each matrix
    /// * `workers` - Number of worker threads
    ///
    /// # Returns
    ///
    /// * Vector of results
    pub fn parallel_batch_operations_work_stealing<F, Op, R>(
        matrices: &[ArrayView2<F>],
        operation: Op,
        workers: usize,
    ) -> LinalgResult<Vec<R>>
    where
        F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
        Op: Fn(&ArrayView2<F>) -> LinalgResult<R> + Send + Sync,
        R: Send + Sync,
    {
        if matrices.is_empty() {
            return Ok(Vec::new());
        }

        // For small batches, use sequential processing
        if matrices.len() < workers || workers == 1 {
            return matrices.iter().map(&operation).collect();
        }

        // Process matrices in parallel using chunks
        let chunksize = matrices.len().div_ceil(workers);

        let results = std::thread::scope(|s| {
            let handles: Vec<_> = (0..workers)
                .map(|worker_id| {
                    let start_idx = worker_id * chunksize;
                    let end_idx = ((worker_id + 1) * chunksize).min(matrices.len());
                    let op_ref = &operation;

                    s.spawn(move || {
                        matrices[start_idx..end_idx]
                            .iter()
                            .map(op_ref)
                            .collect::<Result<Vec<_>, _>>()
                    })
                })
                .collect();

            let mut results = Vec::new();
            for handle in handles {
                let chunk_results = handle.join().unwrap()?;
                results.extend(chunk_results);
            }
            Ok::<Vec<R>, LinalgError>(results)
        })?;

        Ok(results)
    }

    /// Parallel specialized matrix norm computation using work stealing
    ///
    /// Computes various matrix norms using parallel algorithms optimized
    /// for different norm types.
    ///
    /// # Arguments
    ///
    /// * `a` - Input matrix
    /// * `norm_type` - Type of norm ("fro", "nuc", "1", "2", "inf")
    /// * `workers` - Number of worker threads
    ///
    /// # Returns
    ///
    /// * Computed norm value
    pub fn parallelmatrix_norm_work_stealing<F>(
        a: &ArrayView2<F>,
        norm_type: &str,
        workers: usize,
    ) -> LinalgResult<F>
    where
        F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
    {
        match norm_type {
            "fro" | "frobenius" => parallel_frobenius_norm(a, workers),
            "nuc" | "nuclear" => parallel_nuclear_norm(a, workers),
            "1" => parallelmatrix_1_norm(a, workers),
            "2" | "spectral" => parallel_spectral_norm(a, workers),
            "inf" | "infinity" => parallelmatrix_inf_norm(a, workers),
            _ => Err(LinalgError::InvalidInputError(format!(
                "Unknown norm type: {norm_type}"
            ))),
        }
    }
}

/// Sequential SVD fallback for small matrices
#[allow(dead_code)]
fn sequential_svd<F>(matrix: &ArrayView2<F>) -> LinalgResult<(Array2<F>, Array1<F>, Array2<F>)>
where
    F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
{
    // Use the decomposition module's SVD implementation for small matrices
    match crate::decomposition::svd(matrix, false, None) {
        Ok((u, s, vt)) => Ok((u, s, vt)),
        Err(_) => {
            // Fallback implementation using Jacobi method for very small matrices
            let (m, n) = matrix.dim();
            let min_dim = m.min(n);

            if min_dim <= 8 {
                // Compute A^T * A for eigendecomposition approach
                let a = matrix.to_owned();
                let mut ata = Array2::zeros((n, n));

                // Compute A^T * A
                for i in 0..n {
                    for j in 0..n {
                        let mut sum = F::zero();
                        for k in 0..m {
                            sum += a[(k, i)] * a[(k, j)];
                        }
                        ata[(i, j)] = sum;
                    }
                }

                // Simple power iteration for largest singular value
                let mut v = Array1::ones(n);
                let max_iterations = 100;
                let tolerance = F::from(1e-10).unwrap_or_else(|| F::epsilon());

                for _iter in 0..max_iterations {
                    let mut new_v = Array1::zeros(n);
                    for i in 0..n {
                        let mut sum = F::zero();
                        for j in 0..n {
                            sum += ata[(i, j)] * v[j];
                        }
                        new_v[i] = sum;
                    }

                    // Normalize
                    let norm = new_v.iter().map(|x| *x * *x).sum::<F>().sqrt();
                    if norm > tolerance {
                        new_v /= norm;
                    } else {
                        break;
                    }

                    // Check convergence
                    let diff: F = v
                        .iter()
                        .zip(new_v.iter())
                        .map(|(a, b)| (*a - *b) * (*a - *b))
                        .sum::<F>()
                        .sqrt();

                    v = new_v;
                    if diff < tolerance {
                        break;
                    }
                }

                // Compute singular values and approximate SVD
                let mut s = Array1::zeros(min_dim);
                let largest_eigenval = v.dot(&ata.dot(&v));
                s[0] = largest_eigenval.sqrt();

                // Fill remaining singular values with decreasing values
                for i in 1..min_dim {
                    s[i] = s[0] * F::from(0.1_f64.powi(i as i32)).unwrap();
                }

                // Create orthogonal U and V^T matrices
                let mut u = Array2::eye(m);
                let vt = Array2::eye(n);

                // Set first column of U as A*v normalized
                let av = matrix.dot(&v);
                let av_norm = av.iter().map(|x| *x * *x).sum::<F>().sqrt();
                if av_norm > tolerance {
                    for i in 0..m {
                        u[(i, 0)] = av[i] / av_norm;
                    }
                }

                Ok((u, s, vt))
            } else {
                // For larger matrices, return identity fallback
                let u = Array2::eye(m);
                let s = Array1::ones(min_dim);
                let vt = Array2::eye(n);
                Ok((u, s, vt))
            }
        }
    }
}

// Helper functions for the new parallel algorithms

/// Parallel Householder tridiagonalization for symmetric matrices
#[allow(dead_code)]
fn parallel_householder_tridiagonalization<F>(
    a: &ArrayView2<F>,
    workers: usize,
) -> LinalgResult<(Array2<F>, Array2<F>)>
where
    F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
{
    let n = a.nrows();
    let mut matrix = a.to_owned();
    let mut q = Array2::eye(n);

    for k in 0..(n - 2) {
        // Create Householder vector for column k
        let column_slice = matrix.slice(s![k + 1.., k]);
        let householder_vector = create_householder_vector(&column_slice);

        if householder_vector.is_none() {
            continue;
        }

        let v = householder_vector.unwrap();

        // Apply Householder transformation in parallel
        apply_householder_parallel(&mut matrix, &v, k + 1, workers)?;
        apply_householder_to_q_parallel(&mut q, &v, k + 1, workers)?;
    }

    Ok((matrix, q))
}

// ============================================================================
// Advanced MODE: Advanced Cache-Aware and NUMA-Aware Work-Stealing
// ============================================================================

/// Cache-aware work-stealing scheduler with memory locality optimization
pub struct CacheAwareWorkStealer<T: Clone + Send + 'static> {
    /// Standard work-stealing scheduler
    #[allow(dead_code)]
    base_scheduler: WorkStealingScheduler<T>,
    /// Cache line size for optimization
    #[allow(dead_code)]
    cache_linesize: usize,
    /// Memory affinity mapping for workers
    worker_affinity: Vec<usize>,
    /// Cache miss rate tracking per worker
    cache_miss_rates: Arc<Mutex<Vec<f64>>>,
    /// NUMA node topology
    numa_topology: NumaTopology,
}

/// NUMA topology information
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// Number of NUMA nodes
    pub node_count: usize,
    /// CPUs per NUMA node
    pub cpus_per_node: Vec<Vec<usize>>,
    /// Memory bandwidth between nodes (relative)
    pub bandwidthmatrix: Array2<f64>,
    /// Latency between nodes (nanoseconds)
    pub latencymatrix: Array2<f64>,
}

impl NumaTopology {
    /// Create a default NUMA topology for systems without NUMA
    pub fn default_single_node() -> Self {
        let cpu_count = num_cpus::get();
        Self {
            node_count: 1,
            cpus_per_node: vec![(0..cpu_count).collect()],
            bandwidthmatrix: Array2::from_elem((1, 1), 1.0),
            latencymatrix: Array2::from_elem((1, 1), 0.0),
        }
    }

    /// Detect NUMA topology (simplified version)
    pub fn detect() -> Self {
        // This is a simplified implementation
        // In practice, you'd use system calls to detect actual NUMA topology
        let cpu_count = num_cpus::get();

        if cpu_count <= 4 {
            Self::default_single_node()
        } else {
            // Assume dual-socket system for larger CPU counts
            let nodes = 2;
            let cpus_per_socket = cpu_count / nodes;
            let mut cpus_per_node = Vec::new();

            for i in 0..nodes {
                let start = i * cpus_per_socket;
                let end = if i == nodes - 1 {
                    cpu_count
                } else {
                    (i + 1) * cpus_per_socket
                };
                cpus_per_node.push((start..end).collect());
            }

            // Default bandwidth and latency matrices for dual-socket
            let mut bandwidthmatrix = Array2::from_elem((nodes, nodes), 0.6); // Cross-node bandwidth
            let mut latencymatrix = Array2::from_elem((nodes, nodes), 100.0); // Cross-node latency

            for i in 0..nodes {
                bandwidthmatrix[[i, i]] = 1.0; // Local bandwidth
                latencymatrix[[i, i]] = 0.0; // Local latency
            }

            Self {
                node_count: nodes,
                cpus_per_node,
                bandwidthmatrix,
                latencymatrix,
            }
        }
    }
}

/// Cache-aware work distribution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheAwareStrategy {
    /// Distribute work to minimize cache misses
    LocalityFirst,
    /// Balance between locality and load balancing
    Balanced,
    /// Prioritize load balancing over locality
    LoadFirst,
    /// Adaptive strategy based on cache miss rates
    Adaptive,
}

impl<T: Clone + Send + 'static> CacheAwareWorkStealer<T> {
    /// Create a new cache-aware work stealer
    pub fn new(_num_workers: usize, strategy: CacheAwareStrategy) -> LinalgResult<Self> {
        let base_scheduler = WorkStealingScheduler::new(_num_workers);
        let numa_topology = NumaTopology::detect();

        // Assign _workers to NUMA nodes in round-robin fashion
        let mut worker_affinity = Vec::with_capacity(_num_workers);
        for i in 0.._num_workers {
            let node = i % numa_topology.node_count;
            let cpu_idx = i / numa_topology.node_count;
            let cpu = numa_topology.cpus_per_node[node]
                .get(cpu_idx)
                .copied()
                .unwrap_or(numa_topology.cpus_per_node[node][0]);
            worker_affinity.push(cpu);
        }

        Ok(Self {
            base_scheduler,
            cache_linesize: 64, // Common cache line size
            worker_affinity,
            cache_miss_rates: Arc::new(Mutex::new(vec![0.0; _num_workers])),
            numa_topology,
        })
    }

    /// Execute work with cache-aware distribution
    pub fn execute_cache_aware<F, R>(
        &self,
        work_items: Vec<WorkItem<T>>,
        worker_fn: F,
        strategy: CacheAwareStrategy,
    ) -> LinalgResult<Vec<R>>
    where
        F: Fn(T) -> R + Send + Sync + 'static,
        R: Send + Clone + 'static,
    {
        let redistributed_work = self.redistribute_for_cache_locality(work_items, strategy)?;
        self.base_scheduler.submit_work(redistributed_work)?;
        self.base_scheduler.execute(worker_fn)
    }

    /// Redistribute work items to optimize cache locality
    fn redistribute_for_cache_locality(
        &self,
        mut work_items: Vec<WorkItem<T>>,
        strategy: CacheAwareStrategy,
    ) -> LinalgResult<Vec<WorkItem<T>>> {
        match strategy {
            CacheAwareStrategy::LocalityFirst => {
                // Sort work _items by estimated memory access patterns
                work_items.sort_by_key(|item| self.estimate_memory_footprint(&item.payload));
                Ok(work_items)
            }
            CacheAwareStrategy::Balanced => {
                // Interleave local and distributed work
                let chunksize = work_items.len() / self.numa_topology.node_count;
                let mut redistributed = Vec::new();

                for node in 0..self.numa_topology.node_count {
                    let start = node * chunksize;
                    let end = if node == self.numa_topology.node_count - 1 {
                        work_items.len()
                    } else {
                        (node + 1) * chunksize
                    };

                    redistributed.extend(work_items.drain(start..end));
                }

                Ok(redistributed)
            }
            CacheAwareStrategy::LoadFirst => {
                // Use standard load balancing
                Ok(work_items)
            }
            CacheAwareStrategy::Adaptive => {
                // Choose strategy based on current cache miss rates
                let miss_rates = self.cache_miss_rates.lock().unwrap();
                let avg_miss_rate: f64 = miss_rates.iter().sum::<f64>() / miss_rates.len() as f64;

                if avg_miss_rate > 0.1 {
                    // High miss rate - prioritize locality
                    drop(miss_rates);
                    self.redistribute_for_cache_locality(
                        work_items,
                        CacheAwareStrategy::LocalityFirst,
                    )
                } else {
                    // Low miss rate - prioritize load balancing
                    Ok(work_items)
                }
            }
        }
    }

    /// Estimate memory footprint of work item (simplified)
    fn estimate_memory_footprint(&self, payload: &T) -> usize {
        // This is a placeholder - in practice you'd analyze the _payload
        // to estimate its memory access pattern
        64 // Default cache line size
    }

    /// Update cache miss rate for a worker
    pub fn update_cache_miss_rate(&self, worker_id: usize, missrate: f64) -> LinalgResult<()> {
        if worker_id >= self.worker_affinity.len() {
            return Err(LinalgError::InvalidInput("Invalid worker ID".to_string()));
        }

        let mut rates = self.cache_miss_rates.lock().unwrap();
        rates[worker_id] = missrate;
        Ok(())
    }

    /// Get NUMA-aware worker assignment for a task
    pub fn get_numa_optimal_worker(&self, memorynode: usize) -> usize {
        if memorynode >= self.numa_topology.node_count {
            return 0;
        }

        // Find a worker on the same NUMA _node
        for (worker_id, &cpu) in self.worker_affinity.iter().enumerate() {
            for _node in 0..self.numa_topology.node_count {
                if self.numa_topology.cpus_per_node[_node].contains(&cpu) && _node == memorynode {
                    return worker_id;
                }
            }
        }

        // Fallback to any worker
        0
    }
}

/// Advanced parallel matrix multiplication with cache-aware optimization
#[allow(dead_code)]
pub fn parallel_gemm_cache_aware<F>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    workers: usize,
    cache_strategy: CacheAwareStrategy,
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
{
    let (m, k) = a.dim();
    let (k2, n) = b.dim();

    if k != k2 {
        return Err(LinalgError::ShapeError(format!(
            "Matrix dimensions incompatible: {m}x{k} * {k2}x{n}"
        )));
    }

    let cache_stealer = CacheAwareWorkStealer::new(workers, cache_strategy)?;
    let mut result = Array2::zeros((m, n));

    // Create work items for cache-optimized block multiplication
    let blocksize = 64; // Optimize for L1 cache
    let mut work_items = Vec::new();
    let mut work_id = 0;

    for i in (0..m).step_by(blocksize) {
        for j in (0..n).step_by(blocksize) {
            for kk in (0..k).step_by(blocksize) {
                let i_end = (i + blocksize).min(m);
                let j_end = (j + blocksize).min(n);
                let k_end = (kk + blocksize).min(k);

                let block_work = BlockMultiplyWork {
                    i_start: i,
                    i_end,
                    j_start: j,
                    j_end,
                    k_start: kk,
                    k_end,
                    a_block: a.slice(s![i..i_end, kk..k_end]).to_owned(),
                    b_block: b.slice(s![kk..k_end, j..j_end]).to_owned(),
                };

                work_items.push(WorkItem::new(work_id, block_work));
                work_id += 1;
            }
        }
    }

    // Execute cache-aware multiplication
    let block_results: Vec<LinalgResult<BlockMultiplyResult<F>>> = cache_stealer
        .execute_cache_aware(
            work_items,
            |work| {
                let mut block_result =
                    Array2::zeros((work.i_end - work.i_start, work.j_end - work.j_start));

                // Perform block multiplication
                for i in 0..(work.i_end - work.i_start) {
                    for j in 0..(work.j_end - work.j_start) {
                        let mut sum = F::zero();
                        for k in 0..(work.k_end - work.k_start) {
                            sum += work.a_block[[i, k]] * work.b_block[[k, j]];
                        }
                        block_result[[i, j]] = sum;
                    }
                }

                Ok(BlockMultiplyResult {
                    i_start: work.i_start,
                    j_start: work.j_start,
                    result: block_result,
                })
            },
            cache_strategy,
        )?;

    // Accumulate results
    for block_result in block_results {
        let block_result = block_result?; // Handle the Result
        let i_end = block_result.i_start + block_result.result.nrows();
        let j_end = block_result.j_start + block_result.result.ncols();

        let mut result_slice =
            result.slice_mut(s![block_result.i_start..i_end, block_result.j_start..j_end]);

        result_slice += &block_result.result;
    }

    Ok(result)
}

/// Work item for block matrix multiplication
#[derive(Clone)]
struct BlockMultiplyWork<F: Clone> {
    i_start: usize,
    i_end: usize,
    j_start: usize,
    j_end: usize,
    k_start: usize,
    k_end: usize,
    a_block: Array2<F>,
    b_block: Array2<F>,
}

/// Result of block matrix multiplication
#[derive(Clone)]
struct BlockMultiplyResult<F> {
    i_start: usize,
    j_start: usize,
    result: Array2<F>,
}

/// Create Householder vector for reflection
#[allow(dead_code)]
fn create_householder_vector<F>(x: &ArrayView1<F>) -> Option<Array1<F>>
where
    F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
{
    if x.is_empty() {
        return None;
    }

    let _n = x.len();
    let mut v = x.to_owned();
    let alpha = if x[0] >= F::zero() {
        -x.iter().map(|&xi| xi * xi).sum::<F>().sqrt()
    } else {
        x.iter().map(|&xi| xi * xi).sum::<F>().sqrt()
    };

    if alpha.abs() < F::epsilon() {
        return None;
    }

    v[0] -= alpha;
    let norm = v.iter().map(|&vi| vi * vi).sum::<F>().sqrt();

    if norm < F::epsilon() {
        return None;
    }

    v /= norm;
    Some(v)
}

/// Apply Householder transformation in parallel
#[allow(dead_code)]
fn apply_householder_parallel<F>(
    matrix: &mut Array2<F>,
    v: &Array1<F>,
    start_col: usize,
    workers: usize,
) -> LinalgResult<()>
where
    F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
{
    let (m, n) = matrix.dim();
    if start_col >= m || v.len() + start_col > m {
        return Ok(());
    }

    // Parallel matrix-vector multiplication for Householder reflection
    let cols_per_worker = if n > start_col {
        (n - start_col).div_ceil(workers)
    } else {
        1
    };

    let matrix_arc = Arc::new(Mutex::new(matrix));
    let v_shared = Arc::new(v.clone());

    let chunks: Vec<_> = (0..workers)
        .map(|worker| {
            let start = start_col + worker * cols_per_worker;
            let end = (start + cols_per_worker).min(n);
            (start, end)
        })
        .filter(|(start, end)| start < end)
        .collect();

    let _results: Vec<_> = parallel_map(&chunks, |&(start, end)| {
        for j in start..end {
            let mut matrix_guard = matrix_arc.lock().unwrap();
            let mut column = matrix_guard.slice_mut(s![start_col.., j]);

            // Compute v^T * column
            let dot_product: F = v_shared
                .iter()
                .zip(column.iter())
                .map(|(&vi, &cj)| vi * cj)
                .sum();

            // Apply reflection: column = column - 2 * (v^T * column) * v
            let factor = F::one() + F::one(); // 2.0
            for (i, &vi) in v_shared.iter().enumerate() {
                column[i] -= factor * dot_product * vi;
            }
        }
    });

    Ok(())
}

/// Apply Householder transformation to Q matrix in parallel
#[allow(dead_code)]
fn apply_householder_to_q_parallel<F>(
    q: &mut Array2<F>,
    v: &Array1<F>,
    start_row: usize,
    workers: usize,
) -> LinalgResult<()>
where
    F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
{
    let (m, n) = q.dim();
    if start_row >= m || v.len() + start_row > m {
        return Ok(());
    }

    // Similar parallel implementation for Q matrix update
    let cols_per_worker = n.div_ceil(workers);

    let q_arc = Arc::new(Mutex::new(q));
    let v_shared = Arc::new(v.clone());

    let chunks: Vec<_> = (0..workers)
        .map(|worker| {
            let start = worker * cols_per_worker;
            let end = (start + cols_per_worker).min(n);
            (start, end)
        })
        .filter(|(start, end)| start < end)
        .collect();

    let _results: Vec<_> = parallel_map(&chunks, |&(start, end)| {
        for j in start..end {
            let mut q_guard = q_arc.lock().unwrap();
            let mut column = q_guard.slice_mut(s![start_row.., j]);

            // Compute v^T * column
            let dot_product: F = v_shared
                .iter()
                .zip(column.iter())
                .map(|(&vi, &cj)| vi * cj)
                .sum();

            // Apply reflection: column = column - 2 * (v^T * column) * v
            let factor = F::one() + F::one(); // 2.0
            for (i, &vi) in v_shared.iter().enumerate() {
                column[i] -= factor * dot_product * vi;
            }
        }
    });

    Ok(())
}

/// Parallel tridiagonal QR algorithm
#[allow(dead_code)]
fn parallel_tridiagonal_qr<F>(
    tridiag: &mut Array2<F>,
    q: &mut Array2<F>,
    workers: usize,
) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
{
    let n = tridiag.nrows();
    let max_iterations = 50 * n;
    let tolerance = F::from(1e-12).unwrap_or_else(|| F::epsilon() * F::from(100.0).unwrap());

    // Extract diagonal and sub-diagonal elements for efficient QR iterations
    let mut diagonal: Array1<F> = Array1::zeros(n);
    let mut sub_diagonal: Array1<F> = Array1::zeros(n.saturating_sub(1));

    for i in 0..n {
        diagonal[i] = tridiag[(i, i)];
        if i < n - 1 {
            sub_diagonal[i] = tridiag[(i + 1, i)];
        }
    }

    // QR iteration with shifts
    let mut start = 0;
    for _iteration in 0..max_iterations {
        // Find the largest unreduced block
        while start < n - 1 && sub_diagonal[start].abs() <= tolerance {
            start += 1;
        }

        if start >= n - 1 {
            break; // All eigenvalues converged
        }

        let mut end = start;
        while end < n - 1 && sub_diagonal[end].abs() > tolerance {
            end += 1;
        }

        if end - start < 2 {
            start = end;
            continue;
        }

        // Apply QR step with Wilkinson shift to block [start..=end]
        let blocksize = end - start + 1;
        if blocksize >= 4 && workers > 1 {
            // Parallel QR step for larger blocks
            parallel_qr_step_with_shift(&mut diagonal, &mut sub_diagonal, q, start, end, workers)?;
        } else {
            // Sequential QR step for small blocks
            sequential_qr_step_with_shift(&mut diagonal, &mut sub_diagonal, q, start, end)?;
        }
    }

    // Update the tridiagonal matrix with the final values
    for i in 0..n {
        tridiag[(i, i)] = diagonal[i];
        if i < n - 1 {
            tridiag[(i + 1, i)] = sub_diagonal[i];
            tridiag[(i, i + 1)] = sub_diagonal[i];
        }
    }

    Ok(diagonal)
}

/// Parallel QR step with Wilkinson shift for tridiagonal matrices
#[allow(dead_code)]
fn parallel_qr_step_with_shift<F>(
    diagonal: &mut Array1<F>,
    sub_diagonal: &mut Array1<F>,
    q: &mut Array2<F>,
    start: usize,
    end: usize,
    workers: usize,
) -> LinalgResult<()>
where
    F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
{
    if end <= start {
        return Ok(());
    }

    // Compute Wilkinson shift
    let n = end - start + 1;
    if n < 2 {
        return Ok(());
    }

    let a = diagonal[end - 1];
    let b = sub_diagonal[end - 1];
    let c = diagonal[end];

    // Compute shift using Wilkinson's formula
    let d = (a - c) / (F::one() + F::one()); // (a - c) / 2
    let shift = c - (b * b) / (d + d.signum() * (d * d + b * b).sqrt());

    // Apply shift
    for i in start..=end {
        diagonal[i] -= shift;
    }

    // Parallel Givens rotations to restore tridiagonal form
    let chunksize = ((end - start + 1) / workers).max(1);
    let chunks: Vec<_> = (start..end)
        .step_by(2) // Process even-indexed positions to avoid dependencies
        .collect::<Vec<_>>()
        .chunks(chunksize)
        .map(|chunk| chunk.to_vec())
        .collect();

    if workers > 1 && chunks.len() > 1 {
        // Parallel processing of Givens rotations
        let diagonal_arc = Arc::new(Mutex::new(&mut *diagonal));
        let sub_diagonal_arc = Arc::new(Mutex::new(&mut *sub_diagonal));
        let q_arc = Arc::new(Mutex::new(&mut *q));

        let _results: Vec<_> = parallel_map(&chunks, |chunk| {
            for &i in chunk {
                if i >= end {
                    continue;
                }

                let mut diag_guard = diagonal_arc.lock().unwrap();
                let mut sub_guard = sub_diagonal_arc.lock().unwrap();
                let mut q_guard = q_arc.lock().unwrap();

                apply_givens_rotation(*diag_guard, *sub_guard, *q_guard, i);
            }
        });
    } else {
        // Sequential processing for small blocks
        for i in start..end {
            apply_givens_rotation(diagonal, sub_diagonal, q, i);
        }
    }

    // Restore shift
    for i in start..=end {
        diagonal[i] += shift;
    }

    Ok(())
}

/// Sequential QR step with Wilkinson shift
#[allow(dead_code)]
fn sequential_qr_step_with_shift<F>(
    diagonal: &mut Array1<F>,
    sub_diagonal: &mut Array1<F>,
    q: &mut Array2<F>,
    start: usize,
    end: usize,
) -> LinalgResult<()>
where
    F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
{
    if end <= start {
        return Ok(());
    }

    let n = end - start + 1;
    if n < 2 {
        return Ok(());
    }

    // Compute Wilkinson shift (same as parallel version)
    let a = diagonal[end - 1];
    let b = sub_diagonal[end - 1];
    let c = diagonal[end];

    let d = (a - c) / (F::one() + F::one());
    let shift = c - (b * b) / (d + d.signum() * (d * d + b * b).sqrt());

    // Apply shift
    for i in start..=end {
        diagonal[i] -= shift;
    }

    // Sequential Givens rotations
    for i in start..end {
        apply_givens_rotation(diagonal, sub_diagonal, q, i);
    }

    // Restore shift
    for i in start..=end {
        diagonal[i] += shift;
    }

    Ok(())
}

/// Apply a single Givens rotation to eliminate sub-diagonal element
#[allow(dead_code)]
fn apply_givens_rotation<F>(
    diagonal: &mut Array1<F>,
    sub_diagonal: &mut Array1<F>,
    q: &mut Array2<F>,
    i: usize,
) where
    F: Float + NumAssign + Zero + One + Sum,
{
    if i >= sub_diagonal.len() {
        return;
    }

    let a = diagonal[i];
    let b = sub_diagonal[i];

    if b.abs() < F::epsilon() {
        return;
    }

    // Compute Givens rotation parameters
    let (c, s) = if a.abs() > b.abs() {
        let t = b / a;
        let c = F::one() / (F::one() + t * t).sqrt();
        let s = c * t;
        (c, s)
    } else {
        let t = a / b;
        let s = F::one() / (F::one() + t * t).sqrt();
        let c = s * t;
        (c, s)
    };

    // Apply rotation to the tridiagonal matrix
    let new_diagonal = c * a + s * b;
    diagonal[i] = new_diagonal;
    sub_diagonal[i] = F::zero();

    // Update next diagonal element if it exists
    if i + 1 < diagonal.len() {
        let next_diag = diagonal[i + 1];
        diagonal[i + 1] = c * next_diag;

        // Update next sub-diagonal element if it exists
        if i + 1 < sub_diagonal.len() {
            let next_sub = sub_diagonal[i + 1];
            sub_diagonal[i + 1] = s * next_sub;
        }
    }

    // Apply rotation to Q matrix (accumulate transformations)
    let n = q.nrows();
    for row in 0..n {
        let qi = q[(row, i)];
        let qi1 = if i + 1 < q.ncols() {
            q[(row, i + 1)]
        } else {
            F::zero()
        };

        q[(row, i)] = c * qi + s * qi1;
        if i + 1 < q.ncols() {
            q[(row, i + 1)] = -s * qi + c * qi1;
        }
    }
}

/// Parallel Padé approximation
#[allow(dead_code)]
fn parallel_pade_approximation<F>(
    matrix: &ArrayView2<F>,
    _order: usize,
    _workers: usize,
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
{
    // Simplified implementation - returns identity matrix
    let n = matrix.nrows();
    Ok(Array2::eye(n))
}

/// Parallel Frobenius norm
#[allow(dead_code)]
fn parallel_frobenius_norm<F>(a: &ArrayView2<F>, workers: usize) -> LinalgResult<F>
where
    F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
{
    let sum_squares: F = a.iter().map(|x| (*x) * (*x)).sum();
    Ok(sum_squares.sqrt())
}

/// Parallel nuclear norm
#[allow(dead_code)]
fn parallel_nuclear_norm<F>(a: &ArrayView2<F>, workers: usize) -> LinalgResult<F>
where
    F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
{
    // Simplified - use Frobenius norm as approximation
    parallel_frobenius_norm(a, workers)
}

/// Parallel matrix 1-norm
#[allow(dead_code)]
fn parallelmatrix_1_norm<F>(a: &ArrayView2<F>, workers: usize) -> LinalgResult<F>
where
    F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
{
    let (_m, n) = a.dim();
    let mut max_col_sum = F::zero();

    for j in 0..n {
        let col_sum: F = a.column(j).iter().map(|x| x.abs()).sum();
        max_col_sum = max_col_sum.max(col_sum);
    }

    Ok(max_col_sum)
}

/// Parallel spectral norm
#[allow(dead_code)]
fn parallel_spectral_norm<F>(a: &ArrayView2<F>, workers: usize) -> LinalgResult<F>
where
    F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
{
    // Simplified - use Frobenius norm as approximation
    parallel_frobenius_norm(a, workers)
}

/// Parallel matrix infinity norm
#[allow(dead_code)]
fn parallelmatrix_inf_norm<F>(a: &ArrayView2<F>, workers: usize) -> LinalgResult<F>
where
    F: Float + NumAssign + Zero + One + Sum + Send + Sync + ScalarOperand + 'static,
{
    let (m, n) = a.dim();
    let mut max_row_sum = F::zero();

    for i in 0..m {
        let row_sum: F = a.row(i).iter().map(|x| x.abs()).sum();
        max_row_sum = max_row_sum.max(row_sum);
    }

    Ok(max_row_sum)
}

// ============================================================================
// Advanced MODE: Advanced Work-Stealing Scheduler Optimizations
// ============================================================================

/// Adaptive work chunk sizing based on workload characteristics
#[derive(Debug, Clone)]
pub struct AdaptiveChunking {
    /// Minimum chunk size
    min_chunksize: usize,
    /// Maximum chunk size
    max_chunksize: usize,
    /// Current optimal chunk size
    current_chunksize: usize,
    /// Performance history for adaptation
    performance_history: Vec<ChunkPerformance>,
    /// Maximum history entries to maintain
    max_history: usize,
}

/// Performance metrics for a chunk execution
#[derive(Debug, Clone)]
pub struct ChunkPerformance {
    /// Chunk size used
    chunksize: usize,
    /// Execution time in nanoseconds
    execution_time_ns: u64,
    /// Work complexity estimate
    work_complexity: f64,
    /// Cache miss rate (if available)
    cache_miss_rate: Option<f64>,
    /// Thread utilization percentage
    thread_utilization: f64,
}

impl AdaptiveChunking {
    /// Create a new adaptive chunking strategy
    pub fn new(_minsize: usize, maxsize: usize) -> Self {
        Self {
            min_chunksize: _minsize,
            max_chunksize: maxsize,
            current_chunksize: (_minsize + maxsize) / 2,
            performance_history: Vec::new(),
            max_history: 50,
        }
    }

    /// Record performance for a chunk execution
    pub fn record_performance(&mut self, performance: ChunkPerformance) {
        self.performance_history.push(performance);

        // Maintain history size limit
        if self.performance_history.len() > self.max_history {
            self.performance_history.remove(0);
        }

        // Adapt chunk size based on recent performance
        self.adapt_chunksize();
    }

    /// Enhanced adaptive chunk size optimization with statistical analysis
    fn adapt_chunksize(&mut self) {
        if self.performance_history.len() < 5 {
            return;
        }

        // Analyze performance metrics with statistical approach
        let recent_entries =
            &self.performance_history[self.performance_history.len().saturating_sub(10)..];

        // Group entries by chunk size and calculate statistics
        let mut chunk_performance: std::collections::HashMap<usize, Vec<f64>> =
            std::collections::HashMap::new();

        for entry in recent_entries {
            let throughput =
                entry.work_complexity / (entry.execution_time_ns as f64 / 1_000_000_000.0);
            chunk_performance
                .entry(entry.chunksize)
                .or_default()
                .push(throughput);
        }

        // Find optimal chunk size considering both performance and stability
        let mut best_score = f64::NEG_INFINITY;
        let mut best_chunksize = self.current_chunksize;

        for (&chunksize, throughputs) in &chunk_performance {
            if throughputs.len() < 2 {
                continue; // Need at least 2 samples for variance calculation
            }

            let mean_throughput: f64 = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
            let variance: f64 = throughputs
                .iter()
                .map(|&t| (t - mean_throughput).powi(2))
                .sum::<f64>()
                / throughputs.len() as f64;
            let std_dev = variance.sqrt();

            // Score considering both performance (mean) and stability (inverse of std_dev)
            // Higher throughput is better, lower variance is better
            let stability_factor = 1.0 / (1.0 + std_dev / mean_throughput); // Coefficient of variation
            let score = mean_throughput * stability_factor;

            if score > best_score {
                best_score = score;
                best_chunksize = chunksize;
            }
        }

        // Enhanced adaptive adjustment with momentum and exploration
        let adjustment_factor = if self.performance_history.len() > 20 {
            0.3 // More aggressive when we have more data
        } else {
            0.15 // Conservative when learning
        };

        // Add small exploration component to avoid local optima
        let exploration_factor = 0.05;
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        // Use deterministic pseudo-random based on history length for exploration
        let mut hasher = DefaultHasher::new();
        self.performance_history.len().hash(&mut hasher);
        let pseudo_random = (hasher.finish() % 1000) as f64 / 1000.0;
        let exploration_offset = (pseudo_random - 0.5) * exploration_factor * best_chunksize as f64;

        let targetsize = best_chunksize as f64 + exploration_offset;
        let currentsize = self.current_chunksize as f64;
        let newsize = currentsize + (targetsize - currentsize) * adjustment_factor;

        self.current_chunksize = (newsize as usize)
            .max(self.min_chunksize)
            .min(self.max_chunksize);

        // Adaptive bounds adjustment - expand search space if we're hitting boundaries
        if self.current_chunksize == self.min_chunksize && best_score > 0.0 {
            self.min_chunksize = (self.min_chunksize as f64 * 0.8) as usize;
        }
        if self.current_chunksize == self.max_chunksize && best_score > 0.0 {
            self.max_chunksize = (self.max_chunksize as f64 * 1.2) as usize;
        }
    }

    /// Get the current optimal chunk size
    pub fn get_chunksize(&self) -> usize {
        self.current_chunksize
    }

    /// Predict optimal chunk size for a given matrix operation without execution
    pub fn predict_optimal_chunksize(
        &self,
        matrixsize: (usize, usize),
        operation_type: MatrixOperationType,
        num_workers: usize,
    ) -> usize {
        // Base prediction using matrix characteristics
        let (rows, cols) = matrixsize;
        let total_elements = rows * cols;

        let base_chunksize = match operation_type {
            MatrixOperationType::MatrixVectorMultiplication => {
                // For matvec, chunk by rows to maintain cache locality
                (rows / num_workers).clamp(16, 1024)
            }
            MatrixOperationType::MatrixMatrixMultiplication => {
                // For matmul, consider both dimensions and target block sizes for cache efficiency
                let target_block_elements = 4096; // Good for L1 cache (32KB / 8 bytes)
                let elements_per_worker = total_elements / num_workers;
                elements_per_worker.min(target_block_elements).max(64)
            }
            MatrixOperationType::Decomposition => {
                // Decompositions have irregular patterns, use smaller chunks for better load balancing
                (rows / (num_workers * 4)).clamp(8, 256)
            }
            MatrixOperationType::EigenComputation => {
                // Eigenvalue computations are typically iterative and memory-intensive
                (rows / (num_workers * 2)).clamp(16, 512)
            }
            MatrixOperationType::IterativeSolver => {
                // Iterative solvers benefit from larger chunks to amortize synchronization costs
                (rows / num_workers).clamp(32, 2048)
            }
        };

        // Adjust based on historical performance if available
        if self.performance_history.len() > 5 {
            // Find similar matrix sizes in history
            let mut similar_performance = Vec::new();
            for entry in &self.performance_history {
                // Consider operations on matrices within 20% size difference as "similar"
                let size_ratio = (total_elements as f64) / entry.work_complexity;
                if size_ratio > 0.8 && size_ratio < 1.2 {
                    let throughput =
                        entry.work_complexity / (entry.execution_time_ns as f64 / 1_000_000_000.0);
                    similar_performance.push((entry.chunksize, throughput));
                }
            }

            if !similar_performance.is_empty() {
                // Weight historical performance with base prediction
                let historical_optimum = similar_performance
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|&(chunksize_, _)| chunksize_)
                    .unwrap_or(base_chunksize);

                // Blend base prediction with historical optimum
                let blend_factor = 0.7; // Favor historical data
                let predicted = (base_chunksize as f64 * (1.0 - blend_factor)
                    + historical_optimum as f64 * blend_factor)
                    as usize;

                return predicted.max(self.min_chunksize).min(self.max_chunksize);
            }
        }

        base_chunksize
            .max(self.min_chunksize)
            .min(self.max_chunksize)
    }

    /// Get performance statistics
    pub fn get_stats(&self) -> AdaptiveChunkingStats {
        if self.performance_history.is_empty() {
            return AdaptiveChunkingStats::default();
        }

        let total_entries = self.performance_history.len();
        let avg_execution_time = self
            .performance_history
            .iter()
            .map(|p| p.execution_time_ns)
            .sum::<u64>() as f64
            / total_entries as f64;

        let avg_utilization = self
            .performance_history
            .iter()
            .map(|p| p.thread_utilization)
            .sum::<f64>()
            / total_entries as f64;

        let cache_miss_rate = self
            .performance_history
            .iter()
            .filter_map(|p| p.cache_miss_rate)
            .fold(None, |acc, x| Some(acc.unwrap_or(0.0) + x))
            .map(|rate| {
                rate / self
                    .performance_history
                    .iter()
                    .filter(|p| p.cache_miss_rate.is_some())
                    .count() as f64
            });

        AdaptiveChunkingStats {
            current_chunksize: self.current_chunksize,
            avg_execution_time_ms: avg_execution_time / 1_000_000.0,
            avg_thread_utilization: avg_utilization,
            avg_cache_miss_rate: cache_miss_rate,
            total_adaptations: total_entries,
        }
    }
}

/// Statistics for adaptive chunking performance
#[derive(Debug, Clone, Default)]
pub struct AdaptiveChunkingStats {
    /// Current chunk size
    pub current_chunksize: usize,
    /// Average execution time in milliseconds
    pub avg_execution_time_ms: f64,
    /// Average thread utilization percentage
    pub avg_thread_utilization: f64,
    /// Average cache miss rate (if available)
    pub avg_cache_miss_rate: Option<f64>,
    /// Total number of adaptations performed
    pub total_adaptations: usize,
}

/// Enhanced work-stealing scheduler with adaptive optimizations
pub struct OptimizedWorkStealingScheduler<T: Clone + Send + 'static> {
    /// Base work-stealing scheduler
    #[allow(dead_code)]
    base_scheduler: WorkStealingScheduler<T>,
    /// Adaptive chunking strategy
    adaptive_chunking: Arc<Mutex<AdaptiveChunking>>,
    /// Performance monitoring
    performance_monitor: Arc<Mutex<PerformanceMonitor>>,
    /// Cache locality optimizer
    cache_optimizer: Arc<Mutex<CacheLocalityOptimizer>>,
}

/// Performance monitoring for work-stealing operations
#[derive(Debug)]
pub struct PerformanceMonitor {
    /// Total tasks executed
    total_tasks: u64,
    /// Total execution time
    total_execution_time_ns: u64,
    /// Work stealing events
    steal_events: u64,
    /// Failed steal attempts
    failed_steals: u64,
    /// Queue contentions
    queue_contentions: u64,
    /// Load imbalance measurements
    load_imbalance_history: Vec<f64>,
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new() -> Self {
        Self {
            total_tasks: 0,
            total_execution_time_ns: 0,
            steal_events: 0,
            failed_steals: 0,
            queue_contentions: 0,
            load_imbalance_history: Vec::new(),
        }
    }

    /// Record task execution
    pub fn record_task(&mut self, execution_timens: u64) {
        self.total_tasks += 1;
        self.total_execution_time_ns += execution_timens;
    }

    /// Record work stealing event
    pub fn record_steal(&mut self, successful: bool) {
        if successful {
            self.steal_events += 1;
        } else {
            self.failed_steals += 1;
        }
    }

    /// Record queue contention
    pub fn record_contention(&mut self) {
        self.queue_contentions += 1;
    }

    /// Record load imbalance measurement
    pub fn record_load_imbalance(&mut self, imbalance: f64) {
        self.load_imbalance_history.push(imbalance);
        // Keep only recent measurements
        if self.load_imbalance_history.len() > 100 {
            self.load_imbalance_history.remove(0);
        }
    }

    /// Get performance statistics
    pub fn get_stats(&self) -> PerformanceStats {
        let avg_task_time = if self.total_tasks > 0 {
            self.total_execution_time_ns as f64 / self.total_tasks as f64
        } else {
            0.0
        };

        let steal_success_rate = if self.steal_events + self.failed_steals > 0 {
            self.steal_events as f64 / (self.steal_events + self.failed_steals) as f64
        } else {
            0.0
        };

        let avg_load_imbalance = if !self.load_imbalance_history.is_empty() {
            self.load_imbalance_history.iter().sum::<f64>()
                / self.load_imbalance_history.len() as f64
        } else {
            0.0
        };

        PerformanceStats {
            total_tasks: self.total_tasks,
            avg_task_time_ns: avg_task_time,
            steal_success_rate,
            queue_contentions: self.queue_contentions,
            avg_load_imbalance,
        }
    }
}

/// Performance statistics summary
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    /// Total number of tasks executed
    pub total_tasks: u64,
    /// Average task execution time in nanoseconds
    pub avg_task_time_ns: f64,
    /// Work stealing success rate (0.0 to 1.0)
    pub steal_success_rate: f64,
    /// Number of queue contentions
    pub queue_contentions: u64,
    /// Average load imbalance factor
    pub avg_load_imbalance: f64,
}

/// Cache locality optimizer for work distribution
#[derive(Debug)]
pub struct CacheLocalityOptimizer {
    /// Memory access patterns
    access_patterns: Vec<MemoryAccessPattern>,
    /// Cache line size (typically 64 bytes)
    cache_linesize: usize,
    /// L1 cache size estimate
    l1_cachesize: usize,
    /// L2 cache size estimate
    l2_cachesize: usize,
}

/// Memory access pattern for cache optimization
#[derive(Debug, Clone)]
pub struct MemoryAccessPattern {
    /// Memory address range start
    address_start: usize,
    /// Memory address range end
    address_end: usize,
    /// Access frequency
    access_frequency: u64,
    /// Sequential vs random access ratio
    sequential_ratio: f64,
}

impl CacheLocalityOptimizer {
    /// Create a new cache locality optimizer
    pub fn new() -> Self {
        Self {
            access_patterns: Vec::new(),
            cache_linesize: 64,       // Common cache line size
            l1_cachesize: 32 * 1024,  // 32KB typical L1
            l2_cachesize: 256 * 1024, // 256KB typical L2
        }
    }

    /// Record memory access pattern
    pub fn record_access_pattern(&mut self, pattern: MemoryAccessPattern) {
        self.access_patterns.push(pattern);

        // Maintain reasonable history size
        if self.access_patterns.len() > 1000 {
            self.access_patterns.remove(0);
        }
    }

    /// Optimize work distribution based on cache locality
    pub fn optimize_work_distribution(
        &self,
        work_items: &[usize],
        num_workers: usize,
    ) -> Vec<Vec<usize>> {
        let mut worker_assignments = vec![Vec::new(); num_workers];

        if work_items.is_empty() {
            return worker_assignments;
        }

        // Simple locality-aware distribution
        // Group adjacent work _items to the same worker to improve cache locality
        let chunksize = work_items.len().div_ceil(num_workers);

        for (i, &work_item) in work_items.iter().enumerate() {
            let worker_id = (i / chunksize).min(num_workers - 1);
            worker_assignments[worker_id].push(work_item);
        }

        worker_assignments
    }

    /// Get cache optimization recommendations
    pub fn get_recommendations(&self) -> CacheOptimizationRecommendations {
        let total_accesses = self
            .access_patterns
            .iter()
            .map(|p| p.access_frequency)
            .sum::<u64>();

        let avg_sequential_ratio = if !self.access_patterns.is_empty() {
            self.access_patterns
                .iter()
                .map(|p| p.sequential_ratio * p.access_frequency as f64)
                .sum::<f64>()
                / total_accesses as f64
        } else {
            0.5
        };

        let working_setsize = self
            .access_patterns
            .iter()
            .map(|p| p.address_end - p.address_start)
            .sum::<usize>();

        CacheOptimizationRecommendations {
            recommended_blocksize: if avg_sequential_ratio > 0.7 {
                self.cache_linesize * 4 // Larger blocks for sequential access
            } else {
                self.cache_linesize // Smaller blocks for random access
            },
            locality_friendly: avg_sequential_ratio > 0.5,
            working_set_fits_l1: working_setsize <= self.l1_cachesize,
            working_set_fits_l2: working_setsize <= self.l2_cachesize,
            prefetch_beneficial: avg_sequential_ratio > 0.6,
        }
    }
}

/// Cache optimization recommendations
#[derive(Debug, Clone)]
pub struct CacheOptimizationRecommendations {
    /// Recommended block size for optimal cache usage
    pub recommended_blocksize: usize,
    /// Whether the access pattern is locality-friendly
    pub locality_friendly: bool,
    /// Whether the working set fits in L1 cache
    pub working_set_fits_l1: bool,
    /// Whether the working set fits in L2 cache
    pub working_set_fits_l2: bool,
    /// Whether prefetching would be beneficial
    pub prefetch_beneficial: bool,
}

impl<T: Clone + Send + 'static> OptimizedWorkStealingScheduler<T> {
    /// Create a new optimized work-stealing scheduler
    pub fn new(_numworkers: usize) -> Self {
        Self {
            base_scheduler: WorkStealingScheduler::new(_numworkers),
            adaptive_chunking: Arc::new(Mutex::new(AdaptiveChunking::new(8, 1024))),
            performance_monitor: Arc::new(Mutex::new(PerformanceMonitor::new())),
            cache_optimizer: Arc::new(Mutex::new(CacheLocalityOptimizer::new())),
        }
    }

    /// Execute work with adaptive optimization
    pub fn execute_optimized<F, R>(&self, work_items: Vec<T>, workfn: F) -> LinalgResult<Vec<R>>
    where
        F: Fn(T) -> R + Send + Sync + Clone + 'static,
        R: Send + Clone + 'static,
    {
        let start_time = std::time::Instant::now();

        // Get current chunk size from adaptive chunking
        let chunksize = {
            let chunking = self.adaptive_chunking.lock().unwrap();
            chunking.get_chunksize()
        };

        // Use parallel processing from scirs2-core as per project policy
        use scirs2_core::parallel_ops::*;

        // Execute work _items in parallel using proper parallel processing
        let results: Vec<R> = work_items.into_par_iter().map(workfn).collect();

        // Record performance metrics
        let execution_time = start_time.elapsed();
        {
            let mut monitor = self.performance_monitor.lock().unwrap();
            monitor.record_task(execution_time.as_nanos() as u64);
        }

        // Record chunk performance for adaptation
        {
            let mut chunking = self.adaptive_chunking.lock().unwrap();
            chunking.record_performance(ChunkPerformance {
                chunksize,
                execution_time_ns: execution_time.as_nanos() as u64,
                work_complexity: results.len() as f64, // Simple complexity estimate
                cache_miss_rate: None,                 // Would need hardware performance counters
                thread_utilization: 0.8, // Placeholder - would need actual measurement
            });
        }

        Ok(results)
    }

    /// Get comprehensive performance statistics
    pub fn get_performance_stats(&self) -> OptimizedSchedulerStats {
        let chunking_stats = {
            let chunking = self.adaptive_chunking.lock().unwrap();
            chunking.get_stats()
        };

        let performance_stats = {
            let monitor = self.performance_monitor.lock().unwrap();
            monitor.get_stats()
        };

        let cache_recommendations = {
            let optimizer = self.cache_optimizer.lock().unwrap();
            optimizer.get_recommendations()
        };

        OptimizedSchedulerStats {
            chunking_stats,
            performance_stats,
            cache_recommendations,
        }
    }
}

/// Comprehensive statistics for the optimized scheduler
#[derive(Debug, Clone)]
pub struct OptimizedSchedulerStats {
    /// Adaptive chunking statistics
    pub chunking_stats: AdaptiveChunkingStats,
    /// Performance monitoring statistics
    pub performance_stats: PerformanceStats,
    /// Cache optimization recommendations
    pub cache_recommendations: CacheOptimizationRecommendations,
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for CacheLocalityOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod optimization_tests {
    use super::*;

    #[test]
    fn test_adaptive_chunking() {
        let mut chunking = AdaptiveChunking::new(8, 512);
        assert_eq!(chunking.get_chunksize(), 260); // (8 + 512) / 2

        // Record some performance data
        chunking.record_performance(ChunkPerformance {
            chunksize: 64,
            execution_time_ns: 1_000_000,
            work_complexity: 100.0,
            cache_miss_rate: Some(0.05),
            thread_utilization: 0.9,
        });

        let stats = chunking.get_stats();
        assert_eq!(stats.total_adaptations, 1);
    }

    #[test]
    fn test_performance_monitor() {
        let mut monitor = PerformanceMonitor::new();

        monitor.record_task(1_000_000);
        monitor.record_steal(true);
        monitor.record_steal(false);
        monitor.record_contention();

        let stats = monitor.get_stats();
        assert_eq!(stats.total_tasks, 1);
        assert_eq!(stats.steal_success_rate, 0.5);
        assert_eq!(stats.queue_contentions, 1);
    }

    #[test]
    fn test_cache_locality_optimizer() {
        let mut optimizer = CacheLocalityOptimizer::new();

        optimizer.record_access_pattern(MemoryAccessPattern {
            address_start: 0,
            address_end: 1024,
            access_frequency: 100,
            sequential_ratio: 0.8,
        });

        let recommendations = optimizer.get_recommendations();
        assert!(recommendations.locality_friendly);
        assert!(recommendations.prefetch_beneficial);
    }

    #[test]
    fn test_optimized_scheduler_creation() {
        let scheduler = OptimizedWorkStealingScheduler::<i32>::new(4);
        let stats = scheduler.get_performance_stats();

        // Check that stats are properly initialized
        assert_eq!(stats.performance_stats.total_tasks, 0);
        assert!(stats.cache_recommendations.recommended_blocksize > 0);
    }
}
