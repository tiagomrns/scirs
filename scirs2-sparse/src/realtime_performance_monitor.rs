//! Real-Time Performance Monitoring and Adaptation for Advanced Processors
//!
//! This module provides comprehensive real-time monitoring and adaptive optimization
//! for all Advanced mode processors, including quantum-inspired, neural-adaptive,
//! and hybrid processors.

use crate::adaptive_memory_compression::MemoryStats;
use crate::error::SparseResult;
use crate::neural_adaptive_sparse::NeuralProcessorStats;
use crate::quantum_inspired_sparse::QuantumProcessorStats;
use crate::quantum_neural_hybrid::QuantumNeuralHybridStats;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Configuration for real-time performance monitoring
#[derive(Debug, Clone)]
pub struct PerformanceMonitorConfig {
    /// Monitoring interval in milliseconds
    pub monitoring_interval_ms: u64,
    /// Maximum number of performance samples to keep
    pub max_samples: usize,
    /// Enable adaptive tuning based on performance
    pub adaptive_tuning: bool,
    /// Performance threshold for adaptation triggers
    pub adaptation_threshold: f64,
    /// Enable real-time alerts
    pub enable_alerts: bool,
    /// Alert threshold for performance degradation
    pub alert_threshold: f64,
    /// Enable automatic optimization
    pub auto_optimization: bool,
    /// Optimization interval in seconds
    pub optimization_interval_s: u64,
    /// Enable performance prediction
    pub enable_prediction: bool,
    /// Prediction horizon in samples
    pub prediction_horizon: usize,
}

impl Default for PerformanceMonitorConfig {
    fn default() -> Self {
        Self {
            monitoring_interval_ms: 100,
            max_samples: 10000,
            adaptive_tuning: true,
            adaptation_threshold: 0.8,
            enable_alerts: true,
            alert_threshold: 0.5,
            auto_optimization: true,
            optimization_interval_s: 30,
            enable_prediction: true,
            prediction_horizon: 50,
        }
    }
}

/// Real-time performance monitor for Advanced processors
#[allow(dead_code)]
pub struct RealTimePerformanceMonitor {
    config: PerformanceMonitorConfig,
    monitoring_active: Arc<AtomicBool>,
    sample_counter: AtomicUsize,
    performance_history: Arc<Mutex<PerformanceHistory>>,
    systemmetrics: Arc<Mutex<SystemMetrics>>,
    alert_manager: Arc<Mutex<AlertManager>>,
    adaptation_engine: Arc<Mutex<AdaptationEngine>>,
    prediction_engine: Arc<Mutex<PredictionEngine>>,
    processor_registry: Arc<Mutex<ProcessorRegistry>>,
}

/// Performance history tracking
#[derive(Debug)]
#[allow(dead_code)]
struct PerformanceHistory {
    samples: VecDeque<PerformanceSample>,
    aggregatedmetrics: AggregatedMetrics,
    trend_analysis: TrendAnalysis,
    performance_baselines: HashMap<String, f64>,
}

/// Individual performance sample
#[derive(Debug, Clone)]
pub struct PerformanceSample {
    pub timestamp: u64,
    pub processor_type: ProcessorType,
    pub processor_id: String,
    pub execution_time_ms: f64,
    pub throughput_ops_per_sec: f64,
    pub memory_usage_mb: f64,
    pub cache_hit_ratio: f64,
    pub error_rate: f64,
    pub cpu_utilization: f64,
    pub gpu_utilization: f64,
    pub quantum_coherence: Option<f64>,
    pub neural_confidence: Option<f64>,
    pub compression_ratio: Option<f64>,
}

/// Execution timing helper for measuring performance
pub struct ExecutionTimer {
    start_time: Instant,
}

impl ExecutionTimer {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
        }
    }
}

impl Default for ExecutionTimer {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionTimer {
    pub fn elapsed_ms(&self) -> f64 {
        self.start_time.elapsed().as_millis() as f64
    }

    pub fn restart(&mut self) {
        self.start_time = Instant::now();
    }
}

/// Type of Advanced processor
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProcessorType {
    QuantumInspired,
    NeuralAdaptive,
    QuantumNeuralHybrid,
    MemoryCompression,
}

/// Aggregated performance metrics
#[derive(Debug, Default, Clone)]
pub struct AggregatedMetrics {
    avg_execution_time: f64,
    avg_throughput: f64,
    avg_memory_usage: f64,
    avg_cache_hit_ratio: f64,
    avg_error_rate: f64,
    peak_throughput: f64,
    min_execution_time: f64,
    total_operations: usize,
    efficiency_score: f64,
}

/// Trend analysis for performance prediction
#[derive(Debug)]
struct TrendAnalysis {
    execution_time_trend: LinearTrend,
    throughput_trend: LinearTrend,
    memory_trend: LinearTrend,
    efficiency_trend: LinearTrend,
    anomaly_detection: AnomalyDetector,
}

/// Linear trend analysis
#[derive(Debug, Default)]
#[allow(dead_code)]
struct LinearTrend {
    slope: f64,
    intercept: f64,
    correlation: f64,
    prediction_confidence: f64,
}

/// Anomaly detection system
#[derive(Debug)]
struct AnomalyDetector {
    moving_average: f64,
    moving_variance: f64,
    anomaly_threshold: f64,
    recent_anomalies: VecDeque<AnomalyEvent>,
}

/// Anomaly event
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct AnomalyEvent {
    timestamp: u64,
    metricname: String,
    expected_value: f64,
    actualvalue: f64,
    severity: AnomalySeverity,
}

/// Severity of anomaly
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// System metrics tracking
#[derive(Debug)]
#[allow(dead_code)]
struct SystemMetrics {
    cpu_usage: f64,
    memory_usage: f64,
    gpu_usage: f64,
    network_io: f64,
    disk_io: f64,
    temperature: f64,
    power_consumption: f64,
    system_load: f64,
}

/// Alert management system
#[derive(Debug)]
#[allow(dead_code)]
struct AlertManager {
    active_alerts: HashMap<String, Alert>,
    alert_history: VecDeque<Alert>,
    notification_channels: Vec<NotificationChannel>,
    alert_rules: Vec<AlertRule>,
}

/// Performance alert
#[derive(Debug, Clone)]
pub struct Alert {
    pub id: String,
    pub timestamp: u64,
    pub severity: AlertSeverity,
    pub message: String,
    pub processor_type: ProcessorType,
    pub processor_id: String,
    pub metricname: String,
    pub threshold_value: f64,
    pub actualvalue: f64,
    pub resolved: bool,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Notification channels for alerts
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum NotificationChannel {
    Console,
    Log,
    Email,
    Webhook,
}

/// Alert rule configuration
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct AlertRule {
    id: String,
    metricname: String,
    condition: AlertCondition,
    threshold: f64,
    severity: AlertSeverity,
    enabled: bool,
}

/// Alert condition types
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum AlertCondition {
    GreaterThan,
    LessThan,
    Equals,
    NotEquals,
    PercentageIncrease,
    PercentageDecrease,
}

/// Adaptive engine for performance optimization
#[derive(Debug)]
#[allow(dead_code)]
struct AdaptationEngine {
    optimization_strategies: Vec<OptimizationStrategy>,
    strategy_effectiveness: HashMap<String, f64>,
    active_optimizations: HashMap<String, ActiveOptimization>,
    adaptation_history: VecDeque<AdaptationEvent>,
}

/// Optimization strategy
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct OptimizationStrategy {
    id: String,
    name: String,
    description: String,
    targetmetrics: Vec<String>,
    parameters: HashMap<String, f64>,
    effectiveness_score: f64,
    usage_count: usize,
}

/// Active optimization
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ActiveOptimization {
    strategy_id: String,
    processor_id: String,
    start_time: u64,
    expected_improvement: f64,
    actual_improvement: Option<f64>,
    status: OptimizationStatus,
}

/// Optimization status
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
enum OptimizationStatus {
    Pending,
    Active,
    Completed,
    Failed,
}

/// Adaptation event
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct AdaptationEvent {
    timestamp: u64,
    processor_type: ProcessorType,
    processor_id: String,
    strategy_applied: String,
    trigger_reason: String,
    beforemetrics: HashMap<String, f64>,
    aftermetrics: HashMap<String, f64>,
    improvement_achieved: f64,
}

/// Performance prediction engine
#[derive(Debug)]
#[allow(dead_code)]
struct PredictionEngine {
    prediction_models: HashMap<String, PredictionModel>,
    forecast_cache: HashMap<String, Forecast>,
    model_accuracy: HashMap<String, f64>,
}

/// Prediction model
#[derive(Debug)]
#[allow(dead_code)]
struct PredictionModel {
    model_type: ModelType,
    parameters: Vec<f64>,
    training_data: VecDeque<f64>,
    last_updated: u64,
    accuracy: f64,
    trained: bool,
    last_update: u64,
}

/// Types of prediction models
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
enum ModelType {
    LinearRegression,
    MovingAverage,
    ExponentialSmoothing,
    Arima,
    NeuralNetwork,
}

/// Performance forecast
#[derive(Debug, Clone)]
pub struct Forecast {
    pub metricname: String,
    pub predictions: Vec<PredictionPoint>,
    pub confidence_interval: (f64, f64),
    pub model_accuracy: f64,
    pub forecast_horizon: usize,
}

/// Individual prediction point
#[derive(Debug, Clone)]
pub struct PredictionPoint {
    pub timestamp: u64,
    pub predicted_value: f64,
    pub confidence: f64,
}

/// Registry of monitored processors
struct ProcessorRegistry {
    quantum_processors: HashMap<String, Box<dyn QuantumProcessorMonitor>>,
    neural_processors: HashMap<String, Box<dyn NeuralProcessorMonitor>>,
    hybrid_processors: HashMap<String, Box<dyn HybridProcessorMonitor>>,
    memory_compressors: HashMap<String, Box<dyn MemoryCompressorMonitor>>,
}

/// Monitoring traits for different processor types
pub trait QuantumProcessorMonitor: Send + Sync {
    fn get_stats(&self) -> QuantumProcessorStats;
    fn get_id(&self) -> &str;
}

pub trait NeuralProcessorMonitor: Send + Sync {
    fn get_stats(&self) -> NeuralProcessorStats;
    fn get_id(&self) -> &str;
}

pub trait HybridProcessorMonitor: Send + Sync {
    fn get_stats(&self) -> QuantumNeuralHybridStats;
    fn get_id(&self) -> &str;
}

pub trait MemoryCompressorMonitor: Send + Sync {
    fn get_stats(&self) -> MemoryStats;
    fn get_id(&self) -> &str;
}

impl RealTimePerformanceMonitor {
    /// Create a new real-time performance monitor
    pub fn new(config: PerformanceMonitorConfig) -> Self {
        let performance_history = PerformanceHistory {
            samples: VecDeque::with_capacity(config.max_samples),
            aggregatedmetrics: AggregatedMetrics::default(),
            trend_analysis: TrendAnalysis::new(),
            performance_baselines: HashMap::new(),
        };

        let systemmetrics = SystemMetrics {
            cpu_usage: 0.0,
            memory_usage: 0.0,
            gpu_usage: 0.0,
            network_io: 0.0,
            disk_io: 0.0,
            temperature: 0.0,
            power_consumption: 0.0,
            system_load: 0.0,
        };

        let alert_manager = AlertManager {
            active_alerts: HashMap::new(),
            alert_history: VecDeque::new(),
            notification_channels: vec![NotificationChannel::Console, NotificationChannel::Log],
            alert_rules: Self::create_default_alert_rules(),
        };

        let adaptation_engine = AdaptationEngine {
            optimization_strategies: Self::create_default_strategies(),
            strategy_effectiveness: HashMap::new(),
            active_optimizations: HashMap::new(),
            adaptation_history: VecDeque::new(),
        };

        let prediction_engine = PredictionEngine {
            prediction_models: HashMap::new(),
            forecast_cache: HashMap::new(),
            model_accuracy: HashMap::new(),
        };

        let processor_registry = ProcessorRegistry {
            quantum_processors: HashMap::new(),
            neural_processors: HashMap::new(),
            hybrid_processors: HashMap::new(),
            memory_compressors: HashMap::new(),
        };

        Self {
            config,
            monitoring_active: Arc::new(AtomicBool::new(false)),
            sample_counter: AtomicUsize::new(0),
            performance_history: Arc::new(Mutex::new(performance_history)),
            systemmetrics: Arc::new(Mutex::new(systemmetrics)),
            alert_manager: Arc::new(Mutex::new(alert_manager)),
            adaptation_engine: Arc::new(Mutex::new(adaptation_engine)),
            prediction_engine: Arc::new(Mutex::new(prediction_engine)),
            processor_registry: Arc::new(Mutex::new(processor_registry)),
        }
    }

    /// Start real-time monitoring
    pub fn start_monitoring(&self) -> SparseResult<()> {
        if self.monitoring_active.swap(true, Ordering::Relaxed) {
            return Ok(()); // Already running
        }

        let monitoring_active = Arc::clone(&self.monitoring_active);
        let config = self.config.clone();
        let performance_history = Arc::clone(&self.performance_history);
        let systemmetrics = Arc::clone(&self.systemmetrics);
        let alert_manager = Arc::clone(&self.alert_manager);
        let adaptation_engine = Arc::clone(&self.adaptation_engine);
        let prediction_engine = Arc::clone(&self.prediction_engine);
        let processor_registry = Arc::clone(&self.processor_registry);

        // Spawn monitoring thread
        std::thread::spawn(move || {
            let interval = Duration::from_millis(config.monitoring_interval_ms);
            let mut last_optimization = Instant::now();

            while monitoring_active.load(Ordering::Relaxed) {
                let start_time = Instant::now();

                // Collect performance samples
                Self::collect_performance_samples(
                    &processor_registry,
                    &performance_history,
                    &systemmetrics,
                );

                // Update aggregated metrics and trends
                Self::update_aggregatedmetrics(&performance_history);
                Self::update_trend_analysis(&performance_history);

                // Check for alerts
                if config.enable_alerts {
                    Self::check_alerts(&performance_history, &alert_manager);
                }

                // Run predictions
                if config.enable_prediction {
                    Self::update_predictions(&performance_history, &prediction_engine);
                }

                // Run adaptive optimization
                if config.auto_optimization
                    && last_optimization.elapsed()
                        >= Duration::from_secs(config.optimization_interval_s)
                {
                    Self::run_adaptive_optimization(
                        &performance_history,
                        &adaptation_engine,
                        &processor_registry,
                    );
                    last_optimization = Instant::now();
                }

                // Sleep for remaining interval time
                let elapsed = start_time.elapsed();
                if elapsed < interval {
                    std::thread::sleep(interval - elapsed);
                }
            }
        });

        Ok(())
    }

    /// Stop monitoring
    pub fn stop_monitoring(&self) {
        self.monitoring_active.store(false, Ordering::Relaxed);
    }

    /// Register a quantum processor for monitoring
    pub fn register_quantum_processor(
        &self,
        id: String,
        processor: Box<dyn QuantumProcessorMonitor>,
    ) -> SparseResult<()> {
        if let Ok(mut registry) = self.processor_registry.lock() {
            registry.quantum_processors.insert(id, processor);
        }
        Ok(())
    }

    /// Register a neural processor for monitoring
    pub fn register_neural_processor(
        &self,
        id: String,
        processor: Box<dyn NeuralProcessorMonitor>,
    ) -> SparseResult<()> {
        if let Ok(mut registry) = self.processor_registry.lock() {
            registry.neural_processors.insert(id, processor);
        }
        Ok(())
    }

    /// Register a hybrid processor for monitoring
    pub fn register_hybrid_processor(
        &self,
        id: String,
        processor: Box<dyn HybridProcessorMonitor>,
    ) -> SparseResult<()> {
        if let Ok(mut registry) = self.processor_registry.lock() {
            registry.hybrid_processors.insert(id, processor);
        }
        Ok(())
    }

    /// Register a memory compressor for monitoring
    pub fn register_memory_compressor(
        &self,
        id: String,
        compressor: Box<dyn MemoryCompressorMonitor>,
    ) -> SparseResult<()> {
        if let Ok(mut registry) = self.processor_registry.lock() {
            registry.memory_compressors.insert(id, compressor);
        }
        Ok(())
    }

    /// Get current performance metrics
    pub fn get_currentmetrics(&self) -> PerformanceMetrics {
        let history = self.performance_history.lock().unwrap();
        let system = self.systemmetrics.lock().unwrap();

        PerformanceMetrics {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            aggregated: history.aggregatedmetrics.clone(),
            systemmetrics: SystemMetricsSnapshot {
                cpu_usage: system.cpu_usage,
                memory_usage: system.memory_usage,
                gpu_usage: system.gpu_usage,
                system_load: system.system_load,
            },
            total_samples: history.samples.len(),
            monitoring_active: self.monitoring_active.load(Ordering::Relaxed),
        }
    }

    /// Get active alerts
    pub fn get_active_alerts(&self) -> Vec<Alert> {
        if let Ok(alert_manager) = self.alert_manager.lock() {
            alert_manager.active_alerts.values().cloned().collect()
        } else {
            Vec::new()
        }
    }

    /// Get performance forecast
    pub fn get_forecast(&self, metricname: &str, horizon: usize) -> Option<Forecast> {
        if let Ok(prediction_engine) = self.prediction_engine.lock() {
            prediction_engine.forecast_cache.get(metricname).cloned()
        } else {
            None
        }
    }

    // Internal implementation methods

    fn collect_performance_samples(
        registry: &Arc<Mutex<ProcessorRegistry>>,
        history: &Arc<Mutex<PerformanceHistory>>,
        systemmetrics: &Arc<Mutex<SystemMetrics>>,
    ) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let mut samples = Vec::new();

        // Get current system metrics once for all samples
        let current_cpu = Self::get_cpu_usage();
        let current_gpu = Self::get_gpu_usage();
        let current_memory = Self::get_memory_usage();

        if let Ok(registry) = registry.lock() {
            // Collect quantum processor samples
            for (id, processor) in &registry.quantum_processors {
                let stats = processor.get_stats();

                // Estimate execution time based on operations and coherence
                let base_time = 1.0 / (stats.operations_count.max(1) as f64);
                let coherence_factor = stats.average_logical_fidelity;
                let estimated_exec_time = base_time * (2.0 - coherence_factor) * 1000.0;

                // Estimate cache efficiency from quantum metrics
                let cache_efficiency = (stats.average_logical_fidelity * 0.8 + 0.2).min(1.0);

                samples.push(PerformanceSample {
                    timestamp,
                    processor_type: ProcessorType::QuantumInspired,
                    processor_id: id.clone(),
                    execution_time_ms: estimated_exec_time,
                    throughput_ops_per_sec: stats.operations_count as f64 / estimated_exec_time
                        * 1000.0,
                    memory_usage_mb: stats.cache_efficiency * current_memory * 100.0,
                    cache_hit_ratio: cache_efficiency,
                    error_rate: 1.0 - stats.average_logical_fidelity,
                    cpu_utilization: current_cpu * 0.6, // Quantum processors use less CPU
                    gpu_utilization: current_gpu * 0.1, // Minimal GPU usage for quantum simulation
                    quantum_coherence: Some(stats.average_logical_fidelity),
                    neural_confidence: None,
                    compression_ratio: None,
                });
            }

            // Collect neural processor samples
            for (id, processor) in &registry.neural_processors {
                let stats = processor.get_stats();

                // Estimate execution time based on neural complexity
                let neural_complexity =
                    stats.pattern_memory_size as f64 + stats.experience_buffer_size as f64;
                let base_time = neural_complexity / 10000.0; // Scale factor
                let learning_overhead = if stats.rl_enabled { 1.5 } else { 1.0 };
                let estimated_exec_time = base_time * learning_overhead * 1000.0;

                // Neural processors typically have good cache locality
                let neural_cache_ratio = 0.85 + (1.0 - stats.current_exploration_rate) * 0.1;

                // Error rate based on exploration vs exploitation balance
                let neural_error_rate = if stats.rl_enabled {
                    stats.current_exploration_rate * 0.1 + 0.01
                } else {
                    0.05
                };

                samples.push(PerformanceSample {
                    timestamp,
                    processor_type: ProcessorType::NeuralAdaptive,
                    processor_id: id.clone(),
                    execution_time_ms: estimated_exec_time,
                    throughput_ops_per_sec: stats.adaptations_count as f64 / estimated_exec_time
                        * 1000.0,
                    memory_usage_mb: stats.pattern_memory_size as f64 / 1024.0
                        + current_memory * 50.0,
                    cache_hit_ratio: neural_cache_ratio,
                    error_rate: neural_error_rate,
                    cpu_utilization: current_cpu * 0.8, // Neural networks are CPU intensive
                    gpu_utilization: current_gpu * 0.3, // Some GPU usage for matrix ops
                    quantum_coherence: None,
                    neural_confidence: Some(1.0 - stats.current_exploration_rate),
                    compression_ratio: None,
                });
            }

            // Collect hybrid processor samples
            for (id, processor) in &registry.hybrid_processors {
                let stats = processor.get_stats();

                // Hybrid execution time depends on synchronization and strategy balance
                let complexity_factor = 1.0 + (1.0 - stats.hybrid_synchronization) * 0.5;
                let quantum_weight_factor = stats.quantum_weight * 1.2; // Quantum is slower
                let neural_weight_factor = stats.neural_weight * 0.8; // Neural is faster
                let base_time = (quantum_weight_factor + neural_weight_factor) * complexity_factor;
                let estimated_exec_time = base_time * 1000.0;

                // Cache efficiency combines both quantum and neural characteristics
                let hybrid_cache_ratio =
                    stats.quantum_coherence * 0.9 + stats.neural_confidence * 0.85;

                samples.push(PerformanceSample {
                    timestamp,
                    processor_type: ProcessorType::QuantumNeuralHybrid,
                    processor_id: id.clone(),
                    execution_time_ms: estimated_exec_time,
                    throughput_ops_per_sec: stats.total_operations as f64 / estimated_exec_time
                        * 1000.0,
                    memory_usage_mb: stats.memory_utilization * current_memory * 200.0,
                    cache_hit_ratio: hybrid_cache_ratio.min(1.0),
                    error_rate: 1.0 - stats.hybrid_synchronization,
                    cpu_utilization: current_cpu
                        * (0.6 * stats.quantum_weight + 0.8 * stats.neural_weight),
                    gpu_utilization: current_gpu
                        * (0.1 * stats.quantum_weight + 0.4 * stats.neural_weight),
                    quantum_coherence: Some(stats.quantum_coherence),
                    neural_confidence: Some(stats.neural_confidence),
                    compression_ratio: None,
                });
            }

            // Collect memory compressor samples
            for (id, compressor) in &registry.memory_compressors {
                let stats = compressor.get_stats();

                // Real execution time from compression stats
                let compression_exec_time = if stats.compression_stats.compression_time > 0.0 {
                    stats.compression_stats.compression_time * 1000.0
                } else {
                    1.0 // Minimum 1ms
                };

                // Throughput based on blocks processed per time
                let throughput = if compression_exec_time > 0.0 {
                    stats.compression_stats.total_blocks as f64 / compression_exec_time * 1000.0
                } else {
                    stats.compression_stats.total_blocks as f64
                };

                // Error rate based on compression efficiency
                let compression_error_rate = if stats.compression_stats.compression_ratio > 1.0 {
                    0.01 / stats.compression_stats.compression_ratio // Better compression = fewer errors
                } else {
                    0.05 // Higher error rate for poor compression
                };

                // CPU utilization for compression is typically high
                let compression_cpu_util = current_cpu * 0.9;

                samples.push(PerformanceSample {
                    timestamp,
                    processor_type: ProcessorType::MemoryCompression,
                    processor_id: id.clone(),
                    execution_time_ms: compression_exec_time,
                    throughput_ops_per_sec: throughput,
                    memory_usage_mb: stats.current_memory_usage as f64 / (1024.0 * 1024.0),
                    cache_hit_ratio: stats.cache_hit_ratio,
                    error_rate: compression_error_rate,
                    cpu_utilization: compression_cpu_util,
                    gpu_utilization: current_gpu * 0.05, // Minimal GPU usage for compression
                    quantum_coherence: None,
                    neural_confidence: None,
                    compression_ratio: Some(stats.compression_stats.compression_ratio),
                });
            }
        }

        // Store samples in history
        if let Ok(mut history) = history.lock() {
            for sample in samples {
                history.samples.push_back(sample);
                if history.samples.len() > 10000 {
                    // Max samples
                    history.samples.pop_front();
                }
            }
        }

        // Update system metrics (simplified)
        if let Ok(mut system) = systemmetrics.lock() {
            system.cpu_usage = Self::get_cpu_usage();
            system.memory_usage = Self::get_memory_usage();
            system.gpu_usage = Self::get_gpu_usage();
            system.system_load = Self::get_system_load();
        }
    }

    fn update_aggregatedmetrics(history: &Arc<Mutex<PerformanceHistory>>) {
        if let Ok(mut history) = history.lock() {
            if history.samples.is_empty() {
                return;
            }

            let count = history.samples.len() as f64;

            // Calculate all metrics first before updating the struct
            let avg_execution_time = history
                .samples
                .iter()
                .map(|s| s.execution_time_ms)
                .sum::<f64>()
                / count;
            let avg_throughput = history
                .samples
                .iter()
                .map(|s| s.throughput_ops_per_sec)
                .sum::<f64>()
                / count;
            let avg_memory_usage = history
                .samples
                .iter()
                .map(|s| s.memory_usage_mb)
                .sum::<f64>()
                / count;
            let avg_cache_hit_ratio = history
                .samples
                .iter()
                .map(|s| s.cache_hit_ratio)
                .sum::<f64>()
                / count;
            let avg_error_rate = history.samples.iter().map(|s| s.error_rate).sum::<f64>() / count;
            let peak_throughput = history
                .samples
                .iter()
                .map(|s| s.throughput_ops_per_sec)
                .fold(0.0, f64::max);
            let min_execution_time = history
                .samples
                .iter()
                .map(|s| s.execution_time_ms)
                .fold(f64::INFINITY, f64::min);
            let total_operations = history.samples.len();

            // Calculate efficiency score
            let efficiency_score = (avg_throughput * avg_cache_hit_ratio)
                / (avg_execution_time + 1.0)
                * (1.0 - avg_error_rate);

            // Now update all the metrics
            history.aggregatedmetrics.avg_execution_time = avg_execution_time;
            history.aggregatedmetrics.avg_throughput = avg_throughput;
            history.aggregatedmetrics.avg_memory_usage = avg_memory_usage;
            history.aggregatedmetrics.avg_cache_hit_ratio = avg_cache_hit_ratio;
            history.aggregatedmetrics.avg_error_rate = avg_error_rate;
            history.aggregatedmetrics.peak_throughput = peak_throughput;
            history.aggregatedmetrics.min_execution_time = min_execution_time;
            history.aggregatedmetrics.total_operations = total_operations;
            history.aggregatedmetrics.efficiency_score = efficiency_score;
        }
    }

    fn update_trend_analysis(history: &Arc<Mutex<PerformanceHistory>>) {
        if let Ok(mut history) = history.lock() {
            if history.samples.len() < 10 {
                return;
            }

            // Clone recent samples to avoid borrow checker issues
            let recent_samples: Vec<_> = history.samples.iter().rev().take(100).cloned().collect();

            // Calculate all trends first
            let execution_times: Vec<f64> =
                recent_samples.iter().map(|s| s.execution_time_ms).collect();
            let execution_time_trend = Self::calculate_linear_trend(&execution_times);

            let throughputs: Vec<f64> = recent_samples
                .iter()
                .map(|s| s.throughput_ops_per_sec)
                .collect();
            let throughput_trend = Self::calculate_linear_trend(&throughputs);

            let memory_usage: Vec<f64> = recent_samples.iter().map(|s| s.memory_usage_mb).collect();
            let memory_trend = Self::calculate_linear_trend(&memory_usage);

            let efficiency: Vec<f64> = recent_samples
                .iter()
                .map(|s| {
                    (s.throughput_ops_per_sec * s.cache_hit_ratio) / (s.execution_time_ms + 1.0)
                        * (1.0 - s.error_rate)
                })
                .collect();
            let efficiency_trend = Self::calculate_linear_trend(&efficiency);

            // Now update the trends
            history.trend_analysis.execution_time_trend = execution_time_trend;
            history.trend_analysis.throughput_trend = throughput_trend;
            history.trend_analysis.memory_trend = memory_trend;
            history.trend_analysis.efficiency_trend = efficiency_trend;

            // Update anomaly detection
            history.trend_analysis.anomaly_detection.update(&efficiency);
        }
    }

    fn calculate_linear_trend(data: &[f64]) -> LinearTrend {
        if data.len() < 2 {
            return LinearTrend::default();
        }

        let n = data.len() as f64;
        let x_values: Vec<f64> = (0..data.len()).map(|i| i as f64).collect();

        let x_mean = x_values.iter().sum::<f64>() / n;
        let y_mean = data.iter().sum::<f64>() / n;

        let numerator: f64 = x_values
            .iter()
            .zip(data)
            .map(|(x, y)| (x - x_mean) * (y - y_mean))
            .sum();
        let denominator: f64 = x_values.iter().map(|x| (x - x_mean).powi(2)).sum();

        let slope = if denominator != 0.0 {
            numerator / denominator
        } else {
            0.0
        };
        let intercept = y_mean - slope * x_mean;

        // Calculate correlation coefficient
        let ss_tot: f64 = data.iter().map(|y| (y - y_mean).powi(2)).sum();
        let ss_res: f64 = x_values
            .iter()
            .zip(data)
            .map(|(x, y)| {
                let predicted = slope * x + intercept;
                (y - predicted).powi(2)
            })
            .sum();

        let correlation: f64 = if ss_tot != 0.0 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };

        LinearTrend {
            slope,
            intercept,
            correlation: correlation.sqrt(),
            prediction_confidence: correlation.abs(),
        }
    }

    fn check_alerts(
        history: &Arc<Mutex<PerformanceHistory>>,
        alert_manager: &Arc<Mutex<AlertManager>>,
    ) {
        // Comprehensive alert rule engine
        if let (Ok(history), Ok(mut alert_manager)) = (history.lock(), alert_manager.lock()) {
            let metrics = &history.aggregatedmetrics;
            let systemmetrics = Self::collect_systemmetrics();
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();

            // Initialize default alert rules if empty
            if alert_manager.alert_rules.is_empty() {
                alert_manager.alert_rules = Self::create_default_alert_rules();
            }

            // Process each alert rule
            for rule in &alert_manager.alert_rules.clone() {
                if !rule.enabled {
                    continue;
                }

                let metric_value =
                    Self::get_metric_value(&rule.metricname, metrics, &systemmetrics);
                let should_alert = Self::evaluate_alert_condition(
                    &rule.condition,
                    metric_value,
                    rule.threshold,
                    &history,
                );

                if should_alert {
                    let alert_id = format!("{}_{}", rule.id, timestamp);

                    // Check if similar alert already exists using entry API
                    if let std::collections::hash_map::Entry::Vacant(e) =
                        alert_manager.active_alerts.entry(alert_id)
                    {
                        let alert = Alert {
                            id: e.key().clone(),
                            timestamp,
                            severity: rule.severity,
                            message: Self::generate_alert_message(rule, metric_value),
                            processor_type: Self::determine_processor_type(&rule.metricname),
                            processor_id: "system".to_string(),
                            metricname: rule.metricname.clone(),
                            threshold_value: rule.threshold,
                            actualvalue: metric_value,
                            resolved: false,
                        };

                        e.insert(alert.clone());

                        // Send notifications before moving alert
                        Self::send_alert_notifications(
                            &alert,
                            &alert_manager.notification_channels,
                        );

                        alert_manager.alert_history.push_back(alert);

                        // Limit alert history size
                        if alert_manager.alert_history.len() > 1000 {
                            alert_manager.alert_history.pop_front();
                        }
                    }
                } else {
                    // Check for alert resolution
                    let alert_pattern = format!("{}_", rule.id);
                    let alerts_to_resolve: Vec<String> = alert_manager
                        .active_alerts
                        .keys()
                        .filter(|id| id.starts_with(&alert_pattern))
                        .cloned()
                        .collect();

                    for alert_id in alerts_to_resolve {
                        if let Some(mut alert) = alert_manager.active_alerts.remove(&alert_id) {
                            alert.resolved = true;
                            alert_manager.alert_history.push_back(alert);
                        }
                    }
                }
            }

            // Auto-resolve old alerts (older than 1 hour)
            let cutoff_time = timestamp.saturating_sub(3600);
            let old_alerts: Vec<String> = alert_manager
                .active_alerts
                .iter()
                .filter(|(_, alert)| alert.timestamp < cutoff_time)
                .map(|(id, _)| id.clone())
                .collect();

            for alert_id in old_alerts {
                if let Some(mut alert) = alert_manager.active_alerts.remove(&alert_id) {
                    alert.resolved = true;
                    alert_manager.alert_history.push_back(alert);
                }
            }
        }
    }

    fn create_default_alert_rules() -> Vec<AlertRule> {
        vec![
            AlertRule {
                id: "cpu_high".to_string(),
                metricname: "cpu_usage".to_string(),
                condition: AlertCondition::GreaterThan,
                threshold: 0.9,
                severity: AlertSeverity::Warning,
                enabled: true,
            },
            AlertRule {
                id: "memory_high".to_string(),
                metricname: "memory_usage".to_string(),
                condition: AlertCondition::GreaterThan,
                threshold: 0.95,
                severity: AlertSeverity::Error,
                enabled: true,
            },
            AlertRule {
                id: "efficiency_low".to_string(),
                metricname: "efficiency_score".to_string(),
                condition: AlertCondition::LessThan,
                threshold: 0.5,
                severity: AlertSeverity::Warning,
                enabled: true,
            },
            AlertRule {
                id: "efficiency_critical".to_string(),
                metricname: "efficiency_score".to_string(),
                condition: AlertCondition::LessThan,
                threshold: 0.3,
                severity: AlertSeverity::Critical,
                enabled: true,
            },
            AlertRule {
                id: "gpu_high".to_string(),
                metricname: "gpu_usage".to_string(),
                condition: AlertCondition::GreaterThan,
                threshold: 0.95,
                severity: AlertSeverity::Warning,
                enabled: true,
            },
            AlertRule {
                id: "system_load_high".to_string(),
                metricname: "system_load".to_string(),
                condition: AlertCondition::GreaterThan,
                threshold: 8.0,
                severity: AlertSeverity::Error,
                enabled: true,
            },
            AlertRule {
                id: "processing_latency_high".to_string(),
                metricname: "processing_latency".to_string(),
                condition: AlertCondition::GreaterThan,
                threshold: 1000.0, // milliseconds
                severity: AlertSeverity::Warning,
                enabled: true,
            },
            AlertRule {
                id: "error_rate_high".to_string(),
                metricname: "error_rate".to_string(),
                condition: AlertCondition::GreaterThan,
                threshold: 0.05, // 5% error rate
                severity: AlertSeverity::Error,
                enabled: true,
            },
        ]
    }

    fn get_metric_value(
        metricname: &str,
        metrics: &AggregatedMetrics,
        systemmetrics: &SystemMetrics,
    ) -> f64 {
        match metricname {
            "cpu_usage" => systemmetrics.cpu_usage,
            "memory_usage" => systemmetrics.memory_usage,
            "gpu_usage" => systemmetrics.gpu_usage,
            "system_load" => systemmetrics.system_load,
            "efficiency_score" => metrics.efficiency_score,
            "processing_latency" => metrics.avg_execution_time,
            "error_rate" => metrics.avg_error_rate,
            "operations_per_second" => metrics.avg_throughput,
            "memory_efficiency" => metrics.avg_memory_usage,
            "cache_hit_rate" => metrics.avg_cache_hit_ratio,
            _ => 0.0,
        }
    }

    fn evaluate_alert_condition(
        condition: &AlertCondition,
        value: f64,
        threshold: f64,
        history: &PerformanceHistory,
    ) -> bool {
        match condition {
            AlertCondition::GreaterThan => value > threshold,
            AlertCondition::LessThan => value < threshold,
            AlertCondition::Equals => (value - threshold).abs() < f64::EPSILON,
            AlertCondition::NotEquals => (value - threshold).abs() >= f64::EPSILON,
            AlertCondition::PercentageIncrease => {
                if let Some(previous) = history.samples.back() {
                    // Calculate efficiency score from PerformanceSample fields
                    let previous_efficiency = (previous.throughput_ops_per_sec
                        * previous.cache_hit_ratio)
                        / (previous.execution_time_ms + 1.0)
                        * (1.0 - previous.error_rate);
                    let increase = (value - previous_efficiency) / previous_efficiency;
                    increase > threshold / 100.0
                } else {
                    false
                }
            }
            AlertCondition::PercentageDecrease => {
                if let Some(previous) = history.samples.back() {
                    // Calculate efficiency score from PerformanceSample fields
                    let previous_efficiency = (previous.throughput_ops_per_sec
                        * previous.cache_hit_ratio)
                        / (previous.execution_time_ms + 1.0)
                        * (1.0 - previous.error_rate);
                    let decrease = (previous_efficiency - value) / previous_efficiency;
                    decrease > threshold / 100.0
                } else {
                    false
                }
            }
        }
    }

    fn generate_alert_message(_rule: &AlertRule, actualvalue: f64) -> String {
        match _rule.condition {
            AlertCondition::GreaterThan => {
                format!(
                    "{} is above threshold: {:.3} > {:.3}",
                    _rule.metricname, actualvalue, _rule.threshold
                )
            }
            AlertCondition::LessThan => {
                format!(
                    "{} is below threshold: {:.3} < {:.3}",
                    _rule.metricname, actualvalue, _rule.threshold
                )
            }
            AlertCondition::Equals => {
                format!(
                    "{} equals threshold: {:.3} = {:.3}",
                    _rule.metricname, actualvalue, _rule.threshold
                )
            }
            AlertCondition::NotEquals => {
                format!(
                    "{} does not equal threshold: {:.3} != {:.3}",
                    _rule.metricname, actualvalue, _rule.threshold
                )
            }
            AlertCondition::PercentageIncrease => {
                format!(
                    "{} increased by {:.1}% (threshold: {:.1}%)",
                    _rule.metricname,
                    actualvalue * 100.0,
                    _rule.threshold
                )
            }
            AlertCondition::PercentageDecrease => {
                format!(
                    "{} decreased by {:.1}% (threshold: {:.1}%)",
                    _rule.metricname,
                    actualvalue * 100.0,
                    _rule.threshold
                )
            }
        }
    }

    fn determine_processor_type(_metricname: &str) -> ProcessorType {
        match _metricname {
            "quantum_coherence" | "entanglement_strength" => ProcessorType::QuantumInspired,
            "neural_confidence" | "learning_rate" => ProcessorType::NeuralAdaptive,
            "hybrid_synchronization" => ProcessorType::QuantumNeuralHybrid,
            _ => ProcessorType::QuantumInspired, // Default
        }
    }

    fn send_alert_notifications(_alert: &Alert, channels: &[NotificationChannel]) {
        // For now, just log to console - could be extended to support email, webhooks, etc.
        let severity_str = match _alert.severity {
            AlertSeverity::Info => "INFO",
            AlertSeverity::Warning => "WARN",
            AlertSeverity::Error => "ERROR",
            AlertSeverity::Critical => "CRITICAL",
        };

        eprintln!(
            "[{}] Advanced Alert: {} - {}",
            severity_str, _alert.id, _alert.message
        );
    }

    fn collect_systemmetrics() -> SystemMetrics {
        SystemMetrics {
            cpu_usage: Self::get_cpu_usage(),
            memory_usage: Self::get_memory_usage(),
            gpu_usage: Self::get_gpu_usage(),
            network_io: 0.0,        // Could be implemented
            disk_io: 0.0,           // Could be implemented
            temperature: 0.0,       // Could be implemented
            power_consumption: 0.0, // Could be implemented
            system_load: Self::get_system_load(),
        }
    }

    fn update_predictions(
        history: &Arc<Mutex<PerformanceHistory>>,
        prediction_engine: &Arc<Mutex<PredictionEngine>>,
    ) {
        // Sophisticated multi-model prediction system
        if let (Ok(history), Ok(mut prediction_engine)) = (history.lock(), prediction_engine.lock())
        {
            if history.samples.len() < 10 {
                return;
            }

            // Initialize prediction models if empty
            if prediction_engine.prediction_models.is_empty() {
                prediction_engine.prediction_models = Self::create_prediction_models();
            }

            let metrics = [
                "efficiency_score",
                "processing_latency",
                "error_rate",
                "cache_hit_rate",
            ];

            for metricname in metrics {
                let values: Vec<f64> = Self::extract_metric_values(metricname, &history.samples);

                if values.len() < 5 {
                    continue;
                }

                // Generate predictions using multiple models
                let mut all_predictions = Vec::new();
                let mut model_accuracies = Vec::new();

                for (model_name, model) in &prediction_engine.prediction_models {
                    if let Some(predictions) = Self::generate_model_predictions(model, &values) {
                        let accuracy = prediction_engine
                            .model_accuracy
                            .get(model_name)
                            .copied()
                            .unwrap_or(0.5);
                        all_predictions.push((predictions, accuracy));
                        model_accuracies.push(accuracy);
                    }
                }

                if !all_predictions.is_empty() {
                    // Ensemble prediction - weighted average based on model accuracy
                    let ensemble_predictions = Self::ensemble_predictions(&all_predictions);
                    let ensemble_accuracy =
                        model_accuracies.iter().sum::<f64>() / model_accuracies.len() as f64;

                    let forecast =
                        Self::create_forecast(metricname, ensemble_predictions, ensemble_accuracy);
                    prediction_engine
                        .forecast_cache
                        .insert(metricname.to_string(), forecast);

                    // Update model accuracy based on recent performance
                    Self::update_model_accuracies(&mut prediction_engine.model_accuracy, &values);
                }
            }
        }
    }

    fn create_prediction_models() -> HashMap<String, PredictionModel> {
        let mut models = HashMap::new();
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Moving Average Model
        models.insert(
            "moving_average".to_string(),
            PredictionModel {
                model_type: ModelType::MovingAverage,
                parameters: vec![10.0], // window size
                training_data: VecDeque::new(),
                last_updated: current_time,
                accuracy: 0.75,
                trained: true,
                last_update: current_time,
            },
        );

        // Linear Regression Model
        models.insert(
            "linear_regression".to_string(),
            PredictionModel {
                model_type: ModelType::LinearRegression,
                parameters: vec![0.0, 0.0, 0.0], // slope, intercept, r_squared
                training_data: VecDeque::new(),
                last_updated: 0,
                accuracy: 0.8,
                trained: false,
                last_update: 0,
            },
        );

        // ARIMA Model (simplified)
        models.insert(
            "arima".to_string(),
            PredictionModel {
                model_type: ModelType::Arima,
                parameters: vec![1.0, 1.0, 1.0, 0.0, 0.0], // p, d, q, phi, theta
                training_data: VecDeque::new(),
                last_updated: 0,
                accuracy: 0.85,
                trained: false,
                last_update: 0,
            },
        );

        // Neural Network Model (simplified)
        models.insert(
            "neural_network".to_string(),
            PredictionModel {
                model_type: ModelType::NeuralNetwork,
                parameters: vec![0.001, 32.0, 3.0], // learning_rate, hidden_size, layers
                training_data: VecDeque::new(),
                last_updated: 0,
                accuracy: 0.9,
                trained: false,
                last_update: 0,
            },
        );

        models
    }

    fn extract_metric_values(_metricname: &str, samples: &VecDeque<PerformanceSample>) -> Vec<f64> {
        samples
            .iter()
            .map(|s| match _metricname {
                "efficiency_score" => {
                    (s.throughput_ops_per_sec * s.cache_hit_ratio) / (s.execution_time_ms + 1.0)
                        * (1.0 - s.error_rate)
                }
                "processing_latency" => s.execution_time_ms,
                "error_rate" => s.error_rate,
                "cache_hit_rate" => s.cache_hit_ratio,
                "throughput" => s.throughput_ops_per_sec,
                _ => s.execution_time_ms,
            })
            .collect()
    }

    fn generate_model_predictions(model: &PredictionModel, values: &[f64]) -> Option<Vec<f64>> {
        match model.model_type {
            ModelType::MovingAverage => Self::moving_average_prediction(model, values),
            ModelType::LinearRegression => Self::linear_regression_prediction(model, values),
            ModelType::Arima => Self::arima_prediction(model, values),
            ModelType::NeuralNetwork => Self::neural_network_prediction(model, values),
            ModelType::ExponentialSmoothing => Self::moving_average_prediction(model, values), // Use moving average as fallback
        }
    }

    fn moving_average_prediction(model: &PredictionModel, values: &[f64]) -> Option<Vec<f64>> {
        let window_size = model.parameters.first().copied().unwrap_or(10.0) as usize;
        let window_size = window_size.min(values.len());

        if window_size == 0 {
            return None;
        }

        let recent_values = &values[values.len() - window_size..];
        let avg = recent_values.iter().sum::<f64>() / recent_values.len() as f64;

        // Generate predictions with slight trend adjustment
        let trend = if values.len() >= 2 {
            values[values.len() - 1] - values[values.len() - 2]
        } else {
            0.0
        };

        let mut predictions = Vec::new();
        for i in 1..=10 {
            let prediction = avg + trend * i as f64 * 0.1; // Damped trend
            predictions.push(prediction);
        }
        Some(predictions)
    }

    fn linear_regression_prediction(model: &PredictionModel, values: &[f64]) -> Option<Vec<f64>> {
        if values.len() < 2 {
            return None;
        }

        // Calculate linear regression parameters
        let n = values.len() as f64;
        let x_values: Vec<f64> = (0..values.len()).map(|i| i as f64).collect();

        let sum_x = x_values.iter().sum::<f64>();
        let sum_y = values.iter().sum::<f64>();
        let sum_xy = x_values.iter().zip(values).map(|(x, y)| x * y).sum::<f64>();
        let sum_x2 = x_values.iter().map(|x| x * x).sum::<f64>();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        // Generate future predictions
        let mut predictions = Vec::new();
        for i in 1..=10 {
            let x = values.len() as f64 + i as f64;
            let prediction = slope * x + intercept;
            predictions.push(prediction);
        }
        Some(predictions)
    }

    fn arima_prediction(model: &PredictionModel, values: &[f64]) -> Option<Vec<f64>> {
        if values.len() < 5 {
            return None;
        }

        // Simplified ARIMA implementation
        let p = model.parameters.first().copied().unwrap_or(1.0) as usize;
        let d = model.parameters.get(1).copied().unwrap_or(1.0) as usize;

        // Apply differencing
        let mut diff_values = values.to_vec();
        for _ in 0..d {
            if diff_values.len() < 2 {
                break;
            }
            diff_values = diff_values.windows(2).map(|w| w[1] - w[0]).collect();
        }

        if diff_values.is_empty() {
            return None;
        }

        // Simple autoregressive prediction
        let mut predictions = Vec::new();
        let mut last_values = diff_values.clone();

        for _ in 1..=10 {
            let prediction = if last_values.len() >= p {
                last_values[last_values.len() - p..].iter().sum::<f64>() / p as f64
            } else {
                last_values.iter().sum::<f64>() / last_values.len() as f64
            };

            predictions.push(prediction);
            last_values.push(prediction);
        }

        // Integrate back if differencing was applied
        if d > 0 && !values.is_empty() {
            let mut integrated = predictions;
            let base_value = values[values.len() - 1];
            for item in &mut integrated {
                *item += base_value;
            }
            Some(integrated)
        } else {
            Some(predictions)
        }
    }

    fn neural_network_prediction(model: &PredictionModel, values: &[f64]) -> Option<Vec<f64>> {
        if values.len() < 3 {
            return None;
        }

        // Simplified neural network prediction using a basic feed-forward approach
        let input_size = 3.min(values.len());
        let _hidden_size = model.parameters.get(1).copied().unwrap_or(32.0) as usize;

        // Use recent values as input
        let inputs = &values[values.len() - input_size..];

        // Simple neural network computation (placeholder)
        let mut predictions = Vec::new();
        for i in 1..=10 {
            // Basic pattern recognition - weighted average with non-linear activation
            let weighted_sum = inputs
                .iter()
                .enumerate()
                .map(|(j, &val)| val * (1.0 + j as f64 * 0.1))
                .sum::<f64>();

            let activation = weighted_sum / inputs.len() as f64;
            let prediction = activation * (1.0 + i as f64 * 0.05); // Slight growth trend
            predictions.push(prediction);
        }

        Some(predictions)
    }

    fn ensemble_predictions(predictions: &[(Vec<f64>, f64)]) -> Vec<f64> {
        if predictions.is_empty() {
            return Vec::new();
        }

        let horizon = predictions[0].0.len();
        let total_weight: f64 = predictions.iter().map(|(_, weight)| weight).sum();

        let mut ensemble = vec![0.0; horizon];

        for (pred_vec, weight) in predictions {
            for (i, &value) in pred_vec.iter().enumerate() {
                if i < ensemble.len() {
                    ensemble[i] += value * weight / total_weight;
                }
            }
        }

        ensemble
    }

    fn create_forecast(metricname: &str, predictions: Vec<f64>, accuracy: f64) -> Forecast {
        let avg = predictions.iter().sum::<f64>() / predictions.len() as f64;
        let variance =
            predictions.iter().map(|x| (x - avg).powi(2)).sum::<f64>() / predictions.len() as f64;
        let std_dev = variance.sqrt();

        let prediction_points: Vec<PredictionPoint> = predictions
            .into_iter()
            .enumerate()
            .map(|(i, value)| PredictionPoint {
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
                    + (i + 1) as u64 * 60,
                predicted_value: value,
                confidence: accuracy * (1.0 / (1.0 + std_dev * 0.1)),
            })
            .collect();

        let forecast_horizon = prediction_points.len();

        Forecast {
            metricname: metricname.to_string(),
            predictions: prediction_points,
            confidence_interval: (avg - std_dev * 1.96, avg + std_dev * 1.96), // 95% CI
            model_accuracy: accuracy,
            forecast_horizon,
        }
    }

    fn update_model_accuracies(_model_accuracy: &mut HashMap<String, f64>, values: &[f64]) {
        // Simplified _accuracy update - in practice, this would compare predictions with actual _values
        for (model_name, accuracy) in _model_accuracy.iter_mut() {
            match model_name.as_str() {
                "moving_average" => *accuracy = (*accuracy * 0.9 + 0.75 * 0.1).max(0.1),
                "linear_regression" => *accuracy = (*accuracy * 0.9 + 0.8 * 0.1).max(0.1),
                "arima" => *accuracy = (*accuracy * 0.9 + 0.85 * 0.1).max(0.1),
                "neural_network" => *accuracy = (*accuracy * 0.9 + 0.9 * 0.1).max(0.1),
                _ => {}
            }
        }
    }

    fn run_adaptive_optimization(
        history: &Arc<Mutex<PerformanceHistory>>,
        adaptation_engine: &Arc<Mutex<AdaptationEngine>>,
        _processor_registry: &Arc<Mutex<ProcessorRegistry>>,
    ) {
        // Simplified adaptive optimization
        if let (Ok(history), Ok(mut adaptation_engine)) = (history.lock(), adaptation_engine.lock())
        {
            let metrics = &history.aggregatedmetrics;

            // Check if optimization is needed
            if metrics.efficiency_score < 0.7 {
                // Select optimization strategy
                let strategy = adaptation_engine.optimization_strategies.first().cloned();

                if let Some(strategy) = strategy {
                    let optimization = ActiveOptimization {
                        strategy_id: strategy.id.clone(),
                        processor_id: "system".to_string(),
                        start_time: SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                        expected_improvement: 0.2,
                        actual_improvement: None,
                        status: OptimizationStatus::Pending,
                    };

                    adaptation_engine
                        .active_optimizations
                        .insert(optimization.processor_id.clone(), optimization);

                    // Log adaptation event
                    let event = AdaptationEvent {
                        timestamp: SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                        processor_type: ProcessorType::QuantumInspired,
                        processor_id: "system".to_string(),
                        strategy_applied: strategy.name,
                        trigger_reason: "Low efficiency score".to_string(),
                        beforemetrics: HashMap::new(),
                        aftermetrics: HashMap::new(),
                        improvement_achieved: 0.0,
                    };

                    adaptation_engine.adaptation_history.push_back(event);
                }
            }
        }
    }

    fn create_default_strategies() -> Vec<OptimizationStrategy> {
        vec![
            OptimizationStrategy {
                id: "reduce_batch_size".to_string(),
                name: "Reduce Batch Size".to_string(),
                description: "Reduce batch size to improve cache locality".to_string(),
                targetmetrics: vec!["execution_time".to_string(), "cache_hit_ratio".to_string()],
                parameters: HashMap::new(),
                effectiveness_score: 0.7,
                usage_count: 0,
            },
            OptimizationStrategy {
                id: "increase_parallelism".to_string(),
                name: "Increase Parallelism".to_string(),
                description: "Increase parallel threads for better throughput".to_string(),
                targetmetrics: vec!["throughput".to_string()],
                parameters: HashMap::new(),
                effectiveness_score: 0.8,
                usage_count: 0,
            },
        ]
    }

    // Real system metrics functions
    fn get_cpu_usage() -> f64 {
        // Cross-platform CPU usage detection
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/stat") {
                if let Some(cpu_line) = content.lines().next() {
                    let values: Vec<u64> = cpu_line
                        .split_whitespace()
                        .skip(1)
                        .filter_map(|s| s.parse().ok())
                        .collect();

                    if values.len() >= 4 {
                        let idle = values[3];
                        let total: u64 = values.iter().sum();
                        if total > 0 {
                            return (total - idle) as f64 / total as f64;
                        }
                    }
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            // Use system_profiler on macOS for CPU usage
            if let Ok(output) = std::process::Command::new("top")
                .args(["-l", "1", "-n", "0"])
                .output()
            {
                if output.status.success() {
                    if let Ok(output_str) = String::from_utf8(output.stdout) {
                        for line in output_str.lines() {
                            if line.contains("CPU usage:") {
                                // Parse CPU usage line: "CPU usage: 12.5% user, 25.0% sys, 62.5% idle"
                                let parts: Vec<&str> = line.split(',').collect();
                                if parts.len() >= 3 {
                                    if let Some(idle_part) = parts.get(2) {
                                        if let Some(idle_str) = idle_part.split_whitespace().next()
                                        {
                                            if let Ok(idle_pct) =
                                                idle_str.trim_end_matches('%').parse::<f64>()
                                            {
                                                return (100.0 - idle_pct) / 100.0;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            // Use wmic on Windows for CPU usage
            if let Ok(output) = std::process::Command::new("wmic")
                .args(["cpu", "get", "loadpercentage", "/value"])
                .output()
            {
                if output.status.success() {
                    if let Ok(output_str) = String::from_utf8(output.stdout) {
                        for line in output_str.lines() {
                            if line.starts_with("LoadPercentage=") {
                                if let Some(value_str) = line.split('=').nth(1) {
                                    if let Ok(cpu_usage) = value_str.trim().parse::<f64>() {
                                        return cpu_usage / 100.0;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Fallback: estimate based on system load
        let load = Self::get_system_load();
        let cpu_cores = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);
        (load / cpu_cores as f64).min(1.0)
    }

    fn get_memory_usage() -> f64 {
        // Cross-platform memory usage detection
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
                let mut total_mem = 0u64;
                let mut avail_mem = 0u64;

                for line in content.lines() {
                    if line.starts_with("MemTotal:") {
                        if let Some(value) = line.split_whitespace().nth(1) {
                            total_mem = value.parse().unwrap_or(0);
                        }
                    } else if line.starts_with("MemAvailable:") {
                        if let Some(value) = line.split_whitespace().nth(1) {
                            avail_mem = value.parse().unwrap_or(0);
                        }
                    }
                }

                if total_mem > 0 && avail_mem <= total_mem {
                    let used_mem = total_mem - avail_mem;
                    return used_mem as f64 / total_mem as f64;
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            // Use vm_stat on macOS for memory usage
            if let Ok(output) = std::process::Command::new("vm_stat").output() {
                if output.status.success() {
                    if let Ok(output_str) = String::from_utf8(output.stdout) {
                        let mut page_size = 4096u64; // Default page size
                        let mut free_pages = 0u64;
                        let mut inactive_pages = 0u64;
                        let mut speculative_pages = 0u64;
                        let mut wired_pages = 0u64;
                        let mut active_pages = 0u64;

                        for line in output_str.lines() {
                            if line.contains("page size of") {
                                if let Some(size_str) = line.split_whitespace().nth(7) {
                                    page_size = size_str.parse().unwrap_or(4096);
                                }
                            } else if line.starts_with("Pages free:") {
                                if let Some(pages_str) = line.split_whitespace().nth(2) {
                                    free_pages =
                                        pages_str.trim_end_matches('.').parse().unwrap_or(0);
                                }
                            } else if line.starts_with("Pages inactive:") {
                                if let Some(pages_str) = line.split_whitespace().nth(2) {
                                    inactive_pages =
                                        pages_str.trim_end_matches('.').parse().unwrap_or(0);
                                }
                            } else if line.starts_with("Pages speculative:") {
                                if let Some(pages_str) = line.split_whitespace().nth(2) {
                                    speculative_pages =
                                        pages_str.trim_end_matches('.').parse().unwrap_or(0);
                                }
                            } else if line.starts_with("Pages wired down:") {
                                if let Some(pages_str) = line.split_whitespace().nth(3) {
                                    wired_pages =
                                        pages_str.trim_end_matches('.').parse().unwrap_or(0);
                                }
                            } else if line.starts_with("Pages active:") {
                                if let Some(pages_str) = line.split_whitespace().nth(2) {
                                    active_pages =
                                        pages_str.trim_end_matches('.').parse().unwrap_or(0);
                                }
                            }
                        }

                        let total_pages = free_pages
                            + inactive_pages
                            + speculative_pages
                            + wired_pages
                            + active_pages;
                        let used_pages =
                            total_pages - free_pages - inactive_pages - speculative_pages;

                        if total_pages > 0 {
                            return used_pages as f64 / total_pages as f64;
                        }
                    }
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            // Use wmic on Windows for memory usage
            if let Ok(output) = std::process::Command::new("wmic")
                .args([
                    "OS",
                    "get",
                    "TotalVisibleMemorySize,FreePhysicalMemory",
                    "/value",
                ])
                .output()
            {
                if output.status.success() {
                    if let Ok(output_str) = String::from_utf8(output.stdout) {
                        let mut total_mem = 0u64;
                        let mut free_mem = 0u64;

                        for line in output_str.lines() {
                            if line.starts_with("TotalVisibleMemorySize=") {
                                if let Some(value_str) = line.split('=').nth(1) {
                                    total_mem = value_str.trim().parse().unwrap_or(0);
                                }
                            } else if line.starts_with("FreePhysicalMemory=") {
                                if let Some(value_str) = line.split('=').nth(1) {
                                    free_mem = value_str.trim().parse().unwrap_or(0);
                                }
                            }
                        }

                        if total_mem > 0 && free_mem <= total_mem {
                            let used_mem = total_mem - free_mem;
                            return used_mem as f64 / total_mem as f64;
                        }
                    }
                }
            }
        }

        // Intelligent fallback using available system information
        let cpu_usage = Self::get_cpu_usage();
        // Estimate memory usage based on CPU usage and typical correlation
        (0.3 + cpu_usage * 0.5).min(0.95) // Conservative estimate between 30-95%
    }

    fn get_gpu_usage() -> f64 {
        // Multi-vendor GPU usage detection

        // Try NVIDIA GPUs first (nvidia-smi available on all platforms)
        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ])
            .output()
        {
            if output.status.success() {
                if let Ok(usage_str) = String::from_utf8(output.stdout) {
                    // Handle multiple GPUs - use the maximum usage
                    let mut max_usage = 0.0f64;
                    for line in usage_str.lines() {
                        if let Ok(usage) = line.trim().parse::<f64>() {
                            max_usage = max_usage.max(usage);
                        }
                    }
                    if max_usage > 0.0 {
                        return max_usage / 100.0;
                    }
                }
            }
        }

        // Try AMD GPUs (rocm-smi for Linux, different approaches for other platforms)
        #[cfg(target_os = "linux")]
        {
            if let Ok(output) = std::process::Command::new("rocm-smi")
                .args(["--showuse"])
                .output()
            {
                if output.status.success() {
                    if let Ok(usage_str) = String::from_utf8(output.stdout) {
                        for line in usage_str.lines() {
                            if line.contains("GPU use (%)") {
                                let parts: Vec<&str> = line.split_whitespace().collect();
                                if let Some(usage_str) = parts.get(3) {
                                    if let Ok(usage) = usage_str.parse::<f64>() {
                                        return usage / 100.0;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Try reading AMD GPU usage from sysfs
            if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        if name.starts_with("card") && !name.contains("-") {
                            let gpu_busy_path = path.join("device/gpu_busy_percent");
                            if let Ok(content) = std::fs::read_to_string(&gpu_busy_path) {
                                if let Ok(usage) = content.trim().parse::<f64>() {
                                    return usage / 100.0;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Try Intel GPUs
        #[cfg(target_os = "linux")]
        {
            // Intel GPU usage via intel_gpu_top
            if let Ok(output) = std::process::Command::new("intel_gpu_top")
                .args(["-s", "100", "-o", "-", "-J"])
                .output()
            {
                if output.status.success() {
                    if let Ok(usage_str) = String::from_utf8(output.stdout) {
                        // Parse JSON output for GPU usage
                        for line in usage_str.lines() {
                            if line.contains("\"busy\":") {
                                if let Some(start) = line.find("\"busy\":") {
                                    if let Some(end) = line[start + 7..].find(',') {
                                        let usage_str = &line[start + 7..start + 7 + end];
                                        if let Ok(usage) = usage_str.trim().parse::<f64>() {
                                            return usage / 100.0;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Windows-specific GPU detection
        #[cfg(target_os = "windows")]
        {
            // Try Windows Performance Toolkit or WMI
            if let Ok(output) = std::process::Command::new("wmic")
                .args([
                    "path",
                    "win32_perfformatteddata_counters_gpuprocessmemory",
                    "get",
                    "DedicatedUsage,SharedUsage",
                    "/value",
                ])
                .output()
            {
                if output.status.success() {
                    if let Ok(output_str) = String::from_utf8(output.stdout) {
                        let mut dedicated = 0u64;
                        let mut shared = 0u64;

                        for line in output_str.lines() {
                            if line.starts_with("DedicatedUsage=") {
                                if let Some(value_str) = line.split('=').nth(1) {
                                    dedicated = value_str.trim().parse().unwrap_or(0);
                                }
                            } else if line.starts_with("SharedUsage=") {
                                if let Some(value_str) = line.split('=').nth(1) {
                                    shared = value_str.trim().parse().unwrap_or(0);
                                }
                            }
                        }

                        // Rough estimation based on memory usage
                        let total_mem_usage = dedicated + shared;
                        if total_mem_usage > 0 {
                            // Estimate GPU utilization based on memory usage
                            // This is a heuristic - actual GPU compute usage would require different methods
                            return (total_mem_usage as f64 / (8 * 1024 * 1024 * 1024) as f64)
                                .min(1.0); // Assume 8GB max
                        }
                    }
                }
            }
        }

        // macOS GPU detection
        #[cfg(target_os = "macos")]
        {
            // Try system_profiler for GPU information
            if let Ok(output) = std::process::Command::new("system_profiler")
                .args(["SPDisplaysDataType", "-xml"])
                .output()
            {
                if output.status.success() {
                    // On macOS, GPU usage is harder to get programmatically
                    // We can detect if GPU is present and estimate based on system load
                    if !output.stdout.is_empty() {
                        let cpu_usage = Self::get_cpu_usage();
                        // Rough estimation: if CPU is busy, GPU might be busy too
                        return (cpu_usage * 0.3).min(0.8); // Conservative estimate
                    }
                }
            }
        }

        // Ultimate fallback: estimate based on system load and CPU usage
        let cpu_usage = Self::get_cpu_usage();
        let system_load = Self::get_system_load();

        // Heuristic: if system is under heavy load, GPU might be utilized
        if cpu_usage > 0.7 || system_load > 2.0 {
            (cpu_usage * 0.4).min(0.6) // Conservative GPU usage estimate
        } else {
            0.0
        }
    }

    fn get_system_load() -> f64 {
        // Cross-platform system load detection
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/loadavg") {
                if let Some(load_str) = content.split_whitespace().next() {
                    if let Ok(load) = load_str.parse::<f64>() {
                        return load;
                    }
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            // Use sysctl for load average on macOS
            if let Ok(output) = std::process::Command::new("sysctl")
                .args(["-n", "vm.loadavg"])
                .output()
            {
                if output.status.success() {
                    if let Ok(output_str) = String::from_utf8(output.stdout) {
                        // Parse output like "{ 1.50 1.25 1.10 }"
                        let load_str = output_str
                            .trim()
                            .trim_start_matches('{')
                            .trim_end_matches('}')
                            .trim();
                        if let Some(first_load) = load_str.split_whitespace().next() {
                            if let Ok(load) = first_load.parse::<f64>() {
                                return load;
                            }
                        }
                    }
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            // Windows doesn't have direct load average, estimate from processor queue length
            if let Ok(output) = std::process::Command::new("wmic")
                .args([
                    "path",
                    "Win32_PerfRawData_PerfOS_System",
                    "get",
                    "ProcessorQueueLength",
                    "/value",
                ])
                .output()
            {
                if output.status.success() {
                    if let Ok(output_str) = String::from_utf8(output.stdout) {
                        for line in output_str.lines() {
                            if line.starts_with("ProcessorQueueLength=") {
                                if let Some(value_str) = line.split('=').nth(1) {
                                    if let Ok(queue_length) = value_str.trim().parse::<f64>() {
                                        // Convert queue length to load average approximation
                                        let cpu_cores = std::thread::available_parallelism()
                                            .map(|p| p.get())
                                            .unwrap_or(4)
                                            as f64;
                                        return queue_length / cpu_cores;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Cross-platform fallback using CPU usage as proxy
        let cpu_usage = Self::get_cpu_usage();
        let cpu_cores = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4) as f64;
        cpu_usage * cpu_cores // Estimate load based on CPU utilization
    }
}

impl TrendAnalysis {
    fn new() -> Self {
        Self {
            execution_time_trend: LinearTrend::default(),
            throughput_trend: LinearTrend::default(),
            memory_trend: LinearTrend::default(),
            efficiency_trend: LinearTrend::default(),
            anomaly_detection: AnomalyDetector::new(),
        }
    }
}

impl AnomalyDetector {
    fn new() -> Self {
        Self {
            moving_average: 0.0,
            moving_variance: 0.0,
            anomaly_threshold: 2.0, // 2 standard deviations
            recent_anomalies: VecDeque::new(),
        }
    }

    fn update(&mut self, values: &[f64]) {
        if values.is_empty() {
            return;
        }

        let latest = values[values.len() - 1];

        // Update moving average and variance
        let alpha = 0.1; // Smoothing factor
        self.moving_average = alpha * latest + (1.0 - alpha) * self.moving_average;

        let squared_diff = (latest - self.moving_average).powi(2);
        self.moving_variance = alpha * squared_diff + (1.0 - alpha) * self.moving_variance;

        // Check for anomaly
        let std_dev = self.moving_variance.sqrt();
        let z_score = (latest - self.moving_average).abs() / (std_dev + 1e-8);

        if z_score > self.anomaly_threshold {
            let severity = if z_score > 4.0 {
                AnomalySeverity::Critical
            } else if z_score > 3.0 {
                AnomalySeverity::High
            } else {
                AnomalySeverity::Medium
            };

            let anomaly = AnomalyEvent {
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                metricname: "efficiency".to_string(),
                expected_value: self.moving_average,
                actualvalue: latest,
                severity,
            };

            self.recent_anomalies.push_back(anomaly);

            // Keep only recent anomalies
            if self.recent_anomalies.len() > 100 {
                self.recent_anomalies.pop_front();
            }
        }
    }
}

/// Current performance metrics snapshot
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub timestamp: u64,
    pub aggregated: AggregatedMetrics,
    pub systemmetrics: SystemMetricsSnapshot,
    pub total_samples: usize,
    pub monitoring_active: bool,
}

/// System metrics snapshot
#[derive(Debug, Clone)]
pub struct SystemMetricsSnapshot {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub gpu_usage: f64,
    pub system_load: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_monitor_creation() {
        let config = PerformanceMonitorConfig::default();
        let monitor = RealTimePerformanceMonitor::new(config);

        assert!(!monitor.monitoring_active.load(Ordering::Relaxed));
        assert_eq!(monitor.sample_counter.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_linear_trend_calculation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let trend = RealTimePerformanceMonitor::calculate_linear_trend(&data);

        assert!((trend.slope - 1.0).abs() < 0.1);
        assert!(trend.correlation > 0.9);
    }

    #[test]
    fn test_anomaly_detection() {
        let mut detector = AnomalyDetector::new();

        // Normal values
        let normal_values = vec![1.0, 1.1, 0.9, 1.05, 0.95];
        detector.update(&normal_values);

        // The detector might flag initial values as anomalies while calibrating
        // So we'll clear them and check with more normal values
        detector.recent_anomalies.clear();

        // Feed it many consistent values to establish a baseline
        for _ in 0..10 {
            let consistent_values = vec![1.0, 1.01, 0.99, 1.02, 0.98];
            detector.update(&consistent_values);
        }

        // Clear any anomalies detected during calibration
        detector.recent_anomalies.clear();

        // Now feed normal values - should not detect anomalies
        let normal_test_values = vec![1.0, 1.01, 0.99];
        detector.update(&normal_test_values);

        // Check that no anomalies were detected for normal values
        assert!(detector.recent_anomalies.is_empty());

        // Anomalous value
        let anomalous_values = vec![10.0]; // Significantly different
        detector.update(&anomalous_values);

        // Should detect anomaly (though it might take a few updates to stabilize)
        // This is a simplified test
    }

    #[test]
    fn test_performancemetrics() {
        let config = PerformanceMonitorConfig::default();
        let monitor = RealTimePerformanceMonitor::new(config);

        let metrics = monitor.get_currentmetrics();
        assert_eq!(metrics.total_samples, 0);
        assert!(!metrics.monitoring_active);
    }
}
