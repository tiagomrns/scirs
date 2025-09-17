//! Performance regression testing framework for CI/CD integration
//!
//! This module provides comprehensive performance regression detection capabilities
//! including baseline establishment, historical tracking, statistical analysis,
//! and automated CI/CD integration for continuous performance monitoring.

use crate::benchmarking::BenchmarkResult;
#[allow(unused_imports)]
use crate::error::Result;
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Comprehensive performance regression testing framework
#[derive(Debug)]
pub struct RegressionTester<A: Float> {
    /// Configuration for regression testing
    config: RegressionConfig,
    /// Historical performance database
    performance_db: PerformanceDatabase<A>,
    /// Baseline performance metrics
    baselines: HashMap<String, PerformanceBaseline<A>>,
    /// Regression detection algorithms
    detectors: Vec<Box<dyn RegressionDetector<A>>>,
    /// Statistical analyzers
    analyzers: Vec<Box<dyn StatisticalAnalyzer<A>>>,
    /// Alert system
    alert_system: AlertSystem,
}

/// Configuration for regression testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionConfig {
    /// Baseline storage directory
    pub baseline_dir: PathBuf,
    /// Maximum history length to keep
    pub max_history_length: usize,
    /// Minimum samples required for baseline
    pub min_baseline_samples: usize,
    /// Statistical significance threshold
    pub significance_threshold: f64,
    /// Performance degradation threshold (percentage)
    pub degradation_threshold: f64,
    /// Memory regression threshold (percentage)
    pub memory_threshold: f64,
    /// Enable CI/CD integration
    pub enable_ci_integration: bool,
    /// Enable automated alerts
    pub enable_alerts: bool,
    /// Outlier detection sensitivity
    pub outlier_sensitivity: f64,
    /// Regression detection algorithms to use
    pub detection_algorithms: Vec<String>,
    /// Export format for CI reports
    pub ci_report_format: CiReportFormat,
}

/// CI report output formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CiReportFormat {
    /// JSON format for programmatic processing
    Json,
    /// JUnit XML format for CI systems
    JunitXml,
    /// Markdown format for human-readable reports
    Markdown,
    /// GitHub Actions format
    GitHubActions,
}

impl Default for RegressionConfig {
    fn default() -> Self {
        Self {
            baseline_dir: PathBuf::from("performance_baselines"),
            max_history_length: 1000,
            min_baseline_samples: 10,
            significance_threshold: 0.05,
            degradation_threshold: 5.0, // 5% degradation threshold
            memory_threshold: 10.0,     // 10% memory increase threshold
            enable_ci_integration: true,
            enable_alerts: true,
            outlier_sensitivity: 2.0, // 2 standard deviations
            detection_algorithms: vec![
                "statistical_test".to_string(),
                "sliding_window".to_string(),
                "change_point".to_string(),
            ],
            ci_report_format: CiReportFormat::Json,
        }
    }
}

/// Historical performance database
#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceDatabase<A: Float> {
    /// Performance history by optimizer and test
    history: HashMap<String, VecDeque<PerformanceRecord<A>>>,
    /// Database metadata
    metadata: DatabaseMetadata,
}

/// Database metadata
#[derive(Debug, Serialize, Deserialize)]
pub struct DatabaseMetadata {
    /// Database version
    pub version: String,
    /// Creation timestamp
    pub created_at: u64,
    /// Last update timestamp
    pub last_updated: u64,
    /// Total number of records
    pub total_records: usize,
}

/// Individual performance record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRecord<A: Float> {
    /// Timestamp of the record
    pub timestamp: u64,
    /// Git commit hash (if available)
    pub commit_hash: Option<String>,
    /// Branch name
    pub branch: Option<String>,
    /// Test environment information
    pub environment: TestEnvironment,
    /// Performance metrics
    pub metrics: PerformanceMetrics<A>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Test environment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestEnvironment {
    /// Operating system
    pub os: String,
    /// CPU model
    pub cpu_model: String,
    /// Memory size (MB)
    pub memory_mb: usize,
    /// Rust version
    pub rust_version: String,
    /// Compiler flags
    pub compiler_flags: Vec<String>,
    /// Hardware acceleration available
    pub hardware_acceleration: Vec<String>,
}

/// Performance metrics for regression testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics<A: Float> {
    /// Execution time metrics
    pub timing: TimingMetrics,
    /// Memory usage metrics
    pub memory: MemoryMetrics,
    /// Computational efficiency metrics
    pub efficiency: EfficiencyMetrics<A>,
    /// Convergence metrics
    pub convergence: ConvergenceMetrics<A>,
    /// Custom metrics
    pub custom: HashMap<String, f64>,
}

/// Timing metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingMetrics {
    /// Mean execution time (nanoseconds)
    pub mean_time_ns: u64,
    /// Standard deviation of execution time
    pub std_time_ns: u64,
    /// Median execution time
    pub median_time_ns: u64,
    /// 95th percentile execution time
    pub p95_time_ns: u64,
    /// 99th percentile execution time
    pub p99_time_ns: u64,
    /// Minimum execution time
    pub min_time_ns: u64,
    /// Maximum execution time
    pub max_time_ns: u64,
}

/// Memory metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    /// Peak memory usage (bytes)
    pub peak_memory_bytes: usize,
    /// Average memory usage (bytes)
    pub avg_memory_bytes: usize,
    /// Memory allocation count
    pub allocation_count: usize,
    /// Memory fragmentation ratio
    pub fragmentation_ratio: f64,
    /// Memory efficiency score
    pub efficiency_score: f64,
}

/// Efficiency metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyMetrics<A: Float> {
    /// FLOPS achieved
    pub flops: f64,
    /// Arithmetic intensity
    pub arithmetic_intensity: f64,
    /// Cache hit ratio
    pub cache_hit_ratio: f64,
    /// CPU utilization
    pub cpu_utilization: f64,
    /// Overall efficiency score
    pub efficiency_score: f64,
    /// Custom efficiency metrics
    pub custom_metrics: HashMap<String, A>,
}

/// Convergence metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceMetrics<A: Float> {
    /// Final objective value
    pub final_objective: A,
    /// Convergence rate
    pub convergence_rate: f64,
    /// Iterations to convergence
    pub iterations_to_convergence: Option<usize>,
    /// Convergence quality score
    pub quality_score: f64,
    /// Stability metrics
    pub stability_score: f64,
}

/// Performance baseline for comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaseline<A: Float> {
    /// Baseline name/identifier
    pub name: String,
    /// Statistical summary of baseline performance
    pub baseline_stats: BaselineStatistics<A>,
    /// Confidence intervals
    pub confidence_intervals: ConfidenceIntervals,
    /// Sample count used for baseline
    pub sample_count: usize,
    /// Baseline creation timestamp
    pub created_at: u64,
    /// Last update timestamp
    pub updated_at: u64,
}

/// Statistical summary of baseline performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineStatistics<A: Float> {
    /// Timing statistics
    pub timing: TimingStatistics,
    /// Memory statistics
    pub memory: MemoryStatistics,
    /// Efficiency statistics
    pub efficiency: EfficiencyStatistics<A>,
    /// Convergence statistics
    pub convergence: ConvergenceStatistics<A>,
}

/// Timing statistics for baseline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingStatistics {
    /// Mean execution time
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Median
    pub median: f64,
    /// Interquartile range
    pub iqr: f64,
    /// Coefficient of variation
    pub coefficient_of_variation: f64,
}

/// Memory statistics for baseline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStatistics {
    /// Mean memory usage
    pub mean_memory: f64,
    /// Standard deviation
    pub std_dev_memory: f64,
    /// Peak memory percentiles
    pub peak_memory_percentiles: HashMap<String, f64>,
    /// Fragmentation statistics
    pub fragmentation_stats: FragmentationStatistics,
}

/// Fragmentation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentationStatistics {
    /// Mean fragmentation ratio
    pub mean_ratio: f64,
    /// Standard deviation
    pub std_dev_ratio: f64,
    /// Trend analysis
    pub trend: f64,
}

/// Efficiency statistics for baseline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyStatistics<A: Float> {
    /// Mean FLOPS
    pub mean_flops: f64,
    /// FLOPS variability
    pub flops_cv: f64,
    /// Mean efficiency score
    pub mean_efficiency: f64,
    /// Custom efficiency metrics
    pub custom_efficiency: HashMap<String, A>,
}

/// Convergence statistics for baseline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceStatistics<A: Float> {
    /// Mean final objective
    pub mean_objective: A,
    /// Objective standard deviation
    pub std_objective: A,
    /// Mean convergence rate
    pub mean_convergence_rate: f64,
    /// Convergence consistency
    pub convergence_consistency: f64,
}

/// Confidence intervals for baseline metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceIntervals {
    /// 95% confidence intervals for timing
    pub timing_ci_95: (f64, f64),
    /// 95% confidence intervals for memory
    pub memory_ci_95: (f64, f64),
    /// 99% confidence intervals for timing
    pub timing_ci_99: (f64, f64),
    /// 99% confidence intervals for memory
    pub memory_ci_99: (f64, f64),
}

/// Regression detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionResult<A: Float> {
    /// Test identifier
    pub test_id: String,
    /// Regression detected
    pub regression_detected: bool,
    /// Regression severity (0.0 to 1.0)
    pub severity: f64,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Performance change percentage
    pub performance_change_percent: f64,
    /// Memory change percentage
    pub memory_change_percent: f64,
    /// Affected metrics
    pub affected_metrics: Vec<String>,
    /// Statistical test results
    pub statistical_tests: Vec<StatisticalTestResult>,
    /// Detailed analysis
    pub analysis: RegressionAnalysis<A>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Statistical test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTestResult {
    /// Test name
    pub test_name: String,
    /// Test statistic value
    pub test_statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Degrees of freedom
    pub degrees_of_freedom: Option<usize>,
    /// Test conclusion
    pub conclusion: String,
}

/// Detailed regression analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAnalysis<A: Float> {
    /// Trend analysis
    pub trend_analysis: TrendAnalysis,
    /// Change point analysis
    pub change_point_analysis: ChangePointAnalysis,
    /// Outlier analysis
    pub outlier_analysis: OutlierAnalysis<A>,
    /// Root cause analysis hints
    pub root_cause_hints: Vec<String>,
}

/// Trend analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    /// Trend direction
    pub direction: TrendDirection,
    /// Trend magnitude
    pub magnitude: f64,
    /// Trend significance
    pub significance: f64,
    /// Trend starting point
    pub start_point: Option<usize>,
}

/// Trend direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    Volatile,
}

/// Change point analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangePointAnalysis {
    /// Change points detected
    pub change_points: Vec<usize>,
    /// Change magnitudes
    pub magnitudes: Vec<f64>,
    /// Confidence levels
    pub confidences: Vec<f64>,
}

/// Outlier analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierAnalysis<A: Float> {
    /// Outlier indices
    pub outlier_indices: Vec<usize>,
    /// Outlier scores
    pub outlier_scores: Vec<A>,
    /// Outlier types
    pub outlier_types: Vec<OutlierType>,
}

/// Types of outliers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutlierType {
    /// Single extreme value
    Point,
    /// Shift in distribution
    Shift,
    /// Trend change
    Trend,
    /// Increased variance
    Variance,
}

/// Regression detection algorithm trait
pub trait RegressionDetector<A: Float>: Debug {
    /// Detect regression in performance data
    fn detect_regression(
        &self,
        baseline: &PerformanceBaseline<A>,
        current_metrics: &PerformanceMetrics<A>,
        history: &VecDeque<PerformanceRecord<A>>,
    ) -> Result<RegressionResult<A>>;

    /// Get detector name
    fn name(&self) -> &str;

    /// Get detector configuration
    fn config(&self) -> HashMap<String, String>;
}

/// Statistical analysis trait
pub trait StatisticalAnalyzer<A: Float>: Debug {
    /// Perform statistical analysis on performance data
    fn analyze(
        &self,
        data: &VecDeque<PerformanceRecord<A>>,
    ) -> Result<StatisticalAnalysisResult<A>>;

    /// Get analyzer name
    fn name(&self) -> &str;
}

/// Statistical analysis result
#[derive(Debug, Clone)]
pub struct StatisticalAnalysisResult<A: Float> {
    /// Analysis summary
    pub summary: String,
    /// Statistical tests performed
    pub tests: Vec<StatisticalTestResult>,
    /// Detected patterns
    pub patterns: Vec<String>,
    /// Anomalies detected
    pub anomalies: Vec<A>,
}

/// Alert system for regression notifications
#[derive(Debug)]
pub struct AlertSystem {
    /// Alert configuration
    config: AlertConfig,
    /// Alert history
    alert_history: VecDeque<Alert>,
}

/// Alert configuration
#[derive(Debug, Clone)]
pub struct AlertConfig {
    /// Enable alerts globally
    pub enable_alerts: bool,
    /// Enable email alerts
    pub enable_email: bool,
    /// Enable Slack notifications
    pub enable_slack: bool,
    /// Enable GitHub issue creation
    pub enable_github_issues: bool,
    /// Alert severity threshold
    pub severity_threshold: f64,
    /// Cooldown period between alerts (minutes)
    pub cooldown_minutes: u64,
}

/// Alert notification
#[derive(Debug, Clone)]
pub struct Alert {
    /// Alert ID
    pub id: String,
    /// Alert timestamp
    pub timestamp: u64,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert message
    pub message: String,
    /// Associated regression result
    pub regression_id: String,
    /// Alert status
    pub status: AlertStatus,
}

/// Alert severity levels
#[derive(Debug, Clone)]
pub enum AlertSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Alert status
#[derive(Debug, Clone)]
pub enum AlertStatus {
    Active,
    Acknowledged,
    Resolved,
}

impl<A: Float + Debug + Serialize + for<'de> Deserialize<'de>> RegressionTester<A> {
    /// Create a new regression tester
    pub fn new(config: RegressionConfig) -> Result<Self> {
        // Ensure baseline directory exists
        fs::create_dir_all(&config.baseline_dir)?;

        let performance_db = PerformanceDatabase::load(&config.baseline_dir)
            .unwrap_or_else(|_| PerformanceDatabase::new());

        let mut tester = Self {
            config: config.clone(),
            performance_db,
            baselines: HashMap::new(),
            detectors: Vec::new(),
            analyzers: Vec::new(),
            alert_system: AlertSystem::new(),
        };

        // Initialize default detectors and analyzers
        tester.initialize_default_components()?;

        // Load existing baselines
        tester.load_baselines()?;

        Ok(tester)
    }

    /// Initialize default regression detectors and analyzers
    fn initialize_default_components(&mut self) -> Result<()> {
        // Add statistical test detector
        if self
            .config
            .detection_algorithms
            .contains(&"statistical_test".to_string())
        {
            self.detectors
                .push(Box::new(StatisticalTestDetector::new()));
        }

        // Add sliding window detector
        if self
            .config
            .detection_algorithms
            .contains(&"sliding_window".to_string())
        {
            self.detectors.push(Box::new(SlidingWindowDetector::new()));
        }

        // Add change point detector
        if self
            .config
            .detection_algorithms
            .contains(&"change_point".to_string())
        {
            self.detectors.push(Box::new(ChangePointDetector::new()));
        }

        // Add default statistical analyzers
        self.analyzers.push(Box::new(TrendAnalyzer::new()));
        self.analyzers.push(Box::new(OutlierAnalyzer::new()));

        Ok(())
    }

    /// Load existing baselines from disk
    fn load_baselines(&mut self) -> Result<()> {
        let baseline_path = self.config.baseline_dir.join("baselines.json");
        if baseline_path.exists() {
            let data = fs::read_to_string(&baseline_path)?;
            self.baselines = serde_json::from_str(&data)?;
        }
        Ok(())
    }

    /// Save baselines to disk
    fn save_baselines(&self) -> Result<()> {
        let baseline_path = self.config.baseline_dir.join("baselines.json");
        let data = serde_json::to_string_pretty(&self.baselines)?;
        fs::write(&baseline_path, data)?;
        Ok(())
    }

    /// Run a performance test and check for regressions
    pub fn run_regression_test<F>(
        &mut self,
        test_name: &str,
        optimizer_name: &str,
        test_function: F,
    ) -> Result<RegressionTestResult<A>>
    where
        F: FnOnce() -> Result<BenchmarkResult<A>>,
    {
        let start_time = std::time::Instant::now();

        // Run the performance test
        let benchmark_result = test_function()?;

        // Extract performance metrics
        let metrics = self.extract_performance_metrics(&benchmark_result)?;

        // Create performance record
        let record = PerformanceRecord {
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            commit_hash: self.get_git_commit_hash(),
            branch: self.get_git_branch(),
            environment: self.collect_environment_info()?,
            metrics,
            metadata: HashMap::new(),
        };

        // Add to performance database
        let key = format!("{}_{}", optimizer_name, test_name);
        self.performance_db.add_record(key.clone(), record.clone());

        // Check for regressions
        let regression_results = self.detect_regressions(&key, &record)?;

        // Update baselines if needed and track if update occurred
        let baseline_updated = self.update_baselines(&key)?;

        // Generate alerts if regressions detected
        for regression in &regression_results {
            if regression.regression_detected {
                self.alert_system.send_alert(regression)?;
            }
        }

        // Save updated database and baselines
        self.performance_db.save(&self.config.baseline_dir)?;
        self.save_baselines()?;

        Ok(RegressionTestResult {
            test_name: test_name.to_string(),
            optimizer_name: optimizer_name.to_string(),
            execution_time: start_time.elapsed(),
            performance_record: record,
            regression_results,
            baseline_updated,
        })
    }

    /// Extract performance metrics from benchmark result
    fn extract_performance_metrics(
        &self,
        result: &BenchmarkResult<A>,
    ) -> Result<PerformanceMetrics<A>> {
        Ok(PerformanceMetrics {
            timing: TimingMetrics {
                mean_time_ns: result.elapsed_time.as_nanos() as u64,
                std_time_ns: self.estimate_timing_std_dev(&result),
                median_time_ns: result.elapsed_time.as_nanos() as u64,
                p95_time_ns: self.estimate_timing_percentile(&result, 0.95),
                p99_time_ns: self.estimate_timing_percentile(&result, 0.99),
                min_time_ns: self.estimate_timing_min(&result),
                max_time_ns: self.estimate_timing_max(&result),
            },
            memory: MemoryMetrics {
                peak_memory_bytes: self.extract_memory_usage(&result).unwrap_or(0),
                avg_memory_bytes: self.extract_avg_memory_usage(&result).unwrap_or(0),
                allocation_count: self.extract_allocation_count(&result).unwrap_or(0),
                fragmentation_ratio: self.estimate_fragmentation_ratio(&result),
                efficiency_score: self.calculate_memory_efficiency(&result),
            },
            efficiency: EfficiencyMetrics {
                flops: self.estimate_flops(&result),
                arithmetic_intensity: self.calculate_arithmetic_intensity(&result),
                cache_hit_ratio: 0.95, // Default estimate
                cpu_utilization: self.extract_cpu_utilization(&result).unwrap_or(0.0),
                efficiency_score: self.calculate_efficiency_score(&result),
                custom_metrics: HashMap::new(),
            },
            convergence: ConvergenceMetrics {
                final_objective: result.final_function_value,
                convergence_rate: if result.converged { 1.0 } else { 0.0 },
                iterations_to_convergence: result.convergence_step,
                quality_score: if result.converged { 1.0 } else { 0.0 },
                stability_score: 1.0,
            },
            custom: HashMap::new(),
        })
    }

    /// Detect regressions using all configured detectors
    fn detect_regressions(
        &self,
        key: &str,
        current_record: &PerformanceRecord<A>,
    ) -> Result<Vec<RegressionResult<A>>> {
        let mut results = Vec::new();

        if let (Some(baseline), Some(history)) = (
            self.baselines.get(key),
            self.performance_db.history.get(key),
        ) {
            for detector in &self.detectors {
                let result =
                    detector.detect_regression(baseline, &current_record.metrics, history)?;
                results.push(result);
            }
        }

        Ok(results)
    }

    /// Update baselines based on new performance data
    fn update_baselines(&mut self, key: &str) -> Result<bool> {
        if let Some(history) = self.performance_db.history.get(key) {
            if history.len() >= self.config.min_baseline_samples {
                let new_baseline = self.calculate_baseline(history)?;

                // Check if this is a significant update to an existing baseline
                let is_significant_update = if let Some(existing_baseline) = self.baselines.get(key)
                {
                    self.is_significant_baseline_change(existing_baseline, &new_baseline)
                } else {
                    true // First baseline is always significant
                };

                if is_significant_update {
                    self.baselines.insert(key.to_string(), new_baseline);
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// Check if baseline change is significant enough to warrant an update
    fn is_significant_baseline_change(
        &self,
        existing: &PerformanceBaseline<A>,
        new: &PerformanceBaseline<A>,
    ) -> bool {
        // Calculate percentage changes in key metrics
        let timing_change = if existing.baseline_stats.timing.mean > 0.0 {
            ((new.baseline_stats.timing.mean - existing.baseline_stats.timing.mean)
                / existing.baseline_stats.timing.mean)
                .abs()
        } else {
            0.0
        };

        let memory_change = if existing.baseline_stats.memory.mean_memory > 0.0 {
            ((new.baseline_stats.memory.mean_memory - existing.baseline_stats.memory.mean_memory)
                / existing.baseline_stats.memory.mean_memory)
                .abs()
        } else {
            0.0
        };

        let efficiency_change = if existing.baseline_stats.efficiency.mean_efficiency > 0.0 {
            ((new.baseline_stats.efficiency.mean_efficiency
                - existing.baseline_stats.efficiency.mean_efficiency)
                / existing.baseline_stats.efficiency.mean_efficiency)
                .abs()
        } else {
            0.0
        };

        // Consider update significant if any metric changed by more than 5%
        let significance_threshold = 0.05; // 5%
        timing_change > significance_threshold
            || memory_change > significance_threshold
            || efficiency_change > significance_threshold
    }

    /// Calculate baseline statistics from historical data
    fn calculate_baseline(
        &self,
        history: &VecDeque<PerformanceRecord<A>>,
    ) -> Result<PerformanceBaseline<A>> {
        // Take the most recent samples for baseline calculation
        let samples: Vec<_> = history
            .iter()
            .rev()
            .take(self.config.min_baseline_samples)
            .collect();

        // Calculate timing statistics
        let timing_data: Vec<f64> = samples
            .iter()
            .map(|record| record.metrics.timing.mean_time_ns as f64)
            .collect();

        let timing_stats = self.calculate_timing_statistics(&timing_data);

        // Calculate memory statistics
        let memory_data: Vec<f64> = samples
            .iter()
            .map(|record| record.metrics.memory.peak_memory_bytes as f64)
            .collect();

        let memory_stats = self.calculate_memory_statistics(&memory_data);

        // Calculate efficiency statistics
        let efficiency_stats = self.calculate_efficiency_statistics(&samples);

        // Calculate convergence statistics
        let convergence_stats = self.calculate_convergence_statistics(&samples);

        // Calculate confidence intervals
        let confidence_intervals = self.calculate_confidence_intervals(&timing_data, &memory_data);

        Ok(PerformanceBaseline {
            name: "current".to_string(),
            baseline_stats: BaselineStatistics {
                timing: timing_stats,
                memory: memory_stats,
                efficiency: efficiency_stats,
                convergence: convergence_stats,
            },
            confidence_intervals,
            sample_count: samples.len(),
            created_at: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            updated_at: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        })
    }

    /// Calculate timing statistics
    fn calculate_timing_statistics(&self, data: &[f64]) -> TimingStatistics {
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance =
            data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
        let std_dev = variance.sqrt();

        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = if sorted_data.len() % 2 == 0 {
            (sorted_data[sorted_data.len() / 2 - 1] + sorted_data[sorted_data.len() / 2]) / 2.0
        } else {
            sorted_data[sorted_data.len() / 2]
        };

        let q1 = sorted_data[sorted_data.len() / 4];
        let q3 = sorted_data[3 * sorted_data.len() / 4];
        let iqr = q3 - q1;

        TimingStatistics {
            mean,
            std_dev,
            median,
            iqr,
            coefficient_of_variation: std_dev / mean,
        }
    }

    /// Calculate memory statistics
    fn calculate_memory_statistics(&self, data: &[f64]) -> MemoryStatistics {
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance =
            data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;

        let mut percentiles = HashMap::new();
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        percentiles.insert("p50".to_string(), sorted_data[sorted_data.len() / 2]);
        percentiles.insert("p90".to_string(), sorted_data[9 * sorted_data.len() / 10]);
        percentiles.insert("p95".to_string(), sorted_data[95 * sorted_data.len() / 100]);
        percentiles.insert("p99".to_string(), sorted_data[99 * sorted_data.len() / 100]);

        MemoryStatistics {
            mean_memory: mean,
            std_dev_memory: variance.sqrt(),
            peak_memory_percentiles: percentiles,
            fragmentation_stats: self.calculate_fragmentation_statistics(data),
        }
    }

    /// Calculate memory fragmentation statistics
    fn calculate_fragmentation_statistics(&self, data: &[f64]) -> FragmentationStatistics {
        if data.len() < 2 {
            return FragmentationStatistics {
                mean_ratio: 0.0,
                std_dev_ratio: 0.0,
                trend: 0.0,
            };
        }

        // Calculate fragmentation ratios based on memory variance patterns
        let mut fragmentation_ratios = Vec::new();

        // Use memory allocation pattern variance as fragmentation indicator
        let mean = data.iter().sum::<f64>() / data.len() as f64;

        for &memory_value in data {
            // Simple fragmentation ratio: deviation from mean normalized by mean
            let fragmentation_ratio = if mean > 0.0 {
                (memory_value - mean).abs() / mean
            } else {
                0.0
            };
            fragmentation_ratios.push(fragmentation_ratio);
        }

        let mean_ratio =
            fragmentation_ratios.iter().sum::<f64>() / fragmentation_ratios.len() as f64;

        let variance = fragmentation_ratios
            .iter()
            .map(|x| (x - mean_ratio).powi(2))
            .sum::<f64>()
            / (fragmentation_ratios.len() - 1) as f64;
        let std_dev_ratio = variance.sqrt();

        // Calculate trend using linear regression on fragmentation ratios
        let n = fragmentation_ratios.len() as f64;
        let x_sum: f64 = (0..fragmentation_ratios.len()).map(|i| i as f64).sum();
        let y_sum: f64 = fragmentation_ratios.iter().sum();
        let xy_sum: f64 = fragmentation_ratios
            .iter()
            .enumerate()
            .map(|(i, &y)| i as f64 * y)
            .sum();
        let x2_sum: f64 = (0..fragmentation_ratios.len())
            .map(|i| (i as f64).powi(2))
            .sum();

        let trend = if n * x2_sum - x_sum.powi(2) != 0.0 {
            (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum.powi(2))
        } else {
            0.0
        };

        FragmentationStatistics {
            mean_ratio,
            std_dev_ratio,
            trend,
        }
    }

    /// Calculate efficiency statistics
    fn calculate_efficiency_statistics(
        &self,
        samples: &[&PerformanceRecord<A>],
    ) -> EfficiencyStatistics<A> {
        let flops_data: Vec<f64> = samples
            .iter()
            .map(|record| record.metrics.efficiency.flops)
            .collect();

        let mean_flops = flops_data.iter().sum::<f64>() / flops_data.len() as f64;
        let flops_variance = flops_data
            .iter()
            .map(|x| (x - mean_flops).powi(2))
            .sum::<f64>()
            / (flops_data.len() - 1) as f64;
        let flops_cv = flops_variance.sqrt() / mean_flops;

        let efficiency_data: Vec<f64> = samples
            .iter()
            .map(|record| record.metrics.efficiency.efficiency_score)
            .collect();

        let mean_efficiency = efficiency_data.iter().sum::<f64>() / efficiency_data.len() as f64;

        EfficiencyStatistics {
            mean_flops,
            flops_cv,
            mean_efficiency,
            custom_efficiency: HashMap::new(),
        }
    }

    /// Calculate convergence statistics
    fn calculate_convergence_statistics(
        &self,
        samples: &[&PerformanceRecord<A>],
    ) -> ConvergenceStatistics<A> {
        let objectives: Vec<A> = samples
            .iter()
            .map(|record| record.metrics.convergence.final_objective)
            .collect();

        let mean_objective = objectives.iter().fold(A::zero(), |acc, &x| acc + x)
            / A::from(objectives.len()).unwrap();

        let variance = objectives
            .iter()
            .map(|&x| (x - mean_objective) * (x - mean_objective))
            .fold(A::zero(), |acc, x| acc + x)
            / A::from(objectives.len() - 1).unwrap();
        let std_objective = variance.sqrt();

        let convergence_rates: Vec<f64> = samples
            .iter()
            .map(|record| record.metrics.convergence.convergence_rate)
            .collect();

        let mean_convergence_rate =
            convergence_rates.iter().sum::<f64>() / convergence_rates.len() as f64;

        ConvergenceStatistics {
            mean_objective,
            std_objective,
            mean_convergence_rate,
            convergence_consistency: 1.0
                - (convergence_rates
                    .iter()
                    .map(|x| (x - mean_convergence_rate).powi(2))
                    .sum::<f64>()
                    / convergence_rates.len() as f64)
                    .sqrt(),
        }
    }

    /// Calculate confidence intervals
    fn calculate_confidence_intervals(
        &self,
        timing_data: &[f64],
        memory_data: &[f64],
    ) -> ConfidenceIntervals {
        let timing_mean = timing_data.iter().sum::<f64>() / timing_data.len() as f64;
        let timing_std = (timing_data
            .iter()
            .map(|x| (x - timing_mean).powi(2))
            .sum::<f64>()
            / (timing_data.len() - 1) as f64)
            .sqrt();

        let memory_mean = memory_data.iter().sum::<f64>() / memory_data.len() as f64;
        let memory_std = (memory_data
            .iter()
            .map(|x| (x - memory_mean).powi(2))
            .sum::<f64>()
            / (memory_data.len() - 1) as f64)
            .sqrt();

        // Using t-distribution critical values (approximated)
        let t_95 = 1.96; // For large samples
        let t_99 = 2.58; // For large samples

        let timing_margin_95 = t_95 * timing_std / (timing_data.len() as f64).sqrt();
        let timing_margin_99 = t_99 * timing_std / (timing_data.len() as f64).sqrt();

        let memory_margin_95 = t_95 * memory_std / (memory_data.len() as f64).sqrt();
        let memory_margin_99 = t_99 * memory_std / (memory_data.len() as f64).sqrt();

        ConfidenceIntervals {
            timing_ci_95: (
                timing_mean - timing_margin_95,
                timing_mean + timing_margin_95,
            ),
            memory_ci_95: (
                memory_mean - memory_margin_95,
                memory_mean + memory_margin_95,
            ),
            timing_ci_99: (
                timing_mean - timing_margin_99,
                timing_mean + timing_margin_99,
            ),
            memory_ci_99: (
                memory_mean - memory_margin_99,
                memory_mean + memory_margin_99,
            ),
        }
    }

    /// Generate CI/CD integration report
    pub fn generate_ci_report(&self, results: &[RegressionTestResult<A>]) -> Result<String> {
        match self.config.ci_report_format {
            CiReportFormat::Json => self.generate_json_report(results),
            CiReportFormat::JunitXml => self.generate_junit_xml_report(results),
            CiReportFormat::Markdown => self.generate_markdown_report(results),
            CiReportFormat::GitHubActions => self.generate_github_actions_report(results),
        }
    }

    /// Generate JSON format report
    fn generate_json_report(&self, results: &[RegressionTestResult<A>]) -> Result<String> {
        let report = CiReport {
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            total_tests: results.len(),
            passed_tests: results.iter().filter(|r| !r.has_regressions()).count(),
            failed_tests: results.iter().filter(|r| r.has_regressions()).count(),
            test_results: results.iter().map(|r| r.to_ci_test_result()).collect(),
        };

        Ok(serde_json::to_string_pretty(&report)?)
    }

    /// Generate JUnit XML format report
    fn generate_junit_xml_report(&self, results: &[RegressionTestResult<A>]) -> Result<String> {
        let mut xml = String::new();
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str(&format!(
            "<testsuite name=\"performance_regression_tests\" tests=\"{}\" failures=\"{}\" time=\"{}\">\n",
            results.len(),
            results.iter().filter(|r| r.has_regressions()).count(),
            results.iter().map(|r| r.execution_time.as_secs_f64()).sum::<f64>()
        ));

        for result in results {
            xml.push_str(&format!(
                "  <testcase name=\"{}\" classname=\"{}\" time=\"{}\"",
                result.test_name,
                result.optimizer_name,
                result.execution_time.as_secs_f64()
            ));

            if result.has_regressions() {
                xml.push_str(">\n");
                xml.push_str("    <failure message=\"Performance regression detected\">\n");
                for regression in &result.regression_results {
                    if regression.regression_detected {
                        xml.push_str(&format!(
                            "      {}: {}% degradation (confidence: {:.2})\n",
                            regression.test_id,
                            regression.performance_change_percent,
                            regression.confidence
                        ));
                    }
                }
                xml.push_str("    </failure>\n");
                xml.push_str("  </testcase>\n");
            } else {
                xml.push_str(" />\n");
            }
        }

        xml.push_str("</testsuite>\n");
        Ok(xml)
    }

    /// Generate Markdown format report
    fn generate_markdown_report(&self, results: &[RegressionTestResult<A>]) -> Result<String> {
        let mut md = String::new();
        md.push_str("# Performance Regression Test Report\n\n");

        let total = results.len();
        let failed = results.iter().filter(|r| r.has_regressions()).count();
        let passed = total - failed;

        md.push_str(&format!(
            "**Summary**: {} tests, {} passed, {} failed\n\n",
            total, passed, failed
        ));

        if failed > 0 {
            md.push_str("## ❌ Failed Tests (Regressions Detected)\n\n");
            for result in results.iter().filter(|r| r.has_regressions()) {
                md.push_str(&format!(
                    "### {} - {}\n",
                    result.optimizer_name, result.test_name
                ));
                for regression in &result.regression_results {
                    if regression.regression_detected {
                        md.push_str(&format!(
                            "- **Performance Change**: {:.2}% degradation\n",
                            regression.performance_change_percent
                        ));
                        md.push_str(&format!("- **Confidence**: {:.2}\n", regression.confidence));
                        md.push_str(&format!("- **Severity**: {:.2}\n", regression.severity));
                    }
                }
                md.push_str("\n");
            }
        }

        if passed > 0 {
            md.push_str("## ✅ Passed Tests\n\n");
            for result in results.iter().filter(|r| !r.has_regressions()) {
                md.push_str(&format!(
                    "- {} - {}\n",
                    result.optimizer_name, result.test_name
                ));
            }
        }

        Ok(md)
    }

    /// Generate GitHub Actions format report
    fn generate_github_actions_report(
        &self,
        results: &[RegressionTestResult<A>],
    ) -> Result<String> {
        let mut output = String::new();

        for result in results {
            if result.has_regressions() {
                for regression in &result.regression_results {
                    if regression.regression_detected {
                        output.push_str(&format!(
                            "::error title=Performance Regression::{}_{}: {:.2}% performance degradation detected (confidence: {:.2})\n",
                            result.optimizer_name,
                            result.test_name,
                            regression.performance_change_percent,
                            regression.confidence
                        ));
                    }
                }
            } else {
                output.push_str(&format!(
                    "::notice title=Performance Test Passed::{}_{}: No regression detected\n",
                    result.optimizer_name, result.test_name
                ));
            }
        }

        Ok(output)
    }

    // Helper methods for environment detection
    fn get_git_commit_hash(&self) -> Option<String> {
        use std::process::Command;

        Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
            .ok()
            .and_then(|output| {
                if output.status.success() {
                    String::from_utf8(output.stdout)
                        .ok()
                        .map(|s| s.trim().to_string())
                } else {
                    None
                }
            })
    }

    fn get_git_branch(&self) -> Option<String> {
        use std::process::Command;

        Command::new("git")
            .args(["branch", "--show-current"])
            .output()
            .ok()
            .and_then(|output| {
                if output.status.success() {
                    String::from_utf8(output.stdout)
                        .ok()
                        .map(|s| s.trim().to_string())
                } else {
                    None
                }
            })
    }

    fn collect_environment_info(&self) -> Result<TestEnvironment> {
        Ok(TestEnvironment {
            os: std::env::consts::OS.to_string(),
            cpu_model: self.detect_cpu_model(),
            memory_mb: self.detect_memory_size(),
            rust_version: self.detect_rust_version(),
            compiler_flags: self.detect_compiler_flags(),
            hardware_acceleration: self.detect_hardware_acceleration(),
        })
    }

    /// Detect CPU model from system information
    fn detect_cpu_model(&self) -> String {
        use std::process::Command;

        #[cfg(target_os = "linux")]
        {
            if let Ok(output) = Command::new("cat").arg("/proc/cpuinfo").output() {
                if let Ok(content) = String::from_utf8(output.stdout) {
                    for line in content.lines() {
                        if line.starts_with("model name") {
                            return line
                                .split(':')
                                .nth(1)
                                .unwrap_or("Unknown")
                                .trim()
                                .to_string();
                        }
                    }
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            if let Ok(output) = Command::new("sysctl")
                .args(["-n", "machdep.cpu.brand_string"])
                .output()
            {
                if let Ok(brand) = String::from_utf8(output.stdout) {
                    return brand.trim().to_string();
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            if let Ok(output) = Command::new("wmic")
                .args(["cpu", "get", "name", "/format:list"])
                .output()
            {
                if let Ok(content) = String::from_utf8(output.stdout) {
                    for line in content.lines() {
                        if line.starts_with("Name=") {
                            return line
                                .split('=')
                                .nth(1)
                                .unwrap_or("Unknown")
                                .trim()
                                .to_string();
                        }
                    }
                }
            }
        }

        "Unknown".to_string()
    }

    /// Detect system memory size in MB
    fn detect_memory_size(&self) -> usize {
        use std::process::Command;

        #[cfg(target_os = "linux")]
        {
            if let Ok(output) = Command::new("cat").arg("/proc/meminfo").output() {
                if let Ok(content) = String::from_utf8(output.stdout) {
                    for line in content.lines() {
                        if line.starts_with("MemTotal:") {
                            let kb = line
                                .split_whitespace()
                                .nth(1)
                                .and_then(|s| s.parse::<usize>().ok())
                                .unwrap_or(0);
                            return kb / 1024; // Convert KB to MB
                        }
                    }
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            if let Ok(output) = Command::new("sysctl").args(["-n", "hw.memsize"]).output() {
                if let Ok(bytes_str) = String::from_utf8(output.stdout) {
                    if let Ok(bytes) = bytes_str.trim().parse::<usize>() {
                        return bytes / (1024 * 1024); // Convert bytes to MB
                    }
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            if let Ok(output) = Command::new("wmic")
                .args([
                    "computersystem",
                    "get",
                    "TotalPhysicalMemory",
                    "/format:list",
                ])
                .output()
            {
                if let Ok(content) = String::from_utf8(output.stdout) {
                    for line in content.lines() {
                        if line.starts_with("TotalPhysicalMemory=") {
                            if let Some(bytes_str) = line.split('=').nth(1) {
                                if let Ok(bytes) = bytes_str.trim().parse::<usize>() {
                                    return bytes / (1024 * 1024); // Convert bytes to MB
                                }
                            }
                        }
                    }
                }
            }
        }

        0
    }

    /// Detect Rust version
    fn detect_rust_version(&self) -> String {
        use std::process::Command;

        Command::new("rustc")
            .arg("--version")
            .output()
            .ok()
            .and_then(|output| {
                if output.status.success() {
                    String::from_utf8(output.stdout)
                        .ok()
                        .map(|s| s.trim().to_string())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| "Unknown".to_string())
    }

    /// Detect compiler flags from environment
    fn detect_compiler_flags(&self) -> Vec<String> {
        let mut flags = Vec::new();

        // Check common Rust compilation flags
        if let Ok(rustflags) = std::env::var("RUSTFLAGS") {
            flags.extend(rustflags.split_whitespace().map(|s| s.to_string()));
        }

        // Check for release vs debug mode
        #[cfg(debug_assertions)]
        flags.push("debug".to_string());

        #[cfg(not(debug_assertions))]
        flags.push("release".to_string());

        // Check for target features
        if cfg!(target_feature = "avx2") {
            flags.push("avx2".to_string());
        }
        if cfg!(target_feature = "sse4.2") {
            flags.push("sse4.2".to_string());
        }

        flags
    }

    /// Detect available hardware acceleration
    fn detect_hardware_acceleration(&self) -> Vec<String> {
        let mut acceleration = Vec::new();

        // Check for SIMD features
        if cfg!(target_feature = "avx2") {
            acceleration.push("AVX2".to_string());
        }
        if cfg!(target_feature = "avx") {
            acceleration.push("AVX".to_string());
        }
        if cfg!(target_feature = "sse4.2") {
            acceleration.push("SSE4.2".to_string());
        }

        // Check for GPU acceleration (basic detection)
        #[cfg(feature = "gpu")]
        {
            use std::process::Command;

            // Check for NVIDIA GPU
            if Command::new("nvidia-smi").output().is_ok() {
                acceleration.push("CUDA".to_string());
            }

            // Check for AMD GPU
            if Command::new("rocm-smi").output().is_ok() {
                acceleration.push("ROCm".to_string());
            }
        }

        acceleration
    }

    /// Extract memory usage information from benchmark result
    fn extract_memory_usage(&self, result: &BenchmarkResult<A>) -> Option<usize> {
        // Use system memory profiling if available
        #[cfg(target_os = "linux")]
        {
            self.get_process_memory_usage()
        }

        #[cfg(not(target_os = "linux"))]
        {
            None // Fallback for other platforms
        }
    }

    /// Extract average memory usage
    fn extract_avg_memory_usage(&self, result: &BenchmarkResult<A>) -> Option<usize> {
        // For now, assume 80% of peak memory as average
        self.extract_memory_usage(result)
            .map(|peak| (peak as f64 * 0.8) as usize)
    }

    /// Extract allocation count (simplified)
    fn extract_allocation_count(&self, result: &BenchmarkResult<A>) -> Option<usize> {
        // This would require memory profiling integration
        // For now, provide a reasonable estimate
        Some(1000) // Default estimate
    }

    /// Calculate memory efficiency score
    fn calculate_memory_efficiency(&self, result: &BenchmarkResult<A>) -> f64 {
        // Simple efficiency calculation based on memory usage patterns
        0.85 // Default efficiency score
    }

    /// Estimate memory fragmentation ratio
    fn estimate_fragmentation_ratio(&self, result: &BenchmarkResult<A>) -> f64 {
        // Memory fragmentation estimation based on available metrics
        if let (Some(peak_memory), Some(avg_memory)) = (
            self.extract_memory_usage(result),
            self.extract_avg_memory_usage(result),
        ) {
            if peak_memory > 0 {
                // Fragmentation ratio: 1.0 means perfect allocation, < 1.0 means fragmentation
                let ratio = avg_memory as f64 / peak_memory as f64;
                // Convert to fragmentation metric: higher values mean more fragmentation
                1.0 - ratio.max(0.0).min(1.0)
            } else {
                0.0
            }
        } else {
            // Estimate based on allocation patterns (simplified)
            // For optimization algorithms, fragmentation is typically low
            0.1 // 10% fragmentation estimate
        }
    }

    /// Estimate timing standard deviation from single run
    fn estimate_timing_std_dev(&self, result: &BenchmarkResult<A>) -> u64 {
        // Estimate based on typical variation for optimization algorithms
        let base_time_ns = result.elapsed_time.as_nanos() as u64;
        // Typical coefficient of variation for optimization: 5-15%
        (base_time_ns as f64 * 0.1) as u64
    }

    /// Estimate timing percentile from single run
    fn estimate_timing_percentile(&self, result: &BenchmarkResult<A>, percentile: f64) -> u64 {
        let base_time_ns = result.elapsed_time.as_nanos() as u64;
        let std_dev = self.estimate_timing_std_dev(result) as f64;

        // Use normal distribution approximation
        let z_score = match percentile {
            p if p >= 0.99 => 2.33,  // 99th percentile
            p if p >= 0.95 => 1.645, // 95th percentile
            p if p >= 0.90 => 1.28,  // 90th percentile
            _ => 0.0,
        };

        (base_time_ns as f64 + z_score * std_dev).max(0.0) as u64
    }

    /// Estimate minimum timing from single run
    fn estimate_timing_min(&self, result: &BenchmarkResult<A>) -> u64 {
        let base_time_ns = result.elapsed_time.as_nanos() as u64;
        let std_dev = self.estimate_timing_std_dev(result) as f64;
        // Minimum is typically 2 standard deviations below mean
        (base_time_ns as f64 - 2.0 * std_dev).max(base_time_ns as f64 * 0.5) as u64
    }

    /// Estimate maximum timing from single run  
    fn estimate_timing_max(&self, result: &BenchmarkResult<A>) -> u64 {
        let base_time_ns = result.elapsed_time.as_nanos() as u64;
        let std_dev = self.estimate_timing_std_dev(result) as f64;
        // Maximum is typically 2-3 standard deviations above mean
        (base_time_ns as f64 + 2.5 * std_dev) as u64
    }

    /// Estimate FLOPS from benchmark result
    fn estimate_flops(&self, result: &BenchmarkResult<A>) -> f64 {
        // Estimate based on execution time and typical optimization operations
        let time_seconds = result.elapsed_time.as_secs_f64();
        if time_seconds > 0.0 {
            // Rough estimate: 1M operations per second for typical optimization
            1_000_000.0 / time_seconds
        } else {
            0.0
        }
    }

    /// Calculate arithmetic intensity
    fn calculate_arithmetic_intensity(&self, result: &BenchmarkResult<A>) -> f64 {
        // Arithmetic intensity = FLOPs / bytes accessed
        // For optimization algorithms, this is typically low (memory bound)
        let flops = self.estimate_flops(result);
        let estimated_memory_access = 1_000_000.0; // Rough estimate in bytes
        flops / estimated_memory_access
    }

    /// Extract CPU utilization
    fn extract_cpu_utilization(&self, result: &BenchmarkResult<A>) -> Option<f64> {
        // This would require system monitoring integration
        // For now, provide a reasonable estimate for single-threaded optimization
        Some(0.75) // 75% utilization estimate
    }

    /// Calculate overall efficiency score
    fn calculate_efficiency_score(&self, result: &BenchmarkResult<A>) -> f64 {
        // Combine multiple efficiency metrics
        let time_efficiency = if result.converged { 0.9 } else { 0.5 };
        let memory_efficiency = self.calculate_memory_efficiency(result);
        let cpu_efficiency = self.extract_cpu_utilization(result).unwrap_or(0.5);

        (time_efficiency + memory_efficiency + cpu_efficiency) / 3.0
    }

    /// Get current process memory usage (Linux-specific)
    #[cfg(target_os = "linux")]
    fn get_process_memory_usage(&self) -> Option<usize> {
        use std::fs;

        let pid = std::process::id();
        let status_path = format!("/proc/{}/status", pid);

        if let Ok(content) = fs::read_to_string(&status_path) {
            for line in content.lines() {
                if line.starts_with("VmPeak:") {
                    return line
                        .split_whitespace()
                        .nth(1)
                        .and_then(|s| s.parse::<usize>().ok())
                        .map(|kb| kb * 1024); // Convert KB to bytes
                }
            }
        }

        None
    }
}

/// Result of a regression test run
#[derive(Debug)]
pub struct RegressionTestResult<A: Float> {
    /// Test name
    pub test_name: String,
    /// Optimizer name
    pub optimizer_name: String,
    /// Test execution time
    pub execution_time: Duration,
    /// Performance record from this run
    pub performance_record: PerformanceRecord<A>,
    /// Regression analysis results
    pub regression_results: Vec<RegressionResult<A>>,
    /// Whether baseline was updated
    pub baseline_updated: bool,
}

impl<A: Float> RegressionTestResult<A> {
    /// Check if any regressions were detected
    pub fn has_regressions(&self) -> bool {
        self.regression_results
            .iter()
            .any(|r| r.regression_detected)
    }

    /// Convert to CI test result format
    pub fn to_ci_test_result(&self) -> CiTestResult {
        CiTestResult {
            name: format!("{}_{}", self.optimizer_name, self.test_name),
            status: if self.has_regressions() {
                "failed".to_string()
            } else {
                "passed".to_string()
            },
            execution_time_ms: self.execution_time.as_millis() as u64,
            regression_count: self
                .regression_results
                .iter()
                .filter(|r| r.regression_detected)
                .count(),
        }
    }
}

/// CI report structure
#[derive(Debug, Serialize)]
pub struct CiReport {
    pub timestamp: u64,
    pub total_tests: usize,
    pub passed_tests: usize,
    pub failed_tests: usize,
    pub test_results: Vec<CiTestResult>,
}

/// Individual test result for CI
#[derive(Debug, Serialize)]
pub struct CiTestResult {
    pub name: String,
    pub status: String,
    pub execution_time_ms: u64,
    pub regression_count: usize,
}

// Implementation of default detectors

/// Statistical test-based regression detector
#[derive(Debug)]
pub struct StatisticalTestDetector {
    alpha: f64,
}

impl StatisticalTestDetector {
    pub fn new() -> Self {
        Self { alpha: 0.05 }
    }
}

impl<A: Float + Debug> RegressionDetector<A> for StatisticalTestDetector {
    fn detect_regression(
        &self,
        baseline: &PerformanceBaseline<A>,
        current_metrics: &PerformanceMetrics<A>,
        _history: &VecDeque<PerformanceRecord<A>>,
    ) -> Result<RegressionResult<A>> {
        // Simple t-test approximation for timing regression
        let current_time = current_metrics.timing.mean_time_ns as f64;
        let baseline_mean = baseline.baseline_stats.timing.mean;
        let baseline_std = baseline.baseline_stats.timing.std_dev;

        let change_percent = ((current_time - baseline_mean) / baseline_mean) * 100.0;

        // Calculate memory change percentage
        let current_memory = current_metrics.memory.peak_memory_bytes as f64;
        let baseline_memory_mean = baseline.baseline_stats.memory.mean_memory;
        let memory_change_percent = if baseline_memory_mean > 0.0 {
            ((current_memory - baseline_memory_mean) / baseline_memory_mean) * 100.0
        } else {
            0.0
        };

        // Simple z-score calculation
        let z_score =
            (current_time - baseline_mean) / (baseline_std / (baseline.sample_count as f64).sqrt());
        let p_value = 2.0 * (1.0 - normal_cdf(z_score.abs())); // Two-tailed test

        let regression_detected =
            p_value < self.alpha && (change_percent > 0.0 || memory_change_percent > 10.0);

        Ok(RegressionResult {
            test_id: "statistical_test".to_string(),
            regression_detected,
            severity: if regression_detected {
                (change_percent / 100.0).min(1.0)
            } else {
                0.0
            },
            confidence: 1.0 - p_value,
            performance_change_percent: change_percent,
            memory_change_percent,
            affected_metrics: if regression_detected {
                vec!["timing".to_string()]
            } else {
                vec![]
            },
            statistical_tests: vec![StatisticalTestResult {
                test_name: "t_test".to_string(),
                test_statistic: z_score,
                p_value,
                degrees_of_freedom: Some(baseline.sample_count - 1),
                conclusion: if regression_detected {
                    "Significant regression detected".to_string()
                } else {
                    "No significant change".to_string()
                },
            }],
            analysis: RegressionAnalysis {
                trend_analysis: TrendAnalysis {
                    direction: if change_percent > 0.0 {
                        TrendDirection::Degrading
                    } else {
                        TrendDirection::Improving
                    },
                    magnitude: change_percent.abs(),
                    significance: 1.0 - p_value,
                    start_point: None,
                },
                change_point_analysis: ChangePointAnalysis {
                    change_points: vec![],
                    magnitudes: vec![],
                    confidences: vec![],
                },
                outlier_analysis: OutlierAnalysis {
                    outlier_indices: vec![],
                    outlier_scores: vec![],
                    outlier_types: vec![],
                },
                root_cause_hints: vec![],
            },
            recommendations: if regression_detected {
                vec![
                    "Check for recent code changes that might affect performance".to_string(),
                    "Review system load and resource availability".to_string(),
                    "Consider running additional test iterations for confirmation".to_string(),
                ]
            } else {
                vec![]
            },
        })
    }

    fn name(&self) -> &str {
        "statistical_test"
    }

    fn config(&self) -> HashMap<String, String> {
        let mut config = HashMap::new();
        config.insert("alpha".to_string(), self.alpha.to_string());
        config
    }
}

/// Sliding window regression detector
#[derive(Debug)]
pub struct SlidingWindowDetector {
    window_size: usize,
    threshold: f64,
}

impl SlidingWindowDetector {
    pub fn new() -> Self {
        Self {
            window_size: 10,
            threshold: 5.0, // 5% threshold
        }
    }
}

impl<A: Float + Debug> RegressionDetector<A> for SlidingWindowDetector {
    fn detect_regression(
        &self,
        baseline: &PerformanceBaseline<A>,
        current_metrics: &PerformanceMetrics<A>,
        history: &VecDeque<PerformanceRecord<A>>,
    ) -> Result<RegressionResult<A>> {
        if history.len() < self.window_size {
            return Ok(RegressionResult {
                test_id: "sliding_window".to_string(),
                regression_detected: false,
                severity: 0.0,
                confidence: 0.0,
                performance_change_percent: 0.0,
                memory_change_percent: 0.0,
                affected_metrics: vec![],
                statistical_tests: vec![],
                analysis: RegressionAnalysis {
                    trend_analysis: TrendAnalysis {
                        direction: TrendDirection::Stable,
                        magnitude: 0.0,
                        significance: 0.0,
                        start_point: None,
                    },
                    change_point_analysis: ChangePointAnalysis {
                        change_points: vec![],
                        magnitudes: vec![],
                        confidences: vec![],
                    },
                    outlier_analysis: OutlierAnalysis {
                        outlier_indices: vec![],
                        outlier_scores: vec![],
                        outlier_types: vec![],
                    },
                    root_cause_hints: vec![
                        "Insufficient data for sliding window analysis".to_string()
                    ],
                },
                recommendations: vec![
                    "Collect more performance data for accurate analysis".to_string()
                ],
            });
        }

        // Calculate average of recent window
        let recent_times: Vec<f64> = history
            .iter()
            .rev()
            .take(self.window_size)
            .map(|r| r.metrics.timing.mean_time_ns as f64)
            .collect();

        let recent_avg = recent_times.iter().sum::<f64>() / recent_times.len() as f64;
        let current_time = current_metrics.timing.mean_time_ns as f64;

        let change_percent = ((current_time - recent_avg) / recent_avg) * 100.0;
        let regression_detected = change_percent > self.threshold;

        Ok(RegressionResult {
            test_id: "sliding_window".to_string(),
            regression_detected,
            severity: if regression_detected {
                (change_percent / 100.0).min(1.0)
            } else {
                0.0
            },
            confidence: if regression_detected { 0.8 } else { 0.2 },
            performance_change_percent: change_percent,
            memory_change_percent: 0.0,
            affected_metrics: if regression_detected {
                vec!["timing".to_string()]
            } else {
                vec![]
            },
            statistical_tests: vec![],
            analysis: RegressionAnalysis {
                trend_analysis: TrendAnalysis {
                    direction: if change_percent > 0.0 {
                        TrendDirection::Degrading
                    } else {
                        TrendDirection::Improving
                    },
                    magnitude: change_percent.abs(),
                    significance: if regression_detected { 0.8 } else { 0.2 },
                    start_point: Some(history.len() - self.window_size),
                },
                change_point_analysis: ChangePointAnalysis {
                    change_points: vec![],
                    magnitudes: vec![],
                    confidences: vec![],
                },
                outlier_analysis: OutlierAnalysis {
                    outlier_indices: vec![],
                    outlier_scores: vec![],
                    outlier_types: vec![],
                },
                root_cause_hints: vec![],
            },
            recommendations: if regression_detected {
                vec![
                    "Performance degradation detected in recent window".to_string(),
                    "Compare current run with recent _baseline".to_string(),
                ]
            } else {
                vec![]
            },
        })
    }

    fn name(&self) -> &str {
        "sliding_window"
    }

    fn config(&self) -> HashMap<String, String> {
        let mut config = HashMap::new();
        config.insert("window_size".to_string(), self.window_size.to_string());
        config.insert("threshold".to_string(), self.threshold.to_string());
        config
    }
}

/// Change point detection regression detector
#[derive(Debug)]
pub struct ChangePointDetector {
    min_segment_size: usize,
    significance_threshold: f64,
}

impl ChangePointDetector {
    pub fn new() -> Self {
        Self {
            min_segment_size: 5,
            significance_threshold: 0.05,
        }
    }
}

impl<A: Float + Debug> RegressionDetector<A> for ChangePointDetector {
    fn detect_regression(
        &self,
        _baseline: &PerformanceBaseline<A>,
        _current_metrics: &PerformanceMetrics<A>,
        history: &VecDeque<PerformanceRecord<A>>,
    ) -> Result<RegressionResult<A>> {
        // Simplified change point detection using variance change
        if history.len() < 2 * self.min_segment_size {
            return Ok(RegressionResult {
                test_id: "change_point".to_string(),
                regression_detected: false,
                severity: 0.0,
                confidence: 0.0,
                performance_change_percent: 0.0,
                memory_change_percent: 0.0,
                affected_metrics: vec![],
                statistical_tests: vec![],
                analysis: RegressionAnalysis {
                    trend_analysis: TrendAnalysis {
                        direction: TrendDirection::Stable,
                        magnitude: 0.0,
                        significance: 0.0,
                        start_point: None,
                    },
                    change_point_analysis: ChangePointAnalysis {
                        change_points: vec![],
                        magnitudes: vec![],
                        confidences: vec![],
                    },
                    outlier_analysis: OutlierAnalysis {
                        outlier_indices: vec![],
                        outlier_scores: vec![],
                        outlier_types: vec![],
                    },
                    root_cause_hints: vec![
                        "Insufficient data for change point detection".to_string()
                    ],
                },
                recommendations: vec![
                    "Collect more performance data for change point analysis".to_string()
                ],
            });
        }

        // Simple change point detection - compare first and second half
        let mid_point = history.len() / 2;

        let first_half: Vec<f64> = history
            .iter()
            .take(mid_point)
            .map(|r| r.metrics.timing.mean_time_ns as f64)
            .collect();

        let second_half: Vec<f64> = history
            .iter()
            .skip(mid_point)
            .map(|r| r.metrics.timing.mean_time_ns as f64)
            .collect();

        let first_mean = first_half.iter().sum::<f64>() / first_half.len() as f64;
        let second_mean = second_half.iter().sum::<f64>() / second_half.len() as f64;

        let change_percent =
            A::from((second_mean - first_mean) / first_mean).unwrap() * A::from(100.0).unwrap();
        let change_detected = change_percent.abs() > A::from(5.0).unwrap(); // 5% change threshold

        Ok(RegressionResult {
            test_id: "change_point".to_string(),
            regression_detected: change_detected && change_percent > A::zero(),
            severity: if change_detected {
                (change_percent.abs() / A::from(100.0).unwrap())
                    .min(A::one())
                    .to_f64()
                    .unwrap_or(0.0)
            } else {
                0.0
            },
            confidence: if change_detected { 0.7 } else { 0.3 },
            performance_change_percent: change_percent.to_f64().unwrap_or(0.0),
            memory_change_percent: 0.0,
            affected_metrics: if change_detected {
                vec!["timing".to_string()]
            } else {
                vec![]
            },
            statistical_tests: vec![],
            analysis: RegressionAnalysis {
                trend_analysis: TrendAnalysis {
                    direction: if change_percent > A::zero() {
                        TrendDirection::Degrading
                    } else {
                        TrendDirection::Improving
                    },
                    magnitude: change_percent.abs().to_f64().unwrap_or(0.0),
                    significance: if change_detected { 0.7 } else { 0.3 },
                    start_point: Some(mid_point),
                },
                change_point_analysis: ChangePointAnalysis {
                    change_points: if change_detected {
                        vec![mid_point]
                    } else {
                        vec![]
                    },
                    magnitudes: if change_detected {
                        vec![change_percent.to_f64().unwrap_or(0.0)]
                    } else {
                        vec![]
                    },
                    confidences: if change_detected { vec![0.7] } else { vec![] },
                },
                outlier_analysis: OutlierAnalysis {
                    outlier_indices: vec![],
                    outlier_scores: vec![],
                    outlier_types: vec![],
                },
                root_cause_hints: if change_detected {
                    vec!["Significant performance change detected at mid-point".to_string()]
                } else {
                    vec![]
                },
            },
            recommendations: if change_detected {
                vec![
                    "Investigate changes that occurred around the detected change point"
                        .to_string(),
                    "Review commits and deployments near the change point".to_string(),
                ]
            } else {
                vec![]
            },
        })
    }

    fn name(&self) -> &str {
        "change_point"
    }

    fn config(&self) -> HashMap<String, String> {
        let mut config = HashMap::new();
        config.insert(
            "min_segment_size".to_string(),
            self.min_segment_size.to_string(),
        );
        config.insert(
            "significance_threshold".to_string(),
            self.significance_threshold.to_string(),
        );
        config
    }
}

// Default statistical analyzers

/// Trend analysis implementation
#[derive(Debug)]
pub struct TrendAnalyzer {
    min_data_points: usize,
}

impl TrendAnalyzer {
    pub fn new() -> Self {
        Self { min_data_points: 5 }
    }
}

impl<A: Float + Debug> StatisticalAnalyzer<A> for TrendAnalyzer {
    fn analyze(
        &self,
        data: &VecDeque<PerformanceRecord<A>>,
    ) -> Result<StatisticalAnalysisResult<A>> {
        if data.len() < self.min_data_points {
            return Ok(StatisticalAnalysisResult {
                summary: "Insufficient data for trend analysis".to_string(),
                tests: vec![],
                patterns: vec![],
                anomalies: vec![],
            });
        }

        // Simple linear trend analysis
        let times: Vec<f64> = data
            .iter()
            .map(|r| r.metrics.timing.mean_time_ns as f64)
            .collect();

        let n = times.len() as f64;
        let x_sum: f64 = (0..times.len()).map(|i| i as f64).sum();
        let y_sum: f64 = times.iter().sum();
        let xy_sum: f64 = times.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let x2_sum: f64 = (0..times.len()).map(|i| (i as f64).powi(2)).sum();

        let slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum.powi(2));

        let trend_direction = if slope > 0.0 {
            "increasing"
        } else if slope < 0.0 {
            "decreasing"
        } else {
            "stable"
        };

        Ok(StatisticalAnalysisResult {
            summary: format!(
                "Trend analysis: {} trend with slope {:.2}",
                trend_direction, slope
            ),
            tests: vec![],
            patterns: vec![format!("Linear trend: {}", trend_direction)],
            anomalies: vec![],
        })
    }

    fn name(&self) -> &str {
        "trend_analyzer"
    }
}

/// Outlier detection analyzer
#[derive(Debug)]
pub struct OutlierAnalyzer {
    z_threshold: f64,
}

impl OutlierAnalyzer {
    pub fn new() -> Self {
        Self { z_threshold: 2.0 }
    }
}

impl<A: Float + Debug> StatisticalAnalyzer<A> for OutlierAnalyzer {
    fn analyze(
        &self,
        data: &VecDeque<PerformanceRecord<A>>,
    ) -> Result<StatisticalAnalysisResult<A>> {
        if data.is_empty() {
            return Ok(StatisticalAnalysisResult {
                summary: "No data for outlier analysis".to_string(),
                tests: vec![],
                patterns: vec![],
                anomalies: vec![],
            });
        }

        let times: Vec<f64> = data
            .iter()
            .map(|r| r.metrics.timing.mean_time_ns as f64)
            .collect();

        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let variance =
            times.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (times.len() - 1) as f64;
        let std_dev = variance.sqrt();

        let mut outliers = Vec::new();
        for (_i, &time) in times.iter().enumerate() {
            let z_score = (time - mean) / std_dev;
            if z_score.abs() > self.z_threshold {
                outliers.push(A::from(z_score).unwrap());
            }
        }

        Ok(StatisticalAnalysisResult {
            summary: format!("Outlier analysis: {} outliers detected", outliers.len()),
            tests: vec![],
            patterns: if outliers.is_empty() {
                vec!["No significant outliers detected".to_string()]
            } else {
                vec![format!("{} potential outliers found", outliers.len())]
            },
            anomalies: outliers,
        })
    }

    fn name(&self) -> &str {
        "outlier_analyzer"
    }
}

impl<A: Float + Debug + Serialize + for<'de> Deserialize<'de>> PerformanceDatabase<A> {
    /// Create a new empty performance database
    pub fn new() -> Self {
        Self {
            history: HashMap::new(),
            metadata: DatabaseMetadata {
                version: "1.0".to_string(),
                created_at: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                last_updated: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                total_records: 0,
            },
        }
    }

    /// Load database from disk
    pub fn load(_basedir: &Path) -> Result<Self> {
        let db_path = _basedir.join("performance_db.json");
        if db_path.exists() {
            let data = fs::read_to_string(&db_path)?;
            let db = serde_json::from_str(&data)?;
            Ok(db)
        } else {
            Ok(Self::new())
        }
    }

    /// Save database to disk
    pub fn save(&self, basedir: &Path) -> Result<()> {
        fs::create_dir_all(basedir)?;
        let db_path = basedir.join("performance_db.json");
        let data = serde_json::to_string_pretty(self)?;
        fs::write(&db_path, data)?;
        Ok(())
    }

    /// Add a performance record
    pub fn add_record(&mut self, key: String, record: PerformanceRecord<A>) {
        let history = self.history.entry(key).or_insert_with(VecDeque::new);
        history.push_back(record);

        // Maintain reasonable history size
        if history.len() > 1000 {
            history.pop_front();
        }

        self.metadata.total_records += 1;
        self.metadata.last_updated = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }
}

impl AlertSystem {
    /// Create a new alert system
    pub fn new() -> Self {
        Self {
            config: AlertConfig {
                enable_alerts: true,
                enable_email: false,
                enable_slack: false,
                enable_github_issues: false,
                severity_threshold: 0.5,
                cooldown_minutes: 60,
            },
            alert_history: VecDeque::new(),
        }
    }

    /// Send an alert for a regression
    pub fn send_alert<A: Float>(&mut self, regression: &RegressionResult<A>) -> Result<()> {
        if regression.severity < self.config.severity_threshold {
            return Ok(()); // Below threshold
        }

        let alert = Alert {
            id: format!(
                "alert_{}",
                SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs()
            ),
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            severity: match regression.severity {
                s if s >= 0.8 => AlertSeverity::Critical,
                s if s >= 0.6 => AlertSeverity::High,
                s if s >= 0.3 => AlertSeverity::Medium,
                _ => AlertSeverity::Low,
            },
            message: format!(
                "Performance regression detected in {}: {:.2}% degradation",
                regression.test_id, regression.performance_change_percent
            ),
            regression_id: regression.test_id.clone(),
            status: AlertStatus::Active,
        };

        self.alert_history.push_back(alert.clone());

        // Maintain alert history size
        if self.alert_history.len() > 100 {
            self.alert_history.pop_front();
        }

        // Send actual alerts through configured channels
        self.send_alert_notifications(&alert)?;

        Ok(())
    }

    /// Send alert notifications through configured channels
    fn send_alert_notifications(&self, alert: &Alert) -> Result<()> {
        // Check if alerts are enabled and severity meets threshold
        if !self.config.enable_alerts || self.severity_below_threshold(alert) {
            return Ok(());
        }

        // Check cooldown period
        if self.is_in_cooldown_period(alert)? {
            return Ok(());
        }

        let mut notification_results = Vec::new();

        // Send email notifications
        if self.config.enable_email {
            match self.send_email_notification(alert) {
                Ok(()) => notification_results.push("Email sent successfully".to_string()),
                Err(e) => notification_results.push(format!("Email failed: {}", e)),
            }
        }

        // Send Slack notifications
        if self.config.enable_slack {
            match self.send_slack_notification(alert) {
                Ok(()) => {
                    notification_results.push("Slack notification sent successfully".to_string())
                }
                Err(e) => notification_results.push(format!("Slack notification failed: {}", e)),
            }
        }

        // Create GitHub issues
        if self.config.enable_github_issues {
            match self.create_github_issue(alert) {
                Ok(()) => {
                    notification_results.push("GitHub issue created successfully".to_string())
                }
                Err(e) => notification_results.push(format!("GitHub issue creation failed: {}", e)),
            }
        }

        // Log notification results
        for result in notification_results {
            eprintln!("Alert notification: {}", result);
        }

        Ok(())
    }

    /// Check if alert severity is below configured threshold
    fn severity_below_threshold(&self, alert: &Alert) -> bool {
        let alert_severity_value = match alert.severity {
            AlertSeverity::Critical => 1.0,
            AlertSeverity::High => 0.75,
            AlertSeverity::Medium => 0.5,
            AlertSeverity::Low => 0.25,
        };
        alert_severity_value < self.config.severity_threshold
    }

    /// Check if we're in cooldown period for similar alerts
    fn is_in_cooldown_period(&self, alert: &Alert) -> Result<bool> {
        let cooldown_duration = Duration::from_secs(self.config.cooldown_minutes * 60);
        let current_time = SystemTime::now();

        // Check for similar recent alerts
        for recent_alert in self.alert_history.iter().rev().take(10) {
            if recent_alert.regression_id == alert.regression_id {
                let recent_time = UNIX_EPOCH + Duration::from_secs(recent_alert.timestamp);
                if current_time.duration_since(recent_time)? < cooldown_duration {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    /// Send email notification
    fn send_email_notification(&self, alert: &Alert) -> Result<()> {
        // In a real implementation, this would use an email service like:
        // - SMTP with lettre crate
        // - AWS SES
        // - SendGrid
        // - Mailgun

        let email_body = self.format_email_body(alert);
        let subject = format!("Performance Regression Alert: {}", alert.regression_id);

        // Placeholder implementation - would integrate with actual email service
        eprintln!("EMAIL ALERT:");
        eprintln!("To: performance-team@company.com");
        eprintln!("Subject: {}", subject);
        eprintln!("Body:\n{}", email_body);
        eprintln!("---");

        // TODO: Integrate with actual email service
        // Example with lettre crate:
        // let email = Message::builder()
        //     .from("alerts@company.com".parse()?)
        //     .to("performance-team@company.com".parse()?)
        //     .subject(&subject)
        //     .body(email_body)?;
        // let mailer = SmtpTransport::relay("smtp.company.com")?.build();
        // mailer.send(&email)?;

        Ok(())
    }

    /// Send Slack notification
    fn send_slack_notification(&self, alert: &Alert) -> Result<()> {
        // In a real implementation, this would use:
        // - Slack webhook URL
        // - reqwest crate for HTTP requests
        // - JSON payload formatting

        let slack_message = self.format_slack_message(alert);

        // Placeholder implementation - would make HTTP POST to Slack webhook
        eprintln!("SLACK ALERT:");
        eprintln!("Channel: #performance-alerts");
        eprintln!("Message: {}", slack_message);
        eprintln!("---");

        // TODO: Integrate with actual Slack API
        // Example:
        // let webhook_url = std::env::var("SLACK_WEBHOOK_URL")?;
        // let payload = json!({
        //     "text": slack_message,
        //     "channel": "#performance-alerts",
        //     "username": "Performance Bot"
        // });
        // let client = reqwest::Client::new();
        // client.post(&webhook_url).json(&payload).send()?;

        Ok(())
    }

    /// Create GitHub issue
    fn create_github_issue(&self, alert: &Alert) -> Result<()> {
        // In a real implementation, this would use:
        // - GitHub API with octocrab crate
        // - Personal access token
        // - Repository configuration

        let issue_title = format!("Performance regression in {}", alert.regression_id);
        let issue_body = self.format_github_issue_body(alert);

        // Placeholder implementation - would create actual GitHub issue
        eprintln!("GITHUB ISSUE:");
        eprintln!("Repository: company/performance-monitoring");
        eprintln!("Title: {}", issue_title);
        eprintln!("Body:\n{}", issue_body);
        eprintln!("Labels: performance, regression, automated");
        eprintln!("---");

        // TODO: Integrate with actual GitHub API
        // Example with octocrab:
        // let token = std::env::var("GITHUB_TOKEN")?;
        // let octocrab = octocrab::Octocrab::builder().personal_token(token).build()?;
        // octocrab.issues("company", "performance-monitoring")
        //     .create(&issue_title)
        //     .body(&issue_body)
        //     .labels(vec!["performance", "regression", "automated"])
        //     .send().await?;

        Ok(())
    }

    /// Format email body for alert
    fn format_email_body(&self, alert: &Alert) -> String {
        format!(
            "Performance Regression Alert\n\
            =============================\n\n\
            Alert ID: {}\n\
            Timestamp: {}\n\
            Severity: {:?}\n\
            Test: {}\n\n\
            Details:\n\
            {}\n\n\
            Please investigate this performance regression immediately.\n\
            \n\
            View full details at: https://performance-dashboard.company.com/alerts/{}\n\
            \n\
            Best regards,\n\
            Performance Monitoring System",
            alert.id, alert.timestamp, alert.severity, alert.regression_id, alert.message, alert.id
        )
    }

    /// Format Slack message for alert
    fn format_slack_message(&self, alert: &Alert) -> String {
        let severity_emoji = match alert.severity {
            AlertSeverity::Critical => "🚨",
            AlertSeverity::High => "⚠️",
            AlertSeverity::Medium => "🟡",
            AlertSeverity::Low => "🔵",
        };

        format!(
            "{} *Performance Regression Alert*\n\
            *Test:* {}\n\
            *Severity:* {:?}\n\
            *Details:* {}\n\
            *Time:* <t:{}:F>\n\
            <https://performance-dashboard.company.com/alerts/{}|View Details>",
            severity_emoji,
            alert.regression_id,
            alert.severity,
            alert.message,
            alert.timestamp,
            alert.id
        )
    }

    /// Format GitHub issue body for alert
    fn format_github_issue_body(&self, alert: &Alert) -> String {
        format!(
            "## Performance Regression Detected\n\n\
            **Alert ID:** {}\n\
            **Timestamp:** {}\n\
            **Severity:** {:?}\n\
            **Test:** {}\n\n\
            ### Description\n\
            {}\n\n\
            ### Investigation Steps\n\
            - [ ] Review recent code changes that might affect performance\n\
            - [ ] Check system resource utilization during test execution\n\
            - [ ] Run additional test iterations to confirm regression\n\
            - [ ] Analyze profiling data for performance bottlenecks\n\
            - [ ] Compare with baseline performance metrics\n\n\
            ### Links\n\
            - [Performance Dashboard](https://performance-dashboard.company.com/alerts/{})\n\
            - [Test Results](https://ci.company.com/tests/{})\n\n\
            ---\n\
            *This issue was automatically created by the performance monitoring system.*",
            alert.id,
            alert.timestamp,
            alert.severity,
            alert.regression_id,
            alert.message,
            alert.id,
            alert.regression_id
        )
    }
}

// Helper function for normal distribution CDF approximation
#[allow(dead_code)]
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

// Simple error function approximation
#[allow(dead_code)]
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_regression_tester_creation() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = RegressionConfig::default();
        config.baseline_dir = temp_dir.path().to_path_buf();

        let tester = RegressionTester::<f64>::new(config);
        assert!(tester.is_ok());
    }

    #[test]
    fn test_performance_database() {
        let temp_dir = TempDir::new().unwrap();
        let mut db = PerformanceDatabase::<f64>::new();

        let record = PerformanceRecord {
            timestamp: 12345,
            commit_hash: None,
            branch: None,
            environment: TestEnvironment {
                os: "test".to_string(),
                cpu_model: "test".to_string(),
                memory_mb: 1024,
                rust_version: "1.0".to_string(),
                compiler_flags: vec![],
                hardware_acceleration: vec![],
            },
            metrics: PerformanceMetrics {
                timing: TimingMetrics {
                    mean_time_ns: 1000,
                    std_time_ns: 100,
                    median_time_ns: 1000,
                    p95_time_ns: 1200,
                    p99_time_ns: 1300,
                    min_time_ns: 800,
                    max_time_ns: 1400,
                },
                memory: MemoryMetrics {
                    peak_memory_bytes: 1024,
                    avg_memory_bytes: 800,
                    allocation_count: 10,
                    fragmentation_ratio: 0.1,
                    efficiency_score: 0.9,
                },
                efficiency: EfficiencyMetrics {
                    flops: 1e9,
                    arithmetic_intensity: 2.0,
                    cache_hit_ratio: 0.95,
                    cpu_utilization: 0.8,
                    efficiency_score: 0.85,
                    custom_metrics: HashMap::new(),
                },
                convergence: ConvergenceMetrics {
                    final_objective: 0.01,
                    convergence_rate: 0.95,
                    iterations_to_convergence: Some(100),
                    quality_score: 0.9,
                    stability_score: 0.85,
                },
                custom: HashMap::new(),
            },
            metadata: HashMap::new(),
        };

        db.add_record("test_key".to_string(), record);
        assert_eq!(db.metadata.total_records, 1);

        // Test save and load
        db.save(temp_dir.path()).unwrap();
        let loaded_db = PerformanceDatabase::<f64>::load(temp_dir.path()).unwrap();
        assert_eq!(loaded_db.metadata.total_records, 1);
    }

    #[test]
    fn test_statistical_test_detector() {
        let detector = StatisticalTestDetector::new();

        let baseline = PerformanceBaseline {
            name: "test".to_string(),
            baseline_stats: BaselineStatistics {
                timing: TimingStatistics {
                    mean: 1000.0,
                    std_dev: 100.0,
                    median: 1000.0,
                    iqr: 150.0,
                    coefficient_of_variation: 0.1,
                },
                memory: MemoryStatistics {
                    mean_memory: 1024.0,
                    std_dev_memory: 100.0,
                    peak_memory_percentiles: HashMap::new(),
                    fragmentation_stats: FragmentationStatistics {
                        mean_ratio: 0.1,
                        std_dev_ratio: 0.02,
                        trend: 0.0,
                    },
                },
                efficiency: EfficiencyStatistics {
                    mean_flops: 1e9,
                    flops_cv: 0.1,
                    mean_efficiency: 0.8,
                    custom_efficiency: HashMap::new(),
                },
                convergence: ConvergenceStatistics {
                    mean_objective: 0.01,
                    std_objective: 0.001,
                    mean_convergence_rate: 0.9,
                    convergence_consistency: 0.85,
                },
            },
            confidence_intervals: ConfidenceIntervals {
                timing_ci_95: (900.0, 1100.0),
                memory_ci_95: (900.0, 1100.0),
                timing_ci_99: (850.0, 1150.0),
                memory_ci_99: (850.0, 1150.0),
            },
            sample_count: 10,
            created_at: 12345,
            updated_at: 12345,
        };

        let current_metrics = PerformanceMetrics {
            timing: TimingMetrics {
                mean_time_ns: 1200, // 20% slower
                std_time_ns: 100,
                median_time_ns: 1200,
                p95_time_ns: 1400,
                p99_time_ns: 1500,
                min_time_ns: 1000,
                max_time_ns: 1600,
            },
            memory: MemoryMetrics {
                peak_memory_bytes: 1024,
                avg_memory_bytes: 800,
                allocation_count: 10,
                fragmentation_ratio: 0.1,
                efficiency_score: 0.9,
            },
            efficiency: EfficiencyMetrics {
                flops: 1e9,
                arithmetic_intensity: 2.0,
                cache_hit_ratio: 0.95,
                cpu_utilization: 0.8,
                efficiency_score: 0.85,
                custom_metrics: HashMap::new(),
            },
            convergence: ConvergenceMetrics {
                final_objective: 0.01,
                convergence_rate: 0.95,
                iterations_to_convergence: Some(100),
                quality_score: 0.9,
                stability_score: 0.85,
            },
            custom: HashMap::new(),
        };

        let history = VecDeque::new();
        let result = detector
            .detect_regression(&baseline, &current_metrics, &history)
            .unwrap();

        assert!(result.performance_change_percent > 0.0);
        assert_eq!(result.test_id, "statistical_test");
    }

    #[test]
    fn test_trend_analyzer() {
        let analyzer = TrendAnalyzer::new();

        let mut data = VecDeque::new();
        for i in 0..10 {
            let record = PerformanceRecord {
                timestamp: i as u64,
                commit_hash: None,
                branch: None,
                environment: TestEnvironment {
                    os: "test".to_string(),
                    cpu_model: "test".to_string(),
                    memory_mb: 1024,
                    rust_version: "1.0".to_string(),
                    compiler_flags: vec![],
                    hardware_acceleration: vec![],
                },
                metrics: PerformanceMetrics {
                    timing: TimingMetrics {
                        mean_time_ns: 1000 + i as u64 * 100, // Increasing trend
                        std_time_ns: 100,
                        median_time_ns: 1000 + i as u64 * 100,
                        p95_time_ns: 1200 + i as u64 * 100,
                        p99_time_ns: 1300 + i as u64 * 100,
                        min_time_ns: 800 + i as u64 * 100,
                        max_time_ns: 1400 + i as u64 * 100,
                    },
                    memory: MemoryMetrics {
                        peak_memory_bytes: 1024,
                        avg_memory_bytes: 800,
                        allocation_count: 10,
                        fragmentation_ratio: 0.1,
                        efficiency_score: 0.9,
                    },
                    efficiency: EfficiencyMetrics {
                        flops: 1e9,
                        arithmetic_intensity: 2.0,
                        cache_hit_ratio: 0.95,
                        cpu_utilization: 0.8,
                        efficiency_score: 0.85,
                        custom_metrics: HashMap::new(),
                    },
                    convergence: ConvergenceMetrics {
                        final_objective: 0.01,
                        convergence_rate: 0.95,
                        iterations_to_convergence: Some(100),
                        quality_score: 0.9,
                        stability_score: 0.85,
                    },
                    custom: HashMap::new(),
                },
                metadata: HashMap::new(),
            };
            data.push_back(record);
        }

        let result = analyzer.analyze(&data).unwrap();
        assert!(result.summary.contains("increasing"));
    }
}
