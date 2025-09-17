//! Configuration types and result structures for enhanced Lomb-Scargle validation
//!
//! This module contains all the configuration enums, structs, and result types
//! used throughout the enhanced validation system.

use crate::lombscargle_validation::ValidationResult;

/// Enhanced validation configuration
#[derive(Debug, Clone)]
pub struct EnhancedValidationConfig {
    /// Tolerance for numerical comparisons
    pub tolerance: f64,
    /// Strict tolerance for critical tests
    pub strict_tolerance: f64,
    /// Enable performance benchmarking
    pub benchmark: bool,
    /// Number of benchmark iterations
    pub benchmark_iterations: usize,
    /// Memory usage limit (MB) for large signal tests
    pub memory_limit_mb: usize,
    /// Test irregularly sampled data
    pub test_irregular: bool,
    /// Test with missing data
    pub test_missing: bool,
    /// Test with noise
    pub test_noisy: bool,
    /// Noise level (SNR in dB)
    pub noise_snr_db: f64,
    /// Compare with reference values
    pub compare_reference: bool,
    /// Test extreme parameter values
    pub test_extreme_parameters: bool,
    /// Test multi-frequency signals
    pub test_multi_frequency: bool,
    /// Test cross-platform consistency
    pub test_cross_platform: bool,
    /// Test frequency resolution limits
    pub test_frequency_resolution: bool,
    /// Test statistical significance
    pub test_statistical_significance: bool,
    /// Test memory usage patterns
    pub test_memory_usage: bool,
    /// Test floating-point precision robustness
    pub test_precision_robustness: bool,
    /// Test SIMD vs scalar consistency
    pub test_simd_scalar_consistency: bool,
    /// Test with very long signals
    pub test_very_long_signals: bool,
    /// Test edge frequency cases
    pub test_edge_frequencies: bool,
    /// Test normalization methods
    pub test_normalization_methods: bool,
    /// Test aliasing effects
    pub test_aliasing_effects: bool,
    /// Enable verbose diagnostics
    pub verbose_diagnostics: bool,
    /// Generate detailed reports
    pub generate_reports: bool,
}

impl Default for EnhancedValidationConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-10,
            strict_tolerance: 1e-12,
            benchmark: true,
            benchmark_iterations: 100,
            memory_limit_mb: 1024,
            test_irregular: true,
            test_missing: true,
            test_noisy: true,
            noise_snr_db: 20.0,
            compare_reference: true,
            test_extreme_parameters: true,
            test_multi_frequency: true,
            test_cross_platform: true,
            test_frequency_resolution: true,
            test_statistical_significance: true,
            test_memory_usage: true,
            test_precision_robustness: true,
            test_simd_scalar_consistency: true,
            test_very_long_signals: true,
            test_edge_frequencies: true,
            test_normalization_methods: true,
            test_aliasing_effects: true,
            verbose_diagnostics: false,
            generate_reports: false,
        }
    }
}

/// Enhanced validation result with comprehensive metrics
#[derive(Debug, Clone)]
pub struct EnhancedValidationResult {
    /// Basic validation results
    pub basic_validation: ValidationResult,
    /// Performance metrics
    pub performance: PerformanceMetrics,
    /// Irregular sampling validation results
    pub irregular_sampling: Option<IrregularSamplingResults>,
    /// Missing data handling results
    pub missing_data: Option<MissingDataResults>,
    /// Noise robustness results
    pub noise_robustness: Option<NoiseRobustnessResults>,
    /// Edge case validation results
    pub edge_cases: EdgeCaseRobustnessResults,
    /// Statistical significance results
    pub statistical_significance: Option<StatisticalSignificanceResults>,
    /// Cross-platform consistency results
    pub cross_platform: Option<CrossPlatformResults>,
    /// Memory usage analysis
    pub memory_analysis: Option<MemoryAnalysisResults>,
    /// Frequency resolution validation
    pub frequency_resolution: Option<FrequencyResolutionResults>,
    /// SIMD vs scalar consistency
    pub simd_consistency: Option<SimdScalarConsistencyResults>,
    /// Reference comparison results
    pub reference_comparison: Option<ReferenceComparisonResults>,
    /// Extreme parameter test results
    pub extreme_parameters: Option<ExtremeParameterResults>,
    /// Multi-frequency signal test results
    pub multi_frequency: Option<MultiFrequencyResults>,
    /// Precision robustness results
    pub precision_robustness: Option<PrecisionRobustnessResults>,
    /// Advanced frequency domain analysis
    pub frequency_domain_analysis: Option<FrequencyDomainAnalysisResults>,
    /// Cross-validation results
    pub cross_validation: Option<CrossValidationResults>,
    /// Overall validation score (0-100)
    pub overall_score: f64,
    /// Critical issues found
    pub issues: Vec<String>,
    /// Warnings generated
    pub validation_warnings: Vec<String>,
}

/// Advanced frequency domain analysis results
#[derive(Debug, Clone)]
pub struct FrequencyDomainAnalysisResults {
    /// Spectral leakage measurement
    pub spectral_leakage: f64,
    /// Dynamic range assessment
    pub dynamic_range_db: f64,
    /// Frequency resolution accuracy
    pub frequency_resolution_accuracy: f64,
    /// Alias rejection ratio
    pub alias_rejection_db: f64,
    /// Phase coherence (for complex signals)
    pub phase_coherence: f64,
    /// Spurious-free dynamic range
    pub sfdr_db: f64,
}

/// Cross-validation results
#[derive(Debug, Clone)]
pub struct CrossValidationResults {
    /// K-fold cross-validation score
    pub kfold_score: f64,
    /// Bootstrap validation score
    pub bootstrap_score: f64,
    /// Leave-one-out validation score
    pub loo_score: f64,
    /// Temporal consistency score
    pub temporal_consistency: f64,
    /// Frequency stability across folds
    pub frequency_stability: f64,
    /// Overall cross-validation score
    pub overall_cv_score: f64,
}

/// Edge case robustness results
#[derive(Debug, Clone)]
pub struct EdgeCaseRobustnessResults {
    /// Handles empty signals gracefully
    pub empty_signal_handling: bool,
    /// Handles single-point signals
    pub single_point_handling: bool,
    /// Handles constant signals
    pub constant_signal_handling: bool,
    /// Handles infinite/NaN values
    pub invalid_value_handling: bool,
    /// Handles duplicate time points
    pub duplicate_time_handling: bool,
    /// Handles non-monotonic time series
    pub non_monotonic_handling: bool,
    /// Overall robustness score
    pub overall_robustness: f64,
    /// Handles extreme frequency ranges
    pub extreme_frequency_handling: f64,
    /// Handles numerical edge cases
    pub numerical_edge_cases: f64,
}

/// Performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Mean computation time (ms)
    pub mean_time_ms: f64,
    /// Standard deviation of computation time
    pub std_time_ms: f64,
    /// Throughput (samples per second)
    pub throughput: f64,
    /// Memory efficiency score (0-1)
    pub memory_efficiency: f64,
}

/// Irregular sampling test results
#[derive(Debug, Clone)]
pub struct IrregularSamplingResults {
    /// Frequency resolution degradation
    pub resolution_factor: f64,
    /// Peak detection accuracy
    pub peak_accuracy: f64,
    /// Spectral leakage increase
    pub leakage_factor: f64,
    /// Passed all tests
    pub passed: bool,
}

/// Missing data test results
#[derive(Debug, Clone)]
pub struct MissingDataResults {
    /// Reconstruction accuracy with gaps
    pub gap_reconstruction_error: f64,
    /// Frequency estimation error
    pub frequency_error: f64,
    /// Amplitude estimation error
    pub amplitude_error: f64,
    /// Passed all tests
    pub passed: bool,
}

/// Noise robustness results
#[derive(Debug, Clone)]
pub struct NoiseRobustnessResults {
    /// SNR threshold for reliable detection
    pub snr_threshold_db: f64,
    /// False positive rate
    pub false_positive_rate: f64,
    /// False negative rate
    pub false_negative_rate: f64,
    /// Detection probability curve
    pub detection_curve: Vec<(f64, f64)>, // (SNR, detection_prob)
}

/// Reference comparison results
#[derive(Debug, Clone)]
pub struct ReferenceComparisonResults {
    /// Maximum deviation from reference
    pub max_deviation: f64,
    /// Mean absolute error
    pub mean_absolute_error: f64,
    /// Correlation with reference
    pub correlation: f64,
    /// Spectral distance
    pub spectral_distance: f64,
}

/// Extreme parameter test results
#[derive(Debug, Clone)]
pub struct ExtremeParameterResults {
    /// Handles very small time intervals
    pub small_intervals_ok: bool,
    /// Handles very large time intervals
    pub large_intervals_ok: bool,
    /// Handles high oversampling
    pub high_oversample_ok: bool,
    /// Handles extreme frequency ranges
    pub extreme_freqs_ok: bool,
    /// Overall robustness score
    pub robustness_score: f64,
}

/// Multi-frequency signal test results
#[derive(Debug, Clone)]
pub struct MultiFrequencyResults {
    /// Accuracy of primary frequency detection
    pub primary_freq_accuracy: f64,
    /// Accuracy of secondary frequencies
    pub secondary_freq_accuracy: f64,
    /// Spectral separation resolution
    pub separation_resolution: f64,
    /// Amplitude estimation error
    pub amplitude_error: f64,
    /// Phase estimation error
    pub phase_error: f64,
}

/// Cross-platform consistency results
#[derive(Debug, Clone)]
pub struct CrossPlatformResults {
    /// Numerical consistency across platforms
    pub numerical_consistency: f64,
    /// SIMD vs scalar consistency
    pub simd_consistency: f64,
    /// Floating point precision consistency
    pub precision_consistency: f64,
    /// All platforms consistent
    pub all_consistent: bool,
}

/// Frequency resolution test results
#[derive(Debug, Clone)]
pub struct FrequencyResolutionResults {
    /// Minimum resolvable frequency separation
    pub min_separation: f64,
    /// Resolution limit factor
    pub resolution_limit: f64,
    /// Sidelobe suppression
    pub sidelobe_suppression: f64,
    /// Window function effectiveness
    pub window_effectiveness: f64,
}

/// Statistical significance test results
#[derive(Debug, Clone)]
pub struct StatisticalSignificanceResults {
    /// False alarm probability accuracy
    pub fap_accuracy: f64,
    /// Statistical power estimation
    pub statistical_power: f64,
    /// Significance level calibration
    pub significance_calibration: f64,
    /// Bootstrap CI coverage
    pub bootstrap_coverage: f64,
    /// Theoretical vs empirical FAP comparison
    pub fap_theoretical_empirical_ratio: f64,
    /// P-value distribution uniformity (Kolmogorov-Smirnov test)
    pub pvalue_uniformity_score: f64,
}

/// Memory usage analysis results
#[derive(Debug, Clone)]
pub struct MemoryAnalysisResults {
    /// Peak memory usage (MB)
    pub peak_memory_mb: f64,
    /// Memory efficiency score (0-1)
    pub memory_efficiency: f64,
    /// Memory growth rate with signal size
    pub memory_growth_rate: f64,
    /// Fragmentation score
    pub fragmentation_score: f64,
    /// Cache efficiency estimation
    pub cache_efficiency: f64,
}

/// Precision robustness results
#[derive(Debug, Clone)]
pub struct PrecisionRobustnessResults {
    /// Single vs double precision consistency
    pub f32_f64_consistency: f64,
    /// Numerical stability under scaling
    pub scaling_stability: f64,
    /// Condition number analysis
    pub condition_number_analysis: f64,
    /// Catastrophic cancellation detection
    pub cancellation_robustness: f64,
    /// Denormal handling robustness
    pub denormal_handling: f64,
}

/// SIMD vs scalar consistency results
#[derive(Debug, Clone)]
pub struct SimdScalarConsistencyResults {
    /// Maximum deviation between SIMD and scalar
    pub max_deviation: f64,
    /// Mean absolute deviation
    pub mean_absolute_deviation: f64,
    /// Relative performance comparison
    pub performance_ratio: f64,
    /// SIMD utilization effectiveness
    pub simd_utilization: f64,
    /// All computations consistent
    pub all_consistent: bool,
}