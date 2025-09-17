// Type definitions for Lomb-Scargle validation

/// Validation result structure
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Maximum relative error compared to reference
    pub max_relative_error: f64,
    /// Mean relative error
    pub mean_relative_error: f64,
    /// Numerical stability score (0-1, higher is better)
    pub stability_score: f64,
    /// Peak frequency accuracy
    pub peak_freq_error: f64,
    /// Issues found during validation
    pub issues: Vec<String>,
}

/// Single test result structure
#[derive(Debug, Clone)]
pub struct SingleTestResult {
    /// Relative errors from this test
    pub errors: Vec<f64>,
    /// Peak frequency error
    pub peak_error: f64,
    /// Peak frequency errors (for multiple peaks)
    pub peak_errors: Vec<f64>,
    /// Issues found in this test
    pub issues: Vec<String>,
}

/// Statistical validation result
#[derive(Debug, Clone)]
pub struct StatisticalValidationResult {
    /// Chi-squared test p-value for white noise
    pub white_noise_pvalue: f64,
    /// False alarm rate validation
    pub false_alarm_rate_error: f64,
    /// Bootstrap confidence interval coverage
    pub bootstrap_coverage: f64,
    /// Statistical issues found
    pub statistical_issues: Vec<String>,
}

/// Performance validation result
#[derive(Debug, Clone)]
pub struct PerformanceValidationResult {
    /// Standard implementation time (ms)
    pub standard_time_ms: f64,
    /// Enhanced implementation time (ms)
    pub enhanced_time_ms: f64,
    /// Memory usage (approximate MB)
    pub memory_usage_mb: f64,
    /// Speedup factor
    pub speedup_factor: f64,
    /// Performance issues found
    pub performance_issues: Vec<String>,
}

/// Comprehensive validation result
#[derive(Debug, Clone)]
pub struct ComprehensiveValidationResult {
    /// Basic analytical validation
    pub analytical: ValidationResult,
    /// Statistical properties validation
    pub statistical: StatisticalValidationResult,
    /// Performance validation
    pub performance: PerformanceValidationResult,
    /// Overall validation score (0-100)
    pub overall_score: f64,
    /// All issues combined
    pub all_issues: Vec<String>,
}

/// Robustness validation result
#[derive(Debug, Clone)]
pub struct RobustnessValidationResult {
    /// Overall robustness score
    pub robustness_score: f64,
    /// Memory stress test results
    pub memory_stress_score: f64,
    /// Precision limits test score
    pub precision_limits_score: f64,
    /// Boundary conditions score
    pub boundary_conditions_score: f64,
    /// Issues found
    pub issues: Vec<String>,
}

/// Real-world scenario validation result
#[derive(Debug, Clone)]
pub struct RealWorldValidationResult {
    /// Overall score
    pub score: f64,
    /// Astronomical data test score
    pub astronomical_score: f64,
    /// Physiological data test score
    pub physiological_score: f64,
    /// Environmental data test score
    pub environmental_score: f64,
    /// Issues found
    pub issues: Vec<String>,
}

/// Advanced statistical validation result
#[derive(Debug, Clone)]
pub struct AdvancedStatisticalResult {
    /// Overall score
    pub score: f64,
    /// Non-parametric tests score
    pub nonparametric_score: f64,
    /// Bayesian validation score
    pub bayesian_score: f64,
    /// Information theory metrics score
    pub information_theory_score: f64,
    /// Issues found
    pub issues: Vec<String>,
}

/// Enhanced validation configuration
#[derive(Debug, Clone)]
pub struct EnhancedValidationConfig {
    /// Tolerance for numerical comparisons
    pub tolerance: f64,
    /// Number of test iterations for statistical validation
    pub n_iterations: usize,
    /// Test against SciPy reference implementation
    pub test_scipy_reference: bool,
    /// Test with noisy signals
    pub test_with_noise: bool,
    /// Noise SNR in dB
    pub noise_snr_db: f64,
    /// Test extreme parameter values
    pub test_extreme_params: bool,
    /// Test memory efficiency
    pub test_memory_efficiency: bool,
    /// Test SIMD vs scalar consistency
    pub test_simd_consistency: bool,
}

impl Default for EnhancedValidationConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-10,
            n_iterations: 100,
            test_scipy_reference: true,
            test_with_noise: true,
            noise_snr_db: 20.0,
            test_extreme_params: true,
            test_memory_efficiency: true,
            test_simd_consistency: true,
        }
    }
}

/// Enhanced validation result
#[derive(Debug, Clone)]
pub struct EnhancedValidationResult {
    /// Basic validation metrics
    pub base_validation: ValidationResult,
    /// SciPy comparison metrics
    pub scipy_comparison: SciPyComparisonResult,
    /// Noise robustness score (0-100)
    pub noise_robustness: f64,
    /// Memory efficiency score (0-100)
    pub memory_efficiency: f64,
    /// SIMD consistency score (0-100)
    pub simd_consistency: f64,
    /// Overall enhanced score (0-100)
    pub overall_score: f64,
    /// Detailed recommendations
    pub recommendations: Vec<String>,
}

/// SciPy comparison result
#[derive(Debug, Clone)]
pub struct SciPyComparisonResult {
    /// Maximum relative error vs SciPy reference
    pub max_relative_error: f64,
    /// Mean relative error vs SciPy reference
    pub mean_relative_error: f64,
    /// Correlation coefficient with SciPy
    pub correlation: f64,
    /// Peak detection accuracy
    pub peak_detection_accuracy: f64,
}