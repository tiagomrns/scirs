//! Enhanced Lomb-Scargle validation module
//!
//! This module provides comprehensive validation capabilities for Lomb-Scargle
//! implementations, organized into focused sub-modules for different aspects
//! of validation.
//!
//! # Modules
//!
//! - [`config`] - Configuration types and result structures
//! - [`performance`] - Performance benchmarking and memory analysis
//! - [`edge_cases`] - Edge case testing and robustness validation
//! - [`statistical`] - Statistical validation and significance testing
//! - [`frequency`] - Frequency domain analysis and resolution testing
//! - [`data_quality`] - Data quality assessment (irregular sampling, missing data, noise)
//! - [`platform`] - Platform-specific testing (SIMD, precision, cross-platform)
//! - [`reporting`] - Report generation for validation results
//!
//! # Usage
//!
//! ```rust,ignore
//! use scirs2_signal::lombscargle_enhanced_validation::{
//!     config::EnhancedValidationConfig,
//!     run_enhanced_validation,
//!     reporting::generate_validation_report,
//! };
//!
//! let config = EnhancedValidationConfig::default();
//! let result = run_enhanced_validation("enhanced", config)?;
//! let report = generate_validation_report(&result);
//! println!("{}", report);
//! ```

pub mod config;
pub mod data_quality;
pub mod edge_cases;
pub mod frequency;
pub mod performance;
pub mod platform;
pub mod reporting;
pub mod statistical;

// Re-export key types for convenience
pub use config::{
    CrossValidationResults, EdgeCaseRobustnessResults, EnhancedValidationConfig,
    EnhancedValidationResult, FrequencyDomainAnalysisResults, FrequencyResolutionResults,
    IrregularSamplingResults, MemoryAnalysisResults, MissingDataResults, NoiseRobustnessResults,
    PerformanceMetrics, StatisticalSignificanceResults,
};

// Re-export key functions from each module
pub use data_quality::{test_irregular_sampling, test_missing_data, test_noise_robustness};
pub use edge_cases::{
    test_constant_signal, test_duplicate_times, test_edge_case_robustness, test_empty_signal,
    test_invalid_values, test_non_monotonic_times, test_single_point,
    validate_edge_cases_comprehensive, validate_numerical_robustness_extreme,
};
pub use frequency::{
    assess_frequency_resolution, calculate_spurious_free_dynamic_range,
    calculate_theoretical_resolution, estimate_noise_floor, find_peaks,
    measure_spectral_leakage, test_alias_rejection, test_frequency_domain_analysis,
    test_frequency_resolution, test_phase_coherence, test_spectral_leakage,
};
pub use performance::{
    analyze_memory_complexity, analyze_memory_usage, analyze_timing_complexity,
    benchmark_performance, calculate_cache_efficiency, calculate_fragmentation_score,
    calculate_scaling_efficiency, calculate_std_dev, estimate_complexity,
};
pub use platform::{
    test_catastrophic_cancellation, test_condition_number_robustness,
    test_cross_platform_consistency, test_denormal_handling, test_f32_f64_consistency,
    test_precision_robustness, test_simd_scalar_consistency,
};
pub use reporting::generate_validation_report;
pub use statistical::{
    estimate_statistical_power, kolmogorov_smirnov_uniformity_test,
    perform_bootstrap_validation, perform_kfold_validation, perform_loo_validation,
    test_bootstrap_coverage, test_cross_validation, test_enhanced_bootstrap_coverage,
    test_enhanced_statistical_significance, test_frequency_stability,
    test_significance_calibration, test_temporal_consistency,
};

use crate::error::SignalResult;
use crate::lombscargle_validation::ValidationResult;

/// Main entry point for enhanced validation
///
/// This function orchestrates comprehensive validation of Lomb-Scargle implementations
/// using the provided configuration.
///
/// # Arguments
///
/// * `implementation` - The implementation to test ("standard" or "enhanced")
/// * `config` - Validation configuration specifying which tests to run
///
/// # Returns
///
/// An `EnhancedValidationResult` containing comprehensive test results
///
/// # Example
///
/// ```rust,ignore
/// use scirs2_signal::lombscargle_enhanced_validation::{
///     EnhancedValidationConfig, run_enhanced_validation
/// };
///
/// let config = EnhancedValidationConfig::default();
/// let result = run_enhanced_validation("enhanced", config)?;
/// println!("Overall score: {:.1}/100", result.overall_score);
/// ```
pub fn run_enhanced_validation(
    implementation: &str,
    config: EnhancedValidationConfig,
) -> SignalResult<EnhancedValidationResult> {
    let mut result = EnhancedValidationResult {
        basic_validation: ValidationResult::default(),
        performance: PerformanceMetrics {
            mean_time_ms: 0.0,
            std_time_ms: 0.0,
            throughput: 0.0,
            memory_efficiency: 0.0,
        },
        irregular_sampling: None,
        missing_data: None,
        noise_robustness: None,
        edge_cases: EdgeCaseRobustnessResults {
            empty_signal_handling: false,
            single_point_handling: false,
            constant_signal_handling: false,
            invalid_value_handling: false,
            duplicate_time_handling: false,
            non_monotonic_handling: false,
            overall_robustness: 0.0,
            extreme_frequency_handling: 0.0,
            numerical_edge_cases: 0.0,
        },
        statistical_significance: None,
        cross_platform: None,
        memory_analysis: None,
        frequency_resolution: None,
        simd_consistency: None,
        reference_comparison: None,
        extreme_parameters: None,
        multi_frequency: None,
        precision_robustness: None,
        frequency_domain_analysis: None,
        cross_validation: None,
        overall_score: 0.0,
        issues: Vec::new(),
        validation_warnings: Vec::new(),
    };

    let mut scores = Vec::new();

    // Performance benchmarking
    if config.benchmark {
        match benchmark_performance(implementation, config.benchmark_iterations) {
            Ok(perf) => {
                result.performance = perf;
                scores.push(85.0); // Base performance score
            }
            Err(e) => {
                result.issues.push(format!("Performance benchmark failed: {}", e));
                scores.push(0.0);
            }
        }
    }

    // Memory analysis
    if config.test_memory_usage {
        match analyze_memory_usage(implementation, 10) {
            Ok(mem) => {
                result.memory_analysis = Some(mem);
                scores.push(90.0);
            }
            Err(e) => {
                result.issues.push(format!("Memory analysis failed: {}", e));
                scores.push(0.0);
            }
        }
    }

    // Edge case testing
    match test_edge_case_robustness(implementation) {
        Ok(edge) => {
            let score = edge.overall_robustness * 100.0;
            result.edge_cases = edge;
            scores.push(score);
        }
        Err(e) => {
            result.issues.push(format!("Edge case testing failed: {}", e));
            scores.push(0.0);
        }
    }

    // Irregular sampling
    if config.test_irregular {
        match test_irregular_sampling(implementation, config.tolerance) {
            Ok(irregular) => {
                let score = if irregular.passed { 80.0 } else { 40.0 };
                result.irregular_sampling = Some(irregular);
                scores.push(score);
            }
            Err(e) => {
                result.issues.push(format!("Irregular sampling test failed: {}", e));
                scores.push(0.0);
            }
        }
    }

    // Missing data
    if config.test_missing {
        match test_missing_data(implementation, config.tolerance) {
            Ok(missing) => {
                let score = if missing.passed { 75.0 } else { 35.0 };
                result.missing_data = Some(missing);
                scores.push(score);
            }
            Err(e) => {
                result.issues.push(format!("Missing data test failed: {}", e));
                scores.push(0.0);
            }
        }
    }

    // Noise robustness
    if config.test_noisy {
        match test_noise_robustness(implementation, config.noise_snr_db) {
            Ok(noise) => {
                let score = 100.0 - (noise.false_positive_rate + noise.false_negative_rate) * 50.0;
                result.noise_robustness = Some(noise);
                scores.push(score.max(0.0));
            }
            Err(e) => {
                result.issues.push(format!("Noise robustness test failed: {}", e));
                scores.push(0.0);
            }
        }
    }

    // Statistical significance
    if config.test_statistical_significance {
        match test_enhanced_statistical_significance(implementation, config.tolerance) {
            Ok(stats) => {
                let score = (stats.fap_accuracy + stats.statistical_power + stats.significance_calibration) / 3.0 * 100.0;
                result.statistical_significance = Some(stats);
                scores.push(score);
            }
            Err(e) => {
                result.issues.push(format!("Statistical significance test failed: {}", e));
                scores.push(0.0);
            }
        }
    }

    // Cross-platform consistency
    if config.test_cross_platform {
        match test_cross_platform_consistency(implementation, config.tolerance) {
            Ok(cross) => {
                let score = cross.numerical_consistency * 100.0;
                result.cross_platform = Some(cross);
                scores.push(score);
            }
            Err(e) => {
                result.issues.push(format!("Cross-platform test failed: {}", e));
                scores.push(0.0);
            }
        }
    }

    // Frequency resolution
    if config.test_frequency_resolution {
        match test_frequency_resolution(implementation, config.tolerance) {
            Ok(freq_res) => {
                let score = (freq_res.sidelobe_suppression / 60.0 * 50.0 + freq_res.window_effectiveness * 50.0).min(100.0);
                result.frequency_resolution = Some(freq_res);
                scores.push(score);
            }
            Err(e) => {
                result.issues.push(format!("Frequency resolution test failed: {}", e));
                scores.push(0.0);
            }
        }
    }

    // Frequency domain analysis
    match test_frequency_domain_analysis(implementation, config.tolerance) {
        Ok(freq_domain) => {
            let score = (freq_domain.spectral_leakage + freq_domain.frequency_resolution_accuracy + freq_domain.phase_coherence) / 3.0 * 100.0;
            result.frequency_domain_analysis = Some(freq_domain);
            scores.push(score);
        }
        Err(e) => {
            result.issues.push(format!("Frequency domain analysis failed: {}", e));
            scores.push(0.0);
        }
    }

    // Cross-validation
    match test_cross_validation(implementation, config.tolerance) {
        Ok(cv) => {
            let score = cv.overall_cv_score * 100.0;
            result.cross_validation = Some(cv);
            scores.push(score);
        }
        Err(e) => {
            result.issues.push(format!("Cross-validation failed: {}", e));
            scores.push(0.0);
        }
    }

    // Precision robustness
    if config.test_precision_robustness {
        match test_precision_robustness(implementation, config.tolerance) {
            Ok(precision) => {
                let score = (precision.f32_f64_consistency + precision.scaling_stability +
                           precision.condition_number_analysis + precision.cancellation_robustness +
                           precision.denormal_handling) / 5.0 * 100.0;
                result.precision_robustness = Some(precision);
                scores.push(score);
            }
            Err(e) => {
                result.issues.push(format!("Precision robustness test failed: {}", e));
                scores.push(0.0);
            }
        }
    }

    // SIMD vs scalar consistency
    if config.test_simd_scalar_consistency {
        match test_simd_scalar_consistency(implementation, config.tolerance) {
            Ok(simd) => {
                let score = if simd.all_consistent { 95.0 } else { 50.0 };
                result.simd_consistency = Some(simd);
                scores.push(score);
            }
            Err(e) => {
                result.issues.push(format!("SIMD consistency test failed: {}", e));
                scores.push(0.0);
            }
        }
    }

    // Calculate overall score
    result.overall_score = if scores.is_empty() {
        0.0
    } else {
        scores.iter().sum::<f64>() / scores.len() as f64
    };

    // Add warnings based on score
    if result.overall_score < 70.0 {
        result.validation_warnings.push("Overall validation score is below recommended threshold".to_string());
    }

    if result.issues.len() > 3 {
        result.validation_warnings.push("Multiple critical issues detected".to_string());
    }

    Ok(result)
}