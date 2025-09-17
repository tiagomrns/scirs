//! Report generation for Lomb-Scargle validation results
//!
//! This module provides comprehensive report generation capabilities
//! for enhanced Lomb-Scargle validation results.

use super::config::EnhancedValidationResult;

/// Generate comprehensive validation report
pub fn generate_validation_report(result: &EnhancedValidationResult) -> String {
    let mut report = String::new();

    report.push_str("Enhanced Lomb-Scargle Validation Report\n");
    report.push_str("=======================================\n\n");

    report.push_str(&format!(
        "Overall Score: {:.1}/100\n\n",
        result.overall_score
    ));

    // Basic validation
    report.push_str("Basic Validation:\n");
    report.push_str(&format!(
        "  Max Relative Error: {:.2e}\n",
        result.basic_validation.max_relative_error
    ));
    report.push_str(&format!(
        "  Peak Frequency Error: {:.2e}\n",
        result.basic_validation.peak_freq_error
    ));
    report.push_str(&format!(
        "  Stability Score: {:.2}\n",
        result.basic_validation.stability_score
    ));

    if !result.basic_validation.issues.is_empty() {
        report.push_str("  Issues:\n");
        for issue in &result.basic_validation.issues {
            report.push_str(&format!("    - {}\n", issue));
        }
    }
    report.push_str("\n");

    // Performance
    report.push_str("Performance Metrics:\n");
    report.push_str(&format!(
        "  Mean Time: {:.2} ms\n",
        result.performance.mean_time_ms
    ));
    report.push_str(&format!(
        "  Throughput: {:.0} samples/sec\n",
        result.performance.throughput
    ));
    report.push_str(&format!(
        "  Memory Efficiency: {:.2}\n\n",
        result.performance.memory_efficiency
    ));

    // Irregular sampling
    if let Some(ref irregular) = result.irregular_sampling {
        report.push_str("Irregular Sampling:\n");
        report.push_str(&format!(
            "  Peak Accuracy: {:.2}\n",
            irregular.peak_accuracy
        ));
        report.push_str(&format!(
            "  Leakage Factor: {:.2}\n",
            irregular.leakage_factor
        ));
        report.push_str(&format!("  Passed: {}\n\n", irregular.passed));
    }

    // Noise robustness
    if let Some(ref noise) = result.noise_robustness {
        report.push_str("Noise Robustness:\n");
        report.push_str(&format!(
            "  SNR Threshold: {:.1} dB\n",
            noise.snr_threshold_db
        ));
        report.push_str(&format!(
            "  False Positive Rate: {:.1}%\n",
            noise.false_positive_rate * 100.0
        ));
        report.push_str(&format!(
            "  False Negative Rate: {:.1}%\n\n",
            noise.false_negative_rate * 100.0
        ));
    }

    // Frequency domain analysis
    if let Some(ref freq_analysis) = result.frequency_domain_analysis {
        report.push_str("Frequency Domain Analysis:\n");
        report.push_str(&format!(
            "  Spectral Leakage: {:.3}\n",
            freq_analysis.spectral_leakage
        ));
        report.push_str(&format!(
            "  Dynamic Range: {:.1} dB\n",
            freq_analysis.dynamic_range_db
        ));
        report.push_str(&format!(
            "  Frequency Resolution Accuracy: {:.3}\n",
            freq_analysis.frequency_resolution_accuracy
        ));
        report.push_str(&format!(
            "  Alias Rejection: {:.1} dB\n",
            freq_analysis.alias_rejection_db
        ));
        report.push_str(&format!(
            "  Phase Coherence: {:.3}\n",
            freq_analysis.phase_coherence
        ));
        report.push_str(&format!("  SFDR: {:.1} dB\n\n", freq_analysis.sfdr_db));
    }

    // Cross-validation
    if let Some(ref cv) = result.cross_validation {
        report.push_str("Cross-Validation:\n");
        report.push_str(&format!("  K-Fold Score: {:.3}\n", cv.kfold_score));
        report.push_str(&format!("  Bootstrap Score: {:.3}\n", cv.bootstrap_score));
        report.push_str(&format!("  LOO Score: {:.3}\n", cv.loo_score));
        report.push_str(&format!(
            "  Temporal Consistency: {:.3}\n",
            cv.temporal_consistency
        ));
        report.push_str(&format!(
            "  Frequency Stability: {:.3}\n",
            cv.frequency_stability
        ));
        report.push_str(&format!(
            "  Overall CV Score: {:.3}\n\n",
            cv.overall_cv_score
        ));
    }

    // Edge case robustness
    report.push_str("Edge Case Robustness:\n");
    report.push_str(&format!(
        "  Empty Signal Handling: {}\n",
        result.edge_cases.empty_signal_handling
    ));
    report.push_str(&format!(
        "  Single Point Handling: {}\n",
        result.edge_cases.single_point_handling
    ));
    report.push_str(&format!(
        "  Constant Signal Handling: {}\n",
        result.edge_cases.constant_signal_handling
    ));
    report.push_str(&format!(
        "  Invalid Value Handling: {}\n",
        result.edge_cases.invalid_value_handling
    ));
    report.push_str(&format!(
        "  Duplicate Time Handling: {}\n",
        result.edge_cases.duplicate_time_handling
    ));
    report.push_str(&format!(
        "  Non-Monotonic Handling: {}\n",
        result.edge_cases.non_monotonic_handling
    ));
    report.push_str(&format!(
        "  Overall Robustness: {:.3}\n\n",
        result.edge_cases.overall_robustness
    ));

    // Memory analysis
    if let Some(ref mem) = result.memory_analysis {
        report.push_str("Memory Analysis:\n");
        report.push_str(&format!("  Peak Memory: {:.1} MB\n", mem.peak_memory_mb));
        report.push_str(&format!(
            "  Memory Efficiency: {:.3}\n",
            mem.memory_efficiency
        ));
        report.push_str(&format!("  Growth Rate: {:.3}\n", mem.memory_growth_rate));
        report.push_str(&format!(
            "  Cache Efficiency: {:.3}\n\n",
            mem.cache_efficiency
        ));
    }

    // Statistical significance
    if let Some(ref stats) = result.statistical_significance {
        report.push_str("Statistical Significance:\n");
        report.push_str(&format!(
            "  FAP Accuracy: {:.3}\n",
            stats.fap_accuracy
        ));
        report.push_str(&format!(
            "  Statistical Power: {:.3}\n",
            stats.statistical_power
        ));
        report.push_str(&format!(
            "  Significance Calibration: {:.3}\n",
            stats.significance_calibration
        ));
        report.push_str(&format!(
            "  Bootstrap Coverage: {:.3}\n",
            stats.bootstrap_coverage
        ));
        report.push_str(&format!(
            "  FAP Theoretical/Empirical Ratio: {:.3}\n",
            stats.fap_theoretical_empirical_ratio
        ));
        report.push_str(&format!(
            "  P-value Uniformity Score: {:.3}\n\n",
            stats.pvalue_uniformity_score
        ));
    }

    // Cross-platform results
    if let Some(ref cross_platform) = result.cross_platform {
        report.push_str("Cross-Platform Consistency:\n");
        report.push_str(&format!(
            "  Numerical Consistency: {:.3}\n",
            cross_platform.numerical_consistency
        ));
        report.push_str(&format!(
            "  SIMD Consistency: {:.3}\n",
            cross_platform.simd_consistency
        ));
        report.push_str(&format!(
            "  Precision Consistency: {:.3}\n",
            cross_platform.precision_consistency
        ));
        report.push_str(&format!(
            "  All Consistent: {}\n\n",
            cross_platform.all_consistent
        ));
    }

    // Precision robustness
    if let Some(ref precision) = result.precision_robustness {
        report.push_str("Precision Robustness:\n");
        report.push_str(&format!(
            "  F32/F64 Consistency: {:.3}\n",
            precision.f32_f64_consistency
        ));
        report.push_str(&format!(
            "  Scaling Stability: {:.3}\n",
            precision.scaling_stability
        ));
        report.push_str(&format!(
            "  Condition Number Analysis: {:.3}\n",
            precision.condition_number_analysis
        ));
        report.push_str(&format!(
            "  Cancellation Robustness: {:.3}\n",
            precision.cancellation_robustness
        ));
        report.push_str(&format!(
            "  Denormal Handling: {:.3}\n\n",
            precision.denormal_handling
        ));
    }

    // SIMD consistency
    if let Some(ref simd) = result.simd_consistency {
        report.push_str("SIMD vs Scalar Consistency:\n");
        report.push_str(&format!("  Max Deviation: {:.2e}\n", simd.max_deviation));
        report.push_str(&format!(
            "  Mean Absolute Deviation: {:.2e}\n",
            simd.mean_absolute_deviation
        ));
        report.push_str(&format!(
            "  Performance Ratio: {:.2}x\n",
            simd.performance_ratio
        ));
        report.push_str(&format!(
            "  SIMD Utilization: {:.3}\n",
            simd.simd_utilization
        ));
        report.push_str(&format!("  All Consistent: {}\n\n", simd.all_consistent));
    }

    // Frequency resolution
    if let Some(ref freq_res) = result.frequency_resolution {
        report.push_str("Frequency Resolution:\n");
        report.push_str(&format!(
            "  Min Separation: {:.3}\n",
            freq_res.min_separation
        ));
        report.push_str(&format!(
            "  Resolution Limit: {:.3}\n",
            freq_res.resolution_limit
        ));
        report.push_str(&format!(
            "  Sidelobe Suppression: {:.1} dB\n",
            freq_res.sidelobe_suppression
        ));
        report.push_str(&format!(
            "  Window Effectiveness: {:.3}\n\n",
            freq_res.window_effectiveness
        ));
    }

    // Issues and warnings
    if !result.issues.is_empty() {
        report.push_str("Critical Issues:\n");
        for issue in &result.issues {
            report.push_str(&format!("  - {}\n", issue));
        }
        report.push_str("\n");
    }

    if !result.validation_warnings.is_empty() {
        report.push_str("Warnings:\n");
        for warning in &result.validation_warnings {
            report.push_str(&format!("  - {}\n", warning));
        }
        report.push_str("\n");
    }

    report.push_str("=======================================\n");
    report.push_str("Report generated by SciRS2 Enhanced Validation Suite\n");

    report
}