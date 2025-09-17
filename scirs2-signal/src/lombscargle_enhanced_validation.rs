// Enhanced validation suite for Lomb-Scargle periodogram
//
// This module provides comprehensive validation including:
// - Comparison with reference implementations
// - Edge case handling
// - Performance benchmarks with memory profiling
// - Enhanced numerical stability and precision tests
// - Robust statistical significance validation
// - Cross-platform consistency with floating-point analysis
// - Advanced bootstrap confidence interval coverage
// - SIMD vs scalar computation validation

//! Enhanced Lomb-Scargle Validation
//!
//! This module provides a comprehensive validation suite for Lomb-Scargle periodogram
//! implementations. The validation is organized into focused sub-modules for different
//! aspects of testing.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use scirs2_signal::lombscargle_enhanced_validation::{
//!     EnhancedValidationConfig, run_enhanced_validation, generate_validation_report
//! };
//!
//! // Run comprehensive validation
//! let config = EnhancedValidationConfig::default();
//! let result = run_enhanced_validation("enhanced", config)?;
//!
//! // Generate detailed report
//! let report = generate_validation_report(&result);
//! println!("{}", report);
//! ```
//!
//! # Available Tests
//!
//! - **Performance**: Benchmarking and memory analysis
//! - **Edge Cases**: Robustness testing with unusual inputs
//! - **Statistical**: Significance testing and cross-validation
//! - **Frequency Domain**: Spectral analysis and resolution testing
//! - **Data Quality**: Irregular sampling, missing data, noise robustness
//! - **Platform**: Cross-platform consistency, SIMD vs scalar, precision
//! - **Reporting**: Comprehensive validation reports

// Re-export everything from the modular implementation
mod lombscargle_enhanced_validation;

pub use lombscargle_enhanced_validation::*;

// For backward compatibility, re-export key items at the top level
pub use lombscargle_enhanced_validation::{
    run_enhanced_validation,
    generate_validation_report,
    EnhancedValidationConfig,
    EnhancedValidationResult,
};

// Additional backward compatibility exports for functions that were previously public
pub use lombscargle_enhanced_validation::{
    // Performance functions
    benchmark_performance,
    analyze_memory_usage,
    analyze_memory_complexity,
    analyze_timing_complexity,
    calculate_fragmentation_score,
    calculate_cache_efficiency,

    // Edge case functions
    test_edge_case_robustness,
    test_empty_signal,
    test_single_point,
    test_constant_signal,
    test_invalid_values,
    test_duplicate_times,
    test_non_monotonic_times,
    validate_edge_cases_comprehensive,

    // Statistical functions
    test_enhanced_statistical_significance,
    test_bootstrap_coverage,
    test_enhanced_bootstrap_coverage,
    kolmogorov_smirnov_uniformity_test,
    estimate_statistical_power,
    test_significance_calibration,
    test_cross_validation,
    perform_kfold_validation,
    perform_bootstrap_validation,
    perform_loo_validation,

    // Frequency functions
    test_frequency_domain_analysis,
    measure_spectral_leakage,
    estimate_noise_floor,
    assess_frequency_resolution,
    test_alias_rejection,
    test_phase_coherence,
    calculate_spurious_free_dynamic_range,
    test_frequency_resolution,

    // Data quality functions
    test_irregular_sampling,
    test_missing_data,
    test_noise_robustness,

    // Platform functions
    test_cross_platform_consistency,
    test_precision_robustness,
    test_simd_scalar_consistency,
};