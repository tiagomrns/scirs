// Enhanced multitaper validation example
//
// This example demonstrates the comprehensive validation suite for multitaper
// spectral estimation, including SciPy reference comparison and Advanced enhancements.

use scirs2_signal::{
    generate_multitaper_validation_report, run_scipy_multitaper_validation,
    EnhancedTestSignalConfig, TestSignalType,
};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Enhanced Multitaper Spectral Estimation - Comprehensive Validation");
    println!("================================================================");

    // Run the comprehensive validation suite
    println!("\n🔬 Running comprehensive multitaper validation...");
    let validation_result = run_scipy_multitaper_validation()?;

    // Generate and display the validation report
    let report = generate_multitaper_validation_report(&validation_result);
    println!("\n{}", report);

    // Detailed analysis
    println!("\n📊 Detailed Analysis:");
    println!("---------------------");

    println!("Overall Score: {:.1}/100", validation_result.overall_score);

    if validation_result.overall_score >= 90.0 {
        println!("✅ Excellent - Implementation meets high standards");
    } else if validation_result.overall_score >= 75.0 {
        println!("⚠️  Good - Some improvements needed");
    } else {
        println!("❌ Poor - Significant improvements required");
    }

    // Performance analysis
    println!("\n⚡ Performance Analysis:");
    println!(
        "Speed improvement: {:.1}x",
        validation_result.performance_comparison.speed_ratio
    );
    println!(
        "Memory efficiency: {:.1}x",
        validation_result.performance_comparison.memory_ratio
    );
    println!(
        "SIMD acceleration: {:.1}x",
        validation_result.performance_comparison.simd_speedup
    );
    println!(
        "Parallel efficiency: {:.1}%",
        validation_result.performance_comparison.parallel_efficiency * 100.0
    );

    // Statistical validation
    println!("\n📈 Statistical Validation:");
    println!(
        "Cross-correlation: {:.3}",
        validation_result.statistical_metrics.cross_correlation
    );
    println!(
        "Spectral coherence: {:.3}",
        validation_result.statistical_metrics.spectral_coherence
    );
    println!(
        "KS test p-value: {:.3}",
        validation_result.statistical_metrics.ks_test_pvalue
    );

    // SIMD validation
    println!("\n🚀 SIMD Validation:");
    println!(
        "Correctness: {}",
        if validation_result.simd_validation.correctness_passed {
            "✅"
        } else {
            "❌"
        }
    );
    println!(
        "Platform compatibility: {}",
        if validation_result.simd_validation.platform_compatible {
            "✅"
        } else {
            "❌"
        }
    );
    println!(
        "Precision preserved: {}",
        if validation_result.simd_validation.precision_preserved {
            "✅"
        } else {
            "❌"
        }
    );

    // Critical issues
    if !validation_result.issues.is_empty() {
        println!("\n⚠️ Critical Issues Found:");
        for issue in &validation_result.issues {
            println!("  - {}", issue);
        }
    }

    // Recommendations
    if !validation_result.recommendations.is_empty() {
        println!("\n💡 Recommendations:");
        for recommendation in &validation_result.recommendations {
            println!("  - {}", recommendation);
        }
    }

    // Test-specific details
    println!("\n🔍 Test-Specific Results:");
    for (test_name, test_result) in &validation_result.test_results {
        let status = if test_result.passed { "✅" } else { "❌" };
        println!(
            "  {} {}: {:.4} (threshold: {:.4})",
            status, test_name, test_result.error_metric, test_result.threshold
        );

        // Show additional metrics if available
        if !test_result.additional_metrics.is_empty() {
            for (metric, value) in &test_result.additional_metrics {
                println!("    {} = {:.6}", metric, value);
            }
        }
    }

    println!("\n🎯 Validation complete! See report above for detailed analysis.");

    Ok(())
}
