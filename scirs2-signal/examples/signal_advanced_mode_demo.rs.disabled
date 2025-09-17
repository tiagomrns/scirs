// Advanced Mode Demonstration
//
// This example shows how to use the Advanced mode coordinator for comprehensive
// validation and performance testing of signal processing implementations.

use scirs2_signal::advanced_validation_suite::{
    run_comprehensive_validation, run_quick_comprehensive_validation, ComprehensiveValidationConfig,
};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Advanced Mode Signal Processing Demo");
    println!("=========================================\n");

    // Example 1: Quick validation with default settings
    println!("1. Running quick Advanced validation...");
    match run_quick_comprehensive_validation() {
        Ok(results) => {
            println!("✅ Quick validation completed successfully!");
            println!(
                "   Overall pass rate: {:.1}%",
                results.summary.pass_rate * 100.0
            );
            println!(
                "   SIMD validation score: {:.1}",
                results.simd_results.overall_simd_score
            );
            println!(
                "   Parallel validation score: {:.1}",
                results.parallel_results.overall_parallel_score
            );
            println!(
                "   Memory efficiency score: {:.1}",
                results.memory_results.overall_memory_score
            );
            println!(
                "   Execution time: {:.1} ms",
                results.total_execution_time_ms
            );

            if !results.recommendations.is_empty() {
                println!("   Recommendations:");
                for issue in &results.recommendations {
                    println!("     - {}", issue);
                }
            }
        }
        Err(e) => {
            println!("❌ Quick validation failed: {}", e);
        }
    }

    println!();

    // Example 2: Custom configuration
    println!("2. Running validation with custom configuration...");
    let custom_config = ComprehensiveValidationConfig {
        tolerance: 1e-12,
        exhaustive: false,
        test_lengths: vec![64, 128, 256, 512],
        sampling_frequencies: vec![44100.0, 48000.0],
        random_seed: 42,
        max_test_duration: 30.0,
        benchmark: true,
        memory_profiling: true,
        cross_platform_testing: true,
        simd_validation: true,
        parallel_validation: true,
        monte_carlo_trials: 100,
        snr_levels: vec![10.0, 20.0, 30.0],
        test_complex: true,
        test_edge_cases: true,
    };

    match run_comprehensive_validation(&custom_config) {
        Ok(results) => {
            println!("✅ Custom validation completed!");
            println!("   Validation Results:");
            println!(
                "     - Multitaper accuracy: {:.1}",
                results.multitaper_results.dpss_accuracy_score
            );
            println!(
                "     - Lomb-Scargle accuracy: {:.1}",
                results.lombscargle_results.analytical_accuracy
            );
            println!(
                "     - Parametric AR accuracy: {:.1}",
                results
                    .parametric_results
                    .ar_validation
                    .order_estimation_accuracy
            );
            println!(
                "     - Wavelet 2D accuracy: {:.1}%",
                results.wavelet2d_results.reconstruction_accuracy
            );
            println!(
                "     - SIMD validation: {:.1}",
                results.simd_results.overall_simd_score
            );
            println!(
                "   Overall pass rate: {:.1}%",
                results.summary.pass_rate * 100.0
            );
        }
        Err(e) => {
            println!("❌ Custom validation failed: {}", e);
        }
    }

    println!();

    // Example 3: Performance comparison
    println!("3. Performance analysis over multiple runs...");

    for i in 1..=3 {
        println!("   Run {}...", i);
        if let Ok(_results) = run_quick_comprehensive_validation() {
            println!("   ✅ Run {} completed", i);
        } else {
            println!("   ❌ Run {} failed", i);
        }
    }

    println!("\n   Multiple runs completed successfully!");

    // Example 4: Memory-constrained validation
    println!("\n4. Testing memory-constrained configuration...");
    let memory_constrained_config = ComprehensiveValidationConfig {
        tolerance: 1e-8,
        exhaustive: false,
        test_lengths: vec![64, 128], // Smaller test sizes
        sampling_frequencies: vec![44100.0],
        random_seed: 42,
        max_test_duration: 10.0, // Shorter test duration
        benchmark: false,
        memory_profiling: true,
        cross_platform_testing: false,
        simd_validation: false,
        parallel_validation: false,
        monte_carlo_trials: 10, // Fewer trials
        snr_levels: vec![20.0], // Single SNR level
        test_complex: false,
        test_edge_cases: false,
    };

    match run_comprehensive_validation(&memory_constrained_config) {
        Ok(results) => {
            println!("✅ Memory-constrained validation completed!");
            println!(
                "   Memory efficiency: {:.1}",
                results.memory_results.overall_memory_score
            );
            println!(
                "   Execution time: {:.1} ms",
                results.total_execution_time_ms
            );
        }
        Err(e) => {
            println!("❌ Memory-constrained validation failed: {}", e);
        }
    }

    println!("\n🎉 Advanced mode demonstration completed!");
    println!("\nKey Benefits of Advanced Mode:");
    println!("  ⚡ Enhanced performance through SIMD and parallel processing");
    println!("  🔍 Comprehensive validation and testing");
    println!("  💾 Memory-efficient algorithms for large datasets");
    println!("  🎯 Numerical stability improvements");
    println!("  📊 Detailed performance metrics and reporting");

    Ok(())
}
