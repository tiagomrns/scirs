// Enhanced Modern WPT Validation Demonstration
//
// This example demonstrates the use of the enhanced modern validation
// framework for wavelet packet transforms, showing how to:
// - Configure comprehensive validation testing
// - Run various modern validation techniques
// - Analyze and interpret validation results
// - Generate detailed validation reports

use scirs2_signal::dwt::Wavelet;
use scirs2_signal::wpt_enhanced_modern_validation::{
    generate_enhanced_modern_validation_report, run_enhanced_modern_validation,
    EnhancedModernValidationConfig,
};
use std::f64::consts::PI;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Enhanced Modern WPT Validation Demonstration");
    println!("================================================\n");

    // Create test signals with different characteristics
    let signals = create_test_signals();

    // Configure validation with different levels of testing
    let configs = create_validation_configs();

    // Run validation tests
    for (signal_name, signal) in &signals {
        println!("🧪 Testing signal: {}", signal_name);
        println!("   Signal length: {}", signal.len());

        for (config_name, config) in &configs {
            println!("\n📋 Using configuration: {}", config_name);

            // Run enhanced modern validation
            match run_enhanced_modern_validation(
                signal,
                Wavelet::DB(4),
                4, // max_depth
                config,
            ) {
                Ok(results) => {
                    println!("✅ Validation completed successfully!");
                    print_validation_summary(&results);

                    // Generate and save detailed report
                    let report = generate_enhanced_modern_validation_report(&results);
                    println!("\n📄 Validation Report Preview:");
                    println!("{}", &report[..report.len().min(500)]);
                    if report.len() > 500 {
                        println!("... (truncated)");
                    }

                    // Check for critical issues
                    if !results.critical_findings.is_empty() {
                        println!("\n⚠️ Critical findings detected:");
                        for finding in &results.critical_findings {
                            println!("   - {}", finding);
                        }
                    }
                }
                Err(e) => {
                    println!("❌ Validation failed: {}", e);
                }
            }

            println!("\n{}", "=".repeat(60));
        }
    }

    // Demonstrate specific validation capabilities
    demonstrate_specific_validations()?;

    Ok(())
}

/// Create various test signals with different characteristics
#[allow(dead_code)]
fn create_test_signals() -> Vec<(String, Vec<f64>)> {
    let mut signals = Vec::new();

    // 1. Clean sinusoidal signal
    let clean_signal: Vec<f64> = (0..512)
        .map(|i| {
            let t = i as f64 / 512.0;
            (2.0 * PI * 5.0 * t).sin() + 0.5 * (2.0 * PI * 10.0 * t).sin()
        })
        .collect();
    signals.push(("Clean Sinusoidal".to_string(), clean_signal));

    // 2. Noisy signal
    let mut rng = rand::rng();
    let noisy_signal: Vec<f64> = (0..512)
        .map(|i| {
            let t = i as f64 / 512.0;
            let signal = (2.0 * PI * 3.0 * t).sin();
            let noise = (rng.random::<f64>() - 0.5) * 0.2;
            signal + noise
        })
        .collect();
    signals.push(("Noisy Signal".to_string(), noisy_signal));

    // 3. Chirp signal (frequency sweep)
    let chirp_signal: Vec<f64> = (0..512)
        .map(|i| {
            let t = i as f64 / 512.0;
            let freq = 1.0 + 10.0 * t; // Linear frequency sweep
            (2.0 * PI * freq * t).sin()
        })
        .collect();
    signals.push(("Chirp Signal".to_string(), chirp_signal));

    // 4. Step function (discontinuous)
    let step_signal: Vec<f64> = (0..512).map(|i| if i < 256 { 1.0 } else { -1.0 }).collect();
    signals.push(("Step Function".to_string(), step_signal));

    // 5. Impulse signal (sparse)
    let mut impulse_signal = vec![0.0; 512];
    impulse_signal[100] = 1.0;
    impulse_signal[300] = -0.5;
    signals.push(("Impulse Signal".to_string(), impulse_signal));

    signals
}

/// Create different validation configurations for testing
#[allow(dead_code)]
fn create_validation_configs() -> Vec<(String, EnhancedModernValidationConfig)> {
    let mut configs = Vec::new();

    // 1. Basic configuration (fast)
    let basic_config = EnhancedModernValidationConfig {
        test_gpu: false,
        test_streaming: true,
        enable_anomaly_detection: true,
        test_cross_framework: false,
        validate_precision: true,
        test_edge_cases: false,
        validate_resources: true,
        test_optimizations: false,
        tolerance: 1e-10,
        max_test_duration: 30.0,
        monte_carlo_trials: 50,
    };
    configs.push(("Basic (Fast)".to_string(), basic_config));

    // 2. Comprehensive configuration
    let comprehensive_config = EnhancedModernValidationConfig {
        test_gpu: false, // Keep false for compatibility
        test_streaming: true,
        enable_anomaly_detection: true,
        test_cross_framework: false, // Keep false for this demo
        validate_precision: true,
        test_edge_cases: true,
        validate_resources: true,
        test_optimizations: true,
        tolerance: 1e-12,
        max_test_duration: 120.0,
        monte_carlo_trials: 100,
    };
    configs.push(("Comprehensive".to_string(), comprehensive_config));

    // 3. High precision configuration
    let precision_config = EnhancedModernValidationConfig {
        test_gpu: false,
        test_streaming: false,
        enable_anomaly_detection: false,
        test_cross_framework: false,
        validate_precision: true,
        test_edge_cases: true,
        validate_resources: false,
        test_optimizations: false,
        tolerance: 1e-15,
        max_test_duration: 60.0,
        monte_carlo_trials: 200,
    };
    configs.push(("High Precision".to_string(), precision_config));

    configs
}

/// Print a summary of validation results
#[allow(dead_code)]
fn print_validation_summary(
    results: &scirs2_signal::wpt_enhanced_modern_validation::EnhancedModernValidationResult,
) {
    println!("   📊 Overall Score: {:.1}/100", results.overall_score);
    println!(
        "   🔧 Core Stability: {:.3}",
        results.core_validation.stability_score
    );
    println!(
        "   ⚡ Processing Latency: {:.2} ms",
        results
            .streaming_validation
            .realtime_metrics
            .processing_latency
    );
    println!(
        "   🎯 Anomaly Score: {:.3}",
        results.anomaly_detection.anomaly_score
    );
    println!(
        "   💾 CPU Efficiency: {:.1}%",
        results
            .resource_validation
            .resource_utilization
            .cpu_efficiency
            * 100.0
    );

    // Performance insights
    if results
        .streaming_validation
        .realtime_metrics
        .processing_latency
        < 10.0
    {
        println!("   ✅ Excellent real-time performance");
    } else if results
        .streaming_validation
        .realtime_metrics
        .processing_latency
        < 50.0
    {
        println!("   ⚠️ Acceptable real-time performance");
    } else {
        println!("   ❌ Poor real-time performance");
    }

    if results.anomaly_detection.anomaly_score < 0.1 {
        println!("   ✅ No significant anomalies detected");
    } else {
        println!("   ⚠️ Some anomalies detected - investigate further");
    }
}

/// Demonstrate specific validation capabilities in detail
#[allow(dead_code)]
fn demonstrate_specific_validations() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔬 Detailed Validation Capability Demonstration");
    println!("===============================================\n");

    // Create a specific test signal for detailed analysis
    let test_signal: Vec<f64> = (0..1024)
        .map(|i| {
            let t = i as f64 / 1024.0;
            // Multi-component signal with different frequencies
            (2.0 * PI * 2.0 * t).sin()
                + 0.5 * (2.0 * PI * 8.0 * t).sin()
                + 0.25 * (2.0 * PI * 16.0 * t).sin()
        })
        .collect();

    println!("🧪 Testing with multi-component sinusoidal signal");
    println!("   Components: 2 Hz, 8 Hz (0.5 amplitude), 16 Hz (0.25 amplitude)");
    println!("   Signal length: {}", test_signal.len());

    // Test different wavelet types
    let wavelets = vec![
        ("Haar", Wavelet::Haar),
        ("Daubechies-4", Wavelet::DB(4)),
        ("Daubechies-8", Wavelet::DB(8)),
    ];

    for (wavelet_name, wavelet) in wavelets {
        println!("\n🌊 Testing with {} wavelet:", wavelet_name);

        let config = EnhancedModernValidationConfig {
            test_streaming: true,
            enable_anomaly_detection: true,
            validate_precision: true,
            test_edge_cases: true,
            validate_resources: true,
            tolerance: 1e-12,
            max_test_duration: 60.0,
            monte_carlo_trials: 50,
            ..Default::default()
        };

        match run_enhanced_modern_validation(&test_signal, wavelet, 5, &config) {
            Ok(results) => {
                println!("   ✅ Validation Score: {:.1}/100", results.overall_score);

                // Analyze specific metrics
                analyze_streaming_performance(&results);
                analyze_precision_metrics(&results);
                analyze_resource_usage(&results);
            }
            Err(e) => {
                println!("   ❌ Validation failed: {}", e);
            }
        }
    }

    Ok(())
}

/// Analyze streaming performance metrics in detail
#[allow(dead_code)]
fn analyze_streaming_performance(
    results: &scirs2_signal::wpt_enhanced_modern_validation::EnhancedModernValidationResult,
) {
    let streaming = &results.streaming_validation;

    println!("   📡 Streaming Performance Analysis:");
    println!(
        "      - Average Latency: {:.2} ms",
        streaming.realtime_metrics.processing_latency
    );
    println!(
        "      - Jitter (std dev): {:.2} ms",
        streaming.realtime_metrics.jitter
    );
    println!(
        "      - P95 Latency: {:.2} ms",
        streaming.latency_analysis.p95_latency
    );
    println!(
        "      - P99 Latency: {:.2} ms",
        streaming.latency_analysis.p99_latency
    );
    println!(
        "      - Max Throughput: {:.1} samples/sec",
        streaming.throughput_analysis.max_throughput
    );

    // Performance classification
    let latency = streaming.realtime_metrics.processing_latency;
    let performance_class = if latency < 1.0 {
        "Excellent (Advanced-low latency)"
    } else if latency < 10.0 {
        "Very Good (Low latency)"
    } else if latency < 50.0 {
        "Good (Acceptable for most applications)"
    } else {
        "Poor (High latency)"
    };
    println!("      - Performance Class: {}", performance_class);
}

/// Analyze precision metrics in detail
#[allow(dead_code)]
fn analyze_precision_metrics(
    results: &scirs2_signal::wpt_enhanced_modern_validation::EnhancedModernValidationResult,
) {
    let precision = &results.precision_validation;

    println!("   🎯 Precision Analysis:");
    println!(
        "      - Error Growth Rate: {:.2e}",
        precision.accumulation_errors.error_growth_rate
    );
    println!(
        "      - Bounds Validation: {}",
        if precision.accumulation_errors.bounds_validation {
            "✅ Passed"
        } else {
            "❌ Failed"
        }
    );
    println!(
        "      - Stability Margin: {:.2e}",
        precision.precision_analysis.stability_margin
    );

    if precision.precision_analysis.cancellation_incidents > 0 {
        println!(
            "      - ⚠️ Catastrophic cancellation incidents: {}",
            precision.precision_analysis.cancellation_incidents
        );
    } else {
        println!("      - ✅ No catastrophic cancellation detected");
    }
}

/// Analyze resource usage metrics in detail
#[allow(dead_code)]
fn analyze_resource_usage(
    results: &scirs2_signal::wpt_enhanced_modern_validation::EnhancedModernValidationResult,
) {
    let resources = &results.resource_validation;

    println!("   💾 Resource Usage Analysis:");
    println!(
        "      - CPU Efficiency: {:.1}%",
        resources.resource_utilization.cpu_efficiency * 100.0
    );
    println!(
        "      - Memory Efficiency: {:.1}%",
        resources.resource_utilization.memory_efficiency * 100.0
    );
    println!(
        "      - I/O Efficiency: {:.1}%",
        resources.resource_utilization.io_efficiency * 100.0
    );
    println!(
        "      - Memory Leak Score: {:.3} (lower is better)",
        resources.memory_leaks.leak_score
    );

    // Resource efficiency classification
    let avg_efficiency = (resources.resource_utilization.cpu_efficiency
        + resources.resource_utilization.memory_efficiency)
        / 2.0;
    let efficiency_class = if avg_efficiency > 0.9 {
        "Excellent"
    } else if avg_efficiency > 0.8 {
        "Very Good"
    } else if avg_efficiency > 0.7 {
        "Good"
    } else {
        "Needs Improvement"
    };
    println!(
        "      - Overall Efficiency: {} ({:.1}%)",
        efficiency_class,
        avg_efficiency * 100.0
    );
}
