//! Production Stress Testing Demo
//!
//! This example demonstrates how to use the production stress testing
//! to validate readiness for production deployment.

use scirs2__interpolate::{
    run_production_stress_tests, run_quick_stress_tests, run_stress_tests_with_config,
    IssueSeverity, ProductionReadiness, StressTestConfig, TestStatus,
};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Production Stress Testing Demo ===\n");

    // 1. Quick stress tests for development
    println!("1. Running quick stress tests for development...");
    match run_quick_stress_tests::<f64>() {
        Ok(report) => {
            println!("{}", report);

            match report.production_readiness {
                ProductionReadiness::Ready => {
                    println!("✅ Quick tests show library is production ready!");
                }
                ProductionReadiness::NeedsWork => {
                    println!("⚠️  Quick tests identify areas for improvement");
                }
                ProductionReadiness::NotReady => {
                    println!("❌ Quick tests found critical issues");
                }
            }
            println!();
        }
        Err(e) => println!("Quick stress tests failed: {}\n", e),
    }

    // 2. Comprehensive production stress tests
    println!("2. Running comprehensive production stress tests...");
    match run_production_stress_tests::<f64>() {
        Ok(report) => {
            println!("{}", report);

            // Analyze results in detail
            analyze_stress_test_results(&report);
            println!();
        }
        Err(e) => println!("Comprehensive stress tests failed: {}\n", e),
    }

    // 3. Custom stress testing with different configurations
    println!("3. Running custom stress tests with different configurations...");

    let configs = vec![
        (
            "Conservative",
            StressTestConfig {
                max_data_size: 50_000,
                stress_iterations: 20,
                test_timeout: 120,
                memory_limit: Some(1024 * 1024 * 1024), // 1GB
                test_extreme_cases: false,
                max_performance_degradation: 5.0,
            },
        ),
        (
            "Aggressive",
            StressTestConfig {
                max_data_size: 2_000_000,
                stress_iterations: 200,
                test_timeout: 600,
                memory_limit: Some(16 * 1024 * 1024 * 1024), // 16GB
                test_extreme_cases: true,
                max_performance_degradation: 50.0,
            },
        ),
        (
            "Memory Focused",
            StressTestConfig {
                max_data_size: 500_000,
                stress_iterations: 500,
                test_timeout: 300,
                memory_limit: Some(512 * 1024 * 1024), // 512MB
                test_extreme_cases: true,
                max_performance_degradation: 10.0,
            },
        ),
    ];

    for (name, config) in configs {
        println!("  Running {} stress test configuration...", name);
        match run_stress_tests_with_config::<f64>(config) {
            Ok(report) => {
                println!(
                    "    {}: {:?} ({} passed, {} failed, {} warnings)",
                    name,
                    report.production_readiness,
                    report.passed,
                    report.failed,
                    report.warnings
                );

                if !report.critical_issues.is_empty() {
                    println!("      Critical issues: {}", report.critical_issues.len());
                }
            }
            Err(e) => println!("    {} stress test failed: {}", name, e),
        }
    }
    println!();

    // 4. Demonstrate specific stress test categories
    println!("4. Analyzing stress test categories...");
    match run_production_stress_tests::<f64>() {
        Ok(report) => {
            analyze_by_category(&report);
            println!();
        }
        Err(e) => println!("Category analysis failed: {}\n", e),
    }

    // 5. Production deployment recommendations
    println!("5. Production deployment recommendations:");
    match run_production_stress_tests::<f64>() {
        Ok(report) => {
            provide_deployment_guidance(&report);
        }
        Err(e) => println!("Failed to generate deployment recommendations: {}", e),
    }

    println!("\n=== Stress Testing Complete ===");
    println!(
        "Use these results to validate production readiness and identify areas for improvement."
    );

    Ok(())
}

/// Analyze stress test results in detail
#[allow(dead_code)]
fn analyze_stress_test_results(report: &scirs2, interpolate: StressTestReport) {
    println!("Detailed Analysis:");

    // Performance analysis
    let mut total_degradation = 0.0;
    let mut degradation_count = 0;

    for result in &_report.test_results {
        if result.performance.degradation_factor > 1.0
            && result.performance.degradation_factor < f64::INFINITY
        {
            total_degradation += result.performance.degradation_factor;
            degradation_count += 1;
        }
    }

    if degradation_count > 0 {
        let avg_degradation = total_degradation / degradation_count as f64;
        println!("  Average performance degradation: {:.2}x", avg_degradation);

        if avg_degradation > 10.0 {
            println!("  ⚠️  High performance degradation detected");
        } else if avg_degradation > 5.0 {
            println!("  ⚠️  Moderate performance degradation");
        } else {
            println!("  ✅ Performance degradation within acceptable limits");
        }
    }

    // Memory analysis
    let memory_leak_tests = _report
        .test_results
        .iter()
        .filter(|r| r.memory_usage.leak_detected)
        .count();

    if memory_leak_tests > 0 {
        println!(
            "  ⚠️  {} tests detected potential memory leaks",
            memory_leak_tests
        );
    } else {
        println!("  ✅ No memory leaks detected");
    }

    // Error analysis
    let panic_tests = _report
        .test_results
        .iter()
        .filter(|r| r.issues.iter().any(|i| i.description.contains("panic")))
        .count();

    if panic_tests > 0 {
        println!("  ❌ {} tests caused panics - CRITICAL", panic_tests);
    } else {
        println!("  ✅ No panics detected");
    }

    // Timeout analysis
    let timeout_tests = _report
        .test_results
        .iter()
        .filter(|r| r.status == TestStatus::TimedOut)
        .count();

    if timeout_tests > 0 {
        println!("  ⚠️  {} tests timed out", timeout_tests);
    } else {
        println!("  ✅ No timeouts detected");
    }
}

/// Analyze results by test category
#[allow(dead_code)]
fn analyze_by_category(report: &scirs2, interpolate: StressTestReport) {
    use scirs2__interpolate::StressTestCategory;
    use std::collections::HashMap;

    let mut category_stats: HashMap<String, (usize, usize, usize)> = HashMap::new();

    for result in &_report.test_results {
        let category_name = format!("{:?}", result.category);
        let stats = category_stats.entry(category_name).or_insert((0, 0, 0));

        match result.status {
            TestStatus::Passed => stats.0 += 1,
            TestStatus::Failed | TestStatus::Error | TestStatus::TimedOut => stats.1 += 1,
            TestStatus::PassedWithWarnings => stats.2 += 1,
        }
    }

    println!("Results by Category:");
    for (category, (passed, failed, warnings)) in category_stats {
        let total = passed + failed + warnings;
        if total > 0 {
            let status = if failed > 0 {
                "❌ FAILED"
            } else if warnings > passed / 2 {
                "⚠️  WARNINGS"
            } else {
                "✅ PASSED"
            };

            println!(
                "  {}: {} ({} passed, {} failed, {} warnings)",
                category, status, passed, failed, warnings
            );
        }
    }
}

/// Provide production deployment guidance
#[allow(dead_code)]
fn provide_deployment_guidance(report: &scirs2, interpolate: StressTestReport) {
    match report.production_readiness {
        ProductionReadiness::Ready => {
            println!("✅ READY FOR PRODUCTION DEPLOYMENT");
            println!(
                "  The library has passed stress testing and appears ready for production use."
            );
            println!("  Consider the following for production deployment:");
            println!("  - Implement monitoring for performance and memory usage");
            println!("  - Set up alerting for error conditions");
            println!("  - Plan for graceful degradation under extreme load");
            println!("  - Document known limitations and optimal usage patterns");
        }
        ProductionReadiness::NeedsWork => {
            println!("⚠️  NEEDS WORK BEFORE PRODUCTION");
            println!(
                "  The library has issues that should be addressed before production deployment:"
            );

            if !_report.critical_issues.is_empty() {
                println!("  Critical Issues to Address:");
                for (i, issue) in report.critical_issues.iter().enumerate() {
                    if i < 5 {
                        // Show top 5 critical issues
                        println!("    {}. {}", i + 1, issue.description);
                        if let Some(mitigation) = &issue.mitigation {
                            println!("       → {}", mitigation);
                        }
                    }
                }
                if report.critical_issues.len() > 5 {
                    println!(
                        "    ... and {} more critical issues",
                        report.critical_issues.len() - 5
                    );
                }
            }

            println!("  Recommended Actions:");
            for (i, recommendation) in report.recommendations.iter().enumerate() {
                if i < 3 {
                    // Show top 3 recommendations
                    println!("    {}. {}", i + 1, recommendation);
                }
            }
        }
        ProductionReadiness::NotReady => {
            println!("❌ NOT READY FOR PRODUCTION");
            println!("  CRITICAL: Do not deploy to production - serious issues detected");

            let critical_count = report.critical_issues.len();
            let failed_count = report.failed;

            println!("  Issues Summary:");
            println!("    - {} critical issues", critical_count);
            println!("    - {} failed tests", failed_count);

            println!("  IMMEDIATE ACTIONS REQUIRED:");

            let panics = _report
                .test_results
                .iter()
                .filter(|r| r.issues.iter().any(|i| i.description.contains("panic")))
                .count();

            if panics > 0 {
                println!(
                    "    1. Fix {} tests that cause panics - HIGHEST PRIORITY",
                    panics
                );
            }

            let blocking_issues = _report
                .critical_issues
                .iter()
                .filter(|i| i.production_impact == scirs2_interpolate::ProductionImpact::Blocking)
                .count();

            if blocking_issues > 0 {
                println!("    2. Resolve {} blocking issues", blocking_issues);
            }

            println!("    3. Re-run stress tests after fixes");
            println!("    4. Consider code review and additional testing");
        }
    }

    // Memory recommendations
    let memory_issues = _report
        .test_results
        .iter()
        .filter(|r| r.memory_usage.leak_detected || r.memory_usage.growth_rate > 1000.0)
        .count();

    if memory_issues > 0 {
        println!("  Memory Considerations:");
        println!("    - {} tests showed memory concerns", memory_issues);
        println!("    - Implement memory monitoring in production");
        println!("    - Consider memory limits and cleanup strategies");
    }

    // Performance recommendations
    let slow_tests = _report
        .test_results
        .iter()
        .filter(|r| r.performance.degradation_factor > 10.0)
        .count();

    if slow_tests > 0 {
        println!("  Performance Considerations:");
        println!(
            "    - {} tests showed significant performance degradation",
            slow_tests
        );
        println!("    - Consider data size limits for production use");
        println!("    - Implement timeout handling for long-running operations");
    }
}
