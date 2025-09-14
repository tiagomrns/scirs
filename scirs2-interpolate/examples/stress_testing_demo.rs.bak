//! Demonstrates stress testing of interpolation methods with extreme inputs
//!
//! This example shows how to use the stress testing framework to validate
//! interpolation methods under extreme conditions.

use scirs2__interpolate::stress_testing::{extreme_data, stress_tests, StressTestConfig};

#[allow(dead_code)]
fn main() {
    println!("=== Interpolation Stress Testing Demo ===\n");

    // Configure stress tests
    let mut config = StressTestConfig::default();

    // You can customize which tests to run
    config.max_dataset_size = 100_000; // Test up to 100k points
    config.test_special_values = false; // Skip NaN/Inf tests for this demo

    println!("Running stress tests with configuration:");
    println!("  - Large values: {}", config.test_large_values);
    println!("  - Small values: {}", config.test_small_values);
    println!("  - Large ranges: {}", config.test_large_ranges);
    println!("  - Near collinear: {}", config.test_near_collinear);
    println!("  - Duplicates: {}", config.test_duplicates);
    println!(
        "  - Extreme extrapolation: {}",
        config.test_extreme_extrapolation
    );
    println!("  - Max dataset size: {}", config.max_dataset_size);

    // Run individual stress tests
    println!("\n1. Testing Linear Interpolation:");
    let linear_results = stress_tests::test_linear_interpolation(&config);
    linear_results.print_summary();

    println!("\n2. Testing Cubic Spline:");
    let spline_results = stress_tests::test_cubic_spline(&config);
    spline_results.print_summary();

    println!("\n3. Testing RBF Interpolation:");
    let rbf_results = stress_tests::test_rbf_interpolation(&config);
    rbf_results.print_summary();

    // Demonstrate extreme data generation
    println!("\n=== Extreme Data Examples ===");

    // Large values
    let (x_large, y_large) = extreme_data::large_values_1d(10);
    println!("\nLarge values (first 3 points):");
    for i in 0..3.min(x_large.len()) {
        println!(
            "  x[{}] = {:.3e}, y[{}] = {:.3e}",
            i, x_large[i], i, y_large[i]
        );
    }

    // Small values
    let (x_small, y_small) = extreme_data::small_values_1d(10);
    println!("\nSmall values (first 3 points):");
    for i in 0..3.min(x_small.len()) {
        println!(
            "  x[{}] = {:.3e}, y[{}] = {:.3e}",
            i, x_small[i], i, y_small[i]
        );
    }

    // Large range
    let (x_range, y_range) = extreme_data::large_range_1d(10);
    println!("\nLarge range values (showing extremes):");
    println!("  x[0] = {:.3e}, y[0] = {:.3e}", x_range[0], y_range[0]);
    println!(
        "  x[{}] = {:.3e}, y[{}] = {:.3e}",
        x_range.len() - 1,
        x_range[x_range.len() - 1],
        y_range.len() - 1,
        y_range[y_range.len() - 1]
    );

    // Run all tests with a minimal configuration for quick demo
    println!("\n=== Running All Tests (Quick Mode) ===");
    let quick_config = StressTestConfig {
        test_large_values: true,
        test_small_values: true,
        test_large_ranges: false,
        test_near_collinear: true,
        test_duplicates: true,
        test_special_values: false,
        test_extreme_extrapolation: false,
        test_dense_grids: false,
        test_ill_conditioned: true,
        max_dataset_size: 10_000,
    };

    let all_results = stress_tests::run_all_stress_tests(&quick_config);

    println!("\nOverall Summary:");
    for result in &all_results {
        println!(
            "  {}: {:.1}% success rate",
            result.method_name,
            result.success_rate() * 100.0
        );
    }

    // Demonstrate numerical stability checks
    println!("\n=== Numerical Stability Checks ===");
    use scirs2__interpolate::stress_testing::stability;

    let test_values = vec![
        1.0,
        1e100,
        1e-100,
        std::f64::MAX / 10.0,
        std::f64::MIN_POSITIVE * 10.0,
    ];

    for value in test_values {
        match stability::test_numerical_limits(value) {
            Ok(_) => println!("  {:.3e}: OK", value),
            Err(e) => println!("  {:.3e}: {}", value, e),
        }
    }

    println!("\n=== Condition Number Estimation ===");

    // Well-conditioned points
    let (good_points_) = extreme_data::extreme_2d(25, "random");
    let good_cond = stability::condition_number_estimate(&good_points.view());
    println!(
        "Well-conditioned points: condition number ≈ {:.2e}",
        good_cond
    );

    // Ill-conditioned points
    let (bad_points_) = extreme_data::extreme_2d(25, "ill_conditioned");
    let bad_cond = stability::condition_number_estimate(&bad_points.view());
    println!(
        "Ill-conditioned points: condition number ≈ {:.2e}",
        bad_cond
    );

    println!("\nStress testing complete!");
}
