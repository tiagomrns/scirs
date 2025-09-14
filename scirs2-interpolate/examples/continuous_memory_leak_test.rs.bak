//! Continuous memory leak detection test for production validation
//!
//! This example performs long-running interpolation operations while monitoring
//! memory usage to detect any potential memory leaks or unbounded growth.

use ndarray::{Array1, Array2};
use scirs2__interpolate::{
    advanced::rbf::RBFInterpolator,
    advanced::rbf::RBFKernel,
    error::InterpolateResult,
    interp1d::{linear_interpolate, Interp1d, InterpolationMethod},
    memory_monitor::{get_global_stats, start_monitoring, MemoryMonitor},
    spline::{CubicSpline, SplineBoundaryCondition},
};
use std::thread;
use std::time::{Duration, Instant};

/// Configuration for continuous memory testing
#[derive(Debug)]
struct MemoryTestConfig {
    /// Duration to run the test (in seconds)
    pub test_duration_seconds: u64,
    /// Interval between memory checks (in milliseconds)
    pub check_interval_ms: u64,
    /// Maximum allowed memory growth (in MB)
    pub max_memory_growth_mb: usize,
    /// Number of iterations per cycle
    pub iterations_per_cycle: usize,
    /// Data size for each test
    pub data_size: usize,
}

impl Default for MemoryTestConfig {
    fn default() -> Self {
        Self {
            test_duration_seconds: 300, // 5 minutes
            check_interval_ms: 1000,    // Check every second
            max_memory_growth_mb: 10,   // Allow up to 10MB growth
            iterations_per_cycle: 100,  // 100 operations per cycle
            data_size: 1000,            // 1000 data points
        }
    }
}

/// Results of continuous memory testing
#[derive(Debug)]
struct MemoryTestResults {
    /// Initial memory usage (bytes)
    pub initial_memory: usize,
    /// Final memory usage (bytes)
    pub final_memory: usize,
    /// Peak memory usage (bytes)
    pub peak_memory: usize,
    /// Total memory growth (bytes)
    pub memory_growth: isize,
    /// Memory leak detected
    pub leak_detected: bool,
    /// Test duration
    pub test_duration: Duration,
    /// Total operations performed
    pub total_operations: u64,
    /// Memory efficiency (operations per MB)
    pub operations_per_mb: f64,
}

/// Continuous memory leak detector for interpolation operations
pub struct ContinuousMemoryTester {
    config: MemoryTestConfig,
    monitor: MemoryMonitor,
    results: Option<MemoryTestResults>,
}

impl ContinuousMemoryTester {
    /// Create a new continuous memory tester
    pub fn new(config: MemoryTestConfig) -> Self {
        Self {
            config,
            monitor: MemoryMonitor::new("continuous_test"),
            results: None,
        }
    }

    /// Run the continuous memory leak test
    pub fn run_test(&mut self) -> InterpolateResult<MemoryTestResults> {
        println!("Starting continuous memory leak test...");
        println!(
            "Test duration: {} seconds",
            self.config.test_duration_seconds
        );
        println!("Check interval: {} ms", self.config.check_interval_ms);
        println!(
            "Max allowed growth: {} MB",
            self.config.max_memory_growth_mb
        );

        // Start monitoring
        start_monitoring();

        let start_time = Instant::now();
        let test_duration = Duration::from_secs(self.config.test_duration_seconds);
        let check_interval = Duration::from_millis(self.config.check_interval_ms);

        // Record initial memory state
        let initial_stats = get_global_stats();
        let initial_memory = initial_stats.total_allocated_bytes;
        let mut peak_memory = initial_memory;
        let mut total_operations = 0u64;

        println!(
            "Initial memory usage: {:.2} MB",
            initial_memory as f64 / 1_048_576.0
        );

        // Generate test data once to avoid repeated allocation
        let test_data = self.generate_test_data();

        // Main testing loop
        while start_time.elapsed() < test_duration {
            // Perform a cycle of interpolation operations
            for _ in 0..self.config.iterations_per_cycle {
                self.perform_interpolation_cycle(&test_data)?;
                total_operations += 1;
            }

            // Check memory usage
            let current_stats = get_global_stats();
            let current_memory = current_stats.total_allocated_bytes;
            peak_memory = peak_memory.max(current_memory);

            // Calculate memory growth
            let memory_growth = current_memory as isize - initial_memory as isize;
            let memory_growth_mb = memory_growth as f64 / 1_048_576.0;

            println!(
                "Time: {:6.1}s | Memory: {:8.2} MB | Growth: {:+7.2} MB | Ops: {}",
                start_time.elapsed().as_secs_f64(),
                current_memory as f64 / 1_048_576.0,
                memory_growth_mb,
                total_operations
            );

            // Check for excessive memory growth
            if memory_growth_mb > self.config.max_memory_growth_mb as f64 {
                println!("WARNING: Memory growth exceeded threshold!");
                break;
            }

            // Wait before next check
            thread::sleep(check_interval);
        }

        // Final memory check
        let final_stats = get_global_stats();
        let final_memory = final_stats.total_allocated_bytes;
        let memory_growth = final_memory as isize - initial_memory as isize;
        let leak_detected = memory_growth > (self.config.max_memory_growth_mb * 1_048_576) as isize;

        let results = MemoryTestResults {
            initial_memory,
            final_memory,
            peak_memory,
            memory_growth,
            leak_detected,
            test_duration: start_time.elapsed(),
            total_operations,
            operations_per_mb: total_operations as f64 / (peak_memory as f64 / 1_048_576.0),
        };

        self.results = Some(results.clone());
        self.print_final_report(&results);

        Ok(results)
    }

    /// Generate test data for interpolation
    fn generate_test_data(&self) -> TestData {
        let n = self.config.data_size;
        let x: Array1<f64> = Array1::fromshape_fn(n, |i| i as f64);
        let y: Array1<f64> = Array1::fromshape_fn(n, |i| {
            let x_val = i as f64;
            x_val.sin() + 0.1 * x_val.cos() * x_val
        });

        // Generate evaluation points
        let eval_points: Array1<f64> =
            Array1::fromshape_fn(50, |i| i as f64 * (n - 1) as f64 / 49.0 + 0.5);

        TestData { x, y, eval_points }
    }

    /// Perform one cycle of various interpolation operations
    fn perform_interpolation_cycle(&mut self, data: &TestData) -> InterpolateResult<()> {
        // Track memory before operations
        let mem_before = get_global_stats().total_allocated_bytes;

        // Test 1: Linear interpolation
        {
            let interp =
                Interp1d::new(&data.x.view(), &data.y.view(), InterpolationMethod::Linear)?;
            for &eval_point in data.eval_points.iter() {
                let _ = interp.evaluate(eval_point)?;
            }
        }

        // Test 2: Cubic spline interpolation
        {
            let spline = CubicSpline::new(
                &data.x.view(),
                &data.y.view(),
                SplineBoundaryCondition::Natural,
            )?;
            for &eval_point in data.eval_points.iter() {
                let _ = spline.evaluate(eval_point);
            }
        }

        // Test 3: RBF interpolation (smaller dataset for performance)
        if data.x.len() <= 200 {
            let rbf = RBFInterpolator::new(
                &data.x.slice(s![..200.min(data.x.len())]).view(),
                &data.y.slice(s![..200.min(data.y.len())]).view(),
                RBFKernel::Gaussian { epsilon: 1.0 },
            )?;
            for &eval_point in data.eval_points.iter().take(10) {
                let _ = rbf.evaluate(&Array1::from_vec(vec![eval_point]))?;
            }
        }

        // Track memory after operations
        let mem_after = get_global_stats().total_allocated_bytes;
        let cycle_growth = mem_after as isize - mem_before as isize;

        // Track in monitor
        if cycle_growth > 0 {
            self.monitor
                .track_allocation(cycle_growth as usize, "interpolation_cycle");
        } else if cycle_growth < 0 {
            self.monitor
                .track_deallocation((-cycle_growth) as usize, "interpolation_cycle");
        }

        Ok(())
    }

    /// Print final test report
    fn print_final_report(&self, results: &MemoryTestResults) {
        println!("\n" + "=".repeat(60).as_str());
        println!("CONTINUOUS MEMORY LEAK TEST RESULTS");
        println!("=".repeat(60));

        println!(
            "Test Duration: {:.1} seconds",
            results.test_duration.as_secs_f64()
        );
        println!("Total Operations: {}", results.total_operations);
        println!(
            "Operations per Second: {:.1}",
            results.total_operations as f64 / results.test_duration.as_secs_f64()
        );

        println!("\nMemory Usage:");
        println!(
            "  Initial Memory: {:.2} MB",
            results.initial_memory as f64 / 1_048_576.0
        );
        println!(
            "  Final Memory:   {:.2} MB",
            results.final_memory as f64 / 1_048_576.0
        );
        println!(
            "  Peak Memory:    {:.2} MB",
            results.peak_memory as f64 / 1_048_576.0
        );
        println!(
            "  Memory Growth:  {:+.2} MB",
            results.memory_growth as f64 / 1_048_576.0
        );

        println!("\nPerformance Metrics:");
        println!("  Operations per MB: {:.1}", results.operations_per_mb);
        println!(
            "  Memory Efficiency: {:.2}%",
            (results.operations_per_mb / 1000.0 * 100.0).min(100.0)
        );

        println!("\nLeak Detection:");
        if results.leak_detected {
            println!("  ❌ MEMORY LEAK DETECTED");
            println!(
                "  Memory growth exceeded threshold of {} MB",
                self.config.max_memory_growth_mb
            );
        } else {
            println!("  ✅ NO MEMORY LEAKS DETECTED");
            println!("  Memory usage remained within acceptable bounds");
        }

        println!("=".repeat(60));
    }

    /// Get the test results
    pub fn get_results(&self) -> Option<&MemoryTestResults> {
        self.results.as_ref()
    }
}

/// Test data structure
#[derive(Debug)]
struct TestData {
    pub x: Array1<f64>,
    pub y: Array1<f64>,
    pub eval_points: Array1<f64>,
}

#[allow(dead_code)]
fn main() -> InterpolateResult<()> {
    println!("SciRS2 Interpolation - Continuous Memory Leak Test");
    println!("This test runs various interpolation methods continuously to detect memory leaks.\n");

    // Configuration for different test scenarios
    let configs = vec![
        (
            "Quick Test",
            MemoryTestConfig {
                test_duration_seconds: 30,
                check_interval_ms: 1000,
                max_memory_growth_mb: 5,
                iterations_per_cycle: 50,
                data_size: 100,
            },
        ),
        (
            "Standard Test",
            MemoryTestConfig {
                test_duration_seconds: 120,
                check_interval_ms: 2000,
                max_memory_growth_mb: 10,
                iterations_per_cycle: 100,
                data_size: 500,
            },
        ),
        (
            "Stress Test",
            MemoryTestConfig {
                test_duration_seconds: 300,
                check_interval_ms: 5000,
                max_memory_growth_mb: 20,
                iterations_per_cycle: 200,
                data_size: 1000,
            },
        ),
    ];

    let mut all_passed = true;

    for (test_name, config) in configs {
        println!("\n🔬 Running {}", test_name);
        println!("-".repeat(40));

        let mut tester = ContinuousMemoryTester::new(config);
        match tester.run_test() {
            Ok(results) => {
                if results.leak_detected {
                    all_passed = false;
                    println!("❌ {} FAILED - Memory leak detected", test_name);
                } else {
                    println!("✅ {} PASSED - No memory leaks", test_name);
                }
            }
            Err(e) => {
                all_passed = false;
                println!("❌ {} FAILED - Error: {}", test_name, e);
            }
        }
    }

    println!("\n" + "=".repeat(60));
    if all_passed {
        println!("🎉 ALL TESTS PASSED - Memory management is production-ready!");
    } else {
        println!("⚠️  SOME TESTS FAILED - Memory leaks detected, investigation needed");
    }
    println!("=".repeat(60));

    Ok(())
}

// Add necessary imports for slicing
use ndarray::s;
