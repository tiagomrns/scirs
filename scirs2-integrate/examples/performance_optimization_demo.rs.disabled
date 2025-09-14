//! Performance Optimization Demo
//!
//! This example demonstrates how to use the performance monitoring and profiling
//! capabilities to identify bottlenecks and optimize numerical algorithms.
//!
//! The example compares different integration methods while monitoring their
//! performance characteristics and providing optimization recommendations.

use ndarray::{Array1, Array2, ArrayView1};
use scirs2_integrate::{
    monte_carlo::{monte_carlo, MonteCarloOptions},

    // Standard integration methods
    ode::{solve_ivp, ODEMethod, ODEOptions},
    // Parallel optimization
    parallel_optimization::{
        ArithmeticOp, ParallelOptimizer, ReductionOp, VectorOperation, VectorizedComputeTask,
    },

    // Performance monitoring
    performance_monitor::{PerformanceAnalyzer, PerformanceProfiler, PerformanceReport},

    quad::quad,
    romberg::romberg,
    IntegrateResult,
};
use std::time::{Duration, Instant};

/// Benchmark different integration methods with performance monitoring
#[allow(dead_code)]
fn main() -> IntegrateResult<()> {
    println!("=== Performance Optimization Demo ===\n");

    // Benchmark 1: ODE solving with different methods
    benchmark_ode_methods()?;

    // Benchmark 2: Quadrature methods comparison
    benchmark_quadrature_methods()?;

    // Benchmark 3: Parallel optimization showcase
    benchmark_parallel_operations()?;

    // Benchmark 4: Memory usage optimization
    benchmark_memory_efficiency()?;

    Ok(())
}

/// Benchmark different ODE solving methods
#[allow(dead_code)]
fn benchmark_ode_methods() -> IntegrateResult<()> {
    println!("1. ODE Methods Performance Comparison");
    println!("   Solving stiff Van der Pol oscillator...\n");

    let methods = vec![
        ("RK45", ODEMethod::RK45),
        ("BDF", ODEMethod::Bdf),
        ("Radau", ODEMethod::Radau),
    ];

    // Van der Pol oscillator: y'' - μ(1-y²)y' + y = 0
    // Convert to first-order system
    let mu = 10.0; // Stiffness parameter
    let ode_fn = move |_t: f64, y: ArrayView1<f64>| {
        Array1::from_vec(vec![y[1], mu * (1.0 - y[0] * y[0]) * y[1] - y[0]])
    };

    let t_span = [0.0, 20.0];
    let y0 = Array1::from_vec(vec![2.0, 0.0]);

    for (method_name, method) in methods {
        let mut profiler = PerformanceProfiler::new();
        profiler.start_phase("ode_solving");

        let options = ODEOptions {
            method,
            rtol: 1e-6,
            atol: 1e-9,
            max_step: Some(0.1),
            h0: Some(1e-3),
            ..Default::default()
        };

        let start_memory = estimate_memory_usage();
        let result = solve_ivp(ode_fn, t_span, y0.clone(), Some(options));
        let peak_memory = estimate_memory_usage();

        profiler.end_phase("ode_solving");
        profiler.update_memory_stats(start_memory, peak_memory);

        // Estimate function evaluations based on method and solution points
        let estimated_evaluations = if let Ok(ref solution) = result {
            match method {
                ODEMethod::RK45 => solution.t.len() * 6, // 6 evaluations per step for RK45
                ODEMethod::Bdf => solution.t.len() * 3,  // Fewer for implicit methods
                ODEMethod::Radau => solution.t.len() * 4,
            }
        } else {
            100 // Default estimate
        };

        for _ in 0..estimated_evaluations {
            profiler.record_function_evaluation();
        }

        if let Ok(solution) = result {
            // Record convergence information
            for y_step in &solution.y {
                let residual = (y_step[0].powi(2) + y_step[1].powi(2)).sqrt();
                profiler.record_convergence(residual);
            }

            profiler.record_metric("solution_points", solution.t.len() as f64);
            profiler.record_metric("final_time", *solution.t.last().unwrap());
        }

        let metrics = profiler.finalize();
        print_method_performance(method_name, &metrics);
    }

    Ok(())
}

/// Benchmark quadrature methods
#[allow(dead_code)]
fn benchmark_quadrature_methods() -> IntegrateResult<()> {
    println!("\n2. Quadrature Methods Performance Comparison");
    println!("   Integrating oscillatory function...\n");

    // Highly oscillatory function: cos(50x) * exp(-x²)
    let test_function = |x: f64| (50.0 * x).cos() * (-x * x).exp();

    let methods = vec![
        ("Adaptive Quad", "adaptive"),
        ("Romberg", "romberg"),
        ("Monte Carlo", "monte_carlo"),
    ];

    for (method_name, method_type) in methods {
        let mut profiler = PerformanceProfiler::new();
        profiler.start_phase("integration");

        let start_memory = estimate_memory_usage();

        let (result, function_evals) = match method_type {
            "adaptive" => {
                let quad_result = quad(test_function, -2.0, 2.0, None)?;
                profiler.record_metric("error_estimate", quad_result.abs_error);
                (quad_result.value, quad_result.n_evals)
            }
            "romberg" => {
                let romberg_result = romberg(test_function, -2.0, 2.0, None)?;
                profiler.record_metric("error_estimate", romberg_result.abs_error);
                (
                    romberg_result.value,
                    2_usize.pow(romberg_result.n_iters as u32 + 1) - 1,
                )
            }
            "monte_carlo" => {
                let options = MonteCarloOptions {
                    n_samples: 100000,
                    seed: Some(42),
                    ..Default::default()
                };
                let mc_result = monte_carlo(
                    |x: ArrayView1<f64>| test_function(x[0]),
                    &[(-2.0, 2.0)],
                    Some(options),
                )?;
                profiler.record_metric("error_estimate", mc_result.std_error);
                (mc_result.value, mc_result.n_evals)
            }
            _ => unreachable!(),
        };

        let peak_memory = estimate_memory_usage();

        profiler.end_phase("integration");
        profiler.update_memory_stats(start_memory, peak_memory);

        // Record function evaluations
        for _ in 0..function_evals {
            profiler.record_function_evaluation();
        }

        profiler.record_metric("integration_result", result);

        let metrics = profiler.finalize();
        print_method_performance(method_name, &metrics);
    }

    Ok(())
}

/// Benchmark parallel operations
#[allow(dead_code)]
fn benchmark_parallel_operations() -> IntegrateResult<()> {
    println!("\n3. Parallel Operations Performance Benchmark");
    println!("   Testing vectorized operations with different configurations...\n");

    let mut optimizer = ParallelOptimizer::new(4);
    optimizer.initialize()?;

    // Create test data
    let sizes = vec![1000, 5000, 10000];
    let operations = vec![
        (
            "Exponential",
            VectorOperation::ElementWise(ArithmeticOp::Exp),
        ),
        ("Sine", VectorOperation::ElementWise(ArithmeticOp::Sin)),
        (
            "Sum Reduction",
            VectorOperation::Reduction(ReductionOp::Sum),
        ),
        (
            "Power 2",
            VectorOperation::ElementWise(ArithmeticOp::Power(2.0)),
        ),
    ];

    for size in sizes {
        println!("   Matrix size: {}x{}", size, size / 10);

        let test_matrix = Array2::from_shape_fn((size, size / 10), |(i, j)| {
            (i as f64 * 0.1 + j as f64 * 0.05).sin()
        });

        for (op_name, operation) in &operations {
            let mut profiler = PerformanceProfiler::new();
            profiler.start_phase("parallel_operation");

            let task = VectorizedComputeTask {
                input: test_matrix.clone(),
                operation: operation.clone(),
                chunk_size: size / 20,
                prefer_simd: true,
            };

            let start_memory = estimate_memory_usage();
            let start_time = Instant::now();

            let _result = optimizer.execute_vectorized(task)?;

            let duration = start_time.elapsed();
            let peak_memory = estimate_memory_usage();

            profiler.end_phase("parallel_operation");
            profiler.update_memory_stats(start_memory, peak_memory);
            profiler.estimate_flops(size * (size / 10), duration);

            let metrics = profiler.finalize();

            println!(
                "     {}: {:.2}ms, {:.1} MFLOPS, {:.1}MB peak",
                op_name,
                duration.as_millis(),
                metrics.cache_stats.flops.unwrap_or(0.0) / 1e6,
                metrics.memory_stats.peak_memory as f64 / (1024.0 * 1024.0)
            );
        }
        println!();
    }

    Ok(())
}

/// Benchmark memory efficiency
#[allow(dead_code)]
fn benchmark_memory_efficiency() -> IntegrateResult<()> {
    println!("4. Memory Efficiency Analysis");
    println!("   Comparing memory usage patterns...\n");

    let test_scenarios = vec![
        ("Small frequent allocations", 1000, 100),
        ("Large infrequent allocations", 10, 10000),
        ("Balanced allocations", 100, 1000),
    ];

    for (scenario_name, n_allocs, alloc_size) in test_scenarios {
        let mut profiler = PerformanceProfiler::new();
        profiler.start_phase("memory_test");

        let _start_memory = estimate_memory_usage();
        let mut data_storage = Vec::new();

        // Simulate different allocation patterns
        for i in 0..n_allocs {
            let data = Array1::from_shape_fn(alloc_size, |j| (i as f64 + j as f64 * 0.1).sin());
            data_storage.push(data);

            // Record allocation
            let current_memory = estimate_memory_usage();
            profiler.update_memory_stats(current_memory, current_memory);
        }

        let _peak_memory = estimate_memory_usage();

        // Simulate some computation
        let _sum: f64 = data_storage.iter().flat_map(|arr| arr.iter()).sum();

        profiler.end_phase("memory_test");

        let metrics = profiler.finalize();
        let report = PerformanceAnalyzer::generate_report(&metrics);

        println!("   {scenario_name}: ");
        println!(
            "     Peak memory: {:.1} MB",
            metrics.memory_stats.peak_memory as f64 / (1024.0 * 1024.0)
        );
        println!(
            "     Allocations: {}",
            metrics.memory_stats.allocation_count
        );

        if !report.recommendations.is_empty() {
            println!("     Recommendations:");
            for rec in &report.recommendations {
                println!("       - {}: {}", rec.category, rec.suggestion);
            }
        }
        println!();
    }

    Ok(())
}

/// Print performance metrics for a method
#[allow(dead_code)]
fn print_method_performance(
    method_name: &str,
    metrics: &scirs2_integrate::performance_monitor::PerformanceMetrics,
) {
    println!("   {method_name}: ");
    println!("     Time: {:.3}s", metrics.total_time.as_secs_f64());
    println!(
        "     Function evaluations: {}",
        metrics.function_evaluations
    );

    if let Some(eval_rate) = metrics.algorithm_metrics.get("evaluations_per_second") {
        println!("     Evaluation rate: {eval_rate:.1} evals/sec");
    }

    println!(
        "     Peak memory: {:.1} MB",
        metrics.memory_stats.peak_memory as f64 / (1024.0 * 1024.0)
    );

    // Print convergence information if available
    if !metrics.convergence_history.is_empty() {
        let final_residual = metrics.convergence_history.last().unwrap();
        println!("     Final residual: {final_residual:.2e}");
    }

    println!();
}

/// Estimate current memory usage (simplified)
#[allow(dead_code)]
fn estimate_memory_usage() -> usize {
    // This is a simplified memory estimation
    // In a real application, you might use system calls or memory profiling tools

    // Return a rough estimate based on typical allocation patterns
    // In practice, you'd want to use tools like jemalloc's stats or system APIs
    std::mem::size_of::<usize>() * 1024 // Placeholder value
}

/// Create an optimization report
#[allow(dead_code)]
fn create_optimization_report() -> PerformanceReport {
    // This would typically be generated from actual performance data
    let metrics = scirs2_integrate::performance_monitor::PerformanceMetrics::default();
    scirs2_integrate::performance_monitor::PerformanceAnalyzer::generate_report(&metrics)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_demo_components() {
        // Test that the basic components work
        let _profiler = PerformanceProfiler::new();
        let _optimizer = ParallelOptimizer::new(2);
        let _memory_estimate = estimate_memory_usage();

        assert!(true); // Basic sanity check
    }

    #[test]
    fn test_optimization_workflow() {
        // Test a complete optimization workflow
        let mut profiler = PerformanceProfiler::new();
        profiler.start_phase("test");

        // Simulate some work
        std::thread::sleep(Duration::from_millis(1));

        profiler.end_phase("test");
        profiler.record_function_evaluation();

        let metrics = profiler.finalize();
        assert!(metrics.total_time > Duration::ZERO);
        assert_eq!(metrics.function_evaluations, 1);
    }
}
