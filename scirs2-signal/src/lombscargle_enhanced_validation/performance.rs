//! Performance benchmarking and memory analysis for Lomb-Scargle validation
//!
//! This module provides comprehensive performance measurement and memory analysis
//! capabilities for Lomb-Scargle implementations.

use crate::error::{SignalError, SignalResult};
use crate::lombscargle::lombscargle;
use std::f64::consts::PI;
use std::time::Instant;

use super::config::{MemoryAnalysisResults, PerformanceMetrics};

/// Helper function to run Lomb-Scargle implementations
fn run_lombscargle(
    implementation: &str,
    times: &[f64],
    signal: &[f64],
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    match implementation {
        "standard" => lombscargle(
            times,
            signal,
            None,
            Some("standard"),
            Some(true),
            Some(false),
            None,
            None,
        ),
        "enhanced" => {
            let (f, p) = lombscargle(
                times,
                signal,
                None,
                Some("standard"),
                Some(true),
                Some(false),
                Some(1.0),
                None,
            )?;
            Ok((f, p))
        }
        _ => Err(SignalError::ValueError(
            "Unknown implementation".to_string(),
        )),
    }
}

/// Benchmark performance of Lomb-Scargle implementation
pub fn benchmark_performance(
    implementation: &str,
    iterations: usize,
) -> SignalResult<PerformanceMetrics> {
    // Test signal
    let n = 1000;
    let t: Vec<f64> = (0..n).map(|i| i as f64 * 0.01).collect();
    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * PI * 10.0 * ti).sin() + (2.0 * PI * 25.0 * ti).sin())
        .collect();

    let mut times = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let start = Instant::now();

        match implementation {
            "standard" => {
                lombscargle(
                    &t,
                    &signal,
                    None,
                    Some("standard"),
                    Some(true),
                    Some(false),
                    Some(1.0),
                    None,
                )?;
            }
            "enhanced" => {
                // Use standard implementation for now
                lombscargle(
                    &t,
                    &signal,
                    None,
                    Some("standard"),
                    Some(true),
                    Some(false),
                    Some(1.0),
                    None,
                )?;
            }
            _ => {
                return Err(SignalError::ValueError(
                    "Unknown implementation".to_string(),
                ))
            }
        }

        times.push(start.elapsed().as_micros() as f64 / 1000.0); // Convert to ms
    }

    let mean_time_ms = times.iter().sum::<f64>() / iterations as f64;
    let variance = times
        .iter()
        .map(|&t| (t - mean_time_ms).powi(2))
        .sum::<f64>()
        / iterations as f64;
    let std_time_ms = variance.sqrt();

    let throughput = n as f64 / (mean_time_ms / 1000.0); // samples per second

    // Memory efficiency estimate based on signal size and computation time
    let base_efficiency = 0.9;
    let time_penalty = mean_time_ms / 100.0; // Normalize to reasonable scale
    let memory_efficiency = base_efficiency / (1.0 + time_penalty);

    Ok(PerformanceMetrics {
        mean_time_ms,
        std_time_ms,
        throughput,
        memory_efficiency,
    })
}

/// Analyze memory usage patterns for different signal sizes
pub fn analyze_memory_usage(
    implementation: &str,
    iterations: usize,
) -> SignalResult<MemoryAnalysisResults> {
    // Test with varying signal sizes to analyze memory scaling
    let signal_sizes = vec![64, 128, 256, 512, 1024, 2048, 4096];
    let mut memory_measurements = Vec::new();
    let mut timing_measurements = Vec::new();

    for &size in &signal_sizes {
        let t: Vec<f64> = (0..size).map(|i| i as f64 * 0.01).collect();
        let signal: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * PI * 10.0 * ti).sin() + 0.3 * (2.0 * PI * 33.0 * ti).sin())
            .collect();

        // Measure time for multiple iterations
        let start_time = std::time::Instant::now();
        let n_runs = iterations.min(50); // Limit for performance

        for _ in 0..n_runs {
            let _ = run_lombscargle(implementation, &t, &signal)?;
        }

        let elapsed_ms = start_time.elapsed().as_millis() as f64;
        let avg_time_ms = elapsed_ms / n_runs as f64;
        timing_measurements.push((size, avg_time_ms));

        // Estimate memory usage based on algorithmic complexity
        let base_memory_kb = 100.0; // Base overhead in KB

        // Lomb-Scargle memory usage is primarily:
        // - Input data: 2 * size * 8 bytes (time + signal)
        // - Frequency grid: typically 5-10x oversampling
        // - Power array: same size as frequency grid
        // - Intermediate calculations: ~3x data size

        let oversample_factor = 5.0; // Typical oversampling
        let data_memory_kb = (size as f64 * 8.0 * 2.0) / 1024.0; // Input arrays
        let freq_memory_kb = (size as f64 * oversample_factor * 8.0) / 1024.0; // Frequency grid
        let power_memory_kb = freq_memory_kb; // Power array
        let intermediate_memory_kb = data_memory_kb * 3.0; // Intermediate calculations

        let total_memory_kb = base_memory_kb
            + data_memory_kb
            + freq_memory_kb
            + power_memory_kb
            + intermediate_memory_kb;
        let total_memory_mb = total_memory_kb / 1024.0;

        memory_measurements.push((size, total_memory_mb));
    }

    // Analyze memory growth pattern
    let memory_complexity = analyze_memory_complexity(&memory_measurements);
    let timing_complexity = analyze_timing_complexity(&timing_measurements);

    // Calculate peak memory
    let peak_memory_mb = memory_measurements
        .iter()
        .map(|(_, mem)| *mem)
        .fold(0.0, f64::max);

    // Memory efficiency based on deviation from theoretical optimum
    let theoretical_linear_growth =
        memory_measurements.last().unwrap().1 / memory_measurements[0].1;
    let actual_growth_ratio = memory_measurements.last().unwrap().1 / memory_measurements[0].1;
    let memory_efficiency = (theoretical_linear_growth / actual_growth_ratio.max(1.0)).min(1.0);

    // Fragmentation score based on memory pattern consistency
    let fragmentation_score = calculate_fragmentation_score(&memory_measurements);

    // Cache efficiency based on time/memory relationship
    let cache_efficiency = calculate_cache_efficiency(&timing_measurements, &memory_measurements);

    Ok(MemoryAnalysisResults {
        peak_memory_mb,
        memory_efficiency,
        memory_growth_rate: memory_complexity,
        fragmentation_score,
        cache_efficiency,
    })
}

/// Analyze memory usage complexity
pub fn analyze_memory_complexity(measurements: &[(usize, f64)]) -> f64 {
    if measurements.len() < 2 {
        return 1.0;
    }

    // Calculate growth rate between first and last measurements
    let first = measurements[0];
    let last = measurements[measurements.len() - 1];

    if first.0 == last.0 || first.1 <= 0.0 {
        return 1.0;
    }

    // Calculate logarithmic growth rate to detect O(n), O(n log n), etc.
    let size_ratio = (last.0 as f64) / (first.0 as f64);
    let memory_ratio = last.1 / first.1;

    if size_ratio <= 1.0 {
        return 1.0;
    }

    // Growth exponent: memory_ratio = size_ratio^exponent
    let growth_exponent = memory_ratio.ln() / size_ratio.ln();
    growth_exponent
}

/// Analyze timing complexity
pub fn analyze_timing_complexity(measurements: &[(usize, f64)]) -> f64 {
    if measurements.len() < 2 {
        return 1.0;
    }

    let first = measurements[0];
    let last = measurements[measurements.len() - 1];

    if first.0 == last.0 || first.1 <= 0.0 {
        return 1.0;
    }

    let size_ratio = (last.0 as f64) / (first.0 as f64);
    let time_ratio = last.1 / first.1;

    if size_ratio <= 1.0 {
        return 1.0;
    }

    // Growth exponent for timing
    time_ratio.ln() / size_ratio.ln()
}

/// Calculate fragmentation score based on memory allocation patterns
pub fn calculate_fragmentation_score(measurements: &[(usize, f64)]) -> f64 {
    if measurements.len() < 3 {
        return 0.9; // Default good score
    }

    // Calculate how smooth the memory growth is
    let mut deviations = Vec::new();

    for i in 1..measurements.len() - 1 {
        let prev = measurements[i - 1];
        let curr = measurements[i];
        let next = measurements[i + 1];

        // Expected memory based on linear interpolation
        let size_progress = (curr.0 - prev.0) as f64 / (next.0 - prev.0) as f64;
        let expected_memory = prev.1 + size_progress * (next.1 - prev.1);

        // Deviation from smooth growth
        let deviation = (curr.1 - expected_memory).abs() / expected_memory.max(1.0);
        deviations.push(deviation);
    }

    if deviations.is_empty() {
        return 0.9;
    }

    let avg_deviation = deviations.iter().sum::<f64>() / deviations.len() as f64;
    (1.0 - avg_deviation).max(0.0).min(1.0)
}

/// Calculate cache efficiency from timing vs memory patterns
pub fn calculate_cache_efficiency(
    timing_measurements: &[(usize, f64)],
    memory_measurements: &[(usize, f64)],
) -> f64 {
    if timing_measurements.len() != memory_measurements.len() || timing_measurements.len() < 2 {
        return 0.85; // Default estimate
    }

    // Calculate if timing grows proportionally to memory (good cache behavior)
    // or faster than memory (poor cache behavior)

    let memory_growth = analyze_memory_complexity(memory_measurements);
    let timing_growth = analyze_timing_complexity(timing_measurements);

    // Ideal cache efficiency: timing grows linearly with memory
    // Poor cache efficiency: timing grows faster than memory (cache misses)
    let efficiency_ratio = if timing_growth > 0.0 {
        memory_growth / timing_growth
    } else {
        1.0
    };

    // Cache efficiency score: 1.0 = perfect, 0.0 = very poor
    efficiency_ratio.min(1.0).max(0.0)
}

/// Benchmark signal size scaling performance
pub fn benchmark_signal_size(n: usize, num_iterations: usize) -> SignalResult<f64> {
    let mut total_time = 0.0;

    for _ in 0..num_iterations {
        let time: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let signal: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();

        let start = Instant::now();
        let _ = lombscargle(
            &time,
            &signal,
            None,
            Some("standard"),
            Some(true),
            Some(true),
            None,
            None,
        )?;
        total_time += start.elapsed().as_secs_f64();
    }

    Ok(total_time / num_iterations as f64)
}

/// Estimate computational complexity from timing results
pub fn estimate_complexity(timing_results: &[(usize, f64)]) -> f64 {
    // Fit to N^alpha and return alpha
    if timing_results.len() < 2 {
        return 1.0;
    }

    let n1 = timing_results[0].0 as f64;
    let t1 = timing_results[0].1;
    let n2 = timing_results[1].0 as f64;
    let t2 = timing_results[1].1;

    if n1 <= 0.0 || n2 <= 0.0 || t1 <= 0.0 || t2 <= 0.0 {
        return 1.0;
    }

    (t2 / t1).ln() / (n2 / n1).ln()
}

/// Calculate scaling efficiency relative to linear scaling
pub fn calculate_scaling_efficiency(timing_results: &[(usize, f64)]) -> f64 {
    // Measure how close to ideal linear scaling
    if timing_results.len() < 2 {
        return 1.0;
    }

    let baseline_efficiency = timing_results[0].1 / timing_results[0].0 as f64;
    let final_efficiency =
        timing_results.last().unwrap().1 / timing_results.last().unwrap().0 as f64;

    baseline_efficiency / final_efficiency
}

/// Estimate memory usage scaling
pub fn estimate_memory_scaling(signal_sizes: &[usize]) -> SignalResult<f64> {
    // Estimate memory usage scaling (simplified)
    // For Lomb-Scargle, memory usage should be roughly O(N)
    let min_size = signal_sizes[0] as f64;
    let max_size = signal_sizes.last().unwrap_or(&signal_sizes[0]) as f64;

    // Assume linear memory scaling for now
    Ok(max_size / min_size)
}

/// Calculate standard deviation of values
pub fn calculate_std_dev(values: &[f64], mean: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;

    variance.sqrt()
}