// Demonstration of advanced SIMD operations and memory optimization
//
// This example shows how to use the new SIMD-optimized signal processing
// functions and memory-optimized algorithms for processing large signals.

use crate::error::SignalResult;
use scirs2_signal::{
    error::SignalResult,
    memory_optimized::{memory_optimized_fir_filter, memory_optimized_spectrogram, MemoryConfig},
    simd_advanced::{
        benchmark_simd_operations, simd_apply_window, simd_autocorrelation, simd_cross_correlation,
        simd_fir_filter, SimdConfig,
    },
    simd_memory_optimization::{
        benchmark_simd_memory_operations, simd_optimized_convolution, simd_optimized_fir_filter,
        SimdMemoryConfig,
    },
};
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;

#[allow(dead_code)]
fn main() -> SignalResult<()> {
    println!("SIMD and Memory Optimization Demo");
    println!("==================================");

    // Demo 1: SIMD-optimized FIR filtering
    demo_simd_fir_filter()?;

    // Demo 2: SIMD autocorrelation
    demo_simd_autocorrelation()?;

    // Demo 3: SIMD cross-correlation
    demo_simd_cross_correlation()?;

    // Demo 4: SIMD windowing
    demo_simd_windowing()?;

    // Demo 5: Performance benchmarking
    demo_simd_benchmarks()?;

    // Demo 6: Memory-optimized filtering for large signals
    demo_memory_optimized_filtering()?;

    // Demo 7: Memory-optimized spectrogram
    demo_memory_optimized_spectrogram()?;

    // Demo 8: Advanced Mode SIMD Memory Optimization
    demo_advanced_simd_memory()?;

    println!("\nDemo completed successfully!");
    Ok(())
}

#[allow(dead_code)]
fn demo_simd_fir_filter() -> SignalResult<()> {
    println!("\n1. SIMD-Optimized FIR Filtering");
    println!("-------------------------------");

    // Create test signal: sine wave with noise
    let n = 1024;
    let fs = 1000.0;
    let signal_freq = 50.0;

    let signal: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            (2.0 * PI * signal_freq * t).sin() + 0.1 * (i as f64 * 17.0).sin() // Add some noise
        })
        .collect();

    // Design a simple low-pass FIR filter (5-tap)
    let coeffs = vec![0.1, 0.2, 0.4, 0.2, 0.1]; // Normalized to sum to 1

    let mut output = vec![0.0; signal.len()];

    // Configure SIMD with auto-detection
    let simd_config = SimdConfig {
        force_scalar: false,
        simd_threshold: 32,
        align_memory: true,
        use_advanced: true,
    };

    // Apply SIMD FIR filter
    simd_fir_filter(&signal, &coeffs, &mut output, &simd_config)?;

    // Compare with scalar version
    let mut scalar_output = vec![0.0; signal.len()];
    let scalar_config = SimdConfig {
        force_scalar: true,
        ..simd_config
    };

    simd_fir_filter(&signal, &coeffs, &mut scalar_output, &scalar_config)?;

    // Verify results are equivalent
    let max_error = output
        .iter()
        .zip(scalar_output.iter())
        .map(|(simd, scalar)| (simd - scalar).abs())
        .fold(0.0, f64::max);

    println!("  Signal length: {}", n);
    println!("  Filter taps: {}", coeffs.len());
    println!("  Max SIMD vs Scalar error: {:.2e}", max_error);
    println!("  Input RMS: {:.4}", rms(&signal));
    println!("  Output RMS: {:.4}", rms(&output));

    if max_error < 1e-10 {
        println!("  ✓ SIMD and scalar results match!");
    } else {
        println!("  ⚠ SIMD and scalar results differ");
    }

    Ok(())
}

#[allow(dead_code)]
fn demo_simd_autocorrelation() -> SignalResult<()> {
    println!("\n2. SIMD-Optimized Autocorrelation");
    println!("---------------------------------");

    // Create test signal with known autocorrelation properties
    let n = 512;
    let period = 64;

    let signal: Vec<f64> = (0..n)
        .map(|i| {
            let phase = 2.0 * PI * (i % period) as f64 / period as f64;
            phase.sin() + 0.5 * (2.0 * phase).sin()
        })
        .collect();

    let max_lag = 128;
    let config = SimdConfig::default();

    // Compute autocorrelation using SIMD
    let autocorr = simd_autocorrelation(&signal, max_lag, &config)?;

    // Compute scalar version for comparison
    let scalar_config = SimdConfig {
        force_scalar: true,
        ..config
    };
    let scalar_autocorr = simd_autocorrelation(&signal, max_lag, &scalar_config)?;

    // Find peak at expected lag (period)
    let peak_idx = autocorr[1..]
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i_)| i + 1)
        .unwrap_or(0);

    println!("  Signal length: {}", n);
    println!("  Max lag computed: {}", max_lag);
    println!("  Expected peak at lag: {}", period);
    println!("  Actual peak at lag: {}", peak_idx);
    println!("  Zero-lag autocorr: {:.4}", autocorr[0]);
    println!("  Peak autocorr: {:.4}", autocorr[peak_idx]);

    // Verify SIMD vs scalar
    let max_diff = autocorr
        .iter()
        .zip(scalar_autocorr.iter())
        .map(|(simd, scalar)| (simd - scalar).abs())
        .fold(0.0, f64::max);

    println!("  Max SIMD vs Scalar diff: {:.2e}", max_diff);

    if (peak_idx as i32 - period as i32).abs() <= 2 {
        println!("  ✓ Peak detected at expected lag!");
    }

    Ok(())
}

#[allow(dead_code)]
fn demo_simd_cross_correlation() -> SignalResult<()> {
    println!("\n3. SIMD-Optimized Cross-Correlation");
    println!("-----------------------------------");

    // Create two related signals with a known delay
    let n = 256;
    let delay = 32;

    let signal1: Vec<f64> = (0..n)
        .map(|i| (i as f64 * 0.1).sin() + 0.5 * (i as f64 * 0.2).cos())
        .collect();

    // Signal2 is signal1 delayed by 'delay' samples
    let mut signal2 = vec![0.0; n];
    for i in delay..n {
        signal2[i] = signal1[i - delay];
    }
    // Add some noise
    for i in 0..n {
        signal2[i] += 0.05 * (i as f64 * 13.0).sin();
    }

    let config = SimdConfig::default();

    // Compute cross-correlation (full mode)
    let xcorr = simd_cross_correlation(&signal1, &signal2, "full", &config)?;

    // Find peak (indicates delay)
    let peak_idx = xcorr
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i_)| i)
        .unwrap_or(0);

    // In full mode, zero delay is at index (n2-1)
    let detected_delay = peak_idx as i32 - (signal2.len() as i32 - 1);

    println!("  Signal1 length: {}", signal1.len());
    println!("  Signal2 length: {}", signal2.len());
    println!("  Expected delay: {}", delay);
    println!("  Detected delay: {}", detected_delay);
    println!("  Cross-corr length: {}", xcorr.len());
    println!("  Peak value: {:.4}", xcorr[peak_idx]);

    if (detected_delay - delay as i32).abs() <= 2 {
        println!("  ✓ Delay detected correctly!");
    }

    Ok(())
}

#[allow(dead_code)]
fn demo_simd_windowing() -> SignalResult<()> {
    println!("\n4. SIMD-Optimized Windowing");
    println!("---------------------------");

    let n = 512;

    // Create test signal
    let signal: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * i as f64 / n as f64 * 10.0).sin())
        .collect();

    // Create Hann window
    let window: Vec<f64> = (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (n as f64 - 1.0)).cos()))
        .collect();

    let mut windowed = vec![0.0; n];
    let config = SimdConfig::default();

    // Apply window using SIMD
    simd_apply_window(&signal, &window, &mut windowed, &config)?;

    // Check results
    let original_energy: f64 = signal.iter().map(|x| x * x).sum();
    let windowed_energy: f64 = windowed.iter().map(|x| x * x).sum();
    let window_gain: f64 = window.iter().map(|x| x * x).sum::<f64>() / n as f64;

    println!("  Signal length: {}", n);
    println!("  Original energy: {:.2}", original_energy);
    println!("  Windowed energy: {:.2}", windowed_energy);
    println!("  Window gain: {:.4}", window_gain);
    println!("  Energy ratio: {:.4}", windowed_energy / original_energy);

    // Verify that windowing preserves the expected energy relationship
    let expected_ratio = window_gain;
    let actual_ratio = windowed_energy / original_energy;

    if (actual_ratio - expected_ratio).abs() < 0.01 {
        println!("  ✓ Windowing energy preserved correctly!");
    }

    Ok(())
}

#[allow(dead_code)]
fn demo_simd_benchmarks() -> SignalResult<()> {
    println!("\n5. SIMD Performance Benchmarking");
    println!("--------------------------------");

    let signal_lengths = vec![256, 1024, 4096, 16384];

    for &length in &signal_lengths {
        println!("  Benchmarking signal length: {}", length);
        benchmark_simd_operations(length)?;
        println!();
    }

    Ok(())
}

#[allow(dead_code)]
fn demo_memory_optimized_filtering() -> SignalResult<()> {
    println!("\n6. Memory-Optimized FIR Filtering");
    println!("---------------------------------");

    // Create a test signal file
    let input_file = "/tmp/large_signal_input.dat";
    let output_file = "/tmp/large_signal_output.dat";

    let n_samples = 100_000; // 100K samples
    let fs = 44100.0;

    println!("  Creating test signal file with {} samples...", n_samples);

    // Generate test signal and write to file
    {
        let mut file = File::create(input_file)?;
        for i in 0..n_samples {
            let t = i as f64 / fs;
            let sample = (2.0 * PI * 440.0 * t).sin() + 0.1 * (2.0 * PI * 1000.0 * t).sin();
            file.write_all(&sample.to_le_bytes())?;
        }
        file.flush()?;
    }

    // Design filter coefficients (31-tap low-pass filter)
    let coeffs: Vec<f64> = (0..31)
        .map(|i| {
            let n = i as f64 - 15.0; // Center around 0
            if n == 0.0 {
                0.3 // Cutoff frequency (normalized)
            } else {
                (PI * 0.3 * n).sin() / (PI * n) * 0.54 - 0.46 * (2.0 * PI * i as f64 / 30.0).cos()
                // Hamming window
            }
        })
        .collect();

    // Configure memory optimization
    let memory_config = MemoryConfig {
        max_memory_bytes: 10 * 1024 * 1024, // 10MB limit
        chunk_size: 4096,
        overlap_size: 64, // Larger than filter length
        use_mmap: false,  // Use standard I/O for this demo
        temp_dir: Some("/tmp".to_string()),
        compress_temp: false,
        cache_size: 1024 * 1024, // 1MB cache
    };

    println!("  Applying FIR filter with memory optimization...");

    // Apply memory-optimized filtering
    let start_time = std::time::Instant::now();
    let result = memory_optimized_fir_filter(input_file, output_file, &coeffs, &memory_config)?;
    let elapsed = start_time.elapsed();

    println!("  Processing completed in: {:?}", elapsed);
    println!(
        "  Peak memory usage: {:.2} MB",
        result.memory_stats.peak_memory as f64 / (1024.0 * 1024.0)
    );
    println!(
        "  Average memory usage: {:.2} MB",
        result.memory_stats.avg_memory as f64 / (1024.0 * 1024.0)
    );
    println!(
        "  Disk I/O operations: {}",
        result.memory_stats.disk_operations
    );
    println!("  Total time: {} ms", result.timing_stats.total_time_ms);
    println!("  I/O time: {} ms", result.timing_stats.io_time_ms);
    println!("  Compute time: {} ms", result.timing_stats.compute_time_ms);

    // Verify output file was created
    let output_metadata = std::fs::metadata(output_file)?;
    let expected_size = n_samples * std::mem::size_of::<f64>();

    println!(
        "  Output file size: {} bytes (expected: {})",
        output_metadata.len(),
        expected_size
    );

    if output_metadata.len() == expected_size as u64 {
        println!("  ✓ Output file has correct size!");
    }

    // Clean up
    let _ = std::fs::remove_file(input_file);
    let _ = std::fs::remove_file(output_file);

    Ok(())
}

#[allow(dead_code)]
fn demo_memory_optimized_spectrogram() -> SignalResult<()> {
    println!("\n7. Memory-Optimized Spectrogram");
    println!("-------------------------------");

    let input_file = "/tmp/test_signal_spec.dat";
    let output_file = "/tmp/test_spectrogram.dat";

    let n_samples = 50_000;
    let fs = 8000.0;

    println!("  Creating test signal for spectrogram...");

    // Generate test signal: chirp from 100 Hz to 2000 Hz
    {
        let mut file = File::create(input_file)?;
        for i in 0..n_samples {
            let t = i as f64 / fs;
            let f_start = 100.0;
            let f_end = 2000.0;
            let duration = n_samples as f64 / fs;
            let instantaneous_freq = f_start + (f_end - f_start) * t / duration;
            let phase = 2.0 * PI * instantaneous_freq * t;
            let sample = phase.sin();
            file.write_all(&sample.to_le_bytes())?;
        }
        file.flush()?;
    }

    // Spectrogram parameters
    let window_size = 512;
    let hop_size = 256;
    let n_frames = (n_samples - window_size) / hop_size + 1;
    let n_freqs = window_size / 2 + 1;

    let memory_config = MemoryConfig {
        max_memory_bytes: 5 * 1024 * 1024, // 5MB limit
        chunk_size: 1024,
        overlap_size: 0,
        use_mmap: false,
        temp_dir: Some("/tmp".to_string()),
        compress_temp: false,
        cache_size: 512 * 1024, // 512KB cache
    };

    println!("  Computing spectrogram with memory optimization...");
    println!("    Window size: {}", window_size);
    println!("    Hop size: {}", hop_size);
    println!("    Expected frames: {}", n_frames);
    println!("    Frequency bins: {}", n_freqs);

    let start_time = std::time::Instant::now();
    let result = memory_optimized_spectrogram(
        input_file,
        output_file,
        window_size,
        hop_size,
        &memory_config,
    )?;
    let elapsed = start_time.elapsed();

    println!("  Spectrogram computed in: {:?}", elapsed);
    println!(
        "  Peak memory usage: {:.2} MB",
        result.memory_stats.peak_memory as f64 / (1024.0 * 1024.0)
    );
    println!("  Total time: {} ms", result.timing_stats.total_time_ms);
    println!("  I/O time: {} ms", result.timing_stats.io_time_ms);
    println!("  Compute time: {} ms", result.timing_stats.compute_time_ms);

    // Verify output
    let output_metadata = std::fs::metadata(output_file)?;
    let expected_size = n_frames * n_freqs * std::mem::size_of::<f64>();

    println!(
        "  Spectrogram size: {} bytes (expected: {})",
        output_metadata.len(),
        expected_size
    );

    if output_metadata.len() == expected_size as u64 {
        println!("  ✓ Spectrogram has correct dimensions!");
    }

    // Clean up
    let _ = std::fs::remove_file(input_file);
    let _ = std::fs::remove_file(output_file);

    Ok(())
}

#[allow(dead_code)]
fn demo_advanced_simd_memory() -> SignalResult<()> {
    println!("\n8. Advanced Mode SIMD Memory Optimization");
    println!("===========================================");

    // Configure advanced SIMD memory optimization
    let config = SimdMemoryConfig {
        enable_simd: true,
        enable_parallel: true,
        cache_block_size: 16384,
        vector_size: 8,
        memory_alignment: 64,
        enable_prefetch: true,
    };

    println!("🚀 Advanced Mode Configuration:");
    println!("  - SIMD Vectorization: {}", config.enable_simd);
    println!("  - Parallel Processing: {}", config.enable_parallel);
    println!("  - Cache Block Size: {} bytes", config.cache_block_size);
    println!("  - Vector Size: {}", config.vector_size);
    println!("  - Memory Alignment: {} bytes", config.memory_alignment);

    // Generate large test signal
    let signal_size = 100_000;
    let signal: Vec<f64> = (0..signal_size)
        .map(|i| {
            let t = i as f64 / signal_size as f64;
            let freq1 = 50.0;
            let freq2 = 120.0;
            (2.0 * PI * freq1 * t).sin() + 0.5 * (2.0 * PI * freq2 * t).sin()
        })
        .collect();

    // Generate filter kernel
    let kernel_size = 512;
    let kernel: Vec<f64> = (0..kernel_size)
        .map(|i| {
            let x = (i as f64 - kernel_size as f64 / 2.0) / (kernel_size as f64 / 8.0);
            (-x * x / 2.0).exp() / (2.0 * PI).sqrt()
        })
        .collect();

    println!("\n📊 Test Data:");
    println!("  - Signal size: {} samples", signal_size);
    println!("  - Kernel size: {} samples", kernel_size);
    println!(
        "  - Expected output size: {} samples",
        signal_size + kernel_size - 1
    );

    // Test SIMD-optimized convolution
    println!("\n🔄 Testing SIMD-Optimized Convolution...");
    let signal_array = ndarray::Array1::from_vec(signal.clone());
    let kernel_array = ndarray::Array1::from_vec(kernel.clone());

    let conv_result =
        simd_optimized_convolution(&signal_array.view(), &kernel_array.view(), &config)?;

    println!("✅ Convolution Results:");
    println!(
        "  - Processing time: {:.2} ms",
        conv_result.processing_time_ms
    );
    println!(
        "  - Memory efficiency: {:.1}%",
        conv_result.memory_efficiency * 100.0
    );
    println!(
        "  - SIMD acceleration: {:.1}x",
        conv_result.simd_acceleration
    );
    println!(
        "  - Cache hit ratio: {:.1}%",
        conv_result.cache_hit_ratio * 100.0
    );
    println!("  - Output samples: {}", conv_result.data.len());

    // Test SIMD-optimized FIR filtering
    println!("\n🎛️  Testing SIMD-Optimized FIR Filter...");
    let fir_coeffs: Vec<f64> = (0..64)
        .map(|i| {
            let n = i as f64 - 31.5;
            if n == 0.0 {
                0.3 // Lowpass cutoff
            } else {
                let sinc = (PI * 0.3 * n).sin() / (PI * n);
                let window = 0.54 - 0.46 * (2.0 * PI * i as f64 / 63.0).cos();
                sinc * window
            }
        })
        .collect();

    let fir_coeffs_array = ndarray::Array1::from_vec(fir_coeffs);

    let fir_result =
        simd_optimized_fir_filter(&signal_array.view(), &fir_coeffs_array.view(), &config)?;

    println!("✅ FIR Filter Results:");
    println!(
        "  - Processing time: {:.2} ms",
        fir_result.processing_time_ms
    );
    println!(
        "  - Memory efficiency: {:.1}%",
        fir_result.memory_efficiency * 100.0
    );
    println!(
        "  - SIMD acceleration: {:.1}x",
        fir_result.simd_acceleration
    );
    println!(
        "  - Cache hit ratio: {:.1}%",
        fir_result.cache_hit_ratio * 100.0
    );

    // Performance benchmarking across different sizes
    println!("\n📈 Performance Benchmarking...");
    let test_sizes = vec![1000, 5000, 10000, 50000];

    let benchmark_results = benchmark_simd_memory_operations(&test_sizes, &config)?;

    println!("Signal Size | Processing Time | Speedup");
    println!("------------|-----------------|--------");
    for (size, time, speedup) in benchmark_results {
        println!("{:>11} | {:>15.2} ms | {:>6.1}x", size, time, speedup);
    }

    // Calculate total performance metrics
    let total_operations = signal_size * 2; // Convolution + FIR
    let total_time = conv_result.processing_time_ms + fir_result.processing_time_ms;
    let throughput = (total_operations as f64) / (total_time / 1000.0);

    println!("\n🎯 Advanced Mode Performance Summary:");
    println!("  - Total operations: {} samples", total_operations);
    println!("  - Total processing time: {:.2} ms", total_time);
    println!("  - Throughput: {:.0} samples/second", throughput);
    println!(
        "  - Average acceleration: {:.1}x",
        (conv_result.simd_acceleration + fir_result.simd_acceleration) / 2.0
    );
    println!(
        "  - Average memory efficiency: {:.1}%",
        (conv_result.memory_efficiency + fir_result.memory_efficiency) * 50.0
    );

    println!("\n⚡ Optimizations Applied:");
    println!("  ✓ SIMD vectorization for parallel computation");
    println!("  ✓ Cache-friendly memory access patterns");
    println!("  ✓ Memory alignment for optimal SIMD performance");
    println!("  ✓ Adaptive block sizing for cache efficiency");
    println!("  ✓ Prefetching for improved memory bandwidth");

    // Verify correctness with a simple test
    println!("\n🔍 Correctness Verification:");
    let small_signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let small_kernel = vec![1.0, 0.5];
    let small_signal_array = ndarray::Array1::from_vec(small_signal);
    let small_kernel_array = ndarray::Array1::from_vec(small_kernel);

    let small_result = simd_optimized_convolution(
        &small_signal_array.view(),
        &small_kernel_array.view(),
        &config,
    )?;

    // Expected: [1.0, 2.5, 4.0, 5.5, 7.0, 2.5]
    let expected = vec![1.0, 2.5, 4.0, 5.5, 7.0, 2.5];
    let mut all_correct = true;

    for (i, (&actual, &expected_val)) in small_result.data.iter().zip(expected.iter()).enumerate() {
        let error = (actual - expected_val).abs();
        if error > 1e-10 {
            println!(
                "  ❌ Mismatch at index {}: got {:.6}, expected {:.6}",
                i, actual, expected_val
            );
            all_correct = false;
        }
    }

    if all_correct {
        println!("  ✅ All values match expected results!");
    }

    println!("\n🚀 Advanced Mode SIMD Memory Optimization completed!");

    Ok(())
}

/// Calculate RMS (Root Mean Square) of a signal
#[allow(dead_code)]
fn rms(signal: &[f64]) -> f64 {
    let sum_squares: f64 = signal.iter().map(|&x| x * x).sum();
    (sum_squares / signal.len() as f64).sqrt()
}
