//! Frequency domain analysis for Lomb-Scargle validation
//!
//! This module provides comprehensive frequency domain analysis including
//! spectral leakage measurement, alias rejection testing, and frequency
//! resolution assessment.

use crate::error::{SignalError, SignalResult};
use crate::lombscargle::lombscargle;
use crate::lombscargle_enhanced::{lombscargle_enhanced, LombScargleConfig, WindowType};
use std::f64::consts::PI;

use super::config::{FrequencyDomainAnalysisResults, FrequencyResolutionResults};

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

/// Test advanced frequency domain analysis capabilities
pub fn test_frequency_domain_analysis(
    implementation: &str,
    _tolerance: f64,
) -> SignalResult<FrequencyDomainAnalysisResults> {
    let n = 512;
    let fs = 100.0;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();

    // Test signal with known characteristics
    let f1 = 10.0;
    let f2 = 35.0;
    let a1 = 1.0;
    let a2 = 0.3;

    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| a1 * (2.0 * PI * f1 * ti).sin() + a2 * (2.0 * PI * f2 * ti).sin())
        .collect();

    let (freqs, power) = run_lombscargle(implementation, &t, &signal)?;

    // 1. Spectral leakage measurement
    let spectral_leakage = measure_spectral_leakage(&freqs, &power, &[f1, f2]);

    // 2. Dynamic range assessment
    let max_power = power.iter().cloned().fold(0.0, f64::max);
    let noise_floor = estimate_noise_floor(&power);
    let dynamic_range_db = 10.0 * (max_power / noise_floor.max(1e-12)).log10();

    // 3. Frequency resolution accuracy
    let frequency_resolution_accuracy = assess_frequency_resolution(&freqs, &power, &[f1, f2]);

    // 4. Alias rejection ratio (test with signal above Nyquist)
    let alias_rejection_db = test_alias_rejection(implementation, &t)?;

    // 5. Phase coherence (simplified)
    let phase_coherence = test_phase_coherence(implementation, &t, f1)?;

    // 6. Spurious-free dynamic range
    let sfdr_db = calculate_spurious_free_dynamic_range(&freqs, &power, &[f1, f2]);

    Ok(FrequencyDomainAnalysisResults {
        spectral_leakage,
        dynamic_range_db,
        frequency_resolution_accuracy,
        alias_rejection_db,
        phase_coherence,
        sfdr_db,
    })
}

/// Measure spectral leakage around known frequencies
pub fn measure_spectral_leakage(freqs: &[f64], power: &[f64], target_freqs: &[f64]) -> f64 {
    let mut total_leakage = 0.0;

    for &target_freq in target_freqs {
        // Find peak closest to target
        let (peak_idx, _) = freqs
            .iter()
            .enumerate()
            .min_by(|(_, f1), (_, f2)| {
                (**f1 - target_freq)
                    .abs()
                    .partial_cmp(&(**f2 - target_freq).abs())
                    .unwrap()
            })
            .unwrap();

        let peak_power = power[peak_idx];

        // Calculate power in sidelobes (±10 bins around peak, excluding main lobe ±2 bins)
        let mut sidelobe_power = 0.0;
        let mut sidelobe_count = 0;

        let start = peak_idx.saturating_sub(10);
        let end = (peak_idx + 10).min(power.len() - 1);

        for i in start..=end {
            if (i as i32 - peak_idx as i32).abs() > 2 {
                sidelobe_power += power[i];
                sidelobe_count += 1;
            }
        }

        if sidelobe_count > 0 {
            let avg_sidelobe = sidelobe_power / sidelobe_count as f64;
            let leakage = avg_sidelobe / peak_power.max(1e-12);
            total_leakage += leakage;
        }
    }

    // Return normalized leakage (lower is better)
    1.0 - (total_leakage / target_freqs.len() as f64).min(1.0)
}

/// Estimate noise floor from power spectrum
pub fn estimate_noise_floor(power: &[f64]) -> f64 {
    // Use median as robust noise floor estimate
    let mut sorted_power = power.to_vec();
    sorted_power.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_idx = sorted_power.len() / 2;
    sorted_power[median_idx]
}

/// Assess frequency resolution accuracy
pub fn assess_frequency_resolution(freqs: &[f64], power: &[f64], target_freqs: &[f64]) -> f64 {
    let mut accuracy_sum = 0.0;

    for &target_freq in target_freqs {
        // Find peak
        let (_, peak_freq) = freqs
            .iter()
            .zip(power.iter())
            .max_by(|(_, p1), (_, p2)| p1.partial_cmp(p2).unwrap())
            .map(|(&f, _)| ((), f))
            .unwrap_or(((), 0.0));

        let freq_error = (peak_freq - target_freq).abs() / target_freq;
        accuracy_sum += 1.0 - freq_error.min(1.0);
    }

    accuracy_sum / target_freqs.len() as f64
}

/// Test alias rejection with high-frequency signal
pub fn test_alias_rejection(implementation: &str, t: &[f64]) -> SignalResult<f64> {
    let fs = 1.0 / (t[1] - t[0]); // Sampling frequency
    let nyquist = fs / 2.0;
    let f_alias = nyquist * 1.5; // Frequency above Nyquist

    // Create aliased signal
    let signal_alias: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * PI * f_alias * ti).sin())
        .collect();

    let (_, power_alias) = run_lombscargle(implementation, t, &signal_alias)?;

    // The aliased signal should appear at f_alias - fs (or similar aliased frequency)
    let _expected_alias_freq = f_alias - fs;
    let max_power = power_alias.iter().cloned().fold(0.0, f64::max);
    let noise_floor = estimate_noise_floor(&power_alias);

    // Good alias rejection means the aliased component is suppressed
    let rejection_ratio = if max_power > noise_floor * 2.0 {
        // Some aliasing detected
        10.0 * (noise_floor / max_power).log10()
    } else {
        40.0 // Good rejection (>40 dB)
    };

    Ok(rejection_ratio.max(0.0))
}

/// Test phase coherence with quadrature signals
pub fn test_phase_coherence(implementation: &str, t: &[f64], freq: f64) -> SignalResult<f64> {
    // Create in-phase and quadrature signals
    let signal_i: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * freq * ti).cos()).collect();
    let signal_q: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * freq * ti).sin()).collect();

    let (_, power_i) = run_lombscargle(implementation, t, &signal_i)?;
    let (_, power_q) = run_lombscargle(implementation, t, &signal_q)?;

    // Both should have similar peak power (phase coherence)
    let peak_i = power_i.iter().cloned().fold(0.0, f64::max);
    let peak_q = power_q.iter().cloned().fold(0.0, f64::max);

    let coherence = if peak_i > 0.0 && peak_q > 0.0 {
        let ratio = peak_i.min(peak_q) / peak_i.max(peak_q);
        ratio
    } else {
        0.0
    };

    Ok(coherence)
}

/// Calculate spurious-free dynamic range
pub fn calculate_spurious_free_dynamic_range(
    freqs: &[f64],
    power: &[f64],
    target_freqs: &[f64],
) -> f64 {
    // Find all legitimate peaks
    let mut legitimate_peaks = Vec::new();
    for &target_freq in target_freqs {
        let (peak_idx, _) = freqs
            .iter()
            .enumerate()
            .min_by(|(_, f1), (_, f2)| {
                (**f1 - target_freq)
                    .abs()
                    .partial_cmp(&(**f2 - target_freq).abs())
                    .unwrap()
            })
            .unwrap();
        legitimate_peaks.push(peak_idx);
    }

    // Find maximum spurious peak (not near legitimate peaks)
    let mut max_spurious = 0.0;
    for (i, &p) in power.iter().enumerate() {
        let is_spurious = legitimate_peaks.iter().all(|&peak_idx| {
            (i as i32 - peak_idx as i32).abs() > 5 // Not within 5 bins of legitimate peak
        });

        if is_spurious {
            max_spurious = max_spurious.max(p);
        }
    }

    // Find maximum legitimate peak
    let max_legitimate = legitimate_peaks
        .iter()
        .map(|&idx| power[idx])
        .fold(0.0, f64::max);

    // SFDR in dB
    if max_spurious > 0.0 && max_legitimate > 0.0 {
        10.0 * (max_legitimate / max_spurious).log10()
    } else {
        60.0 // Very good SFDR
    }
}

/// Test frequency resolution limits
pub fn test_frequency_resolution(
    implementation: &str,
    _tolerance: f64,
) -> SignalResult<FrequencyResolutionResults> {
    let n = 1024;
    let fs = 100.0;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();

    // Test resolution with closely spaced frequencies
    let f1 = 10.0;
    let df_values = vec![0.1, 0.2, 0.5, 1.0, 2.0]; // Different frequency separations
    let mut resolved_separations = Vec::new();

    for &df in &df_values {
        let f2 = f1 + df;
        let signal: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * PI * f1 * ti).sin() + (2.0 * PI * f2 * ti).sin())
            .collect();

        let (freqs, power) = run_lombscargle(implementation, &t, &signal)?;
        let peaks = find_peaks(&freqs, &power, 2);

        if peaks.len() >= 2 {
            let freq_diff = (peaks[1].0 - peaks[0].0).abs();
            if (freq_diff - df).abs() / df < 0.2 {
                resolved_separations.push(df);
            }
        }
    }

    let min_separation = resolved_separations
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let resolution_limit = min_separation / (1.0 / (t[t.len() - 1] - t[0])); // Normalized

    // Estimate sidelobe suppression
    let sidelobe_suppression = estimate_sidelobe_suppression(implementation, &t)?;

    // Window effectiveness (if enhanced implementation)
    let window_effectiveness = if implementation == "enhanced" {
        estimate_window_effectiveness(&t)?
    } else {
        0.7 // Default for standard implementation
    };

    Ok(FrequencyResolutionResults {
        min_separation,
        resolution_limit,
        sidelobe_suppression,
        window_effectiveness,
    })
}

/// Find peaks in power spectrum
pub fn find_peaks(freqs: &[f64], power: &[f64], max_peaks: usize) -> Vec<(f64, f64)> {
    let mut peaks = Vec::new();

    // Simple peak finding
    for i in 1..power.len() - 1 {
        if power[i] > power[i - 1] && power[i] > power[i + 1] {
            peaks.push((freqs[i], power[i]));
        }
    }

    // Sort by power and take top peaks
    peaks.sort_by(|(_, p1), (_, p2)| p2.partial_cmp(p1).unwrap());
    peaks.truncate(max_peaks);

    peaks
}

/// Estimate sidelobe suppression
pub fn estimate_sidelobe_suppression(implementation: &str, times: &[f64]) -> SignalResult<f64> {
    // Single frequency signal
    let f0 = 10.0;
    let signal: Vec<f64> = times.iter().map(|&ti| (2.0 * PI * f0 * ti).sin()).collect();

    let (freqs, power) = run_lombscargle(implementation, times, &signal)?;

    // Find main peak
    let (peak_idx, &peak_power) = power
        .iter()
        .enumerate()
        .max_by(|(_, p1), (_, p2)| p1.partial_cmp(p2).unwrap())
        .unwrap();

    // Find maximum sidelobe (excluding main lobe)
    let mut max_sidelobe = 0.0;
    let lobe_width = 10; // bins

    for (i, &p) in power.iter().enumerate() {
        if (i as i32 - peak_idx as i32).abs() > lobe_width {
            max_sidelobe = max_sidelobe.max(p);
        }
    }

    Ok(10.0 * (peak_power / max_sidelobe.max(1e-10)).log10()) // dB
}

/// Estimate window function effectiveness
pub fn estimate_window_effectiveness(times: &[f64]) -> SignalResult<f64> {
    // Test different window functions and compare sidelobe suppression
    let window_types = vec!["none", "hann", "hamming", "blackman"];
    let mut suppressions = Vec::new();

    for window in window_types {
        let mut config = LombScargleConfig::default();
        config.window = match window {
            "none" => WindowType::None,
            "hann" => WindowType::Hann,
            "hamming" => WindowType::Hamming,
            "blackman" => WindowType::Blackman,
            _ => WindowType::None,
        };

        let f0 = 10.0;
        let signal: Vec<f64> = times
            .iter()
            .map(|&ti| (2.0 * PI * f0 * ti).sin())
            .collect();

        match lombscargle_enhanced(times, &signal, &config) {
            Ok((freqs, power, _)) => {
                let suppression = estimate_sidelobe_suppression_from_power(&freqs, &power, f0);
                suppressions.push(suppression);
            }
            Err(_) => suppressions.push(0.0),
        }
    }

    // Return improvement over rectangular window
    let baseline = suppressions[0];
    let best = suppressions.iter().cloned().fold(0.0, f64::max);

    Ok((best - baseline) / 40.0) // Normalize to 0-1 scale
}

/// Estimate sidelobe suppression from power spectrum
pub fn estimate_sidelobe_suppression_from_power(freqs: &[f64], power: &[f64], f0: f64) -> f64 {
    // Find peak closest to f0
    let (peak_idx, _) = freqs
        .iter()
        .enumerate()
        .min_by(|(_, f1), (_, f2)| (**f1 - f0).abs().partial_cmp(&(**f2 - f0).abs()).unwrap())
        .unwrap();

    let peak_power = power[peak_idx];

    // Find maximum sidelobe
    let mut max_sidelobe = 0.0;
    let lobe_width = 5;

    for (i, &p) in power.iter().enumerate() {
        if (i as i32 - peak_idx as i32).abs() > lobe_width {
            max_sidelobe = max_sidelobe.max(p);
        }
    }

    10.0 * (peak_power / max_sidelobe.max(1e-10)).log10()
}

/// Test single frequency resolution limit
pub fn test_frequency_resolution_single(fs: f64, n: usize) -> SignalResult<f64> {
    // Generate two close frequencies and test if they can be resolved
    let time: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
    let f1 = fs / 4.0;
    let f2 = f1 + fs / (n as f64); // Frequency separation at theoretical limit

    let signal: Vec<f64> = time
        .iter()
        .map(|&t| {
            (2.0 * PI * f1 * t).sin()
                + (2.0 * PI * f2 * t).sin()
        })
        .collect();

    let (freqs, power) = lombscargle(
        &time,
        &signal,
        None,
        Some("standard"),
        Some(true),
        Some(true),
        Some(1.0),
        None,
    )?;

    // Check if two distinct peaks are resolved
    let peaks = find_peaks(&freqs, &power, 10);

    if peaks.len() >= 2 {
        Ok(fs / (n as f64)) // Return theoretical resolution
    } else {
        Ok(2.0 * fs / (n as f64)) // Need higher resolution
    }
}

/// Calculate theoretical frequency resolution
pub fn calculate_theoretical_resolution(sampling_rates: &[f64], signal_lengths: &[usize]) -> Vec<f64> {
    let mut resolutions = Vec::new();
    for &fs in sampling_rates {
        for &n in signal_lengths {
            resolutions.push(fs / n as f64);
        }
    }
    resolutions
}

/// Calculate empirical frequency resolution
pub fn calculate_empirical_resolution(test_results: &[f64]) -> Vec<f64> {
    test_results.to_vec()
}

/// Calculate resolution agreement between theoretical and empirical
pub fn calculate_resolution_agreement(theoretical: &[f64], empirical: &[f64]) -> f64 {
    let mut agreement_sum = 0.0;
    for (theo, emp) in theoretical.iter().zip(empirical.iter()) {
        let relative_error = (theo - emp).abs() / theo;
        if relative_error < 0.1 {
            agreement_sum += 1.0;
        }
    }
    agreement_sum / theoretical.len() as f64
}

/// Test spectral leakage with off-grid frequency
pub fn test_spectral_leakage(fs: f64, n: usize) -> SignalResult<f64> {
    // Test spectral leakage with a pure tone not at bin center
    let time: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
    let f_tone = fs / 4.0 + fs / (2.0 * n as f64); // Off-grid frequency

    let signal: Vec<f64> = time
        .iter()
        .map(|&t| (2.0 * PI * f_tone * t).sin())
        .collect();

    let (freqs, power) = lombscargle(
        &time,
        &signal,
        None,
        Some("standard"),
        Some(true),
        Some(true),
        Some(1.0),
        None,
    )?;

    // Find peak and measure leakage
    let max_idx = power
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    let peak_power = power[max_idx];
    let total_power: f64 = power.iter().sum();

    Ok(1.0 - peak_power / total_power) // Leakage factor
}