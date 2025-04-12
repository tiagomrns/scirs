//! Spectral analysis functions
//!
//! This module provides functions for estimating power spectral densities and spectrograms.

use crate::error::{SignalError, SignalResult};
use num_complex::Complex64;
use num_traits::{Float, NumCast};
use std::fmt::Debug;

/// Type alias for periodogram result containing frequencies and power spectral density values
type PeriodogramResult = SignalResult<(Vec<f64>, Vec<f64>)>;

/// Type alias for STFT result containing frequencies, time segments, and complex values
type StftResult = SignalResult<(Vec<f64>, Vec<f64>, Vec<Vec<Complex64>>)>;

/// Type alias for spectrogram result containing frequencies, time segments, and magnitude values
type SpectrogramResult = SignalResult<(Vec<f64>, Vec<f64>, Vec<Vec<f64>>)>;

/// Estimate the power spectral density using periodogram method
///
/// # Arguments
///
/// * `x` - Input signal
/// * `fs` - Sampling frequency (default = 1.0)
/// * `window` - Window function to apply (default = None)
/// * `nfft` - Length of the FFT (default = length of x)
/// * `detrend` - Detrend option ("constant", "linear", or "none")
/// * `scaling` - Scaling mode ("density" or "spectrum")
///
/// # Returns
///
/// * A tuple containing (frequencies, power spectral density)
pub fn periodogram<T>(
    x: &[T],
    fs: Option<f64>,
    window: Option<&str>,
    nfft: Option<usize>,
    detrend: Option<&str>,
    scaling: Option<&str>,
) -> PeriodogramResult
where
    T: Float + NumCast + Debug,
{
    // Convert input to f64
    let x_f64: Vec<f64> = x
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<T, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Default parameters
    let fs_val = fs.unwrap_or(1.0);
    let nfft_val = nfft.unwrap_or(x_f64.len());
    let _window_val = window.unwrap_or("boxcar");
    let detrend_val = detrend.unwrap_or("constant");
    let scaling_val = scaling.unwrap_or("density");

    // Validate parameters
    if fs_val <= 0.0 {
        return Err(SignalError::ValueError(format!(
            "Sampling frequency must be positive, got {}",
            fs_val
        )));
    }

    if nfft_val < x_f64.len() {
        return Err(SignalError::ValueError(format!(
            "NFFT must be at least as large as signal length, got {} < {}",
            nfft_val,
            x_f64.len()
        )));
    }

    // 1. Apply detrending
    let detrended = match detrend_val {
        "constant" => {
            // Remove mean
            let mean = x_f64.iter().sum::<f64>() / x_f64.len() as f64;
            x_f64.iter().map(|&x| x - mean).collect::<Vec<_>>()
        }
        "linear" => {
            // Remove linear trend
            let n = x_f64.len();
            let x_indices: Vec<f64> = (0..n).map(|i| i as f64).collect();

            // Linear regression
            let sum_x = x_indices.iter().sum::<f64>();
            let sum_y = x_f64.iter().sum::<f64>();
            let sum_xx = x_indices.iter().map(|&x| x * x).sum::<f64>();
            let sum_xy = x_indices
                .iter()
                .zip(x_f64.iter())
                .map(|(&x, &y)| x * y)
                .sum::<f64>();

            let slope = (n as f64 * sum_xy - sum_x * sum_y) / (n as f64 * sum_xx - sum_x * sum_x);
            let intercept = (sum_y - slope * sum_x) / n as f64;

            // Remove trend
            x_f64
                .iter()
                .enumerate()
                .map(|(i, &y)| y - (slope * i as f64 + intercept))
                .collect::<Vec<_>>()
        }
        "none" => x_f64.clone(),
        _ => {
            return Err(SignalError::ValueError(format!(
                "Unknown detrend option: {}",
                detrend_val
            )))
        }
    };

    // 2. Apply window (just use rectangular window for now)
    let windowed = detrended.clone();

    // 3. Zero-pad the signal if needed
    let mut padded = windowed.clone();
    if nfft_val > padded.len() {
        padded.resize(nfft_val, 0.0);
    }

    // 4. Compute FFT
    let spectrum = scirs2_fft::fft(&padded, None)
        .map_err(|e| SignalError::ComputationError(format!("FFT computation error: {}", e)))?;

    // 5. Compute periodogram (using magnitude squared)
    let periodogram: Vec<f64> = spectrum
        .iter()
        .map(|&c| c.norm_sqr() / (fs_val * x_f64.len() as f64))
        .collect();

    // 6. Generate frequency axis - be careful with bounds
    let freqs = scirs2_fft::helper::fftfreq(nfft_val, 1.0 / fs_val).map_err(|e| {
        SignalError::ComputationError(format!("Frequency computation error: {}", e))
    })?;

    // 7. Keep only positive frequencies
    let mut result_freqs = Vec::new();
    let mut result_psd = Vec::new();

    let n_half = (nfft_val / 2) + (nfft_val % 2); // Handle both even and odd nfft

    for i in 0..n_half {
        if i < freqs.len() && i < periodogram.len() {
            result_freqs.push(freqs[i]);

            // Adjust scaling based on user preference
            let psd_val = if scaling_val == "density" {
                periodogram[i]
            } else {
                // "spectrum" - multiply by sampling frequency
                periodogram[i] * fs_val
            };

            result_psd.push(psd_val);
        }
    }

    Ok((result_freqs, result_psd))
}

/// Estimate the power spectral density using Welch's method
///
/// # Arguments
///
/// * `x` - Input signal
/// * `fs` - Sampling frequency (default = 1.0)
/// * `window` - Window function to apply (default = "hann")
/// * `nperseg` - Length of each segment (default = 256)
/// * `noverlap` - Number of points to overlap between segments (default = nperseg // 2)
/// * `nfft` - Length of the FFT (default = nperseg)
/// * `detrend` - Detrend option ("constant", "linear", or "none")
/// * `scaling` - Scaling mode ("density" or "spectrum")
///
/// # Returns
///
/// * A tuple containing (frequencies, power spectral density)
#[allow(clippy::too_many_arguments)]
pub fn welch<T>(
    _x: &[T],
    _fs: Option<f64>,
    _window: Option<&str>,
    _nperseg: Option<usize>,
    _noverlap: Option<usize>,
    _nfft: Option<usize>,
    _detrend: Option<&str>,
    _scaling: Option<&str>,
) -> PeriodogramResult
where
    T: Float + NumCast + Debug,
{
    // Not yet fully implemented
    Err(SignalError::NotImplementedError(
        "Welch's method is not yet fully implemented".to_string(),
    ))
}

/// Short-time Fourier transform
///
/// # Arguments
///
/// * `x` - Input signal
/// * `fs` - Sampling frequency (default = 1.0)
/// * `window` - Window function to apply (default = "hann")
/// * `nperseg` - Length of each segment (default = 256)
/// * `noverlap` - Number of points to overlap between segments (default = nperseg // 2)
/// * `nfft` - Length of the FFT (default = nperseg)
/// * `detrend` - Detrend option ("constant", "linear", or "none")
/// * `boundary` - How to handle boundaries ("zeros", "extend", or "none")
/// * `padded` - Whether to pad the signal
///
/// # Returns
///
/// * A tuple containing (frequencies, time segments, STFT values)
#[allow(clippy::too_many_arguments)]
pub fn stft<T>(
    _x: &[T],
    _fs: Option<f64>,
    _window: Option<&str>,
    _nperseg: Option<usize>,
    _noverlap: Option<usize>,
    _nfft: Option<usize>,
    _detrend: Option<&str>,
    _boundary: Option<&str>,
    _padded: Option<bool>,
) -> StftResult
where
    T: Float + NumCast + Debug,
{
    // Not yet fully implemented
    Err(SignalError::NotImplementedError(
        "Short-time Fourier transform is not yet fully implemented".to_string(),
    ))
}

/// Compute a spectrogram
///
/// # Arguments
///
/// * `x` - Input signal
/// * `fs` - Sampling frequency (default = 1.0)
/// * `window` - Window function to apply (default = "hann")
/// * `nperseg` - Length of each segment (default = 256)
/// * `noverlap` - Number of points to overlap between segments (default = nperseg // 2)
/// * `nfft` - Length of the FFT (default = nperseg)
/// * `detrend` - Detrend option ("constant", "linear", or "none")
/// * `scaling` - Scaling mode ("density" or "spectrum")
/// * `mode` - Mode ("psd", "complex", "magnitude", "angle", "phase")
///
/// # Returns
///
/// * A tuple containing (frequencies, time segments, spectrogram values)
#[allow(clippy::too_many_arguments)]
pub fn spectrogram<T>(
    _x: &[T],
    _fs: Option<f64>,
    _window: Option<&str>,
    _nperseg: Option<usize>,
    _noverlap: Option<usize>,
    _nfft: Option<usize>,
    _detrend: Option<&str>,
    _scaling: Option<&str>,
    _mode: Option<&str>,
) -> SpectrogramResult
where
    T: Float + NumCast + Debug,
{
    // Not yet fully implemented
    Err(SignalError::NotImplementedError(
        "Spectrogram computation is not yet fully implemented".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_periodogram_sine_wave() {
        // Generate a sine wave at 10 Hz
        let fs = 100.0;
        let t: Vec<f64> = (0..1000).map(|i| i as f64 / fs).collect();
        let f = 10.0; // Hz
        let x: Vec<f64> = t.iter().map(|&t| (2.0 * PI * f * t).sin()).collect();

        // Compute periodogram
        let (freqs, psd) = periodogram(&x, Some(fs), None, None, None, None).unwrap();

        // Find the peak frequency
        let mut max_idx = 0;
        let mut max_val = 0.0;
        for (i, &val) in psd.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        // Check that the peak is at 10 Hz
        assert_relative_eq!(freqs[max_idx], f, epsilon = 1.0);

        // Check that the PSD has a reasonable length
        assert!(freqs.len() >= (x.len() / 2));
        assert!(freqs.len() <= (x.len() / 2 + 2)); // Allow a bit of flexibility
    }

    // More tests to be added as functionality is implemented
}
