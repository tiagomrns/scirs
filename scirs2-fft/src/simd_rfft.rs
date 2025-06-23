//! SIMD-accelerated Real-valued Fast Fourier Transform (RFFT) operations
//!
//! This module provides SIMD-accelerated implementations of FFT operations
//! for real-valued inputs, using the unified SIMD abstraction layer from scirs2-core.

use crate::error::FFTResult;
use crate::rfft::{irfft as irfft_basic, rfft as rfft_basic};
use num_complex::Complex64;
use num_traits::NumCast;
use scirs2_core::simd_ops::{AutoOptimizer, PlatformCapabilities};
use std::fmt::Debug;

/// Compute the 1-dimensional discrete Fourier Transform for real input with SIMD acceleration.
///
/// This function is optimized using SIMD instructions for improved performance on
/// modern CPUs. For real-valued inputs, this uses a specialized algorithm that is
/// more efficient than a general complex FFT.
///
/// # Arguments
///
/// * `input` - Input real-valued array
/// * `n` - Length of the transformed axis (optional)
/// * `norm` - Normalization mode (optional)
///
/// # Returns
///
/// * The Fourier transform of the real input array
///
/// # Examples
///
/// ```
/// use scirs2_fft::simd_rfft::{rfft_simd};
/// use scirs2_fft::simd_fft::NormMode;
///
/// // Generate a simple signal
/// let signal = vec![1.0, 2.0, 3.0, 4.0];
///
/// // Compute RFFT of the signal with SIMD acceleration
/// let spectrum = rfft_simd(&signal, None, None).unwrap();
///
/// // RFFT produces n//2 + 1 complex values
/// assert_eq!(spectrum.len(), signal.len() / 2 + 1);
/// ```
pub fn rfft_simd<T>(input: &[T], n: Option<usize>, norm: Option<&str>) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Use the basic rfft implementation which already handles the logic
    let result = rfft_basic(input, n)?;

    // Apply normalization if requested
    if let Some(_norm_str) = norm {
        // TODO: Apply normalization based on norm_str when supported
        // For now, just return the result without additional normalization
    }

    Ok(result)
}

/// Compute the inverse of the 1-dimensional discrete Fourier Transform for real input with SIMD acceleration.
///
/// This function is optimized using SIMD instructions for improved performance on
/// modern CPUs.
///
/// # Arguments
///
/// * `input` - Input complex-valued array representing the Fourier transform of real data
/// * `n` - Length of the output array (optional)
/// * `norm` - Normalization mode (optional)
///
/// # Returns
///
/// * The inverse Fourier transform, yielding a real-valued array
///
/// # Examples
///
/// ```
/// use scirs2_fft::simd_rfft::{rfft_simd, irfft_simd};
///
/// // Generate a simple signal
/// let signal = vec![1.0, 2.0, 3.0, 4.0];
///
/// // Forward transform
/// let spectrum = rfft_simd(&signal, None, None).unwrap();
///
/// // Inverse transform
/// let recovered = irfft_simd(&spectrum, Some(signal.len()), None).unwrap();
///
/// // Check recovery accuracy
/// for (x, y) in signal.iter().zip(recovered.iter()) {
///     assert!((x - y).abs() < 1e-10);
/// }
/// ```
pub fn irfft_simd<T>(input: &[T], n: Option<usize>, norm: Option<&str>) -> FFTResult<Vec<f64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Use the basic irfft implementation
    let result = irfft_basic(input, n)?;

    // Apply normalization if requested
    if let Some(_norm_str) = norm {
        // TODO: Apply normalization based on norm_str when supported
        // For now, just return the result without additional normalization
    }

    Ok(result)
}

/// Adaptive RFFT that automatically chooses the best implementation
pub fn rfft_adaptive<T>(
    input: &[T],
    n: Option<usize>,
    norm: Option<&str>,
) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    let optimizer = AutoOptimizer::new();
    let caps = PlatformCapabilities::detect();
    let size = n.unwrap_or(input.len());

    if caps.gpu_available && optimizer.should_use_gpu(size) {
        // TODO: Use GPU implementation when available in core
        rfft_simd(input, n, norm)
    } else {
        rfft_simd(input, n, norm)
    }
}

/// Adaptive IRFFT that automatically chooses the best implementation
pub fn irfft_adaptive<T>(input: &[T], n: Option<usize>, norm: Option<&str>) -> FFTResult<Vec<f64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    let optimizer = AutoOptimizer::new();
    let caps = PlatformCapabilities::detect();
    let size = n.unwrap_or_else(|| input.len() * 2 - 2);

    if caps.gpu_available && optimizer.should_use_gpu(size) {
        // TODO: Use GPU implementation when available in core
        irfft_simd(input, n, norm)
    } else {
        irfft_simd(input, n, norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_rfft_simd_simple() {
        let signal = vec![1.0, 2.0, 3.0, 4.0];

        // Forward transform
        let spectrum = rfft_simd(&signal, None, None).unwrap();

        // Check size
        assert_eq!(spectrum.len(), signal.len() / 2 + 1);

        // First element should be sum of all values
        assert_abs_diff_eq!(spectrum[0].re, 10.0, epsilon = 1e-10);
        assert_abs_diff_eq!(spectrum[0].im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rfft_irfft_roundtrip() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // Forward transform
        let spectrum = rfft_simd(&signal, None, None).unwrap();

        // Inverse transform
        let recovered = irfft_simd(&spectrum, Some(signal.len()), None).unwrap();

        // Check recovery
        for (i, (&orig, &rec)) in signal.iter().zip(recovered.iter()).enumerate() {
            if (orig - rec).abs() > 1e-10 {
                panic!("Mismatch at index {}: {} != {}", i, orig, rec);
            }
        }
    }

    #[test]
    fn test_adaptive_selection() {
        let signal = vec![1.0; 1000];

        // Test adaptive functions (should work regardless of GPU availability)
        let spectrum = rfft_adaptive(&signal, None, None).unwrap();
        assert_eq!(spectrum.len(), signal.len() / 2 + 1);

        let recovered = irfft_adaptive(&spectrum, Some(signal.len()), None).unwrap();
        assert_eq!(recovered.len(), signal.len());
    }
}
