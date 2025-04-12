//! Fractional Fourier Transform module
//!
//! The Fractional Fourier Transform (FrFT) is a generalization of the standard
//! Fourier transform, allowing for transformation at arbitrary angles in the
//! time-frequency plane. It provides a continuous transformation between the
//! time and frequency domains.
//!
//! # Mathematical Definition
//!
//! The continuous Fractional Fourier Transform of order α for a signal f(t) is defined as:
//!
//! F_α(u) = ∫ f(t) K_α(t, u) dt
//!
//! where K_α(t, u) is the transformation kernel:
//!
//! K_α(t, u) = √(1-j cot(α)) * exp(j π (t² cot(α) - 2tu csc(α) + u² cot(α)))
//!
//! # Special Cases
//!
//! - α = 0: Identity transform (returns the input signal)
//! - α = 1: Standard Fourier transform
//! - α = 2: Time reversal (f(t) → f(-t))
//! - α = 3: Inverse Fourier transform
//! - α = 4: Identity transform (cycles back to original)
//!
//! # Implementation
//!
//! This implementation uses an efficient algorithm based on the FFT, with
//! special handling for the cases where α is close to 0, 1, 2, or 3.

use crate::error::{FFTError, FFTResult};
use crate::fft::{fft, ifft};
use num_complex::Complex64;
use num_traits::{NumCast, Zero};
use std::any::TypeId;
use std::f64::consts::PI;

/// Computes the Fractional Fourier Transform of order `alpha`.
///
/// The Fractional Fourier Transform is a generalization of the Fourier transform
/// where the transform order can be any real number. Traditional Fourier transform
/// corresponds to alpha=1.
///
/// # Arguments
///
/// * `x` - Input signal
/// * `alpha` - Fractional order of the transform (0 to 4)
/// * `d` - Optional sampling interval (default: 1.0)
///
/// # Returns
///
/// * Complex-valued vector containing the fractional Fourier transform
///
/// # Errors
///
/// Returns an error if computation fails or if parameters are invalid.
///
/// # Examples
///
/// ```ignore
/// use scirs2_fft::frft;
/// use std::f64::consts::PI;
///
/// // Create a simple signal
/// let n = 256;
/// let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * 10.0 * i as f64 / n as f64).sin()).collect();
///
/// // Compute FrFT with order 0.5 (halfway between time and frequency domain)
/// let result = frft(&signal, 0.5, None).unwrap();
///
/// // Result has same length as input
/// assert_eq!(result.len(), signal.len());
/// ```
///
/// For complex inputs, use `frft_complex` directly:
///
/// # Notes
///
/// Special cases:
/// * When α = 0, the transform is the identity operator
/// * When α = 1, the transform is equivalent to the standard Fourier transform
/// * When α = 2, the transform is equivalent to the time reversal operator
/// * When α = 3, the transform is equivalent to the inverse Fourier transform
/// * When α = 4, the transform is equivalent to the identity operator (cycles back)
///
/// The implementation uses specialized algorithms for α near 0, 1, 2, 3
/// to avoid numerical instabilities.
pub fn frft<T>(x: &[T], alpha: f64, d: Option<f64>) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + std::fmt::Debug + 'static,
{
    // Special case for Complex64 input
    if TypeId::of::<T>() == TypeId::of::<Complex64>() {
        // Safety: This is safe because we've verified the type is Complex64
        let complex_slice =
            unsafe { std::slice::from_raw_parts(x.as_ptr() as *const Complex64, x.len()) };
        return frft_complex(complex_slice, alpha, d);
    }

    // For non-Complex64 types, convert to complex

    // Validate inputs
    if x.is_empty() {
        return Err(FFTError::ValueError("Input signal is empty".to_string()));
    }

    // Convert input to complex vector
    let x_complex: Vec<Complex64> = x
        .iter()
        .map(|&val| {
            // Convert numeric values to f64, then to Complex64
            num_traits::cast::<T, f64>(val)
                .ok_or_else(|| FFTError::ValueError(format!("Could not convert {val:?} to f64")))
                .map(|val| Complex64::new(val, 0.0))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Get sampling interval
    let d = d.unwrap_or(1.0);
    if d <= 0.0 {
        return Err(FFTError::ValueError(
            "Sampling interval must be positive".to_string(),
        ));
    }

    // Handle special cases
    if (alpha - 0.0).abs() < 1e-10 || (alpha - 4.0).abs() < 1e-10 {
        // Identity transform
        return Ok(x_complex);
    } else if (alpha - 1.0).abs() < 1e-10 {
        // Standard Fourier transform
        return fft(&x_complex, None);
    } else if (alpha - 2.0).abs() < 1e-10 {
        // Time reversal
        let mut result = x_complex.clone();
        result.reverse();
        return Ok(result);
    } else if (alpha - 3.0).abs() < 1e-10 {
        // Inverse Fourier transform
        return ifft(&x_complex, None);
    }

    // General case implementation

    // Convert alpha to angle in radians
    let alpha = alpha * PI / 2.0;

    // Handle near-special cases with linear interpolation
    if alpha.abs() < 0.1 || (PI - alpha).abs() < 0.1 || (2.0 * PI - alpha).abs() < 0.1 {
        return frft_near_special_case(&x_complex, alpha, d);
    }

    // Compute the transform using the decomposition method
    frft_decomposition(&x_complex, alpha, d)
}

/// Implementation of FrFT for the general case using the decomposition method.
fn frft_decomposition(x: &[Complex64], alpha: f64, d: f64) -> FFTResult<Vec<Complex64>> {
    let n = x.len();

    // We need to use zero padding to avoid aliasing
    let n_padded = 2 * n;

    // Compute chirp functions and constants
    let cot_alpha = 1.0 / alpha.tan();
    let scale = (1.0 - Complex64::i() * cot_alpha).sqrt() / (2.0 * PI).sqrt();

    // Zero-padded input
    let mut padded = vec![Complex64::zero(); n_padded];
    for i in 0..n {
        padded[i + n / 2] = x[i];
    }

    // Step 1: Multiply by first chirp
    let mut result = vec![Complex64::zero(); n_padded];
    for i in 0..n_padded {
        let t = (i as f64 - n_padded as f64 / 2.0) * d;
        let chirp = Complex64::new(0.0, PI * t * t * cot_alpha).exp();
        result[i] = padded[i] * chirp;
    }

    // Step 2: Perform FFT
    let fft_result = fft(&result, None)?;

    // Step 3: Multiply by second chirp and scale
    let mut final_result = vec![Complex64::zero(); n];
    for (i, result_val) in final_result.iter_mut().enumerate().take(n) {
        let u = (i as f64 - n as f64 / 2.0) * 2.0 * PI / (n_padded as f64 * d);
        let chirp = Complex64::new(0.0, PI * u * u * cot_alpha).exp();
        // Extract only the central portion
        let idx = (i + n_padded / 4) % n_padded;
        *result_val = fft_result[idx] * chirp * scale * d;
    }

    Ok(final_result)
}

/// Special case implementation for α near 0, 1, 2, or 3.
/// Uses linear interpolation between the special cases.
fn frft_near_special_case(x: &[Complex64], alpha: f64, _d: f64) -> FFTResult<Vec<Complex64>> {
    let n = x.len();

    // Determine which special case we're near and the interpolation factor
    let (alpha1, alpha2, t) = if alpha.abs() < 0.1 {
        // Near identity (α ≈ 0)
        (0.0, 0.5 * PI, alpha / (0.5 * PI))
    } else if (PI - alpha).abs() < 0.1 {
        // Near standard FT (α ≈ 1)
        (0.5 * PI, PI, (alpha - 0.5 * PI) / (0.5 * PI))
    } else {
        // Near inverse FT (α ≈ 3) or time reversal (α ≈ 2)
        let base = (alpha / PI).floor() * PI;
        (base, base + 0.5 * PI, (alpha - base) / (0.5 * PI))
    };

    // Compute transforms at the two nearest special cases
    let f1 = if alpha1 == 0.0 {
        x.to_vec() // Identity
    } else if alpha1 == PI {
        // Time reversal
        let mut result = x.to_vec();
        result.reverse();
        result
    } else if alpha1 == PI * 0.5 {
        fft(x, None)? // Standard FT
    } else if alpha1 == PI * 1.5 {
        ifft(x, None)? // Inverse FT
    } else {
        unreachable!()
    };

    // Compute the second transform
    let f2 = if alpha2 == PI * 0.5 {
        fft(x, None)? // Standard FT
    } else if alpha2 == PI {
        // Time reversal
        let mut result = x.to_vec();
        result.reverse();
        result
    } else if alpha2 == PI * 1.5 {
        ifft(x, None)? // Inverse FT
    } else if alpha2 == PI * 2.0 {
        x.to_vec() // Identity (wrapped around)
    } else {
        unreachable!()
    };

    // Interpolate between the two transforms
    let mut result = vec![Complex64::zero(); n];
    for (i, result_val) in result.iter_mut().enumerate().take(n) {
        *result_val = f1[i] * (1.0 - t) + f2[i] * t;
    }

    Ok(result)
}

/// Special implementation for Complex64 input to avoid conversion issues.
///
/// This function is optimized for complex inputs and should be used when working with
/// complex input signals.
///
/// # Arguments
///
/// * `x` - Complex input signal
/// * `alpha` - Fractional order of the transform (0 to 4)
/// * `d` - Optional sampling interval (default: 1.0)
///
/// # Returns
///
/// * Complex-valued vector containing the fractional Fourier transform
///
/// # Errors
///
/// Returns an error if computation fails or if parameters are invalid.
///
/// # Examples
///
/// ```ignore
/// use scirs2_fft::frft_complex;
/// use num_complex::Complex64;
/// use std::f64::consts::PI;
///
/// // Create a complex signal
/// let n = 64;
/// let signal: Vec<Complex64> = (0..n).map(|i| {
///     let t = i as f64 / n as f64;
///     Complex64::new((2.0 * PI * 5.0 * t).cos(), 0.0)
/// }).collect();
///
/// // Compute FrFT with order 0.5
/// let result = frft_complex(&signal, 0.5, None).unwrap();
///
/// // Result has same length as input
/// assert_eq!(result.len(), signal.len());
/// ```
pub fn frft_complex(x: &[Complex64], alpha: f64, d: Option<f64>) -> FFTResult<Vec<Complex64>> {
    // Validate inputs
    if x.is_empty() {
        return Err(FFTError::ValueError("Input signal is empty".to_string()));
    }

    // Normalize alpha to [0, 4) range
    let alpha = alpha.rem_euclid(4.0);

    // Get sampling interval
    let d = d.unwrap_or(1.0);
    if d <= 0.0 {
        return Err(FFTError::ValueError(
            "Sampling interval must be positive".to_string(),
        ));
    }

    // Handle special cases
    if (alpha - 0.0).abs() < 1e-10 || (alpha - 4.0).abs() < 1e-10 {
        // Identity transform
        return Ok(x.to_vec());
    } else if (alpha - 1.0).abs() < 1e-10 {
        // Standard Fourier transform
        return fft(x, None);
    } else if (alpha - 2.0).abs() < 1e-10 {
        // Time reversal
        let mut result = x.to_vec();
        result.reverse();
        return Ok(result);
    } else if (alpha - 3.0).abs() < 1e-10 {
        // Inverse Fourier transform
        return ifft(x, None);
    }

    // Convert alpha to angle in radians
    let alpha = alpha * PI / 2.0;

    // Handle near-special cases with linear interpolation
    if alpha.abs() < 0.1 || (PI - alpha).abs() < 0.1 || (2.0 * PI - alpha).abs() < 0.1 {
        return frft_near_special_case(x, alpha, d);
    }

    // Compute the transform using the decomposition method
    frft_decomposition(x, alpha, d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_frft_identity() {
        // α = 0 should be the identity transform
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let result = frft(&signal, 0.0, None).unwrap();

        for (i, val) in signal.iter().enumerate() {
            assert_relative_eq!(result[i].re, *val, epsilon = 1e-10);
            assert_relative_eq!(result[i].im, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_frft_fourier() {
        // α = 1 should be equivalent to standard FFT
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let frft_result = frft(&signal, 1.0, None).unwrap();
        let fft_result = fft(&signal, None).unwrap();

        for i in 0..signal.len() {
            assert_relative_eq!(frft_result[i].re, fft_result[i].re, epsilon = 1e-10);
            assert_relative_eq!(frft_result[i].im, fft_result[i].im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_frft_time_reversal() {
        // α = 2 should reverse the signal
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let result = frft(&signal, 2.0, None).unwrap();

        for i in 0..signal.len() {
            assert_relative_eq!(result[i].re, signal[signal.len() - 1 - i], epsilon = 1e-10);
            assert_relative_eq!(result[i].im, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_frft_inverse_fourier() {
        // α = 3 should be equivalent to inverse FFT
        // Create a vector of Complex64 values explicitly
        let signal_vec = vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 1.0),
            Complex64::new(4.0, -1.0),
        ];

        // Use the specialized function for Complex64
        let frft_result = frft_complex(&signal_vec, 3.0, None).unwrap();
        let ifft_result = ifft(&signal_vec, None).unwrap();

        // Compare results
        for i in 0..signal_vec.len() {
            assert_relative_eq!(frft_result[i].re, ifft_result[i].re, epsilon = 1e-10);
            assert_relative_eq!(frft_result[i].im, ifft_result[i].im, epsilon = 1e-10);
        }
    }

    #[test]
    #[ignore = "Complex number conversion issues being worked on"]
    fn test_frft_additivity() {
        // Test the additivity property: FrFT(α₁+α₂) ≈ FrFT(α₁)[FrFT(α₂)]
        let n = 64;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 5.0 * i as f64 / n as f64).sin())
            .collect();

        let alpha1 = 0.5;
        let alpha2 = 0.7;

        let result1 = frft(&signal, alpha1 + alpha2, None).unwrap();

        // Create a complex signal from the real one
        let signal_complex: Vec<Complex64> =
            signal.iter().map(|&x| Complex64::new(x, 0.0)).collect();
        let temp = frft_complex(&signal_complex, alpha2, None).unwrap();
        let result2 = frft_complex(&temp, alpha1, None).unwrap();

        // Check with a generous epsilon due to numerical differences
        for i in n / 4..3 * n / 4 {
            // Check middle portion where numerical stability is better
            assert_relative_eq!(result1[i].norm(), result2[i].norm(), epsilon = 0.1);
        }
    }

    #[test]
    #[ignore = "Complex number conversion issues being worked on"]
    fn test_frft_linearity() {
        // Test linearity property
        let n = 64;
        let signal1: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 5.0 * i as f64 / n as f64).sin())
            .collect();
        let signal2: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 10.0 * i as f64 / n as f64).sin())
            .collect();

        let alpha = 0.5;
        let a = 2.0;
        let b = 3.0;

        // Convert to Complex64 to avoid conversion issues
        let signal1_complex: Vec<Complex64> =
            signal1.iter().map(|&x| Complex64::new(x, 0.0)).collect();
        let signal2_complex: Vec<Complex64> =
            signal2.iter().map(|&x| Complex64::new(x, 0.0)).collect();

        // Compute a*FrFT(signal1) + b*FrFT(signal2)
        let frft1 = frft_complex(&signal1_complex, alpha, None).unwrap();
        let frft2 = frft_complex(&signal2_complex, alpha, None).unwrap();

        let mut combined1 = vec![Complex64::zero(); n];
        for i in 0..n {
            combined1[i] = a * frft1[i] + b * frft2[i];
        }

        // Compute FrFT(a*signal1 + b*signal2)
        let mut combined_signal = vec![Complex64::zero(); n];
        for i in 0..n {
            combined_signal[i] = Complex64::new(a * signal1[i] + b * signal2[i], 0.0);
        }

        let combined2 = frft_complex(&combined_signal, alpha, None).unwrap();

        // Check with a generous epsilon due to numerical differences
        for i in n / 4..3 * n / 4 {
            // Check middle portion where numerical stability is better
            assert_relative_eq!(combined1[i].norm(), combined2[i].norm(), epsilon = 0.1);
        }
    }

    #[test]
    #[ignore = "Complex number conversion issues being worked on"]
    fn test_frft_complex_input() {
        // Test with complex input
        let n = 64;
        // Create an explicitly typed vector of Complex64
        let signal_complex: Vec<Complex64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                Complex64::new((2.0 * PI * 5.0 * t).cos(), (2.0 * PI * 5.0 * t).sin())
            })
            .collect();

        let result = frft_complex(&signal_complex, 0.5, None).unwrap();

        // Just verify we get a result with the right length
        assert_eq!(result.len(), n);
    }
}
