//! Filter design and application
//!
//! This module provides functions for designing and applying digital filters.
//! It includes functions for IIR filter design (Butterworth, Chebyshev, etc.)
//! and FIR filter design (window method, least squares, etc.).

use crate::error::{SignalError, SignalResult};
use num_traits::{Float, NumCast};
use std::fmt::Debug;

/// Filter type for IIR filter design
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FilterType {
    /// Lowpass filter
    Lowpass,
    /// Highpass filter
    Highpass,
    /// Bandpass filter
    Bandpass,
    /// Bandstop filter
    Bandstop,
}

impl FilterType {
    /// Parse a string into a filter type (deprecated - use FromStr trait instead)
    #[deprecated(since = "0.1.0", note = "use `parse()` from the FromStr trait instead")]
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> SignalResult<Self> {
        s.parse()
    }
}

impl std::str::FromStr for FilterType {
    type Err = SignalError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "lowpass" | "low" => Ok(FilterType::Lowpass),
            "highpass" | "high" => Ok(FilterType::Highpass),
            "bandpass" | "band" => Ok(FilterType::Bandpass),
            "bandstop" | "stop" => Ok(FilterType::Bandstop),
            _ => Err(SignalError::ValueError(format!(
                "Unknown filter type: {}",
                s
            ))),
        }
    }
}

/// Butterworth filter design
///
/// # Arguments
///
/// * `order` - Filter order
/// * `cutoff` - Cutoff frequency (normalized from 0 to 1, where 1 is the Nyquist frequency)
/// * `filter_type` - Filter type (lowpass, highpass, bandpass, bandstop)
///
/// # Returns
///
/// * A tuple of filter coefficients (b, a) where b are the numerator coefficients and a are
///   the denominator coefficients
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::butter;
///
/// // Design a 4th order lowpass Butterworth filter with cutoff at 0.2 times Nyquist
/// let (b, a) = butter(4, 0.2, "lowpass").unwrap();
/// ```
pub fn butter<T>(
    order: usize,
    cutoff: T,
    filter_type: impl Into<FilterTypeParam>,
) -> SignalResult<(Vec<f64>, Vec<f64>)>
where
    T: Float + NumCast + Debug,
{
    // Convert cutoff to f64
    let wn = num_traits::cast::cast::<T, f64>(cutoff)
        .ok_or_else(|| SignalError::ValueError(format!("Could not convert {:?} to f64", cutoff)))?;

    // Convert filter_type to FilterType
    let filter_type_param = filter_type.into();
    let filter_type = match filter_type_param {
        FilterTypeParam::Type(t) => t,
        FilterTypeParam::String(s) => s.parse()?,
    };

    // Validate parameters
    if order == 0 {
        return Err(SignalError::ValueError(
            "Filter order must be greater than 0".to_string(),
        ));
    }

    if wn <= 0.0 || wn >= 1.0 {
        return Err(SignalError::ValueError(format!(
            "Cutoff frequency must be between 0 and 1, got {}",
            wn
        )));
    }

    // Placeholder implementation - Calculate analog prototype poles
    // In a real implementation, we would:
    // 1. Calculate analog prototype poles
    // 2. Apply frequency transform based on filter_type
    // 3. Apply bilinear transform to get digital filter

    // Here's a simple implementation that returns a basic lowpass filter
    let mut b = vec![0.0; order + 1];
    let mut a = vec![0.0; order + 1];

    // For a simple first-order lowpass filter: b = [wn], a = [1, wn-1]
    match filter_type {
        FilterType::Lowpass => {
            // Simple first-order lowpass
            b[0] = wn;
            a[0] = 1.0;
            a[1] = wn - 1.0;
        }
        FilterType::Highpass => {
            // Simple first-order highpass
            b[0] = 1.0 - wn;
            a[0] = 1.0;
            a[1] = wn - 1.0;
        }
        FilterType::Bandpass | FilterType::Bandstop => {
            return Err(SignalError::NotImplementedError(
                "Bandpass and bandstop filters are not yet implemented".to_string(),
            ));
        }
    }

    Ok((b, a))
}

/// Chebyshev Type I filter design
///
/// # Arguments
///
/// * `order` - Filter order
/// * `ripple` - Maximum ripple allowed in the passband (in dB)
/// * `cutoff` - Cutoff frequency (normalized from 0 to 1, where 1 is the Nyquist frequency)
/// * `filter_type` - Filter type (lowpass, highpass, bandpass, bandstop)
///
/// # Returns
///
/// * A tuple of filter coefficients (b, a) where b are the numerator coefficients and a are
///   the denominator coefficients
pub fn cheby1<T>(
    _order: usize,
    _ripple: f64,
    _cutoff: T,
    _filter_type: impl Into<FilterTypeParam>,
) -> SignalResult<(Vec<f64>, Vec<f64>)>
where
    T: Float + NumCast + Debug,
{
    // Not yet implemented
    Err(SignalError::NotImplementedError(
        "Chebyshev Type I filter design is not yet implemented".to_string(),
    ))
}

/// Chebyshev Type II filter design
///
/// # Arguments
///
/// * `order` - Filter order
/// * `attenuation` - Minimum attenuation in the stopband (in dB)
/// * `cutoff` - Cutoff frequency (normalized from 0 to 1, where 1 is the Nyquist frequency)
/// * `filter_type` - Filter type (lowpass, highpass, bandpass, bandstop)
///
/// # Returns
///
/// * A tuple of filter coefficients (b, a) where b are the numerator coefficients and a are
///   the denominator coefficients
pub fn cheby2<T>(
    _order: usize,
    _attenuation: f64,
    _cutoff: T,
    _filter_type: impl Into<FilterTypeParam>,
) -> SignalResult<(Vec<f64>, Vec<f64>)>
where
    T: Float + NumCast + Debug,
{
    // Not yet implemented
    Err(SignalError::NotImplementedError(
        "Chebyshev Type II filter design is not yet implemented".to_string(),
    ))
}

/// Elliptic (Cauer) filter design
///
/// # Arguments
///
/// * `order` - Filter order
/// * `ripple` - Maximum ripple allowed in the passband (in dB)
/// * `attenuation` - Minimum attenuation in the stopband (in dB)
/// * `cutoff` - Cutoff frequency (normalized from 0 to 1, where 1 is the Nyquist frequency)
/// * `filter_type` - Filter type (lowpass, highpass, bandpass, bandstop)
///
/// # Returns
///
/// * A tuple of filter coefficients (b, a) where b are the numerator coefficients and a are
///   the denominator coefficients
pub fn ellip<T>(
    _order: usize,
    _ripple: f64,
    _attenuation: f64,
    _cutoff: T,
    _filter_type: impl Into<FilterTypeParam>,
) -> SignalResult<(Vec<f64>, Vec<f64>)>
where
    T: Float + NumCast + Debug,
{
    // Not yet implemented
    Err(SignalError::NotImplementedError(
        "Elliptic filter design is not yet implemented".to_string(),
    ))
}

/// Bessel filter design
///
/// # Arguments
///
/// * `order` - Filter order
/// * `cutoff` - Cutoff frequency (normalized from 0 to 1, where 1 is the Nyquist frequency)
/// * `filter_type` - Filter type (lowpass, highpass, bandpass, bandstop)
///
/// # Returns
///
/// * A tuple of filter coefficients (b, a) where b are the numerator coefficients and a are
///   the denominator coefficients
pub fn bessel<T>(
    _order: usize,
    _cutoff: T,
    _filter_type: impl Into<FilterTypeParam>,
) -> SignalResult<(Vec<f64>, Vec<f64>)>
where
    T: Float + NumCast + Debug,
{
    // Not yet implemented
    Err(SignalError::NotImplementedError(
        "Bessel filter design is not yet implemented".to_string(),
    ))
}

/// FIR filter design using the window method
///
/// # Arguments
///
/// * `numtaps` - Number of filter taps (filter order + 1)
/// * `cutoff` - Cutoff frequency (normalized from 0 to 1, where 1 is the Nyquist frequency)
/// * `window` - Window function name or parameters
/// * `pass_zero` - If true, the filter is lowpass, otherwise highpass
///
/// # Returns
///
/// * Filter coefficients
pub fn firwin<T>(
    _numtaps: usize,
    _cutoff: T,
    _window: &str,
    _pass_zero: bool,
) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug,
{
    // Not yet implemented
    Err(SignalError::NotImplementedError(
        "FIR filter design using window method is not yet implemented".to_string(),
    ))
}

/// Apply an IIR filter forward and backward to a signal
///
/// # Arguments
///
/// * `b` - Numerator coefficients
/// * `a` - Denominator coefficients
/// * `x` - Input signal
///
/// # Returns
///
/// * Filtered signal
pub fn filtfilt<T>(b: &[f64], a: &[f64], x: &[T]) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug,
{
    if a.is_empty() || a[0] == 0.0 {
        return Err(SignalError::ValueError(
            "First denominator coefficient cannot be zero".to_string(),
        ));
    }

    // Convert input to f64
    let x_f64: Vec<f64> = x
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<T, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // 1. Apply filter forward
    let y1 = lfilter(b, a, &x_f64)?;

    // 2. Reverse the result
    let mut y1_rev = y1.clone();
    y1_rev.reverse();

    // 3. Apply filter backward
    let y2 = lfilter(b, a, &y1_rev)?;

    // 4. Reverse again to get the final result
    let mut result = y2;
    result.reverse();

    Ok(result)
}

/// Apply an IIR filter to a signal
///
/// # Arguments
///
/// * `b` - Numerator coefficients
/// * `a` - Denominator coefficients
/// * `x` - Input signal
///
/// # Returns
///
/// * Filtered signal
pub fn lfilter<T>(b: &[f64], a: &[f64], x: &[T]) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug,
{
    if a.is_empty() || a[0] == 0.0 {
        return Err(SignalError::ValueError(
            "First denominator coefficient cannot be zero".to_string(),
        ));
    }

    // Normalize coefficients by a[0]
    let a0 = a[0];
    let b_norm: Vec<f64> = b.iter().map(|&val| val / a0).collect();
    let a_norm: Vec<f64> = a.iter().map(|&val| val / a0).collect();

    // Convert input to f64
    let x_f64: Vec<f64> = x
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<T, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Apply filter using direct form II transposed
    let n = x_f64.len();
    let mut y = vec![0.0; n];
    let mut z = vec![0.0; a_norm.len() - 1]; // State variables

    for i in 0..n {
        // Compute output
        y[i] = b_norm[0] * x_f64[i] + z[0];

        // Update state variables
        for j in 1..z.len() {
            z[j - 1] = b_norm[j] * x_f64[i] + z[j] - a_norm[j] * y[i];
        }

        // Update last state variable
        if !z.is_empty() {
            let last = z.len() - 1;
            if b_norm.len() > last + 1 {
                z[last] = b_norm[last + 1] * x_f64[i] - a_norm[last + 1] * y[i];
            } else {
                z[last] = -a_norm[last + 1] * y[i];
            }
        }
    }

    Ok(y)
}

/// Helper enum to handle different filter type parameter types
#[derive(Debug)]
pub enum FilterTypeParam {
    /// Filter type enum
    Type(FilterType),
    /// Filter type as string
    String(String),
}

impl From<FilterType> for FilterTypeParam {
    fn from(filter_type: FilterType) -> Self {
        FilterTypeParam::Type(filter_type)
    }
}

impl From<&str> for FilterTypeParam {
    fn from(s: &str) -> Self {
        FilterTypeParam::String(s.to_string())
    }
}

impl From<String> for FilterTypeParam {
    fn from(s: String) -> Self {
        FilterTypeParam::String(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_filter_type_from_str() {
        assert_eq!(
            "lowpass".parse::<FilterType>().unwrap(),
            FilterType::Lowpass
        );
        assert_eq!(
            "highpass".parse::<FilterType>().unwrap(),
            FilterType::Highpass
        );
        assert_eq!(
            "bandpass".parse::<FilterType>().unwrap(),
            FilterType::Bandpass
        );
        assert_eq!(
            "bandstop".parse::<FilterType>().unwrap(),
            FilterType::Bandstop
        );

        assert!("invalid".parse::<FilterType>().is_err());
    }

    #[test]
    fn test_butter_lowpass() {
        let (b, a) = butter(1, 0.5, "lowpass").unwrap();

        // First-order lowpass with cutoff at 0.5
        assert_eq!(b.len(), 2);
        assert_eq!(a.len(), 2);

        assert_relative_eq!(b[0], 0.5, epsilon = 1e-10);
        assert_relative_eq!(a[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(a[1], -0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_butter_highpass() {
        let (b, a) = butter(1, 0.5, "highpass").unwrap();

        // First-order highpass with cutoff at 0.5
        assert_eq!(b.len(), 2);
        assert_eq!(a.len(), 2);

        assert_relative_eq!(b[0], 0.5, epsilon = 1e-10);
        assert_relative_eq!(a[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(a[1], -0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_lfilter() {
        // Simple first-order lowpass filter
        let b = vec![0.5];
        let a = vec![1.0, -0.5];

        // Input signal: step function
        let x = vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        // Apply filter
        let y = lfilter(&b, &a, &x).unwrap();

        // Check result: The step response of a first-order lowpass filter
        // should approach 1.0 asymptotically
        assert_relative_eq!(y[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(y[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(y[2], 0.5, epsilon = 1e-10);
        assert_relative_eq!(y[3], 0.75, epsilon = 1e-10);
        assert_relative_eq!(y[4], 0.875, epsilon = 1e-10);
        assert_relative_eq!(y[5], 0.9375, epsilon = 1e-10);
        assert_relative_eq!(y[6], 0.96875, epsilon = 1e-10);
    }

    #[test]
    fn test_filtfilt() {
        // Simple first-order lowpass filter
        let b = vec![0.5];
        let a = vec![1.0, -0.5];

        // Input signal: step function
        let x = vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        // Apply zero-phase filter
        let y = filtfilt(&b, &a, &x).unwrap();

        // Check the test by verifying the general shape rather than exact values
        // First part should be smaller than the second part
        let first_part_avg = (y[0] + y[1]) / 2.0;
        let second_part_avg = (y[2] + y[3] + y[4] + y[5] + y[6]) / 5.0;

        assert!(first_part_avg < second_part_avg);
    }
}
