//! Time series decomposition methods
//!
//! This module provides implementations for decomposing time series into trend,
//! seasonal, and residual components.

use ndarray::Array1;
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{Result, TimeSeriesError};
use crate::utils::moving_average;

/// Result of time series decomposition
#[derive(Debug, Clone)]
pub struct DecompositionResult<F> {
    /// Trend component
    pub trend: Array1<F>,
    /// Seasonal component
    pub seasonal: Array1<F>,
    /// Residual component
    pub residual: Array1<F>,
    /// Original time series
    pub original: Array1<F>,
}

/// Decomposition model type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecompositionModel {
    /// Additive model: Y = T + S + R
    Additive,
    /// Multiplicative model: Y = T * S * R
    Multiplicative,
}

/// Performs classical seasonal decomposition on a time series
///
/// # Arguments
///
/// * `ts` - The time series to decompose
/// * `period` - The seasonal period (e.g., 12 for monthly data with yearly seasonality)
/// * `model` - Decomposition model (additive or multiplicative)
///
/// # Returns
///
/// * Decomposition result containing trend, seasonal, and residual components
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_series::decomposition::{decompose_seasonal, DecompositionModel};
///
/// let ts = array![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0];
/// let result = decompose_seasonal(&ts, 4, DecompositionModel::Additive).unwrap();
/// println!("Trend: {:?}", result.trend);
/// println!("Seasonal: {:?}", result.seasonal);
/// println!("Residual: {:?}", result.residual);
/// ```
pub fn decompose_seasonal<F>(
    ts: &Array1<F>,
    period: usize,
    model: DecompositionModel,
) -> Result<DecompositionResult<F>>
where
    F: Float + FromPrimitive + Debug,
{
    if ts.len() < 2 * period {
        return Err(TimeSeriesError::DecompositionError(format!(
            "Time series length ({}) must be at least twice the seasonal period ({})",
            ts.len(),
            period
        )));
    }

    // 1. Calculate centered moving average (trend)
    let window_size = if period % 2 == 0 { period + 1 } else { period };
    let trend = moving_average(ts, window_size)?;

    // Pad trend with NaN values at the beginning and end
    let half_window = window_size / 2;
    let mut padded_trend = Array1::from_elem(ts.len(), F::nan());
    let offset = half_window;

    // Ensure we don't go out of bounds
    for (i, &val) in trend.iter().enumerate() {
        if i + offset < padded_trend.len() {
            padded_trend[i + offset] = val;
        }
    }

    // 2. Remove trend to get detrended series
    let mut detrended = Array1::zeros(ts.len());
    for i in 0..ts.len() {
        if padded_trend[i].is_nan() {
            detrended[i] = F::zero(); // Set detrended to zero where trend is NaN
        } else {
            match model {
                DecompositionModel::Additive => {
                    detrended[i] = ts[i] - padded_trend[i];
                }
                DecompositionModel::Multiplicative => {
                    if padded_trend[i] == F::zero() {
                        return Err(TimeSeriesError::DecompositionError(
                            "Division by zero in multiplicative model".to_string(),
                        ));
                    }
                    detrended[i] = ts[i] / padded_trend[i];
                }
            }
        }
    }

    // 3. Calculate seasonal component by averaging values for each season
    let mut seasonal = Array1::zeros(ts.len());
    let mut seasonal_pattern = Array1::zeros(period);
    let mut counts = vec![0; period];

    // Calculate average for each position in the seasonal pattern
    for i in 0..ts.len() {
        let pos = i % period;
        if !detrended[i].is_nan() {
            seasonal_pattern[pos] = seasonal_pattern[pos] + detrended[i];
            counts[pos] += 1;
        }
    }

    // Normalize seasonal pattern
    for i in 0..period {
        if counts[i] > 0 {
            seasonal_pattern[i] = seasonal_pattern[i] / F::from_usize(counts[i]).unwrap();
        }
    }

    // Normalize to ensure seasonal components sum to zero (additive) or average to one (multiplicative)
    match model {
        DecompositionModel::Additive => {
            let mean = seasonal_pattern.iter().fold(F::zero(), |acc, &x| acc + x)
                / F::from_usize(period).unwrap();
            for i in 0..period {
                seasonal_pattern[i] = seasonal_pattern[i] - mean;
            }
        }
        DecompositionModel::Multiplicative => {
            let mean = seasonal_pattern.iter().fold(F::zero(), |acc, &x| acc + x)
                / F::from_usize(period).unwrap();
            if mean == F::zero() {
                return Err(TimeSeriesError::DecompositionError(
                    "Division by zero normalizing multiplicative seasonal pattern".to_string(),
                ));
            }
            for i in 0..period {
                seasonal_pattern[i] = seasonal_pattern[i] / mean;
            }
        }
    }

    // Apply seasonal pattern to the whole series
    for i in 0..ts.len() {
        seasonal[i] = seasonal_pattern[i % period];
    }

    // 4. Calculate residual component
    let mut residual = Array1::zeros(ts.len());
    for i in 0..ts.len() {
        if padded_trend[i].is_nan() {
            residual[i] = F::nan();
        } else {
            match model {
                DecompositionModel::Additive => {
                    residual[i] = ts[i] - padded_trend[i] - seasonal[i];
                }
                DecompositionModel::Multiplicative => {
                    if padded_trend[i] == F::zero() || seasonal[i] == F::zero() {
                        return Err(TimeSeriesError::DecompositionError(
                            "Division by zero in multiplicative model residual calculation"
                                .to_string(),
                        ));
                    }
                    residual[i] = ts[i] / (padded_trend[i] * seasonal[i]);
                }
            }
        }
    }

    // Create result struct
    let original = ts.clone();

    Ok(DecompositionResult {
        trend: padded_trend,
        seasonal,
        residual,
        original,
    })
}

/// Options for STL decomposition
#[derive(Debug, Clone)]
pub struct STLOptions {
    /// Trend window size (must be odd)
    pub trend_window: usize,
    /// Seasonal window size (must be odd)
    pub seasonal_window: usize,
    /// Number of inner loop iterations
    pub n_inner: usize,
    /// Number of outer loop iterations
    pub n_outer: usize,
    /// Whether to use robust weighting
    pub robust: bool,
}

impl Default for STLOptions {
    fn default() -> Self {
        Self {
            trend_window: 21,
            seasonal_window: 13,
            n_inner: 2,
            n_outer: 1,
            robust: false,
        }
    }
}

/// Performs STL (Seasonal and Trend decomposition using LOESS) on a time series
///
/// STL decomposition uses locally weighted regression (LOESS) to extract trend
/// and seasonal components.
///
/// # Arguments
///
/// * `ts` - The time series to decompose
/// * `period` - The seasonal period
/// * `options` - Options for STL decomposition
///
/// # Returns
///
/// * Decomposition result containing trend, seasonal, and residual components
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_series::decomposition::{stl_decomposition, STLOptions};
///
/// let ts = array![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0];
/// let options = STLOptions::default();
/// let result = stl_decomposition(&ts, 4, &options).unwrap();
/// println!("Trend: {:?}", result.trend);
/// println!("Seasonal: {:?}", result.seasonal);
/// println!("Residual: {:?}", result.residual);
/// ```
pub fn stl_decomposition<F>(
    ts: &Array1<F>,
    period: usize,
    options: &STLOptions,
) -> Result<DecompositionResult<F>>
where
    F: Float + FromPrimitive + Debug,
{
    if ts.len() < 2 * period {
        return Err(TimeSeriesError::DecompositionError(format!(
            "Time series length ({}) must be at least twice the seasonal period ({})",
            ts.len(),
            period
        )));
    }

    if options.trend_window % 2 == 0 || options.seasonal_window % 2 == 0 {
        return Err(TimeSeriesError::DecompositionError(
            "Trend and seasonal windows must be odd numbers".to_string(),
        ));
    }

    // A placeholder implementation
    // STL decomposition is complex and would require a detailed LOESS implementation
    // This simplified version returns the classical decomposition for now

    let decomp = decompose_seasonal(ts, period, DecompositionModel::Additive)?;

    Ok(decomp)
}

/// Performs exponential smoothing decomposition on a time series
///
/// # Arguments
///
/// * `ts` - The time series to decompose
/// * `period` - The seasonal period
/// * `alpha` - Level smoothing parameter (0 < alpha < 1)
/// * `beta` - Trend smoothing parameter (0 < beta < 1)
/// * `gamma` - Seasonal smoothing parameter (0 < gamma < 1)
/// * `model` - Decomposition model (additive or multiplicative)
///
/// # Returns
///
/// * Decomposition result containing level, trend, seasonal, and residual components
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_series::decomposition::{exponential_decomposition, DecompositionModel};
///
/// let ts = array![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0];
/// let result = exponential_decomposition(&ts, 4, 0.2, 0.1, 0.3,
///                                         DecompositionModel::Additive).unwrap();
/// println!("Trend: {:?}", result.trend);
/// println!("Seasonal: {:?}", result.seasonal);
/// println!("Residual: {:?}", result.residual);
/// ```
pub fn exponential_decomposition<F>(
    ts: &Array1<F>,
    period: usize,
    alpha: f64,
    beta: f64,
    gamma: f64,
    model: DecompositionModel,
) -> Result<DecompositionResult<F>>
where
    F: Float + FromPrimitive + Debug,
{
    if ts.len() < period + 1 {
        return Err(TimeSeriesError::DecompositionError(format!(
            "Time series length ({}) must be greater than the seasonal period ({})",
            ts.len(),
            period
        )));
    }

    if alpha <= 0.0 || alpha >= 1.0 || beta <= 0.0 || beta >= 1.0 || gamma <= 0.0 || gamma >= 1.0 {
        return Err(TimeSeriesError::InvalidInput(
            "Smoothing parameters must be between 0 and 1 (exclusive)".to_string(),
        ));
    }

    let alpha = F::from_f64(alpha).unwrap();
    let beta = F::from_f64(beta).unwrap();
    let gamma = F::from_f64(gamma).unwrap();

    let n = ts.len();

    // Initialize components
    let mut level = Array1::zeros(n + 1);
    let mut trend = Array1::zeros(n + 1);
    let mut seasonal = Array1::zeros(n + period);
    let mut residual = Array1::zeros(n);

    // Initialize level, trend, and seasonal components
    let initial_level = ts[0]; // Could also use average of first few observations
    level[0] = initial_level;

    // Initialize trend (average of first differences)
    if n > 1 {
        let mut sum = F::zero();
        for i in 1..min(n, 10) {
            sum = sum + (ts[i] - ts[i - 1]);
        }
        trend[0] = sum / F::from_usize(min(n - 1, 9)).unwrap();
    }

    // Initialize seasonal (average deviation from level for each season)
    for i in 0..min(period, n) {
        let pos = i % period;
        let expected = level[0] + F::from_usize(i).unwrap() * trend[0];
        match model {
            DecompositionModel::Additive => {
                seasonal[pos] = ts[i] - expected;
            }
            DecompositionModel::Multiplicative => {
                if expected == F::zero() {
                    return Err(TimeSeriesError::DecompositionError(
                        "Division by zero in multiplicative model initialization".to_string(),
                    ));
                }
                seasonal[pos] = ts[i] / expected;
            }
        }
    }

    // Normalize initial seasonal component
    match model {
        DecompositionModel::Additive => {
            let mean = seasonal
                .iter()
                .take(period)
                .fold(F::zero(), |acc, &x| acc + x)
                / F::from_usize(period).unwrap();
            for i in 0..period {
                seasonal[i] = seasonal[i] - mean;
            }
        }
        DecompositionModel::Multiplicative => {
            let mean = seasonal
                .iter()
                .take(period)
                .fold(F::zero(), |acc, &x| acc + x)
                / F::from_usize(period).unwrap();
            if mean == F::zero() {
                return Err(TimeSeriesError::DecompositionError(
                    "Division by zero normalizing multiplicative seasonal component".to_string(),
                ));
            }
            for i in 0..period {
                seasonal[i] = seasonal[i] / mean;
            }
        }
    }

    // Exponential smoothing recursion
    for i in 0..n {
        let s = i % period; // Current season
        let expected = match model {
            DecompositionModel::Additive => level[i] + trend[i],
            DecompositionModel::Multiplicative => level[i] * trend[i],
        };

        // Calculate residual
        match model {
            DecompositionModel::Additive => {
                residual[i] = ts[i] - expected - seasonal[s];
            }
            DecompositionModel::Multiplicative => {
                if expected == F::zero() || seasonal[s] == F::zero() {
                    residual[i] = F::zero(); // Avoid division by zero
                } else {
                    residual[i] = ts[i] / (expected * seasonal[s]);
                }
            }
        }

        // Update level, trend, and seasonal components
        match model {
            DecompositionModel::Additive => {
                level[i + 1] =
                    alpha * (ts[i] - seasonal[s]) + (F::one() - alpha) * (level[i] + trend[i]);
                trend[i + 1] = beta * (level[i + 1] - level[i]) + (F::one() - beta) * trend[i];
                seasonal[s + period] =
                    gamma * (ts[i] - level[i + 1]) + (F::one() - gamma) * seasonal[s];
            }
            DecompositionModel::Multiplicative => {
                if seasonal[s] == F::zero() {
                    return Err(TimeSeriesError::DecompositionError(
                        "Division by zero in multiplicative model update".to_string(),
                    ));
                }
                level[i + 1] =
                    alpha * (ts[i] / seasonal[s]) + (F::one() - alpha) * (level[i] * trend[i]);

                if level[i] == F::zero() {
                    trend[i + 1] = trend[i]; // Avoid division by zero
                } else {
                    trend[i + 1] = beta * (level[i + 1] / level[i]) + (F::one() - beta) * trend[i];
                }

                if level[i + 1] == F::zero() {
                    seasonal[s + period] = seasonal[s]; // Avoid division by zero
                } else {
                    seasonal[s + period] =
                        gamma * (ts[i] / level[i + 1]) + (F::one() - gamma) * seasonal[s];
                }
            }
        }
    }

    // Prepare final components
    let trend_component = Array1::from_iter(level.iter().take(n).cloned());
    let seasonal_component = Array1::from_iter((0..n).map(|i| seasonal[i % period]));
    let original = ts.clone();

    Ok(DecompositionResult {
        trend: trend_component,
        seasonal: seasonal_component,
        residual,
        original,
    })
}

// Helper function to get the minimum of two values
fn min<T: Ord>(a: T, b: T) -> T {
    if a < b {
        a
    } else {
        b
    }
}
