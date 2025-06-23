//! Calibration utilities for quantization
//!
//! This module provides utilities for finding optimal quantization parameters.
//! Quantization calibration is the process of determining the optimal scaling
//! factors and zero points for a given dataset and quantization method.
//!
//! The module includes:
//!
//! * Histogram-based methods for range calibration
//! * Entropy-based methods using KL divergence minimization
//! * Per-channel calibration strategies
//! * Dynamic calibration based on data statistics

use super::{
    calibration_ema::{
        calibrate_matrix_ema, calibrate_matrix_per_channel_ema, calibrate_vector_ema,
    },
    QuantizationMethod, QuantizationParams,
};
use crate::error::{LinalgError, LinalgResult};
use ndarray::{ArrayView1, ArrayView2};
use std::fmt::Debug;
// Using num_traits through qualified path in code

/// Calibration method for determining quantization parameters
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationMethod {
    /// Simple min-max calibration that uses the full range of the data
    MinMax,

    /// Moving average min-max that excludes outliers
    MovingAverageMinMax,

    /// Entropy-based calibration that minimizes the KL divergence
    EntropyCalibration,

    /// Percentile-based calibration that excludes outliers based on percentiles
    PercentileCalibration,

    /// Mean squared error minimization for finding optimal scale
    MSEOptimization,

    /// Exponential moving average for dynamic calibration
    ExponentialMovingAverage,
}

/// Configuration for quantization calibration
#[derive(Debug, Clone)]
pub struct CalibrationConfig {
    /// Method used for calibration
    pub method: CalibrationMethod,

    /// Number of histogram bins for entropy-based methods
    pub num_bins: usize,

    /// Percentile value for percentile-based methods (0.0 to 1.0)
    pub percentile: f32,

    /// Moving average window size for min-max methods
    pub window_size: usize,

    /// Whether to use per-channel calibration
    pub per_channel: bool,

    /// Whether to use symmetric quantization
    pub symmetric: bool,

    /// Smoothing factor for exponential moving average (0.0 to 1.0)
    /// Higher values give more weight to recent observations
    pub ema_factor: f32,

    /// Number of calibration iterations for dynamic methods
    pub max_iterations: usize,

    /// Convergence threshold for iterative calibration methods
    pub convergence_threshold: f32,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        CalibrationConfig {
            method: CalibrationMethod::MinMax,
            num_bins: 2048,
            percentile: 0.999,
            window_size: 10,
            per_channel: false,
            symmetric: true,
            ema_factor: 0.1,
            max_iterations: 10,
            convergence_threshold: 1e-6,
        }
    }
}

/// Calibrate quantization parameters for a matrix using the specified method
///
/// # Arguments
///
/// * `matrix` - Input matrix to calibrate
/// * `bits` - Bit width for quantization
/// * `config` - Calibration configuration
///
/// # Returns
///
/// * Calibrated quantization parameters
pub fn calibrate_matrix<F>(
    matrix: &ArrayView2<F>,
    bits: u8,
    config: &CalibrationConfig,
) -> LinalgResult<QuantizationParams>
where
    F: num_traits::Float + Debug + num_traits::AsPrimitive<f32> + num_traits::FromPrimitive,
    f32: num_traits::AsPrimitive<F>,
{
    match config.method {
        CalibrationMethod::MinMax => {
            if config.per_channel {
                calibrate_matrix_per_channel_minmax(matrix, bits, config.symmetric)
            } else {
                calibrate_matrix_minmax(matrix, bits, config.symmetric)
            }
        }
        CalibrationMethod::MovingAverageMinMax => {
            if config.per_channel {
                calibrate_matrix_per_channel_moving_average(
                    matrix,
                    bits,
                    config.window_size,
                    config.symmetric,
                )
            } else {
                calibrate_matrix_moving_average(matrix, bits, config.window_size, config.symmetric)
            }
        }
        CalibrationMethod::PercentileCalibration => {
            if config.per_channel {
                calibrate_matrix_per_channel_percentile(
                    matrix,
                    bits,
                    config.percentile,
                    config.symmetric,
                )
            } else {
                calibrate_matrix_percentile(matrix, bits, config.percentile, config.symmetric)
            }
        }
        CalibrationMethod::EntropyCalibration => {
            if config.per_channel {
                calibrate_matrix_per_channel_entropy(
                    matrix,
                    bits,
                    config.num_bins,
                    config.symmetric,
                )
            } else {
                calibrate_matrix_entropy(matrix, bits, config.num_bins, config.symmetric)
            }
        }
        CalibrationMethod::MSEOptimization => {
            if config.per_channel {
                calibrate_matrix_per_channel_mse(matrix, bits, config.symmetric)
            } else {
                calibrate_matrix_mse(matrix, bits, config.symmetric)
            }
        }
        CalibrationMethod::ExponentialMovingAverage => {
            if config.per_channel {
                calibrate_matrix_per_channel_ema(
                    matrix,
                    bits,
                    config.ema_factor,
                    config.max_iterations,
                    config.convergence_threshold,
                    config.symmetric,
                )
            } else {
                calibrate_matrix_ema(
                    matrix,
                    bits,
                    config.ema_factor,
                    config.max_iterations,
                    config.convergence_threshold,
                    config.symmetric,
                )
            }
        }
    }
}

/// Calibrate quantization parameters for a vector using the specified method
///
/// # Arguments
///
/// * `vector` - Input vector to calibrate
/// * `bits` - Bit width for quantization
/// * `config` - Calibration configuration
///
/// # Returns
///
/// * Calibrated quantization parameters
pub fn calibrate_vector<F>(
    vector: &ArrayView1<F>,
    bits: u8,
    config: &CalibrationConfig,
) -> LinalgResult<QuantizationParams>
where
    F: num_traits::Float + Debug + num_traits::AsPrimitive<f32> + num_traits::FromPrimitive,
    f32: num_traits::AsPrimitive<F>,
{
    // Modify configuration to disable per-channel for vectors
    let mut config = config.clone();
    config.per_channel = false;

    match config.method {
        CalibrationMethod::MinMax => calibrate_vector_minmax(vector, bits, config.symmetric),
        CalibrationMethod::MovingAverageMinMax => {
            calibrate_vector_moving_average(vector, bits, config.window_size, config.symmetric)
        }
        CalibrationMethod::PercentileCalibration => {
            calibrate_vector_percentile(vector, bits, config.percentile, config.symmetric)
        }
        CalibrationMethod::EntropyCalibration => {
            calibrate_vector_entropy(vector, bits, config.num_bins, config.symmetric)
        }
        CalibrationMethod::MSEOptimization => calibrate_vector_mse(vector, bits, config.symmetric),
        CalibrationMethod::ExponentialMovingAverage => calibrate_vector_ema(
            vector,
            bits,
            config.ema_factor,
            config.max_iterations,
            config.convergence_threshold,
            config.symmetric,
        ),
    }
}

/// Helper function to get recommended calibration configuration for neural network weights
///
/// Neural network weights typically benefit from symmetric quantization with
/// entropy-based or percentile-based calibration to handle outliers.
///
/// # Arguments
///
/// * `bits` - Bit width to use for quantization
/// * `aggressive` - Whether to use more aggressive quantization (percentile vs entropy)
///
/// # Returns
///
/// Calibration configuration optimized for neural network weights
pub fn get_weight_calibration_config(_bits: u8, aggressive: bool) -> CalibrationConfig {
    if aggressive {
        // More aggressive calibration - clips outliers more
        CalibrationConfig {
            method: CalibrationMethod::PercentileCalibration,
            symmetric: true,
            percentile: 0.99,  // Exclude 1% outliers on each tail
            per_channel: true, // Per-channel calibration often better for weights
            ..Default::default()
        }
    } else {
        // Default calibration - preserves more of the distribution
        CalibrationConfig {
            method: CalibrationMethod::EntropyCalibration,
            symmetric: true,
            num_bins: 2048, // Higher bin count for better precision
            per_channel: true,
            ..Default::default()
        }
    }
}

/// Helper function to get recommended calibration configuration for neural network activations
///
/// Activations typically benefit from asymmetric quantization, especially for
/// non-negative activation functions like ReLU.
///
/// # Arguments
///
/// * `bits` - Bit width to use for quantization
/// * `non_negative` - Whether the activations are known to be non-negative (e.g., from ReLU)
/// * `outlier_sensitive` - Whether the activations contain important outliers
///
/// # Returns
///
/// Calibration configuration optimized for neural network activations
pub fn get_activation_calibration_config(
    _bits: u8,
    non_negative: bool,
    outlier_sensitive: bool,
) -> CalibrationConfig {
    let mut config = if outlier_sensitive {
        // Outlier-sensitive activations benefit from MSE optimization
        CalibrationConfig {
            method: CalibrationMethod::MSEOptimization,
            num_bins: 1024,
            per_channel: false, // Activations usually don't need per-channel
            ..Default::default()
        }
    } else {
        // Standard activations benefit from percentile calibration to ignore outliers
        CalibrationConfig {
            method: CalibrationMethod::PercentileCalibration,
            percentile: 0.9995, // Keep more of the distribution than for weights
            per_channel: false,
            ..Default::default()
        }
    };

    // For activations like ReLU outputs, asymmetric is better
    // For activations with both positive and negative values, symmetric may be better
    config.symmetric = !non_negative;

    config
}

// -------------------------------------------------------------------------
// Matrix calibration implementations
// -------------------------------------------------------------------------

/// Simple min-max calibration for matrices
fn calibrate_matrix_minmax<F>(
    matrix: &ArrayView2<F>,
    bits: u8,
    symmetric: bool,
) -> LinalgResult<QuantizationParams>
where
    F: num_traits::Float + Debug + num_traits::AsPrimitive<f32> + num_traits::FromPrimitive,
    f32: num_traits::AsPrimitive<F>,
{
    // Find min and max values in the matrix
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;

    for &val in matrix.iter() {
        let val_f32 = val.as_();
        if val_f32.is_finite() {
            min_val = min_val.min(val_f32);
            max_val = max_val.max(val_f32);
        }
    }

    // Handle edge cases
    if !min_val.is_finite() || !max_val.is_finite() {
        return Err(LinalgError::ValueError(
            "Matrix contains non-finite values".to_string(),
        ));
    }

    if min_val == max_val {
        min_val -= 1.0;
        max_val += 1.0;
    }

    create_params_from_range(bits, min_val, max_val, symmetric)
}

/// Moving average min-max calibration for matrices
fn calibrate_matrix_moving_average<F>(
    matrix: &ArrayView2<F>,
    bits: u8,
    window_size: usize,
    symmetric: bool,
) -> LinalgResult<QuantizationParams>
where
    F: num_traits::Float + Debug + num_traits::AsPrimitive<f32> + num_traits::FromPrimitive,
    f32: num_traits::AsPrimitive<F>,
{
    // Convert matrix to a flattened vector of finite f32 values
    let mut values: Vec<f32> = matrix
        .iter()
        .filter_map(|&x| {
            let val = x.as_();
            if val.is_finite() {
                Some(val)
            } else {
                None
            }
        })
        .collect();

    if values.is_empty() {
        return Err(LinalgError::ValueError(
            "Matrix contains no finite values".to_string(),
        ));
    }

    // Sort values
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Calculate moving averages to find stable min/max
    if values.len() <= window_size {
        // Not enough data for moving average, fall back to min-max
        let min_val = *values.first().unwrap();
        let max_val = *values.last().unwrap();
        create_params_from_range(bits, min_val, max_val, symmetric)
    } else {
        // Calculate moving averages
        let min_val = values.iter().take(window_size).sum::<f32>() / window_size as f32;
        let max_val = values.iter().rev().take(window_size).sum::<f32>() / window_size as f32;

        create_params_from_range(bits, min_val, max_val, symmetric)
    }
}

/// Percentile-based calibration for matrices
fn calibrate_matrix_percentile<F>(
    matrix: &ArrayView2<F>,
    bits: u8,
    percentile: f32,
    symmetric: bool,
) -> LinalgResult<QuantizationParams>
where
    F: num_traits::Float + Debug + num_traits::AsPrimitive<f32> + num_traits::FromPrimitive,
    f32: num_traits::AsPrimitive<F>,
{
    if !(0.0..=1.0).contains(&percentile) {
        return Err(LinalgError::ValueError(
            "Percentile must be between 0.0 and 1.0".to_string(),
        ));
    }

    // Convert matrix to a flattened vector of finite f32 values
    let mut values: Vec<f32> = matrix
        .iter()
        .filter_map(|&x| {
            let val = x.as_();
            if val.is_finite() {
                Some(val)
            } else {
                None
            }
        })
        .collect();

    if values.is_empty() {
        return Err(LinalgError::ValueError(
            "Matrix contains no finite values".to_string(),
        ));
    }

    // Sort values
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Compute percentile indexes
    let low_idx = ((1.0 - percentile) * (values.len() as f32)).round() as usize;
    let high_idx = ((percentile) * (values.len() as f32)).round() as usize;

    // Get percentile values
    let min_val = values[low_idx.min(values.len() - 1)];
    let max_val = values[high_idx.min(values.len() - 1)];

    create_params_from_range(bits, min_val, max_val, symmetric)
}

/// Entropy-based calibration for matrices
fn calibrate_matrix_entropy<F>(
    matrix: &ArrayView2<F>,
    bits: u8,
    num_bins: usize,
    symmetric: bool,
) -> LinalgResult<QuantizationParams>
where
    F: num_traits::Float + Debug + num_traits::AsPrimitive<f32> + num_traits::FromPrimitive,
    f32: num_traits::AsPrimitive<F>,
{
    // First get min-max range to create histogram
    let (min_val, max_val) = find_min_max(matrix);

    // Create histogram of the data
    let histogram = create_histogram(matrix, min_val, max_val, num_bins);

    // Use KL divergence minimization to find optimal thresholds
    let (opt_min, opt_max) =
        optimize_thresholds_kl_divergence(&histogram, min_val, max_val, bits, symmetric);

    create_params_from_range(bits, opt_min, opt_max, symmetric)
}

/// MSE-based calibration for matrices
fn calibrate_matrix_mse<F>(
    matrix: &ArrayView2<F>,
    bits: u8,
    symmetric: bool,
) -> LinalgResult<QuantizationParams>
where
    F: num_traits::Float + Debug + num_traits::AsPrimitive<f32> + num_traits::FromPrimitive,
    f32: num_traits::AsPrimitive<F>,
{
    // Start with min-max calibration
    let mut base_params = calibrate_matrix_minmax(matrix, bits, symmetric)?;

    // Define a range of scale factors to try
    let scales = if symmetric {
        optimize_symmetric_scale(matrix, bits, base_params.scale)
    } else {
        let (scale, zero_point) =
            optimize_affine_params(matrix, bits, base_params.scale, base_params.zero_point);
        base_params.scale = scale;
        base_params.zero_point = zero_point;
        base_params.scale
    };

    // Create QuantizationParams with optimized scale
    let mut opt_params = base_params.clone();
    opt_params.scale = scales;

    Ok(opt_params)
}

// -------------------------------------------------------------------------
// Per-channel matrix calibration implementations
// -------------------------------------------------------------------------

/// Per-channel min-max calibration for matrices
fn calibrate_matrix_per_channel_minmax<F>(
    matrix: &ArrayView2<F>,
    bits: u8,
    symmetric: bool,
) -> LinalgResult<QuantizationParams>
where
    F: num_traits::Float + Debug + num_traits::AsPrimitive<f32> + num_traits::FromPrimitive,
    f32: num_traits::AsPrimitive<F>,
{
    let (_rows, cols) = matrix.dim();

    // Global min/max for the entire matrix
    let (global_min, global_max) = find_min_max(matrix);

    // Per-channel scales and zero points
    let mut channel_scales = Vec::with_capacity(cols);
    let mut channel_zero_points = Vec::with_capacity(if symmetric { 0 } else { cols });

    // For each channel (column)
    for col_idx in 0..cols {
        let column = matrix.column(col_idx);

        // Find min/max for this channel
        let mut col_min = f32::MAX;
        let mut col_max = f32::MIN;

        for &val in column.iter() {
            let val_f32 = val.as_();
            if val_f32.is_finite() {
                col_min = col_min.min(val_f32);
                col_max = col_max.max(val_f32);
            }
        }

        // Handle edge cases
        if !col_min.is_finite() || !col_max.is_finite() {
            col_min = 0.0;
            col_max = 1.0;
        }

        if col_min == col_max {
            col_min -= 1.0;
            col_max += 1.0;
        }

        // Calculate scale and zero point for this channel
        let (scale, zero_point) = if symmetric {
            let abs_max = col_max.abs().max(col_min.abs());
            let scale = abs_max / ((1 << (bits - 1)) - 1) as f32;
            (scale, 0)
        } else {
            let scale = (col_max - col_min) / ((1 << bits) - 1) as f32;
            let zero_point = (-col_min / scale).round() as i32;
            (scale, zero_point)
        };

        channel_scales.push(scale);
        if !symmetric {
            channel_zero_points.push(zero_point);
        }
    }

    // Create quantization params
    let q_method = if symmetric {
        QuantizationMethod::PerChannelSymmetric
    } else {
        QuantizationMethod::PerChannelAffine
    };

    Ok(QuantizationParams {
        bits,
        scale: 0.0,    // Not used for per-channel
        zero_point: 0, // Not used for per-channel symmetric
        min_val: global_min,
        max_val: global_max,
        method: q_method,
        data_type: determine_data_type(bits),
        channel_scales: Some(channel_scales),
        channel_zero_points: if symmetric {
            None
        } else {
            Some(channel_zero_points)
        },
    })
}

/// Per-channel moving average calibration for matrices
fn calibrate_matrix_per_channel_moving_average<F>(
    matrix: &ArrayView2<F>,
    bits: u8,
    window_size: usize,
    symmetric: bool,
) -> LinalgResult<QuantizationParams>
where
    F: num_traits::Float + Debug + num_traits::AsPrimitive<f32> + num_traits::FromPrimitive,
    f32: num_traits::AsPrimitive<F>,
{
    let (_rows, cols) = matrix.dim();

    // Global min/max for the entire matrix
    let (global_min, global_max) = find_min_max(matrix);

    // Per-channel scales and zero points
    let mut channel_scales = Vec::with_capacity(cols);
    let mut channel_zero_points = Vec::with_capacity(if symmetric { 0 } else { cols });

    // For each channel (column)
    for col_idx in 0..cols {
        let column = matrix.column(col_idx);

        // Convert column to a vector of finite f32 values
        let mut values: Vec<f32> = column
            .iter()
            .filter_map(|&x| {
                let val = x.as_();
                if val.is_finite() {
                    Some(val)
                } else {
                    None
                }
            })
            .collect();

        if values.is_empty() {
            // Handle empty or non-finite column
            channel_scales.push(1.0);
            if !symmetric {
                channel_zero_points.push(0);
            }
            continue;
        }

        // Sort values
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate moving averages to find stable min/max
        let (col_min, col_max) = if values.len() <= window_size {
            // Not enough data for moving average, fall back to min-max
            (*values.first().unwrap(), *values.last().unwrap())
        } else {
            // Calculate moving averages
            let min_val = values.iter().take(window_size).sum::<f32>() / window_size as f32;
            let max_val = values.iter().rev().take(window_size).sum::<f32>() / window_size as f32;
            (min_val, max_val)
        };

        // Calculate scale and zero point for this channel
        let (scale, zero_point) = if symmetric {
            let abs_max = col_max.abs().max(col_min.abs());
            let scale = abs_max / ((1 << (bits - 1)) - 1) as f32;
            (scale, 0)
        } else {
            let scale = (col_max - col_min) / ((1 << bits) - 1) as f32;
            let zero_point = (-col_min / scale).round() as i32;
            (scale, zero_point)
        };

        channel_scales.push(scale);
        if !symmetric {
            channel_zero_points.push(zero_point);
        }
    }

    // Create quantization params
    let q_method = if symmetric {
        QuantizationMethod::PerChannelSymmetric
    } else {
        QuantizationMethod::PerChannelAffine
    };

    Ok(QuantizationParams {
        bits,
        scale: 0.0,    // Not used for per-channel
        zero_point: 0, // Not used for per-channel symmetric
        min_val: global_min,
        max_val: global_max,
        method: q_method,
        data_type: determine_data_type(bits),
        channel_scales: Some(channel_scales),
        channel_zero_points: if symmetric {
            None
        } else {
            Some(channel_zero_points)
        },
    })
}

/// Per-channel percentile calibration for matrices
fn calibrate_matrix_per_channel_percentile<F>(
    matrix: &ArrayView2<F>,
    bits: u8,
    percentile: f32,
    symmetric: bool,
) -> LinalgResult<QuantizationParams>
where
    F: num_traits::Float + Debug + num_traits::AsPrimitive<f32> + num_traits::FromPrimitive,
    f32: num_traits::AsPrimitive<F>,
{
    if !(0.0..=1.0).contains(&percentile) {
        return Err(LinalgError::ValueError(
            "Percentile must be between 0.0 and 1.0".to_string(),
        ));
    }

    let (_rows, cols) = matrix.dim();

    // Global min/max for the entire matrix
    let (global_min, global_max) = find_min_max(matrix);

    // Per-channel scales and zero points
    let mut channel_scales = Vec::with_capacity(cols);
    let mut channel_zero_points = Vec::with_capacity(if symmetric { 0 } else { cols });

    // For each channel (column)
    for col_idx in 0..cols {
        let column = matrix.column(col_idx);

        // Convert column to a vector of finite f32 values
        let mut values: Vec<f32> = column
            .iter()
            .filter_map(|&x| {
                let val = x.as_();
                if val.is_finite() {
                    Some(val)
                } else {
                    None
                }
            })
            .collect();

        if values.is_empty() {
            // Handle empty or non-finite column
            channel_scales.push(1.0);
            if !symmetric {
                channel_zero_points.push(0);
            }
            continue;
        }

        // Sort values
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Compute percentile indexes
        let low_idx = ((1.0 - percentile) * (values.len() as f32)) as usize;
        let high_idx = ((percentile) * (values.len() as f32)) as usize;

        // Get percentile values
        let col_min = values[low_idx.min(values.len() - 1)];
        let col_max = values[high_idx.min(values.len() - 1)];

        // Calculate scale and zero point for this channel
        let (scale, zero_point) = if symmetric {
            let abs_max = col_max.abs().max(col_min.abs());
            let scale = abs_max / ((1 << (bits - 1)) - 1) as f32;
            (scale, 0)
        } else {
            let scale = (col_max - col_min) / ((1 << bits) - 1) as f32;
            let zero_point = (-col_min / scale).round() as i32;
            (scale, zero_point)
        };

        channel_scales.push(scale);
        if !symmetric {
            channel_zero_points.push(zero_point);
        }
    }

    // Create quantization params
    let q_method = if symmetric {
        QuantizationMethod::PerChannelSymmetric
    } else {
        QuantizationMethod::PerChannelAffine
    };

    Ok(QuantizationParams {
        bits,
        scale: 0.0,    // Not used for per-channel
        zero_point: 0, // Not used for per-channel symmetric
        min_val: global_min,
        max_val: global_max,
        method: q_method,
        data_type: determine_data_type(bits),
        channel_scales: Some(channel_scales),
        channel_zero_points: if symmetric {
            None
        } else {
            Some(channel_zero_points)
        },
    })
}

/// Per-channel entropy calibration for matrices
fn calibrate_matrix_per_channel_entropy<F>(
    matrix: &ArrayView2<F>,
    bits: u8,
    num_bins: usize,
    symmetric: bool,
) -> LinalgResult<QuantizationParams>
where
    F: num_traits::Float + Debug + num_traits::AsPrimitive<f32> + num_traits::FromPrimitive,
    f32: num_traits::AsPrimitive<F>,
{
    let (_rows, cols) = matrix.dim();

    // Global min/max for the entire matrix
    let (global_min, global_max) = find_min_max(matrix);

    // Per-channel scales and zero points
    let mut channel_scales = Vec::with_capacity(cols);
    let mut channel_zero_points = Vec::with_capacity(if symmetric { 0 } else { cols });

    // For each channel (column)
    for col_idx in 0..cols {
        let column = matrix.column(col_idx);

        // Find min/max for this channel
        let (col_min, col_max) = find_min_max_vec(&column);

        // Create histogram of the channel data
        let histogram = create_histogram_vec(&column, col_min, col_max, num_bins);

        // Use KL divergence minimization to find optimal thresholds
        let (opt_min, opt_max) =
            optimize_thresholds_kl_divergence(&histogram, col_min, col_max, bits, symmetric);

        // Calculate scale and zero point for this channel
        let (scale, zero_point) = if symmetric {
            let abs_max = opt_max.abs().max(opt_min.abs());
            let scale = abs_max / ((1 << (bits - 1)) - 1) as f32;
            (scale, 0)
        } else {
            let scale = (opt_max - opt_min) / ((1 << bits) - 1) as f32;
            let zero_point = (-opt_min / scale).round() as i32;
            (scale, zero_point)
        };

        channel_scales.push(scale);
        if !symmetric {
            channel_zero_points.push(zero_point);
        }
    }

    // Create quantization params
    let q_method = if symmetric {
        QuantizationMethod::PerChannelSymmetric
    } else {
        QuantizationMethod::PerChannelAffine
    };

    Ok(QuantizationParams {
        bits,
        scale: 0.0,    // Not used for per-channel
        zero_point: 0, // Not used for per-channel symmetric
        min_val: global_min,
        max_val: global_max,
        method: q_method,
        data_type: determine_data_type(bits),
        channel_scales: Some(channel_scales),
        channel_zero_points: if symmetric {
            None
        } else {
            Some(channel_zero_points)
        },
    })
}

/// Per-channel MSE calibration for matrices
fn calibrate_matrix_per_channel_mse<F>(
    matrix: &ArrayView2<F>,
    bits: u8,
    symmetric: bool,
) -> LinalgResult<QuantizationParams>
where
    F: num_traits::Float + Debug + num_traits::AsPrimitive<f32> + num_traits::FromPrimitive,
    f32: num_traits::AsPrimitive<F>,
{
    let (_rows, cols) = matrix.dim();

    // Global min/max for the entire matrix
    let (global_min, global_max) = find_min_max(matrix);

    // Per-channel scales and zero points
    let mut channel_scales = Vec::with_capacity(cols);
    let mut channel_zero_points = Vec::with_capacity(if symmetric { 0 } else { cols });

    // For each channel (column)
    for col_idx in 0..cols {
        let column = matrix.column(col_idx);

        // Start with min-max calibration for this channel
        let (col_min, col_max) = find_min_max_vec(&column);

        let base_scale = if symmetric {
            let abs_max = col_max.abs().max(col_min.abs());
            abs_max / ((1 << (bits - 1)) - 1) as f32
        } else {
            (col_max - col_min) / ((1 << bits) - 1) as f32
        };

        let base_zero_point = if symmetric {
            0
        } else {
            (-col_min / base_scale).round() as i32
        };

        // Optimize parameters for this channel
        if symmetric {
            let scale = optimize_symmetric_scale_vec(&column, bits, base_scale);
            channel_scales.push(scale);
        } else {
            let (scale, zero_point) =
                optimize_affine_params_vec(&column, bits, base_scale, base_zero_point);
            channel_scales.push(scale);
            channel_zero_points.push(zero_point);
        }
    }

    // Create quantization params
    let q_method = if symmetric {
        QuantizationMethod::PerChannelSymmetric
    } else {
        QuantizationMethod::PerChannelAffine
    };

    Ok(QuantizationParams {
        bits,
        scale: 0.0,    // Not used for per-channel
        zero_point: 0, // Not used for per-channel symmetric
        min_val: global_min,
        max_val: global_max,
        method: q_method,
        data_type: determine_data_type(bits),
        channel_scales: Some(channel_scales),
        channel_zero_points: if symmetric {
            None
        } else {
            Some(channel_zero_points)
        },
    })
}

// -------------------------------------------------------------------------
// Vector calibration implementations
// -------------------------------------------------------------------------

/// Simple min-max calibration for vectors
fn calibrate_vector_minmax<F>(
    vector: &ArrayView1<F>,
    bits: u8,
    symmetric: bool,
) -> LinalgResult<QuantizationParams>
where
    F: num_traits::Float + Debug + num_traits::AsPrimitive<f32> + num_traits::FromPrimitive,
    f32: num_traits::AsPrimitive<F>,
{
    // Find min and max values in the vector
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;

    for &val in vector.iter() {
        let val_f32 = val.as_();
        if val_f32.is_finite() {
            min_val = min_val.min(val_f32);
            max_val = max_val.max(val_f32);
        }
    }

    // Handle edge cases
    if !min_val.is_finite() || !max_val.is_finite() {
        return Err(LinalgError::ValueError(
            "Vector contains non-finite values".to_string(),
        ));
    }

    if min_val == max_val {
        min_val -= 1.0;
        max_val += 1.0;
    }

    create_params_from_range(bits, min_val, max_val, symmetric)
}

/// Moving average min-max calibration for vectors
fn calibrate_vector_moving_average<F>(
    vector: &ArrayView1<F>,
    bits: u8,
    window_size: usize,
    symmetric: bool,
) -> LinalgResult<QuantizationParams>
where
    F: num_traits::Float + Debug + num_traits::AsPrimitive<f32> + num_traits::FromPrimitive,
    f32: num_traits::AsPrimitive<F>,
{
    // Convert vector to a flattened vector of finite f32 values
    let mut values: Vec<f32> = vector
        .iter()
        .filter_map(|&x| {
            let val = x.as_();
            if val.is_finite() {
                Some(val)
            } else {
                None
            }
        })
        .collect();

    if values.is_empty() {
        return Err(LinalgError::ValueError(
            "Vector contains no finite values".to_string(),
        ));
    }

    // Sort values
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Calculate moving averages to find stable min/max
    if values.len() <= window_size {
        // Not enough data for moving average, fall back to min-max
        let min_val = *values.first().unwrap();
        let max_val = *values.last().unwrap();
        create_params_from_range(bits, min_val, max_val, symmetric)
    } else {
        // Calculate moving averages
        let min_val = values.iter().take(window_size).sum::<f32>() / window_size as f32;
        let max_val = values.iter().rev().take(window_size).sum::<f32>() / window_size as f32;

        create_params_from_range(bits, min_val, max_val, symmetric)
    }
}

/// Percentile-based calibration for vectors
fn calibrate_vector_percentile<F>(
    vector: &ArrayView1<F>,
    bits: u8,
    percentile: f32,
    symmetric: bool,
) -> LinalgResult<QuantizationParams>
where
    F: num_traits::Float + Debug + num_traits::AsPrimitive<f32> + num_traits::FromPrimitive,
    f32: num_traits::AsPrimitive<F>,
{
    if !(0.0..=1.0).contains(&percentile) {
        return Err(LinalgError::ValueError(
            "Percentile must be between 0.0 and 1.0".to_string(),
        ));
    }

    // Convert vector to a flattened vector of finite f32 values
    let mut values: Vec<f32> = vector
        .iter()
        .filter_map(|&x| {
            let val = x.as_();
            if val.is_finite() {
                Some(val)
            } else {
                None
            }
        })
        .collect();

    if values.is_empty() {
        return Err(LinalgError::ValueError(
            "Vector contains no finite values".to_string(),
        ));
    }

    // Sort values
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Compute percentile indexes
    let low_idx = ((1.0 - percentile) * (values.len() as f32)).round() as usize;
    let high_idx = ((percentile) * (values.len() as f32)).round() as usize;

    // Get percentile values
    let min_val = values[low_idx.min(values.len() - 1)];
    let max_val = values[high_idx.min(values.len() - 1)];

    create_params_from_range(bits, min_val, max_val, symmetric)
}

/// Entropy-based calibration for vectors
fn calibrate_vector_entropy<F>(
    vector: &ArrayView1<F>,
    bits: u8,
    num_bins: usize,
    symmetric: bool,
) -> LinalgResult<QuantizationParams>
where
    F: num_traits::Float + Debug + num_traits::AsPrimitive<f32> + num_traits::FromPrimitive,
    f32: num_traits::AsPrimitive<F>,
{
    // First get min-max range to create histogram
    let (min_val, max_val) = find_min_max_vec(vector);

    // Create histogram of the data
    let histogram = create_histogram_vec(vector, min_val, max_val, num_bins);

    // Use KL divergence minimization to find optimal thresholds
    let (opt_min, opt_max) =
        optimize_thresholds_kl_divergence(&histogram, min_val, max_val, bits, symmetric);

    create_params_from_range(bits, opt_min, opt_max, symmetric)
}

/// MSE-based calibration for vectors
fn calibrate_vector_mse<F>(
    vector: &ArrayView1<F>,
    bits: u8,
    symmetric: bool,
) -> LinalgResult<QuantizationParams>
where
    F: num_traits::Float + Debug + num_traits::AsPrimitive<f32> + num_traits::FromPrimitive,
    f32: num_traits::AsPrimitive<F>,
{
    // Start with min-max calibration
    let mut base_params = calibrate_vector_minmax(vector, bits, symmetric)?;

    // Define a range of scale factors to try
    let scales = if symmetric {
        optimize_symmetric_scale_vec(vector, bits, base_params.scale)
    } else {
        let (scale, zero_point) =
            optimize_affine_params_vec(vector, bits, base_params.scale, base_params.zero_point);
        base_params.scale = scale;
        base_params.zero_point = zero_point;
        base_params.scale
    };

    // Create QuantizationParams with optimized scale
    let mut opt_params = base_params.clone();
    opt_params.scale = scales;

    Ok(opt_params)
}

// -------------------------------------------------------------------------
// Helper functions
// -------------------------------------------------------------------------

/// Find the minimum and maximum values in a matrix
pub fn find_min_max<F>(matrix: &ArrayView2<F>) -> (f32, f32)
where
    F: num_traits::Float + num_traits::AsPrimitive<f32>,
{
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;

    for &val in matrix.iter() {
        let val_f32 = val.as_();
        if val_f32.is_finite() {
            min_val = min_val.min(val_f32);
            max_val = max_val.max(val_f32);
        }
    }

    // Handle edge cases
    if !min_val.is_finite() || !max_val.is_finite() {
        min_val = 0.0;
        max_val = 1.0;
    }

    if min_val == max_val {
        min_val -= 1.0;
        max_val += 1.0;
    }

    (min_val, max_val)
}

/// Find the minimum and maximum values in a vector
pub fn find_min_max_vec<F>(vector: &ArrayView1<F>) -> (f32, f32)
where
    F: num_traits::Float + num_traits::AsPrimitive<f32>,
{
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;

    for &val in vector.iter() {
        let val_f32 = val.as_();
        if val_f32.is_finite() {
            min_val = min_val.min(val_f32);
            max_val = max_val.max(val_f32);
        }
    }

    // Handle edge cases
    if !min_val.is_finite() || !max_val.is_finite() {
        min_val = 0.0;
        max_val = 1.0;
    }

    if min_val == max_val {
        min_val -= 1.0;
        max_val += 1.0;
    }

    (min_val, max_val)
}

/// Create a histogram of values from a matrix
fn create_histogram<F>(
    matrix: &ArrayView2<F>,
    min_val: f32,
    max_val: f32,
    num_bins: usize,
) -> Vec<usize>
where
    F: num_traits::Float + num_traits::AsPrimitive<f32>,
{
    let mut histogram = vec![0; num_bins];
    let bin_width = (max_val - min_val) / num_bins as f32;

    if bin_width == 0.0 {
        // All values are the same, put them all in the middle bin
        histogram[num_bins / 2] = matrix.len();
        return histogram;
    }

    for &val in matrix.iter() {
        let val_f32 = val.as_();
        if val_f32.is_finite() {
            let bin_idx = ((val_f32 - min_val) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(num_bins - 1); // Ensure we don't go out of bounds
            histogram[bin_idx] += 1;
        }
    }

    histogram
}

/// Create a histogram of values from a vector
fn create_histogram_vec<F>(
    vector: &ArrayView1<F>,
    min_val: f32,
    max_val: f32,
    num_bins: usize,
) -> Vec<usize>
where
    F: num_traits::Float + num_traits::AsPrimitive<f32>,
{
    let mut histogram = vec![0; num_bins];
    let bin_width = (max_val - min_val) / num_bins as f32;

    if bin_width == 0.0 {
        // All values are the same, put them all in the middle bin
        histogram[num_bins / 2] = vector.len();
        return histogram;
    }

    for &val in vector.iter() {
        let val_f32 = val.as_();
        if val_f32.is_finite() {
            let bin_idx = ((val_f32 - min_val) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(num_bins - 1); // Ensure we don't go out of bounds
            histogram[bin_idx] += 1;
        }
    }

    histogram
}

/// Optimize thresholds using KL divergence
fn optimize_thresholds_kl_divergence(
    histogram: &[usize],
    min_val: f32,
    max_val: f32,
    bits: u8,
    symmetric: bool,
) -> (f32, f32) {
    let num_bins = histogram.len();
    let bin_width = (max_val - min_val) / num_bins as f32;

    // Convert histogram to probability distribution
    let total_count = histogram.iter().sum::<usize>() as f32;
    let distribution: Vec<f32> = histogram
        .iter()
        .map(|&count| count as f32 / total_count)
        .collect();

    // Number of quantization levels
    let levels = if symmetric {
        (1 << (bits - 1)) as usize // For signed integers
    } else {
        (1 << bits) as usize // For unsigned integers
    };

    // For symmetric quantization, we want to find the optimal abs_max
    if symmetric {
        // Search for the optimal abs_max that minimizes KL divergence
        let mut best_abs_max = max_val.abs().max(min_val.abs());
        let mut min_kl = f32::MAX;

        // Try different abs_max values
        let step = (best_abs_max / 20.0).max(1e-6);
        for i in 0..40 {
            let abs_max = best_abs_max - 20.0 * step + i as f32 * step;
            if abs_max <= 0.0 {
                continue;
            }

            // Calculate quantization step
            let quantization_step = abs_max / (levels - 1) as f32;

            // Calculate KL divergence for this abs_max
            let kl = calculate_kl_divergence_symmetric(
                &distribution,
                min_val,
                max_val,
                bin_width,
                abs_max,
                quantization_step,
            );

            if kl < min_kl {
                min_kl = kl;
                best_abs_max = abs_max;
            }
        }

        // Return symmetric range
        (-best_abs_max, best_abs_max)
    } else {
        // For asymmetric quantization, find the best min and max
        let mut best_min = min_val;
        let mut best_max = max_val;
        let mut min_kl = f32::MAX;

        // Grid search for optimal min/max
        let min_step = (max_val - min_val) / 40.0;
        let max_step = min_step;

        for i in 0..10 {
            let trial_min = min_val + i as f32 * min_step;

            for j in 0..10 {
                let trial_max = max_val - j as f32 * max_step;

                if trial_min >= trial_max {
                    continue;
                }

                // Calculate quantization step
                let quantization_step = (trial_max - trial_min) / (levels - 1) as f32;

                // Calculate KL divergence for this range
                let kl = calculate_kl_divergence_asymmetric(
                    &distribution,
                    min_val,
                    max_val,
                    bin_width,
                    trial_min,
                    trial_max,
                    quantization_step,
                );

                if kl < min_kl {
                    min_kl = kl;
                    best_min = trial_min;
                    best_max = trial_max;
                }
            }
        }

        (best_min, best_max)
    }
}

/// Calculate KL divergence for symmetric quantization
fn calculate_kl_divergence_symmetric(
    distribution: &[f32],
    min_val: f32,
    _max_val: f32,
    bin_width: f32,
    abs_max: f32,
    quantization_step: f32,
) -> f32 {
    let num_bins = distribution.len();

    // Create quantized probability distribution
    let mut quantized_dist = vec![0.0; num_bins];

    for (bin_idx, &prob) in distribution.iter().enumerate() {
        // Original value at the center of this bin
        let orig_val = min_val + (bin_idx as f32 + 0.5) * bin_width;

        // Quantize the value
        let quantized_val = if orig_val > abs_max {
            abs_max
        } else if orig_val < -abs_max {
            -abs_max
        } else {
            // Round to nearest quantization step
            (orig_val / quantization_step).round() * quantization_step
        };

        // Map back to bin index
        let new_bin_idx = ((quantized_val - min_val) / bin_width).floor() as i32;

        if new_bin_idx >= 0 && new_bin_idx < num_bins as i32 {
            quantized_dist[new_bin_idx as usize] += prob;
        }
    }

    // Calculate KL divergence: sum(p * log(p / q))
    let mut kl = 0.0;
    for (i, &p) in distribution.iter().enumerate() {
        if p > 0.0 {
            let q = quantized_dist[i].max(1e-10); // Avoid division by zero
            kl += p * (p / q).ln();
        }
    }

    kl
}

/// Calculate KL divergence for asymmetric quantization
fn calculate_kl_divergence_asymmetric(
    distribution: &[f32],
    min_val: f32,
    _max_val: f32,
    bin_width: f32,
    quant_min: f32,
    quant_max: f32,
    quantization_step: f32,
) -> f32 {
    let num_bins = distribution.len();

    // Create quantized probability distribution
    let mut quantized_dist = vec![0.0; num_bins];

    for (bin_idx, &prob) in distribution.iter().enumerate() {
        // Original value at the center of this bin
        let orig_val = min_val + (bin_idx as f32 + 0.5) * bin_width;

        // Quantize the value
        let quantized_val = if orig_val > quant_max {
            quant_max
        } else if orig_val < quant_min {
            quant_min
        } else {
            // Round to nearest quantization step
            let steps = ((orig_val - quant_min) / quantization_step).round();
            quant_min + steps * quantization_step
        };

        // Map back to bin index
        let new_bin_idx = ((quantized_val - min_val) / bin_width).floor() as i32;

        if new_bin_idx >= 0 && new_bin_idx < num_bins as i32 {
            quantized_dist[new_bin_idx as usize] += prob;
        }
    }

    // Calculate KL divergence: sum(p * log(p / q))
    let mut kl = 0.0;
    for (i, &p) in distribution.iter().enumerate() {
        if p > 0.0 {
            let q = quantized_dist[i].max(1e-10); // Avoid division by zero
            kl += p * (p / q).ln();
        }
    }

    kl
}

/// Optimize symmetric scale factor using MSE
fn optimize_symmetric_scale<F>(matrix: &ArrayView2<F>, bits: u8, base_scale: f32) -> f32
where
    F: num_traits::Float + Debug + num_traits::AsPrimitive<f32> + num_traits::FromPrimitive,
    f32: num_traits::AsPrimitive<F>,
{
    let num_trials = 20;
    let scales: Vec<f32> = (0..num_trials)
        .map(|i| {
            let factor = 0.5 + 1.5 * (i as f32 / (num_trials - 1) as f32);
            base_scale * factor
        })
        .collect();

    let mut best_scale = base_scale;
    let mut min_mse = f32::MAX;

    // Test each scale factor
    for &scale in &scales {
        // Create temporary quantization parameters
        let abs_max = matrix
            .mapv(|x| x.as_().abs())
            .fold(0.0, |a: f32, &b| a.max(b));
        let params = QuantizationParams {
            bits,
            scale,
            zero_point: 0,
            min_val: -abs_max,
            max_val: abs_max,
            method: if bits == 4 {
                QuantizationMethod::Int4
            } else {
                QuantizationMethod::Symmetric
            },
            data_type: determine_data_type(bits),
            channel_scales: None,
            channel_zero_points: None,
        };

        // Manually simulate quantization and dequantization for F type
        let matrix_f32 = matrix.mapv(|x| x.as_());
        let scale = params.scale;
        let dequantized = matrix_f32.mapv(|x| {
            let quantized = (x / scale)
                .round()
                .clamp(-(1 << (bits - 1)) as f32, ((1 << (bits - 1)) - 1) as f32);
            quantized * scale
        });

        // Calculate MSE
        let mse = (&matrix_f32 - &dequantized).mapv(|x| x * x).sum() / matrix.len() as f32;

        if mse < min_mse {
            min_mse = mse;
            best_scale = scale;
        }
    }

    best_scale
}

/// Optimize symmetric scale factor for vectors using MSE
fn optimize_symmetric_scale_vec<F>(vector: &ArrayView1<F>, bits: u8, base_scale: f32) -> f32
where
    F: num_traits::Float + Debug + num_traits::AsPrimitive<f32> + num_traits::FromPrimitive,
    f32: num_traits::AsPrimitive<F>,
{
    let num_trials = 20;
    let scales: Vec<f32> = (0..num_trials)
        .map(|i| {
            let factor = 0.5 + 1.5 * (i as f32 / (num_trials - 1) as f32);
            base_scale * factor
        })
        .collect();

    let mut best_scale = base_scale;
    let mut min_mse = f32::MAX;

    // Test each scale factor
    for &scale in &scales {
        // Create temporary QuantizationParams
        let abs_max = vector
            .mapv(|x| x.as_().abs())
            .fold(0.0, |a: f32, &b| a.max(b));
        let params = QuantizationParams {
            bits,
            scale,
            zero_point: 0,
            min_val: -abs_max,
            max_val: abs_max,
            method: if bits == 4 {
                QuantizationMethod::Int4
            } else {
                QuantizationMethod::Symmetric
            },
            data_type: determine_data_type(bits),
            channel_scales: None,
            channel_zero_points: None,
        };

        // Manually simulate quantization and dequantization for F type
        let vector_f32 = vector.mapv(|x| x.as_());
        let scale = params.scale;
        let dequantized = vector_f32.mapv(|x| {
            let quantized = (x / scale)
                .round()
                .clamp(-(1 << (bits - 1)) as f32, ((1 << (bits - 1)) - 1) as f32);
            quantized * scale
        });

        // Calculate MSE
        let mse = (&vector_f32 - &dequantized).mapv(|x| x * x).sum() / vector.len() as f32;

        if mse < min_mse {
            min_mse = mse;
            best_scale = scale;
        }
    }

    best_scale
}

/// Optimize affine quantization parameters (scale and zero point) using MSE
fn optimize_affine_params<F>(
    matrix: &ArrayView2<F>,
    bits: u8,
    base_scale: f32,
    base_zero_point: i32,
) -> (f32, i32)
where
    F: num_traits::Float + Debug + num_traits::AsPrimitive<f32> + num_traits::FromPrimitive,
    f32: num_traits::AsPrimitive<F>,
{
    let num_scale_trials = 10;
    let num_zp_trials = 5;

    let scales: Vec<f32> = (0..num_scale_trials)
        .map(|i| {
            let factor = 0.8 + 0.4 * (i as f32 / (num_scale_trials - 1) as f32);
            base_scale * factor
        })
        .collect();

    let zero_points: Vec<i32> = (0..num_zp_trials)
        .map(|i| {
            let offset = -2 + i;
            base_zero_point + offset
        })
        .collect();

    let mut best_scale = base_scale;
    let mut best_zero_point = base_zero_point;
    let mut min_mse = f32::MAX;

    // Test each combination of scale and zero point
    for &scale in &scales {
        for &zero_point in &zero_points {
            // Create temporary QuantizationParams
            let mut params = QuantizationParams {
                bits,
                scale,
                zero_point,
                min_val: 0.0, // Will be set by quantize_matrix
                max_val: 0.0, // Will be set by quantize_matrix
                method: QuantizationMethod::Affine,
                data_type: determine_data_type(bits),
                channel_scales: None,
                channel_zero_points: None,
            };

            // Manually simulate affine quantization and dequantization for F type
            let matrix_f32 = matrix.mapv(|x| x.as_());
            let scale = params.scale;
            let zero_point = params.zero_point;

            // Find min/max values for the matrix
            let mut min_val = f32::MAX;
            let mut max_val = f32::MIN;
            for &val in matrix_f32.iter() {
                if val.is_finite() {
                    min_val = min_val.min(val);
                    max_val = max_val.max(val);
                }
            }
            params.min_val = min_val;
            params.max_val = max_val;

            let dequantized = matrix_f32.mapv(|x| {
                let quantized = ((x / scale) + zero_point as f32)
                    .round()
                    .clamp(0.0, ((1 << bits) - 1) as f32);
                (quantized - zero_point as f32) * scale
            });

            // Calculate MSE
            let mse = (&matrix_f32 - &dequantized).mapv(|x| x * x).sum() / matrix.len() as f32;

            if mse < min_mse {
                min_mse = mse;
                best_scale = scale;
                best_zero_point = zero_point;
            }
        }
    }

    (best_scale, best_zero_point)
}

/// Optimize affine quantization parameters for vectors using MSE
fn optimize_affine_params_vec<F>(
    vector: &ArrayView1<F>,
    bits: u8,
    base_scale: f32,
    base_zero_point: i32,
) -> (f32, i32)
where
    F: num_traits::Float + Debug + num_traits::AsPrimitive<f32> + num_traits::FromPrimitive,
    f32: num_traits::AsPrimitive<F>,
{
    let num_scale_trials = 10;
    let num_zp_trials = 5;

    let scales: Vec<f32> = (0..num_scale_trials)
        .map(|i| {
            let factor = 0.8 + 0.4 * (i as f32 / (num_scale_trials - 1) as f32);
            base_scale * factor
        })
        .collect();

    let zero_points: Vec<i32> = (0..num_zp_trials)
        .map(|i| {
            let offset = -2 + i;
            base_zero_point + offset
        })
        .collect();

    let mut best_scale = base_scale;
    let mut best_zero_point = base_zero_point;
    let mut min_mse = f32::MAX;

    // Test each combination of scale and zero point
    for &scale in &scales {
        for &zero_point in &zero_points {
            // Create temporary QuantizationParams
            let mut params = QuantizationParams {
                bits,
                scale,
                zero_point,
                min_val: 0.0, // Will be set by quantize_vector
                max_val: 0.0, // Will be set by quantize_vector
                method: QuantizationMethod::Affine,
                data_type: determine_data_type(bits),
                channel_scales: None,
                channel_zero_points: None,
            };

            // Manually simulate affine quantization and dequantization for F type
            let vector_f32 = vector.mapv(|x| x.as_());
            let scale = params.scale;
            let zero_point = params.zero_point;

            // Find min/max values for the vector
            let mut min_val = f32::MAX;
            let mut max_val = f32::MIN;
            for &val in vector_f32.iter() {
                if val.is_finite() {
                    min_val = min_val.min(val);
                    max_val = max_val.max(val);
                }
            }
            params.min_val = min_val;
            params.max_val = max_val;

            let dequantized = vector_f32.mapv(|x| {
                let quantized = ((x / scale) + zero_point as f32)
                    .round()
                    .clamp(0.0, ((1 << bits) - 1) as f32);
                (quantized - zero_point as f32) * scale
            });

            // Calculate MSE
            let mse = (&vector_f32 - &dequantized).mapv(|x| x * x).sum() / vector.len() as f32;

            if mse < min_mse {
                min_mse = mse;
                best_scale = scale;
                best_zero_point = zero_point;
            }
        }
    }

    (best_scale, best_zero_point)
}

/// Create QuantizationParams from a min-max range
pub fn create_params_from_range(
    bits: u8,
    min_val: f32,
    max_val: f32,
    symmetric: bool,
) -> LinalgResult<QuantizationParams> {
    let (method, scale, zero_point) = if symmetric {
        let abs_max = max_val.abs().max(min_val.abs());
        let scale = abs_max / ((1 << (bits - 1)) - 1) as f32;
        (QuantizationMethod::Symmetric, scale, 0)
    } else {
        let method = QuantizationMethod::Affine;
        let scale = (max_val - min_val) / ((1 << bits) - 1) as f32;
        let zero_point = (-min_val / scale).round() as i32;
        (method, scale, zero_point)
    };

    Ok(QuantizationParams {
        bits,
        scale,
        zero_point,
        min_val,
        max_val,
        method,
        data_type: determine_data_type(bits),
        channel_scales: None,
        channel_zero_points: None,
    })
}

/// Determine the appropriate data type based on bit width
pub fn determine_data_type(bits: u8) -> super::QuantizedDataType {
    use super::QuantizedDataType;

    match bits {
        4 => QuantizedDataType::Int4,     // Default to Int4 for 4-bit
        8 => QuantizedDataType::Int8,     // Default to Int8 for 8-bit
        16 => QuantizedDataType::Float16, // Default to Float16 for 16-bit
        _ => QuantizedDataType::Int8,     // Default to Int8 for other cases
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantization::{dequantize_matrix, quantize_matrix};
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_calibrate_matrix_minmax() {
        let matrix = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        // Symmetric quantization
        let params = calibrate_matrix_minmax(&matrix.view(), 8, true).unwrap();
        assert_eq!(params.method, QuantizationMethod::Symmetric);
        assert_eq!(params.bits, 8);
        assert_eq!(params.min_val, 1.0);
        assert_eq!(params.max_val, 9.0);
        assert_relative_eq!(params.scale, 9.0 / 127.0, epsilon = 1e-6);
        assert_eq!(params.zero_point, 0);

        // Affine quantization
        let params = calibrate_matrix_minmax(&matrix.view(), 8, false).unwrap();
        assert_eq!(params.method, QuantizationMethod::Affine);
        assert_eq!(params.bits, 8);
        assert_eq!(params.min_val, 1.0);
        assert_eq!(params.max_val, 9.0);
        assert_relative_eq!(params.scale, (9.0 - 1.0) / 255.0, epsilon = 1e-6);
        assert_eq!(
            params.zero_point,
            (-params.min_val / params.scale).round() as i32
        );
    }

    #[test]
    fn test_calibrate_matrix_percentile() {
        let matrix = array![
            [1.0f32, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 100.0] // Outlier at 100.0
        ];

        // With percentile 0.8, the outlier should be ignored
        let params = calibrate_matrix_percentile(&matrix.view(), 8, 0.8, true).unwrap();
        assert!(params.max_val < 100.0); // Max should be less than the outlier

        // With percentile 1.0, the outlier should be included
        let params = calibrate_matrix_percentile(&matrix.view(), 8, 1.0, true).unwrap();
        assert_eq!(params.max_val, 100.0); // Max should include the outlier
    }

    #[test]
    fn test_calibrate_vector_minmax() {
        let vector = array![1.0f32, 2.0, 3.0, 4.0, 5.0];

        // Symmetric quantization
        let params = calibrate_vector_minmax(&vector.view(), 8, true).unwrap();
        assert_eq!(params.method, QuantizationMethod::Symmetric);
        assert_eq!(params.bits, 8);
        assert_eq!(params.min_val, 1.0);
        assert_eq!(params.max_val, 5.0);
        assert_relative_eq!(params.scale, 5.0 / 127.0, epsilon = 1e-6);
        assert_eq!(params.zero_point, 0);

        // Affine quantization
        let params = calibrate_vector_minmax(&vector.view(), 8, false).unwrap();
        assert_eq!(params.method, QuantizationMethod::Affine);
        assert_eq!(params.bits, 8);
        assert_eq!(params.min_val, 1.0);
        assert_eq!(params.max_val, 5.0);
        assert_relative_eq!(params.scale, (5.0 - 1.0) / 255.0, epsilon = 1e-6);
        assert_eq!(
            params.zero_point,
            (-params.min_val / params.scale).round() as i32
        );
    }

    #[test]
    fn test_calibrate_matrix_per_channel() {
        // Create a matrix with very different scales in each column
        let matrix = array![
            [0.1f32, 10.0, 100.0],
            [0.2, 20.0, 200.0],
            [0.3, 30.0, 300.0]
        ];

        // Test per-channel symmetric calibration
        let params = calibrate_matrix_per_channel_minmax(&matrix.view(), 8, true).unwrap();
        assert_eq!(params.method, QuantizationMethod::PerChannelSymmetric);

        // Should have 3 different scales (one per column)
        assert!(params.channel_scales.is_some());
        let scales = params.channel_scales.as_ref().unwrap();
        assert_eq!(scales.len(), 3);

        // Scales should be very different for each column
        assert!(scales[0] < scales[1]);
        assert!(scales[1] < scales[2]);

        // Roughly check the expected scale values
        assert_relative_eq!(scales[0], 0.3 / 127.0, epsilon = 1e-5);
        assert_relative_eq!(scales[1], 30.0 / 127.0, epsilon = 1e-5);
        assert_relative_eq!(scales[2], 300.0 / 127.0, epsilon = 1e-5);
    }

    #[test]
    fn test_calibration_end_to_end() {
        // Test the full calibration and quantization pipeline
        let matrix = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        // Configure calibration
        let config = CalibrationConfig {
            method: CalibrationMethod::MinMax,
            symmetric: true,
            ..Default::default()
        };

        // Calibrate
        let params = calibrate_matrix(&matrix.view(), 8, &config).unwrap();

        // Quantize and dequantize
        let (quantized, _) = quantize_matrix(&matrix.view(), 8, params.method);
        let dequantized = dequantize_matrix(&quantized, &params);

        // Check that dequantized values are close to original
        for ((i, j), &orig) in matrix.indexed_iter() {
            let deq = dequantized[[i, j]];
            // Allow some quantization error
            assert!(
                (orig - deq).abs() < 0.1,
                "Values at [{}, {}] differ: original = {}, dequantized = {}",
                i,
                j,
                orig,
                deq
            );
        }
    }
}
