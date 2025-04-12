//! Data normalization and standardization utilities
//!
//! This module provides functions for normalizing and standardizing data,
//! which is often a preprocessing step for machine learning algorithms.

use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2};
use num_traits::{Float, NumCast};

use crate::error::{Result, TransformError};

// Define a small value to use for comparison with zero
const EPSILON: f64 = 1e-10;

/// Method of normalization to apply
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormalizationMethod {
    /// Min-max normalization (scales values to [0, 1] range)
    MinMax,
    /// Min-max normalization to custom range
    MinMaxCustom(f64, f64),
    /// Z-score standardization (zero mean, unit variance)
    ZScore,
    /// Max absolute scaling (scales by maximum absolute value)
    MaxAbs,
    /// L1 normalization (divide by sum of absolute values)
    L1,
    /// L2 normalization (divide by Euclidean norm)
    L2,
}

/// Normalizes a 2D array along a specified axis
///
/// # Arguments
/// * `array` - The input 2D array to normalize
/// * `method` - The normalization method to apply
/// * `axis` - The axis along which to normalize (0 for columns, 1 for rows)
///
/// # Returns
/// * `Result<Array2<f64>>` - The normalized array
///
/// # Examples
/// ```
/// use ndarray::array;
/// use scirs2_transform::normalize::{normalize_array, NormalizationMethod};
///
/// let data = array![[1.0, 2.0, 3.0],
///                   [4.0, 5.0, 6.0],
///                   [7.0, 8.0, 9.0]];
///                   
/// // Normalize columns (axis 0) using min-max normalization
/// let normalized = normalize_array(&data, NormalizationMethod::MinMax, 0).unwrap();
/// ```
pub fn normalize_array<S>(
    array: &ArrayBase<S, Ix2>,
    method: NormalizationMethod,
    axis: usize,
) -> Result<Array2<f64>>
where
    S: Data,
    S::Elem: Float + NumCast,
{
    let array_f64 = array.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

    if !array_f64.is_standard_layout() {
        return Err(TransformError::InvalidInput(
            "Input array must be in standard memory layout".to_string(),
        ));
    }

    if array_f64.ndim() != 2 {
        return Err(TransformError::InvalidInput(
            "Only 2D arrays are supported".to_string(),
        ));
    }

    if axis >= array_f64.ndim() {
        return Err(TransformError::InvalidInput(format!(
            "Invalid axis {} for array with {} dimensions",
            axis,
            array_f64.ndim()
        )));
    }

    let shape = array_f64.shape();
    let mut normalized = Array2::zeros((shape[0], shape[1]));

    match method {
        NormalizationMethod::MinMax => {
            let min = array_f64.map_axis(Axis(axis), |view| {
                view.fold(f64::INFINITY, |acc, &x| acc.min(x))
            });

            let max = array_f64.map_axis(Axis(axis), |view| {
                view.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x))
            });

            let range = &max - &min;

            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    let value = array_f64[[i, j]];
                    let idx = if axis == 0 { j } else { i };

                    if range[idx].abs() > EPSILON {
                        normalized[[i, j]] = (value - min[idx]) / range[idx];
                    } else {
                        normalized[[i, j]] = 0.5; // Default for constant features
                    }
                }
            }
        }
        NormalizationMethod::MinMaxCustom(new_min, new_max) => {
            let min = array_f64.map_axis(Axis(axis), |view| {
                view.fold(f64::INFINITY, |acc, &x| acc.min(x))
            });

            let max = array_f64.map_axis(Axis(axis), |view| {
                view.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x))
            });

            let range = &max - &min;
            let new_range = new_max - new_min;

            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    let value = array_f64[[i, j]];
                    let idx = if axis == 0 { j } else { i };

                    if range[idx].abs() > EPSILON {
                        normalized[[i, j]] = (value - min[idx]) / range[idx] * new_range + new_min;
                    } else {
                        normalized[[i, j]] = (new_min + new_max) / 2.0; // Default for constant features
                    }
                }
            }
        }
        NormalizationMethod::ZScore => {
            let mean = array_f64.map_axis(Axis(axis), |view| {
                view.iter().sum::<f64>() / view.len() as f64
            });

            let std_dev = array_f64.map_axis(Axis(axis), |view| {
                let m = view.iter().sum::<f64>() / view.len() as f64;
                let variance =
                    view.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / view.len() as f64;
                variance.sqrt()
            });

            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    let value = array_f64[[i, j]];
                    let idx = if axis == 0 { j } else { i };

                    if std_dev[idx] > EPSILON {
                        normalized[[i, j]] = (value - mean[idx]) / std_dev[idx];
                    } else {
                        normalized[[i, j]] = 0.0; // Default for constant features
                    }
                }
            }
        }
        NormalizationMethod::MaxAbs => {
            let max_abs = array_f64.map_axis(Axis(axis), |view| {
                view.fold(0.0, |acc, &x| acc.max(x.abs()))
            });

            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    let value = array_f64[[i, j]];
                    let idx = if axis == 0 { j } else { i };

                    if max_abs[idx] > EPSILON {
                        normalized[[i, j]] = value / max_abs[idx];
                    } else {
                        normalized[[i, j]] = 0.0; // Default for constant features
                    }
                }
            }
        }
        NormalizationMethod::L1 => {
            let l1_norm =
                array_f64.map_axis(Axis(axis), |view| view.fold(0.0, |acc, &x| acc + x.abs()));

            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    let value = array_f64[[i, j]];
                    let idx = if axis == 0 { j } else { i };

                    if l1_norm[idx] > EPSILON {
                        normalized[[i, j]] = value / l1_norm[idx];
                    } else {
                        normalized[[i, j]] = 0.0; // Default for constant features
                    }
                }
            }
        }
        NormalizationMethod::L2 => {
            let l2_norm = array_f64.map_axis(Axis(axis), |view| {
                let sum_squares = view.iter().fold(0.0, |acc, &x| acc + x * x);
                sum_squares.sqrt()
            });

            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    let value = array_f64[[i, j]];
                    let idx = if axis == 0 { j } else { i };

                    if l2_norm[idx] > EPSILON {
                        normalized[[i, j]] = value / l2_norm[idx];
                    } else {
                        normalized[[i, j]] = 0.0; // Default for constant features
                    }
                }
            }
        }
    }

    Ok(normalized)
}

/// Normalizes a 1D array
///
/// # Arguments
/// * `array` - The input 1D array to normalize
/// * `method` - The normalization method to apply
///
/// # Returns
/// * `Result<Array1<f64>>` - The normalized array
///
/// # Examples
/// ```
/// use ndarray::array;
/// use scirs2_transform::normalize::{normalize_vector, NormalizationMethod};
///
/// let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
///                   
/// // Normalize vector using min-max normalization
/// let normalized = normalize_vector(&data, NormalizationMethod::MinMax).unwrap();
/// ```
pub fn normalize_vector<S>(
    array: &ArrayBase<S, Ix1>,
    method: NormalizationMethod,
) -> Result<Array1<f64>>
where
    S: Data,
    S::Elem: Float + NumCast,
{
    let array_f64 = array.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

    if array_f64.is_empty() {
        return Err(TransformError::InvalidInput(
            "Input array is empty".to_string(),
        ));
    }

    let mut normalized = Array1::zeros(array_f64.len());

    match method {
        NormalizationMethod::MinMax => {
            let min = array_f64.fold(f64::INFINITY, |acc, &x| acc.min(x));
            let max = array_f64.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
            let range = max - min;

            if range.abs() > EPSILON {
                for (i, &value) in array_f64.iter().enumerate() {
                    normalized[i] = (value - min) / range;
                }
            } else {
                normalized.fill(0.5); // Default for constant features
            }
        }
        NormalizationMethod::MinMaxCustom(new_min, new_max) => {
            let min = array_f64.fold(f64::INFINITY, |acc, &x| acc.min(x));
            let max = array_f64.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
            let range = max - min;
            let new_range = new_max - new_min;

            if range.abs() > EPSILON {
                for (i, &value) in array_f64.iter().enumerate() {
                    normalized[i] = (value - min) / range * new_range + new_min;
                }
            } else {
                normalized.fill((new_min + new_max) / 2.0); // Default for constant features
            }
        }
        NormalizationMethod::ZScore => {
            let mean = array_f64.iter().sum::<f64>() / array_f64.len() as f64;
            let variance =
                array_f64.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / array_f64.len() as f64;
            let std_dev = variance.sqrt();

            if std_dev > EPSILON {
                for (i, &value) in array_f64.iter().enumerate() {
                    normalized[i] = (value - mean) / std_dev;
                }
            } else {
                normalized.fill(0.0); // Default for constant features
            }
        }
        NormalizationMethod::MaxAbs => {
            let max_abs = array_f64.fold(0.0, |acc, &x| acc.max(x.abs()));

            if max_abs > EPSILON {
                for (i, &value) in array_f64.iter().enumerate() {
                    normalized[i] = value / max_abs;
                }
            } else {
                normalized.fill(0.0); // Default for constant features
            }
        }
        NormalizationMethod::L1 => {
            let l1_norm = array_f64.fold(0.0, |acc, &x| acc + x.abs());

            if l1_norm > EPSILON {
                for (i, &value) in array_f64.iter().enumerate() {
                    normalized[i] = value / l1_norm;
                }
            } else {
                normalized.fill(0.0); // Default for constant features
            }
        }
        NormalizationMethod::L2 => {
            let sum_squares = array_f64.iter().fold(0.0, |acc, &x| acc + x * x);
            let l2_norm = sum_squares.sqrt();

            if l2_norm > EPSILON {
                for (i, &value) in array_f64.iter().enumerate() {
                    normalized[i] = value / l2_norm;
                }
            } else {
                normalized.fill(0.0); // Default for constant features
            }
        }
    }

    Ok(normalized)
}

/// Represents a fitted normalization model that can transform new data
pub struct Normalizer {
    /// The normalization method to apply
    #[allow(dead_code)]
    method: NormalizationMethod,
    /// The axis along which to normalize (0 for columns, 1 for rows)
    axis: usize,
    /// Parameters from the fit (depends on method)
    params: NormalizerParams,
}

/// Parameters for different normalization methods
#[derive(Clone)]
enum NormalizerParams {
    /// Min and max values for MinMax normalization
    MinMax {
        min: Array1<f64>,
        max: Array1<f64>,
        new_min: f64,
        new_max: f64,
    },
    /// Mean and standard deviation for ZScore normalization
    ZScore {
        mean: Array1<f64>,
        std_dev: Array1<f64>,
    },
    /// Maximum absolute values for MaxAbs normalization
    MaxAbs { max_abs: Array1<f64> },
    /// L1 norms for L1 normalization
    L1 { l1_norm: Array1<f64> },
    /// L2 norms for L2 normalization
    L2 { l2_norm: Array1<f64> },
}

impl Normalizer {
    /// Creates a new Normalizer with the specified method and axis
    ///
    /// # Arguments
    /// * `method` - The normalization method to apply
    /// * `axis` - The axis along which to normalize (0 for columns, 1 for rows)
    ///
    /// # Returns
    /// * A new Normalizer instance
    pub fn new(method: NormalizationMethod, axis: usize) -> Self {
        let params = match method {
            NormalizationMethod::MinMax => NormalizerParams::MinMax {
                min: Array1::zeros(0),
                max: Array1::zeros(0),
                new_min: 0.0,
                new_max: 1.0,
            },
            NormalizationMethod::MinMaxCustom(min, max) => NormalizerParams::MinMax {
                min: Array1::zeros(0),
                max: Array1::zeros(0),
                new_min: min,
                new_max: max,
            },
            NormalizationMethod::ZScore => NormalizerParams::ZScore {
                mean: Array1::zeros(0),
                std_dev: Array1::zeros(0),
            },
            NormalizationMethod::MaxAbs => NormalizerParams::MaxAbs {
                max_abs: Array1::zeros(0),
            },
            NormalizationMethod::L1 => NormalizerParams::L1 {
                l1_norm: Array1::zeros(0),
            },
            NormalizationMethod::L2 => NormalizerParams::L2 {
                l2_norm: Array1::zeros(0),
            },
        };

        Normalizer {
            method,
            axis,
            params,
        }
    }

    /// Fits the normalizer to the input data
    ///
    /// # Arguments
    /// * `array` - The input 2D array to fit the normalizer to
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    pub fn fit<S>(&mut self, array: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let array_f64 = array.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

        if !array_f64.is_standard_layout() {
            return Err(TransformError::InvalidInput(
                "Input array must be in standard memory layout".to_string(),
            ));
        }

        if array_f64.ndim() != 2 {
            return Err(TransformError::InvalidInput(
                "Only 2D arrays are supported".to_string(),
            ));
        }

        if self.axis >= array_f64.ndim() {
            return Err(TransformError::InvalidInput(format!(
                "Invalid axis {} for array with {} dimensions",
                self.axis,
                array_f64.ndim()
            )));
        }

        let _size = if self.axis == 0 {
            array_f64.shape()[1]
        } else {
            array_f64.shape()[0]
        };

        match &mut self.params {
            NormalizerParams::MinMax {
                min,
                max,
                new_min: _,
                new_max: _,
            } => {
                *min = array_f64.map_axis(Axis(self.axis), |view| {
                    view.fold(f64::INFINITY, |acc, &x| acc.min(x))
                });

                *max = array_f64.map_axis(Axis(self.axis), |view| {
                    view.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x))
                });
            }
            NormalizerParams::ZScore { mean, std_dev } => {
                *mean = array_f64.map_axis(Axis(self.axis), |view| {
                    view.iter().sum::<f64>() / view.len() as f64
                });

                *std_dev = array_f64.map_axis(Axis(self.axis), |view| {
                    let m = view.iter().sum::<f64>() / view.len() as f64;
                    let variance =
                        view.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / view.len() as f64;
                    variance.sqrt()
                });
            }
            NormalizerParams::MaxAbs { max_abs } => {
                *max_abs = array_f64.map_axis(Axis(self.axis), |view| {
                    view.fold(0.0, |acc, &x| acc.max(x.abs()))
                });
            }
            NormalizerParams::L1 { l1_norm } => {
                *l1_norm = array_f64.map_axis(Axis(self.axis), |view| {
                    view.fold(0.0, |acc, &x| acc + x.abs())
                });
            }
            NormalizerParams::L2 { l2_norm } => {
                *l2_norm = array_f64.map_axis(Axis(self.axis), |view| {
                    let sum_squares = view.iter().fold(0.0, |acc, &x| acc + x * x);
                    sum_squares.sqrt()
                });
            }
        }

        Ok(())
    }

    /// Transforms the input data using the fitted normalizer
    ///
    /// # Arguments
    /// * `array` - The input 2D array to transform
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The transformed array
    pub fn transform<S>(&self, array: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let array_f64 = array.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

        if !array_f64.is_standard_layout() {
            return Err(TransformError::InvalidInput(
                "Input array must be in standard memory layout".to_string(),
            ));
        }

        if array_f64.ndim() != 2 {
            return Err(TransformError::InvalidInput(
                "Only 2D arrays are supported".to_string(),
            ));
        }

        // Check the dimension along the normalization axis
        let expected_size = match &self.params {
            NormalizerParams::MinMax { min, .. } => min.len(),
            NormalizerParams::ZScore { mean, .. } => mean.len(),
            NormalizerParams::MaxAbs { max_abs } => max_abs.len(),
            NormalizerParams::L1 { l1_norm } => l1_norm.len(),
            NormalizerParams::L2 { l2_norm } => l2_norm.len(),
        };

        let actual_size = if self.axis == 0 {
            array_f64.shape()[1]
        } else {
            array_f64.shape()[0]
        };

        if expected_size != actual_size {
            return Err(TransformError::InvalidInput(format!(
                "Expected {} features, got {}",
                expected_size, actual_size
            )));
        }

        let shape = array_f64.shape();
        let mut transformed = Array2::zeros((shape[0], shape[1]));

        match &self.params {
            NormalizerParams::MinMax {
                min,
                max,
                new_min,
                new_max,
            } => {
                let range = max - min;
                let new_range = new_max - new_min;

                for i in 0..shape[0] {
                    for j in 0..shape[1] {
                        let value = array_f64[[i, j]];
                        let idx = if self.axis == 0 { j } else { i };

                        if range[idx].abs() > EPSILON {
                            transformed[[i, j]] =
                                (value - min[idx]) / range[idx] * new_range + new_min;
                        } else {
                            transformed[[i, j]] = (new_min + new_max) / 2.0; // Default for constant features
                        }
                    }
                }
            }
            NormalizerParams::ZScore { mean, std_dev } => {
                for i in 0..shape[0] {
                    for j in 0..shape[1] {
                        let value = array_f64[[i, j]];
                        let idx = if self.axis == 0 { j } else { i };

                        if std_dev[idx] > EPSILON {
                            transformed[[i, j]] = (value - mean[idx]) / std_dev[idx];
                        } else {
                            transformed[[i, j]] = 0.0; // Default for constant features
                        }
                    }
                }
            }
            NormalizerParams::MaxAbs { max_abs } => {
                for i in 0..shape[0] {
                    for j in 0..shape[1] {
                        let value = array_f64[[i, j]];
                        let idx = if self.axis == 0 { j } else { i };

                        if max_abs[idx] > EPSILON {
                            transformed[[i, j]] = value / max_abs[idx];
                        } else {
                            transformed[[i, j]] = 0.0; // Default for constant features
                        }
                    }
                }
            }
            NormalizerParams::L1 { l1_norm } => {
                for i in 0..shape[0] {
                    for j in 0..shape[1] {
                        let value = array_f64[[i, j]];
                        let idx = if self.axis == 0 { j } else { i };

                        if l1_norm[idx] > EPSILON {
                            transformed[[i, j]] = value / l1_norm[idx];
                        } else {
                            transformed[[i, j]] = 0.0; // Default for constant features
                        }
                    }
                }
            }
            NormalizerParams::L2 { l2_norm } => {
                for i in 0..shape[0] {
                    for j in 0..shape[1] {
                        let value = array_f64[[i, j]];
                        let idx = if self.axis == 0 { j } else { i };

                        if l2_norm[idx] > EPSILON {
                            transformed[[i, j]] = value / l2_norm[idx];
                        } else {
                            transformed[[i, j]] = 0.0; // Default for constant features
                        }
                    }
                }
            }
        }

        Ok(transformed)
    }

    /// Fits the normalizer to the input data and transforms it
    ///
    /// # Arguments
    /// * `array` - The input 2D array to fit and transform
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The transformed array
    pub fn fit_transform<S>(&mut self, array: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        self.fit(array)?;
        self.transform(array)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array;

    #[test]
    fn test_normalize_vector_minmax() {
        let data = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let normalized = normalize_vector(&data, NormalizationMethod::MinMax).unwrap();

        let expected = Array::from_vec(vec![0.0, 0.25, 0.5, 0.75, 1.0]);

        for (a, b) in normalized.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_normalize_vector_zscore() {
        let data = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let normalized = normalize_vector(&data, NormalizationMethod::ZScore).unwrap();

        let mean = 3.0;
        let std_dev = (10.0 / 5.0_f64).sqrt();
        let expected = data.mapv(|x| (x - mean) / std_dev);

        for (a, b) in normalized.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_normalize_array_minmax() {
        let data = Array::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .unwrap();

        // Normalize columns (axis 0)
        let normalized = normalize_array(&data, NormalizationMethod::MinMax, 0).unwrap();

        let expected =
            Array::from_shape_vec((3, 3), vec![0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0])
                .unwrap();

        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(normalized[[i, j]], expected[[i, j]], epsilon = 1e-10);
            }
        }

        // Normalize rows (axis 1)
        let normalized = normalize_array(&data, NormalizationMethod::MinMax, 1).unwrap();

        let expected =
            Array::from_shape_vec((3, 3), vec![0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
                .unwrap();

        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(normalized[[i, j]], expected[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_normalizer_fit_transform() {
        let data = Array::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .unwrap();

        // Test MinMax normalization
        let mut normalizer = Normalizer::new(NormalizationMethod::MinMax, 0);
        let transformed = normalizer.fit_transform(&data).unwrap();

        let expected =
            Array::from_shape_vec((3, 3), vec![0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0])
                .unwrap();

        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(transformed[[i, j]], expected[[i, j]], epsilon = 1e-10);
            }
        }

        // Test with separate fit and transform
        let data2 = Array::from_shape_vec((2, 3), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();

        let transformed2 = normalizer.transform(&data2).unwrap();

        let expected2 = Array::from_shape_vec(
            (2, 3),
            vec![
                1.0 / 6.0,
                1.0 / 6.0,
                1.0 / 6.0,
                2.0 / 3.0,
                2.0 / 3.0,
                2.0 / 3.0,
            ],
        )
        .unwrap();

        for i in 0..2 {
            for j in 0..3 {
                assert_abs_diff_eq!(transformed2[[i, j]], expected2[[i, j]], epsilon = 1e-10);
            }
        }
    }
}
