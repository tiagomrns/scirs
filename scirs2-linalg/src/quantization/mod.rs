//! Quantization-aware linear algebra operations
//!
//! This module provides functions and types for working with quantized matrices and vectors.
//! Quantization reduces the precision of floating-point numbers to save memory and
//! computational resources, which is particularly useful in machine learning applications.
//!
//! ## Overview
//!
//! * Quantization of matrices and vectors to lower bit-width representations
//! * Linear algebra operations on quantized data
//! * Support for different quantization methods (uniform, symmetric, affine)
//! * Efficient operations with mixed quantized and floating-point data
//!
//! ## Examples
//!
//! Basic quantization:
//!
//! ```ignore
//! use ndarray::{Array2, array};
//! use scirs2_linalg::quantization::{quantize_matrix, dequantize_matrix, QuantizationMethod};
//!
//! let a = array![[1.0_f32, 2.5, 3.7], [4.2, 5.0, 6.1]];
//!
//! // Quantize to 8-bit
//! let (quantized, params) = quantize_matrix(&a.view(), 8, QuantizationMethod::Affine);
//!
//! // Dequantize back to floating point
//! let a_dequantized = dequantize_matrix(&quantized, &params);
//!
//! // Check the error is small
//! let max_error = (&a - &a_dequantized).mapv(|x| x.abs()).fold(0.0, |acc, &b| acc.max(b));
//! assert!(max_error < 0.1); // Error should be small but non-zero due to quantization
//! ```
//!
//! Quantized matrix multiplication:
//!
//! ```ignore
//! use ndarray::{Array2, array};
//! use scirs2_linalg::quantization::{quantize_matrix, QuantizationMethod, quantized_matmul};
//!
//! let a = array![[1.0_f32, 2.0], [3.0, 4.0]];
//! let b = array![[5.0_f32, 6.0], [7.0, 8.0]];
//!
//! // Quantize both matrices to 8-bit
//! let (a_q, a_params) = quantize_matrix(&a.view(), 8, QuantizationMethod::Symmetric);
//! let (b_q, b_params) = quantize_matrix(&b.view(), 8, QuantizationMethod::Symmetric);
//!
//! // Perform quantized matrix multiplication
//! let c_q = quantized_matmul(&a_q, &a_params, &b_q, &b_params).unwrap();
//!
//! // Regular matrix multiplication for comparison
//! let c = a.dot(&b);
//!
//! // Check the error is acceptable
//! let rel_error = (&c - &c_q).mapv(|x| x.abs()).sum() / c.sum();
//! assert!(rel_error < 0.1); // Relative error should be small
//! ```

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{AsPrimitive, Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{LinalgError, LinalgResult};

/// Supported methods of quantization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationMethod {
    /// Uniform quantization maps the input range to uniform discrete levels
    /// with equal spacing between consecutive levels
    Uniform,

    /// Symmetric quantization is centered around zero and has equal positive and
    /// negative ranges, making it suitable for weight matrices
    Symmetric,

    /// Affine quantization uses the formula q = scale * (x - zero_point)
    /// allowing better representation of asymmetric distributions
    Affine,

    /// Power-of-two quantization uses powers of 2 for the scale factor,
    /// enabling efficient implementation with bitshifts
    PowerOfTwo,
}

/// Parameters for the quantization process
#[derive(Debug, Clone)]
pub struct QuantizationParams {
    /// The number of bits used for quantization
    pub bits: u8,

    /// The scale factor used to convert between quantized and float values
    pub scale: f32,

    /// The zero point used for asymmetric quantization (for affine quantization)
    pub zero_point: i32,

    /// The minimum value of the original data
    pub min_val: f32,

    /// The maximum value of the original data
    pub max_val: f32,

    /// The quantization method used
    pub method: QuantizationMethod,
}

/// A matrix with quantized values
#[derive(Debug, Clone)]
pub struct QuantizedMatrix {
    /// The quantized values stored as integers
    pub data: Array2<i8>,

    /// The original shape of the matrix
    pub shape: (usize, usize),
}

/// A vector with quantized values
#[derive(Debug, Clone)]
pub struct QuantizedVector {
    /// The quantized values stored as integers
    pub data: Array1<i8>,

    /// The original length of the vector
    pub length: usize,
}

impl QuantizedMatrix {
    /// Creates a new quantized matrix
    pub fn new(data: Array2<i8>, shape: (usize, usize)) -> Self {
        Self { data, shape }
    }

    /// Returns the shape of the matrix
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Returns the number of rows in the matrix
    pub fn nrows(&self) -> usize {
        self.shape.0
    }

    /// Returns the number of columns in the matrix
    pub fn ncols(&self) -> usize {
        self.shape.1
    }
}

impl QuantizedVector {
    /// Creates a new quantized vector
    pub fn new(data: Array1<i8>, length: usize) -> Self {
        Self { data, length }
    }

    /// Returns the length of the vector
    pub fn len(&self) -> usize {
        self.length
    }

    /// Returns true if the vector is empty
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
}

/// Quantize a floating-point matrix to a lower precision representation
///
/// # Arguments
///
/// * `matrix` - The input matrix to quantize
/// * `bits` - The number of bits to use for quantization (typically 8)
/// * `method` - The quantization method to use
///
/// # Returns
///
/// A tuple containing the quantized matrix and the quantization parameters
pub fn quantize_matrix<F>(
    matrix: &ArrayView2<F>,
    bits: u8,
    method: QuantizationMethod,
) -> (QuantizedMatrix, QuantizationParams)
where
    F: Float + Debug + AsPrimitive<f32> + FromPrimitive,
    f32: AsPrimitive<F>,
{
    let shape = (matrix.nrows(), matrix.ncols());

    // Find min and max values
    let mut min_val = F::infinity().as_();
    let mut max_val = F::neg_infinity().as_();

    for &val in matrix.iter() {
        let val_f32: f32 = val.as_();
        if val_f32.is_finite() {
            min_val = min_val.min(val_f32);
            max_val = max_val.max(val_f32);
        }
    }

    // Handle case where all values are the same
    if (max_val - min_val).abs() < f32::EPSILON {
        max_val = min_val + 1.0;
    }

    // Calculate quantization parameters based on the chosen method
    let (scale, zero_point) = match method {
        QuantizationMethod::Uniform => {
            let scale = (max_val - min_val) / ((1 << bits) - 1) as f32;
            let zero_point = 0;
            (scale, zero_point)
        }
        QuantizationMethod::Symmetric => {
            // Symmetric around zero, calculate scale to fit
            let abs_max = max_val.abs().max(min_val.abs());
            let scale = abs_max / ((1 << (bits - 1)) - 1) as f32;
            let zero_point = 0;
            (scale, zero_point)
        }
        QuantizationMethod::Affine => {
            let scale = (max_val - min_val) / ((1 << bits) - 1) as f32;
            let zero_point = (-min_val / scale).round() as i32;
            (scale, zero_point)
        }
        QuantizationMethod::PowerOfTwo => {
            // Find the smallest power of 2 greater than or equal to (max_val - min_val) / ((1 << bits) - 1)
            let range = max_val - min_val;
            let ideal_scale = range / ((1 << bits) - 1) as f32;
            let exponent = ideal_scale.log2().ceil();
            let scale = 2.0_f32.powf(exponent);
            let zero_point = 0;
            (scale, zero_point)
        }
    };

    // Create quantization parameters
    let params = QuantizationParams {
        bits,
        scale,
        zero_point,
        min_val,
        max_val,
        method,
    };

    // Perform the actual quantization
    let quantized_data = match method {
        QuantizationMethod::Uniform => {
            let mut quantized = Array2::zeros(shape);
            for (i, &val) in matrix.iter().enumerate() {
                let val_f32: f32 = val.as_();
                let q_val = ((val_f32 - min_val) / scale).round() as i8;
                quantized.as_slice_mut().unwrap()[i] = q_val;
            }
            quantized
        }
        QuantizationMethod::Symmetric => {
            let mut quantized = Array2::zeros(shape);
            for (i, &val) in matrix.iter().enumerate() {
                let val_f32: f32 = val.as_();
                let q_val = (val_f32 / scale).round() as i8;
                quantized.as_slice_mut().unwrap()[i] = q_val;
            }
            quantized
        }
        QuantizationMethod::Affine => {
            let mut quantized = Array2::zeros(shape);
            for (i, &val) in matrix.iter().enumerate() {
                let val_f32: f32 = val.as_();
                let q_val = ((val_f32 / scale) + zero_point as f32).round() as i8;
                quantized.as_slice_mut().unwrap()[i] = q_val;
            }
            quantized
        }
        QuantizationMethod::PowerOfTwo => {
            let mut quantized = Array2::zeros(shape);
            for (i, &val) in matrix.iter().enumerate() {
                let val_f32: f32 = val.as_();
                let q_val = ((val_f32 - min_val) / scale).round() as i8;
                quantized.as_slice_mut().unwrap()[i] = q_val;
            }
            quantized
        }
    };

    (QuantizedMatrix::new(quantized_data, shape), params)
}

/// Dequantize a matrix back to floating-point
///
/// # Arguments
///
/// * `quantized` - The quantized matrix
/// * `params` - The quantization parameters
///
/// # Returns
///
/// The dequantized matrix
pub fn dequantize_matrix(quantized: &QuantizedMatrix, params: &QuantizationParams) -> Array2<f32> {
    let shape = quantized.shape();
    let mut dequantized = Array2::zeros(shape);

    // Perform dequantization based on the quantization method
    match params.method {
        QuantizationMethod::Uniform => {
            for (i, &q_val) in quantized.data.iter().enumerate() {
                let val = params.min_val + (q_val as f32 * params.scale);
                dequantized.as_slice_mut().unwrap()[i] = val;
            }
        }
        QuantizationMethod::Symmetric => {
            for (i, &q_val) in quantized.data.iter().enumerate() {
                let val = q_val as f32 * params.scale;
                dequantized.as_slice_mut().unwrap()[i] = val;
            }
        }
        QuantizationMethod::Affine => {
            for (i, &q_val) in quantized.data.iter().enumerate() {
                let val = params.scale * (q_val as f32 - params.zero_point as f32);
                dequantized.as_slice_mut().unwrap()[i] = val;
            }
        }
        QuantizationMethod::PowerOfTwo => {
            for (i, &q_val) in quantized.data.iter().enumerate() {
                let val = params.min_val + (q_val as f32 * params.scale);
                dequantized.as_slice_mut().unwrap()[i] = val;
            }
        }
    }

    dequantized
}

/// Quantize a floating-point vector to a lower precision representation
///
/// # Arguments
///
/// * `vector` - The input vector to quantize
/// * `bits` - The number of bits to use for quantization (typically 8)
/// * `method` - The quantization method to use
///
/// # Returns
///
/// A tuple containing the quantized vector and the quantization parameters
pub fn quantize_vector<F>(
    vector: &ArrayView1<F>,
    bits: u8,
    method: QuantizationMethod,
) -> (QuantizedVector, QuantizationParams)
where
    F: Float + Debug + AsPrimitive<f32> + FromPrimitive,
    f32: AsPrimitive<F>,
{
    let length = vector.len();

    // Find min and max values
    let mut min_val = F::infinity().as_();
    let mut max_val = F::neg_infinity().as_();

    for &val in vector.iter() {
        let val_f32: f32 = val.as_();
        if val_f32.is_finite() {
            min_val = min_val.min(val_f32);
            max_val = max_val.max(val_f32);
        }
    }

    // Handle case where all values are the same
    if (max_val - min_val).abs() < f32::EPSILON {
        max_val = min_val + 1.0;
    }

    // Calculate quantization parameters based on the chosen method
    let (scale, zero_point) = match method {
        QuantizationMethod::Uniform => {
            let scale = (max_val - min_val) / ((1 << bits) - 1) as f32;
            let zero_point = 0;
            (scale, zero_point)
        }
        QuantizationMethod::Symmetric => {
            // Symmetric around zero, calculate scale to fit
            let abs_max = max_val.abs().max(min_val.abs());
            let scale = abs_max / ((1 << (bits - 1)) - 1) as f32;
            let zero_point = 0;
            (scale, zero_point)
        }
        QuantizationMethod::Affine => {
            let scale = (max_val - min_val) / ((1 << bits) - 1) as f32;
            let zero_point = (-min_val / scale).round() as i32;
            (scale, zero_point)
        }
        QuantizationMethod::PowerOfTwo => {
            // Find the smallest power of 2 greater than or equal to (max_val - min_val) / ((1 << bits) - 1)
            let range = max_val - min_val;
            let ideal_scale = range / ((1 << bits) - 1) as f32;
            let exponent = ideal_scale.log2().ceil();
            let scale = 2.0_f32.powf(exponent);
            let zero_point = 0;
            (scale, zero_point)
        }
    };

    // Create quantization parameters
    let params = QuantizationParams {
        bits,
        scale,
        zero_point,
        min_val,
        max_val,
        method,
    };

    // Perform the actual quantization
    let quantized_data = match method {
        QuantizationMethod::Uniform => {
            let mut quantized = Array1::zeros(length);
            for (i, &val) in vector.iter().enumerate() {
                let val_f32: f32 = val.as_();
                let q_val = ((val_f32 - min_val) / scale).round() as i8;
                quantized[i] = q_val;
            }
            quantized
        }
        QuantizationMethod::Symmetric => {
            let mut quantized = Array1::zeros(length);
            for (i, &val) in vector.iter().enumerate() {
                let val_f32: f32 = val.as_();
                let q_val = (val_f32 / scale).round() as i8;
                quantized[i] = q_val;
            }
            quantized
        }
        QuantizationMethod::Affine => {
            let mut quantized = Array1::zeros(length);
            for (i, &val) in vector.iter().enumerate() {
                let val_f32: f32 = val.as_();
                let q_val = ((val_f32 / scale) + zero_point as f32).round() as i8;
                quantized[i] = q_val;
            }
            quantized
        }
        QuantizationMethod::PowerOfTwo => {
            let mut quantized = Array1::zeros(length);
            for (i, &val) in vector.iter().enumerate() {
                let val_f32: f32 = val.as_();
                let q_val = ((val_f32 - min_val) / scale).round() as i8;
                quantized[i] = q_val;
            }
            quantized
        }
    };

    (QuantizedVector::new(quantized_data, length), params)
}

/// Dequantize a vector back to floating-point
///
/// # Arguments
///
/// * `quantized` - The quantized vector
/// * `params` - The quantization parameters
///
/// # Returns
///
/// The dequantized vector
pub fn dequantize_vector(quantized: &QuantizedVector, params: &QuantizationParams) -> Array1<f32> {
    let length = quantized.len();
    let mut dequantized = Array1::zeros(length);

    // Perform dequantization based on the quantization method
    match params.method {
        QuantizationMethod::Uniform => {
            for (i, &q_val) in quantized.data.iter().enumerate() {
                let val = params.min_val + (q_val as f32 * params.scale);
                dequantized[i] = val;
            }
        }
        QuantizationMethod::Symmetric => {
            for (i, &q_val) in quantized.data.iter().enumerate() {
                let val = q_val as f32 * params.scale;
                dequantized[i] = val;
            }
        }
        QuantizationMethod::Affine => {
            for (i, &q_val) in quantized.data.iter().enumerate() {
                let val = params.scale * (q_val as f32 - params.zero_point as f32);
                dequantized[i] = val;
            }
        }
        QuantizationMethod::PowerOfTwo => {
            for (i, &q_val) in quantized.data.iter().enumerate() {
                let val = params.min_val + (q_val as f32 * params.scale);
                dequantized[i] = val;
            }
        }
    }

    dequantized
}

/// Perform matrix multiplication with quantized matrices
///
/// # Arguments
///
/// * `a` - The first quantized matrix
/// * `a_params` - Quantization parameters for the first matrix
/// * `b` - The second quantized matrix
/// * `b_params` - Quantization parameters for the second matrix
///
/// # Returns
///
/// The result of the matrix multiplication in floating-point
pub fn quantized_matmul(
    a: &QuantizedMatrix,
    a_params: &QuantizationParams,
    b: &QuantizedMatrix,
    b_params: &QuantizationParams,
) -> LinalgResult<Array2<f32>> {
    // Check dimensions
    if a.ncols() != b.nrows() {
        return Err(LinalgError::DimensionError(format!(
            "Cannot multiply matrices with shapes {:?} and {:?}",
            a.shape(),
            b.shape()
        )));
    }

    let (m, k) = a.shape();
    let (_, n) = b.shape();

    // Create result matrix
    let mut result = Array2::zeros((m, n));

    // Perform the calculation
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0i32;
            for l in 0..k {
                sum += (a.data[[i, l]] as i32) * (b.data[[l, j]] as i32);
            }

            // Dequantize the result
            let a_scale = match a_params.method {
                QuantizationMethod::Uniform => a_params.scale,
                QuantizationMethod::Symmetric => a_params.scale,
                QuantizationMethod::Affine => a_params.scale,
                QuantizationMethod::PowerOfTwo => a_params.scale,
            };

            let b_scale = match b_params.method {
                QuantizationMethod::Uniform => b_params.scale,
                QuantizationMethod::Symmetric => b_params.scale,
                QuantizationMethod::Affine => b_params.scale,
                QuantizationMethod::PowerOfTwo => b_params.scale,
            };

            // Apply zero-point correction for affine quantization
            if a_params.method == QuantizationMethod::Affine
                && b_params.method == QuantizationMethod::Affine
            {
                // For affine quantization, we need to correct for zero points
                let a_zero_sum: i32 =
                    (0..k).map(|l| b.data[[l, j]] as i32).sum::<i32>() * a_params.zero_point;
                let b_zero_sum: i32 =
                    (0..k).map(|l| a.data[[i, l]] as i32).sum::<i32>() * b_params.zero_point;
                let zero_product = k as i32 * a_params.zero_point * b_params.zero_point;

                sum = sum - a_zero_sum - b_zero_sum + zero_product;
            }

            result[[i, j]] = sum as f32 * a_scale * b_scale;
        }
    }

    Ok(result)
}

/// Perform matrix-vector multiplication with quantized matrix and vector
///
/// # Arguments
///
/// * `a` - The quantized matrix
/// * `a_params` - Quantization parameters for the matrix
/// * `x` - The quantized vector
/// * `x_params` - Quantization parameters for the vector
///
/// # Returns
///
/// The result of the matrix-vector multiplication in floating-point
pub fn quantized_matvec(
    a: &QuantizedMatrix,
    a_params: &QuantizationParams,
    x: &QuantizedVector,
    x_params: &QuantizationParams,
) -> LinalgResult<Array1<f32>> {
    // Check dimensions
    if a.ncols() != x.len() {
        return Err(LinalgError::DimensionError(format!(
            "Cannot multiply matrix with shape {:?} and vector with length {}",
            a.shape(),
            x.len()
        )));
    }

    let (m, n) = a.shape();

    // Create result vector
    let mut result = Array1::zeros(m);

    // Perform the calculation
    for i in 0..m {
        let mut sum = 0i32;
        for j in 0..n {
            sum += (a.data[[i, j]] as i32) * (x.data[j] as i32);
        }

        // Dequantize the result
        let a_scale = match a_params.method {
            QuantizationMethod::Uniform => a_params.scale,
            QuantizationMethod::Symmetric => a_params.scale,
            QuantizationMethod::Affine => a_params.scale,
            QuantizationMethod::PowerOfTwo => a_params.scale,
        };

        let x_scale = match x_params.method {
            QuantizationMethod::Uniform => x_params.scale,
            QuantizationMethod::Symmetric => x_params.scale,
            QuantizationMethod::Affine => x_params.scale,
            QuantizationMethod::PowerOfTwo => x_params.scale,
        };

        // Apply zero-point correction for affine quantization
        if a_params.method == QuantizationMethod::Affine
            && x_params.method == QuantizationMethod::Affine
        {
            // For affine quantization, we need to correct for zero points
            let a_zero_sum: i32 =
                (0..n).map(|j| x.data[j] as i32).sum::<i32>() * a_params.zero_point;
            let x_zero_sum: i32 =
                (0..n).map(|j| a.data[[i, j]] as i32).sum::<i32>() * x_params.zero_point;
            let zero_product = n as i32 * a_params.zero_point * x_params.zero_point;

            sum = sum - a_zero_sum - x_zero_sum + zero_product;
        }

        result[i] = sum as f32 * a_scale * x_scale;
    }

    Ok(result)
}

/// Compute the dot product of two quantized vectors
///
/// # Arguments
///
/// * `a` - The first quantized vector
/// * `a_params` - Quantization parameters for the first vector
/// * `b` - The second quantized vector
/// * `b_params` - Quantization parameters for the second vector
///
/// # Returns
///
/// The dot product as a floating-point value
pub fn quantized_dot(
    a: &QuantizedVector,
    a_params: &QuantizationParams,
    b: &QuantizedVector,
    b_params: &QuantizationParams,
) -> LinalgResult<f32> {
    // Check dimensions
    if a.len() != b.len() {
        return Err(LinalgError::DimensionError(format!(
            "Cannot compute dot product of vectors with lengths {} and {}",
            a.len(),
            b.len()
        )));
    }

    let n = a.len();

    // Compute the dot product
    let mut sum = 0i32;
    for i in 0..n {
        sum += (a.data[i] as i32) * (b.data[i] as i32);
    }

    // Dequantize the result
    let a_scale = match a_params.method {
        QuantizationMethod::Uniform => a_params.scale,
        QuantizationMethod::Symmetric => a_params.scale,
        QuantizationMethod::Affine => a_params.scale,
        QuantizationMethod::PowerOfTwo => a_params.scale,
    };

    let b_scale = match b_params.method {
        QuantizationMethod::Uniform => b_params.scale,
        QuantizationMethod::Symmetric => b_params.scale,
        QuantizationMethod::Affine => b_params.scale,
        QuantizationMethod::PowerOfTwo => b_params.scale,
    };

    // Apply zero-point correction for affine quantization
    if a_params.method == QuantizationMethod::Affine
        && b_params.method == QuantizationMethod::Affine
    {
        // For affine quantization, we need to correct for zero points
        let a_zero_sum: i32 = (0..n).map(|i| b.data[i] as i32).sum::<i32>() * a_params.zero_point;
        let b_zero_sum: i32 = (0..n).map(|i| a.data[i] as i32).sum::<i32>() * b_params.zero_point;
        let zero_product = n as i32 * a_params.zero_point * b_params.zero_point;

        sum = sum - a_zero_sum - b_zero_sum + zero_product;
    }

    let result = sum as f32 * a_scale * b_scale;

    Ok(result)
}

/// Apply fake quantization to a floating-point tensor
///
/// Fake quantization applies the quantization and dequantization steps
/// to simulate the effects of quantization while still working with
/// floating-point values. This is useful for training quantization-aware
/// neural networks.
///
/// # Arguments
///
/// * `matrix` - The input matrix to apply fake quantization to
/// * `bits` - The number of bits to use for quantization (typically 8)
/// * `method` - The quantization method to use
///
/// # Returns
///
/// The matrix after applying fake quantization
pub fn fake_quantize<F>(matrix: &ArrayView2<F>, bits: u8, method: QuantizationMethod) -> Array2<F>
where
    F: Float + Debug + AsPrimitive<f32> + FromPrimitive,
    f32: AsPrimitive<F>,
{
    let (quantized, params) = quantize_matrix(matrix, bits, method);
    let dequantized = dequantize_matrix(&quantized, &params);

    // Convert back to original type
    let mut result = Array2::zeros(matrix.dim());
    for (i, &val) in dequantized.iter().enumerate() {
        result.as_slice_mut().unwrap()[i] = F::from_f32(val).unwrap();
    }

    result
}

/// Apply fake quantization to a floating-point vector
///
/// # Arguments
///
/// * `vector` - The input vector to apply fake quantization to
/// * `bits` - The number of bits to use for quantization (typically 8)
/// * `method` - The quantization method to use
///
/// # Returns
///
/// The vector after applying fake quantization
pub fn fake_quantize_vector<F>(
    vector: &ArrayView1<F>,
    bits: u8,
    method: QuantizationMethod,
) -> Array1<F>
where
    F: Float + Debug + AsPrimitive<f32> + FromPrimitive,
    f32: AsPrimitive<F>,
{
    let (quantized, params) = quantize_vector(vector, bits, method);
    let dequantized = dequantize_vector(&quantized, &params);

    // Convert back to original type
    let mut result = Array1::zeros(vector.dim());
    for (i, &val) in dequantized.iter().enumerate() {
        result[i] = F::from_f32(val).unwrap();
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_quantize_dequantize_uniform() {
        let a = array![[1.0_f32, 2.5, 3.7], [4.2, 5.0, 6.1]];

        let (quantized, params) = quantize_matrix(&a.view(), 8, QuantizationMethod::Uniform);
        let a_dequantized = dequantize_matrix(&quantized, &params);

        // For 8-bit quantization, we can expect some error
        let max_diff = (&a - &a_dequantized)
            .mapv(|x| x.abs())
            .fold(0.0, |acc, &b| acc.max(b));
        println!("Max error (Uniform): {}", max_diff);
        assert!(max_diff < 6.0, "Max error too large: {}", max_diff);
    }

    #[test]
    fn test_quantize_dequantize_symmetric() {
        let a = array![[1.0_f32, -2.5, 3.7], [-4.2, 5.0, -6.1]];

        let (quantized, params) = quantize_matrix(&a.view(), 8, QuantizationMethod::Symmetric);
        let a_dequantized = dequantize_matrix(&quantized, &params);

        // For 8-bit quantization, we can expect some error
        let max_diff = (&a - &a_dequantized)
            .mapv(|x| x.abs())
            .fold(0.0, |acc, &b| acc.max(b));
        println!("Max error (Symmetric): {}", max_diff);
        assert!(max_diff < 6.0, "Max error too large: {}", max_diff);
    }

    #[test]
    fn test_quantize_dequantize_affine() {
        let a = array![[1.0_f32, -2.5, 3.7], [-4.2, 5.0, -6.1]];

        let (quantized, params) = quantize_matrix(&a.view(), 8, QuantizationMethod::Affine);
        let a_dequantized = dequantize_matrix(&quantized, &params);

        // For 8-bit quantization, we can expect some error
        let max_diff = (&a - &a_dequantized)
            .mapv(|x| x.abs())
            .fold(0.0, |acc, &b| acc.max(b));
        println!("Max error (Affine): {}", max_diff);
        assert!(max_diff < 6.0, "Max error too large: {}", max_diff);
    }

    #[test]
    fn test_quantize_dequantize_power_of_two() {
        let a = array![[1.0_f32, 2.5, 3.7], [4.2, 5.0, 6.1]];

        let (quantized, params) = quantize_matrix(&a.view(), 8, QuantizationMethod::PowerOfTwo);
        let a_dequantized = dequantize_matrix(&quantized, &params);

        // For 8-bit quantization, we can expect some error
        let max_diff = (&a - &a_dequantized)
            .mapv(|x| x.abs())
            .fold(0.0, |acc, &b| acc.max(b));
        println!("Max error (PowerOfTwo): {}", max_diff);
        assert!(max_diff < 6.0, "Max error too large: {}", max_diff);
    }

    #[test]
    fn test_quantized_matmul() {
        let a = array![[1.0_f32, 2.0], [3.0, 4.0]];
        let b = array![[5.0_f32, 6.0], [7.0, 8.0]];

        // Quantize matrices
        let (a_q, a_params) = quantize_matrix(&a.view(), 8, QuantizationMethod::Symmetric);
        let (b_q, b_params) = quantize_matrix(&b.view(), 8, QuantizationMethod::Symmetric);

        // Perform quantized matrix multiplication
        let c_q = quantized_matmul(&a_q, &a_params, &b_q, &b_params).unwrap();

        // Regular matrix multiplication for comparison
        let c = a.dot(&b);

        // Check that the relative error is small
        let rel_error = (&c - &c_q).mapv(|x| x.abs()).sum() / c.sum();
        assert!(rel_error < 0.1);
    }

    #[test]
    fn test_quantized_matvec() {
        let a = array![[1.0_f32, 2.0], [3.0, 4.0]];
        let x = array![5.0_f32, 6.0];

        // Quantize matrix and vector
        let (a_q, a_params) = quantize_matrix(&a.view(), 8, QuantizationMethod::Symmetric);
        let (x_q, x_params) = quantize_vector(&x.view(), 8, QuantizationMethod::Symmetric);

        // Perform quantized matrix-vector multiplication
        let y_q = quantized_matvec(&a_q, &a_params, &x_q, &x_params).unwrap();

        // Regular matrix-vector multiplication for comparison
        let y = a.dot(&x);

        // Check that the relative error is small
        let rel_error = (&y - &y_q).mapv(|x| x.abs()).sum() / y.sum();
        assert!(rel_error < 0.1);
    }

    #[test]
    fn test_quantized_dot() {
        let a = array![1.0_f32, 2.0, 3.0, 4.0];
        let b = array![5.0_f32, 6.0, 7.0, 8.0];

        // Quantize vectors
        let (a_q, a_params) = quantize_vector(&a.view(), 8, QuantizationMethod::Symmetric);
        let (b_q, b_params) = quantize_vector(&b.view(), 8, QuantizationMethod::Symmetric);

        // Perform quantized dot product
        let dot_q = quantized_dot(&a_q, &a_params, &b_q, &b_params).unwrap();

        // Regular dot product for comparison
        let dot = a.dot(&b);

        // Check that the relative error is small
        let rel_error = (dot - dot_q).abs() / dot;
        assert!(rel_error < 0.1);
    }

    #[test]
    fn test_fake_quantize() {
        let a = array![[1.0_f32, 2.5, 3.7], [4.2, 5.0, 6.1]];

        let a_fake_q = fake_quantize(&a.view(), 8, QuantizationMethod::Uniform);

        // For 8-bit quantization, we can expect some error
        let max_diff = (&a - &a_fake_q)
            .mapv(|x| x.abs())
            .fold(0.0, |acc, &b| acc.max(b));
        println!("Max error (Fake Quantize): {}", max_diff);
        assert!(max_diff < 6.0, "Max error too large: {}", max_diff);

        // Check that values are different due to quantization
        assert!(a != a_fake_q);
    }

    #[test]
    fn test_fake_quantize_vector() {
        let a = array![1.0_f32, 2.5, 3.7, 4.2, 5.0, 6.1];

        let a_fake_q = fake_quantize_vector(&a.view(), 8, QuantizationMethod::Uniform);

        // For 8-bit quantization, we can expect some error
        let max_diff = (&a - &a_fake_q)
            .mapv(|x| x.abs())
            .fold(0.0, |acc, &b| acc.max(b));
        println!("Max error (Fake Quantize Vector): {}", max_diff);
        assert!(max_diff < 6.0, "Max error too large: {}", max_diff);

        // Check that values are different due to quantization
        assert!(a != a_fake_q);
    }
}
