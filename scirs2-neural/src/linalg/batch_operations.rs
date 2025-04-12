//! Batch matrix operations optimized for neural networks
//!
//! This module provides specialized batch matrix operations that are commonly
//! used in neural network computations, such as batch matrix multiplication,
//! batch normalization operations, convolution operations, attention mechanisms,
//! and RNN/LSTM operations.

use ndarray::{Array, Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayView3, ArrayView4, 
              ArrayView, Axis, Dimension, Ix1, Ix2, Ix3, Ix4, Ix5, IxDyn, s};
use ndarray_stats::QuantileExt;
use num_traits::{Float, NumAssign, Zero, One};
use std::fmt::Debug;
use std::ops::{Add, Mul, Div, Sub};

use crate::error::{NeuralError, Result};

/// Perform batch matrix multiplication for neural network operations.
///
/// This function multiplies batches of matrices efficiently, which is common
/// in neural network computations like batch processing of fully connected layers.
///
/// # Arguments
///
/// * `a` - First batch of matrices with shape [batch_size, m, k]
/// * `b` - Second batch of matrices with shape [batch_size, k, n]
///
/// # Returns
///
/// * Result matrix with shape [batch_size, m, n]
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array, Ix3};
/// use scirs2_neural::linalg::batch_matmul;
///
/// // Create batch of 2x2x3 matrices (batch_size=2, m=2, k=3)
/// let a = Array::from_shape_vec(
///     (2, 2, 3),
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
/// ).unwrap();
///
/// // Create batch of 2x3x2 matrices (batch_size=2, k=3, n=2)
/// let b = Array::from_shape_vec(
///     (2, 3, 2),
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
/// ).unwrap();
///
/// // Result should be shape [2, 2, 2]
/// let c = batch_matmul(&a.view(), &b.view()).unwrap();
/// assert_eq!(c.shape(), &[2, 2, 2]);
/// ```
pub fn batch_matmul<F>(
    a: &ArrayView<F, Ix3>,
    b: &ArrayView<F, Ix3>,
) -> Result<Array<F, Ix3>>
where
    F: Float + Debug,
{
    // Check shape compatibility
    let batch_size_a = a.shape()[0];
    let batch_size_b = b.shape()[0];
    
    if batch_size_a != batch_size_b {
        return Err(NeuralError::ShapeError(format!(
            "Batch sizes don't match: {} vs {}",
            batch_size_a, batch_size_b
        )));
    }
    
    let m = a.shape()[1];
    let k_a = a.shape()[2];
    let k_b = b.shape()[1];
    let n = b.shape()[2];
    
    if k_a != k_b {
        return Err(NeuralError::ShapeError(format!(
            "Inner dimensions don't match: {} vs {}",
            k_a, k_b
        )));
    }
    
    // Initialize result array
    let mut result = Array::zeros((batch_size_a, m, n));
    
    // Perform batch matrix multiplication
    for batch in 0..batch_size_a {
        for i in 0..m {
            for j in 0..n {
                let mut sum = F::zero();
                for k in 0..k_a {
                    sum = sum + a[[batch, i, k]] * b[[batch, k, j]];
                }
                result[[batch, i, j]] = sum;
            }
        }
    }
    
    Ok(result)
}

/// Perform batch vector-matrix multiplication for neural network operations.
///
/// This is commonly used in RNN and attention mechanisms.
///
/// # Arguments
///
/// * `v` - Batch of vectors with shape [batch_size, k]
/// * `m` - Matrix with shape [k, n]
///
/// # Returns
///
/// * Result vectors with shape [batch_size, n]
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array, Ix2};
/// use scirs2_neural::linalg::batch_vecmat;
///
/// // Create batch of vectors [batch_size=2, k=3]
/// let v = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
///
/// // Create matrix [k=3, n=2]
/// let m = array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]];
///
/// // Result should be shape [2, 2]
/// let result = batch_vecmat(&v.view(), &m.view()).unwrap();
/// assert_eq!(result.shape(), &[2, 2]);
/// ```
pub fn batch_vecmat<F>(
    v: &ArrayView<F, Ix2>,
    m: &ArrayView<F, Ix2>,
) -> Result<Array<F, Ix2>>
where
    F: Float + Debug,
{
    // Check shape compatibility
    let batch_size = v.shape()[0];
    let k_v = v.shape()[1];
    let k_m = m.shape()[0];
    let n = m.shape()[1];
    
    if k_v != k_m {
        return Err(NeuralError::ShapeError(format!(
            "Dimensions don't match: {} vs {}",
            k_v, k_m
        )));
    }
    
    // Initialize result array
    let mut result = Array::zeros((batch_size, n));
    
    // Perform batch vector-matrix multiplication
    for batch in 0..batch_size {
        for j in 0..n {
            let mut sum = F::zero();
            for k in 0..k_v {
                sum = sum + v[[batch, k]] * m[[k, j]];
            }
            result[[batch, j]] = sum;
        }
    }
    
    Ok(result)
}

/// Perform batch normalization forward pass.
///
/// Normalizes each feature across the batch dimension to zero mean and unit variance,
/// then scales and shifts the result using gamma and beta parameters.
///
/// # Arguments
///
/// * `x` - Input tensor with shape [batch_size, features]
/// * `gamma` - Scale parameter with shape [features]
/// * `beta` - Shift parameter with shape [features]
/// * `eps` - Small constant for numerical stability
///
/// # Returns
///
/// * Normalized output, batch mean, and batch variance
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array1};
/// use scirs2_neural::linalg::batch_norm_forward;
///
/// // Input: batch_size=2, features=3
/// let x = array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
/// let gamma = Array1::ones(3);
/// let beta = Array1::zeros(3);
///
/// let (y, mean, var) = batch_norm_forward(&x.view(), &gamma.view(), &beta.view(), 1e-5).unwrap();
/// assert_eq!(y.shape(), &[2, 3]);
/// ```
pub fn batch_norm_forward<F>(
    x: &ArrayView2<F>,
    gamma: &ArrayView1<F>,
    beta: &ArrayView1<F>,
    eps: F,
) -> Result<(Array<F, Ix2>, Array<F, Ix1>, Array<F, Ix1>)>
where
    F: Float + Debug,
{
    // Check shape compatibility
    let batch_size = x.shape()[0];
    let features = x.shape()[1];
    
    if gamma.len() != features {
        return Err(NeuralError::ShapeError(format!(
            "gamma size ({}) doesn't match features ({})",
            gamma.len(), features
        )));
    }
    
    if beta.len() != features {
        return Err(NeuralError::ShapeError(format!(
            "beta size ({}) doesn't match features ({})",
            beta.len(), features
        )));
    }
    
    // Compute batch mean and variance
    let mut mean = Array1::zeros(features);
    let mut var = Array1::zeros(features);
    
    // Compute mean
    for j in 0..features {
        let mut sum = F::zero();
        for i in 0..batch_size {
            sum = sum + x[[i, j]];
        }
        mean[j] = sum / F::from(batch_size).unwrap();
    }
    
    // Compute variance
    for j in 0..features {
        let mut sum_sq = F::zero();
        for i in 0..batch_size {
            let diff = x[[i, j]] - mean[j];
            sum_sq = sum_sq + diff * diff;
        }
        var[j] = sum_sq / F::from(batch_size).unwrap();
    }
    
    // Normalize, scale, and shift
    let mut y = Array::zeros(x.dim());
    for j in 0..features {
        let std_dev = (var[j] + eps).sqrt();
        for i in 0..batch_size {
            let x_norm = (x[[i, j]] - mean[j]) / std_dev;
            y[[i, j]] = gamma[j] * x_norm + beta[j];
        }
    }
    
    Ok((y, mean, var))
}

/// Perform batch normalization backward pass.
///
/// Computes gradients with respect to the input and parameters.
///
/// # Arguments
///
/// * `dout` - Gradient from upstream with shape [batch_size, features]
/// * `x` - Input tensor with shape [batch_size, features]
/// * `gamma` - Scale parameter with shape [features]
/// * `mean` - Batch mean with shape [features]
/// * `var` - Batch variance with shape [features]
/// * `eps` - Small constant for numerical stability
///
/// # Returns
///
/// * Gradients with respect to input, gamma, and beta
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array1};
/// use scirs2_neural::linalg::{batch_norm_forward, batch_norm_backward};
///
/// // Input: batch_size=2, features=3
/// let x = array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
/// let gamma = Array1::ones(3);
/// let beta = Array1::zeros(3);
/// let eps = 1e-5;
///
/// let (y, mean, var) = batch_norm_forward(&x.view(), &gamma.view(), &beta.view(), eps).unwrap();
///
/// // Upstream gradient
/// let dout = array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]];
///
/// let (dx, dgamma, dbeta) = batch_norm_backward(
///     &dout.view(), &x.view(), &gamma.view(), &mean.view(), &var.view(), eps
/// ).unwrap();
///
/// assert_eq!(dx.shape(), &[2, 3]);
/// assert_eq!(dgamma.shape(), &[3]);
/// assert_eq!(dbeta.shape(), &[3]);
/// ```
pub fn batch_norm_backward<F>(
    dout: &ArrayView2<F>,
    x: &ArrayView2<F>,
    gamma: &ArrayView1<F>,
    mean: &ArrayView1<F>,
    var: &ArrayView1<F>,
    eps: F,
) -> Result<(Array<F, Ix2>, Array<F, Ix1>, Array<F, Ix1>)>
where
    F: Float + Debug,
{
    // Check shape compatibility
    let batch_size = x.shape()[0];
    let features = x.shape()[1];
    
    if dout.shape() != x.shape() {
        return Err(NeuralError::ShapeError(format!(
            "dout shape ({:?}) doesn't match x shape ({:?})",
            dout.shape(), x.shape()
        )));
    }
    
    if gamma.len() != features {
        return Err(NeuralError::ShapeError(format!(
            "gamma size ({}) doesn't match features ({})",
            gamma.len(), features
        )));
    }
    
    if mean.len() != features {
        return Err(NeuralError::ShapeError(format!(
            "mean size ({}) doesn't match features ({})",
            mean.len(), features
        )));
    }
    
    if var.len() != features {
        return Err(NeuralError::ShapeError(format!(
            "var size ({}) doesn't match features ({})",
            var.len(), features
        )));
    }
    
    // Compute gradients
    let mut dx = Array::zeros(x.dim());
    let mut dgamma = Array1::zeros(features);
    let mut dbeta = Array1::zeros(features);
    
    // Compute dgamma and dbeta
    for j in 0..features {
        let std_dev = (var[j] + eps).sqrt();
        let mut dgamma_j = F::zero();
        let mut dbeta_j = F::zero();
        
        for i in 0..batch_size {
            let x_norm = (x[[i, j]] - mean[j]) / std_dev;
            dgamma_j = dgamma_j + dout[[i, j]] * x_norm;
            dbeta_j = dbeta_j + dout[[i, j]];
        }
        
        dgamma[j] = dgamma_j;
        dbeta[j] = dbeta_j;
    }
    
    // Compute dx
    for j in 0..features {
        let std_dev = (var[j] + eps).sqrt();
        let inv_std = F::one() / std_dev;
        let inv_batch_size = F::one() / F::from(batch_size).unwrap();
        
        for i in 0..batch_size {
            let x_centered = x[[i, j]] - mean[j];
            
            // Gradient with respect to x_i directly from dout
            let dxhat = dout[[i, j]] * gamma[j];
            
            // Contribution to dx_i from the direct path
            let dx_direct = dxhat * inv_std;
            
            // Contribution to dx_i from the mean term
            let dx_mean = -inv_std;
            
            // Contribution to dx_i from the variance term
            let dx_var = -F::from(0.5).unwrap() * inv_std * inv_std * inv_std * x_centered;
            
            // Compute dvar/dx_i = 2(x_i - mean)/batch_size
            let dvar_dx = F::from(2.0).unwrap() * x_centered * inv_batch_size;
            
            // Compute dmean/dx_i = 1/batch_size
            let dmean_dx = inv_batch_size;
            
            // Compute total gradient contribution for df/dvar and df/dmean
            let mut dx_var_term = F::zero();
            let mut dx_mean_term = F::zero();
            
            for k in 0..batch_size {
                // Gradient of normalized x_k with respect to variance
                let dxhat_k = dout[[k, j]] * gamma[j];
                dx_var_term = dx_var_term + dxhat_k * dx_var;
                dx_mean_term = dx_mean_term + dxhat_k * dx_mean;
            }
            
            // Total gradient for x_i
            dx[[i, j]] = dx_direct + dvar_dx * dx_var_term + dmean_dx * dx_mean_term;
        }
    }
    
    Ok((dx, dgamma, dbeta))
}

/// Perform 2D convolution operation for convolutional neural networks.
///
/// This function applies multiple filters to a batch of input images,
/// which is a fundamental operation in convolutional neural networks.
///
/// # Arguments
///
/// * `input` - Input with shape [batch_size, in_channels, height, width]
/// * `filters` - Filters with shape [out_channels, in_channels, kernel_height, kernel_width]
/// * `stride` - Stride (vertical, horizontal)
/// * `padding` - Padding mode: "valid" (no padding) or "same" (padding to maintain size)
///
/// # Returns
///
/// * Result with shape [batch_size, out_channels, output_height, output_width]
///
/// # Examples
///
/// ```
/// use ndarray::{Array, Array4};
/// use scirs2_neural::linalg::conv2d;
///
/// // Create a batch of 2 images, each with 3 channels and 5x5 size
/// let input = Array4::<f32>::zeros((2, 3, 5, 5));
/// 
/// // Create 4 filters, each for 3 input channels and 3x3 kernel size
/// let filters = Array4::<f32>::zeros((4, 3, 3, 3));
///
/// // Apply convolution with stride 1 and 'valid' padding
/// let output = conv2d(&input.view(), &filters.view(), (1, 1), "valid").unwrap();
///
/// // Output shape should be [2, 4, 3, 3]
/// assert_eq!(output.shape(), &[2, 4, 3, 3]);
/// ```
pub fn conv2d<F>(
    input: &ArrayView4<F>,
    filters: &ArrayView4<F>,
    stride: (usize, usize),
    padding: &str,
) -> Result<Array4<F>>
where
    F: Float + Debug + NumAssign + Zero + One,
{
    // Extract dimensions
    let batch_size = input.shape()[0];
    let in_channels = input.shape()[1];
    let in_height = input.shape()[2];
    let in_width = input.shape()[3];
    
    let out_channels = filters.shape()[0];
    let filter_in_channels = filters.shape()[1];
    let filter_height = filters.shape()[2];
    let filter_width = filters.shape()[3];
    
    let stride_h = stride.0;
    let stride_w = stride.1;
    
    // Check shape compatibility
    if in_channels != filter_in_channels {
        return Err(NeuralError::Other(format!(
            "Input channels ({}) don't match filter input channels ({})",
            in_channels, filter_in_channels
        )));
    }
    
    // Calculate output dimensions based on padding
    let (pad_h, pad_w, out_height, out_width) = match padding {
        "valid" => {
            let out_height = (in_height - filter_height) / stride_h + 1;
            let out_width = (in_width - filter_width) / stride_w + 1;
            (0, 0, out_height, out_width)
        },
        "same" => {
            let out_height = (in_height + stride_h - 1) / stride_h;
            let out_width = (in_width + stride_w - 1) / stride_w;
            
            let pad_h_total = (out_height - 1) * stride_h + filter_height - in_height;
            let pad_w_total = (out_width - 1) * stride_w + filter_width - in_width;
            
            let pad_h = pad_h_total / 2;
            let pad_w = pad_w_total / 2;
            
            (pad_h, pad_w, out_height, out_width)
        },
        _ => {
            return Err(NeuralError::Other(format!(
                "Invalid padding mode: {}. Should be 'valid' or 'same'.",
                padding
            )));
        }
    };
    
    // Initialize result array
    let mut output = Array4::zeros((batch_size, out_channels, out_height, out_width));
    
    // Perform convolution
    for b in 0..batch_size {
        for oc in 0..out_channels {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let h_start = oh * stride_h as usize;
                    let w_start = ow * stride_w as usize;
                    
                    let mut sum = F::zero();
                    
                    for ic in 0..in_channels {
                        for fh in 0..filter_height {
                            for fw in 0..filter_width {
                                let ih = h_start + fh;
                                let iw = w_start + fw;
                                
                                // Handle padding by skipping out-of-bounds inputs
                                if ih >= pad_h && ih < in_height + pad_h &&
                                   iw >= pad_w && iw < in_width + pad_w {
                                    let input_h = ih - pad_h;
                                    let input_w = iw - pad_w;
                                    
                                    if input_h < in_height && input_w < in_width {
                                        sum = sum + input[[b, ic, input_h, input_w]] * 
                                              filters[[oc, ic, fh, fw]];
                                    }
                                }
                            }
                        }
                    }
                    
                    output[[b, oc, oh, ow]] = sum;
                }
            }
        }
    }
    
    Ok(output)
}

/// Perform max pooling operation for convolutional neural networks.
///
/// This function downsamples the input by taking the maximum value in each pooling window.
///
/// # Arguments
///
/// * `input` - Input with shape [batch_size, channels, height, width]
/// * `pool_size` - Size of the pooling window (height, width)
/// * `stride` - Stride (vertical, horizontal)
///
/// # Returns
///
/// * Result with shape [batch_size, channels, output_height, output_width]
/// * Indices of maximum values used for backpropagation
///
/// # Examples
///
/// ```
/// use ndarray::Array4;
/// use scirs2_neural::linalg::max_pool2d;
///
/// // Create a batch of 2 feature maps, each with 3 channels and 4x4 size
/// let input = Array4::<f32>::zeros((2, 3, 4, 4));
///
/// // Apply max pooling with 2x2 pool size and stride 2
/// let (output, indices) = max_pool2d(&input.view(), (2, 2), (2, 2)).unwrap();
///
/// // Output shape should be [2, 3, 2, 2]
/// assert_eq!(output.shape(), &[2, 3, 2, 2]);
/// ```
pub fn max_pool2d<F>(
    input: &ArrayView4<F>,
    pool_size: (usize, usize),
    stride: (usize, usize),
) -> Result<(Array4<F>, Array4<usize>)>
where
    F: Float + Debug + PartialOrd,
{
    // Extract dimensions
    let batch_size = input.shape()[0];
    let channels = input.shape()[1];
    let in_height = input.shape()[2];
    let in_width = input.shape()[3];
    
    let pool_h = pool_size.0;
    let pool_w = pool_size.1;
    let stride_h = stride.0;
    let stride_w = stride.1;
    
    // Calculate output dimensions
    let out_height = (in_height - pool_h) / stride_h + 1;
    let out_width = (in_width - pool_w) / stride_w + 1;
    
    // Initialize result arrays
    let mut output = Array4::zeros((batch_size, channels, out_height, out_width));
    let mut indices = Array4::zeros((batch_size, channels, out_height, out_width));
    
    // Perform max pooling
    for b in 0..batch_size {
        for c in 0..channels {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let h_start = oh * stride_h;
                    let w_start = ow * stride_w;
                    
                    let mut max_val = F::neg_infinity();
                    let mut max_idx = 0;
                    
                    for ph in 0..pool_h {
                        for pw in 0..pool_w {
                            let ih = h_start + ph;
                            let iw = w_start + pw;
                            
                            if ih < in_height && iw < in_width {
                                let val = input[[b, c, ih, iw]];
                                if val > max_val {
                                    max_val = val;
                                    max_idx = ph * pool_w + pw;
                                }
                            }
                        }
                    }
                    
                    output[[b, c, oh, ow]] = max_val;
                    indices[[b, c, oh, ow]] = max_idx;
                }
            }
        }
    }
    
    Ok((output, indices))
}

/// Calculate attention scores and apply them to values in an attention mechanism.
///
/// This function implements the core of the attention mechanism used in Transformer models.
///
/// # Arguments
///
/// * `queries` - Query tensor with shape [batch_size, seq_len_q, depth_q]
/// * `keys` - Key tensor with shape [batch_size, seq_len_k, depth_k]
/// * `values` - Value tensor with shape [batch_size, seq_len_k, depth_v]
/// * `mask` - Optional mask tensor with shape [batch_size, seq_len_q, seq_len_k]
///
/// # Returns
///
/// * Attention output with shape [batch_size, seq_len_q, depth_v]
/// * Attention weights with shape [batch_size, seq_len_q, seq_len_k]
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array3};
/// use scirs2_neural::linalg::scaled_dot_product_attention;
///
/// // Create queries, keys, and values for batch_size=2, with sequences of 3 and 4 elements
/// let queries = Array3::<f32>::zeros((2, 3, 8));  // [batch, seq_len_q, depth]
/// let keys = Array3::<f32>::zeros((2, 4, 8));     // [batch, seq_len_k, depth]
/// let values = Array3::<f32>::zeros((2, 4, 16));  // [batch, seq_len_k, depth_v]
///
/// // Calculate attention (no mask)
/// let (output, weights) = scaled_dot_product_attention(
///     &queries.view(), &keys.view(), &values.view(), None
/// ).unwrap();
///
/// // Output shape should be [2, 3, 16]
/// assert_eq!(output.shape(), &[2, 3, 16]);
/// // Weights shape should be [2, 3, 4]
/// assert_eq!(weights.shape(), &[2, 3, 4]);
/// ```
pub fn scaled_dot_product_attention<F>(
    queries: &ArrayView3<F>,
    keys: &ArrayView3<F>,
    values: &ArrayView3<F>,
    mask: Option<&ArrayView3<F>>,
) -> Result<(Array3<F>, Array3<F>)>
where
    F: Float + Debug + NumAssign,
{
    // Extract dimensions
    let batch_size = queries.shape()[0];
    let seq_len_q = queries.shape()[1];
    let depth_q = queries.shape()[2];
    
    let seq_len_k = keys.shape()[1];
    let depth_k = keys.shape()[2];
    
    let depth_v = values.shape()[2];
    
    // Check shape compatibility
    if batch_size != keys.shape()[0] || batch_size != values.shape()[0] {
        return Err(NeuralError::Other(format!(
            "Batch sizes don't match: queries ({}), keys ({}), values ({})",
            batch_size, keys.shape()[0], values.shape()[0]
        )));
    }
    
    if depth_q != depth_k {
        return Err(NeuralError::Other(format!(
            "Query depth ({}) doesn't match key depth ({})",
            depth_q, depth_k
        )));
    }
    
    if seq_len_k != values.shape()[1] {
        return Err(NeuralError::Other(format!(
            "Key sequence length ({}) doesn't match value sequence length ({})",
            seq_len_k, values.shape()[1]
        )));
    }
    
    // Calculate scaling factor
    let scale = F::from(depth_k as f64).unwrap().sqrt().recip();
    
    // Initialize attention scores
    let mut scores = Array3::zeros((batch_size, seq_len_q, seq_len_k));
    
    // Calculate attention scores
    for b in 0..batch_size {
        for q in 0..seq_len_q {
            for k in 0..seq_len_k {
                let mut sum = F::zero();
                for d in 0..depth_k {
                    sum = sum + queries[[b, q, d]] * keys[[b, k, d]];
                }
                scores[[b, q, k]] = sum * scale;
            }
        }
    }
    
    // Apply mask if provided
    if let Some(mask_array) = mask {
        if mask_array.shape() != scores.shape() {
            return Err(NeuralError::Other(format!(
                "Mask shape {:?} doesn't match scores shape {:?}",
                mask_array.shape(), scores.shape()
            )));
        }
        
        let neg_inf = F::neg_infinity();
        for b in 0..batch_size {
            for q in 0..seq_len_q {
                for k in 0..seq_len_k {
                    if mask_array[[b, q, k]] == F::zero() {
                        scores[[b, q, k]] = neg_inf;
                    }
                }
            }
        }
    }
    
    // Apply softmax to get attention weights
    let mut weights = Array3::zeros(scores.dim());
    for b in 0..batch_size {
        for q in 0..seq_len_q {
            // Find max value for numerical stability
            let mut max_val = F::neg_infinity();
            for k in 0..seq_len_k {
                if scores[[b, q, k]] > max_val {
                    max_val = scores[[b, q, k]];
                }
            }
            
            // Compute exponentials and sum
            let mut exp_sum = F::zero();
            for k in 0..seq_len_k {
                let exp_val = (scores[[b, q, k]] - max_val).exp();
                weights[[b, q, k]] = exp_val;
                exp_sum = exp_sum + exp_val;
            }
            
            // Normalize
            if exp_sum > F::zero() {
                for k in 0..seq_len_k {
                    weights[[b, q, k]] = weights[[b, q, k]] / exp_sum;
                }
            }
        }
    }
    
    // Calculate attention output
    let mut output = Array3::zeros((batch_size, seq_len_q, depth_v));
    for b in 0..batch_size {
        for q in 0..seq_len_q {
            for d in 0..depth_v {
                let mut sum = F::zero();
                for k in 0..seq_len_k {
                    sum = sum + weights[[b, q, k]] * values[[b, k, d]];
                }
                output[[b, q, d]] = sum;
            }
        }
    }
    
    Ok((output, weights))
}

/// Apply multi-head attention mechanism.
///
/// This function implements the multi-head attention used in Transformer models.
///
/// # Arguments
///
/// * `queries` - Query tensor with shape [batch_size, seq_len_q, model_dim]
/// * `keys` - Key tensor with shape [batch_size, seq_len_k, model_dim]
/// * `values` - Value tensor with shape [batch_size, seq_len_k, model_dim]
/// * `w_q` - Query weights with shape [num_heads, model_dim, head_dim]
/// * `w_k` - Key weights with shape [num_heads, model_dim, head_dim]
/// * `w_v` - Value weights with shape [num_heads, model_dim, head_dim]
/// * `w_o` - Output weights with shape [model_dim, num_heads * head_dim]
/// * `mask` - Optional mask tensor with shape [batch_size, seq_len_q, seq_len_k]
///
/// # Returns
///
/// * Multi-head attention output with shape [batch_size, seq_len_q, model_dim]
///
/// # Examples
///
/// ```
/// use ndarray::{Array2, Array3, Array4};
/// use scirs2_neural::linalg::multi_head_attention;
///
/// // Initialize parameters for 2-head attention with model_dim=64 and head_dim=32
/// let batch_size = 2;
/// let seq_len_q = 3;
/// let seq_len_k = 4;
/// let model_dim = 64;
/// let head_dim = 32;
/// let num_heads = 2;
///
/// let queries = Array3::<f32>::zeros((batch_size, seq_len_q, model_dim));
/// let keys = Array3::<f32>::zeros((batch_size, seq_len_k, model_dim));
/// let values = Array3::<f32>::zeros((batch_size, seq_len_k, model_dim));
///
/// let w_q = Array3::<f32>::zeros((num_heads, model_dim, head_dim));
/// let w_k = Array3::<f32>::zeros((num_heads, model_dim, head_dim));
/// let w_v = Array3::<f32>::zeros((num_heads, model_dim, head_dim));
/// let w_o = Array2::<f32>::zeros((model_dim, num_heads * head_dim));
///
/// // Calculate multi-head attention
/// let output = multi_head_attention(
///     &queries.view(), &keys.view(), &values.view(),
///     &w_q.view(), &w_k.view(), &w_v.view(), &w_o.view(),
///     None
/// ).unwrap();
///
/// // Output shape should be [2, 3, 64]
/// assert_eq!(output.shape(), &[2, 3, 64]);
/// ```
pub fn multi_head_attention<F>(
    queries: &ArrayView3<F>,
    keys: &ArrayView3<F>,
    values: &ArrayView3<F>,
    w_q: &ArrayView3<F>,
    w_k: &ArrayView3<F>,
    w_v: &ArrayView3<F>,
    w_o: &ArrayView2<F>,
    mask: Option<&ArrayView3<F>>,
) -> Result<Array3<F>>
where
    F: Float + Debug + NumAssign,
{
    // Extract dimensions
    let batch_size = queries.shape()[0];
    let seq_len_q = queries.shape()[1];
    let model_dim = queries.shape()[2];
    
    let seq_len_k = keys.shape()[1];
    
    let num_heads = w_q.shape()[0];
    let head_dim = w_q.shape()[2];
    
    // Check shape compatibility
    if w_q.shape()[1] != model_dim || w_k.shape()[1] != model_dim || w_v.shape()[1] != model_dim {
        return Err(NeuralError::Other(format!(
            "Weight matrices first dimension must match model_dim ({})",
            model_dim
        )));
    }
    
    if w_k.shape()[0] != num_heads || w_v.shape()[0] != num_heads {
        return Err(NeuralError::Other(format!(
            "Number of heads doesn't match across weight matrices: {} vs {} vs {}",
            w_q.shape()[0], w_k.shape()[0], w_v.shape()[0]
        )));
    }
    
    if w_k.shape()[2] != head_dim || w_v.shape()[2] != head_dim {
        return Err(NeuralError::Other(format!(
            "Head dimensions don't match across weight matrices: {} vs {} vs {}",
            w_q.shape()[2], w_k.shape()[2], w_v.shape()[2]
        )));
    }
    
    if w_o.shape() != &[model_dim, num_heads * head_dim] {
        return Err(NeuralError::Other(format!(
            "Output weight matrix shape {:?} doesn't match expected ({}, {})",
            w_o.shape(), model_dim, num_heads * head_dim
        )));
    }
    
    // Project queries, keys, and values for each head
    let mut q_proj = Array4::zeros((batch_size, num_heads, seq_len_q, head_dim));
    let mut k_proj = Array4::zeros((batch_size, num_heads, seq_len_k, head_dim));
    let mut v_proj = Array4::zeros((batch_size, num_heads, seq_len_k, head_dim));
    
    // Compute projections
    for b in 0..batch_size {
        for h in 0..num_heads {
            for q in 0..seq_len_q {
                for d_out in 0..head_dim {
                    let mut sum = F::zero();
                    for d_in in 0..model_dim {
                        sum = sum + queries[[b, q, d_in]] * w_q[[h, d_in, d_out]];
                    }
                    q_proj[[b, h, q, d_out]] = sum;
                }
            }
            
            for k in 0..seq_len_k {
                for d_out in 0..head_dim {
                    let mut sum = F::zero();
                    for d_in in 0..model_dim {
                        sum = sum + keys[[b, k, d_in]] * w_k[[h, d_in, d_out]];
                    }
                    k_proj[[b, h, k, d_out]] = sum;
                }
            }
            
            for v in 0..seq_len_k {
                for d_out in 0..head_dim {
                    let mut sum = F::zero();
                    for d_in in 0..model_dim {
                        sum = sum + values[[b, v, d_in]] * w_v[[h, d_in, d_out]];
                    }
                    v_proj[[b, h, v, d_out]] = sum;
                }
            }
        }
    }
    
    // Apply scaled dot-product attention for each head
    let mut head_outputs = Array4::zeros((batch_size, num_heads, seq_len_q, head_dim));
    
    for b in 0..batch_size {
        for h in 0..num_heads {
            // Calculate scaling factor
            let scale = F::from(head_dim as f64).unwrap().sqrt().recip();
            
            // Initialize attention scores
            let mut scores = Array2::zeros((seq_len_q, seq_len_k));
            
            // Calculate attention scores
            for q in 0..seq_len_q {
                for k in 0..seq_len_k {
                    let mut sum = F::zero();
                    for d in 0..head_dim {
                        sum = sum + q_proj[[b, h, q, d]] * k_proj[[b, h, k, d]];
                    }
                    scores[[q, k]] = sum * scale;
                }
            }
            
            // Apply mask if provided
            if let Some(mask_array) = mask {
                let neg_inf = F::neg_infinity();
                for q in 0..seq_len_q {
                    for k in 0..seq_len_k {
                        if mask_array[[b, q, k]] == F::zero() {
                            scores[[q, k]] = neg_inf;
                        }
                    }
                }
            }
            
            // Apply softmax to get attention weights
            let mut weights = Array2::zeros(scores.dim());
            for q in 0..seq_len_q {
                // Find max value for numerical stability
                let mut max_val = F::neg_infinity();
                for k in 0..seq_len_k {
                    if scores[[q, k]] > max_val {
                        max_val = scores[[q, k]];
                    }
                }
                
                // Compute exponentials and sum
                let mut exp_sum = F::zero();
                for k in 0..seq_len_k {
                    let exp_val = (scores[[q, k]] - max_val).exp();
                    weights[[q, k]] = exp_val;
                    exp_sum = exp_sum + exp_val;
                }
                
                // Normalize
                if exp_sum > F::zero() {
                    for k in 0..seq_len_k {
                        weights[[q, k]] = weights[[q, k]] / exp_sum;
                    }
                }
            }
            
            // Calculate attention output for this head
            for q in 0..seq_len_q {
                for d in 0..head_dim {
                    let mut sum = F::zero();
                    for k in 0..seq_len_k {
                        sum = sum + weights[[q, k]] * v_proj[[b, h, k, d]];
                    }
                    head_outputs[[b, h, q, d]] = sum;
                }
            }
        }
    }
    
    // Concatenate head outputs and apply final projection
    let mut output = Array3::zeros((batch_size, seq_len_q, model_dim));
    
    for b in 0..batch_size {
        for q in 0..seq_len_q {
            for d_out in 0..model_dim {
                let mut sum = F::zero();
                for h in 0..num_heads {
                    for d_in in 0..head_dim {
                        let concat_idx = h * head_dim + d_in;
                        sum = sum + head_outputs[[b, h, q, d_in]] * w_o[[d_out, concat_idx]];
                    }
                }
                output[[b, q, d_out]] = sum;
            }
        }
    }
    
    Ok(output)
}

/// Applies a single LSTM cell update.
///
/// This function implements the computation performed by a single LSTM cell,
/// which is a fundamental building block for recurrent neural networks.
///
/// # Arguments
///
/// * `x` - Input tensor with shape [batch_size, input_size]
/// * `h_prev` - Previous hidden state with shape [batch_size, hidden_size]
/// * `c_prev` - Previous cell state with shape [batch_size, hidden_size]
/// * `w_ih` - Input-hidden weights with shape [4*hidden_size, input_size]
/// * `w_hh` - Hidden-hidden weights with shape [4*hidden_size, hidden_size]
/// * `b_ih` - Input-hidden bias with shape [4*hidden_size]
/// * `b_hh` - Hidden-hidden bias with shape [4*hidden_size]
///
/// # Returns
///
/// * New hidden state with shape [batch_size, hidden_size]
/// * New cell state with shape [batch_size, hidden_size]
///
/// # Examples
///
/// ```
/// use ndarray::{Array1, Array2};
/// use scirs2_neural::linalg::lstm_cell;
///
/// // Initialize parameters for LSTM cell with batch_size=2, input_size=3, hidden_size=4
/// let batch_size = 2;
/// let input_size = 3;
/// let hidden_size = 4;
///
/// let x = Array2::<f32>::zeros((batch_size, input_size));
/// let h_prev = Array2::<f32>::zeros((batch_size, hidden_size));
/// let c_prev = Array2::<f32>::zeros((batch_size, hidden_size));
///
/// let w_ih = Array2::<f32>::zeros((4 * hidden_size, input_size));
/// let w_hh = Array2::<f32>::zeros((4 * hidden_size, hidden_size));
/// let b_ih = Array1::<f32>::zeros(4 * hidden_size);
/// let b_hh = Array1::<f32>::zeros(4 * hidden_size);
///
/// // Calculate LSTM cell
/// let (h_new, c_new) = lstm_cell(
///     &x.view(), &h_prev.view(), &c_prev.view(),
///     &w_ih.view(), &w_hh.view(), &b_ih.view(), &b_hh.view()
/// ).unwrap();
///
/// // Output shapes should be [2, 4]
/// assert_eq!(h_new.shape(), &[2, 4]);
/// assert_eq!(c_new.shape(), &[2, 4]);
/// ```
pub fn lstm_cell<F>(
    x: &ArrayView2<F>,
    h_prev: &ArrayView2<F>,
    c_prev: &ArrayView2<F>,
    w_ih: &ArrayView2<F>,
    w_hh: &ArrayView2<F>,
    b_ih: &ArrayView1<F>,
    b_hh: &ArrayView1<F>,
) -> Result<(Array2<F>, Array2<F>)>
where
    F: Float + Debug + NumAssign,
{
    // Extract dimensions
    let batch_size = x.shape()[0];
    let input_size = x.shape()[1];
    let hidden_size = h_prev.shape()[1];
    
    // Check shape compatibility
    if h_prev.shape()[0] != batch_size || c_prev.shape()[0] != batch_size {
        return Err(NeuralError::Other(format!(
            "Batch sizes don't match: x ({}), h_prev ({}), c_prev ({})",
            batch_size, h_prev.shape()[0], c_prev.shape()[0]
        )));
    }
    
    if c_prev.shape()[1] != hidden_size {
        return Err(NeuralError::Other(format!(
            "Cell state dimension ({}) doesn't match hidden state dimension ({})",
            c_prev.shape()[1], hidden_size
        )));
    }
    
    if w_ih.shape() != &[4 * hidden_size, input_size] {
        return Err(NeuralError::Other(format!(
            "Input-hidden weight matrix shape {:?} doesn't match expected ({}, {})",
            w_ih.shape(), 4 * hidden_size, input_size
        )));
    }
    
    if w_hh.shape() != &[4 * hidden_size, hidden_size] {
        return Err(NeuralError::Other(format!(
            "Hidden-hidden weight matrix shape {:?} doesn't match expected ({}, {})",
            w_hh.shape(), 4 * hidden_size, hidden_size
        )));
    }
    
    if b_ih.len() != 4 * hidden_size || b_hh.len() != 4 * hidden_size {
        return Err(NeuralError::Other(format!(
            "Bias dimensions don't match: b_ih ({}), b_hh ({}), expected ({})",
            b_ih.len(), b_hh.len(), 4 * hidden_size
        )));
    }
    
    // Initialize new hidden and cell states
    let mut h_new = Array2::zeros((batch_size, hidden_size));
    let mut c_new = Array2::zeros((batch_size, hidden_size));
    
    // Compute gates for each batch
    for b in 0..batch_size {
        // Compute input, forget, cell, and output gates
        let mut gates = Array1::zeros(4 * hidden_size);
        
        // Apply input-hidden transformation
        for g in 0..4 * hidden_size {
            let mut sum = F::zero();
            for i in 0..input_size {
                sum = sum + w_ih[[g, i]] * x[[b, i]];
            }
            gates[g] = sum + b_ih[g];
        }
        
        // Apply hidden-hidden transformation
        for g in 0..4 * hidden_size {
            let mut sum = F::zero();
            for h in 0..hidden_size {
                sum = sum + w_hh[[g, h]] * h_prev[[b, h]];
            }
            gates[g] = gates[g] + sum + b_hh[g];
        }
        
        // Split gates into slices
        let i_gate_start = 0;
        let f_gate_start = hidden_size;
        let g_gate_start = 2 * hidden_size;
        let o_gate_start = 3 * hidden_size;
        
        for h in 0..hidden_size {
            // Apply sigmoid activation to input, forget, and output gates
            let i_gate = sigmoid(gates[i_gate_start + h]);
            let f_gate = sigmoid(gates[f_gate_start + h]);
            let g_gate = gates[g_gate_start + h].tanh();  // Apply tanh to cell input
            let o_gate = sigmoid(gates[o_gate_start + h]);
            
            // Update cell state
            c_new[[b, h]] = f_gate * c_prev[[b, h]] + i_gate * g_gate;
            
            // Update hidden state
            h_new[[b, h]] = o_gate * c_new[[b, h]].tanh();
        }
    }
    
    Ok((h_new, c_new))
}

/// Helper function to apply the sigmoid activation
fn sigmoid<F>(x: F) -> F
where
    F: Float,
{
    let one = F::one();
    one / (one + (-x).exp())
}

/// Compute gradients for batch matrix operations efficiently.
///
/// This function computes the gradient of C = A @ B with respect to A,
/// given the gradient of the loss with respect to C.
///
/// # Arguments
///
/// * `grad_output` - Gradient of loss with respect to C with shape [batch_size, m, n]
/// * `a` - First batch of matrices with shape [batch_size, m, k]
/// * `b` - Second batch of matrices with shape [batch_size, k, n]
/// * `compute_grad_a` - Whether to compute gradient with respect to A
/// * `compute_grad_b` - Whether to compute gradient with respect to B
///
/// # Returns
///
/// * Tuple of (grad_a, grad_b), gradients with respect to A and B
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array3};
/// use scirs2_neural::linalg::batch_matmul_backward;
///
/// // Create batch of 2x2x3 matrices (batch_size=2, m=2, k=3)
/// let a = Array3::<f32>::ones((2, 2, 3));
///
/// // Create batch of 2x3x2 matrices (batch_size=2, k=3, n=2)
/// let b = Array3::<f32>::ones((2, 3, 2));
///
/// // Gradient of loss with respect to C (shape [2, 2, 2])
/// let grad_output = Array3::<f32>::ones((2, 2, 2));
///
/// // Compute gradients
/// let (grad_a, grad_b) = batch_matmul_backward(
///     &grad_output.view(), &a.view(), &b.view(), true, true
/// ).unwrap();
///
/// // Gradient shapes should match input shapes
/// assert_eq!(grad_a.shape(), &[2, 2, 3]);
/// assert_eq!(grad_b.shape(), &[2, 3, 2]);
/// ```
pub fn batch_matmul_backward<F>(
    grad_output: &ArrayView<F, Ix3>,
    a: &ArrayView<F, Ix3>,
    b: &ArrayView<F, Ix3>,
    compute_grad_a: bool,
    compute_grad_b: bool,
) -> Result<(Option<Array<F, Ix3>>, Option<Array<F, Ix3>>)>
where
    F: Float + Debug + NumAssign,
{
    // Extract dimensions
    let batch_size = grad_output.shape()[0];
    let m = grad_output.shape()[1];
    let n = grad_output.shape()[2];
    
    let k_a = a.shape()[2];
    let k_b = b.shape()[1];
    
    // Check shape compatibility
    if a.shape()[0] != batch_size || b.shape()[0] != batch_size {
        return Err(NeuralError::Other(format!(
            "Batch sizes don't match: grad_output ({}), a ({}), b ({})",
            batch_size, a.shape()[0], b.shape()[0]
        )));
    }
    
    if a.shape()[1] != m {
        return Err(NeuralError::Other(format!(
            "First matrix rows ({}) don't match grad_output rows ({})",
            a.shape()[1], m
        )));
    }
    
    if b.shape()[2] != n {
        return Err(NeuralError::Other(format!(
            "Second matrix columns ({}) don't match grad_output columns ({})",
            b.shape()[2], n
        )));
    }
    
    if k_a != k_b {
        return Err(NeuralError::Other(format!(
            "Inner dimensions don't match: a cols ({}) vs b rows ({})",
            k_a, k_b
        )));
    }
    
    // Initialize gradients
    let mut grad_a = None;
    let mut grad_b = None;
    
    if compute_grad_a {
        let mut da = Array::zeros((batch_size, m, k_a));
        
        // Compute gradient for A: dL/dA = dL/dC @ B^T
        for batch in 0..batch_size {
            for i in 0..m {
                for k in 0..k_a {
                    let mut sum = F::zero();
                    for j in 0..n {
                        sum = sum + grad_output[[batch, i, j]] * b[[batch, k, j]];
                    }
                    da[[batch, i, k]] = sum;
                }
            }
        }
        
        grad_a = Some(da);
    }
    
    if compute_grad_b {
        let mut db = Array::zeros((batch_size, k_b, n));
        
        // Compute gradient for B: dL/dB = A^T @ dL/dC
        for batch in 0..batch_size {
            for k in 0..k_b {
                for j in 0..n {
                    let mut sum = F::zero();
                    for i in 0..m {
                        sum = sum + a[[batch, i, k]] * grad_output[[batch, i, j]];
                    }
                    db[[batch, k, j]] = sum;
                }
            }
        }
        
        grad_b = Some(db);
    }
    
    Ok((grad_a, grad_b))
}

/// Perform layer normalization forward pass.
///
/// Normalizes each input across the feature dimension (last dimension)
/// to zero mean and unit variance, then scales and shifts the result using
/// gamma and beta parameters.
///
/// # Arguments
///
/// * `x` - Input tensor with shape [batch_size, sequence_length, features]
/// * `gamma` - Scale parameter with shape [features]
/// * `beta` - Shift parameter with shape [features]
/// * `eps` - Small constant for numerical stability
///
/// # Returns
///
/// * Normalized output
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array1, Array3};
/// use scirs2_neural::linalg::layer_norm;
///
/// // Input: batch_size=2, seq_len=1, features=3
/// let x = array![[[0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6]]];
/// let gamma = Array1::ones(3);
/// let beta = Array1::zeros(3);
///
/// let y = layer_norm(&x.view(), &gamma.view(), &beta.view(), 1e-5).unwrap();
/// assert_eq!(y.shape(), &[2, 1, 3]);
/// ```
pub fn layer_norm<F>(
    x: &ArrayView3<F>,
    gamma: &ArrayView1<F>,
    beta: &ArrayView1<F>,
    eps: F,
) -> Result<Array3<F>>
where
    F: Float + Debug + NumAssign,
{
    // Extract dimensions
    let batch_size = x.shape()[0];
    let seq_len = x.shape()[1];
    let features = x.shape()[2];
    
    // Check shape compatibility
    if gamma.len() != features {
        return Err(NeuralError::Other(format!(
            "gamma size ({}) doesn't match features ({})",
            gamma.len(), features
        )));
    }
    
    if beta.len() != features {
        return Err(NeuralError::Other(format!(
            "beta size ({}) doesn't match features ({})",
            beta.len(), features
        )));
    }
    
    // Initialize result array
    let mut y = Array3::zeros(x.dim());
    
    // Compute layer normalization for each batch and sequence position
    for b in 0..batch_size {
        for s in 0..seq_len {
            // Compute mean
            let mut mean = F::zero();
            for j in 0..features {
                mean = mean + x[[b, s, j]];
            }
            mean = mean / F::from(features).unwrap();
            
            // Compute variance
            let mut var = F::zero();
            for j in 0..features {
                let diff = x[[b, s, j]] - mean;
                var = var + diff * diff;
            }
            var = var / F::from(features).unwrap();
            
            // Apply normalization, scaling, and shifting
            let std_dev = (var + eps).sqrt();
            for j in 0..features {
                let x_norm = (x[[b, s, j]] - mean) / std_dev;
                y[[b, s, j]] = gamma[j] * x_norm + beta[j];
            }
        }
    }
    
    Ok(y)
}

/// Apply dropout during training phase.
///
/// Randomly sets elements of the input tensor to zero with probability p.
///
/// # Arguments
///
/// * `x` - Input tensor of any shape
/// * `p` - Dropout probability (between 0 and 1)
/// * `training` - Whether to apply dropout (true) or not (false)
///
/// # Returns
///
/// * Output tensor with dropped values
/// * Dropout mask (1 for kept elements, 0 for dropped)
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use scirs2_neural::linalg::dropout;
/// use rand::Rng;
///
/// // Input tensor
/// let x = Array2::<f32>::ones((2, 3));
/// let mut rng = rand::thread_rng();
/// 
/// // During training
/// let (y_train, mask) = dropout(&x.view(), 0.5, true, &mut rng).unwrap();
/// assert_eq!(y_train.shape(), &[2, 3]);
/// 
/// // During inference
/// let (y_test, _) = dropout(&x.view(), 0.5, false, &mut rng).unwrap();
/// // During inference, values are scaled by 1/(1-p)
/// for &val in y_test.iter() {
///     assert!((val - 1.0).abs() < 1e-6);
/// }
/// ```
pub fn dropout<F, D, R>(
    x: &ArrayView<F, D>,
    p: F,
    training: bool,
    rng: &mut R,
) -> Result<(Array<F, D>, Array<F, D>)>
where
    F: Float + Debug + NumAssign,
    D: Dimension,
    R: rand::Rng,
{
    // Check dropout probability
    if p < F::zero() || p >= F::one() {
        return Err(NeuralError::Other(format!(
            "Dropout probability ({:?}) must be between 0 and 1",
            p
        )));
    }
    
    let mut output = Array::zeros(x.dim());
    let mut mask = Array::ones(x.dim());
    
    if training && p > F::zero() {
        // Create dropout mask during training
        for (i, val) in mask.iter_mut().enumerate() {
            let rand_val: f64 = rng.gen();
            if rand_val < p.to_f64().unwrap() {
                *val = F::zero();
            } else {
                *val = F::one() / (F::one() - p);
            }
        }
        
        // Apply mask
        for (i, val) in x.iter().enumerate() {
            let idx = output.dim().default_index_for_elem(i);
            output[idx.clone()] = *val * mask[idx];
        }
    } else {
        // During inference, just copy the input
        for (i, val) in x.iter().enumerate() {
            let idx = output.dim().default_index_for_elem(i);
            output[idx] = *val;
        }
    }
    
    Ok((output, mask))
}

/// Compute log-softmax function along the specified axis.
///
/// Computes a numerically stable version of log(softmax(x)),
/// which is useful for classification models and cross-entropy loss.
///
/// # Arguments
///
/// * `x` - Input tensor of any shape
/// * `axis` - Axis along which to apply log-softmax (default: last axis)
///
/// # Returns
///
/// * Output tensor with log-softmax applied
///
/// # Examples
///
/// ```
/// use ndarray::{array, Axis};
/// use scirs2_neural::linalg::log_softmax;
///
/// // Input: batch_size=2, classes=3
/// let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
///
/// // Apply log-softmax along class dimension (axis=1)
/// let y = log_softmax(&x.view(), Axis(1)).unwrap();
/// assert_eq!(y.shape(), &[2, 3]);
/// 
/// // Sum of exp(log_softmax) values should be very close to 1 for each batch
/// for i in 0..2 {
///     let mut sum = 0.0;
///     for j in 0..3 {
///         sum += y[[i, j]].exp();
///     }
///     assert!((sum - 1.0).abs() < 1e-6);
/// }
/// ```
pub fn log_softmax<F, D>(
    x: &ArrayView<F, D>,
    axis: Axis,
) -> Result<Array<F, D>>
where
    F: Float + Debug + NumAssign,
    D: Dimension,
{
    // Check that the specified axis is valid
    let axis_idx = axis.index();
    if axis_idx >= x.ndim() {
        return Err(NeuralError::Other(format!(
            "Axis {} is out of bounds for array with {} dimensions",
            axis_idx, x.ndim()
        )));
    }
    
    let mut output = Array::zeros(x.dim());
    
    // Apply log-softmax for each slice along the specified axis
    for (i, mut out_subview) in output.lanes_mut(axis).into_iter().enumerate() {
        let in_subview = x.lanes(axis).into_iter().nth(i).unwrap();
        
        // Find max value for numerical stability
        let mut max_val = F::neg_infinity();
        for &val in in_subview.iter() {
            if val > max_val {
                max_val = val;
            }
        }
        
        // Compute log-sum-exp
        let mut sum = F::zero();
        for &val in in_subview.iter() {
            sum = sum + (val - max_val).exp();
        }
        let log_sum = sum.ln() + max_val;
        
        // Compute log-softmax
        for (j, &val) in in_subview.iter().enumerate() {
            let idx = out_subview.dim().default_index_for_elem(j);
            out_subview[idx] = val - log_sum;
        }
    }
    
    Ok(output)
}

/// Perform position encoding for transformer-based models.
///
/// This function computes sinusoidal position encodings
/// as described in the "Attention Is All You Need" paper.
///
/// # Arguments
///
/// * `seq_len` - Maximum sequence length to encode
/// * `model_dim` - Embedding dimension
///
/// # Returns
///
/// * Position encoding matrix with shape [seq_len, model_dim]
///
/// # Examples
///
/// ```
/// use scirs2_neural::linalg::positional_encoding;
///
/// // Generate positional encoding for sequences up to length 10 with embedding dim 16
/// let pos_encoding = positional_encoding(10, 16).unwrap();
/// assert_eq!(pos_encoding.shape(), &[10, 16]);
/// ```
pub fn positional_encoding<F: Float + Debug>(
    seq_len: usize,
    model_dim: usize,
) -> Result<Array2<F>> {
    // Check dimensions
    if model_dim % 2 != 0 {
        return Err(NeuralError::Other(format!(
            "Model dimension ({}) must be even for positional encoding",
            model_dim
        )));
    }
    
    let mut encoding = Array2::zeros((seq_len, model_dim));
    
    for pos in 0..seq_len {
        for i in 0..model_dim/2 {
            let denom = F::from(10000.0).unwrap().powf(F::from(2.0 * i as f64 / model_dim as f64).unwrap());
            let pos_f = F::from(pos as f64).unwrap();
            
            // Even indices get sine, odd indices get cosine
            let angle = pos_f / denom;
            encoding[[pos, 2*i]] = angle.sin();
            encoding[[pos, 2*i+1]] = angle.cos();
        }
    }
    
    Ok(encoding)
}

/// Perform feed-forward network computation for Transformer models.
///
/// This function implements the position-wise feed-forward network used in
/// Transformer models, consisting of two linear transformations with a ReLU
/// activation in between.
///
/// # Arguments
///
/// * `x` - Input tensor with shape [batch_size, seq_len, model_dim]
/// * `w1` - First weight matrix with shape [model_dim, ff_dim]
/// * `b1` - First bias vector with shape [ff_dim]
/// * `w2` - Second weight matrix with shape [ff_dim, model_dim]
/// * `b2` - Second bias vector with shape [model_dim]
///
/// # Returns
///
/// * Output tensor with shape [batch_size, seq_len, model_dim]
///
/// # Examples
///
/// ```
/// use ndarray::{Array1, Array2, Array3};
/// use scirs2_neural::linalg::transformer_ffn;
///
/// // Setup parameters
/// let batch_size = 2;
/// let seq_len = 3;
/// let model_dim = 4;
/// let ff_dim = 8;
///
/// // Input and weights
/// let x = Array3::<f32>::ones((batch_size, seq_len, model_dim));
/// let w1 = Array2::<f32>::ones((model_dim, ff_dim));
/// let b1 = Array1::<f32>::zeros(ff_dim);
/// let w2 = Array2::<f32>::ones((ff_dim, model_dim));
/// let b2 = Array1::<f32>::zeros(model_dim);
///
/// // Apply feed-forward network
/// let output = transformer_ffn(
///     &x.view(), &w1.view(), &b1.view(), &w2.view(), &b2.view()
/// ).unwrap();
///
/// // Output shape should match input shape
/// assert_eq!(output.shape(), &[batch_size, seq_len, model_dim]);
/// ```
pub fn transformer_ffn<F>(
    x: &ArrayView3<F>,
    w1: &ArrayView2<F>,
    b1: &ArrayView1<F>,
    w2: &ArrayView2<F>,
    b2: &ArrayView1<F>,
) -> Result<Array3<F>>
where
    F: Float + Debug + NumAssign,
{
    // Extract dimensions
    let batch_size = x.shape()[0];
    let seq_len = x.shape()[1];
    let model_dim = x.shape()[2];
    
    let ff_dim = w1.shape()[1];
    
    // Check shape compatibility
    if w1.shape()[0] != model_dim {
        return Err(NeuralError::Other(format!(
            "First weight matrix rows ({}) don't match model dimension ({})",
            w1.shape()[0], model_dim
        )));
    }
    
    if b1.len() != ff_dim {
        return Err(NeuralError::Other(format!(
            "First bias length ({}) doesn't match ff dimension ({})",
            b1.len(), ff_dim
        )));
    }
    
    if w2.shape()[0] != ff_dim || w2.shape()[1] != model_dim {
        return Err(NeuralError::Other(format!(
            "Second weight matrix shape {:?} doesn't match expected ({}, {})",
            w2.shape(), ff_dim, model_dim
        )));
    }
    
    if b2.len() != model_dim {
        return Err(NeuralError::Other(format!(
            "Second bias length ({}) doesn't match model dimension ({})",
            b2.len(), model_dim
        )));
    }
    
    // Initialize intermediate and output arrays
    let mut hidden = Array3::zeros((batch_size, seq_len, ff_dim));
    let mut output = Array3::zeros((batch_size, seq_len, model_dim));
    
    // First linear transformation + ReLU
    for b in 0..batch_size {
        for s in 0..seq_len {
            for j in 0..ff_dim {
                let mut sum = F::zero();
                for i in 0..model_dim {
                    sum = sum + x[[b, s, i]] * w1[[i, j]];
                }
                sum = sum + b1[j];
                
                // ReLU activation
                hidden[[b, s, j]] = if sum > F::zero() { sum } else { F::zero() };
            }
        }
    }
    
    // Second linear transformation
    for b in 0..batch_size {
        for s in 0..seq_len {
            for j in 0..model_dim {
                let mut sum = F::zero();
                for i in 0..ff_dim {
                    sum = sum + hidden[[b, s, i]] * w2[[i, j]];
                }
                output[[b, s, j]] = sum + b2[j];
            }
        }
    }
    
    Ok(output)
}

/// Perform im2col operation for convolutional neural networks.
///
/// This function converts image patches to columns, which is a common
/// preprocessing step for efficient convolution implementation using
/// matrix multiplication.
///
/// # Arguments
///
/// * `input` - Input tensor with shape [batch_size, in_channels, height, width]
/// * `kernel_size` - Size of the kernel (height, width)
/// * `stride` - Stride (vertical, horizontal)
/// * `padding` - Padding mode: "valid" (no padding) or "same" (padding to maintain size)
///
/// # Returns
///
/// * Columns tensor with shape [batch_size, out_height * out_width, in_channels * kernel_height * kernel_width]
///
/// # Examples
///
/// ```
/// use ndarray::Array4;
/// use scirs2_neural::linalg::im2col;
///
/// // Create a batch of 2 images, each with 3 channels and 5x5 size
/// let input = Array4::<f32>::zeros((2, 3, 5, 5));
///
/// // Apply im2col with 3x3 kernel and stride 1
/// let cols = im2col(&input.view(), (3, 3), (1, 1), "valid").unwrap();
///
/// // Output shape: [batch_size, out_height * out_width, in_channels * kernel_height * kernel_width]
/// // For 5x5 input, 3x3 kernel, stride 1, valid padding: out_height=out_width=3
/// // So shape is [2, 9, 27]
/// assert_eq!(cols.shape(), &[2, 9, 27]);
/// ```
pub fn im2col<F>(
    input: &ArrayView4<F>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: &str,
) -> Result<Array3<F>>
where
    F: Float + Debug + NumAssign + Zero,
{
    // Extract dimensions
    let batch_size = input.shape()[0];
    let in_channels = input.shape()[1];
    let in_height = input.shape()[2];
    let in_width = input.shape()[3];
    
    let kernel_h = kernel_size.0;
    let kernel_w = kernel_size.1;
    let stride_h = stride.0;
    let stride_w = stride.1;
    
    // Calculate padding and output dimensions
    let (pad_h, pad_w, out_height, out_width) = match padding {
        "valid" => {
            let out_height = (in_height - kernel_h) / stride_h + 1;
            let out_width = (in_width - kernel_w) / stride_w + 1;
            (0, 0, out_height, out_width)
        },
        "same" => {
            let out_height = (in_height + stride_h - 1) / stride_h;
            let out_width = (in_width + stride_w - 1) / stride_w;
            
            let pad_h_total = (out_height - 1) * stride_h + kernel_h - in_height;
            let pad_w_total = (out_width - 1) * stride_w + kernel_w - in_width;
            
            let pad_h = pad_h_total / 2;
            let pad_w = pad_w_total / 2;
            
            (pad_h, pad_w, out_height, out_width)
        },
        _ => {
            return Err(NeuralError::Other(format!(
                "Invalid padding mode: {}. Should be 'valid' or 'same'.",
                padding
            )));
        }
    };
    
    // Initialize output tensor
    let out_size = out_height * out_width;
    let col_size = in_channels * kernel_h * kernel_w;
    let mut cols = Array3::zeros((batch_size, out_size, col_size));
    
    // Perform im2col
    for b in 0..batch_size {
        let mut col_idx = 0;
        for oh in 0..out_height {
            for ow in 0..out_width {
                let mut patch_idx = 0;
                for c in 0..in_channels {
                    for kh in 0..kernel_h {
                        for kw in 0..kernel_w {
                            let h = oh * stride_h + kh;
                            let w = ow * stride_w + kw;
                            
                            let value = if h >= pad_h && h < in_height + pad_h && 
                                          w >= pad_w && w < in_width + pad_w {
                                let input_h = h - pad_h;
                                let input_w = w - pad_w;
                                if input_h < in_height && input_w < in_width {
                                    input[[b, c, input_h, input_w]]
                                } else {
                                    F::zero()
                                }
                            } else {
                                F::zero()
                            };
                            
                            cols[[b, col_idx, patch_idx]] = value;
                            patch_idx += 1;
                        }
                    }
                }
                col_idx += 1;
            }
        }
    }
    
    Ok(cols)
}

/// Perform col2im operation for convolutional neural networks.
///
/// This function converts columns back to image patches, which is used
/// during the backward pass of convolutional layers.
///
/// # Arguments
///
/// * `cols` - Columns tensor with shape [batch_size, out_height * out_width, in_channels * kernel_height * kernel_width]
/// * `output_shape` - Shape of the output tensor [batch_size, in_channels, height, width]
/// * `kernel_size` - Size of the kernel (height, width)
/// * `stride` - Stride (vertical, horizontal)
/// * `padding` - Padding mode: "valid" (no padding) or "same" (padding to maintain size)
///
/// # Returns
///
/// * Output tensor with shape [batch_size, in_channels, height, width]
///
/// # Examples
///
/// ```
/// use ndarray::{Array3, Array4};
/// use scirs2_neural::linalg::{im2col, col2im};
///
/// // Create a batch of 2 images, each with 3 channels and 5x5 size
/// let input = Array4::<f32>::ones((2, 3, 5, 5));
///
/// // Apply im2col with 3x3 kernel and stride 1
/// let cols = im2col(&input.view(), (3, 3), (1, 1), "valid").unwrap();
///
/// // Convert back to image
/// let output = col2im(&cols.view(), (2, 3, 5, 5), (3, 3), (1, 1), "valid").unwrap();
///
/// // Output shape should match input shape
/// assert_eq!(output.shape(), input.shape());
/// ```
pub fn col2im<F>(
    cols: &ArrayView3<F>,
    output_shape: (usize, usize, usize, usize),
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: &str,
) -> Result<Array4<F>>
where
    F: Float + Debug + NumAssign + Zero,
{
    // Extract dimensions
    let batch_size = cols.shape()[0];
    let (_, in_channels, in_height, in_width) = output_shape;
    
    let kernel_h = kernel_size.0;
    let kernel_w = kernel_size.1;
    let stride_h = stride.0;
    let stride_w = stride.1;
    
    // Calculate padding and output dimensions
    let (pad_h, pad_w, out_height, out_width) = match padding {
        "valid" => {
            let out_height = (in_height - kernel_h) / stride_h + 1;
            let out_width = (in_width - kernel_w) / stride_w + 1;
            (0, 0, out_height, out_width)
        },
        "same" => {
            let out_height = (in_height + stride_h - 1) / stride_h;
            let out_width = (in_width + stride_w - 1) / stride_w;
            
            let pad_h_total = (out_height - 1) * stride_h + kernel_h - in_height;
            let pad_w_total = (out_width - 1) * stride_w + kernel_w - in_width;
            
            let pad_h = pad_h_total / 2;
            let pad_w = pad_w_total / 2;
            
            (pad_h, pad_w, out_height, out_width)
        },
        _ => {
            return Err(NeuralError::Other(format!(
                "Invalid padding mode: {}. Should be 'valid' or 'same'.",
                padding
            )));
        }
    };
    
    // Check shape compatibility
    let expected_cols_shape = (batch_size, out_height * out_width, in_channels * kernel_h * kernel_w);
    if cols.shape() != &[expected_cols_shape.0, expected_cols_shape.1, expected_cols_shape.2] {
        return Err(NeuralError::Other(format!(
            "Columns shape {:?} doesn't match expected {:?}",
            cols.shape(), expected_cols_shape
        )));
    }
    
    // Initialize output tensor
    let mut output = Array4::zeros(output_shape);
    
    // Perform col2im by accumulating patches
    for b in 0..batch_size {
        let mut col_idx = 0;
        for oh in 0..out_height {
            for ow in 0..out_width {
                let mut patch_idx = 0;
                for c in 0..in_channels {
                    for kh in 0..kernel_h {
                        for kw in 0..kernel_w {
                            let h = oh * stride_h + kh;
                            let w = ow * stride_w + kw;
                            
                            if h >= pad_h && h < in_height + pad_h && 
                               w >= pad_w && w < in_width + pad_w {
                                let input_h = h - pad_h;
                                let input_w = w - pad_w;
                                if input_h < in_height && input_w < in_width {
                                    output[[b, c, input_h, input_w]] = 
                                        output[[b, c, input_h, input_w]] + cols[[b, col_idx, patch_idx]];
                                }
                            }
                            
                            patch_idx += 1;
                        }
                    }
                }
                col_idx += 1;
            }
        }
    }
    
    Ok(output)
}

/// Compute adaptive average pooling for neural networks.
///
/// This function performs adaptive average pooling, which resizes the input to a specified output size
/// by averaging values in dynamically sized windows.
///
/// # Arguments
///
/// * `input` - Input tensor with shape [batch_size, channels, height, width]
/// * `output_size` - Desired output size (height, width)
///
/// # Returns
///
/// * Result tensor with shape [batch_size, channels, output_height, output_width]
///
/// # Examples
///
/// ```
/// use ndarray::Array4;
/// use scirs2_neural::linalg::adaptive_avg_pool2d;
///
/// // Create a batch of 2 feature maps, each with 3 channels and 7x7 size
/// let input = Array4::<f32>::ones((2, 3, 7, 7));
///
/// // Apply adaptive average pooling to get 3x3 output
/// let output = adaptive_avg_pool2d(&input.view(), (3, 3)).unwrap();
///
/// // Output shape should be [2, 3, 3, 3]
/// assert_eq!(output.shape(), &[2, 3, 3, 3]);
/// ```
pub fn adaptive_avg_pool2d<F>(
    input: &ArrayView4<F>,
    output_size: (usize, usize),
) -> Result<Array4<F>>
where
    F: Float + Debug + NumAssign,
{
    // Extract dimensions
    let batch_size = input.shape()[0];
    let channels = input.shape()[1];
    let in_height = input.shape()[2];
    let in_width = input.shape()[3];
    
    let out_height = output_size.0;
    let out_width = output_size.1;
    
    // Check output size validity
    if out_height == 0 || out_width == 0 {
        return Err(NeuralError::Other(format!(
            "Output size must be greater than 0, got {:?}",
            output_size
        )));
    }
    
    // Initialize output tensor
    let mut output = Array4::zeros((batch_size, channels, out_height, out_width));
    
    // Calculate the bin sizes (how many input elements map to one output element)
    for b in 0..batch_size {
        for c in 0..channels {
            for oh in 0..out_height {
                // Calculate the start and end indices in the input height dimension
                let h_start = (oh * in_height) / out_height;
                let h_end = ((oh + 1) * in_height) / out_height;
                
                for ow in 0..out_width {
                    // Calculate the start and end indices in the input width dimension
                    let w_start = (ow * in_width) / out_width;
                    let w_end = ((ow + 1) * in_width) / out_width;
                    
                    // Count the number of elements in this bin
                    let bin_size = (h_end - h_start) * (w_end - w_start);
                    
                    // Sum the elements in this bin
                    let mut sum = F::zero();
                    for h in h_start..h_end {
                        for w in w_start..w_end {
                            sum = sum + input[[b, c, h, w]];
                        }
                    }
                    
                    // Calculate the average
                    output[[b, c, oh, ow]] = sum / F::from(bin_size).unwrap();
                }
            }
        }
    }
    
    Ok(output)
}

/// Apply gradient clipping by norm for neural network training.
///
/// This function clips gradients to have a maximum L2 norm,
/// which helps prevent exploding gradients during training.
///
/// # Arguments
///
/// * `grads` - Array of gradient tensors of any shape
/// * `max_norm` - Maximum allowed L2 norm
///
/// # Returns
///
/// * Clipped gradient tensors
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array1, Array2};
/// use scirs2_neural::linalg::clip_grad_norm;
///
/// // Create gradient tensors
/// let grad1 = array![3.0, 4.0]; // norm = 5
/// let grad2 = array![[5.0, 0.0], [0.0, 12.0]]; // norm = 13
///
/// // Clip gradients with max_norm = 10.0
/// let grads = vec![grad1.view(), grad2.view()];
/// let clipped_grads = clip_grad_norm(&grads, 10.0).unwrap();
///
/// // The combined L2 norm of the clipped gradients should be approximately max_norm
/// let combined_norm_squared = clipped_grads[0].iter().map(|&x| x*x).sum::<f64>() +
///                             clipped_grads[1].iter().map(|&x| x*x).sum::<f64>();
/// let combined_norm = combined_norm_squared.sqrt();
/// assert!((combined_norm - 10.0).abs() < 1e-6);
/// ```
pub fn clip_grad_norm<F, D>(
    grads: &[ArrayView<F, D>],
    max_norm: F,
) -> Result<Vec<Array<F, D>>>
where
    F: Float + Debug + NumAssign,
    D: Dimension,
{
    if grads.is_empty() {
        return Ok(Vec::new());
    }
    
    if max_norm <= F::zero() {
        return Err(NeuralError::Other(format!(
            "Maximum norm must be positive, got {:?}",
            max_norm
        )));
    }
    
    // Calculate the combined L2 norm of all gradients
    let mut total_norm_sq = F::zero();
    for grad in grads.iter() {
        for &val in grad.iter() {
            total_norm_sq = total_norm_sq + val * val;
        }
    }
    let total_norm = total_norm_sq.sqrt();
    
    // If the total norm is less than the max norm, no clipping is needed
    if total_norm <= max_norm {
        return Ok(grads.iter().map(|g| g.to_owned()).collect());
    }
    
    // Calculate the scaling factor
    let scale = max_norm / total_norm;
    
    // Apply the scaling factor to all gradients
    let mut clipped_grads = Vec::with_capacity(grads.len());
    for grad in grads.iter() {
        let mut clipped = Array::zeros(grad.dim());
        for (i, &val) in grad.iter().enumerate() {
            let idx = clipped.dim().default_index_for_elem(i);
            clipped[idx] = val * scale;
        }
        clipped_grads.push(clipped);
    }
    
    Ok(clipped_grads)
}

/// Compute Adam optimization updates for neural network parameters.
///
/// This function implements the Adam optimization algorithm, which
/// adapts learning rates for each parameter based on historical gradients.
///
/// # Arguments
///
/// * `param` - Parameter tensor of any shape
/// * `grad` - Gradient tensor of the same shape as param
/// * `m` - First moment (mean) tensor of the same shape as param
/// * `v` - Second moment (variance) tensor of the same shape as param
/// * `lr` - Learning rate
/// * `beta1` - Exponential decay rate for first moment
/// * `beta2` - Exponential decay rate for second moment
/// * `eps` - Small constant for numerical stability
/// * `t` - Current timestep (starts at 1)
///
/// # Returns
///
/// * Updated parameter tensor
/// * Updated first moment tensor
/// * Updated second moment tensor
///
/// # Examples
///
/// ```
/// use ndarray::Array1;
/// use scirs2_neural::linalg::adam_update;
///
/// // Initialize parameters
/// let param = Array1::<f32>::zeros(3);
/// let grad = Array1::<f32>::from_elem(3, 0.1);
/// let m = Array1::<f32>::zeros(3);
/// let v = Array1::<f32>::zeros(3);
///
/// // Apply Adam update
/// let (param_new, m_new, v_new) = adam_update(
///     &param.view(), &grad.view(), &m.view(), &v.view(),
///     0.001, 0.9, 0.999, 1e-8, 1
/// ).unwrap();
///
/// assert_eq!(param_new.shape(), &[3]);
/// assert_eq!(m_new.shape(), &[3]);
/// assert_eq!(v_new.shape(), &[3]);
/// ```
pub fn adam_update<F, D>(
    param: &ArrayView<F, D>,
    grad: &ArrayView<F, D>,
    m: &ArrayView<F, D>,
    v: &ArrayView<F, D>,
    lr: F,
    beta1: F,
    beta2: F,
    eps: F,
    t: usize,
) -> Result<(Array<F, D>, Array<F, D>, Array<F, D>)>
where
    F: Float + Debug + NumAssign,
    D: Dimension,
{
    // Check shape compatibility
    if param.shape() != grad.shape() {
        return Err(NeuralError::Other(format!(
            "Parameter shape {:?} doesn't match gradient shape {:?}",
            param.shape(), grad.shape()
        )));
    }
    
    if param.shape() != m.shape() {
        return Err(NeuralError::Other(format!(
            "Parameter shape {:?} doesn't match first moment shape {:?}",
            param.shape(), m.shape()
        )));
    }
    
    if param.shape() != v.shape() {
        return Err(NeuralError::Other(format!(
            "Parameter shape {:?} doesn't match second moment shape {:?}",
            param.shape(), v.shape()
        )));
    }
    
    // Validate hyperparameters
    if lr <= F::zero() {
        return Err(NeuralError::Other(format!(
            "Learning rate must be positive, got {:?}",
            lr
        )));
    }
    
    if beta1 < F::zero() || beta1 >= F::one() {
        return Err(NeuralError::Other(format!(
            "beta1 must be in [0, 1), got {:?}",
            beta1
        )));
    }
    
    if beta2 < F::zero() || beta2 >= F::one() {
        return Err(NeuralError::Other(format!(
            "beta2 must be in [0, 1), got {:?}",
            beta2
        )));
    }
    
    if t == 0 {
        return Err(NeuralError::Other(
            "Timestep must be greater than 0".to_string()
        ));
    }
    
    // Initialize updated tensors
    let mut m_new = Array::zeros(param.dim());
    let mut v_new = Array::zeros(param.dim());
    let mut param_new = Array::zeros(param.dim());
    
    // Calculate bias correction factors
    let t_f = F::from(t).unwrap();
    let correction1 = F::one() - beta1.powi(t as i32);
    let correction2 = F::one() - beta2.powi(t as i32);
    
    // Update parameters element-wise
    for i in 0..param.len() {
        let idx = param_new.dim().default_index_for_elem(i);
        
        // Update biased first moment estimate
        m_new[idx.clone()] = beta1 * m[idx.clone()] + (F::one() - beta1) * grad[idx.clone()];
        
        // Update biased second raw moment estimate
        v_new[idx.clone()] = beta2 * v[idx.clone()] + (F::one() - beta2) * grad[idx.clone()] * grad[idx.clone()];
        
        // Compute bias-corrected first moment estimate
        let m_hat = m_new[idx.clone()] / correction1;
        
        // Compute bias-corrected second raw moment estimate
        let v_hat = v_new[idx.clone()] / correction2;
        
        // Update parameters
        param_new[idx.clone()] = param[idx.clone()] - lr * m_hat / (v_hat.sqrt() + eps);
    }
    
    Ok((param_new, m_new, v_new))
}

/// Compute the backward pass for convolution operation.
///
/// This function calculates the gradients of the loss with respect to
/// input, filters, and bias for a 2D convolution operation.
///
/// # Arguments
///
/// * `dout` - Gradient from upstream with shape [batch_size, out_channels, out_height, out_width]
/// * `input` - Input tensor with shape [batch_size, in_channels, in_height, in_width]
/// * `filters` - Filters tensor with shape [out_channels, in_channels, filter_height, filter_width]
/// * `bias` - Optional bias tensor with shape [out_channels]
/// * `stride` - Stride (vertical, horizontal)
/// * `padding` - Padding mode: "valid" (no padding) or "same" (padding to maintain size)
///
/// # Returns
///
/// * Tuple of (dinput, dfilters, dbias) gradients
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array1, Array4};
/// use scirs2_neural::linalg::conv2d_backward;
///
/// // Create a batch of 2 images, each with 1 channel and 4x4 size
/// let input = Array4::<f32>::ones((2, 1, 4, 4));
/// 
/// // Create 2 filters, each for 1 input channel and 3x3 kernel size
/// let filters = Array4::<f32>::ones((2, 1, 3, 3));
///
/// // Bias
/// let bias = Array1::<f32>::zeros(2);
///
/// // Gradient from upstream
/// let dout = Array4::<f32>::ones((2, 2, 2, 2));
///
/// // Calculate gradients
/// let (dinput, dfilters, dbias) = conv2d_backward(
///     &dout.view(), &input.view(), &filters.view(), Some(&bias.view()), (1, 1), "valid"
/// ).unwrap();
///
/// // Check gradient shapes
/// assert_eq!(dinput.shape(), input.shape());
/// assert_eq!(dfilters.shape(), filters.shape());
/// assert_eq!(dbias.shape(), bias.shape());
/// ```
pub fn conv2d_backward<F>(
    dout: &ArrayView4<F>,
    input: &ArrayView4<F>,
    filters: &ArrayView4<F>,
    bias: Option<&ArrayView1<F>>,
    stride: (usize, usize),
    padding: &str,
) -> Result<(Array4<F>, Array4<F>, Option<Array1<F>>)>
where
    F: Float + Debug + NumAssign + Zero,
{
    // Extract dimensions
    let batch_size = input.shape()[0];
    let in_channels = input.shape()[1];
    let in_height = input.shape()[2];
    let in_width = input.shape()[3];
    
    let out_channels = filters.shape()[0];
    let filter_height = filters.shape()[2];
    let filter_width = filters.shape()[3];
    
    // Check shape compatibility
    if dout.shape()[0] != batch_size {
        return Err(NeuralError::Other(format!(
            "dout batch size ({}) doesn't match input batch size ({})",
            dout.shape()[0], batch_size
        )));
    }
    
    if dout.shape()[1] != out_channels {
        return Err(NeuralError::Other(format!(
            "dout channels ({}) don't match filter out_channels ({})",
            dout.shape()[1], out_channels
        )));
    }
    
    // Initialize gradients
    let mut dinput = Array4::zeros(input.dim());
    let mut dfilters = Array4::zeros(filters.dim());
    let mut dbias = match bias {
        Some(b) => Some(Array1::zeros(b.dim())),
        None => None,
    };
    
    // Use im2col approach for more efficient convolution gradient computation
    
    // 1. Convert input to columns
    let cols = im2col(input, (filter_height, filter_width), stride, padding)?;
    
    // 2. Reshape filters for matrix multiplication
    let filters_reshaped = filters.clone().into_shape((out_channels, in_channels * filter_height * filter_width)).unwrap();
    
    // 3. Compute gradient with respect to filters
    for b in 0..batch_size {
        // Reshape dout for this batch
        let dout_b = dout.slice(s![b, .., .., ..]).to_owned();
        let dout_reshaped = dout_b.into_shape((out_channels, dout.shape()[2] * dout.shape()[3])).unwrap();
        
        // Get columns for this batch
        let cols_b = cols.slice(s![b, .., ..]).to_owned();
        
        // Compute gradients for filters (out_channels × (in_channels * filter_height * filter_width))
        for oc in 0..out_channels {
            for i in 0..in_channels * filter_height * filter_width {
                let mut sum = F::zero();
                for j in 0..dout.shape()[2] * dout.shape()[3] {
                    sum = sum + dout_reshaped[[oc, j]] * cols_b[[j, i]];
                }
                let filter_idx = unravel_index(i, (in_channels, filter_height, filter_width));
                dfilters[[oc, filter_idx.0, filter_idx.1, filter_idx.2]] = 
                    dfilters[[oc, filter_idx.0, filter_idx.1, filter_idx.2]] + sum;
            }
        }
    }
    
    // 4. Compute gradient with respect to bias
    if let Some(db) = dbias.as_mut() {
        for oc in 0..out_channels {
            let mut sum = F::zero();
            for b in 0..batch_size {
                for h in 0..dout.shape()[2] {
                    for w in 0..dout.shape()[3] {
                        sum = sum + dout[[b, oc, h, w]];
                    }
                }
            }
            db[oc] = sum;
        }
    }
    
    // 5. Compute gradient with respect to input
    // Reshape filters for multiplication (flip 180 degrees for convolution)
    let mut filters_flipped = Array4::zeros((out_channels, in_channels, filter_height, filter_width));
    for oc in 0..out_channels {
        for ic in 0..in_channels {
            for h in 0..filter_height {
                for w in 0..filter_width {
                    filters_flipped[[oc, ic, h, w]] = filters[[oc, ic, filter_height - h - 1, filter_width - w - 1]];
                }
            }
        }
    }
    
    // Pad dout for full convolution
    let pad_h = filter_height - 1;
    let pad_w = filter_width - 1;
    let mut dout_padded = Array4::zeros((batch_size, out_channels, dout.shape()[2] + 2 * pad_h, dout.shape()[3] + 2 * pad_w));
    for b in 0..batch_size {
        for oc in 0..out_channels {
            for h in 0..dout.shape()[2] {
                for w in 0..dout.shape()[3] {
                    dout_padded[[b, oc, h + pad_h, w + pad_w]] = dout[[b, oc, h, w]];
                }
            }
        }
    }
    
    // Perform convolution with flipped filters and padded dout
    for b in 0..batch_size {
        for ic in 0..in_channels {
            for h in 0..in_height {
                for w in 0..in_width {
                    let mut sum = F::zero();
                    for oc in 0..out_channels {
                        for fh in 0..filter_height {
                            for fw in 0..filter_width {
                                let dout_h = h + fh;
                                let dout_w = w + fw;
                                if dout_h < dout_padded.shape()[2] && dout_w < dout_padded.shape()[3] {
                                    sum = sum + dout_padded[[b, oc, dout_h, dout_w]] * filters_flipped[[oc, ic, fh, fw]];
                                }
                            }
                        }
                    }
                    dinput[[b, ic, h, w]] = sum;
                }
            }
        }
    }
    
    Ok((dinput, dfilters, dbias))
}

/// Helper function to convert a flattened index to a 3D index
fn unravel_index(index: usize, shape: (usize, usize, usize)) -> (usize, usize, usize) {
    let (ic, fh, fw) = shape;
    let layer_size = fh * fw;
    
    let ic_idx = index / layer_size;
    let remainder = index % layer_size;
    let fh_idx = remainder / fw;
    let fw_idx = remainder % fw;
    
    (ic_idx, fh_idx, fw_idx)
}

/// Compute the backward pass for max pooling operation.
///
/// This function calculates the gradients of the loss with respect to
/// input for a max pooling operation.
///
/// # Arguments
///
/// * `dout` - Gradient from upstream with shape [batch_size, channels, out_height, out_width]
/// * `input` - Input tensor with shape [batch_size, channels, in_height, in_width]
/// * `indices` - Indices of maximum values from forward pass with shape [batch_size, channels, out_height, out_width]
/// * `pool_size` - Size of the pooling window (height, width)
/// * `stride` - Stride (vertical, horizontal)
///
/// # Returns
///
/// * Gradient with respect to input
///
/// # Examples
///
/// ```
/// use ndarray::Array4;
/// use scirs2_neural::linalg::{max_pool2d, max_pool2d_backward};
///
/// // Create a batch of 2 feature maps, each with 3 channels and 4x4 size
/// let input = Array4::<f32>::ones((2, 3, 4, 4));
///
/// // Apply max pooling with 2x2 pool size and stride 2
/// let (output, indices) = max_pool2d(&input.view(), (2, 2), (2, 2)).unwrap();
///
/// // Gradient from upstream
/// let dout = Array4::<f32>::ones(output.dim());
///
/// // Calculate gradients
/// let dinput = max_pool2d_backward(
///     &dout.view(), &input.view(), &indices.view(), (2, 2), (2, 2)
/// ).unwrap();
///
/// // Check gradient shape
/// assert_eq!(dinput.shape(), input.shape());
/// ```
pub fn max_pool2d_backward<F>(
    dout: &ArrayView4<F>,
    input: &ArrayView4<F>,
    indices: &ArrayView4<usize>,
    pool_size: (usize, usize),
    stride: (usize, usize),
) -> Result<Array4<F>>
where
    F: Float + Debug + NumAssign + Zero,
{
    // Extract dimensions
    let batch_size = input.shape()[0];
    let channels = input.shape()[1];
    let in_height = input.shape()[2];
    let in_width = input.shape()[3];
    
    let pool_h = pool_size.0;
    let pool_w = pool_size.1;
    let stride_h = stride.0;
    let stride_w = stride.1;
    
    let out_height = dout.shape()[2];
    let out_width = dout.shape()[3];
    
    // Check shape compatibility
    if dout.shape()[0] != batch_size || indices.shape()[0] != batch_size {
        return Err(NeuralError::Other(format!(
            "Batch sizes don't match: dout ({}), input ({}), indices ({})",
            dout.shape()[0], batch_size, indices.shape()[0]
        )));
    }
    
    if dout.shape()[1] != channels || indices.shape()[1] != channels {
        return Err(NeuralError::Other(format!(
            "Channel counts don't match: dout ({}), input ({}), indices ({})",
            dout.shape()[1], channels, indices.shape()[1]
        )));
    }
    
    // Initialize gradient tensor
    let mut dinput = Array4::zeros(input.dim());
    
    // Distribute gradients to the max elements in the original input
    for b in 0..batch_size {
        for c in 0..channels {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let h_start = oh * stride_h;
                    let w_start = ow * stride_w;
                    
                    // Get the 1D index and convert to 2D
                    let max_idx = indices[[b, c, oh, ow]];
                    let ph = max_idx / pool_w;
                    let pw = max_idx % pool_w;
                    
                    // Compute the position in the original input
                    let ih = h_start + ph;
                    let iw = w_start + pw;
                    
                    // Pass the gradient to the max element
                    if ih < in_height && iw < in_width {
                        dinput[[b, c, ih, iw]] = dinput[[b, c, ih, iw]] + dout[[b, c, oh, ow]];
                    }
                }
            }
        }
    }
    
    Ok(dinput)
}

/// Compute the backward pass for layer normalization.
///
/// This function calculates the gradients of the loss with respect to
/// input, gamma, and beta for a layer normalization operation.
///
/// # Arguments
///
/// * `dout` - Gradient from upstream with shape [batch_size, seq_len, features]
/// * `x` - Input tensor with shape [batch_size, seq_len, features]
/// * `gamma` - Scale parameter with shape [features]
/// * `mean` - Mean values with shape [batch_size, seq_len]
/// * `var` - Variance values with shape [batch_size, seq_len]
/// * `eps` - Small constant for numerical stability
///
/// # Returns
///
/// * Tuple of (dx, dgamma, dbeta) gradients
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array1, Array2, Array3};
/// use scirs2_neural::linalg::{layer_norm, layer_norm_backward};
///
/// // Input: batch_size=2, seq_len=3, features=4
/// let x = Array3::<f32>::ones((2, 3, 4));
/// let gamma = Array1::<f32>::ones(4);
/// let beta = Array1::<f32>::zeros(4);
/// let eps = 1e-5;
///
/// // Calculate means and variances
/// let mut mean = Array2::<f32>::zeros((2, 3));
/// let mut var = Array2::<f32>::zeros((2, 3));
/// for b in 0..2 {
///     for s in 0..3 {
///         // Compute mean
///         let mut sum = 0.0;
///         for f in 0..4 {
///             sum += x[[b, s, f]];
///         }
///         mean[[b, s]] = sum / 4.0;
///         
///         // Compute variance
///         let mut sum_sq = 0.0;
///         for f in 0..4 {
///             let diff = x[[b, s, f]] - mean[[b, s]];
///             sum_sq += diff * diff;
///         }
///         var[[b, s]] = sum_sq / 4.0;
///     }
/// }
///
/// // Gradient from upstream
/// let dout = Array3::<f32>::ones((2, 3, 4));
///
/// // Calculate gradients
/// let (dx, dgamma, dbeta) = layer_norm_backward(
///     &dout.view(), &x.view(), &gamma.view(), &mean.view(), &var.view(), eps
/// ).unwrap();
///
/// // Check gradient shapes
/// assert_eq!(dx.shape(), x.shape());
/// assert_eq!(dgamma.shape(), gamma.shape());
/// assert_eq!(dbeta.shape(), beta.shape());
/// ```
pub fn layer_norm_backward<F>(
    dout: &ArrayView3<F>,
    x: &ArrayView3<F>,
    gamma: &ArrayView1<F>,
    mean: &ArrayView2<F>,
    var: &ArrayView2<F>,
    eps: F,
) -> Result<(Array3<F>, Array1<F>, Array1<F>)>
where
    F: Float + Debug + NumAssign,
{
    // Extract dimensions
    let batch_size = x.shape()[0];
    let seq_len = x.shape()[1];
    let features = x.shape()[2];
    
    // Check shape compatibility
    if dout.shape() != x.shape() {
        return Err(NeuralError::Other(format!(
            "dout shape ({:?}) doesn't match x shape ({:?})",
            dout.shape(), x.shape()
        )));
    }
    
    if gamma.len() != features {
        return Err(NeuralError::Other(format!(
            "gamma size ({}) doesn't match features ({})",
            gamma.len(), features
        )));
    }
    
    if mean.shape() != &[batch_size, seq_len] {
        return Err(NeuralError::Other(format!(
            "mean shape ({:?}) doesn't match expected ({:?})",
            mean.shape(), [batch_size, seq_len]
        )));
    }
    
    if var.shape() != &[batch_size, seq_len] {
        return Err(NeuralError::Other(format!(
            "var shape ({:?}) doesn't match expected ({:?})",
            var.shape(), [batch_size, seq_len]
        )));
    }
    
    // Initialize gradients
    let mut dx = Array3::zeros(x.dim());
    let mut dgamma = Array1::zeros(features);
    let mut dbeta = Array1::zeros(features);
    
    // Compute dgamma and dbeta
    for j in 0..features {
        let mut dgamma_j = F::zero();
        let mut dbeta_j = F::zero();
        
        for b in 0..batch_size {
            for s in 0..seq_len {
                let std_dev = (var[[b, s]] + eps).sqrt();
                let x_norm = (x[[b, s, j]] - mean[[b, s]]) / std_dev;
                
                dgamma_j = dgamma_j + dout[[b, s, j]] * x_norm;
                dbeta_j = dbeta_j + dout[[b, s, j]];
            }
        }
        
        dgamma[j] = dgamma_j;
        dbeta[j] = dbeta_j;
    }
    
    // Compute dx
    for b in 0..batch_size {
        for s in 0..seq_len {
            let std_dev = (var[[b, s]] + eps).sqrt();
            let inv_std = F::one() / std_dev;
            let inv_features = F::one() / F::from(features).unwrap();
            
            // Compute intermediate values for the gradient
            let mut dxhat = Array1::zeros(features);
            let mut sum_dxhat = F::zero();
            let mut sum_dxhat_x = F::zero();
            
            for j in 0..features {
                let x_centered = x[[b, s, j]] - mean[[b, s]];
                dxhat[j] = dout[[b, s, j]] * gamma[j];
                sum_dxhat = sum_dxhat + dxhat[j];
                sum_dxhat_x = sum_dxhat_x + dxhat[j] * x_centered;
            }
            
            // Compute final gradient for each feature
            for j in 0..features {
                let x_centered = x[[b, s, j]] - mean[[b, s]];
                
                // Term 1: dL/dxhat * dxhat/dx
                let term1 = dxhat[j] * inv_std;
                
                // Term 2: dL/dvar * dvar/dx
                let dvar = sum_dxhat_x * F::from(-0.5).unwrap() * inv_std.powi(3);
                let dvar_dx = F::from(2.0).unwrap() * x_centered * inv_features;
                let term2 = dvar * dvar_dx;
                
                // Term 3: dL/dmean * dmean/dx
                let dmean = sum_dxhat * (-inv_std) + dvar * F::from(-2.0).unwrap() * 
                            inv_features * sum_dxhat_x;
                let dmean_dx = inv_features;
                let term3 = dmean * dmean_dx;
                
                // Combine terms
                dx[[b, s, j]] = term1 + term2 + term3;
            }
        }
    }
    
    Ok((dx, dgamma, dbeta))
}

/// Compute the backward pass for the transformer feed-forward network.
///
/// This function calculates the gradients of the loss with respect to
/// input and parameters for a transformer feed-forward network.
///
/// # Arguments
///
/// * `dout` - Gradient from upstream with shape [batch_size, seq_len, model_dim]
/// * `x` - Input tensor with shape [batch_size, seq_len, model_dim]
/// * `w1` - First weight matrix with shape [model_dim, ff_dim]
/// * `b1` - First bias vector with shape [ff_dim]
/// * `w2` - Second weight matrix with shape [ff_dim, model_dim]
/// * `b2` - Second bias vector with shape [model_dim]
/// * `hidden` - Intermediate hidden state with shape [batch_size, seq_len, ff_dim]
///
/// # Returns
///
/// * Tuple of (dx, dw1, db1, dw2, db2) gradients
///
/// # Examples
///
/// ```
/// use ndarray::{Array1, Array2, Array3};
/// use scirs2_neural::linalg::{transformer_ffn, transformer_ffn_backward};
///
/// // Setup parameters for a small feed-forward network
/// let batch_size = 2;
/// let seq_len = 3;
/// let model_dim = 4;
/// let ff_dim = 8;
///
/// // Input and parameters
/// let x = Array3::<f32>::ones((batch_size, seq_len, model_dim));
/// let w1 = Array2::<f32>::ones((model_dim, ff_dim));
/// let b1 = Array1::<f32>::zeros(ff_dim);
/// let w2 = Array2::<f32>::ones((ff_dim, model_dim));
/// let b2 = Array1::<f32>::zeros(model_dim);
///
/// // Forward pass
/// let mut hidden = Array3::<f32>::zeros((batch_size, seq_len, ff_dim));
/// for b in 0..batch_size {
///     for s in 0..seq_len {
///         for j in 0..ff_dim {
///             let mut sum = 0.0;
///             for i in 0..model_dim {
///                 sum += x[[b, s, i]] * w1[[i, j]];
///             }
///             let val = sum + b1[j];
///             // ReLU activation
///             hidden[[b, s, j]] = if val > 0.0 { val } else { 0.0 };
///         }
///     }
/// }
///
/// // Gradient from upstream
/// let dout = Array3::<f32>::ones((batch_size, seq_len, model_dim));
///
/// // Calculate gradients
/// let (dx, dw1, db1, dw2, db2) = transformer_ffn_backward(
///     &dout.view(), &x.view(), &w1.view(), &b1.view(), &w2.view(), &b2.view(), &hidden.view()
/// ).unwrap();
///
/// // Check gradient shapes
/// assert_eq!(dx.shape(), x.shape());
/// assert_eq!(dw1.shape(), w1.shape());
/// assert_eq!(db1.shape(), b1.shape());
/// assert_eq!(dw2.shape(), w2.shape());
/// assert_eq!(db2.shape(), b2.shape());
/// ```
pub fn transformer_ffn_backward<F>(
    dout: &ArrayView3<F>,
    x: &ArrayView3<F>,
    w1: &ArrayView2<F>,
    b1: &ArrayView1<F>,
    w2: &ArrayView2<F>,
    b2: &ArrayView1<F>,
    hidden: &ArrayView3<F>,
) -> Result<(Array3<F>, Array2<F>, Array1<F>, Array2<F>, Array1<F>)>
where
    F: Float + Debug + NumAssign,
{
    // Extract dimensions
    let batch_size = x.shape()[0];
    let seq_len = x.shape()[1];
    let model_dim = x.shape()[2];
    let ff_dim = w1.shape()[1];
    
    // Check shape compatibility
    if dout.shape() != x.shape() {
        return Err(NeuralError::Other(format!(
            "dout shape ({:?}) doesn't match x shape ({:?})",
            dout.shape(), x.shape()
        )));
    }
    
    if w1.shape()[0] != model_dim {
        return Err(NeuralError::Other(format!(
            "First weight matrix rows ({}) don't match model dimension ({})",
            w1.shape()[0], model_dim
        )));
    }
    
    if b1.len() != ff_dim {
        return Err(NeuralError::Other(format!(
            "First bias length ({}) doesn't match ff dimension ({})",
            b1.len(), ff_dim
        )));
    }
    
    if w2.shape()[0] != ff_dim || w2.shape()[1] != model_dim {
        return Err(NeuralError::Other(format!(
            "Second weight matrix shape {:?} doesn't match expected ({}, {})",
            w2.shape(), ff_dim, model_dim
        )));
    }
    
    if b2.len() != model_dim {
        return Err(NeuralError::Other(format!(
            "Second bias length ({}) doesn't match model dimension ({})",
            b2.len(), model_dim
        )));
    }
    
    if hidden.shape() != &[batch_size, seq_len, ff_dim] {
        return Err(NeuralError::Other(format!(
            "Hidden state shape ({:?}) doesn't match expected ({:?})",
            hidden.shape(), [batch_size, seq_len, ff_dim]
        )));
    }
    
    // Initialize gradients
    let mut dx = Array3::zeros(x.dim());
    let mut dw1 = Array2::zeros(w1.dim());
    let mut db1 = Array1::zeros(b1.dim());
    let mut dw2 = Array2::zeros(w2.dim());
    let mut db2 = Array1::zeros(b2.dim());
    
    // First, compute gradients for the second linear layer (no activation)
    
    // Compute dw2 and db2
    for i in 0..ff_dim {
        for j in 0..model_dim {
            let mut sum = F::zero();
            
            for b in 0..batch_size {
                for s in 0..seq_len {
                    sum = sum + hidden[[b, s, i]] * dout[[b, s, j]];
                }
            }
            
            dw2[[i, j]] = sum;
        }
    }
    
    for j in 0..model_dim {
        let mut sum = F::zero();
        
        for b in 0..batch_size {
            for s in 0..seq_len {
                sum = sum + dout[[b, s, j]];
            }
        }
        
        db2[j] = sum;
    }
    
    // Compute gradient with respect to hidden state
    let mut dhidden = Array3::zeros(hidden.dim());
    
    for b in 0..batch_size {
        for s in 0..seq_len {
            for i in 0..ff_dim {
                let mut sum = F::zero();
                
                for j in 0..model_dim {
                    sum = sum + dout[[b, s, j]] * w2[[i, j]];
                }
                
                dhidden[[b, s, i]] = sum;
            }
        }
    }
    
    // Apply ReLU derivative
    for b in 0..batch_size {
        for s in 0..seq_len {
            for i in 0..ff_dim {
                // ReLU derivative: 1 if x > 0, 0 otherwise
                if hidden[[b, s, i]] <= F::zero() {
                    dhidden[[b, s, i]] = F::zero();
                }
            }
        }
    }
    
    // Compute gradients for the first linear layer
    
    // Compute dw1 and db1
    for i in 0..model_dim {
        for j in 0..ff_dim {
            let mut sum = F::zero();
            
            for b in 0..batch_size {
                for s in 0..seq_len {
                    sum = sum + x[[b, s, i]] * dhidden[[b, s, j]];
                }
            }
            
            dw1[[i, j]] = sum;
        }
    }
    
    for j in 0..ff_dim {
        let mut sum = F::zero();
        
        for b in 0..batch_size {
            for s in 0..seq_len {
                sum = sum + dhidden[[b, s, j]];
            }
        }
        
        db1[j] = sum;
    }
    
    // Compute gradient with respect to input
    for b in 0..batch_size {
        for s in 0..seq_len {
            for i in 0..model_dim {
                let mut sum = F::zero();
                
                for j in 0..ff_dim {
                    sum = sum + dhidden[[b, s, j]] * w1[[i, j]];
                }
                
                dx[[b, s, i]] = sum;
            }
        }
    }
    
    Ok((dx, dw1, db1, dw2, db2))
}

/// Compute the backward pass for dropout.
///
/// This function calculates the gradient of the loss with respect to
/// the input of a dropout operation.
///
/// # Arguments
///
/// * `dout` - Gradient from upstream with same shape as output
/// * `mask` - Dropout mask from forward pass
///
/// # Returns
///
/// * Gradient with respect to input
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use scirs2_neural::linalg::{dropout, dropout_backward};
/// use rand::Rng;
///
/// // Input tensor
/// let x = Array2::<f32>::ones((2, 3));
/// let mut rng = rand::thread_rng();
/// 
/// // Forward pass during training
/// let (y_train, mask) = dropout(&x.view(), 0.5, true, &mut rng).unwrap();
///
/// // Gradient from upstream
/// let dout = Array2::<f32>::ones(y_train.dim());
///
/// // Calculate gradients
/// let dx = dropout_backward(&dout.view(), &mask.view()).unwrap();
///
/// // Check gradient shape
/// assert_eq!(dx.shape(), x.shape());
/// ```
pub fn dropout_backward<F, D>(
    dout: &ArrayView<F, D>,
    mask: &ArrayView<F, D>,
) -> Result<Array<F, D>>
where
    F: Float + Debug,
    D: Dimension,
{
    // Check shape compatibility
    if dout.shape() != mask.shape() {
        return Err(NeuralError::Other(format!(
            "dout shape ({:?}) doesn't match mask shape ({:?})",
            dout.shape(), mask.shape()
        )));
    }
    
    // Calculate gradient by applying mask
    let mut dx = Array::zeros(dout.dim());
    
    for (i, (&dout_val, &mask_val)) in dout.iter().zip(mask.iter()).enumerate() {
        let idx = dx.dim().default_index_for_elem(i);
        dx[idx] = dout_val * mask_val;
    }
    
    Ok(dx)
}

/// Compute the backward pass for adaptive average pooling.
///
/// This function calculates the gradient of the loss with respect to
/// the input of an adaptive average pooling operation.
///
/// # Arguments
///
/// * `dout` - Gradient from upstream with shape [batch_size, channels, out_height, out_width]
/// * `input` - Input tensor with shape [batch_size, channels, in_height, in_width]
/// * `output_size` - Output size from forward pass (out_height, out_width)
///
/// # Returns
///
/// * Gradient with respect to input
///
/// # Examples
///
/// ```
/// use ndarray::Array4;
/// use scirs2_neural::linalg::{adaptive_avg_pool2d, adaptive_avg_pool2d_backward};
///
/// // Create a batch of 2 feature maps, each with 3 channels and 4x4 size
/// let input = Array4::<f32>::ones((2, 3, 4, 4));
///
/// // Apply adaptive average pooling to get 2x2 output
/// let output = adaptive_avg_pool2d(&input.view(), (2, 2)).unwrap();
///
/// // Gradient from upstream
/// let dout = Array4::<f32>::ones(output.dim());
///
/// // Calculate gradients
/// let dinput = adaptive_avg_pool2d_backward(
///     &dout.view(), &input.view(), (2, 2)
/// ).unwrap();
///
/// // Check gradient shape
/// assert_eq!(dinput.shape(), input.shape());
/// ```
pub fn adaptive_avg_pool2d_backward<F>(
    dout: &ArrayView4<F>,
    input: &ArrayView4<F>,
    output_size: (usize, usize),
) -> Result<Array4<F>>
where
    F: Float + Debug + NumAssign,
{
    // Extract dimensions
    let batch_size = input.shape()[0];
    let channels = input.shape()[1];
    let in_height = input.shape()[2];
    let in_width = input.shape()[3];
    
    let out_height = output_size.0;
    let out_width = output_size.1;
    
    // Check shape compatibility
    if dout.shape()[0] != batch_size || dout.shape()[1] != channels {
        return Err(NeuralError::Other(format!(
            "dout dimensions don't match input: {:?} vs {:?}",
            dout.shape(), input.shape()
        )));
    }
    
    if dout.shape()[2] != out_height || dout.shape()[3] != out_width {
        return Err(NeuralError::Other(format!(
            "dout output size {:?} doesn't match expected {:?}",
            (dout.shape()[2], dout.shape()[3]), output_size
        )));
    }
    
    // Initialize gradient tensor
    let mut dinput = Array4::zeros(input.dim());
    
    // Calculate the gradient for each input element
    for b in 0..batch_size {
        for c in 0..channels {
            for oh in 0..out_height {
                // Calculate bin boundaries for height
                let h_start = (oh * in_height) / out_height;
                let h_end = ((oh + 1) * in_height) / out_height;
                let bin_h_size = h_end - h_start;
                
                for ow in 0..out_width {
                    // Calculate bin boundaries for width
                    let w_start = (ow * in_width) / out_width;
                    let w_end = ((ow + 1) * in_width) / out_width;
                    let bin_w_size = w_end - w_start;
                    
                    // Get gradient from output
                    let grad_val = dout[[b, c, oh, ow]];
                    
                    // Calculate the gradient for each element in the bin
                    // Distribute gradient uniformly across all elements in the bin
                    let bin_size = bin_h_size * bin_w_size;
                    let grad_per_element = grad_val / F::from(bin_size).unwrap();
                    
                    // Distribute gradient to all elements in the bin
                    for h in h_start..h_end {
                        for w in w_start..w_end {
                            dinput[[b, c, h, w]] = dinput[[b, c, h, w]] + grad_per_element;
                        }
                    }
                }
            }
        }
    }
    
    Ok(dinput)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array, Array1, Array2, Array3, Array4, Axis, Ix2, Ix3, Ix4};
    use approx::assert_relative_eq;
    use rand::rngs::StdRng;
    use rand::{SeedableRng, Rng};

    #[test]
    fn test_batch_matmul() {
        // Create batch of 2x2x3 matrices (batch_size=2, m=2, k=3)
        let a = Array::from_shape_vec(
            (2, 2, 3),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        ).unwrap();

        // Create batch of 2x3x2 matrices (batch_size=2, k=3, n=2)
        let b = Array::from_shape_vec(
            (2, 3, 2),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        ).unwrap();

        let c = batch_matmul(&a.view(), &b.view()).unwrap();

        // Verify shape
        assert_eq!(c.shape(), &[2, 2, 2]);

        // Check results for first batch
        assert_relative_eq!(c[[0, 0, 0]], 22.0);
        assert_relative_eq!(c[[0, 0, 1]], 28.0);
        assert_relative_eq!(c[[0, 1, 0]], 49.0);
        assert_relative_eq!(c[[0, 1, 1]], 64.0);

        // Check results for second batch
        assert_relative_eq!(c[[1, 0, 0]], 184.0);
        assert_relative_eq!(c[[1, 0, 1]], 202.0);
        assert_relative_eq!(c[[1, 1, 0]], 211.0);
        assert_relative_eq!(c[[1, 1, 1]], 238.0);
    }

    #[test]
    fn test_batch_vecmat() {
        // Create batch of vectors [batch_size=2, k=3]
        let v = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // Create matrix [k=3, n=2]
        let m = array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]];

        let result = batch_vecmat(&v.view(), &m.view()).unwrap();

        // Verify shape
        assert_eq!(result.shape(), &[2, 2]);

        // Check results for first batch
        assert_relative_eq!(result[[0, 0]], 1.0*0.1 + 2.0*0.3 + 3.0*0.5, epsilon = 1e-6);
        assert_relative_eq!(result[[0, 1]], 1.0*0.2 + 2.0*0.4 + 3.0*0.6, epsilon = 1e-6);

        // Check results for second batch
        assert_relative_eq!(result[[1, 0]], 4.0*0.1 + 5.0*0.3 + 6.0*0.5, epsilon = 1e-6);
        assert_relative_eq!(result[[1, 1]], 4.0*0.2 + 5.0*0.4 + 6.0*0.6, epsilon = 1e-6);
    }

    #[test]
    fn test_batch_norm() {
        // Input: batch_size=2, features=3
        let x = array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
        let gamma = Array1::ones(3);
        let beta = Array1::zeros(3);
        let eps = 1e-5;

        let (y, mean, var) = batch_norm_forward(&x.view(), &gamma.view(), &beta.view(), eps).unwrap();

        // Check shapes
        assert_eq!(y.shape(), &[2, 3]);
        assert_eq!(mean.shape(), &[3]);
        assert_eq!(var.shape(), &[3]);

        // Check means
        assert_relative_eq!(mean[0], 0.25, epsilon = 1e-6);
        assert_relative_eq!(mean[1], 0.35, epsilon = 1e-6);
        assert_relative_eq!(mean[2], 0.45, epsilon = 1e-6);

        // Check variances
        assert_relative_eq!(var[0], 0.045, epsilon = 1e-6);
        assert_relative_eq!(var[1], 0.045, epsilon = 1e-6);
        assert_relative_eq!(var[2], 0.045, epsilon = 1e-6);

        // Check that the mean of normalized values is 0 and variance is 1
        for j in 0..3 {
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            
            for i in 0..2 {
                let norm_value = (x[[i, j]] - mean[j]) / (var[j] + eps).sqrt();
                sum += norm_value;
                sum_sq += norm_value * norm_value;
            }
            
            let norm_mean = sum / 2.0;
            let norm_var = sum_sq / 2.0 - norm_mean * norm_mean;
            
            assert_relative_eq!(norm_mean, 0.0, epsilon = 1e-6);
            assert_relative_eq!(norm_var, 1.0, epsilon = 1e-6);
        }

        // Test backward pass
        let dout = array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]];
        let (dx, dgamma, dbeta) = batch_norm_backward(
            &dout.view(), &x.view(), &gamma.view(), &mean.view(), &var.view(), eps
        ).unwrap();

        // Check shapes
        assert_eq!(dx.shape(), &[2, 3]);
        assert_eq!(dgamma.shape(), &[3]);
        assert_eq!(dbeta.shape(), &[3]);

        // Check that the sum of dx is close to 0 (derivation of mean constraint)
        for j in 0..3 {
            let dx_sum = dx[[0, j]] + dx[[1, j]];
            assert_relative_eq!(dx_sum, 0.0, epsilon = 1e-6);
        }

        // Check dbeta (should be the sum of dout)
        for j in 0..3 {
            assert_relative_eq!(dbeta[j], 2.0, epsilon = 1e-6);
        }
    }
    
    #[test]
    fn test_conv2d() {
        // Create a batch of 2 images, each with 1 channel and 3x3 size
        let input = array![
            // Batch 1
            [[
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0]
            ]],
            // Batch 2
            [[
                [9.0, 8.0, 7.0],
                [6.0, 5.0, 4.0],
                [3.0, 2.0, 1.0]
            ]]
        ];
        
        // Create 2 filters, each for 1 input channel and 2x2 kernel size
        let filters = array![
            // Filter 1
            [[
                [1.0, 0.0],
                [0.0, 1.0]
            ]],
            // Filter 2
            [[
                [0.0, 1.0],
                [1.0, 0.0]
            ]]
        ];
        
        // Apply convolution with stride 1 and 'valid' padding
        let output = conv2d(&input.view(), &filters.view(), (1, 1), "valid").unwrap();
        
        // Output shape should be [2, 2, 2, 2]
        assert_eq!(output.shape(), &[2, 2, 2, 2]);
        
        // Check results for first batch, first filter
        assert_relative_eq!(output[[0, 0, 0, 0]], 1.0 * 1.0 + 2.0 * 0.0 + 4.0 * 0.0 + 5.0 * 1.0, epsilon = 1e-6);
        assert_relative_eq!(output[[0, 0, 0, 1]], 2.0 * 1.0 + 3.0 * 0.0 + 5.0 * 0.0 + 6.0 * 1.0, epsilon = 1e-6);
        assert_relative_eq!(output[[0, 0, 1, 0]], 4.0 * 1.0 + 5.0 * 0.0 + 7.0 * 0.0 + 8.0 * 1.0, epsilon = 1e-6);
        assert_relative_eq!(output[[0, 0, 1, 1]], 5.0 * 1.0 + 6.0 * 0.0 + 8.0 * 0.0 + 9.0 * 1.0, epsilon = 1e-6);
        
        // Apply convolution with stride 1 and 'same' padding
        let output_same = conv2d(&input.view(), &filters.view(), (1, 1), "same").unwrap();
        
        // Output shape should be [2, 2, 3, 3]
        assert_eq!(output_same.shape(), &[2, 2, 3, 3]);
    }
    
    #[test]
    fn test_max_pool2d() {
        // Create a batch of 2 feature maps, each with 1 channel and 4x4 size
        let input = array![
            // Batch 1
            [[
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0]
            ]],
            // Batch 2
            [[
                [16.0, 15.0, 14.0, 13.0],
                [12.0, 11.0, 10.0, 9.0],
                [8.0, 7.0, 6.0, 5.0],
                [4.0, 3.0, 2.0, 1.0]
            ]]
        ];
        
        // Apply max pooling with 2x2 pool size and stride 2
        let (output, indices) = max_pool2d(&input.view(), (2, 2), (2, 2)).unwrap();
        
        // Output shape should be [2, 1, 2, 2]
        assert_eq!(output.shape(), &[2, 1, 2, 2]);
        
        // Check results for first batch
        assert_relative_eq!(output[[0, 0, 0, 0]], 6.0, epsilon = 1e-6);  // max of [1, 2, 5, 6]
        assert_relative_eq!(output[[0, 0, 0, 1]], 8.0, epsilon = 1e-6);  // max of [3, 4, 7, 8]
        assert_relative_eq!(output[[0, 0, 1, 0]], 14.0, epsilon = 1e-6); // max of [9, 10, 13, 14]
        assert_relative_eq!(output[[0, 0, 1, 1]], 16.0, epsilon = 1e-6); // max of [11, 12, 15, 16]
        
        // Check results for second batch
        assert_relative_eq!(output[[1, 0, 0, 0]], 16.0, epsilon = 1e-6); // max of [16, 15, 12, 11]
        assert_relative_eq!(output[[1, 0, 0, 1]], 14.0, epsilon = 1e-6); // max of [14, 13, 10, 9]
        assert_relative_eq!(output[[1, 0, 1, 0]], 8.0, epsilon = 1e-6);  // max of [8, 7, 4, 3]
        assert_relative_eq!(output[[1, 0, 1, 1]], 6.0, epsilon = 1e-6);  // max of [6, 5, 2, 1]
    }
    
    #[test]
    fn test_scaled_dot_product_attention() {
        // Create queries, keys, and values for a single batch
        let queries = array![[[1.0, 0.0], [0.0, 1.0]]]; // batch_size=1, seq_len_q=2, depth=2
        let keys = array![[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]; // batch_size=1, seq_len_k=3, depth=2
        let values = array![[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]; // batch_size=1, seq_len_k=3, depth_v=2
        
        // Calculate attention
        let (output, weights) = scaled_dot_product_attention(
            &queries.view(), &keys.view(), &values.view(), None
        ).unwrap();
        
        // Check shapes
        assert_eq!(output.shape(), &[1, 2, 2]);
        assert_eq!(weights.shape(), &[1, 2, 3]);
        
        // For query [1, 0] with keys [[1, 0], [0, 1], [1, 1]], dot products are [1, 0, 1]
        // After scaling by 1/sqrt(2) and softmax, weights should favor the first and third keys
        // For query [0, 1] with keys [[1, 0], [0, 1], [1, 1]], dot products are [0, 1, 1]
        // After scaling and softmax, weights should favor the second and third keys
        
        // Check that the weights follow the expected pattern
        assert!(weights[[0, 0, 0]] > 0.3); // First query matches first key strongly
        assert!(weights[[0, 0, 1]] < 0.1); // First query doesn't match second key
        assert!(weights[[0, 0, 2]] > 0.3); // First query matches third key strongly
        
        assert!(weights[[0, 1, 0]] < 0.1); // Second query doesn't match first key
        assert!(weights[[0, 1, 1]] > 0.3); // Second query matches second key strongly
        assert!(weights[[0, 1, 2]] > 0.3); // Second query matches third key strongly
        
        // Test with mask
        let mask = array![[[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]]; // mask out key 1 for query 0, and key 2 for query 1
        
        let (output_masked, weights_masked) = scaled_dot_product_attention(
            &queries.view(), &keys.view(), &values.view(), Some(&mask.view())
        ).unwrap();
        
        // Check that masked attention weights are zero
        assert_relative_eq!(weights_masked[[0, 0, 1]], 0.0, epsilon = 1e-6);
        assert_relative_eq!(weights_masked[[0, 1, 2]], 0.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_multi_head_attention() {
        // Setup parameters for a simple 2-head attention with model_dim=4 and head_dim=2
        let batch_size = 1;
        let seq_len_q = 2;
        let seq_len_k = 3;
        let model_dim = 4;
        let head_dim = 2;
        let num_heads = 2;
        
        // Create input tensors
        let queries = Array3::<f32>::ones((batch_size, seq_len_q, model_dim));
        let keys = Array3::<f32>::ones((batch_size, seq_len_k, model_dim));
        let values = Array3::<f32>::ones((batch_size, seq_len_k, model_dim));
        
        // Create simple projection weights
        // For the first head, project to the first half of the dimension
        // For the second head, project to the second half
        let mut w_q = Array3::<f32>::zeros((num_heads, model_dim, head_dim));
        let mut w_k = Array3::<f32>::zeros((num_heads, model_dim, head_dim));
        let mut w_v = Array3::<f32>::zeros((num_heads, model_dim, head_dim));
        
        // First head projections
        w_q[[0, 0, 0]] = 1.0; w_q[[0, 1, 1]] = 1.0;
        w_k[[0, 0, 0]] = 1.0; w_k[[0, 1, 1]] = 1.0;
        w_v[[0, 0, 0]] = 1.0; w_v[[0, 1, 1]] = 1.0;
        
        // Second head projections
        w_q[[1, 2, 0]] = 1.0; w_q[[1, 3, 1]] = 1.0;
        w_k[[1, 2, 0]] = 1.0; w_k[[1, 3, 1]] = 1.0;
        w_v[[1, 2, 0]] = 1.0; w_v[[1, 3, 1]] = 1.0;
        
        // Output projection: identity mapping
        let mut w_o = Array2::<f32>::zeros((model_dim, num_heads * head_dim));
        w_o[[0, 0]] = 1.0; w_o[[1, 1]] = 1.0; w_o[[2, 2]] = 1.0; w_o[[3, 3]] = 1.0;
        
        // Calculate multi-head attention
        let output = multi_head_attention(
            &queries.view(), &keys.view(), &values.view(),
            &w_q.view(), &w_k.view(), &w_v.view(), &w_o.view(),
            None
        ).unwrap();
        
        // Check output shape
        assert_eq!(output.shape(), &[batch_size, seq_len_q, model_dim]);
        
        // Since inputs are all ones and weights are set to create an identity-like mapping,
        // the output should be close to the input for each query
        for b in 0..batch_size {
            for q in 0..seq_len_q {
                for d in 0..model_dim {
                    // Each query attends to all keys equally, so output approximates the average of values
                    assert!(output[[b, q, d]] > 0.0);
                }
            }
        }
    }
    
    #[test]
    fn test_lstm_cell() {
        // Setup parameters for a small LSTM cell
        let batch_size = 2;
        let input_size = 3;
        let hidden_size = 2;
        
        // Create input and states with known values
        let x = array![[0.5, 0.4, 0.3], [0.2, 0.1, 0.0]];
        let h_prev = array![[0.1, 0.2], [0.3, 0.4]];
        let c_prev = array![[0.2, 0.3], [0.4, 0.5]];
        
        // Create weights that will produce predictable outputs
        // We set all weights to small values to ensure the activations don't saturate
        let w_ih = Array2::<f32>::from_elem((4 * hidden_size, input_size), 0.1);
        let w_hh = Array2::<f32>::from_elem((4 * hidden_size, hidden_size), 0.1);
        
        // Bias terms
        let b_ih = Array1::<f32>::from_elem(4 * hidden_size, 0.0);
        let b_hh = Array1::<f32>::from_elem(4 * hidden_size, 0.0);
        
        // Calculate LSTM cell forward pass
        let (h_new, c_new) = lstm_cell(
            &x.view(), &h_prev.view(), &c_prev.view(),
            &w_ih.view(), &w_hh.view(), &b_ih.view(), &b_hh.view()
        ).unwrap();
        
        // Check output shapes
        assert_eq!(h_new.shape(), &[batch_size, hidden_size]);
        assert_eq!(c_new.shape(), &[batch_size, hidden_size]);
        
        // Since we used small positive weights and zero biases:
        // - Input and forget gates should open partially (sigmoid(small positive) ≈ 0.5-0.6)
        // - Cell input should be small positive (tanh(small positive) ≈ small positive)
        // - Output gate should open partially
        // - New cell state should increase slightly from previous
        // - New hidden state should be positive
        
        // Verify cell state changes
        for b in 0..batch_size {
            for h in 0..hidden_size {
                // Cell state should be different from previous
                assert!(c_new[[b, h]] != c_prev[[b, h]]);
                
                // Hidden state should be positive
                assert!(h_new[[b, h]] > 0.0);
            }
        }
        
        // The first batch has larger inputs than the second,
        // so its influence on the gate activations should be stronger
        assert!(c_new[[0, 0]].abs() > c_new[[1, 0]].abs() || 
                c_new[[0, 1]].abs() > c_new[[1, 1]].abs());
    }
    
    #[test]
    fn test_batch_matmul_backward() {
        // Create batch of 2x2x3 matrices (batch_size=2, m=2, k=3)
        let a = array![
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
        ];
        
        // Create batch of 2x3x2 matrices (batch_size=2, k=3, n=2)
        let b = array![
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
        ];
        
        // Compute forward pass
        let c = batch_matmul(&a.view(), &b.view()).unwrap();
        
        // Create a loss gradient (in a real scenario, this would come from the loss function)
        let grad_output = Array3::<f32>::ones(c.dim());
        
        // Compute backward pass
        let (grad_a, grad_b) = batch_matmul_backward(
            &grad_output.view(), &a.view(), &b.view(), true, true
        ).unwrap();
        
        // Check shapes
        assert!(grad_a.is_some());
        assert!(grad_b.is_some());
        let grad_a = grad_a.unwrap();
        let grad_b = grad_b.unwrap();
        assert_eq!(grad_a.shape(), &[2, 2, 3]);
        assert_eq!(grad_b.shape(), &[2, 3, 2]);
        
        // Verify gradients using the formula:
        // dL/dA_ijk = sum_m dL/dC_ijm * B_jkm
        // dL/dB_ijk = sum_m dL/dC_mik * A_mjk
        
        // Since grad_output is all ones, we can simplify:
        // dL/dA_ijk = sum_m B_jkm
        // dL/dB_ijk = sum_m A_mjk
        
        // Check a few specific gradient values
        assert_relative_eq!(
            grad_a[[0, 0, 0]], 
            b[[0, 0, 0]] + b[[0, 0, 1]], 
            epsilon = 1e-6
        );
        
        assert_relative_eq!(
            grad_b[[1, 2, 1]], 
            a[[1, 0, 2]] + a[[1, 1, 2]], 
            epsilon = 1e-6
        );
    }
    
    #[test]
    fn test_layer_norm() {
        // Input: batch_size=2, seq_len=2, features=3
        let x = array![
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]
        ];
        let gamma = array![1.0, 0.5, 2.0];
        let beta = array![0.0, 0.1, -0.1];
        
        let y = layer_norm(&x.view(), &gamma.view(), &beta.view(), 1e-5).unwrap();
        
        // Check output shape
        assert_eq!(y.shape(), &[2, 2, 3]);
        
        // Check that for each sequence position, the normalized values (before gamma/beta)
        // have mean ≈ 0 and variance ≈ 1
        for b in 0..2 {
            for s in 0..2 {
                // Calculate mean of the original inputs at this sequence position
                let mean_x = (x[[b, s, 0]] + x[[b, s, 1]] + x[[b, s, 2]]) / 3.0;
                
                // Calculate variance
                let var_x = ((x[[b, s, 0]] - mean_x).powi(2) +
                            (x[[b, s, 1]] - mean_x).powi(2) +
                            (x[[b, s, 2]] - mean_x).powi(2)) / 3.0;
                
                let std_dev = (var_x + 1e-5).sqrt();
                
                // Normalized values (manually calculated)
                let x_norm_0 = (x[[b, s, 0]] - mean_x) / std_dev;
                let x_norm_1 = (x[[b, s, 1]] - mean_x) / std_dev;
                let x_norm_2 = (x[[b, s, 2]] - mean_x) / std_dev;
                
                // Expected output after applying gamma and beta
                let expected_0 = x_norm_0 * gamma[0] + beta[0];
                let expected_1 = x_norm_1 * gamma[1] + beta[1];
                let expected_2 = x_norm_2 * gamma[2] + beta[2];
                
                // Check that the output matches our expectations
                assert_relative_eq!(y[[b, s, 0]], expected_0, epsilon = 1e-6);
                assert_relative_eq!(y[[b, s, 1]], expected_1, epsilon = 1e-6);
                assert_relative_eq!(y[[b, s, 2]], expected_2, epsilon = 1e-6);
                
                // Verify mean ≈ 0 and variance ≈ 1 for normalized values
                let mean_norm = (x_norm_0 + x_norm_1 + x_norm_2) / 3.0;
                let var_norm = ((x_norm_0 - mean_norm).powi(2) +
                                (x_norm_1 - mean_norm).powi(2) +
                                (x_norm_2 - mean_norm).powi(2)) / 3.0;
                
                assert_relative_eq!(mean_norm, 0.0, epsilon = 1e-6);
                assert_relative_eq!(var_norm, 1.0, epsilon = 1e-6);
            }
        }
    }
    
    #[test]
    fn test_dropout() {
        // Use a fixed seed for reproducibility
        let mut rng = StdRng::seed_from_u64(42);
        
        // Create input tensor
        let x = array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]];
        
        // Test training mode with 50% dropout
        let dropout_prob = 0.5;
        let (y_train, mask) = dropout(&x.view(), dropout_prob, true, &mut rng).unwrap();
        
        // Check shapes
        assert_eq!(y_train.shape(), &[2, 3]);
        assert_eq!(mask.shape(), &[2, 3]);
        
        // Count number of dropped elements
        let mut dropped_count = 0;
        for &val in mask.iter() {
            if val == 0.0 {
                dropped_count += 1;
            } else {
                // Check that non-dropped values are scaled by 1/(1-p)
                assert_relative_eq!(val, 1.0 / (1.0 - dropout_prob), epsilon = 1e-6);
            }
        }
        
        // With 6 elements and 50% dropout, expect roughly 3 dropped elements
        // but due to randomness, it could vary
        assert!(dropped_count > 0 && dropped_count < 6);
        
        // For non-dropped elements, check that output = input * mask
        for i in 0..2 {
            for j in 0..3 {
                assert_relative_eq!(y_train[[i, j]], x[[i, j]] * mask[[i, j]], epsilon = 1e-6);
            }
        }
        
        // Test inference mode (no dropout)
        let (y_test, mask_test) = dropout(&x.view(), dropout_prob, false, &mut rng).unwrap();
        
        // Check that all values pass through unchanged in inference mode
        for i in 0..2 {
            for j in 0..3 {
                assert_relative_eq!(y_test[[i, j]], x[[i, j]], epsilon = 1e-6);
                assert_relative_eq!(mask_test[[i, j]], 1.0, epsilon = 1e-6);
            }
        }
    }
    
    #[test]
    fn test_log_softmax() {
        // Input: batch_size=2, classes=3
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ];
        
        // Apply log-softmax along class dimension (axis=1)
        let y = log_softmax(&x.view(), Axis(1)).unwrap();
        
        // Check shape
        assert_eq!(y.shape(), &[2, 3]);
        
        // Manually compute expected log-softmax values for the first batch
        let max_val_0 = 3.0; // max of [1.0, 2.0, 3.0]
        let sum_exp_0 = (1.0 - max_val_0).exp() + (2.0 - max_val_0).exp() + (3.0 - max_val_0).exp();
        let log_sum_exp_0 = sum_exp_0.ln() + max_val_0;
        
        let expected_0_0 = 1.0 - log_sum_exp_0;
        let expected_0_1 = 2.0 - log_sum_exp_0;
        let expected_0_2 = 3.0 - log_sum_exp_0;
        
        // Check first batch values
        assert_relative_eq!(y[[0, 0]], expected_0_0, epsilon = 1e-6);
        assert_relative_eq!(y[[0, 1]], expected_0_1, epsilon = 1e-6);
        assert_relative_eq!(y[[0, 2]], expected_0_2, epsilon = 1e-6);
        
        // Manually compute expected log-softmax values for the second batch
        let max_val_1 = 6.0; // max of [4.0, 5.0, 6.0]
        let sum_exp_1 = (4.0 - max_val_1).exp() + (5.0 - max_val_1).exp() + (6.0 - max_val_1).exp();
        let log_sum_exp_1 = sum_exp_1.ln() + max_val_1;
        
        let expected_1_0 = 4.0 - log_sum_exp_1;
        let expected_1_1 = 5.0 - log_sum_exp_1;
        let expected_1_2 = 6.0 - log_sum_exp_1;
        
        // Check second batch values
        assert_relative_eq!(y[[1, 0]], expected_1_0, epsilon = 1e-6);
        assert_relative_eq!(y[[1, 1]], expected_1_1, epsilon = 1e-6);
        assert_relative_eq!(y[[1, 2]], expected_1_2, epsilon = 1e-6);
        
        // Verify that sum(exp(log_softmax)) ≈ 1 for each batch
        for i in 0..2 {
            let sum_exp = y[[i, 0]].exp() + y[[i, 1]].exp() + y[[i, 2]].exp();
            assert_relative_eq!(sum_exp, 1.0, epsilon = 1e-6);
        }
    }
    
    #[test]
    fn test_positional_encoding() {
        // Test for sequence length 10 and model dimension 16
        let seq_len = 10;
        let model_dim = 16;
        
        let pos_encoding = positional_encoding::<f32>(seq_len, model_dim).unwrap();
        
        // Check shape
        assert_eq!(pos_encoding.shape(), &[seq_len, model_dim]);
        
        // Check that sine is used for even indices and cosine for odd indices
        // For position 0, sin(0) = 0 and cos(0) = 1
        assert_relative_eq!(pos_encoding[[0, 0]], 0.0, epsilon = 1e-6); // sin(0)
        assert_relative_eq!(pos_encoding[[0, 1]], 1.0, epsilon = 1e-6); // cos(0)
        
        // Check periodicity properties
        // For large dimensions, the period is very long
        // For dimension 0 (i=0), the period is 2π * 10000^0 = 2π
        // For dimension 2 (i=1), the period is 2π * 10000^(2/16) ≈ 5.8π
        
        // Values at position 1 should be different from position 0
        assert!(pos_encoding[[1, 0]] != pos_encoding[[0, 0]]);
        assert!(pos_encoding[[1, 1]] != pos_encoding[[0, 1]]);
        
        // Different dimensions should have different frequencies
        let freq_dim0 = pos_encoding[[1, 0]].atan2(pos_encoding[[1, 1]]);
        let freq_dim2 = pos_encoding[[1, 2]].atan2(pos_encoding[[1, 3]]);
        assert!(freq_dim0 != freq_dim2);
    }
    
    #[test]
    fn test_transformer_ffn() {
        // Setup parameters for a small feed-forward network
        let batch_size = 2;
        let seq_len = 3;
        let model_dim = 4;
        let ff_dim = 8;
        
        // Input and parameters with known values
        let x = Array3::<f32>::ones((batch_size, seq_len, model_dim));
        let w1 = Array2::<f32>::from_elem((model_dim, ff_dim), 0.1);
        let b1 = Array1::<f32>::zeros(ff_dim);
        let w2 = Array2::<f32>::from_elem((ff_dim, model_dim), 0.1);
        let b2 = Array1::<f32>::zeros(model_dim);
        
        // Apply feed-forward network
        let output = transformer_ffn(
            &x.view(), &w1.view(), &b1.view(), &w2.view(), &b2.view()
        ).unwrap();
        
        // Check output shape
        assert_eq!(output.shape(), &[batch_size, seq_len, model_dim]);
        
        // For input tensor of 1s, with weights 0.1:
        // First layer output before ReLU: 1*0.1*4 = 0.4 for each element in hidden layer
        // After ReLU: 0.4 (unchanged, since positive)
        // Second layer output: 0.4*0.1*8 = 0.32 for each element in output layer
        
        // Verify output values
        for b in 0..batch_size {
            for s in 0..seq_len {
                for d in 0..model_dim {
                    assert_relative_eq!(output[[b, s, d]], 0.32, epsilon = 1e-6);
                }
            }
        }
    }
    
    #[test]
    fn test_im2col_col2im() {
        // Create a small input tensor with known values
        let input = array![
            // Batch 1
            [[
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0]
            ]]
        ];
        
        // Apply im2col with 2x2 kernel and stride 1
        let cols = im2col(&input.view(), (2, 2), (1, 1), "valid").unwrap();
        
        // Expected shape: [batch_size, out_height * out_width, in_channels * kernel_height * kernel_width]
        // For 3x3 input, 2x2 kernel, stride 1, valid padding: out_height=out_width=2
        // So shape is [1, 4, 4]
        assert_eq!(cols.shape(), &[1, 4, 4]);
        
        // Check the first patch (top-left 2x2)
        assert_relative_eq!(cols[[0, 0, 0]], 1.0, epsilon = 1e-6);
        assert_relative_eq!(cols[[0, 0, 1]], 2.0, epsilon = 1e-6);
        assert_relative_eq!(cols[[0, 0, 2]], 4.0, epsilon = 1e-6);
        assert_relative_eq!(cols[[0, 0, 3]], 5.0, epsilon = 1e-6);
        
        // Apply col2im to convert back
        let output = col2im(&cols.view(), (1, 1, 3, 3), (2, 2), (1, 1), "valid").unwrap();
        
        // Output shape should match input shape
        assert_eq!(output.shape(), input.shape());
        
        // Values should match the original input, but accumulated where patches overlap
        // For stride 1, each pixel appears in up to 4 patches (if it's in the middle)
        assert_relative_eq!(output[[0, 0, 0, 0]], 1.0, epsilon = 1e-6); // top-left (appears in 1 patch)
        assert_relative_eq!(output[[0, 0, 0, 1]], 2.0 * 2.0, epsilon = 1e-6); // top-middle (appears in 2 patches)
        assert_relative_eq!(output[[0, 0, 1, 1]], 5.0 * 4.0, epsilon = 1e-6); // middle (appears in 4 patches)
    }
    
    #[test]
    fn test_adaptive_avg_pool2d() {
        // Create a batch of 2 feature maps, each with 1 channel and 4x4 size
        let input = array![
            // Batch 1
            [[
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0]
            ]],
            // Batch 2
            [[
                [16.0, 15.0, 14.0, 13.0],
                [12.0, 11.0, 10.0, 9.0],
                [8.0, 7.0, 6.0, 5.0],
                [4.0, 3.0, 2.0, 1.0]
            ]]
        ];
        
        // Apply adaptive average pooling to get 2x2 output
        let output = adaptive_avg_pool2d(&input.view(), (2, 2)).unwrap();
        
        // Check output shape
        assert_eq!(output.shape(), &[2, 1, 2, 2]);
        
        // For 4x4 input to 2x2 output, each output pixel should average a 2x2 block
        
        // Check output values for first batch
        // Top-left: average of [1, 2, 5, 6]
        assert_relative_eq!(output[[0, 0, 0, 0]], (1.0 + 2.0 + 5.0 + 6.0) / 4.0, epsilon = 1e-6);
        
        // Top-right: average of [3, 4, 7, 8]
        assert_relative_eq!(output[[0, 0, 0, 1]], (3.0 + 4.0 + 7.0 + 8.0) / 4.0, epsilon = 1e-6);
        
        // Bottom-left: average of [9, 10, 13, 14]
        assert_relative_eq!(output[[0, 0, 1, 0]], (9.0 + 10.0 + 13.0 + 14.0) / 4.0, epsilon = 1e-6);
        
        // Bottom-right: average of [11, 12, 15, 16]
        assert_relative_eq!(output[[0, 0, 1, 1]], (11.0 + 12.0 + 15.0 + 16.0) / 4.0, epsilon = 1e-6);
        
        // Test with non-divisible input size
        let output_odd = adaptive_avg_pool2d(&input.view(), (3, 3)).unwrap();
        assert_eq!(output_odd.shape(), &[2, 1, 3, 3]);
    }
    
    #[test]
    fn test_clip_grad_norm() {
        // Create gradient tensors with known L2 norms
        let grad1 = array![3.0, 4.0]; // L2 norm = 5
        let grad2 = array![[5.0, 0.0], [0.0, 12.0]]; // L2 norm = 13
        
        // Combined L2 norm = sqrt(5^2 + 13^2) = sqrt(194) ≈ 13.93
        
        // Clip gradients with max_norm = 10.0
        let grads = vec![grad1.view(), grad2.view()];
        let clipped_grads = clip_grad_norm(&grads, 10.0).unwrap();
        
        // Check shapes
        assert_eq!(clipped_grads[0].shape(), grad1.shape());
        assert_eq!(clipped_grads[1].shape(), grad2.shape());
        
        // Calculate the scaling factor
        let orig_norm = (5.0f64.powi(2) + 13.0f64.powi(2)).sqrt();
        let scale = 10.0 / orig_norm;
        
        // Check that gradients are scaled correctly
        assert_relative_eq!(clipped_grads[0][0], 3.0 * scale, epsilon = 1e-6);
        assert_relative_eq!(clipped_grads[0][1], 4.0 * scale, epsilon = 1e-6);
        assert_relative_eq!(clipped_grads[1][[0, 0]], 5.0 * scale, epsilon = 1e-6);
        assert_relative_eq!(clipped_grads[1][[1, 1]], 12.0 * scale, epsilon = 1e-6);
        
        // Check that the combined L2 norm is equal to max_norm
        let combined_norm_squared = clipped_grads[0].iter().map(|&x| x*x).sum::<f64>() +
                                    clipped_grads[1].iter().map(|&x| x*x).sum::<f64>();
        let combined_norm = combined_norm_squared.sqrt();
        assert_relative_eq!(combined_norm, 10.0, epsilon = 1e-6);
        
        // Test with grad norm < max_norm (no clipping should occur)
        let grad3 = array![1.0, 2.0]; // L2 norm = sqrt(5) ≈ 2.24
        let grads_small = vec![grad3.view()];
        let clipped_grads_small = clip_grad_norm(&grads_small, 10.0).unwrap();
        
        // Grads should be unchanged
        assert_relative_eq!(clipped_grads_small[0][0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(clipped_grads_small[0][1], 2.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_adam_update() {
        // Initialize parameters with known values
        let param = array![0.0, 0.0, 0.0];
        let grad = array![0.1, 0.2, 0.3];
        let m = array![0.0, 0.0, 0.0];
        let v = array![0.0, 0.0, 0.0];
        
        // Adam hyperparameters
        let lr = 0.001;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;
        let t = 1;
        
        // Apply Adam update
        let (param_new, m_new, v_new) = adam_update(
            &param.view(), &grad.view(), &m.view(), &v.view(),
            lr, beta1, beta2, eps, t
        ).unwrap();
        
        // Check output shapes
        assert_eq!(param_new.shape(), param.shape());
        assert_eq!(m_new.shape(), m.shape());
        assert_eq!(v_new.shape(), v.shape());
        
        // Compute expected values
        
        // First moment: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        let expected_m = array![
            beta1 * 0.0 + (1.0 - beta1) * 0.1,
            beta1 * 0.0 + (1.0 - beta1) * 0.2,
            beta1 * 0.0 + (1.0 - beta1) * 0.3
        ];
        
        // Second moment: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        let expected_v = array![
            beta2 * 0.0 + (1.0 - beta2) * 0.1 * 0.1,
            beta2 * 0.0 + (1.0 - beta2) * 0.2 * 0.2,
            beta2 * 0.0 + (1.0 - beta2) * 0.3 * 0.3
        ];
        
        // Bias correction
        let correction1 = 1.0 - beta1.powi(t as i32);
        let correction2 = 1.0 - beta2.powi(t as i32);
        
        let expected_m_hat = array![
            expected_m[0] / correction1,
            expected_m[1] / correction1,
            expected_m[2] / correction1
        ];
        
        let expected_v_hat = array![
            expected_v[0] / correction2,
            expected_v[1] / correction2,
            expected_v[2] / correction2
        ];
        
        // Parameter update: theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)
        let expected_param = array![
            param[0] - lr * expected_m_hat[0] / (expected_v_hat[0].sqrt() + eps),
            param[1] - lr * expected_m_hat[1] / (expected_v_hat[1].sqrt() + eps),
            param[2] - lr * expected_m_hat[2] / (expected_v_hat[2].sqrt() + eps)
        ];
        
        // Verify results
        for i in 0..3 {
            assert_relative_eq!(m_new[i], expected_m[i], epsilon = 1e-6);
            assert_relative_eq!(v_new[i], expected_v[i], epsilon = 1e-6);
            assert_relative_eq!(param_new[i], expected_param[i], epsilon = 1e-6);
        }
    }
    
    #[test]
    fn test_conv2d_backward() {
        // Create a small input tensor
        let input = Array4::<f32>::ones((2, 1, 4, 4));
        
        // Create a small filter tensor
        let filters = Array4::<f32>::ones((2, 1, 3, 3));
        
        // Bias
        let bias = Array1::<f32>::zeros(2);
        
        // Calculate output shape for 'valid' padding
        let out_height = 4 - 3 + 1; // 2
        let out_width = 4 - 3 + 1;  // 2
        
        // Create gradient from upstream
        let dout = Array4::<f32>::ones((2, 2, out_height, out_width));
        
        // Calculate gradients
        let (dinput, dfilters, dbias) = conv2d_backward(
            &dout.view(), &input.view(), &filters.view(), Some(&bias.view()), (1, 1), "valid"
        ).unwrap();
        
        // Check shapes
        assert_eq!(dinput.shape(), input.shape());
        assert_eq!(dfilters.shape(), filters.shape());
        assert!(dbias.is_some());
        assert_eq!(dbias.unwrap().shape(), bias.shape());
        
        // For this simple case with all ones, we can compute expected values:
        // Each element in dinput receives gradients from all filter positions that include it
        // Each element in dfilters receives gradients from all input positions it's applied to
        // Each element in dbias receives the sum of gradients for its output channel
        
        // Verify dbias
        let dbias_expected = array![4.0, 4.0]; // 2 batches * 2 output pixels for each channel
    }
    
    #[test]
    fn test_max_pool2d_backward() {
        // Create a batch of 1 feature map, 1 channel, 4x4 size with increasing values
        let mut input = Array4::<f32>::zeros((1, 1, 4, 4));
        for i in 0..4 {
            for j in 0..4 {
                input[[0, 0, i, j]] = (i * 4 + j + 1) as f32;
            }
        }
        // Input:
        // [[ 1,  2,  3,  4],
        //  [ 5,  6,  7,  8],
        //  [ 9, 10, 11, 12],
        //  [13, 14, 15, 16]]
        
        // Apply max pooling
        let (output, indices) = max_pool2d(&input.view(), (2, 2), (2, 2)).unwrap();
        
        // Output should be:
        // [[ 6, 8],
        //  [14, 16]]
        
        // Indices should mark positions of 6, 8, 14, 16 in each 2x2 block
        
        // Gradient from upstream
        let dout = array![[[[1.0, 2.0], [3.0, 4.0]]]];
        
        // Calculate gradients
        let dinput = max_pool2d_backward(
            &dout.view(), &input.view(), &indices.view(), (2, 2), (2, 2)
        ).unwrap();
        
        // Check shape
        assert_eq!(dinput.shape(), input.shape());
        
        // Verify that only the positions of max values received gradients
        let mut expected = Array4::<f32>::zeros((1, 1, 4, 4));
        expected[[0, 0, 1, 1]] = 1.0; // Position of 6
        expected[[0, 0, 1, 3]] = 2.0; // Position of 8
        expected[[0, 0, 3, 1]] = 3.0; // Position of 14
        expected[[0, 0, 3, 3]] = 4.0; // Position of 16
        
        for i in 0..4 {
            for j in 0..4 {
                assert_relative_eq!(
                    dinput[[0, 0, i, j]], 
                    expected[[0, 0, i, j]], 
                    epsilon = 1e-6
                );
            }
        }
    }
    
    #[test]
    fn test_layer_norm_backward() {
        // Input: batch_size=2, seq_len=1, features=3
        let x = array![[[0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6]]];
        let gamma = array![1.0, 0.5, 2.0];
        let beta = array![0.0, 0.1, -0.1];
        let eps = 1e-5;
        
        // Calculate means and variances for each (batch, sequence) position
        let mut mean = Array2::<f32>::zeros((2, 1));
        let mut var = Array2::<f32>::zeros((2, 1));
        
        for b in 0..2 {
            for s in 0..1 {
                // Compute mean
                let mut sum = 0.0;
                for f in 0..3 {
                    sum += x[[b, s, f]];
                }
                mean[[b, s]] = sum / 3.0;
                
                // Compute variance
                let mut sum_sq = 0.0;
                for f in 0..3 {
                    let diff = x[[b, s, f]] - mean[[b, s]];
                    sum_sq += diff * diff;
                }
                var[[b, s]] = sum_sq / 3.0;
            }
        }
        
        // Gradient from upstream
        let dout = Array3::<f32>::ones((2, 1, 3));
        
        // Calculate gradients
        let (dx, dgamma, dbeta) = layer_norm_backward(
            &dout.view(), &x.view(), &gamma.view(), &mean.view(), &var.view(), eps
        ).unwrap();
        
        // Check shapes
        assert_eq!(dx.shape(), x.shape());
        assert_eq!(dgamma.shape(), gamma.shape());
        assert_eq!(dbeta.shape(), beta.shape());
        
        // For each feature, dbeta should be the sum of dout across batch and sequence dimensions
        assert_relative_eq!(dbeta[0], 2.0, epsilon = 1e-6); // 2 batches * 1 sequence position
        assert_relative_eq!(dbeta[1], 2.0, epsilon = 1e-6);
        assert_relative_eq!(dbeta[2], 2.0, epsilon = 1e-6);
        
        // The sum of dx along the feature dimension should be approximately zero
        // This is a property of layer normalization's gradient
        for b in 0..2 {
            for s in 0..1 {
                let sum_dx = dx[[b, s, 0]] + dx[[b, s, 1]] + dx[[b, s, 2]];
                assert_relative_eq!(sum_dx, 0.0, epsilon = 1e-5);
            }
        }
    }
    
    #[test]
    fn test_transformer_ffn_backward() {
        // Setup parameters for a small feed-forward network
        let batch_size = 2;
        let seq_len = 1;
        let model_dim = 3;
        let ff_dim = 4;
        
        // Input tensor
        let x = array![[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]];
        
        // Weight matrices and biases
        let w1 = Array2::<f32>::ones((model_dim, ff_dim));
        let b1 = Array1::<f32>::zeros(ff_dim);
        let w2 = Array2::<f32>::ones((ff_dim, model_dim));
        let b2 = Array1::<f32>::zeros(model_dim);
        
        // Forward pass for hidden state
        let mut hidden = Array3::<f32>::zeros((batch_size, seq_len, ff_dim));
        for b in 0..batch_size {
            for s in 0..seq_len {
                for j in 0..ff_dim {
                    let mut sum = 0.0;
                    for i in 0..model_dim {
                        sum += x[[b, s, i]] * w1[[i, j]];
                    }
                    let val = sum + b1[j];
                    // ReLU activation
                    hidden[[b, s, j]] = if val > 0.0 { val } else { 0.0 };
                }
            }
        }
        
        // Gradient from upstream
        let dout = Array3::<f32>::ones((batch_size, seq_len, model_dim));
        
        // Calculate gradients
        let (dx, dw1, db1, dw2, db2) = transformer_ffn_backward(
            &dout.view(), &x.view(), &w1.view(), &b1.view(), &w2.view(), &b2.view(), &hidden.view()
        ).unwrap();
        
        // Check shapes
        assert_eq!(dx.shape(), x.shape());
        assert_eq!(dw1.shape(), w1.shape());
        assert_eq!(db1.shape(), b1.shape());
        assert_eq!(dw2.shape(), w2.shape());
        assert_eq!(db2.shape(), b2.shape());
        
        // For second layer weights (no activation):
        // dw2[i,j] = sum_{b,s} hidden[b,s,i] * dout[b,s,j]
        // For this test case with dout all ones:
        // dw2[i,j] = sum_{b,s} hidden[b,s,i]
        for i in 0..ff_dim {
            for j in 0..model_dim {
                let mut expected = 0.0;
                for b in 0..batch_size {
                    for s in 0..seq_len {
                        expected += hidden[[b, s, i]];
                    }
                }
                assert_relative_eq!(dw2[[i, j]], expected, epsilon = 1e-6);
            }
        }
        
        // For second layer bias:
        // db2[j] = sum_{b,s} dout[b,s,j]
        // For this test case with dout all ones:
        // db2[j] = batch_size * seq_len
        for j in 0..model_dim {
            assert_relative_eq!(db2[j], (batch_size * seq_len) as f32, epsilon = 1e-6);
        }
    }
    
    #[test]
    fn test_dropout_backward() {
        // Input tensor
        let x = Array2::<f32>::ones((2, 3));
        let dropout_prob = 0.5;
        
        // Create a fixed mask for testing
        let mut mask = Array2::<f32>::zeros((2, 3));
        mask[[0, 0]] = 2.0; // 1.0 / (1.0 - dropout_prob)
        mask[[0, 2]] = 2.0;
        mask[[1, 1]] = 2.0;
        
        // Gradient from upstream
        let dout = Array2::<f32>::ones((2, 3));
        
        // Calculate gradient
        let dx = dropout_backward(&dout.view(), &mask.view()).unwrap();
        
        // Check shape
        assert_eq!(dx.shape(), x.shape());
        
        // Verify that gradient is passed only through non-zero mask elements
        for i in 0..2 {
            for j in 0..3 {
                assert_relative_eq!(dx[[i, j]], mask[[i, j]], epsilon = 1e-6);
            }
        }
    }
    
    #[test]
    fn test_adaptive_avg_pool2d_backward() {
        // Create a batch of 1 feature map, 1 channel, 4x4 size
        let input = Array4::<f32>::ones((1, 1, 4, 4));
        
        // Apply adaptive average pooling to get 2x2 output
        let output = adaptive_avg_pool2d(&input.view(), (2, 2)).unwrap();
        
        // Gradient from upstream
        let mut dout = Array4::<f32>::zeros((1, 1, 2, 2));
        dout[[0, 0, 0, 0]] = 1.0;
        dout[[0, 0, 0, 1]] = 2.0;
        dout[[0, 0, 1, 0]] = 3.0;
        dout[[0, 0, 1, 1]] = 4.0;
        
        // Calculate gradient
        let dinput = adaptive_avg_pool2d_backward(
            &dout.view(), &input.view(), (2, 2)
        ).unwrap();
        
        // Check shape
        assert_eq!(dinput.shape(), input.shape());
        
        // For 4x4 to 2x2, each output pixel corresponds to a 2x2 block in the input
        // The gradient for each input pixel should be dout / block_size
        
        // Check for the top-left block (0,0)
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(dinput[[0, 0, i, j]], 1.0 / 4.0, epsilon = 1e-6);
            }
        }
        
        // Check for the top-right block (0,1)
        for i in 0..2 {
            for j in 2..4 {
                assert_relative_eq!(dinput[[0, 0, i, j]], 2.0 / 4.0, epsilon = 1e-6);
            }
        }
        
        // Check for the bottom-left block (1,0)
        for i in 2..4 {
            for j in 0..2 {
                assert_relative_eq!(dinput[[0, 0, i, j]], 3.0 / 4.0, epsilon = 1e-6);
            }
        }
        
        // Check for the bottom-right block (1,1)
        for i in 2..4 {
            for j in 2..4 {
                assert_relative_eq!(dinput[[0, 0, i, j]], 4.0 / 4.0, epsilon = 1e-6);
            }
        }
    }
}