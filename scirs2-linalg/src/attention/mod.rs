//! Optimized attention mechanisms for transformer models
//!
//! This module provides efficient implementations of various attention mechanisms
//! commonly used in transformer-based neural networks. It includes standard
//! scaled dot-product attention, multi-head attention, flash attention for memory
//! efficiency, sparse attention patterns, and various position-aware attention variants.
//!
//! ## Overview
//!
//! * Basic attention - scaled dot-product attention, masked attention
//! * Memory-efficient implementations - flash attention
//! * Sparsity-aware implementations - sparse attention patterns
//! * Position-aware variants - relative position encoding, rotary embeddings, ALiBi
//! * Advanced patterns - grouped query attention, linear attention
//!
//! ## Examples
//!
//! Basic scaled dot-product attention:
//!
//! ```
//! use ndarray::{Array2, Array3};
//! use scirs2_linalg::attention::{scaled_dot_product_attention, AttentionMask};
//!
//! // Create query, key, value matrices
//! let batch_size = 2;
//! let seq_len = 4;
//! let d_model = 8;
//!
//! // Random matrices for demonstration
//! let query = Array3::<f32>::ones((batch_size, seq_len, d_model));
//! let key = Array3::<f32>::ones((batch_size, seq_len, d_model));
//! let value = Array3::<f32>::ones((batch_size, seq_len, d_model));
//!
//! // Compute attention
//! let output = scaled_dot_product_attention(
//!     &query.view(),
//!     &key.view(),
//!     &value.view(),
//!     None,
//!     1.0 / (d_model as f32).sqrt()
//! ).unwrap();
//!
//! assert_eq!(output.shape(), &[batch_size, seq_len, d_model]);
//! ```
//!
//! Multi-head attention:
//!
//! ```
//! use ndarray::{Array2, Array3};
//! use scirs2_linalg::attention::{multi_head_attention, AttentionConfig};
//!
//! // Create query, key, value matrices
//! let batch_size = 2;
//! let seq_len = 4;
//! let d_model = 64;
//! let num_heads = 8;
//! let head_dim = d_model / num_heads;
//!
//! // Random matrices for demonstration
//! let query = Array3::<f32>::ones((batch_size, seq_len, d_model));
//! let key = Array3::<f32>::ones((batch_size, seq_len, d_model));
//! let value = Array3::<f32>::ones((batch_size, seq_len, d_model));
//!
//! // Linear projection weights
//! let wq = Array2::<f32>::ones((d_model, d_model));
//! let wk = Array2::<f32>::ones((d_model, d_model));
//! let wv = Array2::<f32>::ones((d_model, d_model));
//! let wo = Array2::<f32>::ones((d_model, d_model));
//!
//! // Configure attention
//! let config = AttentionConfig {
//!     num_heads,
//!     head_dim,
//!     dropout_prob: 0.0,
//!     causal: false,
//!     scale: Some(1.0 / (head_dim as f32).sqrt()),
//! };
//!
//! // Compute multi-head attention
//! let output = multi_head_attention(
//!     &query.view(),
//!     &key.view(),
//!     &value.view(),
//!     &wq.view(),
//!     &wk.view(),
//!     &wv.view(),
//!     &wo.view(),
//!     None,
//!     &config
//! ).unwrap();
//!
//! assert_eq!(output.shape(), &[batch_size, seq_len, d_model]);
//! ```

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3};
use num_traits::{Float, NumAssignOps, Zero};
use std::ops::{Add, Div, Mul, Sub};

use crate::blas_accelerated;
use crate::error::{check_dimensions, LinalgError, LinalgResult};

/// Mask types for attention mechanisms
#[derive(Debug, Clone)]
pub enum AttentionMask {
    /// Additive mask (added to attention scores before softmax)
    /// Shape: [batch_size, seq_len_q, seq_len_k] or [1, seq_len_q, seq_len_k]
    Additive(Array3<f32>),

    /// Multiplicative mask (multiplied with attention scores after softmax)
    /// Shape: [batch_size, seq_len_q, seq_len_k] or [1, seq_len_q, seq_len_k]
    Multiplicative(Array3<f32>),

    /// Boolean mask (True means attend, False means don't attend)
    /// Shape: [batch_size, seq_len_q, seq_len_k] or [1, seq_len_q, seq_len_k]
    Boolean(Array3<bool>),

    /// Causal mask (upper triangular with -inf)
    /// Automatically sized to match sequence lengths
    Causal,
}

/// Configuration for attention mechanisms
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Number of attention heads
    pub num_heads: usize,

    /// Dimension of each attention head
    pub head_dim: usize,

    /// Dropout probability (0.0 means no dropout)
    pub dropout_prob: f32,

    /// Whether to use causal masking (for autoregressive models)
    pub causal: bool,

    /// Custom scaling factor (default is 1/sqrt(d_k))
    pub scale: Option<f32>,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 8,
            head_dim: 64,
            dropout_prob: 0.0,
            causal: false,
            scale: None,
        }
    }
}

/// Basic attention function - the building block for all other attention variants
///
/// This implements the standard attention mechanism: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
///
/// # Arguments
///
/// * `query` - Query tensor of shape [batch_size, seq_len_q, d_model]
/// * `key` - Key tensor of shape [batch_size, seq_len_k, d_model]
/// * `value` - Value tensor of shape [batch_size, seq_len_k, d_model]
/// * `mask` - Optional mask to apply to attention weights
/// * `scale` - Scaling factor for dot product (default is 1/sqrt(d_k))
///
/// # Returns
///
/// * Output tensor of shape [batch_size, seq_len_q, d_model]
pub fn attention<F>(
    query: &ArrayView3<F>,
    key: &ArrayView3<F>,
    value: &ArrayView3<F>,
    mask: Option<&AttentionMask>,
    scale: F,
) -> LinalgResult<Array3<F>>
where
    F: Float + Add + Mul + Div + Sub + NumAssignOps + Zero + std::fmt::Debug,
{
    // Validate input dimensions
    let (batch_size, seq_len_q, d_model_q) = (query.shape()[0], query.shape()[1], query.shape()[2]);
    let (batch_size_k, seq_len_k, d_model_k) = (key.shape()[0], key.shape()[1], key.shape()[2]);
    let (batch_size_v, seq_len_v, d_model_v) =
        (value.shape()[0], value.shape()[1], value.shape()[2]);

    check_dimensions(
        batch_size == batch_size_k && batch_size == batch_size_v,
        format!(
            "Batch sizes must match: {}, {}, {}",
            batch_size, batch_size_k, batch_size_v
        ),
    )?;

    check_dimensions(
        seq_len_k == seq_len_v,
        format!(
            "Key and value sequence lengths must match: {}, {}",
            seq_len_k, seq_len_v
        ),
    )?;

    check_dimensions(
        d_model_q == d_model_k,
        format!(
            "Query and key dimensions must match: {}, {}",
            d_model_q, d_model_k
        ),
    )?;

    let mut result = Array3::<F>::zeros((batch_size, seq_len_q, d_model_v));

    for b in 0..batch_size {
        // Calculate attention scores: QK^T [seq_len_q, seq_len_k]
        let q_b = query.slice(ndarray::s![b, .., ..]);
        let k_b = key.slice(ndarray::s![b, .., ..]);
        let v_b = value.slice(ndarray::s![b, .., ..]);

        // Compute scores as matrix multiplication: query @ key.transpose()
        let mut scores = Array2::<F>::zeros((seq_len_q, seq_len_k));

        for i in 0..seq_len_q {
            for j in 0..seq_len_k {
                let mut dot_product = F::zero();
                for k in 0..d_model_q {
                    dot_product += q_b[[i, k]] * k_b[[j, k]];
                }
                scores[[i, j]] = dot_product * scale;
            }
        }

        // Apply mask if provided
        if let Some(mask_ref) = mask {
            apply_mask(&mut scores, mask_ref, b)?;
        }

        // Apply softmax along the last dimension
        for i in 0..seq_len_q {
            let mut row = scores.slice_mut(ndarray::s![i, ..]);

            // Compute softmax manually for numerical stability
            // First find the maximum value for numerical stability
            let max_val = row.fold(F::neg_infinity(), |max, &x| if x > max { x } else { max });

            // Subtract max value and exponentiate
            let mut sum = F::zero();
            for j in 0..seq_len_k {
                let exp_val = (row[j] - max_val).exp();
                row[j] = exp_val;
                sum += exp_val;
            }

            // Normalize
            if sum > F::zero() {
                for j in 0..seq_len_k {
                    row[j] /= sum;
                }
            }
        }

        // Calculate output: scores @ value
        let mut output = Array2::<F>::zeros((seq_len_q, d_model_v));

        for i in 0..seq_len_q {
            for j in 0..d_model_v {
                let mut sum = F::zero();
                for k in 0..seq_len_k {
                    sum += scores[[i, k]] * v_b[[k, j]];
                }
                output[[i, j]] = sum;
            }
        }

        // Store the result for this batch
        result.slice_mut(ndarray::s![b, .., ..]).assign(&output);
    }

    Ok(result)
}

/// Apply attention mask to scores
fn apply_mask<F>(scores: &mut Array2<F>, mask: &AttentionMask, batch_idx: usize) -> LinalgResult<()>
where
    F: Float + Add + Mul + Div + Sub + NumAssignOps + Zero + std::fmt::Debug,
{
    let (seq_len_q, seq_len_k) = (scores.shape()[0], scores.shape()[1]);

    match mask {
        AttentionMask::Additive(mask_tensor) => {
            let batch_dim = mask_tensor.shape()[0];
            let mask_idx = if batch_dim == 1 { 0 } else { batch_idx };

            if mask_tensor.shape()[1] != seq_len_q || mask_tensor.shape()[2] != seq_len_k {
                return Err(LinalgError::DimensionError(format!(
                    "Mask shape {:?} doesn't match scores shape [{}, {}]",
                    mask_tensor.shape(),
                    seq_len_q,
                    seq_len_k
                )));
            }

            let mask_slice = mask_tensor.slice(ndarray::s![mask_idx, .., ..]);

            for i in 0..seq_len_q {
                for j in 0..seq_len_k {
                    // Convert f32 to F (handling float type conversion)
                    let mask_val = F::from(mask_slice[[i, j]]).unwrap_or(F::zero());
                    scores[[i, j]] += mask_val;
                }
            }
        }

        AttentionMask::Multiplicative(mask_tensor) => {
            let batch_dim = mask_tensor.shape()[0];
            let mask_idx = if batch_dim == 1 { 0 } else { batch_idx };

            if mask_tensor.shape()[1] != seq_len_q || mask_tensor.shape()[2] != seq_len_k {
                return Err(LinalgError::DimensionError(format!(
                    "Mask shape {:?} doesn't match scores shape [{}, {}]",
                    mask_tensor.shape(),
                    seq_len_q,
                    seq_len_k
                )));
            }

            let mask_slice = mask_tensor.slice(ndarray::s![mask_idx, .., ..]);

            for i in 0..seq_len_q {
                for j in 0..seq_len_k {
                    // Convert f32 to F
                    let mask_val = F::from(mask_slice[[i, j]]).unwrap_or(F::zero());
                    scores[[i, j]] *= mask_val;
                }
            }
        }

        AttentionMask::Boolean(mask_tensor) => {
            let batch_dim = mask_tensor.shape()[0];
            let mask_idx = if batch_dim == 1 { 0 } else { batch_idx };

            if mask_tensor.shape()[1] != seq_len_q || mask_tensor.shape()[2] != seq_len_k {
                return Err(LinalgError::DimensionError(format!(
                    "Mask shape {:?} doesn't match scores shape [{}, {}]",
                    mask_tensor.shape(),
                    seq_len_q,
                    seq_len_k
                )));
            }

            let mask_slice = mask_tensor.slice(ndarray::s![mask_idx, .., ..]);

            for i in 0..seq_len_q {
                for j in 0..seq_len_k {
                    if !mask_slice[[i, j]] {
                        scores[[i, j]] = F::neg_infinity();
                    }
                }
            }
        }

        AttentionMask::Causal => {
            for i in 0..seq_len_q {
                for j in 0..seq_len_k {
                    if j > i {
                        scores[[i, j]] = F::neg_infinity();
                    }
                }
            }
        }
    }

    Ok(())
}

/// Scaled Dot-Product Attention
///
/// The standard attention mechanism used in Transformer models:
/// Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
///
/// # Arguments
///
/// * `query` - Query tensor of shape [batch_size, seq_len_q, d_model]
/// * `key` - Key tensor of shape [batch_size, seq_len_k, d_model]
/// * `value` - Value tensor of shape [batch_size, seq_len_k, d_model]
/// * `mask` - Optional mask to apply to attention weights
/// * `scale` - Scaling factor for dot product (default is 1/sqrt(d_k))
///
/// # Returns
///
/// * Output tensor of shape [batch_size, seq_len_q, d_model]
pub fn scaled_dot_product_attention<F>(
    query: &ArrayView3<F>,
    key: &ArrayView3<F>,
    value: &ArrayView3<F>,
    mask: Option<&AttentionMask>,
    scale: F,
) -> LinalgResult<Array3<F>>
where
    F: Float + Add + Mul + Div + Sub + NumAssignOps + Zero + std::fmt::Debug + 'static,
{
    // Special case for f32 - using runtime type checking
    if let Some(f32_result) = try_f32_attention(query, key, value, mask, scale) {
        return f32_result;
    }

    // Fall back to the generic implementation
    attention(query, key, value, mask, scale)
}

/// Try to use an optimized implementation for f32 type
fn try_f32_attention<F>(
    query: &ArrayView3<F>,
    key: &ArrayView3<F>,
    value: &ArrayView3<F>,
    mask: Option<&AttentionMask>,
    scale: F,
) -> Option<LinalgResult<Array3<F>>>
where
    F: Float + Add + Mul + Div + Sub + NumAssignOps + Zero + std::fmt::Debug + 'static,
{
    // Check if F is f32 and if we can use BLAS acceleration
    if std::any::TypeId::of::<F>() == std::any::TypeId::of::<f32>() && mask.is_none() {
        // SAFETY: We've already verified that F is f32, so this transmute is safe
        let query_f32: &ArrayView3<f32> = unsafe { std::mem::transmute(query) };
        let key_f32: &ArrayView3<f32> = unsafe { std::mem::transmute(key) };
        let value_f32: &ArrayView3<f32> = unsafe { std::mem::transmute(value) };
        let scale_f32: f32 = unsafe { *(&scale as *const F as *const f32) };

        // Use BLAS-accelerated attention for unmasked attention with f32
        let result = blas_attention_f32(query_f32, key_f32, value_f32, scale_f32);

        // Convert the result back to F type
        return Some(unsafe {
            std::mem::transmute::<Result<Array3<f32>, LinalgError>, Result<Array3<F>, LinalgError>>(
                result,
            )
        });
    }

    None
}

/// BLAS-accelerated attention implementation for f32
///
/// Uses BLAS for matrix multiplications to speed up the attention computation
fn blas_attention_f32(
    query: &ArrayView3<f32>,
    key: &ArrayView3<f32>,
    value: &ArrayView3<f32>,
    scale: f32,
) -> LinalgResult<Array3<f32>> {
    // Validate input dimensions
    let (batch_size, seq_len_q, d_model_q) = (query.shape()[0], query.shape()[1], query.shape()[2]);
    let (batch_size_k, seq_len_k, d_model_k) = (key.shape()[0], key.shape()[1], key.shape()[2]);
    let (batch_size_v, seq_len_v, d_model_v) =
        (value.shape()[0], value.shape()[1], value.shape()[2]);

    check_dimensions(
        batch_size == batch_size_k && batch_size == batch_size_v,
        format!(
            "Batch sizes must match: {}, {}, {}",
            batch_size, batch_size_k, batch_size_v
        ),
    )?;

    check_dimensions(
        seq_len_k == seq_len_v,
        format!(
            "Key and value sequence lengths must match: {}, {}",
            seq_len_k, seq_len_v
        ),
    )?;

    check_dimensions(
        d_model_q == d_model_k,
        format!(
            "Query and key dimensions must match: {}, {}",
            d_model_q, d_model_k
        ),
    )?;

    let mut result = Array3::<f32>::zeros((batch_size, seq_len_q, d_model_v));

    for b in 0..batch_size {
        // Extract batch slices
        let q_b = query.slice(ndarray::s![b, .., ..]);
        let k_b = key.slice(ndarray::s![b, .., ..]);
        let v_b = value.slice(ndarray::s![b, .., ..]);

        // Compute scores using BLAS matrix multiplication: QK^T
        // First transpose the key matrix
        let k_b_t = k_b.t();

        // Use BLAS to compute Q @ K^T
        // We need to convert our views to the correct type for BLAS
        let q_b_view = q_b.view();
        let k_b_t_view = k_b_t.view();
        let scores = blas_accelerated::matmul(&q_b_view, &k_b_t_view)?;

        // Scale the scores
        let mut scores_scaled = scores.mapv(|x| x * scale);

        // Apply softmax along the last dimension
        for i in 0..seq_len_q {
            let mut row = scores_scaled.slice_mut(ndarray::s![i, ..]);

            // Find the maximum value for numerical stability
            let max_val = row.fold(f32::NEG_INFINITY, |max, &x| max.max(x));

            // Subtract max value and exponentiate
            let mut sum = 0.0;
            for j in 0..seq_len_k {
                let exp_val = (row[j] - max_val).exp();
                row[j] = exp_val;
                sum += exp_val;
            }

            // Normalize
            if sum > 0.0 {
                for j in 0..seq_len_k {
                    row[j] /= sum;
                }
            }
        }

        // Use BLAS to compute attention_weights @ V
        let scores_view = scores_scaled.view();
        let v_b_view = v_b.view();
        let output = blas_accelerated::matmul(&scores_view, &v_b_view)?;

        // Store the result for this batch
        result.slice_mut(ndarray::s![b, .., ..]).assign(&output);
    }

    Ok(result)
}

// We're removing the SIMD implementation for now due to complexity
// It can be added back in a future PR with proper ndarray integration

/// Multi-Head Attention
///
/// Computes multiple attention heads in parallel and concatenates the results:
/// MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
/// where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
///
/// # Arguments
///
/// * `query` - Query tensor of shape [batch_size, seq_len_q, d_model]
/// * `key` - Key tensor of shape [batch_size, seq_len_k, d_model]
/// * `value` - Value tensor of shape [batch_size, seq_len_v, d_model]
/// * `wq` - Query projection weights [d_model, d_model]
/// * `wk` - Key projection weights [d_model, d_model]
/// * `wv` - Value projection weights [d_model, d_model]
/// * `wo` - Output projection weights [d_model, d_model]
/// * `mask` - Optional mask to apply to attention weights
/// * `config` - Configuration for the attention mechanism
///
/// # Returns
///
/// * Output tensor of shape [batch_size, seq_len_q, d_model]
#[allow(clippy::too_many_arguments)]
pub fn multi_head_attention<F>(
    query: &ArrayView3<F>,
    key: &ArrayView3<F>,
    value: &ArrayView3<F>,
    wq: &ArrayView2<F>,
    wk: &ArrayView2<F>,
    wv: &ArrayView2<F>,
    wo: &ArrayView2<F>,
    mask: Option<&AttentionMask>,
    config: &AttentionConfig,
) -> LinalgResult<Array3<F>>
where
    F: Float + Add + Mul + Div + Sub + NumAssignOps + Zero + std::fmt::Debug,
{
    // Extract dimensions
    let (batch_size, seq_len_q, d_model) = (query.shape()[0], query.shape()[1], query.shape()[2]);
    let seq_len_k = key.shape()[1];
    let seq_len_v = value.shape()[1];

    // Validate dimensions
    if key.shape()[2] != d_model || value.shape()[2] != d_model {
        return Err(LinalgError::DimensionError(format!(
            "Model dimensions must match: {}, {}, {}",
            d_model,
            key.shape()[2],
            value.shape()[2]
        )));
    }

    if wq.shape() != [d_model, d_model]
        || wk.shape() != [d_model, d_model]
        || wv.shape() != [d_model, d_model]
        || wo.shape() != [d_model, d_model]
    {
        return Err(LinalgError::DimensionError(
            "Weight matrices must have shape [d_model, d_model]".to_string(),
        ));
    }

    // Extract attention configuration
    let num_heads = config.num_heads;
    let head_dim = config.head_dim;
    let scale = match config.scale {
        Some(s) => F::from(s).unwrap_or_else(|| F::from(1.0 / (head_dim as f64).sqrt()).unwrap()),
        None => F::from(1.0 / (head_dim as f64).sqrt()).unwrap(),
    };

    // Verify that d_model is compatible with num_heads and head_dim
    if d_model != num_heads * head_dim {
        return Err(LinalgError::ValueError(format!(
            "Model dimension ({}) must equal num_heads ({}) * head_dim ({})",
            d_model, num_heads, head_dim
        )));
    }

    // Project query, key, and value
    let mut q_proj = Array3::<F>::zeros((batch_size, seq_len_q, d_model));
    let mut k_proj = Array3::<F>::zeros((batch_size, seq_len_k, d_model));
    let mut v_proj = Array3::<F>::zeros((batch_size, seq_len_v, d_model));

    // Perform projections batch by batch
    for b in 0..batch_size {
        // Project query
        for i in 0..seq_len_q {
            for j in 0..d_model {
                let mut sum = F::zero();
                for k in 0..d_model {
                    sum += query[[b, i, k]] * wq[[k, j]];
                }
                q_proj[[b, i, j]] = sum;
            }
        }

        // Project key
        for i in 0..seq_len_k {
            for j in 0..d_model {
                let mut sum = F::zero();
                for k in 0..d_model {
                    sum += key[[b, i, k]] * wk[[k, j]];
                }
                k_proj[[b, i, j]] = sum;
            }
        }

        // Project value
        for i in 0..seq_len_v {
            for j in 0..d_model {
                let mut sum = F::zero();
                for k in 0..d_model {
                    sum += value[[b, i, k]] * wv[[k, j]];
                }
                v_proj[[b, i, j]] = sum;
            }
        }
    }

    // Reshape for multi-head attention
    // We need to effectively reshape to [batch_size, num_heads, seq_len, head_dim]
    // but will use separate tensors for each head to avoid complex reshaping
    let mut head_outputs = Vec::with_capacity(num_heads);

    for h in 0..num_heads {
        // Extract head-specific portions of the projected tensors
        let start_idx = h * head_dim;
        let _end_idx = start_idx + head_dim; // Used for debugging/clarity

        let q_head = q_proj.slice(ndarray::s![.., .., start_idx..(start_idx + head_dim)]);
        let k_head = k_proj.slice(ndarray::s![.., .., start_idx..(start_idx + head_dim)]);
        let v_head = v_proj.slice(ndarray::s![.., .., start_idx..(start_idx + head_dim)]);

        // Compute attention for this head
        let head_output = attention(&q_head, &k_head, &v_head, mask, scale)?;
        head_outputs.push(head_output);
    }

    // Concatenate head outputs along the last dimension
    let mut concat_output = Array3::<F>::zeros((batch_size, seq_len_q, d_model));

    for (h, head_output) in head_outputs.iter().enumerate().take(num_heads) {
        let start_idx = h * head_dim;
        let _end_idx = start_idx + head_dim; // Used for clarity

        for b in 0..batch_size {
            for i in 0..seq_len_q {
                for j in 0..head_dim {
                    concat_output[[b, i, start_idx + j]] = head_output[[b, i, j]];
                }
            }
        }
    }

    // Apply output projection
    let mut output = Array3::<F>::zeros((batch_size, seq_len_q, d_model));

    for b in 0..batch_size {
        for i in 0..seq_len_q {
            for j in 0..d_model {
                let mut sum = F::zero();
                for k in 0..d_model {
                    sum += concat_output[[b, i, k]] * wo[[k, j]];
                }
                output[[b, i, j]] = sum;
            }
        }
    }

    Ok(output)
}

/// Flash Attention - Memory-efficient attention implementation
///
/// Implements the Flash Attention algorithm which reduces memory usage by computing
/// attention in blocks, avoiding the materialization of the full attention matrix.
///
/// # Arguments
///
/// * `query` - Query tensor of shape [batch_size, seq_len_q, d_model]
/// * `key` - Key tensor of shape [batch_size, seq_len_k, d_model]
/// * `value` - Value tensor of shape [batch_size, seq_len_k, d_model]
/// * `mask` - Optional mask to apply to attention weights
/// * `scale` - Scaling factor for dot product
/// * `block_size` - Block size for tiling (affects performance but not results)
///
/// # Returns
///
/// * Output tensor of shape [batch_size, seq_len_q, d_model]
pub fn flash_attention<F>(
    query: &ArrayView3<F>,
    key: &ArrayView3<F>,
    value: &ArrayView3<F>,
    mask: Option<&AttentionMask>,
    scale: F,
    block_size: usize,
) -> LinalgResult<Array3<F>>
where
    F: Float + Add + Mul + Div + Sub + NumAssignOps + Zero + std::fmt::Debug,
{
    // Validate dimensions
    let (batch_size, seq_len_q, d_model) = (query.shape()[0], query.shape()[1], query.shape()[2]);
    let (batch_size_k, seq_len_k, d_model_k) = (key.shape()[0], key.shape()[1], key.shape()[2]);
    let (batch_size_v, seq_len_v, d_model_v) =
        (value.shape()[0], value.shape()[1], value.shape()[2]);

    check_dimensions(
        batch_size == batch_size_k,
        format!("Batch sizes must match: {} != {}", batch_size, batch_size_k),
    )?;
    check_dimensions(
        batch_size == batch_size_v,
        format!("Batch sizes must match: {} != {}", batch_size, batch_size_v),
    )?;
    check_dimensions(
        seq_len_k == seq_len_v,
        format!(
            "Key and value sequence lengths must match: {} != {}",
            seq_len_k, seq_len_v
        ),
    )?;
    check_dimensions(
        d_model == d_model_k,
        format!(
            "Query and key dimensions must match: {} != {}",
            d_model, d_model_k
        ),
    )?;

    // Determine block sizes
    let block_size_q = block_size.min(seq_len_q);
    let block_size_k = block_size.min(seq_len_k);

    // Initialize output
    let mut output = Array3::<F>::zeros((batch_size, seq_len_q, d_model_v));

    // Process batch by batch
    for b in 0..batch_size {
        // Process query blocks
        for q_start in (0..seq_len_q).step_by(block_size_q) {
            let q_end = (q_start + block_size_q).min(seq_len_q);
            let q_block = query.slice(ndarray::s![b, q_start..q_end, ..]);

            // For each query block, process all key/value blocks
            let mut m_block = Array1::<F>::from_elem(q_end - q_start, F::neg_infinity());
            let mut l_block = Array1::<F>::zeros(q_end - q_start);

            for k_start in (0..seq_len_k).step_by(block_size_k) {
                let k_end = (k_start + block_size_k).min(seq_len_k);
                let k_block = key.slice(ndarray::s![b, k_start..k_end, ..]);
                let v_block = value.slice(ndarray::s![b, k_start..k_end, ..]);

                // Compute block of attention scores
                let mut scores_block = Array2::<F>::zeros((q_end - q_start, k_end - k_start));

                for i in 0..(q_end - q_start) {
                    for j in 0..(k_end - k_start) {
                        let mut dot_product = F::zero();
                        for k in 0..d_model {
                            dot_product += q_block[[i, k]] * k_block[[j, k]];
                        }
                        scores_block[[i, j]] = dot_product * scale;
                    }
                }

                // Apply mask if provided
                if let Some(mask_ref) = mask {
                    match mask_ref {
                        AttentionMask::Causal => {
                            for i in 0..(q_end - q_start) {
                                let q_idx = q_start + i;
                                for j in 0..(k_end - k_start) {
                                    let k_idx = k_start + j;
                                    if k_idx > q_idx {
                                        scores_block[[i, j]] = F::neg_infinity();
                                    }
                                }
                            }
                        }
                        // For other mask types, we would need to extract the relevant portion
                        // This is a simplified implementation for demonstration
                        _ => {
                            return Err(LinalgError::NotImplementedError(
                                "Flash attention currently only supports causal masks".to_string(),
                            ))
                        }
                    }
                }

                // Update max values and compute exponentials
                for i in 0..(q_end - q_start) {
                    let row = scores_block.slice(ndarray::s![i, ..]);
                    let max_val =
                        row.fold(F::neg_infinity(), |max, &x| if x > max { x } else { max });

                    if max_val > m_block[i] {
                        // Update scaling factors
                        let m_prev = m_block[i];
                        let m_new = max_val;

                        // Update l_i <- l_i * exp(m_i - m'_i)
                        if m_prev != F::neg_infinity() {
                            l_block[i] *= (m_prev - m_new).exp();
                        }

                        // Update output with scaling
                        if l_block[i] > F::zero() {
                            let scale_factor = (m_prev - m_new).exp() / l_block[i];
                            for j in 0..d_model_v {
                                output[[b, q_start + i, j]] *= scale_factor;
                            }
                        }

                        // Update m_i
                        m_block[i] = m_new;
                    }

                    // Compute contribution of current key-value block
                    let mut block_sum = F::zero();
                    let mut block_output = Array1::<F>::zeros(d_model_v);

                    for j in 0..(k_end - k_start) {
                        let exp_val = (scores_block[[i, j]] - m_block[i]).exp();
                        block_sum += exp_val;

                        // Update output
                        for k in 0..d_model_v {
                            block_output[k] += exp_val * v_block[[j, k]];
                        }
                    }

                    // Update l_i and output
                    l_block[i] += block_sum;
                    for j in 0..d_model_v {
                        output[[b, q_start + i, j]] += block_output[j];
                    }
                }
            }

            // Normalize output by l_block
            for i in 0..(q_end - q_start) {
                if l_block[i] > F::zero() {
                    for j in 0..d_model_v {
                        output[[b, q_start + i, j]] /= l_block[i];
                    }
                }
            }
        }
    }

    Ok(output)
}

/// Sparse Attention - Implements attention with sparse patterns
///
/// Computes attention using predetermined sparse attention patterns to reduce
/// computational complexity from O(n²) to O(n * log(n)) or even O(n).
///
/// # Arguments
///
/// * `query` - Query tensor of shape [batch_size, seq_len_q, d_model]
/// * `key` - Key tensor of shape [batch_size, seq_len_k, d_model]
/// * `value` - Value tensor of shape [batch_size, seq_len_k, d_model]
/// * `pattern_mask` - Boolean mask defining the sparse attention pattern [seq_len_q, seq_len_k]
/// * `scale` - Scaling factor for dot product
///
/// # Returns
///
/// * Output tensor of shape [batch_size, seq_len_q, d_model]
pub fn sparse_attention<F>(
    query: &ArrayView3<F>,
    key: &ArrayView3<F>,
    value: &ArrayView3<F>,
    pattern_mask: &ArrayView2<bool>,
    scale: F,
) -> LinalgResult<Array3<F>>
where
    F: Float + Add + Mul + Div + Sub + NumAssignOps + Zero + std::fmt::Debug,
{
    // Validate dimensions
    let (batch_size, seq_len_q, d_model) = (query.shape()[0], query.shape()[1], query.shape()[2]);
    let (_, seq_len_k, _) = (key.shape()[0], key.shape()[1], key.shape()[2]);
    let (_, _, d_model_v) = (value.shape()[0], value.shape()[1], value.shape()[2]);

    if pattern_mask.shape() != [seq_len_q, seq_len_k] {
        return Err(LinalgError::DimensionError(format!(
            "Pattern mask shape {:?} doesn't match query and key sequence lengths [{}, {}]",
            pattern_mask.shape(),
            seq_len_q,
            seq_len_k
        )));
    }

    // Initialize output
    let mut output = Array3::<F>::zeros((batch_size, seq_len_q, d_model_v));

    // Process batch by batch
    for b in 0..batch_size {
        let q_b = query.slice(ndarray::s![b, .., ..]);
        let k_b = key.slice(ndarray::s![b, .., ..]);
        let v_b = value.slice(ndarray::s![b, .., ..]);

        // For each query position
        for i in 0..seq_len_q {
            let q_i = q_b.slice(ndarray::s![i, ..]);

            // Calculate sparse attention scores
            let mut scores = Vec::new();
            let mut indices = Vec::new();

            // Only compute scores for positions allowed by the pattern mask
            for j in 0..seq_len_k {
                if pattern_mask[[i, j]] {
                    let k_j = k_b.slice(ndarray::s![j, ..]);

                    // Compute dot product
                    let mut dot_product = F::zero();
                    for k in 0..d_model {
                        dot_product += q_i[k] * k_j[k];
                    }

                    scores.push(dot_product * scale);
                    indices.push(j);
                }
            }

            // If no attention connections, continue to next query position
            if scores.is_empty() {
                continue;
            }

            // Apply softmax to the sparse scores
            let max_val = scores
                .iter()
                .fold(F::neg_infinity(), |max, &x| if x > max { x } else { max });

            let mut exp_scores = Vec::with_capacity(scores.len());
            let mut sum = F::zero();

            for &score in &scores {
                let exp_val = (score - max_val).exp();
                exp_scores.push(exp_val);
                sum += exp_val;
            }

            // Normalize scores
            if sum > F::zero() {
                for exp_score in &mut exp_scores {
                    *exp_score /= sum;
                }
            }

            // Compute weighted sum of values
            for j in 0..d_model_v {
                let mut weighted_sum = F::zero();

                for k in 0..indices.len() {
                    let v_idx = indices[k];
                    weighted_sum += exp_scores[k] * v_b[[v_idx, j]];
                }

                output[[b, i, j]] = weighted_sum;
            }
        }
    }

    Ok(output)
}

/// Masked Attention - Applies a custom mask to attention
///
/// # Arguments
///
/// * `query` - Query tensor of shape [batch_size, seq_len_q, d_model]
/// * `key` - Key tensor of shape [batch_size, seq_len_k, d_model]
/// * `value` - Value tensor of shape [batch_size, seq_len_k, d_model]
/// * `mask` - The mask to apply to attention weights
/// * `scale` - Scaling factor for dot product
///
/// # Returns
///
/// * Output tensor of shape [batch_size, seq_len_q, d_model]
pub fn masked_attention<F>(
    query: &ArrayView3<F>,
    key: &ArrayView3<F>,
    value: &ArrayView3<F>,
    mask: &AttentionMask,
    scale: F,
) -> LinalgResult<Array3<F>>
where
    F: Float + Add + Mul + Div + Sub + NumAssignOps + Zero + std::fmt::Debug,
{
    attention(query, key, value, Some(mask), scale)
}

/// Causal Attention - Implements attention with causal masking
///
/// Ensures each position can only attend to previous positions (and itself),
/// which is necessary for autoregressive models like GPT.
///
/// # Arguments
///
/// * `query` - Query tensor of shape [batch_size, seq_len_q, d_model]
/// * `key` - Key tensor of shape [batch_size, seq_len_k, d_model]
/// * `value` - Value tensor of shape [batch_size, seq_len_k, d_model]
/// * `scale` - Scaling factor for dot product
///
/// # Returns
///
/// * Output tensor of shape [batch_size, seq_len_q, d_model]
pub fn causal_attention<F>(
    query: &ArrayView3<F>,
    key: &ArrayView3<F>,
    value: &ArrayView3<F>,
    scale: F,
) -> LinalgResult<Array3<F>>
where
    F: Float + Add + Mul + Div + Sub + NumAssignOps + Zero + std::fmt::Debug,
{
    let mask = AttentionMask::Causal;
    attention(query, key, value, Some(&mask), scale)
}

/// Attention with ALiBi (Attention with Linear Biases)
///
/// Implements attention with linear biases as described in the paper
/// "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
///
/// # Arguments
///
/// * `query` - Query tensor of shape [batch_size, seq_len_q, d_model]
/// * `key` - Key tensor of shape [batch_size, seq_len_k, d_model]
/// * `value` - Value tensor of shape [batch_size, seq_len_k, d_model]
/// * `slopes` - Tensor of slope values for each attention head
/// * `scale` - Scaling factor for dot product
/// * `causal` - Whether to apply causal masking
///
/// # Returns
///
/// * Output tensor of shape [batch_size, seq_len_q, d_model]
pub fn attention_with_alibi<F>(
    query: &ArrayView3<F>,
    key: &ArrayView3<F>,
    value: &ArrayView3<F>,
    slopes: &ArrayView1<F>,
    scale: F,
    causal: bool,
) -> LinalgResult<Array3<F>>
where
    F: Float + Add + Mul + Div + Sub + NumAssignOps + Zero + std::fmt::Debug,
{
    // Validate dimensions
    let (batch_size, seq_len_q, d_model) = (query.shape()[0], query.shape()[1], query.shape()[2]);
    let (_, seq_len_k, _) = (key.shape()[0], key.shape()[1], key.shape()[2]);
    let (_, _, d_model_v) = (value.shape()[0], value.shape()[1], value.shape()[2]);

    // Calculate attention scores (QK^T)
    let mut result = Array3::<F>::zeros((batch_size, seq_len_q, d_model_v));

    for b in 0..batch_size {
        // Calculate QK^T
        let q_b = query.slice(ndarray::s![b, .., ..]);
        let k_b = key.slice(ndarray::s![b, .., ..]);
        let v_b = value.slice(ndarray::s![b, .., ..]);

        let mut scores = Array2::<F>::zeros((seq_len_q, seq_len_k));

        // Compute dot products
        for i in 0..seq_len_q {
            for j in 0..seq_len_k {
                let mut dot_product = F::zero();
                for k in 0..d_model {
                    dot_product += q_b[[i, k]] * k_b[[j, k]];
                }
                scores[[i, j]] = dot_product * scale;
            }
        }

        // Apply ALiBi bias (linear bias based on position difference)
        for i in 0..seq_len_q {
            for j in 0..seq_len_k {
                // In ALiBi, the bias is -slope * |i - j|
                // For simplicity, we'll use a single slope here
                let pos_diff = F::from((i as isize - j as isize).abs() as f64).unwrap();
                let slope = slopes[0]; // Using first slope for simplicity
                scores[[i, j]] -= slope * pos_diff;
            }
        }

        // Apply causal mask if requested
        if causal {
            for i in 0..seq_len_q {
                for j in 0..seq_len_k {
                    if j > i {
                        scores[[i, j]] = F::neg_infinity();
                    }
                }
            }
        }

        // Apply softmax to each row
        for i in 0..seq_len_q {
            let mut row = scores.slice_mut(ndarray::s![i, ..]);

            // Find max for numerical stability
            let max_val = row.fold(F::neg_infinity(), |max, &x| if x > max { x } else { max });

            // Compute exp and sum
            let mut sum = F::zero();
            for j in 0..seq_len_k {
                let exp_val = (row[j] - max_val).exp();
                row[j] = exp_val;
                sum += exp_val;
            }

            // Normalize
            if sum > F::zero() {
                for j in 0..seq_len_k {
                    row[j] /= sum;
                }
            }
        }

        // Compute output: scores @ value
        let mut output = Array2::<F>::zeros((seq_len_q, d_model_v));

        for i in 0..seq_len_q {
            for j in 0..d_model_v {
                let mut sum = F::zero();
                for k in 0..seq_len_k {
                    sum += scores[[i, k]] * v_b[[k, j]];
                }
                output[[i, j]] = sum;
            }
        }

        // Store result
        result.slice_mut(ndarray::s![b, .., ..]).assign(&output);
    }

    Ok(result)
}

// Additional attention implementations will be added below with similar patterns
// including relative position attention, rotary embeddings, etc.

/// Rotary Position Embeddings (RoPE)
///
/// Applies rotary position embeddings to query and key tensors as described in
/// "RoFormer: Enhanced Transformer with Rotary Position Embedding"
///
/// # Arguments
///
/// * `x` - Input tensor of shape [batch_size, seq_len, d_model]
/// * `freq_base` - Base frequency for the rotations (default: 10000.0)
///
/// # Returns
///
/// * Tensor with rotary position embeddings applied
pub fn rotary_embedding<F>(x: &ArrayView3<F>, freq_base: F) -> LinalgResult<Array3<F>>
where
    F: Float + Add + Mul + Div + Sub + NumAssignOps + Zero + std::fmt::Debug,
{
    let (batch_size, seq_len, d_model) = (x.shape()[0], x.shape()[1], x.shape()[2]);

    // Ensure dimension is even for proper pairing of dimensions
    if d_model % 2 != 0 {
        return Err(LinalgError::ValueError(
            "Dimension must be even for rotary embeddings".to_string(),
        ));
    }

    let mut result = Array3::<F>::zeros((batch_size, seq_len, d_model));

    // Create position frequencies
    let half_dim = d_model / 2;
    let mut freqs = Vec::with_capacity(half_dim);

    for i in 0..half_dim {
        let freq = F::one() / (freq_base.powf(F::from(2.0 * i as f64 / d_model as f64).unwrap()));
        freqs.push(freq);
    }

    // Apply rotary embeddings
    for b in 0..batch_size {
        for pos in 0..seq_len {
            for (i, _) in freqs.iter().enumerate().take(half_dim) {
                let i2 = 2 * i;

                // Get current values
                let x_i = x[[b, pos, i2]];
                let x_i_plus_1 = x[[b, pos, i2 + 1]];

                // Calculate rotation
                let theta = F::from(pos as f64).unwrap() * freqs[i];
                let cos_theta = theta.cos();
                let sin_theta = theta.sin();

                // Apply rotation
                result[[b, pos, i2]] = x_i * cos_theta - x_i_plus_1 * sin_theta;
                result[[b, pos, i2 + 1]] = x_i * sin_theta + x_i_plus_1 * cos_theta;
            }
        }
    }

    Ok(result)
}

/// Linear Attention
///
/// Implements linear attention which reduces computational complexity from O(n²) to O(n)
/// by using a linearized kernel function.
///
/// # Arguments
///
/// * `query` - Query tensor of shape [batch_size, seq_len_q, d_model]
/// * `key` - Key tensor of shape [batch_size, seq_len_k, d_model]
/// * `value` - Value tensor of shape [batch_size, seq_len_k, d_model]
/// * `scale` - Scaling factor for dot product
///
/// # Returns
///
/// * Output tensor of shape [batch_size, seq_len_q, d_model]
pub fn linear_attention<F>(
    query: &ArrayView3<F>,
    key: &ArrayView3<F>,
    value: &ArrayView3<F>,
    scale: F,
) -> LinalgResult<Array3<F>>
where
    F: Float + Add + Mul + Div + Sub + NumAssignOps + Zero + std::fmt::Debug,
{
    let (batch_size, seq_len_q, d_model) = (query.shape()[0], query.shape()[1], query.shape()[2]);
    let (_, seq_len_k, _) = (key.shape()[0], key.shape()[1], key.shape()[2]);
    let (_, _, d_model_v) = (value.shape()[0], value.shape()[1], value.shape()[2]);

    let mut result = Array3::<F>::zeros((batch_size, seq_len_q, d_model_v));

    // Apply elu + 1 feature mapping to query and key
    for b in 0..batch_size {
        // Feature map: elu(x) + 1
        let mut q_prime = Array2::<F>::zeros((seq_len_q, d_model));
        let mut k_prime = Array2::<F>::zeros((seq_len_k, d_model));

        // Apply feature map to query
        for i in 0..seq_len_q {
            for j in 0..d_model {
                let x = query[[b, i, j]];
                q_prime[[i, j]] = if x > F::zero() {
                    x
                } else {
                    (x.exp() - F::one()) + F::one()
                };
            }
        }

        // Apply feature map to key
        for i in 0..seq_len_k {
            for j in 0..d_model {
                let x = key[[b, i, j]] * scale;
                k_prime[[i, j]] = if x > F::zero() {
                    x
                } else {
                    (x.exp() - F::one()) + F::one()
                };
            }
        }

        // Compute KV first (linear complexity)
        let mut kv = Array2::<F>::zeros((d_model, d_model_v));

        for i in 0..d_model {
            for j in 0..d_model_v {
                let mut sum = F::zero();
                for k in 0..seq_len_k {
                    sum += k_prime[[k, i]] * value[[b, k, j]];
                }
                kv[[i, j]] = sum;
            }
        }

        // Compute normalization factor
        let mut z = Array1::<F>::zeros(seq_len_q);

        for i in 0..seq_len_q {
            let mut sum = F::zero();
            for j in 0..d_model {
                let mut k_sum = F::zero();
                for k in 0..seq_len_k {
                    k_sum += k_prime[[k, j]];
                }
                sum += q_prime[[i, j]] * k_sum;
            }
            z[i] = sum;
        }

        // Compute final output: (Q·(K^T·V)) / z
        for i in 0..seq_len_q {
            for j in 0..d_model_v {
                let mut sum = F::zero();
                for k in 0..d_model {
                    sum += q_prime[[i, k]] * kv[[k, j]];
                }

                if z[i] > F::zero() {
                    result[[b, i, j]] = sum / z[i];
                }
            }
        }
    }

    Ok(result)
}

/// Relative Position Attention
///
/// Implements attention with relative position encodings as described in
/// "Self-Attention with Relative Position Representations"
///
/// # Arguments
///
/// * `query` - Query tensor of shape [batch_size, seq_len_q, d_model]
/// * `key` - Key tensor of shape [batch_size, seq_len_k, d_model]
/// * `value` - Value tensor of shape [batch_size, seq_len_k, d_model]
/// * `rel_emb` - Relative position embeddings of shape [2*max_len-1, d_model]
/// * `scale` - Scaling factor for dot product
///
/// # Returns
///
/// * Output tensor of shape [batch_size, seq_len_q, d_model]
pub fn relative_position_attention<F>(
    query: &ArrayView3<F>,
    key: &ArrayView3<F>,
    value: &ArrayView3<F>,
    rel_emb: &ArrayView2<F>,
    scale: F,
) -> LinalgResult<Array3<F>>
where
    F: Float + Add + Mul + Div + Sub + NumAssignOps + Zero + std::fmt::Debug,
{
    let (batch_size, seq_len_q, d_model) = (query.shape()[0], query.shape()[1], query.shape()[2]);
    let (_, seq_len_k, _) = (key.shape()[0], key.shape()[1], key.shape()[2]);
    let (_, _, d_model_v) = (value.shape()[0], value.shape()[1], value.shape()[2]);

    // Validate relative embedding dimensions
    let expected_rel_emb_len = 2 * seq_len_k.max(seq_len_q) - 1;
    if rel_emb.shape()[0] != expected_rel_emb_len || rel_emb.shape()[1] != d_model {
        return Err(LinalgError::DimensionError(format!(
            "Relative embedding shape should be [{}, {}], got {:?}",
            expected_rel_emb_len,
            d_model,
            rel_emb.shape()
        )));
    }

    let mut result = Array3::<F>::zeros((batch_size, seq_len_q, d_model_v));

    // Process batch by batch
    for b in 0..batch_size {
        // Calculate content-content attention: QK^T
        let mut content_scores = Array2::<F>::zeros((seq_len_q, seq_len_k));

        for i in 0..seq_len_q {
            for j in 0..seq_len_k {
                let mut dot_product = F::zero();
                for k in 0..d_model {
                    dot_product += query[[b, i, k]] * key[[b, j, k]];
                }
                content_scores[[i, j]] = dot_product * scale;
            }
        }

        // Calculate content-position attention: QR^T
        let mut pos_scores = Array2::<F>::zeros((seq_len_q, seq_len_k));

        for i in 0..seq_len_q {
            for j in 0..seq_len_k {
                let rel_pos = (seq_len_k - 1) + i - j; // Offset for zero-indexing
                let mut dot_product = F::zero();
                for k in 0..d_model {
                    dot_product += query[[b, i, k]] * rel_emb[[rel_pos, k]];
                }
                pos_scores[[i, j]] = dot_product * scale;
            }
        }

        // Combine scores and apply softmax
        let mut combined_scores = Array2::<F>::zeros((seq_len_q, seq_len_k));

        for i in 0..seq_len_q {
            // Combine content and position scores
            for j in 0..seq_len_k {
                combined_scores[[i, j]] = content_scores[[i, j]] + pos_scores[[i, j]];
            }

            // Apply softmax
            let mut row = combined_scores.slice_mut(ndarray::s![i, ..]);
            let max_val = row.fold(F::neg_infinity(), |max, &x| if x > max { x } else { max });

            let mut sum = F::zero();
            for j in 0..seq_len_k {
                let exp_val = (row[j] - max_val).exp();
                row[j] = exp_val;
                sum += exp_val;
            }

            // Normalize
            if sum > F::zero() {
                for j in 0..seq_len_k {
                    row[j] /= sum;
                }
            }
        }

        // Compute output: combined_scores @ value
        let mut output = Array2::<F>::zeros((seq_len_q, d_model_v));

        for i in 0..seq_len_q {
            for j in 0..d_model_v {
                let mut sum = F::zero();
                for k in 0..seq_len_k {
                    sum += combined_scores[[i, k]] * value[[b, k, j]];
                }
                output[[i, j]] = sum;
            }
        }

        // Store the result for this batch
        result.slice_mut(ndarray::s![b, .., ..]).assign(&output);
    }

    Ok(result)
}

/// Grouped Query Attention (GQA)
///
/// Implements grouped query attention as described in papers like
/// "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
///
/// # Arguments
///
/// * `query` - Query tensor of shape [batch_size, seq_len_q, d_model]
/// * `key` - Key tensor of shape [batch_size, seq_len_k, d_model]
/// * `value` - Value tensor of shape [batch_size, seq_len_k, d_model]
/// * `wq` - Query projection weights [d_model, d_model]
/// * `wk` - Key projection weights [d_model, kv_dim]
/// * `wv` - Value projection weights [d_model, kv_dim]
/// * `wo` - Output projection weights [d_model, d_model]
/// * `mask` - Optional mask to apply to attention weights
/// * `num_heads` - Number of query heads
/// * `num_kv_heads` - Number of key/value heads
/// * `scale` - Scaling factor for dot product
///
/// # Returns
///
/// * Output tensor of shape [batch_size, seq_len_q, d_model]
#[allow(clippy::too_many_arguments)]
pub fn grouped_query_attention<F>(
    query: &ArrayView3<F>,
    key: &ArrayView3<F>,
    value: &ArrayView3<F>,
    wq: &ArrayView2<F>,
    wk: &ArrayView2<F>,
    wv: &ArrayView2<F>,
    wo: &ArrayView2<F>,
    mask: Option<&AttentionMask>,
    num_heads: usize,
    num_kv_heads: usize,
    scale: F,
) -> LinalgResult<Array3<F>>
where
    F: Float + Add + Mul + Div + Sub + NumAssignOps + Zero + std::fmt::Debug,
{
    // Validate dimensions
    let (batch_size, seq_len_q, d_model) = (query.shape()[0], query.shape()[1], query.shape()[2]);
    let seq_len_k = key.shape()[1];

    // Validate kv head configuration
    if num_heads % num_kv_heads != 0 {
        return Err(LinalgError::ValueError(format!(
            "Number of query heads ({}) must be divisible by number of KV heads ({})",
            num_heads, num_kv_heads
        )));
    }

    let heads_per_kv = num_heads / num_kv_heads;
    let head_dim = d_model / num_heads;
    let kv_dim = num_kv_heads * head_dim;

    // Validate weight dimensions
    if wq.shape() != [d_model, d_model]
        || wk.shape() != [d_model, kv_dim]
        || wv.shape() != [d_model, kv_dim]
        || wo.shape() != [d_model, d_model]
    {
        return Err(LinalgError::DimensionError(
            "Weight matrices have incorrect dimensions".to_string(),
        ));
    }

    // Project query, key, and value
    let mut q_proj = Array3::<F>::zeros((batch_size, seq_len_q, d_model));
    let mut k_proj = Array3::<F>::zeros((batch_size, seq_len_k, kv_dim));
    let mut v_proj = Array3::<F>::zeros((batch_size, seq_len_k, kv_dim));

    // Perform projections batch by batch
    for b in 0..batch_size {
        // Project query
        for i in 0..seq_len_q {
            for j in 0..d_model {
                let mut sum = F::zero();
                for k in 0..d_model {
                    sum += query[[b, i, k]] * wq[[k, j]];
                }
                q_proj[[b, i, j]] = sum;
            }
        }

        // Project key
        for i in 0..seq_len_k {
            for j in 0..kv_dim {
                let mut sum = F::zero();
                for k in 0..d_model {
                    sum += key[[b, i, k]] * wk[[k, j]];
                }
                k_proj[[b, i, j]] = sum;
            }
        }

        // Project value
        for i in 0..seq_len_k {
            for j in 0..kv_dim {
                let mut sum = F::zero();
                for k in 0..d_model {
                    sum += value[[b, i, k]] * wv[[k, j]];
                }
                v_proj[[b, i, j]] = sum;
            }
        }
    }

    // Initialize output
    let mut concat_output = Array3::<F>::zeros((batch_size, seq_len_q, d_model));

    // Process each head
    for h in 0..num_heads {
        let kv_head_idx = h / heads_per_kv;

        // Extract head-specific portions of the projected tensors
        let q_start = h * head_dim;
        let q_end = q_start + head_dim;

        let kv_start = kv_head_idx * head_dim;
        let kv_end = kv_start + head_dim;

        // Extract query head
        let q_head = q_proj.slice(ndarray::s![.., .., q_start..q_end]);

        // Extract key and value heads (shared across multiple query heads)
        let k_head = k_proj.slice(ndarray::s![.., .., kv_start..kv_end]);
        let v_head = v_proj.slice(ndarray::s![.., .., kv_start..kv_end]);

        // Compute attention for this head
        let head_output = attention(&q_head, &k_head, &v_head, mask, scale)?;

        // Add to concatenated output
        for b in 0..batch_size {
            for i in 0..seq_len_q {
                for j in 0..head_dim {
                    concat_output[[b, i, q_start + j]] = head_output[[b, i, j]];
                }
            }
        }
    }

    // Apply output projection
    let mut output = Array3::<F>::zeros((batch_size, seq_len_q, d_model));

    for b in 0..batch_size {
        for i in 0..seq_len_q {
            for j in 0..d_model {
                let mut sum = F::zero();
                for k in 0..d_model {
                    sum += concat_output[[b, i, k]] * wo[[k, j]];
                }
                output[[b, i, j]] = sum;
            }
        }
    }

    Ok(output)
}

/// Attention with Relative Position Encodings (RPE)
///
/// Flexible interface for relative position encoding that supports
/// multiple implementation variants.
///
/// # Arguments
///
/// * `query` - Query tensor [batch_size, seq_len_q, d_model]
/// * `key` - Key tensor [batch_size, seq_len_k, d_model]
/// * `value` - Value tensor [batch_size, seq_len_k, d_model]
/// * `rel_emb` - Relative embeddings tensor
/// * `scale` - Scaling factor for attention
/// * `use_xpos` - Whether to use the xPos style of relative positioning
///
/// # Returns
///
/// * Output tensor of shape [batch_size, seq_len_q, d_model]
pub fn attention_with_rpe<F>(
    query: &ArrayView3<F>,
    key: &ArrayView3<F>,
    value: &ArrayView3<F>,
    rel_emb: &ArrayView2<F>,
    scale: F,
    use_xpos: bool,
) -> LinalgResult<Array3<F>>
where
    F: Float + Add + Mul + Div + Sub + NumAssignOps + Zero + std::fmt::Debug,
{
    // If using xPos, apply the specialized implementation
    if use_xpos {
        // This is a simplified implementation of xPos relative position encoding
        // In a full implementation, we would implement the complete xPos algorithm

        let (batch_size, seq_len_q, d_model) =
            (query.shape()[0], query.shape()[1], query.shape()[2]);
        let (_, seq_len_k, _) = (key.shape()[0], key.shape()[1], key.shape()[2]);
        let (_, _, _d_model_v) = (value.shape()[0], value.shape()[1], value.shape()[2]); // For future use

        // Create scaled arrays for computation

        // Apply xPos scaling to query and key based on position
        let mut q_scaled = Array3::<F>::zeros((batch_size, seq_len_q, d_model));
        let mut k_scaled = Array3::<F>::zeros((batch_size, seq_len_k, d_model));

        // Apply rotary-style position encoding with xPos modifications
        for b in 0..batch_size {
            for i in 0..seq_len_q {
                let pos_i = F::from(i as f64 + 1.0).unwrap(); // 1-indexed position
                for j in 0..d_model {
                    // Apply position-dependent scaling
                    let dim_factor = F::from(j as f64 / d_model as f64).unwrap();
                    let scale_factor = F::one() / pos_i.powf(dim_factor);
                    q_scaled[[b, i, j]] = query[[b, i, j]] * scale_factor;
                }
            }

            for i in 0..seq_len_k {
                let pos_i = F::from(i as f64 + 1.0).unwrap(); // 1-indexed position
                for j in 0..d_model {
                    // Apply position-dependent scaling
                    let dim_factor = F::from(j as f64 / d_model as f64).unwrap();
                    let scale_factor = F::one() / pos_i.powf(dim_factor);
                    k_scaled[[b, i, j]] = key[[b, i, j]] * scale_factor;
                }
            }
        }

        // Now compute attention with the scaled tensors
        return attention(&q_scaled.view(), &k_scaled.view(), value, None, scale);
    }

    // Otherwise use standard relative position embeddings
    relative_position_attention(query, key, value, rel_emb, scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_basic_attention() {
        // Simple 2x2x2 test case - consistent inputs for reproducible tests
        let query = array![[[1.0, 1.0], [1.0, 1.0]]]
            .into_shape_with_order((1, 2, 2))
            .unwrap();
        let key = array![[[1.0, 1.0], [1.0, 1.0]]]
            .into_shape_with_order((1, 2, 2))
            .unwrap();
        let value = array![[[5.0, 6.0], [7.0, 8.0]]]
            .into_shape_with_order((1, 2, 2))
            .unwrap();

        // Scale factor (1/sqrt(d_k))
        let scale = 1.0 / (2.0_f64).sqrt() as f64;

        let result = attention(&query.view(), &key.view(), &value.view(), None, scale).unwrap();

        // The shape should be the same as the query
        assert_eq!(result.shape(), &[1, 2, 2]);

        // With identical query and key vectors, we expect the attention weights to be equal
        // This means each position should get the average of the value vectors
        let expected_first_pos = [(5.0 + 7.0) / 2.0, (6.0 + 8.0) / 2.0];
        let expected_second_pos = [(5.0 + 7.0) / 2.0, (6.0 + 8.0) / 2.0];

        // Check approximate equality with a more generous tolerance
        assert!((result[[0, 0, 0]] - expected_first_pos[0]).abs() < 1e-5);
        assert!((result[[0, 0, 1]] - expected_first_pos[1]).abs() < 1e-5);
        assert!((result[[0, 1, 0]] - expected_second_pos[0]).abs() < 1e-5);
        assert!((result[[0, 1, 1]] - expected_second_pos[1]).abs() < 1e-5);
    }

    #[test]
    fn test_causal_attention() {
        // Create a simple test case
        let query = array![[[1.0, 1.0], [1.0, 1.0]]]
            .into_shape_with_order((1, 2, 2))
            .unwrap();
        let key = array![[[1.0, 1.0], [1.0, 1.0]]]
            .into_shape_with_order((1, 2, 2))
            .unwrap();
        let value = array![[[1.0, 2.0], [3.0, 4.0]]]
            .into_shape_with_order((1, 2, 2))
            .unwrap();

        let scale = 1.0 / (2.0_f64).sqrt() as f64;

        let result = causal_attention(&query.view(), &key.view(), &value.view(), scale).unwrap();

        // First position can only attend to itself
        assert!((result[[0, 0, 0]] - 1.0).abs() < 1e-6);
        assert!((result[[0, 0, 1]] - 2.0).abs() < 1e-6);

        // Second position can attend to both positions
        // Since the attention weights are equal due to identical query and key,
        // the result should be the average of the two value vectors
        assert!((result[[0, 1, 0]] - 2.0).abs() < 1e-6);
        assert!((result[[0, 1, 1]] - 3.0).abs() < 1e-6);
    }
}
