// Parallel convolution and correlation functions
//
// This module provides parallel implementations of convolution and correlation
// operations for improved performance on multi-core systems.

use crate::error::{SignalError, SignalResult};
use ndarray::{Array2, ArrayView2};
use num_traits::{Float, NumCast};
use scirs2_core::parallel_ops::*;
use std::fmt::Debug;

#[allow(unused_imports)]
// Temporary replacement for par_iter_with_setup
fn par_iter_with_setup<I, IT, S, F, R, RF, E>(
    items: I,
    _setup: S,
    map_fn: F,
    reduce_fn: RF,
) -> Result<Vec<R>, E>
where
    I: IntoIterator<Item = IT>,
    IT: Copy,
    S: Fn() -> (),
    F: Fn((), IT) -> Result<R, E>,
    RF: Fn(&mut Vec<R>, Result<R, E>) -> Result<(), E>,
    E: std::fmt::Debug,
{
    let mut results = Vec::new();
    for item in items {
        let result = map_fn((), item);
        reduce_fn(&mut results, result)?;
    }
    Ok(results)
}

/// Parallel 1D convolution with automatic chunking
///
/// # Arguments
///
/// * `a` - First input array
/// * `v` - Second input array (kernel)
/// * `mode` - Convolution mode ("full", "same", "valid")
///
/// # Returns
///
/// * Convolution result
#[allow(dead_code)]
pub fn parallel_convolve1d<T, U>(a: &[T], v: &[U], mode: &str) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug + Send + Sync,
    U: Float + NumCast + Debug + Send + Sync,
{
    // Convert inputs to f64
    let a_vec: Vec<f64> = a
        .iter()
        .map(|&val| NumCast::from(val).unwrap_or(0.0))
        .collect();

    let v_vec: Vec<f64> = v
        .iter()
        .map(|&val| NumCast::from(val).unwrap_or(0.0))
        .collect();

    // Determine if parallel processing is beneficial
    let use_parallel = a_vec.len() > 1000 || (a_vec.len() > 100 && v_vec.len() > 10);

    if use_parallel {
        parallel_convolve_impl(&a_vec, &v_vec, mode)
    } else {
        // Fall back to sequential for small inputs
        crate::convolve::convolve(a, v, mode)
    }
}

/// Core parallel convolution implementation
#[allow(dead_code)]
fn parallel_convolve_impl(a: &[f64], v: &[f64], mode: &str) -> SignalResult<Vec<f64>> {
    let na = a.len();
    let nv = v.len();

    if na == 0 || nv == 0 {
        return Ok(vec![]);
    }

    // Full convolution length
    let n_full = na + nv - 1;

    // Use different strategies based on kernel size
    let result = if nv <= 32 {
        // Small kernel: parallelize over output elements
        parallel_direct_conv(a, v, n_full)
    } else {
        // Large kernel: use overlap-save method
        parallel_overlap_save_conv(a, v, n_full)
    };

    // Apply mode
    apply_conv_mode(result, na, nv, mode)
}

/// Direct parallel convolution for small kernels
#[allow(dead_code)]
fn parallel_direct_conv(a: &[f64], v: &[f64], nfull: usize) -> Vec<f64> {
    let na = a.len();
    let nv = v.len();

    // Parallel computation of output elements
    let result: Vec<f64> = par_iter_with_setup(
        0..n_full,
        || {},
        |_, i| {
            let mut sum = 0.0;

            // Compute valid range for convolution at position i
            let j_start = i.saturating_sub(na - 1);
            let j_end = (i + 1).min(nv);

            for j in j_start..j_end {
                let a_idx = i - j;
                if a_idx < na {
                    sum += a[a_idx] * v[j];
                }
            }

            Ok(sum)
        },
        |results: &mut Vec<f64>, val: Result<f64, SignalError>| {
            results.push(val?);
            Ok(())
        },
    )
    .unwrap_or_else(|_| vec![0.0; n_full]);

    result
}

/// Overlap-save parallel convolution for large kernels
#[allow(dead_code)]
fn parallel_overlap_save_conv(a: &[f64], v: &[f64], nfull: usize) -> Vec<f64> {
    let na = a.len();
    let nv = v.len();

    // Choose chunk size (power of 2 for potential FFT optimization)
    let chunk_size = 4096.max(nv * 4);
    let overlap = nv - 1;
    let step = chunk_size - overlap;

    // Number of chunks
    let n_chunks = (na + step - 1) / step;

    // Process chunks in parallel
    let chunk_results: Vec<Vec<f64>> = par_iter_with_setup(
        0..n_chunks,
        || {},
        |_, chunk_idx| {
            let start = chunk_idx * step;
            let end = (start + chunk_size).min(na + overlap);

            // Create padded chunk
            let mut chunk = vec![0.0; chunk_size];
            for i in start..end.min(na) {
                chunk[i - start] = a[i];
            }

            // Convolve chunk with kernel
            let mut chunk_result = vec![0.0; chunk_size + nv - 1];
            for i in 0..chunk_size {
                for j in 0..nv {
                    chunk_result[i + j] += chunk[i] * v[j];
                }
            }

            Ok(chunk_result)
        },
        |results, res| {
            results.push(res?);
            Ok(())
        },
    )
    .unwrap_or_else(|_: SignalError| vec![]);

    // Combine chunk results
    let mut result = vec![0.0; n_full];
    for (chunk_idx, chunk_result) in chunk_results.iter().enumerate() {
        let start = chunk_idx * step;

        // Copy non-overlapping portion
        let copy_start = if chunk_idx == 0 { 0 } else { overlap };
        let copy_end = if chunk_idx == n_chunks - 1 {
            chunk_result.len()
        } else {
            step + overlap
        };

        for i in copy_start..copy_end.min(chunk_result.len()) {
            if start + i < n_full {
                result[start + i] = chunk_result[i];
            }
        }
    }

    result
}

/// Apply convolution mode (full, same, valid)
#[allow(dead_code)]
fn apply_conv_mode(result: Vec<f64>, na: usize, nv: usize, mode: &str) -> SignalResult<Vec<f64>> {
    match mode {
        "full" => Ok(_result),
        "same" => {
            let start = (nv - 1) / 2;
            let end = start + na;
            if end <= result.len() {
                Ok(_result[start..end].to_vec())
            } else {
                Ok(_result)
            }
        }
        "valid" => {
            if nv > na {
                return Err(SignalError::ValueError(
                    "In 'valid' mode, kernel size must not exceed signal size".to_string(),
                ));
            }
            let start = nv - 1;
            let end = result.len() - (nv - 1);
            if start < end && end <= result.len() {
                Ok(_result[start..end].to_vec())
            } else {
                Ok(vec![])
            }
        }
        _ => Err(SignalError::ValueError(format!("Unknown mode: {}", mode))),
    }
}

/// Parallel cross-correlation of two 1D arrays
///
/// # Arguments
///
/// * `a` - First input array
/// * `v` - Second input array
/// * `mode` - Correlation mode ("full", "same", "valid")
///
/// # Returns
///
/// * Cross-correlation result
#[allow(dead_code)]
pub fn parallel_correlate<T, U>(a: &[T], v: &[U], mode: &str) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug + Send + Sync,
    U: Float + NumCast + Debug + Send + Sync,
{
    // Convert inputs to f64
    let a_vec: Vec<f64> = a
        .iter()
        .map(|&val| NumCast::from(val).unwrap_or(0.0))
        .collect();

    let mut v_vec: Vec<f64> = v
        .iter()
        .map(|&val| NumCast::from(val).unwrap_or(0.0))
        .collect();

    // Correlation is convolution with reversed kernel
    v_vec.reverse();

    parallel_convolve_impl(&a_vec, &v_vec, mode)
}

/// Parallel 2D convolution
///
/// # Arguments
///
/// * `image` - 2D input array
/// * `kernel` - 2D convolution kernel
/// * `mode` - Convolution mode
///
/// # Returns
///
/// * 2D convolution result
#[allow(dead_code)]
pub fn parallel_convolve2d_ndarray(
    image: ArrayView2<f64>,
    kernel: ArrayView2<f64>,
    mode: &str,
) -> SignalResult<Array2<f64>> {
    let (img_rows, img_cols) = image.dim();
    let (ker_rows, ker_cols) = kernel.dim();

    if ker_rows > img_rows || ker_cols > img_cols {
        return Err(SignalError::ValueError(
            "Kernel dimensions must not exceed image dimensions".to_string(),
        ));
    }

    // Determine output dimensions
    let (out_rows, out_cols) = match mode {
        "full" => (img_rows + ker_rows - 1, img_cols + ker_cols - 1),
        "same" => (img_rows, img_cols),
        "valid" => (img_rows - ker_rows + 1, img_cols - ker_cols + 1),
        _ => return Err(SignalError::ValueError(format!("Unknown mode: {}", mode))),
    };

    // Parallel processing over output rows
    let row_results: Vec<Vec<f64>> = par_iter_with_setup(
        0..out_rows,
        || {},
        |_, out_i| {
            let mut row = vec![0.0; out_cols];

            // Determine input row range based on mode
            let (i_start, i_offset) = match mode {
                "full" => (0, out_i as isize),
                "same" => (0, out_i as isize - (ker_rows / 2) as isize),
                "valid" => (out_i, 0),
                _ => (0, 0),
            };

            for out_j in 0..out_cols {
                // Determine input column range
                let (j_start, j_offset) = match mode {
                    "full" => (0, out_j as isize),
                    "same" => (0, out_j as isize - (ker_cols / 2) as isize),
                    "valid" => (out_j, 0),
                    _ => (0, 0),
                };

                let mut sum = 0.0;

                // Perform 2D convolution at this output position
                for ki in 0..ker_rows {
                    let img_i = (i_offset + ki as isize) as usize;
                    if img_i >= img_rows {
                        continue;
                    }

                    for kj in 0..ker_cols {
                        let img_j = (j_offset + kj as isize) as usize;
                        if img_j >= img_cols {
                            continue;
                        }

                        // Flip kernel for convolution
                        sum +=
                            image[[img_i, img_j]] * kernel[[ker_rows - 1 - ki, ker_cols - 1 - kj]];
                    }
                }

                row[out_j] = sum;
            }

            Ok(row)
        },
        |results, row: Result<Vec<f64>, SignalError>| {
            results.push(row?);
            Ok(())
        },
    )?;

    // Convert to Array2
    let mut output = Array2::zeros((out_rows, out_cols));
    for (i, row) in row_results.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            output[[i, j]] = val;
        }
    }

    Ok(output)
}

/// Parallel separable 2D convolution (for separable kernels)
///
/// For kernels that can be separated into row and column vectors,
/// this is much more efficient than general 2D convolution.
///
/// # Arguments
///
/// * `image` - 2D input array
/// * `row_kernel` - 1D row kernel
/// * `col_kernel` - 1D column kernel
/// * `mode` - Convolution mode
///
/// # Returns
///
/// * 2D convolution result
#[allow(dead_code)]
pub fn parallel_separable_convolve2d(
    image: ArrayView2<f64>,
    row_kernel: &[f64],
    col_kernel: &[f64],
    mode: &str,
) -> SignalResult<Array2<f64>> {
    let (img_rows, img_cols) = image.dim();

    // First, convolve each row with row_kernel
    let row_convolved: Vec<Vec<f64>> = par_iter_with_setup(
        0..img_rows,
        || {},
        |_, i| {
            let row = image.row(i);
            let row_vec: Vec<f64> = row.to_vec();
            parallel_convolve_impl(&row_vec, row_kernel, mode)
        },
        |results, row| {
            results.push(row?);
            Ok(())
        },
    )?;

    // Determine intermediate dimensions
    let inter_cols = row_convolved[0].len();

    // Convert to Array2 for column processing
    let mut intermediate = Array2::zeros((img_rows, inter_cols));
    for (i, row) in row_convolved.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            intermediate[[i, j]] = val;
        }
    }

    // Then, convolve each column with col_kernel
    let col_convolved: Vec<Vec<f64>> = par_iter_with_setup(
        0..inter_cols,
        || {},
        |_, j| {
            let col = intermediate.column(j);
            let col_vec: Vec<f64> = col.to_vec();
            parallel_convolve_impl(&col_vec, col_kernel, mode)
        },
        |results, col| {
            results.push(col?);
            Ok(())
        },
    )?;

    // Determine final dimensions
    let final_rows = col_convolved[0].len();
    let final_cols = inter_cols;

    // Transpose to get final result
    let mut output = Array2::zeros((final_rows, final_cols));
    for (j, col) in col_convolved.iter().enumerate() {
        for (i, &val) in col.iter().enumerate() {
            output[[i, j]] = val;
        }
    }

    Ok(output)
}
