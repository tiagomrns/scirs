//! SIMD accelerated operations for numerical computations
//!
//! This module provides SIMD-accelerated versions of common numerical operations
//! for improved performance on modern CPUs. These implementations use the `wide`
//! crate to leverage SIMD instructions available on the target architecture.

use ndarray::{Array, Array1, ArrayView1, Dimension, Zip};
use num_traits::Float;
use std::ops::{Add, Div, Mul, Sub};
use wide::{f32x4, f32x8, f64x2, f64x4, CmpGt, CmpLt};

/// Trait for types that can be processed with SIMD operations
pub trait SimdOps:
    Copy
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
}

impl SimdOps for f32 {}
impl SimdOps for f64 {}

/// Apply element-wise operation on arrays using SIMD instructions
///
/// # Arguments
///
/// * `a` - First input array
/// * `b` - Second input array
/// * `op` - Operation to apply (add, subtract, multiply, divide)
///
/// # Returns
///
/// * Result array after applying the operation
pub fn simd_binary_op<F, S1, S2, D>(
    a: &ndarray::ArrayBase<S1, D>,
    b: &ndarray::ArrayBase<S2, D>,
    op: fn(F, F) -> F,
) -> Array<F, D>
where
    F: SimdOps + Float,
    S1: ndarray::Data<Elem = F>,
    S2: ndarray::Data<Elem = F>,
    D: Dimension,
{
    let mut result = Array::zeros(a.raw_dim());
    Zip::from(&mut result)
        .and(a)
        .and(b)
        .for_each(|r, &a, &b| *r = op(a, b));
    result
}

/// Compute element-wise maximum of two f32 arrays using SIMD
///
/// This function uses SIMD instructions for better performance when
/// processing large arrays of f32 values.
///
/// # Arguments
///
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
///
/// * Element-wise maximum array
#[cfg(feature = "simd")]
pub fn simd_maximum_f32(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    let n = a.len();
    let mut result = Array1::zeros(n);

    // Ensure arrays are the same length
    assert_eq!(
        a.len(),
        b.len(),
        "Arrays must have the same length for SIMD operations"
    );

    // Get raw pointers to the data for SIMD processing
    let a_slice = a.as_slice().unwrap();
    let b_slice = b.as_slice().unwrap();
    let result_slice = result.as_slice_mut().unwrap();

    // Process 8 elements at a time with SIMD
    let mut i = 0;
    let chunk_size = 8;

    while i + chunk_size <= n {
        // Create arrays for SIMD operations
        let a_arr = [
            a_slice[i],
            a_slice[i + 1],
            a_slice[i + 2],
            a_slice[i + 3],
            a_slice[i + 4],
            a_slice[i + 5],
            a_slice[i + 6],
            a_slice[i + 7],
        ];
        let b_arr = [
            b_slice[i],
            b_slice[i + 1],
            b_slice[i + 2],
            b_slice[i + 3],
            b_slice[i + 4],
            b_slice[i + 5],
            b_slice[i + 6],
            b_slice[i + 7],
        ];

        let a_vec = f32x8::new(a_arr);
        let b_vec = f32x8::new(b_arr);

        // Compute element-wise maximum
        let max_vec = a_vec.cmp_gt(b_vec).blend(a_vec, b_vec);

        // Extract results
        let result_arr: [f32; 8] = max_vec.into();
        result_slice[i..(chunk_size + i)].copy_from_slice(&result_arr[..chunk_size]);

        i += chunk_size;
    }

    // Process remaining elements
    for j in i..n {
        result[j] = a[j].max(b[j]);
    }

    result
}

/// Compute element-wise maximum of two f64 arrays using SIMD
///
/// This function uses SIMD instructions for better performance when
/// processing large arrays of f64 values.
///
/// # Arguments
///
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
///
/// * Element-wise maximum array
#[cfg(feature = "simd")]
pub fn simd_maximum_f64(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
    let n = a.len();
    let mut result = Array1::zeros(n);

    // Ensure arrays are the same length
    assert_eq!(
        a.len(),
        b.len(),
        "Arrays must have the same length for SIMD operations"
    );

    // Get raw pointers to the data for SIMD processing
    let a_slice = a.as_slice().unwrap();
    let b_slice = b.as_slice().unwrap();
    let result_slice = result.as_slice_mut().unwrap();

    // Process 4 elements at a time with SIMD
    let mut i = 0;
    let chunk_size = 4;

    while i + chunk_size <= n {
        // Create arrays for SIMD operations
        let a_arr = [a_slice[i], a_slice[i + 1], a_slice[i + 2], a_slice[i + 3]];
        let b_arr = [b_slice[i], b_slice[i + 1], b_slice[i + 2], b_slice[i + 3]];

        let a_vec = f64x4::new(a_arr);
        let b_vec = f64x4::new(b_arr);

        // Compute element-wise maximum
        let max_vec = a_vec.cmp_gt(b_vec).blend(a_vec, b_vec);

        // Extract results
        let result_arr: [f64; 4] = max_vec.into();
        result_slice[i..(chunk_size + i)].copy_from_slice(&result_arr[..chunk_size]);

        i += chunk_size;
    }

    // Process remaining elements
    for j in i..n {
        result[j] = a[j].max(b[j]);
    }

    result
}

/// Compute element-wise minimum of two f32 arrays using SIMD
///
/// This function uses SIMD instructions for better performance when
/// processing large arrays of f32 values.
///
/// # Arguments
///
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
///
/// * Element-wise minimum array
#[cfg(feature = "simd")]
pub fn simd_minimum_f32(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    let n = a.len();
    let mut result = Array1::zeros(n);

    // Ensure arrays are the same length
    assert_eq!(
        a.len(),
        b.len(),
        "Arrays must have the same length for SIMD operations"
    );

    // Get raw pointers to the data for SIMD processing
    let a_slice = a.as_slice().unwrap();
    let b_slice = b.as_slice().unwrap();
    let result_slice = result.as_slice_mut().unwrap();

    // Process 8 elements at a time with SIMD
    let mut i = 0;
    let chunk_size = 8;

    while i + chunk_size <= n {
        // Create arrays for SIMD operations
        let a_arr = [
            a_slice[i],
            a_slice[i + 1],
            a_slice[i + 2],
            a_slice[i + 3],
            a_slice[i + 4],
            a_slice[i + 5],
            a_slice[i + 6],
            a_slice[i + 7],
        ];
        let b_arr = [
            b_slice[i],
            b_slice[i + 1],
            b_slice[i + 2],
            b_slice[i + 3],
            b_slice[i + 4],
            b_slice[i + 5],
            b_slice[i + 6],
            b_slice[i + 7],
        ];

        let a_vec = f32x8::new(a_arr);
        let b_vec = f32x8::new(b_arr);

        // Compute element-wise minimum
        let min_vec = a_vec.cmp_lt(b_vec).blend(a_vec, b_vec);

        // Extract results
        let result_arr: [f32; 8] = min_vec.into();
        result_slice[i..(chunk_size + i)].copy_from_slice(&result_arr[..chunk_size]);

        i += chunk_size;
    }

    // Process remaining elements
    for j in i..n {
        result[j] = a[j].min(b[j]);
    }

    result
}

/// Compute element-wise minimum of two f64 arrays using SIMD
///
/// This function uses SIMD instructions for better performance when
/// processing large arrays of f64 values.
///
/// # Arguments
///
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
///
/// * Element-wise minimum array
#[cfg(feature = "simd")]
pub fn simd_minimum_f64(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
    let n = a.len();
    let mut result = Array1::zeros(n);

    // Ensure arrays are the same length
    assert_eq!(
        a.len(),
        b.len(),
        "Arrays must have the same length for SIMD operations"
    );

    // Get raw pointers to the data for SIMD processing
    let a_slice = a.as_slice().unwrap();
    let b_slice = b.as_slice().unwrap();
    let result_slice = result.as_slice_mut().unwrap();

    // Process 4 elements at a time with SIMD
    let mut i = 0;
    let chunk_size = 4;

    while i + chunk_size <= n {
        // Create arrays for SIMD operations
        let a_arr = [a_slice[i], a_slice[i + 1], a_slice[i + 2], a_slice[i + 3]];
        let b_arr = [b_slice[i], b_slice[i + 1], b_slice[i + 2], b_slice[i + 3]];

        let a_vec = f64x4::new(a_arr);
        let b_vec = f64x4::new(b_arr);

        // Compute element-wise minimum
        let min_vec = a_vec.cmp_lt(b_vec).blend(a_vec, b_vec);

        // Extract results
        let result_arr: [f64; 4] = min_vec.into();
        result_slice[i..(chunk_size + i)].copy_from_slice(&result_arr[..chunk_size]);

        i += chunk_size;
    }

    // Process remaining elements
    for j in i..n {
        result[j] = a[j].min(b[j]);
    }

    result
}

/// SIMD accelerated linspace function for f32 values
///
/// Creates a linearly spaced array between start and end (inclusive)
/// using SIMD instructions for better performance.
///
/// # Arguments
///
/// * `start` - Start value
/// * `end` - End value (inclusive)
/// * `num` - Number of points
///
/// # Returns
///
/// * Array of linearly spaced values
#[cfg(feature = "simd")]
pub fn simd_linspace_f32(start: f32, end: f32, num: usize) -> Array1<f32> {
    if num < 2 {
        return Array1::from_vec(vec![start]);
    }

    let mut result = Array1::zeros(num);
    let step = (end - start) / (num as f32 - 1.0);
    let result_slice = result.as_slice_mut().unwrap();

    // Use SIMD to calculate 4 elements at a time
    let mut i = 0;
    let chunk_size = 4;

    while i + chunk_size <= num {
        let indices = [i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32];
        let indices_vec = f32x4::new(indices);
        let steps = indices_vec * f32x4::splat(step);
        let values = f32x4::splat(start) + steps;

        // Extract results
        let result_arr: [f32; 4] = values.into();
        result_slice[i..(chunk_size + i)].copy_from_slice(&result_arr[..chunk_size]);

        i += chunk_size;
    }

    // Process remaining elements
    for (j, elem) in result_slice.iter_mut().enumerate().skip(i).take(num - i) {
        *elem = start + step * j as f32;
    }

    // Make sure the last value is exactly end to avoid floating point precision issues
    if let Some(last) = result.last_mut() {
        *last = end;
    }

    result
}

/// SIMD accelerated linspace function for f64 values
///
/// Creates a linearly spaced array between start and end (inclusive)
/// using SIMD instructions for better performance.
///
/// # Arguments
///
/// * `start` - Start value
/// * `end` - End value (inclusive)
/// * `num` - Number of points
///
/// # Returns
///
/// * Array of linearly spaced values
#[cfg(feature = "simd")]
pub fn simd_linspace_f64(start: f64, end: f64, num: usize) -> Array1<f64> {
    if num < 2 {
        return Array1::from_vec(vec![start]);
    }

    let mut result = Array1::zeros(num);
    let step = (end - start) / (num as f64 - 1.0);
    let result_slice = result.as_slice_mut().unwrap();

    // Use SIMD to calculate 2 elements at a time
    let mut i = 0;
    let chunk_size = 2;

    while i + chunk_size <= num {
        let indices = [i as f64, (i + 1) as f64];
        let indices_vec = f64x2::new(indices);
        let steps = indices_vec * f64x2::splat(step);
        let values = f64x2::splat(start) + steps;

        // Extract results
        let result_arr: [f64; 2] = values.into();
        result_slice[i..(chunk_size + i)].copy_from_slice(&result_arr[..chunk_size]);

        i += chunk_size;
    }

    // Process remaining elements
    for (j, elem) in result_slice.iter_mut().enumerate().skip(i).take(num - i) {
        *elem = start + step * j as f64;
    }

    // Make sure the last value is exactly end to avoid floating point precision issues
    if let Some(last) = result.last_mut() {
        *last = end;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::arr1;

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_maximum_f32() {
        let a = arr1(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let b = arr1(&[9.0f32, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);

        let result = simd_maximum_f32(&a.view(), &b.view());
        let expected = arr1(&[9.0f32, 8.0, 7.0, 6.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-5);
        }
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_maximum_f64() {
        let a = arr1(&[1.0f64, 2.0, 3.0, 4.0, 5.0]);
        let b = arr1(&[5.0f64, 4.0, 3.0, 2.0, 1.0]);

        let result = simd_maximum_f64(&a.view(), &b.view());
        let expected = arr1(&[5.0f64, 4.0, 3.0, 4.0, 5.0]);

        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_minimum_f32() {
        let a = arr1(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let b = arr1(&[9.0f32, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);

        let result = simd_minimum_f32(&a.view(), &b.view());
        let expected = arr1(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0]);

        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-5);
        }
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_minimum_f64() {
        let a = arr1(&[1.0f64, 2.0, 3.0, 4.0, 5.0]);
        let b = arr1(&[5.0f64, 4.0, 3.0, 2.0, 1.0]);

        let result = simd_minimum_f64(&a.view(), &b.view());
        let expected = arr1(&[1.0f64, 2.0, 3.0, 2.0, 1.0]);

        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_linspace_f32() {
        let result = simd_linspace_f32(0.0, 1.0, 5);
        let expected = arr1(&[0.0f32, 0.25, 0.5, 0.75, 1.0]);

        assert_eq!(result.len(), 5);
        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-6);
        }

        // Test endpoints
        assert_relative_eq!(result[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(result[4], 1.0, epsilon = 1e-6);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_linspace_f64() {
        let result = simd_linspace_f64(0.0, 1.0, 5);
        let expected = arr1(&[0.0f64, 0.25, 0.5, 0.75, 1.0]);

        assert_eq!(result.len(), 5);
        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-14);
        }

        // Test endpoints
        assert_relative_eq!(result[0], 0.0, epsilon = 1e-14);
        assert_relative_eq!(result[4], 1.0, epsilon = 1e-14);
    }
}
