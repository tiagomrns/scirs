//! Parallel processing utilities for improved performance
//!
//! This module provides parallel processing utilities and memory-efficient algorithms
//! for improved performance on multi-core systems. These implementations use the `rayon`
//! crate to leverage multi-threading capabilities.
//!
//! The module includes functions for:
//!
//! * General-purpose parallel operations (map, filter, reduce, for_each)
//! * Parallel array operations (element-wise max/min, linear space, etc.)
//! * Memory-efficient parallel algorithms for large datasets
//!
//! # Examples
//!
//! ```
//! use scirs2_core::parallel::{parallel_map, parallel_filter};
//!
//! // Parallel map example
//! let input = vec![1, 2, 3, 4, 5];
//! let squared = parallel_map(&input, |x| x * x);
//! assert_eq!(squared, vec![1, 4, 9, 16, 25]);
//!
//! // Parallel filter example
//! let filtered = parallel_filter(&input, |&x| x % 2 == 0);
//! assert_eq!(filtered, vec![2, 4]);
//! ```

use crate::error::{CoreError, CoreResult, ErrorContext};
use ndarray::{Array, Array1, Array2, Axis, Dimension, Zip};
use num_traits::Zero;
use rayon::prelude::*;
use std::ops::{Add, Div, Mul, Sub};

/// Apply element-wise operation on arrays using parallel execution
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
#[cfg(feature = "parallel")]
pub fn par_binary_op<F, S1, S2, D>(
    a: &ndarray::ArrayBase<S1, D>,
    b: &ndarray::ArrayBase<S2, D>,
    op: fn(F, F) -> F,
) -> Array<F, D>
where
    F: Send
        + Sync
        + Copy
        + Add<Output = F>
        + Sub<Output = F>
        + Mul<Output = F>
        + Div<Output = F>
        + Zero,
    S1: ndarray::Data<Elem = F>,
    S2: ndarray::Data<Elem = F>,
    D: Dimension,
{
    let mut result = Array::zeros(a.raw_dim());
    Zip::from(&mut result)
        .and(a)
        .and(b)
        .par_for_each(|r, &a, &b| *r = op(a, b));
    result
}

/// Compute element-wise maximum of two arrays using parallel execution
///
/// # Arguments
///
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
///
/// * Element-wise maximum array
#[cfg(feature = "parallel")]
pub fn par_maximum<S1, S2, D, T>(
    a: &ndarray::ArrayBase<S1, D>,
    b: &ndarray::ArrayBase<S2, D>,
) -> Array<T, D>
where
    S1: ndarray::Data<Elem = T>,
    S2: ndarray::Data<Elem = T>,
    D: Dimension,
    T: Send + Sync + Copy + PartialOrd + Zero,
{
    assert_eq!(
        a.shape(),
        b.shape(),
        "Arrays must have the same shape for element-wise maximum"
    );

    let mut result = a.to_owned();
    Zip::from(&mut result).and(b).par_for_each(|a, &b| {
        if b > *a {
            *a = b;
        }
    });

    result
}

/// Compute element-wise minimum of two arrays using parallel execution
///
/// # Arguments
///
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
///
/// * Element-wise minimum array
#[cfg(feature = "parallel")]
pub fn par_minimum<S1, S2, D, T>(
    a: &ndarray::ArrayBase<S1, D>,
    b: &ndarray::ArrayBase<S2, D>,
) -> Array<T, D>
where
    S1: ndarray::Data<Elem = T>,
    S2: ndarray::Data<Elem = T>,
    D: Dimension,
    T: Send + Sync + Copy + PartialOrd + Zero,
{
    assert_eq!(
        a.shape(),
        b.shape(),
        "Arrays must have the same shape for element-wise minimum"
    );

    let mut result = a.to_owned();
    Zip::from(&mut result).and(b).par_for_each(|a, &b| {
        if b < *a {
            *a = b;
        }
    });

    result
}

/// Creates a linearly spaced array between start and end (inclusive) using parallel execution
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
#[cfg(feature = "parallel")]
pub fn par_linspace(start: f64, end: f64, num: usize) -> Array1<f64> {
    if num < 2 {
        return Array1::from_vec(vec![start]);
    }

    let step = (end - start) / (num - 1) as f64;

    // Create the array with parallel iteration
    let result: Vec<f64> = (0..num)
        .into_par_iter()
        .map(|i| start + step * (i as f64))
        .collect();

    let mut result_array = Array1::from_vec(result);

    // Make sure the last value is exactly end to avoid floating point precision issues
    if let Some(last) = result_array.last_mut() {
        *last = end;
    }

    result_array
}

// Note: The par_reduce function was problematic and has been removed.
// It's better to directly use ndarray-parallel's fold feature for reduction operations.

/// Apply a chunk-wise operation to reduce memory usage for 2D arrays
///
/// Processes array in chunks to minimize memory usage for large arrays
///
/// # Arguments
///
/// * `array` - Input array
/// * `chunk_size` - Size of chunks to process
/// * `op` - Operation to apply to each chunk
///
/// # Returns
///
/// * Result after applying operation to all chunks
#[cfg(feature = "parallel")]
pub fn chunk_wise_op_2d<F, T, S>(
    array: &ndarray::ArrayBase<S, ndarray::Ix2>,
    chunk_size: usize,
    op: F,
) -> CoreResult<Array2<T>>
where
    F: Fn(&ndarray::ArrayView<S::Elem, ndarray::Ix2>) -> Array2<T> + Sync,
    S: ndarray::Data,
    T: Clone + Zero,
{
    // Initialize result with appropriate dimensions
    let mut result = Array2::zeros(array.raw_dim());

    // Process array in chunks
    for (chunk_idx, chunk) in array.axis_chunks_iter(Axis(0), chunk_size).enumerate() {
        let chunk_result = op(&chunk);

        // Calculate slice for this chunk in the result array
        let start = chunk_idx * chunk_size;
        let end = start + chunk.len_of(Axis(0));

        // Ensure we don't try to slice outside the bounds of the result array
        if end > result.len_of(Axis(0)) {
            return Err(CoreError::ValueError(ErrorContext::new(format!(
                "Chunk operation produced out-of-bounds result: offset={}, end={}, array_len={}",
                start,
                end,
                result.len_of(Axis(0))
            ))));
        }

        // Copy data manually since we can't use generic slicing
        for i in 0..chunk_result.len_of(Axis(0)) {
            if start + i < result.len_of(Axis(0)) {
                for j in 0..chunk_result.shape()[1] {
                    if j < result.shape()[1] {
                        // Use standard indexing which works for 2D arrays
                        result[[start + i, j]] = chunk_result[[i, j]].clone();
                    }
                }
            }
        }
    }

    Ok(result)
}

/// Compute cumulative sum along an axis for 1D arrays with minimal memory overhead
///
/// # Arguments
///
/// * `array` - Input 1D array
///
/// # Returns
///
/// * Array with cumulative sum
pub fn memory_efficient_cumsum_1d<T>(array: &Array1<T>) -> Array1<T>
where
    T: Copy + Add<Output = T> + Zero,
{
    let mut result = array.to_owned();
    let mut sum = T::zero();

    for i in 0..array.len() {
        sum = sum + result[i];
        result[i] = sum;
    }

    result
}

/// Compute cumulative sum along rows (axis 0) for 2D arrays
///
/// # Arguments
///
/// * `array` - Input 2D array
///
/// # Returns
///
/// * Array with cumulative sum along rows
pub fn cumsum_rows_2d<T>(array: &Array2<T>) -> Array2<T>
where
    T: Copy + Add<Output = T> + Zero,
{
    let mut result = array.to_owned();
    let (rows, cols) = (array.shape()[0], array.shape()[1]);

    for col in 0..cols {
        let mut sum = T::zero();
        for row in 0..rows {
            sum = sum + result[[row, col]];
            result[[row, col]] = sum;
        }
    }

    result
}

/// Compute cumulative sum along columns (axis 1) for 2D arrays
///
/// # Arguments
///
/// * `array` - Input 2D array
///
/// # Returns
///
/// * Array with cumulative sum along columns
pub fn cumsum_cols_2d<T>(array: &Array2<T>) -> Array2<T>
where
    T: Copy + Add<Output = T> + Zero,
{
    let mut result = array.to_owned();
    let (rows, cols) = (array.shape()[0], array.shape()[1]);

    for row in 0..rows {
        let mut sum = T::zero();
        for col in 0..cols {
            sum = sum + result[[row, col]];
            result[[row, col]] = sum;
        }
    }

    result
}

//----------------------------------------------------------
// General-purpose parallel processing functions
//----------------------------------------------------------

/// Apply a function to each element of a collection in parallel
///
/// # Arguments
///
/// * `input` - Input collection
/// * `f` - Function to apply to each element
///
/// # Returns
///
/// * Vector of results
///
/// # Examples
///
/// ```
/// use scirs2_core::parallel::parallel_map;
///
/// let input = vec![1, 2, 3, 4, 5];
/// let squared = parallel_map(&input, |x| x * x);
/// assert_eq!(squared, vec![1, 4, 9, 16, 25]);
/// ```
#[cfg(feature = "parallel")]
pub fn parallel_map<T, U, F>(input: &[T], f: F) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(&T) -> U + Sync + Send,
{
    input.par_iter().map(f).collect()
}

/// Filter elements of a collection in parallel
///
/// # Arguments
///
/// * `input` - Input collection
/// * `predicate` - Predicate function to determine which elements to keep
///
/// # Returns
///
/// * Vector of elements that satisfy the predicate
///
/// # Examples
///
/// ```
/// use scirs2_core::parallel::parallel_filter;
///
/// let input = vec![1, 2, 3, 4, 5];
/// let even_numbers = parallel_filter(&input, |&x| x % 2 == 0);
/// assert_eq!(even_numbers, vec![2, 4]);
/// ```
#[cfg(feature = "parallel")]
pub fn parallel_filter<T, F>(input: &[T], predicate: F) -> Vec<T>
where
    T: Sync + Clone + Send,
    F: Fn(&T) -> bool + Sync + Send,
{
    input
        .par_iter()
        .filter(|item| predicate(item))
        .map(|item| item.clone())
        .collect()
}

/// Apply a function to each element of a collection in parallel with no return value
///
/// # Arguments
///
/// * `input` - Input collection
/// * `f` - Function to apply to each element
///
/// # Examples
///
/// ```
/// use scirs2_core::parallel::parallel_for_each;
/// use std::sync::atomic::{AtomicUsize, Ordering};
///
/// let counter = AtomicUsize::new(0);
/// let input = vec![1, 2, 3, 4, 5];
///
/// parallel_for_each(&input, |&x| {
///     counter.fetch_add(x as usize, Ordering::Relaxed);
/// });
///
/// assert_eq!(counter.load(Ordering::Relaxed), 15);
/// ```
#[cfg(feature = "parallel")]
pub fn parallel_for_each<T, F>(input: &[T], f: F)
where
    T: Sync,
    F: Fn(&T) + Sync + Send,
{
    input.par_iter().for_each(f);
}

/// Apply a function to each mutable element of a collection in parallel
///
/// # Arguments
///
/// * `input` - Mutable input collection
/// * `f` - Function to apply to each element
///
/// # Examples
///
/// ```
/// use scirs2_core::parallel::parallel_for_each_mut;
///
/// let mut input = vec![1, 2, 3, 4, 5];
/// parallel_for_each_mut(&mut input, |x| *x *= 2);
/// assert_eq!(input, vec![2, 4, 6, 8, 10]);
/// ```
#[cfg(feature = "parallel")]
pub fn parallel_for_each_mut<T, F>(input: &mut [T], f: F)
where
    T: Send,
    F: Fn(&mut T) + Sync + Send,
{
    input.par_iter_mut().for_each(f);
}

/// Compute the sum of a collection in parallel
///
/// # Arguments
///
/// * `input` - Input collection
///
/// # Returns
///
/// * Sum of all elements
///
/// # Examples
///
/// ```
/// use scirs2_core::parallel::parallel_sum;
///
/// let input = vec![1, 2, 3, 4, 5];
/// let sum = parallel_sum(&input);
/// assert_eq!(sum, 15);
/// ```
#[cfg(feature = "parallel")]
pub fn parallel_sum<T>(input: &[T]) -> T
where
    T: Sync + Send + Clone + std::ops::Add<Output = T> + std::iter::Sum,
{
    input.par_iter().cloned().sum()
}

/// Compute the product of a collection in parallel
///
/// # Arguments
///
/// * `input` - Input collection
///
/// # Returns
///
/// * Product of all elements
///
/// # Examples
///
/// ```
/// use scirs2_core::parallel::parallel_product;
///
/// let input = vec![1, 2, 3, 4, 5];
/// let product = parallel_product(&input);
/// assert_eq!(product, 120);
/// ```
#[cfg(feature = "parallel")]
pub fn parallel_product<T>(input: &[T]) -> T
where
    T: Sync + Send + Clone + std::ops::Mul<Output = T> + std::iter::Product,
{
    input.par_iter().cloned().product()
}

/// Try to apply a function to each element of a collection in parallel, collecting the results
///
/// # Arguments
///
/// * `input` - Input collection
/// * `f` - Function to apply to each element
///
/// # Returns
///
/// * Result containing a vector of results, or the first error encountered
///
/// # Examples
///
/// ```
/// use scirs2_core::parallel::try_parallel_map;
///
/// let input = vec![0, 1, 2, 3, 4];
///
/// let result = try_parallel_map(&input, |&x| {
///     if x == 0 {
///         Err("Cannot process zero".to_string())
///     } else {
///         Ok(10 / x)
///     }
/// });
///
/// assert!(result.is_err());
///
/// let input = vec![1, 2, 3, 4, 5];
/// let result = try_parallel_map(&input, |&x| -> Result<i32, String> { Ok(10 / x) }).unwrap();
/// assert_eq!(result, vec![10, 5, 3, 2, 2]);
/// ```
#[cfg(feature = "parallel")]
pub fn try_parallel_map<T, U, F, E>(input: &[T], f: F) -> Result<Vec<U>, E>
where
    T: Sync,
    U: Send,
    E: Send + Sync + Clone,
    F: Fn(&T) -> Result<U, E> + Sync + Send,
{
    let results: Vec<Result<U, E>> = input.par_iter().map(f).collect();

    // Convert Vec<Result<U, E>> to Result<Vec<U>, E>
    let mut output = Vec::with_capacity(results.len());
    for result in results {
        match result {
            Ok(value) => output.push(value),
            Err(err) => return Err(err),
        }
    }

    Ok(output)
}

/// Try to apply a function to each element of a collection in parallel with no return value
///
/// # Arguments
///
/// * `input` - Input collection
/// * `f` - Function to apply to each element
///
/// # Returns
///
/// * Ok(()) if all operations succeeded, or the first error encountered
///
/// # Examples
///
/// ```
/// use scirs2_core::parallel::try_parallel_for_each;
/// use std::sync::atomic::{AtomicUsize, Ordering};
///
/// let counter = AtomicUsize::new(0);
/// let input = vec![1, 2, 3, 4, 5];
///
/// let result = try_parallel_for_each(&input, |&x| {
///     if x > 4 {
///         Err("Value too large".to_string())
///     } else {
///         counter.fetch_add(x as usize, Ordering::Relaxed);
///         Ok(())
///     }
/// });
///
/// assert!(result.is_err());
/// ```
#[cfg(feature = "parallel")]
pub fn try_parallel_for_each<T, F, E>(input: &[T], f: F) -> Result<(), E>
where
    T: Sync,
    E: Send + Sync + Clone,
    F: Fn(&T) -> Result<(), E> + Sync + Send,
{
    let results: Vec<Result<(), E>> = input.par_iter().map(f).collect();

    // Return the first error, if any
    for result in results {
        result?;
    }

    Ok(())
}

//----------------------------------------------------------
// Parallel array-specific operations
//----------------------------------------------------------

/// Parallel chunk-wise operations on 1D arrays
///
/// This function processes a 1D array in parallel chunks to minimize memory usage
/// while leveraging multiple CPU cores.
///
/// # Arguments
///
/// * `array` - Input 1D array
/// * `chunk_size` - Size of chunks to process in parallel
/// * `op` - Operation to apply to each chunk
///
/// # Returns
///
/// * Result array after applying operations to all chunks
///
/// # Examples
///
/// ```
/// use scirs2_core::parallel::parallel_chunk_op_1d;
/// use ndarray::arr1;
///
/// let array = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
/// let result = parallel_chunk_op_1d(&array, 2, |chunk| chunk.mapv(|x| x * x)).unwrap();
///
/// assert_eq!(result, arr1(&[1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]));
/// ```
#[cfg(feature = "parallel")]
pub fn parallel_chunk_op_1d<F, T, S>(
    array: &ndarray::ArrayBase<S, ndarray::Ix1>,
    chunk_size: usize,
    op: F,
) -> CoreResult<Array1<T>>
where
    F: Fn(&ndarray::ArrayView<S::Elem, ndarray::Ix1>) -> Array1<T> + Sync + Send,
    S: ndarray::Data,
    T: Clone + Zero + Send,
    S::Elem: Sync,
{
    if array.is_empty() {
        return Err(CoreError::ValueError(ErrorContext::new(
            "Input array is empty",
        )));
    }

    if chunk_size == 0 {
        return Err(CoreError::ValueError(ErrorContext::new(
            "Chunk size must be positive",
        )));
    }

    // Initialize result with appropriate dimensions
    let mut result = Array1::zeros(array.raw_dim());

    // Process array in chunks in parallel
    let chunks: Vec<_> = array.axis_chunks_iter(Axis(0), chunk_size).collect();
    let chunk_results: Vec<_> = chunks
        .par_iter()
        .map(|chunk| (chunk.len(), op(chunk)))
        .collect();

    // Combine chunk results
    let mut current_offset = 0;
    for (chunk_len, chunk_result) in chunk_results {
        let end = current_offset + chunk_len;

        if end > result.len() {
            return Err(CoreError::IndexError(ErrorContext::new(format!(
                "Chunk operation produced out-of-bounds result: offset={}, end={}, array_len={}",
                current_offset,
                end,
                result.len()
            ))));
        }

        // Copy data from chunk result to main result
        for i in 0..chunk_result.len() {
            result[current_offset + i] = chunk_result[i].clone();
        }

        current_offset = end;
    }

    Ok(result)
}

/// Parallel chunk-wise operations on 2D arrays
///
/// This function processes a 2D array in parallel chunks to minimize memory usage
/// while leveraging multiple CPU cores.
///
/// # Arguments
///
/// * `array` - Input 2D array
/// * `chunk_size` - Size of chunks to process in parallel
/// * `op` - Operation to apply to each chunk
///
/// # Returns
///
/// * Result array after applying operations to all chunks
///
/// # Examples
///
/// ```
/// use scirs2_core::parallel::parallel_chunk_op_2d;
/// use ndarray::array;
///
/// let array = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
/// let result = parallel_chunk_op_2d(&array, 2, |chunk| chunk.mapv(|x| x * x)).unwrap();
///
/// assert_eq!(result, array![[1.0, 4.0], [9.0, 16.0], [25.0, 36.0], [49.0, 64.0]]);
/// ```
#[cfg(feature = "parallel")]
pub fn parallel_chunk_op_2d<F, T, S>(
    array: &ndarray::ArrayBase<S, ndarray::Ix2>,
    chunk_size: usize,
    op: F,
) -> CoreResult<Array2<T>>
where
    F: Fn(&ndarray::ArrayView<S::Elem, ndarray::Ix2>) -> Array2<T> + Sync + Send,
    S: ndarray::Data,
    T: Clone + Zero + Send,
    S::Elem: Sync,
{
    if array.is_empty() {
        return Err(CoreError::ValueError(ErrorContext::new(
            "Input array is empty",
        )));
    }

    if chunk_size == 0 {
        return Err(CoreError::ValueError(ErrorContext::new(
            "Chunk size must be positive",
        )));
    }

    // Initialize result with appropriate dimensions
    let mut result = Array2::zeros(array.raw_dim());

    // Process array in chunks in parallel
    let chunks: Vec<_> = array.axis_chunks_iter(Axis(0), chunk_size).collect();
    let chunk_results: Vec<_> = chunks
        .par_iter()
        .map(|chunk| (chunk.shape()[0], op(chunk)))
        .collect();

    // Combine chunk results
    let mut current_offset = 0;
    for (chunk_rows, chunk_result) in chunk_results {
        let end = current_offset + chunk_rows;

        if end > result.shape()[0] {
            return Err(CoreError::IndexError(ErrorContext::new(format!(
                "Chunk operation produced out-of-bounds result: offset={}, end={}, array_rows={}",
                current_offset,
                end,
                result.shape()[0]
            ))));
        }

        // Copy data from chunk result to main result
        for i in 0..chunk_rows {
            for j in 0..result.shape()[1] {
                if j < chunk_result.shape()[1] {
                    result[[current_offset + i, j]] = chunk_result[[i, j]].clone();
                }
            }
        }

        current_offset = end;
    }

    Ok(result)
}

/// Apply a binary operation to 1D arrays in parallel
///
/// # Arguments
///
/// * `a` - First array
/// * `b` - Second array
/// * `op` - Binary operation to apply
///
/// # Returns
///
/// * Result of applying the operation element-wise
///
/// # Examples
///
/// ```
/// use scirs2_core::parallel::parallel_array_op_1d;
/// use ndarray::arr1;
///
/// let a = arr1(&[1.0, 2.0, 3.0, 4.0]);
/// let b = arr1(&[5.0, 6.0, 7.0, 8.0]);
///
/// let sum = parallel_array_op_1d(&a, &b, |x, y| x + y).unwrap();
/// assert_eq!(sum, arr1(&[6.0, 8.0, 10.0, 12.0]));
/// ```
#[cfg(feature = "parallel")]
pub fn parallel_array_op_1d<F, S1, S2, T, R>(
    a: &ndarray::ArrayBase<S1, ndarray::Ix1>,
    b: &ndarray::ArrayBase<S2, ndarray::Ix1>,
    op: F,
) -> CoreResult<Array1<R>>
where
    F: Fn(T, T) -> R + Sync + Send,
    S1: ndarray::Data<Elem = T>,
    S2: ndarray::Data<Elem = T>,
    T: Send + Sync + Copy,
    R: Send + Clone + Zero,
{
    if a.shape() != b.shape() {
        return Err(CoreError::DimensionError(ErrorContext::new(format!(
            "Arrays have different shapes: {:?} vs {:?}",
            a.shape(),
            b.shape()
        ))));
    }

    let mut result = Array1::zeros(a.raw_dim());

    Zip::from(&mut result)
        .and(a)
        .and(b)
        .par_for_each(|r, &a_val, &b_val| {
            *r = op(a_val, b_val);
        });

    Ok(result)
}

/// Apply a binary operation to 2D arrays in parallel
///
/// # Arguments
///
/// * `a` - First array
/// * `b` - Second array
/// * `op` - Binary operation to apply
///
/// # Returns
///
/// * Result of applying the operation element-wise
///
/// # Examples
///
/// ```
/// use scirs2_core::parallel::parallel_array_op_2d;
/// use ndarray::array;
///
/// let a = array![[1.0, 2.0], [3.0, 4.0]];
/// let b = array![[5.0, 6.0], [7.0, 8.0]];
///
/// let sum = parallel_array_op_2d(&a, &b, |x, y| x + y).unwrap();
/// assert_eq!(sum, array![[6.0, 8.0], [10.0, 12.0]]);
/// ```
#[cfg(feature = "parallel")]
pub fn parallel_array_op_2d<F, S1, S2, T, R>(
    a: &ndarray::ArrayBase<S1, ndarray::Ix2>,
    b: &ndarray::ArrayBase<S2, ndarray::Ix2>,
    op: F,
) -> CoreResult<Array2<R>>
where
    F: Fn(T, T) -> R + Sync + Send,
    S1: ndarray::Data<Elem = T>,
    S2: ndarray::Data<Elem = T>,
    T: Send + Sync + Copy,
    R: Send + Clone + Zero,
{
    if a.shape() != b.shape() {
        return Err(CoreError::DimensionError(ErrorContext::new(format!(
            "Arrays have different shapes: {:?} vs {:?}",
            a.shape(),
            b.shape()
        ))));
    }

    let mut result = Array2::zeros(a.raw_dim());

    Zip::from(&mut result)
        .and(a)
        .and(b)
        .par_for_each(|r, &a_val, &b_val| {
            *r = op(a_val, b_val);
        });

    Ok(result)
}

/// Parallel apply a function to every element in a 1D array
///
/// # Arguments
///
/// * `array` - Input 1D array
/// * `f` - Function to apply to each element
///
/// # Returns
///
/// * Result array after applying the function
///
/// # Examples
///
/// ```
/// use scirs2_core::parallel::parallel_map_array_1d;
/// use ndarray::arr1;
///
/// let a = arr1(&[1.0f64, 2.0, 3.0, 4.0]);
/// let result = parallel_map_array_1d(&a, |x: f64| x.powi(2)).unwrap();
/// assert_eq!(result, arr1(&[1.0, 4.0, 9.0, 16.0]));
/// ```
#[cfg(feature = "parallel")]
pub fn parallel_map_array_1d<F, S, T, R>(
    array: &ndarray::ArrayBase<S, ndarray::Ix1>,
    f: F,
) -> CoreResult<Array1<R>>
where
    F: Fn(T) -> R + Sync + Send,
    S: ndarray::Data<Elem = T>,
    T: Send + Sync + Copy,
    R: Send + Clone + Zero,
{
    if array.is_empty() {
        return Err(CoreError::ValueError(ErrorContext::new(
            "Input array is empty",
        )));
    }

    let mut result = Array1::zeros(array.raw_dim());

    Zip::from(&mut result).and(array).par_for_each(|r, &val| {
        *r = f(val);
    });

    Ok(result)
}

/// Parallel apply a function to every element in a 2D array
///
/// # Arguments
///
/// * `array` - Input 2D array
/// * `f` - Function to apply to each element
///
/// # Returns
///
/// * Result array after applying the function
///
/// # Examples
///
/// ```
/// use scirs2_core::parallel::parallel_map_array_2d;
/// use ndarray::array;
///
/// let a = array![[1.0f64, 2.0], [3.0, 4.0]];
/// let result = parallel_map_array_2d(&a, |x: f64| x.powi(2)).unwrap();
/// assert_eq!(result, array![[1.0, 4.0], [9.0, 16.0]]);
/// ```
#[cfg(feature = "parallel")]
pub fn parallel_map_array_2d<F, S, T, R>(
    array: &ndarray::ArrayBase<S, ndarray::Ix2>,
    f: F,
) -> CoreResult<Array2<R>>
where
    F: Fn(T) -> R + Sync + Send,
    S: ndarray::Data<Elem = T>,
    T: Send + Sync + Copy,
    R: Send + Clone + Zero,
{
    if array.is_empty() {
        return Err(CoreError::ValueError(ErrorContext::new(
            "Input array is empty",
        )));
    }

    let mut result = Array2::zeros(array.raw_dim());

    Zip::from(&mut result).and(array).par_for_each(|r, &val| {
        *r = f(val);
    });

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{arr1, arr2};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    #[cfg(feature = "parallel")]
    fn test_par_maximum() {
        let a = arr2(&[[1, 2], [3, 4]]);
        let b = arr2(&[[5, 1], [7, 2]]);

        let result = par_maximum(&a, &b);
        let expected = arr2(&[[5, 2], [7, 4]]);

        assert_eq!(result, expected);
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_par_minimum() {
        let a = arr2(&[[1, 2], [3, 4]]);
        let b = arr2(&[[5, 1], [7, 2]]);

        let result = par_minimum(&a, &b);
        let expected = arr2(&[[1, 1], [3, 2]]);

        assert_eq!(result, expected);
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_par_linspace() {
        let result = par_linspace(0.0, 1.0, 5);
        let expected = arr1(&[0.0, 0.25, 0.5, 0.75, 1.0]);
        assert_eq!(result.len(), 5);

        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-14);
        }

        // Test endpoints
        assert_relative_eq!(result[0], 0.0);
        assert_relative_eq!(result[4], 1.0);
    }

    #[test]
    fn test_memory_efficient_cumsum_1d() {
        let a = arr1(&[1.0, 2.0, 3.0, 4.0]);
        let result = memory_efficient_cumsum_1d(&a);
        let expected = arr1(&[1.0, 3.0, 6.0, 10.0]);

        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_cumsum_rows_and_cols_2d() {
        // Test 2D array
        let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);

        // Along rows (axis 0)
        let result = cumsum_rows_2d(&a);
        let expected = arr2(&[[1.0, 2.0], [4.0, 6.0]]);

        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-14);
        }

        // Along columns (axis 1)
        let result = cumsum_cols_2d(&a);
        let expected = arr2(&[[1.0, 3.0], [3.0, 7.0]]);

        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-14);
        }
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_map() {
        let input = vec![1, 2, 3, 4, 5];
        let squared = parallel_map(&input, |&x| x * x);
        assert_eq!(squared, vec![1, 4, 9, 16, 25]);
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_filter() {
        let input = vec![1, 2, 3, 4, 5];
        let evens = parallel_filter(&input, |&x| x % 2 == 0);
        assert_eq!(evens, vec![2, 4]);
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_for_each() {
        let counter = AtomicUsize::new(0);
        let input = vec![1, 2, 3, 4, 5];

        parallel_for_each(&input, |&x| {
            counter.fetch_add(x as usize, Ordering::Relaxed);
        });

        assert_eq!(counter.load(Ordering::Relaxed), 15);
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_for_each_mut() {
        let mut input = vec![1, 2, 3, 4, 5];
        parallel_for_each_mut(&mut input, |x| *x *= 2);
        assert_eq!(input, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_sum() {
        let input = vec![1, 2, 3, 4, 5];
        let sum = parallel_sum(&input);
        assert_eq!(sum, 15);
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_product() {
        let input = vec![1, 2, 3, 4, 5];
        let product = parallel_product(&input);
        assert_eq!(product, 120);
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_try_parallel_map() {
        // Test successful case
        let input = vec![1, 2, 3, 4, 5];
        let result = try_parallel_map(&input, |&x| -> Result<_, String> { Ok(x * 2) }).unwrap();
        assert_eq!(result, vec![2, 4, 6, 8, 10]);

        // Test failure case
        let result = try_parallel_map(&input, |&x| -> Result<_, String> {
            if x > 3 {
                Err("Value too large".to_string())
            } else {
                Ok(x * 2)
            }
        });
        assert!(result.is_err());
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_array_op_2d() {
        let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let b = arr2(&[[5.0, 6.0], [7.0, 8.0]]);

        let sum = parallel_array_op_2d(&a, &b, |x, y| x + y).unwrap();
        let expected_sum = arr2(&[[6.0, 8.0], [10.0, 12.0]]);
        assert_eq!(sum, expected_sum);

        let product = parallel_array_op_2d(&a, &b, |x, y| x * y).unwrap();
        let expected_product = arr2(&[[5.0, 12.0], [21.0, 32.0]]);
        assert_eq!(product, expected_product);
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_map_array_2d() {
        let a = arr2(&[[1.0f64, 2.0], [3.0, 4.0]]);
        let result = parallel_map_array_2d(&a, |x: f64| x.powi(2)).unwrap();
        let expected = arr2(&[[1.0, 4.0], [9.0, 16.0]]);
        assert_eq!(result, expected);
    }
}
