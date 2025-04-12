//! Basic statistical functions for ndarray arrays
//!
//! This module provides basic statistical functions for ndarray arrays, similar to NumPy's
//! statistical functions.

use ndarray::{Array, ArrayView, Axis, Dimension, Ix1, Ix2};
use num_traits::{Float, FromPrimitive};

/// Result type for histogram function
pub type HistogramResult<T> = Result<(Array<usize, Ix1>, Array<T, Ix1>), &'static str>;

/// Result type for histogram2d function
pub type Histogram2dResult<T> =
    Result<(Array<usize, Ix2>, Array<T, Ix1>, Array<T, Ix1>), &'static str>;

/// Calculate the mean of array elements (2D arrays)
///
/// # Arguments
///
/// * `array` - The input 2D array
/// * `axis` - Optional axis along which to compute the mean (None for global mean)
///
/// # Returns
///
/// The mean of the array elements
///
/// # Examples
///
/// ```
/// use ndarray::{array, Axis};
/// use scirs2_core::ndarray_ext::stats::mean_2d;
///
/// let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
///
/// // Global mean
/// let global_mean = mean_2d(&a.view(), None).unwrap();
/// assert_eq!(global_mean[0], 3.5);
///
/// // Mean along axis 0 (columns)
/// let col_means = mean_2d(&a.view(), Some(Axis(0))).unwrap();
/// assert_eq!(col_means.len(), 3);
/// assert_eq!(col_means[0], 2.5);
/// assert_eq!(col_means[1], 3.5);
/// assert_eq!(col_means[2], 4.5);
/// ```
pub fn mean_2d<T>(
    array: &ArrayView<T, Ix2>,
    axis: Option<Axis>,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float + FromPrimitive,
{
    if array.is_empty() {
        return Err("Cannot compute mean of an empty array");
    }

    match axis {
        Some(ax) => {
            let (rows, cols) = (array.shape()[0], array.shape()[1]);

            match ax.index() {
                0 => {
                    // Mean along axis 0 (columns)
                    let mut result = Array::<T, Ix1>::zeros(cols);
                    let n = T::from_usize(rows).unwrap();

                    for j in 0..cols {
                        let mut sum = T::zero();
                        for i in 0..rows {
                            sum = sum + array[[i, j]];
                        }
                        result[j] = sum / n;
                    }

                    Ok(result)
                }
                1 => {
                    // Mean along axis 1 (rows)
                    let mut result = Array::<T, Ix1>::zeros(rows);
                    let n = T::from_usize(cols).unwrap();

                    for i in 0..rows {
                        let mut sum = T::zero();
                        for j in 0..cols {
                            sum = sum + array[[i, j]];
                        }
                        result[i] = sum / n;
                    }

                    Ok(result)
                }
                _ => Err("Axis index out of bounds for 2D array"),
            }
        }
        None => {
            // Global mean
            let total_elements = array.len();
            let mut sum = T::zero();

            for &val in array.iter() {
                sum = sum + val;
            }

            let count = T::from_usize(total_elements).ok_or("Cannot convert array length to T")?;
            Ok(Array::from_elem(1, sum / count))
        }
    }
}

/// Calculate the median of array elements (2D arrays)
///
/// # Arguments
///
/// * `array` - The input 2D array
/// * `axis` - Optional axis along which to compute the median (None for global median)
///
/// # Returns
///
/// The median of the array elements
///
/// # Examples
///
/// ```
/// use ndarray::{array, Axis};
/// use scirs2_core::ndarray_ext::stats::median_2d;
///
/// let a = array![[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]];
///
/// // Global median
/// let global_median = median_2d(&a.view(), None).unwrap();
/// assert_eq!(global_median[0], 3.5);
///
/// // Median along axis 0 (columns)
/// let col_medians = median_2d(&a.view(), Some(Axis(0))).unwrap();
/// assert_eq!(col_medians.len(), 3);
/// assert_eq!(col_medians[0], 1.5);
/// assert_eq!(col_medians[1], 3.5);
/// assert_eq!(col_medians[2], 5.5);
/// ```
pub fn median_2d<T>(
    array: &ArrayView<T, Ix2>,
    axis: Option<Axis>,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float + FromPrimitive,
{
    if array.is_empty() {
        return Err("Cannot compute median of an empty array");
    }

    match axis {
        Some(ax) => {
            let (rows, cols) = (array.shape()[0], array.shape()[1]);

            match ax.index() {
                0 => {
                    // Median along axis 0 (columns)
                    let mut result = Array::<T, Ix1>::zeros(cols);

                    for j in 0..cols {
                        let mut column_values = Vec::with_capacity(rows);
                        for i in 0..rows {
                            column_values.push(array[[i, j]]);
                        }

                        column_values
                            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                        let median_value = if column_values.len() % 2 == 0 {
                            let mid = column_values.len() / 2;
                            (column_values[mid - 1] + column_values[mid])
                                / T::from_f64(2.0).unwrap()
                        } else {
                            column_values[column_values.len() / 2]
                        };

                        result[j] = median_value;
                    }

                    Ok(result)
                }
                1 => {
                    // Median along axis 1 (rows)
                    let mut result = Array::<T, Ix1>::zeros(rows);

                    for i in 0..rows {
                        let mut row_values = Vec::with_capacity(cols);
                        for j in 0..cols {
                            row_values.push(array[[i, j]]);
                        }

                        row_values
                            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                        let median_value = if row_values.len() % 2 == 0 {
                            let mid = row_values.len() / 2;
                            (row_values[mid - 1] + row_values[mid]) / T::from_f64(2.0).unwrap()
                        } else {
                            row_values[row_values.len() / 2]
                        };

                        result[i] = median_value;
                    }

                    Ok(result)
                }
                _ => Err("Axis index out of bounds for 2D array"),
            }
        }
        None => {
            // Global median
            let mut values: Vec<_> = array.iter().copied().collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let median_value = if values.len() % 2 == 0 {
                let mid = values.len() / 2;
                (values[mid - 1] + values[mid]) / T::from_f64(2.0).unwrap()
            } else {
                values[values.len() / 2]
            };

            Ok(Array::from_elem(1, median_value))
        }
    }
}

/// Calculate the standard deviation of array elements (2D arrays)
///
/// # Arguments
///
/// * `array` - The input 2D array
/// * `axis` - Optional axis along which to compute the std dev (None for global std dev)
/// * `ddof` - Delta degrees of freedom (default 0)
///
/// # Returns
///
/// The standard deviation of the array elements
///
/// # Examples
///
/// ```
/// use ndarray::{array, Axis};
/// use scirs2_core::ndarray_ext::stats::std_dev_2d;
///
/// let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
///
/// // Global standard deviation
/// let global_std = std_dev_2d(&a.view(), None, 1).unwrap();
/// assert!((global_std[0] - 1.87082869339_f64).abs() < 1e-10);
///
/// // Standard deviation along axis 0 (columns)
/// let col_stds = std_dev_2d(&a.view(), Some(Axis(0)), 1).unwrap();
/// assert_eq!(col_stds.len(), 3);
/// ```
pub fn std_dev_2d<T>(
    array: &ArrayView<T, Ix2>,
    axis: Option<Axis>,
    ddof: usize,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float + FromPrimitive,
{
    let var_result = variance_2d(array, axis, ddof)?;
    Ok(var_result.mapv(|x| x.sqrt()))
}

/// Calculate the variance of array elements (2D arrays)
///
/// # Arguments
///
/// * `array` - The input 2D array
/// * `axis` - Optional axis along which to compute the variance (None for global variance)
/// * `ddof` - Delta degrees of freedom (default 0)
///
/// # Returns
///
/// The variance of the array elements
///
/// # Examples
///
/// ```
/// use ndarray::{array, Axis};
/// use scirs2_core::ndarray_ext::stats::variance_2d;
///
/// let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
///
/// // Global variance
/// let global_var = variance_2d(&a.view(), None, 1).unwrap();
/// assert!((global_var[0] - 3.5_f64).abs() < 1e-10);
///
/// // Variance along axis 0 (columns)
/// let col_vars = variance_2d(&a.view(), Some(Axis(0)), 1).unwrap();
/// assert_eq!(col_vars.len(), 3);
/// ```
pub fn variance_2d<T>(
    array: &ArrayView<T, Ix2>,
    axis: Option<Axis>,
    ddof: usize,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float + FromPrimitive,
{
    if array.is_empty() {
        return Err("Cannot compute variance of an empty array");
    }

    match axis {
        Some(ax) => {
            let (rows, cols) = (array.shape()[0], array.shape()[1]);

            match ax.index() {
                0 => {
                    // Variance along axis 0 (columns)
                    let means = mean_2d(array, Some(ax))?;

                    if rows <= ddof {
                        return Err(
                            "Not enough data points for variance calculation with given ddof",
                        );
                    }

                    let mut result = Array::<T, Ix1>::zeros(cols);

                    for j in 0..cols {
                        let mut sum_sq_diff = T::zero();
                        for i in 0..rows {
                            let diff = array[[i, j]] - means[j];
                            sum_sq_diff = sum_sq_diff + (diff * diff);
                        }

                        let divisor = T::from_usize(rows - ddof).unwrap();
                        result[j] = sum_sq_diff / divisor;
                    }

                    Ok(result)
                }
                1 => {
                    // Variance along axis 1 (rows)
                    let means = mean_2d(array, Some(ax))?;

                    if cols <= ddof {
                        return Err(
                            "Not enough data points for variance calculation with given ddof",
                        );
                    }

                    let mut result = Array::<T, Ix1>::zeros(rows);

                    for i in 0..rows {
                        let mut sum_sq_diff = T::zero();
                        for j in 0..cols {
                            let diff = array[[i, j]] - means[i];
                            sum_sq_diff = sum_sq_diff + (diff * diff);
                        }

                        let divisor = T::from_usize(cols - ddof).unwrap();
                        result[i] = sum_sq_diff / divisor;
                    }

                    Ok(result)
                }
                _ => Err("Axis index out of bounds for 2D array"),
            }
        }
        None => {
            // Global variance
            let total_elements = array.len();

            if total_elements <= ddof {
                return Err("Not enough data points for variance calculation with given ddof");
            }

            // Calculate global mean
            let global_mean = mean_2d(array, None)?[0];

            // Calculate sum of squared differences from the mean
            let mut sum_sq_diff = T::zero();
            for &val in array.iter() {
                let diff = val - global_mean;
                sum_sq_diff = sum_sq_diff + (diff * diff);
            }

            let divisor = T::from_usize(total_elements - ddof).unwrap();

            Ok(Array::from_elem(1, sum_sq_diff / divisor))
        }
    }
}

/// Calculate the minimum value(s) of array elements (2D arrays)
///
/// # Arguments
///
/// * `array` - The input 2D array
/// * `axis` - Optional axis along which to compute the minimum (None for global minimum)
///
/// # Returns
///
/// The minimum value(s) of the array elements
///
/// # Examples
///
/// ```
/// use ndarray::{array, Axis};
/// use scirs2_core::ndarray_ext::stats::min_2d;
///
/// let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 0.5]];
///
/// // Global minimum
/// let global_min = min_2d(&a.view(), None).unwrap();
/// assert_eq!(global_min[0], 0.5);
///
/// // Minimum along axis 0 (columns)
/// let col_mins = min_2d(&a.view(), Some(Axis(0))).unwrap();
/// assert_eq!(col_mins.len(), 3);
/// assert_eq!(col_mins[0], 1.0);
/// assert_eq!(col_mins[2], 0.5);
/// ```
pub fn min_2d<T>(
    array: &ArrayView<T, Ix2>,
    axis: Option<Axis>,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float,
{
    if array.is_empty() {
        return Err("Cannot compute minimum of an empty array");
    }

    match axis {
        Some(ax) => {
            let (rows, cols) = (array.shape()[0], array.shape()[1]);

            match ax.index() {
                0 => {
                    // Minimum along axis 0 (columns)
                    let mut result = Array::<T, Ix1>::zeros(cols);

                    for j in 0..cols {
                        let mut min_val = T::infinity();
                        for i in 0..rows {
                            let val = array[[i, j]];
                            if val < min_val {
                                min_val = val;
                            }
                        }
                        result[j] = min_val;
                    }

                    Ok(result)
                }
                1 => {
                    // Minimum along axis 1 (rows)
                    let mut result = Array::<T, Ix1>::zeros(rows);

                    for i in 0..rows {
                        let mut min_val = T::infinity();
                        for j in 0..cols {
                            let val = array[[i, j]];
                            if val < min_val {
                                min_val = val;
                            }
                        }
                        result[i] = min_val;
                    }

                    Ok(result)
                }
                _ => Err("Axis index out of bounds for 2D array"),
            }
        }
        None => {
            // Global minimum
            let mut min_val = T::infinity();
            for &val in array.iter() {
                if val < min_val {
                    min_val = val;
                }
            }

            Ok(Array::from_elem(1, min_val))
        }
    }
}

/// Calculate the maximum value(s) of array elements (2D arrays)
///
/// # Arguments
///
/// * `array` - The input 2D array
/// * `axis` - Optional axis along which to compute the maximum (None for global maximum)
///
/// # Returns
///
/// The maximum value(s) of the array elements
///
/// # Examples
///
/// ```
/// use ndarray::{array, Axis};
/// use scirs2_core::ndarray_ext::stats::max_2d;
///
/// let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 0.5]];
///
/// // Global maximum
/// let global_max = max_2d(&a.view(), None).unwrap();
/// assert_eq!(global_max[0], 5.0);
///
/// // Maximum along axis 0 (columns)
/// let col_maxs = max_2d(&a.view(), Some(Axis(0))).unwrap();
/// assert_eq!(col_maxs.len(), 3);
/// assert_eq!(col_maxs[0], 4.0);
/// assert_eq!(col_maxs[2], 3.0);
/// ```
pub fn max_2d<T>(
    array: &ArrayView<T, Ix2>,
    axis: Option<Axis>,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float,
{
    if array.is_empty() {
        return Err("Cannot compute maximum of an empty array");
    }

    match axis {
        Some(ax) => {
            let (rows, cols) = (array.shape()[0], array.shape()[1]);

            match ax.index() {
                0 => {
                    // Maximum along axis 0 (columns)
                    let mut result = Array::<T, Ix1>::zeros(cols);

                    for j in 0..cols {
                        let mut max_val = T::neg_infinity();
                        for i in 0..rows {
                            let val = array[[i, j]];
                            if val > max_val {
                                max_val = val;
                            }
                        }
                        result[j] = max_val;
                    }

                    Ok(result)
                }
                1 => {
                    // Maximum along axis 1 (rows)
                    let mut result = Array::<T, Ix1>::zeros(rows);

                    for i in 0..rows {
                        let mut max_val = T::neg_infinity();
                        for j in 0..cols {
                            let val = array[[i, j]];
                            if val > max_val {
                                max_val = val;
                            }
                        }
                        result[i] = max_val;
                    }

                    Ok(result)
                }
                _ => Err("Axis index out of bounds for 2D array"),
            }
        }
        None => {
            // Global maximum
            let mut max_val = T::neg_infinity();
            for &val in array.iter() {
                if val > max_val {
                    max_val = val;
                }
            }

            Ok(Array::from_elem(1, max_val))
        }
    }
}

/// Calculate the sum of array elements (2D arrays)
///
/// # Arguments
///
/// * `array` - The input 2D array
/// * `axis` - Optional axis along which to compute the sum (None for global sum)
///
/// # Returns
///
/// The sum of the array elements
///
/// # Examples
///
/// ```
/// use ndarray::{array, Axis};
/// use scirs2_core::ndarray_ext::stats::sum_2d;
///
/// let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
///
/// // Global sum
/// let global_sum = sum_2d(&a.view(), None).unwrap();
/// assert_eq!(global_sum[0], 21.0);
///
/// // Sum along axis 0 (columns)
/// let col_sums = sum_2d(&a.view(), Some(Axis(0))).unwrap();
/// assert_eq!(col_sums.len(), 3);
/// assert_eq!(col_sums[0], 5.0);
/// assert_eq!(col_sums[1], 7.0);
/// assert_eq!(col_sums[2], 9.0);
/// ```
pub fn sum_2d<T>(
    array: &ArrayView<T, Ix2>,
    axis: Option<Axis>,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float,
{
    if array.is_empty() {
        return Err("Cannot compute sum of an empty array");
    }

    match axis {
        Some(ax) => {
            let (rows, cols) = (array.shape()[0], array.shape()[1]);

            match ax.index() {
                0 => {
                    // Sum along axis 0 (columns)
                    let mut result = Array::<T, Ix1>::zeros(cols);

                    for j in 0..cols {
                        let mut sum_val = T::zero();
                        for i in 0..rows {
                            sum_val = sum_val + array[[i, j]];
                        }
                        result[j] = sum_val;
                    }

                    Ok(result)
                }
                1 => {
                    // Sum along axis 1 (rows)
                    let mut result = Array::<T, Ix1>::zeros(rows);

                    for i in 0..rows {
                        let mut sum_val = T::zero();
                        for j in 0..cols {
                            sum_val = sum_val + array[[i, j]];
                        }
                        result[i] = sum_val;
                    }

                    Ok(result)
                }
                _ => Err("Axis index out of bounds for 2D array"),
            }
        }
        None => {
            // Global sum
            let mut sum_val = T::zero();
            for &val in array.iter() {
                sum_val = sum_val + val;
            }

            Ok(Array::from_elem(1, sum_val))
        }
    }
}

/// Calculate the product of array elements (2D arrays)
///
/// # Arguments
///
/// * `array` - The input 2D array
/// * `axis` - Optional axis along which to compute the product (None for global product)
///
/// # Returns
///
/// The product of the array elements
///
/// # Examples
///
/// ```
/// use ndarray::{array, Axis};
/// use scirs2_core::ndarray_ext::stats::product_2d;
///
/// let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
///
/// // Global product
/// let global_prod = product_2d(&a.view(), None).unwrap();
/// assert_eq!(global_prod[0], 720.0);
///
/// // Product along axis 0 (columns)
/// let col_prods = product_2d(&a.view(), Some(Axis(0))).unwrap();
/// assert_eq!(col_prods.len(), 3);
/// assert_eq!(col_prods[0], 4.0);
/// assert_eq!(col_prods[1], 10.0);
/// assert_eq!(col_prods[2], 18.0);
/// ```
pub fn product_2d<T>(
    array: &ArrayView<T, Ix2>,
    axis: Option<Axis>,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float,
{
    if array.is_empty() {
        return Err("Cannot compute product of an empty array");
    }

    match axis {
        Some(ax) => {
            let (rows, cols) = (array.shape()[0], array.shape()[1]);

            match ax.index() {
                0 => {
                    // Product along axis 0 (columns)
                    let mut result = Array::<T, Ix1>::zeros(cols);

                    for j in 0..cols {
                        let mut prod_val = T::one();
                        for i in 0..rows {
                            prod_val = prod_val * array[[i, j]];
                        }
                        result[j] = prod_val;
                    }

                    Ok(result)
                }
                1 => {
                    // Product along axis 1 (rows)
                    let mut result = Array::<T, Ix1>::zeros(rows);

                    for i in 0..rows {
                        let mut prod_val = T::one();
                        for j in 0..cols {
                            prod_val = prod_val * array[[i, j]];
                        }
                        result[i] = prod_val;
                    }

                    Ok(result)
                }
                _ => Err("Axis index out of bounds for 2D array"),
            }
        }
        None => {
            // Global product
            let mut prod_val = T::one();
            for &val in array.iter() {
                prod_val = prod_val * val;
            }

            Ok(Array::from_elem(1, prod_val))
        }
    }
}

/// Calculate the percentile of array elements (2D arrays)
///
/// # Arguments
///
/// * `array` - The input 2D array
/// * `q` - Percentile to compute, which must be between 0 and 100 inclusive
/// * `axis` - Optional axis along which to compute the percentile (None for global percentile)
///
/// # Returns
///
/// The percentile of the array elements
///
/// # Examples
///
/// ```
/// use ndarray::{array, Axis};
/// use scirs2_core::ndarray_ext::stats::percentile_2d;
///
/// let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
///
/// // Global median (50th percentile)
/// let global_median = percentile_2d(&a.view(), 50.0, None).unwrap();
/// assert_eq!(global_median[0], 3.5);
///
/// // 75th percentile along axis 0 (columns)
/// let col_q75 = percentile_2d(&a.view(), 75.0, Some(Axis(0))).unwrap();
/// assert_eq!(col_q75.len(), 3);
/// assert_eq!(col_q75[0], 3.25);
/// ```
pub fn percentile_2d<T>(
    array: &ArrayView<T, Ix2>,
    q: f64,
    axis: Option<Axis>,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float + FromPrimitive,
{
    if array.is_empty() {
        return Err("Cannot compute percentile of an empty array");
    }

    if !(0.0..=100.0).contains(&q) {
        return Err("Percentile must be between 0 and 100");
    }

    // Convert percentile to a fraction
    let q_frac = q / 100.0;

    match axis {
        Some(ax) => {
            let (rows, cols) = (array.shape()[0], array.shape()[1]);

            match ax.index() {
                0 => {
                    // Percentile along axis 0 (columns)
                    let mut result = Array::<T, Ix1>::zeros(cols);

                    for j in 0..cols {
                        let mut column_values = Vec::with_capacity(rows);
                        for i in 0..rows {
                            column_values.push(array[[i, j]]);
                        }

                        column_values
                            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                        let percentile_value = compute_percentile(&column_values, q_frac)?;
                        result[j] = percentile_value;
                    }

                    Ok(result)
                }
                1 => {
                    // Percentile along axis 1 (rows)
                    let mut result = Array::<T, Ix1>::zeros(rows);

                    for i in 0..rows {
                        let mut row_values = Vec::with_capacity(cols);
                        for j in 0..cols {
                            row_values.push(array[[i, j]]);
                        }

                        row_values
                            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                        let percentile_value = compute_percentile(&row_values, q_frac)?;
                        result[i] = percentile_value;
                    }

                    Ok(result)
                }
                _ => Err("Axis index out of bounds for 2D array"),
            }
        }
        None => {
            // Global percentile
            let mut values: Vec<_> = array.iter().copied().collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let percentile_value = compute_percentile(&values, q_frac)?;
            Ok(Array::from_elem(1, percentile_value))
        }
    }
}

// Helper function to compute a percentile from a sorted array
fn compute_percentile<T>(sorted_values: &[T], q: f64) -> Result<T, &'static str>
where
    T: Clone + Float + FromPrimitive,
{
    let n = sorted_values.len();
    if n == 0 {
        return Err("Cannot compute percentile of an empty array");
    }

    if n == 1 {
        return Ok(sorted_values[0]);
    }

    // Calculate the index as a float
    let idx = q * (n as f64 - 1.0);

    // Get the lower index and fraction
    let idx_floor = idx.floor();
    let idx_lower = idx_floor as usize;
    let fraction = idx - idx_floor;

    // Handle edge cases
    if idx_lower >= n - 1 {
        return Ok(sorted_values[n - 1]);
    }

    // Linear interpolation between the values
    let lower_val = sorted_values[idx_lower];
    let upper_val = sorted_values[idx_lower + 1];

    let fraction_t = T::from_f64(fraction).ok_or("Cannot convert fraction to T")?;
    let one_minus_fraction = T::from_f64(1.0 - fraction).ok_or("Cannot convert 1-fraction to T")?;

    Ok(lower_val * one_minus_fraction + upper_val * fraction_t)
}

/// Calculate a histogram of a 1D array of data
///
/// # Arguments
///
/// * `array` - The input 1D array
/// * `bins` - Either the number of bins (usize) or array of bin edges
/// * `range` - Optional range (min, max) to use. If None, the range is (min(array), max(array))
/// * `weights` - Optional array of weights for each data point
///
/// # Returns
///
/// A tuple containing (histogram, bin_edges)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::stats::histogram;
///
/// let data = array![1.0, 1.0, 2.0, 2.0, 2.5, 3.0, 3.0, 4.0, 4.5, 5.0];
/// let (hist, bin_edges) = histogram(data.view(), 5, None, None).unwrap();
///
/// assert_eq!(hist.len(), 5);
/// assert_eq!(bin_edges.len(), 6);
/// assert_eq!(hist[0], 2); // Two values in the first bin
/// assert_eq!(hist[1], 3); // Three values in the second bin
/// ```
pub fn histogram<T>(
    array: ArrayView<T, Ix1>,
    bins: usize,
    range: Option<(T, T)>,
    weights: Option<ArrayView<T, Ix1>>,
) -> HistogramResult<T>
where
    T: Clone + Float + FromPrimitive,
{
    if array.is_empty() {
        return Err("Cannot compute histogram of an empty array");
    }

    if bins == 0 {
        return Err("Number of bins must be positive");
    }

    // Get range (min, max) of the data
    let (min_val, max_val) = match range {
        Some(r) => r,
        None => {
            let mut min_val = T::infinity();
            let mut max_val = T::neg_infinity();

            for &val in array.iter() {
                if val < min_val {
                    min_val = val;
                }
                if val > max_val {
                    max_val = val;
                }
            }
            (min_val, max_val)
        }
    };

    if min_val >= max_val {
        return Err("Range must be (min, max) with min < max");
    }

    // Create bin edges
    let mut bin_edges = Array::<T, Ix1>::zeros(bins + 1);
    let bin_width = (max_val - min_val) / T::from_usize(bins).unwrap();

    for i in 0..=bins {
        bin_edges[i] = min_val + bin_width * T::from_usize(i).unwrap();
    }

    // Ensure the last bin edge is exactly max_val
    bin_edges[bins] = max_val;

    // Initialize histogram array
    let mut hist = Array::<usize, Ix1>::zeros(bins);

    // Fill histogram
    match weights {
        Some(w) => {
            if w.len() != array.len() {
                return Err("Weights array must have the same length as the data array");
            }

            for (&val, &weight) in array.iter().zip(w.iter()) {
                // Skip values outside the range
                if val < min_val || val > max_val {
                    continue;
                }

                // Handle edge case where val == max_val (include in the last bin)
                if val == max_val {
                    hist[bins - 1] += 1;
                    continue;
                }

                // Find bin index
                let scaled_val = (val - min_val) / bin_width;
                let bin_idx = scaled_val.to_usize().unwrap_or(0);
                let bin_idx = bin_idx.min(bins - 1); // Ensure index is in bounds

                // Add to histogram (with weight)
                let weight_int = weight.to_usize().unwrap_or(1);
                hist[bin_idx] += weight_int;
            }
        }
        None => {
            for &val in array.iter() {
                // Skip values outside the range
                if val < min_val || val > max_val {
                    continue;
                }

                // Handle edge case where val == max_val (include in the last bin)
                if val == max_val {
                    hist[bins - 1] += 1;
                    continue;
                }

                // Find bin index
                let scaled_val = (val - min_val) / bin_width;
                let bin_idx = scaled_val.to_usize().unwrap_or(0);
                let bin_idx = bin_idx.min(bins - 1); // Ensure index is in bounds

                // Add to histogram
                hist[bin_idx] += 1;
            }
        }
    }

    Ok((hist, bin_edges))
}

/// Calculate a 2D histogram of data
///
/// # Arguments
///
/// * `x` - The x coordinates of the data points
/// * `y` - The y coordinates of the data points
/// * `bins` - Either a tuple (x_bins, y_bins) for the number of bins, or None for 10 bins in each direction
/// * `range` - Optional tuple ((x_min, x_max), (y_min, y_max)) to use. If None, the range is based on data
/// * `weights` - Optional array of weights for each data point
///
/// # Returns
///
/// A tuple containing (histogram, x_edges, y_edges)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::stats::histogram2d;
///
/// let x = array![0.1, 0.5, 1.3, 2.5, 3.1, 3.8, 4.2, 4.9];
/// let y = array![0.2, 0.8, 1.5, 2.0, 3.0, 3.2, 3.5, 4.5];
/// let (hist, x_edges, y_edges) = histogram2d(x.view(), y.view(), Some((4, 4)), None, None).unwrap();
///
/// assert_eq!(hist.shape(), &[4, 4]);
/// assert_eq!(x_edges.len(), 5);
/// assert_eq!(y_edges.len(), 5);
/// ```
pub fn histogram2d<T>(
    x: ArrayView<T, Ix1>,
    y: ArrayView<T, Ix1>,
    bins: Option<(usize, usize)>,
    range: Option<((T, T), (T, T))>,
    weights: Option<ArrayView<T, Ix1>>,
) -> Histogram2dResult<T>
where
    T: Clone + Float + FromPrimitive,
{
    if x.is_empty() || y.is_empty() {
        return Err("Cannot compute histogram of empty arrays");
    }

    if x.len() != y.len() {
        return Err("x and y arrays must have the same length");
    }

    // Default to 10 bins in each direction if not specified
    let (x_bins, y_bins) = bins.unwrap_or((10, 10));

    if x_bins == 0 || y_bins == 0 {
        return Err("Number of bins must be positive");
    }

    // Get range for x and y
    let ((x_min, x_max), (y_min, y_max)) = match range {
        Some(r) => r,
        None => {
            let mut x_min = T::infinity();
            let mut x_max = T::neg_infinity();
            let mut y_min = T::infinity();
            let mut y_max = T::neg_infinity();

            for (&x_val, &y_val) in x.iter().zip(y.iter()) {
                if x_val < x_min {
                    x_min = x_val;
                }
                if x_val > x_max {
                    x_max = x_val;
                }
                if y_val < y_min {
                    y_min = y_val;
                }
                if y_val > y_max {
                    y_max = y_val;
                }
            }
            ((x_min, x_max), (y_min, y_max))
        }
    };

    if x_min >= x_max || y_min >= y_max {
        return Err("Range must be (min, max) with min < max");
    }

    // Create bin edges
    let mut x_edges = Array::<T, Ix1>::zeros(x_bins + 1);
    let mut y_edges = Array::<T, Ix1>::zeros(y_bins + 1);

    let x_bin_width = (x_max - x_min) / T::from_usize(x_bins).unwrap();
    let y_bin_width = (y_max - y_min) / T::from_usize(y_bins).unwrap();

    for i in 0..=x_bins {
        x_edges[i] = x_min + x_bin_width * T::from_usize(i).unwrap();
    }

    for i in 0..=y_bins {
        y_edges[i] = y_min + y_bin_width * T::from_usize(i).unwrap();
    }

    // Ensure the last bin edges are exactly max values
    x_edges[x_bins] = x_max;
    y_edges[y_bins] = y_max;

    // Initialize histogram array
    let mut hist = Array::<usize, Ix2>::zeros((y_bins, x_bins));

    // Fill histogram
    match weights {
        Some(w) => {
            if w.len() != x.len() {
                return Err("Weights array must have the same length as the data arrays");
            }

            for ((&x_val, &y_val), &weight) in x.iter().zip(y.iter()).zip(w.iter()) {
                // Skip values outside the range
                if x_val < x_min || x_val > x_max || y_val < y_min || y_val > y_max {
                    continue;
                }

                // Find bin indices
                let x_scaled = (x_val - x_min) / x_bin_width;
                let y_scaled = (y_val - y_min) / y_bin_width;

                let mut x_idx = x_scaled.to_usize().unwrap_or(0);
                let mut y_idx = y_scaled.to_usize().unwrap_or(0);

                // Handle edge cases where val == max_val
                if x_val == x_max {
                    x_idx = x_bins - 1;
                } else {
                    x_idx = x_idx.min(x_bins - 1);
                }

                if y_val == y_max {
                    y_idx = y_bins - 1;
                } else {
                    y_idx = y_idx.min(y_bins - 1);
                }

                // Add to histogram (with weight)
                let weight_int = weight.to_usize().unwrap_or(1);
                hist[[y_idx, x_idx]] += weight_int;
            }
        }
        None => {
            for (&x_val, &y_val) in x.iter().zip(y.iter()) {
                // Skip values outside the range
                if x_val < x_min || x_val > x_max || y_val < y_min || y_val > y_max {
                    continue;
                }

                // Find bin indices
                let x_scaled = (x_val - x_min) / x_bin_width;
                let y_scaled = (y_val - y_min) / y_bin_width;

                let mut x_idx = x_scaled.to_usize().unwrap_or(0);
                let mut y_idx = y_scaled.to_usize().unwrap_or(0);

                // Handle edge cases where val == max_val
                if x_val == x_max {
                    x_idx = x_bins - 1;
                } else {
                    x_idx = x_idx.min(x_bins - 1);
                }

                if y_val == y_max {
                    y_idx = y_bins - 1;
                } else {
                    y_idx = y_idx.min(y_bins - 1);
                }

                // Add to histogram
                hist[[y_idx, x_idx]] += 1;
            }
        }
    }

    Ok((hist, x_edges, y_edges))
}

/// Calculate the quantile values from a 1D array
///
/// # Arguments
///
/// * `array` - The input 1D array
/// * `q` - The quantile or array of quantiles to compute (between 0 and 1)
/// * `method` - The interpolation method to use: "linear" (default), "lower", "higher", "midpoint", or "nearest"
///
/// # Returns
///
/// An array containing the quantile values
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::stats::quantile;
///
/// let data = array![1.0, 3.0, 5.0, 7.0, 9.0];
///
/// // Median (50th percentile)
/// let median = quantile(data.view(), array![0.5].view(), Some("linear")).unwrap();
/// assert_eq!(median[0], 5.0);
///
/// // Multiple quantiles
/// let quartiles = quantile(data.view(), array![0.25, 0.5, 0.75].view(), None).unwrap();
/// assert_eq!(quartiles[0], 3.0);  // 25th percentile
/// assert_eq!(quartiles[1], 5.0);  // 50th percentile
/// assert_eq!(quartiles[2], 7.0);  // 75th percentile
/// ```
///
/// This function is equivalent to NumPy's `np.quantile` function.
pub fn quantile<T>(
    array: ArrayView<T, Ix1>,
    q: ArrayView<T, Ix1>,
    method: Option<&str>,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float + FromPrimitive,
{
    if array.is_empty() {
        return Err("Cannot compute quantile of an empty array");
    }

    // Validate q values
    for &val in q.iter() {
        if val < T::from_f64(0.0).unwrap() || val > T::from_f64(1.0).unwrap() {
            return Err("Quantile values must be between 0 and 1");
        }
    }

    // Clone and sort the array
    let mut sorted: Vec<T> = array.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    let mut result = Array::<T, Ix1>::zeros(q.len());

    // The interpolation method to use
    let method = method.unwrap_or("linear");

    for (i, &q_val) in q.iter().enumerate() {
        if q_val == T::from_f64(0.0).unwrap() {
            result[i] = sorted[0];
            continue;
        }

        if q_val == T::from_f64(1.0).unwrap() {
            result[i] = sorted[n - 1];
            continue;
        }

        // Calculate the position in the sorted array
        let h = T::from_usize(n - 1).unwrap() * q_val;
        let h_floor = h.floor();
        let idx_low = h_floor.to_usize().unwrap_or(0).min(n - 1);
        let idx_high = (idx_low + 1).min(n - 1);

        match method {
            "linear" => {
                let weight = h - h_floor;
                result[i] = sorted[idx_low] * (T::from_f64(1.0).unwrap() - weight) + sorted[idx_high] * weight;
            }
            "lower" => {
                result[i] = sorted[idx_low];
            }
            "higher" => {
                result[i] = sorted[idx_high];
            }
            "midpoint" => {
                result[i] = (sorted[idx_low] + sorted[idx_high]) / T::from_f64(2.0).unwrap();
            }
            "nearest" => {
                let weight = h - h_floor;
                if weight < T::from_f64(0.5).unwrap() {
                    result[i] = sorted[idx_low];
                } else {
                    result[i] = sorted[idx_high];
                }
            }
            _ => return Err("Invalid interpolation method. Use 'linear', 'lower', 'higher', 'midpoint', or 'nearest'"),
        }
    }

    Ok(result)
}

/// Count number of occurrences of each value in array of non-negative ints.
///
/// # Arguments
///
/// * `array` - Input array of non-negative integers
/// * `minlength` - Minimum number of bins for the output array. If None, the output array length is determined by the maximum value in `array`.
/// * `weights` - Optional weights array. If specified, must have same shape as `array`.
///
/// # Returns
///
/// An array of counts
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::stats::bincount;
///
/// let data = array![1, 2, 3, 1, 2, 1, 0, 1, 3, 2];
/// let counts = bincount(data.view(), None, None).unwrap();
/// assert_eq!(counts.len(), 4);
/// assert_eq!(counts[0], 1.0); // '0' occurs once
/// assert_eq!(counts[1], 4.0); // '1' occurs four times
/// assert_eq!(counts[2], 3.0); // '2' occurs three times
/// assert_eq!(counts[3], 2.0); // '3' occurs twice
/// ```
///
/// This function is equivalent to NumPy's `np.bincount` function.
pub fn bincount(
    array: ArrayView<usize, Ix1>,
    minlength: Option<usize>,
    weights: Option<ArrayView<f64, Ix1>>,
) -> Result<Array<f64, Ix1>, &'static str> {
    if array.is_empty() {
        return Err("Cannot compute bincount of an empty array");
    }

    // Find maximum value to determine number of bins
    let mut max_val = 0;
    for &val in array.iter() {
        if val > max_val {
            max_val = val;
        }
    }

    // Determine length of output array
    let length = if let Some(min_len) = minlength {
        max_val.max(min_len - 1) + 1
    } else {
        max_val + 1
    };

    let mut result = Array::<f64, Ix1>::zeros(length);

    match weights {
        Some(w) => {
            if w.len() != array.len() {
                return Err("Weights array must have same length as input array");
            }
            for (&idx, &weight) in array.iter().zip(w.iter()) {
                result[idx] += weight;
            }
        }
        None => {
            for &idx in array.iter() {
                result[idx] += 1.0;
            }
        }
    }

    Ok(result)
}

/// Calculate correlation coefficient between two 1D arrays
///
/// # Arguments
///
/// * `x` - First input array
/// * `y` - Second input array (must have same shape as x)
///
/// # Returns
///
/// Pearson's correlation coefficient
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::stats::corrcoef;
///
/// let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = array![5.0, 4.0, 3.0, 2.0, 1.0];
/// let corr = corrcoef(x.view(), y.view()).unwrap();
/// assert!((corr + 1.0_f64).abs() < 1e-10); // Perfect negative correlation (-1.0)
/// ```
///
/// This function is similar to NumPy's `np.corrcoef` function but returns a single value.
pub fn corrcoef<T>(x: ArrayView<T, Ix1>, y: ArrayView<T, Ix1>) -> Result<T, &'static str>
where
    T: Clone + Float + FromPrimitive,
{
    if x.is_empty() || y.is_empty() {
        return Err("Cannot compute correlation of empty arrays");
    }

    if x.len() != y.len() {
        return Err("Arrays must have the same length");
    }

    // Calculate means
    let n = T::from_usize(x.len()).unwrap();
    let mut sum_x = T::zero();
    let mut sum_y = T::zero();

    for (&x_val, &y_val) in x.iter().zip(y.iter()) {
        sum_x = sum_x + x_val;
        sum_y = sum_y + y_val;
    }

    let mean_x = sum_x / n;
    let mean_y = sum_y / n;

    // Calculate covariance and variances
    let mut cov_xy = T::zero();
    let mut var_x = T::zero();
    let mut var_y = T::zero();

    for (&x_val, &y_val) in x.iter().zip(y.iter()) {
        let dx = x_val - mean_x;
        let dy = y_val - mean_y;
        cov_xy = cov_xy + dx * dy;
        var_x = var_x + dx * dx;
        var_y = var_y + dy * dy;
    }

    // Calculate correlation coefficient
    if var_x.is_zero() || var_y.is_zero() {
        return Err("Correlation coefficient is not defined when either array has zero variance");
    }

    Ok(cov_xy / (var_x * var_y).sqrt())
}

/// Calculate the covariance matrix of a 2D array
///
/// # Arguments
///
/// * `array` - The input 2D array where rows are observations and columns are variables
/// * `ddof` - Delta degrees of freedom (default 1)
///
/// # Returns
///
/// The covariance matrix
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::stats::cov;
///
/// let data = array![
///     [1.0, 2.0, 3.0],
///     [4.0, 5.0, 6.0],
///     [7.0, 8.0, 9.0],
///     [10.0, 11.0, 12.0]
/// ];
/// let cov_matrix = cov(data.view(), 1).unwrap();
/// assert_eq!(cov_matrix.shape(), &[3, 3]);
/// ```
///
/// This function is equivalent to NumPy's `np.cov` function.
pub fn cov<T>(array: ArrayView<T, Ix2>, ddof: usize) -> Result<Array<T, Ix2>, &'static str>
where
    T: Clone + Float + FromPrimitive,
{
    if array.is_empty() {
        return Err("Cannot compute covariance of an empty array");
    }

    let (n_samples, n_features) = (array.shape()[0], array.shape()[1]);

    if n_samples <= ddof {
        return Err("Not enough data points for covariance calculation with given ddof");
    }

    // Calculate means for each feature
    let mut feature_means = Array::<T, Ix1>::zeros(n_features);

    for j in 0..n_features {
        let mut sum = T::zero();
        for i in 0..n_samples {
            sum = sum + array[[i, j]];
        }
        feature_means[j] = sum / T::from_usize(n_samples).unwrap();
    }

    // Calculate covariance matrix
    let mut cov_matrix = Array::<T, Ix2>::zeros((n_features, n_features));
    let scale = T::from_usize(n_samples - ddof).unwrap();

    for i in 0..n_features {
        for j in 0..=i {
            let mut cov_ij = T::zero();

            for k in 0..n_samples {
                let dev_i = array[[k, i]] - feature_means[i];
                let dev_j = array[[k, j]] - feature_means[j];
                cov_ij = cov_ij + dev_i * dev_j;
            }

            cov_ij = cov_ij / scale;
            cov_matrix[[i, j]] = cov_ij;

            // Fill symmetric part
            if i != j {
                cov_matrix[[j, i]] = cov_ij;
            }
        }
    }

    Ok(cov_matrix)
}

/// Bin values into discrete intervals
///
/// # Arguments
///
/// * `array` - Input array to be binned
/// * `bins` - Array of bin edges
/// * `right` - If true, intervals include the right bin edge, otherwise the left
/// * `result_type` - 'indices' to return bin indices or 'values' to return bin values
///
/// # Returns
///
/// Array of indices or bin values
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::stats::digitize;
///
/// let x = array![0.2, 1.4, 2.5, 6.2, 9.7, 2.1];
/// let bins = array![0.0, 1.0, 2.5, 4.0, 10.0];
/// let indices = digitize(x.view(), bins.view(), true, "indices").unwrap();
/// assert_eq!(indices, array![1, 2, 2, 4, 4, 2]);
/// ```
///
/// This function is equivalent to NumPy's `np.digitize` function.
pub fn digitize<T>(
    array: ArrayView<T, Ix1>,
    bins: ArrayView<T, Ix1>,
    right: bool,
    result_type: &str,
) -> Result<Array<usize, Ix1>, &'static str>
where
    T: Clone + Float + FromPrimitive,
{
    if array.is_empty() {
        return Err("Cannot digitize an empty array");
    }

    if bins.is_empty() {
        return Err("Bins array cannot be empty");
    }

    // Check that bins are monotonically increasing
    for i in 1..bins.len() {
        if bins[i] <= bins[i - 1] {
            return Err("Bins must be monotonically increasing");
        }
    }

    let mut result = Array::<usize, Ix1>::zeros(array.len());

    for (i, &val) in array.iter().enumerate() {
        let mut bin_idx = 0;

        if right {
            // Right inclusive: val <= edge
            for j in 0..bins.len() {
                if val <= bins[j] {
                    bin_idx = j;
                    break;
                }
                bin_idx = bins.len();
            }
        } else {
            // Left inclusive: val < edge
            for j in 0..bins.len() {
                if val < bins[j] {
                    bin_idx = j;
                    break;
                }
                bin_idx = bins.len();
            }
        }

        result[i] = bin_idx;
    }

    if result_type == "indices" {
        Ok(result)
    } else if result_type == "values" {
        Err("'values' result_type is not yet implemented")
    } else {
        Err("result_type must be 'indices' or 'values'")
    }
}

/// Calculate the mean of array elements
///
/// # Arguments
///
/// * `array` - The input array (1D or 2D)
/// * `axis` - Option to calculate along an axis (None for global mean)
///
/// # Returns
///
/// The mean of the array elements
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::stats::mean;
///
/// // 1D array example
/// let a1d = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let mean_1d = mean(&a1d.view(), None).unwrap();
/// assert_eq!(mean_1d[0], 3.0);
///
/// // 2D array example
/// let a2d = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
/// let mean_global = mean(&a2d.view(), None).unwrap();
/// assert_eq!(mean_global[0], 3.5);
/// ```
pub fn mean<T, D>(
    array: &ArrayView<T, D>,
    axis: Option<Axis>,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float + FromPrimitive,
    D: Dimension,
{
    match array.ndim() {
        1 => {
            // Handle 1D arrays
            if array.is_empty() {
                return Err("Cannot compute mean of an empty array");
            }

            let total_elements = array.len();
            let mut sum = T::zero();

            for &val in array.iter() {
                sum = sum + val;
            }

            let count = T::from_usize(total_elements).ok_or("Cannot convert array length to T")?;
            Ok(Array::from_elem(1, sum / count))
        }
        2 => {
            // Call the 2D implementation for 2D arrays
            let array_2d = array
                .clone()
                .into_dimensionality::<Ix2>()
                .map_err(|_| "Failed to convert array to 2D")?;
            mean_2d(&array_2d.view(), axis)
        }
        _ => Err("Array dimension not supported, must be 1D or 2D"),
    }
}

/// Calculate the median of array elements
///
/// # Arguments
///
/// * `array` - The input array (1D or 2D)
/// * `axis` - Option to calculate along an axis (None for global median)
///
/// # Returns
///
/// The median of the array elements
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::stats::median;
///
/// // 1D array example
/// let a1d = array![1.0, 3.0, 5.0, 7.0, 9.0];
/// let median_1d = median(&a1d.view(), None).unwrap();
/// assert_eq!(median_1d[0], 5.0);
///
/// // 2D array example
/// let a2d = array![[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]];
/// let median_global = median(&a2d.view(), None).unwrap();
/// assert_eq!(median_global[0], 3.5);
/// ```
pub fn median<T, D>(
    array: &ArrayView<T, D>,
    axis: Option<Axis>,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float + FromPrimitive,
    D: Dimension,
{
    match array.ndim() {
        1 => {
            // Handle 1D arrays
            if array.is_empty() {
                return Err("Cannot compute median of an empty array");
            }

            let mut values: Vec<_> = array.iter().copied().collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let median_value = if values.len() % 2 == 0 {
                let mid = values.len() / 2;
                (values[mid - 1] + values[mid]) / T::from_f64(2.0).unwrap()
            } else {
                values[values.len() / 2]
            };

            Ok(Array::from_elem(1, median_value))
        }
        2 => {
            // Call the 2D implementation for 2D arrays
            let array_2d = array
                .clone()
                .into_dimensionality::<Ix2>()
                .map_err(|_| "Failed to convert array to 2D")?;
            median_2d(&array_2d.view(), axis)
        }
        _ => Err("Array dimension not supported, must be 1D or 2D"),
    }
}

/// Calculate the variance of array elements
///
/// # Arguments
///
/// * `array` - The input array (1D or 2D)
/// * `axis` - Option to calculate along an axis (None for global variance)
/// * `ddof` - Delta degrees of freedom (default 0)
///
/// # Returns
///
/// The variance of the array elements
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::stats::variance;
///
/// // 1D array example
/// let a1d = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let var_1d = variance(&a1d.view(), None, 1).unwrap();
/// assert!((var_1d[0] - 2.5_f64).abs() < 1e-10);
///
/// // 2D array example
/// let a2d = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
/// let var_global = variance(&a2d.view(), None, 1).unwrap();
/// assert!((var_global[0] - 3.5_f64).abs() < 1e-10);
/// ```
pub fn variance<T, D>(
    array: &ArrayView<T, D>,
    axis: Option<Axis>,
    ddof: usize,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float + FromPrimitive,
    D: Dimension,
{
    match array.ndim() {
        1 => {
            // Handle 1D arrays
            if array.is_empty() {
                return Err("Cannot compute variance of an empty array");
            }

            let total_elements = array.len();

            if total_elements <= ddof {
                return Err("Not enough data points for variance calculation with given ddof");
            }

            // Calculate mean
            let mean_val = mean(array, None)?[0];

            // Calculate sum of squared differences from the mean
            let mut sum_sq_diff = T::zero();
            for &val in array.iter() {
                let diff = val - mean_val;
                sum_sq_diff = sum_sq_diff + (diff * diff);
            }

            let divisor = T::from_usize(total_elements - ddof).unwrap();

            Ok(Array::from_elem(1, sum_sq_diff / divisor))
        }
        2 => {
            // Call the 2D implementation for 2D arrays
            let array_2d = array
                .clone()
                .into_dimensionality::<Ix2>()
                .map_err(|_| "Failed to convert array to 2D")?;
            variance_2d(&array_2d.view(), axis, ddof)
        }
        _ => Err("Array dimension not supported, must be 1D or 2D"),
    }
}

/// Calculate the standard deviation of array elements
///
/// # Arguments
///
/// * `array` - The input array (1D or 2D)
/// * `axis` - Option to calculate along an axis (None for global std dev)
/// * `ddof` - Delta degrees of freedom (default 0)
///
/// # Returns
///
/// The standard deviation of the array elements
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::stats::std_dev;
///
/// // 1D array example
/// let a1d = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let std_1d = std_dev(&a1d.view(), None, 1).unwrap();
/// assert!((std_1d[0] - 1.5811388300841898_f64).abs() < 1e-10);
///
/// // 2D array example
/// let a2d = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
/// let std_global = std_dev(&a2d.view(), None, 1).unwrap();
/// assert!((std_global[0] - 1.8708286933869707_f64).abs() < 1e-10);
/// ```
pub fn std_dev<T, D>(
    array: &ArrayView<T, D>,
    axis: Option<Axis>,
    ddof: usize,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float + FromPrimitive,
    D: Dimension,
{
    let var_result = variance(array, axis, ddof)?;
    Ok(var_result.mapv(|x| x.sqrt()))
}

/// Calculate the minimum value of array elements
///
/// # Arguments
///
/// * `array` - The input array (1D or 2D)
/// * `axis` - Option to calculate along an axis (None for global minimum)
///
/// # Returns
///
/// The minimum value of the array elements
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::stats::min;
///
/// // 1D array example
/// let a1d = array![3.0, 1.0, 4.0, 1.0, 5.0];
/// let min_1d = min(&a1d.view(), None).unwrap();
/// assert_eq!(min_1d[0], 1.0);
///
/// // 2D array example
/// let a2d = array![[1.0, 2.0, 3.0], [4.0, 5.0, 0.5]];
/// let min_global = min(&a2d.view(), None).unwrap();
/// assert_eq!(min_global[0], 0.5);
/// ```
pub fn min<T, D>(array: &ArrayView<T, D>, axis: Option<Axis>) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float,
    D: Dimension,
{
    match array.ndim() {
        1 => {
            // Handle 1D arrays
            if array.is_empty() {
                return Err("Cannot compute minimum of an empty array");
            }

            let mut min_val = T::infinity();
            for &val in array.iter() {
                if val < min_val {
                    min_val = val;
                }
            }

            Ok(Array::from_elem(1, min_val))
        }
        2 => {
            // Call the 2D implementation for 2D arrays
            let array_2d = array
                .clone()
                .into_dimensionality::<Ix2>()
                .map_err(|_| "Failed to convert array to 2D")?;
            min_2d(&array_2d.view(), axis)
        }
        _ => Err("Array dimension not supported, must be 1D or 2D"),
    }
}

/// Calculate the maximum value of array elements
///
/// # Arguments
///
/// * `array` - The input array (1D or 2D)
/// * `axis` - Option to calculate along an axis (None for global maximum)
///
/// # Returns
///
/// The maximum value of the array elements
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::stats::max;
///
/// // 1D array example
/// let a1d = array![3.0, 1.0, 4.0, 1.0, 5.0];
/// let max_1d = max(&a1d.view(), None).unwrap();
/// assert_eq!(max_1d[0], 5.0);
///
/// // 2D array example
/// let a2d = array![[1.0, 2.0, 3.0], [4.0, 5.0, 0.5]];
/// let max_global = max(&a2d.view(), None).unwrap();
/// assert_eq!(max_global[0], 5.0);
/// ```
pub fn max<T, D>(array: &ArrayView<T, D>, axis: Option<Axis>) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float,
    D: Dimension,
{
    match array.ndim() {
        1 => {
            // Handle 1D arrays
            if array.is_empty() {
                return Err("Cannot compute maximum of an empty array");
            }

            let mut max_val = T::neg_infinity();
            for &val in array.iter() {
                if val > max_val {
                    max_val = val;
                }
            }

            Ok(Array::from_elem(1, max_val))
        }
        2 => {
            // Call the 2D implementation for 2D arrays
            let array_2d = array
                .clone()
                .into_dimensionality::<Ix2>()
                .map_err(|_| "Failed to convert array to 2D")?;
            max_2d(&array_2d.view(), axis)
        }
        _ => Err("Array dimension not supported, must be 1D or 2D"),
    }
}

/// Calculate the percentile of array elements
///
/// # Arguments
///
/// * `array` - The input array (1D or 2D)
/// * `q` - Percentile to compute, which must be between 0 and 100 inclusive
/// * `axis` - Option to calculate along an axis (None for global percentile)
///
/// # Returns
///
/// The percentile of the array elements
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::stats::percentile;
///
/// // 1D array example
/// let a1d = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let median_1d = percentile(&a1d.view(), 50.0, None).unwrap();
/// assert_eq!(median_1d[0], 3.0);
///
/// // 2D array example
/// let a2d = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
/// let median_global = percentile(&a2d.view(), 50.0, None).unwrap();
/// assert_eq!(median_global[0], 3.5);
/// ```
pub fn percentile<T, D>(
    array: &ArrayView<T, D>,
    q: f64,
    axis: Option<Axis>,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Float + FromPrimitive,
    D: Dimension,
{
    if !(0.0..=100.0).contains(&q) {
        return Err("Percentile must be between 0 and 100");
    }

    match array.ndim() {
        1 => {
            // Handle 1D arrays
            if array.is_empty() {
                return Err("Cannot compute percentile of an empty array");
            }

            // Convert percentile to a fraction
            let q_frac = q / 100.0;

            // Clone and sort the array
            let mut values: Vec<T> = array.iter().copied().collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let percentile_value = compute_percentile(&values, q_frac)?;
            Ok(Array::from_elem(1, percentile_value))
        }
        2 => {
            // Call the 2D implementation for 2D arrays
            let array_2d = array
                .clone()
                .into_dimensionality::<Ix2>()
                .map_err(|_| "Failed to convert array to 2D")?;
            percentile_2d(&array_2d.view(), q, axis)
        }
        _ => Err("Array dimension not supported, must be 1D or 2D"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_mean_2d() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // Global mean
        let global_mean = mean_2d(&a.view(), None).unwrap();
        assert_eq!(global_mean[0], 3.5);

        // Mean along axis 0 (columns)
        let col_means = mean_2d(&a.view(), Some(Axis(0))).unwrap();
        assert_eq!(col_means.len(), 3);
        assert_eq!(col_means[0], 2.5);
        assert_eq!(col_means[1], 3.5);
        assert_eq!(col_means[2], 4.5);

        // Mean along axis 1 (rows)
        let row_means = mean_2d(&a.view(), Some(Axis(1))).unwrap();
        assert_eq!(row_means.len(), 2);
        assert_eq!(row_means[0], 2.0);
        assert_eq!(row_means[1], 5.0);
    }

    #[test]
    fn test_median_2d() {
        let a = array![[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]];

        // Global median
        let global_median = median_2d(&a.view(), None).unwrap();
        assert_eq!(global_median[0], 3.5);

        // Even number of elements
        let a_even = array![[1.0, 2.0, 3.0, 4.0]];
        let median_even = median_2d(&a_even.view(), None).unwrap();
        assert_eq!(median_even[0], 2.5);

        // Odd number of elements
        let a_odd = array![[1.0, 2.0, 3.0, 4.0, 5.0]];
        let median_odd = median_2d(&a_odd.view(), None).unwrap();
        assert_eq!(median_odd[0], 3.0);

        // Median along axis 0 (columns)
        let col_medians = median_2d(&a.view(), Some(Axis(0))).unwrap();
        assert_eq!(col_medians.len(), 3);
        assert_eq!(col_medians[0], 1.5);
        assert_eq!(col_medians[1], 3.5);
        assert_eq!(col_medians[2], 5.5);
    }

    #[test]
    fn test_variance_std_dev() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // Global variance (population)
        let global_var_pop = variance_2d(&a.view(), None, 0).unwrap();
        assert!((global_var_pop[0] - 2.9166666666666665_f64).abs() < 1e-10);

        // Global variance (sample)
        let global_var_sample = variance_2d(&a.view(), None, 1).unwrap();
        assert!((global_var_sample[0] - 3.5_f64).abs() < 1e-10);

        // Standard deviation (sample)
        let global_std = std_dev_2d(&a.view(), None, 1).unwrap();
        assert!((global_std[0] - 1.8708286933869707_f64).abs() < 1e-10);

        // Variance along axis 0 (columns, sample)
        let col_vars = variance_2d(&a.view(), Some(Axis(0)), 1).unwrap();
        assert_eq!(col_vars.len(), 3);
        assert!((col_vars[0] - 4.5_f64).abs() < 1e-10);
        assert!((col_vars[1] - 4.5_f64).abs() < 1e-10);
        assert!((col_vars[2] - 4.5_f64).abs() < 1e-10);
    }

    #[test]
    fn test_min_max() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 0.5]];

        // Global min/max
        let global_min = min_2d(&a.view(), None).unwrap();
        assert_eq!(global_min[0], 0.5);

        let global_max = max_2d(&a.view(), None).unwrap();
        assert_eq!(global_max[0], 5.0);

        // Min/max along axis 0 (columns)
        let col_mins = min_2d(&a.view(), Some(Axis(0))).unwrap();
        assert_eq!(col_mins.len(), 3);
        assert_eq!(col_mins[0], 1.0);
        assert_eq!(col_mins[1], 2.0);
        assert_eq!(col_mins[2], 0.5);

        let col_maxs = max_2d(&a.view(), Some(Axis(0))).unwrap();
        assert_eq!(col_maxs.len(), 3);
        assert_eq!(col_maxs[0], 4.0);
        assert_eq!(col_maxs[1], 5.0);
        assert_eq!(col_maxs[2], 3.0);
    }

    #[test]
    fn test_sum_product() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // Global sum/product
        let global_sum = sum_2d(&a.view(), None).unwrap();
        assert_eq!(global_sum[0], 21.0);

        let global_prod = product_2d(&a.view(), None).unwrap();
        assert_eq!(global_prod[0], 720.0);

        // Sum/product along axis 0 (columns)
        let col_sums = sum_2d(&a.view(), Some(Axis(0))).unwrap();
        assert_eq!(col_sums.len(), 3);
        assert_eq!(col_sums[0], 5.0);
        assert_eq!(col_sums[1], 7.0);
        assert_eq!(col_sums[2], 9.0);

        let col_prods = product_2d(&a.view(), Some(Axis(0))).unwrap();
        assert_eq!(col_prods.len(), 3);
        assert_eq!(col_prods[0], 4.0);
        assert_eq!(col_prods[1], 10.0);
        assert_eq!(col_prods[2], 18.0);
    }

    #[test]
    fn test_percentile() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // Global median (50th percentile)
        let global_median = percentile_2d(&a.view(), 50.0, None).unwrap();
        assert_eq!(global_median[0], 3.5);

        // 25th percentile
        let global_q25 = percentile_2d(&a.view(), 25.0, None).unwrap();
        assert_eq!(global_q25[0], 2.25);

        // 75th percentile
        let global_q75 = percentile_2d(&a.view(), 75.0, None).unwrap();
        assert_eq!(global_q75[0], 4.75);

        // Percentiles along axis 0 (columns)
        let col_q50 = percentile_2d(&a.view(), 50.0, Some(Axis(0))).unwrap();
        assert_eq!(col_q50.len(), 3);
        assert_eq!(col_q50[0], 2.5);
        assert_eq!(col_q50[1], 3.5);
        assert_eq!(col_q50[2], 4.5);
    }

    #[test]
    fn test_histogram() {
        let data = array![1.0, 1.0, 2.0, 2.0, 2.5, 3.0, 3.0, 4.0, 4.5, 5.0];
        let (hist, bin_edges) = histogram(data.view(), 5, None, None).unwrap();

        assert_eq!(hist.len(), 5);
        assert_eq!(bin_edges.len(), 6);
        assert_eq!(hist[0], 2); // Two values in the first bin
        assert_eq!(hist[1], 3); // Three values in the second bin

        // Test with specified range
        let (hist, bin_edges) = histogram(data.view(), 4, Some((0.0, 8.0)), None).unwrap();
        assert_eq!(hist.len(), 4);
        assert_eq!(bin_edges.len(), 5);
        assert_eq!(bin_edges[0], 0.0);
        assert_eq!(bin_edges[4], 8.0);

        // Test invalid parameters
        assert!(histogram(data.view(), 0, None, None).is_err()); // Zero bins
        let empty_arr: Array<f64, Ix1> = array![];
        assert!(histogram(empty_arr.view(), 5, None, None).is_err()); // Empty array
    }

    #[test]
    fn test_histogram2d() {
        let x = array![0.1, 0.5, 1.3, 2.5, 3.1, 3.8, 4.2, 4.9];
        let y = array![0.2, 0.8, 1.5, 2.0, 3.0, 3.2, 3.5, 4.5];

        let (hist, x_edges, y_edges) =
            histogram2d(x.view(), y.view(), Some((4, 4)), None, None).unwrap();

        assert_eq!(hist.shape(), &[4, 4]);
        assert_eq!(x_edges.len(), 5);
        assert_eq!(y_edges.len(), 5);

        // Test with specified range
        let range = ((0.0, 5.0), (0.0, 5.0));
        let (hist, x_edges, y_edges) =
            histogram2d(x.view(), y.view(), Some((5, 5)), Some(range), None).unwrap();

        assert_eq!(hist.shape(), &[5, 5]);
        assert_eq!(x_edges[0], 0.0);
        assert_eq!(x_edges[5], 5.0);
        assert_eq!(y_edges[0], 0.0);
        assert_eq!(y_edges[5], 5.0);

        // Test invalid parameters
        let short_y = array![0.2, 0.8, 1.5];
        assert!(histogram2d(x.view(), short_y.view(), None, None, None).is_err()); // Mismatched lengths
        assert!(histogram2d(x.view(), y.view(), Some((0, 5)), None, None).is_err());
        // Zero bins
    }

    #[test]
    fn test_quantile() {
        let data = array![1.0, 3.0, 5.0, 7.0, 9.0];

        // Test median (50th percentile)
        let median = quantile(data.view(), array![0.5].view(), Some("linear")).unwrap();
        assert_eq!(median[0], 5.0);

        // Test quartiles
        let quartiles = quantile(data.view(), array![0.25, 0.5, 0.75].view(), None).unwrap();
        assert_eq!(quartiles[0], 3.0); // 25th percentile
        assert_eq!(quartiles[1], 5.0); // 50th percentile
        assert_eq!(quartiles[2], 7.0); // 75th percentile

        // Test different interpolation methods
        let methods = ["linear", "lower", "higher", "midpoint", "nearest"];
        for method in methods {
            let q = quantile(data.view(), array![0.5].view(), Some(method)).unwrap();
            assert!(q[0] >= 3.0 && q[0] <= 7.0);
        }

        // Test edge cases
        let q_edge = quantile(data.view(), array![0.0, 1.0].view(), None).unwrap();
        assert_eq!(q_edge[0], 1.0); // Minimum
        assert_eq!(q_edge[1], 9.0); // Maximum

        // Test invalid parameters
        let empty_arr: Array<f64, Ix1> = array![];
        assert!(quantile(empty_arr.view(), array![0.5].view(), None).is_err()); // Empty array
        assert!(quantile(data.view(), array![-0.1].view(), None).is_err()); // q < 0
        assert!(quantile(data.view(), array![1.1].view(), None).is_err()); // q > 1
        assert!(quantile(data.view(), array![0.5].view(), Some("invalid")).is_err());
        // Invalid method
    }

    #[test]
    fn test_bincount() {
        let data = array![1, 2, 3, 1, 2, 1, 0, 1, 3, 2];

        // Test with default parameters
        let counts = bincount(data.view(), None, None).unwrap();
        assert_eq!(counts.len(), 4);
        assert_eq!(counts[0], 1.0); // '0' occurs once
        assert_eq!(counts[1], 4.0); // '1' occurs four times
        assert_eq!(counts[2], 3.0); // '2' occurs three times
        assert_eq!(counts[3], 2.0); // '3' occurs twice

        // Test with minlength parameter
        let counts_min = bincount(data.view(), Some(6), None).unwrap();
        assert_eq!(counts_min.len(), 6);
        assert_eq!(counts_min[0], 1.0);
        assert_eq!(counts_min[1], 4.0);
        assert_eq!(counts_min[2], 3.0);
        assert_eq!(counts_min[3], 2.0);
        assert_eq!(counts_min[4], 0.0);
        assert_eq!(counts_min[5], 0.0);

        // Test with weights
        let weights = array![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
        let weighted = bincount(data.view(), None, Some(weights.view())).unwrap();
        assert_eq!(weighted.len(), 4);
        assert_eq!(weighted[0], 3.5); // only the 7th element (weight 3.5)
        assert_eq!(weighted[1], 9.5); // sum of weights: 0.5 + 2.0 + 3.0 + 4.0 = 9.5
        assert_eq!(weighted[2], 8.5); // sum of weights: 1.0 + 2.5 + 5.0 = 8.5
        assert_eq!(weighted[3], 6.0); // sum of weights: 1.5 + 4.5 = 6.0

        // Test with empty array
        let empty_arr: Array<usize, Ix1> = array![];
        assert!(bincount(empty_arr.view(), None, None).is_err());

        // Test with mismatched weights length
        let short_weights = array![1.0, 2.0];
        assert!(bincount(data.view(), None, Some(short_weights.view())).is_err());
    }

    #[test]
    fn test_corrcoef() {
        // Test perfect positive correlation
        let x1 = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y1 = array![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr1 = corrcoef(x1.view(), y1.view()).unwrap();
        assert!((corr1 - 1.0_f64).abs() < 1e-10);

        // Test perfect negative correlation
        let x2 = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y2 = array![5.0, 4.0, 3.0, 2.0, 1.0];
        let corr2 = corrcoef(x2.view(), y2.view()).unwrap();
        assert!((corr2 + 1.0_f64).abs() < 1e-10);

        // Test no correlation
        let x3 = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y3 = array![5.0, 2.0, 8.0, 1.0, 4.0];
        let corr3 = corrcoef(x3.view(), y3.view()).unwrap();
        assert!(corr3.abs() < 0.5); // Not strongly correlated

        // Test invalid inputs
        let empty_arr: Array<f64, Ix1> = array![];
        assert!(corrcoef(empty_arr.view(), array![1.0].view()).is_err()); // Empty array
        assert!(corrcoef(array![1.0].view(), array![1.0, 2.0].view()).is_err()); // Mismatched lengths

        // Test zero variance case
        let x4 = array![3.0, 3.0, 3.0];
        let y4 = array![1.0, 2.0, 3.0];
        assert!(corrcoef(x4.view(), y4.view()).is_err()); // Zero variance in x
    }

    #[test]
    fn test_cov() {
        // Test covariance calculation
        let data = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ];

        let cov_matrix = cov(data.view(), 1).unwrap();
        assert_eq!(cov_matrix.shape(), &[3, 3]);

        // Expected values - diagonals should contain variances
        assert!((cov_matrix[[0, 0]] - 15.0_f64).abs() < 1e-10); // Variance of first column
        assert!((cov_matrix[[1, 1]] - 15.0_f64).abs() < 1e-10); // Variance of second column
        assert!((cov_matrix[[2, 2]] - 15.0_f64).abs() < 1e-10); // Variance of third column

        // Cross-covariances should be positive (all variables increase together)
        assert!(cov_matrix[[0, 1]] > 0.0);
        assert!(cov_matrix[[0, 2]] > 0.0);
        assert!(cov_matrix[[1, 2]] > 0.0);

        // Covariance matrix should be symmetric
        assert_eq!(cov_matrix[[0, 1]], cov_matrix[[1, 0]]);
        assert_eq!(cov_matrix[[0, 2]], cov_matrix[[2, 0]]);
        assert_eq!(cov_matrix[[1, 2]], cov_matrix[[2, 1]]);

        // Test with different ddof
        let cov_matrix_ddof0 = cov(data.view(), 0).unwrap();
        assert!((cov_matrix_ddof0[[0, 0]] - 11.25_f64).abs() < 1e-10); // Variance with ddof=0

        // Test invalid inputs
        assert!(cov(Array::<f64, _>::zeros((0, 0)).view(), 1).is_err()); // Empty array
        assert!(cov(array![[1.0], [2.0]].view(), 2).is_err()); // Not enough samples for ddof
    }

    #[test]
    fn test_digitize() {
        let x = array![0.2, 1.4, 2.5, 6.2, 9.7, 2.1];
        let bins = array![0.0, 1.0, 2.5, 4.0, 10.0];

        // Test right=true (right inclusive)
        let indices_right = digitize(x.view(), bins.view(), true, "indices").unwrap();
        assert_eq!(indices_right[0], 1); // 0.2 is in bin [0.0, 1.0]
        assert_eq!(indices_right[1], 2); // 1.4 is in bin [1.0, 2.5]
        assert_eq!(indices_right[2], 2); // 2.5 is in bin [1.0, 2.5] (right inclusive)
        assert_eq!(indices_right[3], 4); // 6.2 is in bin [4.0, 10.0]
        assert_eq!(indices_right[4], 4); // 9.7 is in bin [4.0, 10.0]
        assert_eq!(indices_right[5], 2); // 2.1 is in bin [1.0, 2.5]

        // Test right=false (left inclusive)
        let indices_left = digitize(x.view(), bins.view(), false, "indices").unwrap();
        assert_eq!(indices_left[0], 1); // 0.2 is in bin [0.0, 1.0)
        assert_eq!(indices_left[1], 2); // 1.4 is in bin [1.0, 2.5)
        assert_eq!(indices_left[2], 3); // 2.5 is in bin [2.5, 4.0) (left inclusive)
        assert_eq!(indices_left[3], 4); // 6.2 is in bin [4.0, 10.0)
        assert_eq!(indices_left[4], 4); // 9.7 is in the last bin [4.0, 10.0), or it counts as bin 4
        assert_eq!(indices_left[5], 2); // 2.1 is in bin [1.0, 2.5)

        // Test invalid inputs
        let empty_arr: Array<f64, Ix1> = array![];
        assert!(digitize(empty_arr.view(), bins.view(), true, "indices").is_err()); // Empty array
        let empty_bins: Array<f64, Ix1> = array![];
        assert!(digitize(x.view(), empty_bins.view(), true, "indices").is_err()); // Empty bins

        // Test non-monotonic bins
        let bad_bins = array![0.0, 2.0, 1.0, 4.0];
        assert!(digitize(x.view(), bad_bins.view(), true, "indices").is_err());

        // Test invalid result type
        assert!(digitize(x.view(), bins.view(), true, "invalid").is_err());
    }

    #[test]
    fn test_1d_functions() {
        // Test 1D array functions
        let a = array![1.0, 2.0, 3.0, 4.0, 5.0];

        // Test mean
        let mean_val = mean(&a.view(), None).unwrap();
        assert_eq!(mean_val[0], 3.0);

        // Test median
        let median_val = median(&a.view(), None).unwrap();
        assert_eq!(median_val[0], 3.0);

        // Test variance
        let var_val = variance(&a.view(), None, 1).unwrap();
        assert!((var_val[0] - 2.5_f64).abs() < 1e-10);

        // Test standard deviation
        let std_val = std_dev(&a.view(), None, 1).unwrap();
        assert!((std_val[0] - 1.5811388300841898_f64).abs() < 1e-10);

        // Test min/max
        let min_val = min(&a.view(), None).unwrap();
        assert_eq!(min_val[0], 1.0);

        let max_val = max(&a.view(), None).unwrap();
        assert_eq!(max_val[0], 5.0);

        // Test percentile
        let p25 = percentile(&a.view(), 25.0, None).unwrap();
        assert_eq!(p25[0], 2.0);

        let p50 = percentile(&a.view(), 50.0, None).unwrap();
        assert_eq!(p50[0], 3.0);

        let p75 = percentile(&a.view(), 75.0, None).unwrap();
        assert_eq!(p75[0], 4.0);
    }
}
