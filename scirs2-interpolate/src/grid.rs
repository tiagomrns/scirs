//! Grid transformation and resampling utilities
//!
//! This module provides functions for transforming irregular data onto regular grids
//! and vice versa, as well as various grid manipulation utilities.

use crate::error::{InterpolateError, InterpolateResult};
use ndarray::{Array1, ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

/// Grid transformation methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GridTransformMethod {
    /// Nearest neighbor assignment
    Nearest,
    /// Linear interpolation
    Linear,
    /// Cubic spline interpolation
    Cubic,
}

/// Create a grid of evenly spaced coordinates
///
/// # Arguments
///
/// * `bounds` - Min and max bounds for each dimension
/// * `shape` - Number of points in each dimension
///
/// # Returns
///
/// Vector of coordinate arrays, one for each dimension
///
/// # Examples
///
/// ```
/// use scirs2_interpolate::grid::create_regular_grid;
///
/// // Create a 10×20 grid from (0,0) to (1,2)
/// let grid_coords = create_regular_grid(
///     &[(0.0, 1.0), (0.0, 2.0)],
///     &[10, 20]
/// ).unwrap();
///
/// assert_eq!(grid_coords.len(), 2);
/// assert_eq!(grid_coords[0].len(), 10);
/// assert_eq!(grid_coords[1].len(), 20);
/// ```
pub fn create_regular_grid<F>(
    bounds: &[(F, F)],
    shape: &[usize],
) -> InterpolateResult<Vec<Array1<F>>>
where
    F: Float + FromPrimitive + Debug,
{
    if bounds.len() != shape.len() {
        return Err(InterpolateError::ValueError(
            "bounds and shape must have the same length".to_string(),
        ));
    }

    let n_dims = bounds.len();
    let mut grid_coords = Vec::with_capacity(n_dims);

    for i in 0..n_dims {
        let (min, max) = bounds[i];
        let n_points = shape[i];

        if min >= max {
            return Err(InterpolateError::ValueError(
                "min bound must be less than max bound".to_string(),
            ));
        }

        if n_points < 2 {
            return Err(InterpolateError::ValueError(
                "grid shape must have at least 2 points in each dimension".to_string(),
            ));
        }

        let mut coords = Array1::zeros(n_points);

        if n_points == 1 {
            coords[0] = min;
        } else {
            let step = (max - min) / F::from_usize(n_points - 1).unwrap();
            for j in 0..n_points {
                coords[j] = min + F::from_usize(j).unwrap() * step;
            }
        }

        grid_coords.push(coords);
    }

    Ok(grid_coords)
}

/// Resample irregular (scattered) data onto a regular grid
///
/// # Arguments
///
/// * `points` - Coordinates of scattered data points (n_points × n_dimensions)
/// * `values` - Values at the scattered data points (n_points)
/// * `grid_shape` - Shape of the output grid (number of points in each dimension)
/// * `grid_bounds` - Min and max bounds for each dimension of the grid
/// * `method` - Interpolation method to use
/// * `fill_value` - Value to use for grid points outside the convex hull of input points
///
/// # Returns
///
/// A tuple containing:
/// * The grid coordinates for each dimension (vector of arrays)
/// * The resampled values on the regular grid
pub fn resample_to_grid<F>(
    _points: &ArrayView2<F>,
    _values: &ArrayView1<F>,
    grid_shape: &[usize],
    grid_bounds: &[(F, F)],
    _method: GridTransformMethod,
    fill_value: F,
) -> InterpolateResult<(Vec<Array1<F>>, ndarray::ArrayD<F>)>
where
    F: Float + FromPrimitive + Debug,
{
    // This is a placeholder implementation
    // In a full implementation, we would:
    // 1. Create a regular grid based on the grid_shape and grid_bounds
    // 2. Use an appropriate interpolation method to resample the scattered data onto the grid

    // For now, just create the grid coordinates and return an array filled with the fill_value
    let grid_coords = create_regular_grid(grid_bounds, grid_shape)?;

    // Create a multidimensional array with the specified shape
    let mut shape = Vec::with_capacity(grid_shape.len());
    for &s in grid_shape {
        shape.push(s);
    }

    let grid_values = ndarray::ArrayD::from_elem(shape, fill_value);

    // In the future, implement proper interpolation here

    Ok((grid_coords, grid_values))
}

/// Resample a regular grid to another regular grid with different resolution or bounds
///
/// # Arguments
///
/// * `src_coords` - Source grid coordinates (vector of arrays, one per dimension)
/// * `src_values` - Source grid values
/// * `dst_coords` - Destination grid coordinates (vector of arrays, one per dimension)
/// * `method` - Interpolation method to use
/// * `fill_value` - Value to use for grid points outside the source grid
///
/// # Returns
///
/// Resampled values on the destination grid
pub fn resample_grid_to_grid<F, D>(
    _src_coords: &[Array1<F>],
    _src_values: &ndarray::ArrayView<F, D>,
    dst_coords: &[Array1<F>],
    _method: GridTransformMethod,
    fill_value: F,
) -> InterpolateResult<ndarray::ArrayD<F>>
where
    F: Float + FromPrimitive + Debug,
    D: ndarray::Dimension,
{
    // This is a placeholder implementation
    // In a full implementation, we would:
    // 1. Create interpolator from the source grid
    // 2. Evaluate at all points in the destination grid

    // For now, just return an array filled with the fill_value
    let mut shape = Vec::with_capacity(dst_coords.len());
    for coords in dst_coords {
        shape.push(coords.len());
    }

    let dst_values = ndarray::ArrayD::from_elem(shape, fill_value);

    // In the future, implement proper interpolation here

    Ok(dst_values)
}

/// Maps values from a regular grid to arbitrary points using interpolation
///
/// # Arguments
///
/// * `grid_coords` - Grid coordinates (vector of arrays, one per dimension)
/// * `grid_values` - Values on the regular grid
/// * `query_points` - Points at which to evaluate (n_points × n_dimensions)
/// * `method` - Interpolation method to use
/// * `fill_value` - Value to use for query points outside the grid
///
/// # Returns
///
/// Values at the query points
pub fn map_grid_to_points<F, D>(
    _grid_coords: &[Array1<F>],
    _grid_values: &ndarray::ArrayView<F, D>,
    query_points: &ArrayView2<F>,
    _method: GridTransformMethod,
    fill_value: F,
) -> InterpolateResult<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
    D: ndarray::Dimension,
{
    // This is a placeholder implementation
    // In a full implementation, we would:
    // 1. Create interpolator from the grid
    // 2. Evaluate at all query points

    // For now, just return an array filled with the fill_value
    let n_points = query_points.shape()[0];
    let result = Array1::from_elem(n_points, fill_value);

    // In the future, implement proper interpolation here

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    // テスト関数で使用するモジュール

    #[test]
    fn test_create_regular_grid() {
        // 1D grid
        let grid_1d = create_regular_grid(&[(0.0, 1.0)], &[5]).unwrap();

        assert_eq!(grid_1d.len(), 1);
        assert_eq!(grid_1d[0].len(), 5);
        assert_abs_diff_eq!(grid_1d[0][0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(grid_1d[0][4], 1.0, epsilon = 1e-10);

        // 2D grid
        let grid_2d = create_regular_grid(&[(0.0, 1.0), (-1.0, 1.0)], &[3, 5]).unwrap();

        assert_eq!(grid_2d.len(), 2);
        assert_eq!(grid_2d[0].len(), 3);
        assert_eq!(grid_2d[1].len(), 5);
        assert_abs_diff_eq!(grid_2d[0][0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(grid_2d[0][2], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(grid_2d[1][0], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(grid_2d[1][4], 1.0, epsilon = 1e-10);
    }
}
