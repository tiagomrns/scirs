//! Median filtering functions for n-dimensional arrays

use ndarray::{Array, Array1, Array2, Dimension};
use num_traits::{Float, FromPrimitive};
use scirs2_core::validation::{check_1d, check_2d, check_positive};
use std::fmt::Debug;

use super::{pad_array, BorderMode};
use crate::error::{NdimageError, Result};

/// Apply a median filter to an n-dimensional array
///
/// # Arguments
///
/// * `input` - Input array to filter
/// * `size` - Size of the filter kernel in each dimension
/// * `mode` - Border handling mode (defaults to Reflect)
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Filtered array
pub fn median_filter<T, D>(
    input: &Array<T, D>,
    size: &[usize],
    mode: Option<BorderMode>,
) -> Result<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + PartialOrd + Clone,
    D: Dimension,
{
    let border_mode = mode.unwrap_or(BorderMode::Reflect);

    // Validate that size array has same dimensions as input
    if size.len() != input.ndim() {
        return Err(NdimageError::DimensionError(format!(
            "Size must have same length as input dimensions (got {} expected {})",
            size.len(),
            input.ndim()
        )));
    }

    // Validate all kernel sizes are positive
    for (i, &s) in size.iter().enumerate() {
        check_positive(s, format!("Kernel size in dimension {}", i)).map_err(NdimageError::from)?;
    }

    // Handle scalar or constant case
    if input.len() <= 1 {
        return Ok(input.to_owned());
    }

    // Calculate kernel radii (half size)
    let _radii: Vec<usize> = size.iter().map(|&s| s / 2).collect();

    // Dispatch to the appropriate implementation based on dimensionality
    match input.ndim() {
        1 => {
            // Handle 1D array
            let input_1d = input
                .to_owned()
                .into_dimensionality::<ndarray::Ix1>()
                .map_err(|_| {
                    NdimageError::DimensionError("Failed to convert to 1D array".into())
                })?;

            // Validate that the input is 1D (redundant but for consistency)
            check_1d(&input_1d, "input").map_err(NdimageError::from)?;

            let result_1d = median_filter_1d(&input_1d, size[0], &border_mode)?;

            result_1d.into_dimensionality::<D>().map_err(|_| {
                NdimageError::DimensionError("Failed to convert back from 1D array".into())
            })
        }
        2 => {
            // Handle 2D array
            let input_2d = input
                .to_owned()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| {
                    NdimageError::DimensionError("Failed to convert to 2D array".into())
                })?;

            // Validate that the input is 2D (redundant but for consistency)
            check_2d(&input_2d, "input").map_err(NdimageError::from)?;

            let result_2d = median_filter_2d(&input_2d, size, &border_mode)?;

            result_2d.into_dimensionality::<D>().map_err(|_| {
                NdimageError::DimensionError("Failed to convert back from 2D array".into())
            })
        }
        _ => {
            // For higher dimensions, use a general implementation
            median_filter_nd(input, size, &border_mode)
        }
    }
}

/// Apply a median filter to a 1D array
fn median_filter_1d<T>(input: &Array1<T>, size: usize, mode: &BorderMode) -> Result<Array1<T>>
where
    T: Float + FromPrimitive + Debug + PartialOrd + Clone,
{
    let radius = size / 2;

    // Create output array
    let mut output = Array1::zeros(input.len());

    // Pad input for border handling
    let pad_width = vec![(radius, radius)];
    let padded_input = pad_array(input, &pad_width, mode, None)?;

    // Apply median filter to each position
    for i in 0..input.len() {
        let center = i + radius;

        // Extract window
        let mut window = Vec::with_capacity(size);
        for k in 0..size {
            window.push(padded_input[center - radius + k]);
        }

        // Sort window and find median
        window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        output[i] = window[size / 2];
    }

    Ok(output)
}

/// Apply a median filter to a 2D array
fn median_filter_2d<T>(input: &Array2<T>, size: &[usize], mode: &BorderMode) -> Result<Array2<T>>
where
    T: Float + FromPrimitive + Debug + PartialOrd + Clone,
{
    let rows = input.shape()[0];
    let cols = input.shape()[1];
    let radius_y = size[0] / 2;
    let radius_x = size[1] / 2;
    let window_size = size[0] * size[1];

    // Create output array
    let mut output = Array2::zeros((rows, cols));

    // Pad input for border handling
    let pad_width = vec![(radius_y, radius_y), (radius_x, radius_x)];
    let padded_input = pad_array(input, &pad_width, mode, None)?;

    // Apply median filter to each position
    for i in 0..rows {
        for j in 0..cols {
            // Calculate padded coordinates
            let center_y = i + radius_y;
            let center_x = j + radius_x;

            // Extract window
            let mut window = Vec::with_capacity(window_size);
            for ky in 0..size[0] {
                for kx in 0..size[1] {
                    let y = center_y - radius_y + ky;
                    let x = center_x - radius_x + kx;
                    window.push(padded_input[[y, x]]);
                }
            }

            // Sort window and find median
            window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            output[[i, j]] = window[window_size / 2];
        }
    }

    Ok(output)
}

/// Apply a median filter to an n-dimensional array with arbitrary dimensionality
fn median_filter_nd<T, D>(
    input: &Array<T, D>,
    size: &[usize],
    mode: &BorderMode,
) -> Result<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + PartialOrd + Clone,
    D: Dimension,
{
    // Calculate radii
    let radii: Vec<usize> = size.iter().map(|&s| s / 2).collect();

    // Calculate total window size
    let window_size: usize = size.iter().product();

    // Create output array and convert to dynamic dimensions for efficient indexing
    let mut output_dyn = Array::<T, ndarray::IxDyn>::zeros(ndarray::IxDyn(input.shape()));

    // Pad the input array
    let pad_width: Vec<(usize, usize)> = radii.iter().map(|&r| (r, r)).collect();
    let padded_input_array = pad_array(input, &pad_width, mode, None)?;

    // Convert to dynamic dimension once for efficiency
    let padded_input = padded_input_array
        .clone()
        .into_dimensionality::<ndarray::IxDyn>()
        .unwrap();

    // Use a cartesian_product-like approach to iterate through all elements
    let mut indices = vec![0; input.ndim()];
    let shape = input.shape();

    // Process each position in the output array
    loop {
        // Get center indices in the padded array
        let center_indices: Vec<usize> = indices.iter().zip(&radii).map(|(&i, &r)| i + r).collect();

        // Extract and sort the window values
        let mut window = Vec::with_capacity(window_size);

        // Recursively iterate through all positions in the window
        fn add_window_values<T: Clone>(
            window: &mut Vec<T>,
            padded_input: &Array<T, ndarray::IxDyn>,
            center_indices: &[usize],
            radii: &[usize],
            size: &[usize],
            dim_index: usize,
            current_indices: &mut Vec<usize>,
        ) {
            if dim_index == padded_input.ndim() {
                // At leaf level, add value to window
                window.push(
                    padded_input
                        .get(current_indices.as_slice())
                        .unwrap()
                        .clone(),
                );
                return;
            }

            let center = center_indices[dim_index];
            let radius = radii[dim_index];
            let dim_size = size[dim_index];

            for k in 0..dim_size {
                current_indices[dim_index] = center - radius + k;
                add_window_values(
                    window,
                    padded_input,
                    center_indices,
                    radii,
                    size,
                    dim_index + 1,
                    current_indices,
                );
            }
        }

        // Initialize current indices for recursion
        let mut current_indices = center_indices.clone();

        // Get all values in the window
        add_window_values(
            &mut window,
            &padded_input,
            &center_indices,
            &radii,
            size,
            0,
            &mut current_indices,
        );

        // Sort and find median
        window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = window[window_size / 2];

        // Set output value directly in dynamic array
        output_dyn[indices.as_slice()] = median;

        // Move to next position
        let mut increment_done = false;
        for i in (0..indices.len()).rev() {
            indices[i] += 1;
            if indices[i] < shape[i] {
                increment_done = true;
                break;
            }
            indices[i] = 0;
        }

        if !increment_done {
            break;
        }
    }

    // Convert back to original dimensionality at the end
    output_dyn.into_dimensionality::<D>().map_err(|_| {
        NdimageError::DimensionError("Failed to convert back from dynamic dimensions".into())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_median_filter_1d() {
        // Create a 1D array with an outlier
        let array = Array1::from_vec(vec![1.0, 2.0, 3.0, 100.0, 5.0]);

        // Apply median filter with size 3
        let result = median_filter(&array, &[3], None).unwrap();

        // Check dimensions
        assert_eq!(result.shape(), array.shape());

        // Check that outlier is smoothed out
        assert_eq!(result[3], 5.0); // [2, 3, 100] -> median = 3
    }

    #[test]
    fn test_median_filter_2d() {
        // Create a simple test image with an outlier
        let mut image = Array2::zeros((5, 5));
        image[[2, 2]] = 100.0; // Center pixel is an outlier

        // Apply filter
        let result = median_filter(&image, &[3, 3], None).unwrap();

        // Check that result has the same shape
        assert_eq!(result.shape(), image.shape());

        // Check that outlier is removed (should be 0.0 after median filtering)
        assert_eq!(result[[2, 2]], 0.0);
    }

    #[test]
    fn test_median_filter_noise_removal() {
        // Create an array with salt and pepper noise
        let array = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0, 100.0, 1.0, 0.0, 1.0, 0.0]);

        // Apply median filter with size 3
        let result = median_filter(&array, &[3], None).unwrap();

        // Check that noise is reduced
        assert_eq!(result[4], 1.0); // [0, 1, 100] -> median = 1
    }

    #[test]
    fn test_median_filter_invalid_size() {
        // Create a simple test image
        let image: Array2<f64> = Array2::eye(5);

        // Apply filter with wrong size dimensionality
        let result = median_filter(&image, &[3], None);

        // Check that it returns an error
        assert!(result.is_err());
    }

    #[test]
    fn test_median_filter_even_kernel() {
        // Create a simple array
        let array = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        // Apply median filter with an even size (should still work, using the middle-right value)
        let result = median_filter(&array, &[4], None).unwrap();

        // Should still have the same dimensions
        assert_eq!(result.shape(), array.shape());
    }

    #[test]
    fn test_median_filter_zero_size() {
        // Create a simple array
        let array = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        // Apply median filter with zero size (should fail)
        let result = median_filter(&array, &[0], None);

        // Check that it returns an error
        assert!(result.is_err());
    }

    #[test]
    fn test_median_filter_3d() {
        use ndarray::Array3;

        // Create a 3D array with an outlier
        let mut cube = Array3::<f64>::zeros((3, 3, 3));
        cube[[1, 1, 1]] = 100.0; // Center voxel is an outlier

        // Apply median filter with size [3, 3, 3]
        let result = median_filter(&cube, &[3, 3, 3], None).unwrap();

        // Check that result has the same shape
        assert_eq!(result.shape(), cube.shape());

        // Check that outlier is removed (should be 0.0 after median filtering)
        assert_eq!(result[[1, 1, 1]], 0.0);

        // Create a 3D array with noise
        let mut noise_cube = Array3::<f64>::zeros((5, 5, 5));

        // Add some random outliers
        noise_cube[[1, 2, 3]] = 100.0;
        noise_cube[[3, 1, 2]] = -100.0;
        noise_cube[[2, 3, 1]] = 50.0;

        // Apply median filter
        let filtered = median_filter(&noise_cube, &[3, 3, 3], None).unwrap();

        // Check that outliers are removed or reduced
        assert!(filtered[[1, 2, 3]].abs() < 100.0);
        assert!(filtered[[3, 1, 2]].abs() < 100.0);
        assert!(filtered[[2, 3, 1]].abs() < 50.0);
    }
}
