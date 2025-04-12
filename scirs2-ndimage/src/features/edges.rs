//! Edge detection algorithms
//!
//! This module provides functions for detecting edges in n-dimensional arrays,
//! including gradient-based methods, zero-crossing methods, and other edge detection techniques.

use crate::error::Result;
use crate::filters::{convolve, gaussian_filter_f32, sobel, BorderMode};
use ndarray::{Array, ArrayD, Dimension, Ix2};

/// Canny edge detector
///
/// The Canny edge detector is an edge detection operator that uses a multi-stage algorithm
/// to detect a wide range of edges in images. It was developed by John F. Canny in 1986.
///
/// # Arguments
///
/// * `image` - Input array, 2D array
/// * `sigma` - Standard deviation of the Gaussian filter
/// * `low_threshold` - Lower threshold for hysteresis
/// * `high_threshold` - Upper threshold for hysteresis
///
/// # Returns
///
/// * Array of edges, where non-zero values indicate detected edges
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_ndimage::features::canny;
///
/// let image = array![
///     [0.0, 0.0, 0.0, 0.0, 0.0],
///     [0.0, 0.0, 0.0, 0.0, 0.0],
///     [0.0, 0.0, 1.0, 0.0, 0.0],
///     [0.0, 0.0, 0.0, 0.0, 0.0],
///     [0.0, 0.0, 0.0, 0.0, 0.0],
/// ];
///
/// let edges = canny(&image, 1.0, 0.1, 0.2);
/// ```
pub fn canny(
    image: &Array<f32, Ix2>,
    sigma: f32,
    low_threshold: f32,
    high_threshold: f32,
) -> Array<bool, Ix2> {
    // Convert to ArrayD for processing
    let image_dim = image.raw_dim();
    let mut image_d = ArrayD::from_elem(image_dim.into_dyn(), 0.0f32);
    for (idx, &val) in image.indexed_iter() {
        let idx_d = [idx.0, idx.1];
        image_d[idx_d] = val;
    }

    // Step 1: Gaussian filter to reduce noise
    let smoothed = gaussian_filter_f32(&image_d, sigma, Some(BorderMode::Reflect), None).unwrap();

    // Step 2: Calculate gradients using Sobel
    let gradient_y = sobel(&smoothed, 0, Some(BorderMode::Reflect)).unwrap();
    let gradient_x = sobel(&smoothed, 1, Some(BorderMode::Reflect)).unwrap();

    // Convert back to 2D arrays for the rest of the processing
    let mut magnitude = Array::zeros(image_dim);
    let mut direction = Array::zeros(image_dim);

    // Calculate gradient magnitude and direction
    for idx in image.indexed_iter() {
        let pos = idx.0;
        let idx_d = [pos.0, pos.1];
        let gx = gradient_x[idx_d.as_ref()];
        let gy = gradient_y[idx_d.as_ref()];

        // Calculate magnitude using Euclidean distance
        magnitude[pos] = (gx * gx + gy * gy).sqrt();

        // Calculate direction in degrees and convert to one of four directions (0, 45, 90, 135)
        let angle = gy.atan2(gx) * 180.0 / std::f32::consts::PI;
        let angle = if angle < 0.0 { angle + 180.0 } else { angle };

        // Quantize the angle to 4 directions (0, 45, 90, 135 degrees)
        if (angle < 22.5) || (angle >= 157.5) {
            direction[pos] = 0.0; // 0 degrees (horizontal)
        } else if (angle >= 22.5) && (angle < 67.5) {
            direction[pos] = 45.0; // 45 degrees
        } else if (angle >= 67.5) && (angle < 112.5) {
            direction[pos] = 90.0; // 90 degrees (vertical)
        } else {
            direction[pos] = 135.0; // 135 degrees
        }
    }

    // Step 3: Non-maximum suppression
    let mut suppressed = Array::zeros(image_dim);
    let shape = image_dim.into_pattern();

    for row in 1..(shape.0 - 1) {
        for col in 1..(shape.1 - 1) {
            let dir = direction[(row, col)];
            let mag = magnitude[(row, col)];

            let (neighbor1, neighbor2) = get_gradient_neighbors(row, col, dir, &magnitude);

            // If the current pixel is a local maximum, keep it, otherwise suppress it
            if mag >= neighbor1 && mag >= neighbor2 {
                suppressed[(row, col)] = mag;
            } else {
                suppressed[(row, col)] = 0.0;
            }
        }
    }

    // Step 4: Double thresholding and edge tracking by hysteresis
    let mut strong_edges = Array::from_elem(image_dim, false);
    let mut weak_edges = Array::from_elem(image_dim, false);

    for idx in suppressed.indexed_iter() {
        let pos = idx.0;
        let val = *idx.1;

        if val >= high_threshold {
            strong_edges[pos] = true;
        } else if val >= low_threshold {
            weak_edges[pos] = true;
        }
    }

    // Step 5: Edge tracking by hysteresis
    let mut result = strong_edges.clone();
    let shape = image_dim.into_pattern();

    // Recursive edge tracking (implemented iteratively for simplicity)
    let mut changed = true;
    while changed {
        changed = false;

        for row in 1..(shape.0 - 1) {
            for col in 1..(shape.1 - 1) {
                if weak_edges[(row, col)] && !result[(row, col)] {
                    // Check if this weak edge is connected to a strong edge
                    if is_connected_to_strong_edge(row, col, &result) {
                        result[(row, col)] = true;
                        changed = true;
                    }
                }
            }
        }
    }

    result
}

/// Helper function to get the neighbors in the gradient direction
fn get_gradient_neighbors(
    row: usize,
    col: usize,
    direction: f32,
    magnitude: &Array<f32, Ix2>,
) -> (f32, f32) {
    // 0 degrees (horizontal)
    if direction == 0.0 {
        (magnitude[(row, col - 1)], magnitude[(row, col + 1)])
    }
    // 45 degrees
    else if direction == 45.0 {
        (magnitude[(row - 1, col + 1)], magnitude[(row + 1, col - 1)])
    }
    // 90 degrees (vertical)
    else if direction == 90.0 {
        (magnitude[(row - 1, col)], magnitude[(row + 1, col)])
    }
    // 135 degrees
    else {
        (magnitude[(row - 1, col - 1)], magnitude[(row + 1, col + 1)])
    }
}

/// Helper function to check if a pixel is connected to a strong edge
fn is_connected_to_strong_edge(row: usize, col: usize, edges: &Array<bool, Ix2>) -> bool {
    for i in (row.saturating_sub(1))..=(row + 1) {
        for j in (col.saturating_sub(1))..=(col + 1) {
            if i < edges.shape()[0]
                && j < edges.shape()[1]
                && !(i == row && j == col)
                && edges[(i, j)]
            {
                return true;
            }
        }
    }
    false
}

/// Laplacian edge detector
///
/// The Laplacian operator is used for edge detection where the second derivative changes sign.
///
/// # Arguments
///
/// * `image` - Input array
/// * `ksize` - Size of the Laplace kernel
/// * `sigma` - Standard deviation of the Gaussian filter (pre-processing step)
///
/// # Returns
///
/// * Result containing the detected edges
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_ndimage::features::laplacian_edges;
///
/// let image = array![
///     [0.0, 0.0, 0.0, 0.0, 0.0],
///     [0.0, 0.0, 0.0, 0.0, 0.0],
///     [0.0, 0.0, 1.0, 0.0, 0.0],
///     [0.0, 0.0, 0.0, 0.0, 0.0],
///     [0.0, 0.0, 0.0, 0.0, 0.0],
/// ];
///
/// let edges = laplacian_edges(&image.into_dyn(), 3, 1.0).unwrap();
/// ```
pub fn laplacian_edges(image: &ArrayD<f32>, ksize: usize, sigma: f32) -> Result<ArrayD<f32>> {
    // First, apply Gaussian filter to reduce noise
    let smoothed = gaussian_filter_f32(image, sigma, Some(BorderMode::Reflect), None)?;

    // Create Laplace kernel
    let mut kernel = ArrayD::zeros(vec![ksize; image.ndim()]);
    let center = ksize / 2;

    // Fill kernel with the discrete Laplacian operator
    // For 2D: [0 1 0; 1 -4 1; 0 1 0]
    // Center value depends on the dimension
    let center_value = -2.0 * image.ndim() as f32;

    // Set center to the sum of all other values (negative)
    let ndim = image.ndim();
    let center_idx = vec![center; ndim];
    kernel[center_idx.as_slice()] = center_value;

    // Set direct neighbors to 1
    for dim in 0..image.ndim() {
        let mut idx = vec![center; image.ndim()];

        // Set previous neighbor
        if center > 0 {
            idx[dim] = center - 1;
            kernel[idx.as_slice()] = 1.0;
        }

        // Set next neighbor
        idx[dim] = center + 1;
        if idx[dim] < ksize {
            kernel[idx.as_slice()] = 1.0;
        }
    }

    // Apply convolution
    convolve(&smoothed, &kernel, Some(BorderMode::Reflect))
}

/// Sobel edge detector
///
/// A simplified wrapper around the sobel function that returns the magnitude of edges.
///
/// # Arguments
///
/// * `image` - Input array
///
/// # Returns
///
/// * Result containing the magnitude of edges
pub fn sobel_edges(image: &ArrayD<f32>) -> Result<ArrayD<f32>> {
    let gradient_y = sobel(image, 0, Some(BorderMode::Reflect))?;
    let gradient_x = sobel(image, 1, Some(BorderMode::Reflect))?;

    // Calculate the magnitude
    let mut magnitude = gradient_x.mapv(|x| x * x);
    for (idx, &y) in gradient_y.indexed_iter() {
        magnitude[idx.slice()] += y * y;
    }

    Ok(magnitude.mapv(|x| x.sqrt()))
}
