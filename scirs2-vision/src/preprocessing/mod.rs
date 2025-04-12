//! Image preprocessing module
//!
//! This module provides functionality for preprocessing images before further analysis.

pub mod morphology;

pub use morphology::*;

use crate::error::{Result, VisionError};
use crate::feature::image_to_array;
use image::{DynamicImage, GrayImage, ImageBuffer, Luma};
use ndarray::Array2;

/// Convert an image to grayscale
///
/// # Arguments
///
/// * `img` - Input image
///
/// # Returns
///
/// * Grayscale image
pub fn to_grayscale(img: &DynamicImage) -> GrayImage {
    img.to_luma8()
}

/// Normalize image brightness and contrast
///
/// # Arguments
///
/// * `img` - Input image
/// * `min_out` - Minimum output intensity (0.0 to 1.0)
/// * `max_out` - Maximum output intensity (0.0 to 1.0)
///
/// # Returns
///
/// * Result containing the normalized image
pub fn normalize_brightness(
    img: &DynamicImage,
    min_out: f32,
    max_out: f32,
) -> Result<DynamicImage> {
    if !(0.0..=1.0).contains(&min_out) || !(0.0..=1.0).contains(&max_out) || min_out >= max_out {
        return Err(VisionError::InvalidParameter(
            "Output intensity range must be within [0, 1] and min_out < max_out".to_string(),
        ));
    }

    // Convert to grayscale and then to array
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();

    // Find min and max values
    let mut min_val = 255;
    let mut max_val = 0;

    for pixel in gray.pixels() {
        let val = pixel[0];
        if val < min_val {
            min_val = val;
        }
        if val > max_val {
            max_val = val;
        }
    }

    // Handle edge case: if all pixels have the same value
    if min_val == max_val {
        return Ok(img.clone());
    }

    // Create output image
    let mut result = ImageBuffer::new(width, height);

    // Map input range to output range
    let scale = (max_out - min_out) / (max_val as f32 - min_val as f32);
    let offset = min_out - min_val as f32 * scale;

    for y in 0..height {
        for x in 0..width {
            let val = gray.get_pixel(x, y)[0];
            let new_val = (val as f32 * scale + offset) * 255.0;
            result.put_pixel(x, y, Luma([new_val.clamp(0.0, 255.0) as u8]));
        }
    }

    Ok(DynamicImage::ImageLuma8(result))
}

/// Apply histogram equalization to enhance contrast
///
/// # Arguments
///
/// * `img` - Input image
///
/// # Returns
///
/// * Result containing the contrast-enhanced image
pub fn equalize_histogram(img: &DynamicImage) -> Result<DynamicImage> {
    // Convert to grayscale
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();
    let total_pixels = width * height;

    // Calculate histogram
    let mut histogram = [0u32; 256];
    for pixel in gray.pixels() {
        histogram[pixel[0] as usize] += 1;
    }

    // Calculate cumulative distribution function
    let mut cdf = [0u32; 256];
    cdf[0] = histogram[0];
    for i in 1..256 {
        cdf[i] = cdf[i - 1] + histogram[i];
    }

    // Find first non-zero value in CDF
    let cdf_min = cdf.iter().find(|&&x| x > 0).unwrap_or(&0);

    // Create mapping function
    let mut mapping = [0u8; 256];
    for i in 0..256 {
        mapping[i] =
            (((cdf[i] - cdf_min) as f32 / (total_pixels - cdf_min) as f32) * 255.0).round() as u8;
    }

    // Apply mapping to create equalized image
    let mut result = ImageBuffer::new(width, height);
    for (x, y, pixel) in gray.enumerate_pixels() {
        result.put_pixel(x, y, Luma([mapping[pixel[0] as usize]]));
    }

    Ok(DynamicImage::ImageLuma8(result))
}

/// Apply Gaussian blur to reduce noise
///
/// # Arguments
///
/// * `img` - Input image
/// * `sigma` - Standard deviation of the Gaussian kernel
///
/// # Returns
///
/// * Result containing the blurred image
pub fn gaussian_blur(img: &DynamicImage, sigma: f32) -> Result<DynamicImage> {
    if sigma <= 0.0 {
        return Err(VisionError::InvalidParameter(
            "Sigma must be positive".to_string(),
        ));
    }

    // Convert to array
    let array = image_to_array(img)?;
    let (height, width) = array.dim();

    // Determine kernel size based on sigma (3-sigma rule)
    let kernel_radius = (3.0 * sigma).ceil() as usize;
    let kernel_size = 2 * kernel_radius + 1;

    // Create Gaussian kernel
    let mut kernel = Array2::zeros((kernel_size, kernel_size));
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut sum = 0.0;

    for y in 0..kernel_size {
        for x in 0..kernel_size {
            let dy = (y as isize - kernel_radius as isize) as f32;
            let dx = (x as isize - kernel_radius as isize) as f32;
            let exponent = -(dx * dx + dy * dy) / two_sigma_sq;
            let value = exponent.exp();
            kernel[[y, x]] = value;
            sum += value;
        }
    }

    // Normalize kernel
    kernel.mapv_inplace(|x| x / sum);

    // Apply convolution
    let mut result = Array2::zeros((height, width));

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            let mut weight_sum = 0.0;

            for ky in 0..kernel_size {
                let iy = y as isize + (ky as isize - kernel_radius as isize);
                if iy < 0 || iy >= height as isize {
                    continue;
                }

                for kx in 0..kernel_size {
                    let ix = x as isize + (kx as isize - kernel_radius as isize);
                    if ix < 0 || ix >= width as isize {
                        continue;
                    }

                    let weight = kernel[[ky, kx]];
                    sum += array[[iy as usize, ix as usize]] * weight;
                    weight_sum += weight;
                }
            }

            // Normalize by weight sum to handle border properly
            result[[y, x]] = sum / weight_sum;
        }
    }

    // Convert back to image
    let mut blurred = ImageBuffer::new(width as u32, height as u32);

    for y in 0..height {
        for x in 0..width {
            let value = (result[[y, x]] * 255.0).clamp(0.0, 255.0) as u8;
            blurred.put_pixel(x as u32, y as u32, Luma([value]));
        }
    }

    Ok(DynamicImage::ImageLuma8(blurred))
}

/// Apply unsharp masking to enhance edges
///
/// # Arguments
///
/// * `img` - Input image
/// * `sigma` - Standard deviation of the Gaussian blur
/// * `amount` - Strength of sharpening (typically 0.5 to 2.0)
///
/// # Returns
///
/// * Result containing the sharpened image
pub fn unsharp_mask(img: &DynamicImage, sigma: f32, amount: f32) -> Result<DynamicImage> {
    if amount < 0.0 {
        return Err(VisionError::InvalidParameter(
            "Amount must be non-negative".to_string(),
        ));
    }

    // Apply Gaussian blur
    let blurred = gaussian_blur(img, sigma)?;

    // Get original as grayscale
    let original = img.to_luma8();
    let (width, height) = original.dimensions();

    // Create sharpened image: original + amount * (original - blurred)
    let blurred_gray = blurred.to_luma8();
    let mut sharpened = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let orig_val = original.get_pixel(x, y)[0] as f32;
            let blur_val = blurred_gray.get_pixel(x, y)[0] as f32;

            // Calculate sharpened value and clamp to valid range
            let sharp_val = orig_val + amount * (orig_val - blur_val);
            let final_val = sharp_val.clamp(0.0, 255.0) as u8;

            sharpened.put_pixel(x, y, Luma([final_val]));
        }
    }

    Ok(DynamicImage::ImageLuma8(sharpened))
}
