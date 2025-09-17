//! FAST (Features from Accelerated Segment Test) corner detector
//!
//! FAST is a corner detection method designed to be computational efficient
//! while maintaining good detection quality. It examines a circle of pixels
//! around a candidate point to determine if it's a corner.
//!
//! # References
//!
//! - Rosten, E. and Drummond, T., 2006, May. Machine learning for high-speed corner detection. In European conference on computer vision (pp. 430-443). Springer, Berlin, Heidelberg.

use crate::error::Result;
use crate::feature::image_to_array;
use image::{DynamicImage, GrayImage};
use ndarray::Array2;

/// FAST corner detection
///
/// Detects corners using the FAST algorithm. A pixel is considered a corner
/// if there exist N contiguous pixels in a circle of 16 pixels around it
/// that are all brighter or all darker than the center pixel by a threshold.
///
/// # Arguments
///
/// * `img` - Input image
/// * `threshold` - Intensity difference threshold
/// * `n_consecutive` - Number of consecutive pixels required (typically 9 or 12)
/// * `non_max_suppression` - Apply non-maximum suppression to results
///
/// # Returns
///
/// * Result containing corner points
///
/// # Example
///
/// ```rust
/// use scirs2_vision::feature::fast_corners;
/// use image::DynamicImage;
///
/// # fn main() -> scirs2_vision::error::Result<()> {
/// let img = image::open("examples/input/input.jpg").unwrap();
/// let corners = fast_corners(&img, 20.0, 9, true)?;
/// # Ok(())
/// # }
/// ```
#[allow(dead_code)]
pub fn fast_corners(
    img: &DynamicImage,
    threshold: f32,
    n_consecutive: usize,
    non_max_suppression: bool,
) -> Result<GrayImage> {
    let array = image_to_array(img)?;
    let (height, width) = array.dim();

    // Convert threshold to normalized range since image_to_array normalizes to 0-1
    let normalized_threshold = threshold / 255.0;

    // Results array
    let mut corners = Array2::zeros((height, width));

    // The circle of 16 pixels (relative coordinates)
    // Standard FAST-9 circle pattern
    let circle16: [(i32, i32); 16] = [
        (0, -3),  // 0  - top
        (1, -3),  // 1
        (2, -2),  // 2
        (3, -1),  // 3
        (3, 0),   // 4  - right
        (3, 1),   // 5
        (2, 2),   // 6
        (1, 3),   // 7
        (0, 3),   // 8  - bottom
        (-1, 3),  // 9
        (-2, 2),  // 10
        (-3, 1),  // 11
        (-3, 0),  // 12 - left
        (-3, -1), // 13
        (-2, -2), // 14
        (-1, -3), // 15
    ];

    // Process each pixel (skip border pixels)
    for y in 3..(height - 3) {
        for x in 3..(width - 3) {
            let center = array[[y, x]];

            // Quick rejection test using pixels 0, 4, 8, 12
            // These are at the cardinal directions
            let p0 = array[[
                (y as i32 + circle16[0].1) as usize,
                (x as i32 + circle16[0].0) as usize,
            ]];
            let p4 = array[[
                (y as i32 + circle16[4].1) as usize,
                (x as i32 + circle16[4].0) as usize,
            ]];
            let p8 = array[[
                (y as i32 + circle16[8].1) as usize,
                (x as i32 + circle16[8].0) as usize,
            ]];
            let p12 = array[[
                (y as i32 + circle16[12].1) as usize,
                (x as i32 + circle16[12].0) as usize,
            ]];

            // At least 3 of these 4 pixels must be either all brighter or all darker
            let mut brighter_count = 0;
            let mut darker_count = 0;

            if p0 > center + normalized_threshold {
                brighter_count += 1;
            }
            if p0 < center - normalized_threshold {
                darker_count += 1;
            }
            if p4 > center + normalized_threshold {
                brighter_count += 1;
            }
            if p4 < center - normalized_threshold {
                darker_count += 1;
            }
            if p8 > center + normalized_threshold {
                brighter_count += 1;
            }
            if p8 < center - normalized_threshold {
                darker_count += 1;
            }
            if p12 > center + normalized_threshold {
                brighter_count += 1;
            }
            if p12 < center - normalized_threshold {
                darker_count += 1;
            }

            // Quick rejection
            if brighter_count < 3 && darker_count < 3 {
                continue;
            }

            // Full circle test
            let mut pixels = [0.0; 16];
            for i in 0..16 {
                let ny = (y as i32 + circle16[i].1) as usize;
                let nx = (x as i32 + circle16[i].0) as usize;
                pixels[i] = array[[ny, nx]];
            }

            // Check for _consecutive brighter or darker pixels
            let is_corner =
                check_consecutive_pixels(&pixels, center, normalized_threshold, n_consecutive);

            if is_corner {
                // Compute corner score for non-maximum _suppression
                let score = compute_corner_score(&pixels, center, normalized_threshold);
                corners[[y, x]] = score;
            }
        }
    }

    // Apply non-maximum _suppression if requested
    if non_max_suppression {
        corners = apply_non_max_suppression(&corners, 3);
    }

    // Convert to binary image
    crate::feature::array_to_image(&corners.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }))
}

/// Check if there are N consecutive pixels that are all brighter or darker
#[allow(dead_code)]
fn check_consecutive_pixels(
    pixels: &[f32; 16],
    center: f32,
    threshold: f32,
    n_consecutive: usize,
) -> bool {
    // We need to check circular sequences, so we extend the check
    // Check for _consecutive brighter pixels
    let mut brighter_count = 0;
    let mut max_brighter = 0;

    // Check twice around the circle to handle wrap-around cases
    for i in 0..32 {
        let idx = i % 16;

        if pixels[idx] > center + threshold {
            brighter_count += 1;
            if brighter_count > max_brighter {
                max_brighter = brighter_count;
            }
            if brighter_count >= n_consecutive {
                return true;
            }
        } else {
            brighter_count = 0;
        }
    }

    // Check for _consecutive darker pixels
    let mut darker_count = 0;
    let mut max_darker = 0;

    for i in 0..32 {
        let idx = i % 16;

        if pixels[idx] < center - threshold {
            darker_count += 1;
            if darker_count > max_darker {
                max_darker = darker_count;
            }
            if darker_count >= n_consecutive {
                return true;
            }
        } else {
            darker_count = 0;
        }
    }

    false
}

/// Compute corner score based on the sum of absolute differences
#[allow(dead_code)]
fn compute_corner_score(pixels: &[f32; 16], center: f32, threshold: f32) -> f32 {
    let mut score = 0.0;

    for &pixel in pixels.iter() {
        let diff = (pixel - center).abs();
        if diff > threshold {
            score += diff - threshold;
        }
    }

    score
}

/// Apply non-maximum suppression to corner scores
#[allow(dead_code)]
fn apply_non_max_suppression(corners: &Array2<f32>, windowsize: usize) -> Array2<f32> {
    let (height, width) = corners.dim();
    let mut result = Array2::zeros((height, width));
    let radius = windowsize / 2;

    for y in radius..(height - radius) {
        for x in radius..(width - radius) {
            let center_score = corners[[y, x]];
            if center_score == 0.0 {
                continue;
            }

            let mut is_maximum = true;

            'window: for dy in -(radius as i32)..=(radius as i32) {
                for dx in -(radius as i32)..=(radius as i32) {
                    if dx == 0 && dy == 0 {
                        continue;
                    }

                    let ny = (y as i32 + dy) as usize;
                    let nx = (x as i32 + dx) as usize;

                    if corners[[ny, nx]] > center_score {
                        is_maximum = false;
                        break 'window;
                    }
                }
            }

            if is_maximum {
                result[[y, x]] = center_score;
            }
        }
    }

    result
}

/// Simplified FAST corner detection with default parameters
///
/// Uses n_consecutive = 9 and applies non-maximum suppression by default.
///
/// # Arguments
///
/// * `img` - Input image
/// * `threshold` - Intensity difference threshold
///
/// # Returns
///
/// * Result containing corner points
#[allow(dead_code)]
pub fn fast_corners_simple(img: &DynamicImage, threshold: f32) -> Result<GrayImage> {
    fast_corners(img, threshold, 9, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn test_fast_corner_detection() {
        // First test the consecutive pixels function directly
        let pixels: [f32; 16] = [
            0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, // 8 bright pixels
            0.8, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, // 1 bright + 7 dark
        ];

        // This should detect 9 consecutive brighter pixels
        assert!(check_consecutive_pixels(&pixels, 0.5, 0.2, 9));

        // Test that we can detect a simple corner pattern
        // For now, just verify the algorithm runs without panicking
        let mut img = GrayImage::new(20, 20);

        // Create a simple high-contrast pattern
        for y in 0..20 {
            for x in 0..20 {
                if x < 10 && y < 10 {
                    img.put_pixel(x, y, Luma([255u8])); // bright top-left
                } else {
                    img.put_pixel(x, y, Luma([0u8])); // dark elsewhere
                }
            }
        }

        let dynamic_img = DynamicImage::ImageLuma8(img);
        let result = fast_corners(&dynamic_img, 50.0, 9, true);

        assert!(
            result.is_ok(),
            "FAST algorithm should complete without errors"
        );

        // The algorithm should at least produce a valid output image
        let corners = result.unwrap();
        assert_eq!(corners.dimensions(), (20, 20));
    }

    #[test]
    fn test_fast_parameters() {
        let img = GrayImage::new(20, 20);
        let dynamic_img = DynamicImage::ImageLuma8(img);

        // Test with different parameters
        let result1 = fast_corners(&dynamic_img, 20.0, 9, true);
        let result2 = fast_corners(&dynamic_img, 20.0, 12, false);

        assert!(result1.is_ok());
        assert!(result2.is_ok());
    }

    #[test]
    fn test_consecutive_pixel_check() {
        let pixels = [
            150.0, 150.0, 150.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
            150.0, 150.0, 150.0, 150.0,
        ];
        let center = 125.0;
        let threshold = 20.0;

        // Should find 9 consecutive darker pixels
        assert!(check_consecutive_pixels(&pixels, center, threshold, 9));

        // Shouldn't find 10 consecutive darker pixels
        assert!(!check_consecutive_pixels(&pixels, center, threshold, 10));
    }
}
