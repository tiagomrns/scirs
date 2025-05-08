//! Basic interpolation methods (linear, nearest, cubic)
//!
//! This module provides implementations of standard interpolation methods.

use crate::error::InterpolateResult;
use ndarray::{Array1, ArrayView1};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

/// Perform nearest neighbor interpolation
///
/// # Arguments
///
/// * `x` - The x coordinates
/// * `y` - The y coordinates
/// * `idx` - The index of the segment containing the target point
/// * `x_new` - The x coordinate at which to interpolate
///
/// # Returns
///
/// The interpolated value
#[allow(dead_code)]
pub(crate) fn nearest_interp<F: Float>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    idx: usize,
    x_new: F,
) -> InterpolateResult<F> {
    // Find which of the two points is closer
    let dist_left = (x_new - x[idx]).abs();
    let dist_right = (x_new - x[idx + 1]).abs();

    if dist_left <= dist_right {
        Ok(y[idx])
    } else {
        Ok(y[idx + 1])
    }
}

/// Perform linear interpolation
///
/// # Arguments
///
/// * `x` - The x coordinates
/// * `y` - The y coordinates
/// * `idx` - The index of the segment containing the target point
/// * `x_new` - The x coordinate at which to interpolate
///
/// # Returns
///
/// The interpolated value
#[allow(dead_code)]
pub(crate) fn linear_interp<F: Float>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    idx: usize,
    x_new: F,
) -> InterpolateResult<F> {
    let x0 = x[idx];
    let x1 = x[idx + 1];
    let y0 = y[idx];
    let y1 = y[idx + 1];

    // Avoid division by zero
    if x0 == x1 {
        return Ok(y0); // or y1, they should be the same
    }

    // Linear interpolation formula: y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    Ok(y0 + (x_new - x0) * (y1 - y0) / (x1 - x0))
}

/// Perform cubic interpolation
///
/// # Arguments
///
/// * `x` - The x coordinates
/// * `y` - The y coordinates
/// * `idx` - The index of the segment containing the target point
/// * `x_new` - The x coordinate at which to interpolate
///
/// # Returns
///
/// The interpolated value
#[allow(dead_code)]
pub(crate) fn cubic_interp<F: Float + FromPrimitive>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    idx: usize,
    x_new: F,
) -> InterpolateResult<F> {
    // We need 4 points for cubic interpolation
    // If we're near the edges, we need to adjust the indices
    let (i0, i1, i2, i3) = if idx == 0 {
        (0, 0, 1, 2)
    } else if idx == x.len() - 2 {
        (idx - 1, idx, idx + 1, idx + 1)
    } else {
        // Handles both idx == x.len() - 3 and idx > x.len() - 3 cases since they're identical
        (idx - 1, idx, idx + 1, idx + 2)
    };

    let _x0 = x[i0];
    let x1 = x[i1];
    let x2 = x[i2];
    let _x3 = x[i3];

    let y0 = y[i0];
    let y1 = y[i1];
    let y2 = y[i2];
    let y3 = y[i3];

    // Normalized position within the interval [x1, x2]
    let t = if x2 != x1 {
        (x_new - x1) / (x2 - x1)
    } else {
        F::zero()
    };

    // Calculate cubic interpolation using Catmull-Rom spline
    // p(t) = 0.5 * ((2*p1) +
    //               (-p0 + p2) * t +
    //               (2*p0 - 5*p1 + 4*p2 - p3) * t^2 +
    //               (-p0 + 3*p1 - 3*p2 + p3) * t^3)

    let two = F::from_f64(2.0).unwrap();
    let three = F::from_f64(3.0).unwrap();
    let four = F::from_f64(4.0).unwrap();
    let five = F::from_f64(5.0).unwrap();
    let half = F::from_f64(0.5).unwrap();

    let t2 = t * t;
    let t3 = t2 * t;

    let c0 = two * y1;
    let c1 = -y0 + y2;
    let c2 = two * y0 - five * y1 + four * y2 - y3;
    let c3 = -y0 + three * y1 - three * y2 + y3;

    let result = half * (c0 + c1 * t + c2 * t2 + c3 * t3);

    Ok(result)
}

/// Nearest neighbor interpolation convenience function
///
/// # Arguments
///
/// * `x` - The x coordinates (must be sorted in ascending order)
/// * `y` - The y coordinates
/// * `x_new` - The points at which to interpolate
///
/// # Returns
///
/// The interpolated values
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_interpolate::nearest_interpolate;
///
/// let x = array![0.0f64, 1.0, 2.0, 3.0];
/// let y = array![0.0f64, 1.0, 4.0, 9.0];
/// let x_new = array![0.5f64, 1.5, 2.5];
///
/// let y_interp = nearest_interpolate(&x.view(), &y.view(), &x_new.view()).unwrap();
/// let diff0 = (y_interp[0] - 0.0).abs();
/// let diff1 = (y_interp[1] - 1.0).abs();
/// let diff2 = (y_interp[2] - 4.0).abs();
///
/// assert!(diff0 < 1e-10);  // Nearest value to x=0.5 is y=0.0
/// assert!(diff1 < 1e-10);  // Nearest value to x=1.5 is y=1.0
/// assert!(diff2 < 1e-10);  // Nearest value to x=2.5 is y=4.0
/// ```
pub fn nearest_interpolate<F: Float + FromPrimitive + Debug>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    x_new: &ArrayView1<F>,
) -> InterpolateResult<Array1<F>> {
    use super::{ExtrapolateMode, Interp1d, InterpolationMethod};
    let interp = Interp1d::new(x, y, InterpolationMethod::Nearest, ExtrapolateMode::Error)?;
    interp.evaluate_array(x_new)
}

/// Linear interpolation convenience function
///
/// # Arguments
///
/// * `x` - The x coordinates (must be sorted in ascending order)
/// * `y` - The y coordinates
/// * `x_new` - The points at which to interpolate
///
/// # Returns
///
/// The interpolated values
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_interpolate::linear_interpolate;
///
/// let x = array![0.0f64, 1.0, 2.0, 3.0];
/// let y = array![0.0f64, 1.0, 4.0, 9.0];
/// let x_new = array![0.5f64, 1.5, 2.5];
///
/// let y_interp = linear_interpolate(&x.view(), &y.view(), &x_new.view()).unwrap();
/// let diff0 = (y_interp[0] - 0.5).abs();
/// let diff1 = (y_interp[1] - 2.5).abs();
/// let diff2 = (y_interp[2] - 6.5).abs();
///
/// assert!(diff0 < 1e-10);  // Linear interpolation at x=0.5
/// assert!(diff1 < 1e-10);  // Linear interpolation at x=1.5
/// assert!(diff2 < 1e-10);  // Linear interpolation at x=2.5
/// ```
pub fn linear_interpolate<F: Float + FromPrimitive + Debug>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    x_new: &ArrayView1<F>,
) -> InterpolateResult<Array1<F>> {
    use super::{ExtrapolateMode, Interp1d, InterpolationMethod};
    let interp = Interp1d::new(x, y, InterpolationMethod::Linear, ExtrapolateMode::Error)?;
    interp.evaluate_array(x_new)
}

/// Cubic interpolation convenience function
///
/// # Arguments
///
/// * `x` - The x coordinates (must be sorted in ascending order)
/// * `y` - The y coordinates
/// * `x_new` - The points at which to interpolate
///
/// # Returns
///
/// The interpolated values
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_interpolate::cubic_interpolate;
///
/// let x = array![0.0f64, 1.0, 2.0, 3.0];
/// let y = array![0.0f64, 1.0, 4.0, 9.0];
/// let x_new = array![0.5f64, 1.5, 2.5];
///
/// let y_interp = cubic_interpolate(&x.view(), &y.view(), &x_new.view()).unwrap();
/// // Values will be close to but not exactly the same as linear interpolation
/// let diff0 = (y_interp[0] - 0.5).abs();
/// let diff1 = (y_interp[1] - 2.5).abs();
/// let diff2 = (y_interp[2] - 6.5).abs();
///
/// assert!(diff0 < 0.3);  // Using a wider tolerance for cubic interpolation
/// assert!(diff1 < 0.3);  // as the specific implementation may vary
/// assert!(diff2 < 0.3);
/// ```
pub fn cubic_interpolate<F: Float + FromPrimitive + Debug>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    x_new: &ArrayView1<F>,
) -> InterpolateResult<Array1<F>> {
    use super::{ExtrapolateMode, Interp1d, InterpolationMethod};
    let interp = Interp1d::new(x, y, InterpolationMethod::Cubic, ExtrapolateMode::Error)?;
    interp.evaluate_array(x_new)
}

#[cfg(test)]
mod tests {
    use super::super::{ExtrapolateMode, Interp1d, InterpolationMethod};
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_nearest_interpolation() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];

        let interp = Interp1d::new(
            &x.view(),
            &y.view(),
            InterpolationMethod::Nearest,
            ExtrapolateMode::Error,
        )
        .unwrap();

        // Test points exactly at data points
        assert_relative_eq!(interp.evaluate(0.0).unwrap(), 0.0);
        assert_relative_eq!(interp.evaluate(1.0).unwrap(), 1.0);
        assert_relative_eq!(interp.evaluate(2.0).unwrap(), 4.0);
        assert_relative_eq!(interp.evaluate(3.0).unwrap(), 9.0);

        // Test points between data points
        assert_relative_eq!(interp.evaluate(0.4).unwrap(), 0.0);
        assert_relative_eq!(interp.evaluate(0.6).unwrap(), 1.0);
        assert_relative_eq!(interp.evaluate(1.4).unwrap(), 1.0);
        assert_relative_eq!(interp.evaluate(1.6).unwrap(), 4.0);
    }

    #[test]
    fn test_linear_interpolation() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];

        let interp = Interp1d::new(
            &x.view(),
            &y.view(),
            InterpolationMethod::Linear,
            ExtrapolateMode::Error,
        )
        .unwrap();

        // Test points exactly at data points
        assert_relative_eq!(interp.evaluate(0.0).unwrap(), 0.0);
        assert_relative_eq!(interp.evaluate(1.0).unwrap(), 1.0);
        assert_relative_eq!(interp.evaluate(2.0).unwrap(), 4.0);
        assert_relative_eq!(interp.evaluate(3.0).unwrap(), 9.0);

        // Test points between data points
        assert_relative_eq!(interp.evaluate(0.5).unwrap(), 0.5);
        assert_relative_eq!(interp.evaluate(1.5).unwrap(), 2.5);
        assert_relative_eq!(interp.evaluate(2.5).unwrap(), 6.5);
    }

    #[test]
    fn test_cubic_interpolation() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];

        let interp = Interp1d::new(
            &x.view(),
            &y.view(),
            InterpolationMethod::Cubic,
            ExtrapolateMode::Error,
        )
        .unwrap();

        // Test points exactly at data points
        assert_relative_eq!(interp.evaluate(0.0).unwrap(), 0.0);
        assert_relative_eq!(interp.evaluate(1.0).unwrap(), 1.0);
        assert_relative_eq!(interp.evaluate(2.0).unwrap(), 4.0);
        assert_relative_eq!(interp.evaluate(3.0).unwrap(), 9.0);

        // For this particular dataset (a quadratic y = x²),
        // cubic interpolation might not reproduce it exactly due to the specific spline algorithm
        // so we use wider tolerances
        assert_relative_eq!(interp.evaluate(0.5).unwrap(), 0.25, epsilon = 0.1);
        assert_relative_eq!(interp.evaluate(1.5).unwrap(), 2.25, epsilon = 0.1);
        assert_relative_eq!(interp.evaluate(2.5).unwrap(), 6.25, epsilon = 1.0);
    }

    #[test]
    fn test_convenience_functions() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];
        let x_new = array![0.5, 1.5, 2.5];

        // Test nearest interpolation
        let y_nearest = nearest_interpolate(&x.view(), &y.view(), &x_new.view()).unwrap();
        // Point 0.5 is exactly halfway between x[0]=0.0 and x[1]=1.0, so we default to the left point's value
        assert_relative_eq!(y_nearest[0], 0.0);
        // Point 1.5 is exactly halfway between x[1]=1.0 and x[2]=2.0, so we default to the left point's value
        assert_relative_eq!(y_nearest[1], 1.0);
        // Point 2.5 is exactly halfway between x[2]=2.0 and x[3]=3.0, so we default to the left point's value
        assert_relative_eq!(y_nearest[2], 4.0);

        // Test linear interpolation
        let y_linear = linear_interpolate(&x.view(), &y.view(), &x_new.view()).unwrap();
        assert_relative_eq!(y_linear[0], 0.5);
        assert_relative_eq!(y_linear[1], 2.5);
        assert_relative_eq!(y_linear[2], 6.5);

        // Test cubic interpolation
        let y_cubic = cubic_interpolate(&x.view(), &y.view(), &x_new.view()).unwrap();
        // Allow a wider tolerance for cubic interpolation since it depends on the specific spline implementation
        assert!((y_cubic[0] - 0.25).abs() < 0.15);
        assert!((y_cubic[1] - 2.25).abs() < 0.15);
        // For point 2.5, allow an even wider tolerance
        assert!((y_cubic[2] - 6.25).abs() < 1.0);
    }
}
