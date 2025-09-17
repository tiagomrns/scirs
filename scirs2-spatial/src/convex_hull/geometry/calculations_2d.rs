//! 2D geometric calculations for convex hull operations
//!
//! This module provides utility functions for 2D geometric computations
//! commonly used in convex hull algorithms.

use crate::error::SpatialResult;
use ndarray::{Array2, ArrayView2};

/// Compute cross product for three 2D points (returns z-component of 3D cross product)
///
/// # Arguments
///
/// * `p1` - First point [x, y]
/// * `p2` - Second point [x, y]
/// * `p3` - Third point [x, y]
///
/// # Returns
///
/// * Cross product value. Positive indicates counterclockwise turn,
///   negative indicates clockwise turn, zero indicates collinear points.
///
/// # Examples
///
/// ```
/// use scirs2_spatial::convex_hull::geometry::calculations_2d::cross_product_2d;
///
/// let p1 = [0.0, 0.0];
/// let p2 = [1.0, 0.0];
/// let p3 = [0.0, 1.0];
///
/// let cross = cross_product_2d(p1, p2, p3);
/// assert!(cross > 0.0); // Counterclockwise turn
/// ```
pub fn cross_product_2d(p1: [f64; 2], p2: [f64; 2], p3: [f64; 2]) -> f64 {
    (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
}

/// Compute squared distance between two 2D points
///
/// # Arguments
///
/// * `p1` - First point [x, y]
/// * `p2` - Second point [x, y]
///
/// # Returns
///
/// * Squared Euclidean distance between the points
///
/// # Examples
///
/// ```
/// use scirs2_spatial::convex_hull::geometry::calculations_2d::distance_squared_2d;
///
/// let p1 = [0.0, 0.0];
/// let p2 = [3.0, 4.0];
///
/// let dist_sq = distance_squared_2d(p1, p2);
/// assert_eq!(dist_sq, 25.0); // 3² + 4² = 25
/// ```
pub fn distance_squared_2d(p1: [f64; 2], p2: [f64; 2]) -> f64 {
    (p2[0] - p1[0]).powi(2) + (p2[1] - p1[1]).powi(2)
}

/// Compute facet equations for a 2D convex hull
///
/// Each equation represents a line in the form: ax + by + c = 0
/// where (a, b) is the normal vector and c is the offset.
///
/// # Arguments
///
/// * `points` - Input points array
/// * `vertex_indices` - Indices of hull vertices in counterclockwise order
///
/// # Returns
///
/// * Array2 of shape (n_edges, 3) containing line equations [a, b, c]
///
/// # Examples
///
/// ```
/// use scirs2_spatial::convex_hull::geometry::calculations_2d::compute_2d_hull_equations;
/// use ndarray::array;
///
/// let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
/// let vertices = vec![0, 1, 2];
///
/// let equations = compute_2d_hull_equations(&points.view(), &vertices);
/// assert_eq!(equations.nrows(), 3); // Three edges
/// assert_eq!(equations.ncols(), 3); // [a, b, c] format
/// ```
pub fn compute_2d_hull_equations(
    points: &ArrayView2<'_, f64>,
    vertex_indices: &[usize],
) -> Array2<f64> {
    let n = vertex_indices.len();
    let mut equations = Array2::zeros((n, 3)); // 2D equations: ax + by + c = 0

    for i in 0..n {
        let j = (i + 1) % n;
        let p1 = [
            points[[vertex_indices[i], 0]],
            points[[vertex_indices[i], 1]],
        ];
        let p2 = [
            points[[vertex_indices[j], 0]],
            points[[vertex_indices[j], 1]],
        ];

        // Line equation: (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0
        let a = p2[1] - p1[1];
        let b = p1[0] - p2[0];
        let c = (p2[0] - p1[0]) * p1[1] - (p2[1] - p1[1]) * p1[0];

        equations[[i, 0]] = a;
        equations[[i, 1]] = b;
        equations[[i, 2]] = c;
    }

    equations
}

/// Compute the area of a 2D polygon using the shoelace formula
///
/// # Arguments
///
/// * `points` - Input points array
/// * `vertex_indices` - Indices of polygon vertices in order
///
/// # Returns
///
/// * Result containing the polygon area
///
/// # Examples
///
/// ```
/// use scirs2_spatial::convex_hull::geometry::calculations_2d::compute_polygon_area;
/// use ndarray::array;
///
/// // Unit square
/// let points = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
/// let vertices = vec![0, 1, 2, 3];
///
/// let area = compute_polygon_area(&points.view(), &vertices).unwrap();
/// assert!((area - 1.0).abs() < 1e-10);
/// ```
pub fn compute_polygon_area(
    points: &ArrayView2<'_, f64>,
    vertex_indices: &[usize],
) -> SpatialResult<f64> {
    if vertex_indices.len() < 3 {
        return Ok(0.0);
    }

    let mut area = 0.0;
    let n = vertex_indices.len();

    for i in 0..n {
        let j = (i + 1) % n;
        let xi = points[[vertex_indices[i], 0]];
        let yi = points[[vertex_indices[i], 1]];
        let xj = points[[vertex_indices[j], 0]];
        let yj = points[[vertex_indices[j], 1]];

        area += xi * yj - xj * yi;
    }

    Ok(area.abs() / 2.0)
}

/// Compute the perimeter of a 2D polygon
///
/// # Arguments
///
/// * `points` - Input points array
/// * `vertex_indices` - Indices of polygon vertices in order
///
/// # Returns
///
/// * Result containing the polygon perimeter
///
/// # Examples
///
/// ```
/// use scirs2_spatial::convex_hull::geometry::calculations_2d::compute_polygon_perimeter;
/// use ndarray::array;
///
/// // Unit square
/// let points = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
/// let vertices = vec![0, 1, 2, 3];
///
/// let perimeter = compute_polygon_perimeter(&points.view(), &vertices).unwrap();
/// assert!((perimeter - 4.0).abs() < 1e-10);
/// ```
pub fn compute_polygon_perimeter(
    points: &ArrayView2<'_, f64>,
    vertex_indices: &[usize],
) -> SpatialResult<f64> {
    if vertex_indices.len() < 2 {
        return Ok(0.0);
    }

    let mut perimeter = 0.0;
    let n = vertex_indices.len();

    for i in 0..n {
        let j = (i + 1) % n;
        let xi = points[[vertex_indices[i], 0]];
        let yi = points[[vertex_indices[i], 1]];
        let xj = points[[vertex_indices[j], 0]];
        let yj = points[[vertex_indices[j], 1]];

        let dx = xj - xi;
        let dy = yj - yi;
        perimeter += (dx * dx + dy * dy).sqrt();
    }

    Ok(perimeter)
}

/// Check if three points are ordered counterclockwise
///
/// # Arguments
///
/// * `p1` - First point [x, y]
/// * `p2` - Second point [x, y]
/// * `p3` - Third point [x, y]
///
/// # Returns
///
/// * true if points are in counterclockwise order, false otherwise
///
/// # Examples
///
/// ```
/// use scirs2_spatial::convex_hull::geometry::calculations_2d::is_counterclockwise;
///
/// let p1 = [0.0, 0.0];
/// let p2 = [1.0, 0.0];
/// let p3 = [0.0, 1.0];
///
/// assert!(is_counterclockwise(p1, p2, p3));
/// assert!(!is_counterclockwise(p1, p3, p2));
/// ```
pub fn is_counterclockwise(p1: [f64; 2], p2: [f64; 2], p3: [f64; 2]) -> bool {
    cross_product_2d(p1, p2, p3) > 0.0
}

/// Calculate polar angle from a reference point
///
/// # Arguments
///
/// * `reference` - Reference point [x, y]
/// * `point` - Target point [x, y]
///
/// # Returns
///
/// * Polar angle in radians
///
/// # Examples
///
/// ```
/// use scirs2_spatial::convex_hull::geometry::calculations_2d::polar_angle;
///
/// let origin = [0.0, 0.0];
/// let point = [1.0, 1.0];
///
/// let angle = polar_angle(origin, point);
/// assert!((angle - std::f64::consts::PI / 4.0).abs() < 1e-10);
/// ```
pub fn polar_angle(reference: [f64; 2], point: [f64; 2]) -> f64 {
    (point[1] - reference[1]).atan2(point[0] - reference[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_cross_product_2d() {
        let p1 = [0.0, 0.0];
        let p2 = [1.0, 0.0];
        let p3 = [0.0, 1.0];

        let cross = cross_product_2d(p1, p2, p3);
        assert!(cross > 0.0); // Counterclockwise

        let cross_cw = cross_product_2d(p1, p3, p2);
        assert!(cross_cw < 0.0); // Clockwise
    }

    #[test]
    fn test_distance_squared_2d() {
        let p1 = [0.0, 0.0];
        let p2 = [3.0, 4.0];

        let dist_sq = distance_squared_2d(p1, p2);
        assert_eq!(dist_sq, 25.0);
    }

    #[test]
    fn test_polygon_area() {
        // Unit square
        let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        let vertices = vec![0, 1, 2, 3];

        let area = compute_polygon_area(&points.view(), &vertices).unwrap();
        assert!((area - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_polygon_perimeter() {
        // Unit square
        let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        let vertices = vec![0, 1, 2, 3];

        let perimeter = compute_polygon_perimeter(&points.view(), &vertices).unwrap();
        assert!((perimeter - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_counterclockwise() {
        let p1 = [0.0, 0.0];
        let p2 = [1.0, 0.0];
        let p3 = [0.0, 1.0];

        assert!(is_counterclockwise(p1, p2, p3));
        assert!(!is_counterclockwise(p1, p3, p2));
    }

    #[test]
    fn test_polar_angle() {
        let origin = [0.0, 0.0];
        let point = [1.0, 1.0];

        let angle = polar_angle(origin, point);
        assert!((angle - std::f64::consts::PI / 4.0).abs() < 1e-10);
    }
}
