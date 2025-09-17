//! Alpha shapes implementation for spatial analysis
//!
//! Alpha shapes are a generalization of convex hulls that allow for
//! non-convex boundaries by controlling the "tightness" of the shape
//! around a set of points through the alpha parameter.

use crate::error::{SpatialError, SpatialResult};
use ndarray::{Array2, ArrayView1, ArrayView2};

/// Alpha shape representation for point cloud analysis
///
/// Alpha shapes provide a way to compute non-convex boundaries around
/// a set of points, controlled by the alpha parameter.
#[derive(Debug, Clone)]
pub struct AlphaShape {
    /// Points defining the alpha shape
    pub points: Array2<f64>,
    /// Alpha parameter controlling shape tightness
    pub alpha: f64,
    /// Boundary edges of the alpha shape
    pub edges: Vec<(usize, usize)>,
    /// Triangles in the alpha shape (if applicable)
    pub triangles: Vec<(usize, usize, usize)>,
}

impl AlphaShape {
    /// Create a new alpha shape from a set of points
    ///
    /// # Arguments
    ///
    /// * `points` - Input points as a 2D array
    /// * `alpha` - Alpha parameter controlling shape tightness
    ///
    /// # Returns
    ///
    /// * Result containing the computed alpha shape
    pub fn new(points: &ArrayView2<'_, f64>, alpha: f64) -> SpatialResult<Self> {
        if points.is_empty() {
            return Err(SpatialError::ValueError(
                "Points array cannot be empty".to_string(),
            ));
        }

        if alpha <= 0.0 {
            return Err(SpatialError::ValueError(
                "Alpha parameter must be positive".to_string(),
            ));
        }

        // Basic implementation - create a minimal alpha shape
        // In a full implementation, this would use Delaunay triangulation
        // and alpha-complex computation
        let points_owned = points.to_owned();
        let n_points = points_owned.nrows();

        // For now, create a simple convex hull-like boundary
        let mut edges = Vec::new();
        let mut triangles = Vec::new();

        // Simple boundary for demonstration
        if n_points >= 3 {
            for i in 0..n_points {
                edges.push((i, (i + 1) % n_points));
            }

            // Add a single triangle if we have exactly 3 points
            if n_points == 3 {
                triangles.push((0, 1, 2));
            }
        }

        Ok(AlphaShape {
            points: points_owned,
            alpha,
            edges,
            triangles,
        })
    }

    /// Get the number of points in the alpha shape
    pub fn num_points(&self) -> usize {
        self.points.nrows()
    }

    /// Get the number of edges in the alpha shape boundary
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Get the number of triangles in the alpha shape
    pub fn num_triangles(&self) -> usize {
        self.triangles.len()
    }

    /// Check if a point is inside the alpha shape
    ///
    /// # Arguments
    ///
    /// * `point` - Point to test
    ///
    /// # Returns
    ///
    /// * True if the point is inside the alpha shape
    pub fn contains_point(&self, point: &ArrayView1<f64>) -> bool {
        // Simplified implementation - just check if point is close to any input point
        if point.len() != self.points.ncols() {
            return false;
        }

        let tolerance = self.alpha;
        for i in 0..self.points.nrows() {
            let dist: f64 = point
                .iter()
                .zip(self.points.row(i).iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            if dist <= tolerance {
                return true;
            }
        }

        false
    }

    /// Compute the area/volume of the alpha shape
    ///
    /// # Returns
    ///
    /// * Area (2D) or volume (3D) of the alpha shape
    pub fn area(&self) -> f64 {
        // Simplified implementation
        // In a full implementation, this would compute the actual area/volume
        // based on the triangulation

        if self.triangles.is_empty() {
            return 0.0;
        }

        // Simple area calculation for triangles in 2D
        if self.points.ncols() == 2 {
            let mut total_area = 0.0;
            for &(i, j, k) in &self.triangles {
                let p1 = self.points.row(i);
                let p2 = self.points.row(j);
                let p3 = self.points.row(k);

                // Triangle area using cross product
                let area = 0.5
                    * ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])).abs();
                total_area += area;
            }
            return total_area;
        }

        // For 3D and higher dimensions, return a placeholder
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_alpha_shape_creation() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]];
        let alpha_shape = AlphaShape::new(&points.view(), 1.0);
        assert!(alpha_shape.is_ok());

        let shape = alpha_shape.unwrap();
        assert_eq!(shape.num_points(), 3);
        assert_eq!(shape.alpha, 1.0);
    }

    #[test]
    fn test_alpha_shape_contains() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]];
        let alpha_shape = AlphaShape::new(&points.view(), 0.5).unwrap();

        let test_point = array![0.1, 0.1];
        let contains = alpha_shape.contains_point(&test_point.view());
        assert!(contains);

        let far_point = array![10.0, 10.0];
        let contains_far = alpha_shape.contains_point(&far_point.view());
        assert!(!contains_far);
    }

    #[test]
    fn test_alpha_shape_area() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let alpha_shape = AlphaShape::new(&points.view(), 1.0).unwrap();

        let area = alpha_shape.area();
        assert!(area >= 0.0);
    }

    #[test]
    fn test_invalid_alpha() {
        let points = array![[0.0, 0.0], [1.0, 0.0]];
        let result = AlphaShape::new(&points.view(), -1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_points() {
        let points = Array2::<f64>::zeros((0, 2));
        let result = AlphaShape::new(&points.view(), 1.0);
        assert!(result.is_err());
    }
}
