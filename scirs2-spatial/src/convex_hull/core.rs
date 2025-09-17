//! Core ConvexHull structure and basic methods
//!
//! This module contains the main ConvexHull struct and its fundamental
//! operations, providing a foundation for convex hull computations.

use crate::error::{SpatialError, SpatialResult};
use ndarray::{Array2, ArrayView2};
use qhull::Qh;

/// Algorithms available for computing convex hulls
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConvexHullAlgorithm {
    /// Use QHull (default) - works for any dimension
    #[default]
    QHull,
    /// Graham scan algorithm - only for 2D
    GrahamScan,
    /// Jarvis march (gift wrapping) algorithm - only for 2D
    JarvisMarch,
}

/// A ConvexHull represents the convex hull of a set of points.
///
/// It provides methods to access hull properties, check if points are
/// inside the hull, and access hull facets and vertices.
pub struct ConvexHull {
    /// Input points
    pub(crate) points: Array2<f64>,
    /// QHull instance
    #[allow(dead_code)]
    pub(crate) qh: Qh<'static>,
    /// Vertex indices of the convex hull (indices into the original points array)
    pub(crate) vertex_indices: Vec<usize>,
    /// Simplex indices (facets) of the convex hull
    pub(crate) simplices: Vec<Vec<usize>>,
    /// Equations of the hull facets (shape: n_facets x (n_dim+1))
    pub(crate) equations: Option<Array2<f64>>,
}

impl ConvexHull {
    /// Create a new ConvexHull from a set of points.
    ///
    /// # Arguments
    ///
    /// * `points` - Input points (shape: npoints x n_dim)
    ///
    /// # Returns
    ///
    /// * Result containing a ConvexHull instance or an error
    ///
    /// # Errors
    ///
    /// * Returns error if hull computation fails or input is invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::convex_hull::ConvexHull;
    /// use ndarray::array;
    ///
    /// let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]];
    /// let hull = ConvexHull::new(&points.view()).unwrap();
    /// ```
    pub fn new(points: &ArrayView2<'_, f64>) -> SpatialResult<Self> {
        Self::new_with_algorithm(points, ConvexHullAlgorithm::default())
    }

    /// Create a new ConvexHull from a set of points using a specific algorithm.
    ///
    /// # Arguments
    ///
    /// * `points` - Input points (shape: npoints x n_dim)
    /// * `algorithm` - Algorithm to use for convex hull computation
    ///
    /// # Returns
    ///
    /// * Result containing a ConvexHull instance or an error
    ///
    /// # Errors
    ///
    /// * Returns error if hull computation fails or input is invalid
    /// * Returns error if algorithm is not supported for the given dimensionality
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::convex_hull::{ConvexHull, ConvexHullAlgorithm};
    /// use ndarray::array;
    ///
    /// let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]];
    /// let hull = ConvexHull::new_with_algorithm(&points.view(), ConvexHullAlgorithm::GrahamScan).unwrap();
    /// ```
    pub fn new_with_algorithm(
        points: &ArrayView2<'_, f64>,
        algorithm: ConvexHullAlgorithm,
    ) -> SpatialResult<Self> {
        let npoints = points.nrows();
        let ndim = points.ncols();

        // For 1D, allow at least 1 point (degenerate case)
        // For 2D, need at least 3 points for a proper hull, but allow 2 for degenerate line case
        // For 3D and higher, require at least ndim + 1 points
        let min_points = match ndim {
            1 => 1, // Allow single points in 1D
            2 => 3, // Need at least 3 points for a 2D convex hull
            _ => ndim + 1,
        };

        if npoints < min_points {
            return Err(SpatialError::ValueError(format!(
                "Need at least {} points to construct a {}D convex hull",
                min_points, ndim
            )));
        }

        // Check if algorithm is compatible with dimensionality
        match algorithm {
            ConvexHullAlgorithm::GrahamScan | ConvexHullAlgorithm::JarvisMarch => {
                if ndim != 2 {
                    return Err(SpatialError::ValueError(format!(
                        "{algorithm:?} algorithm only supports 2D points, got {ndim}D"
                    )));
                }
            }
            ConvexHullAlgorithm::QHull => {
                // QHull supports any dimension
            }
        }

        match algorithm {
            ConvexHullAlgorithm::GrahamScan => {
                crate::convex_hull::algorithms::graham_scan::compute_graham_scan(points)
            }
            ConvexHullAlgorithm::JarvisMarch => {
                crate::convex_hull::algorithms::jarvis_march::compute_jarvis_march(points)
            }
            ConvexHullAlgorithm::QHull => {
                crate::convex_hull::algorithms::qhull::compute_qhull(points)
            }
        }
    }

    /// Get the vertices of the convex hull
    ///
    /// # Returns
    ///
    /// * Array of vertices of the convex hull
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::convex_hull::ConvexHull;
    /// use ndarray::array;
    ///
    /// let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]];
    /// let hull = ConvexHull::new(&points.view()).unwrap();
    ///
    /// // The hull vertices should be the corners, not the interior point
    /// // The number of vertices can vary depending on QHull implementation
    /// assert!(hull.vertices().len() >= 3);
    /// ```
    pub fn vertices(&self) -> Vec<Vec<f64>> {
        self.vertex_indices
            .iter()
            .map(|&idx| self.points.row(idx).to_vec())
            .collect()
    }

    /// Get the vertices of the convex hull as an Array2
    ///
    /// # Returns
    ///
    /// * Array2 of shape (n_vertices, n_dim) containing the vertices of the convex hull
    pub fn vertices_array(&self) -> Array2<f64> {
        let ndim = self.points.ncols();
        let n_vertices = self.vertex_indices.len();
        let mut vertices = Array2::zeros((n_vertices, ndim));

        for (i, &idx) in self.vertex_indices.iter().enumerate() {
            // Make sure the index is valid
            if idx < self.points.nrows() {
                for j in 0..ndim {
                    vertices[[i, j]] = self.points[[idx, j]];
                }
            } else {
                // If we have an invalid index (shouldn't happen, but for safety)
                for j in 0..ndim {
                    vertices[[i, j]] = 0.0;
                }
            }
        }

        vertices
    }

    /// Get the indices of the vertices of the convex hull
    ///
    /// # Returns
    ///
    /// * Vector of indices of the hull vertices (into the original points array)
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::convex_hull::ConvexHull;
    /// use ndarray::array;
    ///
    /// let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]];
    /// let hull = ConvexHull::new(&points.view()).unwrap();
    ///
    /// // Get the vertex indices of the hull
    /// let vertices = hull.vertex_indices();
    /// println!("Convex hull vertices: {:?}", vertices);
    /// ```
    pub fn vertex_indices(&self) -> &[usize] {
        &self.vertex_indices
    }

    /// Get the simplices (facets) of the convex hull
    ///
    /// # Returns
    ///
    /// * Vector of simplices, where each simplex is a vector of vertex indices
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::convex_hull::ConvexHull;
    /// use ndarray::array;
    ///
    /// let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]];
    /// let hull = ConvexHull::new(&points.view()).unwrap();
    ///
    /// // For a 2D hull, examine the simplices
    /// for simplex in hull.simplices() {
    ///     // Print each simplex
    ///     println!("Simplex: {:?}", simplex);
    /// }
    /// ```
    pub fn simplices(&self) -> &[Vec<usize>] {
        &self.simplices
    }

    /// Get the dimensionality of the convex hull
    ///
    /// # Returns
    ///
    /// * Number of dimensions of the convex hull
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::convex_hull::ConvexHull;
    /// use ndarray::array;
    ///
    /// let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]];
    /// let hull = ConvexHull::new(&points.view()).unwrap();
    ///
    /// assert_eq!(hull.ndim(), 2);
    /// ```
    pub fn ndim(&self) -> usize {
        self.points.ncols()
    }

    /// Check if a point is inside the convex hull
    ///
    /// # Arguments
    ///
    /// * `point` - The point to check
    ///
    /// # Returns
    ///
    /// * Result containing a boolean indicating if the point is inside the hull
    ///
    /// # Errors
    ///
    /// * Returns error if point dimension doesn't match hull dimension
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::convex_hull::ConvexHull;
    /// use ndarray::array;
    ///
    /// let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    /// let hull = ConvexHull::new(&points.view()).unwrap();
    ///
    /// // Check a point inside the hull
    /// // Result may vary depending on QHull implementation and special case handling
    /// let inside = hull.contains(&[0.25, 0.25]).unwrap();
    /// println!("Point [0.25, 0.25] inside: {}", inside);
    ///
    /// // Check a point outside the hull
    /// assert!(!hull.contains(&[2.0, 2.0]).unwrap());
    /// ```
    pub fn contains<T: AsRef<[f64]>>(&self, point: T) -> SpatialResult<bool> {
        crate::convex_hull::properties::containment::check_point_containment(self, point)
    }

    /// Get the volume of the convex hull
    ///
    /// For 2D hulls, this returns the area. For 3D hulls, this returns the volume.
    /// For higher dimensions, this returns the hypervolume.
    ///
    /// # Returns
    ///
    /// * Result containing the volume/area of the convex hull
    ///
    /// # Errors
    ///
    /// * Returns error if volume computation fails
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::convex_hull::ConvexHull;
    /// use ndarray::array;
    ///
    /// // Create a square with area 1
    /// let points = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    /// let hull = ConvexHull::new(&points.view()).unwrap();
    ///
    /// let area = hull.volume().unwrap();
    /// assert!((area - 1.0).abs() < 1e-10);
    /// ```
    pub fn volume(&self) -> SpatialResult<f64> {
        crate::convex_hull::properties::volume::compute_volume(self)
    }

    /// Get the surface area of the convex hull
    ///
    /// For 2D hulls, this returns the perimeter. For 3D hulls, this returns the surface area.
    /// For higher dimensions, this returns the surface hypervolume.
    ///
    /// # Returns
    ///
    /// * Result containing the surface area/perimeter of the convex hull
    ///
    /// # Errors
    ///
    /// * Returns error if area computation fails
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::convex_hull::ConvexHull;
    /// use ndarray::array;
    ///
    /// // Create a 3D cube
    /// let points = array![
    ///     [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
    ///     [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]
    /// ];
    /// let hull = ConvexHull::new(&points.view()).unwrap();
    ///
    /// // Surface area of the cube should be 6 square units
    /// let area = hull.area().unwrap();
    /// assert!((area - 6.0).abs() < 1e-10);
    /// ```
    pub fn area(&self) -> SpatialResult<f64> {
        crate::convex_hull::properties::surface_area::compute_surface_area(self)
    }

    /// Extract facet equations from the Qhull instance
    ///
    /// # Arguments
    ///
    /// * `qh` - Qhull instance
    /// * `ndim` - Dimensionality of the hull
    ///
    /// # Returns
    ///
    /// * Option containing an Array2 of shape (n_facets, ndim+1) with the equations of the hull facets
    ///   or None if equations cannot be extracted
    pub fn extract_equations(qh: &Qh, ndim: usize) -> Option<Array2<f64>> {
        // Get facets from qhull
        let facets: Vec<_> = qh.facets().collect();
        let n_facets = facets.len();

        // Allocate array for equations
        let mut equations = Array2::zeros((n_facets, ndim + 1));

        // Extract facet equations
        for (i, facet) in facets.iter().enumerate() {
            // Qhull facet equation format: normal coefficients followed by offset
            // Ax + By + Cz + offset <= 0 for points inside the hull
            if let Some(normal) = facet.normal() {
                // Fill in normal coefficients
                for j in 0..ndim {
                    equations[[i, j]] = normal[j];
                }

                // Fill in offset (last column)
                equations[[i, ndim]] = facet.offset();
            } else {
                // If we can't get a facet's equation, we can't provide all equations
                return None;
            }
        }

        Some(equations)
    }
}
