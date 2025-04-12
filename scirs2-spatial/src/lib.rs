//! Spatial algorithms module
//!
//! This module provides implementations of various spatial algorithms,
//! similar to SciPy's `spatial` module.
//!
//! ## Overview
//!
//! * Distance computations
//! * KD-trees for efficient spatial queries
//! * Voronoi diagrams and Delaunay triangulation
//! * Convex hulls
//! * Spatial transformations
//!
//! ## Examples
//!
//! ```
//! use scirs2_spatial::distance::euclidean;
//!
//! let point1 = &[1.0, 2.0, 3.0];
//! let point2 = &[4.0, 5.0, 6.0];
//!
//! let dist = euclidean(point1, point2);
//! println!("Euclidean distance: {}", dist);
//! ```

// Export error types
pub mod error;
pub use error::{SpatialError, SpatialResult};

// Distance metrics
pub mod distance;
pub use distance::*;

// KD-Tree for efficient nearest neighbor searches
pub mod kdtree;
pub use kdtree::KDTree;

// Voronoi diagrams and Delaunay triangulation
pub mod voronoi;

// Convex hulls
pub mod convex_hull;

// Utility functions
mod utils;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
