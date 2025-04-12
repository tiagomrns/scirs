//! Clustering algorithms module for SciRS2
//!
//! This module provides implementations of various clustering algorithms such as:
//! - Vector quantization (k-means, etc.)
//! - Hierarchical clustering
//! - Density-based clustering (DBSCAN, etc.)
//!
//! ## Features
//!
//! * **Vector Quantization**: K-means and K-means++ for partitioning data
//! * **Hierarchical Clustering**: Agglomerative clustering with various linkage methods
//! * **Density-based Clustering**: DBSCAN for finding clusters of arbitrary shape
//!
//! ## Examples
//!
//! ```
//! use ndarray::{Array2, ArrayView2};
//! use scirs2_cluster::vq::kmeans;
//!
//! // Example data with two clusters
//! let data = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 2.0,
//!     1.2, 1.8,
//!     0.8, 1.9,
//!     3.7, 4.2,
//!     3.9, 3.9,
//!     4.2, 4.1,
//! ]).unwrap();
//!
//! // Run k-means with k=2
//! let (centroids, labels) = kmeans(data.view(), 2, None).unwrap();
//!
//! // Print the results
//! println!("Centroids: {:?}", centroids);
//! println!("Cluster assignments: {:?}", labels);
//! ```

#![warn(missing_docs)]

pub mod density;
pub mod error;
pub mod hierarchy;
pub mod vq;

// Re-exports
pub use density::*;
pub use hierarchy::*;
pub use vq::*;
