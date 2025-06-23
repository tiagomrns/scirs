//! SIMD-accelerated distance computations for clustering algorithms
//!
//! This module provides highly optimized distance calculations using the unified
//! SIMD operations from scirs2-core, with fallbacks to standard implementations.

use ndarray::{Array1, Array2, ArrayView2};
use num_traits::{Float, FromPrimitive};
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::{AutoOptimizer, PlatformCapabilities, SimdUnifiedOps};
use std::fmt::Debug;

/// Compute Euclidean distances between all pairs of points using SIMD when available
///
/// # Arguments
///
/// * `data` - Input data (n_samples × n_features)
///
/// # Returns
///
/// * Condensed distance matrix as a 1D array
pub fn pairwise_euclidean_simd<F>(data: ArrayView2<F>) -> Array1<F>
where
    F: Float + FromPrimitive + Debug + Send + Sync + SimdUnifiedOps,
{
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];
    let n_distances = n_samples * (n_samples - 1) / 2;
    let mut distances = Array1::zeros(n_distances);

    let caps = PlatformCapabilities::detect();
    let optimizer = AutoOptimizer::new();

    if caps.simd_available && optimizer.should_use_simd(n_samples * n_features) {
        pairwise_euclidean_simd_optimized(data, &mut distances);
    } else {
        pairwise_euclidean_standard(data, &mut distances);
    }

    distances
}

/// Standard pairwise Euclidean distance computation
fn pairwise_euclidean_standard<F>(data: ArrayView2<F>, distances: &mut Array1<F>)
where
    F: Float + FromPrimitive + Debug,
{
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];

    let mut idx = 0;
    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            let mut sum_sq = F::zero();
            for k in 0..n_features {
                let diff = data[[i, k]] - data[[j, k]];
                sum_sq = sum_sq + diff * diff;
            }
            distances[idx] = sum_sq.sqrt();
            idx += 1;
        }
    }
}

/// SIMD-optimized pairwise Euclidean distance computation using unified operations
fn pairwise_euclidean_simd_optimized<F>(data: ArrayView2<F>, distances: &mut Array1<F>)
where
    F: Float + FromPrimitive + Debug + SimdUnifiedOps,
{
    let n_samples = data.shape()[0];

    let mut idx = 0;
    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            let row_i = data.row(i);
            let row_j = data.row(j);

            // Use SIMD operations for vector subtraction and norm calculation
            let diff = F::simd_sub(&row_i, &row_j);
            let distance = F::simd_norm(&diff.view());

            distances[idx] = distance;
            idx += 1;
        }
    }
}

/// Compute distances from each point to a set of centroids using SIMD
///
/// # Arguments
///
/// * `data` - Input data (n_samples × n_features)
/// * `centroids` - Cluster centroids (n_clusters × n_features)
///
/// # Returns
///
/// * Distance matrix (n_samples × n_clusters)
pub fn distance_to_centroids_simd<F>(data: ArrayView2<F>, centroids: ArrayView2<F>) -> Array2<F>
where
    F: Float + FromPrimitive + Debug + Send + Sync + SimdUnifiedOps,
{
    let n_samples = data.shape()[0];
    let n_clusters = centroids.shape()[0];
    let n_features = data.shape()[1];

    if centroids.shape()[1] != n_features {
        panic!("Data and centroids must have the same number of features");
    }

    let mut distances = Array2::zeros((n_samples, n_clusters));

    let caps = PlatformCapabilities::detect();
    let optimizer = AutoOptimizer::new();

    if caps.simd_available && optimizer.should_use_simd(n_samples * n_features) {
        distance_to_centroids_simd_optimized(data, centroids, &mut distances);
    } else {
        distance_to_centroids_standard(data, centroids, &mut distances);
    }

    distances
}

/// Standard distance to centroids computation
fn distance_to_centroids_standard<F>(
    data: ArrayView2<F>,
    centroids: ArrayView2<F>,
    distances: &mut Array2<F>,
) where
    F: Float + FromPrimitive + Debug,
{
    let n_samples = data.shape()[0];
    let n_clusters = centroids.shape()[0];
    let n_features = data.shape()[1];

    for i in 0..n_samples {
        for j in 0..n_clusters {
            let mut sum_sq = F::zero();
            for k in 0..n_features {
                let diff = data[[i, k]] - centroids[[j, k]];
                sum_sq = sum_sq + diff * diff;
            }
            distances[[i, j]] = sum_sq.sqrt();
        }
    }
}

/// SIMD-optimized distance to centroids computation using unified operations
fn distance_to_centroids_simd_optimized<F>(
    data: ArrayView2<F>,
    centroids: ArrayView2<F>,
    distances: &mut Array2<F>,
) where
    F: Float + FromPrimitive + Debug + SimdUnifiedOps,
{
    let n_samples = data.shape()[0];
    let n_clusters = centroids.shape()[0];

    for i in 0..n_samples {
        for j in 0..n_clusters {
            let data_row = data.row(i);
            let centroid_row = centroids.row(j);

            // Use SIMD operations for vector subtraction and norm calculation
            let diff = F::simd_sub(&data_row, &centroid_row);
            let distance = F::simd_norm(&diff.view());

            distances[[i, j]] = distance;
        }
    }
}

/// Parallel distance matrix computation using core parallel operations
///
/// # Arguments
///
/// * `data` - Input data (n_samples × n_features)
///
/// # Returns
///
/// * Condensed distance matrix
pub fn pairwise_euclidean_parallel<F>(data: ArrayView2<F>) -> Array1<F>
where
    F: Float + FromPrimitive + Debug + Send + Sync + SimdUnifiedOps,
{
    let n_samples = data.shape()[0];
    let n_distances = n_samples * (n_samples - 1) / 2;

    // Create index pairs
    let mut pairs = Vec::with_capacity(n_distances);
    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            pairs.push((i, j));
        }
    }

    // Use parallel operations from core
    if is_parallel_enabled() && pairs.len() > 100 {
        // Compute distances in parallel using core abstractions
        let distances: Vec<F> = pairs
            .into_par_iter()
            .map(|(i, j)| {
                let row_i = data.row(i);
                let row_j = data.row(j);

                // Use SIMD operations for distance calculation
                let diff = F::simd_sub(&row_i, &row_j);
                F::simd_norm(&diff.view())
            })
            .collect();
        Array1::from_vec(distances)
    } else {
        // Fallback to SIMD version for small problems or when parallel is disabled
        pairwise_euclidean_simd(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array2;

    #[test]
    fn test_pairwise_euclidean_simd() {
        let data =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        let distances = pairwise_euclidean_simd(data.view());

        // Expected distances: (0,1)=1.0, (0,2)=1.0, (0,3)=√2, (1,2)=√2, (1,3)=1.0, (2,3)=1.0
        assert_eq!(distances.len(), 6);
        assert_abs_diff_eq!(distances[0], 1.0, epsilon = 1e-10); // (0,1)
        assert_abs_diff_eq!(distances[1], 1.0, epsilon = 1e-10); // (0,2)
        assert_abs_diff_eq!(distances[2], 2.0_f64.sqrt(), epsilon = 1e-10); // (0,3)
        assert_abs_diff_eq!(distances[3], 2.0_f64.sqrt(), epsilon = 1e-10); // (1,2)
        assert_abs_diff_eq!(distances[4], 1.0, epsilon = 1e-10); // (1,3)
        assert_abs_diff_eq!(distances[5], 1.0, epsilon = 1e-10); // (2,3)
    }

    #[test]
    fn test_distance_to_centroids_simd() {
        let data =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        let centroids = Array2::from_shape_vec((2, 2), vec![0.5, 0.0, 0.5, 1.0]).unwrap();

        let distances = distance_to_centroids_simd(data.view(), centroids.view());

        assert_eq!(distances.shape(), &[4, 2]);

        // Check some expected distances
        assert_abs_diff_eq!(distances[[0, 0]], 0.5, epsilon = 1e-10); // (0,0) to centroid 0
        assert_abs_diff_eq!(distances[[3, 1]], 0.5, epsilon = 1e-10); // (1,1) to centroid 1
    }

    #[test]
    fn test_parallel_vs_standard() {
        let data = Array2::from_shape_vec(
            (6, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                9.0, 10.0,
            ],
        )
        .unwrap();

        let distances_simd = pairwise_euclidean_simd(data.view());
        let distances_parallel = pairwise_euclidean_parallel(data.view());

        assert_eq!(distances_simd.len(), distances_parallel.len());

        for i in 0..distances_simd.len() {
            assert_abs_diff_eq!(distances_simd[i], distances_parallel[i], epsilon = 1e-10);
        }
    }
}
