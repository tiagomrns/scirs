//! SIMD-accelerated distance calculations for spatial operations
//!
//! This module provides high-performance implementations of distance calculations
//! using the unified SIMD operations from scirs2-core. These optimized
//! functions can provide significant performance improvements for operations
//! involving large datasets.
//!
//! # Features
//!
//! - **SIMD-accelerated distance metrics**: Euclidean, Manhattan, Minkowski
//! - **Parallel distance matrix computation**: Multi-threaded pdist and cdist
//! - **Batch nearest neighbor queries**: Optimized for large point sets
//! - **Memory-efficient operations**: Minimized memory allocation and copying
//!
//! # Architecture Support
//!
//! The SIMD operations are automatically detected and optimized by scirs2-core
//! based on the available hardware capabilities.
//!
//! # Examples
//!
//! ```
//! use scirs2_spatial::simd_distance::{simd_euclidean_distance_batch, parallel_pdist};
//! use ndarray::array;
//!
//! // SIMD batch distance calculation
//! let points1 = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
//! let points2 = array![[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]];
//!
//! let distances = simd_euclidean_distance_batch(&points1.view(), &points2.view()).unwrap();
//! println!("Batch distances: {:?}", distances);
//!
//! // Parallel distance matrix
//! let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
//! let dist_matrix = parallel_pdist(&points.view(), "euclidean").unwrap();
//! println!("Distance matrix shape: {:?}", dist_matrix.shape());
//! ```

use crate::error::{SpatialError, SpatialResult};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::{PlatformCapabilities, SimdUnifiedOps};

/// Supported distance metrics for SIMD operations
#[derive(Debug, Clone, Copy)]
pub enum SimdMetric {
    /// Euclidean distance (L2 norm)
    Euclidean,
    /// Manhattan distance (L1 norm)  
    Manhattan,
    /// Squared Euclidean distance (no square root)
    SquaredEuclidean,
    /// Chebyshev distance (L∞ norm)
    Chebyshev,
}

impl SimdMetric {
    /// Get the string name of the metric
    pub fn name(&self) -> &'static str {
        match self {
            SimdMetric::Euclidean => "euclidean",
            SimdMetric::Manhattan => "manhattan",
            SimdMetric::SquaredEuclidean => "sqeuclidean",
            SimdMetric::Chebyshev => "chebyshev",
        }
    }
}

/// SIMD-accelerated Euclidean distance between two points
///
/// # Arguments
/// * `a` - First point
/// * `b` - Second point
///
/// # Returns
/// * Euclidean distance between the points
pub fn simd_euclidean_distance(a: &[f64], b: &[f64]) -> SpatialResult<f64> {
    if a.len() != b.len() {
        return Err(SpatialError::ValueError(
            "Points must have the same dimension".to_string(),
        ));
    }

    // Convert slices to ArrayView for core ops
    let a_view = ArrayView1::from(a);
    let b_view = ArrayView1::from(b);

    // Compute squared distance using SIMD operations
    let diff = f64::simd_sub(&a_view, &b_view);
    let squared = f64::simd_mul(&diff.view(), &diff.view());
    let sum = f64::simd_sum(&squared.view());
    Ok(sum.sqrt())
}

/// SIMD-accelerated squared Euclidean distance between two points
pub fn simd_squared_euclidean_distance(a: &[f64], b: &[f64]) -> SpatialResult<f64> {
    if a.len() != b.len() {
        return Err(SpatialError::ValueError(
            "Points must have the same dimension".to_string(),
        ));
    }

    let a_view = ArrayView1::from(a);
    let b_view = ArrayView1::from(b);

    // Compute squared distance using SIMD operations
    let diff = f64::simd_sub(&a_view, &b_view);
    let squared = f64::simd_mul(&diff.view(), &diff.view());
    Ok(f64::simd_sum(&squared.view()))
}

/// SIMD-accelerated Manhattan distance between two points
pub fn simd_manhattan_distance(a: &[f64], b: &[f64]) -> SpatialResult<f64> {
    if a.len() != b.len() {
        return Err(SpatialError::ValueError(
            "Points must have the same dimension".to_string(),
        ));
    }

    let a_view = ArrayView1::from(a);
    let b_view = ArrayView1::from(b);

    // Compute Manhattan distance using SIMD operations
    let diff = f64::simd_sub(&a_view, &b_view);
    let abs_diff = f64::simd_abs(&diff.view());
    Ok(f64::simd_sum(&abs_diff.view()))
}

/// SIMD-accelerated Chebyshev distance
pub fn simd_chebyshev_distance(a: &[f64], b: &[f64]) -> SpatialResult<f64> {
    if a.len() != b.len() {
        return Err(SpatialError::ValueError(
            "Points must have the same dimension".to_string(),
        ));
    }

    let a_view = ArrayView1::from(a);
    let b_view = ArrayView1::from(b);

    // Compute Chebyshev distance using SIMD operations
    let diff = f64::simd_sub(&a_view, &b_view);
    let abs_diff = f64::simd_abs(&diff.view());
    Ok(f64::simd_max_element(&abs_diff.view()))
}

/// Batch SIMD-accelerated Euclidean distance calculation
///
/// Computes distances between corresponding rows of two matrices
///
/// # Arguments
/// * `points1` - First set of points, shape (n, d)
/// * `points2` - Second set of points, shape (n, d)
///
/// # Returns
/// * Array of distances, shape (n,)
pub fn simd_euclidean_distance_batch(
    points1: &ArrayView2<f64>,
    points2: &ArrayView2<f64>,
) -> SpatialResult<Array1<f64>> {
    if points1.shape() != points2.shape() {
        return Err(SpatialError::ValueError(
            "Point arrays must have the same shape".to_string(),
        ));
    }

    let n_points = points1.nrows();

    // Use parallel computation with SIMD operations
    let distances_vec: Result<Vec<f64>, SpatialError> = (0..n_points)
        .into_par_iter()
        .map(|i| -> SpatialResult<f64> {
            let p1 = points1.row(i);
            let p2 = points2.row(i);
            let diff = f64::simd_sub(&p1, &p2);
            let squared = f64::simd_mul(&diff.view(), &diff.view());
            let sum = f64::simd_sum(&squared.view());
            Ok(sum.sqrt())
        })
        .collect();

    Ok(Array1::from(distances_vec?))
}

/// Parallel computation of pairwise distance matrix
///
/// # Arguments
/// * `points` - Array of points, shape (n, d)
/// * `metric` - Distance metric to use
///
/// # Returns
/// * Condensed distance matrix, shape (n*(n-1)/2,)
pub fn parallel_pdist(points: &ArrayView2<f64>, metric: &str) -> SpatialResult<Array1<f64>> {
    let n_points = points.nrows();
    if n_points < 2 {
        return Err(SpatialError::ValueError(
            "Need at least 2 points for distance computation".to_string(),
        ));
    }

    let n_distances = n_points * (n_points - 1) / 2;

    let metric_enum = match metric {
        "euclidean" => SimdMetric::Euclidean,
        "manhattan" => SimdMetric::Manhattan,
        "sqeuclidean" => SimdMetric::SquaredEuclidean,
        "chebyshev" => SimdMetric::Chebyshev,
        _ => {
            return Err(SpatialError::ValueError(format!(
                "Unsupported metric: {}",
                metric
            )))
        }
    };

    // Parallel computation of distance pairs
    let distances_vec: Result<Vec<f64>, SpatialError> = (0..n_distances)
        .into_par_iter()
        .map(|idx| -> SpatialResult<f64> {
            // Convert linear index to (i, j) pair
            let (i, j) = linear_to_condensed_indices(idx, n_points);

            let p1 = points.row(i);
            let p2 = points.row(j);

            match metric_enum {
                SimdMetric::Euclidean => {
                    let diff = f64::simd_sub(&p1, &p2);
                    let squared = f64::simd_mul(&diff.view(), &diff.view());
                    let sum = f64::simd_sum(&squared.view());
                    Ok(sum.sqrt())
                }
                SimdMetric::Manhattan => {
                    let diff = f64::simd_sub(&p1, &p2);
                    let abs_diff = f64::simd_abs(&diff.view());
                    Ok(f64::simd_sum(&abs_diff.view()))
                }
                SimdMetric::SquaredEuclidean => {
                    let diff = f64::simd_sub(&p1, &p2);
                    let squared = f64::simd_mul(&diff.view(), &diff.view());
                    Ok(f64::simd_sum(&squared.view()))
                }
                SimdMetric::Chebyshev => {
                    let diff = f64::simd_sub(&p1, &p2);
                    let abs_diff = f64::simd_abs(&diff.view());
                    Ok(f64::simd_max_element(&abs_diff.view()))
                }
            }
        })
        .collect();

    Ok(Array1::from(distances_vec?))
}

/// Parallel computation of cross-distance matrix
///
/// # Arguments
/// * `points1` - First set of points, shape (n, d)
/// * `points2` - Second set of points, shape (m, d)
/// * `metric` - Distance metric to use
///
/// # Returns
/// * Distance matrix, shape (n, m)
pub fn parallel_cdist(
    points1: &ArrayView2<f64>,
    points2: &ArrayView2<f64>,
    metric: &str,
) -> SpatialResult<Array2<f64>> {
    if points1.ncols() != points2.ncols() {
        return Err(SpatialError::ValueError(
            "Point arrays must have the same number of dimensions".to_string(),
        ));
    }

    let n1 = points1.nrows();
    let n2 = points2.nrows();
    let mut distances = Array2::zeros((n1, n2));

    let metric_enum = match metric {
        "euclidean" => SimdMetric::Euclidean,
        "manhattan" => SimdMetric::Manhattan,
        "sqeuclidean" => SimdMetric::SquaredEuclidean,
        "chebyshev" => SimdMetric::Chebyshev,
        _ => {
            return Err(SpatialError::ValueError(format!(
                "Unsupported metric: {}",
                metric
            )))
        }
    };

    // Parallel computation over rows of first matrix
    distances
        .outer_iter_mut()
        .enumerate()
        .par_bridge()
        .try_for_each(|(i, mut row)| -> SpatialResult<()> {
            let p1 = points1.row(i);

            for (j, dist) in row.iter_mut().enumerate() {
                let p2 = points2.row(j);

                *dist = match metric_enum {
                    SimdMetric::Euclidean => {
                        let diff = f64::simd_sub(&p1, &p2);
                        let squared = f64::simd_mul(&diff.view(), &diff.view());
                        let sum = f64::simd_sum(&squared.view());
                        sum.sqrt()
                    }
                    SimdMetric::Manhattan => {
                        let diff = f64::simd_sub(&p1, &p2);
                        let abs_diff = f64::simd_abs(&diff.view());
                        f64::simd_sum(&abs_diff.view())
                    }
                    SimdMetric::SquaredEuclidean => {
                        let diff = f64::simd_sub(&p1, &p2);
                        let squared = f64::simd_mul(&diff.view(), &diff.view());
                        f64::simd_sum(&squared.view())
                    }
                    SimdMetric::Chebyshev => {
                        let diff = f64::simd_sub(&p1, &p2);
                        let abs_diff = f64::simd_abs(&diff.view());
                        f64::simd_max_element(&abs_diff.view())
                    }
                };
            }
            Ok(())
        })?;

    Ok(distances)
}

/// SIMD-accelerated k-nearest neighbors search
///
/// # Arguments
/// * `query_points` - Points to query, shape (n_queries, d)
/// * `data_points` - Data points to search, shape (n_data, d)
/// * `k` - Number of nearest neighbors
/// * `metric` - Distance metric to use
///
/// # Returns
/// * (indices, distances) where both have shape (n_queries, k)
pub fn simd_knn_search(
    query_points: &ArrayView2<f64>,
    data_points: &ArrayView2<f64>,
    k: usize,
    metric: &str,
) -> SpatialResult<(Array2<usize>, Array2<f64>)> {
    if query_points.ncols() != data_points.ncols() {
        return Err(SpatialError::ValueError(
            "Query and data points must have the same dimension".to_string(),
        ));
    }

    if k > data_points.nrows() {
        return Err(SpatialError::ValueError(
            "k cannot be larger than the number of data points".to_string(),
        ));
    }

    let n_queries = query_points.nrows();
    let n_data = data_points.nrows();

    let mut indices = Array2::zeros((n_queries, k));
    let mut distances = Array2::zeros((n_queries, k));

    let metric_enum = match metric {
        "euclidean" => SimdMetric::Euclidean,
        "manhattan" => SimdMetric::Manhattan,
        "sqeuclidean" => SimdMetric::SquaredEuclidean,
        "chebyshev" => SimdMetric::Chebyshev,
        _ => {
            return Err(SpatialError::ValueError(format!(
                "Unsupported metric: {}",
                metric
            )))
        }
    };

    // Process queries in parallel
    indices
        .outer_iter_mut()
        .zip(distances.outer_iter_mut())
        .enumerate()
        .par_bridge()
        .try_for_each(
            |(query_idx, (mut idx_row, mut dist_row))| -> SpatialResult<()> {
                let query_point = query_points.row(query_idx);

                // Compute all distances for this query using SIMD
                let mut all_distances: Vec<(f64, usize)> = (0..n_data)
                    .map(|data_idx| {
                        let data_point = data_points.row(data_idx);
                        let dist = match metric_enum {
                            SimdMetric::Euclidean => {
                                let diff = f64::simd_sub(&query_point, &data_point);
                                let squared = f64::simd_mul(&diff.view(), &diff.view());
                                let sum = f64::simd_sum(&squared.view());
                                sum.sqrt()
                            }
                            SimdMetric::Manhattan => {
                                let diff = f64::simd_sub(&query_point, &data_point);
                                let abs_diff = f64::simd_abs(&diff.view());
                                f64::simd_sum(&abs_diff.view())
                            }
                            SimdMetric::SquaredEuclidean => {
                                let diff = f64::simd_sub(&query_point, &data_point);
                                let squared = f64::simd_mul(&diff.view(), &diff.view());
                                f64::simd_sum(&squared.view())
                            }
                            SimdMetric::Chebyshev => {
                                let diff = f64::simd_sub(&query_point, &data_point);
                                let abs_diff = f64::simd_abs(&diff.view());
                                f64::simd_max_element(&abs_diff.view())
                            }
                        };
                        (dist, data_idx)
                    })
                    .collect();

                // Partial sort to get k smallest
                all_distances.select_nth_unstable_by(k - 1, |a, b| a.0.partial_cmp(&b.0).unwrap());
                all_distances[..k].sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                // Fill result arrays
                for (i, (dist, idx)) in all_distances[..k].iter().enumerate() {
                    dist_row[i] = *dist;
                    idx_row[i] = *idx;
                }

                Ok(())
            },
        )?;

    Ok((indices, distances))
}

/// Convert linear index to (i, j) indices for condensed distance matrix
fn linear_to_condensed_indices(linear_idx: usize, n: usize) -> (usize, usize) {
    // For condensed distance matrix where entry (i,j) with i < j is stored
    // at position (n-1-i)*(n-i)/2 + (j-i-1)
    let mut k = linear_idx;
    let mut i = 0;

    while k >= n - i - 1 {
        k -= n - i - 1;
        i += 1;
    }

    let j = k + i + 1;
    (i, j)
}

/// Performance benchmarking utilities
pub mod bench {
    use super::*;
    use std::time::Instant;

    /// Benchmark SIMD vs scalar distance computation
    pub fn benchmark_distance_computation(
        points1: &ArrayView2<f64>,
        points2: &ArrayView2<f64>,
        iterations: usize,
    ) -> (f64, f64) {
        // Scalar benchmark
        let start = Instant::now();
        for _ in 0..iterations {
            for (row1, row2) in points1.outer_iter().zip(points2.outer_iter()) {
                let _dist =
                    crate::distance::euclidean(row1.as_slice().unwrap(), row2.as_slice().unwrap());
            }
        }
        let scalar_time = start.elapsed().as_secs_f64();

        // SIMD benchmark
        let start = Instant::now();
        for _ in 0..iterations {
            let _distances = simd_euclidean_distance_batch(points1, points2).unwrap();
        }
        let simd_time = start.elapsed().as_secs_f64();

        (scalar_time, simd_time)
    }

    /// Report available SIMD features
    pub fn report_simd_features() {
        println!("SIMD Features Available:");

        let caps = PlatformCapabilities::detect();
        println!("  SIMD Available: {}", caps.simd_available);
        println!("  GPU Available: {}", caps.gpu_available);

        if caps.simd_available {
            println!("  AVX2: {}", caps.avx2_available);
            println!("  AVX512: {}", caps.avx512_available);
            println!("  NEON: {}", caps.neon_available);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_simd_euclidean_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let simd_dist = simd_euclidean_distance(&a, &b).unwrap();
        let scalar_dist = crate::distance::euclidean(&a, &b);

        assert_relative_eq!(simd_dist, scalar_dist, epsilon = 1e-10);
    }

    #[test]
    fn test_simd_manhattan_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let simd_dist = simd_manhattan_distance(&a, &b).unwrap();
        let scalar_dist = crate::distance::manhattan(&a, &b);

        assert_relative_eq!(simd_dist, scalar_dist, epsilon = 1e-10);
    }

    #[test]
    fn test_simd_batch_distance() {
        let points1 = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let points2 = array![[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]];

        let distances = simd_euclidean_distance_batch(&points1.view(), &points2.view()).unwrap();

        assert_eq!(distances.len(), 3);
        for &dist in distances.iter() {
            assert!(dist > 0.0);
            assert!(dist.is_finite());
        }

        // Check against scalar computation
        for i in 0..3 {
            let p1 = points1.row(i).to_vec();
            let p2 = points2.row(i).to_vec();
            let expected = crate::distance::euclidean(&p1, &p2);
            assert_relative_eq!(distances[i], expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_parallel_pdist() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];

        let distances = parallel_pdist(&points.view(), "euclidean").unwrap();

        // Should have n*(n-1)/2 = 6 distances
        assert_eq!(distances.len(), 6);

        // All distances should be positive
        for &dist in distances.iter() {
            assert!(dist > 0.0);
            assert!(dist.is_finite());
        }
    }

    #[test]
    fn test_parallel_cdist() {
        let points1 = array![[0.0, 0.0], [1.0, 1.0]];
        let points2 = array![[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]];

        let distances = parallel_cdist(&points1.view(), &points2.view(), "euclidean").unwrap();

        assert_eq!(distances.shape(), &[2, 3]);

        // All distances should be non-negative
        for &dist in distances.iter() {
            assert!(dist >= 0.0);
            assert!(dist.is_finite());
        }
    }

    #[test]
    fn test_simd_knn_search() {
        let data_points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0]];
        let query_points = array![[0.5, 0.5], [1.5, 1.5]];

        let (indices, distances) =
            simd_knn_search(&query_points.view(), &data_points.view(), 3, "euclidean").unwrap();

        assert_eq!(indices.shape(), &[2, 3]);
        assert_eq!(distances.shape(), &[2, 3]);

        // Distances should be sorted in ascending order
        for row in distances.outer_iter() {
            for i in 1..row.len() {
                assert!(row[i] >= row[i - 1]);
            }
        }

        // All indices should be valid
        for &idx in indices.iter() {
            assert!(idx < data_points.nrows());
        }
    }

    #[test]
    fn test_linear_to_condensed_indices() {
        // For n=4, condensed matrix has 6 elements
        let n = 4;
        let expected_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

        for (linear_idx, expected) in expected_pairs.iter().enumerate() {
            let result = linear_to_condensed_indices(linear_idx, n);
            assert_eq!(result, *expected);
        }
    }

    #[test]
    fn test_simd_chebyshev_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 1.0];

        let dist = simd_chebyshev_distance(&a, &b).unwrap();

        // Max difference should be |2 - 5| = 3
        assert_relative_eq!(dist, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_different_metrics() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];

        let metrics = ["euclidean", "manhattan", "sqeuclidean", "chebyshev"];

        for metric in &metrics {
            let distances = parallel_pdist(&points.view(), metric).unwrap();
            assert_eq!(distances.len(), 3); // n*(n-1)/2 = 3

            for &dist in distances.iter() {
                assert!(dist >= 0.0);
                assert!(dist.is_finite());
            }
        }
    }

    #[test]
    fn test_error_handling() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0]; // Different length

        let result = simd_euclidean_distance(&a, &b);
        assert!(result.is_err());

        let result = simd_manhattan_distance(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_large_dimension_vectors() {
        // Simple test case first
        let a = vec![0.0, 1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];

        let simd_dist = simd_euclidean_distance(&a, &b).unwrap();
        let scalar_dist = crate::distance::euclidean(&a, &b);

        // Expected: sqrt(3 * 1^2) = sqrt(3) ≈ 1.732
        assert_relative_eq!(simd_dist, scalar_dist, epsilon = 1e-10);

        // Test with larger vectors
        let dim = 1000; // Restore original size
        let a: Vec<f64> = (0..dim).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..dim).map(|i| (i + 1) as f64).collect();

        let simd_dist = simd_euclidean_distance(&a, &b).unwrap();
        let scalar_dist = crate::distance::euclidean(&a, &b);

        // Expected: sqrt(1000 * 1^2) = sqrt(1000) ≈ 31.62
        assert_relative_eq!(simd_dist, scalar_dist, epsilon = 1e-10);
    }
}
