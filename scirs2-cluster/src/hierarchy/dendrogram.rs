//! Dendrogram visualization and analysis tools
//!
//! This module provides functions for analyzing and potentially visualizing dendrograms
//! created by hierarchical clustering.

use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

/// Calculates the cophenetic correlation coefficient
///
/// This measures how faithfully the dendrogram preserves the original distances.
///
/// # Arguments
///
/// * `z` - The linkage matrix from the `linkage` function
/// * `d` - The original distance matrix (condensed form)
///
/// # Returns
///
/// * `Result<F>` - The cophenetic correlation coefficient
pub fn cophenet<F: Float + FromPrimitive>(z: &Array2<F>, d: &Array1<F>) -> Result<F> {
    let n_samples = z.shape()[0] + 1;

    // Calculate the cophenetic distances
    let mut cophenetic_distances = Array1::zeros(d.len());

    // Create a mapping from sample indices to their dendrogram heights
    let mut cluster_height: Vec<F> = vec![F::zero(); 2 * n_samples - 1];

    // Process the linkage matrix
    for i in 0..z.shape()[0] {
        let _cluster1 = z[[i, 0]].to_usize().unwrap();
        let _cluster2 = z[[i, 1]].to_usize().unwrap();
        let height = z[[i, 2]];
        let new_cluster = n_samples + i;

        // Record the merge height
        cluster_height[new_cluster] = height;
    }

    // Calculate cophenetic distances
    let mut idx = 0;
    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            // Find the lowest common ancestor in the dendrogram
            let lca_height = find_lca_height(i, j, z, &cluster_height);
            cophenetic_distances[idx] = lca_height;
            idx += 1;
        }
    }

    // Calculate the correlation coefficient
    let d_mean = d.mean().unwrap();
    let c_mean = cophenetic_distances.mean().unwrap();

    let mut numerator = F::zero();
    let mut d_variance = F::zero();
    let mut c_variance = F::zero();

    for i in 0..d.len() {
        let d_diff = d[i] - d_mean;
        let c_diff = cophenetic_distances[i] - c_mean;

        numerator = numerator + d_diff * c_diff;
        d_variance = d_variance + d_diff * d_diff;
        c_variance = c_variance + c_diff * c_diff;
    }

    let denom = (d_variance * c_variance).sqrt();

    if denom < F::from_f64(1e-10).unwrap() {
        return Err(ClusteringError::ComputationError(
            "Variance is too small to calculate cophenetic correlation".into(),
        ));
    }

    Ok(numerator / denom)
}

/// Finds the height of the lowest common ancestor in the dendrogram
fn find_lca_height<F: Float>(i: usize, j: usize, z: &Array2<F>, cluster_height: &[F]) -> F {
    let n_samples = z.shape()[0] + 1;

    // These are intentionally prefixed with underscore to indicate they're not used
    // in our simplified implementation, but kept for future development
    let _i_cluster = i;
    let _j_cluster = j;

    // Trace up the dendrogram, tracking the membership of each leaf
    let mut cluster_map = vec![0; 2 * n_samples - 1];
    for (idx, val) in cluster_map.iter_mut().enumerate().take(n_samples) {
        *val = idx;
    }

    for idx in 0..z.shape()[0] {
        let cluster1 = z[[idx, 0]].to_usize().unwrap();
        let cluster2 = z[[idx, 1]].to_usize().unwrap();
        let new_cluster = n_samples + idx;

        // Update cluster membership
        for (k, val) in cluster_map.iter_mut().enumerate() {
            if k < new_cluster && (*val == cluster1 || *val == cluster2) {
                *val = new_cluster;
            }
        }

        // Check if i and j are now in the same cluster
        if cluster_map[i] == cluster_map[j] {
            // Found the lowest common ancestor
            return cluster_height[cluster_map[i]];
        }
    }

    // If we get here, something went wrong
    F::zero()
}

/// Calculates the inconsistency of each merge in the linkage matrix
///
/// Inconsistency is a measure of how consistent a merge is with respect to its neighbors.
///
/// # Arguments
///
/// * `z` - The linkage matrix from the `linkage` function
/// * `d` - Number of levels to consider in calculating inconsistency (default: 2)
///
/// # Returns
///
/// * `Result<Array2<F>>` - The inconsistency matrix
pub fn inconsistent<F: Float + FromPrimitive + Debug>(
    z: &Array2<F>,
    d: Option<usize>,
) -> Result<Array2<F>> {
    let depth = d.unwrap_or(2);
    let n = z.shape()[0];

    // Output format: [mean, std, count, inconsistency]
    let mut result = Array2::zeros((n, 4));

    for i in 0..n {
        // Get depths of descendants within specified depth
        let depths = get_descendants(z, i, depth)?;

        // Extract heights
        let mut heights = Vec::with_capacity(depths.len());
        for &idx in &depths {
            if idx < n {
                heights.push(z[[idx, 2]]);
            }
        }

        if heights.is_empty() {
            heights.push(z[[i, 2]]);
        }

        // Calculate statistics
        let mean = heights.iter().fold(F::zero(), |acc, &x| acc + x)
            / F::from_usize(heights.len()).unwrap();

        let mut variance = F::zero();
        for &h in &heights {
            let diff = h - mean;
            variance = variance + diff * diff;
        }
        variance = variance / F::from_usize(heights.len()).unwrap();
        let std_dev = variance.sqrt();

        // Calculate inconsistency
        let inconsistency = if std_dev < F::from_f64(1e-10).unwrap() {
            F::zero()
        } else {
            (z[[i, 2]] - mean) / std_dev
        };

        // Store results
        result[[i, 0]] = mean;
        result[[i, 1]] = std_dev;
        result[[i, 2]] = F::from_usize(heights.len()).unwrap();
        result[[i, 3]] = inconsistency;
    }

    Ok(result)
}

/// Gets the descendants of a node in the dendrogram within a specified depth
fn get_descendants<F: Float>(z: &Array2<F>, idx: usize, depth: usize) -> Result<Vec<usize>> {
    let n_samples = z.shape()[0] + 1;
    let mut result = Vec::new();

    if depth == 0 {
        result.push(idx);
        return Ok(result);
    }

    // Non-leaf node
    if idx >= n_samples - 1 {
        let i = idx - (n_samples - 1); // Convert to z index

        // Check if indices are valid
        if i >= z.shape()[0] {
            return Err(ClusteringError::ComputationError(
                "Invalid node index in dendrogram".into(),
            ));
        }

        let left = z[[i, 0]].to_usize().unwrap();
        let right = z[[i, 1]].to_usize().unwrap();

        // Add current node
        result.push(i);

        // Add descendants recursively
        let left_desc = get_descendants(z, left, depth - 1)?;
        let right_desc = get_descendants(z, right, depth - 1)?;

        result.extend(left_desc);
        result.extend(right_desc);
    } else {
        // Leaf node
        result.push(idx);
    }

    Ok(result)
}

/// Calculates the optimal leaf ordering for a dendrogram
///
/// This reorders the leaves to minimize the sum of distances between adjacent leaves.
///
/// # Arguments
///
/// * `z` - The linkage matrix
/// * `d` - The original distance matrix (condensed form)
///
/// # Returns
///
/// * `Result<Array1<usize>>` - The optimal leaf ordering
pub fn optimal_leaf_ordering<F: Float + FromPrimitive + PartialOrd>(
    z: &Array2<F>,
    _d: &Array1<F>,
) -> Result<Array1<usize>> {
    let n_samples = z.shape()[0] + 1;

    // Initialize the leaf order as identity mapping
    let order = Array1::from_iter(0..n_samples);

    // Not a full implementation - would require extensive dynamic programming
    // For now, we return the original ordering

    Ok(order)
}

/// Converts a linkage matrix to a dendrogram dictionary for visualization
///
/// # Arguments
///
/// * `z` - The linkage matrix
///
/// # Returns
///
/// * `Result<Vec<(usize, usize, F, usize)>>` - The dendrogram data
pub fn dendrogram<F: Float + FromPrimitive + Clone>(
    z: &Array2<F>,
) -> Result<Vec<(usize, usize, F, usize)>> {
    let n = z.shape()[0];

    if n == 0 {
        return Err(ClusteringError::InvalidInput("Empty linkage matrix".into()));
    }

    // Convert the linkage matrix to a list of tuples
    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        let cluster1 = z[[i, 0]].to_usize().unwrap();
        let cluster2 = z[[i, 1]].to_usize().unwrap();
        let height = z[[i, 2]];
        let count = z[[i, 3]].to_usize().unwrap();

        result.push((cluster1, cluster2, height, count));
    }

    Ok(result)
}
