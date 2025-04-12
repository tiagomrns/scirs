// Clustering metrics module
//!
//! This module provides functions for evaluating clustering algorithms, including
//! silhouette score, Davies-Bouldin index, and Calinski-Harabasz index.

use ndarray::{Array1, ArrayBase, Data, Dimension, Ix2};
use num_traits::{Float, NumCast};
use std::collections::HashMap;

use crate::error::{MetricsError, Result};

/// Calculates the silhouette score for a clustering
///
/// The silhouette score measures how similar an object is to its own cluster
/// compared to other clusters. The silhouette score ranges from -1 to 1, where
/// a high value indicates that the object is well matched to its own cluster
/// and poorly matched to neighboring clusters.
///
/// # Arguments
///
/// * `x` - Array of shape (n_samples, n_features) - The data
/// * `labels` - Array of shape (n_samples,) - Predicted labels for each sample
/// * `metric` - Distance metric to use. Currently only 'euclidean' is supported.
///
/// # Returns
///
/// * The mean silhouette coefficient for all samples
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array2};
/// use scirs2_metrics::clustering::silhouette_score;
///
/// // Create a small dataset with 2 clusters
/// let x = Array2::from_shape_vec((6, 2), vec![
///     1.0, 2.0,
///     1.5, 1.8,
///     1.2, 2.2,
///     5.0, 6.0,
///     5.2, 5.8,
///     5.5, 6.2,
/// ]).unwrap();
///
/// let labels = array![0, 0, 0, 1, 1, 1];
///
/// let score = silhouette_score(&x, &labels, "euclidean").unwrap();
/// assert!(score > 0.8); // High score for well-separated clusters
/// ```
pub fn silhouette_score<F, S1, S2, D>(
    x: &ArrayBase<S1, Ix2>,
    labels: &ArrayBase<S2, D>,
    metric: &str,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug + ndarray::ScalarOperand,
    S1: Data<Elem = F>,
    S2: Data<Elem = usize>,
    D: Dimension,
{
    // Check that the metric is supported
    if metric != "euclidean" {
        return Err(MetricsError::InvalidInput(format!(
            "Unsupported metric: {}. Only 'euclidean' is currently supported.",
            metric
        )));
    }

    // Check that x and labels have the same number of samples
    let n_samples = x.shape()[0];
    if n_samples != labels.len() {
        return Err(MetricsError::InvalidInput(format!(
            "x has {} samples, but labels has {} samples",
            n_samples,
            labels.len()
        )));
    }

    // Check that there are at least 2 samples
    if n_samples < 2 {
        return Err(MetricsError::InvalidInput(
            "n_samples must be at least 2".to_string(),
        ));
    }

    // Group samples by label
    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &label) in labels.iter().enumerate() {
        clusters.entry(label).or_default().push(i);
    }

    // Check that there are at least 2 clusters
    if clusters.len() < 2 {
        return Err(MetricsError::InvalidInput(
            "Number of labels is 1. Silhouette score is undefined for a single cluster."
                .to_string(),
        ));
    }

    // Check that all clusters have at least 1 sample
    let empty_clusters: Vec<_> = clusters
        .iter()
        .filter(|(_, samples)| samples.is_empty())
        .map(|(&label, _)| label)
        .collect();

    if !empty_clusters.is_empty() {
        return Err(MetricsError::InvalidInput(format!(
            "Empty clusters found: {:?}",
            empty_clusters
        )));
    }

    // Compute silhouette scores for each sample
    let mut silhouette_values = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let label_i = labels.iter().nth(i).ok_or_else(|| {
            MetricsError::InvalidInput(format!("Could not access index {} in labels", i))
        })?;
        let cluster_i = &clusters[label_i];

        // Calculate the mean intra-cluster distance (a)
        let mut a = F::zero();
        let mut count_a = 0;

        for &j in cluster_i {
            if i == j {
                continue;
            }
            a = a + euclidean_distance(x.row(i).to_owned(), x.row(j).to_owned());
            count_a += 1;
        }

        // Handle single sample in cluster (set a to 0)
        if count_a > 0 {
            a = a / F::from(count_a).unwrap();
        }

        // Calculate the mean nearest-cluster distance (b)
        let mut b = None;

        for (label_j, cluster_j) in &clusters {
            if *label_j == *label_i {
                continue;
            }

            // Calculate mean distance to this cluster
            let mut cluster_dist = F::zero();
            for &j in cluster_j {
                cluster_dist =
                    cluster_dist + euclidean_distance(x.row(i).to_owned(), x.row(j).to_owned());
            }
            let cluster_dist = cluster_dist / F::from(cluster_j.len()).unwrap();

            // Update b if this is the closest cluster
            if let Some(current_b) = b {
                if cluster_dist < current_b {
                    b = Some(cluster_dist);
                }
            } else {
                b = Some(cluster_dist);
            }
        }

        // Calculate silhouette score
        let s = if let Some(b) = b {
            if a < b {
                F::one() - a / b
            } else if a > b {
                b / a - F::one()
            } else {
                F::zero()
            }
        } else {
            F::zero() // Will never happen if there are at least 2 clusters
        };

        silhouette_values.push(s);
    }

    // Calculate mean silhouette score
    let sum = silhouette_values
        .iter()
        .fold(F::zero(), |acc, &val| acc + val);
    Ok(sum / F::from(n_samples).unwrap())
}

/// Calculates the Davies-Bouldin index for a clustering
///
/// The Davies-Bouldin index measures the average similarity between clusters,
/// where the similarity is a ratio of within-cluster distances to between-cluster distances.
/// The lower the value, the better the clustering.
///
/// # Arguments
///
/// * `x` - Array of shape (n_samples, n_features) - The data
/// * `labels` - Array of shape (n_samples,) - Predicted labels for each sample
///
/// # Returns
///
/// * The Davies-Bouldin index
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array2};
/// use scirs2_metrics::clustering::davies_bouldin_score;
///
/// // Create a small dataset with 2 clusters
/// let x = Array2::from_shape_vec((6, 2), vec![
///     1.0, 2.0,
///     1.5, 1.8,
///     1.2, 2.2,
///     5.0, 6.0,
///     5.2, 5.8,
///     5.5, 6.2,
/// ]).unwrap();
///
/// let labels = array![0, 0, 0, 1, 1, 1];
///
/// let score = davies_bouldin_score(&x, &labels).unwrap();
/// assert!(score < 0.5); // Low score for well-separated clusters
/// ```
pub fn davies_bouldin_score<F, S1, S2, D>(
    x: &ArrayBase<S1, Ix2>,
    labels: &ArrayBase<S2, D>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug + ndarray::ScalarOperand,
    S1: Data<Elem = F>,
    S2: Data<Elem = usize>,
    D: Dimension,
{
    // Check that x and labels have the same number of samples
    let n_samples = x.shape()[0];
    if n_samples != labels.len() {
        return Err(MetricsError::InvalidInput(format!(
            "x has {} samples, but labels has {} samples",
            n_samples,
            labels.len()
        )));
    }

    // Check that there are at least 2 samples
    if n_samples < 2 {
        return Err(MetricsError::InvalidInput(
            "n_samples must be at least 2".to_string(),
        ));
    }

    // Group samples by label and compute cluster centroids
    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &label) in labels.iter().enumerate() {
        clusters.entry(label).or_default().push(i);
    }

    // Check that there are at least 2 clusters
    if clusters.len() < 2 {
        return Err(MetricsError::InvalidInput(
            "Number of labels is 1. Davies-Bouldin index is undefined for a single cluster."
                .to_string(),
        ));
    }

    // Compute centroids for each cluster
    let mut centroids = HashMap::new();
    for (&label, indices) in &clusters {
        let mut centroid = Array1::<F>::zeros(x.shape()[1]);
        for &idx in indices {
            centroid = centroid + x.row(idx).to_owned();
        }
        centroid = centroid / F::from(indices.len()).unwrap();
        centroids.insert(label, centroid);
    }

    // Compute average distance to centroid for each cluster
    let mut avg_distances = HashMap::new();
    for (&label, indices) in &clusters {
        let centroid = centroids.get(&label).unwrap();
        let mut total_distance = F::zero();
        for &idx in indices {
            total_distance =
                total_distance + euclidean_distance(x.row(idx).to_owned(), centroid.view());
        }
        let avg_distance = total_distance / F::from(indices.len()).unwrap();
        avg_distances.insert(label, avg_distance);
    }

    // Compute Davies-Bouldin index
    let mut db_index = F::zero();
    let labels_vec: Vec<_> = clusters.keys().cloned().collect();

    for i in 0..labels_vec.len() {
        let label_i = labels_vec[i];
        let centroid_i = centroids.get(&label_i).unwrap();
        let avg_dist_i = avg_distances.get(&label_i).unwrap();

        let mut max_ratio = F::zero();
        for (j, &label_j) in labels_vec.iter().enumerate() {
            if i == j {
                continue;
            }
            let centroid_j = centroids.get(&label_j).unwrap();
            let avg_dist_j = avg_distances.get(&label_j).unwrap();

            // Distance between centroids
            let centroid_dist = euclidean_distance(centroid_i.view(), centroid_j.view());

            // Ratio of sum of intra-cluster distances to inter-cluster distance
            let ratio = (*avg_dist_i + *avg_dist_j) / centroid_dist;

            // Update max ratio
            if ratio > max_ratio {
                max_ratio = ratio;
            }
        }

        db_index = db_index + max_ratio;
    }

    // Normalize by number of clusters
    Ok(db_index / F::from(labels_vec.len()).unwrap())
}

/// Calculates the Calinski-Harabasz index (Variance Ratio Criterion)
///
/// The Calinski-Harabasz index is defined as the ratio of the between-clusters
/// dispersion and the within-cluster dispersion. Higher values indicate better clustering.
///
/// # Arguments
///
/// * `x` - Array of shape (n_samples, n_features) - The data
/// * `labels` - Array of shape (n_samples,) - Predicted labels for each sample
///
/// # Returns
///
/// * The Calinski-Harabasz index
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array2};
/// use scirs2_metrics::clustering::calinski_harabasz_score;
///
/// // Create a small dataset with 2 clusters
/// let x = Array2::from_shape_vec((6, 2), vec![
///     1.0, 2.0,
///     1.5, 1.8,
///     1.2, 2.2,
///     5.0, 6.0,
///     5.2, 5.8,
///     5.5, 6.2,
/// ]).unwrap();
///
/// let labels = array![0, 0, 0, 1, 1, 1];
///
/// let score = calinski_harabasz_score(&x, &labels).unwrap();
/// assert!(score > 50.0); // High score for well-separated clusters
/// ```
pub fn calinski_harabasz_score<F, S1, S2, D>(
    x: &ArrayBase<S1, Ix2>,
    labels: &ArrayBase<S2, D>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug + ndarray::ScalarOperand,
    S1: Data<Elem = F>,
    S2: Data<Elem = usize>,
    D: Dimension,
{
    // Check that x and labels have the same number of samples
    let n_samples = x.shape()[0];
    if n_samples != labels.len() {
        return Err(MetricsError::InvalidInput(format!(
            "x has {} samples, but labels has {} samples",
            n_samples,
            labels.len()
        )));
    }

    // Check that there are at least 2 samples
    if n_samples < 2 {
        return Err(MetricsError::InvalidInput(
            "n_samples must be at least 2".to_string(),
        ));
    }

    // Group samples by label
    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &label) in labels.iter().enumerate() {
        clusters.entry(label).or_default().push(i);
    }

    // Check that there are at least 2 clusters
    if clusters.len() < 2 {
        return Err(MetricsError::InvalidInput(
            "Number of labels is 1. Calinski-Harabasz index is undefined for a single cluster."
                .to_string(),
        ));
    }

    // Compute the global centroid
    let mut global_centroid = Array1::<F>::zeros(x.shape()[1]);
    for i in 0..n_samples {
        global_centroid = global_centroid + x.row(i).to_owned();
    }
    global_centroid = global_centroid / F::from(n_samples).unwrap();

    // Compute centroids for each cluster
    let mut centroids = HashMap::new();
    for (&label, indices) in &clusters {
        let mut centroid = Array1::<F>::zeros(x.shape()[1]);
        for &idx in indices {
            centroid = centroid + x.row(idx).to_owned();
        }
        centroid = centroid / F::from(indices.len()).unwrap();
        centroids.insert(label, centroid);
    }

    // Compute between-cluster dispersion
    let mut between_disp = F::zero();
    for (label, indices) in &clusters {
        let cluster_size = F::from(indices.len()).unwrap();
        let centroid = centroids.get(label).unwrap();

        let dist_to_global = squared_euclidean_distance(centroid.view(), global_centroid.view());
        between_disp = between_disp + cluster_size * dist_to_global;
    }

    // Compute within-cluster dispersion
    let mut within_disp = F::zero();
    for (label, indices) in &clusters {
        let centroid = centroids.get(label).unwrap();

        let mut cluster_disp = F::zero();
        for &idx in indices {
            let dist_to_centroid =
                squared_euclidean_distance(x.row(idx).to_owned(), centroid.view());
            cluster_disp = cluster_disp + dist_to_centroid;
        }

        within_disp = within_disp + cluster_disp;
    }

    // Handle edge cases
    if within_disp <= F::epsilon() {
        return Err(MetricsError::CalculationError(
            "Within-cluster dispersion is zero".to_string(),
        ));
    }

    // Calculate Calinski-Harabasz index
    let n_clusters = F::from(clusters.len()).unwrap();
    Ok(
        between_disp * (F::from(n_samples - clusters.len()).unwrap())
            / (within_disp * (n_clusters - F::one())),
    )
}

/// Helper function to calculate the Euclidean distance between two vectors
fn euclidean_distance<F, S1, S2, D1, D2>(a: ArrayBase<S1, D1>, b: ArrayBase<S2, D2>) -> F
where
    F: Float + NumCast + std::fmt::Debug + ndarray::ScalarOperand,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    let mut sum = F::zero();
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = *x - *y;
        sum = sum + diff * diff;
    }
    sum.sqrt()
}

/// Helper function to calculate the squared Euclidean distance between two vectors
fn squared_euclidean_distance<F, S1, S2, D1, D2>(a: ArrayBase<S1, D1>, b: ArrayBase<S2, D2>) -> F
where
    F: Float + NumCast + std::fmt::Debug + ndarray::ScalarOperand,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    let mut sum = F::zero();
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = *x - *y;
        sum = sum + diff * diff;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array2};

    fn create_test_data() -> (Array2<f64>, Array1<usize>) {
        // Create a small dataset with 2 well-separated clusters
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.5, 1.8, 1.2, 2.2, 5.0, 6.0, 5.2, 5.8, 5.5, 6.2],
        )
        .unwrap();

        let labels = array![0, 0, 0, 1, 1, 1];

        (x, labels)
    }

    #[test]
    fn test_silhouette_score() {
        let (x, labels) = create_test_data();

        let score = silhouette_score(&x, &labels, "euclidean").unwrap();
        assert!(score > 0.8); // High score for well-separated clusters
    }

    #[test]
    fn test_davies_bouldin_score() {
        let (x, labels) = create_test_data();

        let score = davies_bouldin_score(&x, &labels).unwrap();
        assert!(score < 0.5); // Low score for well-separated clusters
    }

    #[test]
    fn test_calinski_harabasz_score() {
        let (x, labels) = create_test_data();

        let score = calinski_harabasz_score(&x, &labels).unwrap();
        assert!(score > 50.0); // High score for well-separated clusters
    }

    #[test]
    fn test_distance_functions() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![4.0, 5.0, 6.0];

        let dist = euclidean_distance(a.view(), b.view());
        assert_abs_diff_eq!(dist, 5.196152422706632, epsilon = 1e-10);

        let sq_dist = squared_euclidean_distance(a.view(), b.view());
        assert_abs_diff_eq!(sq_dist, 27.0, epsilon = 1e-10);
    }
}
