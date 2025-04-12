//! K-means clustering implementation

use ndarray::{s, Array1, Array2, ArrayView2};
use num_traits::{Float, FromPrimitive};
use rand::{rngs::ThreadRng, Rng};
use std::fmt::Debug;

use super::{euclidean_distance, vq};
use crate::error::{ClusteringError, Result};

/// Options for K-means clustering
#[derive(Debug, Clone)]
pub struct KMeansOptions<F: Float> {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence threshold for centroid movement
    pub tol: F,
    /// Random seed for initialization
    pub random_seed: Option<u64>,
    /// Number of different initializations to try
    pub n_init: usize,
}

impl<F: Float + FromPrimitive> Default for KMeansOptions<F> {
    fn default() -> Self {
        Self {
            max_iter: 300,
            tol: F::from(1e-4).unwrap(),
            random_seed: None,
            n_init: 10,
        }
    }
}

/// K-means clustering algorithm
///
/// # Arguments
///
/// * `data` - Input data (n_samples × n_features)
/// * `k` - Number of clusters
/// * `options` - Optional parameters
///
/// # Returns
///
/// * Tuple of (centroids, labels) where:
///   - centroids: Array of shape (k × n_features)
///   - labels: Array of shape (n_samples,) with cluster assignments
///
/// # Examples
///
/// ```
/// use ndarray::{Array2, ArrayView2};
/// use scirs2_cluster::vq::kmeans;
///
/// let data = Array2::from_shape_vec((6, 2), vec![
///     1.0, 2.0,
///     1.2, 1.8,
///     0.8, 1.9,
///     3.7, 4.2,
///     3.9, 3.9,
///     4.2, 4.1,
/// ]).unwrap();
///
/// let (centroids, labels) = kmeans(ArrayView2::from(&data), 2, None).unwrap();
/// ```
pub fn kmeans<F>(
    data: ArrayView2<F>,
    k: usize,
    options: Option<KMeansOptions<F>>,
) -> Result<(Array2<F>, Array1<usize>)>
where
    F: Float + FromPrimitive + Debug,
{
    if k == 0 {
        return Err(ClusteringError::InvalidInput(
            "Number of clusters must be greater than 0".to_string(),
        ));
    }

    let n_samples = data.shape()[0];
    if n_samples == 0 {
        return Err(ClusteringError::InvalidInput(
            "Input data is empty".to_string(),
        ));
    }

    if k > n_samples {
        return Err(ClusteringError::InvalidInput(format!(
            "Number of clusters ({}) cannot be greater than number of data points ({})",
            k, n_samples
        )));
    }

    let opts = options.unwrap_or_default();
    // We create the RNG but don't use it directly in this function
    if opts.random_seed.is_some() {
        #[allow(unused_variables)]
        let seed = opts.random_seed.unwrap();
        // In a real implementation, we would seed the RNG here
    } else {
        // In a real implementation, we would use thread_rng here
    };

    let mut best_centroids = None;
    let mut best_labels = None;
    let mut best_inertia = F::infinity();

    for _ in 0..opts.n_init {
        // Initialize centroids using k-means++
        let centroids = kmeans_plus_plus(data, k, opts.random_seed)?;

        // Run k-means
        let (centroids, labels, inertia) = _kmeans_single(data, centroids.view(), &opts)?;

        if inertia < best_inertia {
            best_centroids = Some(centroids);
            best_labels = Some(labels);
            best_inertia = inertia;
        }
    }

    Ok((best_centroids.unwrap(), best_labels.unwrap()))
}

/// Run a single k-means clustering iteration
fn _kmeans_single<F>(
    data: ArrayView2<F>,
    init_centroids: ArrayView2<F>,
    opts: &KMeansOptions<F>,
) -> Result<(Array2<F>, Array1<usize>, F)>
where
    F: Float + FromPrimitive + Debug,
{
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];
    let k = init_centroids.shape()[0];

    let mut centroids = init_centroids.to_owned();
    let mut labels = Array1::zeros(n_samples);
    let mut prev_centroid_diff = F::infinity();

    for _iter in 0..opts.max_iter {
        // Assign samples to nearest centroid
        let (new_labels, distances) = vq(data, centroids.view())?;
        labels = new_labels;

        // Compute new centroids
        let mut new_centroids = Array2::zeros((k, n_features));
        let mut counts = Array1::zeros(k);

        for i in 0..n_samples {
            let cluster = labels[i];
            let point = data.slice(s![i, ..]);

            for j in 0..n_features {
                new_centroids[[cluster, j]] = new_centroids[[cluster, j]] + point[j];
            }

            counts[cluster] += 1;
        }

        // If a cluster is empty, reinitialize it
        for i in 0..k {
            if counts[i] == 0 {
                // Find the point furthest from its centroid
                let mut max_dist = F::zero();
                let mut far_idx = 0;

                for j in 0..n_samples {
                    let dist = distances[j];
                    if dist > max_dist {
                        max_dist = dist;
                        far_idx = j;
                    }
                }

                // Move this point to the empty cluster
                for j in 0..n_features {
                    new_centroids[[i, j]] = data[[far_idx, j]];
                }

                counts[i] = 1;
            } else {
                // Normalize by the number of points in the cluster
                for j in 0..n_features {
                    new_centroids[[i, j]] = new_centroids[[i, j]] / F::from(counts[i]).unwrap();
                }
            }
        }

        // Check for convergence
        let mut centroid_diff = F::zero();
        for i in 0..k {
            let dist =
                euclidean_distance(centroids.slice(s![i, ..]), new_centroids.slice(s![i, ..]));
            centroid_diff = centroid_diff + dist;
        }

        centroids = new_centroids;

        if centroid_diff <= opts.tol || centroid_diff >= prev_centroid_diff {
            break;
        }

        prev_centroid_diff = centroid_diff;
    }

    // Calculate inertia (sum of squared distances to nearest centroid)
    let mut inertia = F::zero();
    for i in 0..n_samples {
        let cluster = labels[i];
        let dist = euclidean_distance(data.slice(s![i, ..]), centroids.slice(s![cluster, ..]));
        inertia = inertia + dist * dist;
    }

    Ok((centroids, labels, inertia))
}

/// K-means++ initialization algorithm
///
/// # Arguments
///
/// * `data` - Input data (n_samples × n_features)
/// * `k` - Number of clusters
/// * `_random_seed` - Optional random seed
///
/// # Returns
///
/// * Array of shape (k × n_features) with initial centroids
pub fn kmeans_plus_plus<F>(
    data: ArrayView2<F>,
    k: usize,
    _random_seed: Option<u64>,
) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];

    if k == 0 || k > n_samples {
        return Err(ClusteringError::InvalidInput(format!(
            "Number of clusters ({}) must be between 1 and number of samples ({})",
            k, n_samples
        )));
    }

    let mut rng: ThreadRng = rand::rng();

    let mut centroids = Array2::zeros((k, n_features));

    // Choose the first centroid randomly
    let first_idx = rng.random_range(0..n_samples);
    for j in 0..n_features {
        centroids[[0, j]] = data[[first_idx, j]];
    }

    if k == 1 {
        return Ok(centroids);
    }

    // Choose remaining centroids using the k-means++ algorithm
    for i in 1..k {
        // Compute distances to closest centroid for each point
        let mut min_distances = Array1::from_elem(n_samples, F::infinity());

        for sample_idx in 0..n_samples {
            let sample = data.slice(s![sample_idx, ..]);

            for centroid_idx in 0..i {
                let centroid = centroids.slice(s![centroid_idx, ..]);
                let dist = euclidean_distance(sample, centroid);

                if dist < min_distances[sample_idx] {
                    min_distances[sample_idx] = dist;
                }
            }
        }

        // Square the distances to get the probability distribution
        let mut weights = min_distances.mapv(|d| d * d);

        // Normalize the weights to create a probability distribution
        let sum_weights = weights.sum();
        if sum_weights > F::zero() {
            weights.mapv_inplace(|w| w / sum_weights);
        } else {
            // If all weights are zero, use uniform distribution
            weights.fill(F::from(1.0 / n_samples as f64).unwrap());
        }

        // Convert weights to cumulative distribution
        let mut cum_weights = weights.clone();
        for j in 1..n_samples {
            cum_weights[j] = cum_weights[j] + cum_weights[j - 1];
        }

        // Sample the next centroid based on the probability distribution
        let rand_val = F::from(rng.random_range(0.0..1.0)).unwrap();
        let mut next_idx = 0;

        for j in 0..n_samples {
            if rand_val <= cum_weights[j] {
                next_idx = j;
                break;
            }
        }

        // Add the new centroid
        for j in 0..n_features {
            centroids[[i, j]] = data[[next_idx, j]];
        }
    }

    Ok(centroids)
}
