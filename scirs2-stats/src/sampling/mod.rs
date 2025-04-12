//! Statistical sampling
//!
//! This module provides functions for statistical sampling,
//! following SciPy's `stats.sampling` module.

use crate::error::{StatsError, StatsResult};
use crate::random;
use ndarray::{Array1, Array2, ArrayView1};
use num_traits::Float;
use rand::prelude::*;
use rand::SeedableRng;

/// Distribution trait for statistical distributions that can be sampled
pub trait SampleableDistribution<T> {
    /// Generate random samples from the distribution
    fn rvs(&self, size: usize) -> StatsResult<Vec<T>>;
}

/// Sample from a distribution
///
/// # Arguments
///
/// * `dist` - Distribution to sample from
/// * `size` - Number of samples to generate
///
/// # Returns
///
/// * Array of samples
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_stats::{sampling, distributions};
///
/// // Create a normal distribution
/// let normal = distributions::norm(0.0f64, 1.0).unwrap();
///
/// // Sample from it
/// let samples = sampling::sample_distribution(&normal, 100).unwrap();
/// assert_eq!(samples.len(), 100);
/// ```
pub fn sample_distribution<T, D>(dist: &D, size: usize) -> StatsResult<Array1<T>>
where
    T: Float + std::iter::Sum<T> + std::ops::Div<Output = T>,
    D: SampleableDistribution<T>,
{
    if size == 0 {
        return Err(StatsError::InvalidArgument(
            "Size must be positive".to_string(),
        ));
    }

    let samples = dist.rvs(size)?;
    Ok(Array1::from_vec(samples))
}

/// Nonparametric bootstrap
///
/// # Arguments
///
/// * `x` - Input array
/// * `n_resamples` - Number of bootstrap samples to generate
/// * `seed` - Optional seed for reproducibility
///
/// # Returns
///
/// * Bootstrap samples with replacement
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_stats::sampling;
///
/// // Create an array
/// let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
///
/// // Generate bootstrap samples
/// let samples = sampling::bootstrap(&data.view(), 10, Some(42)).unwrap();
/// assert_eq!(samples.shape(), &[10, 5]);
/// ```
pub fn bootstrap<T>(
    x: &ArrayView1<T>,
    n_resamples: usize,
    seed: Option<u64>,
) -> StatsResult<Array2<T>>
where
    T: Copy + num_traits::Zero,
{
    random::bootstrap_sample(x, n_resamples, seed)
}

/// Random permutation
///
/// # Arguments
///
/// * `x` - Input array
/// * `seed` - Optional seed for reproducibility
///
/// # Returns
///
/// * Randomly permuted array
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_stats::sampling;
///
/// // Create an array
/// let data = array![1, 2, 3, 4, 5];
///
/// // Generate a permutation
/// let perm = sampling::permutation(&data.view(), Some(42)).unwrap();
/// assert_eq!(perm.len(), 5);
/// ```
pub fn permutation<T>(x: &ArrayView1<T>, seed: Option<u64>) -> StatsResult<Array1<T>>
where
    T: Copy,
{
    random::permutation(x, seed)
}

/// Generate stratified random sample
///
/// # Arguments
///
/// * `x` - Input array
/// * `groups` - Group labels for each element in x
/// * `size` - Number of samples per group
/// * `seed` - Optional seed for reproducibility
///
/// # Returns
///
/// * Stratified sample indices
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_stats::sampling;
///
/// // Create an array and group labels
/// let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let groups = array![0, 0, 1, 1, 2, 2];
///
/// // Generate a stratified sample with 1 sample per group
/// let indices = sampling::stratified_sample(&data.view(), &groups.view(), 1, Some(42)).unwrap();
/// assert_eq!(indices.len(), 3);  // 3 groups with 1 sample each
/// ```
pub fn stratified_sample<T, G>(
    x: &ArrayView1<T>,
    groups: &ArrayView1<G>,
    size: usize,
    seed: Option<u64>,
) -> StatsResult<Array1<usize>>
where
    T: Copy,
    G: Copy + Eq + std::hash::Hash,
{
    if x.len() != groups.len() {
        return Err(StatsError::DimensionMismatch(
            "Input array and group array must have the same length".to_string(),
        ));
    }

    if size == 0 {
        return Err(StatsError::InvalidArgument(
            "Size must be positive".to_string(),
        ));
    }

    // Get unique groups
    let mut unique_groups = std::collections::HashSet::new();
    for &g in groups.iter() {
        unique_groups.insert(g);
    }

    let n_groups = unique_groups.len();

    // Create map of group -> indices
    let mut group_indices = std::collections::HashMap::new();
    for (i, &g) in groups.iter().enumerate() {
        group_indices.entry(g).or_insert_with(Vec::new).push(i);
    }

    // Initialize RNG
    let mut rng = match seed {
        Some(seed_value) => rand::rngs::StdRng::seed_from_u64(seed_value),
        None => {
            // Get a seed from the system RNG
            let mut system_rng = rand::rng();
            let seed = system_rng.random::<u64>();
            rand::rngs::StdRng::seed_from_u64(seed)
        }
    };

    // Sample from each group
    let mut result = Vec::with_capacity(n_groups * size);

    for (_, indices) in group_indices.iter() {
        if indices.len() < size {
            return Err(StatsError::InvalidArgument(format!(
                "Group size {} is smaller than requested sample size {}",
                indices.len(),
                size
            )));
        }

        // Sample without replacement using Fisher-Yates shuffle
        let mut indices_copy = indices.clone();
        for i in 0..size {
            let j = rng.random_range(i..indices_copy.len());
            indices_copy.swap(i, j);
            result.push(indices_copy[i]);
        }
    }

    Ok(Array1::from_vec(result))
}
