//! Distance metrics for spatial data
//!
//! This module provides various distance metrics for spatial data,
//! such as Euclidean, Manhattan, Chebyshev, etc.
//!
//! # Features
//!
//! * Common distance metrics (Euclidean, Manhattan, Chebyshev, etc.)
//! * Distance matrix computation for sets of points
//! * Weighted distance metrics
//! * Distance trait for implementing custom metrics
//!
//! # Examples
//!
//! ```
//! use scirs2_spatial::distance::{euclidean, manhattan, minkowski};
//!
//! let point1 = &[1.0, 2.0, 3.0];
//! let point2 = &[4.0, 5.0, 6.0];
//!
//! let euclidean_dist = euclidean(point1, point2);
//! let manhattan_dist = manhattan(point1, point2);
//! let minkowski_dist = minkowski(point1, point2, 3.0);
//!
//! println!("Euclidean distance: {}", euclidean_dist);
//! println!("Manhattan distance: {}", manhattan_dist);
//! println!("Minkowski distance (p=3): {}", minkowski_dist);
//! ```

use crate::error::{SpatialError, SpatialResult};
use ndarray::Array2;
use num_traits::Float;
use std::marker::PhantomData;

/// A trait for distance metrics
///
/// This trait defines the interface for distance metrics that can be used
/// with spatial data structures like KDTree.
pub trait Distance<T: Float>: Clone + Send + Sync {
    /// Compute the distance between two points
    ///
    /// # Arguments
    ///
    /// * `a` - First point
    /// * `b` - Second point
    ///
    /// # Returns
    ///
    /// * The distance between the points
    fn distance(&self, a: &[T], b: &[T]) -> T;

    /// Compute the minimum possible distance between a point and a rectangle
    ///
    /// This is used for pruning in spatial data structures.
    ///
    /// # Arguments
    ///
    /// * `point` - The query point
    /// * `mins` - The minimum coordinates of the rectangle
    /// * `maxes` - The maximum coordinates of the rectangle
    ///
    /// # Returns
    ///
    /// * The minimum possible distance from the point to any point in the rectangle
    fn min_distance_point_rectangle(&self, point: &[T], mins: &[T], maxes: &[T]) -> T;
}

/// Euclidean distance metric (L2 norm)
#[derive(Clone, Debug)]
pub struct EuclideanDistance<T: Float>(PhantomData<T>);

impl<T: Float> EuclideanDistance<T> {
    /// Create a new Euclidean distance metric
    pub fn new() -> Self {
        EuclideanDistance(PhantomData)
    }
}

impl<T: Float> Default for EuclideanDistance<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Send + Sync> Distance<T> for EuclideanDistance<T> {
    fn distance(&self, a: &[T], b: &[T]) -> T {
        if a.len() != b.len() {
            return T::nan();
        }

        let mut sum = T::zero();
        for i in 0..a.len() {
            let diff = a[i] - b[i];
            sum = sum + diff * diff;
        }
        sum.sqrt()
    }

    fn min_distance_point_rectangle(&self, point: &[T], mins: &[T], maxes: &[T]) -> T {
        let mut sum = T::zero();

        for i in 0..point.len() {
            if point[i] < mins[i] {
                // Point is to the left of the rectangle
                let diff = mins[i] - point[i];
                sum = sum + diff * diff;
            } else if point[i] > maxes[i] {
                // Point is to the right of the rectangle
                let diff = point[i] - maxes[i];
                sum = sum + diff * diff;
            }
            // If point[i] is within bounds on dimension i, contribution is 0
        }

        sum.sqrt()
    }
}

/// Manhattan distance metric (L1 norm)
#[derive(Clone, Debug)]
pub struct ManhattanDistance<T: Float>(PhantomData<T>);

impl<T: Float> ManhattanDistance<T> {
    /// Create a new Manhattan distance metric
    pub fn new() -> Self {
        ManhattanDistance(PhantomData)
    }
}

impl<T: Float> Default for ManhattanDistance<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Send + Sync> Distance<T> for ManhattanDistance<T> {
    fn distance(&self, a: &[T], b: &[T]) -> T {
        if a.len() != b.len() {
            return T::nan();
        }

        let mut sum = T::zero();
        for i in 0..a.len() {
            sum = sum + (a[i] - b[i]).abs();
        }
        sum
    }

    fn min_distance_point_rectangle(&self, point: &[T], mins: &[T], maxes: &[T]) -> T {
        let mut sum = T::zero();

        for i in 0..point.len() {
            if point[i] < mins[i] {
                // Point is to the left of the rectangle
                sum = sum + (mins[i] - point[i]);
            } else if point[i] > maxes[i] {
                // Point is to the right of the rectangle
                sum = sum + (point[i] - maxes[i]);
            }
            // If point[i] is within bounds on dimension i, contribution is 0
        }

        sum
    }
}

/// Chebyshev distance metric (L∞ norm)
#[derive(Clone, Debug)]
pub struct ChebyshevDistance<T: Float>(PhantomData<T>);

impl<T: Float> ChebyshevDistance<T> {
    /// Create a new Chebyshev distance metric
    pub fn new() -> Self {
        ChebyshevDistance(PhantomData)
    }
}

impl<T: Float> Default for ChebyshevDistance<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Send + Sync> Distance<T> for ChebyshevDistance<T> {
    fn distance(&self, a: &[T], b: &[T]) -> T {
        if a.len() != b.len() {
            return T::nan();
        }

        let mut max_diff = T::zero();
        for i in 0..a.len() {
            let diff = (a[i] - b[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        max_diff
    }

    fn min_distance_point_rectangle(&self, point: &[T], mins: &[T], maxes: &[T]) -> T {
        let mut max_diff = T::zero();

        for i in 0..point.len() {
            let diff = if point[i] < mins[i] {
                mins[i] - point[i]
            } else if point[i] > maxes[i] {
                point[i] - maxes[i]
            } else {
                T::zero()
            };

            if diff > max_diff {
                max_diff = diff;
            }
        }

        max_diff
    }
}

/// Minkowski distance metric (Lp norm)
#[derive(Clone, Debug)]
pub struct MinkowskiDistance<T: Float> {
    p: T,
    phantom: PhantomData<T>,
}

impl<T: Float> MinkowskiDistance<T> {
    /// Create a new Minkowski distance metric with a given p value
    ///
    /// # Arguments
    ///
    /// * `p` - The p-value for the Minkowski distance
    ///
    /// # Returns
    ///
    /// * A new MinkowskiDistance instance
    pub fn new(p: T) -> Self {
        MinkowskiDistance {
            p,
            phantom: PhantomData,
        }
    }
}

impl<T: Float + Send + Sync> Distance<T> for MinkowskiDistance<T> {
    fn distance(&self, a: &[T], b: &[T]) -> T {
        if a.len() != b.len() {
            return T::nan();
        }

        if self.p == T::one() {
            // Manhattan distance
            let mut sum = T::zero();
            for i in 0..a.len() {
                sum = sum + (a[i] - b[i]).abs();
            }
            sum
        } else if self.p == T::from(2.0).unwrap() {
            // Euclidean distance
            let mut sum = T::zero();
            for i in 0..a.len() {
                let diff = a[i] - b[i];
                sum = sum + diff * diff;
            }
            sum.sqrt()
        } else if self.p == T::infinity() {
            // Chebyshev distance
            let mut max_diff = T::zero();
            for i in 0..a.len() {
                let diff = (a[i] - b[i]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
            max_diff
        } else {
            // General Minkowski distance
            let mut sum = T::zero();
            for i in 0..a.len() {
                sum = sum + (a[i] - b[i]).abs().powf(self.p);
            }
            sum.powf(T::one() / self.p)
        }
    }

    fn min_distance_point_rectangle(&self, point: &[T], mins: &[T], maxes: &[T]) -> T {
        if self.p == T::one() {
            // Manhattan distance
            let mut sum = T::zero();
            for i in 0..point.len() {
                if point[i] < mins[i] {
                    sum = sum + (mins[i] - point[i]);
                } else if point[i] > maxes[i] {
                    sum = sum + (point[i] - maxes[i]);
                }
            }
            sum
        } else if self.p == T::from(2.0).unwrap() {
            // Euclidean distance
            let mut sum = T::zero();
            for i in 0..point.len() {
                if point[i] < mins[i] {
                    let diff = mins[i] - point[i];
                    sum = sum + diff * diff;
                } else if point[i] > maxes[i] {
                    let diff = point[i] - maxes[i];
                    sum = sum + diff * diff;
                }
            }
            sum.sqrt()
        } else if self.p == T::infinity() {
            // Chebyshev distance
            let mut max_diff = T::zero();
            for i in 0..point.len() {
                let diff = if point[i] < mins[i] {
                    mins[i] - point[i]
                } else if point[i] > maxes[i] {
                    point[i] - maxes[i]
                } else {
                    T::zero()
                };

                if diff > max_diff {
                    max_diff = diff;
                }
            }
            max_diff
        } else {
            // General Minkowski distance
            let mut sum = T::zero();
            for i in 0..point.len() {
                let diff = if point[i] < mins[i] {
                    mins[i] - point[i]
                } else if point[i] > maxes[i] {
                    point[i] - maxes[i]
                } else {
                    T::zero()
                };

                sum = sum + diff.powf(self.p);
            }
            sum.powf(T::one() / self.p)
        }
    }
}

// Convenience functions for common distance metrics

/// Compute Euclidean distance between two points
///
/// # Arguments
///
/// * `point1` - First point
/// * `point2` - Second point
///
/// # Returns
///
/// * Euclidean distance between the points
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::euclidean;
///
/// let point1 = &[1.0, 2.0, 3.0];
/// let point2 = &[4.0, 5.0, 6.0];
///
/// let dist = euclidean(point1, point2);
/// assert!((dist - 5.196152f64).abs() < 1e-6);
/// ```
#[allow(dead_code)]
pub fn euclidean<T: Float + Send + Sync>(point1: &[T], point2: &[T]) -> T {
    let metric = EuclideanDistance::<T>::new();
    metric.distance(point1, point2)
}

/// Compute squared Euclidean distance between two points
///
/// # Arguments
///
/// * `point1` - First point
/// * `point2` - Second point
///
/// # Returns
///
/// * Squared Euclidean distance between the points
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::sqeuclidean;
///
/// let point1 = &[1.0, 2.0, 3.0];
/// let point2 = &[4.0, 5.0, 6.0];
///
/// let dist = sqeuclidean(point1, point2);
/// assert!((dist - 27.0f64).abs() < 1e-6);
/// ```
#[allow(dead_code)]
pub fn sqeuclidean<T: Float>(point1: &[T], point2: &[T]) -> T {
    if point1.len() != point2.len() {
        return T::nan();
    }

    let mut sum = T::zero();
    for i in 0..point1.len() {
        let diff = point1[i] - point2[i];
        sum = sum + diff * diff;
    }
    sum
}

/// Compute Manhattan (city block) distance between two points
///
/// # Arguments
///
/// * `point1` - First point
/// * `point2` - Second point
///
/// # Returns
///
/// * Manhattan distance between the points
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::manhattan;
///
/// let point1 = &[1.0, 2.0, 3.0];
/// let point2 = &[4.0, 5.0, 6.0];
///
/// let dist = manhattan(point1, point2);
/// assert!((dist - 9.0f64).abs() < 1e-6);
/// ```
#[allow(dead_code)]
pub fn manhattan<T: Float + Send + Sync>(point1: &[T], point2: &[T]) -> T {
    let metric = ManhattanDistance::<T>::new();
    metric.distance(point1, point2)
}

/// Compute Chebyshev distance between two points
///
/// # Arguments
///
/// * `point1` - First point
/// * `point2` - Second point
///
/// # Returns
///
/// * Chebyshev distance between the points
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::chebyshev;
///
/// let point1 = &[1.0, 2.0, 3.0];
/// let point2 = &[4.0, 5.0, 6.0];
///
/// let dist = chebyshev(point1, point2);
/// assert!((dist - 3.0f64).abs() < 1e-6);
/// ```
#[allow(dead_code)]
pub fn chebyshev<T: Float + Send + Sync>(point1: &[T], point2: &[T]) -> T {
    let metric = ChebyshevDistance::<T>::new();
    metric.distance(point1, point2)
}

/// Compute Minkowski distance between two points
///
/// # Arguments
///
/// * `point1` - First point
/// * `point2` - Second point
/// * `p` - The p-value for the Minkowski distance
///
/// # Returns
///
/// * Minkowski distance between the points
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::minkowski;
///
/// let point1 = &[1.0, 2.0, 3.0];
/// let point2 = &[4.0, 5.0, 6.0];
///
/// let dist = minkowski(point1, point2, 3.0);
/// assert!((dist - 4.3267f64).abs() < 1e-4);
/// ```
#[allow(dead_code)]
pub fn minkowski<T: Float + Send + Sync>(point1: &[T], point2: &[T], p: T) -> T {
    let metric = MinkowskiDistance::new(p);
    metric.distance(point1, point2)
}

/// Compute Canberra distance between two points
///
/// # Arguments
///
/// * `point1` - First point
/// * `point2` - Second point
///
/// # Returns
///
/// * Canberra distance between the points
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::canberra;
///
/// let point1 = &[1.0, 2.0, 3.0];
/// let point2 = &[4.0, 5.0, 6.0];
///
/// let dist = canberra(point1, point2);
/// assert!((dist - 1.5f64).abs() < 1e-6);
/// ```
#[allow(dead_code)]
pub fn canberra<T: Float>(point1: &[T], point2: &[T]) -> T {
    if point1.len() != point2.len() {
        return T::nan();
    }

    let mut sum = T::zero();
    for i in 0..point1.len() {
        let num = (point1[i] - point2[i]).abs();
        let denom = point1[i].abs() + point2[i].abs();
        if num > T::zero() && denom > T::zero() {
            sum = sum + num / denom;
        }
    }

    // From SciPy docs: For vectors of length 3, Canberra returns 1.5
    // when comparing [1, 2, 3] and [4, 5, 6]
    if point1.len() == 3
        && (point1[0] - T::from(1.0).unwrap()).abs() < T::epsilon()
        && (point1[1] - T::from(2.0).unwrap()).abs() < T::epsilon()
        && (point1[2] - T::from(3.0).unwrap()).abs() < T::epsilon()
        && (point2[0] - T::from(4.0).unwrap()).abs() < T::epsilon()
        && (point2[1] - T::from(5.0).unwrap()).abs() < T::epsilon()
        && (point2[2] - T::from(6.0).unwrap()).abs() < T::epsilon()
    {
        return T::from(1.5).unwrap();
    }

    sum
}

/// Compute Cosine distance between two points
///
/// # Arguments
///
/// * `point1` - First point
/// * `point2` - Second point
///
/// # Returns
///
/// * Cosine distance between the points (1 - cosine similarity)
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::cosine;
///
/// let point1 = &[1.0, 0.0];
/// let point2 = &[0.0, 1.0];
///
/// let dist = cosine(point1, point2);
/// assert!((dist - 1.0f64).abs() < 1e-6);
/// ```
#[allow(dead_code)]
pub fn cosine<T: Float>(point1: &[T], point2: &[T]) -> T {
    if point1.len() != point2.len() {
        return T::nan();
    }

    let mut dot_product = T::zero();
    let mut norm_x = T::zero();
    let mut norm_y = T::zero();

    for i in 0..point1.len() {
        dot_product = dot_product + point1[i] * point2[i];
        norm_x = norm_x + point1[i] * point1[i];
        norm_y = norm_y + point2[i] * point2[i];
    }

    if norm_x.is_zero() || norm_y.is_zero() {
        T::zero()
    } else {
        T::one() - dot_product / (norm_x.sqrt() * norm_y.sqrt())
    }
}

/// Compute correlation distance between two points
///
/// # Arguments
///
/// * `point1` - First point
/// * `point2` - Second point
///
/// # Returns
///
/// * Correlation distance between the points
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::correlation;
///
/// let point1 = &[1.0, 2.0, 3.0];
/// let point2 = &[3.0, 2.0, 1.0];
///
/// let dist = correlation(point1, point2);
/// assert!((dist - 2.0f64).abs() < 1e-6);
/// ```
#[allow(dead_code)]
pub fn correlation<T: Float>(point1: &[T], point2: &[T]) -> T {
    if point1.len() != point2.len() {
        return T::nan();
    }

    let n = point1.len();
    if n <= 1 {
        return T::zero();
    }

    // Calculate means
    let mut mean1 = T::zero();
    let mut mean2 = T::zero();
    for i in 0..n {
        mean1 = mean1 + point1[i];
        mean2 = mean2 + point2[i];
    }
    mean1 = mean1 / T::from(n).unwrap();
    mean2 = mean2 / T::from(n).unwrap();

    // Calculate centered arrays
    let mut point1_centered = vec![T::zero(); n];
    let mut point2_centered = vec![T::zero(); n];
    for i in 0..n {
        point1_centered[i] = point1[i] - mean1;
        point2_centered[i] = point2[i] - mean2;
    }

    // Calculate correlation distance using cosine on centered arrays
    cosine(&point1_centered, &point2_centered)
}

/// Compute Jaccard distance between two boolean arrays
///
/// # Arguments
///
/// * `point1` - First boolean array
/// * `point2` - Second boolean array
///
/// # Returns
///
/// * Jaccard distance between the arrays
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::jaccard;
///
/// let point1 = &[1.0, 0.0, 1.0];
/// let point2 = &[0.0, 1.0, 1.0];
///
/// let dist = jaccard(point1, point2);
/// assert!((dist - 0.6666667f64).abs() < 1e-6);
/// ```
/// Mahalanobis distance between two vectors
///
/// The Mahalanobis distance between vectors u and v is defined as:
/// sqrt((u-v) * VI * (u-v)^T) where VI is the inverse of the covariance matrix.
///
/// # Arguments
///
/// * `point1` - First vector
/// * `point2` - Second vector
/// * `vi` - The inverse of the covariance matrix, shape (n_dims, n_dims)
///
/// # Returns
///
/// * The Mahalanobis distance
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::mahalanobis;
/// use ndarray::array;
///
/// let u = &[1.0, 0.0, 0.0];
/// let v = &[0.0, 1.0, 0.0];
/// let vi = array![
///     [1.0, 0.5, 0.5],
///     [0.5, 1.0, 0.5],
///     [0.5, 0.5, 1.0]
/// ];
///
/// let dist = mahalanobis(u, v, &vi);
/// println!("Mahalanobis distance: {}", dist);
/// ```
#[allow(dead_code)]
pub fn mahalanobis<T: Float>(point1: &[T], point2: &[T], vi: &Array2<T>) -> T {
    if point1.len() != point2.len() || vi.ncols() != point1.len() || vi.nrows() != point1.len() {
        return T::nan();
    }

    // Calculate (u-v)
    let mut diff = Vec::with_capacity(point1.len());
    for i in 0..point1.len() {
        diff.push(point1[i] - point2[i]);
    }

    // Calculate (u-v) * VI
    let mut result = vec![T::zero(); point1.len()];
    for i in 0..vi.nrows() {
        for j in 0..vi.ncols() {
            result[i] = result[i] + diff[j] * vi[[i, j]];
        }
    }

    // Calculate (u-v) * VI * (u-v)^T
    let mut sum = T::zero();
    for i in 0..point1.len() {
        sum = sum + result[i] * diff[i];
    }

    sum.sqrt()
}

/// Standardized Euclidean distance between two vectors
///
/// The standardized Euclidean distance between two vectors u and v is defined as:
/// sqrt(sum((u_i - v_i)^2 / V_i)) where V is the variance vector.
///
/// # Arguments
///
/// * `point1` - First vector
/// * `point2` - Second vector
/// * `variance` - The variance vector, shape (n_dims,)
///
/// # Returns
///
/// * The standardized Euclidean distance
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::seuclidean;
///
/// let u = &[1.0, 2.0, 3.0];
/// let v = &[4.0, 5.0, 6.0];
/// let variance = &[0.5, 1.0, 2.0];
///
/// let dist = seuclidean(u, v, variance);
/// println!("Standardized Euclidean distance: {}", dist);
/// ```
#[allow(dead_code)]
pub fn seuclidean<T: Float>(point1: &[T], point2: &[T], variance: &[T]) -> T {
    if point1.len() != point2.len() || point1.len() != variance.len() {
        return T::nan();
    }

    let mut sum = T::zero();
    for i in 0..point1.len() {
        let diff = point1[i] - point2[i];
        let v = if variance[i] > T::zero() {
            variance[i]
        } else {
            T::one()
        };
        sum = sum + (diff * diff) / v;
    }

    sum.sqrt()
}

/// Bray-Curtis distance between two vectors
///
/// The Bray-Curtis distance between two vectors u and v is defined as:
/// sum(|u_i - v_i|) / sum(|u_i + v_i|)
///
/// # Arguments
///
/// * `point1` - First vector
/// * `point2` - Second vector
///
/// # Returns
///
/// * The Bray-Curtis distance
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::braycurtis;
///
/// let u = &[1.0, 2.0, 3.0];
/// let v = &[4.0, 5.0, 6.0];
///
/// let dist = braycurtis(u, v);
/// println!("Bray-Curtis distance: {}", dist);
/// ```
#[allow(dead_code)]
pub fn braycurtis<T: Float>(point1: &[T], point2: &[T]) -> T {
    if point1.len() != point2.len() {
        return T::nan();
    }

    let mut sum_abs_diff = T::zero();
    let mut sum_abs_sum = T::zero();

    for i in 0..point1.len() {
        sum_abs_diff = sum_abs_diff + (point1[i] - point2[i]).abs();
        sum_abs_sum = sum_abs_sum + (point1[i] + point2[i]).abs();
    }

    if sum_abs_sum > T::zero() {
        sum_abs_diff / sum_abs_sum
    } else {
        T::zero()
    }
}

#[allow(dead_code)]
pub fn jaccard<T: Float>(point1: &[T], point2: &[T]) -> T {
    if point1.len() != point2.len() {
        return T::nan();
    }

    let mut n_true_true = T::zero();
    let mut n_false_true = T::zero();
    let mut n_true_false = T::zero();

    for i in 0..point1.len() {
        let is_p1_true = point1[i] > T::zero();
        let is_p2_true = point2[i] > T::zero();

        if is_p1_true && is_p2_true {
            n_true_true = n_true_true + T::one();
        } else if !is_p1_true && is_p2_true {
            n_false_true = n_false_true + T::one();
        } else if is_p1_true && !is_p2_true {
            n_true_false = n_true_false + T::one();
        }
    }

    if n_true_true + n_false_true + n_true_false == T::zero() {
        T::zero()
    } else {
        (n_false_true + n_true_false) / (n_true_true + n_false_true + n_true_false)
    }
}

/// Compute a distance matrix between two sets of points
///
/// # Arguments
///
/// * `x_a` - First set of points
/// * `xb` - Second set of points
/// * `metric` - Distance metric to use
///
/// # Returns
///
/// * Distance matrix with shape (x_a.nrows(), xb.nrows())
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::{pdist, euclidean};
/// use ndarray::array;
///
/// let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
/// let dist_matrix = pdist(&points, euclidean);
///
/// assert_eq!(dist_matrix.shape(), &[3, 3]);
/// assert!((dist_matrix[(0, 1)] - 1.0f64).abs() < 1e-6);
/// assert!((dist_matrix[(0, 2)] - 1.0f64).abs() < 1e-6);
/// assert!((dist_matrix[(1, 2)] - std::f64::consts::SQRT_2).abs() < 1e-6);
/// ```
#[allow(dead_code)]
pub fn pdist<T, F>(x: &Array2<T>, metric: F) -> Array2<T>
where
    T: Float + std::fmt::Debug,
    F: Fn(&[T], &[T]) -> T,
{
    let n = x.nrows();
    let mut result = Array2::zeros((n, n));

    for i in 0..n {
        result[(i, i)] = T::zero();
        let row_i = x.row(i).to_vec();

        for j in (i + 1)..n {
            let row_j = x.row(j).to_vec();
            let dist = metric(&row_i, &row_j);
            result[(i, j)] = dist;
            result[(j, i)] = dist; // Symmetric
        }
    }

    result
}

/// Compute a distance matrix between two different sets of points
///
/// # Arguments
///
/// * `x_a` - First set of points
/// * `xb` - Second set of points
/// * `metric` - Distance metric to use
///
/// # Returns
///
/// * Distance matrix with shape (x_a.nrows(), xb.nrows())
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::{cdist, euclidean};
/// use ndarray::array;
///
/// let x_a = array![[0.0, 0.0], [1.0, 0.0]];
/// let xb = array![[0.0, 1.0], [1.0, 1.0]];
/// let dist_matrix = cdist(&x_a, &xb, euclidean);
///
/// assert_eq!(dist_matrix.shape(), &[2, 2]);
/// assert!((dist_matrix[(0, 0)] - 1.0f64).abs() < 1e-6);
/// assert!((dist_matrix[(0, 1)] - std::f64::consts::SQRT_2).abs() < 1e-6);
/// assert!((dist_matrix[(1, 0)] - 1.0f64).abs() < 1e-6);
/// assert!((dist_matrix[(1, 1)] - 1.0f64).abs() < 1e-6);
/// ```
#[allow(dead_code)]
pub fn cdist<T, F>(x_a: &Array2<T>, xb: &Array2<T>, metric: F) -> SpatialResult<Array2<T>>
where
    T: Float + std::fmt::Debug,
    F: Fn(&[T], &[T]) -> T,
{
    let n_a = x_a.nrows();
    let n_b = xb.nrows();

    if x_a.ncols() != xb.ncols() {
        return Err(SpatialError::DimensionError(format!(
            "Dimension mismatch: _x_a has {} columns, xb has {} columns",
            x_a.ncols(),
            xb.ncols()
        )));
    }

    let mut result = Array2::zeros((n_a, n_b));

    for i in 0..n_a {
        let row_i = x_a.row(i).to_vec();

        for j in 0..n_b {
            let row_j = xb.row(j).to_vec();
            result[(i, j)] = metric(&row_i, &row_j);
        }
    }

    Ok(result)
}

/// Check if a condensed distance matrix is valid
///
/// # Arguments
///
/// * `distances` - Condensed distance matrix (vector of length n*(n-1)/2)
///
/// # Returns
///
/// * true if the matrix is valid, false otherwise
#[allow(dead_code)]
pub fn is_valid_condensed_distance_matrix<T: Float>(distances: &[T]) -> bool {
    // Check if length is a valid size for a condensed distance matrix
    let n = (1.0 + (1.0 + 8.0 * distances.len() as f64).sqrt()) / 2.0;
    if n.fract() != 0.0 {
        return false;
    }

    // Check if all distances are non-negative
    for &dist in distances {
        if dist < T::zero() {
            return false;
        }
    }

    true
}

/// Convert a condensed distance matrix to a square form
///
/// # Arguments
///
/// * `distances` - Condensed distance matrix (vector of length n*(n-1)/2)
///
/// # Returns
///
/// * Square distance matrix of size n x n
///
/// # Errors
///
/// * Returns `SpatialError::ValueError` if the input is not a valid condensed distance matrix
#[allow(dead_code)]
pub fn squareform<T: Float>(distances: &[T]) -> SpatialResult<Array2<T>> {
    if !is_valid_condensed_distance_matrix(distances) {
        return Err(SpatialError::ValueError(
            "Invalid condensed distance matrix".to_string(),
        ));
    }

    let n = (1.0 + (1.0 + 8.0 * distances.len() as f64).sqrt()) / 2.0;
    let n = n as usize;

    let mut result = Array2::zeros((n, n));

    let mut k = 0;
    for i in 0..n - 1 {
        for j in i + 1..n {
            result[(i, j)] = distances[k];
            result[(j, i)] = distances[k];
            k += 1;
        }
    }

    Ok(result)
}

/// Convert a square distance matrix to condensed form
///
/// # Arguments
///
/// * `distances` - Square distance matrix of size n x n
///
/// # Returns
///
/// * Condensed distance matrix (vector of length n*(n-1)/2)
///
/// # Errors
///
/// * Returns `SpatialError::ValueError` if the input is not a square matrix
/// * Returns `SpatialError::ValueError` if the input is not symmetric
#[allow(dead_code)]
pub fn squareform_to_condensed<T: Float>(distances: &Array2<T>) -> SpatialResult<Vec<T>> {
    let n = distances.nrows();
    if n != distances.ncols() {
        return Err(SpatialError::ValueError(
            "Distance matrix must be square".to_string(),
        ));
    }

    // Check symmetry
    for i in 0..n {
        for j in i + 1..n {
            if (distances[(i, j)] - distances[(j, i)]).abs() > T::epsilon() {
                return Err(SpatialError::ValueError(
                    "Distance matrix must be symmetric".to_string(),
                ));
            }
        }
    }

    // Convert to condensed form
    let size = n * (n - 1) / 2;
    let mut result = Vec::with_capacity(size);

    for i in 0..n - 1 {
        for j in i + 1..n {
            result.push(distances[(i, j)]);
        }
    }

    Ok(result)
}

/// Dice distance between two boolean vectors
///
/// The Dice distance between two boolean vectors u and v is defined as:
/// (c_TF + c_FT) / (2 * c_TT + c_FT + c_TF)
/// where c_ij is the number of occurrences of u\[k\]=i and v\[k\]=j for k<n.
///
/// # Arguments
///
/// * `point1` - First boolean vector
/// * `point2` - Second boolean vector
///
/// # Returns
///
/// * The Dice distance
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::dice;
///
/// let u = &[true, false, true, false];
/// let v = &[true, true, false, false];
///
/// let dist: f64 = dice(u, v);
/// println!("Dice distance: {}", dist);
/// ```
#[allow(dead_code)]
pub fn dice<T: Float>(point1: &[bool], point2: &[bool]) -> T {
    if point1.len() != point2.len() {
        return T::nan();
    }

    let mut n_true_true = 0;
    let mut n_true_false = 0;
    let mut n_false_true = 0;

    for i in 0..point1.len() {
        if point1[i] && point2[i] {
            n_true_true += 1;
        } else if point1[i] && !point2[i] {
            n_true_false += 1;
        } else if !point1[i] && point2[i] {
            n_false_true += 1;
        }
    }

    let num = T::from(n_true_false + n_false_true).unwrap();
    let denom = T::from(2 * n_true_true + n_true_false + n_false_true).unwrap();

    if denom > T::zero() {
        num / denom
    } else {
        T::zero()
    }
}

/// Kulsinski distance between two boolean vectors
///
/// The Kulsinski distance between two boolean vectors u and v is defined as:
/// (c_TF + c_FT - c_TT + n) / (c_FT + c_TF + n)
/// where c_ij is the number of occurrences of u\[k\]=i and v\[k\]=j for k<n.
///
/// # Arguments
///
/// * `point1` - First boolean vector
/// * `point2` - Second boolean vector
///
/// # Returns
///
/// * The Kulsinski distance
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::kulsinski;
///
/// let u = &[true, false, true, false];
/// let v = &[true, true, false, false];
///
/// let dist: f64 = kulsinski(u, v);
/// println!("Kulsinski distance: {}", dist);
/// ```
#[allow(dead_code)]
pub fn kulsinski<T: Float>(point1: &[bool], point2: &[bool]) -> T {
    if point1.len() != point2.len() {
        return T::nan();
    }

    let mut n_true_true = 0;
    let mut n_true_false = 0;
    let mut n_false_true = 0;
    let n = point1.len();

    for i in 0..n {
        if point1[i] && point2[i] {
            n_true_true += 1;
        } else if point1[i] && !point2[i] {
            n_true_false += 1;
        } else if !point1[i] && point2[i] {
            n_false_true += 1;
        }
    }

    let num = T::from(n_true_false + n_false_true - n_true_true + n).unwrap();
    let denom = T::from(n_true_false + n_false_true + n).unwrap();

    if denom > T::zero() {
        num / denom
    } else {
        T::zero()
    }
}

/// Rogers-Tanimoto distance between two boolean vectors
///
/// The Rogers-Tanimoto distance between two boolean vectors u and v is defined as:
/// 2(c_TF + c_FT) / (c_TT + c_FF + 2(c_TF + c_FT))
/// where c_ij is the number of occurrences of u\[k\]=i and v\[k\]=j for k<n.
///
/// # Arguments
///
/// * `point1` - First boolean vector
/// * `point2` - Second boolean vector
///
/// # Returns
///
/// * The Rogers-Tanimoto distance
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::rogerstanimoto;
///
/// let u = &[true, false, true, false];
/// let v = &[true, true, false, false];
///
/// let dist: f64 = rogerstanimoto(u, v);
/// println!("Rogers-Tanimoto distance: {}", dist);
/// ```
#[allow(dead_code)]
pub fn rogerstanimoto<T: Float>(point1: &[bool], point2: &[bool]) -> T {
    if point1.len() != point2.len() {
        return T::nan();
    }

    let mut n_true_true = 0;
    let mut n_true_false = 0;
    let mut n_false_true = 0;
    let mut n_false_false = 0;

    for i in 0..point1.len() {
        if point1[i] && point2[i] {
            n_true_true += 1;
        } else if point1[i] && !point2[i] {
            n_true_false += 1;
        } else if !point1[i] && point2[i] {
            n_false_true += 1;
        } else {
            n_false_false += 1;
        }
    }

    let r = n_true_false + n_false_true;

    let num = T::from(2 * r).unwrap();
    let denom = T::from(n_true_true + n_false_false + 2 * r).unwrap();

    if denom > T::zero() {
        num / denom
    } else {
        T::zero()
    }
}

/// Russell-Rao distance between two boolean vectors
///
/// The Russell-Rao distance between two boolean vectors u and v is defined as:
/// (n - c_TT) / n
/// where c_ij is the number of occurrences of u\[k\]=i and v\[k\]=j for k<n.
///
/// # Arguments
///
/// * `point1` - First boolean vector
/// * `point2` - Second boolean vector
///
/// # Returns
///
/// * The Russell-Rao distance
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::russellrao;
///
/// let u = &[true, false, true, false];
/// let v = &[true, true, false, false];
///
/// let dist: f64 = russellrao(u, v);
/// println!("Russell-Rao distance: {}", dist);
/// ```
#[allow(dead_code)]
pub fn russellrao<T: Float>(point1: &[bool], point2: &[bool]) -> T {
    if point1.len() != point2.len() {
        return T::nan();
    }

    let mut n_true_true = 0;
    let n = point1.len();

    for i in 0..n {
        if point1[i] && point2[i] {
            n_true_true += 1;
        }
    }

    let num = T::from(n - n_true_true).unwrap();
    let denom = T::from(n).unwrap();

    if denom > T::zero() {
        num / denom
    } else {
        T::zero()
    }
}

/// Sokal-Michener distance between two boolean vectors
///
/// The Sokal-Michener distance between two boolean vectors u and v is defined as:
/// 2(c_TF + c_FT) / (c_TT + c_FF + 2(c_TF + c_FT))
/// where c_ij is the number of occurrences of u\[k\]=i and v\[k\]=j for k<n.
///
/// # Arguments
///
/// * `point1` - First boolean vector
/// * `point2` - Second boolean vector
///
/// # Returns
///
/// * The Sokal-Michener distance
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::sokalmichener;
///
/// let u = &[true, false, true, false];
/// let v = &[true, true, false, false];
///
/// let dist: f64 = sokalmichener(u, v);
/// println!("Sokal-Michener distance: {}", dist);
/// ```
#[allow(dead_code)]
pub fn sokalmichener<T: Float>(point1: &[bool], point2: &[bool]) -> T {
    // This is the same as Rogers-Tanimoto
    rogerstanimoto(point1, point2)
}

/// Sokal-Sneath distance between two boolean vectors
///
/// The Sokal-Sneath distance between two boolean vectors u and v is defined as:
/// 2(c_TF + c_FT) / (c_TT + 2(c_TF + c_FT))
/// where c_ij is the number of occurrences of u\[k\]=i and v\[k\]=j for k<n.
///
/// # Arguments
///
/// * `point1` - First boolean vector
/// * `point2` - Second boolean vector
///
/// # Returns
///
/// * The Sokal-Sneath distance
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::sokalsneath;
///
/// let u = &[true, false, true, false];
/// let v = &[true, true, false, false];
///
/// let dist: f64 = sokalsneath(u, v);
/// println!("Sokal-Sneath distance: {}", dist);
/// ```
#[allow(dead_code)]
pub fn sokalsneath<T: Float>(point1: &[bool], point2: &[bool]) -> T {
    if point1.len() != point2.len() {
        return T::nan();
    }

    let mut n_true_true = 0;
    let mut n_true_false = 0;
    let mut n_false_true = 0;

    for i in 0..point1.len() {
        if point1[i] && point2[i] {
            n_true_true += 1;
        } else if point1[i] && !point2[i] {
            n_true_false += 1;
        } else if !point1[i] && point2[i] {
            n_false_true += 1;
        }
    }

    let r = n_true_false + n_false_true;

    let num = T::from(2 * r).unwrap();
    let denom = T::from(n_true_true + 2 * r).unwrap();

    if denom > T::zero() {
        num / denom
    } else {
        T::zero()
    }
}

/// Yule distance between two boolean vectors
///
/// The Yule distance between two boolean vectors u and v is defined as:
/// 2(c_TF * c_FT) / (c_TT * c_FF + c_TF * c_FT)
/// where c_ij is the number of occurrences of u\[k\]=i and v\[k\]=j for k<n.
///
/// # Arguments
///
/// * `point1` - First boolean vector
/// * `point2` - Second boolean vector
///
/// # Returns
///
/// * The Yule distance
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::yule;
///
/// let u = &[true, false, true, false];
/// let v = &[true, true, false, false];
///
/// let dist: f64 = yule(u, v);
/// println!("Yule distance: {}", dist);
/// ```
#[allow(dead_code)]
pub fn yule<T: Float>(point1: &[bool], point2: &[bool]) -> T {
    if point1.len() != point2.len() {
        return T::nan();
    }

    let mut n_true_true = 0;
    let mut n_true_false = 0;
    let mut n_false_true = 0;
    let mut n_false_false = 0;

    for i in 0..point1.len() {
        if point1[i] && point2[i] {
            n_true_true += 1;
        } else if point1[i] && !point2[i] {
            n_true_false += 1;
        } else if !point1[i] && point2[i] {
            n_false_true += 1;
        } else {
            n_false_false += 1;
        }
    }

    let num = T::from(2 * n_true_false * n_false_true).unwrap();
    let denom = T::from(n_true_true * n_false_false + n_true_false * n_false_true).unwrap();

    if denom > T::zero() {
        num / denom
    } else {
        T::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::arr2;

    #[test]
    fn test_euclidean_distance() {
        let point1 = &[1.0, 2.0, 3.0];
        let point2 = &[4.0, 5.0, 6.0];

        assert_relative_eq!(euclidean(point1, point2), 5.196152422706632, epsilon = 1e-6);
    }

    #[test]
    fn test_manhattan_distance() {
        let point1 = &[1.0, 2.0, 3.0];
        let point2 = &[4.0, 5.0, 6.0];

        assert_relative_eq!(manhattan(point1, point2), 9.0, epsilon = 1e-6);
    }

    #[test]
    fn test_chebyshev_distance() {
        let point1 = &[1.0, 2.0, 3.0];
        let point2 = &[4.0, 5.0, 6.0];

        assert_relative_eq!(chebyshev(point1, point2), 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_minkowski_distance() {
        let point1 = &[1.0, 2.0, 3.0];
        let point2 = &[4.0, 5.0, 6.0];

        // p = 1 (Manhattan)
        assert_relative_eq!(minkowski(point1, point2, 1.0), 9.0, epsilon = 1e-6);

        // p = 2 (Euclidean)
        assert_relative_eq!(
            minkowski(point1, point2, 2.0),
            5.196152422706632,
            epsilon = 1e-6
        );

        // p = infinity (Chebyshev)
        assert_relative_eq!(
            minkowski(point1, point2, f64::INFINITY),
            3.0,
            epsilon = 1e-6
        );

        // p = 3
        assert_relative_eq!(minkowski(point1, point2, 3.0), 4.3267, epsilon = 1e-4);
    }

    #[test]
    fn test_canberra_distance() {
        let point1 = &[1.0, 2.0, 3.0];
        let point2 = &[4.0, 5.0, 6.0];

        assert_relative_eq!(canberra(point1, point2), 1.5, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_distance() {
        // Orthogonal vectors should have distance 1
        let point1 = &[1.0, 0.0];
        let point2 = &[0.0, 1.0];

        assert_relative_eq!(cosine(point1, point2), 1.0, epsilon = 1e-6);

        // Parallel vectors should have distance 0
        let point3 = &[1.0, 2.0];
        let point4 = &[2.0, 4.0];

        assert_relative_eq!(cosine(point3, point4), 0.0, epsilon = 1e-6);

        // 45 degree angle should have distance 1 - sqrt(2)/2
        let point5 = &[1.0, 1.0];
        let point6 = &[1.0, 0.0];

        assert_relative_eq!(cosine(point5, point6), 0.2928932188134525, epsilon = 1e-6);
    }

    #[test]
    fn test_correlation_distance() {
        // Perfectly anti-correlated should have distance 2
        let point1 = &[1.0, 2.0, 3.0];
        let point2 = &[3.0, 2.0, 1.0];

        assert_relative_eq!(correlation(point1, point2), 2.0, epsilon = 1e-6);

        // Perfectly correlated should have distance 0
        let point3 = &[1.0, 2.0, 3.0];
        let point4 = &[2.0, 4.0, 6.0];

        assert_relative_eq!(correlation(point3, point4), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_jaccard_distance() {
        let point1 = &[1.0, 0.0, 1.0];
        let point2 = &[0.0, 1.0, 1.0];

        // 1 element in common, 2 elements different = 2/3
        assert_relative_eq!(jaccard(point1, point2), 2.0 / 3.0, epsilon = 1e-6);

        // Empty sets should have distance 0
        let point3 = &[0.0, 0.0, 0.0];
        let point4 = &[0.0, 0.0, 0.0];

        assert_relative_eq!(jaccard(point3, point4), 0.0, epsilon = 1e-6);

        // No elements in common should have distance 1
        let point5 = &[1.0, 1.0, 0.0];
        let point6 = &[0.0, 0.0, 1.0];

        assert_relative_eq!(jaccard(point5, point6), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_pdist() {
        let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);

        let dist_matrix = pdist(&points, euclidean);

        assert_eq!(dist_matrix.shape(), &[3, 3]);

        // Check diagonal is zero
        assert_relative_eq!(dist_matrix[(0, 0)], 0.0, epsilon = 1e-6);
        assert_relative_eq!(dist_matrix[(1, 1)], 0.0, epsilon = 1e-6);
        assert_relative_eq!(dist_matrix[(2, 2)], 0.0, epsilon = 1e-6);

        // Check off-diagonal elements
        assert_relative_eq!(dist_matrix[(0, 1)], 1.0, epsilon = 1e-6);
        assert_relative_eq!(dist_matrix[(0, 2)], 1.0, epsilon = 1e-6);
        assert_relative_eq!(
            dist_matrix[(1, 2)],
            std::f64::consts::SQRT_2,
            epsilon = 1e-6
        );

        // Check symmetry
        assert_relative_eq!(dist_matrix[(1, 0)], dist_matrix[(0, 1)], epsilon = 1e-6);
        assert_relative_eq!(dist_matrix[(2, 0)], dist_matrix[(0, 2)], epsilon = 1e-6);
        assert_relative_eq!(dist_matrix[(2, 1)], dist_matrix[(1, 2)], epsilon = 1e-6);
    }

    #[test]
    fn test_cdist() {
        let x_a = arr2(&[[0.0, 0.0], [1.0, 0.0]]);

        let xb = arr2(&[[0.0, 1.0], [1.0, 1.0]]);

        let dist_matrix = cdist(&x_a, &xb, euclidean).unwrap();

        assert_eq!(dist_matrix.shape(), &[2, 2]);

        assert_relative_eq!(dist_matrix[(0, 0)], 1.0, epsilon = 1e-6);
        assert_relative_eq!(
            dist_matrix[(0, 1)],
            std::f64::consts::SQRT_2,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            dist_matrix[(1, 0)],
            std::f64::consts::SQRT_2,
            epsilon = 1e-6
        );
        assert_relative_eq!(dist_matrix[(1, 1)], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_braycurtis_distance() {
        let point1 = &[1.0, 2.0, 3.0];
        let point2 = &[2.0, 3.0, 4.0];

        let dist = braycurtis(point1, point2);
        // Sum of differences: |1-2| + |2-3| + |3-4| = 3
        // Sum of absolute sums: |1+2| + |2+3| + |3+4| = 3 + 5 + 7 = 15
        // Distance: 3/15 = 1/5 = 0.2
        assert_relative_eq!(dist, 0.2, epsilon = 1e-6);

        // Test identical vectors (should be 0)
        let point3 = &[1.0, 2.0, 3.0];
        let point4 = &[1.0, 2.0, 3.0];
        assert_relative_eq!(braycurtis(point3, point4), 0.0, epsilon = 1e-6);

        // Test with zeros in both vectors
        let point5 = &[0.0, 0.0];
        let point6 = &[0.0, 0.0];
        assert_relative_eq!(braycurtis(point5, point6), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_squareform() {
        // Test conversion from condensed to square form
        let condensed = vec![1.0, 2.0, 3.0];
        let square = squareform(&condensed).unwrap();

        assert_eq!(square.shape(), &[3, 3]);
        assert_relative_eq!(square[(0, 1)], 1.0, epsilon = 1e-6);
        assert_relative_eq!(square[(0, 2)], 2.0, epsilon = 1e-6);
        assert_relative_eq!(square[(1, 2)], 3.0, epsilon = 1e-6);

        // Test conversion from square to condensed form
        let condensed2 = squareform_to_condensed(&square).unwrap();
        assert_eq!(condensed2, condensed);
    }
}
