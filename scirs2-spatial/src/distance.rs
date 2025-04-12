//! Distance metrics for spatial data
//!
//! This module provides various distance metrics for spatial data,
//! such as Euclidean, Manhattan, Chebyshev, etc.

use crate::error::{SpatialError, SpatialResult};
use ndarray::{Array1, Array2};
use num_traits::Float;

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
pub fn euclidean<T: Float>(point1: &[T], point2: &[T]) -> T {
    if point1.len() != point2.len() {
        return T::nan();
    }

    let mut sum = T::zero();
    for i in 0..point1.len() {
        let diff = point1[i] - point2[i];
        sum = sum + diff * diff;
    }
    sum.sqrt()
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
pub fn manhattan<T: Float>(point1: &[T], point2: &[T]) -> T {
    if point1.len() != point2.len() {
        return T::nan();
    }

    let mut sum = T::zero();
    for i in 0..point1.len() {
        sum = sum + (point1[i] - point2[i]).abs();
    }
    sum
}

/// Compute Chebyshev (maximum) distance between two points
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
pub fn chebyshev<T: Float>(point1: &[T], point2: &[T]) -> T {
    if point1.len() != point2.len() {
        return T::nan();
    }

    let mut max_dist = T::zero();
    for i in 0..point1.len() {
        let diff = (point1[i] - point2[i]).abs();
        if diff > max_dist {
            max_dist = diff;
        }
    }
    max_dist
}

/// Compute Minkowski distance between two points
///
/// # Arguments
///
/// * `point1` - First point
/// * `point2` - Second point
/// * `p` - Order of the Minkowski metric
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
/// // Minkowski with p=2 is equivalent to Euclidean
/// let dist = minkowski(point1, point2, 2.0);
/// assert!((dist - 5.196152f64).abs() < 1e-6);
///
/// // Minkowski with p=1 is equivalent to Manhattan
/// let dist = minkowski(point1, point2, 1.0);
/// assert!((dist - 9.0f64).abs() < 1e-6);
///
/// // Minkowski with p=inf is equivalent to Chebyshev
/// let dist = minkowski(point1, point2, f64::INFINITY);
/// assert!((dist - 3.0f64).abs() < 1e-6);
/// ```
pub fn minkowski<T: Float>(point1: &[T], point2: &[T], p: T) -> T {
    if point1.len() != point2.len() {
        return T::nan();
    }

    if p == T::one() {
        return manhattan(point1, point2);
    } else if p == T::infinity() {
        return chebyshev(point1, point2);
    } else if p == T::from(2.0).unwrap() {
        return euclidean(point1, point2);
    }

    let mut sum = T::zero();
    for i in 0..point1.len() {
        sum = sum + (point1[i] - point2[i]).abs().powf(p);
    }
    sum.powf(T::one() / p)
}

/// Compute Hamming distance between two points (proportion of coordinates that differ)
///
/// # Arguments
///
/// * `point1` - First point
/// * `point2` - Second point
///
/// # Returns
///
/// * Hamming distance between the points
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::hamming;
///
/// let point1 = &[1.0, 2.0, 3.0];
/// let point2 = &[1.0, 5.0, 3.0];
///
/// let dist = hamming(point1, point2);
/// assert!((dist - 0.3333333f64).abs() < 1e-6);
/// ```
pub fn hamming<T: Float>(point1: &[T], point2: &[T]) -> T {
    if point1.len() != point2.len() {
        return T::nan();
    }

    let mut count = T::zero();
    for i in 0..point1.len() {
        if point1[i] != point2[i] {
            count = count + T::one();
        }
    }
    count / T::from(point1.len()).unwrap()
}

/// Compute Euclidean distance between each pair of points in two collections
///
/// # Arguments
///
/// * `xa` - First collection of points
/// * `xb` - Second collection of points
///
/// # Returns
///
/// * 2D array of distances between each pair of points
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::cdist;
/// use ndarray::array;
///
/// let points1 = array![[1.0, 2.0], [3.0, 4.0]];
/// let points2 = array![[5.0, 6.0], [7.0, 8.0]];
///
/// let distances = cdist(&points1, &points2).unwrap();
/// // distances[0, 0] is distance between [1,2] and [5,6]
/// // distances[0, 1] is distance between [1,2] and [7,8]
/// // distances[1, 0] is distance between [3,4] and [5,6]
/// // distances[1, 1] is distance between [3,4] and [7,8]
/// ```
pub fn cdist(xa: &Array2<f64>, xb: &Array2<f64>) -> SpatialResult<Array2<f64>> {
    if xa.ncols() != xb.ncols() {
        return Err(SpatialError::DimensionError(
            "Points in both collections must have the same dimensionality".to_string(),
        ));
    }

    let na = xa.nrows();
    let nb = xb.nrows();
    let mut result = Array2::zeros((na, nb));

    for i in 0..na {
        for j in 0..nb {
            let a_row = xa.row(i).to_vec();
            let b_row = xb.row(j).to_vec();
            result[[i, j]] = euclidean(&a_row, &b_row);
        }
    }

    Ok(result)
}

/// Compute the distance matrix for a collection of points
///
/// # Arguments
///
/// * `x` - Collection of points
///
/// # Returns
///
/// * 2D array of pairwise distances between points
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::pdist;
/// use ndarray::array;
///
/// let points = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
///
/// let distances = pdist(&points).unwrap();
/// // Number of pairwise distances should be n*(n-1)/2 = 3
/// assert_eq!(distances.len(), 3);
/// ```
pub fn pdist(x: &Array2<f64>) -> SpatialResult<Array1<f64>> {
    let n = x.nrows();
    let num_dists = n * (n - 1) / 2;
    let mut result = Array1::zeros(num_dists);

    let mut idx = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let a_row = x.row(i).to_vec();
            let b_row = x.row(j).to_vec();
            result[idx] = euclidean(&a_row, &b_row);
            idx += 1;
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::arr2;

    #[test]
    fn test_euclidean() {
        let point1 = [1.0, 2.0, 3.0];
        let point2 = [4.0, 5.0, 6.0];
        let dist = euclidean(&point1, &point2);
        assert_relative_eq!(dist, 5.196152, epsilon = 1e-6);
    }

    #[test]
    fn test_sqeuclidean() {
        let point1 = [1.0, 2.0, 3.0];
        let point2 = [4.0, 5.0, 6.0];
        let dist = sqeuclidean(&point1, &point2);
        assert_relative_eq!(dist, 27.0, epsilon = 1e-6);
    }

    #[test]
    fn test_manhattan() {
        let point1 = [1.0, 2.0, 3.0];
        let point2 = [4.0, 5.0, 6.0];
        let dist = manhattan(&point1, &point2);
        assert_relative_eq!(dist, 9.0, epsilon = 1e-6);
    }

    #[test]
    fn test_chebyshev() {
        let point1 = [1.0, 2.0, 3.0];
        let point2 = [4.0, 5.0, 6.0];
        let dist = chebyshev(&point1, &point2);
        assert_relative_eq!(dist, 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_minkowski() {
        let point1 = [1.0, 2.0, 3.0];
        let point2 = [4.0, 5.0, 6.0];

        // p=1 (Manhattan)
        let dist = minkowski(&point1, &point2, 1.0);
        assert_relative_eq!(dist, 9.0, epsilon = 1e-6);

        // p=2 (Euclidean)
        let dist = minkowski(&point1, &point2, 2.0);
        assert_relative_eq!(dist, 5.196152, epsilon = 1e-6);

        // p=inf (Chebyshev)
        let dist = minkowski(&point1, &point2, f64::INFINITY);
        assert_relative_eq!(dist, 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_hamming() {
        let point1 = [1.0, 2.0, 3.0];
        let point2 = [1.0, 5.0, 3.0];
        let dist = hamming(&point1, &point2);
        assert_relative_eq!(dist, 1.0 / 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cdist() {
        let points1 = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let points2 = arr2(&[[5.0, 6.0], [7.0, 8.0]]);

        let distances = cdist(&points1, &points2).unwrap();
        assert_eq!(distances.shape(), &[2, 2]);

        assert_relative_eq!(distances[[0, 0]], 5.656854, epsilon = 1e-6); // dist([1,2], [5,6])
        assert_relative_eq!(distances[[0, 1]], 8.485281, epsilon = 1e-6); // dist([1,2], [7,8])
        assert_relative_eq!(distances[[1, 0]], 2.828427, epsilon = 1e-6); // dist([3,4], [5,6])
        assert_relative_eq!(distances[[1, 1]], 5.656854, epsilon = 1e-6); // dist([3,4], [7,8])
    }

    #[test]
    fn test_pdist() {
        let points = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);

        let distances = pdist(&points).unwrap();
        assert_eq!(distances.len(), 3);

        assert_relative_eq!(distances[0], 2.828427, epsilon = 1e-6); // dist([1,2], [3,4])
        assert_relative_eq!(distances[1], 5.656854, epsilon = 1e-6); // dist([1,2], [5,6])
        assert_relative_eq!(distances[2], 2.828427, epsilon = 1e-6); // dist([3,4], [5,6])
    }
}
