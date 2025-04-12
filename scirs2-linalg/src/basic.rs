//! Basic matrix operations

use crate::error::{LinalgError, LinalgResult};
use ndarray::{Array2, ArrayView2};
use num_traits::Float;

/// Compute the determinant of a square matrix.
///
/// # Arguments
///
/// * `a` - Input square matrix
///
/// # Returns
///
/// * Determinant of the matrix
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::det;
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
/// let d = det(&a.view()).unwrap();
/// assert!((d - (-2.0)).abs() < 1e-10);
/// ```
pub fn det<F: Float>(a: &ArrayView2<F>) -> LinalgResult<F> {
    if a.nrows() != a.ncols() {
        return Err(LinalgError::DimensionError(format!(
            "Matrix must be square to compute determinant, got shape {:?}",
            a.shape()
        )));
    }

    // Simple implementation for 2x2 and 3x3 matrices
    match a.nrows() {
        0 => Ok(F::one()),
        1 => Ok(a[[0, 0]]),
        2 => Ok(a[[0, 0]] * a[[1, 1]] - a[[0, 1]] * a[[1, 0]]),
        3 => {
            let det = a[[0, 0]] * (a[[1, 1]] * a[[2, 2]] - a[[1, 2]] * a[[2, 1]])
                - a[[0, 1]] * (a[[1, 0]] * a[[2, 2]] - a[[1, 2]] * a[[2, 0]])
                + a[[0, 2]] * (a[[1, 0]] * a[[2, 1]] - a[[1, 1]] * a[[2, 0]]);
            Ok(det)
        }
        _ => {
            // For larger matrices, we would implement LU decomposition with partial pivoting
            // This is a placeholder that will be replaced with a proper implementation
            Err(LinalgError::NotImplementedError(
                "Determinant for matrices larger than 3x3 not yet implemented".to_string(),
            ))
        }
    }
}

/// Compute the inverse of a square matrix.
///
/// # Arguments
///
/// * `a` - Input square matrix
///
/// # Returns
///
/// * Inverse of the matrix
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::inv;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 2.0]];
/// let a_inv = inv(&a.view()).unwrap();
/// assert!((a_inv[[0, 0]] - 1.0).abs() < 1e-10);
/// assert!((a_inv[[1, 1]] - 0.5).abs() < 1e-10);
/// ```
pub fn inv<F: Float>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>> {
    if a.nrows() != a.ncols() {
        return Err(LinalgError::DimensionError(format!(
            "Matrix must be square to compute inverse, got shape {:?}",
            a.shape()
        )));
    }

    // Simple implementation for 2x2 matrices
    if a.nrows() == 2 {
        let det_val = det(a)?;
        if det_val.abs() < F::epsilon() {
            return Err(LinalgError::SingularMatrixError(
                "Matrix is singular, cannot compute inverse".to_string(),
            ));
        }

        let inv_det = F::one() / det_val;
        let mut result = Array2::zeros((2, 2));
        result[[0, 0]] = a[[1, 1]] * inv_det;
        result[[0, 1]] = -a[[0, 1]] * inv_det;
        result[[1, 0]] = -a[[1, 0]] * inv_det;
        result[[1, 1]] = a[[0, 0]] * inv_det;
        return Ok(result);
    }

    // For larger matrices, we would implement more efficient algorithms
    // This is a placeholder that will be replaced with a proper implementation
    Err(LinalgError::NotImplementedError(
        "Matrix inverse for matrices larger than 2x2 not yet implemented".to_string(),
    ))
}

/// Raise a square matrix to the given power.
///
/// # Arguments
///
/// * `a` - Input square matrix
/// * `n` - Power (can be positive, negative, or zero)
///
/// # Returns
///
/// * Matrix raised to the power n
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::matrix_power;
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
///
/// // Identity matrix for n=0
/// let a_0 = matrix_power(&a.view(), 0).unwrap();
/// assert!((a_0[[0, 0]] - 1.0).abs() < 1e-10);
/// assert!((a_0[[0, 1]] - 0.0).abs() < 1e-10);
/// assert!((a_0[[1, 0]] - 0.0).abs() < 1e-10);
/// assert!((a_0[[1, 1]] - 1.0).abs() < 1e-10);
/// ```
pub fn matrix_power<F: Float>(a: &ArrayView2<F>, n: i32) -> LinalgResult<Array2<F>> {
    if a.nrows() != a.ncols() {
        return Err(LinalgError::DimensionError(format!(
            "Matrix must be square to compute power, got shape {:?}",
            a.shape()
        )));
    }

    let dim = a.nrows();

    // Handle special cases
    if n == 0 {
        // Return identity matrix
        let mut result = Array2::zeros((dim, dim));
        for i in 0..dim {
            result[[i, i]] = F::one();
        }
        return Ok(result);
    }

    if n == 1 {
        // Return copy of the matrix
        return Ok(a.to_owned());
    }

    if n == -1 {
        // Return inverse
        return inv(a);
    }

    if n.abs() > 1 {
        // For higher powers, we would implement more efficient algorithms
        // using matrix decompositions or binary exponentiation
        // This is a placeholder that will be replaced with a proper implementation
        return Err(LinalgError::NotImplementedError(
            "Matrix power for |n| > 1 not yet implemented".to_string(),
        ));
    }

    // This should never be reached
    Err(LinalgError::ComputationError(
        "Unexpected error in matrix power calculation".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_det_2x2() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let d = det(&a.view()).unwrap();
        assert!((d - (-2.0)).abs() < 1e-10);

        let b = array![[2.0, 0.0], [0.0, 3.0]];
        let d = det(&b.view()).unwrap();
        assert!((d - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_det_3x3() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let d = det(&a.view()).unwrap();
        assert!((d - 0.0).abs() < 1e-10);

        let b = array![[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]];
        let d = det(&b.view()).unwrap();
        assert!((d - 24.0).abs() < 1e-10);
    }

    #[test]
    fn test_inv_2x2() {
        let a = array![[1.0, 0.0], [0.0, 2.0]];
        let a_inv = inv(&a.view()).unwrap();
        assert_relative_eq!(a_inv[[0, 0]], 1.0);
        assert_relative_eq!(a_inv[[0, 1]], 0.0);
        assert_relative_eq!(a_inv[[1, 0]], 0.0);
        assert_relative_eq!(a_inv[[1, 1]], 0.5);

        let b = array![[1.0, 2.0], [3.0, 4.0]];
        let b_inv = inv(&b.view()).unwrap();
        assert_relative_eq!(b_inv[[0, 0]], -2.0);
        assert_relative_eq!(b_inv[[0, 1]], 1.0);
        assert_relative_eq!(b_inv[[1, 0]], 1.5);
        assert_relative_eq!(b_inv[[1, 1]], -0.5);
    }

    #[test]
    fn test_matrix_power() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];

        // Power 0 should give identity matrix
        let a_0 = matrix_power(&a.view(), 0).unwrap();
        assert_relative_eq!(a_0[[0, 0]], 1.0);
        assert_relative_eq!(a_0[[0, 1]], 0.0);
        assert_relative_eq!(a_0[[1, 0]], 0.0);
        assert_relative_eq!(a_0[[1, 1]], 1.0);

        // Power 1 should return the original matrix
        let a_1 = matrix_power(&a.view(), 1).unwrap();
        assert_relative_eq!(a_1[[0, 0]], a[[0, 0]]);
        assert_relative_eq!(a_1[[0, 1]], a[[0, 1]]);
        assert_relative_eq!(a_1[[1, 0]], a[[1, 0]]);
        assert_relative_eq!(a_1[[1, 1]], a[[1, 1]]);
    }
}
