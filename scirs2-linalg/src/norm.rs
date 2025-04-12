//! Matrix and vector norms

use ndarray::{ArrayView1, ArrayView2};
use num_traits::{Float, NumAssign};
use std::iter::Sum;

use crate::error::{LinalgError, LinalgResult};

/// Compute a matrix norm.
///
/// # Arguments
///
/// * `a` - Input matrix
/// * `ord` - Order of the norm:
///   * 'fro' or 'f': Frobenius norm
///   * '1': 1-norm (maximum column sum)
///   * 'inf': Infinity norm (maximum row sum)
///   * '2': 2-norm (largest singular value)
///
/// # Returns
///
/// * Value of the norm
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::matrix_norm;
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
/// let norm_fro = matrix_norm(&a.view(), "fro").unwrap();
/// assert!((norm_fro - 5.477225575051661).abs() < 1e-10);
/// ```
pub fn matrix_norm<F>(a: &ArrayView2<F>, ord: &str) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum,
{
    match ord {
        "fro" | "f" => {
            // Frobenius norm
            let mut sum_sq = F::zero();
            for i in 0..a.nrows() {
                for j in 0..a.ncols() {
                    sum_sq += a[[i, j]] * a[[i, j]];
                }
            }
            Ok(sum_sq.sqrt())
        }
        "1" => {
            // 1-norm (maximum column sum)
            let mut max_col_sum = F::zero();
            for j in 0..a.ncols() {
                let col = a.column(j);
                let col_sum = col.fold(F::zero(), |acc, &x| acc + x.abs());
                if col_sum > max_col_sum {
                    max_col_sum = col_sum;
                }
            }
            Ok(max_col_sum)
        }
        "inf" => {
            // Infinity norm (maximum row sum)
            let mut max_row_sum = F::zero();
            for i in 0..a.nrows() {
                let row = a.row(i);
                let row_sum = row.fold(F::zero(), |acc, &x| acc + x.abs());
                if row_sum > max_row_sum {
                    max_row_sum = row_sum;
                }
            }
            Ok(max_row_sum)
        }
        "2" => {
            // 2-norm (largest singular value)
            // This would be computed using SVD
            Err(LinalgError::ImplementationError(
                "2-norm not yet implemented".to_string(),
            ))
        }
        _ => Err(LinalgError::ShapeError(format!(
            "Invalid norm order: {}, must be one of 'fro', 'f', '1', 'inf', '2'",
            ord
        ))),
    }
}

/// Compute a vector norm.
///
/// # Arguments
///
/// * `x` - Input vector
/// * `ord` - Order of the norm:
///   * 1: 1-norm (sum of absolute values)
///   * 2: 2-norm (Euclidean norm)
///   * usize::MAX: Infinity norm (maximum absolute value)
///
/// # Returns
///
/// * Value of the norm
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::vector_norm;
///
/// let x = array![3.0_f64, 4.0];
/// let norm_2 = vector_norm(&x.view(), 2).unwrap();
/// assert!((norm_2 - 5.0).abs() < 1e-10);
/// ```
pub fn vector_norm<F>(x: &ArrayView1<F>, ord: usize) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum,
{
    match ord {
        1 => {
            // 1-norm (sum of absolute values)
            let sum_abs = x.fold(F::zero(), |acc, &val| acc + val.abs());
            Ok(sum_abs)
        }
        2 => {
            // 2-norm (Euclidean norm)
            let sum_sq = x.fold(F::zero(), |acc, &val| acc + val * val);
            Ok(sum_sq.sqrt())
        }
        usize::MAX => {
            // Infinity norm (maximum absolute value)
            let max_abs = x.fold(F::zero(), |acc, &val| {
                let abs_val = val.abs();
                if abs_val > acc {
                    abs_val
                } else {
                    acc
                }
            });
            Ok(max_abs)
        }
        _ => Err(LinalgError::ShapeError(format!(
            "Invalid norm order: {}, must be one of 1, 2, or inf",
            ord
        ))),
    }
}

/// Compute the condition number of a matrix.
///
/// # Arguments
///
/// * `a` - Input matrix
/// * `p` - Order of the norm:
///   * None: 2-norm
///   * "fro" or "f": Frobenius norm
///   * "1": 1-norm
///   * "inf": Infinity norm
///
/// # Returns
///
/// * Condition number of the matrix
///
/// # Examples
///
/// ```no_run
/// use ndarray::array;
/// use scirs2_linalg::cond;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 2.0]];
/// let c = cond(&a.view(), None).unwrap();
/// assert!((c - 2.0).abs() < 1e-10);
/// ```
pub fn cond<F>(_a: &ArrayView2<F>, _p: Option<&str>) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum,
{
    // This would compute the condition number using SVD or other methods
    // For now, it's a placeholder
    Err(LinalgError::ImplementationError(
        "Condition number computation not yet implemented".to_string(),
    ))
}

/// Compute the rank of a matrix.
///
/// # Arguments
///
/// * `a` - Input matrix
/// * `tol` - Tolerance for singular values
///
/// # Returns
///
/// * Rank of the matrix
///
/// # Examples
///
/// ```no_run
/// use ndarray::array;
/// use scirs2_linalg::matrix_rank;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
/// let r = matrix_rank(&a.view(), None).unwrap();
/// assert_eq!(r, 2);
/// ```
pub fn matrix_rank<F>(_a: &ArrayView2<F>, _tol: Option<F>) -> LinalgResult<usize>
where
    F: Float + NumAssign + Sum,
{
    // This would compute the rank using SVD and counting singular values above tolerance
    // For now, it's a placeholder
    Err(LinalgError::ImplementationError(
        "Matrix rank computation not yet implemented".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_matrix_norm_frobenius() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let norm = matrix_norm(&a.view(), "fro").unwrap();
        // sqrt(1^2 + 2^2 + 3^2 + 4^2) = sqrt(30) ≈ 5.477
        assert_relative_eq!(norm, 5.477225575051661, epsilon = 1e-10);
    }

    #[test]
    fn test_matrix_norm_1() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let norm = matrix_norm(&a.view(), "1").unwrap();
        // max(1+3, 2+4) = max(4, 6) = 6
        assert_relative_eq!(norm, 6.0);
    }

    #[test]
    fn test_matrix_norm_inf() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let norm = matrix_norm(&a.view(), "inf").unwrap();
        // max(1+2, 3+4) = max(3, 7) = 7
        assert_relative_eq!(norm, 7.0);
    }

    #[test]
    fn test_vector_norm_1() {
        let x = array![1.0, -2.0, 3.0];
        let norm = vector_norm(&x.view(), 1).unwrap();
        // |1| + |-2| + |3| = 6
        assert_relative_eq!(norm, 6.0);
    }

    #[test]
    fn test_vector_norm_2() {
        let x = array![3.0, 4.0];
        let norm = vector_norm(&x.view(), 2).unwrap();
        // sqrt(3^2 + 4^2) = 5
        assert_relative_eq!(norm, 5.0);
    }

    #[test]
    fn test_vector_norm_inf() {
        let x = array![1.0, -5.0, 3.0];
        let norm = vector_norm(&x.view(), usize::MAX).unwrap();
        // max(|1|, |-5|, |3|) = 5
        assert_relative_eq!(norm, 5.0);
    }
}
