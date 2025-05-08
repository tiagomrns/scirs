//! Extended precision matrix operations
//!
//! This module provides functions for performing matrix operations with extended precision.
//!
//! ## Overview
//!
//! This module provides utilities for higher precision matrix computations:
//!
//! * Basic operations with extended precision (matrix-vector, matrix-matrix multiply)
//! * Linear system solvers with extended precision
//! * Eigenvalue computations with extended precision
//! * Matrix factorizations (LU, QR, Cholesky, SVD) with extended precision
//!
//! Extended precision operations are useful for computations involving ill-conditioned matrices,
//! or when working with matrices that have entries with widely different magnitudes.
//!
//! ## Examples
//!
//! Basic operations:
//!
//! ```ignore
//! use ndarray::{array, ArrayView2, ArrayView1};
//! use scirs2_linalg::extended_precision::{extended_matmul, extended_matvec};
//!
//! // Create a matrix and vectors in f32 precision
//! let a = array![[1.0_f32, 2.0_f32], [3.0_f32, 4.0_f32]];
//! let x = array![0.1_f32, 0.2_f32];
//!
//! // Compute matrix-vector product with higher precision
//! let y = extended_matvec::<_, f64>(&a.view(), &x.view()).unwrap();
//!
//! // Compute matrix-matrix product with higher precision
//! let b = array![[5.0_f32, 6.0_f32], [7.0_f32, 8.0_f32]];
//! let c = extended_matmul::<_, f64>(&a.view(), &b.view()).unwrap();
//! ```
//!
//! Matrix factorizations with extended precision:
//!
//! ```ignore
//! use ndarray::{array, ArrayView2};
//! use scirs2_linalg::extended_precision::factorizations::{extended_lu, extended_qr, extended_cholesky};
//!
//! let a = array![
//!     [2.0_f32, 1.0_f32, 1.0_f32],
//!     [4.0_f32, 3.0_f32, 3.0_f32],
//!     [8.0_f32, 7.0_f32, 9.0_f32]
//! ];
//!
//! // LU decomposition with extended precision
//! let (p, l, u) = extended_lu::<_, f64>(&a.view()).unwrap();
//!
//! // QR decomposition with extended precision
//! let (q, r) = extended_qr::<_, f64>(&a.view()).unwrap();
//!
//! // Cholesky decomposition with extended precision
//! let spd_matrix = array![
//!     [4.0_f32, 1.0_f32, 1.0_f32],
//!     [1.0_f32, 5.0_f32, 2.0_f32],
//!     [1.0_f32, 2.0_f32, 6.0_f32]
//! ];
//! let l = extended_cholesky::<_, f64>(&spd_matrix.view()).unwrap();
//! ```

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, One, Zero};

use crate::error::LinalgResult;

// Extended precision eigenvalue computations - temporarily disabled
// pub mod eigen;

// Extended precision matrix factorizations - temporarily disabled
// pub mod factorizations;

/// Trait for promoting numerical types to higher precision
pub trait PromotableTo<T> {
    /// Convert to a higher precision type
    fn promote(self) -> T;
}

// Implement promotions for common types
impl PromotableTo<f64> for f32 {
    fn promote(self) -> f64 {
        self as f64
    }
}

impl PromotableTo<f64> for f64 {
    fn promote(self) -> f64 {
        self // No promotion needed
    }
}

/// Trait for demoting numerical types to lower precision
pub trait DemotableTo<T> {
    /// Convert to a lower precision type
    fn demote(self) -> T;
}

impl DemotableTo<f32> for f64 {
    fn demote(self) -> f32 {
        self as f32
    }
}

impl DemotableTo<f32> for f32 {
    fn demote(self) -> f32 {
        self // No demotion needed
    }
}

/// Matrix-vector multiplication using higher precision arithmetic
///
/// Computes y = A*x using higher precision intermediate calculations
pub fn extended_matvec<A, I>(a: &ArrayView2<A>, x: &ArrayView1<A>) -> LinalgResult<Array1<A>>
where
    A: Float + Zero + PromotableTo<I> + Copy,
    I: Float + Zero + DemotableTo<A> + Copy,
{
    if a.ncols() != x.len() {
        return Err(crate::error::LinalgError::ShapeError(format!(
            "Incompatible dimensions for matrix-vector multiplication: {:?} and {:?}",
            a.shape(),
            x.shape()
        )));
    }

    let m = a.nrows();
    let n = a.ncols();

    let mut y = Array1::zeros(m);

    for i in 0..m {
        // Compute dot product in higher precision
        let mut sum = I::zero();
        for j in 0..n {
            let a_high = a[[i, j]].promote();
            let x_high = x[j].promote();
            sum = sum + a_high * x_high;
        }

        // Convert back to original precision
        y[i] = sum.demote();
    }

    Ok(y)
}

/// Matrix-matrix multiplication using higher precision arithmetic
///
/// Computes C = A*B using higher precision intermediate calculations
pub fn extended_matmul<A, I>(a: &ArrayView2<A>, b: &ArrayView2<A>) -> LinalgResult<Array2<A>>
where
    A: Float + Zero + PromotableTo<I> + Copy,
    I: Float + Zero + DemotableTo<A> + Copy,
{
    if a.ncols() != b.nrows() {
        return Err(crate::error::LinalgError::ShapeError(format!(
            "Incompatible dimensions for matrix multiplication: {:?} and {:?}",
            a.shape(),
            b.shape()
        )));
    }

    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();

    let mut c = Array2::zeros((m, n));

    for i in 0..m {
        for j in 0..n {
            // Compute dot product in higher precision
            let mut sum = I::zero();
            for l in 0..k {
                let a_high = a[[i, l]].promote();
                let b_high = b[[l, j]].promote();
                sum = sum + a_high * b_high;
            }

            // Convert back to original precision
            c[[i, j]] = sum.demote();
        }
    }

    Ok(c)
}

/// Solve a linear system Ax = b using extended precision
///
/// This implementation uses Gaussian elimination with partial pivoting
/// and higher precision intermediate calculations for better accuracy.
pub fn extended_solve<A, I>(a: &ArrayView2<A>, b: &ArrayView1<A>) -> LinalgResult<Array1<A>>
where
    A: Float + Zero + One + PromotableTo<I> + Copy,
    I: Float + Zero + One + DemotableTo<A> + Copy + PartialOrd,
{
    if a.nrows() != a.ncols() {
        return Err(crate::error::LinalgError::ShapeError(format!(
            "Expected square matrix, got shape {:?}",
            a.shape()
        )));
    }

    if a.nrows() != b.len() {
        return Err(crate::error::LinalgError::ShapeError(format!(
            "Incompatible dimensions: A is {:?}, b is {:?}",
            a.shape(),
            b.shape()
        )));
    }

    let n = a.nrows();

    // Convert all inputs to higher precision
    let mut a_high = Array2::zeros((n, n));
    let mut b_high = Array1::zeros(n);

    for i in 0..n {
        b_high[i] = b[i].promote();
        for j in 0..n {
            a_high[[i, j]] = a[[i, j]].promote();
        }
    }

    // Gaussian elimination with partial pivoting in higher precision
    for k in 0..n - 1 {
        // Find pivot
        let mut pivot_row = k;
        let mut max_val = a_high[[k, k]].abs();

        for i in k + 1..n {
            let val = a_high[[i, k]].abs();
            if val > max_val {
                max_val = val;
                pivot_row = i;
            }
        }

        // Check for singular matrix
        if max_val < I::epsilon() {
            return Err(crate::error::LinalgError::SingularMatrixError(
                "Matrix is singular or nearly singular".to_string(),
            ));
        }

        // Swap rows if necessary
        if pivot_row != k {
            for j in k..n {
                let temp = a_high[[k, j]];
                a_high[[k, j]] = a_high[[pivot_row, j]];
                a_high[[pivot_row, j]] = temp;
            }

            let temp = b_high[k];
            b_high[k] = b_high[pivot_row];
            b_high[pivot_row] = temp;
        }

        // Elimination
        for i in k + 1..n {
            let factor = a_high[[i, k]] / a_high[[k, k]];

            for j in k + 1..n {
                a_high[[i, j]] = a_high[[i, j]] - factor * a_high[[k, j]];
            }

            b_high[i] = b_high[i] - factor * b_high[k];
            a_high[[i, k]] = I::zero(); // Not necessary but makes structure clear
        }
    }

    // Back substitution in higher precision
    let mut x_high = Array1::zeros(n);

    for i in (0..n).rev() {
        let mut sum = I::zero();
        for j in i + 1..n {
            sum = sum + a_high[[i, j]] * x_high[j];
        }

        x_high[i] = (b_high[i] - sum) / a_high[[i, i]];
    }

    // Convert back to original precision
    let mut x = Array1::zeros(n);
    for i in 0..n {
        x[i] = x_high[i].demote();
    }

    Ok(x)
}
