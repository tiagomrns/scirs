//! Eigenvalue solvers for symmetric matrices
//!
//! This module provides specialized solvers for symmetric matrices,
//! which are more efficient than general eigenvalue solvers.

use ndarray::{Array1, Array2, ArrayView2};
use num_traits::{Float, NumAssign};
use std::iter::Sum;

use crate::error::{LinalgError, LinalgResult};

/// Compute eigenvalues and eigenvectors of a symmetric matrix.
///
/// This algorithm transforms the symmetric matrix to tridiagonal form
/// and then applies the tridiagonal eigensolver.
///
/// # Arguments
///
/// * `a` - Input symmetric matrix
///
/// # Returns
///
/// * Tuple containing eigenvalues and eigenvectors
pub fn symmetric_eigh<F>(a: &ArrayView2<F>) -> LinalgResult<(Array1<F>, Array2<F>)>
where
    F: Float + NumAssign + Sum + 'static,
{
    let n = a.nrows();

    if a.ncols() != n {
        return Err(LinalgError::ShapeError(format!(
            "Expected square matrix, got shape {:?}",
            a.shape()
        )));
    }

    // Check if the matrix is symmetric
    for i in 0..n {
        for j in i + 1..n {
            let diff = (a[[i, j]] - a[[j, i]]).abs();
            if diff > F::epsilon() * F::from(10.0).unwrap() {
                return Err(LinalgError::ShapeError(
                    "Matrix must be symmetric for symmetric_eigh function".to_string(),
                ));
            }
        }
    }

    // Convert to tridiagonal form using Householder reflections
    let (diagonal, off_diagonal) = tridiagonalize(a)?;

    // Use tridiagonal solver
    crate::eigen_specialized::tridiagonal::tridiagonal_eigh(&diagonal.view(), &off_diagonal.view())
}

/// Compute just the eigenvalues of a symmetric matrix.
///
/// This is similar to `symmetric_eigh` but only returns the eigenvalues.
///
/// # Arguments
///
/// * `a` - Input symmetric matrix
///
/// # Returns
///
/// * Vector of eigenvalues
pub fn symmetric_eigvalsh<F>(a: &ArrayView2<F>) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Sum + 'static,
{
    let n = a.nrows();

    if a.ncols() != n {
        return Err(LinalgError::ShapeError(format!(
            "Expected square matrix, got shape {:?}",
            a.shape()
        )));
    }

    // Check if the matrix is symmetric
    for i in 0..n {
        for j in i + 1..n {
            let diff = (a[[i, j]] - a[[j, i]]).abs();
            if diff > F::epsilon() * F::from(10.0).unwrap() {
                return Err(LinalgError::ShapeError(
                    "Matrix must be symmetric for symmetric_eigvalsh function".to_string(),
                ));
            }
        }
    }

    // Convert to tridiagonal form using Householder reflections
    let (diagonal, off_diagonal) = tridiagonalize(a)?;

    // Use tridiagonal solver
    crate::eigen_specialized::tridiagonal::tridiagonal_eigvalsh(
        &diagonal.view(),
        &off_diagonal.view(),
    )
}

/// Tridiagonalize a symmetric matrix using Householder reflections
///
/// # Arguments
///
/// * `a` - Input symmetric matrix
///
/// # Returns
///
/// * Diagonal and off-diagonal elements of the tridiagonal matrix
fn tridiagonalize<F>(a: &ArrayView2<F>) -> LinalgResult<(Array1<F>, Array1<F>)>
where
    F: Float + NumAssign + Sum + 'static,
{
    let n = a.nrows();
    let mut working_matrix = a.to_owned();
    let mut diagonal = Array1::zeros(n);
    let mut off_diagonal = Array1::zeros(n - 1);

    // Apply Householder transformations to reduce to tridiagonal form
    for i in 0..n - 2 {
        let mut alpha = F::zero();

        // Compute norm of the subdiagonal column
        for j in i + 1..n {
            alpha += working_matrix[[j, i]] * working_matrix[[j, i]];
        }
        alpha = alpha.sqrt();

        // Set the diagonal element
        diagonal[i] = working_matrix[[i, i]];

        if alpha < F::epsilon() {
            // Subdiagonal is already zero, no need for Householder reflection
            off_diagonal[i] = F::zero();
            continue;
        }

        // Choose sign to avoid cancellation
        let sgn = if working_matrix[[i + 1, i]] < F::zero() {
            F::one()
        } else {
            -F::one()
        };
        let alpha = -sgn * alpha;
        off_diagonal[i] = alpha;

        // Householder vector
        let mut v = Array1::zeros(n);
        v[i + 1] = working_matrix[[i + 1, i]] - alpha;
        for j in i + 2..n {
            v[j] = working_matrix[[j, i]];
        }

        // Normalize the Householder vector
        let vnorm = v.iter().map(|&x| x * x).sum::<F>().sqrt();
        if vnorm > F::epsilon() {
            for j in i + 1..n {
                v[j] /= vnorm;
            }
        }

        // Apply Householder reflection H = I - 2*v*v' to working_matrix
        // Formula: A' = H*A*H

        // Compute w = A*v
        let mut w = Array1::zeros(n);
        for j in 0..n {
            for k in i + 1..n {
                w[j] += working_matrix[[j, k]] * v[k];
            }
        }

        // Compute z = v'*A*v
        let mut z = F::zero();
        for j in i + 1..n {
            z += v[j] * w[j];
        }

        // Update the matrix: A' = A - 2*v*w' - 2*w*v' + 4*z*v*v'
        for j in 0..n {
            for k in j..n {
                working_matrix[[j, k]] = working_matrix[[j, k]]
                    - F::from(2.0).unwrap() * (v[j] * w[k] + w[j] * v[k])
                    + F::from(4.0).unwrap() * z * v[j] * v[k];
                working_matrix[[k, j]] = working_matrix[[j, k]]; // Maintain symmetry
            }
        }
    }

    // Set the last two diagonal elements
    match n.cmp(&1) {
        std::cmp::Ordering::Greater => {
            diagonal[n - 2] = working_matrix[[n - 2, n - 2]];
            diagonal[n - 1] = working_matrix[[n - 1, n - 1]];
            off_diagonal[n - 2] = working_matrix[[n - 1, n - 2]];
        }
        std::cmp::Ordering::Equal => {
            diagonal[0] = working_matrix[[0, 0]];
        }
        std::cmp::Ordering::Less => {
            // No action needed for n = 0
        }
    }

    Ok((diagonal, off_diagonal))
}
