//! Sparse matrix module
//!
//! This module provides implementations of various sparse matrix formats and operations,
//! similar to SciPy's `sparse` module.
//!
//! ## Overview
//!
//! * Sparse matrix formats (CSR, CSC, COO, DOK, LIL, DIA, BSR, etc.)
//! * Basic matrix operations (addition, multiplication, etc.)
//! * Sparse linear system solvers
//! * Sparse eigenvalue computation
//! * Conversion between different formats
//!
//! ## Examples
//!
//! ```
//! use scirs2_sparse::csr::CsrMatrix;
//!
//! // Create a sparse matrix in CSR format
//! let rows = vec![0, 0, 1, 2, 2];
//! let cols = vec![0, 2, 2, 0, 1];
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let shape = (3, 3);
//!
//! let matrix = CsrMatrix::new(data, rows, cols, shape).unwrap();
//! ```

// Export error types
pub mod error;
pub use error::{SparseError, SparseResult};

// Sparse matrix formats
pub mod csr;
pub use csr::CsrMatrix;

pub mod csc;
pub use csc::CscMatrix;

pub mod coo;
pub use coo::CooMatrix;

pub mod dok;
pub use dok::DokMatrix;

pub mod lil;
pub use lil::LilMatrix;

pub mod dia;
pub use dia::DiaMatrix;

pub mod bsr;
pub use bsr::BsrMatrix;

// Utility functions
pub mod utils;

// Linear algebra with sparse matrices
pub mod linalg;

// Format conversions
pub mod convert;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
