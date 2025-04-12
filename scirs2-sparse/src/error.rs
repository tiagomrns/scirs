//! Error types for the SciRS2 sparse matrix module

use thiserror::Error;

/// Sparse matrix error type
#[derive(Error, Debug)]
pub enum SparseError {
    /// Computation error (generic error)
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// Dimension mismatch error
    #[error("Dimension mismatch error: {0}")]
    DimensionError(String),

    /// Singular matrix error
    #[error("Singular matrix error: {0}")]
    SingularMatrixError(String),

    /// Value error (invalid value)
    #[error("Value error: {0}")]
    ValueError(String),

    /// Not implemented error
    #[error("Not implemented: {0}")]
    NotImplementedError(String),
}

/// Result type for sparse matrix operations
pub type SparseResult<T> = Result<T, SparseError>;
