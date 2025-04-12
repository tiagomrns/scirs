//! Error types for the SciRS2 optimization module

use thiserror::Error;

/// Optimization error type
#[derive(Error, Debug)]
pub enum OptimizeError {
    /// Computation error (generic error)
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// Convergence error (algorithm did not converge)
    #[error("Convergence error: {0}")]
    ConvergenceError(String),

    /// Value error (invalid value)
    #[error("Value error: {0}")]
    ValueError(String),

    /// Not implemented error
    #[error("Not implemented: {0}")]
    NotImplementedError(String),
}

/// Result type for optimization operations
pub type OptimizeResult<T> = Result<T, OptimizeError>;
