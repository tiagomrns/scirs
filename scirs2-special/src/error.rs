//! Error types for the SciRS2 special functions module

use scirs2_core::error::CoreError;
use thiserror::Error;

/// Special functions error type
#[derive(Error, Debug)]
pub enum SpecialError {
    /// Computation error (generic error)
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// Domain error (input outside valid domain)
    #[error("Domain error: {0}")]
    DomainError(String),

    /// Value error (invalid value)
    #[error("Value error: {0}")]
    ValueError(String),

    /// Not implemented error
    #[error("Not implemented: {0}")]
    NotImplementedError(String),

    /// Convergence error (algorithm did not converge)
    #[error("Convergence error: {0}")]
    ConvergenceError(String),

    /// Core error propagation
    #[error("Core error: {0}")]
    CoreError(#[from] CoreError),
}

/// Result type for special functions operations
pub type SpecialResult<T> = Result<T, SpecialError>;

/// Convert from std::num::ParseFloatError
impl From<std::num::ParseFloatError> for SpecialError {
    fn from(err: std::num::ParseFloatError) -> Self {
        SpecialError::ValueError(format!("Failed to parse float: {}", err))
    }
}

/// Convert from std::io::Error
impl From<std::io::Error> for SpecialError {
    fn from(err: std::io::Error) -> Self {
        SpecialError::ComputationError(format!("IO error: {}", err))
    }
}
