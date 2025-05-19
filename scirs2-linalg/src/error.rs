//! Error types for the SciRS2 linear algebra module

use scirs2_core::error::CoreError;
use thiserror::Error;

/// Linear algebra error type
#[derive(Error, Debug)]
pub enum LinalgError {
    /// Core error
    #[error(transparent)]
    CoreError(#[from] CoreError),

    /// Computation error (generic error)
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// Domain error (input outside valid domain)
    #[error("Domain error: {0}")]
    DomainError(String),

    /// Convergence error (algorithm did not converge)
    #[error("Convergence error: {0}")]
    ConvergenceError(String),

    /// Dimension mismatch error
    #[error("Dimension mismatch error: {0}")]
    DimensionError(String),

    /// Shape error (matrices/arrays have incompatible shapes)
    #[error("Shape error: {0}")]
    ShapeError(String),

    /// Out of bounds error
    #[error("Index out of bounds: {0}")]
    IndexError(String),

    /// Singular matrix error
    #[error("Singular matrix error: {0}")]
    SingularMatrixError(String),

    /// Non-positive definite matrix error
    #[error("Non-positive definite matrix error: {0}")]
    NonPositiveDefiniteError(String),

    /// Not implemented error
    #[error("Not implemented: {0}")]
    NotImplementedError(String),

    /// Implementation error (method exists but not fully implemented yet)
    #[error("Implementation error: {0}")]
    ImplementationError(String),

    /// Value error (invalid value)
    #[error("Value error: {0}")]
    ValueError(String),

    /// Invalid input error
    #[error("Invalid input error: {0}")]
    InvalidInputError(String),

    /// Numerical error (e.g., overflow, underflow, division by zero)
    #[error("Numerical error: {0}")]
    NumericalError(String),
}

/// Result type for linear algebra operations
pub type LinalgResult<T> = Result<T, LinalgError>;

/// Checks if a condition is true, otherwise returns a domain error
///
/// # Arguments
///
/// * `condition` - The condition to check
/// * `message` - The error message if the condition is false
///
/// # Returns
///
/// * `Ok(())` if the condition is true
/// * `Err(LinalgError::DomainError)` if the condition is false
pub fn check_domain<S: AsRef<str>>(condition: bool, message: S) -> LinalgResult<()> {
    if condition {
        Ok(())
    } else {
        Err(LinalgError::DomainError(message.as_ref().to_string()))
    }
}

/// Checks matrix dimensions
///
/// # Arguments
///
/// * `condition` - The condition to check
/// * `message` - The error message if the condition is false
///
/// # Returns
///
/// * `Ok(())` if the condition is true
/// * `Err(LinalgError::DimensionError)` if the condition is false
///
/// # Note
///
/// This is a linalg-specific wrapper around scirs2_core::validation functions.
/// For new code, consider using scirs2_core::validation functions directly when possible.
pub fn check_dimensions<S: AsRef<str>>(condition: bool, message: S) -> LinalgResult<()> {
    if condition {
        Ok(())
    } else {
        Err(LinalgError::DimensionError(message.as_ref().to_string()))
    }
}

/// Checks if a value is valid
///
/// # Arguments
///
/// * `condition` - The condition to check
/// * `message` - The error message if the condition is false
///
/// # Returns
///
/// * `Ok(())` if the condition is true
/// * `Err(LinalgError::ValueError)` if the condition is false
///
/// # Note
///
/// This is a linalg-specific wrapper around scirs2_core::validation functions.
/// For new code, consider using scirs2_core::validation functions directly when possible.
pub fn check_value<S: AsRef<str>>(condition: bool, message: S) -> LinalgResult<()> {
    if condition {
        Ok(())
    } else {
        Err(LinalgError::ValueError(message.as_ref().to_string()))
    }
}
