//! Error types for the SciRS2 statistics module

use scirs2_core::error::{CoreError, ErrorContext, ErrorLocation};
use thiserror::Error;

/// Statistics error type
#[derive(Error, Debug)]
pub enum StatsError {
    /// Computation error (generic error)
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// Domain error (input outside valid domain)
    #[error("Domain error: {0}")]
    DomainError(String),

    /// Dimension mismatch error
    #[error("Dimension mismatch error: {0}")]
    DimensionMismatch(String),

    /// Value error (invalid value)
    #[error("Value error: {0}")]
    InvalidArgument(String),

    /// Not implemented error
    #[error("Not implemented: {0}")]
    NotImplementedError(String),

    /// Core error (propagated from scirs2-core)
    #[error("{0}")]
    CoreError(#[from] CoreError),
}

// The #[from] attribute in the CoreError variant handles the conversion automatically

/// Result type for statistics operations
pub type StatsResult<T> = Result<T, StatsError>;

/// Create a function to convert from StatsResult to CoreError::ValidationError
pub fn convert_to_validation_error<T, S: Into<String>>(
    result: StatsResult<T>,
    message: S,
) -> Result<T, CoreError> {
    match result {
        Ok(val) => Ok(val),
        Err(err) => Err(CoreError::ValidationError(
            ErrorContext::new(format!("{}: {}", message.into(), err))
                .with_location(ErrorLocation::new(file!(), line!())),
        )),
    }
}
