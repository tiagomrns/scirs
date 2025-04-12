//! Error types for the SciRS2 signal processing module

use thiserror::Error;

/// Signal processing error type
#[derive(Error, Debug)]
pub enum SignalError {
    /// Computation error (generic error)
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// Dimension mismatch error
    #[error("Dimension mismatch error: {0}")]
    DimensionError(String),

    /// Value error (invalid value)
    #[error("Value error: {0}")]
    ValueError(String),

    /// Not implemented error
    #[error("Not implemented: {0}")]
    NotImplementedError(String),
}

/// Result type for signal processing operations
pub type SignalResult<T> = Result<T, SignalError>;
