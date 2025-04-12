//! Error types for the `SciRS2` FFT module

use thiserror::Error;

/// FFT error type
#[derive(Error, Debug)]
pub enum FFTError {
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

/// Result type for FFT operations
pub type FFTResult<T> = Result<T, FFTError>;
