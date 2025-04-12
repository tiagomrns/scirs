//! Error types for the time series module

// No imports needed here, thiserror handles the implementations
use thiserror::Error;

/// Error type for time series operations
#[derive(Debug, Error)]
pub enum TimeSeriesError {
    /// Invalid input data
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Forecasting error
    #[error("Forecasting error: {0}")]
    ForecastingError(String),

    /// Decomposition error
    #[error("Decomposition error: {0}")]
    DecompositionError(String),

    /// Feature extraction error
    #[error("Feature extraction error: {0}")]
    FeatureExtractionError(String),

    /// Statistical error
    #[error("Statistical error: {0}")]
    StatisticalError(String),

    /// Other error
    #[error("Error: {0}")]
    Other(String),
}

/// Result type for time series operations
pub type Result<T> = std::result::Result<T, TimeSeriesError>;
