//! Error types for the metrics module

use thiserror::Error;

/// Error type for metrics operations
#[derive(Error, Debug)]
pub enum MetricsError {
    /// Invalid input data
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Calculation error
    #[error("Calculation error: {0}")]
    CalculationError(String),

    /// Statistics error
    #[error("Statistics error: {0}")]
    StatsError(String),

    /// Linear algebra error
    #[error("Linear algebra error: {0}")]
    LinalgError(String),

    /// Core error
    #[error("Core error: {0}")]
    CoreError(#[from] scirs2_core::error::CoreError),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Other error
    #[error("Error: {0}")]
    Other(String),
}

/// Result type for metrics operations
pub type Result<T> = std::result::Result<T, MetricsError>;
