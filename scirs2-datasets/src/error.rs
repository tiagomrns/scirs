//! Error types for the datasets module

use std::io;
use thiserror::Error;

/// Error type for datasets operations
#[derive(Error, Debug)]
pub enum DatasetsError {
    /// Invalid data format
    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    /// Data loading error
    #[error("Loading error: {0}")]
    LoadingError(String),

    /// Download error
    #[error("Download error: {0}")]
    DownloadError(String),

    /// Cache error
    #[error("Cache error: {0}")]
    CacheError(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] io::Error),

    /// Serialization/Deserialization error
    #[error("Serialization error: {0}")]
    SerdeError(String),

    /// Other error
    #[error("Error: {0}")]
    Other(String),
}

/// Result type for datasets operations
pub type Result<T> = std::result::Result<T, DatasetsError>;
