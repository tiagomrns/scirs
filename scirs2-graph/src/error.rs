//! Error types for the graph processing module

use thiserror::Error;

/// Error type for graph processing operations
#[derive(Error, Debug)]
pub enum GraphError {
    /// Invalid graph structure
    #[error("Invalid graph: {0}")]
    InvalidGraph(String),

    /// Algorithm error
    #[error("Algorithm error: {0}")]
    AlgorithmError(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Linear algebra error
    #[error("Linear algebra error: {0}")]
    LinAlgError(String),

    /// Sparse matrix error
    #[error("Sparse matrix error: {0}")]
    SparseError(#[from] scirs2_sparse::error::SparseError),

    /// Core error
    #[error("Core error: {0}")]
    CoreError(#[from] scirs2_core::error::CoreError),

    /// Other error
    #[error("Error: {0}")]
    Other(String),
}

/// Result type for graph processing operations
pub type Result<T> = std::result::Result<T, GraphError>;
