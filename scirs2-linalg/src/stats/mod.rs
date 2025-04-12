//! Statistical functions for matrices
//!
//! This module provides statistical functions for operating on matrices,
//! including covariance and correlation computation, dimensionality reduction,
//! and multivariate distributions.

pub mod covariance;

// Re-export key functions
pub use covariance::{correlation_matrix, covariance_matrix};
