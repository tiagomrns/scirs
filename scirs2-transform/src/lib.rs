//! Data transformation module for SciRS2
//!
//! This module provides utilities for transforming data in ways that are useful
//! for machine learning and data analysis. The main functionalities include:
//!
//! - Data normalization and standardization
//! - Feature engineering
//! - Dimensionality reduction

#![warn(missing_docs)]

pub mod error;
pub mod features;
pub mod normalize;
pub mod reduction;

// Re-export important types and functions
pub use error::{Result, TransformError};
pub use features::{
    binarize, discretize_equal_frequency, discretize_equal_width, log_transform, power_transform,
    PolynomialFeatures,
};
pub use normalize::{normalize_array, normalize_vector, NormalizationMethod, Normalizer};
pub use reduction::{TruncatedSVD, LDA, PCA};
