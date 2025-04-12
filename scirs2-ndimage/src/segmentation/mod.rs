//! Image segmentation module
//!
//! This module provides functions for segmenting images into regions
//! or partitioning images into meaningful parts.

mod thresholding;
mod watershed;

// Re-export submodule components
pub use self::thresholding::{
    adaptive_threshold, otsu_threshold, threshold_binary, AdaptiveMethod,
};
pub use self::watershed::{marker_watershed, watershed};
