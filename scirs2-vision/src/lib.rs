//! Computer vision module for SciRS2
//!
//! This module provides computer vision functionality that builds on top of the
//! scirs2-ndimage module, including image processing, feature detection, and segmentation.

#![warn(missing_docs)]

pub mod color;
pub mod error;
pub mod feature;
pub mod preprocessing;
pub mod segmentation;

// Re-export commonly used items
pub use error::{Result, VisionError};

// Re-export feature functionality
pub use feature::*;

// Re-export segmentation functionality
pub use segmentation::*;

// Re-export preprocessing functionality
pub use preprocessing::*;

// Re-export color functionality
pub use color::*;
