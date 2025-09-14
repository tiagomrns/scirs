//! Convolutional neural network layers implementation
//!
//! This module provides implementations of convolution layers for neural networks,
//! including Conv2D, Conv3D, and their transpose versions, as well as comprehensive
//! pooling layers for 1D, 2D, and 3D data.
//! # Module Organization
//! - [`common`] - Common types, enums, and utility functions
//! - [`conv2d`] - 2D convolution implementation with im2col operations
//! - [`pooling`] - All pooling layer implementations (standard and adaptive)

pub mod common;
pub mod conv2d;
pub mod pooling;

// Re-export main types and functions for backward compatibility
pub use common::PaddingMode;
pub use conv2d::Conv2D;
pub use pooling::{
    MaxPool2D,
    // AdaptiveAvgPool1D, AdaptiveAvgPool2D, AdaptiveAvgPool3D, AdaptiveMaxPool1D, AdaptiveMaxPool2D,
    // AdaptiveMaxPool3D, GlobalAvgPool2D,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::Layer;

    #[test]
    fn test_conv2d_basic() {
        let conv = Conv2D::<f64>::new(3, 8, (3, 3), (1, 1), None).unwrap();
        assert_eq!(conv.layer_type(), "Conv2D");
        assert!(conv.parameter_count() > 0);
    }

    #[test]
    fn test_maxpool2d_basic() {
        let pool = MaxPool2D::<f64>::new((2, 2), (2, 2), None).unwrap();
        assert_eq!(pool.layer_type(), "MaxPool2D");
        assert_eq!(pool.parameter_count(), 0);
    }

    /*
    #[test]
    fn test_adaptive_pools() {
        let adaptive_avg = AdaptiveAvgPool2D::<f64>::new((7, 7), None).unwrap();
        assert_eq!(adaptive_avg.layer_type(), "AdaptiveAvgPool2D");

        let adaptive_max = AdaptiveMaxPool2D::<f64>::new((7, 7), None).unwrap();
        assert_eq!(adaptive_max.layer_type(), "AdaptiveMaxPool2D");

        let global_pool = GlobalAvgPool2D::<f64>::new(None).unwrap();
        assert_eq!(global_pool.layer_type(), "GlobalAvgPool2D");
    }
    */
}
