//! Feature detection module
//!
//! This module provides functions for detecting features in n-dimensional arrays,
//! including edge detection, corner detection, and other local feature detection methods.

mod corners;
mod edges;
mod ml_detection;

// Re-export submodule components
pub use self::corners::{fast_corners, harris_corners};
pub use self::edges::{
    canny, edge_detector, edge_detector_simple, gradient_edges, laplacian_edges, sobel_edges,
    EdgeDetectionAlgorithm, EdgeDetectionConfig, GradientMethod,
};

// Machine learning-based detection
pub use self::ml_detection::{
    BatchNormParams, FeatureDetectorWeights, LearnedEdgeDetector, LearnedKeypointDescriptor,
    MLDetectorConfig, ObjectProposal, ObjectProposalGenerator, SemanticFeatureExtractor,
};
