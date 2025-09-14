//! Enhanced visualization capabilities for clustering results
//!
//! This module provides comprehensive visualization tools for clustering algorithms,
//! including 2D/3D scatter plots, animations, interactive visualizations, real-time
//! streaming displays, and various export formats.
//!
//! The visualization module has been restructured into submodules for better organization:
//! - `animation`: Animation and streaming visualization capabilities
//! - `interactive`: Interactive 3D visualization with camera controls
//! - `export`: Export functionality for multiple formats
//!
//! This file re-exports the main functionality for backward compatibility.

// Re-export everything from the module structure
mod visualization_mod {
    pub use crate::visualization::*;
}

// For backward compatibility, include the module here
#[path = "visualization/mod.rs"]
pub mod visualization;

// Re-export main types and functions
pub use visualization::*;
