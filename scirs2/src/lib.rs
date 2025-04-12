//! # SciRS2
//!
//! SciRS2 is a comprehensive scientific computing library for Rust,
//! aiming to provide functionality similar to SciPy in Python.
//!
//! ## Overview
//!
//! SciRS2 provides a collection of modules for various scientific computing tasks:
//!
//! * Linear algebra operations - [`linalg`](linalg)
//! * Statistical functions - [`stats`](stats)
//! * Optimization algorithms - [`optimize`](optimize)
//! * Numerical integration - [`integrate`](integrate)
//! * Fast Fourier Transform - [`fft`](fft)
//! * Signal processing - [`signal`](signal)
//! * Special functions - [`special`](special)
//! * Sparse matrix operations - [`sparse`](sparse)
//! * Spatial algorithms - [`spatial`](spatial)
//! * Clustering algorithms - [`cluster`](cluster)
//! * N-dimensional image processing - [`ndimage`](ndimage)
//! * Physical and mathematical constants - [`constants`](constants)
//! * Input/output utilities - [`io`](io)
//! * And more...
//!
//! ## Examples
//!
//! ```rust,no_run
//! // Example will be provided as modules are implemented
//! ```

// Re-export from scirs2-core
pub use scirs2_core::{constants, error, utils};

// Optional modules (enabled via features)
#[cfg(feature = "linalg")]
pub use scirs2_linalg as linalg;

#[cfg(feature = "integrate")]
pub use scirs2_integrate as integrate;

#[cfg(feature = "interpolate")]
pub use scirs2_interpolate as interpolate;

#[cfg(feature = "optimize")]
pub use scirs2_optimize as optimize;

#[cfg(feature = "fft")]
pub use scirs2_fft as fft;

#[cfg(feature = "stats")]
pub use scirs2_stats as stats;

#[cfg(feature = "special")]
pub use scirs2_special as special;

#[cfg(feature = "signal")]
pub use scirs2_signal as signal;

#[cfg(feature = "sparse")]
pub use scirs2_sparse as sparse;

#[cfg(feature = "spatial")]
pub use scirs2_spatial as spatial;

// Optional advanced modules
#[cfg(feature = "cluster")]
pub use scirs2_cluster as cluster;

#[cfg(feature = "ndimage")]
pub use scirs2_ndimage as ndimage;

#[cfg(feature = "io")]
pub use scirs2_io as io;

#[cfg(feature = "datasets")]
pub use scirs2_datasets as datasets;

// Optional AI/ML modules
#[cfg(feature = "neural")]
pub use scirs2_neural as neural;

#[cfg(feature = "optim")]
pub use scirs2_optim as optim;

#[cfg(feature = "graph")]
pub use scirs2_graph as graph;

#[cfg(feature = "transform")]
pub use scirs2_transform as transform;

#[cfg(feature = "metrics")]
pub use scirs2_metrics as metrics;

#[cfg(feature = "text")]
pub use scirs2_text as text;

#[cfg(feature = "vision")]
pub use scirs2_vision as vision;

#[cfg(feature = "series")]
pub use scirs2_series as series;

// Public API
/// SciRS2 version information
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
