//! Financial utilities and helper functions
//!
//! This module provides common utilities, mathematical functions, and helper tools
//! used throughout the finance module.
//!
//! # Modules
//! - `math`: Mathematical utilities for finance
//! - `calibration`: Model calibration tools
//! - `simulation`: Simulation utilities and random generators

pub mod calibration;
pub mod math;
pub mod simulation;

// Re-export commonly used utilities
pub use calibration::{CalibrationResult, Calibrator};
pub use math::{interpolate_smile, vol_surface_arbitrage_free};
pub use simulation::{PathGenerator, RandomNumberGenerator};

// TODO: Add common financial utilities
// - Date/time handling for financial contracts
// - Day count conventions
// - Business calendar implementations
// - Market data structures and conversions
