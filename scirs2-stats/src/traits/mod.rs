//! Trait definitions for distributions and statistical objects
//!
//! This module provides traits that define common interfaces for statistical
//! distributions and other objects used throughout the library.

pub mod distribution;

// Re-export all traits at the module level
pub use distribution::*;
