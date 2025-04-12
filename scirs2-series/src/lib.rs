//! Time series analysis module for SciRS2
//!
//! This module provides functionality for time series analysis, including:
//! - Time series decomposition (trend, seasonality, residual)
//! - Forecasting methods (ARIMA, exponential smoothing)
//! - Feature extraction for time series
//! - Utility functions for time series operations

#![warn(missing_docs)]

pub mod decomposition;
pub mod error;
pub mod features;
pub mod forecasting;
pub mod utils;
