//! Quasi-Monte Carlo
//!
//! This module provides functions for quasi-Monte Carlo integration,
//! following SciPy's `stats.qmc` module.

use crate::error::{StatsError, StatsResult};

/// Sobol sequence (placeholder)
pub fn sobol() -> StatsResult<()> {
    Err(StatsError::NotImplementedError(
        "Sobol sequence not yet implemented".to_string(),
    ))
}

/// Halton sequence (placeholder)
pub fn halton() -> StatsResult<()> {
    Err(StatsError::NotImplementedError(
        "Halton sequence not yet implemented".to_string(),
    ))
}

/// Latin hypercube sampling (placeholder)
pub fn latin_hypercube() -> StatsResult<()> {
    Err(StatsError::NotImplementedError(
        "Latin hypercube sampling not yet implemented".to_string(),
    ))
}
