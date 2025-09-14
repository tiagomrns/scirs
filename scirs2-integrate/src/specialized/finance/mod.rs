//! Financial modeling solvers for stochastic PDEs
//!
//! This module provides specialized solvers for quantitative finance applications,
//! including Black-Scholes, stochastic volatility models, and jump-diffusion processes.

pub mod derivatives;
pub mod ml;
pub mod models;
pub mod pricing;
pub mod risk;
pub mod solvers;
pub mod types;
pub mod utils;

// Re-export commonly used types
pub use models::*;
pub use risk::greeks::Greeks;
pub use solvers::stochastic_pde::StochasticPDESolver;
pub use types::*;

// Re-export pricing methods
pub use pricing::{black_scholes::*, finite_difference::*, monte_carlo::*};

// Re-export main solver for backwards compatibility
pub use solvers::StochasticPDESolver as FinancialPDESolver;
