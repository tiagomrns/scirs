//! Markov Chain Monte Carlo (MCMC) methods
//!
//! This module provides implementations of MCMC algorithms for sampling from
//! complex probability distributions including:
//! - Metropolis-Hastings
//! - Gibbs sampling
//! - Hamiltonian Monte Carlo
//! - Advanced methods (Multiple-try Metropolis, Parallel Tempering, Slice Sampling, Ensemble Methods)

mod advanced;
mod enhanced_hamiltonian;
mod gibbs;
mod hamiltonian;
mod metropolis;

pub use advanced::*;
pub use enhanced_hamiltonian::*;
pub use gibbs::*;
pub use hamiltonian::*;
pub use metropolis::*;

#[allow(unused_imports)]
use crate::error::StatsResult as Result;
#[allow(unused_imports)]
use scirs2_core::{simd_ops::SimdUnifiedOps, validation::*};
