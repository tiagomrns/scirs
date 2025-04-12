//! Numerical integration module
//!
//! This module provides implementations of various numerical integration methods.
//! These methods are used to approximate the value of integrals numerically and
//! solve ordinary differential equations (ODEs).
//!
//! ## Overview
//!
//! * Numerical quadrature methods for definite integrals (`quad` module)
//!   * Basic methods (trapezoid rule, Simpson's rule)
//!   * Adaptive quadrature for improved accuracy
//!   * Gaussian quadrature for high accuracy with fewer function evaluations
//!   * Romberg integration for accelerated convergence
//!   * Monte Carlo methods for high-dimensional integrals
//! * ODE solvers for initial value problems (`ode` module)
//!   * Euler and Runge-Kutta methods
//!   * Support for first-order ODE systems
//!
//! ## Usage Examples
//!
//! ### Basic Numerical Integration
//!
//! ```ignore
//! use scirs2_integrate::quad;
//!
//! // Integrate f(x) = x² from 0 to 1 (exact result: 1/3)
//! let result = quad(|x: f64| x * x, 0.0, 1.0, None).unwrap();
//! assert!((result.value - 1.0/3.0).abs() < 1e-8);
//! ```ignore
//!
//! ### Gaussian Quadrature
//!
//! ```ignore
//! use scirs2_integrate::gaussian::gauss_legendre;
//!
//! // Integrate f(x) = x² from 0 to 1 (exact result: 1/3)
//! let result = gauss_legendre(|x: f64| x * x, 0.0, 1.0, 5).unwrap();
//! assert!((result - 1.0/3.0).abs() < 1e-10);
//! ```ignore
//!
//! ### Romberg Integration
//!
//! ```ignore
//! use scirs2_integrate::romberg::romberg;
//!
//! // Integrate f(x) = x² from 0 to 1 (exact result: 1/3)
//! let result = romberg(|x: f64| x * x, 0.0, 1.0, None).unwrap();
//! assert!((result.value - 1.0/3.0).abs() < 1e-10);
//! ```ignore
//!
//! ### Monte Carlo Integration
//!
//! ```ignore
//! use scirs2_integrate::monte_carlo::{monte_carlo, MonteCarloOptions};
//! use ndarray::ArrayView1;
//!
//! // Integrate f(x) = x² from 0 to 1 (exact result: 1/3)
//! let options = MonteCarloOptions {
//!     n_samples: 10000,
//!     seed: Some(42),  // For reproducibility
//!     ..Default::default()
//! };
//!
//! let result = monte_carlo(
//!     |x: ArrayView1<f64>| x[0] * x[0],
//!     &[(0.0, 1.0)],
//!     Some(options)
//! ).unwrap();
//!
//! // Monte Carlo has statistical error, so we use a loose tolerance
//! assert!((result.value - 1.0/3.0).abs() < 0.02);
//! ```ignore
//!
//! ### ODE Solving
//!
//! ```ignore
//! use ndarray::array;
//! use scirs2_integrate::ode::{solve_ivp, ODEOptions, ODEMethod};
//!
//! // Solve y'(t) = -y with initial condition y(0) = 1
//! let result = solve_ivp(
//!     |_: f64, y| array![-y[0]],
//!     [0.0, 1.0],
//!     array![1.0],
//!     None
//! ).unwrap();
//!
//! // Final value should be close to e^(-1) ≈ 0.368
//! let final_y = result.y.last().unwrap()[0];
//! assert!((final_y - 0.368).abs() < 1e-2);
//! ```ignore

// Export error types
pub mod error;
pub use error::{IntegrateError, IntegrateResult};

// Integration modules
pub mod gaussian;
pub mod monte_carlo;
pub mod ode;
pub mod quad;
pub mod romberg;

// Re-exports for convenience
pub use ode::{solve_ivp, ODEMethod, ODEOptions, ODEResult};
pub use quad::{quad, simpson, trapezoid};

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
