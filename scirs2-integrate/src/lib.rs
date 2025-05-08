//! Numerical integration module
//!
//! This module provides implementations of various numerical integration methods.
//! These methods are used to approximate the value of integrals numerically and
//! solve ordinary differential equations (ODEs) including initial value problems (IVPs)
//! and boundary value problems (BVPs).
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
//!   * Variable step-size methods (RK45, RK23)
//!   * Implicit methods for stiff equations (BDF)
//!   * Support for first-order ODE systems
//! * Boundary value problem solvers (`bvp` module)
//!   * Two-point boundary value problems
//!   * Support for Dirichlet and Neumann boundary conditions
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
//! ```
//!
//! ### Gaussian Quadrature
//!
//! ```ignore
//! use scirs2_integrate::gaussian::gauss_legendre;
//!
//! // Integrate f(x) = x² from 0 to 1 (exact result: 1/3)
//! let result = gauss_legendre(|x: f64| x * x, 0.0, 1.0, 5).unwrap();
//! assert!((result - 1.0/3.0).abs() < 1e-10);
//! ```
//!
//! ### Romberg Integration
//!
//! ```ignore
//! use scirs2_integrate::romberg::romberg;
//!
//! // Integrate f(x) = x² from 0 to 1 (exact result: 1/3)
//! let result = romberg(|x: f64| x * x, 0.0, 1.0, None).unwrap();
//! assert!((result.value - 1.0/3.0).abs() < 1e-10);
//! ```
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
//! ```
//!
//! ### ODE Solving (Initial Value Problem)
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
//! ```
//!
//! ### Boundary Value Problem Solving
//!
//! ```ignore
//! use ndarray::{array, ArrayView1};
//! use scirs2_integrate::bvp::{solve_bvp, BVPOptions};
//! use std::f64::consts::PI;
//!
//! // Solve the harmonic oscillator ODE: y'' + y = 0
//! // as a first-order system: y0' = y1, y1' = -y0
//! // with boundary conditions y0(0) = 0, y0(pi) = 0
//!
//! let fun = |_x: f64, y: ArrayView1<f64>| array![y[1], -y[0]];
//!
//! let bc = |ya: ArrayView1<f64>, yb: ArrayView1<f64>| {
//!     // Boundary conditions: y0(0) = 0, y0(pi) = 0
//!     array![ya[0], yb[0]]
//! };
//!
//! // Initial mesh: 5 points from 0 to π
//! let x = vec![0.0, PI/4.0, PI/2.0, 3.0*PI/4.0, PI];
//!
//! // Initial guess: zeros
//! let y_init = vec![
//!     array![0.0, 0.0],
//!     array![0.0, 0.0],
//!     array![0.0, 0.0],
//!     array![0.0, 0.0],
//!     array![0.0, 0.0],
//! ];
//!
//! let result = solve_bvp(fun, bc, Some(x), y_init, None).unwrap();
//!
//! // The solution should approximate sin(x)
//! // Check at x = π/2 where sin(π/2) = 1.0
//! let idx_mid = result.x.len() / 2;
//! let scale = result.y[idx_mid][0]; // Scale factor
//!
//! // Check solution at a specific point
//! let i = 1; // Check at x = π/4
//! let y_val = result.y[i][0];
//! let sin_val = scale * x[i].sin();
//! assert!((y_val - sin_val).abs() < 1e-2);
//! ```

// Export error types
pub mod error;
pub use error::{IntegrateError, IntegrateResult};

// Integration modules
pub mod bvp;
pub mod gaussian;
pub mod monte_carlo;
pub mod ode;
pub mod quad;
pub mod romberg;
pub mod utils;

// Re-exports for convenience
pub use bvp::{solve_bvp, solve_bvp_auto, BVPOptions, BVPResult};
pub use ode::{solve_ivp, ODEMethod, ODEOptions, ODEResult};
pub use quad::{quad, simpson, trapezoid};

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
