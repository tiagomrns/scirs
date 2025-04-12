//! Optimization module for SciRS
//!
//! This module provides implementations of various optimization algorithms,
//! modeled after SciPy's `optimize` module.
//!

#![allow(clippy::field_reassign_with_default)]
//! ## Submodules
//!
//! * `unconstrained`: Unconstrained optimization algorithms
//! * `constrained`: Constrained optimization algorithms
//! * `least_squares`: Least squares minimization
//! * `roots`: Root finding algorithms
//!
//! ## Examples
//!
//! ```
//! // Example of minimizing a function using BFGS
//! use ndarray::array;
//! use scirs2_optimize::unconstrained::{minimize, Method};
//!
//! fn rosenbrock(x: &[f64]) -> f64 {
//!     let a = 1.0;
//!     let b = 100.0;
//!     let x0 = x[0];
//!     let x1 = x[1];
//!     (a - x0).powi(2) + b * (x1 - x0.powi(2)).powi(2)
//! }
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let initial_guess = array![0.0, 0.0];
//! let result = minimize(rosenbrock, &initial_guess, Method::BFGS, None)?;
//!
//! println!("Solution: {:?}", result.x);
//! println!("Function value at solution: {}", result.fun);
//! println!("Number of iterations: {}", result.nit);
//! println!("Success: {}", result.success);
//! # Ok(())
//! # }
//! ```

// Export error types
pub mod error;
pub use error::{OptimizeError, OptimizeResult};

// Module structure
pub mod constrained;
pub mod least_squares;
pub mod roots;
pub mod roots_anderson;
pub mod roots_krylov;
pub mod unconstrained;

// Common optimization result structure
pub mod result;
pub use result::OptimizeResults;

// Convenience re-exports for common functions
pub use constrained::minimize_constrained;
pub use least_squares::least_squares;
pub use roots::root;
pub use unconstrained::minimize;

// Prelude module for convenient imports
pub mod prelude {
    pub use crate::constrained::{minimize_constrained, Method as ConstrainedMethod};
    pub use crate::error::{OptimizeError, OptimizeResult};
    pub use crate::least_squares::{least_squares, Method as LeastSquaresMethod};
    pub use crate::result::OptimizeResults;
    pub use crate::roots::{root, Method as RootMethod};
    pub use crate::unconstrained::{minimize, Method as UnconstrainedMethod};
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
