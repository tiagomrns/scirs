//! Special functions module
//!
//! This module provides implementations of various special mathematical functions,
//! following SciPy's `special` module.
//!
//! ## Overview
//!
//! * Gamma and Beta functions
//! * Bessel functions
//! * Orthogonal polynomials
//! * Error functions
//! * Airy functions
//! * Exponential and logarithmic integrals
//! * Elliptic integrals and functions
//! * Hypergeometric functions
//! * Spherical harmonics
//! * Mathieu functions
//! * Zeta functions
//!
//! ## Examples
//!
//! ```
//! use scirs2_special::gamma;
//!
//! let gamma_value = gamma(5.0f64);
//! assert!((gamma_value - 24.0).abs() < 1e-10);
//! ```

// Export error types
pub mod error;
pub use error::{SpecialError, SpecialResult};

// Modules
mod airy;
mod bessel;
mod elliptic;
mod gamma;
mod hypergeometric;
mod mathieu;
mod orthogonal;
mod spherical_harmonics;
mod zeta;

// Re-export common functions
// Note: These functions require various trait bounds in their implementation,
// including Float, FromPrimitive, Debug, AddAssign, etc.
pub use airy::{ai, aip, bi, bip};
pub use bessel::{i0, i1, iv, j0, j1, jn, k0, k1, kv, y0, y1, yn};
pub use elliptic::{
    elliptic_e, elliptic_e_inc, elliptic_f, elliptic_k, elliptic_pi, jacobi_cn, jacobi_dn,
    jacobi_sn,
};
pub use gamma::{
    beta, betainc, betainc_regularized, betaincinv, betaln, digamma, gamma, gammaln, loggamma,
};
pub use hypergeometric::{hyp1f1, hyp2f1, ln_pochhammer, pochhammer};
pub use mathieu::{
    mathieu_a, mathieu_b, mathieu_cem, mathieu_even_coef, mathieu_odd_coef, mathieu_sem,
};
pub use orthogonal::{
    chebyshev, gegenbauer, hermite, hermite_prob, jacobi, laguerre, laguerre_generalized, legendre,
    legendre_assoc,
};
pub use spherical_harmonics::{sph_harm, sph_harm_complex};
pub use zeta::{hurwitz_zeta, zeta, zetac};

// Error function and related functions
pub mod erf;
pub use erf::{erf, erfc, erfcinv, erfinv};

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_gamma_function() {
        // Test integer values
        assert_relative_eq!(gamma(1.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(gamma(2.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(gamma(3.0), 2.0, epsilon = 1e-10);
        assert_relative_eq!(gamma(4.0), 6.0, epsilon = 1e-10);
        assert_relative_eq!(gamma(5.0), 24.0, epsilon = 1e-10);
    }
}
