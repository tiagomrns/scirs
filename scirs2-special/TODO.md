# scirs2-special TODO

This module provides special functions similar to SciPy's special module.

## Current Status

- [x] Set up module structure
- [x] Error handling
- [x] Bessel functions
  - [x] J₀, J₁, Jₙ (first kind)
  - [x] Y₀, Y₁, Yₙ (second kind)
  - [x] I₀, I₁, Iᵥ (modified, first kind)
  - [x] K₀, K₁, Kᵥ (modified, second kind)
- [x] Gamma and related functions
  - [x] Gamma function
  - [x] Log gamma function
  - [x] Digamma function
  - [x] Beta function
  - [x] Incomplete beta function
- [x] Error functions
  - [x] Error function (erf)
  - [x] Complementary error function (erfc)
  - [x] Inverse error function (erfinv)
  - [x] Inverse complementary error function (erfcinv)
- [x] Orthogonal polynomials
  - [x] Legendre polynomials
  - [x] Associated Legendre polynomials
  - [x] Laguerre polynomials
  - [x] Generalized Laguerre polynomials
  - [x] Hermite polynomials
  - [x] Chebyshev polynomials
  - [x] Gegenbauer polynomials
  - [x] Jacobi polynomials
- [x] Example for getting function values

## Future Tasks

- [x] Fix Clippy warning for needless_range_loop in orthogonal.rs
- [ ] Add more special functions
  - [x] Airy functions
  - [x] Elliptic integrals and functions
  - [x] Hypergeometric functions
  - [x] Spherical harmonics
  - [x] Mathieu functions
  - [x] Zeta functions
- [ ] Enhance numerical precision
  - [ ] Improved algorithms for edge cases
  - [ ] Better handling of overflow and underflow
  - [ ] Extended precision options
- [ ] Optimize performance
  - [ ] Use more efficient algorithms for function evaluation
  - [ ] Precomputed coefficients and lookup tables where appropriate
- [ ] Add more examples and documentation
  - [ ] Tutorial for common special function applications
  - [ ] Visual examples showing function behavior
- [ ] Fix ignored doctests

## Long-term Goals

- [ ] Performance comparable to or better than SciPy's special
- [ ] Integration with statistical and physics modules
- [ ] Support for arbitrary precision computation
- [ ] Comprehensive coverage of all SciPy special functions
- [ ] Advanced visualization tools for special functions
- [ ] Domain-specific packages for physics, engineering, and statistics