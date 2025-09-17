//! Numeric traits and utilities for ``SciRS2``
//!
//! This module provides traits and utilities for working with numeric types
//! in scientific computing contexts.

use crate::error::{CoreError, CoreResult, ErrorContext};
use num_traits::{Float, Num, NumCast, One, Zero};
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A trait for numeric types that can be used in scientific calculations
///
/// This trait combines common numeric traits required for scientific computing
/// operations, providing a unified trait bound for generic code.
pub trait ScientificNumber:
    Num
    + Clone
    + Copy
    + PartialOrd
    + Debug
    + Zero
    + One
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + NumCast
{
    /// Absolute value
    #[must_use]
    fn abs(self) -> Self;

    /// Square root
    #[must_use]
    fn sqrt(self) -> Self;

    /// Square
    #[must_use]
    fn square(self) -> Self {
        self * self
    }

    /// Maximum of two values
    #[must_use]
    fn max(self, other: Self) -> Self;

    /// Minimum of two values
    #[must_use]
    fn min(self, other: Self) -> Self;

    /// Check if the value is finite
    #[must_use]
    fn is_finite(self) -> bool;

    /// Convert to f64
    #[must_use]
    fn to_f64(self) -> Option<f64>;

    /// Convert from f64
    #[must_use]
    fn from_f64(value: f64) -> Option<Self>;

    /// Convert from little-endian bytes
    #[must_use]
    fn from_le_bytes(bytes: &[u8]) -> Self;

    /// Convert from big-endian bytes
    #[must_use]
    fn from_be_bytes(bytes: &[u8]) -> Self;

    /// Convert to little-endian bytes
    #[must_use]
    fn to_le_bytes(self) -> Vec<u8>;

    /// Convert to big-endian bytes
    #[must_use]
    fn to_be_bytes(self) -> Vec<u8>;
}

/// A trait for real-valued floating point types
pub trait RealNumber: ScientificNumber + Float {
    /// Returns the machine epsilon (the difference between 1.0 and the least value greater than 1.0)
    #[must_use]
    fn epsilon() -> Self;

    /// Exponential function (e^x)
    #[must_use]
    fn exp(self) -> Self;

    /// Natural logarithm (ln(x))
    #[must_use]
    fn ln(self) -> Self;

    /// Base-10 logarithm
    #[must_use]
    fn log10(self) -> Self;

    /// Base-2 logarithm
    #[must_use]
    fn log2(self) -> Self;

    /// Sine function
    #[must_use]
    fn sin(self) -> Self;

    /// Cosine function
    #[must_use]
    fn cos(self) -> Self;

    /// Tangent function
    #[must_use]
    fn tan(self) -> Self;

    /// Hyperbolic sine
    #[must_use]
    fn sinh(self) -> Self;

    /// Hyperbolic cosine
    #[must_use]
    fn cosh(self) -> Self;

    /// Hyperbolic tangent
    #[must_use]
    fn tanh(self) -> Self;

    /// Inverse sine
    #[must_use]
    fn asin(self) -> Self;

    /// Inverse cosine
    #[must_use]
    fn acos(self) -> Self;

    /// Inverse tangent
    #[must_use]
    fn atan(self) -> Self;

    /// Inverse tangent of y/x with correct quadrant
    #[must_use]
    fn atan2(self, other: Self) -> Self;

    /// Power function
    #[must_use]
    fn powf(self, n: Self) -> Self;

    /// Integer power function
    #[must_use]
    fn powi(self, n: i32) -> Self;

    /// Factorial function (approximation for non-integers)
    #[must_use]
    fn factorial(self) -> Self;
}

/// A trait for complex number types
pub trait ComplexNumber: ScientificNumber {
    /// The real part of the complex number
    type RealPart: RealNumber;

    /// Returns the real part of the complex number
    #[must_use]
    fn re(&self) -> Self::RealPart;

    /// Returns the imaginary part of the complex number
    #[must_use]
    fn im(&self) -> Self::RealPart;

    /// Create a new complex number from real and imaginary parts
    #[must_use]
    fn from_parts(re: Self::RealPart, im: Self::RealPart) -> Self;

    /// Returns the complex conjugate
    #[must_use]
    fn conj(self) -> Self;

    /// Returns the magnitude (absolute value)
    #[must_use]
    fn abs(self) -> Self::RealPart;

    /// Returns the argument (phase)
    #[must_use]
    fn arg(self) -> Self::RealPart;

    /// Returns the complex number in exponential form (r, theta)
    #[must_use]
    fn to_polar(self) -> (Self::RealPart, Self::RealPart);

    /// Creates a complex number from polar coordinates
    #[must_use]
    fn from_polar(r: Self::RealPart, theta: Self::RealPart) -> Self;

    /// Exponential function
    #[must_use]
    fn exp(self) -> Self;

    /// Natural logarithm
    #[must_use]
    fn ln(self) -> Self;

    /// Power function with complex exponent
    #[must_use]
    fn powc(self, exp: Self) -> Self;

    /// Power function with real exponent
    #[must_use]
    fn powf(self, exp: Self::RealPart) -> Self;

    /// Square root
    #[must_use]
    fn sqrt(self) -> Self;
}

/// A trait for integers that can be used in scientific calculations
pub trait ScientificInteger: ScientificNumber + Eq {
    /// Greatest common divisor
    #[must_use]
    fn gcd(self, other: Self) -> Self;

    /// Least common multiple
    #[must_use]
    fn lcm(self, other: Self) -> Self;

    /// Check if the number is prime
    #[must_use]
    fn is_prime(self) -> bool;

    /// Check if the number is even
    #[must_use]
    fn is_even(self) -> bool;

    /// Check if the number is odd
    #[must_use]
    fn is_odd(self) -> bool;

    /// Modular exponentiation (self^exp mod modulus)
    #[must_use]
    fn mod_pow(self, exp: Self, modulus: Self) -> Self;

    /// Factorial
    fn factorial(self) -> CoreResult<Self>;

    /// Binomial coefficient (n choose k)
    #[must_use]
    fn binomial(self, k: Self) -> Self;
}

// Implement ScientificNumber for f32
impl ScientificNumber for f32 {
    fn abs(self) -> Self {
        self.abs()
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn max(self, other: Self) -> Self {
        self.max(other)
    }

    fn min(self, other: Self) -> Self {
        self.min(other)
    }

    fn is_finite(self) -> bool {
        self.is_finite()
    }

    fn to_f64(self) -> Option<f64> {
        Some(self as f64)
    }

    fn from_f64(value: f64) -> Option<Self> {
        if value.is_finite() && value <= Self::MAX as f64 && value >= Self::MIN as f64 {
            Some(value as f32)
        } else {
            None
        }
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        let mut array = [0u8; 4];
        array.copy_from_slice(&bytes[..4]);
        f32::from_le_bytes(array)
    }

    fn from_be_bytes(bytes: &[u8]) -> Self {
        let mut array = [0u8; 4];
        array.copy_from_slice(&bytes[..4]);
        f32::from_be_bytes(array)
    }

    fn to_le_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }

    fn to_be_bytes(self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
    }
}

// Implement RealNumber for f32
impl RealNumber for f32 {
    fn epsilon() -> Self {
        Self::EPSILON
    }

    fn exp(self) -> Self {
        self.exp()
    }

    fn ln(self) -> Self {
        self.ln()
    }

    fn log10(self) -> Self {
        self.log10()
    }

    fn log2(self) -> Self {
        self.log2()
    }

    fn sin(self) -> Self {
        self.sin()
    }

    fn cos(self) -> Self {
        self.cos()
    }

    fn tan(self) -> Self {
        self.tan()
    }

    fn sinh(self) -> Self {
        self.sinh()
    }

    fn cosh(self) -> Self {
        self.cosh()
    }

    fn tanh(self) -> Self {
        self.tanh()
    }

    fn asin(self) -> Self {
        self.asin()
    }

    fn acos(self) -> Self {
        self.acos()
    }

    fn atan(self) -> Self {
        self.atan()
    }

    fn atan2(self, other: Self) -> Self {
        self.atan2(other)
    }

    fn powf(self, n: Self) -> Self {
        self.powf(n)
    }

    fn powi(self, n: i32) -> Self {
        self.powi(n)
    }

    fn factorial(self) -> Self {
        if self < 0.0 {
            return Self::NAN;
        }

        // Use Stirling's approximation for non-integers or large values
        if self != self.trunc() || self > 100.0 {
            const SQRT_TWO_PI: f32 = 2.506_628_3;
            return SQRT_TWO_PI * self.powf(self + 0.5) * (-self).exp();
        }

        let mut result = 1.0;
        let n = self as u32;
        for i in 2..=n {
            result *= i as f32;
        }

        result
    }
}

// Implement ScientificNumber for f64
impl ScientificNumber for f64 {
    fn abs(self) -> Self {
        self.abs()
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn max(self, other: Self) -> Self {
        self.max(other)
    }

    fn min(self, other: Self) -> Self {
        self.min(other)
    }

    fn is_finite(self) -> bool {
        self.is_finite()
    }

    fn to_f64(self) -> Option<f64> {
        Some(self)
    }

    fn from_f64(value: f64) -> Option<Self> {
        Some(value)
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        let mut array = [0u8; 8];
        array.copy_from_slice(&bytes[..8]);
        f64::from_le_bytes(array)
    }

    fn from_be_bytes(bytes: &[u8]) -> Self {
        let mut array = [0u8; 8];
        array.copy_from_slice(&bytes[..8]);
        f64::from_be_bytes(array)
    }

    fn to_le_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }

    fn to_be_bytes(self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
    }
}

// Implement RealNumber for f64
impl RealNumber for f64 {
    fn epsilon() -> Self {
        Self::EPSILON
    }

    fn exp(self) -> Self {
        self.exp()
    }

    fn ln(self) -> Self {
        self.ln()
    }

    fn log10(self) -> Self {
        self.log10()
    }

    fn log2(self) -> Self {
        self.log2()
    }

    fn sin(self) -> Self {
        self.sin()
    }

    fn cos(self) -> Self {
        self.cos()
    }

    fn tan(self) -> Self {
        self.tan()
    }

    fn sinh(self) -> Self {
        self.sinh()
    }

    fn cosh(self) -> Self {
        self.cosh()
    }

    fn tanh(self) -> Self {
        self.tanh()
    }

    fn asin(self) -> Self {
        self.asin()
    }

    fn acos(self) -> Self {
        self.acos()
    }

    fn atan(self) -> Self {
        self.atan()
    }

    fn atan2(self, other: Self) -> Self {
        self.atan2(other)
    }

    fn powf(self, n: Self) -> Self {
        self.powf(n)
    }

    fn powi(self, n: i32) -> Self {
        self.powi(n)
    }

    fn factorial(self) -> Self {
        if self < 0.0 {
            return Self::NAN;
        }

        // Use Stirling's approximation for non-integers or large values
        if self != self.trunc() || self > 170.0 {
            const SQRT_TWO_PI: f64 = 2.506_628_274_631_000_2;
            return SQRT_TWO_PI * self.powf(self + 0.5) * (-self).exp();
        }

        let mut result = 1.0;
        let n = self as u32;
        for i in 2..=n {
            result *= i as f64;
        }

        result
    }
}

// Implement ScientificNumber for i32
impl ScientificNumber for i32 {
    fn abs(self) -> Self {
        self.abs()
    }

    fn sqrt(self) -> Self {
        (self as f64).sqrt() as i32
    }

    fn max(self, other: Self) -> Self {
        std::cmp::max(self, other)
    }

    fn min(self, other: Self) -> Self {
        std::cmp::min(self, other)
    }

    fn is_finite(self) -> bool {
        true // integers are always finite
    }

    fn to_f64(self) -> Option<f64> {
        Some(self as f64)
    }

    fn from_f64(value: f64) -> Option<Self> {
        if value.is_finite() && value <= i32::MAX as f64 && value >= i32::MIN as f64 {
            Some(value as i32)
        } else {
            None
        }
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        let mut array = [0u8; 4];
        array.copy_from_slice(&bytes[..4]);
        i32::from_le_bytes(array)
    }

    fn from_be_bytes(bytes: &[u8]) -> Self {
        let mut array = [0u8; 4];
        array.copy_from_slice(&bytes[..4]);
        i32::from_be_bytes(array)
    }

    fn to_le_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }

    fn to_be_bytes(self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
    }
}

// Implement ScientificInteger for i32
impl ScientificInteger for i32 {
    fn gcd(self, other: Self) -> Self {
        let mut a = self.abs();
        let mut b = other.abs();

        if a == 0 {
            return b;
        }
        if b == 0 {
            return a;
        }

        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }

        a
    }

    fn lcm(self, other: Self) -> Self {
        if self == 0 || other == 0 {
            return 0;
        }

        let gcd = self.gcd(other);

        (self / gcd) * other
    }

    fn is_prime(self) -> bool {
        if self <= 1 {
            return false;
        }
        if self <= 3 {
            return true;
        }
        if self % 2 == 0 || self % 3 == 0 {
            return false;
        }

        let mut i = 5;
        while i * i <= self {
            if self % i == 0 || self % (i + 2) == 0 {
                return false;
            }
            i += 6;
        }

        true
    }

    fn is_even(self) -> bool {
        self % 2 == 0
    }

    fn is_odd(self) -> bool {
        self % 2 != 0
    }

    fn mod_pow(self, exp: Self, modulus: Self) -> Self {
        if modulus == 1 {
            return 0;
        }

        let mut base = self % modulus;
        let mut result = 1;
        let mut exponent = exp;

        while exponent > 0 {
            if exponent % 2 == 1 {
                result = (result * base) % modulus;
            }
            exponent >>= 1;
            base = (base * base) % modulus;
        }

        result
    }

    fn factorial(self) -> CoreResult<Self> {
        if self < 0 {
            return Err(CoreError::ValueError(ErrorContext::new(
                "Factorial not defined for negative numbers".to_string(),
            )));
        }

        let mut result = 1;
        for i in 2..=self {
            result *= i;
        }

        Ok(result)
    }

    fn binomial(self, k: Self) -> Self {
        if k < 0 || k > self {
            return 0;
        }

        let k = std::cmp::min(k, self - k);
        let mut result = 1;

        for i in 0..k {
            result = result * (self - i) / (i + 1);
        }

        result
    }
}

/// Conversion between different numeric types
pub trait NumericConversion<T> {
    /// Try to convert to the target type
    fn try_convert(self) -> Option<T>;

    /// Convert to the target type, or a default value if conversion fails
    fn convert_or(self, default: T) -> T;

    /// Convert to the target type, or panic if conversion fails
    fn convert(self) -> T;
}

/// Implement NumericConversion for all types that implement NumCast
impl<F, T> NumericConversion<T> for F
where
    F: NumCast,
    T: NumCast,
{
    fn try_convert(self) -> Option<T> {
        num_traits::cast(self)
    }

    fn convert_or(self, default: T) -> T {
        self.try_convert().unwrap_or(default)
    }

    fn convert(self) -> T {
        self.try_convert().expect("Numeric conversion failed")
    }
}

/// Trait for converting between degrees and radians
pub trait AngleConversion: Sized {
    /// Convert from degrees to radians
    fn to_radians(&self) -> CoreResult<Self>
    where
        Self: std::marker::Sized;

    /// Convert from radians to degrees
    fn to_degrees(&self) -> CoreResult<Self>
    where
        Self: std::marker::Sized;
}

/// Implement AngleConversion for all RealNumber types
impl<T: RealNumber> AngleConversion for T {
    fn to_radians(&self) -> CoreResult<Self> {
        let pi = T::from_f64(std::f64::consts::PI).ok_or_else(|| {
            CoreError::ValueError(ErrorContext::new(
                "Failed to convert PI constant to target type".to_string(),
            ))
        })?;
        let one_eighty = T::from_f64(180.0).ok_or_else(|| {
            CoreError::ValueError(ErrorContext::new(
                "Failed to convert 180.0 to target type".to_string(),
            ))
        })?;
        Ok(*self * pi / one_eighty)
    }

    fn to_degrees(&self) -> CoreResult<Self> {
        let pi = T::from_f64(std::f64::consts::PI).ok_or_else(|| {
            CoreError::ValueError(ErrorContext::new(
                "Failed to convert PI constant to target type".to_string(),
            ))
        })?;
        let one_eighty = T::from_f64(180.0).ok_or_else(|| {
            CoreError::ValueError(ErrorContext::new(
                "Failed to convert 180.0 to target type".to_string(),
            ))
        })?;
        Ok(*self * one_eighty / pi)
    }
}

/// Type-safe representation of a unitless quantity
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Scalar<T: ScientificNumber>(pub T);

impl<T: ScientificNumber> Scalar<T> {
    /// Create a new scalar
    pub fn new(value: T) -> Self {
        Scalar(value)
    }

    /// Get the underlying value
    pub fn value(&self) -> T {
        self.0
    }
}

impl<T: ScientificNumber> From<T> for Scalar<T> {
    fn from(value: T) -> Self {
        Scalar(value)
    }
}

impl<T: ScientificNumber> Add for Scalar<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Scalar(self.0 + other.0)
    }
}

impl<T: ScientificNumber> Sub for Scalar<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Scalar(self.0 - other.0)
    }
}

impl<T: ScientificNumber> Mul for Scalar<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        Scalar(self.0 * other.0)
    }
}

impl<T: ScientificNumber> Div for Scalar<T> {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        Scalar(self.0 / other.0)
    }
}

impl<T: ScientificNumber + Neg<Output = T>> Neg for Scalar<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Scalar(self.0.neg())
    }
}

/// Automated precision tracking for numerical computations
pub mod precision_tracking;

/// Specialized numeric types for scientific domains
pub mod scientific_types;

/// Arbitrary precision numerical computation support
#[cfg(feature = "arbitrary-precision")]
pub mod arbitrary_precision;

/// Numerical stability improvements
pub mod stability;

/// Stable numerical algorithms
pub mod stable_algorithms;

/// Advanced-optimized SIMD operations for numerical computations
///
/// This module provides vectorized implementations of common mathematical operations
/// using the highest available SIMD instruction sets for maximum performance.
pub mod advanced_simd {
    #[allow(unused_imports)]
    use super::*;

    #[cfg(target_arch = "aarch64")]
    use std::arch::aarch64::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    /// Optimized vectorized addition for f32 arrays
    #[inline]
    pub fn add_f32_advanced(a: &[f32], b: &[f32], result: &mut [f32]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), result.len());

        let len = a.len();
        let mut i = 0;

        // AVX2 path - process 8 elements at a time
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    while i + 8 <= len {
                        let va = _mm256_loadu_ps(a.as_ptr().add(i));
                        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                        let vr = _mm256_add_ps(va, vb);
                        _mm256_storeu_ps(result.as_mut_ptr().add(i), vr);
                        i += 8;
                    }
                }
            }
            // SSE path - process 4 elements at a time
            else if is_x86_feature_detected!("sse") {
                unsafe {
                    while i + 4 <= len {
                        let va = _mm_loadu_ps(a.as_ptr().add(i));
                        let vb = _mm_loadu_ps(b.as_ptr().add(i));
                        let vr = _mm_add_ps(va, vb);
                        _mm_storeu_ps(result.as_mut_ptr().add(i), vr);
                        i += 4;
                    }
                }
            }
        }

        // ARM NEON path - process 4 elements at a time
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                unsafe {
                    while i + 4 <= len {
                        let va = vld1q_f32(a.as_ptr().add(i));
                        let vb = vld1q_f32(b.as_ptr().add(i));
                        let vr = vaddq_f32(va, vb);
                        vst1q_f32(result.as_mut_ptr().add(i), vr);
                        i += 4;
                    }
                }
            }
        }

        // Scalar fallback for remaining elements
        while i < len {
            result[i] = a[i] + b[i];
            i += 1;
        }
    }

    /// Optimized vectorized multiplication for f32 arrays
    #[inline]
    pub fn mul_f32_advanced(a: &[f32], b: &[f32], result: &mut [f32]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), result.len());

        let len = a.len();
        let mut i = 0;

        // AVX2 + FMA path for maximum performance
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe {
                    while i + 8 <= len {
                        let va = _mm256_loadu_ps(a.as_ptr().add(i));
                        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                        let vr = _mm256_mul_ps(va, vb);
                        _mm256_storeu_ps(result.as_mut_ptr().add(i), vr);
                        i += 8;
                    }
                }
            } else if is_x86_feature_detected!("sse") {
                unsafe {
                    while i + 4 <= len {
                        let va = _mm_loadu_ps(a.as_ptr().add(i));
                        let vb = _mm_loadu_ps(b.as_ptr().add(i));
                        let vr = _mm_mul_ps(va, vb);
                        _mm_storeu_ps(result.as_mut_ptr().add(i), vr);
                        i += 4;
                    }
                }
            }
        }

        // ARM NEON path
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                unsafe {
                    while i + 4 <= len {
                        let va = vld1q_f32(a.as_ptr().add(i));
                        let vb = vld1q_f32(b.as_ptr().add(i));
                        let vr = vmulq_f32(va, vb);
                        vst1q_f32(result.as_mut_ptr().add(i), vr);
                        i += 4;
                    }
                }
            }
        }

        // Scalar fallback
        while i < len {
            result[i] = a[i] * b[i];
            i += 1;
        }
    }

    /// Optimized fused multiply-add (a * b + c) for f32 arrays
    #[inline]
    pub fn fma_f32_advanced(a: &[f32], b: &[f32], c: &[f32], result: &mut [f32]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), c.len());
        debug_assert_eq!(a.len(), result.len());

        let len = a.len();
        let mut i = 0;

        // AVX2 + FMA path for optimal performance
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe {
                    while i + 8 <= len {
                        let va = _mm256_loadu_ps(a.as_ptr().add(i));
                        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                        let vc = _mm256_loadu_ps(c.as_ptr().add(i));
                        let vr = _mm256_fmadd_ps(va, vb, vc);
                        _mm256_storeu_ps(result.as_mut_ptr().add(i), vr);
                        i += 8;
                    }
                }
            } else if is_x86_feature_detected!("sse") {
                unsafe {
                    while i + 4 <= len {
                        let va = _mm_loadu_ps(a.as_ptr().add(i));
                        let vb = _mm_loadu_ps(b.as_ptr().add(i));
                        let vc = _mm_loadu_ps(c.as_ptr().add(i));
                        let vr = _mm_add_ps(_mm_mul_ps(va, vb), vc);
                        _mm_storeu_ps(result.as_mut_ptr().add(i), vr);
                        i += 4;
                    }
                }
            }
        }

        // ARM NEON path with FMA
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                unsafe {
                    while i + 4 <= len {
                        let va = vld1q_f32(a.as_ptr().add(i));
                        let vb = vld1q_f32(b.as_ptr().add(i));
                        let vc = vld1q_f32(c.as_ptr().add(i));
                        let vr = vfmaq_f32(vc, va, vb);
                        vst1q_f32(result.as_mut_ptr().add(i), vr);
                        i += 4;
                    }
                }
            }
        }

        // Scalar fallback
        while i < len {
            result[i] = a[i] * b[i] + c[0];
            i += 1;
        }
    }

    /// Optimized vectorized dot product for f32 arrays
    #[inline]
    pub fn dot_product_f32_advanced(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        let len = a.len();
        let mut i = 0;
        let mut sum = 0.0f32;

        // AVX2 + FMA path for maximum throughput
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe {
                    let mut acc = _mm256_setzero_ps();
                    while i + 8 <= len {
                        let va = _mm256_loadu_ps(a.as_ptr().add(i));
                        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                        acc = _mm256_fmadd_ps(va, vb, acc);
                        i += 8;
                    }
                    // Horizontal sum of 8 floats
                    let hi = _mm256_extractf128_ps(acc, 1);
                    let lo = _mm256_castps256_ps128(acc);
                    let sum4 = _mm_add_ps(hi, lo);
                    let sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
                    let sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 1));
                    sum = _mm_cvtss_f32(sum1);
                }
            } else if is_x86_feature_detected!("sse") {
                unsafe {
                    let mut acc = _mm_setzero_ps();
                    while i + 4 <= len {
                        let va = _mm_loadu_ps(a.as_ptr().add(i));
                        let vb = _mm_loadu_ps(b.as_ptr().add(i));
                        acc = _mm_add_ps(acc, _mm_mul_ps(va, vb));
                        i += 4;
                    }
                    // Horizontal sum of 4 floats
                    let sum2 = _mm_add_ps(acc, _mm_movehl_ps(acc, acc));
                    let sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 1));
                    sum = _mm_cvtss_f32(sum1);
                }
            }
        }

        // ARM NEON path with accumulation
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                unsafe {
                    let mut acc = vdupq_n_f32(0.0);
                    while i + 4 <= len {
                        let va = vld1q_f32(a.as_ptr().add(i));
                        let vb = vld1q_f32(b.as_ptr().add(i));
                        acc = vfmaq_f32(acc, va, vb);
                        i += 4;
                    }
                    // Horizontal sum
                    sum = vaddvq_f32(acc);
                }
            }
        }

        // Scalar accumulation for remaining elements
        while i < len {
            sum += a[i] * b[i];
            i += 1;
        }

        sum
    }

    /// Optimized vectorized sum reduction for f32 arrays
    #[inline]
    pub fn sum_f32_advanced(data: &[f32]) -> f32 {
        let len = data.len();
        let mut i = 0;
        let mut sum = 0.0f32;

        // AVX2 path
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    let mut acc = _mm256_setzero_ps();
                    while i + 8 <= len {
                        let v = _mm256_loadu_ps(data.as_ptr().add(i));
                        acc = _mm256_add_ps(acc, v);
                        i += 8;
                    }
                    // Horizontal sum
                    let hi = _mm256_extractf128_ps(acc, 1);
                    let lo = _mm256_castps256_ps128(acc);
                    let sum4 = _mm_add_ps(hi, lo);
                    let sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
                    let sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 1));
                    sum = _mm_cvtss_f32(sum1);
                }
            } else if is_x86_feature_detected!("sse") {
                unsafe {
                    let mut acc = _mm_setzero_ps();
                    while i + 4 <= len {
                        let v = _mm_loadu_ps(data.as_ptr().add(i));
                        acc = _mm_add_ps(acc, v);
                        i += 4;
                    }
                    let sum2 = _mm_add_ps(acc, _mm_movehl_ps(acc, acc));
                    let sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 1));
                    sum = _mm_cvtss_f32(sum1);
                }
            }
        }

        // ARM NEON path
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                unsafe {
                    let mut acc = vdupq_n_f32(0.0);
                    while i + 4 <= len {
                        let v = vld1q_f32(data.as_ptr().add(i));
                        acc = vaddq_f32(acc, v);
                        i += 4;
                    }
                    sum = vaddvq_f32(acc);
                }
            }
        }

        // Scalar accumulation
        while i < len {
            sum += data[i];
            i += 1;
        }

        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scientific_number_f32() {
        let a: f32 = 3.0;
        let b: f32 = 4.0;

        assert_eq!(a.abs(), 3.0);
        assert_eq!(a.sqrt(), (3.0_f32).sqrt());
        assert_eq!(a.square(), 9.0);
        assert_eq!(a.max(b), 4.0);
        assert_eq!(a.min(b), 3.0);
        assert!(a.is_finite());
        assert_eq!(a.to_f64(), Some(3.0));
        assert_eq!(f32::from_f64(3.5), Some(3.5));
    }

    #[test]
    fn test_scientific_number_f64() {
        let a: f64 = 3.0;
        let b: f64 = 4.0;

        assert_eq!(a.abs(), 3.0);
        assert_eq!(a.sqrt(), (3.0_f64).sqrt());
        assert_eq!(a.square(), 9.0);
        assert_eq!(a.max(b), 4.0);
        assert_eq!(a.min(b), 3.0);
        assert!(a.is_finite());
        assert_eq!(a.to_f64(), Some(3.0));
        assert_eq!(f64::from_f64(3.5), Some(3.5));
    }

    #[test]
    fn test_real_number_f32() {
        let a: f32 = 3.0;

        assert_eq!(<f32 as RealNumber>::epsilon(), f32::EPSILON);
        assert_eq!(a.exp(), (3.0_f32).exp());
        assert_eq!(a.ln(), (3.0_f32).ln());
        assert_eq!(a.sin(), (3.0_f32).sin());
        assert_eq!(a.cos(), (3.0_f32).cos());
        assert_eq!(a.powf(2.0), 9.0);
    }

    #[test]
    fn test_real_number_f64() {
        let a: f64 = 3.0;

        assert_eq!(<f64 as RealNumber>::epsilon(), f64::EPSILON);
        assert_eq!(a.exp(), (3.0_f64).exp());
        assert_eq!(a.ln(), (3.0_f64).ln());
        assert_eq!(a.sin(), (3.0_f64).sin());
        assert_eq!(a.cos(), (3.0_f64).cos());
        assert_eq!(a.powf(2.0), 9.0);
    }

    #[test]
    fn test_scientific_integer_i32() {
        let a: i32 = 12;
        let b: i32 = 8;

        assert_eq!(a.gcd(b), 4);
        assert_eq!(a.lcm(b), 24);
        assert!(!a.is_prime());
        assert!(11_i32.is_prime());
        assert!(a.is_even());
        assert!(!a.is_odd());
        assert_eq!(a.mod_pow(2, 10), 4); // 12^2 mod 10 = 4
        assert_eq!(5_i32.factorial().unwrap(), 120);
        assert_eq!(5_i32.binomial(2), 10); // 5 choose 2 = 10
    }

    #[test]
    fn test_numeric_conversion() {
        let a: f64 = 3.5;
        let b: i32 = a.try_convert().expect("3.5 should convert to i32 as 3");
        assert_eq!(b, 3);

        let c: f32 = 100.5;
        let d: f64 = c.convert();
        assert_eq!(d, 100.5);

        let e: i32 = 100;
        let f: f32 = e.convert();
        assert_eq!(f, 100.0);
    }

    #[test]
    fn test_angle_conversion() {
        let degrees: f64 = 180.0;
        let radians = <f64 as AngleConversion>::to_radians(&degrees).unwrap();
        assert!((radians - std::f64::consts::PI).abs() < 1e-10);

        let radians: f64 = std::f64::consts::PI / 2.0;
        let degrees = <f64 as AngleConversion>::to_degrees(&radians).unwrap();
        assert!((degrees - 90.0).abs() < 1e-10);
    }

    #[test]
    fn test_scalar() {
        let a = Scalar::new(3.0_f64);
        let b = Scalar::new(4.0_f64);

        assert_eq!((a + b).value(), 7.0);
        assert_eq!((a - b).value(), -1.0);
        assert_eq!((a * b).value(), 12.0);
        assert_eq!((a / b).value(), 0.75);
        assert_eq!((-a).value(), -3.0);

        let c: Scalar<f64> = 5.0.into();
        assert_eq!(c.value(), 5.0);
    }
}
