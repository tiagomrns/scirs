//! # Type Conversions
//!
//! This module provides robust numeric type conversions and complex number interoperability.
//!
//! ## Features
//!
//! * Robust numeric type conversions with error handling
//! * Complex number interoperability
//! * Type-safe conversion traits
//! * Support for scientific computing types
//!
//! ## Usage
//!
//! ```rust,no_run
//! use scirs2_core::types::{NumericConversion, ComplexOps, ComplexExt};
//! use num_complex::Complex64;
//!
//! // Convert between numeric types with error handling
//! let x: f64 = 100.5;
//! let y: i32 = x.to_numeric().unwrap();
//! assert_eq!(y, 100);
//!
//! // Convert with boundary checking
//! let large: f64 = 1e20;
//! let result: Result<i32, _> = large.to_numeric();
//! assert!(result.is_err()); // Error: value out of range
//!
//! // Complex number operations
//! let z1 = Complex64::new(1.0, 2.0);
//! let z2 = Complex64::new(3.0, 4.0);
//!
//! // Use additional complex number operations
//! let magnitude = z1.magnitude();
//! let phase = z1.phase();
//! let distance = z1.distance(z2);
//!
//! // Convert between complex types
//! let z_f32: num_complex::Complex32 = z1.convert_complex().unwrap();
//! ```

use num_complex::{Complex, Complex32, Complex64};
use num_traits::{Bounded, Float, NumCast, Zero};
use std::fmt;
use thiserror::Error;

/// Error type for numeric conversions
#[derive(Error, Debug, Clone, PartialEq)]
pub enum NumericConversionError {
    /// Value is too large for target type
    #[error("Value {value} is too large for the target type (max: {max})")]
    Overflow {
        /// The value that caused the overflow
        value: String,
        /// The maximum value of the target type
        max: String,
    },

    /// Value is too small for target type
    #[error("Value {value} is too small for the target type (min: {min})")]
    Underflow {
        /// The value that caused the underflow
        value: String,
        /// The minimum value of the target type
        min: String,
    },

    /// Value cannot be represented in target type
    #[error("Value {value} cannot be represented in the target type")]
    NotRepresentable {
        /// The value that cannot be represented
        value: String,
    },

    /// Precision loss during conversion
    #[error("Precision loss when converting {value} to target type")]
    PrecisionLoss {
        /// The value that would lose precision
        value: String,
    },

    /// NaN or infinite value cannot be converted
    #[error("NaN or infinite value cannot be converted to the target type")]
    NanOrInfinite,

    /// Generic conversion error
    #[error("Failed to convert value: {0}")]
    Other(String),
}

/// Trait for checked numeric conversions
pub trait NumericConversion {
    /// Convert to another numeric type with error handling
    fn to_numeric<T>(&self) -> Result<T, NumericConversionError>
    where
        T: Bounded + NumCast + PartialOrd + fmt::Display;

    /// Convert to another numeric type, clamping to the target type's bounds
    fn to_numeric_clamped<T>(&self) -> T
    where
        T: Bounded + NumCast + PartialOrd + num_traits::Zero;

    /// Convert to another numeric type, rounding to the nearest valid value
    fn to_numeric_rounded<T>(&self) -> T
    where
        T: Bounded + NumCast + PartialOrd + num_traits::Zero;

    /// Helper to check whether a value is a floating-point type
    fn is_float_type<T>() -> bool {
        let name = std::any::type_name::<T>();
        name.contains("f32") || name.contains("f64")
    }
}

/// Implementation of NumericConversion for floating-point types
impl<S> NumericConversion for S
where
    S: Copy + NumCast + PartialOrd + fmt::Display,
{
    /// Implementation of is_float_type
    fn is_float_type<T>() -> bool {
        let name = std::any::type_name::<T>();
        name.contains("f32") || name.contains("f64")
    }
    fn to_numeric<T>(&self) -> Result<T, NumericConversionError>
    where
        T: Bounded + NumCast + PartialOrd + fmt::Display,
    {
        let value = *self;

        // Handle NaN and infinite values for floating-point sources
        if let Some(float_val) = NumCast::from(value) {
            let float_val: f64 = float_val;
            if float_val.is_nan() || float_val.is_infinite() {
                return Err(NumericConversionError::NanOrInfinite);
            }
        }

        // Try the conversion
        let result = T::from(value);

        match result {
            Some(result) => {
                // Check for integer overflow/underflow when converting from float to int
                if let (Some(float_val), Some(max), Some(min)) = (
                    NumCast::from(value),
                    NumCast::from(T::max_value()),
                    NumCast::from(T::min_value()),
                ) {
                    let float_val: f64 = float_val;
                    let max: f64 = max;
                    let min: f64 = min;

                    if float_val > max {
                        return Err(NumericConversionError::Overflow {
                            value: format!("{}", value),
                            max: format!("{}", T::max_value()),
                        });
                    }

                    if float_val < min {
                        return Err(NumericConversionError::Underflow {
                            value: format!("{}", value),
                            min: format!("{}", T::min_value()),
                        });
                    }

                    // Check for precision loss when converting float to int
                    if float_val.fract() != 0.0 && !Self::is_float_type::<T>() {
                        return Err(NumericConversionError::PrecisionLoss {
                            value: format!("{}", value),
                        });
                    }
                }

                Ok(result)
            }
            None => {
                // Handle conversion failure
                if let Some(max_s) = NumCast::from(T::max_value()) {
                    let max_s: S = max_s;
                    if value > max_s {
                        return Err(NumericConversionError::Overflow {
                            value: format!("{}", value),
                            max: format!("{}", T::max_value()),
                        });
                    }
                }

                if let Some(min_s) = NumCast::from(T::min_value()) {
                    let min_s: S = min_s;
                    if value < min_s {
                        return Err(NumericConversionError::Underflow {
                            value: format!("{}", value),
                            min: format!("{}", T::min_value()),
                        });
                    }
                }

                Err(NumericConversionError::NotRepresentable {
                    value: format!("{}", value),
                })
            }
        }
    }

    fn to_numeric_clamped<T>(&self) -> T
    where
        T: Bounded + NumCast + PartialOrd + num_traits::Zero,
    {
        let value = *self;

        // Try the conversion first
        if let Some(result) = T::from(value) {
            return result;
        }

        // If conversion failed, clamp to bounds
        if let Some(max_s) = NumCast::from(T::max_value()) {
            let max_s: S = max_s;
            if value > max_s {
                return T::max_value();
            }
        }

        if let Some(min_s) = NumCast::from(T::min_value()) {
            let min_s: S = min_s;
            if value < min_s {
                return T::min_value();
            }
        }

        // Fallback to zero if unable to determine bounds
        // We know T implements Zero from the trait bound, so we can call this safely
        <T as num_traits::Zero>::zero()
    }

    fn to_numeric_rounded<T>(&self) -> T
    where
        T: Bounded + NumCast + PartialOrd + num_traits::Zero,
    {
        let value = *self;

        // For floating-point sources, round to nearest integer before conversion
        if let Some(float_val) = NumCast::from(value) {
            let float_val: f64 = float_val;
            let rounded = float_val.round();

            if let Some(result) = T::from(rounded) {
                return result;
            }
        }

        // If rounding failed, fall back to same logic as in to_numeric_clamped
        // But implement directly to avoid dependency issues
        if let Some(result) = T::from(value) {
            return result;
        }

        // If conversion failed, clamp to bounds
        if let Some(max_s) = NumCast::from(T::max_value()) {
            let max_s: S = max_s;
            if value > max_s {
                return T::max_value();
            }
        }

        if let Some(min_s) = NumCast::from(T::min_value()) {
            let min_s: S = min_s;
            if value < min_s {
                return T::min_value();
            }
        }

        // Fallback to zero
        <T as num_traits::Zero>::zero()
    }
}

/// Error type for complex number conversions
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ComplexConversionError {
    /// Error in real part conversion
    #[error("Error converting real part: {0}")]
    RealPartError(#[from] NumericConversionError),

    /// Error in imaginary part conversion
    #[error("Error converting imaginary part: {0}")]
    ImaginaryPartError(String),

    /// Generic conversion error
    #[error("Failed to convert complex value: {0}")]
    Other(String),
}

/// Extended operations for complex numbers
pub trait ComplexOps<T>
where
    T: Float,
{
    /// Calculate the magnitude (absolute value) of the complex number
    fn magnitude(&self) -> T;

    /// Calculate the phase (argument) of the complex number
    fn phase(&self) -> T;

    /// Calculate the distance to another complex number
    fn distance(&self, other: Complex<T>) -> T;

    /// Normalize the complex number (make its magnitude 1)
    fn normalize(&self) -> Complex<T>;

    /// Rotate the complex number by the given phase (in radians)
    fn rotate(&self, phase: T) -> Complex<T>;

    /// Calculate the complex conjugate
    fn conjugate(&self) -> Complex<T>;

    /// Convert to polar form (magnitude, phase)
    fn to_polar(&self) -> (T, T);

    /// Create from polar form (magnitude, phase)
    fn from_polar(magnitude: T, phase: T) -> Complex<T>;

    /// Check if the complex number is approximately equal to another
    fn approx_eq(&self, other: Complex<T>, epsilon: T) -> bool;
}

impl<T> ComplexOps<T> for Complex<T>
where
    T: Float,
{
    fn magnitude(&self) -> T {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    fn phase(&self) -> T {
        self.im.atan2(self.re)
    }

    fn distance(&self, other: Complex<T>) -> T {
        let diff = *self - other;
        diff.magnitude()
    }

    fn normalize(&self) -> Complex<T> {
        let mag = self.magnitude();
        if mag.is_zero() {
            return Complex::new(T::zero(), T::zero());
        }
        Complex::new(self.re / mag, self.im / mag)
    }

    fn rotate(&self, phase: T) -> Complex<T> {
        let (mag, current_phase) = self.to_polar();
        Self::from_polar(mag, current_phase + phase)
    }

    fn conjugate(&self) -> Complex<T> {
        Complex::new(self.re, -self.im)
    }

    fn to_polar(&self) -> (T, T) {
        (self.magnitude(), self.phase())
    }

    fn from_polar(magnitude: T, phase: T) -> Complex<T> {
        Complex::new(magnitude * phase.cos(), magnitude * phase.sin())
    }

    fn approx_eq(&self, other: Complex<T>, epsilon: T) -> bool {
        (self.re - other.re).abs() < epsilon && (self.im - other.im).abs() < epsilon
    }
}

/// Extension trait for complex number conversions
pub trait ComplexExt<T>
where
    T: Float,
{
    /// Convert to another complex type with error handling
    fn convert_complex<U>(&self) -> Result<Complex<U>, ComplexConversionError>
    where
        U: Float + Bounded + NumCast + PartialOrd + fmt::Display;

    /// Convert to another complex type, clamping values to bounds
    fn convert_complex_clamped<U>(&self) -> Complex<U>
    where
        U: Float + Bounded + NumCast + PartialOrd;

    /// Convert to string in algebraic form (a+bi)
    fn to_algebraic_string(&self) -> String;

    /// Convert to string in polar form (r∠θ)
    fn to_polar_string(&self) -> String;

    /// Check if this complex number is approximately zero
    fn is_approx_zero(&self, epsilon: T) -> bool;
}

impl<T> ComplexExt<T> for Complex<T>
where
    T: Float + fmt::Display,
{
    fn convert_complex<U>(&self) -> Result<Complex<U>, ComplexConversionError>
    where
        U: Float + Bounded + NumCast + PartialOrd + fmt::Display,
    {
        let real = self
            .re
            .to_numeric()
            .map_err(|e| ComplexConversionError::RealPartError(e))?;

        let imag = self
            .im
            .to_numeric()
            .map_err(|e| ComplexConversionError::ImaginaryPartError(e.to_string()))?;

        Ok(Complex::new(real, imag))
    }

    fn convert_complex_clamped<U>(&self) -> Complex<U>
    where
        U: Float + Bounded + NumCast + PartialOrd,
    {
        let real = self.re.to_numeric_clamped();
        let imag = self.im.to_numeric_clamped();
        Complex::new(real, imag)
    }

    fn to_algebraic_string(&self) -> String {
        if self.im.is_zero() {
            format!("{}", self.re)
        } else if self.im.is_sign_positive() {
            format!("{}+{}i", self.re, self.im)
        } else {
            format!("{}{}i", self.re, self.im)
        }
    }

    fn to_polar_string(&self) -> String {
        let (mag, phase) = self.to_polar();
        format!("{:.4}∠{:.4}rad", mag, phase)
    }

    fn is_approx_zero(&self, epsilon: T) -> bool {
        self.re.abs() < epsilon && self.im.abs() < epsilon
    }
}

/// Helper functions for common type conversions
pub mod convert {
    use super::*;

    /// Convert a slice to a specified numeric type with error checking
    pub fn slice_to_numeric<S, T>(slice: &[S]) -> Result<Vec<T>, NumericConversionError>
    where
        S: Copy + NumCast + PartialOrd + fmt::Display,
        T: Bounded + NumCast + PartialOrd + fmt::Display,
    {
        slice.iter().map(|&x| x.to_numeric()).collect()
    }

    /// Convert a slice to a specified numeric type with clamping
    pub fn slice_to_numeric_clamped<S, T>(slice: &[S]) -> Vec<T>
    where
        S: Copy + NumCast + PartialOrd + NumericConversion,
        T: Bounded + NumCast + PartialOrd + Zero,
    {
        slice.iter().map(|&x| x.to_numeric_clamped()).collect()
    }

    /// Convert a complex slice to another complex type with error checking
    pub fn complex_slice_to_complex<S, T>(
        slice: &[Complex<S>],
    ) -> Result<Vec<Complex<T>>, ComplexConversionError>
    where
        S: Float + fmt::Display,
        T: Float + Bounded + NumCast + PartialOrd + fmt::Display,
    {
        slice.iter().map(|x| x.convert_complex()).collect()
    }

    /// Convert a real slice to a complex slice with zero imaginary part
    pub fn real_to_complex<S, T>(slice: &[S]) -> Result<Vec<Complex<T>>, NumericConversionError>
    where
        S: Copy + NumCast + PartialOrd + fmt::Display,
        T: Float + Bounded + NumCast + PartialOrd + fmt::Display,
    {
        slice
            .iter()
            .map(|&x| x.to_numeric().map(|real| Complex::new(real, T::zero())))
            .collect()
    }

    /// Convert between commonly used complex types
    pub fn complex64_to_complex32(z: Complex64) -> Complex32 {
        Complex32::new(z.re as f32, z.im as f32)
    }

    /// Convert between commonly used complex types
    pub fn complex32_to_complex64(z: Complex32) -> Complex64 {
        Complex64::new(z.re as f64, z.im as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numeric_conversion() {
        // Valid conversions
        assert_eq!(42.5f64.to_numeric::<i32>().unwrap(), 42);
        assert_eq!((-42.0f64).to_numeric::<i32>().unwrap(), -42);

        // Overflow
        assert!(1e20f64.to_numeric::<i32>().is_err());

        // Underflow
        assert!((-1e20f64).to_numeric::<i32>().is_err());

        // NaN and Infinity
        assert!(f64::NAN.to_numeric::<i32>().is_err());
        assert!(f64::INFINITY.to_numeric::<i32>().is_err());

        // Clamping
        assert_eq!(1e20f64.to_numeric_clamped::<i32>(), i32::MAX);
        assert_eq!((-1e20f64).to_numeric_clamped::<i32>(), i32::MIN);
    }

    #[test]
    fn test_complex_operations() {
        let z1 = Complex64::new(3.0, 4.0);

        // Magnitude and phase
        assert_eq!(z1.magnitude(), 5.0);
        assert!((z1.phase() - 0.9272952180016122).abs() < 1e-10);

        // Distance
        let z2 = Complex64::new(0.0, 0.0);
        assert_eq!(z1.distance(z2), 5.0);

        // Normalization
        let z_norm = z1.normalize();
        assert!((z_norm.magnitude() - 1.0).abs() < 1e-10);

        // Rotation
        let z_rot = z1.rotate(std::f64::consts::PI / 2.0);
        assert!((z_rot.re + 4.0).abs() < 1e-10);
        assert!((z_rot.im - 3.0).abs() < 1e-10);

        // Polar form
        let (mag, phase) = z1.to_polar();
        let z_back = Complex64::from_polar(mag, phase);
        assert!((z1.re - z_back.re).abs() < 1e-10);
        assert!((z1.im - z_back.im).abs() < 1e-10);
    }

    #[test]
    fn test_complex_conversion() {
        let z1 = Complex64::new(3.0, 4.0);

        // Valid conversion
        let z2 = z1.convert_complex::<f32>().unwrap();
        assert_eq!(z2.re, 3.0f32);
        assert_eq!(z2.im, 4.0f32);

        // Large value conversion
        let z_large = Complex64::new(1e30, 1e30);
        assert!(z_large.convert_complex::<f32>().is_err());

        // Clamped conversion
        let z_clamped = z_large.convert_complex_clamped::<f32>();
        assert_eq!(z_clamped.re, f32::MAX);
        assert_eq!(z_clamped.im, f32::MAX);
    }

    #[test]
    fn test_string_representation() {
        let z1 = Complex64::new(3.0, 4.0);
        assert_eq!(z1.to_algebraic_string(), "3+4i");

        let z2 = Complex64::new(3.0, -4.0);
        assert_eq!(z2.to_algebraic_string(), "3-4i");

        let z3 = Complex64::new(3.0, 0.0);
        assert_eq!(z3.to_algebraic_string(), "3");
    }
}
