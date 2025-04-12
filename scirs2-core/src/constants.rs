//! Physical and mathematical constants
//!
//! This module provides commonly used constants for scientific computing.

/// Mathematical constants
pub mod math {
    /// Pi (π)
    pub const PI: f64 = std::f64::consts::PI;

    /// Euler's number (e)
    pub const E: f64 = std::f64::consts::E;

    /// Euler-Mascheroni constant (γ)
    pub const EULER: f64 = 0.577_215_664_901_532_9;

    /// Golden ratio (φ)
    pub const GOLDEN: f64 = 1.618_033_988_749_895;

    /// Square root of 2
    pub const SQRT2: f64 = std::f64::consts::SQRT_2;

    /// Square root of π
    pub const SQRTPI: f64 = 1.772_453_850_905_516;

    /// Natural logarithm of 2
    pub const LN2: f64 = std::f64::consts::LN_2;

    /// Natural logarithm of 10
    pub const LN10: f64 = std::f64::consts::LN_10;
}

/// Physical constants
pub mod physical {
    /// Speed of light in vacuum (m/s)
    pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;

    /// Gravitational constant (m^3 kg^-1 s^-2)
    pub const GRAVITATIONAL_CONSTANT: f64 = 6.674_30e-11;

    /// Planck constant (J s)
    pub const PLANCK: f64 = 6.626_070_15e-34;

    /// Reduced Planck constant (J s)
    pub const HBAR: f64 = 1.054_571_817e-34;

    /// Elementary charge (C)
    pub const ELEMENTARY_CHARGE: f64 = 1.602_176_634e-19;

    /// Electron mass (kg)
    pub const ELECTRON_MASS: f64 = 9.109_383_701_5e-31;

    /// Proton mass (kg)
    pub const PROTON_MASS: f64 = 1.672_621_923_69e-27;

    /// Fine-structure constant
    pub const FINE_STRUCTURE: f64 = 7.297_352_569_3e-3;

    /// Rydberg constant (1/m)
    pub const RYDBERG: f64 = 10_973_731.568_160;

    /// Avogadro constant (1/mol)
    pub const AVOGADRO: f64 = 6.022_140_76e23;

    /// Gas constant (J/(mol K))
    pub const GAS_CONSTANT: f64 = 8.314_462_618_153_24;

    /// Boltzmann constant (J/K)
    pub const BOLTZMANN: f64 = 1.380_649e-23;

    /// Stefan-Boltzmann constant (W/(m^2 K^4))
    pub const STEFAN_BOLTZMANN: f64 = 5.670_374_419e-8;

    /// Electron volt (J)
    pub const ELECTRON_VOLT: f64 = 1.602_176_634e-19;

    /// Astronomical unit (m)
    pub const ASTRONOMICAL_UNIT: f64 = 1.495_978_707e11;

    /// Light year (m)
    pub const LIGHT_YEAR: f64 = 9.460_730_472_580_8e15;

    /// Parsec (m)
    pub const PARSEC: f64 = 3.085_677_581_49e16;
}

/// Unit conversions
pub mod conversions {
    /// Degrees to radians conversion factor
    pub const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0;

    /// Radians to degrees conversion factor
    pub const RAD_TO_DEG: f64 = 180.0 / std::f64::consts::PI;

    /// Inches to meters
    pub const INCH_TO_METER: f64 = 0.0254;

    /// Feet to meters
    pub const FOOT_TO_METER: f64 = 0.3048;

    /// Miles to meters
    pub const MILE_TO_METER: f64 = 1609.344;

    /// Pounds to kilograms
    pub const POUND_TO_KG: f64 = 0.45359237;

    /// Gallons (US) to cubic meters
    pub const GALLON_TO_CUBIC_METER: f64 = 0.003_785_411_784;
}
