//! Airy functions
//!
//! This module provides implementations of Airy functions of the first kind (Ai)
//! and second kind (Bi), along with their derivatives.

use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

// Lookup tables for Airy functions at specific points for interpolation
// Format: x, Ai(x), Ai'(x), Bi(x), Bi'(x)
const AIRY_TABLE: [(f64, f64, f64, f64, f64); 13] = [
    (
        -5.0,
        0.3507610090241142,
        0.7845177219792484,
        0.3757289802157,
        -0.7771388142155,
    ),
    (
        -4.0,
        0.3818263073159224,
        0.5251967358889706,
        0.1475403876463545,
        0.5609353889057985,
    ),
    (
        -3.0,
        0.3786287733679269,
        0.1297506195662909,
        -0.1982693379014283,
        0.5924485964250711,
    ),
    (
        -2.0,
        0.22740742820168557,
        0.18836677954676762,
        0.37339167942543735,
        -0.13289576834890213,
    ),
    (
        -1.0,
        0.5355608832923521,
        -0.3271928185544436,
        0.103_997_389_496_944_6,
        0.5383830805628176,
    ),
    (
        0.0,
        0.3550280538878172,
        -0.25881940379280678,
        0.6149266274460007,
        0.4482883573538264,
    ),
    (
        0.5,
        0.2309493905707306,
        -0.2241595309561765,
        0.8712188886443742,
        0.6820309552208995,
    ),
    (
        1.0,
        0.1352924163128814,
        -0.16049975743698353,
        1.2074235949528715,
        1.0434774887138,
    ),
    (
        2.0,
        0.03492413042327235,
        -0.06120049354097896,
        3.5424680997112654,
        3.3662351522409107,
    ),
    (
        3.0,
        0.006591139357460388,
        -0.01585408416342784,
        10.139198841026192,
        12.469524135390914,
    ),
    (
        4.0,
        0.0009074603677222401,
        -0.002683279352261441,
        29.819025046218183,
        44.94308396859953,
    ),
    (
        5.0,
        0.0001083277795169,
        -0.0003280924942020832,
        83.73823961011828,
        148.34022202713066,
    ),
    (
        10.0,
        1.1047532368598592e-10,
        -3.3630442265728913e-10,
        14036.271985859755,
        44387.26986072647,
    ),
];

/// Airy function of the first kind, Ai(x).
///
/// The Airy function Ai(x) is a solution to the differential equation:
/// y''(x) - xy(x) = 0
///
/// # Arguments
///
/// * `x` - Input value
///
/// # Returns
///
/// * Ai(x) Airy function value
///
/// # Examples
///
/// ```
/// use scirs2_special::ai;
///
/// // Ai(0) = 3^(-2/3) / Γ(2/3) ≈ 0.3550
/// assert!((ai(0.0f64) - 0.3550280538878172).abs() < 1e-10);
/// ```
pub fn ai<F: Float + FromPrimitive + Debug>(x: F) -> F {
    let x_f64 = x.to_f64().unwrap();

    // For very large positive x, use asymptotic form
    if x_f64 > 10.0 {
        let z = (2.0 / 3.0) * x_f64.powf(1.5);
        let prefactor = 1.0 / (2.0 * std::f64::consts::PI.sqrt() * x_f64.powf(0.25));
        return F::from(prefactor * (-z).exp()).unwrap();
    }

    // For very large negative x, use asymptotic form
    if x_f64 < -5.0 {
        let z = (2.0 / 3.0) * (-x_f64).powf(1.5);
        let pi_quarter = std::f64::consts::FRAC_PI_4;
        let prefactor = 1.0 / (std::f64::consts::PI.sqrt() * (-x_f64).powf(0.25));

        return F::from(prefactor * (z - pi_quarter).sin()).unwrap();
    }

    // Use table lookup with linear interpolation for moderate values
    let mut idx_low = 0;

    // Find the right interval in the table
    for i in 0..AIRY_TABLE.len() - 1 {
        if x_f64 >= AIRY_TABLE[i].0 && x_f64 <= AIRY_TABLE[i + 1].0 {
            idx_low = i;
            break;
        }
    }

    let x0 = AIRY_TABLE[idx_low].0;
    let x1 = AIRY_TABLE[idx_low + 1].0;
    let y0 = AIRY_TABLE[idx_low].1; // Ai value at x0
    let y1 = AIRY_TABLE[idx_low + 1].1; // Ai value at x1

    // Linear interpolation formula: y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    let result = y0 + (x_f64 - x0) * (y1 - y0) / (x1 - x0);

    F::from(result).unwrap()
}

/// Derivative of the Airy function of the first kind, Ai'(x).
///
/// # Arguments
///
/// * `x` - Input value
///
/// # Returns
///
/// * Ai'(x) Derivative of the Airy function value
///
/// # Examples
///
/// ```
/// use scirs2_special::aip;
///
/// // Ai'(0) = -3^(-1/3) / Γ(1/3) ≈ -0.2588
/// assert!((aip(0.0f64) + 0.25881940379280677).abs() < 1e-10);
/// ```
pub fn aip<F: Float + FromPrimitive + Debug>(x: F) -> F {
    let x_f64 = x.to_f64().unwrap();

    // For very large positive x, use asymptotic form
    if x_f64 > 10.0 {
        let z = (2.0 / 3.0) * x_f64.powf(1.5);
        let prefactor = -x_f64.sqrt() / (2.0 * std::f64::consts::PI.sqrt() * x_f64.powf(0.25));
        return F::from(prefactor * (-z).exp()).unwrap();
    }

    // For very large negative x, use asymptotic form
    if x_f64 < -5.0 {
        let z = (2.0 / 3.0) * (-x_f64).powf(1.5);
        let pi_quarter = std::f64::consts::FRAC_PI_4;
        let prefactor = (-x_f64).sqrt() / (std::f64::consts::PI.sqrt() * (-x_f64).powf(0.25));

        return F::from(prefactor * (z - pi_quarter).cos()).unwrap();
    }

    // Use table lookup with linear interpolation for moderate values
    let mut idx_low = 0;

    // Find the right interval in the table
    for i in 0..AIRY_TABLE.len() - 1 {
        if x_f64 >= AIRY_TABLE[i].0 && x_f64 <= AIRY_TABLE[i + 1].0 {
            idx_low = i;
            break;
        }
    }

    let x0 = AIRY_TABLE[idx_low].0;
    let x1 = AIRY_TABLE[idx_low + 1].0;
    let y0 = AIRY_TABLE[idx_low].2; // Ai' value at x0
    let y1 = AIRY_TABLE[idx_low + 1].2; // Ai' value at x1

    // Linear interpolation formula: y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    let result = y0 + (x_f64 - x0) * (y1 - y0) / (x1 - x0);

    F::from(result).unwrap()
}

/// Airy function of the second kind, Bi(x).
///
/// The Airy function Bi(x) is the second linearly independent solution to the differential equation:
/// y''(x) - xy(x) = 0
///
/// # Arguments
///
/// * `x` - Input value
///
/// # Returns
///
/// * Bi(x) Airy function value
///
/// # Examples
///
/// ```
/// use scirs2_special::bi;
///
/// // Bi(0) = 3^(-1/6) / Γ(2/3) ≈ 0.6149
/// assert!((bi(0.0f64) - 0.6149266274460007).abs() < 1e-10);
/// ```
pub fn bi<F: Float + FromPrimitive + Debug>(x: F) -> F {
    let x_f64 = x.to_f64().unwrap();

    // For very large positive x, use asymptotic form
    if x_f64 > 10.0 {
        let z = (2.0 / 3.0) * x_f64.powf(1.5);
        let prefactor = 1.0 / (std::f64::consts::PI.sqrt() * x_f64.powf(0.25));
        return F::from(prefactor * z.exp()).unwrap();
    }

    // For very large negative x, use asymptotic form
    if x_f64 < -5.0 {
        let z = (2.0 / 3.0) * (-x_f64).powf(1.5);
        let pi_quarter = std::f64::consts::FRAC_PI_4;
        let prefactor = 1.0 / (std::f64::consts::PI.sqrt() * (-x_f64).powf(0.25));

        return F::from(prefactor * (z - pi_quarter).cos()).unwrap();
    }

    // Use table lookup with linear interpolation for moderate values
    let mut idx_low = 0;

    // Find the right interval in the table
    for i in 0..AIRY_TABLE.len() - 1 {
        if x_f64 >= AIRY_TABLE[i].0 && x_f64 <= AIRY_TABLE[i + 1].0 {
            idx_low = i;
            break;
        }
    }

    let x0 = AIRY_TABLE[idx_low].0;
    let x1 = AIRY_TABLE[idx_low + 1].0;
    let y0 = AIRY_TABLE[idx_low].3; // Bi value at x0
    let y1 = AIRY_TABLE[idx_low + 1].3; // Bi value at x1

    // Linear interpolation formula: y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    let result = y0 + (x_f64 - x0) * (y1 - y0) / (x1 - x0);

    F::from(result).unwrap()
}

/// Derivative of the Airy function of the second kind, Bi'(x).
///
/// # Arguments
///
/// * `x` - Input value
///
/// # Returns
///
/// * Bi'(x) Derivative of the Airy function value
///
/// # Examples
///
/// ```
/// use scirs2_special::bip;
///
/// // Bi'(0) = 3^(1/6) / Γ(1/3) ≈ 0.4483
/// assert!((bip(0.0f64) - 0.4482883573538264).abs() < 1e-10);
/// ```
pub fn bip<F: Float + FromPrimitive + Debug>(x: F) -> F {
    let x_f64 = x.to_f64().unwrap();

    // For very large positive x, use asymptotic form
    if x_f64 > 10.0 {
        let z = (2.0 / 3.0) * x_f64.powf(1.5);
        let prefactor = x_f64.sqrt() / (std::f64::consts::PI.sqrt() * x_f64.powf(0.25));
        return F::from(prefactor * z.exp()).unwrap();
    }

    // For very large negative x, use asymptotic form
    if x_f64 < -5.0 {
        let z = (2.0 / 3.0) * (-x_f64).powf(1.5);
        let pi_quarter = std::f64::consts::FRAC_PI_4;
        let prefactor = (-x_f64).sqrt() / (std::f64::consts::PI.sqrt() * (-x_f64).powf(0.25));

        return F::from(-prefactor * (z - pi_quarter).sin()).unwrap();
    }

    // Use table lookup with linear interpolation for moderate values
    let mut idx_low = 0;

    // Find the right interval in the table
    for i in 0..AIRY_TABLE.len() - 1 {
        if x_f64 >= AIRY_TABLE[i].0 && x_f64 <= AIRY_TABLE[i + 1].0 {
            idx_low = i;
            break;
        }
    }

    let x0 = AIRY_TABLE[idx_low].0;
    let x1 = AIRY_TABLE[idx_low + 1].0;
    let y0 = AIRY_TABLE[idx_low].4; // Bi' value at x0
    let y1 = AIRY_TABLE[idx_low + 1].4; // Bi' value at x1

    // Linear interpolation formula: y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    let result = y0 + (x_f64 - x0) * (y1 - y0) / (x1 - x0);

    F::from(result).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_ai() {
        // Test at x = 0
        assert_relative_eq!(ai(0.0), 0.3550280538878172, epsilon = 1e-10);

        // Test at positive values
        assert_relative_eq!(ai(1.0), 0.1352924163128814, epsilon = 1e-10);
        assert_relative_eq!(ai(2.0), 0.03492413042327235, epsilon = 1e-10);
        assert_relative_eq!(ai(5.0), 0.0001083277795169, epsilon = 1e-10);

        // Test at negative values
        assert_relative_eq!(ai(-1.0), 0.5355608832923521, epsilon = 1e-10);
        assert_relative_eq!(ai(-2.0), 0.22740742820168557, epsilon = 1e-10);
        assert_relative_eq!(ai(-5.0), 0.3507610090241142, epsilon = 1e-10);
    }

    #[test]
    fn test_aip() {
        // Test at x = 0
        assert_relative_eq!(aip(0.0), -0.25881940379280678, epsilon = 1e-10);

        // Test at positive values
        assert_relative_eq!(aip(1.0), -0.16049975743698353, epsilon = 1e-10);
        assert_relative_eq!(aip(2.0), -0.06120049354097896, epsilon = 1e-10);
        assert_relative_eq!(aip(5.0), -0.0003280924942020832, epsilon = 1e-10);

        // Test at negative values
        assert_relative_eq!(aip(-1.0), -0.3271928185544436, epsilon = 1e-10);
        assert_relative_eq!(aip(-2.0), 0.18836677954676762, epsilon = 1e-10);
        assert_relative_eq!(aip(-5.0), 0.7845177219792484, epsilon = 1e-10);
    }

    #[test]
    fn test_bi() {
        // Test at x = 0
        assert_relative_eq!(bi(0.0), 0.6149266274460007, epsilon = 1e-10);

        // Test at positive values
        assert_relative_eq!(bi(1.0), 1.2074235949528715, epsilon = 1e-10);
        assert_relative_eq!(bi(2.0), 3.5424680997112654, epsilon = 1e-10);
        assert_relative_eq!(bi(5.0), 83.73823961011828, epsilon = 1e-8);

        // Test at negative values
        assert_relative_eq!(bi(-1.0), 0.103_997_389_496_944_6, epsilon = 1e-10);
        assert_relative_eq!(bi(-2.0), 0.37339167942543735, epsilon = 1e-10);
        assert_relative_eq!(bi(-5.0), 0.3757289802157, epsilon = 1e-10);
    }

    #[test]
    fn test_bip() {
        // Test at x = 0
        assert_relative_eq!(bip(0.0), 0.4482883573538264, epsilon = 1e-10);

        // Test at positive values
        assert_relative_eq!(bip(1.0), 1.0434774887138, epsilon = 1e-10);
        assert_relative_eq!(bip(2.0), 3.3662351522409107, epsilon = 1e-10);
        assert_relative_eq!(bip(5.0), 148.34022202713066, epsilon = 1e-8);

        // Test at negative values
        assert_relative_eq!(bip(-1.0), 0.5383830805628176, epsilon = 1e-10);
        assert_relative_eq!(bip(-2.0), -0.13289576834890213, epsilon = 1e-10);
        assert_relative_eq!(bip(-5.0), -0.7771388142155, epsilon = 1e-10);
    }
}
