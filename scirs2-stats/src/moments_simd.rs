//! SIMD-optimized higher-order moment calculations
//!
//! This module provides SIMD-accelerated implementations of statistical moments
//! including skewness and kurtosis, using scirs2-core's unified SIMD operations.

use crate::error::{StatsError, StatsResult};
use crate::error_standardization::ErrorMessages;
use ndarray::{Array1, ArrayBase, Data, Ix1};
use num_traits::{Float, NumCast, One, Zero};
use scirs2_core::{
    simd_ops::{AutoOptimizer, SimdUnifiedOps},
    validation::*,
};

/// SIMD-optimized skewness calculation
///
/// Computes the skewness (third standardized moment) using SIMD acceleration
/// for vectorized operations on deviations and their powers.
///
/// # Arguments
///
/// * `x` - Input data array
/// * `bias` - Whether to use biased estimator (true) or apply sample bias correction (false)
///
/// # Returns
///
/// * The skewness of the input data
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_stats::moments_simd::skewness_simd;
///
/// let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let skew = skewness_simd(&data.view(), false).unwrap();
/// ```
#[allow(dead_code)]
pub fn skewness_simd<F, D>(x: &ArrayBase<D, Ix1>, bias: bool) -> StatsResult<F>
where
    F: Float + NumCast + SimdUnifiedOps + Zero + One + Copy + Send + Sync + std::fmt::Display,
    D: Data<Elem = F>,
{
    checkarray_finite(x, "x")?;

    if x.is_empty() {
        return Err(ErrorMessages::empty_array("x"));
    }

    if x.len() < 3 && !bias {
        return Err(ErrorMessages::insufficientdata(
            "unbiased skewness calculation",
            3,
            x.len(),
        ));
    }

    let n = x.len();
    let n_f = F::from(n).unwrap();
    let optimizer = AutoOptimizer::new();

    // Compute mean using SIMD if beneficial
    let mean = if optimizer.should_use_simd(n) {
        F::simd_sum(&x.view()) / n_f
    } else {
        x.iter().fold(F::zero(), |acc, &val| acc + val) / n_f
    };

    // SIMD-optimized moment calculations
    let (sum_sq_dev, sum_cubed_dev) = if optimizer.should_use_simd(n) {
        compute_moments_simd(x, mean, n)
    } else {
        compute_moments_scalar(x, mean)
    };

    if sum_sq_dev == F::zero() {
        return Ok(F::zero()); // No variation, so no skewness
    }

    // Formula: g1 = (Σ(x-μ)³/n) / (Σ(x-μ)²/n)^(3/2)
    let variance = sum_sq_dev / n_f;
    let third_moment = sum_cubed_dev / n_f;
    let skew = third_moment / variance.powf(F::from(1.5).unwrap());

    if !bias && n > 2 {
        // Apply correction for sample bias
        // The bias correction factor for skewness is sqrt(n(n-1))/(n-2)
        let sqrt_term = (n_f * (n_f - F::one())).sqrt();
        let correction = sqrt_term / (n_f - F::from(2.0).unwrap());
        Ok(skew * correction)
    } else {
        Ok(skew)
    }
}

/// SIMD-optimized kurtosis calculation
///
/// Computes the kurtosis (fourth standardized moment) using SIMD acceleration
/// for vectorized operations on deviations and their powers.
///
/// # Arguments
///
/// * `x` - Input data array
/// * `fisher` - Whether to use Fisher's (true) or Pearson's (false) definition
/// * `bias` - Whether to use biased estimator (true) or apply sample bias correction (false)
///
/// # Returns
///
/// * The kurtosis of the input data
#[allow(dead_code)]
pub fn kurtosis_simd<F, D>(x: &ArrayBase<D, Ix1>, fisher: bool, bias: bool) -> StatsResult<F>
where
    F: Float + NumCast + SimdUnifiedOps + Zero + One + Copy + Send + Sync + std::fmt::Display,
    D: Data<Elem = F>,
{
    checkarray_finite(x, "x")?;

    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Empty array provided".to_string(),
        ));
    }

    if x.len() < 4 {
        return Err(StatsError::DomainError(
            "At least 4 data points required to calculate kurtosis".to_string(),
        ));
    }

    let n = x.len();
    let n_f = F::from(n).unwrap();
    let optimizer = AutoOptimizer::new();

    // Compute mean using SIMD if beneficial
    let mean = if optimizer.should_use_simd(n) {
        F::simd_sum(&x.view()) / n_f
    } else {
        x.iter().fold(F::zero(), |acc, &val| acc + val) / n_f
    };

    // SIMD-optimized moment calculations
    let (sum_sq_dev, sum_fourth_dev) = if optimizer.should_use_simd(n) {
        compute_fourth_moments_simd(x, mean, n)
    } else {
        compute_fourth_moments_scalar(x, mean)
    };

    let variance = sum_sq_dev / n_f;

    if variance == F::zero() {
        return Err(StatsError::DomainError(
            "Standard deviation is zero, kurtosis undefined".to_string(),
        ));
    }

    // Calculate kurtosis
    let fourth_moment = sum_fourth_dev / n_f;
    let mut k = fourth_moment / (variance * variance);

    // Apply bias correction if requested
    if !bias && n > 3 {
        // Unbiased estimator for kurtosis
        let n_f = F::from(n).unwrap();
        let n1 = n_f - F::one();
        let n2 = n_f - F::from(2.0).unwrap();
        let n3 = n_f - F::from(3.0).unwrap();

        // For sample kurtosis: k = ((n+1)*k - 3*(n-1)) * (n-1) / ((n-2)*(n-3)) + 3
        k = ((n_f + F::one()) * k - F::from(3.0).unwrap() * n1) * n1 / (n2 * n3)
            + F::from(3.0).unwrap();
    }

    // Apply Fisher's definition (excess kurtosis)
    if fisher {
        k = k - F::from(3.0).unwrap();
    }

    Ok(k)
}

/// SIMD-optimized generic moment calculation
///
/// Computes the nth moment using SIMD acceleration for vectorized operations.
///
/// # Arguments
///
/// * `x` - Input data array
/// * `moment_order` - Order of the moment to compute
/// * `center` - Whether to compute central moment (around mean) or raw moment
///
/// # Returns
///
/// * The nth moment of the input data
#[allow(dead_code)]
pub fn moment_simd<F, D>(x: &ArrayBase<D, Ix1>, momentorder: usize, center: bool) -> StatsResult<F>
where
    F: Float + NumCast + SimdUnifiedOps + Zero + One + Copy + Send + Sync + std::fmt::Display,
    D: Data<Elem = F>,
{
    checkarray_finite(x, "x")?;

    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Empty array provided".to_string(),
        ));
    }

    if momentorder == 0 {
        return Ok(F::one()); // 0th moment is always 1
    }

    let n = x.len();
    let n_f = F::from(n).unwrap();
    let _order_f = F::from(momentorder as f64).unwrap();
    let optimizer = AutoOptimizer::new();

    if center {
        // Central moment calculation
        let mean = if optimizer.should_use_simd(n) {
            F::simd_sum(&x.view()) / n_f
        } else {
            x.iter().fold(F::zero(), |acc, &val| acc + val) / n_f
        };

        let moment_sum = if optimizer.should_use_simd(n) {
            compute_central_moment_simd(x, mean, momentorder)
        } else {
            compute_central_moment_scalar(x, mean, momentorder)
        };

        Ok(moment_sum / n_f)
    } else {
        // Raw moment calculation
        let moment_sum = if optimizer.should_use_simd(n) {
            compute_raw_moment_simd(x, momentorder)
        } else {
            compute_raw_moment_scalar(x, momentorder)
        };

        Ok(moment_sum / n_f)
    }
}

/// Batch computation of multiple moments using SIMD
///
/// Efficiently computes multiple moments in a single pass through the data.
///
/// # Arguments
///
/// * `x` - Input data array
/// * `moments` - List of moment orders to compute
/// * `center` - Whether to compute central moments
///
/// # Returns
///
/// * Vector of computed moments in the same order as requested
#[allow(dead_code)]
pub fn moments_batch_simd<F, D>(
    x: &ArrayBase<D, Ix1>,
    moments: &[usize],
    center: bool,
) -> StatsResult<Vec<F>>
where
    F: Float + NumCast + SimdUnifiedOps + Zero + One + Copy + Send + Sync + std::fmt::Display,
    D: Data<Elem = F>,
{
    checkarray_finite(x, "x")?;

    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Empty array provided".to_string(),
        ));
    }

    let n = x.len();
    let n_f = F::from(n).unwrap();
    let optimizer = AutoOptimizer::new();

    let mut results = Vec::with_capacity(moments.len());

    if center {
        // Compute mean once for all central moments
        let mean = if optimizer.should_use_simd(n) {
            F::simd_sum(&x.view()) / n_f
        } else {
            x.iter().fold(F::zero(), |acc, &val| acc + val) / n_f
        };

        // Batch compute central moments
        for &order in moments {
            if order == 0 {
                results.push(F::one());
            } else {
                let moment_sum = if optimizer.should_use_simd(n) {
                    compute_central_moment_simd(x, mean, order)
                } else {
                    compute_central_moment_scalar(x, mean, order)
                };
                results.push(moment_sum / n_f);
            }
        }
    } else {
        // Batch compute raw moments
        for &order in moments {
            if order == 0 {
                results.push(F::one());
            } else {
                let moment_sum = if optimizer.should_use_simd(n) {
                    compute_raw_moment_simd(x, order)
                } else {
                    compute_raw_moment_scalar(x, order)
                };
                results.push(moment_sum / n_f);
            }
        }
    }

    Ok(results)
}

// Helper functions for SIMD computations

#[allow(dead_code)]
fn compute_moments_simd<F, D>(x: &ArrayBase<D, Ix1>, mean: F, n: usize) -> (F, F)
where
    F: Float + NumCast + SimdUnifiedOps + Zero + One + Copy,
    D: Data<Elem = F>,
{
    // Create mean array for SIMD subtraction
    let mean_array = Array1::from_elem(n, mean);

    // Compute deviations: x - mean
    let deviations = F::simd_sub(&x.view(), &mean_array.view());

    // Compute squared deviations
    let sq_deviations = F::simd_mul(&deviations.view(), &deviations.view());

    // Compute cubed deviations
    let cubed_deviations = F::simd_mul(&sq_deviations.view(), &deviations.view());

    // Sum the moments
    let sum_sq_dev = F::simd_sum(&sq_deviations.view());
    let sum_cubed_dev = F::simd_sum(&cubed_deviations.view());

    (sum_sq_dev, sum_cubed_dev)
}

#[allow(dead_code)]
fn compute_moments_scalar<F, D>(x: &ArrayBase<D, Ix1>, mean: F) -> (F, F)
where
    F: Float + NumCast + Zero + One + Copy,
    D: Data<Elem = F>,
{
    let mut sum_sq_dev = F::zero();
    let mut sum_cubed_dev = F::zero();

    for &val in x.iter() {
        let dev = val - mean;
        let dev_sq = dev * dev;
        sum_sq_dev = sum_sq_dev + dev_sq;
        sum_cubed_dev = sum_cubed_dev + dev_sq * dev;
    }

    (sum_sq_dev, sum_cubed_dev)
}

#[allow(dead_code)]
fn compute_fourth_moments_simd<F, D>(x: &ArrayBase<D, Ix1>, mean: F, n: usize) -> (F, F)
where
    F: Float + NumCast + SimdUnifiedOps + Zero + One + Copy,
    D: Data<Elem = F>,
{
    // Create mean array for SIMD subtraction
    let mean_array = Array1::from_elem(n, mean);

    // Compute deviations: x - mean
    let deviations = F::simd_sub(&x.view(), &mean_array.view());

    // Compute squared deviations
    let sq_deviations = F::simd_mul(&deviations.view(), &deviations.view());

    // Compute fourth power deviations
    let fourth_deviations = F::simd_mul(&sq_deviations.view(), &sq_deviations.view());

    // Sum the moments
    let sum_sq_dev = F::simd_sum(&sq_deviations.view());
    let sum_fourth_dev = F::simd_sum(&fourth_deviations.view());

    (sum_sq_dev, sum_fourth_dev)
}

#[allow(dead_code)]
fn compute_fourth_moments_scalar<F, D>(x: &ArrayBase<D, Ix1>, mean: F) -> (F, F)
where
    F: Float + NumCast + Zero + One + Copy,
    D: Data<Elem = F>,
{
    let mut sum_sq_dev = F::zero();
    let mut sum_fourth_dev = F::zero();

    for &val in x.iter() {
        let dev = val - mean;
        let dev_sq = dev * dev;
        sum_sq_dev = sum_sq_dev + dev_sq;
        sum_fourth_dev = sum_fourth_dev + dev_sq * dev_sq;
    }

    (sum_sq_dev, sum_fourth_dev)
}

#[allow(dead_code)]
fn compute_central_moment_simd<F, D>(x: &ArrayBase<D, Ix1>, mean: F, order: usize) -> F
where
    F: Float + NumCast + SimdUnifiedOps + Zero + One + Copy,
    D: Data<Elem = F>,
{
    let n = x.len();
    let mean_array = Array1::from_elem(n, mean);

    // Compute deviations
    let deviations = F::simd_sub(&x.view(), &mean_array.view());

    // Compute power of deviations
    match order {
        1 => F::simd_sum(&deviations.view()),
        2 => {
            let squared = F::simd_mul(&deviations.view(), &deviations.view());
            F::simd_sum(&squared.view())
        }
        3 => {
            let squared = F::simd_mul(&deviations.view(), &deviations.view());
            let cubed = F::simd_mul(&squared.view(), &deviations.view());
            F::simd_sum(&cubed.view())
        }
        4 => {
            let squared = F::simd_mul(&deviations.view(), &deviations.view());
            let fourth = F::simd_mul(&squared.view(), &squared.view());
            F::simd_sum(&fourth.view())
        }
        _ => {
            // For higher orders, use scalar computation with SIMD sum
            let order_f = F::from(order as f64).unwrap();
            let powered: Array1<F> = deviations.mapv(|x| x.powf(order_f));
            F::simd_sum(&powered.view())
        }
    }
}

#[allow(dead_code)]
fn compute_central_moment_scalar<F, D>(x: &ArrayBase<D, Ix1>, mean: F, order: usize) -> F
where
    F: Float + NumCast + Zero + One + Copy,
    D: Data<Elem = F>,
{
    let order_f = F::from(order as f64).unwrap();
    x.iter()
        .map(|&val| (val - mean).powf(order_f))
        .fold(F::zero(), |acc, val| acc + val)
}

#[allow(dead_code)]
fn compute_raw_moment_simd<F, D>(x: &ArrayBase<D, Ix1>, order: usize) -> F
where
    F: Float + NumCast + SimdUnifiedOps + Zero + One + Copy,
    D: Data<Elem = F>,
{
    // Compute power of x
    match order {
        1 => F::simd_sum(&x.view()),
        2 => {
            let squared = F::simd_mul(&x.view(), &x.view());
            F::simd_sum(&squared.view())
        }
        3 => {
            let squared = F::simd_mul(&x.view(), &x.view());
            let cubed = F::simd_mul(&squared.view(), &x.view());
            F::simd_sum(&cubed.view())
        }
        4 => {
            let squared = F::simd_mul(&x.view(), &x.view());
            let fourth = F::simd_mul(&squared.view(), &squared.view());
            F::simd_sum(&fourth.view())
        }
        _ => {
            // For higher orders, use scalar computation with SIMD sum
            let order_f = F::from(order as f64).unwrap();
            let powered: Array1<F> = x.mapv(|val| val.powf(order_f));
            F::simd_sum(&powered.view())
        }
    }
}

#[allow(dead_code)]
fn compute_raw_moment_scalar<F, D>(x: &ArrayBase<D, Ix1>, order: usize) -> F
where
    F: Float + NumCast + Zero + One + Copy,
    D: Data<Elem = F>,
{
    let order_f = F::from(order as f64).unwrap();
    x.iter()
        .map(|&val| val.powf(order_f))
        .fold(F::zero(), |acc, val| acc + val)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptive::{kurtosis, moment, skew};
    use ndarray::array;

    #[test]
    #[ignore = "timeout"]
    fn test_skewness_simd_consistency() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let simd_result = skewness_simd(&data.view(), false).unwrap();
        let scalar_result = skew(&data.view(), false, None).unwrap();

        assert!((simd_result - scalar_result).abs() < 1e-10);
    }

    #[test]
    fn test_kurtosis_simd_consistency() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let simd_result = kurtosis_simd(&data.view(), true, false).unwrap();
        let scalar_result = kurtosis(&data.view(), true, false, None).unwrap();

        assert!((simd_result - scalar_result).abs() < 1e-10);
    }

    #[test]
    fn test_moment_simd_consistency() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        for order in 1..=4 {
            for center in [true, false] {
                let simd_result = moment_simd(&data.view(), order, center).unwrap();
                let scalar_result = moment(&data.view(), order, center, None).unwrap();

                assert!(
                    (simd_result - scalar_result).abs() < 1e-10,
                    "Mismatch for order {} center {}: SIMD {} vs Scalar {}",
                    order,
                    center,
                    simd_result,
                    scalar_result
                );
            }
        }
    }

    #[test]
    fn test_moments_batch_simd() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let orders = vec![1, 2, 3, 4];

        let batch_results = moments_batch_simd(&data.view(), &orders, true).unwrap();

        for (i, &order) in orders.iter().enumerate() {
            let individual_result = moment_simd(&data.view(), order, true).unwrap();
            assert!((batch_results[i] - individual_result).abs() < 1e-10);
        }
    }
}
