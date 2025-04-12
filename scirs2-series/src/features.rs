//! Time series feature extraction
//!
//! This module provides functions to extract meaningful features from time series data
//! for classification, clustering, and other machine learning tasks.

use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::Debug;

use crate::error::{Result, TimeSeriesError};
use crate::utils::{autocorrelation, is_stationary, partial_autocorrelation};

/// Statistical features of a time series
#[derive(Debug, Clone)]
pub struct TimeSeriesFeatures<F> {
    /// Mean value
    pub mean: F,
    /// Standard deviation
    pub std_dev: F,
    /// Skewness (measure of asymmetry)
    pub skewness: F,
    /// Kurtosis (measure of "tailedness")
    pub kurtosis: F,
    /// Minimum value
    pub min: F,
    /// Maximum value
    pub max: F,
    /// Range (max - min)
    pub range: F,
    /// Median value
    pub median: F,
    /// First quartile (25th percentile)
    pub q1: F,
    /// Third quartile (75th percentile)
    pub q3: F,
    /// Interquartile range (IQR = Q3 - Q1)
    pub iqr: F,
    /// Coefficient of variation (std / mean)
    pub cv: F,
    /// Trend strength
    pub trend_strength: F,
    /// Seasonality strength
    pub seasonality_strength: Option<F>,
    /// First autocorrelation coefficient
    pub acf1: F,
    /// Autocorrelation function values
    pub acf: Array1<F>,
    /// Partial autocorrelation function values
    pub pacf: Array1<F>,
    /// ADF test statistic
    pub adf_stat: F,
    /// ADF test p-value
    pub adf_pvalue: F,
    /// Additional features
    pub additional: HashMap<String, F>,
}

/// Feature extraction options
#[derive(Debug, Clone)]
pub struct FeatureExtractionOptions {
    /// Maximum lag for autocorrelation
    pub max_lag: Option<usize>,
    /// Seasonal period (if known)
    pub seasonal_period: Option<usize>,
    /// Whether to calculate entropy features
    pub calculate_entropy: bool,
    /// Whether to calculate frequency domain features
    pub calculate_frequency_features: bool,
    /// Whether to calculate trend and seasonality strength
    pub calculate_decomposition_features: bool,
}

impl Default for FeatureExtractionOptions {
    fn default() -> Self {
        Self {
            max_lag: None,
            seasonal_period: None,
            calculate_entropy: false,
            calculate_frequency_features: false,
            calculate_decomposition_features: true,
        }
    }
}

/// Extract features from a time series
///
/// # Arguments
///
/// * `ts` - The time series data
/// * `options` - Feature extraction options
///
/// # Returns
///
/// * Time series features
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_series::features::{extract_features, FeatureExtractionOptions};
///
/// let ts = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let options = FeatureExtractionOptions::default();
/// let features = extract_features(&ts, &options).unwrap();
///
/// println!("Mean: {}", features.mean);
/// println!("Std Dev: {}", features.std_dev);
/// println!("Trend Strength: {}", features.trend_strength);
/// ```
pub fn extract_features<F>(
    ts: &Array1<F>,
    options: &FeatureExtractionOptions,
) -> Result<TimeSeriesFeatures<F>>
where
    F: Float + FromPrimitive + Debug,
{
    if ts.len() < 3 {
        return Err(TimeSeriesError::FeatureExtractionError(
            "Time series must have at least 3 points for feature extraction".to_string(),
        ));
    }

    let n = ts.len();

    // Calculate basic statistical features
    let mean = ts.iter().fold(F::zero(), |acc, &x| acc + x) / F::from_usize(n).unwrap();

    let mut sum_sq_dev = F::zero();
    let mut sum_cube_dev = F::zero();
    let mut sum_quart_dev = F::zero();

    for &x in ts.iter() {
        let dev = x - mean;
        let dev_sq = dev * dev;
        sum_sq_dev = sum_sq_dev + dev_sq;
        sum_cube_dev = sum_cube_dev + dev_sq * dev;
        sum_quart_dev = sum_quart_dev + dev_sq * dev_sq;
    }

    let variance = sum_sq_dev / F::from_usize(n).unwrap();
    let std_dev = variance.sqrt();

    // Avoid division by zero for skewness and kurtosis
    let (skewness, kurtosis) = if std_dev == F::zero() {
        (F::zero(), F::zero())
    } else {
        let skewness = sum_cube_dev / (F::from_usize(n).unwrap() * std_dev.powi(3));
        let kurtosis = sum_quart_dev / (F::from_usize(n).unwrap() * variance.powi(2))
            - F::from_f64(3.0).unwrap();
        (skewness, kurtosis)
    };

    // Min, max, range
    let min = ts.iter().fold(F::infinity(), |a, &b| a.min(b));
    let max = ts.iter().fold(F::neg_infinity(), |a, &b| a.max(b));
    let range = max - min;

    // Calculate median and quartiles
    let mut sorted = Vec::with_capacity(n);
    for &x in ts.iter() {
        sorted.push(x);
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let median = if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / F::from_f64(2.0).unwrap()
    } else {
        sorted[n / 2]
    };

    let q1_idx = n / 4;
    let q3_idx = 3 * n / 4;

    let q1 = if n % 4 == 0 {
        (sorted[q1_idx - 1] + sorted[q1_idx]) / F::from_f64(2.0).unwrap()
    } else {
        sorted[q1_idx]
    };

    let q3 = if 3 * n % 4 == 0 {
        (sorted[q3_idx - 1] + sorted[q3_idx]) / F::from_f64(2.0).unwrap()
    } else {
        sorted[q3_idx]
    };

    let iqr = q3 - q1;

    // Coefficient of variation
    let cv = if mean != F::zero() {
        std_dev / mean.abs()
    } else {
        F::infinity()
    };

    // ACF and PACF
    let max_lag = options.max_lag.unwrap_or(std::cmp::min(n / 4, 10));
    let acf = autocorrelation(ts, Some(max_lag))?;
    let pacf = partial_autocorrelation(ts, Some(max_lag))?;
    let acf1 = acf[1]; // First autocorrelation

    // Stationarity test
    let (adf_stat, adf_pvalue) = is_stationary(ts, None)?;

    // Trend and seasonality strength
    let (trend_strength, seasonality_strength) = if options.calculate_decomposition_features {
        calculate_trend_seasonality_strength(ts, options.seasonal_period)?
    } else {
        (F::zero(), None)
    };

    // Additional features
    let mut additional = HashMap::new();

    // Entropy features
    if options.calculate_entropy {
        let approx_entropy =
            calculate_approximate_entropy(ts, 2, F::from_f64(0.2).unwrap() * std_dev)?;
        additional.insert("approx_entropy".to_string(), approx_entropy);

        let sample_entropy = calculate_sample_entropy(ts, 2, F::from_f64(0.2).unwrap() * std_dev)?;
        additional.insert("sample_entropy".to_string(), sample_entropy);
    }

    // Frequency domain features
    if options.calculate_frequency_features {
        let spectral_features = calculate_spectral_features(ts)?;
        for (key, value) in spectral_features {
            additional.insert(key, value);
        }
    }

    Ok(TimeSeriesFeatures {
        mean,
        std_dev,
        skewness,
        kurtosis,
        min,
        max,
        range,
        median,
        q1,
        q3,
        iqr,
        cv,
        trend_strength,
        seasonality_strength,
        acf1,
        acf,
        pacf,
        adf_stat,
        adf_pvalue,
        additional,
    })
}

/// Extract multiple features from multiple time series
///
/// # Arguments
///
/// * `ts_collection` - Collection of time series
/// * `options` - Feature extraction options
///
/// # Returns
///
/// * Vector of time series features
///
/// # Example
///
/// ```
/// use ndarray::{array, Array2};
/// use scirs2_series::features::{extract_features_batch, FeatureExtractionOptions};
///
/// let ts1 = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let ts2 = array![5.0, 4.0, 3.0, 2.0, 1.0];
/// let ts_collection = Array2::from_shape_vec((2, 5),
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0]).unwrap();
///
/// let options = FeatureExtractionOptions::default();
/// let features = extract_features_batch(&ts_collection, &options).unwrap();
///
/// println!("Number of feature sets: {}", features.len());
/// println!("First time series mean: {}", features[0].mean);
/// println!("Second time series mean: {}", features[1].mean);
/// ```
pub fn extract_features_batch<F>(
    ts_collection: &Array2<F>,
    options: &FeatureExtractionOptions,
) -> Result<Vec<TimeSeriesFeatures<F>>>
where
    F: Float + FromPrimitive + Debug,
{
    // Verify that the input array has at least 2 dimensions
    if ts_collection.ndim() < 2 {
        return Err(TimeSeriesError::FeatureExtractionError(
            "Expected a collection of time series (2D array)".to_string(),
        ));
    }

    let n_series = ts_collection.shape()[0];
    let series_length = ts_collection.shape()[1];

    if series_length < 3 {
        return Err(TimeSeriesError::FeatureExtractionError(
            "Time series must have at least 3 points for feature extraction".to_string(),
        ));
    }

    let mut results = Vec::with_capacity(n_series);

    for i in 0..n_series {
        // Extract the i-th time series
        let ts = Array1::from_iter(ts_collection.slice(ndarray::s![i, ..]).iter().cloned());

        // Extract features
        let features = extract_features(&ts, options)?;
        results.push(features);
    }

    Ok(results)
}

/// Calculate trend and seasonality strength
fn calculate_trend_seasonality_strength<F>(
    ts: &Array1<F>,
    seasonal_period: Option<usize>,
) -> Result<(F, Option<F>)>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();

    // Calculate first differences (for trend)
    let mut diff1 = Vec::with_capacity(n - 1);
    for i in 1..n {
        diff1.push(ts[i] - ts[i - 1]);
    }

    // Variance of the original time series
    let ts_mean = ts.iter().fold(F::zero(), |acc, &x| acc + x) / F::from_usize(n).unwrap();
    let ts_var = ts
        .iter()
        .fold(F::zero(), |acc, &x| acc + (x - ts_mean).powi(2))
        / F::from_usize(n).unwrap();

    if ts_var == F::zero() {
        return Ok((F::zero(), None));
    }

    // Variance of the differenced series
    let diff_mean =
        diff1.iter().fold(F::zero(), |acc, &x| acc + x) / F::from_usize(diff1.len()).unwrap();
    let diff_var = diff1
        .iter()
        .fold(F::zero(), |acc, &x| acc + (x - diff_mean).powi(2))
        / F::from_usize(diff1.len()).unwrap();

    // Trend strength
    let trend_strength = F::one() - (diff_var / ts_var);

    // Seasonality strength (if seasonal period is provided)
    let seasonality_strength = if let Some(period) = seasonal_period {
        if n <= period {
            return Err(TimeSeriesError::FeatureExtractionError(
                "Time series length must be greater than seasonal period".to_string(),
            ));
        }

        // Calculate seasonal differences
        let mut seasonal_diff = Vec::with_capacity(n - period);
        for i in period..n {
            seasonal_diff.push(ts[i] - ts[i - period]);
        }

        // Variance of seasonal differences
        let s_diff_mean = seasonal_diff.iter().fold(F::zero(), |acc, &x| acc + x)
            / F::from_usize(seasonal_diff.len()).unwrap();
        let s_diff_var = seasonal_diff
            .iter()
            .fold(F::zero(), |acc, &x| acc + (x - s_diff_mean).powi(2))
            / F::from_usize(seasonal_diff.len()).unwrap();

        // Seasonality strength
        let s_strength = F::one() - (s_diff_var / ts_var);

        // Constrain to [0, 1] range
        Some(s_strength.max(F::zero()).min(F::one()))
    } else {
        None
    };

    // Constrain trend strength to [0, 1] range
    let trend_strength = trend_strength.max(F::zero()).min(F::one());

    Ok((trend_strength, seasonality_strength))
}

/// Calculate approximate entropy
fn calculate_approximate_entropy<F>(ts: &Array1<F>, m: usize, r: F) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    if ts.len() < m + 1 {
        return Err(TimeSeriesError::FeatureExtractionError(
            "Time series too short for approximate entropy calculation".to_string(),
        ));
    }

    let n = ts.len();

    // Create embedding vectors
    let mut phi_m = F::zero();
    let mut phi_m_plus_1 = F::zero();

    // Phi(m)
    for i in 0..=n - m {
        let mut count = F::zero();

        for j in 0..=n - m {
            // Check if vectors are within tolerance r
            let mut is_match = true;

            for k in 0..m {
                let x = *ts.get(i + k).unwrap();
                let y = *ts.get(j + k).unwrap();
                if (x - y).abs() > r {
                    is_match = false;
                    break;
                }
            }

            if is_match {
                count = count + F::one();
            }
        }

        phi_m = phi_m + (count / F::from_usize(n - m + 1).unwrap()).ln();
    }

    phi_m = phi_m / F::from_usize(n - m + 1).unwrap();

    // Phi(m+1)
    for i in 0..=n - m - 1 {
        let mut count = F::zero();

        for j in 0..=n - m - 1 {
            // Check if vectors are within tolerance r
            let mut is_match = true;

            for k in 0..m + 1 {
                let x = *ts.get(i + k).unwrap();
                let y = *ts.get(j + k).unwrap();
                if (x - y).abs() > r {
                    is_match = false;
                    break;
                }
            }

            if is_match {
                count = count + F::one();
            }
        }

        phi_m_plus_1 = phi_m_plus_1 + (count / F::from_usize(n - m).unwrap()).ln();
    }

    phi_m_plus_1 = phi_m_plus_1 / F::from_usize(n - m).unwrap();

    // Approximate entropy is phi_m - phi_(m+1)
    Ok(phi_m - phi_m_plus_1)
}

/// Calculate sample entropy
fn calculate_sample_entropy<F>(ts: &Array1<F>, m: usize, r: F) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    if ts.len() < m + 2 {
        return Err(TimeSeriesError::FeatureExtractionError(
            "Time series too short for sample entropy calculation".to_string(),
        ));
    }

    let n = ts.len();

    // Count matches for m and m+1
    let mut a = F::zero(); // Number of template matches of length m+1
    let mut b = F::zero(); // Number of template matches of length m

    for i in 0..n - m {
        for j in i + 1..n - m {
            // Check match for length m
            let mut is_match_m = true;

            for k in 0..m {
                let x = *ts.get(i + k).unwrap();
                let y = *ts.get(j + k).unwrap();
                if (x - y).abs() > r {
                    is_match_m = false;
                    break;
                }
            }

            if is_match_m {
                b = b + F::one();

                // Check additional element for m+1
                let x = *ts.get(i + m).unwrap();
                let y = *ts.get(j + m).unwrap();
                if (x - y).abs() <= r {
                    a = a + F::one();
                }
            }
        }
    }

    // Calculate sample entropy
    if b == F::zero() {
        return Err(TimeSeriesError::FeatureExtractionError(
            "No matches found for template length m".to_string(),
        ));
    }

    if a == F::zero() {
        // This is actually infinity, but we'll return a large value
        return Ok(F::from_f64(100.0).unwrap());
    }

    Ok(-((a / b).ln()))
}

/// Calculate spectral features from FFT
fn calculate_spectral_features<F>(ts: &Array1<F>) -> Result<HashMap<String, F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();

    if n < 4 {
        return Err(TimeSeriesError::FeatureExtractionError(
            "Time series too short for spectral feature calculation".to_string(),
        ));
    }

    // This is a simplified implementation
    // A full implementation would use FFT from scirs2-fft

    let mut features = HashMap::new();

    // For now, we'll just calculate some simple spectral approximations
    // using autocorrelations

    let acf_values = autocorrelation(ts, Some(n / 2))?;

    // Spectral entropy approximation
    let mut spectral_sum = F::zero();
    for lag in 1..acf_values.len() {
        let val = acf_values[lag].abs();
        spectral_sum = spectral_sum + val;
    }

    if spectral_sum > F::zero() {
        let mut spectral_entropy = F::zero();
        for lag in 1..acf_values.len() {
            let val = acf_values[lag].abs() / spectral_sum;
            if val > F::zero() {
                spectral_entropy = spectral_entropy - val * val.ln();
            }
        }
        features.insert("spectral_entropy".to_string(), spectral_entropy);
    }

    // Find dominant frequency (peak in ACF)
    let mut max_acf = F::neg_infinity();
    let mut dominant_period = 0;

    for lag in 1..acf_values.len() {
        if acf_values[lag] > max_acf {
            max_acf = acf_values[lag];
            dominant_period = lag;
        }
    }

    features.insert(
        "dominant_period".to_string(),
        F::from_usize(dominant_period).unwrap(),
    );
    features.insert(
        "dominant_frequency".to_string(),
        F::one() / F::from_usize(dominant_period.max(1)).unwrap(),
    );

    Ok(features)
}

/// Extract a single feature from a time series
///
/// # Arguments
///
/// * `ts` - The time series data
/// * `feature_name` - Name of the feature to extract
///
/// # Returns
///
/// * Feature value
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_series::features::extract_single_feature;
///
/// let ts = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let mean = extract_single_feature(&ts, "mean").unwrap();
/// let std_dev = extract_single_feature(&ts, "std_dev").unwrap();
///
/// println!("Mean: {}", mean);
/// println!("Std Dev: {}", std_dev);
/// ```
pub fn extract_single_feature<F>(ts: &Array1<F>, feature_name: &str) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    if ts.len() < 3 {
        return Err(TimeSeriesError::FeatureExtractionError(
            "Time series must have at least 3 points for feature extraction".to_string(),
        ));
    }

    let n = ts.len();

    match feature_name {
        "mean" => {
            let mean = ts.iter().fold(F::zero(), |acc, &x| acc + x) / F::from_usize(n).unwrap();
            Ok(mean)
        }
        "std_dev" => {
            let mean = ts.iter().fold(F::zero(), |acc, &x| acc + x) / F::from_usize(n).unwrap();
            let variance = ts
                .iter()
                .fold(F::zero(), |acc, &x| acc + (x - mean).powi(2))
                / F::from_usize(n).unwrap();
            Ok(variance.sqrt())
        }
        "min" => {
            let min = ts.iter().fold(F::infinity(), |a, &b| a.min(b));
            Ok(min)
        }
        "max" => {
            let max = ts.iter().fold(F::neg_infinity(), |a, &b| a.max(b));
            Ok(max)
        }
        "acf1" => {
            let acf = autocorrelation(ts, Some(1))?;
            Ok(acf[1])
        }
        "trend_strength" => {
            let (trend_strength, _) = calculate_trend_seasonality_strength(ts, None)?;
            Ok(trend_strength)
        }
        _ => Err(TimeSeriesError::FeatureExtractionError(format!(
            "Unknown feature: {}",
            feature_name
        ))),
    }
}
