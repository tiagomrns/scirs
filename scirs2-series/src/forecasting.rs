//! Time series forecasting methods
//!
//! This module provides implementations for forecasting future values of time series.

use ndarray::Array1;
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::decomposition::{exponential_decomposition, DecompositionModel};
use crate::error::{Result, TimeSeriesError};
use crate::utils::{is_stationary, transform_to_stationary};

/// Result of time series forecasting
#[derive(Debug, Clone)]
pub struct ForecastResult<F> {
    /// Point forecasts
    pub forecast: Array1<F>,
    /// Lower confidence interval
    pub lower_ci: Array1<F>,
    /// Upper confidence interval
    pub upper_ci: Array1<F>,
}

/// ARIMA model parameters
#[derive(Debug, Clone)]
pub struct ArimaParams {
    /// Autoregressive order (p)
    pub p: usize,
    /// Integration order (d)
    pub d: usize,
    /// Moving average order (q)
    pub q: usize,
    /// Seasonal order (P)
    pub seasonal_p: Option<usize>,
    /// Seasonal integration order (D)
    pub seasonal_d: Option<usize>,
    /// Seasonal moving average order (Q)
    pub seasonal_q: Option<usize>,
    /// Seasonal period
    pub seasonal_period: Option<usize>,
    /// Fit intercept
    pub fit_intercept: bool,
    /// Trend component
    pub trend: Option<String>,
}

impl Default for ArimaParams {
    fn default() -> Self {
        Self {
            p: 1,
            d: 0,
            q: 0,
            seasonal_p: None,
            seasonal_d: None,
            seasonal_q: None,
            seasonal_period: None,
            fit_intercept: true,
            trend: None,
        }
    }
}

/// Exponential smoothing parameters
#[derive(Debug, Clone)]
pub struct ExpSmoothingParams {
    /// Level smoothing parameter (alpha)
    pub alpha: f64,
    /// Trend smoothing parameter (beta)
    pub beta: Option<f64>,
    /// Seasonal smoothing parameter (gamma)
    pub gamma: Option<f64>,
    /// Seasonal period
    pub seasonal_period: Option<usize>,
    /// Whether to use multiplicative trend
    pub multiplicative_trend: bool,
    /// Whether to use multiplicative seasonality
    pub multiplicative_seasonality: bool,
    /// Whether to damp trend
    pub damped_trend: bool,
    /// Damping factor
    pub phi: Option<f64>,
}

impl Default for ExpSmoothingParams {
    fn default() -> Self {
        Self {
            alpha: 0.3,
            beta: None,
            gamma: None,
            seasonal_period: None,
            multiplicative_trend: false,
            multiplicative_seasonality: false,
            damped_trend: false,
            phi: None,
        }
    }
}

/// Forecasts future values using simple moving average
///
/// # Arguments
///
/// * `ts` - The time series data
/// * `window_size` - Size of the moving window
/// * `horizon` - Number of future points to forecast
/// * `conf_level` - Confidence level (0.0-1.0) for prediction intervals
///
/// # Returns
///
/// * Forecast result containing point forecasts and confidence intervals
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_series::forecasting::moving_average_forecast;
///
/// let ts = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let result = moving_average_forecast(&ts, 3, 5, 0.95).unwrap();
/// println!("Forecast: {:?}", result.forecast);
/// println!("Lower CI: {:?}", result.lower_ci);
/// println!("Upper CI: {:?}", result.upper_ci);
/// ```
pub fn moving_average_forecast<F>(
    ts: &Array1<F>,
    window_size: usize,
    horizon: usize,
    conf_level: f64,
) -> Result<ForecastResult<F>>
where
    F: Float + FromPrimitive + Debug,
{
    if ts.len() < window_size {
        return Err(TimeSeriesError::ForecastingError(format!(
            "Time series length ({}) must be at least equal to window size ({})",
            ts.len(),
            window_size
        )));
    }

    if conf_level <= 0.0 || conf_level >= 1.0 {
        return Err(TimeSeriesError::InvalidInput(
            "Confidence level must be between 0 and 1 (exclusive)".to_string(),
        ));
    }

    // Calculate the most recent moving average
    let mut sum = F::zero();
    for i in ts.len() - window_size..ts.len() {
        sum = sum + ts[i];
    }
    let avg = sum / F::from_usize(window_size).unwrap();

    // Create forecast arrays
    let mut forecast = Array1::zeros(horizon);
    let mut lower_ci = Array1::zeros(horizon);
    let mut upper_ci = Array1::zeros(horizon);

    // Calculate the standard error of past forecasts
    let mut sq_errors = Array1::zeros(ts.len() - window_size);

    for i in window_size..ts.len() {
        // Calculate the moving average for past windows
        let mut window_sum = F::zero();
        for j in i - window_size..i {
            window_sum = window_sum + ts[j];
        }
        let window_avg = window_sum / F::from_usize(window_size).unwrap();

        // Calculate squared error
        sq_errors[i - window_size] = (ts[i] - window_avg).powi(2);
    }

    // Calculate standard error
    let mse = sq_errors.iter().fold(F::zero(), |acc, &x| acc + x)
        / F::from_usize(sq_errors.len()).unwrap();
    let std_err = mse.sqrt();

    // Z-score for the given confidence level (approximation)
    let z_score = match conf_level {
        c if c >= 0.99 => F::from_f64(2.576).unwrap(),
        c if c >= 0.98 => F::from_f64(2.326).unwrap(),
        c if c >= 0.95 => F::from_f64(1.96).unwrap(),
        c if c >= 0.90 => F::from_f64(1.645).unwrap(),
        c if c >= 0.85 => F::from_f64(1.44).unwrap(),
        c if c >= 0.80 => F::from_f64(1.282).unwrap(),
        _ => F::from_f64(1.0).unwrap(),
    };

    // Compute forecast and confidence intervals
    for i in 0..horizon {
        forecast[i] = avg;

        // Increase uncertainty with horizon (more uncertainty further into the future)
        // This is a simplified model; in practice, more complex time-dependent error models are used
        let adjustment = F::one() + F::from_f64(0.1).unwrap() * F::from_usize(i).unwrap();
        let ci_width = z_score * std_err * adjustment;

        lower_ci[i] = avg - ci_width;
        upper_ci[i] = avg + ci_width;
    }

    Ok(ForecastResult {
        forecast,
        lower_ci,
        upper_ci,
    })
}

/// Forecasts future values using simple exponential smoothing
///
/// # Arguments
///
/// * `ts` - The time series data
/// * `alpha` - Smoothing parameter (0.0-1.0)
/// * `horizon` - Number of future points to forecast
/// * `conf_level` - Confidence level (0.0-1.0) for prediction intervals
///
/// # Returns
///
/// * Forecast result containing point forecasts and confidence intervals
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_series::forecasting::exponential_smoothing_forecast;
///
/// let ts = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let result = exponential_smoothing_forecast(&ts, 0.3, 5, 0.95).unwrap();
/// println!("Forecast: {:?}", result.forecast);
/// println!("Lower CI: {:?}", result.lower_ci);
/// println!("Upper CI: {:?}", result.upper_ci);
/// ```
pub fn exponential_smoothing_forecast<F>(
    ts: &Array1<F>,
    alpha: f64,
    horizon: usize,
    conf_level: f64,
) -> Result<ForecastResult<F>>
where
    F: Float + FromPrimitive + Debug,
{
    if ts.len() < 2 {
        return Err(TimeSeriesError::ForecastingError(
            "Time series must have at least 2 points for exponential smoothing".to_string(),
        ));
    }

    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(TimeSeriesError::InvalidInput(
            "Smoothing parameter (alpha) must be between 0 and 1 (exclusive)".to_string(),
        ));
    }

    if conf_level <= 0.0 || conf_level >= 1.0 {
        return Err(TimeSeriesError::InvalidInput(
            "Confidence level must be between 0 and 1 (exclusive)".to_string(),
        ));
    }

    // Initialize level and forecast error arrays
    let mut level = Array1::zeros(ts.len() + 1);
    let mut sq_errors = Array1::zeros(ts.len() - 1);

    level[0] = ts[0]; // Initialize with first observation

    // Apply simple exponential smoothing
    for i in 0..ts.len() {
        // Update level
        level[i + 1] =
            F::from_f64(alpha).unwrap() * ts[i] + F::from_f64(1.0 - alpha).unwrap() * level[i];

        // Calculate forecast error for one-step ahead forecast
        if i > 0 {
            sq_errors[i - 1] = (ts[i] - level[i]).powi(2);
        }
    }

    // Calculate mean squared error
    let mse = sq_errors.iter().fold(F::zero(), |acc, &x| acc + x)
        / F::from_usize(sq_errors.len()).unwrap();
    let std_err = mse.sqrt();

    // Z-score for the given confidence level
    let z_score = match conf_level {
        c if c >= 0.99 => F::from_f64(2.576).unwrap(),
        c if c >= 0.98 => F::from_f64(2.326).unwrap(),
        c if c >= 0.95 => F::from_f64(1.96).unwrap(),
        c if c >= 0.90 => F::from_f64(1.645).unwrap(),
        c if c >= 0.85 => F::from_f64(1.44).unwrap(),
        c if c >= 0.80 => F::from_f64(1.282).unwrap(),
        _ => F::from_f64(1.0).unwrap(),
    };

    // Create forecast arrays
    let mut forecast = Array1::zeros(horizon);
    let mut lower_ci = Array1::zeros(horizon);
    let mut upper_ci = Array1::zeros(horizon);

    // Compute forecast and confidence intervals
    for i in 0..horizon {
        forecast[i] = level[ts.len()]; // All forecasts are the same (last level)

        // Increase uncertainty with horizon
        // For SES, theoretical standard error increases with square root of horizon
        let h_adjustment = (F::from_usize(i + 1).unwrap()).sqrt();
        let ci_width = z_score * std_err * h_adjustment;

        lower_ci[i] = forecast[i] - ci_width;
        upper_ci[i] = forecast[i] + ci_width;
    }

    Ok(ForecastResult {
        forecast,
        lower_ci,
        upper_ci,
    })
}

/// Forecasts future values using Holt-Winters exponential smoothing
///
/// # Arguments
///
/// * `ts` - The time series data
/// * `params` - Exponential smoothing parameters
/// * `horizon` - Number of future points to forecast
/// * `conf_level` - Confidence level (0.0-1.0) for prediction intervals
///
/// # Returns
///
/// * Forecast result containing point forecasts and confidence intervals
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_series::forecasting::{holt_winters_forecast, ExpSmoothingParams};
///
/// let ts = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 9.0, 8.0];
///
/// let mut params = ExpSmoothingParams::default();
/// params.alpha = 0.3;
/// params.beta = Some(0.1);
/// params.gamma = Some(0.2);
/// params.seasonal_period = Some(4);
///
/// let result = holt_winters_forecast(&ts, &params, 8, 0.95).unwrap();
/// println!("Forecast: {:?}", result.forecast);
/// ```
pub fn holt_winters_forecast<F>(
    ts: &Array1<F>,
    params: &ExpSmoothingParams,
    horizon: usize,
    conf_level: f64,
) -> Result<ForecastResult<F>>
where
    F: Float + FromPrimitive + Debug,
{
    // Validate inputs
    if params.alpha <= 0.0 || params.alpha >= 1.0 {
        return Err(TimeSeriesError::InvalidInput(
            "Alpha must be between 0 and 1 (exclusive)".to_string(),
        ));
    }

    if let Some(beta) = params.beta {
        if beta <= 0.0 || beta >= 1.0 {
            return Err(TimeSeriesError::InvalidInput(
                "Beta must be between 0 and 1 (exclusive)".to_string(),
            ));
        }
    }

    if let Some(gamma) = params.gamma {
        if gamma <= 0.0 || gamma >= 1.0 {
            return Err(TimeSeriesError::InvalidInput(
                "Gamma must be between 0 and 1 (exclusive)".to_string(),
            ));
        }

        // If gamma is provided, seasonal period must also be provided
        if params.seasonal_period.is_none() {
            return Err(TimeSeriesError::InvalidInput(
                "Seasonal period must be provided when gamma is specified".to_string(),
            ));
        }
    }

    if conf_level <= 0.0 || conf_level >= 1.0 {
        return Err(TimeSeriesError::InvalidInput(
            "Confidence level must be between 0 and 1 (exclusive)".to_string(),
        ));
    }

    if let Some(period) = params.seasonal_period {
        if ts.len() < 2 * period {
            return Err(TimeSeriesError::ForecastingError(format!(
                "Time series length ({}) must be at least twice the seasonal period ({})",
                ts.len(),
                period
            )));
        }
    } else if ts.len() < 3 {
        return Err(TimeSeriesError::ForecastingError(
            "Time series must have at least 3 points for non-seasonal models".to_string(),
        ));
    }

    // For damped trend model, phi must be provided and valid
    if params.damped_trend {
        if let Some(phi) = params.phi {
            if phi <= 0.0 || phi >= 1.0 {
                return Err(TimeSeriesError::InvalidInput(
                    "Damping parameter (phi) must be between 0 and 1 (exclusive)".to_string(),
                ));
            }
        } else {
            return Err(TimeSeriesError::InvalidInput(
                "Damping parameter (phi) must be provided for damped trend models".to_string(),
            ));
        }
    }

    // Determine the model type
    let has_trend = params.beta.is_some();
    let has_seasonal = params.gamma.is_some() && params.seasonal_period.is_some();

    // For multiplicative models, data must be strictly positive
    if (params.multiplicative_trend || params.multiplicative_seasonality)
        && ts.iter().any(|&x| x <= F::zero())
    {
        return Err(TimeSeriesError::InvalidInput(
            "Multiplicative models require strictly positive data".to_string(),
        ));
    }

    // Implementing a simplified version of Holt-Winters
    // This is a placeholder implementation and would need more work for a full implementation

    // Initialize arrays for components
    let n = ts.len();
    let mut level = Array1::zeros(n + 1);
    let mut trend = Array1::zeros(n + 1);
    let mut seasonal = Array1::zeros(n + params.seasonal_period.unwrap_or(1));
    let mut forecast_errors = Array1::zeros(n);

    // For simplicity, we're using an additive model here
    // A full implementation would handle multiplicative models as well

    // Initialize components
    if has_seasonal {
        // We use exponential decomposition which is already implemented
        let period = params.seasonal_period.unwrap();
        let decomp = exponential_decomposition(
            ts,
            period,
            params.alpha,
            params.beta.unwrap_or(0.1),
            params.gamma.unwrap(),
            if params.multiplicative_seasonality {
                DecompositionModel::Multiplicative
            } else {
                DecompositionModel::Additive
            },
        )?;

        // Extract components
        level[n] = decomp.trend[n - 1];
        if has_trend {
            // Simplistically, we use the difference between the last two trend values
            if n >= 2 {
                trend[n] = decomp.trend[n - 1] - decomp.trend[n - 2];
            }
        }

        // Copy the seasonal pattern
        for i in 0..period {
            seasonal[i] = decomp.seasonal[n - period + i];
        }

        // Calculate forecast errors
        for i in period..n {
            let pred = decomp.trend[i - 1] + decomp.seasonal[i];
            forecast_errors[i] = ts[i] - pred;
        }
    } else if has_trend {
        // Simple Holt's method (linear trend, no seasonality)
        level[0] = ts[0];
        if n > 1 {
            trend[0] = ts[1] - ts[0];
        }

        // Apply Holt's method
        let alpha = F::from_f64(params.alpha).unwrap();
        let beta = F::from_f64(params.beta.unwrap()).unwrap();
        let phi = F::from_f64(params.phi.unwrap_or(1.0)).unwrap();

        for i in 1..=n {
            // Calculate expected value
            let expected = level[i - 1]
                + if params.damped_trend {
                    phi * trend[i - 1]
                } else {
                    trend[i - 1]
                };

            // Update level and trend
            if i < n {
                level[i] = alpha * ts[i - 1] + (F::one() - alpha) * expected;
                trend[i] = beta * (level[i] - level[i - 1])
                    + (F::one() - beta)
                        * if params.damped_trend {
                            phi * trend[i - 1]
                        } else {
                            trend[i - 1]
                        };

                // Calculate forecast error
                forecast_errors[i - 1] = ts[i - 1] - expected;
            }
        }
    } else {
        // Simple exponential smoothing (level only)
        return exponential_smoothing_forecast(ts, params.alpha, horizon, conf_level);
    }

    // Calculate standard error for confidence intervals
    let mse = forecast_errors
        .iter()
        .skip(if has_seasonal {
            params.seasonal_period.unwrap()
        } else {
            1
        })
        .fold(F::zero(), |acc, &x| acc + x.powi(2))
        / F::from_usize(if has_seasonal {
            n - params.seasonal_period.unwrap()
        } else {
            n - 1
        })
        .unwrap();
    let std_err = mse.sqrt();

    // Z-score for confidence level
    let z_score = match conf_level {
        c if c >= 0.99 => F::from_f64(2.576).unwrap(),
        c if c >= 0.98 => F::from_f64(2.326).unwrap(),
        c if c >= 0.95 => F::from_f64(1.96).unwrap(),
        c if c >= 0.90 => F::from_f64(1.645).unwrap(),
        c if c >= 0.85 => F::from_f64(1.44).unwrap(),
        c if c >= 0.80 => F::from_f64(1.282).unwrap(),
        _ => F::from_f64(1.0).unwrap(),
    };

    // Create forecast arrays
    let mut forecast = Array1::zeros(horizon);
    let mut lower_ci = Array1::zeros(horizon);
    let mut upper_ci = Array1::zeros(horizon);

    // Generate forecasts
    for h in 0..horizon {
        let mut pred = level[n];

        // Add trend component
        if has_trend {
            let phi = F::from_f64(params.phi.unwrap_or(1.0)).unwrap();
            if params.damped_trend {
                // Sum of damped trend: b * (1 + phi + phi^2 + ... + phi^(h-1))
                let mut sum = F::one();
                let mut term = F::one();
                for _ in 1..h + 1 {
                    term = term * phi;
                    sum = sum + term;
                }
                pred = pred + trend[n] * sum;
            } else {
                // Linear trend: l + h*b
                pred = pred + F::from_usize(h + 1).unwrap() * trend[n];
            }
        }

        // Add seasonal component
        if has_seasonal {
            let period = params.seasonal_period.unwrap();
            let season_idx = (h + n) % period;
            if params.multiplicative_seasonality {
                pred = pred * seasonal[season_idx];
            } else {
                pred = pred + seasonal[season_idx];
            }
        }

        forecast[h] = pred;

        // Confidence intervals (simplified)
        // A full implementation would account for increased uncertainty with horizon
        let h_adjustment = (F::from_usize(h + 1).unwrap()).sqrt();
        let ci_width = z_score * std_err * h_adjustment;

        lower_ci[h] = pred - ci_width;
        upper_ci[h] = pred + ci_width;
    }

    Ok(ForecastResult {
        forecast,
        lower_ci,
        upper_ci,
    })
}

/// Forecasts using ARIMA (Auto-Regressive Integrated Moving Average) models
///
/// # Arguments
///
/// * `ts` - The time series data
/// * `params` - ARIMA model parameters
/// * `horizon` - Number of future points to forecast
/// * `conf_level` - Confidence level (0.0-1.0) for prediction intervals
///
/// # Returns
///
/// * Forecast result containing point forecasts and confidence intervals
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_series::forecasting::{arima_forecast, ArimaParams};
///
/// let ts = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
///
/// let mut params = ArimaParams::default();
/// params.p = 1;
/// params.d = 1;
/// params.q = 1;
///
/// let result = arima_forecast(&ts, &params, 5, 0.95).unwrap();
/// println!("Forecast: {:?}", result.forecast);
/// ```
pub fn arima_forecast<F>(
    ts: &Array1<F>,
    params: &ArimaParams,
    horizon: usize,
    conf_level: f64,
) -> Result<ForecastResult<F>>
where
    F: Float + FromPrimitive + Debug,
{
    // Validate inputs
    if ts.len() <= params.p + params.d + params.q {
        return Err(TimeSeriesError::ForecastingError(format!(
            "Time series length ({}) must be greater than p+d+q ({})",
            ts.len(),
            params.p + params.d + params.q
        )));
    }

    if conf_level <= 0.0 || conf_level >= 1.0 {
        return Err(TimeSeriesError::InvalidInput(
            "Confidence level must be between 0 and 1 (exclusive)".to_string(),
        ));
    }

    // Check for seasonal components
    if let Some(period) = params.seasonal_period {
        if ts.len() < 2 * period {
            return Err(TimeSeriesError::ForecastingError(format!(
                "Time series length ({}) must be at least twice the seasonal period ({})",
                ts.len(),
                period
            )));
        }
    }

    // ARIMA implementation is complex and would require a more detailed implementation
    // Here's a simplified version that handles some basic cases

    // Differencing (if needed)
    let mut data = ts.clone();

    // Apply differencing "d" times
    for _ in 0..params.d {
        let mut diff_data = Array1::zeros(data.len() - 1);
        for i in 0..data.len() - 1 {
            diff_data[i] = data[i + 1] - data[i];
        }
        data = diff_data;
    }

    // Apply seasonal differencing if needed
    if let (Some(s_d), Some(period)) = (params.seasonal_d, params.seasonal_period) {
        for _ in 0..s_d {
            if data.len() <= period {
                return Err(TimeSeriesError::ForecastingError(
                    "Series too short after differencing for seasonal differencing".to_string(),
                ));
            }

            let mut diff_data = Array1::zeros(data.len() - period);
            for i in 0..data.len() - period {
                diff_data[i] = data[i + period] - data[i];
            }
            data = diff_data;
        }
    }

    // A simple AR(p) model implementation
    // For a full ARIMA implementation, we would need to estimate all parameters
    // and handle the MA terms as well

    // Simplified AR coefficients (normally these would be estimated from the data)
    let ar_coeffs: Vec<F> = if params.p > 0 {
        let mut coeffs = Vec::with_capacity(params.p);
        for i in 0..params.p {
            // Decreasing coefficients for higher lags (simplified)
            coeffs.push(F::from_f64(0.8 / (i + 1) as f64).unwrap());
        }
        coeffs
    } else {
        vec![]
    };

    // Create forecast arrays
    let mut forecast = Array1::zeros(horizon);
    let mut lower_ci = Array1::zeros(horizon);
    let mut upper_ci = Array1::zeros(horizon);

    // Generate forecasts
    let n = data.len();

    // Simplified AR model forecasting
    for h in 0..horizon {
        let mut pred = F::zero();

        // AR component
        for i in 0..params.p {
            if h >= i && h - i < n {
                // Use actual values for available history
                pred = pred + ar_coeffs[i] * data[n - 1 - (h - i)];
            } else if h >= i {
                // Use forecasted values for future points
                pred = pred + ar_coeffs[i] * forecast[h - i - 1];
            }
        }

        forecast[h] = pred;
    }

    // Reverse the differencing to get forecasts in original scale
    for _ in 0..params.d {
        // For each step, we need the last value from the previous level
        let last_value = if params.d > 0 {
            ts[ts.len() - 1]
        } else {
            F::zero()
        };

        for h in 0..horizon {
            if h == 0 {
                forecast[h] = last_value + forecast[h];
            } else {
                forecast[h] = forecast[h - 1] + forecast[h];
            }
        }
    }

    // Simplified confidence intervals (would need proper error variance estimation)
    let std_err = F::from_f64(0.5).unwrap(); // Placeholder
    let z_score = match conf_level {
        c if c >= 0.99 => F::from_f64(2.576).unwrap(),
        c if c >= 0.95 => F::from_f64(1.96).unwrap(),
        c if c >= 0.90 => F::from_f64(1.645).unwrap(),
        _ => F::from_f64(1.0).unwrap(),
    };

    for h in 0..horizon {
        let ci_width = z_score * std_err * (F::from_usize(h + 1).unwrap()).sqrt();
        lower_ci[h] = forecast[h] - ci_width;
        upper_ci[h] = forecast[h] + ci_width;
    }

    Ok(ForecastResult {
        forecast,
        lower_ci,
        upper_ci,
    })
}

/// Automatically selects the best ARIMA model parameters
///
/// # Arguments
///
/// * `ts` - The time series data
/// * `max_p` - Maximum autoregressive order to consider
/// * `max_d` - Maximum differencing order to consider
/// * `max_q` - Maximum moving average order to consider
/// * `seasonal` - Whether to include seasonal components
/// * `seasonal_period` - Seasonal period (required if seasonal is true)
///
/// # Returns
///
/// * Optimal ARIMA parameters
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_series::forecasting::auto_arima;
///
/// let ts = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let params = auto_arima(&ts, 2, 1, 2, false, None).unwrap();
/// println!("Optimal p: {}, d: {}, q: {}", params.p, params.d, params.q);
/// ```
pub fn auto_arima<F>(
    ts: &Array1<F>,
    max_p: usize,
    max_d: usize,
    max_q: usize,
    seasonal: bool,
    seasonal_period: Option<usize>,
) -> Result<ArimaParams>
where
    F: Float + FromPrimitive + Debug,
{
    if ts.len() < 10 {
        return Err(TimeSeriesError::ForecastingError(
            "Time series too short for ARIMA parameter selection".to_string(),
        ));
    }

    if seasonal && seasonal_period.is_none() {
        return Err(TimeSeriesError::InvalidInput(
            "Seasonal period must be provided for seasonal models".to_string(),
        ));
    }

    if seasonal && seasonal_period.unwrap() >= ts.len() / 2 {
        return Err(TimeSeriesError::InvalidInput(format!(
            "Seasonal period ({}) must be less than half the time series length ({})",
            seasonal_period.unwrap(),
            ts.len()
        )));
    }

    // First, determine the order of differencing needed for stationarity
    let mut best_d = 0;
    let mut series_is_stationary = false;

    // Check stationarity
    let (_, p_value) = is_stationary(ts, None)?;
    if p_value < F::from_f64(0.05).unwrap() {
        series_is_stationary = true;
    }

    // If not stationary, try differencing
    if !series_is_stationary {
        let mut ts_diff = ts.clone();

        for d in 1..=max_d {
            // Apply differencing
            let diff_ts = transform_to_stationary(&ts_diff, "diff", None)?;

            // Check stationarity of differenced series
            let (_, p_value) = is_stationary(&diff_ts, None)?;
            if p_value < F::from_f64(0.05).unwrap() {
                best_d = d;
                break;
            }

            ts_diff = diff_ts;
        }
    }

    // Simplified model selection
    // In a full implementation, we would fit models with different (p,d,q) combinations
    // and select the best one based on AIC or BIC criteria

    // For this simplified version, we'll use some heuristics
    let best_p = if max_p > 0 { 1 } else { 0 };
    let best_q = if max_q > 0 { 1 } else { 0 };

    // Create the ARIMA parameters
    let mut params = ArimaParams {
        p: best_p,
        d: best_d,
        q: best_q,
        seasonal_p: None,
        seasonal_d: None,
        seasonal_q: None,
        seasonal_period: None,
        fit_intercept: true,
        trend: None,
    };

    // Add seasonal components if requested
    if seasonal {
        // Simplified seasonal parameters
        params.seasonal_p = Some(1);
        params.seasonal_d = Some(1);
        params.seasonal_q = Some(1);
        params.seasonal_period = seasonal_period;
    }

    Ok(params)
}

/// Automatically selects the best exponential smoothing model
///
/// # Arguments
///
/// * `ts` - The time series data
/// * `seasonal_period` - Seasonal period (optional)
///
/// # Returns
///
/// * Optimal exponential smoothing parameters
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_series::forecasting::auto_ets;
///
/// let ts = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let params = auto_ets(&ts, None).unwrap();
/// println!("Alpha: {}", params.alpha);
/// ```
pub fn auto_ets<F>(ts: &Array1<F>, seasonal_period: Option<usize>) -> Result<ExpSmoothingParams>
where
    F: Float + FromPrimitive + Debug,
{
    if ts.len() < 10 {
        return Err(TimeSeriesError::ForecastingError(
            "Time series too short for ETS parameter selection".to_string(),
        ));
    }

    if let Some(period) = seasonal_period {
        if period >= ts.len() / 2 {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Seasonal period ({}) must be less than half the time series length ({})",
                period,
                ts.len()
            )));
        }
    }

    // A full implementation would try different combinations of:
    // - Error type (additive, multiplicative)
    // - Trend type (none, additive, multiplicative, damped)
    // - Seasonal type (none, additive, multiplicative)
    // And select the best model based on AIC or BIC

    // Simplified approach for this implementation

    // Check if data is strictly positive (required for multiplicative models)
    let all_positive = ts.iter().all(|&x| x > F::zero());

    // Check for trend
    let has_trend = {
        // Simple linear regression
        let n = ts.len();
        let mut sum_x = F::zero();
        let mut sum_y = F::zero();
        let mut sum_xy = F::zero();
        let mut sum_xx = F::zero();

        for i in 0..n {
            let x = F::from_usize(i).unwrap();
            let y = ts[i];
            sum_x = sum_x + x;
            sum_y = sum_y + y;
            sum_xy = sum_xy + x * y;
            sum_xx = sum_xx + x * x;
        }

        let slope = (F::from_usize(n).unwrap() * sum_xy - sum_x * sum_y)
            / (F::from_usize(n).unwrap() * sum_xx - sum_x * sum_x);

        // If slope is significantly different from zero, assume there's a trend
        slope.abs() > F::from_f64(0.01).unwrap()
    };

    // Check for seasonality
    let has_seasonality = if let Some(period) = seasonal_period {
        if ts.len() >= 2 * period {
            // Calculate correlation between seasonal lags
            let mut sum_corr = F::zero();
            let mut count = 0;

            for lag in 1..=min(3, ts.len() / period) {
                let lag_p = lag * period;
                if ts.len() > lag_p {
                    let mut sum_xy = F::zero();
                    let mut sum_x = F::zero();
                    let mut sum_y = F::zero();
                    let mut sum_xx = F::zero();
                    let mut sum_yy = F::zero();
                    let mut n = 0;

                    for i in 0..ts.len() - lag_p {
                        let x = ts[i];
                        let y = ts[i + lag_p];
                        sum_x = sum_x + x;
                        sum_y = sum_y + y;
                        sum_xy = sum_xy + x * y;
                        sum_xx = sum_xx + x * x;
                        sum_yy = sum_yy + y * y;
                        n += 1;
                    }

                    if n > 0 {
                        let n_f = F::from_usize(n).unwrap();
                        let denom = ((n_f * sum_xx - sum_x * sum_x)
                            * (n_f * sum_yy - sum_y * sum_y))
                            .sqrt();

                        if denom > F::zero() {
                            let corr = (n_f * sum_xy - sum_x * sum_y) / denom;
                            sum_corr = sum_corr + corr;
                            count += 1;
                        }
                    }
                }
            }

            // If average correlation is high, assume there's seasonality
            count > 0 && (sum_corr / F::from_usize(count).unwrap()) > F::from_f64(0.3).unwrap()
        } else {
            false
        }
    } else {
        false
    };

    // Create parameters based on detected patterns
    let mut params = ExpSmoothingParams {
        alpha: 0.2,
        ..Default::default()
    };

    if has_trend {
        // Add trend component with typical beta value
        params.beta = Some(0.1);

        // Consider multiplicative trend if data is strictly positive and
        // if the data pattern suggests exponential growth/decay
        if all_positive {
            // Calculate first and second half averages
            let half = ts.len() / 2;
            let first_half_avg = ts.iter().take(half).fold(F::zero(), |acc, &x| acc + x)
                / F::from_usize(half).unwrap();
            let second_half_avg = ts.iter().skip(half).fold(F::zero(), |acc, &x| acc + x)
                / F::from_usize(ts.len() - half).unwrap();

            if second_half_avg / first_half_avg > F::from_f64(2.0).unwrap() {
                params.multiplicative_trend = true;
            }
        }

        // Consider damped trend if growth appears to be leveling off
        if ts.len() >= 10 {
            let first_third = ts.len() / 3;
            let second_third = 2 * ts.len() / 3;

            let first_slope = (ts[first_third] - ts[0]) / F::from_usize(first_third).unwrap();
            let second_slope = (ts[second_third] - ts[first_third])
                / F::from_usize(second_third - first_third).unwrap();
            let third_slope = (ts[ts.len() - 1] - ts[second_third])
                / F::from_usize(ts.len() - 1 - second_third).unwrap();

            if (first_slope > second_slope && second_slope > third_slope)
                || (first_slope < second_slope && second_slope < third_slope)
            {
                params.damped_trend = true;
                params.phi = Some(0.9); // Typical damping parameter
            }
        }
    }

    if has_seasonality {
        // Add seasonal component
        params.gamma = Some(0.1);
        params.seasonal_period = seasonal_period;

        // Consider multiplicative seasonality for data with changing seasonal amplitude
        if all_positive {
            let period = seasonal_period.unwrap();
            let num_seasons = ts.len() / period;

            if num_seasons >= 2 {
                let mut seasonal_ranges = Vec::with_capacity(num_seasons);

                for s in 0..num_seasons {
                    let start = s * period;
                    let end = min((s + 1) * period, ts.len());

                    let mut min_val = ts[start];
                    let mut max_val = ts[start];

                    for i in start + 1..end {
                        if ts[i] < min_val {
                            min_val = ts[i];
                        }
                        if ts[i] > max_val {
                            max_val = ts[i];
                        }
                    }

                    seasonal_ranges.push(max_val - min_val);
                }

                // If seasonal ranges vary significantly, use multiplicative seasonality
                if num_seasons >= 3 {
                    let first_range = seasonal_ranges[0];
                    let last_range = seasonal_ranges[num_seasons - 1];

                    if (last_range / first_range > F::from_f64(1.5).unwrap())
                        || (first_range / last_range > F::from_f64(1.5).unwrap())
                    {
                        params.multiplicative_seasonality = true;
                    }
                }
            }
        }
    }

    Ok(params)
}

// Helper function to get the minimum of two values
fn min<T: Ord>(a: T, b: T) -> T {
    if a < b {
        a
    } else {
        b
    }
}
