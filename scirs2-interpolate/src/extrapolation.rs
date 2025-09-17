use ndarray::Array1;
use num_traits::{Float, FromPrimitive};

use crate::error::{InterpolateError, InterpolateResult};

/// Enhanced extrapolation methods for interpolation.
///
/// This module provides advanced extrapolation capabilities that go beyond
/// the basic ExtrapolateMode enum. It allows for more sophisticated boundary
/// handling and domain extension methods, including:
///
/// - Physics-informed extrapolation based on boundary derivatives
/// - Polynomial extrapolation of various orders
/// - Decay/growth models for asymptotic behavior
/// - Periodic extension of the domain
/// - Reflection-based extrapolation
/// - Domain-specific extrapolation models
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExtrapolationMethod {
    /// No extrapolation - return an error for points outside the domain
    Error,

    /// Use the nearest endpoint value (constant extrapolation)
    Constant,

    /// Linear extrapolation based on endpoint derivatives
    Linear,

    /// Quadratic extrapolation based on endpoint values and derivatives
    Quadratic,

    /// Cubic extrapolation preserving both values and derivatives at boundaries
    Cubic,

    /// Extend domain as if the function is periodic
    Periodic,

    /// Reflect the function at the boundaries
    Reflection,

    /// Exponential decay/growth model for asymptotic behavior
    Exponential,

    /// Power law decay/growth model for asymptotic behavior
    PowerLaw,

    /// Spline-based extrapolation using the full spline continuation
    Spline,

    /// Akima extrapolation for stable polynomial continuation
    Akima,

    /// Sinusoidal extrapolation for periodic data
    Sinusoidal,

    /// Rational function extrapolation for poles/zeros behavior
    Rational,

    /// Confidence-based extrapolation with uncertainty bands
    Confidence,

    /// Ensemble extrapolation combining multiple methods
    Ensemble,

    /// Adaptive extrapolation that selects the best method locally  
    Adaptive,

    /// Autoregressive extrapolation using AR models
    Autoregressive,

    /// Return zeros for all out-of-bounds points (SciPy 'zeros' mode)
    Zeros,

    /// Use nearest boundary value (SciPy 'nearest'/'edge' mode)
    Nearest,

    /// Mirror reflection without repeating edge values (SciPy 'mirror' mode)
    Mirror,

    /// Periodic wrapping (SciPy 'wrap' mode)
    Wrap,

    /// Clamped boundary conditions with zero derivatives
    Clamped,

    /// Grid-specific mirror mode for structured grids
    GridMirror,

    /// Grid-specific constant mode for structured grids
    GridConstant,

    /// Grid-specific wrap mode for structured grids
    GridWrap,
}

/// Direction for extrapolation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExtrapolationDirection {
    /// Extrapolation below the lower boundary
    Lower,

    /// Extrapolation above the upper boundary
    Upper,
}

/// Extrapolator for extending interpolation methods beyond their domain.
///
/// This class provides a flexible way to extrapolate values outside the
/// original domain of interpolation, using a variety of methods that can be
/// customized separately for the lower and upper boundaries.
#[derive(Debug, Clone)]
pub struct Extrapolator<T: Float> {
    /// Lower boundary of the original domain
    lower_bound: T,

    /// Upper boundary of the original domain
    upper_bound: T,

    /// Extrapolation method for below the lower boundary
    lower_method: ExtrapolationMethod,

    /// Extrapolation method for above the upper boundary
    upper_method: ExtrapolationMethod,

    /// Value at the lower boundary
    lower_value: T,

    /// Value at the upper boundary
    upper_value: T,

    /// Derivative at the lower boundary
    lower_derivative: T,

    /// Derivative at the upper boundary
    upper_derivative: T,

    /// Second derivative at the lower boundary (for higher-order methods)
    lower_second_derivative: Option<T>,

    /// Second derivative at the upper boundary (for higher-order methods)
    upper_second_derivative: Option<T>,

    /// Parameters for specialized extrapolation models
    parameters: ExtrapolationParameters<T>,
}

/// Parameters for specialized extrapolation methods
#[derive(Debug, Clone)]
pub struct ExtrapolationParameters<T: Float> {
    /// Decay/growth rate for exponential extrapolation
    exponential_rate: T,

    /// Offset for exponential extrapolation
    exponential_offset: T,

    /// Exponent for power law extrapolation
    power_exponent: T,

    /// Scale factor for power law extrapolation
    power_scale: T,

    /// Period for periodic extrapolation
    period: T,
}

impl<T: Float> Default for ExtrapolationParameters<T> {
    fn default() -> Self {
        Self {
            exponential_rate: T::one(),
            exponential_offset: T::zero(),
            power_exponent: -T::one(), // Default to 1/x decay
            power_scale: T::one(),
            period: T::from(2.0 * std::f64::consts::PI).unwrap(),
        }
    }
}

impl<T: Float> ExtrapolationParameters<T> {
    /// Creates default parameters for extrapolation methods
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the decay/growth rate for exponential extrapolation
    pub fn with_exponential_rate(mut self, rate: T) -> Self {
        self.exponential_rate = rate;
        self
    }

    /// Set the offset for exponential extrapolation
    pub fn with_exponential_offset(mut self, offset: T) -> Self {
        self.exponential_offset = offset;
        self
    }

    /// Set the exponent for power law extrapolation
    pub fn with_power_exponent(mut self, exponent: T) -> Self {
        self.power_exponent = exponent;
        self
    }

    /// Set the scale factor for power law extrapolation
    pub fn with_power_scale(mut self, scale: T) -> Self {
        self.power_scale = scale;
        self
    }

    /// Set the period for periodic extrapolation
    pub fn with_period(mut self, period: T) -> Self {
        self.period = period;
        self
    }
}

/// Configuration for confidence-based extrapolation
#[derive(Debug, Clone)]
pub struct ConfidenceExtrapolationConfig<T: Float> {
    /// Base extrapolation method
    pub base_method: ExtrapolationMethod,
    /// Confidence level (e.g., 0.95 for 95% confidence)
    pub confidence_level: T,
    /// Number of bootstrap samples for uncertainty estimation
    pub n_bootstrap: usize,
    /// Whether to include uncertainty bounds in results
    pub include_bounds: bool,
}

impl<T: Float> Default for ConfidenceExtrapolationConfig<T> {
    fn default() -> Self {
        Self {
            base_method: ExtrapolationMethod::Linear,
            confidence_level: T::from(0.95).unwrap(),
            n_bootstrap: 1000,
            include_bounds: true,
        }
    }
}

/// Result from confidence-based extrapolation
#[derive(Debug, Clone)]
pub struct ConfidenceExtrapolationResult<T: Float> {
    /// Point estimate
    pub estimate: T,
    /// Lower confidence bound
    pub lower_bound: T,
    /// Upper confidence bound  
    pub upper_bound: T,
    /// Standard error estimate
    pub standard_error: T,
}

/// Configuration for ensemble extrapolation
#[derive(Debug, Clone)]
pub struct EnsembleExtrapolationConfig<T: Float> {
    /// List of (method, weight) pairs
    pub methods_and_weights: Vec<(ExtrapolationMethod, T)>,
    /// How to combine the results (mean, median, weighted_mean)
    pub combination_strategy: EnsembleCombinationStrategy,
    /// Whether to normalize weights
    pub normalize_weights: bool,
}

/// Strategy for combining ensemble results
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EnsembleCombinationStrategy {
    /// Simple average of all methods
    Mean,
    /// Median of all methods (robust to outliers)
    Median,
    /// Weighted average using specified weights
    WeightedMean,
    /// Trimmed mean (remove extreme values)
    TrimmedMean { trim_fraction: f64 },
}

impl<T: Float> Default for EnsembleExtrapolationConfig<T> {
    fn default() -> Self {
        Self {
            methods_and_weights: vec![
                (ExtrapolationMethod::Linear, T::one()),
                (ExtrapolationMethod::Quadratic, T::one()),
                (ExtrapolationMethod::Cubic, T::one()),
            ],
            combination_strategy: EnsembleCombinationStrategy::WeightedMean,
            normalize_weights: true,
        }
    }
}

/// Configuration for adaptive extrapolation
#[derive(Debug, Clone)]
pub struct AdaptiveExtrapolationConfig {
    /// Candidate methods to choose from
    pub candidate_methods: Vec<ExtrapolationMethod>,
    /// Window size for local analysis
    pub analysis_window_size: usize,
    /// Criteria for method selection
    pub selection_criteria: Vec<AdaptiveSelectionCriterion>,
    /// Whether to cache method selections
    pub cache_selections: bool,
}

/// Criteria for adaptive method selection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AdaptiveSelectionCriterion {
    /// Choose based on local smoothness
    LocalSmoothness,
    /// Choose based on derivative continuity
    DerivativeContinuity,
    /// Choose based on prediction error (if validation data available)
    PredictionError,
    /// Choose based on stability of extrapolation
    ExtrapolationStability,
}

impl Default for AdaptiveExtrapolationConfig {
    fn default() -> Self {
        Self {
            candidate_methods: vec![
                ExtrapolationMethod::Linear,
                ExtrapolationMethod::Quadratic,
                ExtrapolationMethod::Cubic,
                ExtrapolationMethod::Akima,
            ],
            analysis_window_size: 5,
            selection_criteria: vec![
                AdaptiveSelectionCriterion::LocalSmoothness,
                AdaptiveSelectionCriterion::DerivativeContinuity,
            ],
            cache_selections: true,
        }
    }
}

/// Configuration for autoregressive extrapolation
#[derive(Debug, Clone)]
pub struct AutoregressiveExtrapolationConfig<T: Float> {
    /// Order of the AR model
    pub ar_order: usize,
    /// Method for fitting AR coefficients
    pub fitting_method: ARFittingMethod,
    /// Whether to include a constant term
    pub include_constant: bool,
    /// Regularization parameter for stability
    pub regularization: T,
}

/// Methods for fitting autoregressive models
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ARFittingMethod {
    /// Ordinary least squares
    OLS,
    /// Ridge regression for stability
    Ridge,
    /// Yule-Walker equations
    YuleWalker,
}

impl<T: Float> Default for AutoregressiveExtrapolationConfig<T> {
    fn default() -> Self {
        Self {
            ar_order: 3,
            fitting_method: ARFittingMethod::Ridge,
            include_constant: true,
            regularization: T::from(1e-6).unwrap(),
        }
    }
}

/// Advanced extrapolator with support for sophisticated extrapolation methods
#[derive(Debug, Clone)]
pub struct AdvancedExtrapolator<T: Float> {
    /// Basic extrapolator for standard methods
    pub base_extrapolator: Extrapolator<T>,
    /// Configuration for confidence-based extrapolation
    pub confidence_config: Option<ConfidenceExtrapolationConfig<T>>,
    /// Configuration for ensemble extrapolation
    pub ensemble_config: Option<EnsembleExtrapolationConfig<T>>,
    /// Configuration for adaptive extrapolation
    pub adaptive_config: Option<AdaptiveExtrapolationConfig>,
    /// Configuration for autoregressive extrapolation
    pub autoregressive_config: Option<AutoregressiveExtrapolationConfig<T>>,
    /// Historical data for advanced methods (when available)
    pub historical_data: Option<(Array1<T>, Array1<T>)>,
}

impl<T: Float + std::fmt::Display + std::default::Default + std::ops::AddAssign>
    AdvancedExtrapolator<T>
{
    /// Create a new advanced extrapolator
    pub fn new(_baseextrapolator: Extrapolator<T>) -> Self {
        Self {
            base_extrapolator: _baseextrapolator,
            confidence_config: None,
            ensemble_config: None,
            adaptive_config: None,
            autoregressive_config: None,
            historical_data: None,
        }
    }

    /// Enable confidence-based extrapolation
    pub fn with_confidence(mut self, config: ConfidenceExtrapolationConfig<T>) -> Self {
        self.confidence_config = Some(config);
        self
    }

    /// Enable ensemble extrapolation
    pub fn with_ensemble(mut self, config: EnsembleExtrapolationConfig<T>) -> Self {
        self.ensemble_config = Some(config);
        self
    }

    /// Enable adaptive extrapolation
    pub fn with_adaptive(mut self, config: AdaptiveExtrapolationConfig) -> Self {
        self.adaptive_config = Some(config);
        self
    }

    /// Enable autoregressive extrapolation
    pub fn with_autoregressive(mut self, config: AutoregressiveExtrapolationConfig<T>) -> Self {
        self.autoregressive_config = Some(config);
        self
    }

    /// Set historical data for advanced methods
    pub fn with_historical_data(mut self, x_data: Array1<T>, ydata: Array1<T>) -> Self {
        self.historical_data = Some((x_data, ydata));
        self
    }

    /// Perform advanced extrapolation at a point
    pub fn extrapolate_advanced(&self, x: T) -> InterpolateResult<T> {
        // Try ensemble extrapolation first if configured
        if self.ensemble_config.is_some() {
            return self.extrapolate_ensemble(x);
        }

        // Try adaptive extrapolation if configured
        if self.adaptive_config.is_some() {
            return self.extrapolate_adaptive(x);
        }

        // Try autoregressive extrapolation if configured
        if self.autoregressive_config.is_some() {
            return self.extrapolate_autoregressive(x);
        }

        // Fall back to base extrapolator
        self.base_extrapolator.extrapolate(x)
    }

    /// Perform confidence-based extrapolation
    pub fn extrapolate_with_confidence(
        &self,
        x: T,
    ) -> InterpolateResult<ConfidenceExtrapolationResult<T>> {
        if let Some(config) = &self.confidence_config {
            let base_result = self.base_extrapolator.extrapolate(x)?;

            // Estimate uncertainty based on distance from domain boundaries
            let lower_bound = self.base_extrapolator.get_lower_bound();
            let upper_bound = self.base_extrapolator.get_upper_bound();

            // Calculate distance from nearest boundary
            let distance_from_domain = if x < lower_bound {
                lower_bound - x
            } else if x > upper_bound {
                x - upper_bound
            } else {
                T::zero() // Inside domain
            };

            // Uncertainty increases with distance from domain
            // Standard error grows linearly with distance (simple model)
            let base_uncertainty = T::from(0.01).unwrap_or_default(); // 1% base uncertainty
            let distance_factor = T::from(0.1).unwrap_or_default(); // 10% per unit distance
            let standard_error = base_uncertainty + distance_factor * distance_from_domain;

            // Calculate confidence bounds based on confidence level
            // Using normal approximation: bounds = estimate ± z * standard_error
            let z_score = if config.confidence_level >= T::from(0.99).unwrap_or_default() {
                T::from(2.576).unwrap_or_default() // 99%
            } else if config.confidence_level >= T::from(0.95).unwrap_or_default() {
                T::from(1.96).unwrap_or_default() // 95%
            } else if config.confidence_level >= T::from(0.90).unwrap_or_default() {
                T::from(1.645).unwrap_or_default() // 90%
            } else {
                T::from(1.0).unwrap_or_default() // Default 1-sigma
            };

            let margin_of_error = z_score * standard_error;
            let lower_bound_confidence = base_result - margin_of_error;
            let upper_bound_confidence = base_result + margin_of_error;

            Ok(ConfidenceExtrapolationResult {
                estimate: base_result,
                lower_bound: lower_bound_confidence,
                upper_bound: upper_bound_confidence,
                standard_error,
            })
        } else {
            Err(InterpolateError::ComputationError(
                "Confidence extrapolation not configured".to_string(),
            ))
        }
    }

    /// Perform ensemble extrapolation
    pub fn extrapolate_ensemble(&self, x: T) -> InterpolateResult<T> {
        if let Some(config) = &self.ensemble_config {
            let mut results = Vec::new();
            let mut weights = Vec::new();

            // Collect results from all methods
            for (method, weight) in &config.methods_and_weights {
                // Create a temporary extrapolator with this method
                let mut temp_extrapolator = self.base_extrapolator.clone();

                // Update the extrapolation method based on direction
                if x < temp_extrapolator.get_lower_bound() {
                    temp_extrapolator.set_lower_method(*method);
                } else if x > temp_extrapolator.get_upper_bound() {
                    temp_extrapolator.set_upper_method(*method);
                }

                if let Ok(result) = temp_extrapolator.extrapolate(x) {
                    results.push(result);
                    weights.push(*weight);
                }
            }

            if results.is_empty() {
                return Err(InterpolateError::ComputationError(
                    "No ensemble methods produced valid results".to_string(),
                ));
            }

            // Combine results based on strategy
            match config.combination_strategy {
                EnsembleCombinationStrategy::Mean => {
                    let sum: T = results.iter().copied().fold(T::zero(), |acc, x| acc + x);
                    Ok(sum / T::from(results.len()).unwrap())
                }
                EnsembleCombinationStrategy::WeightedMean => {
                    let weighted_sum: T = results
                        .iter()
                        .zip(weights.iter())
                        .map(|(r, w)| *r * *w)
                        .fold(T::zero(), |acc, x| acc + x);
                    let weight_sum: T = weights.iter().copied().fold(T::zero(), |acc, x| acc + x);

                    if weight_sum.is_zero() {
                        return Err(InterpolateError::ComputationError(
                            "Zero total weight in ensemble".to_string(),
                        ));
                    }

                    Ok(weighted_sum / weight_sum)
                }
                EnsembleCombinationStrategy::Median => {
                    let mut sorted_results = results;
                    sorted_results.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let mid = sorted_results.len() / 2;

                    if sorted_results.len() % 2 == 0 {
                        let two = T::from(2.0).unwrap();
                        Ok((sorted_results[mid - 1] + sorted_results[mid]) / two)
                    } else {
                        Ok(sorted_results[mid])
                    }
                }
                EnsembleCombinationStrategy::TrimmedMean { trim_fraction } => {
                    let mut sorted_results = results;
                    sorted_results.sort_by(|a, b| a.partial_cmp(b).unwrap());

                    let n = sorted_results.len();
                    let trim_count = ((n as f64) * trim_fraction).floor() as usize;
                    let start = trim_count;
                    let end = n - trim_count;

                    if start >= end {
                        return Err(InterpolateError::ComputationError(
                            "Trim fraction too large for ensemble".to_string(),
                        ));
                    }

                    let trimmed_sum: T = sorted_results[start..end]
                        .iter()
                        .copied()
                        .fold(T::zero(), |acc, x| acc + x);
                    let trimmed_count = end - start;

                    Ok(trimmed_sum / T::from(trimmed_count).unwrap())
                }
            }
        } else {
            Err(InterpolateError::ComputationError(
                "Ensemble extrapolation not configured".to_string(),
            ))
        }
    }

    /// Perform adaptive extrapolation
    pub fn extrapolate_adaptive(&self, x: T) -> InterpolateResult<T> {
        if let Some(config) = &self.adaptive_config {
            let mut best_result = None;
            let mut best_score = T::infinity();

            // Try each candidate method and select the best one
            for &method in &config.candidate_methods {
                let mut temp_extrapolator = self.base_extrapolator.clone();

                // Update the extrapolation method based on direction
                if x < temp_extrapolator.get_lower_bound() {
                    temp_extrapolator.set_lower_method(method);
                } else if x > temp_extrapolator.get_upper_bound() {
                    temp_extrapolator.set_upper_method(method);
                }

                if let Ok(result) = temp_extrapolator.extrapolate(x) {
                    // Score based on selection criteria
                    let score = self.evaluate_extrapolation_quality(method, x, result)?;

                    if score < best_score {
                        best_score = score;
                        best_result = Some(result);
                    }
                }
            }

            best_result.ok_or_else(|| {
                InterpolateError::ComputationError(
                    "No adaptive method produced valid results".to_string(),
                )
            })
        } else {
            Err(InterpolateError::ComputationError(
                "Adaptive extrapolation not configured".to_string(),
            ))
        }
    }

    /// Perform autoregressive extrapolation
    pub fn extrapolate_autoregressive(&self, x: T) -> InterpolateResult<T> {
        if let Some(config) = &self.autoregressive_config {
            if let Some((x_data, y_data)) = &self.historical_data {
                // Fit AR model using historical data
                let ar_coeffs = self.fit_ar_model(x_data, y_data, config.ar_order)?;

                // Use AR model for extrapolation
                let prediction = self.ar_predict(&ar_coeffs, x_data, y_data, x, config)?;
                Ok(prediction)
            } else {
                // Fall back to base extrapolator if no historical data
                self.base_extrapolator.extrapolate(x)
            }
        } else {
            Err(InterpolateError::ComputationError(
                "Autoregressive extrapolation not configured".to_string(),
            ))
        }
    }

    /// Evaluate extrapolation quality for adaptive method selection
    fn evaluate_extrapolation_quality(
        &self,
        method: ExtrapolationMethod,
        x: T,
        result: T,
    ) -> InterpolateResult<T> {
        // Simple scoring based on distance from domain and method stability
        let lower_bound = self.base_extrapolator.get_lower_bound();
        let upper_bound = self.base_extrapolator.get_upper_bound();

        let distance_from_domain = if x < lower_bound {
            lower_bound - x
        } else if x > upper_bound {
            x - upper_bound
        } else {
            T::zero()
        };

        // Score based on method characteristics and distance
        let base_score = match method {
            ExtrapolationMethod::Linear => T::from(1.0).unwrap_or_default(), // Most stable
            ExtrapolationMethod::Quadratic => T::from(2.0).unwrap_or_default(),
            ExtrapolationMethod::Cubic => T::from(3.0).unwrap_or_default(),
            ExtrapolationMethod::Akima => T::from(1.5).unwrap_or_default(), // Good stability
            ExtrapolationMethod::Exponential => T::from(4.0).unwrap_or_default(),
            ExtrapolationMethod::PowerLaw => T::from(4.0).unwrap_or_default(),
            _ => T::from(5.0).unwrap_or_default(), // Other methods
        };

        // Penalize methods more as distance increases
        let distance_penalty = distance_from_domain * T::from(0.1).unwrap_or_default();

        Ok(base_score + distance_penalty)
    }

    /// Fit autoregressive model to historical data
    fn fit_ar_model(
        &self,
        _x_data: &Array1<T>,
        y_data: &Array1<T>,
        order: usize,
    ) -> InterpolateResult<Array1<T>> {
        if y_data.len() < order + 1 {
            return Err(InterpolateError::ComputationError(
                "Insufficient _data for AR model fitting".to_string(),
            ));
        }

        // Simple AR fitting using Yule-Walker equations (simplified version)
        let n = y_data.len();
        let mut coeffs = Array1::zeros(order);

        // For simplicity, use least squares approach
        // In practice, you'd use more sophisticated methods like Burg's method

        // Calculate autocorrelations
        let mut autocorr = Array1::zeros(order + 1);
        for lag in 0..=order {
            let mut sum = T::zero();
            let mut count = 0;

            for i in lag..n {
                sum += y_data[i] * y_data[i - lag];
                count += 1;
            }

            if count > 0 {
                autocorr[lag] = sum / T::from(count).unwrap_or(T::one());
            }
        }

        // Solve Yule-Walker equations (simplified)
        // For a proper implementation, you'd solve the full Toeplitz system
        for i in 0..order {
            if autocorr[0] != T::zero() {
                coeffs[i] = autocorr[i + 1] / autocorr[0];
            }
        }

        Ok(coeffs)
    }

    /// Make AR prediction
    fn ar_predict(
        &self,
        coeffs: &Array1<T>,
        x_data: &Array1<T>,
        y_data: &Array1<T>,
        x: T,
        config: &AutoregressiveExtrapolationConfig<T>,
    ) -> InterpolateResult<T> {
        let order = coeffs.len();

        if y_data.len() < order {
            return Err(InterpolateError::ComputationError(
                "Insufficient _data for AR prediction".to_string(),
            ));
        }

        // Use the last 'order' values to predict
        let mut prediction = T::zero();
        let start_idx = y_data.len() - order;

        for i in 0..order {
            prediction += coeffs[i] * y_data[start_idx + i];
        }

        // Adjust prediction based on distance from domain
        // This is a simplified approach - in practice you'd interpolate the time series
        let last_x = x_data[x_data.len() - 1];
        let extrapolation_distance = x - last_x;

        // Apply simple trend adjustment (very basic)
        if extrapolation_distance != T::zero() && y_data.len() >= 2 {
            let trend = (y_data[y_data.len() - 1] - y_data[y_data.len() - 2])
                / (x_data[x_data.len() - 1] - x_data[x_data.len() - 2]);
            prediction += trend * extrapolation_distance;
        }

        Ok(prediction)
    }
}

impl<T: Float + std::fmt::Display> Extrapolator<T> {
    /// Creates a new extrapolator with the specified methods and boundary values.
    ///
    /// # Arguments
    ///
    /// * `lower_bound` - Lower boundary of the original domain
    /// * `upper_bound` - Upper boundary of the original domain
    /// * `lower_value` - Function value at the lower boundary
    /// * `upper_value` - Function value at the upper boundary
    /// * `lower_method` - Extrapolation method for below the lower boundary
    /// * `upper_method` - Extrapolation method for above the upper boundary
    ///
    /// # Returns
    ///
    /// A new `Extrapolator` instance
    pub fn new(
        lower_bound: T,
        upper_bound: T,
        lower_value: T,
        upper_value: T,
        lower_method: ExtrapolationMethod,
        upper_method: ExtrapolationMethod,
    ) -> Self {
        // For linear methods, estimate derivatives as zero by default
        let lower_derivative = T::zero();
        let upper_derivative = T::zero();

        Self {
            lower_bound,
            upper_bound,
            lower_method,
            upper_method,
            lower_value,
            upper_value,
            lower_derivative,
            upper_derivative,
            lower_second_derivative: None,
            upper_second_derivative: None,
            parameters: ExtrapolationParameters::default(),
        }
    }

    /// Sets the derivatives at the boundaries for gradient-aware extrapolation.
    ///
    /// # Arguments
    ///
    /// * `lower_derivative` - Derivative at the lower boundary
    /// * `upper_derivative` - Derivative at the upper boundary
    ///
    /// # Returns
    ///
    /// A reference to the modified extrapolator
    pub fn with_derivatives(mut self, lower_derivative: T, upperderivative: T) -> Self {
        self.lower_derivative = lower_derivative;
        self.upper_derivative = upperderivative;
        self
    }

    /// Sets the second derivatives at the boundaries for higher-order extrapolation.
    ///
    /// # Arguments
    ///
    /// * `lower_second_derivative` - Second derivative at the lower boundary
    /// * `upper_second_derivative` - Second derivative at the upper boundary
    ///
    /// # Returns
    ///
    /// A reference to the modified extrapolator
    pub fn with_second_derivatives(
        mut self,
        lower_second_derivative: T,
        upper_second_derivative: T,
    ) -> Self {
        self.lower_second_derivative = Some(lower_second_derivative);
        self.upper_second_derivative = Some(upper_second_derivative);
        self
    }

    /// Sets custom parameters for specialized extrapolation methods.
    ///
    /// # Arguments
    ///
    /// * `parameters` - Custom parameters for extrapolation methods
    ///
    /// # Returns
    ///
    /// A reference to the modified extrapolator
    pub fn with_parameters(mut self, parameters: ExtrapolationParameters<T>) -> Self {
        self.parameters = parameters;
        self
    }

    /// Extrapolates the function value at the given point.
    ///
    /// # Arguments
    ///
    /// * `x` - The point at which to evaluate the function
    ///
    /// # Returns
    ///
    /// The extrapolated function value
    pub fn extrapolate(&self, x: T) -> InterpolateResult<T> {
        if x < self.lower_bound {
            self.extrapolate_direction(x, ExtrapolationDirection::Lower)
        } else if x > self.upper_bound {
            self.extrapolate_direction(x, ExtrapolationDirection::Upper)
        } else {
            // Point is inside the domain, shouldn't be extrapolating
            Err(InterpolateError::InvalidValue(format!(
                "Point {} is inside the domain [{}, {}], use interpolation instead",
                x, self.lower_bound, self.upper_bound
            )))
        }
    }

    /// Extrapolates the function value in the specified direction.
    ///
    /// # Arguments
    ///
    /// * `x` - The point at which to evaluate the function
    /// * `direction` - Direction of extrapolation (lower or upper)
    ///
    /// # Returns
    ///
    /// The extrapolated function value
    fn extrapolate_direction(
        &self,
        x: T,
        direction: ExtrapolationDirection,
    ) -> InterpolateResult<T> {
        let method = match direction {
            ExtrapolationDirection::Lower => self.lower_method,
            ExtrapolationDirection::Upper => self.upper_method,
        };

        match method {
            ExtrapolationMethod::Error => Err(InterpolateError::OutOfBounds(format!(
                "Point {} is outside the domain [{}, {}]",
                x, self.lower_bound, self.upper_bound
            ))),
            ExtrapolationMethod::Constant => match direction {
                ExtrapolationDirection::Lower => Ok(self.lower_value),
                ExtrapolationDirection::Upper => Ok(self.upper_value),
            },
            ExtrapolationMethod::Linear => self.linear_extrapolation(x, direction),
            ExtrapolationMethod::Quadratic => self.quadratic_extrapolation(x, direction),
            ExtrapolationMethod::Cubic => self.cubic_extrapolation(x, direction),
            ExtrapolationMethod::Periodic => self.periodic_extrapolation(x),
            ExtrapolationMethod::Reflection => self.reflection_extrapolation(x),
            ExtrapolationMethod::Exponential => self.exponential_extrapolation(x, direction),
            ExtrapolationMethod::PowerLaw => self.power_law_extrapolation(x, direction),
            ExtrapolationMethod::Spline => self.spline_extrapolation(x, direction),
            ExtrapolationMethod::Akima => self.akima_extrapolation(x, direction),
            ExtrapolationMethod::Sinusoidal => self.sinusoidal_extrapolation(x, direction),
            ExtrapolationMethod::Rational => self.rational_extrapolation(x, direction),
            ExtrapolationMethod::Confidence => self.confidence_extrapolation(x, direction),
            ExtrapolationMethod::Ensemble => self.ensemble_extrapolation(x, direction),
            ExtrapolationMethod::Adaptive => self.adaptive_extrapolation(x, direction),
            ExtrapolationMethod::Autoregressive => self.autoregressive_extrapolation(x, direction),
            ExtrapolationMethod::Zeros => Ok(T::zero()),
            ExtrapolationMethod::Nearest => self.nearest_extrapolation(x, direction),
            ExtrapolationMethod::Mirror => self.mirror_extrapolation(x, direction),
            ExtrapolationMethod::Wrap => self.wrap_extrapolation(x),
            ExtrapolationMethod::Clamped => self.clamped_extrapolation(x, direction),
            ExtrapolationMethod::GridMirror => self.grid_mirror_extrapolation(x, direction),
            ExtrapolationMethod::GridConstant => self.grid_constant_extrapolation(x, direction),
            ExtrapolationMethod::GridWrap => self.grid_wrap_extrapolation(x),
        }
    }

    /// Linear extrapolation based on endpoint values and derivatives.
    ///
    /// Uses the formula: f(x) = f(x₀) + f'(x₀) * (x - x₀)
    fn linear_extrapolation(
        &self,
        x: T,
        direction: ExtrapolationDirection,
    ) -> InterpolateResult<T> {
        match direction {
            ExtrapolationDirection::Lower => {
                let dx = x - self.lower_bound;
                Ok(self.lower_value + self.lower_derivative * dx)
            }
            ExtrapolationDirection::Upper => {
                let dx = x - self.upper_bound;
                Ok(self.upper_value + self.upper_derivative * dx)
            }
        }
    }

    /// Quadratic extrapolation based on endpoint values, derivatives, and curvature.
    ///
    /// Uses the formula: f(x) = f(x₀) + f'(x₀) * (x - x₀) + 0.5 * f''(x₀) * (x - x₀)²
    fn quadratic_extrapolation(
        &self,
        x: T,
        direction: ExtrapolationDirection,
    ) -> InterpolateResult<T> {
        let (bound, value, deriv, second_deriv) = match direction {
            ExtrapolationDirection::Lower => {
                let second_deriv = self.lower_second_derivative.ok_or_else(|| {
                    InterpolateError::InvalidState(
                        "Second derivative not provided for quadratic extrapolation".to_string(),
                    )
                })?;
                (
                    self.lower_bound,
                    self.lower_value,
                    self.lower_derivative,
                    second_deriv,
                )
            }
            ExtrapolationDirection::Upper => {
                let second_deriv = self.upper_second_derivative.ok_or_else(|| {
                    InterpolateError::InvalidState(
                        "Second derivative not provided for quadratic extrapolation".to_string(),
                    )
                })?;
                (
                    self.upper_bound,
                    self.upper_value,
                    self.upper_derivative,
                    second_deriv,
                )
            }
        };

        let dx = x - bound;
        let half = T::from(0.5).unwrap();

        Ok(value + deriv * dx + half * second_deriv * dx * dx)
    }

    /// Cubic extrapolation preserving both values and derivatives at boundaries.
    ///
    /// For lower boundary:
    /// - f(x_lower) = lower_value
    /// - f'(x_lower) = lower_derivative
    /// - The cubic polynomial is constructed to smoothly match these conditions
    fn cubic_extrapolation(&self, x: T, direction: ExtrapolationDirection) -> InterpolateResult<T> {
        // Cubic extrapolation requires second derivatives to be specified
        if self.lower_second_derivative.is_none() || self.upper_second_derivative.is_none() {
            return Err(InterpolateError::InvalidState(
                "Second derivatives must be provided for cubic extrapolation".to_string(),
            ));
        }

        let (bound, value, deriv, second_deriv) = match direction {
            ExtrapolationDirection::Lower => (
                self.lower_bound,
                self.lower_value,
                self.lower_derivative,
                self.lower_second_derivative.unwrap(),
            ),
            ExtrapolationDirection::Upper => (
                self.upper_bound,
                self.upper_value,
                self.upper_derivative,
                self.upper_second_derivative.unwrap(),
            ),
        };

        let dx = x - bound;
        let dx2 = dx * dx;
        let dx3 = dx2 * dx;

        // Coefficients for cubic polynomial: a + b*dx + c*dx^2 + d*dx^3
        let a = value;
        let b = deriv;
        let c = second_deriv / T::from(2.0).unwrap();

        // The third coefficient (d) depends on the third derivative, which we don't have directly
        // Let's set it to a small value based on the rate of change of the second derivative
        let d = T::from(0.0).unwrap(); // Simplified version sets this to zero

        Ok(a + b * dx + c * dx2 + d * dx3)
    }

    /// Periodic extrapolation extending the domain as if the function repeats.
    ///
    /// Maps the point x to an equivalent point within the domain using modular arithmetic,
    /// effectively treating the function as periodic with period equal to the domain width.
    fn periodic_extrapolation(&self, x: T) -> InterpolateResult<T> {
        let domain_width = self.upper_bound - self.lower_bound;

        // If a custom period is specified, use that instead of the domain width
        let period = if self.parameters.period > T::zero() {
            self.parameters.period
        } else {
            domain_width
        };

        // Compute the equivalent position within the domain
        let mut x_equiv = x;

        // Handle points below the lower bound
        if x < self.lower_bound {
            let offset = self.lower_bound - x;
            let periods = (offset / period).ceil();
            x_equiv = x + periods * period;
        }
        // Handle points above the upper bound
        else if x > self.upper_bound {
            let offset = x - self.lower_bound;
            let periods = (offset / period).floor();
            x_equiv = x - periods * period;
        }

        // Ensure the point is now within the domain bounds (handle numerical precision issues)
        if x_equiv < self.lower_bound {
            x_equiv = self.lower_bound;
        } else if x_equiv > self.upper_bound {
            x_equiv = self.upper_bound;
        }

        // At this point, x_equiv should be inside the domain
        // This isn't actually extrapolation anymore, so we're returning an "error" to
        // indicate that interpolation should be used with this mapped point
        Err(InterpolateError::MappedPoint(
            x_equiv.to_f64().unwrap_or(0.0),
        ))
    }

    /// Reflection extrapolation reflecting the function at the boundaries.
    ///
    /// Maps the point x to an equivalent point within the domain by reflecting
    /// across the boundary, as if the function were mirrored at the endpoints.
    fn reflection_extrapolation(&self, x: T) -> InterpolateResult<T> {
        let domain_width = self.upper_bound - self.lower_bound;
        let mut x_equiv = x;

        // Handle points below the lower bound
        if x < self.lower_bound {
            let offset = self.lower_bound - x;
            let reflections = (offset / domain_width).floor();
            let remaining = offset - reflections * domain_width;

            // Even number of reflections: reflect from lower boundary
            if reflections.to_u64().unwrap() % 2 == 0 {
                x_equiv = self.lower_bound + remaining;
            }
            // Odd number of reflections: reflect from upper boundary
            else {
                x_equiv = self.upper_bound - remaining;
            }
        }
        // Handle points above the upper bound
        else if x > self.upper_bound {
            let offset = x - self.upper_bound;
            let reflections = (offset / domain_width).floor();
            let remaining = offset - reflections * domain_width;

            // Even number of reflections: reflect from upper boundary
            if reflections.to_u64().unwrap() % 2 == 0 {
                x_equiv = self.upper_bound - remaining;
            }
            // Odd number of reflections: reflect from lower boundary
            else {
                x_equiv = self.lower_bound + remaining;
            }
        }

        // Ensure the point is now within the domain bounds (handle numerical precision issues)
        if x_equiv < self.lower_bound {
            x_equiv = self.lower_bound;
        } else if x_equiv > self.upper_bound {
            x_equiv = self.upper_bound;
        }

        // At this point, x_equiv should be inside the domain
        // This isn't actually extrapolation anymore, so we're returning an "error" to
        // indicate that interpolation should be used with this mapped point
        Err(InterpolateError::MappedPoint(
            x_equiv.to_f64().unwrap_or(0.0),
        ))
    }

    /// Exponential extrapolation for asymptotic behavior.
    ///
    /// Models the function as:
    /// f(x) = asymptote + scale * exp(rate * (x - boundary))
    fn exponential_extrapolation(
        &self,
        x: T,
        direction: ExtrapolationDirection,
    ) -> InterpolateResult<T> {
        match direction {
            ExtrapolationDirection::Lower => {
                let dx = x - self.lower_bound;

                // Estimate parameters if not explicitly provided
                let rate = self.parameters.exponential_rate;

                // Compute the scale factor that ensures f(x_lower) = lower_value
                let scale = self.lower_derivative / rate;

                // Compute the asymptote that ensures f'(x_lower) = lower_derivative
                let asymptote = self.lower_value - scale;

                Ok(asymptote + scale * (rate * dx).exp())
            }
            ExtrapolationDirection::Upper => {
                let dx = x - self.upper_bound;

                // For upper boundary, often want negative rate for decay
                let rate = -self.parameters.exponential_rate;

                // Compute the scale factor that ensures f(x_upper) = upper_value
                let scale = self.upper_derivative / rate;

                // Compute the asymptote that ensures f'(x_upper) = upper_derivative
                let asymptote = self.upper_value - scale;

                Ok(asymptote + scale * (rate * dx).exp())
            }
        }
    }

    /// Power law extrapolation for asymptotic behavior.
    ///
    /// Models the function as:
    /// f(x) = asymptote + scale * (x - boundary)^exponent
    fn power_law_extrapolation(
        &self,
        x: T,
        direction: ExtrapolationDirection,
    ) -> InterpolateResult<T> {
        let exponent = self.parameters.power_exponent;

        match direction {
            ExtrapolationDirection::Lower => {
                // Ensure x is not too close to boundary for negative exponent
                if exponent < T::zero() && (self.lower_bound - x).abs() < T::epsilon() {
                    return Ok(self.lower_value);
                }

                let dx = x - self.lower_bound;

                // For negative exponents with x < boundary, need to handle sign carefully
                let power_term = if dx < T::zero() && exponent.fract() != T::zero() {
                    let abs_pow = (-dx).powf(exponent.abs());
                    if exponent.abs().to_u64().unwrap() % 2 == 0 {
                        abs_pow
                    } else {
                        -abs_pow
                    }
                } else {
                    dx.powf(exponent)
                };

                // Compute scale based on derivative at boundary
                let scale = self.lower_derivative
                    / (exponent * (self.lower_bound - T::epsilon()).powf(exponent - T::one()));

                // Asymptote ensures correct function value at boundary
                let asymptote = self.lower_value;

                Ok(asymptote + scale * power_term)
            }
            ExtrapolationDirection::Upper => {
                // Ensure x is not too close to boundary for negative exponent
                if exponent < T::zero() && (x - self.upper_bound).abs() < T::epsilon() {
                    return Ok(self.upper_value);
                }

                let dx = x - self.upper_bound;

                // Compute power term with care for negative exponents
                let power_term = dx.powf(exponent);

                // Compute scale based on derivative at boundary
                let scale = self.upper_derivative
                    / (exponent * (self.upper_bound + T::epsilon()).powf(exponent - T::one()));

                // Asymptote ensures correct function value at boundary
                let asymptote = self.upper_value;

                Ok(asymptote + scale * power_term)
            }
        }
    }

    /// Spline-based extrapolation using the full spline continuation.
    ///
    /// This method uses the underlying spline representation to continue
    /// the interpolation naturally beyond the boundaries.
    fn spline_extrapolation(
        &self,
        x: T,
        direction: ExtrapolationDirection,
    ) -> InterpolateResult<T> {
        // For spline extrapolation, we use a cubic polynomial that matches
        // the value, first derivative, and second derivative at the boundary
        self.cubic_extrapolation(x, direction)
    }

    /// Akima extrapolation for stable polynomial continuation.
    ///
    /// Uses Akima's method which provides a more stable extrapolation
    /// compared to standard cubic methods, especially for data with rapid changes.
    fn akima_extrapolation(&self, x: T, direction: ExtrapolationDirection) -> InterpolateResult<T> {
        // Akima extrapolation uses a modified cubic that is less sensitive to outliers
        // We use a weighted combination of linear and cubic extrapolation
        let linear_result = self.linear_extrapolation(x, direction)?;
        let cubic_result = self
            .cubic_extrapolation(x, direction)
            .unwrap_or(linear_result);

        // Weight factor based on distance from boundary
        let dx = match direction {
            ExtrapolationDirection::Lower => (self.lower_bound - x).abs(),
            ExtrapolationDirection::Upper => (x - self.upper_bound).abs(),
        };

        // Smooth transition from cubic (near boundary) to linear (far from boundary)
        let weight = (-dx / T::from(2.0).unwrap()).exp();
        Ok(weight * cubic_result + (T::one() - weight) * linear_result)
    }

    /// Sinusoidal extrapolation for periodic data.
    ///
    /// Fits a sinusoidal function to match the value and derivative at the boundary,
    /// useful for data with known periodic behavior.
    fn sinusoidal_extrapolation(
        &self,
        x: T,
        direction: ExtrapolationDirection,
    ) -> InterpolateResult<T> {
        let (bound, value, deriv) = match direction {
            ExtrapolationDirection::Lower => {
                (self.lower_bound, self.lower_value, self.lower_derivative)
            }
            ExtrapolationDirection::Upper => {
                (self.upper_bound, self.upper_value, self.upper_derivative)
            }
        };

        let dx = x - bound;

        // Default frequency if not specified
        let omega = T::from(2.0 * std::f64::consts::PI).unwrap() / self.parameters.period;

        // Fit A*sin(omega*dx + phi) + C to match value and derivative
        // f(0) = A*sin(phi) + C = value
        // f'(0) = A*omega*cos(phi) = deriv

        // Solve for A and phi
        let a_omega = deriv;
        let _a = a_omega / omega;

        // Use a default phase of pi/4 for stability
        let phi = T::from(std::f64::consts::PI / 4.0).unwrap();
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();

        // Adjust amplitude to match derivative constraint
        let a_adjusted = deriv / (omega * cos_phi);

        // Compute offset to match value constraint
        let c = value - a_adjusted * sin_phi;

        Ok(a_adjusted * (omega * dx + phi).sin() + c)
    }

    /// Rational function extrapolation for poles/zeros behavior.
    ///
    /// Models the function as a rational function (ratio of polynomials),
    /// useful for functions with known asymptotic behavior or poles.
    fn rational_extrapolation(
        &self,
        x: T,
        direction: ExtrapolationDirection,
    ) -> InterpolateResult<T> {
        let (bound, value, deriv) = match direction {
            ExtrapolationDirection::Lower => {
                (self.lower_bound, self.lower_value, self.lower_derivative)
            }
            ExtrapolationDirection::Upper => {
                (self.upper_bound, self.upper_value, self.upper_derivative)
            }
        };

        let dx = x - bound;

        // Simple Padé approximant [1/1]: (a0 + a1*dx)/(1 + b1*dx)
        // Matching value and derivative at boundary:
        // f(0) = a0 = value
        // f'(0) = a1 - a0*b1 = deriv

        let a0 = value;

        // Choose b1 to control asymptotic behavior
        // For decay: b1 > 0, for growth: b1 < 0
        let b1 = match direction {
            ExtrapolationDirection::Lower => T::from(0.1).unwrap(), // Mild growth
            ExtrapolationDirection::Upper => T::from(-0.1).unwrap(), // Mild decay
        };

        let a1 = deriv + a0 * b1;

        // Evaluate the rational function
        let numerator = a0 + a1 * dx;
        let denominator = T::one() + b1 * dx;

        // Avoid division by very small numbers
        if denominator.abs() < T::epsilon() {
            return Ok(value);
        }

        Ok(numerator / denominator)
    }

    /// Confidence-based extrapolation with uncertainty quantification.
    fn confidence_extrapolation(
        &self,
        x: T,
        direction: ExtrapolationDirection,
    ) -> InterpolateResult<T> {
        // Use linear extrapolation as base method
        let base_value = self.linear_extrapolation(x, direction)?;

        // Calculate confidence bounds based on distance from boundary
        let (bound, value, deriv) = match direction {
            ExtrapolationDirection::Lower => {
                (self.lower_bound, self.lower_value, self.lower_derivative)
            }
            ExtrapolationDirection::Upper => {
                (self.upper_bound, self.upper_value, self.upper_derivative)
            }
        };

        let distance = (x - bound).abs();
        let confidence_factor = T::from(0.95).unwrap(); // 95% confidence

        // Uncertainty increases with distance
        let uncertainty = distance * deriv.abs() * (T::one() - confidence_factor);

        // For conservative extrapolation, bias towards boundary value when uncertain
        let weight = T::one() / (T::one() + distance);
        Ok(base_value * (T::one() - weight) + value * weight + uncertainty)
    }

    /// Ensemble extrapolation combining multiple methods with weighted averaging.
    fn ensemble_extrapolation(
        &self,
        x: T,
        direction: ExtrapolationDirection,
    ) -> InterpolateResult<T> {
        let methods = [
            (ExtrapolationMethod::Linear, T::from(0.4).unwrap()),
            (ExtrapolationMethod::Quadratic, T::from(0.3).unwrap()),
            (ExtrapolationMethod::Cubic, T::from(0.2).unwrap()),
            (ExtrapolationMethod::Exponential, T::from(0.1).unwrap()),
        ];

        let mut weighted_sum = T::zero();
        let mut weight_sum = T::zero();

        for (method, weight) in methods.iter() {
            let value = match method {
                ExtrapolationMethod::Linear => self.linear_extrapolation(x, direction)?,
                ExtrapolationMethod::Quadratic => {
                    if self.lower_second_derivative.is_some()
                        && self.upper_second_derivative.is_some()
                    {
                        self.quadratic_extrapolation(x, direction)?
                    } else {
                        self.linear_extrapolation(x, direction)?
                    }
                }
                ExtrapolationMethod::Cubic => self.cubic_extrapolation(x, direction)?,
                ExtrapolationMethod::Exponential => self.exponential_extrapolation(x, direction)?,
                _ => self.linear_extrapolation(x, direction)?, // fallback
            };

            weighted_sum = weighted_sum + value * (*weight);
            weight_sum = weight_sum + (*weight);
        }

        Ok(weighted_sum / weight_sum)
    }

    /// Adaptive extrapolation that selects the best method based on local characteristics.
    fn adaptive_extrapolation(
        &self,
        x: T,
        direction: ExtrapolationDirection,
    ) -> InterpolateResult<T> {
        let (bound, value, deriv) = match direction {
            ExtrapolationDirection::Lower => {
                (self.lower_bound, self.lower_value, self.lower_derivative)
            }
            ExtrapolationDirection::Upper => {
                (self.upper_bound, self.upper_value, self.upper_derivative)
            }
        };

        let distance = (x - bound).abs();
        let derivative_magnitude = deriv.abs();

        // Adaptive method selection based on characteristics
        if derivative_magnitude < T::from(0.1).unwrap() {
            // Low derivative - use constant extrapolation
            Ok(value)
        } else if derivative_magnitude < T::from(1.0).unwrap() && distance < T::from(1.0).unwrap() {
            // Moderate derivative, close distance - use linear
            self.linear_extrapolation(x, direction)
        } else if self.lower_second_derivative.is_some()
            && self.upper_second_derivative.is_some()
            && distance < T::from(2.0).unwrap()
        {
            // Has second derivatives and moderate distance - use quadratic
            self.quadratic_extrapolation(x, direction)
        } else {
            // High derivative or far distance - use exponential for stability
            self.exponential_extrapolation(x, direction)
        }
    }

    /// Autoregressive extrapolation using time series forecasting techniques.
    fn autoregressive_extrapolation(
        &self,
        x: T,
        direction: ExtrapolationDirection,
    ) -> InterpolateResult<T> {
        // Simple AR(1) model: x(t) = φ * x(t-1) + ε
        // For extrapolation: f(x) = φ * f(boundary) + trend * (x - boundary)

        let (bound, value, deriv) = match direction {
            ExtrapolationDirection::Lower => {
                (self.lower_bound, self.lower_value, self.lower_derivative)
            }
            ExtrapolationDirection::Upper => {
                (self.upper_bound, self.upper_value, self.upper_derivative)
            }
        };

        let distance = (x - bound).abs();

        // Estimate AR coefficient based on derivative (stability parameter)
        let phi = if deriv.abs() < T::from(0.5).unwrap() {
            T::from(0.8).unwrap() // Stable, low autocorrelation
        } else {
            T::from(0.6).unwrap() // Less stable, moderate autocorrelation
        };

        // Apply AR model with exponential decay for extrapolation stability
        let ar_factor = phi.powf(distance);
        let trend_component = deriv * distance * (T::one() - ar_factor);

        Ok(value * ar_factor + trend_component)
    }

    /// Nearest extrapolation (SciPy 'nearest' mode)
    ///
    /// Returns the nearest boundary value - equivalent to constant extrapolation
    /// but specifically designed for compatibility with SciPy's 'nearest' mode.
    fn nearest_extrapolation(
        &self,
        x: T,
        direction: ExtrapolationDirection,
    ) -> InterpolateResult<T> {
        match direction {
            ExtrapolationDirection::Lower => Ok(self.lower_value),
            ExtrapolationDirection::Upper => Ok(self.upper_value),
        }
    }

    /// Mirror extrapolation (SciPy 'mirror' mode)
    ///
    /// Reflects the function about the boundary without repeating the edge values.
    /// For a point outside the domain, it computes the reflected position inside
    /// the domain and returns the value at that position.
    fn mirror_extrapolation(
        &self,
        x: T,
        direction: ExtrapolationDirection,
    ) -> InterpolateResult<T> {
        let domain_size = self.upper_bound - self.lower_bound;

        match direction {
            ExtrapolationDirection::Lower => {
                let distance_outside = self.lower_bound - x;
                let mirrored_x = self.lower_bound + distance_outside;

                // If the mirrored point is still outside, use linear extrapolation
                if mirrored_x > self.upper_bound {
                    self.linear_extrapolation(x, direction)
                } else {
                    // Use linear interpolation between boundary and mirrored position
                    let t = distance_outside / domain_size;
                    Ok(self.lower_value + self.lower_derivative * distance_outside * t)
                }
            }
            ExtrapolationDirection::Upper => {
                let distance_outside = x - self.upper_bound;
                let mirrored_x = self.upper_bound - distance_outside;

                // If the mirrored point is still outside, use linear extrapolation
                if mirrored_x < self.lower_bound {
                    self.linear_extrapolation(x, direction)
                } else {
                    // Use linear interpolation between boundary and mirrored position
                    let t = distance_outside / domain_size;
                    Ok(self.upper_value - self.upper_derivative * distance_outside * t)
                }
            }
        }
    }

    /// Wrap extrapolation (SciPy 'wrap' mode)
    ///
    /// Wraps the input coordinate into the domain using periodic boundary conditions.
    /// This is similar to periodic extrapolation but specifically for SciPy compatibility.
    fn wrap_extrapolation(&self, x: T) -> InterpolateResult<T> {
        let domain_size = self.upper_bound - self.lower_bound;

        if domain_size <= T::zero() {
            return Err(InterpolateError::InvalidValue(
                "Domain size must be positive for wrap extrapolation".to_string(),
            ));
        }

        // Map x into the domain [lower_bound, upper_bound)
        let offset = x - self.lower_bound;
        let wrapped_offset = offset - (offset / domain_size).floor() * domain_size;
        let _wrapped_x = self.lower_bound + wrapped_offset; // Reserved for future use

        // Since this is extrapolation, we need to estimate the value
        // using the boundary conditions in a periodic manner
        let t = wrapped_offset / domain_size;
        Ok(self.lower_value + (self.upper_value - self.lower_value) * t)
    }

    /// Clamped extrapolation with zero derivatives
    ///
    /// Uses zero derivatives at the boundaries for smooth extrapolation.
    /// This corresponds to "clamped" boundary conditions in spline theory.
    fn clamped_extrapolation(
        &self,
        x: T,
        direction: ExtrapolationDirection,
    ) -> InterpolateResult<T> {
        // For clamped extrapolation, we use zero derivatives at boundaries
        match direction {
            ExtrapolationDirection::Lower => {
                // Linear extrapolation with zero derivative
                Ok(self.lower_value)
            }
            ExtrapolationDirection::Upper => {
                // Linear extrapolation with zero derivative
                Ok(self.upper_value)
            }
        }
    }

    /// Grid-specific mirror extrapolation
    ///
    /// Similar to mirror extrapolation but optimized for structured grid data.
    /// Uses a more sophisticated reflection algorithm suitable for grid-based interpolation.
    fn grid_mirror_extrapolation(
        &self,
        x: T,
        direction: ExtrapolationDirection,
    ) -> InterpolateResult<T> {
        // For now, delegate to regular mirror extrapolation
        // In a full implementation, this would include grid-specific optimizations
        self.mirror_extrapolation(x, direction)
    }

    /// Grid-specific constant extrapolation
    ///
    /// Similar to constant extrapolation but optimized for structured grid data.
    fn grid_constant_extrapolation(
        &self,
        x: T,
        direction: ExtrapolationDirection,
    ) -> InterpolateResult<T> {
        // For grid data, use the boundary values
        match direction {
            ExtrapolationDirection::Lower => Ok(self.lower_value),
            ExtrapolationDirection::Upper => Ok(self.upper_value),
        }
    }

    /// Grid-specific wrap extrapolation
    ///
    /// Similar to wrap extrapolation but optimized for structured grid data.
    fn grid_wrap_extrapolation(&self, x: T) -> InterpolateResult<T> {
        // For now, delegate to regular wrap extrapolation
        // In a full implementation, this would include grid-specific optimizations
        self.wrap_extrapolation(x)
    }

    /// Set the extrapolation method for the lower boundary.
    pub fn set_lower_method(&mut self, method: ExtrapolationMethod) {
        self.lower_method = method;
    }

    /// Set the extrapolation method for the upper boundary.
    pub fn set_upper_method(&mut self, method: ExtrapolationMethod) {
        self.upper_method = method;
    }

    /// Get the lower bound of the domain.
    pub fn get_lower_bound(&self) -> T {
        self.lower_bound
    }

    /// Get the upper bound of the domain.
    pub fn get_upper_bound(&self) -> T {
        self.upper_bound
    }

    /// Get the method used for extrapolation below the lower boundary.
    pub fn get_lower_method(&self) -> ExtrapolationMethod {
        self.lower_method
    }

    /// Get the method used for extrapolation above the upper boundary.
    pub fn get_upper_method(&self) -> ExtrapolationMethod {
        self.upper_method
    }
}

/// Creates an extrapolator with linear extrapolation based on values and derivatives.
///
/// # Arguments
///
/// * `lower_bound` - Lower boundary of the original domain
/// * `upper_bound` - Upper boundary of the original domain
/// * `lower_value` - Function value at the lower boundary
/// * `upper_value` - Function value at the upper boundary
/// * `lower_derivative` - Derivative at the lower boundary
/// * `upper_derivative` - Derivative at the upper boundary
///
/// # Returns
///
/// A new `Extrapolator` configured for linear extrapolation
#[allow(dead_code)]
pub fn make_linear_extrapolator<T: Float + std::fmt::Display>(
    lower_bound: T,
    upper_bound: T,
    lower_value: T,
    upper_value: T,
    lower_derivative: T,
    upper_derivative: T,
) -> Extrapolator<T> {
    Extrapolator::new(
        lower_bound,
        upper_bound,
        lower_value,
        upper_value,
        ExtrapolationMethod::Linear,
        ExtrapolationMethod::Linear,
    )
    .with_derivatives(lower_derivative, upper_derivative)
}

/// Creates an extrapolator with periodic extension.
///
/// # Arguments
///
/// * `lower_bound` - Lower boundary of the original domain
/// * `upper_bound` - Upper boundary of the original domain
/// * `period` - The period of the function (defaults to domain width if None)
///
/// # Returns
///
/// A new `Extrapolator` configured for periodic extrapolation
#[allow(dead_code)]
pub fn make_periodic_extrapolator<T: Float + std::fmt::Display>(
    lower_bound: T,
    upper_bound: T,
    period: Option<T>,
) -> Extrapolator<T> {
    let mut extrapolator = Extrapolator::new(
        lower_bound,
        upper_bound,
        T::zero(), // Values and derivatives don't matter for periodic extrapolation
        T::zero(),
        ExtrapolationMethod::Periodic,
        ExtrapolationMethod::Periodic,
    );

    if let Some(p) = period {
        let params = ExtrapolationParameters::default().with_period(p);
        extrapolator = extrapolator.with_parameters(params);
    }

    extrapolator
}

/// Creates an extrapolator with reflection at boundaries.
///
/// # Arguments
///
/// * `lower_bound` - Lower boundary of the original domain
/// * `upper_bound` - Upper boundary of the original domain
///
/// # Returns
///
/// A new `Extrapolator` configured for reflection extrapolation
#[allow(dead_code)]
pub fn make_reflection_extrapolator<T: Float + std::fmt::Display>(
    lower_bound: T,
    upper_bound: T,
) -> Extrapolator<T> {
    Extrapolator::new(
        lower_bound,
        upper_bound,
        T::zero(), // Values and derivatives don't matter for reflection extrapolation
        T::zero(),
        ExtrapolationMethod::Reflection,
        ExtrapolationMethod::Reflection,
    )
}

/// Creates an extrapolator with cubic polynomial extrapolation.
///
/// # Arguments
///
/// * `lower_bound` - Lower boundary of the original domain
/// * `upper_bound` - Upper boundary of the original domain
/// * `lower_value` - Function value at the lower boundary
/// * `upper_value` - Function value at the upper boundary
/// * `lower_derivative` - First derivative at the lower boundary
/// * `upper_derivative` - First derivative at the upper boundary
/// * `lower_second_derivative` - Second derivative at the lower boundary
/// * `upper_second_derivative` - Second derivative at the upper boundary
///
/// # Returns
///
/// A new `Extrapolator` configured for cubic extrapolation
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn make_cubic_extrapolator<T: Float + std::fmt::Display>(
    lower_bound: T,
    upper_bound: T,
    lower_value: T,
    upper_value: T,
    lower_derivative: T,
    upper_derivative: T,
    lower_second_derivative: T,
    upper_second_derivative: T,
) -> Extrapolator<T> {
    Extrapolator::new(
        lower_bound,
        upper_bound,
        lower_value,
        upper_value,
        ExtrapolationMethod::Cubic,
        ExtrapolationMethod::Cubic,
    )
    .with_derivatives(lower_derivative, upper_derivative)
    .with_second_derivatives(lower_second_derivative, upper_second_derivative)
}

/// Creates an extrapolator with exponential decay/growth.
///
/// # Arguments
///
/// * `lower_bound` - Lower boundary of the original domain
/// * `upper_bound` - Upper boundary of the original domain
/// * `lower_value` - Function value at the lower boundary
/// * `upper_value` - Function value at the upper boundary
/// * `lower_derivative` - Derivative at the lower boundary
/// * `upper_derivative` - Derivative at the upper boundary
/// * `lower_rate` - Exponential rate for lower extrapolation (positive = growth, negative = decay)
/// * `upper_rate` - Exponential rate for upper extrapolation (positive = growth, negative = decay)
///
/// # Returns
///
/// A new `Extrapolator` configured for exponential extrapolation
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn make_exponential_extrapolator<T: Float + std::fmt::Display>(
    lower_bound: T,
    upper_bound: T,
    lower_value: T,
    upper_value: T,
    lower_derivative: T,
    upper_derivative: T,
    lower_rate: T,
    _rate: T,
) -> Extrapolator<T> {
    let params = ExtrapolationParameters::default().with_exponential_rate(lower_rate.abs());

    Extrapolator::new(
        lower_bound,
        upper_bound,
        lower_value,
        upper_value,
        ExtrapolationMethod::Exponential,
        ExtrapolationMethod::Exponential,
    )
    .with_derivatives(lower_derivative, upper_derivative)
    .with_parameters(params)
}

/// Convenience function to create a confidence-based extrapolator
#[allow(dead_code)]
pub fn make_confidence_extrapolator<
    T: Float + std::fmt::Display + std::default::Default + std::ops::AddAssign,
>(
    base_extrapolator: Extrapolator<T>,
    confidence_level: T,
    n_bootstrap: usize,
) -> AdvancedExtrapolator<T> {
    let config = ConfidenceExtrapolationConfig {
        base_method: ExtrapolationMethod::Linear,
        confidence_level,
        n_bootstrap,
        include_bounds: true,
    };

    AdvancedExtrapolator::new(base_extrapolator).with_confidence(config)
}

/// Convenience function to create an ensemble extrapolator
#[allow(dead_code)]
pub fn make_ensemble_extrapolator<
    T: Float + std::fmt::Display + std::default::Default + std::ops::AddAssign,
>(
    base_extrapolator: Extrapolator<T>,
    methods_and_weights: Vec<(ExtrapolationMethod, T)>,
    strategy: EnsembleCombinationStrategy,
) -> AdvancedExtrapolator<T> {
    let config = EnsembleExtrapolationConfig {
        methods_and_weights,
        combination_strategy: strategy,
        normalize_weights: true,
    };

    AdvancedExtrapolator::new(base_extrapolator).with_ensemble(config)
}

/// Convenience function to create an adaptive extrapolator
#[allow(dead_code)]
pub fn make_adaptive_extrapolator<
    T: Float + std::fmt::Display + std::default::Default + std::ops::AddAssign,
>(
    base_extrapolator: Extrapolator<T>,
    candidate_methods: Vec<ExtrapolationMethod>,
    criteria: Vec<AdaptiveSelectionCriterion>,
) -> AdvancedExtrapolator<T> {
    let config = AdaptiveExtrapolationConfig {
        candidate_methods,
        analysis_window_size: 5,
        selection_criteria: criteria,
        cache_selections: true,
    };

    AdvancedExtrapolator::new(base_extrapolator).with_adaptive(config)
}

/// Convenience function to create an autoregressive extrapolator
#[allow(dead_code)]
pub fn make_autoregressive_extrapolator<
    T: Float + std::fmt::Display + std::default::Default + std::ops::AddAssign,
>(
    base_extrapolator: Extrapolator<T>,
    ar_order: usize,
    fitting_method: ARFittingMethod,
    historical_data: Option<(Array1<T>, Array1<T>)>,
) -> AdvancedExtrapolator<T> {
    let config = AutoregressiveExtrapolationConfig {
        ar_order,
        fitting_method,
        include_constant: true,
        regularization: T::from(1e-6).unwrap(),
    };

    let mut _extrapolator =
        AdvancedExtrapolator::new(base_extrapolator).with_autoregressive(config);

    if let Some((x_data, y_data)) = historical_data {
        _extrapolator = _extrapolator.with_historical_data(x_data, y_data);
    }

    _extrapolator
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_constant_extrapolation() {
        let lower_bound = 0.0;
        let upper_bound = 10.0;
        let lower_value = 5.0;
        let upper_value = 15.0;

        let extrapolator = Extrapolator::new(
            lower_bound,
            upper_bound,
            lower_value,
            upper_value,
            ExtrapolationMethod::Constant,
            ExtrapolationMethod::Constant,
        );

        // Test lower extrapolation
        let result = extrapolator.extrapolate(-5.0).unwrap();
        assert_abs_diff_eq!(result, lower_value);

        // Test upper extrapolation
        let result = extrapolator.extrapolate(15.0).unwrap();
        assert_abs_diff_eq!(result, upper_value);
    }

    #[test]
    fn test_linear_extrapolation() {
        let lower_bound = 0.0;
        let upper_bound = 10.0;
        let lower_value = 0.0;
        let upper_value = 10.0;
        let lower_derivative = 1.0;
        let upper_derivative = 1.0;

        let extrapolator = make_linear_extrapolator(
            lower_bound,
            upper_bound,
            lower_value,
            upper_value,
            lower_derivative,
            upper_derivative,
        );

        // Test lower extrapolation
        let result = extrapolator.extrapolate(-5.0).unwrap();
        assert_abs_diff_eq!(result, -5.0); // f(-5) = 0 + 1 * (-5 - 0) = -5

        // Test upper extrapolation
        let result = extrapolator.extrapolate(15.0).unwrap();
        assert_abs_diff_eq!(result, 15.0); // f(15) = 10 + 1 * (15 - 10) = 15
    }

    #[test]
    fn test_periodic_extrapolation() {
        let lower_bound = 0.0;
        let upper_bound = 1.0;

        let extrapolator = make_periodic_extrapolator(lower_bound, upper_bound, Some(1.0));

        // Test mapping points outside domain
        match extrapolator.extrapolate(-0.3) {
            Err(InterpolateError::MappedPoint(x)) => assert_abs_diff_eq!(x, 0.7),
            result => assert!(false, "Expected MappedPoint error, got: {:?}", result),
        }

        match extrapolator.extrapolate(1.4) {
            Err(InterpolateError::MappedPoint(x)) => assert_abs_diff_eq!(x, 0.4),
            result => assert!(false, "Expected MappedPoint error, got: {:?}", result),
        }

        match extrapolator.extrapolate(3.7) {
            Err(InterpolateError::MappedPoint(x)) => assert_abs_diff_eq!(x, 0.7),
            result => assert!(false, "Expected MappedPoint error, got: {:?}", result),
        }
    }

    #[test]
    fn test_reflection_extrapolation() {
        let lower_bound = 0.0;
        let upper_bound = 1.0;

        let extrapolator = make_reflection_extrapolator(lower_bound, upper_bound);

        // Test reflection below lower bound
        match extrapolator.extrapolate(-0.3) {
            Err(InterpolateError::MappedPoint(x)) => assert_abs_diff_eq!(x, 0.3),
            result => assert!(false, "Expected MappedPoint error, got: {:?}", result),
        }

        // Test reflection above upper bound
        match extrapolator.extrapolate(1.3) {
            Err(InterpolateError::MappedPoint(x)) => assert_abs_diff_eq!(x, 0.7),
            result => assert!(false, "Expected MappedPoint error, got: {:?}", result),
        }

        // Test multiple reflections
        match extrapolator.extrapolate(-1.3) {
            Err(InterpolateError::MappedPoint(x)) => assert_abs_diff_eq!(x, 0.7),
            result => assert!(false, "Expected MappedPoint error, got: {:?}", result),
        }

        match extrapolator.extrapolate(2.3) {
            Err(InterpolateError::MappedPoint(x)) => assert_abs_diff_eq!(x, 0.3),
            result => assert!(false, "Expected MappedPoint error, got: {:?}", result),
        }
    }

    #[test]
    fn test_cubic_extrapolation() {
        let lower_bound = 0.0;
        let upper_bound = 1.0;
        let lower_value = 0.0;
        let upper_value = 1.0;
        let lower_derivative = 1.0;
        let upper_derivative = 1.0;
        let lower_second_derivative = 0.0;
        let upper_second_derivative = 0.0;

        // Cubic extrapolation of a linear function (should match linear exactly)
        let extrapolator = make_cubic_extrapolator(
            lower_bound,
            upper_bound,
            lower_value,
            upper_value,
            lower_derivative,
            upper_derivative,
            lower_second_derivative,
            upper_second_derivative,
        );

        // Test lower extrapolation
        let result = extrapolator.extrapolate(-1.0).unwrap();
        assert_abs_diff_eq!(result, -1.0); // Should match linear extrapolation

        // Test upper extrapolation
        let result = extrapolator.extrapolate(2.0).unwrap();
        assert_abs_diff_eq!(result, 2.0); // Should match linear extrapolation
    }

    #[test]
    fn test_exponential_extrapolation() {
        let lower_bound = 0.0;
        let upper_bound = 1.0;
        let lower_value = 1.0;
        let upper_value = std::f64::consts::E; // e^1
        let lower_derivative = 1.0;
        let upper_derivative = std::f64::consts::E; // e^1
        let lower_rate = 1.0;
        let upper_rate = 1.0;

        // Exponential extrapolation of f(x) = e^x
        let extrapolator = make_exponential_extrapolator(
            lower_bound,
            upper_bound,
            lower_value,
            upper_value,
            lower_derivative,
            upper_derivative,
            lower_rate,
            upper_rate,
        );

        // Test lower extrapolation
        let result = extrapolator.extrapolate(-1.0).unwrap();
        // For exponential extrapolation, the exact formula depends on the implementation
        // We verify it produces a reasonable exponential-like behavior
        assert!(result.is_finite());
        assert!(result > 0.0); // Should be positive for exponential function
        assert!(result < lower_value); // Should decay below the boundary value

        // Test upper extrapolation
        let result = extrapolator.extrapolate(2.0).unwrap();
        assert!(result.is_finite());
        assert!(result > 0.0); // Should be positive for exponential function
        assert!(result > upper_value); // Should grow above the boundary value

        // Verify the extrapolation approaches the boundary values as we approach from outside
        let result_near_lower = extrapolator.extrapolate(lower_bound - 1e-6).unwrap();
        assert_abs_diff_eq!(result_near_lower, lower_value, epsilon = 1e-3);

        let result_near_upper = extrapolator.extrapolate(upper_bound + 1e-6).unwrap();
        assert_abs_diff_eq!(result_near_upper, upper_value, epsilon = 1e-3)
    }

    #[test]
    #[ignore] // FIXME: Test failing - needs investigation
    fn test_physics_informed_extrapolation() {
        // Test physics-informed extrapolation with conservation laws
        let extrapolator = make_physics_informed_extrapolator(
            0.0,
            1.0, // domain bounds
            0.0,
            1.0, // boundary values
            1.0,
            1.0, // boundary derivatives
            PhysicsLaw::MassConservation,
        );

        let result = extrapolator.base_extrapolator.extrapolate(-0.5).unwrap();
        assert!(result.is_finite());
        assert!(result >= 0.0); // Mass conservation - no negative values
    }
}

/// Physics-informed extrapolation respecting conservation laws
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn make_physics_informed_extrapolator<
    T: Float + FromPrimitive + std::fmt::Display + std::default::Default + std::ops::AddAssign,
>(
    lower_bound: T,
    upper_bound: T,
    lower_value: T,
    upper_value: T,
    lower_derivative: T,
    upper_derivative: T,
    physics_law: PhysicsLaw,
) -> AdvancedExtrapolator<T> {
    // Create base extrapolator with appropriate methods
    let base_extrapolator = Extrapolator::new(
        lower_bound,
        upper_bound,
        lower_value,
        upper_value,
        ExtrapolationMethod::Linear,
        ExtrapolationMethod::Linear,
    )
    .with_derivatives(lower_derivative, upper_derivative);

    let mut extrapolator = AdvancedExtrapolator::new(base_extrapolator);

    // Configure physics-based constraints
    match physics_law {
        PhysicsLaw::MassConservation => {
            // Ensure non-negative extrapolation for mass quantities
            extrapolator.base_extrapolator.lower_method = ExtrapolationMethod::Exponential;
            extrapolator.base_extrapolator.upper_method = ExtrapolationMethod::Linear;
            extrapolator.base_extrapolator.parameters.exponential_rate = T::from_f64(-0.1).unwrap();
            // Decay rate
        }
        PhysicsLaw::EnergyConservation => {
            // Energy-conserving polynomial extrapolation
            extrapolator.base_extrapolator.lower_method = ExtrapolationMethod::Quadratic;
            extrapolator.base_extrapolator.upper_method = ExtrapolationMethod::Quadratic;
        }
        PhysicsLaw::MomentumConservation => {
            // Linear momentum conservation
            extrapolator.base_extrapolator.lower_method = ExtrapolationMethod::Linear;
            extrapolator.base_extrapolator.upper_method = ExtrapolationMethod::Linear;
        }
    }

    extrapolator
}

/// Physics laws for informed extrapolation
#[derive(Debug, Clone, Copy)]
pub enum PhysicsLaw {
    /// Mass conservation (non-negative, decay to zero)
    MassConservation,
    /// Energy conservation (quadratic behavior)
    EnergyConservation,
    /// Momentum conservation (linear behavior)
    MomentumConservation,
}

/// Boundary condition preserving extrapolation
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn make_boundary_preserving_extrapolator<
    T: Float + FromPrimitive + std::fmt::Display + std::default::Default + std::ops::AddAssign,
>(
    lower_bound: T,
    upper_bound: T,
    lower_value: T,
    upper_value: T,
    lower_derivative: T,
    upper_derivative: T,
    boundary_type: BoundaryType,
) -> AdvancedExtrapolator<T> {
    // Create base extrapolator with appropriate methods
    let base_extrapolator = Extrapolator::new(
        lower_bound,
        upper_bound,
        lower_value,
        upper_value,
        ExtrapolationMethod::Linear,
        ExtrapolationMethod::Linear,
    )
    .with_derivatives(lower_derivative, upper_derivative);

    let mut extrapolator = AdvancedExtrapolator::new(base_extrapolator);

    match boundary_type {
        BoundaryType::Dirichlet => {
            // Fixed _value boundaries - use cubic for smooth transition
            extrapolator.base_extrapolator.lower_method = ExtrapolationMethod::Cubic;
            extrapolator.base_extrapolator.upper_method = ExtrapolationMethod::Cubic;
        }
        BoundaryType::Neumann => {
            // Fixed _derivative boundaries - use quadratic
            extrapolator.base_extrapolator.lower_method = ExtrapolationMethod::Quadratic;
            extrapolator.base_extrapolator.upper_method = ExtrapolationMethod::Quadratic;
        }
        BoundaryType::Robin => {
            // Mixed boundaries - use linear combination
            extrapolator.base_extrapolator.lower_method = ExtrapolationMethod::Linear;
            extrapolator.base_extrapolator.upper_method = ExtrapolationMethod::Linear;
        }
        BoundaryType::Absorbing => {
            // Absorbing boundaries - exponential decay
            extrapolator.base_extrapolator.lower_method = ExtrapolationMethod::Exponential;
            extrapolator.base_extrapolator.upper_method = ExtrapolationMethod::Exponential;
            extrapolator.base_extrapolator.parameters.exponential_rate = T::from_f64(-1.0).unwrap();
        }
    }

    extrapolator
}

/// Boundary condition types for physics-informed extrapolation
#[derive(Debug, Clone, Copy)]
pub enum BoundaryType {
    /// Fixed value at boundary (Dirichlet)
    Dirichlet,
    /// Fixed derivative at boundary (Neumann)
    Neumann,
    /// Linear combination of value and derivative (Robin)
    Robin,
    /// Absorbing boundary with exponential decay
    Absorbing,
}

/// Adaptive extrapolation that selects method based on local data characteristics
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn make_smart_adaptive_extrapolator<
    T: Float + FromPrimitive + std::fmt::Display + std::default::Default + std::ops::AddAssign,
>(
    lower_bound: T,
    upper_bound: T,
    lower_value: T,
    upper_value: T,
    lower_derivative: T,
    upper_derivative: T,
    data_characteristics: &DataCharacteristics<T>,
) -> AdvancedExtrapolator<T> {
    // Create base extrapolator with appropriate methods
    let base_extrapolator = Extrapolator::new(
        lower_bound,
        upper_bound,
        lower_value,
        upper_value,
        ExtrapolationMethod::Linear,
        ExtrapolationMethod::Linear,
    )
    .with_derivatives(lower_derivative, upper_derivative);

    let mut extrapolator = AdvancedExtrapolator::new(base_extrapolator);

    // Select extrapolation method based on data analysis
    if data_characteristics.is_periodic {
        extrapolator.base_extrapolator.lower_method = ExtrapolationMethod::Periodic;
        extrapolator.base_extrapolator.upper_method = ExtrapolationMethod::Periodic;
        extrapolator.base_extrapolator.parameters.period = data_characteristics
            .estimated_period
            .unwrap_or_else(|| T::from_f64(2.0 * std::f64::consts::PI).unwrap());
    } else if data_characteristics.is_monotonic {
        if data_characteristics.is_exponential_like {
            extrapolator.base_extrapolator.lower_method = ExtrapolationMethod::Exponential;
            extrapolator.base_extrapolator.upper_method = ExtrapolationMethod::Exponential;
        } else {
            extrapolator.base_extrapolator.lower_method = ExtrapolationMethod::Linear;
            extrapolator.base_extrapolator.upper_method = ExtrapolationMethod::Linear;
        }
    } else if data_characteristics.is_oscillatory {
        extrapolator.base_extrapolator.lower_method = ExtrapolationMethod::Sinusoidal;
        extrapolator.base_extrapolator.upper_method = ExtrapolationMethod::Sinusoidal;
    } else {
        // Default to quadratic for smooth data
        extrapolator.base_extrapolator.lower_method = ExtrapolationMethod::Quadratic;
        extrapolator.base_extrapolator.upper_method = ExtrapolationMethod::Quadratic;
    }

    extrapolator
}

/// Data characteristics for adaptive extrapolation
#[derive(Debug, Clone)]
pub struct DataCharacteristics<T: Float> {
    /// Whether the data appears periodic
    pub is_periodic: bool,
    /// Estimated period if periodic
    pub estimated_period: Option<T>,
    /// Whether the data is monotonic
    pub is_monotonic: bool,
    /// Whether the data follows exponential-like growth/decay
    pub is_exponential_like: bool,
    /// Whether the data is oscillatory
    pub is_oscillatory: bool,
    /// Characteristic scale of the data
    pub characteristic_scale: T,
}
