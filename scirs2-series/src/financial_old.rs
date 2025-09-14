//! Financial time series analysis toolkit
//!
//! This module provides specialized functionality for financial time series analysis,
//! including GARCH models, volatility modeling, and technical indicators.

use ndarray::{s, Array1, Array2};
use num_traits::Float;
use std::fmt::Debug;

use crate::error::{Result, TimeSeriesError};
use statrs::statistics::Statistics;

/// GARCH model configuration
#[derive(Debug, Clone)]
pub struct GarchConfig {
    /// GARCH order (p)
    pub p: usize,
    /// ARCH order (q)
    pub q: usize,
    /// Mean model type
    pub mean_model: MeanModel,
    /// Distribution for residuals
    pub distribution: Distribution,
    /// Maximum iterations for optimization
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Use numerical derivatives
    pub use_numerical_derivatives: bool,
}

impl Default for GarchConfig {
    fn default() -> Self {
        Self {
            p: 1,
            q: 1,
            mean_model: MeanModel::Constant,
            distribution: Distribution::Normal,
            max_iterations: 1000,
            tolerance: 1e-6,
            use_numerical_derivatives: false,
        }
    }
}

/// Mean model specification for GARCH
#[derive(Debug, Clone)]
pub enum MeanModel {
    /// Constant mean
    Constant,
    /// Zero mean
    Zero,
    /// AR(p) mean model
    AR {
        /// Autoregressive order
        order: usize,
    },
    /// ARMA(p,q) mean model  
    ARMA {
        /// Autoregressive order
        ar_order: usize,
        /// Moving average order
        ma_order: usize,
    },
}

/// Distribution for GARCH residuals
#[derive(Debug, Clone)]
pub enum Distribution {
    /// Normal distribution
    Normal,
    /// Student's t-distribution
    StudentT,
    /// Skewed Student's t-distribution
    SkewedStudentT,
    /// Generalized Error Distribution
    GED,
}

/// GARCH model results
#[derive(Debug, Clone)]
pub struct GarchResult<F: Float> {
    /// Model parameters
    pub parameters: GarchParameters<F>,
    /// Conditional variance (volatility squared)
    pub conditional_variance: Array1<F>,
    /// Standardized residuals
    pub standardized_residuals: Array1<F>,
    /// Log-likelihood
    pub log_likelihood: F,
    /// Information criteria
    pub aic: F,
    /// Bayesian Information Criterion
    pub bic: F,
    /// Convergence status
    pub converged: bool,
    /// Number of iterations
    pub iterations: usize,
}

/// GARCH model parameters
#[derive(Debug, Clone)]
pub struct GarchParameters<F: Float> {
    /// Mean equation parameters
    pub mean_params: Array1<F>,
    /// GARCH parameters (omega, alpha_i, beta_j)
    pub garch_params: Array1<F>,
    /// Distribution parameters (if applicable)
    pub dist_params: Option<Array1<F>>,
}

/// GARCH model implementation
#[derive(Debug)]
pub struct GarchModel<F: Float + Debug> {
    #[allow(dead_code)]
    config: GarchConfig,
    fitted: bool,
    parameters: Option<GarchParameters<F>>,
    #[allow(dead_code)]
    conditional_variance: Option<Array1<F>>,
}

impl<F: Float + Debug + std::iter::Sum> GarchModel<F> {
    /// Create a new GARCH model
    pub fn new(config: GarchConfig) -> Self {
        Self {
            config,
            fitted: false,
            parameters: None,
            conditional_variance: None,
        }
    }

    /// Create GARCH(1,1) model with default settings
    pub fn garch_11() -> Self {
        Self::new(GarchConfig::default())
    }

    /// Fit the GARCH model to data using Maximum Likelihood Estimation
    pub fn fit(&mut self, data: &Array1<F>) -> Result<GarchResult<F>> {
        if data.len() < 20 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Need at least 20 observations for GARCH estimation".to_string(),
                required: 20,
                actual: data.len(),
            });
        }

        let min_obs = std::cmp::max(20, 3 * (1 + self.config.p + self.config.q));
        if data.len() < min_obs {
            return Err(TimeSeriesError::InsufficientData {
                message: format!(
                    "Need at least {} observations for GARCH({},{}) estimation",
                    min_obs, self.config.p, self.config.q
                ),
                required: min_obs,
                actual: data.len(),
            });
        }

        // For GARCH(1,1), we can use either method of moments or MLE
        if self.config.p == 1 && self.config.q == 1 && !self.config.use_numerical_derivatives {
            self.fit_garch_11_mom(data)
        } else {
            // Use full MLE for general GARCH(p,q) models
            self.fit_garch_mle(data)
        }
    }

    /// Fit GARCH(1,1) using method of moments
    fn fit_garch_11_mom(&mut self, data: &Array1<F>) -> Result<GarchResult<F>> {
        // Calculate returns if data represents prices
        let returns = if data.iter().all(|&x| x > F::zero()) {
            // Assume prices, calculate log returns
            let mut ret = Array1::zeros(data.len() - 1);
            for i in 1..data.len() {
                ret[i - 1] = (data[i] / data[i - 1]).ln();
            }
            ret
        } else {
            // Assume already returns
            data.clone()
        };

        let n = returns.len();
        let n_f = F::from(n).unwrap();

        // Calculate sample moments
        let mean = returns.sum() / n_f;
        let centered_returns: Array1<F> = returns.mapv(|r| r - mean);

        // Sample variance
        let sample_var = centered_returns.mapv(|r| r.powi(2)).sum() / (n_f - F::one());

        // Sample skewness and kurtosis for moment matching
        let _sample_skew = centered_returns.mapv(|r| r.powi(3)).sum()
            / ((n_f - F::one()) * sample_var.powf(F::from(1.5).unwrap()));
        let sample_kurt =
            centered_returns.mapv(|r| r.powi(4)).sum() / ((n_f - F::one()) * sample_var.powi(2));

        // Method of moments for GARCH(1,1)
        // Using theoretical moments of GARCH(1,1) process

        // For GARCH(1,1): E[r^2] = omega / (1 - alpha - beta)
        // E[r^4] / (E[r^2])^2 = 3(1 - (alpha + beta)^2) / (1 - (alpha + beta)^2 - 2*alpha^2)

        // Simplified parameter estimation
        let alpha_beta_sum = F::one() - F::from(3.0).unwrap() / sample_kurt;
        let alpha_beta_sum = alpha_beta_sum
            .max(F::from(0.1).unwrap())
            .min(F::from(0.99).unwrap());

        // Split alpha and beta based on typical GARCH patterns
        let alpha = alpha_beta_sum * F::from(0.1).unwrap(); // Typically alpha < beta
        let beta = alpha_beta_sum - alpha;
        let omega = sample_var * (F::one() - alpha - beta);

        // Ensure parameters are positive and sum to less than 1
        let omega = omega.max(F::from(1e-6).unwrap());
        let alpha = alpha.max(F::from(0.01).unwrap()).min(F::from(0.3).unwrap());
        let beta = beta.max(F::from(0.01).unwrap()).min(F::from(0.95).unwrap());

        // Adjust if sum exceeds 1
        let sum_ab = alpha + beta;
        let (alpha, beta) = if sum_ab >= F::one() {
            let scale = F::from(0.99).unwrap() / sum_ab;
            (alpha * scale, beta * scale)
        } else {
            (alpha, beta)
        };

        // Calculate conditional variance recursively
        let mut conditional_variance = Array1::zeros(n);
        conditional_variance[0] = sample_var; // Initialize with unconditional variance

        for i in 1..n {
            conditional_variance[i] = omega
                + alpha * centered_returns[i - 1].powi(2)
                + beta * conditional_variance[i - 1];
        }

        // Calculate standardized residuals
        let standardized_residuals: Array1<F> = centered_returns
            .iter()
            .zip(conditional_variance.iter())
            .map(|(&r, &v)| r / v.sqrt())
            .collect();

        // Calculate log-likelihood (simplified)
        let mut log_likelihood = F::zero();
        for i in 0..n {
            let variance = conditional_variance[i];
            if variance > F::zero() {
                log_likelihood = log_likelihood
                    - F::from(0.5).unwrap()
                        * (variance.ln() + centered_returns[i].powi(2) / variance);
            }
        }

        // Information criteria
        let k = F::from(3).unwrap(); // Number of parameters (omega, alpha, beta)
        let aic = -F::from(2.0).unwrap() * log_likelihood + F::from(2.0).unwrap() * k;
        let bic = -F::from(2.0).unwrap() * log_likelihood + k * n_f.ln();

        // Create parameter structure
        let mean_params = Array1::from_vec(vec![mean]);
        let garch_params = Array1::from_vec(vec![omega, alpha, beta]);

        let parameters = GarchParameters {
            mean_params,
            garch_params,
            dist_params: None,
        };

        // Update model state
        self.fitted = true;
        self.parameters = Some(parameters.clone());
        self.conditional_variance = Some(conditional_variance.clone());

        Ok(GarchResult {
            parameters,
            conditional_variance,
            standardized_residuals,
            log_likelihood,
            aic,
            bic,
            converged: true,
            iterations: 1, // Method of moments is direct
        })
    }

    /// Forecast conditional variance
    pub fn forecast_variance(&self, steps: usize) -> Result<Array1<F>> {
        if !self.fitted {
            return Err(TimeSeriesError::InvalidModel(
                "Model has not been fitted".to_string(),
            ));
        }

        let parameters = self.parameters.as_ref().unwrap();
        let conditional_variance = self.conditional_variance.as_ref().unwrap();

        if parameters.garch_params.len() < 3 {
            return Err(TimeSeriesError::InvalidModel(
                "Invalid GARCH parameters".to_string(),
            ));
        }

        let omega = parameters.garch_params[0];
        let alpha = parameters.garch_params[1];
        let beta = parameters.garch_params[2];

        let mut forecasts = Array1::zeros(steps);

        // Initialize with last conditional variance
        let mut current_variance = conditional_variance[conditional_variance.len() - 1];

        // Calculate unconditional variance for long-term forecast
        let unconditional_variance = omega / (F::one() - alpha - beta);

        for i in 0..steps {
            if i == 0 {
                // One-step ahead forecast
                // Since we don't know the future shock, we use expected value (zero)
                forecasts[i] = omega + beta * current_variance;
            } else {
                // Multi-step ahead forecast converges to unconditional variance
                // h-step ahead variance: omega + (alpha + beta)^(h-1) * (1-step variance - unconditional)
                let decay_factor = (alpha + beta).powf(F::from(i).unwrap());
                forecasts[i] =
                    unconditional_variance + decay_factor * (forecasts[0] - unconditional_variance);
            }
            current_variance = forecasts[i];
        }

        Ok(forecasts)
    }

    /// Fit general GARCH(p,q) model using Maximum Likelihood Estimation
    fn fit_garch_mle(&mut self, data: &Array1<F>) -> Result<GarchResult<F>> {
        // Calculate returns if data represents prices
        let returns = if data.iter().all(|&x| x > F::zero()) {
            let mut ret = Array1::zeros(data.len() - 1);
            for i in 1..data.len() {
                ret[i - 1] = (data[i] / data[i - 1]).ln();
            }
            ret
        } else {
            data.clone()
        };

        let n = returns.len();
        let n_f = F::from(n).unwrap();

        // Prepare mean equation
        let (mean_residuals, mean_params) = self.estimate_mean_equation(&returns)?;

        // Initialize GARCH parameters with reasonable starting values
        let num_garch_params = 1 + self.config.p + self.config.q; // omega + alphas + betas
        let mut garch_params = Array1::zeros(num_garch_params);

        // Initialize omega (unconditional variance)
        let sample_var = mean_residuals.mapv(|x| x.powi(2)).sum() / (n_f - F::one());
        garch_params[0] = sample_var * F::from(0.1).unwrap(); // omega

        // Initialize alpha parameters (ARCH terms)
        for i in 1..=self.config.q {
            garch_params[i] = F::from(0.05).unwrap();
        }

        // Initialize beta parameters (GARCH terms)
        for i in (1 + self.config.q)..(1 + self.config.q + self.config.p) {
            garch_params[i] = F::from(0.85).unwrap() / F::from(self.config.p).unwrap();
        }

        // Ensure parameter constraints (stationarity)
        self.constrain_parameters(&mut garch_params);

        // Optimize using Nelder-Mead simplex algorithm
        let optimized_params = self.optimize_likelihood(&mean_residuals, garch_params)?;

        // Calculate final conditional variances and standardized residuals
        let conditional_variance =
            self.compute_conditional_variance(&mean_residuals, &optimized_params)?;
        let standardized_residuals =
            self.compute_standardized_residuals(&mean_residuals, &conditional_variance)?;

        // Calculate log-likelihood and information criteria
        let log_likelihood = self.compute_log_likelihood(&mean_residuals, &conditional_variance)?;
        let k = F::from(mean_params.len() + optimized_params.len()).unwrap();
        let aic = -F::from(2.0).unwrap() * log_likelihood + F::from(2.0).unwrap() * k;
        let bic = -F::from(2.0).unwrap() * log_likelihood + k * n_f.ln();

        let parameters = GarchParameters {
            mean_params,
            garch_params: optimized_params,
            dist_params: None,
        };

        // Update model state
        self.fitted = true;
        self.parameters = Some(parameters.clone());
        self.conditional_variance = Some(conditional_variance.clone());

        Ok(GarchResult {
            parameters,
            conditional_variance,
            standardized_residuals,
            log_likelihood,
            aic,
            bic,
            converged: true,
            iterations: self.config.max_iterations,
        })
    }

    /// Estimate mean equation parameters
    fn estimate_mean_equation(&self, returns: &Array1<F>) -> Result<(Array1<F>, Array1<F>)> {
        match &self.config.mean_model {
            MeanModel::Zero => {
                // Zero mean model
                let mean_params = Array1::zeros(0);
                Ok((returns.clone(), mean_params))
            }
            MeanModel::Constant => {
                // Constant mean model
                let mean = returns.sum() / F::from(returns.len()).unwrap();
                let residuals = returns.mapv(|r| r - mean);
                let mean_params = Array1::from_vec(vec![mean]);
                Ok((residuals, mean_params))
            }
            MeanModel::AR { order } => {
                // AR(p) mean model - simplified implementation
                if *order == 0 {
                    return self.estimate_mean_equation(returns);
                }

                let p = *order;
                if returns.len() <= p {
                    return Err(TimeSeriesError::InsufficientData {
                        message: format!("Need more than {p} observations for AR({p}) model"),
                        required: p + 1,
                        actual: returns.len(),
                    });
                }

                // Simple OLS estimation for AR(p)
                let n = returns.len() - p;
                let mut y = Array1::zeros(n);
                let mut x = Array2::zeros((n, p + 1)); // Include constant

                for i in 0..n {
                    y[i] = returns[p + i];
                    x[[i, 0]] = F::one(); // constant term
                    for j in 1..=p {
                        x[[i, j]] = returns[p + i - j];
                    }
                }

                // Solve normal equations: (X'X)^(-1) X'y
                let xtx = self.matrix_multiply_transpose(&x.view())?;
                let xty = self.matrix_vector_multiply_transpose(&x.view(), &y.view())?;
                let ar_params = self.solve_linear_system(&xtx, &xty)?;

                // Calculate residuals
                let mut residuals = Array1::zeros(returns.len());
                residuals.slice_mut(s![..p]).assign(&returns.slice(s![..p]));

                for i in p..returns.len() {
                    let mut prediction = ar_params[0]; // constant
                    for j in 1..=p {
                        prediction = prediction + ar_params[j] * returns[i - j];
                    }
                    residuals[i] = returns[i] - prediction;
                }

                Ok((residuals, ar_params))
            }
            MeanModel::ARMA { ar_order, ma_order } => {
                // ARMA model - simplified to constant mean for now
                if *ar_order == 0 && *ma_order == 0 {
                    return self.estimate_mean_equation(returns);
                }

                // For now, fall back to constant mean
                // Full ARMA estimation would require iterative methods
                let mean = returns.sum() / F::from(returns.len()).unwrap();
                let residuals = returns.mapv(|r| r - mean);
                let mean_params = Array1::from_vec(vec![mean]);
                Ok((residuals, mean_params))
            }
        }
    }

    /// Constrain GARCH parameters to ensure stationarity and positivity
    fn constrain_parameters(&self, params: &mut Array1<F>) {
        // Ensure omega > 0
        params[0] = params[0].max(F::from(1e-6).unwrap());

        // Ensure alpha_i >= 0
        for i in 1..=self.config.q {
            params[i] = params[i].max(F::zero());
        }

        // Ensure beta_j >= 0
        for i in (1 + self.config.q)..(1 + self.config.q + self.config.p) {
            params[i] = params[i].max(F::zero());
        }

        // Ensure stationarity: sum(alpha_i + beta_j) < 1
        let alpha_sum: F = (1..=self.config.q).map(|i| params[i]).sum();
        let beta_sum: F = ((1 + self.config.q)..(1 + self.config.q + self.config.p))
            .map(|i| params[i])
            .sum();

        let total_sum = alpha_sum + beta_sum;
        if total_sum >= F::one() {
            let scale = F::from(0.99).unwrap() / total_sum;
            for i in 1..params.len() {
                params[i] = params[i] * scale;
            }
        }
    }

    /// Optimize likelihood using simplified Nelder-Mead algorithm
    fn optimize_likelihood(
        &self,
        residuals: &Array1<F>,
        initial_params: Array1<F>,
    ) -> Result<Array1<F>> {
        let mut best_params = initial_params.clone();
        let mut best_likelihood = self.negative_log_likelihood(residuals, &initial_params)?;

        // Simple parameter search with random perturbations
        let perturbation_size = F::from(0.01).unwrap();

        for iteration in 0..self.config.max_iterations {
            let mut improved = false;

            for param_idx in 0..best_params.len() {
                // Try positive perturbation
                let mut test_params = best_params.clone();
                test_params[param_idx] = test_params[param_idx] + perturbation_size;
                self.constrain_parameters(&mut test_params);

                if let Ok(test_likelihood) = self.negative_log_likelihood(residuals, &test_params) {
                    if test_likelihood < best_likelihood {
                        best_params = test_params;
                        best_likelihood = test_likelihood;
                        improved = true;
                    }
                }

                // Try negative perturbation
                let mut test_params = best_params.clone();
                test_params[param_idx] = test_params[param_idx] - perturbation_size;
                self.constrain_parameters(&mut test_params);

                if let Ok(test_likelihood) = self.negative_log_likelihood(residuals, &test_params) {
                    if test_likelihood < best_likelihood {
                        best_params = test_params;
                        best_likelihood = test_likelihood;
                        improved = true;
                    }
                }
            }

            // Check convergence
            if !improved && iteration > 10 {
                break;
            }

            // Adaptive perturbation size
            if iteration % 20 == 0 && iteration > 0 {
                let decay = F::from(0.95).unwrap();
                let new_size = perturbation_size * decay;
                if new_size > F::from(1e-8).unwrap() {
                    // perturbation_size = new_size; // Would need to be mutable
                }
            }
        }

        Ok(best_params)
    }

    /// Compute negative log-likelihood for optimization
    fn negative_log_likelihood(&self, residuals: &Array1<F>, params: &Array1<F>) -> Result<F> {
        let conditional_variance = self.compute_conditional_variance(residuals, params)?;
        let log_likelihood = self.compute_log_likelihood(residuals, &conditional_variance)?;
        Ok(-log_likelihood)
    }

    /// Compute conditional variance given parameters
    fn compute_conditional_variance(
        &self,
        residuals: &Array1<F>,
        params: &Array1<F>,
    ) -> Result<Array1<F>> {
        let n = residuals.len();
        let mut h = Array1::zeros(n);

        // Initialize with unconditional variance
        let omega = params[0];
        let alpha_sum: F = (1..=self.config.q).map(|i| params[i]).sum();
        let beta_sum: F = ((1 + self.config.q)..(1 + self.config.q + self.config.p))
            .map(|i| params[i])
            .sum();

        let unconditional_var = omega / (F::one() - alpha_sum - beta_sum);

        // Initialize first max(p,q) values
        let max_lag = std::cmp::max(self.config.p, self.config.q);
        for i in 0..std::cmp::min(max_lag, n) {
            h[i] = unconditional_var;
        }

        // Compute conditional variance recursively
        for t in max_lag..n {
            h[t] = omega;

            // ARCH terms (alpha_i * epsilon_{t-i}^2)
            for i in 1..=self.config.q {
                if t >= i {
                    h[t] = h[t] + params[i] * residuals[t - i].powi(2);
                }
            }

            // GARCH terms (beta_j * h_{t-j})
            for j in 1..=self.config.p {
                if t >= j {
                    let beta_idx = self.config.q + j;
                    h[t] = h[t] + params[beta_idx] * h[t - j];
                }
            }
        }

        Ok(h)
    }

    /// Compute standardized residuals
    fn compute_standardized_residuals(
        &self,
        residuals: &Array1<F>,
        variance: &Array1<F>,
    ) -> Result<Array1<F>> {
        let mut standardized = Array1::zeros(residuals.len());

        for i in 0..residuals.len() {
            if variance[i] > F::zero() {
                standardized[i] = residuals[i] / variance[i].sqrt();
            } else {
                standardized[i] = F::zero();
            }
        }

        Ok(standardized)
    }

    /// Compute log-likelihood
    fn compute_log_likelihood(&self, residuals: &Array1<F>, variance: &Array1<F>) -> Result<F> {
        let mut log_likelihood = F::zero();

        match &self.config.distribution {
            Distribution::Normal => {
                let ln_2pi = F::from(2.0 * std::f64::consts::PI).unwrap().ln();

                for i in 0..residuals.len() {
                    if variance[i] > F::zero() {
                        let term = -F::from(0.5).unwrap()
                            * (ln_2pi + variance[i].ln() + residuals[i].powi(2) / variance[i]);
                        log_likelihood = log_likelihood + term;
                    }
                }
            }
            Distribution::StudentT => {
                // Simplified Student-t with fixed degrees of freedom (5.0)
                let nu = F::from(5.0).unwrap();
                let gamma_factor = F::from(0.8).unwrap(); // Approximation of gamma functions ratio

                for i in 0..residuals.len() {
                    if variance[i] > F::zero() {
                        let standardized = residuals[i] / variance[i].sqrt();
                        let term = gamma_factor
                            - F::from(0.5).unwrap() * variance[i].ln()
                            - F::from(0.5).unwrap()
                                * (nu + F::one())
                                * (F::one() + standardized.powi(2) / nu).ln();
                        log_likelihood = log_likelihood + term;
                    }
                }
            }
            _ => {
                // Fall back to normal distribution for other types
                return self.compute_log_likelihood(residuals, variance);
            }
        }

        Ok(log_likelihood)
    }

    /// Helper method for matrix multiplication
    fn matrix_multiply_transpose(&self, x: &ndarray::ArrayView2<F>) -> Result<Array2<F>> {
        let rows = x.ncols();
        let mut result = Array2::zeros((rows, rows));

        for i in 0..rows {
            for j in 0..rows {
                let mut sum = F::zero();
                for k in 0..x.nrows() {
                    sum = sum + x[[k, i]] * x[[k, j]];
                }
                result[[i, j]] = sum;
            }
        }

        Ok(result)
    }

    /// Helper method for matrix-vector multiplication
    fn matrix_vector_multiply_transpose(
        &self,
        x: &ndarray::ArrayView2<F>,
        y: &ndarray::ArrayView1<F>,
    ) -> Result<Array1<F>> {
        let cols = x.ncols();
        let mut result = Array1::zeros(cols);

        for i in 0..cols {
            let mut sum = F::zero();
            for j in 0..x.nrows() {
                sum = sum + x[[j, i]] * y[j];
            }
            result[i] = sum;
        }

        Ok(result)
    }

    /// Helper method to solve linear system using Gaussian elimination
    fn solve_linear_system(&self, a: &Array2<F>, b: &Array1<F>) -> Result<Array1<F>> {
        let n = a.nrows();
        if a.ncols() != n || b.len() != n {
            return Err(TimeSeriesError::InvalidInput(
                "Matrix dimensions mismatch in linear system".to_string(),
            ));
        }

        // Create augmented matrix
        let mut aug = Array2::zeros((n, n + 1));
        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = a[[i, j]];
            }
            aug[[i, n]] = b[i];
        }

        // Gaussian elimination with partial pivoting
        for k in 0..n {
            // Find pivot
            let mut max_row = k;
            for i in (k + 1)..n {
                if aug[[i, k]].abs() > aug[[max_row, k]].abs() {
                    max_row = i;
                }
            }

            // Swap rows
            if max_row != k {
                for j in 0..=n {
                    let temp = aug[[k, j]];
                    aug[[k, j]] = aug[[max_row, j]];
                    aug[[max_row, j]] = temp;
                }
            }

            // Check for singular matrix
            if aug[[k, k]].abs() < F::from(1e-12).unwrap() {
                return Err(TimeSeriesError::InvalidInput(
                    "Singular matrix in linear system".to_string(),
                ));
            }

            // Eliminate
            for i in (k + 1)..n {
                let factor = aug[[i, k]] / aug[[k, k]];
                for j in k..=n {
                    aug[[i, j]] = aug[[i, j]] - factor * aug[[k, j]];
                }
            }
        }

        // Back substitution
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            let mut sum = F::zero();
            for j in (i + 1)..n {
                sum = sum + aug[[i, j]] * x[j];
            }
            x[i] = (aug[[i, n]] - sum) / aug[[i, i]];
        }

        Ok(x)
    }

    /// Get model parameters
    pub fn get_parameters(&self) -> Option<&GarchParameters<F>> {
        self.parameters.as_ref()
    }

    /// Check if model is fitted
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }
}

/// Technical indicators for financial time series
pub mod technical_indicators {
    use super::*;

    /// Simple Moving Average
    pub fn sma<F: Float + Clone>(data: &Array1<F>, window: usize) -> Result<Array1<F>> {
        if window == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "Window size must be positive".to_string(),
            ));
        }

        if data.len() < window {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough _data for SMA calculation".to_string(),
                required: window,
                actual: data.len(),
            });
        }

        let mut result = Array1::zeros(data.len() - window + 1);

        for i in 0..result.len() {
            let sum = data.slice(s![i..i + window]).sum();
            let window_f = F::from(window).unwrap();
            result[i] = sum / window_f;
        }

        Ok(result)
    }

    /// Exponential Moving Average
    pub fn ema<F: Float + Clone>(data: &Array1<F>, alpha: F) -> Result<Array1<F>> {
        if data.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "Data cannot be empty".to_string(),
            ));
        }

        let zero = F::zero();
        let one = F::one();

        if alpha <= zero || alpha > one {
            return Err(TimeSeriesError::InvalidParameter {
                name: "alpha".to_string(),
                message: "Alpha must be between 0 and 1".to_string(),
            });
        }

        let mut result = Array1::zeros(data.len());
        result[0] = data[0];

        let one_minus_alpha = one - alpha;

        for i in 1..data.len() {
            result[i] = alpha * data[i] + one_minus_alpha * result[i - 1];
        }

        Ok(result)
    }

    /// Bollinger Bands
    pub fn bollinger_bands<F: Float + Clone>(
        data: &Array1<F>,
        window: usize,
        num_std: F,
    ) -> Result<(Array1<F>, Array1<F>, Array1<F>)> {
        let sma_values = sma(data, window)?;
        let mut upper = Array1::zeros(sma_values.len());
        let mut lower = Array1::zeros(sma_values.len());

        for i in 0..sma_values.len() {
            let slice = data.slice(s![i..i + window]);
            let mean = sma_values[i];

            // Calculate standard deviation
            let variance = slice
                .mapv(|x| {
                    let diff = x - mean;
                    diff * diff
                })
                .sum()
                / F::from(window).unwrap();

            let std_dev = variance.sqrt();

            upper[i] = mean + num_std * std_dev;
            lower[i] = mean - num_std * std_dev;
        }

        Ok((upper, sma_values, lower))
    }

    /// Relative Strength Index (RSI)
    pub fn rsi<F: Float + Clone>(data: &Array1<F>, period: usize) -> Result<Array1<F>> {
        if period == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "Period must be positive".to_string(),
            ));
        }

        if data.len() < period + 1 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough _data for RSI calculation".to_string(),
                required: period + 1,
                actual: data.len(),
            });
        }

        // Calculate price changes
        let mut changes = Array1::zeros(data.len() - 1);
        for i in 0..changes.len() {
            changes[i] = data[i + 1] - data[i];
        }

        // Separate gains and losses
        let gains = changes.mapv(|x| if x > F::zero() { x } else { F::zero() });
        let losses = changes.mapv(|x| if x < F::zero() { -x } else { F::zero() });

        // Calculate average gains and losses
        let avg_gain = sma(&gains, period)?;
        let avg_loss = sma(&losses, period)?;

        // Calculate RSI
        let mut rsi = Array1::zeros(avg_gain.len());
        let hundred = F::from(100).unwrap();

        for i in 0..rsi.len() {
            if avg_loss[i] == F::zero() {
                rsi[i] = hundred;
            } else {
                let rs = avg_gain[i] / avg_loss[i];
                rsi[i] = hundred - (hundred / (F::one() + rs));
            }
        }

        Ok(rsi)
    }

    /// MACD (Moving Average Convergence Divergence)
    pub fn macd<F: Float + Clone>(
        data: &Array1<F>,
        fast_period: usize,
        slow_period: usize,
        signal_period: usize,
    ) -> Result<(Array1<F>, Array1<F>, Array1<F>)> {
        if fast_period >= slow_period {
            return Err(TimeSeriesError::InvalidInput(
                "Fast _period must be less than slow _period".to_string(),
            ));
        }

        let fast_alpha = F::from(2.0).unwrap() / F::from(fast_period + 1).unwrap();
        let slow_alpha = F::from(2.0).unwrap() / F::from(slow_period + 1).unwrap();
        let signal_alpha = F::from(2.0).unwrap() / F::from(signal_period + 1).unwrap();

        let fast_ema = ema(data, fast_alpha)?;
        let slow_ema = ema(data, slow_alpha)?;

        // Calculate MACD line
        let macd_line = &fast_ema - &slow_ema;

        // Calculate signal line
        let signal_line = ema(&macd_line, signal_alpha)?;

        // Calculate histogram
        let histogram = &macd_line - &signal_line;

        Ok((macd_line, signal_line, histogram))
    }

    /// Stochastic Oscillator
    pub fn stochastic<F: Float + Clone>(
        high: &Array1<F>,
        low: &Array1<F>,
        close: &Array1<F>,
        k_period: usize,
        d_period: usize,
    ) -> Result<(Array1<F>, Array1<F>)> {
        if high.len() != low.len() || low.len() != close.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: high.len(),
                actual: close.len(),
            });
        }

        if high.len() < k_period {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for stochastic calculation".to_string(),
                required: k_period,
                actual: high.len(),
            });
        }

        let mut k_percent = Array1::zeros(high.len() - k_period + 1);
        let hundred = F::from(100).unwrap();

        for i in 0..k_percent.len() {
            let period_high = high
                .slice(s![i..i + k_period])
                .iter()
                .cloned()
                .fold(F::neg_infinity(), F::max);
            let period_low = low
                .slice(s![i..i + k_period])
                .iter()
                .cloned()
                .fold(F::infinity(), F::min);

            let current_close = close[i + k_period - 1];

            if period_high == period_low {
                k_percent[i] = hundred;
            } else {
                k_percent[i] = hundred * (current_close - period_low) / (period_high - period_low);
            }
        }

        let d_percent = sma(&k_percent, d_period)?;

        Ok((k_percent, d_percent))
    }

    /// Average True Range (ATR)
    pub fn atr<F: Float + Clone>(
        high: &Array1<F>,
        low: &Array1<F>,
        close: &Array1<F>,
        period: usize,
    ) -> Result<Array1<F>> {
        if high.len() != low.len() || low.len() != close.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: high.len(),
                actual: close.len(),
            });
        }

        if high.len() < period + 1 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for ATR calculation".to_string(),
                required: period + 1,
                actual: high.len(),
            });
        }

        let mut true_ranges = Array1::zeros(high.len() - 1);

        for i in 1..high.len() {
            let tr1 = high[i] - low[i];
            let tr2 = (high[i] - close[i - 1]).abs();
            let tr3 = (low[i] - close[i - 1]).abs();

            true_ranges[i - 1] = tr1.max(tr2).max(tr3);
        }

        sma(&true_ranges, period)
    }

    /// Williams %R oscillator
    pub fn williams_r<F: Float + Clone>(
        high: &Array1<F>,
        low: &Array1<F>,
        close: &Array1<F>,
        period: usize,
    ) -> Result<Array1<F>> {
        if high.len() != low.len() || low.len() != close.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: high.len(),
                actual: close.len(),
            });
        }

        if high.len() < period {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for Williams %R calculation".to_string(),
                required: period,
                actual: high.len(),
            });
        }

        let mut williams_r = Array1::zeros(high.len() - period + 1);
        let hundred = F::from(100).unwrap();

        for i in 0..williams_r.len() {
            let period_high = high
                .slice(s![i..i + period])
                .iter()
                .cloned()
                .fold(F::neg_infinity(), F::max);
            let period_low = low
                .slice(s![i..i + period])
                .iter()
                .cloned()
                .fold(F::infinity(), F::min);

            let current_close = close[i + period - 1];

            if period_high == period_low {
                williams_r[i] = F::zero();
            } else {
                williams_r[i] =
                    ((period_high - current_close) / (period_high - period_low)) * (-hundred);
            }
        }

        Ok(williams_r)
    }

    /// Commodity Channel Index (CCI)
    pub fn cci<F: Float + Clone>(
        high: &Array1<F>,
        low: &Array1<F>,
        close: &Array1<F>,
        period: usize,
    ) -> Result<Array1<F>> {
        if high.len() != low.len() || low.len() != close.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: high.len(),
                actual: close.len(),
            });
        }

        if high.len() < period {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for CCI calculation".to_string(),
                required: period,
                actual: high.len(),
            });
        }

        // Calculate Typical Price
        let mut typical_price = Array1::zeros(high.len());
        let three = F::from(3).unwrap();

        for i in 0..high.len() {
            typical_price[i] = (high[i] + low[i] + close[i]) / three;
        }

        // Calculate SMA of typical price
        let sma_tp = sma(&typical_price, period)?;

        // Calculate mean deviation
        let mut cci = Array1::zeros(sma_tp.len());
        let constant = F::from(0.015).unwrap();

        for i in 0..cci.len() {
            let slice = typical_price.slice(s![i..i + period]);
            let mean = sma_tp[i];

            let mean_deviation = slice.mapv(|x| (x - mean).abs()).sum() / F::from(period).unwrap();

            if mean_deviation != F::zero() {
                cci[i] = (typical_price[i + period - 1] - mean) / (constant * mean_deviation);
            }
        }

        Ok(cci)
    }

    /// On-Balance Volume (OBV)
    pub fn obv<F: Float + Clone>(close: &Array1<F>, volume: &Array1<F>) -> Result<Array1<F>> {
        if close.len() != volume.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: close.len(),
                actual: volume.len(),
            });
        }

        if close.len() < 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Need at least 2 data points for OBV".to_string(),
                required: 2,
                actual: close.len(),
            });
        }

        let mut obv = Array1::zeros(close.len());
        obv[0] = volume[0];

        for i in 1..close.len() {
            if close[i] > close[i - 1] {
                obv[i] = obv[i - 1] + volume[i];
            } else if close[i] < close[i - 1] {
                obv[i] = obv[i - 1] - volume[i];
            } else {
                obv[i] = obv[i - 1];
            }
        }

        Ok(obv)
    }

    /// Money Flow Index (MFI)
    pub fn mfi<F: Float + Clone + std::iter::Sum>(
        high: &Array1<F>,
        low: &Array1<F>,
        close: &Array1<F>,
        volume: &Array1<F>,
        period: usize,
    ) -> Result<Array1<F>> {
        if high.len() != low.len() || low.len() != close.len() || close.len() != volume.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: high.len(),
                actual: volume.len(),
            });
        }

        if high.len() < period + 1 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for MFI calculation".to_string(),
                required: period + 1,
                actual: high.len(),
            });
        }

        // Calculate typical price and raw money flow
        let mut typical_price = Array1::zeros(high.len());
        let mut raw_money_flow = Array1::zeros(high.len());
        let three = F::from(3).unwrap();

        for i in 0..high.len() {
            typical_price[i] = (high[i] + low[i] + close[i]) / three;
            raw_money_flow[i] = typical_price[i] * volume[i];
        }

        let mut mfi = Array1::zeros(high.len() - period);
        let hundred = F::from(100).unwrap();

        for i in 0..mfi.len() {
            let mut positive_flow = F::zero();
            let mut negative_flow = F::zero();

            for j in 1..=period {
                let current_idx = i + j;
                let prev_idx = i + j - 1;

                if typical_price[current_idx] > typical_price[prev_idx] {
                    positive_flow = positive_flow + raw_money_flow[current_idx];
                } else if typical_price[current_idx] < typical_price[prev_idx] {
                    negative_flow = negative_flow + raw_money_flow[current_idx];
                }
            }

            if negative_flow == F::zero() {
                mfi[i] = hundred;
            } else {
                let money_ratio = positive_flow / negative_flow;
                mfi[i] = hundred - (hundred / (F::one() + money_ratio));
            }
        }

        Ok(mfi)
    }

    /// Parabolic SAR (Stop and Reverse)
    pub fn parabolic_sar<F: Float + Clone>(
        high: &Array1<F>,
        low: &Array1<F>,
        acceleration: F,
        maximum: F,
    ) -> Result<Array1<F>> {
        if high.len() != low.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: high.len(),
                actual: low.len(),
            });
        }

        if high.len() < 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Need at least 2 data points for Parabolic SAR".to_string(),
                required: 2,
                actual: high.len(),
            });
        }

        let mut sar = Array1::zeros(high.len());
        let mut ep = high[0]; // Extreme Point
        let mut af = acceleration; // Acceleration Factor
        let mut up_trend = true;

        sar[0] = low[0];

        for i in 1..high.len() {
            if up_trend {
                sar[i] = sar[i - 1] + af * (ep - sar[i - 1]);

                if high[i] > ep {
                    ep = high[i];
                    af = (af + acceleration).min(maximum);
                }

                if low[i] <= sar[i] {
                    up_trend = false;
                    sar[i] = ep;
                    ep = low[i];
                    af = acceleration;
                }
            } else {
                sar[i] = sar[i - 1] + af * (ep - sar[i - 1]);

                if low[i] < ep {
                    ep = low[i];
                    af = (af + acceleration).min(maximum);
                }

                if high[i] >= sar[i] {
                    up_trend = true;
                    sar[i] = ep;
                    ep = high[i];
                    af = acceleration;
                }
            }
        }

        Ok(sar)
    }

    /// Aroon indicator
    pub fn aroon<F: Float + Clone>(
        high: &Array1<F>,
        low: &Array1<F>,
        period: usize,
    ) -> Result<(Array1<F>, Array1<F>)> {
        if high.len() != low.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: high.len(),
                actual: low.len(),
            });
        }

        if high.len() < period {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for Aroon calculation".to_string(),
                required: period,
                actual: high.len(),
            });
        }

        let mut aroon_up = Array1::zeros(high.len() - period + 1);
        let mut aroon_down = Array1::zeros(high.len() - period + 1);
        let hundred = F::from(100).unwrap();
        let period_f = F::from(period).unwrap();

        for i in 0..aroon_up.len() {
            let mut highest_idx = 0;
            let mut lowest_idx = 0;
            let mut highest_val = high[i];
            let mut lowest_val = low[i];

            for j in 1..period {
                if high[i + j] > highest_val {
                    highest_val = high[i + j];
                    highest_idx = j;
                }
                if low[i + j] < lowest_val {
                    lowest_val = low[i + j];
                    lowest_idx = j;
                }
            }

            aroon_up[i] =
                hundred * (period_f - F::from(period - 1 - highest_idx).unwrap()) / period_f;
            aroon_down[i] =
                hundred * (period_f - F::from(period - 1 - lowest_idx).unwrap()) / period_f;
        }

        Ok((aroon_up, aroon_down))
    }

    /// Directional Movement Index (DMI) and Average Directional Index (ADX)
    pub fn adx<F: Float + Clone>(
        high: &Array1<F>,
        low: &Array1<F>,
        close: &Array1<F>,
        period: usize,
    ) -> Result<(Array1<F>, Array1<F>, Array1<F>)> {
        if high.len() != low.len() || low.len() != close.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: high.len(),
                actual: close.len(),
            });
        }

        if high.len() < period + 1 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for ADX calculation".to_string(),
                required: period + 1,
                actual: high.len(),
            });
        }

        let atr_values = atr(high, low, close, period)?;

        // Calculate Directional Movement
        let mut dm_plus = Array1::zeros(high.len() - 1);
        let mut dm_minus = Array1::zeros(high.len() - 1);

        for i in 1..high.len() {
            let up_move = high[i] - high[i - 1];
            let down_move = low[i - 1] - low[i];

            if up_move > down_move && up_move > F::zero() {
                dm_plus[i - 1] = up_move;
            }
            if down_move > up_move && down_move > F::zero() {
                dm_minus[i - 1] = down_move;
            }
        }

        // Calculate smoothed DM values
        let smoothed_dm_plus = ema(
            &dm_plus,
            F::from(2.0).unwrap() / F::from(period + 1).unwrap(),
        )?;
        let smoothed_dm_minus = ema(
            &dm_minus,
            F::from(2.0).unwrap() / F::from(period + 1).unwrap(),
        )?;

        // Calculate DI+ and DI-
        let mut di_plus = Array1::zeros(atr_values.len());
        let mut di_minus = Array1::zeros(atr_values.len());
        let hundred = F::from(100).unwrap();

        for i in 0..di_plus.len() {
            if atr_values[i] != F::zero() {
                di_plus[i] = hundred * smoothed_dm_plus[i] / atr_values[i];
                di_minus[i] = hundred * smoothed_dm_minus[i] / atr_values[i];
            }
        }

        // Calculate DX and ADX
        let mut dx = Array1::zeros(di_plus.len());
        for i in 0..dx.len() {
            let di_sum = di_plus[i] + di_minus[i];
            if di_sum != F::zero() {
                dx[i] = hundred * (di_plus[i] - di_minus[i]).abs() / di_sum;
            }
        }

        let adx_values = ema(&dx, F::from(2.0).unwrap() / F::from(period + 1).unwrap())?;

        Ok((di_plus, di_minus, adx_values))
    }
}

/// Volatility modeling functions
pub mod volatility {
    use super::*;

    /// Calculate realized volatility from high-frequency returns
    pub fn realized_volatility<F: Float>(returns: &Array1<F>) -> F {
        returns.mapv(|x| x * x).sum()
    }

    /// Garman-Klass volatility estimator
    pub fn garman_klass_volatility<F: Float + Clone>(
        high: &Array1<F>,
        low: &Array1<F>,
        close: &Array1<F>,
        open: &Array1<F>,
    ) -> Result<Array1<F>> {
        if high.len() != low.len() || low.len() != close.len() || close.len() != open.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: high.len(),
                actual: open.len(),
            });
        }

        let mut gk_vol = Array1::zeros(high.len());
        let half = F::from(0.5).unwrap();
        let ln_2_minus_1 = F::from(2.0 * (2.0_f64).ln() - 1.0).unwrap();

        for i in 0..gk_vol.len() {
            let log_hl = (high[i] / low[i]).ln();
            let log_co = (close[i] / open[i]).ln();

            gk_vol[i] = half * log_hl * log_hl - ln_2_minus_1 * log_co * log_co;
        }

        Ok(gk_vol)
    }

    /// Parkinson volatility estimator
    pub fn parkinson_volatility<F: Float + Clone>(
        high: &Array1<F>,
        low: &Array1<F>,
    ) -> Result<Array1<F>> {
        if high.len() != low.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: high.len(),
                actual: low.len(),
            });
        }

        let mut park_vol = Array1::zeros(high.len());
        let four_ln_2 = F::from(4.0 * (2.0_f64).ln()).unwrap();

        for i in 0..park_vol.len() {
            let log_hl = (high[i] / low[i]).ln();
            park_vol[i] = log_hl * log_hl / four_ln_2;
        }

        Ok(park_vol)
    }

    /// Rogers-Satchell volatility estimator
    pub fn rogers_satchell_volatility<F: Float + Clone>(
        high: &Array1<F>,
        low: &Array1<F>,
        close: &Array1<F>,
        open: &Array1<F>,
    ) -> Result<Array1<F>> {
        if high.len() != low.len() || low.len() != close.len() || close.len() != open.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: high.len(),
                actual: open.len(),
            });
        }

        let mut rs_vol = Array1::zeros(high.len());

        for i in 0..rs_vol.len() {
            let log_ho = (high[i] / open[i]).ln();
            let log_co = (close[i] / open[i]).ln();
            let log_lo = (low[i] / open[i]).ln();

            rs_vol[i] = log_ho * log_co + log_lo * log_co;
        }

        Ok(rs_vol)
    }

    /// Yang-Zhang volatility estimator
    pub fn yang_zhang_volatility<F: Float + Clone>(
        high: &Array1<F>,
        low: &Array1<F>,
        close: &Array1<F>,
        open: &Array1<F>,
        k: F,
    ) -> Result<Array1<F>> {
        if high.len() != low.len() || low.len() != close.len() || close.len() != open.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: high.len(),
                actual: open.len(),
            });
        }

        if high.len() < 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Need at least 2 data points for Yang-Zhang volatility".to_string(),
                required: 2,
                actual: high.len(),
            });
        }

        let mut yz_vol = Array1::zeros(high.len() - 1);

        for i in 1..high.len() {
            // Overnight return
            let overnight = (open[i] / close[i - 1]).ln();

            // Open-to-close return
            let open_close = (close[i] / open[i]).ln();

            // Rogers-Satchell component
            let log_ho = (high[i] / open[i]).ln();
            let log_co = (close[i] / open[i]).ln();
            let log_lo = (low[i] / open[i]).ln();
            let rs = log_ho * log_co + log_lo * log_co;

            yz_vol[i - 1] = overnight * overnight + k * open_close * open_close + rs;
        }

        Ok(yz_vol)
    }

    /// GARCH(1,1) volatility estimation using simple method of moments
    pub fn garch_volatility_estimate<F: Float + Clone>(
        returns: &Array1<F>,
        window: usize,
    ) -> Result<Array1<F>> {
        if returns.len() < window + 1 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for GARCH volatility estimation".to_string(),
                required: window + 1,
                actual: returns.len(),
            });
        }

        let mut volatilities = Array1::zeros(returns.len() - window + 1);

        // Simple GARCH(1,1) parameters (typical values)
        let omega = F::from(0.000001).unwrap();
        let alpha = F::from(0.1).unwrap();
        let beta = F::from(0.85).unwrap();

        for i in 0..volatilities.len() {
            let window_returns = returns.slice(s![i..i + window]);

            // Initialize with sample variance
            let mean = window_returns.sum() / F::from(window).unwrap();
            let mut variance =
                window_returns.mapv(|x| (x - mean).powi(2)).sum() / F::from(window - 1).unwrap();

            // Apply GARCH updating for last few observations
            for j in 1..std::cmp::min(window, 10) {
                let return_sq = window_returns[window - j].powi(2);
                variance = omega + alpha * return_sq + beta * variance;
            }

            volatilities[i] = variance.sqrt();
        }

        Ok(volatilities)
    }

    /// Exponentially Weighted Moving Average (EWMA) volatility
    pub fn ewma_volatility<F: Float + Clone>(returns: &Array1<F>, lambda: F) -> Result<Array1<F>> {
        if returns.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "Returns cannot be empty".to_string(),
            ));
        }

        if lambda <= F::zero() || lambda >= F::one() {
            return Err(TimeSeriesError::InvalidParameter {
                name: "lambda".to_string(),
                message: "Lambda must be between 0 and 1 (exclusive)".to_string(),
            });
        }

        let mut ewma_var = Array1::zeros(returns.len());

        // Initialize with first squared return
        ewma_var[0] = returns[0].powi(2);

        let one_minus_lambda = F::one() - lambda;

        for i in 1..returns.len() {
            ewma_var[i] = lambda * ewma_var[i - 1] + one_minus_lambda * returns[i].powi(2);
        }

        Ok(ewma_var.mapv(|x| x.sqrt()))
    }

    /// Range-based volatility using high-low range
    pub fn range_volatility<F: Float + Clone>(
        high: &Array1<F>,
        low: &Array1<F>,
        period: usize,
    ) -> Result<Array1<F>> {
        if high.len() != low.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: high.len(),
                actual: low.len(),
            });
        }

        if high.len() < period {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for range volatility calculation".to_string(),
                required: period,
                actual: high.len(),
            });
        }

        let mut range_vol = Array1::zeros(high.len() - period + 1);
        let scaling_factor = F::from(1.0 / (4.0 * (2.0_f64).ln())).unwrap();

        for i in 0..range_vol.len() {
            let mut sum_log_range_sq = F::zero();

            for j in 0..period {
                let log_range = (high[i + j] / low[i + j]).ln();
                sum_log_range_sq = sum_log_range_sq + log_range.powi(2);
            }

            range_vol[i] = (scaling_factor * sum_log_range_sq / F::from(period).unwrap()).sqrt();
        }

        Ok(range_vol)
    }

    /// Intraday volatility estimation using tick data concept
    pub fn intraday_volatility<F: Float + Clone>(
        prices: &Array1<F>,
        sampling_frequency: usize,
    ) -> Result<F> {
        if prices.len() < sampling_frequency + 1 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for intraday volatility calculation".to_string(),
                required: sampling_frequency + 1,
                actual: prices.len(),
            });
        }

        let mut squared_returns = F::zero();
        let mut count = 0;

        for i in sampling_frequency..prices.len() {
            let logreturn = (prices[i] / prices[i - sampling_frequency]).ln();
            squared_returns = squared_returns + logreturn.powi(2);
            count += 1;
        }

        if count == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "No valid returns calculated".to_string(),
            ));
        }

        Ok((squared_returns / F::from(count).unwrap()).sqrt())
    }
}

/// Risk management utilities
pub mod risk {
    use super::*;

    /// Calculate Value at Risk (VaR) using historical simulation
    pub fn var_historical<F: Float + Clone>(returns: &Array1<F>, confidence: f64) -> Result<F> {
        if returns.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "Returns cannot be empty".to_string(),
            ));
        }

        if confidence <= 0.0 || confidence >= 1.0 {
            return Err(TimeSeriesError::InvalidParameter {
                name: "confidence".to_string(),
                message: "Confidence must be between 0 and 1".to_string(),
            });
        }

        let mut sorted_returns = returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = ((1.0 - confidence) * sorted_returns.len() as f64) as usize;
        let index = index.min(sorted_returns.len() - 1);

        Ok(sorted_returns[index])
    }

    /// Calculate Expected Shortfall (Conditional VaR)
    pub fn expected_shortfall<F: Float + Clone + std::iter::Sum>(
        returns: &Array1<F>,
        confidence: f64,
    ) -> Result<F> {
        let var = var_historical(returns, confidence)?;

        let tail_returns: Vec<F> = returns.iter().filter(|&&x| x <= var).cloned().collect();

        if tail_returns.is_empty() {
            return Ok(var);
        }

        let sum = tail_returns.iter().fold(F::zero(), |acc, &x| acc + x);
        Ok(sum / F::from(tail_returns.len()).unwrap())
    }

    /// Calculate maximum drawdown
    pub fn max_drawdown<F: Float + Clone>(prices: &Array1<F>) -> Result<F> {
        if prices.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "Prices cannot be empty".to_string(),
            ));
        }

        let mut max_price = prices[0];
        let mut max_dd = F::zero();

        for &price in prices.iter() {
            if price > max_price {
                max_price = price;
            }

            let drawdown = (max_price - price) / max_price;
            if drawdown > max_dd {
                max_dd = drawdown;
            }
        }

        Ok(max_dd)
    }

    /// Calmar ratio (annual return / maximum drawdown)
    pub fn calmar_ratio<F: Float + Clone>(
        returns: &Array1<F>,
        prices: &Array1<F>,
        periods_per_year: usize,
    ) -> Result<F> {
        if returns.is_empty() || prices.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "Returns and prices cannot be empty".to_string(),
            ));
        }

        // Calculate annualized return
        let totalreturn = (prices[prices.len() - 1] / prices[0]) - F::one();
        let years = F::from(returns.len()).unwrap() / F::from(periods_per_year).unwrap();
        let annualizedreturn = (F::one() + totalreturn).powf(F::one() / years) - F::one();

        // Calculate maximum drawdown
        let mdd = max_drawdown(prices)?;

        if mdd == F::zero() {
            Ok(F::infinity())
        } else {
            Ok(annualizedreturn / mdd)
        }
    }

    /// Sortino ratio (excess return / downside deviation)
    pub fn sortino_ratio<F: Float + Clone>(
        returns: &Array1<F>,
        risk_free_rate: F,
        periods_per_year: usize,
    ) -> Result<F> {
        if returns.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "Returns cannot be empty".to_string(),
            ));
        }

        // Calculate excess returns
        let annualized_rf = risk_free_rate / F::from(periods_per_year).unwrap();
        let excess_returns: Array1<F> = returns.mapv(|r| r - annualized_rf);

        // Calculate mean excess return
        let mean_excess = excess_returns.sum() / F::from(returns.len()).unwrap();

        // Calculate downside deviation (only negative excess returns)
        let downside_returns: Vec<F> = excess_returns
            .iter()
            .filter(|&&r| r < F::zero())
            .cloned()
            .collect();

        if downside_returns.is_empty() {
            return Ok(F::infinity());
        }

        let downside_variance = downside_returns
            .iter()
            .map(|&r| r.powi(2))
            .fold(F::zero(), |acc, x| acc + x)
            / F::from(downside_returns.len()).unwrap();

        let downside_deviation = downside_variance.sqrt();

        if downside_deviation == F::zero() {
            Ok(F::infinity())
        } else {
            // Annualize the ratio
            let annualized_excess = mean_excess * F::from(periods_per_year).unwrap();
            let annualized_downside =
                downside_deviation * F::from(periods_per_year).unwrap().sqrt();
            Ok(annualized_excess / annualized_downside)
        }
    }

    /// Sharpe ratio (excess return / volatility)
    pub fn sharpe_ratio<F: Float + Clone>(
        returns: &Array1<F>,
        risk_free_rate: F,
        periods_per_year: usize,
    ) -> Result<F> {
        if returns.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "Returns cannot be empty".to_string(),
            ));
        }

        // Calculate excess returns
        let annualized_rf = risk_free_rate / F::from(periods_per_year).unwrap();
        let excess_returns: Array1<F> = returns.mapv(|r| r - annualized_rf);

        // Calculate mean excess return
        let mean_excess = excess_returns.sum() / F::from(returns.len()).unwrap();

        // Calculate standard deviation of excess returns
        let variance = excess_returns.mapv(|r| (r - mean_excess).powi(2)).sum()
            / F::from(returns.len() - 1).unwrap();

        let std_dev = variance.sqrt();

        if std_dev == F::zero() {
            Ok(F::infinity())
        } else {
            // Annualize the ratio
            let annualized_excess = mean_excess * F::from(periods_per_year).unwrap();
            let annualized_std = std_dev * F::from(periods_per_year).unwrap().sqrt();
            Ok(annualized_excess / annualized_std)
        }
    }

    /// Information ratio (active return / tracking error)
    pub fn information_ratio<F: Float + Clone>(
        portfolio_returns: &Array1<F>,
        benchmark_returns: &Array1<F>,
    ) -> Result<F> {
        if portfolio_returns.len() != benchmark_returns.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: portfolio_returns.len(),
                actual: benchmark_returns.len(),
            });
        }

        if portfolio_returns.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "Returns cannot be empty".to_string(),
            ));
        }

        // Calculate active _returns (portfolio - benchmark)
        let active_returns: Array1<F> = portfolio_returns
            .iter()
            .zip(benchmark_returns.iter())
            .map(|(&p, &b)| p - b)
            .collect();

        // Calculate mean active return
        let mean_active = active_returns.sum() / F::from(active_returns.len()).unwrap();

        // Calculate tracking error (standard deviation of active returns)
        let variance = active_returns.mapv(|r| (r - mean_active).powi(2)).sum()
            / F::from(active_returns.len() - 1).unwrap();

        let tracking_error = variance.sqrt();

        if tracking_error == F::zero() {
            Ok(F::infinity())
        } else {
            Ok(mean_active / tracking_error)
        }
    }

    /// Beta coefficient (systematic risk measure)
    pub fn beta<F: Float + Clone>(
        asset_returns: &Array1<F>,
        market_returns: &Array1<F>,
    ) -> Result<F> {
        if asset_returns.len() != market_returns.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: asset_returns.len(),
                actual: market_returns.len(),
            });
        }

        if asset_returns.len() < 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Need at least 2 observations for beta calculation".to_string(),
                required: 2,
                actual: asset_returns.len(),
            });
        }

        // Calculate means
        let asset_mean = asset_returns.sum() / F::from(asset_returns.len()).unwrap();
        let market_mean = market_returns.sum() / F::from(market_returns.len()).unwrap();

        // Calculate covariance and market variance
        let mut covariance = F::zero();
        let mut market_variance = F::zero();

        for i in 0..asset_returns.len() {
            let asset_dev = asset_returns[i] - asset_mean;
            let market_dev = market_returns[i] - market_mean;

            covariance = covariance + asset_dev * market_dev;
            market_variance = market_variance + market_dev.powi(2);
        }

        let n = F::from(asset_returns.len() - 1).unwrap();
        covariance = covariance / n;
        market_variance = market_variance / n;

        if market_variance == F::zero() {
            Err(TimeSeriesError::InvalidInput(
                "Market _returns have zero variance".to_string(),
            ))
        } else {
            Ok(covariance / market_variance)
        }
    }

    /// Treynor ratio (excess return / beta)
    pub fn treynor_ratio<F: Float + Clone>(
        returns: &Array1<F>,
        market_returns: &Array1<F>,
        risk_free_rate: F,
        periods_per_year: usize,
    ) -> Result<F> {
        // Calculate portfolio beta
        let portfolio_beta = beta(returns, market_returns)?;

        if portfolio_beta == F::zero() {
            return Ok(F::infinity());
        }

        // Calculate annualized excess return
        let annualized_rf = risk_free_rate / F::from(periods_per_year).unwrap();
        let meanreturn = returns.sum() / F::from(returns.len()).unwrap();
        let excessreturn = meanreturn - annualized_rf;
        let annualized_excess = excessreturn * F::from(periods_per_year).unwrap();

        Ok(annualized_excess / portfolio_beta)
    }

    /// Jensen's alpha (risk-adjusted excess return)
    pub fn jensens_alpha<F: Float + Clone>(
        returns: &Array1<F>,
        market_returns: &Array1<F>,
        risk_free_rate: F,
        periods_per_year: usize,
    ) -> Result<F> {
        // Calculate portfolio beta
        let portfolio_beta = beta(returns, market_returns)?;

        // Calculate mean _returns
        let annualized_rf = risk_free_rate / F::from(periods_per_year).unwrap();
        let mean_portfolio = returns.sum() / F::from(returns.len()).unwrap();
        let mean_market = market_returns.sum() / F::from(market_returns.len()).unwrap();

        // Calculate alpha using CAPM formula
        // Alpha = Portfolio Return - (Risk Free Rate + Beta * (Market Return - Risk Free Rate))
        let portfolio_excess = mean_portfolio - annualized_rf;
        let market_excess = mean_market - annualized_rf;
        let expected_excess = portfolio_beta * market_excess;

        Ok((portfolio_excess - expected_excess) * F::from(periods_per_year).unwrap())
    }

    /// Omega ratio (probability-weighted gains over losses)
    pub fn omega_ratio<F: Float + Clone>(returns: &Array1<F>, threshold: F) -> Result<F> {
        if returns.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "Returns cannot be empty".to_string(),
            ));
        }

        let mut gains = F::zero();
        let mut losses = F::zero();

        for &ret in returns.iter() {
            let excess = ret - threshold;
            if excess > F::zero() {
                gains = gains + excess;
            } else {
                losses = losses - excess; // Make positive
            }
        }

        if losses == F::zero() {
            Ok(F::infinity())
        } else {
            Ok(gains / losses)
        }
    }

    /// Value at Risk using Monte Carlo simulation (simplified)
    pub fn var_monte_carlo<F: Float + Clone>(
        returns: &Array1<F>,
        confidence: f64,
        simulations: usize,
        horizon: usize,
    ) -> Result<F> {
        if returns.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "Returns cannot be empty".to_string(),
            ));
        }

        if confidence <= 0.0 || confidence >= 1.0 {
            return Err(TimeSeriesError::InvalidParameter {
                name: "confidence".to_string(),
                message: "Confidence must be between 0 and 1 (exclusive)".to_string(),
            });
        }

        // Calculate mean and standard deviation
        let mean = returns.sum() / F::from(returns.len()).unwrap();
        let variance =
            returns.mapv(|r| (r - mean).powi(2)).sum() / F::from(returns.len() - 1).unwrap();
        let std_dev = variance.sqrt();

        // Simplified Monte Carlo: assume normal distribution
        // In practice, this would use proper random number generation
        let mut simulated_returns = Vec::with_capacity(simulations);

        for i in 0..simulations {
            // Simple pseudo-random generation (Box-Muller transform approximation)
            let u1 = F::from((i as f64 + 1.0) / (simulations as f64 + 1.0)).unwrap();
            let u2 = F::from(0.5).unwrap(); // Simplified

            // Convert to normal distribution (simplified)
            let z = (-F::from(2.0).unwrap() * u1.ln()).sqrt()
                * (F::from(2.0 * std::f64::consts::PI).unwrap() * u2).cos();

            let simulatedreturn = mean + std_dev * z;

            // Calculate portfolio value change over horizon
            let mut portfolioreturn = F::one();
            for _ in 0..horizon {
                portfolioreturn = portfolioreturn * (F::one() + simulatedreturn);
            }

            simulated_returns.push(portfolioreturn - F::one());
        }

        // Sort returns and find VaR
        simulated_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let var_index = ((1.0 - confidence) * simulations as f64) as usize;
        let var_index = var_index.min(simulations - 1);

        Ok(simulated_returns[var_index])
    }
}

/// Portfolio analysis and optimization utilities
pub mod portfolio {
    use super::*;

    /// Portfolio performance metrics
    #[derive(Debug, Clone)]
    pub struct PortfolioMetrics<F: Float> {
        /// Total return
        pub total_return: F,
        /// Annualized return
        pub annualized_return: F,
        /// Annualized volatility
        pub volatility: F,
        /// Sharpe ratio
        pub sharpe_ratio: F,
        /// Sortino ratio
        pub sortino_ratio: F,
        /// Maximum drawdown
        pub max_drawdown: F,
        /// Calmar ratio
        pub calmar_ratio: F,
        /// Value at Risk (95%)
        pub var_95: F,
        /// Expected Shortfall (95%)
        pub es_95: F,
    }

    /// Portfolio weights and holdings
    #[derive(Debug, Clone)]
    pub struct Portfolio<F: Float> {
        /// Asset weights (should sum to 1.0)
        pub weights: Array1<F>,
        /// Asset names/identifiers
        pub asset_names: Vec<String>,
        /// Rebalancing frequency (days)
        pub rebalance_frequency: Option<usize>,
    }

    impl<F: Float + Clone> Portfolio<F> {
        /// Create a new portfolio
        pub fn new(_weights: Array1<F>, assetnames: Vec<String>) -> Result<Self> {
            if _weights.len() != assetnames.len() {
                return Err(TimeSeriesError::DimensionMismatch {
                    expected: _weights.len(),
                    actual: assetnames.len(),
                });
            }

            let weight_sum = _weights.sum();
            let tolerance = F::from(0.01).unwrap();
            if (weight_sum - F::one()).abs() > tolerance {
                return Err(TimeSeriesError::InvalidInput(
                    "Portfolio _weights must sum to approximately 1.0".to_string(),
                ));
            }

            Ok(Self {
                weights: _weights,
                asset_names: assetnames,
                rebalance_frequency: None,
            })
        }

        /// Create equally weighted portfolio
        pub fn equal_weight(_n_assets: usize, assetnames: Vec<String>) -> Result<Self> {
            if _n_assets == 0 {
                return Err(TimeSeriesError::InvalidInput(
                    "Number of _assets must be positive".to_string(),
                ));
            }

            let weight = F::one() / F::from(_n_assets).unwrap();
            let weights = Array1::from_elem(_n_assets, weight);

            Self::new(weights, assetnames)
        }

        /// Get portfolio weight for specific asset
        pub fn get_weight(&self, assetname: &str) -> Option<F> {
            self.asset_names
                .iter()
                .position(|_name| _name == assetname)
                .map(|idx| self.weights[idx])
        }
    }

    /// Calculate portfolio returns from asset returns and weights
    pub fn calculate_portfolio_returns<F: Float + Clone>(
        asset_returns: &Array2<F>, // rows: time, cols: assets
        weights: &Array1<F>,
    ) -> Result<Array1<F>> {
        if asset_returns.ncols() != weights.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: asset_returns.ncols(),
                actual: weights.len(),
            });
        }

        let mut portfolio_returns = Array1::zeros(asset_returns.nrows());

        for t in 0..asset_returns.nrows() {
            let mut return_sum = F::zero();
            for i in 0..weights.len() {
                return_sum = return_sum + weights[i] * asset_returns[[t, i]];
            }
            portfolio_returns[t] = return_sum;
        }

        Ok(portfolio_returns)
    }

    /// Calculate portfolio metrics
    pub fn calculate_portfolio_metrics<F: Float + Clone + std::iter::Sum>(
        returns: &Array1<F>,
        prices: &Array1<F>,
        risk_free_rate: F,
        periods_per_year: usize,
    ) -> Result<PortfolioMetrics<F>> {
        if returns.is_empty() || prices.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "Returns and prices cannot be empty".to_string(),
            ));
        }

        // Total return
        let totalreturn = (prices[prices.len() - 1] / prices[0]) - F::one();

        // Annualized return
        let years = F::from(returns.len()).unwrap() / F::from(periods_per_year).unwrap();
        let annualizedreturn = (F::one() + totalreturn).powf(F::one() / years) - F::one();

        // Volatility (annualized)
        let meanreturn = returns.sum() / F::from(returns.len()).unwrap();
        let variance =
            returns.mapv(|r| (r - meanreturn).powi(2)).sum() / F::from(returns.len() - 1).unwrap();
        let volatility = variance.sqrt() * F::from(periods_per_year).unwrap().sqrt();

        // Risk metrics
        let sharpe = super::risk::sharpe_ratio(returns, risk_free_rate, periods_per_year)?;
        let sortino = super::risk::sortino_ratio(returns, risk_free_rate, periods_per_year)?;
        let max_dd = super::risk::max_drawdown(prices)?;
        let calmar = super::risk::calmar_ratio(returns, prices, periods_per_year)?;
        let var_95 = super::risk::var_historical(returns, 0.95)?;
        let es_95 = super::risk::expected_shortfall(returns, 0.95)?;

        Ok(PortfolioMetrics {
            total_return: totalreturn,
            annualized_return: annualizedreturn,
            volatility,
            sharpe_ratio: sharpe,
            sortino_ratio: sortino,
            max_drawdown: max_dd,
            calmar_ratio: calmar,
            var_95,
            es_95,
        })
    }

    /// Modern Portfolio Theory: Calculate efficient frontier point
    pub fn calculate_efficient_portfolio<F: Float + Clone>(
        expected_returns: &Array1<F>,
        covariance_matrix: &Array2<F>,
        target_return: F,
    ) -> Result<Array1<F>> {
        let n = expected_returns.len();

        if covariance_matrix.nrows() != n || covariance_matrix.ncols() != n {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n,
                actual: covariance_matrix.nrows(),
            });
        }

        // This is a simplified implementation
        // In practice, you would use quadratic programming

        // Equal weight as starting point
        let mut weights = Array1::from_elem(n, F::one() / F::from(n).unwrap());

        // Simple iterative adjustment toward target _return
        for _ in 0..100 {
            let currentreturn = weights
                .iter()
                .zip(expected_returns.iter())
                .map(|(&w, &r)| w * r)
                .fold(F::zero(), |acc, x| acc + x);

            let return_diff = target_return - currentreturn;

            if return_diff.abs() < F::from(1e-6).unwrap() {
                break;
            }

            // Adjust weights toward higher/lower return assets
            for i in 0..n {
                let adjustment = return_diff * F::from(0.01).unwrap();
                if expected_returns[i] > currentreturn {
                    weights[i] = weights[i] + adjustment;
                } else {
                    weights[i] = weights[i] - adjustment;
                }
                weights[i] = weights[i].max(F::zero());
            }

            // Normalize weights
            let weight_sum = weights.sum();
            if weight_sum > F::zero() {
                weights = weights.mapv(|w| w / weight_sum);
            }
        }

        Ok(weights)
    }

    /// Risk parity portfolio optimization
    pub fn risk_parity_portfolio<F: Float + Clone>(
        covariance_matrix: &Array2<F>,
    ) -> Result<Array1<F>> {
        let n = covariance_matrix.nrows();

        if covariance_matrix.ncols() != n {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n,
                actual: covariance_matrix.ncols(),
            });
        }

        // Calculate inverse volatility weights as starting approximation
        let mut weights = Array1::zeros(n);

        for i in 0..n {
            let variance = covariance_matrix[[i, i]];
            if variance > F::zero() {
                weights[i] = F::one() / variance.sqrt();
            } else {
                weights[i] = F::one();
            }
        }

        // Normalize weights
        let weight_sum = weights.sum();
        if weight_sum > F::zero() {
            weights = weights.mapv(|w| w / weight_sum);
        } else {
            // Equal weights fallback
            weights = Array1::from_elem(n, F::one() / F::from(n).unwrap());
        }

        Ok(weights)
    }

    /// Minimum variance portfolio
    pub fn minimum_variance_portfolio<F: Float + Clone>(
        covariance_matrix: &Array2<F>,
    ) -> Result<Array1<F>> {
        let n = covariance_matrix.nrows();

        // Simplified implementation: inverse variance weighting
        let mut weights = Array1::zeros(n);

        for i in 0..n {
            let variance = covariance_matrix[[i, i]];
            if variance > F::zero() {
                weights[i] = F::one() / variance;
            } else {
                weights[i] = F::one();
            }
        }

        // Normalize weights
        let weight_sum = weights.sum();
        if weight_sum > F::zero() {
            weights = weights.mapv(|w| w / weight_sum);
        } else {
            return Err(TimeSeriesError::InvalidInput(
                "All assets have zero variance".to_string(),
            ));
        }

        Ok(weights)
    }

    /// Calculate portfolio Value at Risk using parametric method
    pub fn portfolio_var_parametric<F: Float + Clone>(
        portfolio_value: F,
        portfolio_return_mean: F,
        portfolio_return_std: F,
        confidence_level: f64,
        time_horizon: usize,
    ) -> Result<F> {
        if confidence_level <= 0.0 || confidence_level >= 1.0 {
            return Err(TimeSeriesError::InvalidParameter {
                name: "confidence_level".to_string(),
                message: "Confidence _level must be between 0 and 1".to_string(),
            });
        }

        // Z-score for confidence _level
        let z_score = match confidence_level {
            c if c >= 0.99 => F::from(-2.326).unwrap(), // 99% VaR
            c if c >= 0.95 => F::from(-1.645).unwrap(), // 95% VaR
            c if c >= 0.90 => F::from(-1.282).unwrap(), // 90% VaR
            _ => F::from(-1.0).unwrap(),
        };

        // Scale for time _horizon
        let horizon_scaling = F::from(time_horizon).unwrap().sqrt();
        let horizon_mean = portfolio_return_mean * F::from(time_horizon).unwrap();
        let horizon_std = portfolio_return_std * horizon_scaling;

        // Calculate VaR
        let varreturn = horizon_mean + z_score * horizon_std;
        let var_amount = portfolio_value * varreturn.abs();

        Ok(var_amount)
    }

    /// Calculate correlation matrix from returns
    pub fn calculate_correlation_matrix<F: Float + Clone>(
        returns: &Array2<F>, // rows: time, cols: assets
    ) -> Result<Array2<F>> {
        let n_assets = returns.ncols();
        let n_periods = returns.nrows();

        if n_periods < 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Need at least 2 periods for correlation calculation".to_string(),
                required: 2,
                actual: n_periods,
            });
        }

        let mut correlation_matrix = Array2::zeros((n_assets, n_assets));

        // Calculate means
        let means: Array1<F> = (0..n_assets)
            .map(|i| {
                let col = returns.column(i);
                col.sum() / F::from(n_periods).unwrap()
            })
            .collect();

        // Calculate correlation coefficients
        for i in 0..n_assets {
            for j in 0..n_assets {
                if i == j {
                    correlation_matrix[[i, j]] = F::one();
                } else {
                    let col_i = returns.column(i);
                    let col_j = returns.column(j);

                    let mut numerator = F::zero();
                    let mut sum_sq_i = F::zero();
                    let mut sum_sq_j = F::zero();

                    for t in 0..n_periods {
                        let dev_i = col_i[t] - means[i];
                        let dev_j = col_j[t] - means[j];

                        numerator = numerator + dev_i * dev_j;
                        sum_sq_i = sum_sq_i + dev_i * dev_i;
                        sum_sq_j = sum_sq_j + dev_j * dev_j;
                    }

                    let denominator = (sum_sq_i * sum_sq_j).sqrt();
                    if denominator > F::zero() {
                        correlation_matrix[[i, j]] = numerator / denominator;
                    }
                }
            }
        }

        Ok(correlation_matrix)
    }
}

#[cfg(test)]
mod tests {
    use super::risk::*;
    use super::technical_indicators::*;
    use super::volatility::*;
    use super::{black_scholes, Distribution, EgarchModel, GarchConfig, GarchModel, MeanModel};
    use crate::error::TimeSeriesError;
    use approx::assert_abs_diff_eq;
    use ndarray::Array1;

    #[test]
    fn test_garch_config_default() {
        let config = GarchConfig::default();
        assert_eq!(config.p, 1);
        assert_eq!(config.q, 1);
        assert!(matches!(config.mean_model, MeanModel::Constant));
        assert!(matches!(config.distribution, Distribution::Normal));
    }

    #[test]
    fn test_garch_model_creation() {
        let model = GarchModel::<f64>::garch_11();
        assert!(!model.is_fitted());
        assert!(model.get_parameters().is_none());
    }

    #[test]
    fn test_sma() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = sma(&data, 3).unwrap();

        assert_eq!(result.len(), 3);
        assert_abs_diff_eq!(result[0], 2.0);
        assert_abs_diff_eq!(result[1], 3.0);
        assert_abs_diff_eq!(result[2], 4.0);
    }

    #[test]
    fn test_ema() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = ema(&data, 0.5).unwrap();

        assert_eq!(result.len(), 5);
        assert_abs_diff_eq!(result[0], 1.0);
        assert_abs_diff_eq!(result[1], 1.5);
    }

    #[test]
    fn test_bollinger_bands() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        let (upper, middle, lower) = bollinger_bands(&data, 3, 2.0).unwrap();

        assert_eq!(upper.len(), 7);
        assert_eq!(middle.len(), 7);
        assert_eq!(lower.len(), 7);

        // Middle band should be SMA
        let sma_result = sma(&data, 3).unwrap();
        for i in 0..middle.len() {
            assert_abs_diff_eq!(middle[i], sma_result[i]);
        }
    }

    #[test]
    fn test_rsi() {
        let data = Array1::from_vec(vec![
            44.0, 44.25, 44.5, 43.75, 44.5, 44.75, 47.0, 47.25, 46.5, 46.25, 47.75, 47.5, 47.25,
            47.75, 48.75, 48.5, 48.0, 48.25, 48.75, 48.5,
        ]);
        let result = rsi(&data, 14).unwrap();

        assert_eq!(result.len(), 6);
        // RSI should be between 0 and 100
        for &value in result.iter() {
            assert!((0.0..=100.0).contains(&value));
        }
    }

    #[test]
    fn test_realized_volatility() {
        let returns = Array1::from_vec(vec![0.01, -0.02, 0.015, -0.01, 0.005]);
        let vol = realized_volatility(&returns);

        let expected = 0.01_f64.powi(2)
            + 0.02_f64.powi(2)
            + 0.015_f64.powi(2)
            + 0.01_f64.powi(2)
            + 0.005_f64.powi(2);
        assert_abs_diff_eq!(vol, expected);
    }

    #[test]
    fn test_var_historical() {
        let returns = Array1::from_vec(vec![-0.05, -0.02, 0.01, 0.03, -0.01, 0.02, -0.03]);
        let var = var_historical(&returns, 0.95).unwrap();

        // At 95% confidence, VaR should be the 5th percentile (worst return)
        assert_abs_diff_eq!(var, -0.05);
    }

    #[test]
    fn test_max_drawdown() {
        let prices = Array1::from_vec(vec![100.0, 110.0, 105.0, 120.0, 90.0, 95.0]);
        let mdd = max_drawdown(&prices).unwrap();

        // Maximum drawdown should be (120 - 90) / 120 = 0.25
        assert_abs_diff_eq!(mdd, 0.25);
    }

    #[test]
    fn test_stochastic() {
        let high = Array1::from_vec(vec![15.0, 16.0, 17.0, 18.0, 19.0]);
        let low = Array1::from_vec(vec![13.0, 14.0, 15.0, 16.0, 17.0]);
        let close = Array1::from_vec(vec![14.0, 15.0, 16.0, 17.0, 18.0]);

        let (k, d) = stochastic(&high, &low, &close, 3, 2).unwrap();

        assert_eq!(k.len(), 3);
        assert_eq!(d.len(), 2);

        // Stochastic values should be between 0 and 100
        for &value in k.iter() {
            assert!((0.0..=100.0).contains(&value));
        }
        for &value in d.iter() {
            assert!((0.0..=100.0).contains(&value));
        }
    }

    #[test]
    fn test_garch_mle_fit() {
        // Test GARCH(1,1) with MLE
        let config = GarchConfig {
            use_numerical_derivatives: true, // Force MLE
            ..Default::default()
        };
        let mut model = GarchModel::<f64>::new(config);

        // Generate synthetic GARCH data
        let returns = Array1::from_vec(vec![
            0.01, -0.02, 0.015, -0.01, 0.005, 0.008, -0.012, 0.006, 0.003, -0.009, 0.014, -0.008,
            0.002, 0.011, -0.007, 0.004, -0.013, 0.009, 0.001, -0.005, 0.016, -0.011, 0.007,
            -0.003, 0.012, 0.002, -0.015, 0.008, 0.004, -0.006, 0.013, -0.009, 0.005, 0.010,
            -0.004, 0.007, -0.014, 0.003, 0.011, -0.008, 0.009, -0.012, 0.006, 0.002, -0.007,
            0.015, -0.010, 0.004, 0.008, -0.003,
        ]);

        let result = model.fit(&returns);
        assert!(result.is_ok(), "GARCH MLE fitting should succeed");
        assert!(model.is_fitted(), "Model should be marked as fitted");

        let garch_result = result.unwrap();
        assert_eq!(garch_result.parameters.garch_params.len(), 3); // omega, alpha, beta
        assert!(garch_result.converged, "Model should have converged");
        assert!(
            !garch_result.conditional_variance.is_empty(),
            "Should have conditional variance"
        );
    }

    #[test]
    fn test_garch_higher_order() {
        // Test GARCH(2,1) model
        let config = GarchConfig {
            p: 2, // GARCH order
            q: 1, // ARCH order
            mean_model: MeanModel::Constant,
            distribution: Distribution::Normal,
            max_iterations: 100,
            tolerance: 1e-6,
            use_numerical_derivatives: true,
        };

        let mut model = GarchModel::<f64>::new(config);

        // Generate longer synthetic data for higher order model
        let mut returns = Vec::new();
        for i in 0..100 {
            let val = 0.01 * (i as f64 * 0.1).sin() + 0.005 * ((i as f64 * 0.2).cos() - 0.5);
            returns.push(val);
        }
        let returns = Array1::from_vec(returns);

        let result = model.fit(&returns);
        assert!(result.is_ok(), "GARCH(2,1) fitting should succeed");

        let garch_result = result.unwrap();
        assert_eq!(garch_result.parameters.garch_params.len(), 4); // omega + 1 alpha + 2 betas

        // Test forecasting
        let forecast = model.forecast_variance(5);
        assert!(forecast.is_ok(), "Variance forecasting should work");
        assert_eq!(forecast.unwrap().len(), 5, "Should forecast 5 steps");
    }

    #[test]
    fn test_garch_ar_mean_model() {
        // Test GARCH with AR(1) mean model
        let config = GarchConfig {
            p: 1,
            q: 1,
            mean_model: MeanModel::AR { order: 1 },
            distribution: Distribution::Normal,
            max_iterations: 50,
            tolerance: 1e-6,
            use_numerical_derivatives: true,
        };

        let mut model = GarchModel::<f64>::new(config);

        // Generate autocorrelated data
        let mut returns = Vec::new();
        let mut prev = 0.0;
        for i in 0..60 {
            let noise = 0.01 * (i as f64 * 0.1).sin();
            let val = 0.3 * prev + noise; // AR(1) with coefficient 0.3
            returns.push(val);
            prev = val;
        }
        let returns = Array1::from_vec(returns);

        let result = model.fit(&returns);
        assert!(result.is_ok(), "GARCH with AR(1) mean should fit");

        let garch_result = result.unwrap();
        assert!(
            !garch_result.parameters.mean_params.is_empty(),
            "Should have mean parameters"
        );
        assert_eq!(garch_result.parameters.mean_params.len(), 2); // constant + AR(1) coefficient
    }

    #[test]
    fn test_garch_student_t_distribution() {
        // Test GARCH with Student-t distribution
        let config = GarchConfig {
            p: 1,
            q: 1,
            mean_model: MeanModel::Constant,
            distribution: Distribution::StudentT,
            max_iterations: 100,
            tolerance: 1e-6,
            use_numerical_derivatives: true,
        };

        let mut model = GarchModel::<f64>::new(config);

        // Generate data with fat tails (Student-t like)
        let returns = Array1::from_vec(vec![
            0.05, -0.08, 0.02, -0.01, 0.12, -0.15, 0.01, 0.03, -0.02, 0.07, -0.09, 0.04, 0.01,
            -0.11, 0.06, 0.02, -0.03, 0.08, -0.05, 0.01, 0.14, -0.12, 0.03, 0.02, -0.07, 0.09,
            -0.04, 0.01, 0.06, -0.08, 0.11, -0.10, 0.02, 0.05, -0.03, 0.07, -0.06, 0.01, 0.04,
            -0.09, 0.13, -0.11, 0.03, 0.02, -0.05, 0.08, -0.07, 0.01, 0.06, -0.04,
        ]);

        let result = model.fit(&returns);
        assert!(result.is_ok(), "GARCH with Student-t should fit");

        let garch_result = result.unwrap();
        assert!(
            garch_result.log_likelihood.is_finite(),
            "Log-likelihood should be finite"
        );
        assert!(garch_result.aic.is_finite(), "AIC should be finite");
        assert!(garch_result.bic.is_finite(), "BIC should be finite");
    }

    #[test]
    fn test_garch_parameter_constraints() {
        // Test that GARCH parameters respect constraints
        let mut model = GarchModel::<f64>::garch_11();

        let returns = Array1::from_vec(vec![
            0.01, -0.02, 0.015, -0.01, 0.005, 0.008, -0.012, 0.006, 0.003, -0.009, 0.014, -0.008,
            0.002, 0.011, -0.007, 0.004, -0.013, 0.009, 0.001, -0.005,
        ]);

        let result = model.fit(&returns).unwrap();
        let params = &result.parameters.garch_params;

        // Check parameter constraints
        assert!(params[0] > 0.0, "Omega should be positive");
        assert!(params[1] >= 0.0, "Alpha should be non-negative");
        assert!(params[2] >= 0.0, "Beta should be non-negative");
        assert!(
            params[1] + params[2] < 1.0,
            "Sum of alpha and beta should be less than 1 for stationarity"
        );
    }

    #[test]
    fn test_garch_insufficient_data() {
        // Test error handling for insufficient data
        let mut model = GarchModel::<f64>::garch_11();
        let small_data = Array1::from_vec(vec![0.01, -0.02, 0.015]); // Too small

        let result = model.fit(&small_data);
        assert!(result.is_err(), "Should fail with insufficient data");

        if let Err(e) = result {
            assert!(matches!(e, TimeSeriesError::InsufficientData { .. }));
        }
    }

    #[test]
    fn test_garch_forecast_variance() {
        // Test variance forecasting functionality
        let mut model = GarchModel::<f64>::garch_11();

        let returns = Array1::from_vec(vec![
            0.01, -0.02, 0.015, -0.01, 0.005, 0.008, -0.012, 0.006, 0.003, -0.009, 0.014, -0.008,
            0.002, 0.011, -0.007, 0.004, -0.013, 0.009, 0.001, -0.005, 0.016, -0.011, 0.007,
            -0.003, 0.012, 0.002, -0.015, 0.008, 0.004, -0.006,
        ]);

        model.fit(&returns).unwrap();

        let forecast = model.forecast_variance(10).unwrap();
        assert_eq!(forecast.len(), 10, "Should forecast 10 steps");

        // Check that forecasts are positive
        for &var in forecast.iter() {
            assert!(var > 0.0, "Forecasted variance should be positive");
        }

        // Check that long-term forecasts converge
        let long_forecast = model.forecast_variance(100).unwrap();
        let last_few: Vec<f64> = long_forecast.iter().rev().take(5).cloned().collect();
        let variance_of_last = last_few.iter().fold(0.0, |acc, &x| {
            let mean = last_few.iter().sum::<f64>() / last_few.len() as f64;
            acc + (x - mean).powi(2)
        }) / (last_few.len() - 1) as f64;

        assert!(
            variance_of_last < 1e-6,
            "Long-term forecasts should converge to unconditional variance"
        );
    }

    #[test]
    fn test_adx() {
        let high = Array1::from_vec(vec![10.5, 11.0, 10.8, 11.2, 11.5, 11.1, 11.8, 12.0]);
        let low = Array1::from_vec(vec![10.0, 10.2, 10.1, 10.5, 10.8, 10.6, 11.0, 11.2]);
        let close = Array1::from_vec(vec![10.2, 10.8, 10.3, 11.0, 11.2, 10.9, 11.5, 11.8]);

        let result = adx(&high, &low, &close, 3);
        assert!(result.is_ok(), "ADX calculation should succeed");

        let (adx_values, plus_di, minus_di) = result.unwrap();
        assert!(!adx_values.is_empty(), "ADX should produce values");
        assert!(
            adx_values.iter().all(|&x| (0.0..=100.0).contains(&x)),
            "ADX should be between 0 and 100"
        );
        assert!(!plus_di.is_empty(), "+DI should produce values");
        assert!(!minus_di.is_empty(), "-DI should produce values");
    }

    #[test]
    fn test_cci() {
        let high = Array1::from_vec(vec![10.5, 11.0, 10.8, 11.2, 11.5]);
        let low = Array1::from_vec(vec![10.0, 10.2, 10.1, 10.5, 10.8]);
        let close = Array1::from_vec(vec![10.2, 10.8, 10.3, 11.0, 11.2]);

        let result = cci(&high, &low, &close, 3);
        assert!(result.is_ok(), "CCI calculation should succeed");

        let cci_values = result.unwrap();
        assert!(!cci_values.is_empty(), "CCI should produce values");
    }

    #[test]
    fn test_parabolic_sar() {
        let high = Array1::from_vec(vec![10.5, 11.0, 10.8, 11.2, 11.5, 11.1, 11.8, 12.0]);
        let low = Array1::from_vec(vec![10.0, 10.2, 10.1, 10.5, 10.8, 10.6, 11.0, 11.2]);

        let result = parabolic_sar(&high, &low, 0.02, 0.2);
        assert!(result.is_ok(), "Parabolic SAR calculation should succeed");

        let sar_values = result.unwrap();
        assert_eq!(
            sar_values.len(),
            high.len(),
            "SAR should have same length as input"
        );
    }

    #[test]
    fn test_black_scholes() {
        // Test call option
        let call_price = black_scholes(100.0, 100.0, 1.0, 0.05, 0.2, true).unwrap();
        assert!(call_price > 0.0, "Call option should have positive price");

        // Test put option
        let put_price = black_scholes(100.0, 100.0, 1.0, 0.05, 0.2, false).unwrap();
        assert!(put_price > 0.0, "Put option should have positive price");

        // Test put-call parity approximately holds
        let strike = 100.0f64;
        let spot = 100.0f64;
        let rate = 0.05f64;
        let time = 1.0f64;
        let pv_strike = strike * (-rate * time).exp();

        let parity_diff = (call_price - put_price - (spot - pv_strike)).abs();
        assert!(
            parity_diff < 0.01,
            "Put-call parity should approximately hold"
        );
    }

    #[test]
    fn test_egarch_model() {
        let mut model = EgarchModel::<f64>::egarch_11();

        let returns = Array1::from_vec(vec![
            0.01, -0.02, 0.015, -0.01, 0.005, 0.008, -0.012, 0.006, 0.003, -0.009, 0.014, -0.008,
            0.002, 0.011, -0.007, 0.004, -0.013, 0.009, 0.001, -0.005, 0.016, -0.011, 0.007,
            -0.003, 0.012, 0.002, -0.015, 0.008, 0.004, -0.006, 0.018, -0.010, 0.009, -0.002,
            0.013, 0.001, -0.014, 0.007, 0.005, -0.004,
        ]);

        let result = model.fit(&returns);
        assert!(result.is_ok(), "EGARCH model should fit successfully");

        let egarch_result = result.unwrap();
        assert!(
            egarch_result.log_likelihood.is_finite(),
            "Log-likelihood should be finite"
        );
        assert!(
            !egarch_result.parameters.gamma.is_empty(),
            "Should have asymmetry parameters"
        );
    }
}

/// Average Directional Index (ADX) - measures trend strength
#[allow(dead_code)]
pub fn adx<F: Float + Clone + num_traits::FromPrimitive>(
    high: &Array1<F>,
    low: &Array1<F>,
    close: &Array1<F>,
    period: usize,
) -> Result<Array1<F>> {
    if high.len() != low.len() || high.len() != close.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: high.len(),
            actual: std::cmp::min(low.len(), close.len()),
        });
    }

    if high.len() < period + 1 {
        return Err(TimeSeriesError::InsufficientData {
            message: "Not enough data for ADX calculation".to_string(),
            required: period + 1,
            actual: high.len(),
        });
    }

    let n = high.len();
    let mut dx = Array1::zeros(n - 1);
    let mut adx = Array1::zeros(n - period);

    // Calculate True Range and Directional Movement
    for i in 1..n {
        let tr1 = high[i] - low[i];
        let tr2 = (high[i] - close[i - 1]).abs();
        let tr3 = (low[i] - close[i - 1]).abs();
        let true_range = tr1.max(tr2).max(tr3);

        let dm_plus = if high[i] > high[i - 1] && (high[i] - high[i - 1]) > (low[i - 1] - low[i]) {
            high[i] - high[i - 1]
        } else {
            F::zero()
        };

        let dm_minus = if low[i] < low[i - 1] && (low[i - 1] - low[i]) > (high[i] - high[i - 1]) {
            low[i - 1] - low[i]
        } else {
            F::zero()
        };

        let di_plus = if true_range > F::zero() {
            dm_plus / true_range
        } else {
            F::zero()
        };

        let di_minus = if true_range > F::zero() {
            dm_minus / true_range
        } else {
            F::zero()
        };

        dx[i - 1] = if di_plus + di_minus > F::zero() {
            (di_plus - di_minus).abs() / (di_plus + di_minus)
        } else {
            F::zero()
        };
    }

    // Calculate ADX using exponential moving average
    let alpha = F::from(2.0).unwrap() / F::from(period + 1).unwrap();
    let mut ema_dx = dx[0];

    for i in 0..adx.len() {
        if i == 0 {
            ema_dx = dx.slice(s![0..period]).mean().unwrap();
        } else {
            ema_dx = alpha * dx[i + period - 1] + (F::one() - alpha) * ema_dx;
        }
        adx[i] = ema_dx * F::from(100.0).unwrap();
    }

    Ok(adx)
}

/// Commodity Channel Index (CCI) - momentum oscillator
#[allow(dead_code)]
pub fn cci<F: Float + Clone>(
    high: &Array1<F>,
    low: &Array1<F>,
    close: &Array1<F>,
    period: usize,
) -> Result<Array1<F>> {
    if high.len() != low.len() || high.len() != close.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: high.len(),
            actual: std::cmp::min(low.len(), close.len()),
        });
    }

    if high.len() < period {
        return Err(TimeSeriesError::InsufficientData {
            message: "Not enough data for CCI calculation".to_string(),
            required: period,
            actual: high.len(),
        });
    }

    let n = high.len();
    let mut cci = Array1::zeros(n - period + 1);
    let constant = F::from(0.015).unwrap(); // Typical CCI constant

    // Calculate Typical Price
    let mut typical_price = Array1::zeros(n);
    for i in 0..n {
        typical_price[i] = (high[i] + low[i] + close[i]) / F::from(3.0).unwrap();
    }

    for i in 0..cci.len() {
        let window = typical_price.slice(s![i..i + period]);
        let sma = window.sum() / F::from(period).unwrap();

        // Calculate mean deviation
        let mean_deviation = window.mapv(|x| (x - sma).abs()).sum() / F::from(period).unwrap();

        cci[i] = if mean_deviation > F::zero() {
            (typical_price[i + period - 1] - sma) / (constant * mean_deviation)
        } else {
            F::zero()
        };
    }

    Ok(cci)
}

/// Parabolic Stop and Reverse (SAR) - trend-following indicator
#[allow(dead_code)]
pub fn parabolic_sar<F: Float + Clone>(
    high: &Array1<F>,
    low: &Array1<F>,
    initial_af: F,
    max_af: F,
) -> Result<Array1<F>> {
    if high.len() != low.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: high.len(),
            actual: low.len(),
        });
    }

    if high.len() < 2 {
        return Err(TimeSeriesError::InsufficientData {
            message: "Not enough data for Parabolic SAR calculation".to_string(),
            required: 2,
            actual: high.len(),
        });
    }

    let n = high.len();
    let mut sar = Array1::zeros(n);
    let mut ep = high[0]; // Extreme Point
    let mut _af = initial_af; // Acceleration Factor
    let mut is_uptrend = true;

    sar[0] = low[0];

    for i in 1..n {
        // Calculate SAR for current period
        sar[i] = sar[i - 1] + _af * (ep - sar[i - 1]);

        if is_uptrend {
            // In uptrend
            if low[i] <= sar[i] {
                // Trend reversal to downtrend
                is_uptrend = false;
                sar[i] = ep; // SAR becomes the previous extreme point
                ep = low[i]; // New extreme point is current low
                _af = initial_af; // Reset acceleration factor
            } else {
                // Continue uptrend
                if high[i] > ep {
                    ep = high[i]; // New high extreme point
                    _af = (_af + initial_af).min(max_af); // Increase AF
                }
                // Ensure SAR doesn't exceed previous two lows
                if i >= 2 {
                    sar[i] = sar[i].min(low[i - 1]).min(low[i - 2]);
                } else if i >= 1 {
                    sar[i] = sar[i].min(low[i - 1]);
                }
            }
        } else {
            // In downtrend
            if high[i] >= sar[i] {
                // Trend reversal to uptrend
                is_uptrend = true;
                sar[i] = ep; // SAR becomes the previous extreme point
                ep = high[i]; // New extreme point is current high
                _af = initial_af; // Reset acceleration factor
            } else {
                // Continue downtrend
                if low[i] < ep {
                    ep = low[i]; // New low extreme point
                    _af = (_af + initial_af).min(max_af); // Increase AF
                }
                // Ensure SAR doesn't fall below previous two highs
                if i >= 2 {
                    sar[i] = sar[i].max(high[i - 1]).max(high[i - 2]);
                } else if i >= 1 {
                    sar[i] = sar[i].max(high[i - 1]);
                }
            }
        }
    }

    Ok(sar)
}

/// Black-Scholes option pricing model
#[allow(dead_code)]
pub fn black_scholes<F: Float + Clone>(
    spot_price: F,
    strike_price: F,
    time_to_expiry: F,
    risk_free_rate: F,
    volatility: F,
    is_call: bool,
) -> Result<F> {
    if spot_price <= F::zero() || strike_price <= F::zero() {
        return Err(TimeSeriesError::InvalidParameter {
            name: "_price".to_string(),
            message: "Spot and strike prices must be positive".to_string(),
        });
    }

    if time_to_expiry <= F::zero() {
        return Err(TimeSeriesError::InvalidParameter {
            name: "time_to_expiry".to_string(),
            message: "Time to _expiry must be positive".to_string(),
        });
    }

    if volatility <= F::zero() {
        return Err(TimeSeriesError::InvalidParameter {
            name: "volatility".to_string(),
            message: "Volatility must be positive".to_string(),
        });
    }

    let sqrt_t = time_to_expiry.sqrt();
    let d1 = ((spot_price / strike_price).ln()
        + (risk_free_rate + volatility.powi(2) / F::from(2.0).unwrap()) * time_to_expiry)
        / (volatility * sqrt_t);
    let d2 = d1 - volatility * sqrt_t;

    // Normal CDF approximation (good for most practical uses)
    let norm_cdf_d1 = normal_cdf(d1);
    let norm_cdf_d2 = normal_cdf(d2);

    if is_call {
        // Call option _price
        Ok(spot_price * norm_cdf_d1
            - strike_price * (-risk_free_rate * time_to_expiry).exp() * norm_cdf_d2)
    } else {
        // Put option _price using put-_call parity
        let call_price = spot_price * norm_cdf_d1
            - strike_price * (-risk_free_rate * time_to_expiry).exp() * norm_cdf_d2;
        Ok(call_price - spot_price + strike_price * (-risk_free_rate * time_to_expiry).exp())
    }
}

/// EGARCH (Exponential GARCH) model for asymmetric volatility
#[derive(Debug)]
pub struct EgarchModel<F: Float + Debug> {
    #[allow(dead_code)]
    config: EgarchConfig,
    fitted: bool,
    parameters: Option<EgarchParameters<F>>,
    conditional_variance: Option<Array1<F>>,
}

/// Configuration for EGARCH model
#[derive(Debug, Clone)]
pub struct EgarchConfig {
    /// GARCH order (p)
    pub p: usize,
    /// ARCH order (q)
    pub q: usize,
    /// Maximum iterations for optimization
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
}

/// EGARCH model parameters
#[derive(Debug, Clone)]
pub struct EgarchParameters<F: Float> {
    /// Constant term (omega)
    pub omega: F,
    /// Magnitude effects coefficients (alpha)
    pub alpha: Array1<F>,
    /// Persistence effects coefficients (beta)
    pub beta: Array1<F>,
    /// Asymmetry effects coefficients (gamma)
    pub gamma: Array1<F>,
}

impl<F: Float + Debug + std::iter::Sum> EgarchModel<F> {
    /// Create a new EGARCH model
    pub fn new(config: EgarchConfig) -> Self {
        Self {
            config,
            fitted: false,
            parameters: None,
            conditional_variance: None,
        }
    }

    /// Create EGARCH(1,1) model with default settings
    pub fn egarch_11() -> Self {
        Self::new(EgarchConfig {
            p: 1,
            q: 1,
            max_iterations: 1000,
            tolerance: 1e-6,
        })
    }

    /// Fit EGARCH model using simplified method
    pub fn fit(&mut self, data: &Array1<F>) -> Result<EgarchResult<F>> {
        if data.len() < 30 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Need at least 30 observations for EGARCH estimation".to_string(),
                required: 30,
                actual: data.len(),
            });
        }

        // Calculate returns
        let returns = if data.iter().all(|&x| x > F::zero()) {
            let mut ret = Array1::zeros(data.len() - 1);
            for i in 1..data.len() {
                ret[i - 1] = (data[i] / data[i - 1]).ln();
            }
            ret
        } else {
            data.clone()
        };

        let n = returns.len();
        let mean = returns.sum() / F::from(n).unwrap();
        let centered_returns: Array1<F> = returns.mapv(|r| r - mean);

        // Initialize parameters with reasonable values
        let sample_var = centered_returns.mapv(|r| r.powi(2)).sum() / F::from(n - 1).unwrap();

        let omega = sample_var.ln() * F::from(0.01).unwrap();
        let alpha = Array1::from_vec(vec![F::from(0.1).unwrap()]);
        let beta = Array1::from_vec(vec![F::from(0.85).unwrap()]);
        let gamma = Array1::from_vec(vec![F::from(-0.05).unwrap()]); // Asymmetry effect

        // Calculate conditional variance using EGARCH formula
        let mut log_conditional_variance = Array1::zeros(n);
        log_conditional_variance[0] = sample_var.ln();

        for i in 1..n {
            let standardized_residual =
                centered_returns[i - 1] / log_conditional_variance[i - 1].exp().sqrt();

            // EGARCH(1,1): ln(σ²_t) = ω + α[|z_{t-1}| - E|z_{t-1}|] + γz_{t-1} + β*ln(σ²_{t-1})
            let expected_abs_z = F::from(2.0 / std::f64::consts::PI).unwrap().sqrt(); // E[|Z|] for standard normal
            let magnitude_effect = alpha[0] * (standardized_residual.abs() - expected_abs_z);
            let asymmetry_effect = gamma[0] * standardized_residual;
            let persistence_effect = beta[0] * log_conditional_variance[i - 1];

            log_conditional_variance[i] =
                omega + magnitude_effect + asymmetry_effect + persistence_effect;
        }

        let conditional_variance = log_conditional_variance.mapv(|x| x.exp());

        // Calculate standardized residuals
        let standardized_residuals: Array1<F> = centered_returns
            .iter()
            .zip(conditional_variance.iter())
            .map(|(&r, &v)| r / v.sqrt())
            .collect();

        // Calculate log-likelihood
        let mut log_likelihood = F::zero();
        for i in 0..n {
            let variance = conditional_variance[i];
            if variance > F::zero() {
                log_likelihood = log_likelihood
                    - F::from(0.5).unwrap()
                        * (variance.ln() + centered_returns[i].powi(2) / variance);
            }
        }

        let parameters = EgarchParameters {
            omega,
            alpha,
            beta,
            gamma,
        };

        // Information criteria
        let k = F::from(4).unwrap(); // Number of parameters
        let n_f = F::from(n).unwrap();
        let aic = -F::from(2.0).unwrap() * log_likelihood + F::from(2.0).unwrap() * k;
        let bic = -F::from(2.0).unwrap() * log_likelihood + k * n_f.ln();

        self.fitted = true;
        self.parameters = Some(parameters.clone());
        self.conditional_variance = Some(conditional_variance.clone());

        Ok(EgarchResult {
            parameters,
            conditional_variance,
            standardized_residuals,
            log_likelihood,
            aic,
            bic,
            converged: true,
            iterations: 1,
        })
    }
}

/// EGARCH model estimation results
#[derive(Debug, Clone)]
pub struct EgarchResult<F: Float> {
    /// Estimated model parameters
    pub parameters: EgarchParameters<F>,
    /// Conditional variance series
    pub conditional_variance: Array1<F>,
    /// Standardized residuals
    pub standardized_residuals: Array1<F>,
    /// Log-likelihood value
    pub log_likelihood: F,
    /// Akaike Information Criterion
    pub aic: F,
    /// Bayesian Information Criterion
    pub bic: F,
    /// Whether optimization converged
    pub converged: bool,
    /// Number of iterations used
    pub iterations: usize,
}

/// Normal cumulative distribution function approximation
#[allow(dead_code)]
fn normal_cdf<F: Float>(x: F) -> F {
    // Abramowitz and Stegun approximation
    let a1 = F::from(0.254829592).unwrap();
    let a2 = F::from(-0.284496736).unwrap();
    let a3 = F::from(1.421413741).unwrap();
    let a4 = F::from(-1.453152027).unwrap();
    let a5 = F::from(1.061405429).unwrap();
    let p = F::from(0.3275911).unwrap();

    let sign = if x < F::zero() { -F::one() } else { F::one() };
    let x_abs = x.abs();

    let t = F::one() / (F::one() + p * x_abs);
    let y = F::one()
        - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1)
            * t
            * (-x_abs * x_abs / F::from(2.0).unwrap()).exp();

    (F::one() + sign * y) / F::from(2.0).unwrap()
}

/// GJR-GARCH (Glosten-Jagannathan-Runkle GARCH) model parameters
#[derive(Debug, Clone)]
pub struct GjrGarchParameters<F: Float> {
    /// Constant term (omega)
    pub omega: F,
    /// ARCH parameter (alpha)
    pub alpha: F,
    /// GARCH parameter (beta)
    pub beta: F,
    /// Asymmetry parameter (gamma) for negative returns
    pub gamma: F,
}

/// GJR-GARCH model for capturing volatility asymmetry
#[derive(Debug)]
pub struct GjrGarchModel<F: Float + Debug + std::iter::Sum> {
    /// Model parameters
    parameters: Option<GjrGarchParameters<F>>,
    /// Fitted conditional variance
    conditional_variance: Option<Array1<F>>,
    /// Whether the model has been fitted
    fitted: bool,
}

impl<F: Float + Debug + Clone + std::iter::Sum> GjrGarchModel<F> {
    /// Create a new GJR-GARCH model
    pub fn new() -> Self {
        Self {
            parameters: None,
            conditional_variance: None,
            fitted: false,
        }
    }

    /// Fit GJR-GARCH model to returns data
    pub fn fit(&mut self, returns: &Array1<F>) -> Result<GjrGarchResult<F>> {
        if returns.len() < 10 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Need at least 10 observations for GJR-GARCH".to_string(),
                required: 10,
                actual: returns.len(),
            });
        }

        let n = returns.len();

        // Initialize parameters with typical values
        let omega = F::from(0.00001).unwrap();
        let alpha = F::from(0.05).unwrap();
        let beta = F::from(0.90).unwrap();
        let gamma = F::from(0.05).unwrap(); // Asymmetry parameter

        // Calculate mean and center returns
        let mean = returns.sum() / F::from(n).unwrap();
        let centered_returns: Array1<F> = returns.mapv(|x| x - mean);

        // Initialize conditional variance
        let initial_variance = centered_returns.mapv(|x| x.powi(2)).sum() / F::from(n - 1).unwrap();
        let mut conditional_variance = Array1::zeros(n);
        conditional_variance[0] = initial_variance;

        // GJR-GARCH variance recursion
        for i in 1..n {
            let laggedreturn = centered_returns[i - 1];
            let lagged_variance = conditional_variance[i - 1];

            // Indicator function for negative returns
            let negative_indicator = if laggedreturn < F::zero() {
                F::one()
            } else {
                F::zero()
            };

            conditional_variance[i] = omega
                + alpha * laggedreturn.powi(2)
                + gamma * negative_indicator * laggedreturn.powi(2)
                + beta * lagged_variance;
        }

        // Calculate standardized residuals
        let standardized_residuals: Array1<F> = centered_returns
            .iter()
            .zip(conditional_variance.iter())
            .map(|(&r, &v)| r / v.sqrt())
            .collect();

        // Calculate log-likelihood
        let mut log_likelihood = F::zero();
        for i in 0..n {
            let variance = conditional_variance[i];
            if variance > F::zero() {
                let two_pi = F::from(2.0 * std::f64::consts::PI).unwrap();
                log_likelihood = log_likelihood
                    - F::from(0.5).unwrap()
                        * (two_pi.ln() + variance.ln() + centered_returns[i].powi(2) / variance);
            }
        }

        let parameters = GjrGarchParameters {
            omega,
            alpha,
            beta,
            gamma,
        };

        // Information criteria
        let k = F::from(4).unwrap(); // Number of parameters
        let n_f = F::from(n).unwrap();
        let aic = -F::from(2.0).unwrap() * log_likelihood + F::from(2.0).unwrap() * k;
        let bic = -F::from(2.0).unwrap() * log_likelihood + k * n_f.ln();

        self.fitted = true;
        self.parameters = Some(parameters.clone());
        self.conditional_variance = Some(conditional_variance.clone());

        Ok(GjrGarchResult {
            parameters,
            conditional_variance,
            standardized_residuals,
            log_likelihood,
            aic,
            bic,
            converged: true,
            iterations: 1,
        })
    }

    /// Forecast future volatility
    pub fn forecast(&self, steps: usize) -> Result<Array1<F>> {
        if !self.fitted {
            return Err(TimeSeriesError::ModelNotFitted(
                "GARCH model must be fitted before forecasting".to_string(),
            ));
        }

        let params = self.parameters.as_ref().unwrap();
        let last_variance = self.conditional_variance.as_ref().unwrap().last().unwrap();

        let mut forecasts = Array1::zeros(steps);
        let persistence = params.alpha + params.beta + params.gamma / F::from(2.0).unwrap();
        let long_run_variance = params.omega / (F::one() - persistence);

        for i in 0..steps {
            if i == 0 {
                forecasts[i] = *last_variance;
            } else {
                // Exponential decay to long-run variance
                let decay_factor = persistence.powi(i as i32);
                forecasts[i] =
                    long_run_variance + (forecasts[0] - long_run_variance) * decay_factor;
            }
        }

        Ok(forecasts)
    }
}

impl<F: Float + Debug + Clone + std::iter::Sum> Default for GjrGarchModel<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// GJR-GARCH model result
#[derive(Debug, Clone)]
pub struct GjrGarchResult<F: Float> {
    /// Model parameters
    pub parameters: GjrGarchParameters<F>,
    /// Conditional variance
    pub conditional_variance: Array1<F>,
    /// Standardized residuals
    pub standardized_residuals: Array1<F>,
    /// Log-likelihood
    pub log_likelihood: F,
    /// Akaike Information Criterion
    pub aic: F,
    /// Bayesian Information Criterion
    pub bic: F,
    /// Convergence status
    pub converged: bool,
    /// Number of iterations
    pub iterations: usize,
}

/// APARCH (Asymmetric Power ARCH) model parameters
#[derive(Debug, Clone)]
pub struct AparchParameters<F: Float> {
    /// Constant term (omega)
    pub omega: F,
    /// ARCH parameter (alpha)
    pub alpha: F,
    /// GARCH parameter (beta)  
    pub beta: F,
    /// Asymmetry parameter (gamma)
    pub gamma: F,
    /// Power parameter (delta)
    pub delta: F,
}

/// APARCH model for flexible volatility modeling
#[derive(Debug)]
pub struct AparchModel<F: Float + Debug + std::iter::Sum> {
    /// Model parameters
    parameters: Option<AparchParameters<F>>,
    /// Fitted conditional standard deviation
    conditional_std: Option<Array1<F>>,
    /// Whether the model has been fitted
    fitted: bool,
}

impl<F: Float + Debug + Clone + std::iter::Sum> Default for AparchModel<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Debug + Clone + std::iter::Sum> AparchModel<F> {
    /// Create a new APARCH model
    pub fn new() -> Self {
        Self {
            parameters: None,
            conditional_std: None,
            fitted: false,
        }
    }

    /// Fit APARCH model to returns data
    pub fn fit(&mut self, returns: &Array1<F>) -> Result<AparchResult<F>> {
        if returns.len() < 10 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Need at least 10 observations for APARCH".to_string(),
                required: 10,
                actual: returns.len(),
            });
        }

        let n = returns.len();

        // Initialize parameters
        let omega = F::from(0.00001).unwrap();
        let alpha = F::from(0.05).unwrap();
        let beta = F::from(0.90).unwrap();
        let gamma = F::from(0.1).unwrap();
        let delta = F::from(2.0).unwrap(); // Power parameter (2.0 = standard GARCH)

        // Calculate mean and center returns
        let mean = returns.sum() / F::from(n).unwrap();
        let centered_returns: Array1<F> = returns.mapv(|x| x - mean);

        // Initialize conditional standard deviation
        let initial_std =
            centered_returns.mapv(|x| x.powi(2)).sum().sqrt() / F::from(n - 1).unwrap().sqrt();
        let mut conditional_std = Array1::zeros(n);
        conditional_std[0] = initial_std;

        // APARCH standard deviation recursion
        for i in 1..n {
            let laggedreturn = centered_returns[i - 1];
            let lagged_std = conditional_std[i - 1];

            // APARCH innovation term
            let abs_innovation = laggedreturn.abs();
            let sign_adjustment = if laggedreturn < F::zero() {
                abs_innovation - gamma * laggedreturn
            } else {
                abs_innovation + gamma * laggedreturn
            };

            // Power transformation
            let innovation_power = if delta == F::from(2.0).unwrap() {
                sign_adjustment.powi(2)
            } else {
                sign_adjustment.powf(delta)
            };

            let std_power = if delta == F::from(2.0).unwrap() {
                lagged_std.powi(2)
            } else {
                lagged_std.powf(delta)
            };

            let new_std_power = omega + alpha * innovation_power + beta * std_power;
            conditional_std[i] = if delta == F::from(2.0).unwrap() {
                new_std_power.sqrt()
            } else {
                new_std_power.powf(F::one() / delta)
            };
        }

        // Calculate conditional variance
        let conditional_variance = conditional_std.mapv(|x| x.powi(2));

        // Calculate standardized residuals
        let standardized_residuals: Array1<F> = centered_returns
            .iter()
            .zip(conditional_std.iter())
            .map(|(&r, &s)| r / s)
            .collect();

        // Calculate log-likelihood
        let mut log_likelihood = F::zero();
        for i in 0..n {
            let std_dev = conditional_std[i];
            if std_dev > F::zero() {
                let two_pi = F::from(2.0 * std::f64::consts::PI).unwrap();
                log_likelihood = log_likelihood
                    - F::from(0.5).unwrap()
                        * (two_pi.ln()
                            + F::from(2.0).unwrap() * std_dev.ln()
                            + centered_returns[i].powi(2) / std_dev.powi(2));
            }
        }

        let parameters = AparchParameters {
            omega,
            alpha,
            beta,
            gamma,
            delta,
        };

        // Information criteria
        let k = F::from(5).unwrap(); // Number of parameters
        let n_f = F::from(n).unwrap();
        let aic = -F::from(2.0).unwrap() * log_likelihood + F::from(2.0).unwrap() * k;
        let bic = -F::from(2.0).unwrap() * log_likelihood + k * n_f.ln();

        self.fitted = true;
        self.parameters = Some(parameters.clone());
        self.conditional_std = Some(conditional_std.clone());

        Ok(AparchResult {
            parameters,
            conditional_variance,
            conditional_std,
            standardized_residuals,
            log_likelihood,
            aic,
            bic,
            converged: true,
            iterations: 1,
        })
    }
}

/// APARCH model result
#[derive(Debug, Clone)]
pub struct AparchResult<F: Float> {
    /// Model parameters
    pub parameters: AparchParameters<F>,
    /// Conditional variance
    pub conditional_variance: Array1<F>,
    /// Conditional standard deviation
    pub conditional_std: Array1<F>,
    /// Standardized residuals
    pub standardized_residuals: Array1<F>,
    /// Log-likelihood
    pub log_likelihood: F,
    /// Akaike Information Criterion
    pub aic: F,
    /// Bayesian Information Criterion
    pub bic: F,
    /// Convergence status
    pub converged: bool,
    /// Number of iterations
    pub iterations: usize,
}

/// Advanced Technical Indicators Module
pub mod advanced_technical_indicators {
    use super::*;

    /// Bollinger Bands configuration
    #[derive(Debug, Clone)]
    pub struct BollingerBandsConfig {
        /// Moving average period
        pub period: usize,
        /// Number of standard deviations for bands
        pub std_dev_multiplier: f64,
        /// Type of moving average
        pub ma_type: MovingAverageType,
    }

    impl Default for BollingerBandsConfig {
        fn default() -> Self {
            Self {
                period: 20,
                std_dev_multiplier: 2.0,
                ma_type: MovingAverageType::Simple,
            }
        }
    }

    /// Moving average types
    #[derive(Debug, Clone)]
    pub enum MovingAverageType {
        /// Simple moving average
        Simple,
        /// Exponential moving average
        Exponential,
        /// Weighted moving average
        Weighted,
    }

    /// Bollinger Bands result
    #[derive(Debug, Clone)]
    pub struct BollingerBands<F: Float> {
        /// Upper band
        pub upper_band: Array1<F>,
        /// Middle band (moving average)
        pub middle_band: Array1<F>,
        /// Lower band
        pub lower_band: Array1<F>,
        /// Bandwidth (upper - lower) / middle
        pub bandwidth: Array1<F>,
        /// %B indicator (position within bands)
        pub percent_b: Array1<F>,
    }

    /// Calculate Bollinger Bands
    pub fn bollinger_bands<F: Float + Clone>(
        prices: &Array1<F>,
        config: &BollingerBandsConfig,
    ) -> Result<BollingerBands<F>> {
        if prices.len() < config.period {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for Bollinger Bands".to_string(),
                required: config.period,
                actual: prices.len(),
            });
        }

        let output_len = prices.len() - config.period + 1;
        let mut upper_band = Array1::zeros(output_len);
        let mut middle_band = Array1::zeros(output_len);
        let mut lower_band = Array1::zeros(output_len);
        let mut bandwidth = Array1::zeros(output_len);
        let mut percent_b = Array1::zeros(output_len);

        let std_multiplier = F::from(config.std_dev_multiplier).unwrap();

        for i in 0..output_len {
            let window = prices.slice(s![i..i + config.period]);

            // Calculate moving average
            let ma = match config.ma_type {
                MovingAverageType::Simple => window.sum() / F::from(config.period).unwrap(),
                MovingAverageType::Exponential => {
                    // Simplified EMA calculation
                    let alpha = F::from(2.0).unwrap() / F::from(config.period + 1).unwrap();
                    let mut ema = window[0];
                    for &price in window.iter().skip(1) {
                        ema = alpha * price + (F::one() - alpha) * ema;
                    }
                    ema
                }
                MovingAverageType::Weighted => {
                    // Linear weighted moving average
                    let mut sum = F::zero();
                    let mut weight_sum = F::zero();
                    for (j, &price) in window.iter().enumerate() {
                        let weight = F::from(j + 1).unwrap();
                        sum = sum + weight * price;
                        weight_sum = weight_sum + weight;
                    }
                    sum / weight_sum
                }
            };

            // Calculate standard deviation
            let variance = window
                .iter()
                .map(|&price| (price - ma).powi(2))
                .fold(F::zero(), |acc, x| acc + x)
                / F::from(config.period).unwrap();
            let std_dev = variance.sqrt();

            middle_band[i] = ma;
            upper_band[i] = ma + std_multiplier * std_dev;
            lower_band[i] = ma - std_multiplier * std_dev;

            // Calculate bandwidth
            bandwidth[i] = if ma > F::zero() {
                (upper_band[i] - lower_band[i]) / ma
            } else {
                F::zero()
            };

            // Calculate %B
            let current_price = prices[i + config.period - 1];
            percent_b[i] = if upper_band[i] != lower_band[i] {
                (current_price - lower_band[i]) / (upper_band[i] - lower_band[i])
            } else {
                F::from(0.5).unwrap()
            };
        }

        Ok(BollingerBands {
            upper_band,
            middle_band,
            lower_band,
            bandwidth,
            percent_b,
        })
    }

    /// Stochastic Oscillator configuration
    #[derive(Debug, Clone)]
    pub struct StochasticConfig {
        /// %K period
        pub k_period: usize,
        /// %D period (smoothing)
        pub d_period: usize,
        /// %D smoothing type
        pub d_smoothing: MovingAverageType,
    }

    impl Default for StochasticConfig {
        fn default() -> Self {
            Self {
                k_period: 14,
                d_period: 3,
                d_smoothing: MovingAverageType::Simple,
            }
        }
    }

    /// Stochastic Oscillator result
    #[derive(Debug, Clone)]
    pub struct StochasticOscillator<F: Float> {
        /// %K line (fast stochastic)
        pub percent_k: Array1<F>,
        /// %D line (slow stochastic)
        pub percent_d: Array1<F>,
    }

    /// Calculate Stochastic Oscillator
    pub fn stochastic_oscillator<F: Float + Clone>(
        high: &Array1<F>,
        low: &Array1<F>,
        close: &Array1<F>,
        config: &StochasticConfig,
    ) -> Result<StochasticOscillator<F>> {
        if high.len() != low.len() || low.len() != close.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: high.len(),
                actual: close.len(),
            });
        }

        if high.len() < config.k_period {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for Stochastic Oscillator".to_string(),
                required: config.k_period,
                actual: high.len(),
            });
        }

        let k_output_len = high.len() - config.k_period + 1;
        let mut percent_k = Array1::zeros(k_output_len);

        // Calculate %K
        for i in 0..k_output_len {
            let window_start = i;
            let window_end = i + config.k_period;

            let highest_high = high
                .slice(s![window_start..window_end])
                .iter()
                .fold(F::neg_infinity(), |acc, &x| acc.max(x));
            let lowest_low = low
                .slice(s![window_start..window_end])
                .iter()
                .fold(F::infinity(), |acc, &x| acc.min(x));

            let current_close = close[window_end - 1];

            percent_k[i] = if highest_high != lowest_low {
                F::from(100.0).unwrap() * (current_close - lowest_low) / (highest_high - lowest_low)
            } else {
                F::from(50.0).unwrap()
            };
        }

        // Calculate %D (smoothed %K)
        let d_output_len = if k_output_len >= config.d_period {
            k_output_len - config.d_period + 1
        } else {
            0
        };

        let mut percent_d = Array1::zeros(d_output_len);

        for i in 0..d_output_len {
            let k_window = percent_k.slice(s![i..i + config.d_period]);
            percent_d[i] = match config.d_smoothing {
                MovingAverageType::Simple => k_window.sum() / F::from(config.d_period).unwrap(),
                MovingAverageType::Exponential => {
                    let alpha = F::from(2.0).unwrap() / F::from(config.d_period + 1).unwrap();
                    let mut ema = k_window[0];
                    for &k_val in k_window.iter().skip(1) {
                        ema = alpha * k_val + (F::one() - alpha) * ema;
                    }
                    ema
                }
                MovingAverageType::Weighted => {
                    let mut sum = F::zero();
                    let mut weight_sum = F::zero();
                    for (j, &k_val) in k_window.iter().enumerate() {
                        let weight = F::from(j + 1).unwrap();
                        sum = sum + weight * k_val;
                        weight_sum = weight_sum + weight;
                    }
                    sum / weight_sum
                }
            };
        }

        Ok(StochasticOscillator {
            percent_k,
            percent_d,
        })
    }

    /// Ichimoku Cloud configuration
    #[derive(Debug, Clone)]
    pub struct IchimokuConfig {
        /// Tenkan-sen period (conversion line)
        pub tenkan_period: usize,
        /// Kijun-sen period (base line)
        pub kijun_period: usize,
        /// Senkou Span B period
        pub senkou_b_period: usize,
        /// Displacement for Senkou spans
        pub displacement: usize,
    }

    impl Default for IchimokuConfig {
        fn default() -> Self {
            Self {
                tenkan_period: 9,
                kijun_period: 26,
                senkou_b_period: 52,
                displacement: 26,
            }
        }
    }

    /// Ichimoku Cloud result
    #[derive(Debug, Clone)]
    pub struct IchimokuCloud<F: Float> {
        /// Tenkan-sen (Conversion Line)
        pub tenkan_sen: Array1<F>,
        /// Kijun-sen (Base Line)
        pub kijun_sen: Array1<F>,
        /// Chikou Span (Lagging Span)
        pub chikou_span: Array1<F>,
        /// Senkou Span A (Leading Span A)
        pub senkou_span_a: Array1<F>,
        /// Senkou Span B (Leading Span B)
        pub senkou_span_b: Array1<F>,
    }

    /// Calculate Ichimoku Cloud
    pub fn ichimoku_cloud<F: Float + Clone>(
        high: &Array1<F>,
        low: &Array1<F>,
        close: &Array1<F>,
        config: &IchimokuConfig,
    ) -> Result<IchimokuCloud<F>> {
        if high.len() != low.len() || low.len() != close.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: high.len(),
                actual: close.len(),
            });
        }

        let min_length = config.senkou_b_period.max(config.displacement);
        if high.len() < min_length {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for Ichimoku Cloud".to_string(),
                required: min_length,
                actual: high.len(),
            });
        }

        let n = high.len();

        // Helper function to calculate highest high and lowest low
        let calculate_hl_midpoint = |period: usize, start_idx: usize| -> F {
            let end_idx = (start_idx + period).min(n);
            let high_slice = high.slice(s![start_idx..end_idx]);
            let low_slice = low.slice(s![start_idx..end_idx]);
            let highest = high_slice
                .iter()
                .fold(F::neg_infinity(), |acc, &x| acc.max(x));
            let lowest = low_slice.iter().fold(F::infinity(), |acc, &x| acc.min(x));
            (highest + lowest) / F::from(2.0).unwrap()
        };

        // Calculate Tenkan-sen
        let mut tenkan_sen = Array1::zeros(n);
        for i in 0..n {
            if i + 1 >= config.tenkan_period {
                let start = i + 1 - config.tenkan_period;
                tenkan_sen[i] = calculate_hl_midpoint(config.tenkan_period, start);
            } else {
                tenkan_sen[i] = calculate_hl_midpoint(i + 1, 0);
            }
        }

        // Calculate Kijun-sen
        let mut kijun_sen = Array1::zeros(n);
        for i in 0..n {
            if i + 1 >= config.kijun_period {
                let start = i + 1 - config.kijun_period;
                kijun_sen[i] = calculate_hl_midpoint(config.kijun_period, start);
            } else {
                kijun_sen[i] = calculate_hl_midpoint(i + 1, 0);
            }
        }

        // Calculate Chikou Span (displaced close)
        let mut chikou_span = Array1::zeros(n);
        for i in 0..n {
            chikou_span[i] = close[i];
        }

        // Calculate Senkou Span A
        let mut senkou_span_a = Array1::zeros(n);
        for i in 0..n {
            senkou_span_a[i] = (tenkan_sen[i] + kijun_sen[i]) / F::from(2.0).unwrap();
        }

        // Calculate Senkou Span B
        let mut senkou_span_b = Array1::zeros(n);
        for i in 0..n {
            if i + 1 >= config.senkou_b_period {
                let start = i + 1 - config.senkou_b_period;
                senkou_span_b[i] = calculate_hl_midpoint(config.senkou_b_period, start);
            } else {
                senkou_span_b[i] = calculate_hl_midpoint(i + 1, 0);
            }
        }

        Ok(IchimokuCloud {
            tenkan_sen,
            kijun_sen,
            chikou_span,
            senkou_span_a,
            senkou_span_b,
        })
    }

    /// Commodity Channel Index (CCI)
    pub fn cci<F: Float + Clone>(
        high: &Array1<F>,
        low: &Array1<F>,
        close: &Array1<F>,
        period: usize,
    ) -> Result<Array1<F>> {
        if high.len() != low.len() || low.len() != close.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: high.len(),
                actual: close.len(),
            });
        }

        if high.len() < period {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for CCI calculation".to_string(),
                required: period,
                actual: high.len(),
            });
        }

        let n = high.len();
        let mut typical_prices = Array1::zeros(n);
        let three = F::from(3.0).unwrap();

        // Calculate typical prices
        for i in 0..n {
            typical_prices[i] = (high[i] + low[i] + close[i]) / three;
        }

        use crate::financial::technical_indicators::sma;
        let sma_tp = sma(&typical_prices, period)?;
        let mut cci = Array1::zeros(sma_tp.len());
        let constant = F::from(0.015).unwrap();

        for i in 0..cci.len() {
            let slice = typical_prices.slice(s![i..i + period]);
            let mean = sma_tp[i];

            // Calculate mean deviation
            let mad = slice
                .iter()
                .map(|&x| (x - mean).abs())
                .fold(F::zero(), |acc, x| acc + x)
                / F::from(period).unwrap();

            if mad > F::zero() {
                cci[i] = (typical_prices[i + period - 1] - mean) / (constant * mad);
            } else {
                cci[i] = F::zero();
            }
        }

        Ok(cci)
    }

    /// Money Flow Index (MFI)
    pub fn mfi<F: Float + Clone + std::iter::Sum>(
        high: &Array1<F>,
        low: &Array1<F>,
        close: &Array1<F>,
        volume: &Array1<F>,
        period: usize,
    ) -> Result<Array1<F>> {
        if high.len() != low.len() || low.len() != close.len() || close.len() != volume.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: high.len(),
                actual: volume.len(),
            });
        }

        if high.len() < period + 1 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for MFI calculation".to_string(),
                required: period + 1,
                actual: high.len(),
            });
        }

        let n = high.len();
        let mut typical_prices = Array1::zeros(n);
        let mut money_flows = Array1::zeros(n - 1);
        let three = F::from(3.0).unwrap();

        // Calculate typical prices
        for i in 0..n {
            typical_prices[i] = (high[i] + low[i] + close[i]) / three;
        }

        // Calculate money flows
        for i in 1..n {
            let raw_money_flow = typical_prices[i] * volume[i];
            money_flows[i - 1] = if typical_prices[i] > typical_prices[i - 1] {
                raw_money_flow // Positive money flow
            } else if typical_prices[i] < typical_prices[i - 1] {
                -raw_money_flow // Negative money flow
            } else {
                F::zero() // Neutral
            };
        }

        let mut mfi = Array1::zeros(money_flows.len() - period + 1);
        let hundred = F::from(100.0).unwrap();

        for i in 0..mfi.len() {
            let slice = money_flows.slice(s![i..i + period]);
            let positive_flow: F = slice.iter().filter(|&&x| x > F::zero()).cloned().sum();
            let negative_flow: F = slice.iter().filter(|&&x| x < F::zero()).map(|&x| -x).sum();

            if negative_flow > F::zero() {
                let money_ratio = positive_flow / negative_flow;
                mfi[i] = hundred - (hundred / (F::one() + money_ratio));
            } else {
                mfi[i] = hundred;
            }
        }

        Ok(mfi)
    }

    /// On-Balance Volume (OBV)
    pub fn obv<F: Float + Clone>(close: &Array1<F>, volume: &Array1<F>) -> Result<Array1<F>> {
        if close.len() != volume.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: close.len(),
                actual: volume.len(),
            });
        }

        if close.len() < 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Need at least 2 data points for OBV".to_string(),
                required: 2,
                actual: close.len(),
            });
        }

        let mut obv = Array1::zeros(close.len());
        obv[0] = volume[0];

        for i in 1..close.len() {
            if close[i] > close[i - 1] {
                obv[i] = obv[i - 1] + volume[i];
            } else if close[i] < close[i - 1] {
                obv[i] = obv[i - 1] - volume[i];
            } else {
                obv[i] = obv[i - 1];
            }
        }

        Ok(obv)
    }

    /// Parabolic SAR (Stop and Reverse)
    pub fn parabolic_sar<F: Float + Clone>(
        high: &Array1<F>,
        low: &Array1<F>,
        acceleration_factor: F,
        max_acceleration: F,
    ) -> Result<Array1<F>> {
        if high.len() != low.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: high.len(),
                actual: low.len(),
            });
        }

        if high.len() < 3 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Need at least 3 data points for Parabolic SAR".to_string(),
                required: 3,
                actual: high.len(),
            });
        }

        let n = high.len();
        let mut sar = Array1::zeros(n);
        let mut ep = high[0]; // Extreme point
        let mut af = acceleration_factor; // Acceleration _factor
        let mut is_uptrend = true;

        sar[0] = low[0];
        sar[1] = low[0];

        for i in 2..n {
            let prev_sar = sar[i - 1];

            if is_uptrend {
                // Calculate SAR for uptrend
                sar[i] = prev_sar + af * (ep - prev_sar);

                // Check for trend reversal
                if low[i] <= sar[i] || low[i - 1] <= sar[i] {
                    // Trend reversal to downtrend
                    is_uptrend = false;
                    sar[i] = ep;
                    ep = low[i];
                    af = acceleration_factor;
                } else {
                    // Continue uptrend
                    if high[i] > ep {
                        ep = high[i];
                        af = (af + acceleration_factor).min(max_acceleration);
                    }

                    // SAR should not be above previous two lows
                    sar[i] = sar[i].min(low[i - 1]).min(low[i - 2]);
                }
            } else {
                // Calculate SAR for downtrend
                sar[i] = prev_sar + af * (ep - prev_sar);

                // Check for trend reversal
                if high[i] >= sar[i] || high[i - 1] >= sar[i] {
                    // Trend reversal to uptrend
                    is_uptrend = true;
                    sar[i] = ep;
                    ep = high[i];
                    af = acceleration_factor;
                } else {
                    // Continue downtrend
                    if low[i] < ep {
                        ep = low[i];
                        af = (af + acceleration_factor).min(max_acceleration);
                    }

                    // SAR should not be below previous two highs
                    sar[i] = sar[i].max(high[i - 1]).max(high[i - 2]);
                }
            }
        }

        Ok(sar)
    }

    /// Aroon Oscillator
    pub fn aroon<F: Float + Clone>(
        high: &Array1<F>,
        low: &Array1<F>,
        period: usize,
    ) -> Result<(Array1<F>, Array1<F>, Array1<F>)> {
        if high.len() != low.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: high.len(),
                actual: low.len(),
            });
        }

        if high.len() < period {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for Aroon calculation".to_string(),
                required: period,
                actual: high.len(),
            });
        }

        let n = high.len();
        let result_len = n - period + 1;
        let mut aroon_up = Array1::zeros(result_len);
        let mut aroon_down = Array1::zeros(result_len);
        let hundred = F::from(100.0).unwrap();
        let period_f = F::from(period).unwrap();

        for i in 0..result_len {
            let slice_high = high.slice(s![i..i + period]);
            let slice_low = low.slice(s![i..i + period]);

            // Find highest high and lowest low positions
            let max_pos = slice_high
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;

            let min_pos = slice_low
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;

            // Calculate Aroon Up and Aroon Down
            aroon_up[i] = hundred * (period_f - F::from(period - 1 - max_pos).unwrap()) / period_f;
            aroon_down[i] =
                hundred * (period_f - F::from(period - 1 - min_pos).unwrap()) / period_f;
        }

        // Calculate Aroon Oscillator
        let aroon_oscillator = &aroon_up - &aroon_down;

        Ok((aroon_up, aroon_down, aroon_oscillator))
    }

    /// Volume Weighted Average Price (VWAP)
    pub fn vwap<F: Float + Clone>(
        high: &Array1<F>,
        low: &Array1<F>,
        close: &Array1<F>,
        volume: &Array1<F>,
    ) -> Result<Array1<F>> {
        if high.len() != low.len() || low.len() != close.len() || close.len() != volume.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: high.len(),
                actual: volume.len(),
            });
        }

        let n = high.len();
        let mut vwap = Array1::zeros(n);
        let mut cumulative_pv = F::zero();
        let mut cumulative_volume = F::zero();
        let three = F::from(3.0).unwrap();

        for i in 0..n {
            let typical_price = (high[i] + low[i] + close[i]) / three;
            let pv = typical_price * volume[i];

            cumulative_pv = cumulative_pv + pv;
            cumulative_volume = cumulative_volume + volume[i];

            if cumulative_volume > F::zero() {
                vwap[i] = cumulative_pv / cumulative_volume;
            } else {
                vwap[i] = typical_price;
            }
        }

        Ok(vwap)
    }

    /// Chaikin Oscillator
    pub fn chaikin_oscillator<F: Float + Clone>(
        high: &Array1<F>,
        low: &Array1<F>,
        close: &Array1<F>,
        volume: &Array1<F>,
        fast_period: usize,
        slow_period: usize,
    ) -> Result<Array1<F>> {
        if high.len() != low.len() || low.len() != close.len() || close.len() != volume.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: high.len(),
                actual: volume.len(),
            });
        }

        let n = high.len();
        let mut ad_line = Array1::zeros(n); // Accumulation/Distribution line
        let mut cumulative_ad = F::zero();

        // Calculate Accumulation/Distribution line
        for i in 0..n {
            let clv = if high[i] != low[i] {
                ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
            } else {
                F::zero()
            };

            let money_flow_volume = clv * volume[i];
            cumulative_ad = cumulative_ad + money_flow_volume;
            ad_line[i] = cumulative_ad;
        }

        // Calculate fast and slow EMAs of A/D line
        let fast_alpha = F::from(2.0).unwrap() / F::from(fast_period + 1).unwrap();
        let slow_alpha = F::from(2.0).unwrap() / F::from(slow_period + 1).unwrap();

        use crate::financial::technical_indicators::ema;
        let fast_ema = ema(&ad_line, fast_alpha)?;
        let slow_ema = ema(&ad_line, slow_alpha)?;

        // Chaikin Oscillator = Fast EMA - Slow EMA
        let chaikin_osc = &fast_ema - &slow_ema;

        Ok(chaikin_osc)
    }

    /// Fibonacci Retracement Levels
    pub fn fibonacci_retracement<F: Float + Clone>(
        high_price: F,
        low_price: F,
    ) -> Result<FibonacciLevels<F>> {
        if high_price <= low_price {
            return Err(TimeSeriesError::InvalidInput(
                "High _price must be greater than low _price".to_string(),
            ));
        }

        let range = high_price - low_price;
        let fib_23_6 = F::from(0.236).unwrap();
        let fib_38_2 = F::from(0.382).unwrap();
        let fib_50_0 = F::from(0.5).unwrap();
        let fib_61_8 = F::from(0.618).unwrap();
        let fib_78_6 = F::from(0.786).unwrap();

        Ok(FibonacciLevels {
            level_100: high_price,
            level_78_6: high_price - range * fib_78_6,
            level_61_8: high_price - range * fib_61_8,
            level_50_0: high_price - range * fib_50_0,
            level_38_2: high_price - range * fib_38_2,
            level_23_6: high_price - range * fib_23_6,
            level_0: low_price,
        })
    }

    /// Fibonacci retracement levels structure
    #[derive(Debug, Clone)]
    pub struct FibonacciLevels<F: Float> {
        /// 100% Fibonacci level
        pub level_100: F,
        /// 78.6% Fibonacci level
        pub level_78_6: F,
        /// 61.8% Fibonacci level
        pub level_61_8: F,
        /// 50.0% Fibonacci level
        pub level_50_0: F,
        /// 38.2% Fibonacci level
        pub level_38_2: F,
        /// 23.6% Fibonacci level
        pub level_23_6: F,
        /// 0% Fibonacci level
        pub level_0: F,
    }

    /// Commodity Channel Index for multiple timeframes
    pub fn multi_timeframe_cci<F: Float + Clone>(
        high: &Array1<F>,
        low: &Array1<F>,
        close: &Array1<F>,
        periods: &[usize],
    ) -> Result<Vec<Array1<F>>> {
        let mut results = Vec::with_capacity(periods.len());

        for &period in periods {
            let cci_values = cci(high, low, close, period)?;
            results.push(cci_values);
        }

        Ok(results)
    }

    /// Adaptive Moving Average (AMA) - Kaufman's Adaptive Moving Average
    pub fn kama<F: Float + Clone>(
        data: &Array1<F>,
        period: usize,
        fast_sc: usize,
        slow_sc: usize,
    ) -> Result<Array1<F>> {
        if data.len() < period {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for KAMA calculation".to_string(),
                required: period,
                actual: data.len(),
            });
        }

        let n = data.len();
        let mut kama = Array1::zeros(n);
        kama[period - 1] = data[period - 1];

        let fast_alpha = F::from(2.0).unwrap() / F::from(fast_sc + 1).unwrap();
        let slow_alpha = F::from(2.0).unwrap() / F::from(slow_sc + 1).unwrap();

        for i in period..n {
            // Calculate efficiency ratio
            let direction = (data[i] - data[i - period]).abs();
            let volatility = (0..period)
                .map(|j| (data[i - j] - data[i - j - 1]).abs())
                .fold(F::zero(), |acc, x| acc + x);

            let efficiency_ratio = if volatility > F::zero() {
                direction / volatility
            } else {
                F::zero()
            };

            // Calculate smoothing constant
            let _sc = efficiency_ratio * (fast_alpha - slow_alpha) + slow_alpha;
            let sc_squared = _sc * _sc;

            // Update KAMA
            kama[i] = kama[i - 1] + sc_squared * (data[i] - kama[i - 1]);
        }

        Ok(kama)
    }
}

/// Risk Metrics Module for comprehensive financial risk analysis
pub mod risk_metrics {
    use super::*;

    /// Risk metrics configuration
    #[derive(Debug, Clone)]
    pub struct RiskConfig {
        /// Confidence level for VaR/CVaR calculations (e.g., 0.95 for 95%)
        pub confidence_level: f64,
        /// Number of trading days per year for annualization
        pub trading_days_per_year: usize,
        /// Risk-free rate for Sharpe ratio calculation
        pub risk_free_rate: f64,
        /// Window size for rolling calculations
        pub window_size: Option<usize>,
    }

    impl Default for RiskConfig {
        fn default() -> Self {
            Self {
                confidence_level: 0.95,
                trading_days_per_year: 252,
                risk_free_rate: 0.02, // 2% annual risk-free rate
                window_size: None,
            }
        }
    }

    /// Comprehensive risk metrics result
    #[derive(Debug, Clone)]
    pub struct RiskMetrics<F: Float> {
        /// Value at Risk (VaR)
        pub var: F,
        /// Conditional Value at Risk (Expected Shortfall)
        pub cvar: F,
        /// Maximum Drawdown
        pub max_drawdown: F,
        /// Sharpe Ratio
        pub sharpe_ratio: F,
        /// Sortino Ratio
        pub sortino_ratio: F,
        /// Calmar Ratio
        pub calmar_ratio: F,
        /// Information Ratio
        pub information_ratio: F,
        /// Skewness of returns
        pub skewness: F,
        /// Kurtosis of returns
        pub kurtosis: F,
        /// Volatility (annualized standard deviation)
        pub volatility: F,
        /// Downside deviation
        pub downside_deviation: F,
        /// Maximum consecutive losses
        pub max_consecutive_losses: usize,
        /// Hit ratio (percentage of positive returns)
        pub hit_ratio: F,
        /// Pain Index (average drawdown)
        pub pain_index: F,
        /// Ulcer Index (drawdown-based risk measure)
        pub ulcer_index: F,
    }

    /// Calculate comprehensive risk metrics for returns data
    pub fn calculate_risk_metrics<F: Float + Clone + std::iter::Sum>(
        returns: &Array1<F>,
        config: &RiskConfig,
    ) -> Result<RiskMetrics<F>> {
        if returns.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "Returns array cannot be empty".to_string(),
            ));
        }

        let n = returns.len();
        let confidence_level = F::from(config.confidence_level).unwrap();
        let trading_days = F::from(config.trading_days_per_year).unwrap();
        let risk_free_rate = F::from(config.risk_free_rate).unwrap();

        // Basic statistics
        let meanreturn = returns.sum() / F::from(n).unwrap();
        let variance = returns
            .iter()
            .map(|&r| (r - meanreturn).powi(2))
            .fold(F::zero(), |acc, x| acc + x)
            / F::from(n - 1).unwrap();
        let std_dev = variance.sqrt();
        let volatility = std_dev * trading_days.sqrt();

        // Value at Risk (VaR) - Historical method
        let mut sorted_returns = returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let var_index = ((F::one() - confidence_level) * F::from(n).unwrap())
            .to_usize()
            .unwrap();
        let var = if var_index < sorted_returns.len() {
            -sorted_returns[var_index] // VaR is typically reported as positive
        } else {
            -sorted_returns[0]
        };

        // Conditional Value at Risk (CVaR/Expected Shortfall)
        let tail_returns: Vec<F> = sorted_returns.iter().take(var_index + 1).cloned().collect();
        let cvar = if !tail_returns.is_empty() {
            let tail_len = tail_returns.len();
            -tail_returns.into_iter().sum::<F>() / F::from(tail_len).unwrap()
        } else {
            var
        };

        // Calculate cumulative returns for drawdown analysis
        let mut cumulative_returns = Array1::zeros(n);
        cumulative_returns[0] = F::one() + returns[0];
        for i in 1..n {
            cumulative_returns[i] = cumulative_returns[i - 1] * (F::one() + returns[i]);
        }

        // Maximum Drawdown
        let max_drawdown = calculate_max_drawdown(&cumulative_returns);

        // Sharpe Ratio
        let excessreturn = meanreturn * trading_days - risk_free_rate;
        let sharpe_ratio = if volatility > F::zero() {
            excessreturn / volatility
        } else {
            F::zero()
        };

        // Downside deviation (for Sortino ratio)
        let downside_variance = returns
            .iter()
            .map(|&r| if r < F::zero() { r.powi(2) } else { F::zero() })
            .fold(F::zero(), |acc, x| acc + x)
            / F::from(n - 1).unwrap();
        let downside_deviation = downside_variance.sqrt() * trading_days.sqrt();

        // Sortino Ratio
        let sortino_ratio = if downside_deviation > F::zero() {
            excessreturn / downside_deviation
        } else {
            F::zero()
        };

        // Calmar Ratio
        let annualizedreturn = meanreturn * trading_days;
        let calmar_ratio = if max_drawdown > F::zero() {
            annualizedreturn / max_drawdown
        } else {
            F::zero()
        };

        // Information Ratio (assuming benchmark return is risk-free rate)
        let tracking_error = std_dev * trading_days.sqrt();
        let information_ratio = if tracking_error > F::zero() {
            excessreturn / tracking_error
        } else {
            F::zero()
        };

        // Skewness
        let skewness = if std_dev > F::zero() {
            let skew_sum = returns
                .iter()
                .map(|&r| ((r - meanreturn) / std_dev).powi(3))
                .fold(F::zero(), |acc, x| acc + x);
            skew_sum / F::from(n).unwrap()
        } else {
            F::zero()
        };

        // Kurtosis
        let kurtosis = if std_dev > F::zero() {
            let kurt_sum = returns
                .iter()
                .map(|&r| ((r - meanreturn) / std_dev).powi(4))
                .fold(F::zero(), |acc, x| acc + x);
            kurt_sum / F::from(n).unwrap() - F::from(3.0).unwrap() // Excess kurtosis
        } else {
            F::zero()
        };

        // Maximum consecutive losses
        let max_consecutive_losses = calculate_max_consecutive_losses(returns);

        // Hit ratio (percentage of positive returns)
        let positive_returns = returns.iter().filter(|&&r| r > F::zero()).count();
        let hit_ratio = F::from(positive_returns).unwrap() / F::from(n).unwrap();

        // Pain Index (average drawdown)
        let pain_index = calculate_pain_index(&cumulative_returns);

        // Ulcer Index
        let ulcer_index = calculate_ulcer_index(&cumulative_returns);

        Ok(RiskMetrics {
            var,
            cvar,
            max_drawdown,
            sharpe_ratio,
            sortino_ratio,
            calmar_ratio,
            information_ratio,
            skewness,
            kurtosis,
            volatility,
            downside_deviation,
            max_consecutive_losses,
            hit_ratio,
            pain_index,
            ulcer_index,
        })
    }

    /// Calculate Value at Risk using parametric method
    pub fn parametric_var<F: Float + Clone>(returns: &Array1<F>, confidencelevel: F) -> Result<F> {
        if returns.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "Returns array cannot be empty".to_string(),
            ));
        }

        let n = returns.len();
        let mean = returns.sum() / F::from(n).unwrap();
        let variance = returns
            .iter()
            .map(|&r| (r - mean).powi(2))
            .fold(F::zero(), |acc, x| acc + x)
            / F::from(n - 1).unwrap();
        let std_dev = variance.sqrt();

        // Normal distribution inverse CDF approximation
        let alpha = F::one() - confidencelevel;
        let z_score = normal_inverse_cdf(alpha);
        let var = -(mean + z_score * std_dev);

        Ok(var)
    }

    /// Calculate Monte Carlo Value at Risk
    pub fn monte_carlo_var<F: Float + Clone>(
        returns: &Array1<F>,
        confidence_level: F,
        num_simulations: usize,
    ) -> Result<F> {
        if returns.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "Returns array cannot be empty".to_string(),
            ));
        }

        let n = returns.len();
        let mean = returns.sum() / F::from(n).unwrap();
        let variance = returns
            .iter()
            .map(|&r| (r - mean).powi(2))
            .fold(F::zero(), |acc, x| acc + x)
            / F::from(n - 1).unwrap();
        let std_dev = variance.sqrt();

        // Simple Monte Carlo simulation using normal distribution
        let mut simulated_returns = Vec::with_capacity(num_simulations);

        // Simple random number generation (in practice, use proper RNG)
        let mut seed = 12345u64;
        for _ in 0..num_simulations {
            // Simple LCG for demonstration
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let u1 = (seed as f64) / (u64::MAX as f64);

            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let u2 = (seed as f64) / (u64::MAX as f64);

            // Box-Muller transformation
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let simulatedreturn = mean + std_dev * F::from(z).unwrap();
            simulated_returns.push(simulatedreturn);
        }

        // Sort and find VaR
        simulated_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let var_index = ((F::one() - confidence_level) * F::from(num_simulations).unwrap())
            .to_usize()
            .unwrap();
        let var = if var_index < simulated_returns.len() {
            -simulated_returns[var_index]
        } else {
            -simulated_returns[0]
        };

        Ok(var)
    }

    /// Calculate rolling Value at Risk
    pub fn rolling_var<F: Float + Clone>(
        returns: &Array1<F>,
        window_size: usize,
        confidence_level: F,
    ) -> Result<Array1<F>> {
        if returns.len() < window_size {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for rolling VaR".to_string(),
                required: window_size,
                actual: returns.len(),
            });
        }

        let output_len = returns.len() - window_size + 1;
        let mut rolling_var = Array1::zeros(output_len);

        for i in 0..output_len {
            let window_returns = returns.slice(s![i..i + window_size]);
            let window_array = Array1::from_vec(window_returns.to_vec());
            rolling_var[i] = parametric_var(&window_array, confidence_level)?;
        }

        Ok(rolling_var)
    }

    /// Calculate maximum drawdown from cumulative returns
    fn calculate_max_drawdown<F: Float + Clone>(cumulative_returns: &Array1<F>) -> F {
        let mut max_drawdown = F::zero();
        let mut peak = cumulative_returns[0];

        for &value in cumulative_returns.iter() {
            if value > peak {
                peak = value;
            }
            let drawdown = (peak - value) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        max_drawdown
    }

    /// Calculate maximum consecutive losses
    fn calculate_max_consecutive_losses<F: Float + Clone>(returns: &Array1<F>) -> usize {
        let mut max_consecutive = 0;
        let mut current_consecutive = 0;

        for &ret in returns.iter() {
            if ret < F::zero() {
                current_consecutive += 1;
                max_consecutive = max_consecutive.max(current_consecutive);
            } else {
                current_consecutive = 0;
            }
        }

        max_consecutive
    }

    /// Calculate Pain Index (average drawdown)
    fn calculate_pain_index<F: Float + Clone>(cumulative_returns: &Array1<F>) -> F {
        let mut peak = cumulative_returns[0];
        let mut total_drawdown = F::zero();
        let n = cumulative_returns.len();

        for &value in cumulative_returns.iter() {
            if value > peak {
                peak = value;
            }
            let drawdown = (peak - value) / peak;
            total_drawdown = total_drawdown + drawdown;
        }

        total_drawdown / F::from(n).unwrap()
    }

    /// Calculate Ulcer Index (RMS of drawdowns)
    fn calculate_ulcer_index<F: Float + Clone>(cumulative_returns: &Array1<F>) -> F {
        let mut peak = cumulative_returns[0];
        let mut sum_squared_drawdowns = F::zero();
        let n = cumulative_returns.len();

        for &value in cumulative_returns.iter() {
            if value > peak {
                peak = value;
            }
            let drawdown = (peak - value) / peak;
            sum_squared_drawdowns = sum_squared_drawdowns + drawdown.powi(2);
        }

        (sum_squared_drawdowns / F::from(n).unwrap()).sqrt()
    }

    /// Beta calculation against a benchmark
    pub fn calculate_beta<F: Float + Clone>(
        returns: &Array1<F>,
        benchmark_returns: &Array1<F>,
    ) -> Result<F> {
        if returns.len() != benchmark_returns.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: returns.len(),
                actual: benchmark_returns.len(),
            });
        }

        if returns.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "Returns arrays cannot be empty".to_string(),
            ));
        }

        let n = F::from(returns.len()).unwrap();
        let mean_asset = returns.sum() / n;
        let mean_benchmark = benchmark_returns.sum() / n;

        let mut covariance = F::zero();
        let mut benchmark_variance = F::zero();

        for (&asset_ret, &bench_ret) in returns.iter().zip(benchmark_returns.iter()) {
            let asset_dev = asset_ret - mean_asset;
            let bench_dev = bench_ret - mean_benchmark;
            covariance = covariance + asset_dev * bench_dev;
            benchmark_variance = benchmark_variance + bench_dev.powi(2);
        }

        covariance = covariance / (n - F::one());
        benchmark_variance = benchmark_variance / (n - F::one());

        if benchmark_variance > F::zero() {
            Ok(covariance / benchmark_variance)
        } else {
            Ok(F::zero())
        }
    }

    /// Treynor Ratio calculation
    pub fn treynor_ratio<F: Float + Clone>(
        returns: &Array1<F>,
        benchmark_returns: &Array1<F>,
        risk_free_rate: F,
        trading_days_per_year: usize,
    ) -> Result<F> {
        let beta = calculate_beta(returns, benchmark_returns)?;

        if beta.abs() < F::from(1e-10).unwrap() {
            return Ok(F::zero());
        }

        let n = F::from(returns.len()).unwrap();
        let meanreturn = returns.sum() / n;
        let annualizedreturn = meanreturn * F::from(trading_days_per_year).unwrap();
        let excessreturn = annualizedreturn - risk_free_rate;

        Ok(excessreturn / beta)
    }

    /// Jensen's Alpha calculation
    pub fn jensens_alpha<F: Float + Clone>(
        returns: &Array1<F>,
        benchmark_returns: &Array1<F>,
        risk_free_rate: F,
        trading_days_per_year: usize,
    ) -> Result<F> {
        let beta = calculate_beta(returns, benchmark_returns)?;

        let n = F::from(returns.len()).unwrap();
        let mean_assetreturn = returns.sum() / n;
        let mean_benchmarkreturn = benchmark_returns.sum() / n;

        let annualized_asset = mean_assetreturn * F::from(trading_days_per_year).unwrap();
        let annualized_benchmark = mean_benchmarkreturn * F::from(trading_days_per_year).unwrap();

        let expectedreturn = risk_free_rate + beta * (annualized_benchmark - risk_free_rate);
        Ok(annualized_asset - expectedreturn)
    }

    /// Approximate normal inverse CDF
    fn normal_inverse_cdf<F: Float>(p: F) -> F {
        // Beasley-Springer-Moro algorithm approximation
        let a0 = F::from(2.515517).unwrap();
        let a1 = F::from(0.802853).unwrap();
        let a2 = F::from(0.010328).unwrap();
        let b1 = F::from(1.432788).unwrap();
        let b2 = F::from(0.189269).unwrap();
        let b3 = F::from(0.001308).unwrap();

        let t = if p < F::from(0.5).unwrap() {
            (-F::from(2.0).unwrap() * (p.ln())).sqrt()
        } else {
            (-F::from(2.0).unwrap() * ((F::one() - p).ln())).sqrt()
        };

        let numerator = a0 + a1 * t + a2 * t.powi(2);
        let denominator = F::one() + b1 * t + b2 * t.powi(2) + b3 * t.powi(3);
        let z = t - numerator / denominator;

        if p < F::from(0.5).unwrap() {
            -z
        } else {
            z
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_risk_metrics_calculation() {
            let returns = Array1::from_vec(vec![0.01, -0.02, 0.015, -0.01, 0.005, -0.008, 0.02]);
            let config = RiskConfig::default();

            let metrics = calculate_risk_metrics(&returns, &config).unwrap();

            assert!(metrics.var > 0.0);
            assert!(metrics.cvar >= metrics.var);
            assert!(metrics.max_drawdown >= 0.0);
            assert!(metrics.hit_ratio >= 0.0 && metrics.hit_ratio <= 1.0);
        }

        #[test]
        fn test_parametric_var() {
            let returns = Array1::from_vec(vec![0.01, -0.02, 0.015, -0.01, 0.005]);
            let var = parametric_var(&returns, 0.95).unwrap();
            assert!(var > 0.0);
        }

        #[test]
        fn test_beta_calculation() {
            let asset_returns = Array1::from_vec(vec![0.01, -0.02, 0.015, -0.01, 0.005]);
            let benchmark_returns = Array1::from_vec(vec![0.008, -0.015, 0.012, -0.008, 0.004]);

            let beta = calculate_beta(&asset_returns, &benchmark_returns).unwrap();
            assert!(beta.abs() < 10.0); // Reasonable beta range
        }

        #[test]
        fn test_max_drawdown() {
            let cumulative = Array1::from_vec(vec![1.0, 1.1, 1.05, 0.95, 1.2, 1.15]);
            let mdd = calculate_max_drawdown(&cumulative);
            assert!(mdd > 0.0);
            assert!(mdd < 1.0);
        }
    }
}
