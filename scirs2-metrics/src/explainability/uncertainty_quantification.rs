//! Uncertainty quantification methods for model predictions

use crate::error::{MetricsError, Result};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_traits::Float;
use statrs::statistics::Statistics;
use std::collections::HashMap;
use std::f64::consts::PI;

/// Uncertainty quantification analyzer
pub struct UncertaintyQuantifier<F: Float> {
    /// Number of Monte Carlo samples
    pub n_mc_samples: usize,
    /// Confidence level for intervals
    pub confidence_level: F,
    /// Bootstrap samples for confidence estimation
    pub n_bootstrap: usize,
    /// Random seed
    pub random_seed: Option<u64>,
    /// Random number generator type
    pub rng_type: RandomNumberGenerator,
    /// Number of conformal calibration samples
    pub n_conformal_calibration: usize,
    /// Enable Bayesian uncertainty estimation
    pub enable_bayesian: bool,
    /// Number of MCMC samples
    pub n_mcmc_samples: usize,
    /// MCMC burn-in samples
    pub mcmc_burn_in: usize,
    /// Enable temperature scaling
    pub enable_temperature_scaling: bool,
    /// Enable SIMD acceleration
    pub enable_simd: bool,
}

impl<F: Float + num_traits::FromPrimitive + std::iter::Sum + ndarray::ScalarOperand> Default
    for UncertaintyQuantifier<F>
{
    fn default() -> Self {
        Self::new()
    }
}

/// Random number generator types
#[derive(Debug, Clone)]
pub enum RandomNumberGenerator {
    /// Linear Congruential Generator (fast, basic quality)
    Lcg,
    /// Xorshift (good balance of speed and quality)
    Xorshift,
    /// Permuted Congruential Generator (high quality)
    Pcg,
    /// ChaCha (cryptographically secure)
    ChaCha,
}

/// Trait for random number generators
pub trait RandomNumberGeneratorTrait {
    fn uniform_01<F: Float + num_traits::FromPrimitive>(&mut self) -> F;
    fn normal<F: Float + num_traits::FromPrimitive>(&mut self) -> F;
    fn seed(&mut self, seed: u64);
}

/// Linear Congruential Generator implementation
pub struct LcgRng {
    state: u64,
}

impl LcgRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }
}

impl RandomNumberGeneratorTrait for LcgRng {
    fn uniform_01<F: Float + num_traits::FromPrimitive>(&mut self) -> F {
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        F::from((self.state >> 16) as f64 / (1u64 << 32) as f64).unwrap()
    }

    fn normal<F: Float + num_traits::FromPrimitive>(&mut self) -> F {
        // Box-Muller transform
        let u1 = self.uniform_01::<F>();
        let u2 = self.uniform_01::<F>();

        (-F::from(2.0).unwrap() * u1.ln()).sqrt() * (F::from(2.0 * PI).unwrap() * u2).cos()
    }

    fn seed(&mut self, seed: u64) {
        self.state = seed;
    }
}

/// Xorshift random number generator
pub struct XorshiftRng {
    state: u64,
}

impl XorshiftRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed.max(1) } // Ensure non-zero state
    }
}

impl RandomNumberGeneratorTrait for XorshiftRng {
    fn uniform_01<F: Float + num_traits::FromPrimitive>(&mut self) -> F {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        F::from(self.state as f64 / u64::MAX as f64).unwrap()
    }

    fn normal<F: Float + num_traits::FromPrimitive>(&mut self) -> F {
        let u1 = self.uniform_01::<F>();
        let u2 = self.uniform_01::<F>();

        (-F::from(2.0).unwrap() * u1.ln()).sqrt() * (F::from(2.0 * PI).unwrap() * u2).cos()
    }

    fn seed(&mut self, seed: u64) {
        self.state = seed.max(1);
    }
}

/// PCG random number generator
pub struct PcgRng {
    state: u64,
    inc: u64,
}

impl PcgRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed,
            inc: 721347520444481703u64,
        }
    }
}

impl RandomNumberGeneratorTrait for PcgRng {
    fn uniform_01<F: Float + num_traits::FromPrimitive>(&mut self) -> F {
        let oldstate = self.state;
        self.state = oldstate
            .wrapping_mul(6364136223846793005u64)
            .wrapping_add(self.inc);
        let xorshifted = ((oldstate >> 18) ^ oldstate) >> 27;
        let rot = oldstate >> 59;
        let result = (xorshifted >> rot) | (xorshifted << ((32u32.wrapping_sub(rot as u32)) & 31));
        F::from(result as f64 / u32::MAX as f64).unwrap()
    }

    fn normal<F: Float + num_traits::FromPrimitive>(&mut self) -> F {
        let u1 = self.uniform_01::<F>();
        let u2 = self.uniform_01::<F>();

        (-F::from(2.0).unwrap() * u1.ln()).sqrt() * (F::from(2.0 * PI).unwrap() * u2).cos()
    }

    fn seed(&mut self, seed: u64) {
        self.state = seed;
    }
}

/// ChaCha random number generator (simplified)
pub struct ChaChaRng {
    state: [u32; 16],
    counter: u64,
}

impl ChaChaRng {
    pub fn new(seed: u64) -> Self {
        let mut state = [0u32; 16];
        state[0] = seed as u32;
        state[1] = (seed >> 32) as u32;
        Self { state, counter: 0 }
    }
}

impl RandomNumberGeneratorTrait for ChaChaRng {
    fn uniform_01<F: Float + num_traits::FromPrimitive>(&mut self) -> F {
        // Simplified ChaCha implementation
        self.counter = self.counter.wrapping_add(1);
        let mut x = self.state[0].wrapping_add(self.counter as u32);
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state[0] = x;

        F::from(x as f64 / u32::MAX as f64).unwrap()
    }

    fn normal<F: Float + num_traits::FromPrimitive>(&mut self) -> F {
        let u1 = self.uniform_01::<F>();
        let u2 = self.uniform_01::<F>();

        (-F::from(2.0).unwrap() * u1.ln()).sqrt() * (F::from(2.0 * PI).unwrap() * u2).cos()
    }

    fn seed(&mut self, seed: u64) {
        self.state[0] = seed as u32;
        self.state[1] = (seed >> 32) as u32;
        self.counter = 0;
    }
}

impl<F: Float + num_traits::FromPrimitive + std::iter::Sum + ndarray::ScalarOperand>
    UncertaintyQuantifier<F>
{
    /// Create new uncertainty quantifier
    pub fn new() -> Self {
        Self {
            n_mc_samples: 1000,
            confidence_level: F::from(0.95).unwrap(),
            n_bootstrap: 100,
            random_seed: None,
            rng_type: RandomNumberGenerator::Lcg,
            n_conformal_calibration: 100,
            enable_bayesian: false,
            n_mcmc_samples: 1000,
            mcmc_burn_in: 200,
            enable_temperature_scaling: false,
            enable_simd: false,
        }
    }

    /// Set number of Monte Carlo samples
    pub fn with_mc_samples(mut self, n: usize) -> Self {
        self.n_mc_samples = n;
        self
    }

    /// Set confidence level
    pub fn with_confidence_level(mut self, level: F) -> Self {
        self.confidence_level = level;
        self
    }

    /// Set number of bootstrap samples
    pub fn with_bootstrap(mut self, n: usize) -> Self {
        self.n_bootstrap = n;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Set random number generator type
    pub fn with_rng_type(mut self, rngtype: RandomNumberGenerator) -> Self {
        self.rng_type = rngtype;
        self
    }

    /// Set conformal calibration parameters
    pub fn with_conformal_calibration(mut self, nsamples: usize) -> Self {
        self.n_conformal_calibration = nsamples;
        self
    }

    /// Enable/disable Bayesian uncertainty estimation
    pub fn with_bayesian(mut self, enable: bool) -> Self {
        self.enable_bayesian = enable;
        self
    }

    /// Set MCMC parameters
    pub fn with_mcmc(mut self, nsamples: usize, burnin: usize) -> Self {
        self.n_mcmc_samples = nsamples;
        self.mcmc_burn_in = burnin;
        self
    }

    /// Enable/disable temperature scaling
    pub fn with_temperature_scaling(mut self, enable: bool) -> Self {
        self.enable_temperature_scaling = enable;
        self
    }

    /// Enable/disable SIMD acceleration
    pub fn with_simd(mut self, enable: bool) -> Self {
        self.enable_simd = enable;
        self
    }

    /// Compute conformal prediction intervals with adaptive methods
    pub fn compute_conformal_prediction<M>(
        &self,
        model: &M,
        x_calibration: &Array2<F>,
        y_calibration: &Array1<F>,
        xtest: &Array2<F>,
        alpha: F,
    ) -> Result<ConformalPrediction<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        // Compute nonconformity scores on _calibration set
        let cal_predictions = model(&x_calibration.view());
        let nonconformity_scores =
            self.compute_nonconformity_scores(&cal_predictions, y_calibration)?;

        // Adaptive conformal prediction with weighted quantiles
        let adaptive_quantile =
            self.compute_adaptive_quantile(&nonconformity_scores, alpha, x_calibration)?;

        // Make predictions on _test set
        let test_predictions = model(&xtest.view());

        // Compute prediction sets with local adaptation
        let mut prediction_sets = Vec::new();
        for i in 0..test_predictions.len() {
            let pred = test_predictions[i];

            // Local conformity adjustment based on input similarity
            let local_adjustment = self.compute_local_conformity_adjustment(
                &xtest.row(i),
                x_calibration,
                &nonconformity_scores,
            )?;

            let adjusted_quantile = adaptive_quantile * local_adjustment;
            let lower = pred - adjusted_quantile;
            let upper = pred + adjusted_quantile;
            let size = upper - lower;

            prediction_sets.push(PredictionSet {
                lower,
                upper,
                size,
                contains_truth: None, // Unknown for _test set
                local_difficulty: local_adjustment,
                adaptive_quantile: adjusted_quantile,
            });
        }

        // Enhanced coverage analysis
        let coverage_analysis = self.compute_enhanced_coverage_analysis(
            &cal_predictions,
            y_calibration,
            &nonconformity_scores,
            adaptive_quantile,
        )?;

        // Compute conditional coverage by difficulty
        let conditional_coverage = self.compute_conditional_coverage_analysis(
            &prediction_sets,
            xtest,
            &cal_predictions,
            y_calibration,
        )?;

        Ok(ConformalPrediction {
            prediction_sets,
            coverage_probability: coverage_analysis.overall_coverage,
            average_set_size: coverage_analysis.average_set_size,
            conditional_coverage,
            coverage_analysis: Some(coverage_analysis),
        })
    }

    /// Advanced Bayesian neural network uncertainty with variational inference
    pub fn compute_variational_uncertainty<M, P>(
        &self,
        model: &M,
        variational_parameters: &P,
        xdata: &Array2<F>,
        y_data: &Array1<F>,
        xtest: &Array2<F>,
    ) -> Result<VariationalUncertainty<F>>
    where
        M: Fn(&ArrayView2<F>, &VariationalParams<F>) -> (Array1<F>, Array1<F>), // Returns (mean, variance)
        P: Fn() -> VariationalParams<F>, // Variational parameter sampler
    {
        let mut posterior_samples = Vec::new();
        let mut kl_divergences = Vec::new();
        let mut elbo_values = Vec::new();

        // Variational inference with multiple samples
        for _ in 0..self.n_mcmc_samples {
            let varparams = variational_parameters();

            // Forward pass with variational _parameters
            let (mean_pred, var_pred) = model(&xtest.view(), &varparams);

            // Compute KL divergence between posterior and prior
            let kl_div = self.compute_kl_divergence_gaussian(&varparams)?;

            // Compute ELBO (Evidence Lower BOund)
            let log_likelihood = self.compute_variational_log_likelihood(
                &mean_pred,
                &var_pred,
                &xdata.view(),
                y_data,
            )?;
            let elbo = log_likelihood - kl_div;

            posterior_samples.push((mean_pred, var_pred));
            kl_divergences.push(kl_div);
            elbo_values.push(elbo);
        }

        // Aggregate results
        let nsamples = xtest.nrows();
        let mut ensemble_mean = Array1::zeros(nsamples);
        let mut epistemic_uncertainty = Array1::zeros(nsamples);
        let mut aleatoric_uncertainty = Array1::zeros(nsamples);

        // Compute ensemble statistics
        for i in 0..nsamples {
            let sample_means: Vec<F> = posterior_samples.iter().map(|(mean, _)| mean[i]).collect();
            let sample_vars: Vec<F> = posterior_samples.iter().map(|(_, var)| var[i]).collect();

            // Ensemble mean
            ensemble_mean[i] =
                sample_means.iter().cloned().sum::<F>() / F::from(sample_means.len()).unwrap();

            // Epistemic uncertainty (variance of means)
            let mean_of_means = ensemble_mean[i];
            epistemic_uncertainty[i] = sample_means
                .iter()
                .map(|&m| (m - mean_of_means) * (m - mean_of_means))
                .sum::<F>()
                / F::from(sample_means.len()).unwrap();

            // Aleatoric uncertainty (mean of variances)
            aleatoric_uncertainty[i] =
                sample_vars.iter().cloned().sum::<F>() / F::from(sample_vars.len()).unwrap();
        }

        // Compute convergence diagnostics
        let mean_elbo =
            elbo_values.iter().cloned().sum::<F>() / F::from(elbo_values.len()).unwrap();
        let elbo_variance = elbo_values
            .iter()
            .map(|&x| (x - mean_elbo) * (x - mean_elbo))
            .sum::<F>()
            / F::from(elbo_values.len()).unwrap();

        let convergence_diagnostic = if elbo_variance < F::from(0.01).unwrap() {
            "Converged".to_string()
        } else {
            "Not converged".to_string()
        };

        Ok(VariationalUncertainty {
            ensemble_mean,
            epistemic_uncertainty,
            aleatoric_uncertainty,
            kl_divergences,
            elbo_values,
            mean_elbo,
            elbo_variance,
            convergence_diagnostic,
        })
    }

    /// Multi-scale uncertainty quantification
    pub fn compute_multiscale_uncertainty<M>(
        &self,
        models: &[M], // Models at different scales/resolutions
        xtest: &Array2<F>,
        scales: &[F],
    ) -> Result<MultiscaleUncertainty<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        if models.len() != scales.len() {
            return Err(MetricsError::InvalidInput(
                "Number of models must match number of scales".to_string(),
            ));
        }

        let mut scale_predictions = Vec::new();
        let mut scale_uncertainties = Vec::new();

        // Compute predictions at each scale
        for (i, model) in models.iter().enumerate() {
            let predictions = model(&xtest.view());

            // Estimate uncertainty at this scale using local variation
            let scale_uncertainty = self.compute_scale_uncertainty(&predictions, scales[i])?;

            scale_predictions.push(predictions);
            scale_uncertainties.push(scale_uncertainty);
        }

        // Multi-scale fusion using weighted averaging
        let weights = self.compute_scale_weights(scales, &scale_uncertainties)?;
        let fused_predictions = self.fuse_multiscale_predictions(&scale_predictions, &weights)?;

        // Cross-scale consistency analysis
        let consistency_scores = self.compute_cross_scale_consistency(&scale_predictions)?;

        // Hierarchical uncertainty decomposition
        let uncertainty_decomposition = self.decompose_multiscale_uncertainty(
            &scale_predictions,
            &scale_uncertainties,
            &weights,
        )?;

        Ok(MultiscaleUncertainty {
            scale_predictions,
            scale_uncertainties,
            fused_predictions,
            consistency_scores,
            uncertainty_decomposition,
            scales: scales.to_vec(),
            weights,
        })
    }

    /// Compute Bayesian uncertainty using MCMC
    pub fn compute_bayesian_uncertainty_mcmc<M, P>(
        &self,
        model: &M,
        prior_sampler: &P,
        xdata: &Array2<F>,
        y_data: &Array1<F>,
        xtest: &Array2<F>,
    ) -> Result<BayesianUncertainty<F>>
    where
        M: Fn(&ArrayView2<F>, &[F]) -> Array1<F>, // Model takes parameters
        P: Fn() -> Vec<F>,                        // Prior sampler
    {
        let mut samples = Vec::new();
        let mut current_params = prior_sampler();
        let mut current_loglik =
            self.compute_log_likelihood(model, &current_params, xdata, y_data)?;

        let mut accepted = 0;
        let step_size = F::from(0.01).unwrap();

        // MCMC sampling using Metropolis-Hastings
        for i in 0..self.n_mcmc_samples + self.mcmc_burn_in {
            // Propose new parameters
            let mut proposed_params = current_params.clone();
            for param in &mut proposed_params {
                let noise = self.sample_gaussian()? * step_size;
                *param = *param + noise;
            }

            // Compute likelihood of proposed parameters
            let proposed_loglik =
                self.compute_log_likelihood(model, &proposed_params, xdata, y_data)?;

            // Acceptance probability
            let log_alpha = proposed_loglik - current_loglik;
            let alpha = log_alpha.exp().min(F::one());

            // Accept or reject
            if self.uniform_01()? < alpha {
                current_params = proposed_params;
                current_loglik = proposed_loglik;
                accepted += 1;
            }

            // Store sample after burn-in
            if i >= self.mcmc_burn_in {
                samples.push(current_params.clone());
            }
        }

        // Compute posterior predictions
        let mut posterior_predictions = Array2::zeros((samples.len(), xtest.nrows()));
        for (i, params) in samples.iter().enumerate() {
            let predictions = model(&xtest.view(), params);
            for j in 0..xtest.nrows() {
                posterior_predictions[[i, j]] = predictions[j];
            }
        }

        // Compute posterior statistics
        let posterior_mean = posterior_predictions.mean_axis(Axis(0)).unwrap();
        let posterior_variance = posterior_predictions.var_axis(Axis(0), F::zero());

        // Compute credible intervals
        let mut credible_intervals = Array2::zeros((xtest.nrows(), 2));
        let alpha = F::from(0.05).unwrap(); // 95% credible interval

        for i in 0..xtest.nrows() {
            let mut column_samples: Vec<F> = posterior_predictions.column(i).to_vec();
            column_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let lower_idx = (alpha / F::from(2.0).unwrap()
                * F::from(column_samples.len()).unwrap())
            .to_usize()
            .unwrap_or(0);
            let upper_idx = ((F::one() - alpha / F::from(2.0).unwrap())
                * F::from(column_samples.len()).unwrap())
            .to_usize()
            .unwrap_or(column_samples.len() - 1);

            credible_intervals[[i, 0]] = column_samples[lower_idx];
            credible_intervals[[i, 1]] = column_samples[upper_idx];
        }

        // Compute model evidence (simplified)
        let model_evidence = current_loglik; // Simplified - should use more sophisticated methods

        // Compute effective sample size
        let effective_sample_size =
            F::from(accepted).unwrap() / F::from(self.n_mcmc_samples).unwrap();

        // Compute R-hat convergence diagnostic (simplified)
        let r_hat = self.compute_r_hat(&samples)?;

        Ok(BayesianUncertainty {
            posterior_mean,
            posterior_variance,
            credible_intervals,
            model_evidence,
            effective_sample_size,
            r_hat,
        })
    }

    /// Compute comprehensive uncertainty metrics
    pub fn compute_uncertainty<M>(
        &self,
        model: &M,
        xtest: &Array2<F>,
        y_test: Option<&Array1<F>>,
    ) -> Result<UncertaintyAnalysis<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        // Epistemic uncertainty via Monte Carlo dropout
        let epistemic_uncertainty = self.compute_epistemic_uncertainty(model, xtest)?;

        // Aleatoric uncertainty estimation
        let aleatoric_uncertainty = self.compute_aleatoric_uncertainty(model, xtest)?;

        // Prediction intervals
        let prediction_intervals = self.compute_prediction_intervals(model, xtest)?;

        // Calibration metrics (if labels available)
        let calibration_metrics = if let Some(y_true) = y_test {
            Some(self.compute_calibration_metrics(model, xtest, y_true)?)
        } else {
            None
        };

        // Confidence estimation
        let confidence_scores = self.compute_confidence_scores(model, xtest)?;

        // Out-of-distribution detection
        let ood_scores = self.compute_ood_scores(model, xtest)?;

        Ok(UncertaintyAnalysis {
            epistemic_uncertainty,
            aleatoric_uncertainty,
            prediction_intervals,
            calibration_metrics,
            confidence_scores,
            ood_scores,
            sample_size: xtest.nrows(),
        })
    }

    /// Compute epistemic uncertainty using Monte Carlo methods
    fn compute_epistemic_uncertainty<M>(
        &self,
        model: &M,
        xtest: &Array2<F>,
    ) -> Result<EpistemicUncertainty<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let mut mcpredictions = Vec::new();

        // Simulate model uncertainty with Monte Carlo sampling
        for _ in 0..self.n_mc_samples {
            // In practice, this would involve dropout or other stochastic elements
            let predictions = model(&xtest.view());
            mcpredictions.push(predictions);
        }

        // Compute statistics across MC samples
        let mean_predictions = self.compute_mc_mean(&mcpredictions)?;
        let prediction_variance = self.compute_mc_variance(&mcpredictions, &mean_predictions)?;
        let prediction_entropy = self.compute_prediction_entropy(&mcpredictions)?;
        let mutual_information = self.compute_mutual_information(&mcpredictions)?;

        Ok(EpistemicUncertainty {
            mean_predictions,
            prediction_variance,
            prediction_entropy,
            mutual_information,
            mc_samples: self.n_mc_samples,
        })
    }

    /// Compute aleatoric uncertainty (data-dependent)
    fn compute_aleatoric_uncertainty<M>(
        &self,
        model: &M,
        xtest: &Array2<F>,
    ) -> Result<AleatoricUncertainty<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        // Estimate aleatoric uncertainty using input perturbations
        let mut perturbed_predictions = Vec::new();
        let noisestd = F::from(0.01).unwrap(); // Small noise for input perturbation

        for _ in 0..50 {
            let mut x_perturbed = xtest.clone();
            self.add_input_noise(&mut x_perturbed, noisestd)?;

            let predictions = model(&x_perturbed.view());
            perturbed_predictions.push(predictions);
        }

        let baseline_predictions = model(&xtest.view());
        let input_sensitivity =
            self.compute_input_sensitivity(&perturbed_predictions, &baseline_predictions)?;
        let data_uncertainty = self.estimate_data_uncertainty(xtest)?;

        Ok(AleatoricUncertainty {
            input_sensitivity,
            data_uncertainty,
            noise_level: noisestd,
        })
    }

    /// Compute prediction intervals
    fn compute_prediction_intervals<M>(
        &self,
        model: &M,
        xtest: &Array2<F>,
    ) -> Result<PredictionIntervals<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        // Bootstrap-based prediction intervals
        let mut bootstrap_predictions = Vec::new();

        for _ in 0..self.n_bootstrap {
            // Bootstrap sample (with replacement)
            let bootstrap_indices = self.generate_bootstrap_indices(xtest.nrows())?;
            let x_bootstrap = self.sample_by_indices(xtest, &bootstrap_indices)?;

            let predictions = model(&x_bootstrap.view());
            bootstrap_predictions.push(predictions);
        }

        let lower_bound = self.compute_percentile(
            &bootstrap_predictions,
            (F::one() - self.confidence_level) / F::from(2).unwrap(),
        )?;
        let upper_bound = self.compute_percentile(
            &bootstrap_predictions,
            F::one() - (F::one() - self.confidence_level) / F::from(2).unwrap(),
        )?;
        let median_prediction =
            self.compute_percentile(&bootstrap_predictions, F::from(0.5).unwrap())?;

        Ok(PredictionIntervals {
            lower_bound,
            upper_bound,
            median_prediction,
            confidence_level: self.confidence_level,
        })
    }

    /// Compute calibration metrics
    fn compute_calibration_metrics<M>(
        &self,
        model: &M,
        xtest: &Array2<F>,
        y_test: &Array1<F>,
    ) -> Result<CalibrationMetrics<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let predictions = model(&xtest.view());

        // Expected Calibration Error (ECE)
        let ece = self.compute_expected_calibration_error(&predictions, y_test)?;

        // Maximum Calibration Error
        let mce = self.compute_maximum_calibration_error(&predictions, y_test)?;

        // Reliability diagram data
        let reliability_data = self.compute_reliability_diagram(&predictions, y_test)?;

        // Brier score decomposition
        let brier_decomposition = self.compute_brier_decomposition(&predictions, y_test)?;

        Ok(CalibrationMetrics {
            expected_calibration_error: ece,
            maximum_calibration_error: mce,
            reliability_data,
            brier_decomposition,
        })
    }

    /// Compute confidence scores for predictions
    fn compute_confidence_scores<M>(
        &self,
        model: &M,
        xtest: &Array2<F>,
    ) -> Result<ConfidenceScores<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let predictions = model(&xtest.view());

        // Max probability as confidence (for classification-like problems)
        let max_confidence = predictions.iter().cloned().fold(F::neg_infinity(), F::max);

        // Entropy-based uncertainty
        let entropy_uncertainty = self.compute_entropy_uncertainty(&predictions)?;

        // Distance-based confidence
        let distance_confidence = self.compute_distance_based_confidence(xtest)?;

        // Ensemble-based confidence (simplified)
        let ensemble_confidence = self.compute_ensemble_confidence(model, xtest)?;

        Ok(ConfidenceScores {
            max_confidence,
            entropy_uncertainty,
            distance_confidence,
            ensemble_confidence,
        })
    }

    /// Compute out-of-distribution detection scores
    fn compute_ood_scores<M>(&self, model: &M, xtest: &Array2<F>) -> Result<OODScores<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        // Mahalanobis distance-based OOD detection
        let mahalanobis_scores = self.compute_mahalanobis_scores(xtest)?;

        // Energy-based OOD detection
        let energy_scores = self.compute_energy_scores(model, xtest)?;

        // Reconstruction error (if applicable)
        let reconstruction_errors = self.compute_reconstruction_errors(xtest)?;

        // Density-based scores
        let density_scores = self.compute_density_scores(xtest)?;

        Ok(OODScores {
            mahalanobis_scores,
            energy_scores,
            reconstruction_errors,
            density_scores,
        })
    }

    // Helper methods

    fn compute_mc_mean(&self, mcpredictions: &[Array1<F>]) -> Result<Array1<F>> {
        if mcpredictions.is_empty() {
            return Err(MetricsError::InvalidInput(
                "No MC _predictions provided".to_string(),
            ));
        }

        let nsamples = mcpredictions[0].len();
        let mut mean_pred = Array1::zeros(nsamples);

        for predictions in mcpredictions {
            mean_pred = mean_pred + predictions;
        }

        mean_pred = mean_pred / F::from(mcpredictions.len()).unwrap();
        Ok(mean_pred)
    }

    fn compute_mc_variance(
        &self,
        mcpredictions: &[Array1<F>],
        mean_pred: &Array1<F>,
    ) -> Result<Array1<F>> {
        let nsamples = mean_pred.len();
        let mut variance = Array1::zeros(nsamples);

        for _predictions in mcpredictions {
            let diff = _predictions - mean_pred;
            variance = variance + &(&diff * &diff);
        }

        variance = variance / F::from(mcpredictions.len()).unwrap();
        Ok(variance)
    }

    fn compute_prediction_entropy(&self, mcpredictions: &[Array1<F>]) -> Result<Array1<F>> {
        let mean_pred = self.compute_mc_mean(mcpredictions)?;
        let mut entropy = Array1::zeros(mean_pred.len());

        for (i, &mean_val) in mean_pred.iter().enumerate() {
            // Simplified entropy calculation
            if mean_val > F::zero() && mean_val < F::one() {
                entropy[i] =
                    -mean_val * mean_val.ln() - (F::one() - mean_val) * (F::one() - mean_val).ln();
            }
        }

        Ok(entropy)
    }

    fn compute_mutual_information(&self, mcpredictions: &[Array1<F>]) -> Result<F> {
        // Simplified mutual information calculation
        let mean_pred = self.compute_mc_mean(mcpredictions)?;
        let variance = self.compute_mc_variance(mcpredictions, &mean_pred)?;

        let avg_variance = variance.mean().unwrap_or(F::zero());
        let avg_entropy = mean_pred
            .iter()
            .map(|&p| {
                if p > F::zero() && p < F::one() {
                    -p * p.ln() - (F::one() - p) * (F::one() - p).ln()
                } else {
                    F::zero()
                }
            })
            .sum::<F>()
            / F::from(mean_pred.len()).unwrap();

        Ok(avg_entropy - avg_variance)
    }

    fn add_input_noise(&self, xdata: &mut Array2<F>, noisestd: F) -> Result<()> {
        for value in xdata.iter_mut() {
            let noise = self.generate_gaussian_noise()? * noisestd;
            *value = *value + noise;
        }
        Ok(())
    }

    fn generate_gaussian_noise(&self) -> Result<F> {
        // Simplified Gaussian noise generation
        let seed = self.random_seed.unwrap_or(42);
        let u1 = F::from((seed % 1000) as f64 / 1000.0).unwrap();
        let u2 = F::from(((seed / 1000) % 1000) as f64 / 1000.0).unwrap();

        let z = (-F::from(2.0).unwrap() * u1.ln()).sqrt()
            * (F::from(2.0).unwrap() * F::from(std::f64::consts::PI).unwrap() * u2).cos();

        Ok(z)
    }

    fn compute_input_sensitivity(
        &self,
        perturbed_preds: &[Array1<F>],
        baseline_pred: &Array1<F>,
    ) -> Result<Array1<F>> {
        let mut sensitivity = Array1::zeros(baseline_pred.len());

        for _pred in perturbed_preds {
            let diff = _pred - baseline_pred;
            sensitivity = sensitivity + &diff.mapv(|x| x.abs());
        }

        sensitivity = sensitivity / F::from(perturbed_preds.len()).unwrap();
        Ok(sensitivity)
    }

    fn estimate_data_uncertainty(&self, xtest: &Array2<F>) -> Result<Array1<F>> {
        // Simplified data uncertainty based on local density
        let mut uncertainty = Array1::zeros(xtest.nrows());

        for i in 0..xtest.nrows() {
            let mut min_distance = F::infinity();

            for j in 0..xtest.nrows() {
                if i != j {
                    let distance = self.compute_euclidean_distance(&xtest.row(i), &xtest.row(j))?;
                    min_distance = min_distance.min(distance);
                }
            }

            uncertainty[i] = min_distance; // Higher distance = higher uncertainty
        }

        Ok(uncertainty)
    }

    fn compute_euclidean_distance(
        &self,
        x1: &ndarray::ArrayView1<F>,
        x2: &ndarray::ArrayView1<F>,
    ) -> Result<F> {
        let squared_diff: F = x1
            .iter()
            .zip(x2.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum();
        Ok(squared_diff.sqrt())
    }

    fn generate_bootstrap_indices(&self, nsamples: usize) -> Result<Vec<usize>> {
        let mut indices = Vec::with_capacity(nsamples);
        for i in 0..nsamples {
            let idx = (self.random_seed.unwrap_or(0) as usize + i) % nsamples;
            indices.push(idx);
        }
        Ok(indices)
    }

    fn sample_by_indices(&self, data: &Array2<F>, indices: &[usize]) -> Result<Array2<F>> {
        let mut sampled = Array2::zeros((indices.len(), data.ncols()));

        for (i, &idx) in indices.iter().enumerate() {
            for j in 0..data.ncols() {
                sampled[[i, j]] = data[[idx, j]];
            }
        }

        Ok(sampled)
    }

    fn compute_percentile(
        &self,
        bootstrap_preds: &[Array1<F>],
        percentile: F,
    ) -> Result<Array1<F>> {
        if bootstrap_preds.is_empty() {
            return Err(MetricsError::InvalidInput(
                "No bootstrap predictions".to_string(),
            ));
        }

        let nsamples = bootstrap_preds[0].len();
        let mut result = Array1::zeros(nsamples);

        for i in 0..nsamples {
            let mut values: Vec<F> = bootstrap_preds.iter().map(|pred| pred[i]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let index = (percentile * F::from(values.len() - 1).unwrap())
                .to_usize()
                .unwrap_or(0);
            result[i] = values[index.min(values.len() - 1)];
        }

        Ok(result)
    }

    fn compute_expected_calibration_error(
        &self,
        predictions: &Array1<F>,
        y_true: &Array1<F>,
    ) -> Result<F> {
        let n_bins = 10;
        let mut ece = F::zero();

        for bin in 0..n_bins {
            let bin_lower = F::from(bin).unwrap() / F::from(n_bins).unwrap();
            let bin_upper = F::from(bin + 1).unwrap() / F::from(n_bins).unwrap();

            let (bin_accuracy, bin_confidence, bin_weight) =
                self.compute_bin_metrics(predictions, y_true, bin_lower, bin_upper)?;

            ece = ece + bin_weight * (bin_accuracy - bin_confidence).abs();
        }

        Ok(ece)
    }

    fn compute_maximum_calibration_error(
        &self,
        predictions: &Array1<F>,
        y_true: &Array1<F>,
    ) -> Result<F> {
        let n_bins = 10;
        let mut mce = F::zero();

        for bin in 0..n_bins {
            let bin_lower = F::from(bin).unwrap() / F::from(n_bins).unwrap();
            let bin_upper = F::from(bin + 1).unwrap() / F::from(n_bins).unwrap();

            let (bin_accuracy, bin_confidence_, _bin_count) =
                self.compute_bin_metrics(predictions, y_true, bin_lower, bin_upper)?;

            let bin_error = (bin_accuracy - bin_confidence_).abs();
            mce = mce.max(bin_error);
        }

        Ok(mce)
    }

    fn compute_bin_metrics(
        &self,
        predictions: &Array1<F>,
        y_true: &Array1<F>,
        bin_lower: F,
        bin_upper: F,
    ) -> Result<(F, F, F)> {
        let mut bin_predictions = Vec::new();
        let mut bin_labels = Vec::new();

        for (i, &pred) in predictions.iter().enumerate() {
            if pred >= bin_lower && pred < bin_upper {
                bin_predictions.push(pred);
                bin_labels.push(y_true[i]);
            }
        }

        if bin_predictions.is_empty() {
            return Ok((F::zero(), F::zero(), F::zero()));
        }

        let bin_accuracy =
            bin_labels.iter().cloned().sum::<F>() / F::from(bin_labels.len()).unwrap();
        let bin_confidence =
            bin_predictions.iter().cloned().sum::<F>() / F::from(bin_predictions.len()).unwrap();
        let bin_weight =
            F::from(bin_predictions.len()).unwrap() / F::from(predictions.len()).unwrap();

        Ok((bin_accuracy, bin_confidence, bin_weight))
    }

    fn compute_reliability_diagram(
        &self,
        predictions: &Array1<F>,
        y_true: &Array1<F>,
    ) -> Result<Vec<(F, F, F)>> {
        let n_bins = 10;
        let mut reliability_data = Vec::new();

        for bin in 0..n_bins {
            let bin_lower = F::from(bin).unwrap() / F::from(n_bins).unwrap();
            let bin_upper = F::from(bin + 1).unwrap() / F::from(n_bins).unwrap();

            let (bin_accuracy, bin_confidence, bin_weight) =
                self.compute_bin_metrics(predictions, y_true, bin_lower, bin_upper)?;

            reliability_data.push((bin_confidence, bin_accuracy, bin_weight));
        }

        Ok(reliability_data)
    }

    fn compute_brier_decomposition(
        &self,
        predictions: &Array1<F>,
        y_true: &Array1<F>,
    ) -> Result<BrierDecomposition<F>> {
        let brier_score = predictions
            .iter()
            .zip(y_true.iter())
            .map(|(&pred, &label)| (pred - label) * (pred - label))
            .sum::<F>()
            / F::from(predictions.len()).unwrap();

        let reliability = self.compute_expected_calibration_error(predictions, y_true)?;

        let mean_pred = predictions.mean().unwrap_or(F::zero());
        let mean_label = y_true.mean().unwrap_or(F::zero());
        let resolution = (mean_pred - mean_label) * (mean_pred - mean_label);

        let uncertainty = mean_label * (F::one() - mean_label);

        Ok(BrierDecomposition {
            brier_score,
            reliability,
            resolution,
            uncertainty,
        })
    }

    fn compute_entropy_uncertainty(&self, predictions: &Array1<F>) -> Result<Array1<F>> {
        let mut entropy = Array1::zeros(predictions.len());

        for (i, &pred) in predictions.iter().enumerate() {
            if pred > F::zero() && pred < F::one() {
                entropy[i] = -pred * pred.ln() - (F::one() - pred) * (F::one() - pred).ln();
            }
        }

        Ok(entropy)
    }

    fn compute_distance_based_confidence(&self, xtest: &Array2<F>) -> Result<Array1<F>> {
        let mut confidence = Array1::zeros(xtest.nrows());

        for i in 0..xtest.nrows() {
            let mut min_distance = F::infinity();

            for j in 0..xtest.nrows() {
                if i != j {
                    let distance = self.compute_euclidean_distance(&xtest.row(i), &xtest.row(j))?;
                    min_distance = min_distance.min(distance);
                }
            }

            confidence[i] = F::one() / (F::one() + min_distance); // Higher distance = lower confidence
        }

        Ok(confidence)
    }

    fn compute_ensemble_confidence<M>(&self, model: &M, xtest: &Array2<F>) -> Result<Array1<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        // Simplified ensemble confidence using prediction variance
        let mut ensemble_predictions = Vec::new();

        for _ in 0..10 {
            let predictions = model(&xtest.view());
            ensemble_predictions.push(predictions);
        }

        let mean_pred = self.compute_mc_mean(&ensemble_predictions)?;
        let variance = self.compute_mc_variance(&ensemble_predictions, &mean_pred)?;

        // Confidence is inverse of variance
        let confidence = variance.mapv(|v| F::one() / (F::one() + v));
        Ok(confidence)
    }

    fn compute_mahalanobis_scores(&self, xtest: &Array2<F>) -> Result<Array1<F>> {
        // Simplified Mahalanobis distance computation
        let mean = xtest.mean_axis(Axis(0)).unwrap();
        let mut scores = Array1::zeros(xtest.nrows());

        for i in 0..xtest.nrows() {
            let diff = &xtest.row(i) - &mean;
            let score = diff.iter().map(|&x| x * x).sum::<F>().sqrt();
            scores[i] = score;
        }

        Ok(scores)
    }

    fn compute_energy_scores<M>(&self, model: &M, xtest: &Array2<F>) -> Result<Array1<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let predictions = model(&xtest.view());

        // Energy score based on prediction magnitude
        let energy_scores = predictions.mapv(|pred| -pred.ln());
        Ok(energy_scores)
    }

    fn compute_reconstruction_errors(&self, xtest: &Array2<F>) -> Result<Array1<F>> {
        // Simplified reconstruction error (assuming identity reconstruction)
        let mut errors = Array1::zeros(xtest.nrows());

        for i in 0..xtest.nrows() {
            let reconstruction_error = xtest.row(i).iter().map(|&x| x * x).sum::<F>().sqrt();
            errors[i] = reconstruction_error;
        }

        Ok(errors)
    }

    fn compute_density_scores(&self, xtest: &Array2<F>) -> Result<Array1<F>> {
        // Simplified density estimation using k-nearest neighbors
        let mut density_scores = Array1::zeros(xtest.nrows());
        let k = 5; // Number of nearest neighbors

        for i in 0..xtest.nrows() {
            let mut distances = Vec::new();

            for j in 0..xtest.nrows() {
                if i != j {
                    let distance = self.compute_euclidean_distance(&xtest.row(i), &xtest.row(j))?;
                    distances.push(distance);
                }
            }

            distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            if distances.len() >= k {
                let kth_distance = distances[k - 1];
                density_scores[i] = F::one() / (F::one() + kth_distance);
            }
        }

        Ok(density_scores)
    }

    // Helper methods for advanced uncertainty quantification

    /// Compute nonconformity scores for conformal prediction
    fn compute_nonconformity_scores(
        &self,
        predictions: &Array1<F>,
        y_true: &Array1<F>,
    ) -> Result<Array1<F>> {
        if predictions.len() != y_true.len() {
            return Err(MetricsError::InvalidInput(
                "Predictions and labels must have same length".to_string(),
            ));
        }

        let mut scores = Array1::zeros(predictions.len());
        for i in 0..predictions.len() {
            scores[i] = (predictions[i] - y_true[i]).abs();
        }

        Ok(scores)
    }

    /// Compute quantile of an array
    fn compute_quantile(&self, values: &Array1<F>, quantile: F) -> Result<F> {
        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if sorted_values.is_empty() {
            return Err(MetricsError::InvalidInput("Empty values array".to_string()));
        }

        let index = (quantile * F::from(sorted_values.len() - 1).unwrap())
            .to_usize()
            .unwrap_or(0);
        let index = index.min(sorted_values.len() - 1);

        Ok(sorted_values[index])
    }

    /// Compute log-likelihood for Bayesian inference
    fn compute_log_likelihood<M>(
        &self,
        model: &M,
        params: &[F],
        xdata: &Array2<F>,
        y_data: &Array1<F>,
    ) -> Result<F>
    where
        M: Fn(&ArrayView2<F>, &[F]) -> Array1<F>,
    {
        let predictions = model(&xdata.view(), params);
        let mut log_likelihood = F::zero();
        let sigma = F::from(0.1).unwrap(); // Noise parameter

        for i in 0..y_data.len() {
            let residual = y_data[i] - predictions[i];
            let log_prob = -F::from(0.5).unwrap() * (residual / sigma) * (residual / sigma)
                - F::from(0.5).unwrap()
                    * (F::from(2.0 * std::f64::consts::PI).unwrap() * sigma * sigma).ln();
            log_likelihood = log_likelihood + log_prob;
        }

        Ok(log_likelihood)
    }

    /// Sample from uniform distribution [0, 1)
    fn uniform_01(&self) -> Result<F> {
        let seed = self.random_seed.unwrap_or(42);
        let u = F::from((seed % 1000) as f64 / 1000.0).unwrap();
        Ok(u)
    }

    /// Sample from standard Gaussian distribution
    fn sample_gaussian(&self) -> Result<F> {
        // Box-Muller transform
        let u1 = self.uniform_01()?;
        let u2 = self.uniform_01()?;

        let z = (-F::from(2.0).unwrap() * u1.ln()).sqrt()
            * (F::from(2.0 * std::f64::consts::PI).unwrap() * u2).cos();

        Ok(z)
    }

    /// Compute R-hat convergence diagnostic for MCMC
    fn compute_r_hat(&self, samples: &[Vec<F>]) -> Result<F> {
        if samples.is_empty() || samples[0].is_empty() {
            return Ok(F::one());
        }

        let _n_samples = samples.len();
        let n_params = samples[0].len();

        // Simplified R-hat computation for single chain
        // In practice, you'd use multiple chains
        let mut r_hat_sum = F::zero();

        for param_idx in 0..n_params {
            let param_values: Vec<F> = samples.iter().map(|s| s[param_idx]).collect();

            // Compute within-chain variance
            let mean =
                param_values.iter().cloned().sum::<F>() / F::from(param_values.len()).unwrap();
            let variance = param_values
                .iter()
                .map(|&x| (x - mean) * (x - mean))
                .sum::<F>()
                / F::from(param_values.len() - 1).unwrap();

            // Simplified R-hat (normally requires multiple chains)
            let r_hat_param = F::one() + variance / (variance + F::from(1e-6).unwrap());
            r_hat_sum = r_hat_sum + r_hat_param;
        }

        Ok(r_hat_sum / F::from(n_params).unwrap())
    }

    // Helper methods for advanced uncertainty quantification

    /// Compute adaptive quantile for conformal prediction
    fn compute_adaptive_quantile(
        &self,
        nonconformity_scores: &Array1<F>,
        alpha: F,
        x_calibration: &Array2<F>,
    ) -> Result<F> {
        // Weight quantiles based on local density
        let mut sorted_scores = nonconformity_scores.to_vec();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Adaptive quantile level based on _calibration data complexity
        let data_complexity = self.estimate_data_complexity(x_calibration)?;
        let adjusted_alpha = alpha * (F::one() + data_complexity * F::from(0.1).unwrap());

        let quantile_level = F::one() - adjusted_alpha.min(F::from(0.95).unwrap());
        let quantile_index = (quantile_level * F::from(sorted_scores.len() - 1).unwrap())
            .to_usize()
            .unwrap_or(0);

        Ok(sorted_scores[quantile_index.min(sorted_scores.len() - 1)])
    }

    /// Compute local conformity adjustment
    fn compute_local_conformity_adjustment(
        &self,
        test_point: &ndarray::ArrayView1<F>,
        x_calibration: &Array2<F>,
        nonconformity_scores: &Array1<F>,
    ) -> Result<F> {
        // Find k nearest neighbors in _calibration set
        let k = 5.min(x_calibration.nrows());
        let mut distances = Vec::new();

        for i in 0..x_calibration.nrows() {
            let distance = self.compute_euclidean_distance(test_point, &x_calibration.row(i))?;
            distances.push((distance, i));
        }

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Compute local adjustment based on neighborhood nonconformity
        let mut local_scores = Vec::new();
        for i in 0..k {
            let idx = distances[i].1;
            local_scores.push(nonconformity_scores[idx]);
        }

        let local_mean =
            local_scores.iter().cloned().sum::<F>() / F::from(local_scores.len()).unwrap();
        let global_mean = nonconformity_scores.mean().unwrap_or(F::one());

        // Adjustment factor: higher if local difficulty is higher than global
        let adjustment = (local_mean / global_mean)
            .max(F::from(0.5).unwrap())
            .min(F::from(2.0).unwrap());

        Ok(adjustment)
    }

    /// Enhanced coverage analysis
    fn compute_enhanced_coverage_analysis(
        &self,
        predictions: &Array1<F>,
        y_true: &Array1<F>,
        nonconformity_scores: &Array1<F>,
        quantile: F,
    ) -> Result<CoverageAnalysis<F>> {
        let nsamples = predictions.len();
        let mut covered = 0;
        let mut set_sizes = Vec::new();
        let mut difficulties = Vec::new();

        // Compute coverage and difficulty analysis
        for i in 0..nsamples {
            let pred = predictions[i];
            let lower = pred - quantile;
            let upper = pred + quantile;
            let set_size = upper - lower;
            set_sizes.push(set_size);

            if y_true[i] >= lower && y_true[i] <= upper {
                covered += 1;
            }

            // Estimate local difficulty based on nonconformity
            difficulties.push(nonconformity_scores[i]);
        }

        let overall_coverage = F::from(covered).unwrap() / F::from(nsamples).unwrap();
        let average_set_size =
            set_sizes.iter().cloned().sum::<F>() / F::from(set_sizes.len()).unwrap();

        // Coverage by difficulty bins
        let mut difficulty_coverage = Vec::new();
        let n_bins = 5;
        let mut sorted_difficulties: Vec<_> = difficulties.iter().enumerate().collect();
        sorted_difficulties
            .sort_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal));

        let bin_size = nsamples / n_bins;
        for bin in 0..n_bins {
            let start = bin * bin_size;
            let end = if bin == n_bins - 1 {
                nsamples
            } else {
                (bin + 1) * bin_size
            };

            let mut bin_covered = 0;
            let mut bin_difficulty_sum = F::zero();

            for &(idx, &difficulty) in &sorted_difficulties[start..end] {
                let pred = predictions[idx];
                let lower = pred - quantile;
                let upper = pred + quantile;

                if y_true[idx] >= lower && y_true[idx] <= upper {
                    bin_covered += 1;
                }
                bin_difficulty_sum = bin_difficulty_sum + difficulty;
            }

            let bin_coverage = F::from(bin_covered).unwrap() / F::from(end - start).unwrap();
            let bin_difficulty = bin_difficulty_sum / F::from(end - start).unwrap();
            difficulty_coverage.push((bin_difficulty, bin_coverage));
        }

        // Adaptive efficiency: smaller sets with maintained coverage are better
        let efficiency = overall_coverage / average_set_size;

        // Local coverage variance
        let coverage_variance = difficulty_coverage
            .iter()
            .map(|(_, cov)| (*cov - overall_coverage) * (*cov - overall_coverage))
            .sum::<F>()
            / F::from(difficulty_coverage.len()).unwrap();

        Ok(CoverageAnalysis {
            overall_coverage,
            average_set_size,
            difficulty_coverage,
            adaptive_efficiency: efficiency,
            local_coverage_variance: coverage_variance,
        })
    }

    /// Compute conditional coverage analysis
    fn compute_conditional_coverage_analysis(
        &self,
        prediction_sets: &[PredictionSet<F>],
        _x_test: &Array2<F>,
        cal_predictions: &Array1<F>,
        y_calibration: &Array1<F>,
    ) -> Result<HashMap<String, F>> {
        let mut conditional_coverage = HashMap::new();

        // Coverage by difficulty level
        let difficulties: Vec<F> = prediction_sets
            .iter()
            .map(|ps| ps.local_difficulty)
            .collect();
        let median_difficulty = {
            let mut sorted_diff = difficulties.clone();
            sorted_diff.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            sorted_diff[sorted_diff.len() / 2]
        };

        // Split into easy and hard cases
        let mut easy_coverage = 0;
        let mut easy_count = 0;
        let mut hard_coverage = 0;
        let mut hard_count = 0;

        for (i, ps) in prediction_sets.iter().enumerate() {
            // Estimate coverage based on prediction set characteristics and _calibration data
            // Use heuristic: coverage is higher for larger _sets and lower difficulty cases
            let base_coverage_prob = F::from(0.85).unwrap(); // Base coverage probability
            let difficulty_factor = F::one()
                - (ps.local_difficulty - median_difficulty).abs()
                    / (difficulties
                        .iter()
                        .copied()
                        .fold(F::zero(), |acc, d| acc.max(d))
                        + F::from(1e-8).unwrap());
            let size_factor = ps.size / F::from(10.0).unwrap(); // Normalize by expected set size

            let coverage_prob = base_coverage_prob * difficulty_factor * size_factor.min(F::one());

            // Use _calibration data as a proxy for expected coverage
            let cal_coverage_estimate = if i < cal_predictions.len() && i < y_calibration.len() {
                let prediction_error = (cal_predictions[i] - y_calibration[i]).abs();
                let mean_error = cal_predictions
                    .iter()
                    .zip(y_calibration.iter())
                    .map(|(&p, &y)| (p - y).abs())
                    .sum::<F>()
                    / F::from(cal_predictions.len()).unwrap();

                if prediction_error <= mean_error {
                    F::from(0.9).unwrap()
                } else {
                    F::from(0.7).unwrap()
                }
            } else {
                coverage_prob
            };

            // Combine heuristics for final coverage estimate
            let estimated_coverage =
                (coverage_prob + cal_coverage_estimate) / F::from(2.0).unwrap();
            let is_covered = estimated_coverage > F::from(0.8).unwrap(); // Threshold for "covered"

            if ps.local_difficulty <= median_difficulty {
                if is_covered {
                    easy_coverage += 1;
                }
                easy_count += 1;
            } else {
                if is_covered {
                    hard_coverage += 1;
                }
                hard_count += 1;
            }
        }

        if easy_count > 0 {
            conditional_coverage.insert(
                "easy_cases".to_string(),
                F::from(easy_coverage).unwrap() / F::from(easy_count).unwrap(),
            );
        }
        if hard_count > 0 {
            conditional_coverage.insert(
                "hard_cases".to_string(),
                F::from(hard_coverage).unwrap() / F::from(hard_count).unwrap(),
            );
        }

        Ok(conditional_coverage)
    }

    /// Estimate data complexity for adaptive methods
    fn estimate_data_complexity(&self, xdata: &Array2<F>) -> Result<F> {
        // Estimate complexity using intrinsic dimensionality and local variation
        let nsamples = xdata.nrows();
        let n_features = xdata.ncols();

        if nsamples < 2 {
            return Ok(F::zero());
        }

        // Compute pairwise distances and estimate intrinsic dimensionality
        let mut distances = Vec::new();
        let sample_size = 100.min(nsamples); // Sample for efficiency

        for i in 0..sample_size {
            for j in (i + 1)..sample_size {
                let distance = self.compute_euclidean_distance(&xdata.row(i), &xdata.row(j))?;
                distances.push(distance);
            }
        }

        if distances.is_empty() {
            return Ok(F::zero());
        }

        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Complexity based on distance distribution
        let median_distance = distances[distances.len() / 2];
        let iqr = distances[distances.len() * 3 / 4] - distances[distances.len() / 4];

        // Normalize complexity by feature dimensionality
        let complexity = (iqr / median_distance) / F::from(n_features).unwrap().sqrt();

        Ok(complexity.max(F::zero()).min(F::one()))
    }

    /// Compute KL divergence for Gaussian variational parameters
    fn compute_kl_divergence_gaussian(&self, varparams: &VariationalParams<F>) -> Result<F> {
        let mut kl_div = F::zero();

        // KL divergence for weights: KL(q(w)||p(w)) where p(w) ~ N(0, I)
        for (&mean, &log_var) in varparams
            .weight_means
            .iter()
            .zip(varparams.weight_log_vars.iter())
        {
            let var = log_var.exp();
            // KL(N(μ,σ²)||N(0,1)) = 0.5 * (σ² + μ² - 1 - log(σ²))
            kl_div = kl_div + F::from(0.5).unwrap() * (var + mean * mean - F::one() - log_var);
        }

        // KL divergence for biases
        for (&mean, &log_var) in varparams
            .bias_means
            .iter()
            .zip(varparams.bias_log_vars.iter())
        {
            let var = log_var.exp();
            kl_div = kl_div + F::from(0.5).unwrap() * (var + mean * mean - F::one() - log_var);
        }

        Ok(kl_div)
    }

    /// Compute variational log likelihood
    fn compute_variational_log_likelihood(
        &self,
        mean_pred: &Array1<F>,
        var_pred: &Array1<F>,
        _x_data: &ArrayView2<F>,
        y_data: &Array1<F>,
    ) -> Result<F> {
        let mut log_lik = F::zero();
        let pi = F::from(std::f64::consts::PI).unwrap();

        for i in 0..y_data.len() {
            let pred_mean = mean_pred[i];
            let pred_var = var_pred[i];
            let y_true = y_data[i];

            // Gaussian likelihood: log p(y|x) = -0.5 * log(2πσ²) - (y-μ)²/(2σ²)
            let log_prob = -F::from(0.5).unwrap() * (F::from(2.0).unwrap() * pi * pred_var).ln()
                - (y_true - pred_mean) * (y_true - pred_mean) / (F::from(2.0).unwrap() * pred_var);

            log_lik = log_lik + log_prob;
        }

        Ok(log_lik)
    }

    /// Compute scale uncertainty for multi-scale methods
    fn compute_scale_uncertainty(&self, predictions: &Array1<F>, scale: F) -> Result<Array1<F>> {
        let mut uncertainties = Array1::zeros(predictions.len());

        // Uncertainty increases with finer scales (higher resolution)
        let scale_factor = F::one() / (F::one() + scale);

        for i in 0..predictions.len() {
            // Local variation-based uncertainty estimate
            let neighbors = if i > 0 && i < predictions.len() - 1 {
                vec![predictions[i - 1], predictions[i], predictions[i + 1]]
            } else if i == 0 && predictions.len() > 1 {
                vec![predictions[i], predictions[i + 1]]
            } else if i == predictions.len() - 1 && predictions.len() > 1 {
                vec![predictions[i - 1], predictions[i]]
            } else {
                vec![predictions[i]]
            };

            let mean = neighbors.iter().cloned().sum::<F>() / F::from(neighbors.len()).unwrap();
            let variance = neighbors
                .iter()
                .map(|&x| (x - mean) * (x - mean))
                .sum::<F>()
                / F::from(neighbors.len()).unwrap();

            uncertainties[i] = variance.sqrt() * scale_factor;
        }

        Ok(uncertainties)
    }

    /// Compute scale weights for multi-scale fusion
    fn compute_scale_weights(
        &self,
        scales: &[F],
        uncertainties: &[Array1<F>],
    ) -> Result<Array1<F>> {
        let _n_samples = uncertainties[0].len();
        let n_scales = scales.len();
        let mut weights = Array1::zeros(n_scales);

        // Compute inverse uncertainty weights
        for i in 0..n_scales {
            let avg_uncertainty = uncertainties[i].mean().unwrap_or(F::one());
            weights[i] = F::one() / (avg_uncertainty + F::from(1e-6).unwrap());
        }

        // Normalize weights
        let weight_sum = weights.sum();
        if weight_sum > F::zero() {
            weights = weights / weight_sum;
        } else {
            weights.fill(F::one() / F::from(n_scales).unwrap());
        }

        Ok(weights)
    }

    /// Fuse multi-scale predictions using weighted averaging
    fn fuse_multiscale_predictions(
        &self,
        predictions: &[Array1<F>],
        weights: &Array1<F>,
    ) -> Result<Array1<F>> {
        let nsamples = predictions[0].len();
        let mut fused = Array1::zeros(nsamples);

        for i in 0..nsamples {
            let mut weighted_sum = F::zero();
            for (j, pred) in predictions.iter().enumerate() {
                weighted_sum = weighted_sum + weights[j] * pred[i];
            }
            fused[i] = weighted_sum;
        }

        Ok(fused)
    }

    /// Compute cross-scale consistency
    fn compute_cross_scale_consistency(&self, predictions: &[Array1<F>]) -> Result<Array1<F>> {
        let nsamples = predictions[0].len();
        let n_scales = predictions.len();
        let mut consistency = Array1::zeros(nsamples);

        for i in 0..nsamples {
            let sample_preds: Vec<F> = predictions.iter().map(|pred| pred[i]).collect();
            let mean = sample_preds.iter().cloned().sum::<F>() / F::from(n_scales).unwrap();
            let variance = sample_preds
                .iter()
                .map(|&x| (x - mean) * (x - mean))
                .sum::<F>()
                / F::from(n_scales).unwrap();

            // Consistency = 1 / (1 + variance)
            consistency[i] = F::one() / (F::one() + variance);
        }

        Ok(consistency)
    }

    /// Decompose multi-scale uncertainty
    fn decompose_multiscale_uncertainty(
        &self,
        predictions: &[Array1<F>],
        uncertainties: &[Array1<F>],
        weights: &Array1<F>,
    ) -> Result<UncertaintyDecomposition<F>> {
        let nsamples = predictions[0].len();
        let n_scales = predictions.len();

        let mut within_scale = Array1::zeros(nsamples);
        let mut between_scale = Array1::zeros(nsamples);
        let mut scale_contributions = vec![F::zero(); n_scales];

        for i in 0..nsamples {
            // Within-scale uncertainty (weighted average of individual uncertainties)
            let mut weighted_within = F::zero();
            for j in 0..n_scales {
                weighted_within = weighted_within + weights[j] * uncertainties[j][i];
                scale_contributions[j] = scale_contributions[j] + weights[j] * uncertainties[j][i];
            }
            within_scale[i] = weighted_within;

            // Between-scale uncertainty (variance of predictions across scales)
            let sample_preds: Vec<F> = predictions.iter().map(|pred| pred[i]).collect();
            let mean = sample_preds.iter().cloned().sum::<F>() / F::from(n_scales).unwrap();
            let variance = sample_preds
                .iter()
                .map(|&x| (x - mean) * (x - mean))
                .sum::<F>()
                / F::from(n_scales).unwrap();
            between_scale[i] = variance;
        }

        // Normalize scale contributions
        let total_contribution = scale_contributions.iter().cloned().sum::<F>();
        if total_contribution > F::zero() {
            for contrib in &mut scale_contributions {
                *contrib = *contrib / total_contribution;
            }
        }

        let total_uncertainty = &within_scale + &between_scale;

        Ok(UncertaintyDecomposition {
            within_scale_uncertainty: within_scale,
            between_scale_uncertainty: between_scale,
            total_uncertainty,
            scale_contributions,
        })
    }
}

/// Comprehensive uncertainty analysis results
#[derive(Debug, Clone)]
pub struct UncertaintyAnalysis<F: Float> {
    /// Epistemic uncertainty (model uncertainty)
    pub epistemic_uncertainty: EpistemicUncertainty<F>,
    /// Aleatoric uncertainty (data uncertainty)
    pub aleatoric_uncertainty: AleatoricUncertainty<F>,
    /// Prediction intervals
    pub prediction_intervals: PredictionIntervals<F>,
    /// Calibration metrics (if labels available)
    pub calibration_metrics: Option<CalibrationMetrics<F>>,
    /// Confidence scores
    pub confidence_scores: ConfidenceScores<F>,
    /// Out-of-distribution detection scores
    pub ood_scores: OODScores<F>,
    /// Sample size
    pub sample_size: usize,
}

/// Epistemic uncertainty metrics
#[derive(Debug, Clone)]
pub struct EpistemicUncertainty<F: Float> {
    /// Mean predictions across MC samples
    pub mean_predictions: Array1<F>,
    /// Prediction variance
    pub prediction_variance: Array1<F>,
    /// Prediction entropy
    pub prediction_entropy: Array1<F>,
    /// Mutual information
    pub mutual_information: F,
    /// Number of MC samples used
    pub mc_samples: usize,
}

/// Aleatoric uncertainty metrics
#[derive(Debug, Clone)]
pub struct AleatoricUncertainty<F: Float> {
    /// Input sensitivity measure
    pub input_sensitivity: Array1<F>,
    /// Data uncertainty estimate
    pub data_uncertainty: Array1<F>,
    /// Noise level used for estimation
    pub noise_level: F,
}

/// Prediction intervals
#[derive(Debug, Clone)]
pub struct PredictionIntervals<F: Float> {
    /// Lower bound of interval
    pub lower_bound: Array1<F>,
    /// Upper bound of interval
    pub upper_bound: Array1<F>,
    /// Median prediction
    pub median_prediction: Array1<F>,
    /// Confidence level
    pub confidence_level: F,
}

/// Calibration metrics
#[derive(Debug, Clone)]
pub struct CalibrationMetrics<F: Float> {
    /// Expected Calibration Error
    pub expected_calibration_error: F,
    /// Maximum Calibration Error
    pub maximum_calibration_error: F,
    /// Reliability diagram data (confidence, accuracy, weight)
    pub reliability_data: Vec<(F, F, F)>,
    /// Brier score decomposition
    pub brier_decomposition: BrierDecomposition<F>,
}

/// Brier score decomposition
#[derive(Debug, Clone)]
pub struct BrierDecomposition<F: Float> {
    /// Overall Brier score
    pub brier_score: F,
    /// Reliability component
    pub reliability: F,
    /// Resolution component
    pub resolution: F,
    /// Uncertainty component
    pub uncertainty: F,
}

/// Confidence scores
#[derive(Debug, Clone)]
pub struct ConfidenceScores<F: Float> {
    /// Maximum probability confidence
    pub max_confidence: F,
    /// Entropy-based uncertainty
    pub entropy_uncertainty: Array1<F>,
    /// Distance-based confidence
    pub distance_confidence: Array1<F>,
    /// Ensemble-based confidence
    pub ensemble_confidence: Array1<F>,
}

/// Out-of-distribution detection scores
#[derive(Debug, Clone)]
pub struct OODScores<F: Float> {
    /// Mahalanobis distance scores
    pub mahalanobis_scores: Array1<F>,
    /// Energy-based scores
    pub energy_scores: Array1<F>,
    /// Reconstruction error scores
    pub reconstruction_errors: Array1<F>,
    /// Density-based scores
    pub density_scores: Array1<F>,
}

/// Advanced uncertainty analysis results
#[derive(Debug, Clone)]
pub struct AdvancedUncertaintyAnalysis<F: Float> {
    /// Epistemic uncertainty (model uncertainty)
    pub epistemic_uncertainty: EpistemicUncertainty<F>,
    /// Aleatoric uncertainty (data uncertainty)
    pub aleatoric_uncertainty: AleatoricUncertainty<F>,
    /// Prediction intervals
    pub prediction_intervals: PredictionIntervals<F>,
    /// Calibration metrics (if labels available)
    pub calibration_metrics: Option<CalibrationMetrics<F>>,
    /// Confidence scores
    pub confidence_scores: ConfidenceScores<F>,
    /// Out-of-distribution detection scores
    pub ood_scores: OODScores<F>,
    /// Conformal prediction results
    pub conformal_prediction: Option<ConformalPrediction<F>>,
    /// Bayesian uncertainty estimation
    pub bayesian_uncertainty: Option<BayesianUncertainty<F>>,
    /// Temperature scaling results
    pub temperature_scaling: Option<TemperatureScaling<F>>,
    /// Deep ensemble uncertainty
    pub deep_ensemble_uncertainty: DeepEnsembleUncertainty<F>,
    /// Sample size
    pub sample_size: usize,
}

/// Enhanced conformal prediction results
#[derive(Debug, Clone)]
pub struct ConformalPrediction<F: Float> {
    /// Prediction sets for each test sample
    pub prediction_sets: Vec<PredictionSet<F>>,
    /// Coverage probability
    pub coverage_probability: F,
    /// Average set size
    pub average_set_size: F,
    /// Conditional coverage by groups
    pub conditional_coverage: HashMap<String, F>,
    /// Enhanced coverage analysis
    pub coverage_analysis: Option<CoverageAnalysis<F>>,
}

/// Enhanced prediction set for conformal prediction
#[derive(Debug, Clone)]
pub struct PredictionSet<F: Float> {
    /// Lower bound
    pub lower: F,
    /// Upper bound
    pub upper: F,
    /// Set size
    pub size: F,
    /// Contains true value (if known)
    pub contains_truth: Option<bool>,
    /// Local difficulty adjustment factor
    pub local_difficulty: F,
    /// Adaptive quantile used
    pub adaptive_quantile: F,
}

/// Bayesian uncertainty estimation
#[derive(Debug, Clone)]
pub struct BayesianUncertainty<F: Float> {
    /// Posterior mean predictions
    pub posterior_mean: Array1<F>,
    /// Posterior variance
    pub posterior_variance: Array1<F>,
    /// Credible intervals
    pub credible_intervals: Array2<F>, // [nsamples, 2] for lower/upper bounds
    /// Model evidence (marginal likelihood)
    pub model_evidence: F,
    /// MCMC effective sample size
    pub effective_sample_size: F,
    /// R-hat convergence diagnostic
    pub r_hat: F,
}

/// Temperature scaling results
#[derive(Debug, Clone)]
pub struct TemperatureScaling<F: Float> {
    /// Optimal temperature parameter
    pub temperature: F,
    /// Calibrated predictions
    pub calibrated_predictions: Array1<F>,
    /// Before calibration error
    pub before_calibration_error: F,
    /// After calibration error
    pub after_calibration_error: F,
    /// Calibration improvement
    pub improvement: F,
}

/// Deep ensemble uncertainty
#[derive(Debug, Clone)]
pub struct DeepEnsembleUncertainty<F: Float> {
    /// Individual model predictions
    pub individual_predictions: Array2<F>, // [n_models, nsamples]
    /// Ensemble mean
    pub ensemble_mean: Array1<F>,
    /// Ensemble variance
    pub ensemble_variance: Array1<F>,
    /// Model disagreement
    pub model_disagreement: Array1<F>,
    /// Diversity scores
    pub diversity_scores: Array1<F>,
}

/// Variational parameters for Bayesian neural networks
#[derive(Debug, Clone)]
pub struct VariationalParams<F: Float> {
    /// Means of weight distributions
    pub weight_means: Vec<F>,
    /// Log variances of weight distributions
    pub weight_log_vars: Vec<F>,
    /// Bias means
    pub bias_means: Vec<F>,
    /// Bias log variances
    pub bias_log_vars: Vec<F>,
}

/// Variational uncertainty results
#[derive(Debug, Clone)]
pub struct VariationalUncertainty<F: Float> {
    /// Ensemble mean predictions
    pub ensemble_mean: Array1<F>,
    /// Epistemic uncertainty
    pub epistemic_uncertainty: Array1<F>,
    /// Aleatoric uncertainty
    pub aleatoric_uncertainty: Array1<F>,
    /// KL divergences for each sample
    pub kl_divergences: Vec<F>,
    /// ELBO values for each sample
    pub elbo_values: Vec<F>,
    /// Mean ELBO
    pub mean_elbo: F,
    /// ELBO variance (convergence indicator)
    pub elbo_variance: F,
    /// Convergence diagnostic
    pub convergence_diagnostic: String,
}

/// Multi-scale uncertainty quantification results
#[derive(Debug, Clone)]
pub struct MultiscaleUncertainty<F: Float> {
    /// Predictions at each scale
    pub scale_predictions: Vec<Array1<F>>,
    /// Uncertainties at each scale
    pub scale_uncertainties: Vec<Array1<F>>,
    /// Fused multi-scale predictions
    pub fused_predictions: Array1<F>,
    /// Cross-scale consistency scores
    pub consistency_scores: Array1<F>,
    /// Hierarchical uncertainty decomposition
    pub uncertainty_decomposition: UncertaintyDecomposition<F>,
    /// Scale values
    pub scales: Vec<F>,
    /// Scale weights used for fusion
    pub weights: Array1<F>,
}

/// Hierarchical uncertainty decomposition
#[derive(Debug, Clone)]
pub struct UncertaintyDecomposition<F: Float> {
    /// Within-scale uncertainty
    pub within_scale_uncertainty: Array1<F>,
    /// Between-scale uncertainty
    pub between_scale_uncertainty: Array1<F>,
    /// Total uncertainty
    pub total_uncertainty: Array1<F>,
    /// Scale-specific contributions
    pub scale_contributions: Vec<F>,
}

/// Enhanced coverage analysis
#[derive(Debug, Clone)]
pub struct CoverageAnalysis<F: Float> {
    /// Overall coverage probability
    pub overall_coverage: F,
    /// Average set size
    pub average_set_size: F,
    /// Coverage by difficulty bins
    pub difficulty_coverage: Vec<(F, F)>, // (difficulty_level, coverage)
    /// Adaptive efficiency metric
    pub adaptive_efficiency: F,
    /// Local coverage variance
    pub local_coverage_variance: F,
}

// Additional convenience functions for uncertainty quantification

/// Compute entropy of probability distribution
#[allow(dead_code)]
pub fn compute_entropy<F: Float + num_traits::FromPrimitive>(probabilities: &Array1<F>) -> F {
    let mut entropy = F::zero();
    let eps = F::from(1e-15).unwrap();

    for &p in probabilities.iter() {
        if p > eps {
            entropy = entropy - p * p.ln();
        }
    }

    entropy
}

/// Compute KL divergence between two distributions
#[allow(dead_code)]
pub fn compute_kl_divergence<F: Float + num_traits::FromPrimitive>(
    p: &Array1<F>,
    q: &Array1<F>,
) -> Result<F> {
    if p.len() != q.len() {
        return Err(MetricsError::InvalidInput(
            "Distributions must have same length".to_string(),
        ));
    }

    let mut kl_div = F::zero();
    let eps = F::from(1e-15).unwrap();

    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if pi > eps && qi > eps {
            kl_div = kl_div + pi * (pi / qi).ln();
        }
    }

    Ok(kl_div)
}

/// Compute Jensen-Shannon divergence
#[allow(dead_code)]
pub fn compute_js_divergence<F: Float + num_traits::FromPrimitive + ndarray::ScalarOperand>(
    p: &Array1<F>,
    q: &Array1<F>,
) -> Result<F> {
    let m = (p + q) / F::from(2.0).unwrap();
    let kl_pm = compute_kl_divergence(p, &m)?;
    let kl_qm = compute_kl_divergence(q, &m)?;

    Ok((kl_pm + kl_qm) / F::from(2.0).unwrap())
}

/// Compute Wasserstein distance (simplified 1D version)
#[allow(dead_code)]
pub fn compute_wasserstein_distance<F: Float + num_traits::FromPrimitive>(
    samples1: &Array1<F>,
    samples2: &Array1<F>,
) -> F {
    let mut s1 = samples1.to_vec();
    let mut s2 = samples2.to_vec();

    s1.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    s2.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let min_len = s1.len().min(s2.len());
    let mut distance = F::zero();

    for i in 0..min_len {
        distance = distance + (s1[i] - s2[i]).abs();
    }

    distance / F::from(min_len).unwrap()
}

/// Compute maximum mean discrepancy (simplified)
#[allow(dead_code)]
pub fn compute_mmd<F: Float + num_traits::FromPrimitive + std::iter::Sum>(
    samples1: &Array2<F>,
    samples2: &Array2<F>,
    gamma: F,
) -> Result<F> {
    if samples1.ncols() != samples2.ncols() {
        return Err(MetricsError::InvalidInput(
            "Samples must have same dimensionality".to_string(),
        ));
    }

    let n1 = samples1.nrows();
    let n2 = samples2.nrows();

    // Compute kernel means
    let mut k11 = F::zero();
    let mut k22 = F::zero();
    let mut k12 = F::zero();

    // K(X, X)
    for i in 0..n1 {
        for j in 0..n1 {
            let dist_sq = samples1
                .row(i)
                .iter()
                .zip(samples1.row(j).iter())
                .map(|(&xi, &xj)| (xi - xj) * (xi - xj))
                .sum::<F>();
            k11 = k11 + (-gamma * dist_sq).exp();
        }
    }
    k11 = k11 / F::from(n1 * n1).unwrap();

    // K(Y, Y)
    for i in 0..n2 {
        for j in 0..n2 {
            let dist_sq = samples2
                .row(i)
                .iter()
                .zip(samples2.row(j).iter())
                .map(|(&yi, &yj)| (yi - yj) * (yi - yj))
                .sum::<F>();
            k22 = k22 + (-gamma * dist_sq).exp();
        }
    }
    k22 = k22 / F::from(n2 * n2).unwrap();

    // K(X, Y)
    for i in 0..n1 {
        for j in 0..n2 {
            let dist_sq = samples1
                .row(i)
                .iter()
                .zip(samples2.row(j).iter())
                .map(|(&xi, &yj)| (xi - yj) * (xi - yj))
                .sum::<F>();
            k12 = k12 + (-gamma * dist_sq).exp();
        }
    }
    k12 = k12 / F::from(n1 * n2).unwrap();

    Ok(k11 + k22 - F::from(2.0).unwrap() * k12)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // Mock model for testing
    fn mock_model(x: &ArrayView2<f64>) -> Array1<f64> {
        x.map_axis(Axis(1), |row| row.sum())
    }

    #[test]
    fn test_uncertainty_quantifier_creation() {
        let quantifier = UncertaintyQuantifier::<f64>::new()
            .with_mc_samples(500)
            .with_confidence_level(0.9)
            .with_bootstrap(50)
            .with_seed(42);

        assert_eq!(quantifier.n_mc_samples, 500);
        assert_eq!(quantifier.confidence_level, 0.9);
        assert_eq!(quantifier.n_bootstrap, 50);
        assert_eq!(quantifier.random_seed, Some(42));
    }

    #[test]
    fn test_epistemic_uncertainty() {
        let quantifier = UncertaintyQuantifier::<f64>::new()
            .with_mc_samples(10)
            .with_seed(42);

        let xtest = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let epistemic = quantifier
            .compute_epistemic_uncertainty(&mock_model, &xtest)
            .unwrap();

        assert_eq!(epistemic.mean_predictions.len(), 3);
        assert_eq!(epistemic.prediction_variance.len(), 3);
        assert_eq!(epistemic.mc_samples, 10);
    }

    #[test]
    fn test_prediction_intervals() {
        let quantifier = UncertaintyQuantifier::<f64>::new()
            .with_bootstrap(10)
            .with_confidence_level(0.95)
            .with_seed(42);

        let xtest = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let intervals = quantifier
            .compute_prediction_intervals(&mock_model, &xtest)
            .unwrap();

        assert_eq!(intervals.lower_bound.len(), 3);
        assert_eq!(intervals.upper_bound.len(), 3);
        assert_eq!(intervals.median_prediction.len(), 3);
        assert_eq!(intervals.confidence_level, 0.95);
    }

    #[test]
    fn test_calibration_metrics() {
        let quantifier = UncertaintyQuantifier::<f64>::new().with_seed(42);

        let _predictions = array![0.1, 0.4, 0.7, 0.9];
        let y_true = array![0.0, 0.0, 1.0, 1.0];

        let calibration = quantifier
            .compute_calibration_metrics(
                &mock_model,
                &array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                &y_true,
            )
            .unwrap();

        assert!(calibration.expected_calibration_error >= 0.0);
        assert!(calibration.maximum_calibration_error >= 0.0);
        assert_eq!(calibration.reliability_data.len(), 10); // 10 bins
    }

    #[test]
    fn test_ood_scores() {
        let quantifier = UncertaintyQuantifier::<f64>::new().with_seed(42);
        let xtest = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let ood_scores = quantifier.compute_ood_scores(&mock_model, &xtest).unwrap();

        assert_eq!(ood_scores.mahalanobis_scores.len(), 3);
        assert_eq!(ood_scores.energy_scores.len(), 3);
        assert_eq!(ood_scores.reconstruction_errors.len(), 3);
        assert_eq!(ood_scores.density_scores.len(), 3);
    }

    #[test]
    fn test_gaussian_noise_generation() {
        let quantifier = UncertaintyQuantifier::<f64>::new().with_seed(42);
        let noise = quantifier.generate_gaussian_noise().unwrap();

        // Just check that noise is finite
        assert!(noise.is_finite());
    }

    #[test]
    fn test_euclidean_distance() {
        let quantifier = UncertaintyQuantifier::<f64>::new();
        let x1 = array![1.0, 2.0, 3.0];
        let x2 = array![4.0, 5.0, 6.0];

        let distance = quantifier
            .compute_euclidean_distance(&x1.view(), &x2.view())
            .unwrap();
        let expected = ((3.0_f64).powi(2) * 3.0).sqrt(); // sqrt(9 + 9 + 9) = sqrt(27)
        assert!((distance - expected).abs() < 1e-10);
    }

    #[test]
    fn test_brier_decomposition() {
        let quantifier = UncertaintyQuantifier::<f64>::new();
        let predictions = array![0.2, 0.8, 0.6, 0.9];
        let y_true = array![0.0, 1.0, 1.0, 1.0];

        let decomposition = quantifier
            .compute_brier_decomposition(&predictions, &y_true)
            .unwrap();

        assert!(decomposition.brier_score >= 0.0);
        assert!(decomposition.reliability >= 0.0);
        assert!(decomposition.uncertainty >= 0.0);
    }

    #[test]
    fn test_comprehensive_uncertainty_analysis() {
        let quantifier = UncertaintyQuantifier::<f64>::new()
            .with_mc_samples(5)
            .with_bootstrap(5)
            .with_seed(42);

        let xtest = array![[1.0, 2.0], [3.0, 4.0]];
        let y_test = array![0.0, 1.0];

        let analysis = quantifier
            .compute_uncertainty(&mock_model, &xtest, Some(&y_test))
            .unwrap();

        assert_eq!(analysis.sample_size, 2);
        assert!(analysis.calibration_metrics.is_some());
        assert_eq!(analysis.epistemic_uncertainty.mean_predictions.len(), 2);
        assert_eq!(analysis.prediction_intervals.lower_bound.len(), 2);
    }

    #[test]
    #[ignore] // FIXME: Test failing - needs investigation
    fn test_random_number_generators() {
        let mut lcg = LcgRng::new(42);
        let mut xorshift = XorshiftRng::new(42);
        let mut pcg = PcgRng::new(42);
        let mut chacha = ChaChaRng::new(42);

        // Test uniform generation
        let lcg_val = lcg.uniform_01::<f64>();
        let xorshift_val = xorshift.uniform_01::<f64>();
        let pcg_val = pcg.uniform_01::<f64>();
        let chacha_val = chacha.uniform_01::<f64>();

        assert!((0.0..=1.0).contains(&lcg_val));
        assert!((0.0..=1.0).contains(&xorshift_val));
        assert!((0.0..=1.0).contains(&pcg_val));
        assert!((0.0..=1.0).contains(&chacha_val));

        // Test normal generation
        let lcg_normal = lcg.normal::<f64>();
        let xorshift_normal = xorshift.normal::<f64>();
        let pcg_normal = pcg.normal::<f64>();
        let chacha_normal = chacha.normal::<f64>();

        assert!(lcg_normal.is_finite());
        assert!(xorshift_normal.is_finite());
        assert!(pcg_normal.is_finite());
        assert!(chacha_normal.is_finite());
    }

    #[test]
    fn test_advanced_uncertainty_quantifier() {
        let quantifier = UncertaintyQuantifier::<f64>::new()
            .with_rng_type(RandomNumberGenerator::Pcg)
            .with_conformal_calibration(50)
            .with_bayesian(true)
            .with_mcmc(100, 20)
            .with_temperature_scaling(true)
            .with_simd(true);

        assert_eq!(quantifier.n_conformal_calibration, 50);
        assert!(quantifier.enable_bayesian);
        assert_eq!(quantifier.n_mcmc_samples, 100);
        assert_eq!(quantifier.mcmc_burn_in, 20);
        assert!(quantifier.enable_temperature_scaling);
        assert!(quantifier.enable_simd);
    }

    #[test]
    fn test_entropy_computation() {
        let probs = array![0.1, 0.2, 0.3, 0.4];
        let entropy = compute_entropy(&probs);
        assert!(entropy > 0.0);

        // Uniform distribution should have maximum entropy
        let uniform_probs = array![0.25, 0.25, 0.25, 0.25];
        let uniform_entropy = compute_entropy(&uniform_probs);
        assert!(uniform_entropy > entropy);
    }

    #[test]
    fn test_kl_divergence() {
        let p = array![0.5, 0.3, 0.2];
        let q = array![0.4, 0.4, 0.2];

        let kl_div = compute_kl_divergence(&p, &q).unwrap();
        assert!(kl_div >= 0.0);

        // KL divergence should be 0 for identical distributions
        let kl_self = compute_kl_divergence(&p, &p).unwrap();
        assert!(kl_self.abs() < 1e-10);
    }

    #[test]
    fn test_js_divergence() {
        let p = array![0.5, 0.3, 0.2];
        let q = array![0.4, 0.4, 0.2];

        let js_div = compute_js_divergence(&p, &q).unwrap();
        assert!(js_div >= 0.0);

        // JS divergence should be 0 for identical distributions
        let js_self = compute_js_divergence(&p, &p).unwrap();
        assert!(js_self.abs() < 1e-10);
    }

    #[test]
    fn test_wasserstein_distance() {
        let samples1 = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let samples2 = array![1.1, 2.1, 3.1, 4.1, 5.1];

        let wasserstein = compute_wasserstein_distance(&samples1, &samples2);
        assert!(wasserstein >= 0.0);
        assert!((wasserstein - 0.1).abs() < 1e-10);
    }
}
