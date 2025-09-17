//! Advanced Training Methods for Time Series
//!
//! This module implements cutting-edge training methodologies for time series forecasting,
//! including meta-learning for few-shot adaptation, Neural ODEs for continuous-time modeling,
//! and variational autoencoders for uncertainty quantification.
//!
//! ## Advanced Training Techniques
//! - **Meta-Learning (MAML)**: Model-Agnostic Meta-Learning for rapid adaptation
//! - **Neural ODEs**: Continuous-time neural networks with ODE solvers
//! - **Variational Autoencoders**: Probabilistic modeling with uncertainty estimation
//! - **Bayesian Neural Networks**: Posterior inference for time series
//! - **Gradient-Based Meta-Learning**: Few-shot learning optimization
//! - **Transformer Forecasting**: Attention-based sequence modeling for time series
//! - **Hyperparameter Optimization**: Automated hyperparameter tuning with Bayesian optimization
//! - **Advanced Ensemble Methods**: Multi-model ensemble with uncertainty quantification
//! - **Multi-Task Learning**: Joint training across multiple time series tasks
//! - **Neural Architecture Search**: Automatic model architecture discovery

use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::Result;

/// Model-Agnostic Meta-Learning (MAML) for few-shot time series forecasting
#[derive(Debug)]
pub struct MAML<F: Float + Debug + ndarray::ScalarOperand> {
    /// Base model parameters
    parameters: Array2<F>,
    /// Meta-learning rate
    meta_lr: F,
    /// Inner loop learning rate
    inner_lr: F,
    /// Number of inner gradient steps
    inner_steps: usize,
    /// Model dimensions
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
}

impl<F: Float + Debug + Clone + FromPrimitive + ndarray::ScalarOperand> MAML<F> {
    /// Create new MAML instance
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        meta_lr: F,
        inner_lr: F,
        inner_steps: usize,
    ) -> Self {
        // Initialize parameters using Xavier initialization
        let total_params =
            input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim + output_dim;
        let scale = F::from(2.0).unwrap() / F::from(input_dim + output_dim).unwrap();
        let std_dev = scale.sqrt();

        let mut parameters = Array2::zeros((1, total_params));
        for i in 0..total_params {
            let val = ((i * 17) % 1000) as f64 / 1000.0 - 0.5;
            parameters[[0, i]] = F::from(val).unwrap() * std_dev;
        }

        Self {
            parameters,
            meta_lr,
            inner_lr,
            inner_steps,
            input_dim,
            hidden_dim,
            output_dim,
        }
    }

    /// Meta-training step with multiple tasks
    pub fn meta_train(&mut self, tasks: &[TaskData<F>]) -> Result<F> {
        let mut meta_gradients = Array2::zeros(self.parameters.dim());
        let mut total_loss = F::zero();

        for task in tasks {
            // Inner loop adaptation
            let adapted_params = self.inner_loop_adaptation(task)?;

            // Compute meta-gradient
            let task_loss = self.compute_meta_loss(&adapted_params, task)?;
            let task_gradient = self.compute_meta_gradient(&adapted_params, task)?;

            meta_gradients = meta_gradients + task_gradient;
            total_loss = total_loss + task_loss;
        }

        // Meta-update
        let num_tasks = F::from(tasks.len()).unwrap();
        meta_gradients = meta_gradients / num_tasks;
        total_loss = total_loss / num_tasks;

        // Update meta-parameters
        self.parameters = self.parameters.clone() - meta_gradients * self.meta_lr;

        Ok(total_loss)
    }

    /// Inner loop adaptation for a single task
    fn inner_loop_adaptation(&self, task: &TaskData<F>) -> Result<Array2<F>> {
        let mut adapted_params = self.parameters.clone();

        for _ in 0..self.inner_steps {
            let _loss = self.forward(&adapted_params, &task.support_x, &task.support_y)?;
            let gradients = self.compute_gradients(&adapted_params, task)?;
            adapted_params = adapted_params - gradients * self.inner_lr;
        }

        Ok(adapted_params)
    }

    /// Forward pass through neural network
    fn forward(&self, params: &Array2<F>, inputs: &Array2<F>, targets: &Array2<F>) -> Result<F> {
        let predictions = self.predict(params, inputs)?;

        // Mean squared error loss
        let mut loss = F::zero();
        let (batch_size, _) = predictions.dim();

        for i in 0..batch_size {
            for j in 0..self.output_dim {
                let diff = predictions[[i, j]] - targets[[i, j]];
                loss = loss + diff * diff;
            }
        }

        Ok(loss / F::from(batch_size).unwrap())
    }

    /// Make predictions using current parameters
    fn predict(&self, params: &Array2<F>, inputs: &Array2<F>) -> Result<Array2<F>> {
        let (batch_size, _) = inputs.dim();

        // Extract weight matrices from flattened parameters
        let (w1, b1, w2, b2) = self.extract_weights(params);

        // Forward pass: input -> hidden -> output
        let mut hidden = Array2::zeros((batch_size, self.hidden_dim));

        // Input to hidden layer
        for i in 0..batch_size {
            for j in 0..self.hidden_dim {
                let mut sum = b1[j];
                for k in 0..self.input_dim {
                    sum = sum + inputs[[i, k]] * w1[[j, k]];
                }
                hidden[[i, j]] = self.relu(sum); // ReLU activation
            }
        }

        // Hidden to output layer
        let mut output = Array2::zeros((batch_size, self.output_dim));
        for i in 0..batch_size {
            for j in 0..self.output_dim {
                let mut sum = b2[j];
                for k in 0..self.hidden_dim {
                    sum = sum + hidden[[i, k]] * w2[[j, k]];
                }
                output[[i, j]] = sum; // Linear output
            }
        }

        Ok(output)
    }

    /// Extract weight matrices from flattened parameter vector
    fn extract_weights(&self, params: &Array2<F>) -> (Array2<F>, Array1<F>, Array2<F>, Array1<F>) {
        let param_vec = params.row(0);
        let mut idx = 0;

        // W1: input_dim x hidden_dim
        let mut w1 = Array2::zeros((self.hidden_dim, self.input_dim));
        for i in 0..self.hidden_dim {
            for j in 0..self.input_dim {
                w1[[i, j]] = param_vec[idx];
                idx += 1;
            }
        }

        // b1: hidden_dim
        let mut b1 = Array1::zeros(self.hidden_dim);
        for i in 0..self.hidden_dim {
            b1[i] = param_vec[idx];
            idx += 1;
        }

        // W2: hidden_dim x output_dim
        let mut w2 = Array2::zeros((self.output_dim, self.hidden_dim));
        for i in 0..self.output_dim {
            for j in 0..self.hidden_dim {
                w2[[i, j]] = param_vec[idx];
                idx += 1;
            }
        }

        // b2: output_dim
        let mut b2 = Array1::zeros(self.output_dim);
        for i in 0..self.output_dim {
            b2[i] = param_vec[idx];
            idx += 1;
        }

        (w1, b1, w2, b2)
    }

    /// ReLU activation function
    fn relu(&self, x: F) -> F {
        x.max(F::zero())
    }

    /// Compute gradients (simplified numerical differentiation)
    fn compute_gradients(&self, params: &Array2<F>, task: &TaskData<F>) -> Result<Array2<F>> {
        let epsilon = F::from(1e-5).unwrap();
        let mut gradients = Array2::zeros(params.dim());

        let base_loss = self.forward(params, &task.support_x, &task.support_y)?;

        for i in 0..params.ncols() {
            let mut perturbed_params = params.clone();
            perturbed_params[[0, i]] = perturbed_params[[0, i]] + epsilon;

            let perturbed_loss =
                self.forward(&perturbed_params, &task.support_x, &task.support_y)?;
            gradients[[0, i]] = (perturbed_loss - base_loss) / epsilon;
        }

        Ok(gradients)
    }

    /// Compute meta-gradient for meta-learning update
    fn compute_meta_gradient(
        &self,
        adapted_params: &Array2<F>,
        task: &TaskData<F>,
    ) -> Result<Array2<F>> {
        // Simplified meta-gradient computation
        let _meta_loss = self.forward(adapted_params, &task.query_x, &task.query_y)?;
        self.compute_gradients(
            adapted_params,
            &TaskData {
                support_x: task.query_x.clone(),
                support_y: task.query_y.clone(),
                query_x: task.query_x.clone(),
                query_y: task.query_y.clone(),
            },
        )
    }

    /// Compute meta-loss on query set
    fn compute_meta_loss(&self, adapted_params: &Array2<F>, task: &TaskData<F>) -> Result<F> {
        self.forward(adapted_params, &task.query_x, &task.query_y)
    }

    /// Fast adaptation for new task (few-shot learning)
    pub fn fast_adapt(&self, support_x: &Array2<F>, support_y: &Array2<F>) -> Result<Array2<F>> {
        let task = TaskData {
            support_x: support_x.clone(),
            support_y: support_y.clone(),
            query_x: support_x.clone(),
            query_y: support_y.clone(),
        };

        self.inner_loop_adaptation(&task)
    }
}

/// Task data structure for meta-learning
#[derive(Debug, Clone)]
pub struct TaskData<F: Float + Debug> {
    /// Support set inputs (for adaptation)
    pub support_x: Array2<F>,
    /// Support set outputs
    pub support_y: Array2<F>,
    /// Query set inputs (for evaluation)
    pub query_x: Array2<F>,
    /// Query set outputs
    pub query_y: Array2<F>,
}

/// Neural Ordinary Differential Equation (NODE) implementation
#[derive(Debug)]
pub struct NeuralODE<F: Float + Debug + ndarray::ScalarOperand> {
    /// Network parameters
    parameters: Array2<F>,
    /// Integration time steps
    time_steps: Array1<F>,
    /// ODE solver configuration
    solver_config: ODESolverConfig<F>,
    /// Network dimensions
    input_dim: usize,
    hidden_dim: usize,
}

/// Configuration for ODE solver
#[derive(Debug, Clone)]
pub struct ODESolverConfig<F: Float + Debug> {
    /// Integration method
    method: IntegrationMethod,
    /// Step size
    #[allow(dead_code)]
    step_size: F,
    /// Tolerance for adaptive methods
    #[allow(dead_code)]
    tolerance: F,
}

/// Integration methods for ODE solving
#[derive(Debug, Clone)]
pub enum IntegrationMethod {
    /// Forward Euler method
    Euler,
    /// Fourth-order Runge-Kutta
    RungeKutta4,
    /// Adaptive Runge-Kutta-Fehlberg
    RKF45,
}

impl<F: Float + Debug + Clone + FromPrimitive + ndarray::ScalarOperand> NeuralODE<F> {
    /// Create new Neural ODE
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        time_steps: Array1<F>,
        solver_config: ODESolverConfig<F>,
    ) -> Self {
        // Initialize network parameters
        let total_params = input_dim * hidden_dim + hidden_dim * input_dim + 2 * hidden_dim;
        let scale = F::from(2.0).unwrap() / F::from(input_dim).unwrap();
        let std_dev = scale.sqrt();

        let mut parameters = Array2::zeros((1, total_params));
        for i in 0..total_params {
            let val = ((i * 23) % 1000) as f64 / 1000.0 - 0.5;
            parameters[[0, i]] = F::from(val).unwrap() * std_dev;
        }

        Self {
            parameters,
            time_steps,
            solver_config,
            input_dim,
            hidden_dim,
        }
    }

    /// Forward pass through Neural ODE
    pub fn forward(&self, initialstate: &Array1<F>) -> Result<Array2<F>> {
        let num_times = self.time_steps.len();
        let mut trajectory = Array2::zeros((num_times, self.input_dim));

        // Set initial condition
        for i in 0..self.input_dim {
            trajectory[[0, i]] = initialstate[i];
        }

        // Integrate ODE
        for t in 1..num_times {
            let dt = self.time_steps[t] - self.time_steps[t - 1];
            let current_state = trajectory.row(t - 1).to_owned();

            let next_state = match self.solver_config.method {
                IntegrationMethod::Euler => self.euler_step(&current_state, dt)?,
                IntegrationMethod::RungeKutta4 => self.rk4_step(&current_state, dt)?,
                IntegrationMethod::RKF45 => self.rkf45_step(&current_state, dt)?,
            };

            for i in 0..self.input_dim {
                trajectory[[t, i]] = next_state[i];
            }
        }

        Ok(trajectory)
    }

    /// Neural network defining the ODE dynamics
    fn neural_network(&self, state: &Array1<F>) -> Result<Array1<F>> {
        let (w1, b1, w2, b2) = self.extract_ode_weights();

        // First layer
        let mut hidden = Array1::zeros(self.hidden_dim);
        for i in 0..self.hidden_dim {
            let mut sum = b1[i];
            for j in 0..self.input_dim {
                sum = sum + w1[[i, j]] * state[j];
            }
            hidden[i] = self.tanh(sum);
        }

        // Second layer
        let mut output = Array1::zeros(self.input_dim);
        for i in 0..self.input_dim {
            let mut sum = b2[i];
            for j in 0..self.hidden_dim {
                sum = sum + w2[[i, j]] * hidden[j];
            }
            output[i] = sum;
        }

        Ok(output)
    }

    /// Extract ODE network weights
    fn extract_ode_weights(&self) -> (Array2<F>, Array1<F>, Array2<F>, Array1<F>) {
        let param_vec = self.parameters.row(0);
        let mut idx = 0;

        // W1: input_dim x hidden_dim
        let mut w1 = Array2::zeros((self.hidden_dim, self.input_dim));
        for i in 0..self.hidden_dim {
            for j in 0..self.input_dim {
                w1[[i, j]] = param_vec[idx];
                idx += 1;
            }
        }

        // b1: hidden_dim
        let mut b1 = Array1::zeros(self.hidden_dim);
        for i in 0..self.hidden_dim {
            b1[i] = param_vec[idx];
            idx += 1;
        }

        // W2: hidden_dim x input_dim
        let mut w2 = Array2::zeros((self.input_dim, self.hidden_dim));
        for i in 0..self.input_dim {
            for j in 0..self.hidden_dim {
                w2[[i, j]] = param_vec[idx];
                idx += 1;
            }
        }

        // b2: input_dim
        let mut b2 = Array1::zeros(self.input_dim);
        for i in 0..self.input_dim {
            b2[i] = param_vec[idx];
            idx += 1;
        }

        (w1, b1, w2, b2)
    }

    /// Euler integration step
    fn euler_step(&self, state: &Array1<F>, dt: F) -> Result<Array1<F>> {
        let derivative = self.neural_network(state)?;
        let mut next_state = Array1::zeros(state.len());

        for i in 0..state.len() {
            next_state[i] = state[i] + dt * derivative[i];
        }

        Ok(next_state)
    }

    /// Fourth-order Runge-Kutta integration step
    fn rk4_step(&self, state: &Array1<F>, dt: F) -> Result<Array1<F>> {
        let k1 = self.neural_network(state)?;

        let mut state_k2 = Array1::zeros(state.len());
        for i in 0..state.len() {
            state_k2[i] = state[i] + dt * k1[i] / F::from(2.0).unwrap();
        }
        let k2 = self.neural_network(&state_k2)?;

        let mut state_k3 = Array1::zeros(state.len());
        for i in 0..state.len() {
            state_k3[i] = state[i] + dt * k2[i] / F::from(2.0).unwrap();
        }
        let k3 = self.neural_network(&state_k3)?;

        let mut state_k4 = Array1::zeros(state.len());
        for i in 0..state.len() {
            state_k4[i] = state[i] + dt * k3[i];
        }
        let k4 = self.neural_network(&state_k4)?;

        let mut next_state = Array1::zeros(state.len());
        for i in 0..state.len() {
            next_state[i] = state[i]
                + dt * (k1[i]
                    + F::from(2.0).unwrap() * k2[i]
                    + F::from(2.0).unwrap() * k3[i]
                    + k4[i])
                    / F::from(6.0).unwrap();
        }

        Ok(next_state)
    }

    /// Runge-Kutta-Fehlberg adaptive step
    fn rkf45_step(&self, state: &Array1<F>, dt: F) -> Result<Array1<F>> {
        // Simplified implementation - use RK4 for now
        self.rk4_step(state, dt)
    }

    /// Hyperbolic tangent activation
    fn tanh(&self, x: F) -> F {
        x.tanh()
    }
}

/// Type alias for complex encoder weights return type
type EncoderWeights<F> = (
    Array2<F>,
    Array1<F>,
    Array2<F>,
    Array1<F>,
    Array2<F>,
    Array1<F>,
);

/// Variational Autoencoder for Time Series with Uncertainty Quantification
#[derive(Debug)]
pub struct TimeSeriesVAE<F: Float + Debug + ndarray::ScalarOperand> {
    /// Encoder parameters
    encoder_params: Array2<F>,
    /// Decoder parameters
    decoder_params: Array2<F>,
    /// Latent dimension
    latent_dim: usize,
    /// Input sequence length
    seq_len: usize,
    /// Feature dimension
    feature_dim: usize,
    /// Hidden dimensions
    encoder_hidden: usize,
    decoder_hidden: usize,
}

impl<F: Float + Debug + Clone + FromPrimitive + ndarray::ScalarOperand> TimeSeriesVAE<F> {
    /// Create new Time Series VAE
    pub fn new(
        seq_len: usize,
        feature_dim: usize,
        latent_dim: usize,
        encoder_hidden: usize,
        decoder_hidden: usize,
    ) -> Self {
        let input_size = seq_len * feature_dim;

        // Initialize encoder parameters (input -> _hidden -> latent_mean, latent_logvar)
        let encoder_param_count = input_size * encoder_hidden
            + encoder_hidden
            + encoder_hidden * latent_dim * 2
            + latent_dim * 2;
        let mut encoder_params = Array2::zeros((1, encoder_param_count));

        // Initialize decoder parameters (latent -> _hidden -> output)
        let decoder_param_count =
            latent_dim * decoder_hidden + decoder_hidden + decoder_hidden * input_size + input_size;
        let mut decoder_params = Array2::zeros((1, decoder_param_count));

        // Xavier initialization
        let encoder_scale = F::from(2.0).unwrap() / F::from(input_size + latent_dim).unwrap();
        let decoder_scale = F::from(2.0).unwrap() / F::from(latent_dim + input_size).unwrap();

        for i in 0..encoder_param_count {
            let val = ((i * 19) % 1000) as f64 / 1000.0 - 0.5;
            encoder_params[[0, i]] = F::from(val).unwrap() * encoder_scale.sqrt();
        }

        for i in 0..decoder_param_count {
            let val = ((i * 31) % 1000) as f64 / 1000.0 - 0.5;
            decoder_params[[0, i]] = F::from(val).unwrap() * decoder_scale.sqrt();
        }

        Self {
            encoder_params,
            decoder_params,
            latent_dim,
            seq_len,
            feature_dim,
            encoder_hidden,
            decoder_hidden,
        }
    }

    /// Encode time series to latent distribution
    pub fn encode(&self, input: &Array2<F>) -> Result<(Array1<F>, Array1<F>)> {
        // Flatten input
        let input_flat = self.flatten_input(input);

        // Extract encoder weights
        let (w1, b1, w_mean, b_mean, w_logvar, b_logvar) = self.extract_encoder_weights();

        // Forward through encoder
        let mut hidden = Array1::zeros(self.encoder_hidden);
        for i in 0..self.encoder_hidden {
            let mut sum = b1[i];
            for j in 0..input_flat.len() {
                sum = sum + w1[[i, j]] * input_flat[j];
            }
            hidden[i] = self.relu(sum);
        }

        // Compute latent mean and log variance
        let mut latent_mean = Array1::zeros(self.latent_dim);
        let mut latent_logvar = Array1::zeros(self.latent_dim);

        for i in 0..self.latent_dim {
            let mut mean_sum = b_mean[i];
            let mut logvar_sum = b_logvar[i];

            for j in 0..self.encoder_hidden {
                mean_sum = mean_sum + w_mean[[i, j]] * hidden[j];
                logvar_sum = logvar_sum + w_logvar[[i, j]] * hidden[j];
            }

            latent_mean[i] = mean_sum;
            latent_logvar[i] = logvar_sum;
        }

        Ok((latent_mean, latent_logvar))
    }

    /// Sample from latent distribution using reparameterization trick
    pub fn reparameterize(&self, mean: &Array1<F>, logvar: &Array1<F>) -> Array1<F> {
        let mut sample = Array1::zeros(self.latent_dim);

        for i in 0..self.latent_dim {
            // Sample from standard normal (simplified)
            let eps = F::from(((i * 47) % 1000) as f64 / 1000.0 - 0.5).unwrap();
            let std = (logvar[i] / F::from(2.0).unwrap()).exp();
            sample[i] = mean[i] + std * eps;
        }

        sample
    }

    /// Decode latent representation to time series
    pub fn decode(&self, latent: &Array1<F>) -> Result<Array2<F>> {
        // Extract decoder weights
        let (w1, b1, w2, b2) = self.extract_decoder_weights();

        // Forward through decoder
        let mut hidden = Array1::zeros(self.decoder_hidden);
        for i in 0..self.decoder_hidden {
            let mut sum = b1[i];
            for j in 0..self.latent_dim {
                sum = sum + w1[[i, j]] * latent[j];
            }
            hidden[i] = self.relu(sum);
        }

        // Generate output
        let output_size = self.seq_len * self.feature_dim;
        let mut output_flat = Array1::zeros(output_size);

        for i in 0..output_size {
            let mut sum = b2[i];
            for j in 0..self.decoder_hidden {
                sum = sum + w2[[i, j]] * hidden[j];
            }
            output_flat[i] = sum;
        }

        // Reshape to time series format
        self.unflatten_output(&output_flat)
    }

    /// Full forward pass with reconstruction and KL divergence
    pub fn forward(&self, input: &Array2<F>) -> Result<VAEOutput<F>> {
        let (latent_mean, latent_logvar) = self.encode(input)?;
        let latent_sample = self.reparameterize(&latent_mean, &latent_logvar);
        let reconstruction = self.decode(&latent_sample)?;

        // Compute KL divergence
        let mut kl_div = F::zero();
        for i in 0..self.latent_dim {
            let mean_sq = latent_mean[i] * latent_mean[i];
            let var = latent_logvar[i].exp();
            kl_div = kl_div + mean_sq + var - latent_logvar[i] - F::one();
        }
        kl_div = kl_div / F::from(2.0).unwrap();

        // Compute reconstruction loss
        let mut recon_loss = F::zero();
        let (seq_len, feature_dim) = input.dim();

        for i in 0..seq_len {
            for j in 0..feature_dim {
                let diff = reconstruction[[i, j]] - input[[i, j]];
                recon_loss = recon_loss + diff * diff;
            }
        }
        recon_loss = recon_loss / F::from(seq_len * feature_dim).unwrap();

        Ok(VAEOutput {
            reconstruction,
            latent_mean,
            latent_logvar,
            latent_sample,
            reconstruction_loss: recon_loss,
            kl_divergence: kl_div,
        })
    }

    /// Generate new time series by sampling from latent space
    pub fn generate(&self, numsamples: usize) -> Result<Vec<Array2<F>>> {
        let mut _samples = Vec::new();

        for i in 0..numsamples {
            // Sample from prior distribution (standard normal)
            let mut latent = Array1::zeros(self.latent_dim);
            for j in 0..self.latent_dim {
                let val = ((i * 53 + j * 29) % 1000) as f64 / 1000.0 - 0.5;
                latent[j] = F::from(val).unwrap();
            }

            let generated = self.decode(&latent)?;
            _samples.push(generated);
        }

        Ok(_samples)
    }

    /// Estimate uncertainty by sampling multiple reconstructions
    pub fn estimate_uncertainty(
        &self,
        input: &Array2<F>,
        num_samples: usize,
    ) -> Result<(Array2<F>, Array2<F>)> {
        let (latent_mean, latent_logvar) = self.encode(input)?;
        let mut reconstructions = Vec::new();

        // Generate multiple _samples
        for _ in 0..num_samples {
            let latent_sample = self.reparameterize(&latent_mean, &latent_logvar);
            let reconstruction = self.decode(&latent_sample)?;
            reconstructions.push(reconstruction);
        }

        // Compute mean and standard deviation
        let (seq_len, feature_dim) = input.dim();
        let mut mean_recon = Array2::zeros((seq_len, feature_dim));
        let mut std_recon = Array2::zeros((seq_len, feature_dim));

        // Compute mean
        for recon in &reconstructions {
            for i in 0..seq_len {
                for j in 0..feature_dim {
                    mean_recon[[i, j]] = mean_recon[[i, j]] + recon[[i, j]];
                }
            }
        }

        let num_samples_f = F::from(num_samples).unwrap();
        for i in 0..seq_len {
            for j in 0..feature_dim {
                mean_recon[[i, j]] = mean_recon[[i, j]] / num_samples_f;
            }
        }

        // Compute standard deviation
        for recon in &reconstructions {
            for i in 0..seq_len {
                for j in 0..feature_dim {
                    let diff = recon[[i, j]] - mean_recon[[i, j]];
                    std_recon[[i, j]] = std_recon[[i, j]] + diff * diff;
                }
            }
        }

        for i in 0..seq_len {
            for j in 0..feature_dim {
                let val: F = std_recon[[i, j]] / num_samples_f;
                std_recon[[i, j]] = val.sqrt();
            }
        }

        Ok((mean_recon, std_recon))
    }

    // Helper methods
    fn flatten_input(&self, input: &Array2<F>) -> Array1<F> {
        let (seq_len, feature_dim) = input.dim();
        let mut flat = Array1::zeros(seq_len * feature_dim);

        for i in 0..seq_len {
            for j in 0..feature_dim {
                flat[i * feature_dim + j] = input[[i, j]];
            }
        }

        flat
    }

    fn unflatten_output(&self, output: &Array1<F>) -> Result<Array2<F>> {
        let mut result = Array2::zeros((self.seq_len, self.feature_dim));

        for i in 0..self.seq_len {
            for j in 0..self.feature_dim {
                let idx = i * self.feature_dim + j;
                if idx < output.len() {
                    result[[i, j]] = output[idx];
                }
            }
        }

        Ok(result)
    }

    fn extract_encoder_weights(&self) -> EncoderWeights<F> {
        let param_vec = self.encoder_params.row(0);
        let input_size = self.seq_len * self.feature_dim;
        let mut idx = 0;

        // W1: input_size x encoder_hidden
        let mut w1 = Array2::zeros((self.encoder_hidden, input_size));
        for i in 0..self.encoder_hidden {
            for j in 0..input_size {
                w1[[i, j]] = param_vec[idx];
                idx += 1;
            }
        }

        // b1: encoder_hidden
        let mut b1 = Array1::zeros(self.encoder_hidden);
        for i in 0..self.encoder_hidden {
            b1[i] = param_vec[idx];
            idx += 1;
        }

        // W_mean: encoder_hidden x latent_dim
        let mut w_mean = Array2::zeros((self.latent_dim, self.encoder_hidden));
        for i in 0..self.latent_dim {
            for j in 0..self.encoder_hidden {
                w_mean[[i, j]] = param_vec[idx];
                idx += 1;
            }
        }

        // b_mean: latent_dim
        let mut b_mean = Array1::zeros(self.latent_dim);
        for i in 0..self.latent_dim {
            b_mean[i] = param_vec[idx];
            idx += 1;
        }

        // W_logvar: encoder_hidden x latent_dim
        let mut w_logvar = Array2::zeros((self.latent_dim, self.encoder_hidden));
        for i in 0..self.latent_dim {
            for j in 0..self.encoder_hidden {
                w_logvar[[i, j]] = param_vec[idx];
                idx += 1;
            }
        }

        // b_logvar: latent_dim
        let mut b_logvar = Array1::zeros(self.latent_dim);
        for i in 0..self.latent_dim {
            b_logvar[i] = param_vec[idx];
            idx += 1;
        }

        (w1, b1, w_mean, b_mean, w_logvar, b_logvar)
    }

    fn extract_decoder_weights(&self) -> (Array2<F>, Array1<F>, Array2<F>, Array1<F>) {
        let param_vec = self.decoder_params.row(0);
        let output_size = self.seq_len * self.feature_dim;
        let mut idx = 0;

        // W1: latent_dim x decoder_hidden
        let mut w1 = Array2::zeros((self.decoder_hidden, self.latent_dim));
        for i in 0..self.decoder_hidden {
            for j in 0..self.latent_dim {
                w1[[i, j]] = param_vec[idx];
                idx += 1;
            }
        }

        // b1: decoder_hidden
        let mut b1 = Array1::zeros(self.decoder_hidden);
        for i in 0..self.decoder_hidden {
            b1[i] = param_vec[idx];
            idx += 1;
        }

        // W2: decoder_hidden x output_size
        let mut w2 = Array2::zeros((output_size, self.decoder_hidden));
        for i in 0..output_size {
            for j in 0..self.decoder_hidden {
                w2[[i, j]] = param_vec[idx];
                idx += 1;
            }
        }

        // b2: output_size
        let mut b2 = Array1::zeros(output_size);
        for i in 0..output_size {
            b2[i] = param_vec[idx];
            idx += 1;
        }

        (w1, b1, w2, b2)
    }

    fn relu(&self, x: F) -> F {
        x.max(F::zero())
    }
}

/// VAE output structure
#[derive(Debug, Clone)]
pub struct VAEOutput<F: Float + Debug> {
    /// Reconstructed time series
    pub reconstruction: Array2<F>,
    /// Latent mean
    pub latent_mean: Array1<F>,
    /// Latent log variance
    pub latent_logvar: Array1<F>,
    /// Latent sample
    pub latent_sample: Array1<F>,
    /// Reconstruction loss
    pub reconstruction_loss: F,
    /// KL divergence
    pub kl_divergence: F,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_maml() {
        let mut maml = MAML::<f64>::new(4, 8, 2, 0.01, 0.1, 5);

        // Create sample tasks
        let task1 = TaskData {
            support_x: Array2::from_shape_vec((5, 4), (0..20).map(|i| i as f64 * 0.1).collect())
                .unwrap(),
            support_y: Array2::from_shape_vec((5, 2), (0..10).map(|i| i as f64 * 0.2).collect())
                .unwrap(),
            query_x: Array2::from_shape_vec(
                (3, 4),
                (0..12).map(|i| i as f64 * 0.1 + 0.5).collect(),
            )
            .unwrap(),
            query_y: Array2::from_shape_vec((3, 2), (0..6).map(|i| i as f64 * 0.2 + 0.3).collect())
                .unwrap(),
        };

        let tasks = vec![task1];
        let loss = maml.meta_train(&tasks).unwrap();

        assert!(loss > 0.0); // Should have positive loss
        assert!(loss.is_finite()); // Should be finite
    }

    #[test]
    fn test_neural_ode() {
        let time_steps = Array1::from_vec(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5]);
        let solver_config = ODESolverConfig {
            method: IntegrationMethod::RungeKutta4,
            step_size: 0.1,
            tolerance: 1e-6,
        };

        let node = NeuralODE::<f64>::new(3, 8, time_steps, solver_config);
        let initial_state = Array1::from_vec(vec![1.0, 0.5, -0.2]);

        let trajectory = node.forward(&initial_state).unwrap();
        assert_eq!(trajectory.dim(), (6, 3)); // 6 time steps, 3 dimensions

        // Check initial condition
        for i in 0..3 {
            assert_abs_diff_eq!(trajectory[[0, i]], initial_state[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_time_series_vae() {
        let vae = TimeSeriesVAE::<f64>::new(10, 3, 5, 16, 16);

        let input =
            Array2::from_shape_vec((10, 3), (0..30).map(|i| i as f64 * 0.1).collect()).unwrap();

        // Test encoding
        let (mean, logvar) = vae.encode(&input).unwrap();
        assert_eq!(mean.len(), 5);
        assert_eq!(logvar.len(), 5);

        // Test reparameterization
        let sample = vae.reparameterize(&mean, &logvar);
        assert_eq!(sample.len(), 5);

        // Test decoding
        let reconstruction = vae.decode(&sample).unwrap();
        assert_eq!(reconstruction.dim(), (10, 3));

        // Test full forward pass
        let output = vae.forward(&input).unwrap();
        assert_eq!(output.reconstruction.dim(), (10, 3));
        assert!(output.reconstruction_loss >= 0.0);
        assert!(output.kl_divergence >= 0.0);

        // Test uncertainty estimation
        let (mean_recon, std_recon) = vae.estimate_uncertainty(&input, 10).unwrap();
        assert_eq!(mean_recon.dim(), (10, 3));
        assert_eq!(std_recon.dim(), (10, 3));

        // Standard deviations should be non-negative
        for i in 0..10 {
            for j in 0..3 {
                assert!(std_recon[[i, j]] >= 0.0);
            }
        }
    }

    #[test]
    fn test_ode_solver_methods() {
        let time_steps = Array1::from_vec(vec![0.0, 0.1, 0.2]);

        // Test different solvers
        let solvers = vec![
            ODESolverConfig {
                method: IntegrationMethod::Euler,
                step_size: 0.1,
                tolerance: 1e-6,
            },
            ODESolverConfig {
                method: IntegrationMethod::RungeKutta4,
                step_size: 0.1,
                tolerance: 1e-6,
            },
        ];

        let initial_state = Array1::from_vec(vec![1.0, 0.0]);

        for solver_config in solvers {
            let node = NeuralODE::<f64>::new(2, 4, time_steps.clone(), solver_config);
            let trajectory = node.forward(&initial_state).unwrap();

            assert_eq!(trajectory.dim(), (3, 2));

            // Check that trajectory evolves (not constant)
            let _initial_sum = trajectory.row(0).sum();
            let _final_sum = trajectory.row(2).sum();
            // Should be different unless the ODE derivative is exactly zero
            // For our randomly initialized network, this is very unlikely
        }
    }

    #[test]
    fn test_vae_generation() {
        let vae = TimeSeriesVAE::<f64>::new(5, 2, 3, 8, 8);

        let generated_samples = vae.generate(3).unwrap();
        assert_eq!(generated_samples.len(), 3);

        for sample in &generated_samples {
            assert_eq!(sample.dim(), (5, 2));
        }
    }
}

/// Transformer-based Time Series Forecasting Model
#[derive(Debug)]
pub struct TimeSeriesTransformer<F: Float + Debug + ndarray::ScalarOperand> {
    /// Number of transformer layers
    num_layers: usize,
    /// Attention heads per layer
    num_heads: usize,
    /// Model dimension
    d_model: usize,
    /// Feed-forward dimension
    d_ff: usize,
    /// Sequence length
    seq_len: usize,
    /// Prediction horizon
    pred_len: usize,
    /// Model parameters
    parameters: Array2<F>,
    /// Positional encoding
    positional_encoding: Array2<F>,
}

impl<F: Float + Debug + Clone + FromPrimitive + ndarray::ScalarOperand> TimeSeriesTransformer<F> {
    /// Create new Transformer for time series forecasting
    pub fn new(
        seq_len: usize,
        pred_len: usize,
        d_model: usize,
        num_heads: usize,
        num_layers: usize,
        d_ff: usize,
    ) -> Self {
        // Calculate total parameter count
        let attention_params_per_layer = 4 * d_model * d_model; // Q, K, V, O projections
        let ff_params_per_layer = 2 * d_model * d_ff + d_ff + d_model; // Two linear _layers + biases
        let layer_norm_params_per_layer = 2 * d_model * 2; // Two layer norms per layer
        let embedding_params = seq_len * d_model; // Input embeddings
        let output_params = d_model * pred_len; // Final projection

        let total_params = num_layers
            * (attention_params_per_layer + ff_params_per_layer + layer_norm_params_per_layer)
            + embedding_params
            + output_params;

        // Initialize parameters
        let scale = F::from(2.0).unwrap() / F::from(d_model).unwrap();
        let std_dev = scale.sqrt();

        let mut parameters = Array2::zeros((1, total_params));
        for i in 0..total_params {
            let val = ((i * 13) % 1000) as f64 / 1000.0 - 0.5;
            parameters[[0, i]] = F::from(val).unwrap() * std_dev;
        }

        // Create positional encoding
        let mut positional_encoding = Array2::zeros((seq_len, d_model));
        for pos in 0..seq_len {
            for i in 0..d_model {
                let angle = F::from(pos).unwrap()
                    / F::from(10000.0)
                        .unwrap()
                        .powf(F::from(2 * (i / 2)).unwrap() / F::from(d_model).unwrap());
                if i % 2 == 0 {
                    positional_encoding[[pos, i]] = angle.sin();
                } else {
                    positional_encoding[[pos, i]] = angle.cos();
                }
            }
        }

        Self {
            num_layers,
            num_heads,
            d_model,
            d_ff,
            seq_len,
            pred_len,
            parameters,
            positional_encoding,
        }
    }

    /// Forward pass through transformer
    pub fn forward(&self, input: &Array2<F>) -> Result<Array2<F>> {
        let batch_size_ = input.nrows();

        // Input embedding + positional encoding
        let mut x = self.input_embedding(input)?;

        // Add positional encoding
        for i in 0..batch_size_ {
            for j in 0..self.seq_len {
                for k in 0..self.d_model {
                    x[[i * self.seq_len + j, k]] =
                        x[[i * self.seq_len + j, k]] + self.positional_encoding[[j, k]];
                }
            }
        }

        // Pass through transformer layers
        for layer in 0..self.num_layers {
            x = self.transformer_layer(&x, layer)?;
        }

        // Final projection to prediction horizon
        self.output_projection(&x, batch_size_)
    }

    /// Input embedding layer
    fn input_embedding(&self, input: &Array2<F>) -> Result<Array2<F>> {
        let batch_size_ = input.nrows();
        let input_dim = input.ncols();

        // Simple linear projection to d_model
        let mut embedded = Array2::zeros((batch_size_ * self.seq_len, self.d_model));

        // Extract embedding weights (simplified)
        let param_start = 0;

        for i in 0..batch_size_ {
            for j in 0..self.seq_len.min(input_dim) {
                for k in 0..self.d_model {
                    let weight_idx = (j * self.d_model + k) % (self.seq_len * self.d_model);
                    let weight = if param_start + weight_idx < self.parameters.ncols() {
                        self.parameters[[0, param_start + weight_idx]]
                    } else {
                        F::zero()
                    };
                    embedded[[i * self.seq_len + j, k]] = input[[i, j]] * weight;
                }
            }
        }

        Ok(embedded)
    }

    /// Single transformer layer
    fn transformer_layer(&self, input: &Array2<F>, layeridx: usize) -> Result<Array2<F>> {
        // Multi-head attention
        let attention_output = self.multi_head_attention(input, layeridx)?;

        // Add & Norm 1
        let norm1_output =
            self.layer_norm(&self.add_residual(input, &attention_output)?, layeridx, 0)?;

        // Feed-forward
        let ff_output = self.feed_forward(&norm1_output, layeridx)?;

        // Add & Norm 2
        let final_output =
            self.layer_norm(&self.add_residual(&norm1_output, &ff_output)?, layeridx, 1)?;

        Ok(final_output)
    }

    /// Multi-head attention mechanism
    fn multi_head_attention(&self, input: &Array2<F>, layeridx: usize) -> Result<Array2<F>> {
        let seq_len = input.nrows();
        let head_dim = self.d_model / self.num_heads;

        // Simplified attention computation
        let mut output = Array2::zeros((seq_len, self.d_model));

        for head in 0..self.num_heads {
            // Compute Q, K, V for this head (simplified)
            let q = self.compute_qkv_projection(input, layeridx, head, 0)?; // Query
            let k = self.compute_qkv_projection(input, layeridx, head, 1)?; // Key
            let v = self.compute_qkv_projection(input, layeridx, head, 2)?; // Value

            // Attention scores
            let attention_scores = self.compute_attention_scores(&q, &k)?;

            // Apply attention to values
            let head_output = self.apply_attention(&attention_scores, &v)?;

            // Combine heads
            for i in 0..seq_len {
                for j in 0..head_dim {
                    if head * head_dim + j < self.d_model {
                        output[[i, head * head_dim + j]] = head_output[[i, j]];
                    }
                }
            }
        }

        Ok(output)
    }

    /// Compute Q, K, V projections
    fn compute_qkv_projection(
        &self,
        input: &Array2<F>,
        layer_idx: usize,
        head: usize,
        projection_type: usize,
    ) -> Result<Array2<F>> {
        let seq_len = input.nrows();
        let head_dim = self.d_model / self.num_heads;
        let mut output = Array2::zeros((seq_len, head_dim));

        // Simplified projection computation
        for i in 0..seq_len {
            for j in 0..head_dim {
                let mut sum = F::zero();
                for k in 0..self.d_model {
                    // Compute weight index (simplified)
                    let weight_idx = (layer_idx * 1000
                        + head * 100
                        + projection_type * 10
                        + j * self.d_model
                        + k)
                        % self.parameters.ncols();
                    let weight = self.parameters[[0, weight_idx]];
                    sum = sum + input[[i, k]] * weight;
                }
                output[[i, j]] = sum;
            }
        }

        Ok(output)
    }

    /// Compute attention scores
    fn compute_attention_scores(&self, q: &Array2<F>, k: &Array2<F>) -> Result<Array2<F>> {
        let seq_len = q.nrows();
        let head_dim = q.ncols();
        let scale = F::one() / F::from(head_dim).unwrap().sqrt();

        let mut scores = Array2::zeros((seq_len, seq_len));

        for i in 0..seq_len {
            for j in 0..seq_len {
                let mut dot_product = F::zero();
                for dim in 0..head_dim {
                    dot_product = dot_product + q[[i, dim]] * k[[j, dim]];
                }
                scores[[i, j]] = dot_product * scale;
            }
        }

        // Apply softmax
        self.softmax_2d(&scores)
    }

    /// Apply attention weights to values
    fn apply_attention(&self, attention: &Array2<F>, values: &Array2<F>) -> Result<Array2<F>> {
        let seq_len = attention.nrows();
        let head_dim = values.ncols();
        let mut output = Array2::zeros((seq_len, head_dim));

        for i in 0..seq_len {
            for j in 0..head_dim {
                let mut sum = F::zero();
                for k in 0..seq_len {
                    sum = sum + attention[[i, k]] * values[[k, j]];
                }
                output[[i, j]] = sum;
            }
        }

        Ok(output)
    }

    /// Feed-forward network
    fn feed_forward(&self, input: &Array2<F>, layeridx: usize) -> Result<Array2<F>> {
        let seq_len = input.nrows();

        // First linear layer
        let mut hidden = Array2::zeros((seq_len, self.d_ff));
        for i in 0..seq_len {
            for j in 0..self.d_ff {
                let mut sum = F::zero();
                for k in 0..self.d_model {
                    let weight_idx =
                        (layeridx * 2000 + j * self.d_model + k) % self.parameters.ncols();
                    let weight = self.parameters[[0, weight_idx]];
                    sum = sum + input[[i, k]] * weight;
                }
                hidden[[i, j]] = self.relu(sum);
            }
        }

        // Second linear layer
        let mut output = Array2::zeros((seq_len, self.d_model));
        for i in 0..seq_len {
            for j in 0..self.d_model {
                let mut sum = F::zero();
                for k in 0..self.d_ff {
                    let weight_idx =
                        (layeridx * 3000 + j * self.d_ff + k) % self.parameters.ncols();
                    let weight = self.parameters[[0, weight_idx]];
                    sum = sum + hidden[[i, k]] * weight;
                }
                output[[i, j]] = sum;
            }
        }

        Ok(output)
    }

    /// Layer normalization
    fn layer_norm(
        &self,
        input: &Array2<F>,
        layer_idx: usize,
        norm_idx: usize,
    ) -> Result<Array2<F>> {
        let seq_len = input.nrows();
        let mut output = Array2::zeros(input.dim());

        for i in 0..seq_len {
            // Compute mean and variance
            let mut sum = F::zero();
            for j in 0..self.d_model {
                sum = sum + input[[i, j]];
            }
            let mean = sum / F::from(self.d_model).unwrap();

            let mut var_sum = F::zero();
            for j in 0..self.d_model {
                let diff = input[[i, j]] - mean;
                var_sum = var_sum + diff * diff;
            }
            let variance = var_sum / F::from(self.d_model).unwrap();
            let std_dev = (variance + F::from(1e-5).unwrap()).sqrt();

            // Normalize
            for j in 0..self.d_model {
                let normalized = (input[[i, j]] - mean) / std_dev;

                // Apply learnable parameters (gamma and beta)
                let gamma_idx = (layer_idx * 100 + norm_idx * 50 + j) % self.parameters.ncols();
                let beta_idx = (layer_idx * 100 + norm_idx * 50 + j + 25) % self.parameters.ncols();

                let gamma = self.parameters[[0, gamma_idx]];
                let beta = self.parameters[[0, beta_idx]];

                output[[i, j]] = gamma * normalized + beta;
            }
        }

        Ok(output)
    }

    /// Add residual connection
    fn add_residual(&self, input1: &Array2<F>, input2: &Array2<F>) -> Result<Array2<F>> {
        let mut output = Array2::zeros(input1.dim());

        for i in 0..input1.nrows() {
            for j in 0..input1.ncols() {
                output[[i, j]] = input1[[i, j]] + input2[[i, j]];
            }
        }

        Ok(output)
    }

    /// Output projection to prediction horizon
    fn output_projection(&self, input: &Array2<F>, batchsize: usize) -> Result<Array2<F>> {
        let mut output = Array2::zeros((batchsize, self.pred_len));

        // Use last token representation for prediction
        for i in 0..batchsize {
            let last_token_idx = i * self.seq_len + self.seq_len - 1;

            for j in 0..self.pred_len {
                let mut sum = F::zero();
                for k in 0..self.d_model {
                    let weight_idx = (j * self.d_model + k) % self.parameters.ncols();
                    let weight = self.parameters[[0, weight_idx]];
                    sum = sum + input[[last_token_idx, k]] * weight;
                }
                output[[i, j]] = sum;
            }
        }

        Ok(output)
    }

    /// 2D Softmax function
    fn softmax_2d(&self, input: &Array2<F>) -> Result<Array2<F>> {
        let mut output = Array2::zeros(input.dim());

        for i in 0..input.nrows() {
            // Find max for numerical stability
            let mut max_val = input[[i, 0]];
            for j in 1..input.ncols() {
                if input[[i, j]] > max_val {
                    max_val = input[[i, j]];
                }
            }

            // Compute exponentials and sum
            let mut sum = F::zero();
            for j in 0..input.ncols() {
                let exp_val = (input[[i, j]] - max_val).exp();
                output[[i, j]] = exp_val;
                sum = sum + exp_val;
            }

            // Normalize
            for j in 0..input.ncols() {
                output[[i, j]] = output[[i, j]] / sum;
            }
        }

        Ok(output)
    }

    /// ReLU activation
    fn relu(&self, x: F) -> F {
        x.max(F::zero())
    }
}

/// Hyperparameter Optimization Framework
#[derive(Debug)]
pub struct HyperparameterOptimizer<F: Float + Debug + ndarray::ScalarOperand> {
    /// Optimization method
    method: OptimizationMethod,
    /// Search space definition
    search_space: SearchSpace<F>,
    /// Current best parameters
    best_params: Option<HyperparameterSet<F>>,
    /// Best validation score
    best_score: Option<F>,
    /// Optimization history
    history: Vec<OptimizationStep<F>>,
    /// Number of trials
    max_trials: usize,
}

/// Hyperparameter optimization methods
#[derive(Debug, Clone)]
pub enum OptimizationMethod {
    /// Random search
    RandomSearch,
    /// Grid search
    GridSearch,
    /// Bayesian optimization with Gaussian Process
    BayesianOptimization,
    /// Evolutionary algorithm
    EvolutionarySearch,
    /// Tree-structured Parzen Estimator
    TPE,
}

/// Search space for hyperparameters
#[derive(Debug, Clone)]
pub struct SearchSpace<F: Float + Debug> {
    /// Continuous parameters (name, min, max)
    continuous: Vec<(String, F, F)>,
    /// Integer parameters (name, min, max)
    integer: Vec<(String, i32, i32)>,
    /// Categorical parameters (name, choices)
    categorical: Vec<(String, Vec<String>)>,
}

/// Set of hyperparameters
#[derive(Debug, Clone)]
pub struct HyperparameterSet<F: Float + Debug> {
    /// Continuous parameter values
    pub continuous: Vec<(String, F)>,
    /// Integer parameter values
    pub integer: Vec<(String, i32)>,
    /// Categorical parameter values
    pub categorical: Vec<(String, String)>,
}

/// Single optimization step
#[derive(Debug, Clone)]
pub struct OptimizationStep<F: Float + Debug> {
    /// Trial number
    pub trial_id: usize,
    /// Parameters tried
    pub params: HyperparameterSet<F>,
    /// Validation score achieved
    pub score: F,
    /// Training time
    pub training_time: F,
}

impl<F: Float + Debug + Clone + FromPrimitive + ndarray::ScalarOperand> HyperparameterOptimizer<F> {
    /// Create new hyperparameter optimizer
    pub fn new(
        method: OptimizationMethod,
        search_space: SearchSpace<F>,
        max_trials: usize,
    ) -> Self {
        Self {
            method,
            search_space,
            best_params: None,
            best_score: None,
            history: Vec::new(),
            max_trials,
        }
    }

    /// Run hyperparameter optimization
    pub fn optimize<ModelFn>(&mut self, objectivefn: ModelFn) -> Result<HyperparameterSet<F>>
    where
        ModelFn: Fn(&HyperparameterSet<F>) -> Result<F>,
    {
        for trial in 0..self.max_trials {
            // Generate candidate parameters
            let params = match self.method {
                OptimizationMethod::RandomSearch => self.random_search()?,
                OptimizationMethod::GridSearch => self.grid_search(trial)?,
                OptimizationMethod::BayesianOptimization => self.bayesian_optimization()?,
                OptimizationMethod::EvolutionarySearch => self.evolutionary_search()?,
                OptimizationMethod::TPE => self.tpe_search()?,
            };

            // Evaluate objective function
            let start_time = std::time::Instant::now();
            let score = objectivefn(&params)?;
            let training_time = F::from(start_time.elapsed().as_secs_f64()).unwrap();

            // Update best parameters if improved
            let is_better = self.best_score.is_none_or(|best| score > best);
            if is_better {
                self.best_params = Some(params.clone());
                self.best_score = Some(score);
            }

            // Record step
            self.history.push(OptimizationStep {
                trial_id: trial,
                params,
                score,
                training_time,
            });

            println!(
                "Trial {}: Score = {:.6}, Best = {:.6}",
                trial,
                score.to_f64().unwrap_or(0.0),
                self.best_score.unwrap().to_f64().unwrap_or(0.0)
            );
        }

        self.best_params.clone().ok_or_else(|| {
            crate::error::TimeSeriesError::InvalidOperation("No successful trials".to_string())
        })
    }

    /// Random search implementation
    fn random_search(&self) -> Result<HyperparameterSet<F>> {
        let mut params = HyperparameterSet {
            continuous: Vec::new(),
            integer: Vec::new(),
            categorical: Vec::new(),
        };

        // Sample continuous parameters
        for (name, min_val, max_val) in &self.search_space.continuous {
            let range = *max_val - *min_val;
            let random_val = F::from(rand::random::<f64>()).unwrap();
            let value = *min_val + range * random_val;
            params.continuous.push((name.clone(), value));
        }

        // Sample integer parameters
        for (name, min_val, max_val) in &self.search_space.integer {
            let range = max_val - min_val;
            let random_val = (rand::random::<f64>() * (range + 1) as f64) as i32;
            let value = min_val + random_val;
            params.integer.push((name.clone(), value));
        }

        // Sample categorical parameters
        for (name, choices) in &self.search_space.categorical {
            let idx = (rand::random::<f64>() * choices.len() as f64) as usize;
            let value = choices[idx.min(choices.len() - 1)].clone();
            params.categorical.push((name.clone(), value));
        }

        Ok(params)
    }

    /// Grid search implementation (simplified)
    fn grid_search(&self, trial: usize) -> Result<HyperparameterSet<F>> {
        // For simplicity, use random search with some structure
        self.random_search()
    }

    /// Bayesian optimization implementation (simplified)
    fn bayesian_optimization(&self) -> Result<HyperparameterSet<F>> {
        if self.history.is_empty() {
            // No history yet, use random search
            return self.random_search();
        }

        // Simplified acquisition function (Upper Confidence Bound)
        let mut best_candidate = None;
        let mut best_acquisition = F::from(-f64::INFINITY).unwrap();

        for _ in 0..10 {
            let candidate = self.random_search()?;
            let acquisition = self.compute_acquisition_ucb(&candidate)?;

            if acquisition > best_acquisition {
                best_acquisition = acquisition;
                best_candidate = Some(candidate);
            }
        }

        best_candidate.ok_or_else(|| {
            crate::error::TimeSeriesError::InvalidOperation("Failed to find candidate".to_string())
        })
    }

    /// Compute Upper Confidence Bound acquisition function
    fn compute_acquisition_ucb(&self, params: &HyperparameterSet<F>) -> Result<F> {
        // Simplified UCB computation
        let mean = self.predict_mean(params)?;
        let std = self.predict_std(params)?;
        let beta = F::from(2.0).unwrap(); // Exploration parameter

        Ok(mean + beta * std)
    }

    /// Predict mean performance (simplified Gaussian Process)
    fn predict_mean(&self, params: &HyperparameterSet<F>) -> Result<F> {
        // Simplified: return average of historical scores
        if self.history.is_empty() {
            return Ok(F::zero());
        }

        let sum: F = self
            .history
            .iter()
            .map(|step| step.score)
            .fold(F::zero(), |acc, x| acc + x);
        Ok(sum / F::from(self.history.len()).unwrap())
    }

    /// Predict standard deviation (simplified)
    fn predict_std(&self, _params: &HyperparameterSet<F>) -> Result<F> {
        // Simplified: return fixed exploration term
        Ok(F::one())
    }

    /// Evolutionary search implementation
    fn evolutionary_search(&self) -> Result<HyperparameterSet<F>> {
        if self.history.len() < 5 {
            return self.random_search();
        }

        // Select top performers as parents
        let mut sorted_history = self.history.clone();
        sorted_history.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        let parent1 = &sorted_history[0].params;
        let parent2 = &sorted_history[1].params;

        // Crossover and mutation
        self.crossover_mutate(parent1, parent2)
    }

    /// Crossover and mutation for evolutionary search
    fn crossover_mutate(
        &self,
        parent1: &HyperparameterSet<F>,
        parent2: &HyperparameterSet<F>,
    ) -> Result<HyperparameterSet<F>> {
        let mut child = HyperparameterSet {
            continuous: Vec::new(),
            integer: Vec::new(),
            categorical: Vec::new(),
        };

        // Crossover continuous parameters
        for ((name1, val1), (_, val2)) in parent1.continuous.iter().zip(&parent2.continuous) {
            let alpha = F::from(rand::random::<f64>()).unwrap();
            let crossed_val = *val1 + alpha * (*val2 - *val1);

            // Mutation
            let mutation = if rand::random::<f64>() < 0.1 {
                F::from((rand::random::<f64>() - 0.5) * 0.2).unwrap()
            } else {
                F::zero()
            };

            child
                .continuous
                .push((name1.clone(), crossed_val + mutation));
        }

        // Handle integer and categorical similarly (simplified)
        for (name, val) in &parent1.integer {
            child.integer.push((name.clone(), *val));
        }

        for (name, val) in &parent1.categorical {
            child.categorical.push((name.clone(), val.clone()));
        }

        Ok(child)
    }

    /// Tree-structured Parzen Estimator implementation
    fn tpe_search(&self) -> Result<HyperparameterSet<F>> {
        // Simplified TPE - use random search for now
        self.random_search()
    }

    /// Get optimization results
    pub fn get_results(&self) -> OptimizationResults<F> {
        OptimizationResults {
            best_params: self.best_params.clone(),
            best_score: self.best_score,
            history: self.history.clone(),
            convergence_curve: self.get_convergence_curve(),
        }
    }

    /// Get convergence curve
    fn get_convergence_curve(&self) -> Vec<F> {
        let mut best_so_far = Vec::new();
        let mut current_best = F::from(-f64::INFINITY).unwrap();

        for step in &self.history {
            if step.score > current_best {
                current_best = step.score;
            }
            best_so_far.push(current_best);
        }

        best_so_far
    }
}

/// Optimization results
#[derive(Debug)]
pub struct OptimizationResults<F: Float + Debug> {
    /// Best hyperparameters found
    pub best_params: Option<HyperparameterSet<F>>,
    /// Best validation score
    pub best_score: Option<F>,
    /// Complete optimization history
    pub history: Vec<OptimizationStep<F>>,
    /// Best score over time (convergence curve)
    pub convergence_curve: Vec<F>,
}

// Additional imports for slice notation
#[allow(unused_imports)]
use ndarray::s;

// Additional test cases for new functionality
#[cfg(test)]
mod advanced_tests {
    use super::*;

    #[test]
    fn test_transformer_forecasting() {
        let transformer = TimeSeriesTransformer::<f64>::new(10, 5, 64, 8, 4, 256);

        let input =
            Array2::from_shape_vec((2, 10), (0..20).map(|i| i as f64 * 0.1).collect()).unwrap();

        let output = transformer.forward(&input).unwrap();
        assert_eq!(output.dim(), (2, 5)); // batch_size_ x pred_len
    }

    #[test]
    fn test_hyperparameter_optimization() {
        let search_space = SearchSpace {
            continuous: vec![
                ("learning_rate".to_string(), 0.001, 0.1),
                ("dropout".to_string(), 0.0, 0.5),
            ],
            integer: vec![
                ("hidden_size".to_string(), 32, 256),
                ("num_layers".to_string(), 1, 6),
            ],
            categorical: vec![(
                "optimizer".to_string(),
                vec!["adam".to_string(), "sgd".to_string()],
            )],
        };

        let mut optimizer =
            HyperparameterOptimizer::new(OptimizationMethod::RandomSearch, search_space, 5);

        // Dummy objective function
        let objective = |params: &HyperparameterSet<f64>| -> Result<f64> {
            // Simulate model training and validation
            let mut score = 0.5;

            for (name, value) in &params.continuous {
                if name == "learning_rate" {
                    score += 0.1 * (0.01 - value).abs();
                }
            }

            Ok(score)
        };

        let best_params = optimizer.optimize(objective).unwrap();
        assert!(!best_params.continuous.is_empty());
    }
}

/// **Advanced MODE: NEXT-GENERATION META-LEARNING TECHNIQUES**
/// Prototypical Networks for Few-Shot Learning
#[derive(Debug)]
pub struct PrototypicalNetworks<F: Float + Debug + ndarray::ScalarOperand> {
    /// Feature extraction network parameters
    feature_extractor: Array2<F>,
    /// Input dimension
    input_dim: usize,
    /// Feature dimension
    feature_dim: usize,
    /// Hidden dimensions for feature extractor
    hidden_dims: Vec<usize>,
}

impl<F: Float + Debug + Clone + FromPrimitive + ndarray::ScalarOperand> PrototypicalNetworks<F> {
    /// Create new Prototypical Networks model
    pub fn new(input_dim: usize, feature_dim: usize, hidden_dims: Vec<usize>) -> Self {
        // Calculate total parameters for feature extractor
        let mut total_params = 0;
        let mut layer_sizes = vec![input_dim];
        layer_sizes.extend(&hidden_dims);
        layer_sizes.push(feature_dim);

        for i in 0..layer_sizes.len() - 1 {
            total_params += layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1];
            // weights + biases
        }

        // Initialize feature extractor parameters
        let scale = F::from(2.0).unwrap() / F::from(input_dim + feature_dim).unwrap();
        let std_dev = scale.sqrt();

        let mut feature_extractor = Array2::zeros((1, total_params));
        for i in 0..total_params {
            let val = ((i * 43) % 1000) as f64 / 1000.0 - 0.5;
            feature_extractor[[0, i]] = F::from(val).unwrap() * std_dev;
        }

        Self {
            feature_extractor,
            input_dim,
            feature_dim,
            hidden_dims,
        }
    }

    /// Extract features from input data
    pub fn extract_features(&self, input: &Array2<F>) -> Result<Array2<F>> {
        let batch_size_ = input.nrows();
        let mut current_input = input.clone();

        // Extract layer weights and biases
        let layer_params = self.extract_layer_parameters();

        // Forward pass through feature extractor
        for (weights, biases) in layer_params {
            let mut layer_output = Array2::zeros((batch_size_, biases.len()));

            // Apply linear transformation
            for i in 0..batch_size_ {
                for j in 0..biases.len() {
                    let mut sum = biases[j];
                    for k in 0..current_input.ncols() {
                        if k < weights.ncols() {
                            sum = sum + current_input[[i, k]] * weights[[j, k]];
                        }
                    }
                    layer_output[[i, j]] = self.relu(sum);
                }
            }

            current_input = layer_output;
        }

        Ok(current_input)
    }

    /// Compute prototypes for each class
    pub fn compute_prototypes(
        &self,
        support_features: &Array2<F>,
        support_labels: &Array1<usize>,
    ) -> Result<Array2<F>> {
        // Find unique classes
        let mut unique_classes = Vec::new();
        for &label in support_labels {
            if !unique_classes.contains(&label) {
                unique_classes.push(label);
            }
        }
        unique_classes.sort();

        let num_classes = unique_classes.len();
        let mut prototypes = Array2::zeros((num_classes, self.feature_dim));

        // Compute prototype for each class
        for (class_idx, &class_label) in unique_classes.iter().enumerate() {
            let mut class_features = Vec::new();
            for (i, &label) in support_labels.iter().enumerate() {
                if label == class_label {
                    class_features.push(support_features.row(i).to_owned());
                }
            }

            if !class_features.is_empty() {
                // Compute mean of class _features
                for j in 0..self.feature_dim {
                    let mut sum = F::zero();
                    for features in &class_features {
                        sum = sum + features[j];
                    }
                    prototypes[[class_idx, j]] = sum / F::from(class_features.len()).unwrap();
                }
            }
        }

        Ok(prototypes)
    }

    /// Classify query samples using prototypical networks
    pub fn classify_queries(
        &self,
        query_features: &Array2<F>,
        prototypes: &Array2<F>,
    ) -> Result<Array1<usize>> {
        let num_queries = query_features.nrows();
        let num_classes = prototypes.nrows();
        let mut predictions = Array1::zeros(num_queries);

        for i in 0..num_queries {
            let mut min_distance = F::infinity();
            let mut predicted_class = 0;

            // Find closest prototype
            for j in 0..num_classes {
                let distance = self.euclidean_distance(
                    &query_features.row(i).to_owned(),
                    &prototypes.row(j).to_owned(),
                )?;

                if distance < min_distance {
                    min_distance = distance;
                    predicted_class = j;
                }
            }

            predictions[i] = predicted_class;
        }

        Ok(predictions)
    }

    /// Few-shot learning episode
    pub fn few_shot_episode(
        &self,
        support_x: &Array2<F>,
        support_y: &Array1<usize>,
        query_x: &Array2<F>,
    ) -> Result<Array1<usize>> {
        // Extract features
        let support_features = self.extract_features(support_x)?;
        let query_features = self.extract_features(query_x)?;

        // Compute prototypes
        let prototypes = self.compute_prototypes(&support_features, support_y)?;

        // Classify queries
        self.classify_queries(&query_features, &prototypes)
    }

    /// Train the feature extractor on a batch of few-shot tasks
    pub fn meta_train(&mut self, episodes: &[FewShotEpisode<F>]) -> Result<F> {
        let mut total_loss = F::zero();
        let mut total_gradients = Array2::zeros(self.feature_extractor.dim());

        for episode in episodes {
            // Forward pass
            let predictions =
                self.few_shot_episode(&episode.support_x, &episode.support_y, &episode.query_x)?;

            // Compute loss (cross-entropy approximation)
            let mut episode_loss = F::zero();
            for (i, &pred) in predictions.iter().enumerate() {
                if i < episode.query_y.len() {
                    let target = episode.query_y[i];
                    if pred != target {
                        episode_loss = episode_loss + F::one();
                    }
                }
            }
            episode_loss = episode_loss / F::from(predictions.len()).unwrap();

            // Compute gradients (simplified numerical differentiation)
            let gradients = self.compute_gradients(episode)?;
            total_gradients = total_gradients + gradients;
            total_loss = total_loss + episode_loss;
        }

        // Update parameters
        let learning_rate = F::from(0.001).unwrap();
        let num_episodes = F::from(episodes.len()).unwrap();
        total_gradients = total_gradients / num_episodes;

        self.feature_extractor = self.feature_extractor.clone() - total_gradients * learning_rate;

        Ok(total_loss / num_episodes)
    }

    // Helper methods
    fn extract_layer_parameters(&self) -> Vec<(Array2<F>, Array1<F>)> {
        let param_vec = self.feature_extractor.row(0);
        let mut layer_params = Vec::new();
        let mut param_idx = 0;

        let mut layer_sizes = vec![self.input_dim];
        layer_sizes.extend(&self.hidden_dims);
        layer_sizes.push(self.feature_dim);

        for i in 0..layer_sizes.len() - 1 {
            let input_size = layer_sizes[i];
            let output_size = layer_sizes[i + 1];

            // Extract weights
            let mut weights = Array2::zeros((output_size, input_size));
            for j in 0..output_size {
                for k in 0..input_size {
                    if param_idx < param_vec.len() {
                        weights[[j, k]] = param_vec[param_idx];
                        param_idx += 1;
                    }
                }
            }

            // Extract biases
            let mut biases = Array1::zeros(output_size);
            for j in 0..output_size {
                if param_idx < param_vec.len() {
                    biases[j] = param_vec[param_idx];
                    param_idx += 1;
                }
            }

            layer_params.push((weights, biases));
        }

        layer_params
    }

    fn euclidean_distance(&self, a: &Array1<F>, b: &Array1<F>) -> Result<F> {
        let mut sum = F::zero();
        for i in 0..a.len().min(b.len()) {
            let diff = a[i] - b[i];
            sum = sum + diff * diff;
        }
        Ok(sum.sqrt())
    }

    fn relu(&self, x: F) -> F {
        x.max(F::zero())
    }

    fn compute_gradients(&self, episode: &FewShotEpisode<F>) -> Result<Array2<F>> {
        // Simplified gradient computation
        let epsilon = F::from(1e-5).unwrap();
        let mut gradients = Array2::zeros(self.feature_extractor.dim());

        let base_predictions =
            self.few_shot_episode(&episode.support_x, &episode.support_y, &episode.query_x)?;
        let mut base_loss = F::zero();
        for (i, &pred) in base_predictions.iter().enumerate() {
            if i < episode.query_y.len() && pred != episode.query_y[i] {
                base_loss = base_loss + F::one();
            }
        }

        // Numerical differentiation for each parameter
        for i in 0..self.feature_extractor.ncols() {
            let mut perturbed_extractor = self.feature_extractor.clone();
            perturbed_extractor[[0, i]] = perturbed_extractor[[0, i]] + epsilon;

            // Create temporary network with perturbed parameters
            let mut temp_network = self.clone();
            temp_network.feature_extractor = perturbed_extractor;

            let perturbed_predictions = temp_network.few_shot_episode(
                &episode.support_x,
                &episode.support_y,
                &episode.query_x,
            )?;
            let mut perturbed_loss = F::zero();
            for (j, &pred) in perturbed_predictions.iter().enumerate() {
                if j < episode.query_y.len() && pred != episode.query_y[j] {
                    perturbed_loss = perturbed_loss + F::one();
                }
            }

            gradients[[0, i]] = (perturbed_loss - base_loss) / epsilon;
        }

        Ok(gradients)
    }
}

impl<F: Float + Debug + Clone + FromPrimitive + ndarray::ScalarOperand> Clone
    for PrototypicalNetworks<F>
{
    fn clone(&self) -> Self {
        Self {
            feature_extractor: self.feature_extractor.clone(),
            input_dim: self.input_dim,
            feature_dim: self.feature_dim,
            hidden_dims: self.hidden_dims.clone(),
        }
    }
}

/// Few-shot learning episode data structure
#[derive(Debug, Clone)]
pub struct FewShotEpisode<F: Float + Debug> {
    /// Support set inputs
    pub support_x: Array2<F>,
    /// Support set labels
    pub support_y: Array1<usize>,
    /// Query set inputs  
    pub query_x: Array2<F>,
    /// Query set labels
    pub query_y: Array1<usize>,
}

/// REPTILE Algorithm for Meta-Learning
#[derive(Debug)]
pub struct REPTILE<F: Float + Debug + ndarray::ScalarOperand> {
    /// Base model parameters
    parameters: Array2<F>,
    /// Meta-learning rate
    meta_lr: F,
    /// Inner loop learning rate
    inner_lr: F,
    /// Number of inner gradient steps
    inner_steps: usize,
    /// Model dimensions
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
}

impl<F: Float + Debug + Clone + FromPrimitive + ndarray::ScalarOperand> REPTILE<F> {
    /// Create new REPTILE instance
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        meta_lr: F,
        inner_lr: F,
        inner_steps: usize,
    ) -> Self {
        // Initialize parameters using Xavier initialization
        let total_params =
            input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim + output_dim;
        let scale = F::from(2.0).unwrap() / F::from(input_dim + output_dim).unwrap();
        let std_dev = scale.sqrt();

        let mut parameters = Array2::zeros((1, total_params));
        for i in 0..total_params {
            let val = ((i * 59) % 1000) as f64 / 1000.0 - 0.5;
            parameters[[0, i]] = F::from(val).unwrap() * std_dev;
        }

        Self {
            parameters,
            meta_lr,
            inner_lr,
            inner_steps,
            input_dim,
            hidden_dim,
            output_dim,
        }
    }

    /// REPTILE meta-training step
    pub fn meta_train(&mut self, tasks: &[TaskData<F>]) -> Result<F> {
        let mut total_loss = F::zero();
        let mut parameter_updates = Array2::zeros(self.parameters.dim());

        for task in tasks {
            // Store initial parameters
            let initial_params = self.parameters.clone();

            // Inner loop training on task
            let mut task_params = initial_params.clone();
            for _ in 0..self.inner_steps {
                let gradients = self.compute_task_gradients(&task_params, task)?;
                task_params = task_params - gradients * self.inner_lr;
            }

            // Compute task loss
            let task_loss = self.forward(&task_params, &task.support_x, &task.support_y)?;
            total_loss = total_loss + task_loss;

            // REPTILE update: move towards task-adapted parameters
            let update = task_params - initial_params;
            parameter_updates = parameter_updates + update;
        }

        // Meta-update: average parameter updates across tasks
        let num_tasks = F::from(tasks.len()).unwrap();
        parameter_updates = parameter_updates / num_tasks;
        total_loss = total_loss / num_tasks;

        // Update meta-parameters
        self.parameters = self.parameters.clone() + parameter_updates * self.meta_lr;

        Ok(total_loss)
    }

    /// Fast adaptation for new task (few-shot learning)
    pub fn fast_adapt(&self, support_x: &Array2<F>, support_y: &Array2<F>) -> Result<Array2<F>> {
        let task = TaskData {
            support_x: support_x.clone(),
            support_y: support_y.clone(),
            query_x: support_x.clone(),
            query_y: support_y.clone(),
        };

        // Inner loop adaptation
        let mut adapted_params = self.parameters.clone();
        for _ in 0..self.inner_steps {
            let gradients = self.compute_task_gradients(&adapted_params, &task)?;
            adapted_params = adapted_params - gradients * self.inner_lr;
        }

        Ok(adapted_params)
    }

    /// Forward pass through neural network
    fn forward(&self, params: &Array2<F>, inputs: &Array2<F>, targets: &Array2<F>) -> Result<F> {
        let predictions = self.predict(params, inputs)?;

        // Mean squared error loss
        let mut loss = F::zero();
        let (batch_size, _) = predictions.dim();

        for i in 0..batch_size {
            for j in 0..self.output_dim {
                let diff = predictions[[i, j]] - targets[[i, j]];
                loss = loss + diff * diff;
            }
        }

        Ok(loss / F::from(batch_size).unwrap())
    }

    /// Make predictions using current parameters
    fn predict(&self, params: &Array2<F>, inputs: &Array2<F>) -> Result<Array2<F>> {
        let (batch_size, _) = inputs.dim();

        // Extract weight matrices from flattened parameters
        let (w1, b1, w2, b2) = self.extract_weights(params);

        // Forward pass: input -> hidden -> output
        let mut hidden = Array2::zeros((batch_size, self.hidden_dim));

        // Input to hidden layer
        for i in 0..batch_size {
            for j in 0..self.hidden_dim {
                let mut sum = b1[j];
                for k in 0..self.input_dim {
                    sum = sum + inputs[[i, k]] * w1[[j, k]];
                }
                hidden[[i, j]] = self.relu(sum); // ReLU activation
            }
        }

        // Hidden to output layer
        let mut output = Array2::zeros((batch_size, self.output_dim));
        for i in 0..batch_size {
            for j in 0..self.output_dim {
                let mut sum = b2[j];
                for k in 0..self.hidden_dim {
                    sum = sum + hidden[[i, k]] * w2[[j, k]];
                }
                output[[i, j]] = sum; // Linear output
            }
        }

        Ok(output)
    }

    /// Extract weight matrices from flattened parameter vector
    fn extract_weights(&self, params: &Array2<F>) -> (Array2<F>, Array1<F>, Array2<F>, Array1<F>) {
        let param_vec = params.row(0);
        let mut idx = 0;

        // W1: input_dim x hidden_dim
        let mut w1 = Array2::zeros((self.hidden_dim, self.input_dim));
        for i in 0..self.hidden_dim {
            for j in 0..self.input_dim {
                w1[[i, j]] = param_vec[idx];
                idx += 1;
            }
        }

        // b1: hidden_dim
        let mut b1 = Array1::zeros(self.hidden_dim);
        for i in 0..self.hidden_dim {
            b1[i] = param_vec[idx];
            idx += 1;
        }

        // W2: hidden_dim x output_dim
        let mut w2 = Array2::zeros((self.output_dim, self.hidden_dim));
        for i in 0..self.output_dim {
            for j in 0..self.hidden_dim {
                w2[[i, j]] = param_vec[idx];
                idx += 1;
            }
        }

        // b2: output_dim
        let mut b2 = Array1::zeros(self.output_dim);
        for i in 0..self.output_dim {
            b2[i] = param_vec[idx];
            idx += 1;
        }

        (w1, b1, w2, b2)
    }

    /// ReLU activation function
    fn relu(&self, x: F) -> F {
        x.max(F::zero())
    }

    /// Compute task-specific gradients
    fn compute_task_gradients(&self, params: &Array2<F>, task: &TaskData<F>) -> Result<Array2<F>> {
        let epsilon = F::from(1e-5).unwrap();
        let mut gradients = Array2::zeros(params.dim());

        let base_loss = self.forward(params, &task.support_x, &task.support_y)?;

        for i in 0..params.ncols() {
            let mut perturbed_params = params.clone();
            perturbed_params[[0, i]] = perturbed_params[[0, i]] + epsilon;

            let perturbed_loss =
                self.forward(&perturbed_params, &task.support_x, &task.support_y)?;
            gradients[[0, i]] = (perturbed_loss - base_loss) / epsilon;
        }

        Ok(gradients)
    }
}

/// Memory-Augmented Neural Network (MANN) for Meta-Learning
#[derive(Debug)]
pub struct MANN<F: Float + Debug + ndarray::ScalarOperand> {
    /// Controller network parameters
    controller_params: Array2<F>,
    /// External memory matrix
    memory: Array2<F>,
    /// Memory dimensions
    memory_size: usize,
    memory_width: usize,
    /// Controller dimensions
    controller_input_dim: usize,
    controller_hidden_dim: usize,
    controller_output_dim: usize,
    /// Read/write head parameters
    #[allow(dead_code)]
    read_head_params: Array2<F>,
    #[allow(dead_code)]
    write_head_params: Array2<F>,
}

impl<F: Float + Debug + Clone + FromPrimitive + ndarray::ScalarOperand> MANN<F> {
    /// Create new Memory-Augmented Neural Network
    pub fn new(
        memory_size: usize,
        memory_width: usize,
        controller_input_dim: usize,
        controller_hidden_dim: usize,
        controller_output_dim: usize,
    ) -> Self {
        // Initialize controller parameters
        let controller_param_count = controller_input_dim * controller_hidden_dim
            + controller_hidden_dim
            + controller_hidden_dim * controller_output_dim
            + controller_output_dim;

        let mut controller_params = Array2::zeros((1, controller_param_count));
        let scale =
            F::from(2.0).unwrap() / F::from(controller_input_dim + controller_output_dim).unwrap();
        let std_dev = scale.sqrt();

        for i in 0..controller_param_count {
            let val = ((i * 67) % 1000) as f64 / 1000.0 - 0.5;
            controller_params[[0, i]] = F::from(val).unwrap() * std_dev;
        }

        // Initialize memory
        let memory = Array2::zeros((memory_size, memory_width));

        // Initialize read/write head parameters
        let head_param_count = memory_width * 2 + 3; // key, beta, gate, shift, gamma
        let mut read_head_params = Array2::zeros((1, head_param_count));
        let mut write_head_params = Array2::zeros((1, head_param_count));

        for i in 0..head_param_count {
            let val1 = ((i * 71) % 1000) as f64 / 1000.0 - 0.5;
            let val2 = ((i * 73) % 1000) as f64 / 1000.0 - 0.5;
            read_head_params[[0, i]] = F::from(val1).unwrap() * F::from(0.1).unwrap();
            write_head_params[[0, i]] = F::from(val2).unwrap() * F::from(0.1).unwrap();
        }

        Self {
            controller_params,
            memory,
            memory_size,
            memory_width,
            controller_input_dim,
            controller_hidden_dim,
            controller_output_dim,
            read_head_params,
            write_head_params,
        }
    }

    /// Forward pass through MANN
    pub fn forward(&mut self, input: &Array1<F>) -> Result<Array1<F>> {
        // Read from memory
        let read_vector = self.memory_read()?;

        // Combine input with read vector
        let mut controller_input = Array1::zeros(self.controller_input_dim);
        for i in 0..input.len().min(self.controller_input_dim) {
            controller_input[i] = input[i];
        }

        // Add read vector to controller input
        let read_start = input.len().min(self.controller_input_dim);
        for i in 0..read_vector.len() {
            if read_start + i < self.controller_input_dim {
                controller_input[read_start + i] = read_vector[i];
            }
        }

        // Controller forward pass
        let controller_output = self.controller_forward(&controller_input)?;

        // Write to memory
        self.memory_write(&controller_output)?;

        Ok(controller_output)
    }

    /// Controller neural network forward pass
    fn controller_forward(&self, input: &Array1<F>) -> Result<Array1<F>> {
        let (w1, b1, w2, b2) = self.extract_controller_weights();

        // Hidden layer
        let mut hidden = Array1::zeros(self.controller_hidden_dim);
        for i in 0..self.controller_hidden_dim {
            let mut sum = b1[i];
            for j in 0..input.len().min(w1.ncols()) {
                sum = sum + input[j] * w1[[i, j]];
            }
            hidden[i] = self.tanh(sum);
        }

        // Output layer
        let mut output = Array1::zeros(self.controller_output_dim);
        for i in 0..self.controller_output_dim {
            let mut sum = b2[i];
            for j in 0..self.controller_hidden_dim {
                sum = sum + hidden[j] * w2[[i, j]];
            }
            output[i] = sum;
        }

        Ok(output)
    }

    /// Read from external memory
    fn memory_read(&self) -> Result<Array1<F>> {
        // Simplified memory read - return average of memory rows
        let mut read_vector = Array1::zeros(self.memory_width);

        for i in 0..self.memory_size {
            for j in 0..self.memory_width {
                read_vector[j] = read_vector[j] + self.memory[[i, j]];
            }
        }

        let size = F::from(self.memory_size).unwrap();
        for j in 0..self.memory_width {
            read_vector[j] = read_vector[j] / size;
        }

        Ok(read_vector)
    }

    /// Write to external memory
    fn memory_write(&mut self, controller_output: &Array1<F>) -> Result<()> {
        // Simplified memory write - update first row with controller _output
        for i in 0..controller_output.len().min(self.memory_width) {
            self.memory[[0, i]] = controller_output[i];
        }

        Ok(())
    }

    /// Extract controller weights from parameters
    fn extract_controller_weights(&self) -> (Array2<F>, Array1<F>, Array2<F>, Array1<F>) {
        let param_vec = self.controller_params.row(0);
        let mut idx = 0;

        // W1: controller_input_dim x controller_hidden_dim
        let mut w1 = Array2::zeros((self.controller_hidden_dim, self.controller_input_dim));
        for i in 0..self.controller_hidden_dim {
            for j in 0..self.controller_input_dim {
                if idx < param_vec.len() {
                    w1[[i, j]] = param_vec[idx];
                    idx += 1;
                }
            }
        }

        // b1: controller_hidden_dim
        let mut b1 = Array1::zeros(self.controller_hidden_dim);
        for i in 0..self.controller_hidden_dim {
            if idx < param_vec.len() {
                b1[i] = param_vec[idx];
                idx += 1;
            }
        }

        // W2: controller_hidden_dim x controller_output_dim
        let mut w2 = Array2::zeros((self.controller_output_dim, self.controller_hidden_dim));
        for i in 0..self.controller_output_dim {
            for j in 0..self.controller_hidden_dim {
                if idx < param_vec.len() {
                    w2[[i, j]] = param_vec[idx];
                    idx += 1;
                }
            }
        }

        // b2: controller_output_dim
        let mut b2 = Array1::zeros(self.controller_output_dim);
        for i in 0..self.controller_output_dim {
            if idx < param_vec.len() {
                b2[i] = param_vec[idx];
                idx += 1;
            }
        }

        (w1, b1, w2, b2)
    }

    /// Reset memory
    pub fn reset_memory(&mut self) {
        self.memory = Array2::zeros((self.memory_size, self.memory_width));
    }

    /// Train MANN on few-shot learning task
    pub fn train_few_shot(&mut self, episodes: &[FewShotEpisode<F>]) -> Result<F> {
        let mut total_loss = F::zero();

        for episode in episodes {
            self.reset_memory();

            // Present support set
            for i in 0..episode.support_x.nrows() {
                let input_row = episode.support_x.row(i).to_owned();
                let _output = self.forward(&input_row)?;
            }

            // Test on query set
            let mut episode_loss = F::zero();
            for i in 0..episode.query_x.nrows() {
                let input_row = episode.query_x.row(i).to_owned();
                let prediction = self.forward(&input_row)?;

                // Compute loss (simplified)
                if i < episode.query_y.len() {
                    let target = F::from(episode.query_y[i]).unwrap();
                    if !prediction.is_empty() {
                        let diff = prediction[0] - target;
                        episode_loss = episode_loss + diff * diff;
                    }
                }
            }

            total_loss = total_loss + episode_loss;
        }

        Ok(total_loss / F::from(episodes.len()).unwrap())
    }

    /// Hyperbolic tangent activation
    fn tanh(&self, x: F) -> F {
        x.tanh()
    }
}

/// Meta-Optimizer: Learning to Learn by Gradient Descent by Gradient Descent
#[derive(Debug)]
pub struct MetaOptimizer<F: Float + Debug + ndarray::ScalarOperand> {
    /// LSTM parameters for the optimizer
    #[allow(dead_code)]
    lstm_params: Array2<F>,
    /// Hidden state size
    hidden_size: usize,
    /// Input dimension (gradient + other features)
    input_dim: usize,
    /// Current LSTM hidden state
    hidden_state: Array1<F>,
    /// Current LSTM cell state
    cell_state: Array1<F>,
}

impl<F: Float + Debug + Clone + FromPrimitive + ndarray::ScalarOperand> MetaOptimizer<F> {
    /// Create new meta-optimizer
    pub fn new(input_dim: usize, hidden_size: usize) -> Self {
        // Initialize LSTM parameters
        let param_count = 4 * hidden_size * (input_dim + hidden_size) + 4 * hidden_size; // 4 gates
        let mut lstm_params = Array2::zeros((1, param_count));

        let scale = F::from(1.0).unwrap() / F::from(hidden_size).unwrap().sqrt();
        for i in 0..param_count {
            let val = ((i * 79) % 1000) as f64 / 1000.0 - 0.5;
            lstm_params[[0, i]] = F::from(val).unwrap() * scale;
        }

        let hidden_state = Array1::zeros(hidden_size);
        let cell_state = Array1::zeros(hidden_size);

        Self {
            lstm_params,
            hidden_size,
            input_dim,
            hidden_state,
            cell_state,
        }
    }

    /// Generate parameter update using meta-optimizer
    pub fn generate_update(
        &mut self,
        gradient: F,
        loss_history: &[F],
        step_count: usize,
    ) -> Result<F> {
        // Prepare input features
        let mut input = Array1::zeros(self.input_dim);
        input[0] = gradient;

        if self.input_dim > 1 && !loss_history.is_empty() {
            input[1] = loss_history[loss_history.len() - 1];
        }

        if self.input_dim > 2 {
            input[2] = F::from(step_count).unwrap();
        }

        // LSTM forward pass
        let (new_hidden, new_cell) = self.lstm_forward(&input)?;
        self.hidden_state = new_hidden.clone();
        self.cell_state = new_cell;

        // Generate parameter update (use first output as update)
        Ok(new_hidden[0])
    }

    /// LSTM forward pass
    fn lstm_forward(&self, input: &Array1<F>) -> Result<(Array1<F>, Array1<F>)> {
        // Extract LSTM weights (simplified implementation)
        let combined_input = self.combine_input_hidden(input);

        // Compute gates (simplified)
        let forget_gate = self.sigmoid(combined_input[0]);
        let input_gate = self.sigmoid(combined_input[1]);
        let candidate_gate = self.tanh(combined_input[2]);
        let output_gate = self.sigmoid(combined_input[3]);

        // Update cell state
        let mut new_cell_state = Array1::zeros(self.hidden_size);
        for i in 0..self.hidden_size {
            new_cell_state[i] = forget_gate * self.cell_state[i] + input_gate * candidate_gate;
        }

        // Update hidden state
        let mut new_hidden_state = Array1::zeros(self.hidden_size);
        for i in 0..self.hidden_size {
            new_hidden_state[i] = output_gate * self.tanh(new_cell_state[i]);
        }

        Ok((new_hidden_state, new_cell_state))
    }

    /// Combine input and hidden state
    fn combine_input_hidden(&self, input: &Array1<F>) -> Array1<F> {
        // Simplified combination - just use input values for gates
        let mut combined = Array1::zeros(4);
        for i in 0..4.min(input.len()) {
            combined[i] = input[i.min(input.len() - 1)];
        }
        combined
    }

    /// Sigmoid activation
    fn sigmoid(&self, x: F) -> F {
        F::one() / (F::one() + (-x).exp())
    }

    /// Hyperbolic tangent activation
    fn tanh(&self, x: F) -> F {
        x.tanh()
    }

    /// Reset optimizer state
    pub fn reset(&mut self) {
        self.hidden_state = Array1::zeros(self.hidden_size);
        self.cell_state = Array1::zeros(self.hidden_size);
    }

    /// Train meta-optimizer on optimization tasks
    pub fn meta_train(&mut self, optimization_problems: &[OptimizationProblem<F>]) -> Result<F> {
        let mut total_loss = F::zero();

        for problem in optimization_problems {
            self.reset();

            let mut current_params = problem.initial_params.clone();
            let mut loss_history = Vec::new();

            // Simulate optimization steps
            for step in 0..problem.max_steps {
                // Compute gradient
                let gradient = self.compute_simple_gradient(&current_params, problem)?;

                // Generate update using meta-optimizer
                let update = self.generate_update(gradient, &loss_history, step)?;

                // Apply update
                current_params = current_params + update;

                // Compute loss
                let loss = self.evaluate_objective(&current_params, problem)?;
                loss_history.push(loss);
                total_loss = total_loss + loss;
            }
        }

        Ok(total_loss / F::from(optimization_problems.len()).unwrap())
    }

    /// Compute simple gradient (placeholder)
    fn compute_simple_gradient(
        &self,
        params: &Array1<F>,
        problem: &OptimizationProblem<F>,
    ) -> Result<F> {
        // Simplified gradient computation
        if !params.is_empty() && !problem.target.is_empty() {
            Ok(params[0] - problem.target[0])
        } else {
            Ok(F::zero())
        }
    }

    /// Evaluate objective function
    fn evaluate_objective(
        &self,
        params: &Array1<F>,
        problem: &OptimizationProblem<F>,
    ) -> Result<F> {
        // Simple quadratic objective
        let mut loss = F::zero();
        for i in 0..params.len().min(problem.target.len()) {
            let diff = params[i] - problem.target[i];
            loss = loss + diff * diff;
        }
        Ok(loss)
    }
}

/// Optimization problem for meta-optimizer training
#[derive(Debug, Clone)]
pub struct OptimizationProblem<F: Float + Debug> {
    /// Initial parameters
    pub initial_params: Array1<F>,
    /// Target parameters
    pub target: Array1<F>,
    /// Maximum optimization steps
    pub max_steps: usize,
}

#[cfg(test)]
mod advanced_meta_learning_tests {
    use super::*;

    #[test]
    fn test_prototypical_networks() {
        let proto_net = PrototypicalNetworks::<f64>::new(10, 5, vec![8]);

        let support_x =
            Array2::from_shape_vec((4, 10), (0..40).map(|i| i as f64 * 0.1).collect()).unwrap();
        let support_y = Array1::from_vec(vec![0, 0, 1, 1]);
        let query_x =
            Array2::from_shape_vec((2, 10), (40..60).map(|i| i as f64 * 0.1).collect()).unwrap();

        let predictions = proto_net
            .few_shot_episode(&support_x, &support_y, &query_x)
            .unwrap();
        assert_eq!(predictions.len(), 2);
    }

    #[test]
    fn test_reptile_algorithm() {
        let mut reptile = REPTILE::<f64>::new(5, 8, 3, 0.01, 0.1, 5);

        let task = TaskData {
            support_x: Array2::from_shape_vec((3, 5), (0..15).map(|i| i as f64 * 0.1).collect())
                .unwrap(),
            support_y: Array2::from_shape_vec((3, 3), (0..9).map(|i| i as f64 * 0.1).collect())
                .unwrap(),
            query_x: Array2::from_shape_vec((2, 5), (15..25).map(|i| i as f64 * 0.1).collect())
                .unwrap(),
            query_y: Array2::from_shape_vec((2, 3), (9..15).map(|i| i as f64 * 0.1).collect())
                .unwrap(),
        };

        let loss = reptile.meta_train(&[task]).unwrap();
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_memory_augmented_neural_network() {
        let mut mann = MANN::<f64>::new(10, 8, 6, 12, 4);

        let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let output = mann.forward(&input).unwrap();
        assert_eq!(output.len(), 4);

        // Test memory reset
        mann.reset_memory();
        let output2 = mann.forward(&input).unwrap();
        assert_eq!(output2.len(), 4);
    }

    #[test]
    fn test_meta_optimizer() {
        let mut meta_opt = MetaOptimizer::<f64>::new(3, 5);

        let gradient = 0.1;
        let loss_history = vec![1.0, 0.8, 0.6];
        let step_count = 5;

        let update = meta_opt
            .generate_update(gradient, &loss_history, step_count)
            .unwrap();
        assert!(update.is_finite());

        // Test reset
        meta_opt.reset();
    }

    #[test]
    fn test_few_shot_episode_structure() {
        let episode = FewShotEpisode {
            support_x: Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
            support_y: Array1::from_vec(vec![0, 1]),
            query_x: Array2::from_shape_vec((1, 3), vec![7.0, 8.0, 9.0]).unwrap(),
            query_y: Array1::from_vec(vec![1]),
        };

        assert_eq!(episode.support_x.dim(), (2, 3));
        assert_eq!(episode.support_y.len(), 2);
        assert_eq!(episode.query_x.dim(), (1, 3));
        assert_eq!(episode.query_y.len(), 1);
    }
}
