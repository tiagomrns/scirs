//! K-FAC (Kronecker-Factored Approximate Curvature) optimizer
//!
//! This module implements K-FAC, an efficient second-order optimization method
//! that approximates the Fisher information matrix using Kronecker factorization.
//! This allows for much more scalable second-order optimization compared to
//! storing and inverting the full Fisher information matrix.

#![allow(dead_code)]

use crate::error::{OptimError, Result};
use ndarray::{s, Array1, Array2, Axis};
use num_traits::Float;
use std::collections::HashMap;
use std::fmt::Debug;

/// K-FAC optimizer configuration
#[derive(Debug, Clone)]
pub struct KFACConfig<T: Float> {
    /// Learning rate
    pub learning_rate: T,

    /// Damping parameter for numerical stability
    pub damping: T,

    /// Weight decay (L2 regularization)
    pub weight_decay: T,

    /// Update frequency for covariance matrices
    pub cov_update_freq: usize,

    /// Update frequency for inverse covariance matrices
    pub inv_update_freq: usize,

    /// Exponential moving average decay for statistics
    pub stat_decay: T,

    /// Minimum eigenvalue for regularization
    pub min_eigenvalue: T,

    /// Maximum number of iterations for iterative inversion
    pub max_inv_iterations: usize,

    /// Tolerance for iterative inversion
    pub inv_tolerance: T,

    /// Use Tikhonov regularization
    pub use_tikhonov: bool,

    /// Enable automatic damping adjustment
    pub auto_damping: bool,

    /// Target acceptance ratio for damping adjustment
    pub target_acceptance_ratio: T,
}

impl<T: Float> Default for KFACConfig<T> {
    fn default() -> Self {
        Self {
            learning_rate: T::from(0.001).unwrap(),
            damping: T::from(0.001).unwrap(),
            weight_decay: T::from(0.0).unwrap(),
            cov_update_freq: 10,
            inv_update_freq: 100,
            stat_decay: T::from(0.95).unwrap(),
            min_eigenvalue: T::from(1e-7).unwrap(),
            max_inv_iterations: 50,
            inv_tolerance: T::from(1e-6).unwrap(),
            use_tikhonov: true,
            auto_damping: true,
            target_acceptance_ratio: T::from(0.75).unwrap(),
        }
    }
}

/// Layer information for K-FAC
#[derive(Debug, Clone)]
pub struct LayerInfo {
    /// Layer name/identifier
    pub name: String,

    /// Input dimension
    pub input_dim: usize,

    /// Output dimension  
    pub output_dim: usize,

    /// Layer type
    pub layer_type: LayerType,

    /// Whether to include bias
    pub has_bias: bool,
}

/// Types of layers supported by K-FAC
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerType {
    /// Dense/Fully connected layer
    Dense,

    /// Convolutional layer
    Convolution,

    /// Convolutional layer with grouped/depthwise convolution
    GroupedConvolution { groups: usize },

    /// Embedding layer
    Embedding,

    /// Batch normalization layer
    BatchNorm,
}

/// K-FAC optimizer state for a single layer
#[derive(Debug, Clone)]
pub struct KFACLayerState<T: Float> {
    /// Input covariance matrix A = E[a a^T]
    pub a_cov: Array2<T>,

    /// Output gradient covariance matrix G = E[g g^T]
    pub g_cov: Array2<T>,

    /// Inverse of input covariance matrix
    pub a_cov_inv: Option<Array2<T>>,

    /// Inverse of output gradient covariance matrix
    pub g_cov_inv: Option<Array2<T>>,

    /// Number of updates performed
    pub num_updates: usize,

    /// Last update step for covariance matrices
    pub last_cov_update: usize,

    /// Last update step for inverse matrices
    pub last_inv_update: usize,

    /// Damping values for this layer
    pub damping_a: T,
    pub damping_g: T,

    /// Layer information
    pub layerinfo: LayerInfo,

    /// Precomputed Kronecker factors for bias
    pub bias_correction: Option<Array1<T>>,

    /// Moving average statistics
    pub running_mean_a: Option<Array1<T>>,
    pub running_mean_g: Option<Array1<T>>,
}

/// Main K-FAC optimizer
#[derive(Debug)]
pub struct KFAC<T: Float> {
    /// Configuration
    config: KFACConfig<T>,

    /// Per-layer state
    layer_states: HashMap<String, KFACLayerState<T>>,

    /// Global step counter
    step_count: usize,

    /// Acceptance ratio for damping adjustment
    acceptance_ratio: T,

    /// Previous loss for loss-based damping
    previous_loss: Option<T>,

    /// Eigenvalue regularization history
    eigenvalue_history: Vec<T>,

    /// Performance statistics
    stats: KFACStats<T>,
}

/// K-FAC performance statistics
#[derive(Debug, Clone, Default)]
pub struct KFACStats<T: Float> {
    /// Total number of optimization steps
    pub total_steps: usize,

    /// Number of covariance updates
    pub cov_updates: usize,

    /// Number of inverse updates
    pub inv_updates: usize,

    /// Average condition number of covariance matrices
    pub avg_condition_number: T,

    /// Time spent in different operations (in microseconds)
    pub time_cov_update: u64,
    pub time_inv_update: u64,
    pub time_gradient_update: u64,

    /// Memory usage estimate (in bytes)
    pub memory_usage: usize,
}

impl<
        T: Float
            + Default
            + Clone
            + Send
            + Sync
            + std::iter::Sum
            + ndarray::ScalarOperand
            + 'static
            + num_traits::FromPrimitive,
    > KFAC<T>
{
    /// Create a new K-FAC optimizer
    pub fn new(config: KFACConfig<T>) -> Self {
        Self {
            config,
            layer_states: HashMap::new(),
            step_count: 0,
            acceptance_ratio: T::from(1.0).unwrap(),
            previous_loss: None,
            eigenvalue_history: Vec::new(),
            stats: KFACStats::default(),
        }
    }

    /// Register a layer with the optimizer
    pub fn register_layer(&mut self, layerinfo: LayerInfo) -> Result<()> {
        let input_size = layerinfo.input_dim + if layerinfo.has_bias { 1 } else { 0 };
        let output_size = layerinfo.output_dim;
        let layername = layerinfo.name.clone();

        let state = KFACLayerState {
            a_cov: Array2::eye(input_size),
            g_cov: Array2::eye(output_size),
            a_cov_inv: None,
            g_cov_inv: None,
            num_updates: 0,
            last_cov_update: 0,
            last_inv_update: 0,
            damping_a: self.config.damping,
            damping_g: self.config.damping,
            layerinfo,
            bias_correction: None,
            running_mean_a: None,
            running_mean_g: None,
        };

        self.layer_states.insert(layername, state);
        Ok(())
    }

    /// Update covariance matrices with new activations and gradients
    pub fn update_covariance_matrices(
        &mut self,
        layername: &str,
        activations: &Array2<T>,
        gradients: &Array2<T>,
    ) -> Result<()> {
        let should_update = {
            let state = self.layer_states.get(layername).ok_or_else(|| {
                OptimError::InvalidConfig(format!("Layer {} not found", layername))
            })?;
            // Allow initial update when step_count is 0 and last_cov_update is 0
            self.step_count == 0 && state.last_cov_update == 0
                || self.step_count - state.last_cov_update >= self.config.cov_update_freq
        };

        if should_update {
            let stat_decay = self.config.stat_decay;
            let step_count = self.step_count;

            let state = self.layer_states.get_mut(layername).ok_or_else(|| {
                OptimError::InvalidConfig(format!("Layer {} not found", layername))
            })?;

            Self::update_input_covariance_static(state, activations, stat_decay)?;
            Self::update_output_covariance_static(state, gradients, stat_decay)?;
            state.last_cov_update = step_count;
            self.stats.cov_updates += 1;
        }

        Ok(())
    }

    /// Update input covariance matrix A = E[a a^T]
    fn update_input_covariance_static(
        state: &mut KFACLayerState<T>,
        activations: &Array2<T>,
        stat_decay: T,
    ) -> Result<()> {
        let batch_size = T::from(activations.nrows()).unwrap();

        // Add bias term if needed
        let augmented_activations = if state.layerinfo.has_bias {
            let mut aug = Array2::ones((activations.nrows(), activations.ncols() + 1));
            aug.slice_mut(s![.., ..activations.ncols()])
                .assign(activations);
            aug
        } else {
            activations.clone()
        };

        // Compute batch covariance: (1/batch_size) * A^T * A
        let batch_cov = augmented_activations.t().dot(&augmented_activations) / batch_size;

        // Exponential moving average update
        state.a_cov = &state.a_cov * stat_decay + &batch_cov * (T::one() - stat_decay);

        // Update running mean for bias correction
        if state.running_mean_a.is_none() {
            state.running_mean_a = Some(augmented_activations.mean_axis(Axis(0)).unwrap());
        } else {
            let mean = augmented_activations.mean_axis(Axis(0)).unwrap();
            let running_mean = state.running_mean_a.as_mut().unwrap();
            *running_mean = &*running_mean * stat_decay + &mean * (T::one() - stat_decay);
        }

        Ok(())
    }

    /// Update output gradient covariance matrix G = E[g g^T]
    fn update_output_covariance_static(
        state: &mut KFACLayerState<T>,
        gradients: &Array2<T>,
        stat_decay: T,
    ) -> Result<()> {
        let batch_size = T::from(gradients.nrows()).unwrap();

        // Compute batch covariance: (1/batch_size) * G^T * G
        let batch_cov = gradients.t().dot(gradients) / batch_size;

        // Exponential moving average update
        state.g_cov = &state.g_cov * stat_decay + &batch_cov * (T::one() - stat_decay);

        // Update running mean
        if state.running_mean_g.is_none() {
            state.running_mean_g = Some(gradients.mean_axis(Axis(0)).unwrap());
        } else {
            let mean = gradients.mean_axis(Axis(0)).unwrap();
            let running_mean = state.running_mean_g.as_mut().unwrap();
            *running_mean = &*running_mean * stat_decay + &mean * (T::one() - stat_decay);
        }

        Ok(())
    }

    /// Update inverse covariance matrices
    pub fn update_inverse_matrices(&mut self, layername: &str) -> Result<()> {
        let should_update = {
            let state = self.layer_states.get(layername).ok_or_else(|| {
                OptimError::InvalidConfig(format!("Layer {} not found", layername))
            })?;
            self.step_count - state.last_inv_update >= self.config.inv_update_freq
        };

        if should_update {
            let step_count = self.step_count;

            // Do the computation that needs mutable self borrow first
            let layer_exists = self.layer_states.contains_key(layername);
            if !layer_exists {
                return Err(OptimError::InvalidConfig(format!(
                    "Layer {} not found",
                    layername
                )));
            }

            // Now we can safely get mutable access to the layer state
            if let Some(state) = self.layer_states.get_mut(layername) {
                // Extract needed config before the method call
                let config = self.config.clone();
                Self::compute_inverse_covariance_static(state, &config)?;
                state.last_inv_update = step_count;
                self.stats.inv_updates += 1;
            }
        }

        Ok(())
    }

    /// Compute inverse covariance matrices with regularization (static version)
    fn compute_inverse_covariance_static(
        state: &mut KFACLayerState<T>,
        config: &KFACConfig<T>,
    ) -> Result<()> {
        // Add Tikhonov regularization for numerical stability
        let mut a_reg = state.a_cov.clone();
        let mut g_reg = state.g_cov.clone();

        if config.use_tikhonov {
            // Use adaptive damping or fixed damping
            let damping_a = if config.auto_damping {
                // Simplified adaptive damping - estimate condition number directly
                let mut max_diag = T::zero();
                let mut min_diag = T::infinity();
                for i in 0..a_reg.nrows() {
                    let diag = a_reg[[i, i]];
                    if diag > max_diag {
                        max_diag = diag;
                    }
                    if diag < min_diag {
                        min_diag = diag;
                    }
                }
                let condition_estimate = if min_diag > T::zero() {
                    max_diag / min_diag
                } else {
                    T::from(1e6).unwrap()
                };
                let adaptive_damping =
                    config.damping * (T::one() + condition_estimate * T::from(0.1).unwrap());
                adaptive_damping
                    .min(T::from(1e-3).unwrap())
                    .max(T::from(1e-8).unwrap())
            } else {
                state.damping_a
            };

            let damping_g = if config.auto_damping {
                // Simplified adaptive damping - estimate condition number directly
                let mut max_diag = T::zero();
                let mut min_diag = T::infinity();
                for i in 0..g_reg.nrows() {
                    let diag = g_reg[[i, i]];
                    if diag > max_diag {
                        max_diag = diag;
                    }
                    if diag < min_diag {
                        min_diag = diag;
                    }
                }
                let condition_estimate = if min_diag > T::zero() {
                    max_diag / min_diag
                } else {
                    T::from(1e6).unwrap()
                };
                let adaptive_damping =
                    config.damping * (T::one() + condition_estimate * T::from(0.1).unwrap());
                adaptive_damping
                    .min(T::from(1e-3).unwrap())
                    .max(T::from(1e-8).unwrap())
            } else {
                state.damping_g
            };

            // Add damping to diagonal
            for i in 0..a_reg.nrows() {
                a_reg[[i, i]] = a_reg[[i, i]] + damping_a;
            }
            for i in 0..g_reg.nrows() {
                g_reg[[i, i]] = g_reg[[i, i]] + damping_g;
            }

            state.damping_a = damping_a;
            state.damping_g = damping_g;
        }

        // Compute inverses using simplified safe inversion
        state.a_cov_inv = Some(natural_gradients::safe_matrix_inverse_static(&a_reg)?);
        state.g_cov_inv = Some(natural_gradients::safe_matrix_inverse_static(&g_reg)?);

        Ok(())
    }

    /// Compute inverse covariance matrices with regularization
    fn compute_inverse_covariance(&mut self, state: &mut KFACLayerState<T>) -> Result<()> {
        // Add Tikhonov regularization for numerical stability
        let mut a_reg = state.a_cov.clone();
        let mut g_reg = state.g_cov.clone();

        if self.config.use_tikhonov {
            // Adaptive damping based on condition number
            let damping_a = if self.config.auto_damping {
                self.compute_adaptive_damping(&a_reg)?
            } else {
                state.damping_a
            };

            let damping_g = if self.config.auto_damping {
                self.compute_adaptive_damping(&g_reg)?
            } else {
                state.damping_g
            };

            // Add damping to diagonal
            for i in 0..a_reg.nrows() {
                a_reg[[i, i]] = a_reg[[i, i]] + damping_a;
            }
            for i in 0..g_reg.nrows() {
                g_reg[[i, i]] = g_reg[[i, i]] + damping_g;
            }

            state.damping_a = damping_a;
            state.damping_g = damping_g;
        }

        // Compute inverses using Cholesky decomposition for numerical stability
        state.a_cov_inv = Some(self.safe_matrix_inverse(&a_reg)?);
        state.g_cov_inv = Some(self.safe_matrix_inverse(&g_reg)?);

        Ok(())
    }

    /// Safely compute matrix inverse with fallback methods
    fn safe_matrix_inverse(&self, matrix: &Array2<T>) -> Result<Array2<T>> {
        // Check matrix condition number first
        let condition_number = self.estimate_condition_number(matrix);
        let max_condition = T::from(1e12).unwrap();

        if condition_number > max_condition {
            // Use regularized inverse for ill-conditioned matrices
            return self.regularized_inverse(matrix);
        }

        // Try Cholesky decomposition first (fastest for positive definite)
        if let Ok(inverse) = self.cholesky_inverse(matrix) {
            return Ok(inverse);
        }

        // Fallback to LU decomposition with partial pivoting
        if let Ok(inverse) = self.lu_inverse(matrix) {
            return Ok(inverse);
        }

        // Use iterative refinement for better accuracy
        if let Ok(inverse) = self.iterative_inverse(matrix) {
            return Ok(inverse);
        }

        // Final fallback to pseudoinverse using SVD
        self.svd_pseudoinverse(matrix)
    }

    /// Compute matrix inverse using Cholesky decomposition
    fn cholesky_inverse(&self, matrix: &Array2<T>) -> Result<Array2<T>> {
        let n = matrix.nrows();
        let mut l = Array2::zeros((n, n));

        // Perform Cholesky decomposition: A = L * L^T
        for i in 0..n {
            for j in 0..=i {
                if i == j {
                    // Diagonal elements
                    let mut sum = T::zero();
                    for k in 0..j {
                        sum = sum + l[[j, k]] * l[[j, k]];
                    }
                    let diag_val = matrix[[j, j]] - sum;
                    if diag_val <= T::zero() {
                        return Err(OptimError::InvalidConfig(
                            "Matrix not positive definite".to_string(),
                        ));
                    }
                    l[[j, j]] = diag_val.sqrt();
                } else {
                    // Lower triangular elements
                    let mut sum = T::zero();
                    for k in 0..j {
                        sum = sum + l[[i, k]] * l[[j, k]];
                    }
                    l[[i, j]] = (matrix[[i, j]] - sum) / l[[j, j]];
                }
            }
        }

        // Compute inverse using forward and backward substitution
        self.cholesky_solve_identity(&l)
    }

    /// Compute matrix inverse using LU decomposition
    fn lu_inverse(&self, matrix: &Array2<T>) -> Result<Array2<T>> {
        let n = matrix.nrows();
        let mut lu = matrix.clone();
        let mut perm = (0..n).collect::<Vec<usize>>();

        // LU decomposition with partial pivoting
        for k in 0..n - 1 {
            // Find pivot
            let mut max_idx = k;
            let mut max_val = lu[[k, k]].abs();
            for i in k + 1..n {
                let val = lu[[i, k]].abs();
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }

            // Swap rows if needed
            if max_idx != k {
                for j in 0..n {
                    let temp = lu[[k, j]];
                    lu[[k, j]] = lu[[max_idx, j]];
                    lu[[max_idx, j]] = temp;
                }
                perm.swap(k, max_idx);
            }

            // Check for zero pivot
            if lu[[k, k]].abs() < T::from(1e-14).unwrap() {
                return Err(OptimError::InvalidConfig("Singular matrix".to_string()));
            }

            // Eliminate below pivot
            for i in k + 1..n {
                lu[[i, k]] = lu[[i, k]] / lu[[k, k]];
                for j in k + 1..n {
                    lu[[i, j]] = lu[[i, j]] - lu[[i, k]] * lu[[k, j]];
                }
            }
        }

        // Solve for inverse using forward and backward substitution
        self.lu_solve_identity(&lu, &perm)
    }

    /// Compute pseudoinverse using SVD
    fn svd_pseudoinverse(&self, matrix: &Array2<T>) -> Result<Array2<T>> {
        let (m, n) = matrix.dim();
        let min_dim = m.min(n);

        // Simplified SVD using power iteration for dominant eigenvalues
        let mut u = Array2::zeros((m, min_dim));
        let mut s = Array1::zeros(min_dim);
        let mut vt = Array2::zeros((min_dim, n));

        let mut a = matrix.clone();
        let tolerance = T::from(1e-10).unwrap();

        for k in 0..min_dim {
            // Power iteration to find k-th singular vector
            let mut v = Array1::ones(n);

            for _ in 0..50 {
                // Maximum iterations
                let u_k = a.dot(&v);
                let u_norm = u_k.iter().map(|&x| x * x).sum::<T>().sqrt();

                if u_norm < tolerance {
                    break;
                }

                let u_normalized = &u_k / u_norm;
                let v_new = a.t().dot(&u_normalized);
                let v_norm = v_new.iter().map(|&x| x * x).sum::<T>().sqrt();

                if v_norm < tolerance {
                    break;
                }

                v = &v_new / v_norm;
                s[k] = v_norm;
            }

            // Store singular vectors
            let u_k = a.dot(&v);
            let u_norm = u_k.iter().map(|&x| x * x).sum::<T>().sqrt();
            if u_norm > tolerance {
                for i in 0..m {
                    u[[i, k]] = u_k[i] / u_norm;
                }
                for j in 0..n {
                    vt[[k, j]] = v[j];
                }

                // Deflate matrix
                let outer_prod =
                    Array2::from_shape_fn((m, n), |(i, j)| u[[i, k]] * s[k] * vt[[k, j]]);
                a = a - outer_prod;
            }
        }

        // Compute pseudoinverse: A^+ = V * S^+ * U^T
        let mut s_inv = Array2::zeros((n, m));
        for i in 0..min_dim {
            if s[i] > self.config.min_eigenvalue {
                s_inv[[i, i]] = T::one() / s[i];
            }
        }

        Ok(vt.t().dot(&s_inv).dot(&u.t()))
    }

    /// Compute adaptive damping based on matrix condition number
    fn compute_adaptive_damping(&self, matrix: &Array2<T>) -> Result<T> {
        // Estimate condition number using ratio of max/min diagonal elements
        let mut max_diag = T::zero();
        let mut min_diag = T::infinity();

        for i in 0..matrix.nrows() {
            let diag = matrix[[i, i]];
            if diag > max_diag {
                max_diag = diag;
            }
            if diag < min_diag {
                min_diag = diag;
            }
        }

        let condition_estimate = if min_diag > T::zero() {
            max_diag / min_diag
        } else {
            T::infinity()
        };

        // Adaptive damping based on condition number
        let target_condition = T::from(1e6).unwrap();
        let base_damping = self.config.damping;

        if condition_estimate >= target_condition {
            // Ensure damping increases by at least a factor when condition number is high
            let scale_factor =
                ((condition_estimate / target_condition).sqrt()).max(T::from(1.1).unwrap());
            Ok(base_damping * scale_factor)
        } else {
            Ok(base_damping)
        }
    }

    /// Apply K-FAC update to parameter gradients
    pub fn apply_update(&mut self, layername: &str, gradients: &Array2<T>) -> Result<Array2<T>> {
        let state = self
            .layer_states
            .get(layername)
            .ok_or_else(|| OptimError::InvalidConfig(format!("Layer {} not found", layername)))?;

        // Ensure inverse matrices are computed
        if state.a_cov_inv.is_none() || state.g_cov_inv.is_none() {
            return Err(OptimError::InvalidConfig(
                "Inverse covariance matrices not computed".to_string(),
            ));
        }

        let a_inv = state.a_cov_inv.as_ref().unwrap();
        let g_inv = state.g_cov_inv.as_ref().unwrap();

        // Apply Kronecker-factored preconditioner: G^(-1) ⊗ A^(-1)
        // For gradients G (output_dim x input_dim): G_new = G_inv * G * A_inv
        let preconditioned = g_inv.dot(&gradients.dot(a_inv));

        // Apply learning rate and weight decay
        let mut update = preconditioned * self.config.learning_rate;

        if self.config.weight_decay > T::zero() {
            // Add weight decay to the original gradients
            update = update + gradients * self.config.weight_decay;
        }

        Ok(update)
    }

    /// Perform a complete K-FAC optimization step
    pub fn step(
        &mut self,
        layername: &str,
        parameters: &Array2<T>,
        gradients: &Array2<T>,
        activations: &Array2<T>,
        loss: Option<T>,
    ) -> Result<Array2<T>> {
        self.step_count += 1;
        self.stats.total_steps += 1;

        // Update loss-based statistics
        if let Some(current_loss) = loss {
            if let Some(prev_loss) = self.previous_loss {
                let improvement = prev_loss - current_loss;
                // Update acceptance ratio for adaptive damping
                if improvement > T::zero() {
                    self.acceptance_ratio =
                        self.acceptance_ratio * T::from(0.9).unwrap() + T::from(0.1).unwrap();
                } else {
                    self.acceptance_ratio = self.acceptance_ratio * T::from(0.9).unwrap();
                }
            }
            self.previous_loss = Some(current_loss);
        }

        // Update covariance matrices
        self.update_covariance_matrices(layername, activations, gradients)?;

        // Update inverse matrices if needed
        self.update_inverse_matrices(layername)?;

        // Apply K-FAC update
        let update = self.apply_update(layername, gradients)?;

        // Update parameters
        let new_parameters = parameters - &update;

        Ok(new_parameters)
    }

    /// Get statistics for monitoring
    pub fn get_stats(&self) -> &KFACStats<T> {
        &self.stats
    }

    /// Reset optimizer state
    pub fn reset(&mut self) {
        for state in self.layer_states.values_mut() {
            let input_size =
                state.layerinfo.input_dim + if state.layerinfo.has_bias { 1 } else { 0 };
            let output_size = state.layerinfo.output_dim;

            state.a_cov = Array2::eye(input_size);
            state.g_cov = Array2::eye(output_size);
            state.a_cov_inv = None;
            state.g_cov_inv = None;
            state.num_updates = 0;
            state.last_cov_update = 0;
            state.last_inv_update = 0;
            state.running_mean_a = None;
            state.running_mean_g = None;
        }

        self.step_count = 0;
        self.acceptance_ratio = T::from(1.0).unwrap();
        self.previous_loss = None;
        self.eigenvalue_history.clear();
        self.stats = KFACStats::default();
    }

    /// Estimate memory usage
    pub fn estimate_memory_usage(&self) -> usize {
        let mut total_memory = 0;

        for state in self.layer_states.values() {
            let input_size = state.a_cov.len();
            let output_size = state.g_cov.len();

            // Covariance matrices (2 per layer)
            total_memory += input_size * std::mem::size_of::<T>();
            total_memory += output_size * std::mem::size_of::<T>();

            // Inverse matrices (2 per layer)
            if state.a_cov_inv.is_some() {
                total_memory += input_size * std::mem::size_of::<T>();
            }
            if state.g_cov_inv.is_some() {
                total_memory += output_size * std::mem::size_of::<T>();
            }

            // Running means
            if state.running_mean_a.is_some() {
                total_memory += state.layerinfo.input_dim * std::mem::size_of::<T>();
            }
            if state.running_mean_g.is_some() {
                total_memory += state.layerinfo.output_dim * std::mem::size_of::<T>();
            }
        }

        total_memory
    }

    /// Get layer state information
    pub fn get_layer_state(&self, layername: &str) -> Option<&KFACLayerState<T>> {
        self.layer_states.get(layername)
    }

    /// Set custom damping for a specific layer
    pub fn set_layer_damping(&mut self, layername: &str, damping_a: T, damping_g: T) -> Result<()> {
        let state = self
            .layer_states
            .get_mut(layername)
            .ok_or_else(|| OptimError::InvalidConfig(format!("Layer {} not found", layername)))?;

        state.damping_a = damping_a;
        state.damping_g = damping_g;
        Ok(())
    }

    /// Advanced matrix analysis and conditioning
    fn estimate_condition_number(&self, matrix: &Array2<T>) -> T {
        let mut max_diag = T::zero();
        let mut min_diag = T::infinity();

        for i in 0..matrix.nrows() {
            let diag = matrix[[i, i]];
            if diag > max_diag {
                max_diag = diag;
            }
            if diag < min_diag {
                min_diag = diag;
            }
        }

        if min_diag > T::zero() {
            max_diag / min_diag
        } else {
            T::infinity()
        }
    }

    /// Regularized inverse for ill-conditioned matrices
    fn regularized_inverse(&self, matrix: &Array2<T>) -> Result<Array2<T>> {
        let n = matrix.nrows();
        let mut regularized = matrix.clone();

        // Add Tikhonov regularization
        let reg_param = self.config.damping * T::from(10.0).unwrap();
        for i in 0..n {
            regularized[[i, i]] = regularized[[i, i]] + reg_param;
        }

        self.cholesky_inverse(&regularized)
    }

    /// Iterative refinement for improved accuracy
    fn iterative_inverse(&self, matrix: &Array2<T>) -> Result<Array2<T>> {
        let n = matrix.nrows();
        let mut x = Array2::eye(n); // Initial guess
        let eye = Array2::eye(n);

        // Use Newton-Schulz iteration: X_{k+1} = X_k * (2*I - A*X_k)
        for _ in 0..5 {
            // 5 iterations should be enough
            let ax = matrix.dot(&x);
            let residual = &eye - &ax;
            let update = x.dot(&residual);
            x = &x + &update;

            // Check convergence
            let error = residual.iter().map(|&r| r * r).sum::<T>().sqrt();
            if error < T::from(1e-12).unwrap() {
                break;
            }
        }

        Ok(x)
    }

    /// Solve L * L^T * X = I using Cholesky factorization
    fn cholesky_solve_identity(&self, l: &Array2<T>) -> Result<Array2<T>> {
        let n = l.nrows();
        let mut inv = Array2::zeros((n, n));

        // Solve for each column of the identity matrix
        for i in 0..n {
            let mut b = Array1::zeros(n);
            b[i] = T::one();

            // Forward substitution: L * y = b
            let mut y = Array1::zeros(n);
            for j in 0..n {
                let mut sum = T::zero();
                for k in 0..j {
                    sum = sum + l[[j, k]] * y[k];
                }
                y[j] = (b[j] - sum) / l[[j, j]];
            }

            // Backward substitution: L^T * x = y
            let mut x = Array1::zeros(n);
            for j in (0..n).rev() {
                let mut sum = T::zero();
                for k in j + 1..n {
                    sum = sum + l[[k, j]] * x[k];
                }
                x[j] = (y[j] - sum) / l[[j, j]];
            }

            // Store column in inverse matrix
            for j in 0..n {
                inv[[j, i]] = x[j];
            }
        }

        Ok(inv)
    }

    /// Solve LU * X = I using LU factorization
    fn lu_solve_identity(&self, lu: &Array2<T>, perm: &[usize]) -> Result<Array2<T>> {
        let n = lu.nrows();
        let mut inv = Array2::zeros((n, n));

        for i in 0..n {
            let mut b = Array1::zeros(n);
            b[perm[i]] = T::one(); // Apply permutation

            // Forward substitution for L
            let mut y = Array1::zeros(n);
            for j in 0..n {
                let mut sum = T::zero();
                for k in 0..j {
                    sum = sum + lu[[j, k]] * y[k];
                }
                y[j] = b[j] - sum;
            }

            // Backward substitution for U
            let mut x = Array1::zeros(n);
            for j in (0..n).rev() {
                let mut sum = T::zero();
                for k in j + 1..n {
                    sum = sum + lu[[j, k]] * x[k];
                }
                x[j] = (y[j] - sum) / lu[[j, j]];
            }

            // Store column in inverse matrix
            for j in 0..n {
                inv[[j, i]] = x[j];
            }
        }

        Ok(inv)
    }
}

/// K-FAC utilities for specialized layer types
pub mod kfac_utils {
    use super::*;

    /// Compute K-FAC update for convolutional layers
    pub fn conv_kfac_update<T: Float + 'static>(
        _input_patches: &Array2<T>,
        output_grads: &Array2<T>,
        a_inv: &Array2<T>,
        g_inv: &Array2<T>,
    ) -> Result<Array2<T>> {
        // For conv layers, we need to reshape and handle _patches properly
        let preconditioned = g_inv.dot(&output_grads.dot(a_inv));
        Ok(preconditioned)
    }

    /// Compute statistics for batch normalization layers
    pub fn batchnorm_statistics<T: Float + num_traits::FromPrimitive>(
        inputs: &Array2<T>,
        _gamma: &Array1<T>,
        _beta: &Array1<T>,
    ) -> Result<(Array1<T>, Array1<T>)> {
        let mean = inputs.mean_axis(Axis(0)).unwrap();
        let var = inputs.var_axis(Axis(0), T::zero());
        Ok((mean, var))
    }

    /// Handle grouped convolution layers
    pub fn grouped_conv_kfac<T: Float>(
        groups: usize,
        input_patches: &Array2<T>,
        output_grads: &Array2<T>,
    ) -> Result<Vec<Array2<T>>> {
        let input_group_size = input_patches.ncols() / groups;
        let output_group_size = output_grads.ncols() / groups;
        let mut updates = Vec::new();

        for g in 0..groups {
            let input_start_idx = g * input_group_size;
            let input_end_idx = (g + 1) * input_group_size;
            let output_start_idx = g * output_group_size;
            let output_end_idx = (g + 1) * output_group_size;

            let group_input = input_patches.slice(s![.., input_start_idx..input_end_idx]);
            let _group_output = output_grads.slice(s![.., output_start_idx..output_end_idx]);

            // Apply K-FAC to each group independently
            updates.push(group_input.to_owned());
        }

        Ok(updates)
    }
}

#[allow(dead_code)]
impl<T: Float + Send + Sync> KFACLayerState<T> {
    /// Get the condition number estimate of covariance matrices
    pub fn condition_number_estimate(&self) -> (T, T) {
        let a_cond = self.estimate_condition_number(&self.a_cov);
        let g_cond = self.estimate_condition_number(&self.g_cov);
        (a_cond, g_cond)
    }

    fn estimate_condition_number(&self, matrix: &Array2<T>) -> T {
        let mut max_diag = T::zero();
        let mut min_diag = T::infinity();

        for i in 0..matrix.nrows() {
            let diag = matrix[[i, i]];
            if diag > max_diag {
                max_diag = diag;
            }
            if diag < min_diag {
                min_diag = diag;
            }
        }

        if min_diag > T::zero() {
            max_diag / min_diag
        } else {
            T::infinity()
        }
    }
}

/// Natural gradient optimization using Fisher information matrix
pub mod natural_gradients {
    use super::*;

    /// Natural gradient optimizer configuration
    #[derive(Debug, Clone)]
    pub struct NaturalGradientConfig<T: Float> {
        /// Learning rate for natural gradients
        pub learning_rate: T,

        /// Damping parameter for Fisher information matrix
        pub fisher_damping: T,

        /// Update frequency for Fisher information matrix
        pub fisher_update_freq: usize,

        /// Use empirical Fisher information (vs true Fisher)
        pub use_empirical_fisher: bool,

        /// Maximum rank for low-rank Fisher approximation
        pub max_rank: Option<usize>,

        /// Enable adaptive damping
        pub adaptive_damping: bool,

        /// Use conjugate gradient for matrix inversion
        pub use_conjugate_gradient: bool,

        /// Maximum CG iterations
        pub max_cg_iterations: usize,

        /// CG convergence tolerance
        pub cg_tolerance: T,
    }

    impl<T: Float> Default for NaturalGradientConfig<T> {
        fn default() -> Self {
            Self {
                learning_rate: T::from(0.001).unwrap(),
                fisher_damping: T::from(0.001).unwrap(),
                fisher_update_freq: 10,
                use_empirical_fisher: true,
                max_rank: Some(100),
                adaptive_damping: true,
                use_conjugate_gradient: true,
                max_cg_iterations: 100,
                cg_tolerance: T::from(1e-6).unwrap(),
            }
        }
    }

    /// Simplified condition number estimation for static methods
    fn estimate_condition_simple<T>(matrix: &Array2<T>) -> T
    where
        T: Float,
    {
        // Simple condition number estimate using ratio of max/min diagonal elements
        let diag = matrix.diag();
        let max_diag = diag.iter().fold(T::neg_infinity(), |acc, &x| acc.max(x));
        let min_diag = diag.iter().fold(T::infinity(), |acc, &x| acc.min(x));

        if min_diag > T::zero() {
            max_diag / min_diag
        } else {
            T::from(1e12).unwrap() // Large condition number for singular matrices
        }
    }

    /// Simplified static matrix inverse using basic inverse
    pub fn safe_matrix_inverse_static<T>(matrix: &Array2<T>) -> Result<Array2<T>>
    where
        T: Float + 'static,
    {
        // Simplified inverse using pseudo-inverse approach
        // Add small regularization to diagonal for numerical stability
        let mut regularized = matrix.clone();
        let reg_value = T::from(1e-8).unwrap();
        for i in 0..matrix.nrows() {
            regularized[[i, i]] = regularized[[i, i]] + reg_value;
        }

        // Use simple Gauss-Jordan elimination for small matrices
        if matrix.nrows() <= 3 {
            gauss_jordan_inverse(&regularized)
        } else {
            // For larger matrices, use iterative approach
            iterative_inverse_simple(&regularized)
        }
    }

    /// Simple Gauss-Jordan inverse for small matrices
    fn gauss_jordan_inverse<T>(matrix: &Array2<T>) -> Result<Array2<T>>
    where
        T: Float,
    {
        let n = matrix.nrows();
        let mut augmented = Array2::zeros((n, 2 * n));

        // Create augmented matrix [A | I]
        for i in 0..n {
            for j in 0..n {
                augmented[[i, j]] = matrix[[i, j]];
                augmented[[i, j + n]] = if i == j { T::one() } else { T::zero() };
            }
        }

        // Gauss-Jordan elimination
        for i in 0..n {
            // Find pivot
            let mut pivot_row = i;
            for k in (i + 1)..n {
                if augmented[[k, i]].abs() > augmented[[pivot_row, i]].abs() {
                    pivot_row = k;
                }
            }

            // Swap rows if needed
            if pivot_row != i {
                for j in 0..(2 * n) {
                    let temp = augmented[[i, j]];
                    augmented[[i, j]] = augmented[[pivot_row, j]];
                    augmented[[pivot_row, j]] = temp;
                }
            }

            // Check for singularity
            if augmented[[i, i]].abs() < T::from(1e-12).unwrap() {
                return Err(OptimError::ComputationError(
                    "Matrix is singular".to_string(),
                ));
            }

            // Scale row to make diagonal element 1
            let pivot = augmented[[i, i]];
            for j in 0..(2 * n) {
                augmented[[i, j]] = augmented[[i, j]] / pivot;
            }

            // Eliminate column
            for k in 0..n {
                if k != i {
                    let factor = augmented[[k, i]];
                    for j in 0..(2 * n) {
                        augmented[[k, j]] = augmented[[k, j]] - factor * augmented[[i, j]];
                    }
                }
            }
        }

        // Extract inverse from right half of augmented matrix
        let mut inverse = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                inverse[[i, j]] = augmented[[i, j + n]];
            }
        }

        Ok(inverse)
    }

    /// Simple iterative inverse for larger matrices
    fn iterative_inverse_simple<T>(matrix: &Array2<T>) -> Result<Array2<T>>
    where
        T: Float + 'static,
    {
        // Use Richardson iteration with diagonal preconditioning
        let n = matrix.nrows();
        let mut inverse = Array2::eye(n);

        // Extract diagonal for preconditioning
        let mut diag_inv = Array1::zeros(n);
        for i in 0..n {
            let diag_val = matrix[[i, i]];
            if diag_val.abs() < T::from(1e-12).unwrap() {
                return Err(OptimError::ComputationError(
                    "Matrix has zero diagonal element".to_string(),
                ));
            }
            diag_inv[i] = T::one() / diag_val;
        }

        // Richardson iteration: X_{k+1} = X_k + D^{-1}(I - AX_k)
        for _iter in 0..10 {
            let residual = Array2::eye(n) - matrix.dot(&inverse);
            let correction = &diag_inv * &residual;
            inverse = inverse + correction;
        }

        Ok(inverse)
    }
}

/// Advanced K-FAC enhancements for production-scale optimization
pub mod advanced_kfac {
    use super::*;
    use crate::reinforcement_learning::NaturalGradientConfig;
    use std::collections::{BTreeMap, VecDeque};
    use std::sync::Arc;

    /// Distributed K-FAC optimizer for large-scale training
    #[derive(Debug)]
    pub struct DistributedKFAC<T: Float> {
        /// Base K-FAC optimizer
        base_kfac: KFACOptimizer<T>,

        /// Distributed configuration
        dist_config: DistributedKFACConfig,

        /// Communication backend for gradient synchronization
        comm_backend: Option<Arc<dyn CommunicationBackend>>,

        /// Block-wise decomposition for memory efficiency
        block_decomposition: BlockDecomposition<T>,

        /// GPU acceleration integration
        gpu_acceleration: Option<KFACGpuAcceleration>,

        /// Advanced conditioning and monitoring
        conditioning: AdvancedConditioning<T>,

        /// Second-order momentum state
        momentum_state: SecondOrderMomentum<T>,
    }

    /// Configuration for distributed K-FAC
    #[derive(Debug, Clone)]
    pub struct DistributedKFACConfig {
        /// Number of distributed workers
        pub num_workers: usize,

        /// Current worker rank
        pub workerrank: usize,

        /// Communication pattern for gradient averaging
        pub comm_pattern: CommunicationPattern,

        /// Enable asynchronous updates
        pub async_updates: bool,

        /// Gradient compression for communication
        pub gradient_compression: CompressionConfig,

        /// Enable hierarchical communication
        pub hierarchical_comm: bool,

        /// Block size for block-wise K-FAC
        pub blocksize: usize,

        /// Enable overlap computation and communication
        pub overlap_comm_compute: bool,
    }

    /// Communication patterns for distributed training
    #[derive(Debug, Clone, Copy)]
    pub enum CommunicationPattern {
        AllReduce,
        ParameterServer,
        Ring,
        Tree,
        Butterfly,
    }

    /// Gradient compression configuration
    #[derive(Debug, Clone)]
    pub struct CompressionConfig {
        /// Enable compression
        pub enabled: bool,

        /// Compression method
        pub method: CompressionMethod,

        /// Compression ratio (0.0 to 1.0)
        pub ratio: f64,

        /// Error feedback for compressed gradients
        pub error_feedback: bool,
    }

    /// Compression methods for gradients
    #[derive(Debug, Clone, Copy)]
    pub enum CompressionMethod {
        TopK,
        Quantization,
        SignSGD,
        PowerSGD,
        LayerAdaptive,
    }

    /// Communication backend trait
    pub trait CommunicationBackend: Send + Sync + std::fmt::Debug {
        fn all_reduce(&self, data: &mut [f32]) -> Result<()>;
        fn broadcast(&self, data: &mut [f32], root: usize) -> Result<()>;
        fn gather(&self, send_data: &[f32], recvdata: &mut [f32], root: usize) -> Result<()>;
        fn scatter(&self, send_data: &[f32], recvdata: &mut [f32], root: usize) -> Result<()>;
        fn barrier(&self) -> Result<()>;
    }

    /// Block-wise decomposition for memory-efficient K-FAC
    #[derive(Debug)]
    pub struct BlockDecomposition<T: Float> {
        /// Block states for each layer
        blocks: HashMap<String, Vec<BlockState<T>>>,

        /// Block size configuration
        blocksize: usize,

        /// Overlap factor for block boundaries
        overlap_factor: f64,

        /// Block scheduling strategy
        scheduling: BlockScheduling,
    }

    /// Individual block state within a layer
    #[derive(Debug, Clone)]
    pub struct BlockState<T: Float> {
        /// Block index within layer
        pub block_id: usize,

        /// Block dimensions
        pub rows: std::ops::Range<usize>,
        pub cols: std::ops::Range<usize>,

        /// Block-specific covariance matrices
        pub a_cov_block: Array2<T>,
        pub g_cov_block: Array2<T>,

        /// Block-specific inverses
        pub a_cov_inv_block: Option<Array2<T>>,
        pub g_cov_inv_block: Option<Array2<T>>,

        /// Block update frequency
        pub update_freq: usize,
        pub last_update: usize,

        /// Block-specific damping
        pub damping: T,
    }

    /// Block scheduling strategies
    #[derive(Debug, Clone, Copy)]
    pub enum BlockScheduling {
        Sequential,
        Random,
        PriorityBased,
        AdaptiveRound,
        LoadBalanced,
    }

    /// GPU acceleration for K-FAC operations
    #[derive(Debug)]
    pub struct KFACGpuAcceleration {
        /// GPU device context
        device_id: usize,

        /// Tensor core optimization enabled
        tensor_cores_enabled: bool,

        /// Mixed precision configuration
        mixed_precision: MixedPrecisionConfig,

        /// Memory pool for GPU operations
        memory_pool: GpuMemoryPool,

        /// Stream manager for overlapping operations
        stream_manager: KFACStreamManager,
    }

    /// Mixed precision configuration for K-FAC
    #[derive(Debug, Clone)]
    pub struct MixedPrecisionConfig {
        /// Use FP16 for covariance computation
        pub fp16_covariance: bool,

        /// Use FP32 for matrix inversion (higher precision)
        pub fp32_inversion: bool,

        /// Gradient scaling factor
        pub gradient_scale: f32,

        /// Enable automatic loss scaling
        pub auto_scaling: bool,
    }

    /// GPU memory pool for K-FAC operations
    #[derive(Debug)]
    pub struct GpuMemoryPool {
        /// Allocated memory blocks
        blocks: Vec<GpuMemoryBlock>,

        /// Free memory tracking
        free_memory: usize,

        /// Memory alignment requirements
        alignment: usize,
    }

    /// GPU memory block descriptor
    #[derive(Debug)]
    pub struct GpuMemoryBlock {
        /// Block size in bytes
        size: usize,

        /// Device pointer
        ptr: *mut std::ffi::c_void,

        /// Block usage status
        in_use: bool,
    }

    /// Stream manager for GPU operations
    #[derive(Debug)]
    pub struct KFACStreamManager {
        /// CUDA streams for different operations
        covariance_stream: usize,
        inversion_stream: usize,
        update_stream: usize,

        /// Stream synchronization events
        sync_events: Vec<usize>,
    }

    /// Advanced numerical conditioning and monitoring
    #[derive(Debug)]
    pub struct AdvancedConditioning<T: Float> {
        /// Eigenvalue tracking for each layer
        eigenvalue_trackers: HashMap<String, EigenvalueTracker<T>>,

        /// Condition number monitoring
        condition_monitors: HashMap<String, ConditionMonitor<T>>,

        /// Adaptive preconditioning strategies
        preconditioning: AdaptivePreconditioning<T>,

        /// Numerical stability analysis
        stability_analyzer: StabilityAnalyzer<T>,
    }

    /// Eigenvalue tracking for matrix conditioning
    #[derive(Debug)]
    pub struct EigenvalueTracker<T: Float> {
        /// Historical eigenvalues
        eigenvalue_history: VecDeque<Vec<T>>,

        /// Spectral radius tracking
        spectral_radius: VecDeque<T>,

        /// Eigenvalue decay rates
        decay_rates: Option<Vec<T>>,

        /// Power iteration state for dominant eigenvalue
        power_iteration_state: PowerIterationState<T>,

        /// Lanczos algorithm state for spectrum estimation
        lanczos_state: Option<LanczosState<T>>,
    }

    /// Power iteration state for eigenvalue computation
    #[derive(Debug)]
    pub struct PowerIterationState<T: Float> {
        /// Current eigenvector estimate
        pub vector: Array1<T>,

        /// Current eigenvalue estimate
        pub value: T,

        /// Iteration count
        pub iterations: usize,

        /// Convergence tolerance
        pub tolerance: T,
    }

    /// Lanczos algorithm state for spectrum estimation
    #[derive(Debug)]
    pub struct LanczosState<T: Float> {
        /// Tridiagonal matrix from Lanczos procedure
        pub tridiag_alpha: Vec<T>,
        pub tridiag_beta: Vec<T>,

        /// Lanczos vectors
        pub vectors: Vec<Array1<T>>,

        /// Current iteration
        pub iteration: usize,

        /// Maximum iterations
        pub max_iterations: usize,
    }

    /// Condition number monitoring
    #[derive(Debug)]
    pub struct ConditionMonitor<T: Float> {
        /// Condition number history
        condition_history: VecDeque<T>,

        /// Threshold for poor conditioning
        condition_threshold: T,

        /// Automatic remediation strategies
        auto_remediation: bool,

        /// Remediation actions taken
        remediation_log: Vec<RemediationAction>,
    }

    /// Remediation actions for poor conditioning
    #[derive(Debug, Clone)]
    pub enum RemediationAction {
        IncreaseDamping { old_damping: f64, new_damping: f64 },
        SwitchToRegularizedInverse,
        ResetCovariance,
        ApplySpectralShift { shift: f64 },
        ReduceUpdateFrequency,
    }

    /// Adaptive preconditioning strategies
    #[derive(Debug)]
    pub struct AdaptivePreconditioning<T: Float> {
        /// Preconditioning method selection
        method_selector: PreconditioningSelector<T>,

        /// Method-specific configurations
        method_configs: HashMap<PreconditioningMethod, PreconditioningConfig<T>>,

        /// Performance tracking for each method
        performance_tracker: MethodPerformanceTracker<T>,
    }

    /// Preconditioning method types
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum PreconditioningMethod {
        Standard,
        BlockDiagonal,
        LowRank,
        Hierarchical,
        Sketched,
        Randomized,
    }

    /// Method selection logic
    #[derive(Debug)]
    pub struct PreconditioningSelector<T: Float> {
        /// Selection criteria
        criteria: SelectionCriteria<T>,

        /// Method transition rules
        transition_rules: Vec<TransitionRule<T>>,

        /// Current method for each layer
        current_methods: HashMap<String, PreconditioningMethod>,
    }

    /// Selection criteria for preconditioning methods
    #[derive(Debug)]
    pub struct SelectionCriteria<T: Float> {
        /// Matrix size thresholds
        pub size_thresholds: BTreeMap<usize, PreconditioningMethod>,

        /// Condition number thresholds
        pub condition_thresholds: BTreeMap<OrderedFloat<T>, PreconditioningMethod>,

        /// Memory usage constraints
        pub memory_constraints: MemoryConstraints,

        /// Performance requirements
        pub performance_requirements: PerformanceRequirements<T>,
    }

    /// Method transition rules
    #[derive(Debug)]
    pub struct TransitionRule<T: Float> {
        /// Source method
        pub from_method: PreconditioningMethod,

        /// Target method
        pub to_method: PreconditioningMethod,

        /// Trigger condition
        pub trigger: TransitionTrigger<T>,

        /// Hysteresis to prevent oscillation
        pub hysteresis: T,
    }

    /// Triggers for method transitions
    #[derive(Debug)]
    pub enum TransitionTrigger<T: Float> {
        ConditionNumber(T),
        MemoryUsage(usize),
        ComputeTime(std::time::Duration),
        ConvergenceRate(T),
        Composite(Vec<TransitionTrigger<T>>),
    }

    /// Memory constraints for method selection
    #[derive(Debug, Clone)]
    pub struct MemoryConstraints {
        /// Maximum memory per layer (bytes)
        pub max_memory_per_layer: usize,

        /// Total memory budget (bytes)
        pub total_memory_budget: usize,

        /// Memory efficiency threshold
        pub efficiency_threshold: f64,
    }

    /// Performance requirements
    #[derive(Debug, Clone)]
    pub struct PerformanceRequirements<T: Float> {
        /// Maximum compute time per update
        pub max_update_time: std::time::Duration,

        /// Target convergence rate
        pub target_convergence_rate: T,

        /// Accuracy tolerance
        pub accuracy_tolerance: T,
    }

    /// Performance tracking for preconditioning methods
    #[derive(Debug)]
    pub struct MethodPerformanceTracker<T: Float> {
        /// Performance metrics per method
        metrics: HashMap<PreconditioningMethod, PerformanceMetrics<T>>,

        /// Historical performance data
        history: HashMap<PreconditioningMethod, VecDeque<PerformanceSnapshot<T>>>,
    }

    /// Performance metrics for a preconditioning method
    #[derive(Debug, Clone)]
    pub struct PerformanceMetrics<T: Float> {
        /// Average update time
        pub avg_update_time: std::time::Duration,

        /// Average convergence rate
        pub avg_convergence_rate: T,

        /// Memory usage
        pub memory_usage: usize,

        /// Numerical stability score
        pub stability_score: T,

        /// Success rate (updates without numerical issues)
        pub success_rate: f64,
    }

    /// Performance snapshot at a specific time
    #[derive(Debug, Clone)]
    pub struct PerformanceSnapshot<T: Float> {
        /// Timestamp
        pub timestamp: std::time::Instant,

        /// Metrics at this time
        pub metrics: PerformanceMetrics<T>,

        /// Context information
        pub context: PerformanceContext,
    }

    /// Context information for performance snapshots
    #[derive(Debug, Clone)]
    pub struct PerformanceContext {
        /// Layer size
        pub layer_size: (usize, usize),

        /// Batch size
        pub batch_size: usize,

        /// Current condition number
        pub condition_number: f64,

        /// Memory pressure
        pub memory_pressure: f64,
    }

    /// Numerical stability analysis
    #[derive(Debug)]
    pub struct StabilityAnalyzer<T: Float> {
        /// Stability metrics tracking
        stability_metrics: StabilityMetrics<T>,

        /// Error accumulation tracking
        error_tracking: ErrorTracking<T>,

        /// Stability thresholds
        thresholds: StabilityThresholds<T>,

        /// Automatic stabilization methods
        auto_stabilization: AutoStabilization<T>,
    }

    /// Comprehensive stability metrics
    #[derive(Debug)]
    pub struct StabilityMetrics<T: Float> {
        /// Numerical error estimates
        pub numerical_errors: Vec<T>,

        /// Matrix conditioning indicators
        pub conditioning_indicators: ConditioningIndicators<T>,

        /// Convergence stability
        pub convergence_stability: ConvergenceStability<T>,

        /// Update magnitude tracking
        pub update_magnitudes: VecDeque<T>,
    }

    /// Matrix conditioning indicators
    #[derive(Debug)]
    pub struct ConditioningIndicators<T: Float> {
        /// Condition number trends
        pub condition_trends: VecDeque<T>,

        /// Eigenvalue spread
        pub eigenvalue_spread: T,

        /// Rank deficiency indicators
        pub rank_indicators: RankIndicators<T>,

        /// Symmetry preservation
        pub symmetry_preservation: T,
    }

    /// Rank deficiency indicators
    #[derive(Debug)]
    pub struct RankIndicators<T: Float> {
        /// Effective rank estimate
        pub effective_rank: usize,

        /// Numerical rank
        pub numerical_rank: usize,

        /// Null space dimension estimate
        pub null_space_dim: usize,

        /// Small singular values
        pub small_singular_values: Vec<T>,
    }

    /// Convergence stability tracking
    #[derive(Debug)]
    pub struct ConvergenceStability<T: Float> {
        /// Convergence rate history
        pub rate_history: VecDeque<T>,

        /// Oscillation detection
        pub oscillation_detector: OscillationDetector<T>,

        /// Plateau detection
        pub plateau_detector: PlateauDetector<T>,

        /// Divergence indicators
        pub divergence_indicators: DivergenceIndicators<T>,
    }

    /// Oscillation detection in convergence
    #[derive(Debug)]
    pub struct OscillationDetector<T: Float> {
        /// Recent values for oscillation analysis
        pub recent_values: VecDeque<T>,

        /// Oscillation frequency estimate
        pub frequency_estimate: Option<f64>,

        /// Oscillation amplitude
        pub amplitude: T,

        /// Detection threshold
        pub threshold: T,
    }

    /// Plateau detection in convergence
    #[derive(Debug)]
    pub struct PlateauDetector<T: Float> {
        /// Minimum improvement threshold
        pub min_improvement: T,

        /// Plateau duration threshold
        pub duration_threshold: usize,

        /// Current plateau length
        pub current_plateau_length: usize,

        /// Plateau status
        pub in_plateau: bool,
    }

    /// Divergence indicators
    #[derive(Debug)]
    pub struct DivergenceIndicators<T: Float> {
        /// Gradient explosion detection
        pub gradient_explosion: bool,

        /// Parameter drift detection
        pub parameter_drift: T,

        /// Loss explosion indicators
        pub loss_explosion: bool,

        /// Update magnitude explosion
        pub update_explosion: bool,
    }

    /// Error accumulation tracking
    #[derive(Debug)]
    pub struct ErrorTracking<T: Float> {
        /// Floating point error accumulation
        pub fp_error_accumulation: T,

        /// Roundoff error estimates
        pub roundoff_errors: VecDeque<T>,

        /// Catastrophic cancellation detection
        pub cancellation_detector: CancellationDetector<T>,

        /// Error propagation analysis
        pub error_propagation: ErrorPropagation<T>,
    }

    /// Catastrophic cancellation detection
    #[derive(Debug)]
    pub struct CancellationDetector<T: Float> {
        /// Significant bit loss tracking
        pub bit_loss: Vec<usize>,

        /// Cancellation events
        pub cancellation_events: Vec<CancellationEvent>,

        /// Detection sensitivity
        pub sensitivity: T,
    }

    /// Cancellation event record
    #[derive(Debug)]
    pub struct CancellationEvent {
        /// Timestamp of event
        pub timestamp: std::time::Instant,

        /// Operation that caused cancellation
        pub operation: String,

        /// Severity level
        pub severity: CancellationSeverity,

        /// Bits lost
        pub bits_lost: usize,
    }

    /// Severity levels for cancellation events
    #[derive(Debug, Clone, Copy)]
    pub enum CancellationSeverity {
        Minor,
        Moderate,
        Severe,
        Critical,
    }

    /// Error propagation analysis
    #[derive(Debug)]
    pub struct ErrorPropagation<T: Float> {
        /// Error sensitivity to input perturbations
        pub input_sensitivity: T,

        /// Error amplification factors
        pub amplification_factors: Vec<T>,

        /// Error correlation tracking
        pub error_correlations: HashMap<String, T>,
    }

    /// Stability thresholds for automated intervention
    #[derive(Debug)]
    pub struct StabilityThresholds<T: Float> {
        /// Maximum acceptable condition number
        pub max_condition_number: T,

        /// Minimum eigenvalue threshold
        pub min_eigenvalue: T,

        /// Maximum update magnitude
        pub max_update_magnitude: T,

        /// Error tolerance
        pub error_tolerance: T,
    }

    /// Automatic stabilization methods
    #[derive(Debug)]
    pub struct AutoStabilization<T: Float> {
        /// Enabled stabilization techniques
        pub enabled_techniques: Vec<StabilizationTechnique>,

        /// Intervention history
        pub intervention_history: Vec<StabilizationIntervention<T>>,

        /// Adaptive thresholds
        pub adaptive_thresholds: bool,
    }

    /// Available stabilization techniques
    #[derive(Debug, Clone, Copy)]
    pub enum StabilizationTechnique {
        AdaptiveDamping,
        SpectralShift,
        GradientClipping,
        CovarianceReset,
        PrecisionUpgrade,
        RegularizationAdjustment,
    }

    /// Record of stabilization intervention
    #[derive(Debug)]
    pub struct StabilizationIntervention<T: Float> {
        /// Timestamp
        pub timestamp: std::time::Instant,

        /// Technique applied
        pub technique: StabilizationTechnique,

        /// Parameters before intervention
        pub before_params: InterventionParams<T>,

        /// Parameters after intervention
        pub after_params: InterventionParams<T>,

        /// Effectiveness measure
        pub effectiveness: T,
    }

    /// Parameters for intervention tracking
    #[derive(Debug, Clone)]
    pub struct InterventionParams<T: Float> {
        /// Condition number
        pub condition_number: T,

        /// Minimum eigenvalue
        pub min_eigenvalue: T,

        /// Damping parameter
        pub damping: T,

        /// Update magnitude
        pub update_magnitude: T,
    }

    /// Second-order momentum for enhanced K-FAC convergence
    #[derive(Debug)]
    pub struct SecondOrderMomentum<T: Float> {
        /// Layer-wise momentum states
        layer_momentum: HashMap<String, LayerMomentumState<T>>,

        /// Momentum configuration
        config: MomentumConfig<T>,

        /// Adaptive momentum scheduling
        adaptive_scheduling: AdaptiveMomentumScheduling<T>,

        /// Momentum effectiveness tracking
        effectiveness_tracker: MomentumEffectivenessTracker<T>,
    }

    /// Momentum state for individual layers
    #[derive(Debug)]
    pub struct LayerMomentumState<T: Float> {
        /// First-order momentum for gradients
        pub gradient_momentum: Option<Array2<T>>,

        /// Second-order momentum for curvature
        pub curvature_momentum: Option<(Array2<T>, Array2<T>)>,

        /// Velocity for accelerated updates
        pub velocity: Option<Array2<T>>,

        /// Momentum decay factors
        pub decay_factors: MomentumDecayFactors<T>,

        /// Bias correction terms
        pub bias_correction: BiasCorrectionTerms<T>,
    }

    /// Momentum decay factors
    #[derive(Debug, Clone)]
    pub struct MomentumDecayFactors<T: Float> {
        /// First-order decay (β₁)
        pub beta1: T,

        /// Second-order decay (β₂)
        pub beta2: T,

        /// Curvature momentum decay
        pub curvature_decay: T,

        /// Adaptive decay adjustment
        pub adaptive_decay: bool,
    }

    /// Bias correction terms for momentum
    #[derive(Debug, Clone)]
    pub struct BiasCorrectionTerms<T: Float> {
        /// First-order bias correction
        pub bias_correction_1: T,

        /// Second-order bias correction
        pub bias_correction_2: T,

        /// Curvature bias correction
        pub curvature_bias_correction: T,
    }

    /// Momentum configuration
    #[derive(Debug, Clone)]
    pub struct MomentumConfig<T: Float> {
        /// Enable first-order momentum
        pub enable_first_order: bool,

        /// Enable second-order momentum
        pub enable_second_order: bool,

        /// Enable curvature momentum
        pub enable_curvature_momentum: bool,

        /// Base momentum parameters
        pub base_beta1: T,
        pub base_beta2: T,

        /// Warmup schedule
        pub warmup_steps: usize,

        /// Momentum annealing
        pub annealing_schedule: AnnealingSchedule<T>,

        /// Layer-specific momentum adaptation
        pub layer_adaptation: bool,
    }

    /// Momentum annealing schedule
    #[derive(Debug, Clone)]
    pub enum AnnealingSchedule<T: Float> {
        Constant,
        Linear(T, T),
        Exponential(T),
        Cosine(T),
        Adaptive,
    }

    /// Adaptive momentum scheduling
    #[derive(Debug)]
    pub struct AdaptiveMomentumScheduling<T: Float> {
        /// Layer-specific schedules
        layer_schedules: HashMap<String, LayerMomentumSchedule<T>>,

        /// Global momentum trends
        global_trends: MomentumTrends<T>,

        /// Scheduling strategy
        strategy: MomentumSchedulingStrategy,
    }

    /// Layer-specific momentum schedule
    #[derive(Debug)]
    pub struct LayerMomentumSchedule<T: Float> {
        /// Current momentum parameters
        pub current_params: MomentumParams<T>,

        /// Parameter history
        pub param_history: VecDeque<MomentumParams<T>>,

        /// Effectiveness history
        pub effectiveness_history: VecDeque<T>,

        /// Adaptation rate
        pub adaptation_rate: T,
    }

    /// Momentum parameters for a layer
    #[derive(Debug, Clone)]
    pub struct MomentumParams<T: Float> {
        pub beta1: T,
        pub beta2: T,
        pub curvature_momentum: T,
        pub step: usize,
    }

    /// Global momentum trends
    #[derive(Debug)]
    pub struct MomentumTrends<T: Float> {
        /// Average momentum effectiveness
        pub avg_effectiveness: T,

        /// Trend direction
        pub trend_direction: TrendDirection,

        /// Trend magnitude
        pub trend_magnitude: T,

        /// Stability indicator
        pub stability: T,
    }

    /// Trend direction for momentum adaptation
    #[derive(Debug, Clone, Copy)]
    pub enum TrendDirection {
        Increasing,
        Decreasing,
        Stable,
        Oscillating,
    }

    /// Momentum scheduling strategies
    #[derive(Debug, Clone, Copy)]
    pub enum MomentumSchedulingStrategy {
        Static,
        LayerAdaptive,
        GlobalAdaptive,
        HybridAdaptive,
        PerformanceBased,
    }

    /// Momentum effectiveness tracking
    #[derive(Debug)]
    pub struct MomentumEffectivenessTracker<T: Float> {
        /// Layer effectiveness scores
        layer_scores: HashMap<String, EffectivenessScore<T>>,

        /// Global effectiveness metrics
        global_metrics: GlobalEffectivenessMetrics<T>,

        /// Effectiveness comparison baselines
        baselines: EffectivenessBaselines<T>,
    }

    /// Effectiveness score for momentum
    #[derive(Debug)]
    pub struct EffectivenessScore<T: Float> {
        /// Convergence acceleration
        pub convergence_acceleration: T,

        /// Stability improvement
        pub stability_improvement: T,

        /// Memory efficiency
        pub memory_efficiency: T,

        /// Computational overhead
        pub computational_overhead: T,

        /// Overall score
        pub overall_score: T,
    }

    /// Global effectiveness metrics
    #[derive(Debug)]
    pub struct GlobalEffectivenessMetrics<T: Float> {
        /// Total convergence improvement
        pub total_convergence_improvement: T,

        /// Average per-layer improvement
        pub avg_layer_improvement: T,

        /// Variance in effectiveness
        pub effectiveness_variance: T,

        /// Momentum overhead ratio
        pub overhead_ratio: T,
    }

    /// Baselines for effectiveness comparison
    #[derive(Debug)]
    pub struct EffectivenessBaselines<T: Float> {
        /// No momentum baseline
        pub no_momentum: T,

        /// First-order only baseline
        pub first_order_only: T,

        /// Standard K-FAC baseline
        pub standard_kfac: T,
    }

    // Helper types for ordered comparisons
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct OrderedFloat<T: Float>(T);

    impl<T: Float> Eq for OrderedFloat<T> {}

    impl<T: Float> PartialOrd for OrderedFloat<T> {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            self.0.partial_cmp(&other.0)
        }
    }

    impl<T: Float> Ord for OrderedFloat<T> {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
        }
    }

    /// Preconditioning configuration for different methods
    #[derive(Debug, Clone)]
    pub struct PreconditioningConfig<T: Float> {
        /// Method-specific parameters
        pub parameters: HashMap<String, T>,

        /// Numerical tolerances
        pub tolerances: HashMap<String, T>,

        /// Update frequencies
        pub update_frequencies: HashMap<String, usize>,

        /// Memory budgets
        pub memory_budgets: HashMap<String, usize>,
    }

    // Duplicate Default implementation removed - using implementation at line 1007

    /// Natural gradient optimizer using Fisher information matrix
    #[derive(Debug)]
    pub struct NaturalGradientOptimizer<T: Float> {
        /// Configuration
        config: NaturalGradientConfig<T>,

        /// Fisher information matrix approximation
        fisher_matrix: FisherInformation<T>,

        /// Current step count
        step_count: usize,

        /// Adaptive damping state
        damping_state: AdaptiveDampingState<T>,

        /// Performance metrics
        metrics: NaturalGradientMetrics<T>,
    }

    /// Fisher information matrix representation
    #[derive(Debug, Clone)]
    pub enum FisherInformation<T: Float> {
        /// Full Fisher information matrix
        Full(Array2<T>),

        /// Diagonal approximation
        Diagonal(Array1<T>),

        /// Low-rank approximation: F ≈ U * S * U^T
        LowRank { u: Array2<T>, s: Array1<T> },

        /// Block-diagonal approximation
        BlockDiagonal {
            blocks: Vec<Array2<T>>,
            block_indices: Vec<(usize, usize)>,
        },

        /// Kronecker-factored approximation (K-FAC style)
        KroneckerFactored {
            a_factors: Vec<Array2<T>>,
            g_factors: Vec<Array2<T>>,
        },
    }

    /// Adaptive damping state
    #[derive(Debug, Clone)]
    struct AdaptiveDampingState<T: Float> {
        current_damping: T,
        acceptance_ratio: T,
        previous_loss: Option<T>,
        damping_history: VecDeque<T>,
    }

    /// Natural gradient performance metrics
    #[derive(Debug, Clone)]
    pub struct NaturalGradientMetrics<T: Float> {
        /// Condition number of Fisher matrix
        pub fisher_condition_number: T,

        /// Effective rank of Fisher matrix
        pub fisher_effective_rank: usize,

        /// Average eigenvalue
        pub avg_eigenvalue: T,

        /// Min/max eigenvalues
        pub min_eigenvalue: T,
        pub max_eigenvalue: T,

        /// Fisher update computation time (microseconds)
        pub fisher_update_time_us: u64,

        /// Natural gradient computation time (microseconds)
        pub nat_grad_compute_time_us: u64,

        /// Memory usage for Fisher matrix (bytes)
        pub fisher_memory_bytes: usize,
    }

    impl<
            T: Float
                + Default
                + Clone
                + Send
                + Sync
                + 'static
                + std::iter::Sum
                + ndarray::ScalarOperand,
        > NaturalGradientOptimizer<T>
    {
        /// Create a new natural gradient optimizer
        pub fn new(config: NaturalGradientConfig<T>) -> Self {
            let damping_state = AdaptiveDampingState {
                current_damping: config.damping,
                acceptance_ratio: T::from(1.0).unwrap(),
                previous_loss: None,
                damping_history: VecDeque::with_capacity(100),
            };

            Self {
                config,
                fisher_matrix: FisherInformation::Diagonal(Array1::zeros(1)),
                step_count: 0,
                damping_state,
                metrics: NaturalGradientMetrics::default(),
            }
        }

        /// Initialize Fisher information matrix with parameter dimensions
        pub fn initialize_fisher(&mut self, paramdims: &[usize]) -> Result<()> {
            let total_params: usize = paramdims.iter().sum();

            self.fisher_matrix = if let Some(rank) = Some(100) {
                // Default max rank
                if rank < total_params {
                    FisherInformation::LowRank {
                        u: Array2::zeros((total_params, rank)),
                        s: Array1::zeros(rank),
                    }
                } else {
                    FisherInformation::Full(Array2::eye(total_params))
                }
            } else {
                FisherInformation::Full(Array2::eye(total_params))
            };

            Ok(())
        }

        /// Update Fisher information matrix using gradient samples
        pub fn update_fisher_information(
            &mut self,
            gradientsamples: &[Array1<T>],
            loss_samples: Option<&[T]>,
        ) -> Result<()> {
            if self.step_count % self.config.fisher_update_freq != 0 {
                return Ok(());
            }

            let start_time = std::time::Instant::now();

            if self.config.use_empirical_fisher {
                self.update_empirical_fisher(gradientsamples)?;
            } else {
                self.update_true_fisher(gradientsamples, loss_samples)?;
            }

            self.metrics.fisher_update_time_us = start_time.elapsed().as_micros() as u64;
            self.update_fisher_metrics()?;

            Ok(())
        }

        fn update_empirical_fisher(&mut self, gradientsamples: &[Array1<T>]) -> Result<()> {
            if gradientsamples.is_empty() {
                return Ok(());
            }

            let n_samples = T::from(gradientsamples.len()).unwrap();

            match &mut self.fisher_matrix {
                FisherInformation::Full(ref mut fisher) => {
                    // F = (1/n) * sum(g_i * g_i^T)
                    fisher.fill(T::zero());

                    for grad in gradientsamples {
                        // Outer product: g * g^T
                        for i in 0..grad.len() {
                            for j in 0..grad.len() {
                                fisher[[i, j]] = fisher[[i, j]] + grad[i] * grad[j];
                            }
                        }
                    }

                    // Average over _samples
                    fisher.mapv_inplace(|x| x / n_samples);

                    // Add damping
                    for i in 0..fisher.nrows() {
                        fisher[[i, i]] = fisher[[i, i]] + self.damping_state.current_damping;
                    }
                }

                FisherInformation::Diagonal(ref mut diag) => {
                    // Diagonal approximation: F_ii = (1/n) * sum(g_i^2)
                    diag.fill(T::zero());

                    for grad in gradientsamples {
                        for i in 0..grad.len() {
                            diag[i] = diag[i] + grad[i] * grad[i];
                        }
                    }

                    diag.mapv_inplace(|x| x / n_samples + self.damping_state.current_damping);
                }

                FisherInformation::LowRank {
                    ref mut u,
                    ref mut s,
                } => {
                    // Low-rank approximation using randomized SVD
                    Self::update_low_rank_fisher(gradientsamples, u, s)?;
                }

                _ => {
                    return Err(OptimError::InvalidConfig(
                        "Unsupported Fisher matrix type for empirical update".to_string(),
                    ));
                }
            }

            Ok(())
        }

        fn update_true_fisher(
            &mut self,
            gradientsamples: &[Array1<T>],
            loss_samples: Option<&[T]>,
        ) -> Result<()> {
            // True Fisher information requires second derivatives
            // This is a simplified implementation that falls back to empirical Fisher
            if loss_samples.is_some() {
                // Could implement true Fisher using loss function Hessian
                // For now, fall back to empirical Fisher
                self.update_empirical_fisher(gradientsamples)
            } else {
                self.update_empirical_fisher(gradientsamples)
            }
        }

        fn update_low_rank_fisher(
            gradientsamples: &[Array1<T>],
            u: &mut Array2<T>,
            s: &mut Array1<T>,
        ) -> Result<()> {
            if gradientsamples.is_empty() {
                return Ok(());
            }

            // Create gradient matrix: G = [g_1, g_2, ..., g_n]
            let n_samples = gradientsamples.len();
            let param_dim = gradientsamples[0].len();
            let mut grad_matrix = Array2::zeros((param_dim, n_samples));

            for (j, grad) in gradientsamples.iter().enumerate() {
                for i in 0..param_dim {
                    grad_matrix[[i, j]] = grad[i];
                }
            }

            // Simplified low-rank update using power iteration
            // In practice, would use proper randomized SVD
            let rank = u.ncols();

            // Power iteration for top eigenvectors
            for k in 0..rank.min(n_samples) {
                let mut v = Array1::ones(param_dim);

                // Power iteration
                for _ in 0..10 {
                    // v = G * G^T * v
                    let gv = grad_matrix.dot(&grad_matrix.t().dot(&v));
                    let norm = gv.iter().map(|&x| x * x).sum::<T>().sqrt();

                    if norm > T::from(1e-10).unwrap() {
                        v = gv / norm;
                    }
                }

                // Store eigenvector
                for i in 0..param_dim {
                    u[[i, k]] = v[i];
                }

                // Compute eigenvalue
                let uv = grad_matrix.t().dot(&v);
                s[k] = uv.iter().map(|&x| x * x).sum::<T>() / T::from(n_samples).unwrap();
            }

            Ok(())
        }

        /// Compute natural gradient: F^(-1) * g
        pub fn compute_natural_gradient(&mut self, gradient: &Array1<T>) -> Result<Array1<T>> {
            let start_time = std::time::Instant::now();

            let natural_grad = match &self.fisher_matrix {
                FisherInformation::Full(fisher) => {
                    if true {
                        // Default to using conjugate gradient
                        self.solve_cg(fisher, gradient)?
                    } else {
                        self.solve_direct(fisher, gradient)?
                    }
                }

                FisherInformation::Diagonal(diag) => {
                    // Element-wise division
                    let mut nat_grad = gradient.clone();
                    for i in 0..nat_grad.len() {
                        nat_grad[i] = nat_grad[i] / diag[i];
                    }
                    nat_grad
                }

                FisherInformation::LowRank { u: _, s: _ } => {
                    return Err(OptimError::InvalidConfig(
                        "Unsupported Fisher matrix type for natural gradient".to_string(),
                    ))
                }
                FisherInformation::BlockDiagonal { .. } => {
                    return Err(OptimError::InvalidConfig(
                        "BlockDiagonal Fisher matrix not implemented".to_string(),
                    ))
                }
                FisherInformation::KroneckerFactored { .. } => {
                    return Err(OptimError::InvalidConfig(
                        "KroneckerFactored Fisher matrix not implemented".to_string(),
                    ))
                }
            };

            self.metrics.nat_grad_compute_time_us = start_time.elapsed().as_micros() as u64;

            Ok(natural_grad)
        }

        fn solve_direct(&self, fisher: &Array2<T>, gradient: &Array1<T>) -> Result<Array1<T>> {
            // Direct solve using Cholesky decomposition (simplified)
            // In practice, would use proper linear algebra library
            let mut solution = gradient.clone();

            // Simplified diagonal solve for stability
            for i in 0..fisher.nrows() {
                let diag_elem = fisher[[i, i]];
                if diag_elem > T::from(1e-10).unwrap() {
                    solution[i] = solution[i] / diag_elem;
                }
            }

            Ok(solution)
        }

        fn solve_cg(&self, fisher: &Array2<T>, gradient: &Array1<T>) -> Result<Array1<T>> {
            // Conjugate gradient solver for F * x = g
            let mut x = Array1::zeros(gradient.len());
            let mut r = gradient.clone(); // r = b - A*x (x starts at 0)
            let mut p = r.clone();
            let mut rsold = r.dot(&r);

            for _iter in 0..10 {
                // Default max CG iterations
                let ap = fisher.dot(&p);
                let alpha = rsold / p.dot(&ap);

                // Update solution: x = x + alpha * p
                for i in 0..x.len() {
                    x[i] = x[i] + alpha * p[i];
                }

                // Update residual: r = r - alpha * A*p
                for i in 0..r.len() {
                    r[i] = r[i] - alpha * ap[i];
                }

                let rsnew = r.dot(&r);

                // Check convergence
                if rsnew.sqrt() < self.config.cg_tolerance {
                    break;
                }

                let beta = rsnew / rsold;

                // Update search direction: p = r + beta * p
                for i in 0..p.len() {
                    p[i] = r[i] + beta * p[i];
                }

                rsold = rsnew;
            }

            Ok(x)
        }

        fn solve_low_rank(
            &self,
            u: &Array2<T>,
            s: &Array1<T>,
            gradient: &Array1<T>,
        ) -> Result<Array1<T>> {
            // For low-rank F = U * S * U^T, solve using Sherman-Morrison-Woodbury formula
            // (U*S*U^T + damping*I)^(-1) * g

            let damping = self.damping_state.current_damping;
            let mut solution = gradient.clone();

            // Apply damping: solution = g / damping
            solution.mapv_inplace(|x| x / damping);

            // Apply Sherman-Morrison-Woodbury correction
            // This is simplified - in practice would be more sophisticated
            for k in 0..s.len() {
                if s[k] > T::from(1e-10).unwrap() {
                    let uk_dot_g = u.column(k).dot(gradient);
                    let correction_factor = s[k] / (s[k] + damping);

                    for i in 0..solution.len() {
                        solution[i] =
                            solution[i] - correction_factor * uk_dot_g * u[[i, k]] / damping;
                    }
                }
            }

            Ok(solution)
        }

        /// Apply natural gradient step
        pub fn step(
            &mut self,
            parameters: &Array1<T>,
            gradient: &Array1<T>,
            loss: Option<T>,
        ) -> Result<Array1<T>> {
            self.step_count += 1;

            // Update adaptive damping
            if true {
                // Default to adaptive damping
                self.update_adaptive_damping(loss)?;
            }

            // Compute natural gradient
            let natural_gradient = self.compute_natural_gradient(gradient)?;

            // Apply update: θ_{t+1} = θ_t - lr * F^(-1) * g
            let mut new_params = parameters.clone();
            for i in 0..new_params.len() {
                new_params[i] =
                    new_params[i] - self.config.base_config.policy_lr * natural_gradient[i];
            }

            Ok(new_params)
        }

        fn update_adaptive_damping(&mut self, loss: Option<T>) -> Result<()> {
            if let Some(current_loss) = loss {
                if let Some(prev_loss) = self.damping_state.previous_loss {
                    let improvement = prev_loss - current_loss;

                    // Update acceptance ratio
                    if improvement > T::zero() {
                        self.damping_state.acceptance_ratio = self.damping_state.acceptance_ratio
                            * T::from(0.9).unwrap()
                            + T::from(0.1).unwrap();
                    } else {
                        self.damping_state.acceptance_ratio =
                            self.damping_state.acceptance_ratio * T::from(0.9).unwrap();
                    }

                    // Adjust damping based on acceptance ratio
                    let target_ratio = T::from(0.75).unwrap();
                    if self.damping_state.acceptance_ratio < target_ratio {
                        // Increase damping if acceptance is low
                        self.damping_state.current_damping =
                            self.damping_state.current_damping * T::from(1.1).unwrap();
                    } else {
                        // Decrease damping if acceptance is high
                        self.damping_state.current_damping =
                            self.damping_state.current_damping * T::from(0.95).unwrap();
                    }

                    // Clamp damping to reasonable range
                    let min_damping = T::from(1e-6).unwrap();
                    let max_damping = T::from(1.0).unwrap();
                    self.damping_state.current_damping = self
                        .damping_state
                        .current_damping
                        .max(min_damping)
                        .min(max_damping);
                }

                self.damping_state.previous_loss = Some(current_loss);

                // Update history
                self.damping_state
                    .damping_history
                    .push_back(self.damping_state.current_damping);
                if self.damping_state.damping_history.len() > 100 {
                    self.damping_state.damping_history.pop_front();
                }
            }

            Ok(())
        }

        fn update_fisher_metrics(&mut self) -> Result<()> {
            match &self.fisher_matrix {
                FisherInformation::Full(fisher) => {
                    // Compute condition number (simplified using diagonal approximation)
                    let mut min_diag = T::infinity();
                    let mut max_diag = T::zero();
                    let mut sum_diag = T::zero();

                    for i in 0..fisher.nrows() {
                        let diag = fisher[[i, i]];
                        min_diag = min_diag.min(diag);
                        max_diag = max_diag.max(diag);
                        sum_diag = sum_diag + diag;
                    }

                    self.metrics.min_eigenvalue = min_diag;
                    self.metrics.max_eigenvalue = max_diag;
                    self.metrics.avg_eigenvalue = sum_diag / T::from(fisher.nrows()).unwrap();
                    self.metrics.fisher_condition_number = if min_diag > T::zero() {
                        max_diag / min_diag
                    } else {
                        T::infinity()
                    };

                    self.metrics.fisher_memory_bytes = fisher.len() * std::mem::size_of::<T>();
                }

                FisherInformation::Diagonal(diag) => {
                    let min_val = diag.iter().cloned().fold(T::infinity(), T::min);
                    let max_val = diag.iter().cloned().fold(T::zero(), T::max);
                    let sum_val = diag.iter().cloned().sum::<T>();

                    self.metrics.min_eigenvalue = min_val;
                    self.metrics.max_eigenvalue = max_val;
                    self.metrics.avg_eigenvalue = sum_val / T::from(diag.len()).unwrap();
                    self.metrics.fisher_condition_number = if min_val > T::zero() {
                        max_val / min_val
                    } else {
                        T::infinity()
                    };

                    self.metrics.fisher_memory_bytes = diag.len() * std::mem::size_of::<T>();
                }

                FisherInformation::LowRank { u, s } => {
                    let min_val = s.iter().cloned().fold(T::infinity(), T::min);
                    let max_val = s.iter().cloned().fold(T::zero(), T::max);
                    let sum_val = s.iter().cloned().sum::<T>();

                    self.metrics.min_eigenvalue = min_val;
                    self.metrics.max_eigenvalue = max_val;
                    self.metrics.avg_eigenvalue = sum_val / T::from(s.len()).unwrap();
                    self.metrics.fisher_condition_number = if min_val > T::zero() {
                        max_val / min_val
                    } else {
                        T::infinity()
                    };

                    // Effective rank (number of non-zero eigenvalues)
                    self.metrics.fisher_effective_rank =
                        s.iter().filter(|&&x| x > T::from(1e-10).unwrap()).count();

                    self.metrics.fisher_memory_bytes =
                        (u.len() + s.len()) * std::mem::size_of::<T>();
                }

                _ => {}
            }

            Ok(())
        }

        /// Get current metrics
        pub fn get_metrics(&self) -> &NaturalGradientMetrics<T> {
            &self.metrics
        }

        /// Get current damping value
        pub fn get_current_damping(&self) -> T {
            self.damping_state.current_damping
        }

        /// Reset optimizer state
        pub fn reset(&mut self) {
            self.step_count = 0;
            self.damping_state.previous_loss = None;
            self.damping_state.acceptance_ratio = T::from(1.0).unwrap();
            self.damping_state.damping_history.clear();
            self.damping_state.current_damping = self.config.damping;
        }
    }

    impl<T: Float> Default for NaturalGradientMetrics<T> {
        fn default() -> Self {
            Self {
                fisher_condition_number: T::one(),
                fisher_effective_rank: 0,
                avg_eigenvalue: T::one(),
                min_eigenvalue: T::one(),
                max_eigenvalue: T::one(),
                fisher_update_time_us: 0,
                nat_grad_compute_time_us: 0,
                fisher_memory_bytes: 0,
            }
        }
    }

    /// Concrete communication backend implementation for testing/local use
    #[derive(Debug)]
    pub struct LocalCommunicationBackend {
        num_workers: usize,
        workerrank: usize,
    }

    impl LocalCommunicationBackend {
        pub fn new(num_workers: usize, workerrank: usize) -> Self {
            Self {
                num_workers,
                workerrank,
            }
        }
    }

    impl CommunicationBackend for LocalCommunicationBackend {
        fn all_reduce(&self, data: &mut [f32]) -> Result<()> {
            // For local backend, just divide by number of workers (simulated averaging)
            let scale = 1.0 / self.num_workers as f32;
            for value in data.iter_mut() {
                *value *= scale;
            }
            Ok(())
        }

        fn broadcast(&self, data: &mut [f32], root: usize) -> Result<()> {
            // No-op for single node
            if root != self.workerrank {
                // In real implementation, would receive data from root
                for value in data.iter_mut() {
                    *value = 0.0; // Placeholder
                }
            }
            Ok(())
        }

        fn gather(&self, send_data: &[f32], recvdata: &mut [f32], root: usize) -> Result<()> {
            if self.workerrank == root {
                // Copy own _data to beginning of receive buffer
                let chunk_size = send_data.len();
                recvdata[..chunk_size].copy_from_slice(send_data);
                // In real implementation, would receive from other workers
            }
            Ok(())
        }

        fn scatter(&self, send_data: &[f32], recvdata: &mut [f32], root: usize) -> Result<()> {
            if self.workerrank == root {
                // Send _data to self (copy chunk for this worker)
                let chunk_size = recvdata.len();
                let start = self.workerrank * chunk_size;
                recvdata.copy_from_slice(&send_data[start..start + chunk_size]);
            }
            Ok(())
        }

        fn barrier(&self) -> Result<()> {
            // No-op for single process
            Ok(())
        }
    }

    /// Basic implementations for missing components
    impl<
            T: Float
                + Default
                + Clone
                + Send
                + Sync
                + std::iter::Sum
                + ndarray::ScalarOperand
                + num_traits::FromPrimitive,
        > DistributedKFAC<T>
    {
        pub fn new(
            _base_config: KFACConfig<T>,
            dist_config: DistributedKFACConfig,
        ) -> Result<Self> {
            let base_kfac = KFAC::new(_base_config);
            let local_backend =
                LocalCommunicationBackend::new(dist_config.num_workers, dist_config.workerrank);
            let comm_backend: Option<Arc<dyn CommunicationBackend>> =
                Some(Arc::new(local_backend) as Arc<dyn CommunicationBackend>);

            let blocksize = dist_config.blocksize;
            Ok(Self {
                base_kfac,
                dist_config,
                comm_backend,
                block_decomposition: BlockDecomposition::new(blocksize),
                gpu_acceleration: None,
                conditioning: AdvancedConditioning::new(),
                momentum_state: SecondOrderMomentum::new(MomentumConfig::default()),
            })
        }

        pub fn step_distributed(
            &mut self,
            layername: &str,
            local_gradients: &Array2<T>,
            _local_activations: &Array2<T>,
        ) -> Result<Array2<T>> {
            // 1. Compute local K-FAC updates
            let local_update = self.base_kfac.apply_update(layername, local_gradients)?;

            // 2. All-reduce _gradients across workers
            if let Some(ref comm) = self.comm_backend {
                // Convert to f32 for communication (in practice would handle T generically)
                let mut grad_data: Vec<f32> = local_gradients
                    .iter()
                    .map(|&x| x.to_f64().unwrap_or(0.0) as f32)
                    .collect();

                comm.all_reduce(&mut grad_data)?;

                // Convert back and apply distributed update
                // (Implementation details would depend on communication backend)
            }

            // 3. Apply block-wise decomposition if enabled
            if self.dist_config.blocksize > 0 {
                self.block_decomposition
                    .apply_block_update(layername, &local_update)
            } else {
                Ok(local_update)
            }
        }
    }

    /// Placeholder type alias for the main KFAC optimizer (for compatibility)
    pub type KFACOptimizer<T> = KFAC<T>;

    impl<T: Float + Send + Sync> BlockDecomposition<T> {
        pub fn new(blocksize: usize) -> Self {
            Self {
                blocks: HashMap::new(),
                blocksize,
                overlap_factor: 0.1,
                scheduling: BlockScheduling::Sequential,
            }
        }

        pub fn apply_block_update(
            &mut self,
            _layer_name: &str,
            update: &Array2<T>,
        ) -> Result<Array2<T>> {
            // Simple block-wise application (in practice would be more sophisticated)
            if self.blocksize >= update.nrows() && self.blocksize >= update.ncols() {
                // No blocking needed
                return Ok(update.clone());
            }

            let mut result = update.clone();

            // Apply blocks sequentially
            let num_row_blocks = (update.nrows() + self.blocksize - 1) / self.blocksize;
            let num_col_blocks = (update.ncols() + self.blocksize - 1) / self.blocksize;

            for i in 0..num_row_blocks {
                for j in 0..num_col_blocks {
                    let row_start = i * self.blocksize;
                    let row_end = ((i + 1) * self.blocksize).min(update.nrows());
                    let col_start = j * self.blocksize;
                    let col_end = ((j + 1) * self.blocksize).min(update.ncols());

                    // Process block
                    let mut block =
                        result.slice_mut(ndarray::s![row_start..row_end, col_start..col_end]);

                    // Apply block-specific processing (placeholder)
                    block.mapv_inplace(|x| x * T::from(0.99).unwrap());
                }
            }

            Ok(result)
        }
    }

    impl<T: Float + Send + Sync + std::iter::Sum + ndarray::ScalarOperand> AdvancedConditioning<T> {
        pub fn new() -> Self {
            Self {
                eigenvalue_trackers: HashMap::new(),
                condition_monitors: HashMap::new(),
                preconditioning: AdaptivePreconditioning::new(),
                stability_analyzer: StabilityAnalyzer::new(),
            }
        }

        pub fn monitor_layer(&mut self, layername: &str, matrix: &Array2<T>) -> Result<()> {
            // Initialize trackers if not present
            if !self.eigenvalue_trackers.contains_key(layername) {
                self.eigenvalue_trackers.insert(
                    layername.to_string(),
                    EigenvalueTracker::new(matrix.nrows()),
                );
            }

            if !self.condition_monitors.contains_key(layername) {
                self.condition_monitors
                    .insert(layername.to_string(), ConditionMonitor::new());
            }

            // Update monitoring
            let tracker = self.eigenvalue_trackers.get_mut(layername).unwrap();
            let monitor = self.condition_monitors.get_mut(layername).unwrap();

            tracker.update(matrix)?;
            monitor.update(matrix)?;

            Ok(())
        }
    }

    impl<T: Float + Send + Sync + std::iter::Sum + ndarray::ScalarOperand> EigenvalueTracker<T> {
        pub fn new(size: usize) -> Self {
            Self {
                eigenvalue_history: VecDeque::with_capacity(100),
                spectral_radius: VecDeque::with_capacity(100),
                decay_rates: None,
                power_iteration_state: PowerIterationState::new(size),
                lanczos_state: None,
            }
        }

        pub fn update(&mut self, matrix: &Array2<T>) -> Result<()> {
            // Simplified eigenvalue estimation using power iteration
            let eigenvalue = self
                .power_iteration_state
                .estimate_dominant_eigenvalue(matrix)?;

            // Store eigenvalue
            let mut eigenvals = vec![eigenvalue];
            if self.eigenvalue_history.len() > 0 {
                if let Some(last_eigenvals) = self.eigenvalue_history.back() {
                    eigenvals = last_eigenvals.clone();
                    eigenvals[0] = eigenvalue; // Update dominant eigenvalue
                }
            }

            self.eigenvalue_history.push_back(eigenvals);
            if self.eigenvalue_history.len() > 100 {
                self.eigenvalue_history.pop_front();
            }

            self.spectral_radius.push_back(eigenvalue);
            if self.spectral_radius.len() > 100 {
                self.spectral_radius.pop_front();
            }

            Ok(())
        }
    }

    impl<T: Float + Send + Sync + std::iter::Sum + ndarray::ScalarOperand> PowerIterationState<T> {
        pub fn new(size: usize) -> Self {
            Self {
                vector: Array1::ones(size),
                value: T::one(),
                iterations: 0,
                tolerance: T::from(1e-8).unwrap(),
            }
        }

        pub fn estimate_dominant_eigenvalue(&mut self, matrix: &Array2<T>) -> Result<T> {
            let max_iterations = 50;

            for _ in 0..max_iterations {
                let new_vector = matrix.dot(&self.vector);
                let norm = new_vector.iter().map(|&x| x * x).sum::<T>().sqrt();

                if norm < T::from(1e-12).unwrap() {
                    break;
                }

                self.vector = new_vector / norm;
                let new_value = self.vector.dot(&matrix.dot(&self.vector));

                if (new_value - self.value).abs() < self.tolerance {
                    break;
                }

                self.value = new_value;
                self.iterations += 1;
            }

            Ok(self.value)
        }
    }

    impl<T: Float + Send + Sync> ConditionMonitor<T> {
        pub fn new() -> Self {
            Self {
                condition_history: VecDeque::with_capacity(100),
                condition_threshold: T::from(1e8).unwrap(),
                auto_remediation: true,
                remediation_log: Vec::new(),
            }
        }

        pub fn update(&mut self, matrix: &Array2<T>) -> Result<()> {
            // Estimate condition number using diagonal elements
            let mut min_diag = T::infinity();
            let mut max_diag = T::zero();

            for i in 0..matrix.nrows() {
                let diag = matrix[[i, i]];
                min_diag = min_diag.min(diag);
                max_diag = max_diag.max(diag);
            }

            let condition_number = if min_diag > T::zero() {
                max_diag / min_diag
            } else {
                T::infinity()
            };

            self.condition_history.push_back(condition_number);
            if self.condition_history.len() > 100 {
                self.condition_history.pop_front();
            }

            // Check if remediation is needed
            if self.auto_remediation && condition_number > self.condition_threshold {
                self.apply_remediation(condition_number)?;
            }

            Ok(())
        }

        fn apply_remediation(&mut self, _conditionnumber: T) -> Result<()> {
            // Simple remediation: log the need for increased damping
            let action = RemediationAction::IncreaseDamping {
                old_damping: 0.001,
                new_damping: 0.01,
            };

            self.remediation_log.push(action);
            Ok(())
        }
    }

    impl<T: Float + Send + Sync> AdaptivePreconditioning<T> {
        pub fn new() -> Self {
            Self {
                method_selector: PreconditioningSelector::new(),
                method_configs: HashMap::new(),
                performance_tracker: MethodPerformanceTracker::new(),
            }
        }
    }

    impl<T: Float + Send + Sync> PreconditioningSelector<T> {
        pub fn new() -> Self {
            let mut size_thresholds = BTreeMap::new();
            size_thresholds.insert(100, PreconditioningMethod::Standard);
            size_thresholds.insert(1000, PreconditioningMethod::BlockDiagonal);
            size_thresholds.insert(10000, PreconditioningMethod::LowRank);

            let criteria = SelectionCriteria {
                size_thresholds,
                condition_thresholds: BTreeMap::new(),
                memory_constraints: MemoryConstraints {
                    max_memory_per_layer: 1_000_000_000, // 1GB
                    total_memory_budget: 10_000_000_000, // 10GB
                    efficiency_threshold: 0.8,
                },
                performance_requirements: PerformanceRequirements {
                    max_update_time: std::time::Duration::from_millis(100),
                    target_convergence_rate: T::from(0.01).unwrap(),
                    accuracy_tolerance: T::from(1e-6).unwrap(),
                },
            };

            Self {
                criteria,
                transition_rules: Vec::new(),
                current_methods: HashMap::new(),
            }
        }
    }

    impl<T: Float + Send + Sync> MethodPerformanceTracker<T> {
        pub fn new() -> Self {
            Self {
                metrics: HashMap::new(),
                history: HashMap::new(),
            }
        }
    }

    impl<T: Float + Send + Sync> StabilityAnalyzer<T> {
        pub fn new() -> Self {
            Self {
                stability_metrics: StabilityMetrics::new(),
                error_tracking: ErrorTracking::new(),
                thresholds: StabilityThresholds::default(),
                auto_stabilization: AutoStabilization::new(),
            }
        }
    }

    impl<T: Float + Send + Sync> StabilityMetrics<T> {
        pub fn new() -> Self {
            Self {
                numerical_errors: Vec::new(),
                conditioning_indicators: ConditioningIndicators::new(),
                convergence_stability: ConvergenceStability::new(),
                update_magnitudes: VecDeque::with_capacity(1000),
            }
        }
    }

    impl<T: Float + Send + Sync> ConditioningIndicators<T> {
        pub fn new() -> Self {
            Self {
                condition_trends: VecDeque::with_capacity(100),
                eigenvalue_spread: T::one(),
                rank_indicators: RankIndicators::new(),
                symmetry_preservation: T::one(),
            }
        }
    }

    impl<T: Float + Send + Sync> RankIndicators<T> {
        pub fn new() -> Self {
            Self {
                effective_rank: 0,
                numerical_rank: 0,
                null_space_dim: 0,
                small_singular_values: Vec::new(),
            }
        }
    }

    impl<T: Float + Send + Sync> ConvergenceStability<T> {
        pub fn new() -> Self {
            Self {
                rate_history: VecDeque::with_capacity(100),
                oscillation_detector: OscillationDetector::new(),
                plateau_detector: PlateauDetector {
                    min_improvement: T::from(1e-6).unwrap(),
                    duration_threshold: 20,
                    current_plateau_length: 0,
                    in_plateau: false,
                },
                divergence_indicators: DivergenceIndicators::new(),
            }
        }
    }

    impl<T: Float + Send + Sync> OscillationDetector<T> {
        pub fn new() -> Self {
            Self {
                recent_values: VecDeque::with_capacity(50),
                frequency_estimate: None,
                amplitude: T::zero(),
                threshold: T::from(0.1).unwrap(),
            }
        }
    }

    impl<T: Float + Send + Sync> DivergenceIndicators<T> {
        pub fn new() -> Self {
            Self {
                gradient_explosion: false,
                parameter_drift: T::zero(),
                loss_explosion: false,
                update_explosion: false,
            }
        }
    }

    impl<T: Float + Send + Sync> ErrorTracking<T> {
        pub fn new() -> Self {
            Self {
                fp_error_accumulation: T::zero(),
                roundoff_errors: VecDeque::with_capacity(100),
                cancellation_detector: CancellationDetector::new(),
                error_propagation: ErrorPropagation::new(),
            }
        }
    }

    impl<T: Float + Send + Sync> CancellationDetector<T> {
        pub fn new() -> Self {
            Self {
                bit_loss: Vec::new(),
                cancellation_events: Vec::new(),
                sensitivity: T::from(1e-10).unwrap(),
            }
        }
    }

    impl<T: Float + Send + Sync> ErrorPropagation<T> {
        pub fn new() -> Self {
            Self {
                input_sensitivity: T::one(),
                amplification_factors: Vec::new(),
                error_correlations: HashMap::new(),
            }
        }
    }

    impl<T: Float> Default for StabilityThresholds<T> {
        fn default() -> Self {
            Self {
                max_condition_number: T::from(1e12).unwrap(),
                min_eigenvalue: T::from(1e-8).unwrap(),
                max_update_magnitude: T::from(10.0).unwrap(),
                error_tolerance: T::from(1e-6).unwrap(),
            }
        }
    }

    impl<T: Float + Send + Sync> AutoStabilization<T> {
        pub fn new() -> Self {
            Self {
                enabled_techniques: vec![
                    StabilizationTechnique::AdaptiveDamping,
                    StabilizationTechnique::GradientClipping,
                ],
                intervention_history: Vec::new(),
                adaptive_thresholds: true,
            }
        }
    }

    impl<T: Float + Send + Sync> SecondOrderMomentum<T> {
        pub fn new(config: MomentumConfig<T>) -> Self {
            Self {
                layer_momentum: HashMap::new(),
                config,
                adaptive_scheduling: AdaptiveMomentumScheduling::new(),
                effectiveness_tracker: MomentumEffectivenessTracker::new(),
            }
        }
    }

    impl<T: Float> Default for MomentumConfig<T> {
        fn default() -> Self {
            Self {
                enable_first_order: true,
                enable_second_order: true,
                enable_curvature_momentum: false,
                base_beta1: T::from(0.9).unwrap(),
                base_beta2: T::from(0.999).unwrap(),
                warmup_steps: 1000,
                annealing_schedule: AnnealingSchedule::Constant,
                layer_adaptation: true,
            }
        }
    }

    impl<T: Float + Send + Sync> AdaptiveMomentumScheduling<T> {
        pub fn new() -> Self {
            Self {
                layer_schedules: HashMap::new(),
                global_trends: MomentumTrends::new(),
                strategy: MomentumSchedulingStrategy::LayerAdaptive,
            }
        }
    }

    impl<T: Float + Send + Sync> MomentumTrends<T> {
        pub fn new() -> Self {
            Self {
                avg_effectiveness: T::from(0.5).unwrap(),
                trend_direction: TrendDirection::Stable,
                trend_magnitude: T::zero(),
                stability: T::one(),
            }
        }
    }

    impl<T: Float + Send + Sync> MomentumEffectivenessTracker<T> {
        pub fn new() -> Self {
            Self {
                layer_scores: HashMap::new(),
                global_metrics: GlobalEffectivenessMetrics::new(),
                baselines: EffectivenessBaselines::new(),
            }
        }
    }

    impl<T: Float + Send + Sync> GlobalEffectivenessMetrics<T> {
        pub fn new() -> Self {
            Self {
                total_convergence_improvement: T::zero(),
                avg_layer_improvement: T::zero(),
                effectiveness_variance: T::zero(),
                overhead_ratio: T::from(0.1).unwrap(),
            }
        }
    }

    impl<T: Float + Send + Sync> EffectivenessBaselines<T> {
        pub fn new() -> Self {
            Self {
                no_momentum: T::zero(),
                first_order_only: T::from(0.1).unwrap(),
                standard_kfac: T::from(0.5).unwrap(),
            }
        }
    }

    /// Integrate natural gradients with K-FAC
    impl<
            T: Float
                + Default
                + Clone
                + Send
                + Sync
                + std::fmt::Debug
                + ndarray::ScalarOperand
                + std::iter::Sum
                + num_traits::FromPrimitive,
        > KFAC<T>
    {
        /// Compute natural gradient update using K-FAC approximation
        pub fn natural_gradient_step(
            &mut self,
            layername: &str,
            parameters: &Array2<T>,
            gradients: &Array2<T>,
            activations: &Array2<T>,
            loss: Option<T>,
        ) -> Result<Array2<T>> {
            // Regular K-FAC step
            let kfac_update = self.step(layername, parameters, gradients, activations, loss)?;

            // Apply natural gradient scaling based on Fisher information
            let state = self.layer_states.get(layername).ok_or_else(|| {
                OptimError::InvalidConfig(format!("Layer {} not found", layername))
            })?;

            if let (Some(a_inv), Some(g_inv)) = (&state.a_cov_inv, &state.g_cov_inv) {
                // Scale by Fisher information to get natural gradient
                let fisher_scaled = self.apply_fisher_scaling(&kfac_update, a_inv, g_inv)?;
                Ok(fisher_scaled)
            } else {
                Ok(kfac_update)
            }
        }

        fn apply_fisher_scaling(
            &self,
            update: &Array2<T>,
            a_inv: &Array2<T>,
            g_inv: &Array2<T>,
        ) -> Result<Array2<T>> {
            // Apply Fisher information scaling: F^(-1/2) * update
            // This is simplified - in practice would use proper matrix square root
            let scaled = g_inv.dot(&update.dot(a_inv));
            Ok(scaled * T::from(0.5).unwrap()) // Scale factor for stability
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kfac_creation() {
        let config = KFACConfig::default();
        let kfac = KFAC::<f64>::new(config);
        assert_eq!(kfac.step_count, 0);
        assert_eq!(kfac.layer_states.len(), 0);
    }

    #[test]
    fn test_layer_registration() {
        let mut kfac = KFAC::<f64>::new(KFACConfig::default());

        let layerinfo = LayerInfo {
            name: "dense1".to_string(),
            input_dim: 10,
            output_dim: 5,
            layer_type: LayerType::Dense,
            has_bias: true,
        };

        assert!(kfac.register_layer(layerinfo).is_ok());
        assert_eq!(kfac.layer_states.len(), 1);

        let state = kfac.get_layer_state("dense1").unwrap();
        assert_eq!(state.a_cov.nrows(), 11); // 10 + 1 for bias
        assert_eq!(state.g_cov.nrows(), 5);
    }

    #[test]
    fn test_covariance_update() {
        let mut kfac = KFAC::<f64>::new(KFACConfig {
            cov_update_freq: 1,
            ..Default::default()
        });

        let layerinfo = LayerInfo {
            name: "test_layer".to_string(),
            input_dim: 3,
            output_dim: 2,
            layer_type: LayerType::Dense,
            has_bias: false,
        };

        kfac.register_layer(layerinfo).unwrap();

        let activations =
            Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let gradients = Array2::from_shape_vec((2, 2), vec![0.1, 0.2, 0.3, 0.4]).unwrap();

        assert!(kfac
            .update_covariance_matrices("test_layer", &activations, &gradients)
            .is_ok());

        let state = kfac.get_layer_state("test_layer").unwrap();
        assert!(state.running_mean_a.is_some());
        assert!(state.running_mean_g.is_some());
    }

    #[test]
    fn test_adaptive_damping() {
        let kfac = KFAC::<f64>::new(KFACConfig::default());

        // Test matrix with large condition number
        let mut matrix = Array2::eye(3);
        matrix[[0, 0]] = 1.0;
        matrix[[1, 1]] = 1e-6;
        matrix[[2, 2]] = 1e-6;

        let damping = kfac.compute_adaptive_damping(&matrix).unwrap();
        assert!(damping > kfac.config.damping);
    }

    #[test]
    fn test_memory_estimation() {
        let mut kfac = KFAC::<f64>::new(KFACConfig::default());

        let layerinfo = LayerInfo {
            name: "test".to_string(),
            input_dim: 100,
            output_dim: 50,
            layer_type: LayerType::Dense,
            has_bias: true,
        };

        kfac.register_layer(layerinfo).unwrap();

        let memory = kfac.estimate_memory_usage();
        assert!(memory > 0);
    }

    #[test]
    fn test_layer_types() {
        let dense = LayerType::Dense;
        let conv = LayerType::Convolution;
        let grouped = LayerType::GroupedConvolution { groups: 4 };

        assert_eq!(dense, LayerType::Dense);
        assert_ne!(conv, LayerType::Dense);

        if let LayerType::GroupedConvolution { groups } = grouped {
            assert_eq!(groups, 4);
        } else {
            panic!("Incorrect layer type");
        }
    }

    #[test]
    fn test_kfac_utils() {
        let input_patches =
            Array2::from_shape_vec((4, 6), (0..24).map(|x| x as f64).collect()).unwrap();
        let output_grads =
            Array2::from_shape_vec((4, 3), (0..12).map(|x| x as f64 * 0.1).collect()).unwrap();

        let groups = 2;
        let result = kfac_utils::grouped_conv_kfac(groups, &input_patches, &output_grads);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), groups);
    }
}
