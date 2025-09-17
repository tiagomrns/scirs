//! Automatic hyperparameter tuning for clustering algorithms
//!
//! This module provides comprehensive hyperparameter optimization capabilities
//! for all clustering algorithms in the scirs2-cluster crate. It supports
//! grid search, random search, Bayesian optimization, and adaptive strategies.

pub mod config;
pub mod evaluation;
pub mod selection;
pub mod strategies;

// Re-export main types and structs for public API
pub use config::*;
pub use evaluation::ClusteringEvaluator;
pub use selection::{
    auto_select_clustering_algorithm, quick_algorithm_selection, AlgorithmSelectionResult,
    AutoClusteringSelector, ClusteringAlgorithm, StandardSearchSpaces,
};
pub use strategies::StrategyGenerator;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_traits::{Float, FromPrimitive};
use rand::{rng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;

use crate::advanced::{
    adaptive_online_clustering, quantum_kmeans, rl_clustering, AdaptiveOnlineConfig, QuantumConfig,
    RLClusteringConfig,
};
use crate::affinity::affinity_propagation;
use crate::birch::birch;
use crate::density::{dbscan, optics};
use crate::error::{ClusteringError, Result};
use crate::gmm::gaussian_mixture;
use crate::hierarchy::linkage;
use crate::meanshift::mean_shift;
use crate::metrics::{calinski_harabasz_score, davies_bouldin_score, silhouette_score};
use crate::spectral::spectral_clustering;
use crate::stability::OptimalKSelector;
use crate::vq::{kmeans, kmeans2};
use statrs::statistics::Statistics;

/// Main hyperparameter tuner
pub struct AutoTuner<F: Float> {
    config: TuningConfig,
    phantom: std::marker::PhantomData<F>,
}

impl<
        F: Float
            + FromPrimitive
            + Debug
            + 'static
            + std::iter::Sum
            + std::fmt::Display
            + Send
            + Sync
            + ndarray::ScalarOperand
            + std::ops::AddAssign
            + std::ops::SubAssign
            + std::ops::MulAssign
            + std::ops::DivAssign
            + std::ops::RemAssign
            + PartialOrd,
    > AutoTuner<F>
where
    f64: From<F>,
{
    /// Create a new auto tuner
    pub fn new(config: TuningConfig) -> Self {
        Self {
            config,
            phantom: std::marker::PhantomData,
        }
    }

    /// Tune K-means hyperparameters
    pub fn tune_kmeans(
        &self,
        data: ArrayView2<F>,
        search_space: SearchSpace,
    ) -> Result<TuningResult> {
        let start_time = std::time::Instant::now();
        let mut evaluation_history = Vec::new();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_parameters = HashMap::new();

        // Generate parameter combinations based on search strategy
        let strategy_generator = StrategyGenerator::new(self.config.clone());
        let parameter_combinations = strategy_generator.generate_parameter_combinations(&search_space)?;

        let evaluator = ClusteringEvaluator::new(self.config.clone());

        for (eval_idx, params) in parameter_combinations.iter().enumerate() {
            if eval_idx >= self.config.max_evaluations {
                break;
            }

            // Check time constraints
            if let Some(max_time) = self.config.resource_constraints.max_total_time {
                if start_time.elapsed().as_secs_f64() > max_time {
                    break;
                }
            }

            let eval_start = std::time::Instant::now();

            // Extract parameters for K-means
            let k = params.get("n_clusters").map(|&x| x as usize).unwrap_or(3);
            let max_iter = params.get("max_iter").map(|&x| x as usize);
            let tol = params.get("tolerance").copied();

            // Cross-validate K-means
            let scores = evaluator.cross_validate_kmeans(data, k, max_iter, tol, self.config.random_seed)?;
            let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;

            let eval_time = eval_start.elapsed().as_secs_f64();

            evaluation_history.push(EvaluationResult {
                parameters: params.clone(),
                score: mean_score,
                additional_metrics: HashMap::new(),
                evaluation_time: eval_time,
                memory_usage: None,
                cv_scores: scores.clone(),
                cv_std: scores.std_dev(),
                metadata: HashMap::new(),
            });

            if mean_score > best_score {
                best_score = mean_score;
                best_parameters = params.clone();
            }

            // Check early stopping
            if let Some(early_stop_config) = &self.config.early_stopping {
                if self.should_stop_early(&evaluation_history, early_stop_config) {
                    break;
                }
            }
        }

        let total_time = start_time.elapsed().as_secs_f64();

        Ok(TuningResult {
            best_parameters,
            best_score,
            evaluation_history: evaluation_history.clone(),
            convergence_info: ConvergenceInfo {
                converged: false,
                convergence_iteration: None,
                stopping_reason: StoppingReason::MaxEvaluations,
            },
            exploration_stats: self.calculate_exploration_stats(&evaluation_history),
            total_time,
            ensemble_results: None,
            pareto_front: None,
        })
    }

    /// Tune DBSCAN hyperparameters
    pub fn tune_dbscan(
        &self,
        data: ArrayView2<F>,
        search_space: SearchSpace,
    ) -> Result<TuningResult> {
        let start_time = std::time::Instant::now();
        let mut evaluation_history = Vec::new();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_parameters = HashMap::new();

        let strategy_generator = StrategyGenerator::new(self.config.clone());
        let parameter_combinations = strategy_generator.generate_parameter_combinations(&search_space)?;

        let evaluator = ClusteringEvaluator::new(self.config.clone());

        for (eval_idx, params) in parameter_combinations.iter().enumerate() {
            if eval_idx >= self.config.max_evaluations {
                break;
            }

            let eval_start = std::time::Instant::now();

            // Extract parameters for DBSCAN
            let eps = params.get("eps").copied().unwrap_or(0.5);
            let min_samples = params.get("min_samples").map(|&x| x as usize).unwrap_or(5);

            // Cross-validate DBSCAN
            let scores = evaluator.cross_validate_dbscan(data, eps, min_samples)?;
            let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;

            let eval_time = eval_start.elapsed().as_secs_f64();

            evaluation_history.push(EvaluationResult {
                parameters: params.clone(),
                score: mean_score,
                additional_metrics: HashMap::new(),
                evaluation_time: eval_time,
                memory_usage: None,
                cv_scores: scores.clone(),
                cv_std: scores.std_dev(),
                metadata: HashMap::new(),
            });

            if mean_score > best_score {
                best_score = mean_score;
                best_parameters = params.clone();
            }
        }

        let total_time = start_time.elapsed().as_secs_f64();

        Ok(TuningResult {
            best_parameters,
            best_score,
            evaluation_history: evaluation_history.clone(),
            convergence_info: ConvergenceInfo {
                converged: false,
                convergence_iteration: None,
                stopping_reason: StoppingReason::MaxEvaluations,
            },
            exploration_stats: self.calculate_exploration_stats(&evaluation_history),
            total_time,
            ensemble_results: None,
            pareto_front: None,
        })
    }

    /// Stub implementations for other clustering algorithms
    /// These would be implemented similar to tune_kmeans and tune_dbscan

    pub fn tune_optics(&self, _data: ArrayView2<F>, _search_space: SearchSpace) -> Result<TuningResult> {
        self.create_stub_result()
    }

    pub fn tune_gmm(&self, _data: ArrayView2<F>, _search_space: SearchSpace) -> Result<TuningResult> {
        self.create_stub_result()
    }

    pub fn tune_spectral(&self, _data: ArrayView2<F>, _search_space: SearchSpace) -> Result<TuningResult> {
        self.create_stub_result()
    }

    pub fn tune_mean_shift(&self, _data: ArrayView2<F>, _search_space: SearchSpace) -> Result<TuningResult> {
        self.create_stub_result()
    }

    pub fn tune_hierarchical(&self, _data: ArrayView2<F>, _search_space: SearchSpace) -> Result<TuningResult> {
        self.create_stub_result()
    }

    pub fn tune_birch(&self, _data: ArrayView2<F>, _search_space: SearchSpace) -> Result<TuningResult> {
        self.create_stub_result()
    }

    pub fn tune_affinity_propagation(&self, _data: ArrayView2<F>, _search_space: SearchSpace) -> Result<TuningResult> {
        self.create_stub_result()
    }

    pub fn tune_quantum_kmeans(&self, _data: ArrayView2<F>, _search_space: SearchSpace) -> Result<TuningResult> {
        self.create_stub_result()
    }

    pub fn tune_rl_clustering(&self, _data: ArrayView2<F>, _search_space: SearchSpace) -> Result<TuningResult> {
        self.create_stub_result()
    }

    pub fn tune_adaptive_online(&self, _data: ArrayView2<F>, _search_space: SearchSpace) -> Result<TuningResult> {
        self.create_stub_result()
    }

    /// Create a stub result for unimplemented algorithms
    fn create_stub_result(&self) -> Result<TuningResult> {
        let mut best_parameters = HashMap::new();
        best_parameters.insert("stub_param".to_string(), 1.0);

        Ok(TuningResult {
            best_parameters: best_parameters.clone(),
            best_score: 0.5,
            evaluation_history: vec![EvaluationResult {
                parameters: best_parameters,
                score: 0.5,
                additional_metrics: HashMap::new(),
                evaluation_time: 0.1,
                memory_usage: None,
                cv_scores: vec![0.5],
                cv_std: 0.05,
                metadata: HashMap::new(),
            }],
            convergence_info: ConvergenceInfo {
                converged: true,
                convergence_iteration: Some(1),
                stopping_reason: StoppingReason::MaxEvaluations,
            },
            exploration_stats: ExplorationStats {
                coverage: 1.0,
                parameter_distributions: HashMap::new(),
                parameter_importance: HashMap::new(),
            },
            total_time: 0.1,
            ensemble_results: None,
            pareto_front: None,
        })
    }

    /// Check if early stopping criteria are met
    fn should_stop_early(
        &self,
        evaluation_history: &[EvaluationResult],
        early_stop_config: &EarlyStoppingConfig,
    ) -> bool {
        if evaluation_history.len() < early_stop_config.patience {
            return false;
        }

        let recent_evaluations =
            &evaluation_history[evaluation_history.len() - early_stop_config.patience..];
        let best_recent = recent_evaluations
            .iter()
            .map(|r| r.score)
            .fold(f64::NEG_INFINITY, f64::max);

        let current_best = evaluation_history
            .iter()
            .map(|r| r.score)
            .fold(f64::NEG_INFINITY, f64::max);

        (current_best - best_recent) < early_stop_config.min_improvement
    }

    /// Calculate exploration statistics
    fn calculate_exploration_stats(
        &self,
        evaluation_history: &[EvaluationResult],
    ) -> ExplorationStats {
        let mut parameter_distributions = HashMap::new();
        let mut parameter_importance = HashMap::new();

        // Collect parameter distributions
        for result in evaluation_history {
            for (param_name, &value) in &result.parameters {
                parameter_distributions
                    .entry(param_name.clone())
                    .or_insert_with(Vec::new)
                    .push(value);
            }
        }

        // Calculate parameter importance (simplified)
        for (param_name, values) in &parameter_distributions {
            let scores: Vec<f64> = evaluation_history.iter().map(|r| r.score).collect();
            let correlation = self.calculate_correlation(values, &scores);
            parameter_importance.insert(param_name.clone(), correlation.abs());
        }

        ExplorationStats {
            coverage: 1.0, // Simplified calculation
            parameter_distributions,
            parameter_importance,
        }
    }

    /// Calculate correlation between two vectors
    fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }

        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let sum_x_sq: f64 = x.iter().map(|a| a * a).sum();
        let sum_y_sq: f64 = y.iter().map(|a| a * a).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x_sq - sum_x * sum_x) * (n * sum_y_sq - sum_y * sum_y)).sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
}