//! Neural network model trainer implementation
//!
//! This module provides a trainer for neural network models, including
//! utilities for batch training, evaluation, and early stopping.

use ndarray::{s, Array, IxDyn, ScalarOperand};
use num_integer::div_ceil;
use num_traits::Float;
use std::fmt::Debug;
use std::time::Instant;

/// Result type for train/validation data split
pub type SplitResult<F> = (
    Array<F, IxDyn>,
    Array<F, IxDyn>,
    Option<Array<F, IxDyn>>,
    Option<Array<F, IxDyn>>,
);

use crate::error::{NeuralError, Result};
use crate::losses::Loss;
use crate::models::Model;
use crate::optimizers::Optimizer;
use crate::utils::metrics::Metric;

/// Configuration for the training process
pub struct TrainingConfig {
    /// Number of training epochs
    pub epochs: usize,
    /// Batch size for training
    pub batch_size: usize,
    /// Whether to shuffle the data between epochs
    pub shuffle: bool,
    /// Validation split (0.0-1.0) for early stopping
    pub validation_split: Option<f64>,
    /// Number of epochs with no improvement after which training will be stopped
    pub early_stopping_patience: Option<usize>,
    /// Minimum change in loss to qualify as an improvement
    pub early_stopping_min_delta: Option<f64>,
    /// Whether to print progress during training
    pub verbose: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 10,
            batch_size: 32,
            shuffle: true,
            validation_split: None,
            early_stopping_patience: None,
            early_stopping_min_delta: None,
            verbose: true,
        }
    }
}

/// History of training metrics
pub struct History<F: Float + ScalarOperand> {
    /// Training loss history
    pub train_loss: Vec<F>,
    /// Validation loss history
    pub val_loss: Vec<F>,
    /// Custom metrics history
    pub metrics: Vec<(String, Vec<F>)>,
}

/// Trainer for neural network models
pub struct Trainer<F: Float + Debug + ScalarOperand, M: Model<F>, L: Loss<F>, O: Optimizer<F>> {
    /// Model to train
    model: M,
    /// Loss function to use
    loss_fn: L,
    /// Optimizer to use
    optimizer: O,
    /// Configuration for training
    config: TrainingConfig,
    /// Metrics to track during training
    metrics: Vec<(String, Box<dyn Metric<F>>)>,
}

impl<F: Float + Debug + ScalarOperand, M: Model<F>, L: Loss<F>, O: Optimizer<F>>
    Trainer<F, M, L, O>
{
    /// Create a new trainer
    pub fn new(model: M, loss_fn: L, optimizer: O) -> Self {
        Self {
            model,
            loss_fn,
            optimizer,
            config: TrainingConfig::default(),
            metrics: Vec::new(),
        }
    }

    /// Set the training configuration
    pub fn with_config(mut self, config: TrainingConfig) -> Self {
        self.config = config;
        self
    }

    /// Add a metric to track during training
    pub fn add_metric<Me: Metric<F> + 'static>(&mut self, name: &str, metric: Me) -> &mut Self {
        self.metrics.push((name.to_string(), Box::new(metric)));
        self
    }

    /// Train the model on the given data
    pub fn fit(&mut self, x: &Array<F, IxDyn>, y: &Array<F, IxDyn>) -> Result<History<F>> {
        let start_time = Instant::now();

        // Initialize history
        let mut history = History {
            train_loss: Vec::with_capacity(self.config.epochs),
            val_loss: Vec::with_capacity(self.config.epochs),
            metrics: self
                .metrics
                .iter()
                .map(|(name, _)| (name.clone(), Vec::with_capacity(self.config.epochs)))
                .collect(),
        };

        // Split data for validation if needed
        let (x_train, y_train, x_val, y_val) = if let Some(val_split) = self.config.validation_split
        {
            self.split_validation_data(x, y, val_split)?
        } else {
            (x.clone(), y.clone(), None, None)
        };

        // Training loop
        let mut best_val_loss = None;
        let mut patience_counter = 0;

        for epoch in 0..self.config.epochs {
            // Train on batches
            let mut epoch_loss = F::zero();
            let batch_count = div_ceil(x_train.shape()[0], self.config.batch_size);

            for batch in 0..batch_count {
                let start_idx = batch * self.config.batch_size;
                let end_idx = std::cmp::min(start_idx + self.config.batch_size, x_train.shape()[0]);

                let batch_x = x_train
                    .slice(s![start_idx..end_idx, ..])
                    .to_owned()
                    .into_dyn();
                let batch_y = y_train
                    .slice(s![start_idx..end_idx, ..])
                    .to_owned()
                    .into_dyn();

                let batch_loss = self.model.train_batch(
                    &batch_x,
                    &batch_y,
                    &self.loss_fn,
                    &mut self.optimizer,
                )?;
                epoch_loss = epoch_loss + batch_loss;
            }

            epoch_loss = epoch_loss / F::from(batch_count).unwrap_or(F::one());
            history.train_loss.push(epoch_loss);

            // Calculate validation loss if validation data is provided
            if let (Some(ref x_val), Some(ref y_val)) = (&x_val, &y_val) {
                let val_loss = self.model.evaluate(x_val, y_val, &self.loss_fn)?;
                history.val_loss.push(val_loss);

                // Early stopping
                if let (Some(patience), Some(min_delta)) = (
                    self.config.early_stopping_patience,
                    self.config.early_stopping_min_delta,
                ) {
                    let min_delta = F::from(min_delta).unwrap_or(F::epsilon());

                    match best_val_loss {
                        None => best_val_loss = Some(val_loss),
                        Some(best_loss) => {
                            if val_loss < best_loss - min_delta {
                                best_val_loss = Some(val_loss);
                                patience_counter = 0;
                            } else {
                                patience_counter += 1;
                                if patience_counter >= patience {
                                    if self.config.verbose {
                                        eprintln!(
                                            "Early stopping triggered after {} epochs",
                                            epoch + 1
                                        );
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            // Calculate metrics
            for (i, (_name, metric)) in self.metrics.iter().enumerate() {
                let metric_value = if let (Some(ref x_val), Some(ref y_val)) = (&x_val, &y_val) {
                    let y_pred = self.model.predict(x_val)?;
                    metric.compute(&y_pred, y_val)?
                } else {
                    let y_pred = self.model.predict(&x_train)?;
                    metric.compute(&y_pred, &y_train)?
                };

                history.metrics[i].1.push(metric_value);
            }

            // Print progress
            if self.config.verbose {
                let elapsed = start_time.elapsed();
                let metrics_str = self
                    .metrics
                    .iter()
                    .enumerate()
                    .map(|(i, (name, _))| {
                        format!("{}: {:.4?}", name, history.metrics[i].1.last().unwrap())
                    })
                    .collect::<Vec<_>>()
                    .join(", ");

                if !history.val_loss.is_empty() {
                    eprintln!(
                        "Epoch {}/{} - {:.2?} - loss: {:.4?} - val_loss: {:.4?} - {}",
                        epoch + 1,
                        self.config.epochs,
                        elapsed,
                        epoch_loss,
                        history.val_loss.last().unwrap(),
                        metrics_str
                    );
                } else {
                    eprintln!(
                        "Epoch {}/{} - {:.2?} - loss: {:.4?} - {}",
                        epoch + 1,
                        self.config.epochs,
                        elapsed,
                        epoch_loss,
                        metrics_str
                    );
                }
            }
        }

        Ok(history)
    }

    /// Split data into training and validation sets
    fn split_validation_data(
        &self,
        x: &Array<F, IxDyn>,
        y: &Array<F, IxDyn>,
        val_split: f64,
    ) -> Result<SplitResult<F>> {
        if val_split <= 0.0 || val_split >= 1.0 {
            return Err(NeuralError::InferenceError(
                "Validation split must be between 0 and 1".to_string(),
            ));
        }

        let n_samples = x.shape()[0];
        let val_size = (n_samples as f64 * val_split).round() as usize;
        let train_size = n_samples - val_size;

        if val_size == 0 || train_size == 0 {
            return Err(NeuralError::InferenceError(
                "Validation split results in empty training or validation set".to_string(),
            ));
        }

        let x_train = x.slice(s![0..train_size, ..]).to_owned().into_dyn();
        let y_train = y.slice(s![0..train_size, ..]).to_owned().into_dyn();

        let x_val = x.slice(s![train_size.., ..]).to_owned();
        let y_val = y.slice(s![train_size.., ..]).to_owned();

        // Convert to dynamic dimensions to match the return type
        let x_val_dyn = x_val.into_dyn();
        let y_val_dyn = y_val.into_dyn();
        Ok((x_train, y_train, Some(x_val_dyn), Some(y_val_dyn)))
    }

    /// Get a reference to the model
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Get a mutable reference to the model
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }
}
