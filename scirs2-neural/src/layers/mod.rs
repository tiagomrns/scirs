//! Neural network layers implementation
//!
//! This module provides implementations of various neural network layers
//! such as dense (fully connected), attention, convolution, pooling, etc.
//! Layers are the fundamental building blocks of neural networks.

use crate::error::Result;
use ndarray::{Array, ScalarOperand};
use num_traits::Float;
use std::fmt::Debug;

/// Base trait for neural network layers
///
/// This trait defines the core interface that all neural network layers must implement.
/// It supports forward propagation, backpropagation, parameter management, and
/// training/evaluation mode switching.
pub trait Layer<F: Float + Debug + ScalarOperand>: Send + Sync {
    /// Forward pass of the layer
    ///
    /// Computes the output of the layer given an input tensor.
    fn forward(&self, input: &Array<F, ndarray::IxDyn>) -> Result<Array<F, ndarray::IxDyn>>;

    /// Backward pass of the layer to compute gradients
    ///
    /// Computes gradients with respect to the layer's input, which is needed
    /// for backpropagation.
    fn backward(
        &self,
        input: &Array<F, ndarray::IxDyn>,
        grad_output: &Array<F, ndarray::IxDyn>,
    ) -> Result<Array<F, ndarray::IxDyn>>;

    /// Update the layer parameters with the given learning rate
    fn update(&mut self, learningrate: F) -> Result<()>;

    /// Get the layer as a dyn Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any;

    /// Get the layer as a mutable dyn Any for downcasting
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;

    /// Get the parameters of the layer
    fn params(&self) -> Vec<Array<F, ndarray::IxDyn>> {
        Vec::new()
    }

    /// Get the gradients of the layer parameters
    fn gradients(&self) -> Vec<Array<F, ndarray::IxDyn>> {
        Vec::new()
    }

    /// Set the gradients of the layer parameters
    fn set_gradients(&mut self, _gradients: &[Array<F, ndarray::IxDyn>]) -> Result<()> {
        Ok(())
    }

    /// Set the parameters of the layer
    fn set_params(&mut self, _params: &[Array<F, ndarray::IxDyn>]) -> Result<()> {
        Ok(())
    }

    /// Set the layer to training mode (true) or evaluation mode (false)
    fn set_training(&mut self, _training: bool) {
        // Default implementation: do nothing
    }

    /// Get the current training mode
    fn is_training(&self) -> bool {
        true // Default implementation: always in training mode
    }

    /// Get the type of the layer (e.g., "Dense", "Conv2D")
    fn layer_type(&self) -> &str {
        "Unknown"
    }

    /// Get the number of trainable parameters in this layer
    fn parameter_count(&self) -> usize {
        0
    }

    /// Get a detailed description of this layer
    fn layer_description(&self) -> String {
        format!("type:{}", self.layer_type())
    }

    /// Get the input shape if known
    fn inputshape(&self) -> Option<Vec<usize>> {
        None
    }

    /// Get the output shape if known  
    fn outputshape(&self) -> Option<Vec<usize>> {
        None
    }

    /// Get the name of the layer if set
    fn name(&self) -> Option<&str> {
        None
    }
}

/// Trait for layers with parameters (weights, biases)
pub trait ParamLayer<F: Float + Debug + ScalarOperand>: Layer<F> {
    /// Get the parameters of the layer as a vector of arrays
    fn get_parameters(&self) -> Vec<Array<F, ndarray::IxDyn>>;

    /// Get the gradients of the parameters
    fn get_gradients(&self) -> Vec<Array<F, ndarray::IxDyn>>;

    /// Set the parameters
    fn set_parameters(&mut self, params: Vec<Array<F, ndarray::IxDyn>>) -> Result<()>;
}

/// Information about a layer for visualization purposes
#[derive(Debug, Clone)]
pub struct LayerInfo {
    /// Index of the layer in the sequence
    pub index: usize,
    /// Name of the layer
    pub name: String,
    /// Type of the layer
    pub layer_type: String,
    /// Number of parameters in the layer
    pub parameter_count: usize,
    /// Input shape of the layer
    pub inputshape: Option<Vec<usize>>,
    /// Output shape of the layer
    pub outputshape: Option<Vec<usize>>,
}

/// Sequential container for neural network layers
///
/// A Sequential model is a linear stack of layers where data flows through
/// each layer in order.
pub struct Sequential<F: Float + Debug + ScalarOperand> {
    layers: Vec<Box<dyn Layer<F> + Send + Sync>>,
    training: bool,
}

impl<F: Float + Debug + ScalarOperand> std::fmt::Debug for Sequential<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sequential")
            .field("num_layers", &self.layers.len())
            .field("training", &self.training)
            .finish()
    }
}

impl<F: Float + Debug + ScalarOperand + 'static> Clone for Sequential<F> {
    fn clone(&self) -> Self {
        // We can't clone the layers, so we just create an empty Sequential
        // with the same training flag
        Self {
            layers: Vec::new(),
            training: self.training,
        }
    }
}

impl<F: Float + Debug + ScalarOperand> Default for Sequential<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Debug + ScalarOperand> Sequential<F> {
    /// Create a new Sequential container
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            training: true,
        }
    }

    /// Add a layer to the container
    pub fn add<L: Layer<F> + Send + Sync + 'static>(&mut self, layer: L) {
        self.layers.push(Box::new(layer));
    }

    /// Get the number of layers
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Check if there are no layers
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Get total parameter count across all layers
    pub fn total_parameters(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.parameter_count())
            .sum()
    }

    /// Get layer information for visualization purposes
    pub fn layer_info(&self) -> Vec<LayerInfo> {
        self.layers
            .iter()
            .enumerate()
            .map(|(i, layer)| LayerInfo {
                index: i,
                name: layer.name().unwrap_or(&format!("Layer_{i}")).to_string(),
                layer_type: layer.layer_type().to_string(),
                parameter_count: layer.parameter_count(),
                inputshape: layer.inputshape(),
                outputshape: layer.outputshape(),
            })
            .collect()
    }
}

impl<F: Float + Debug + ScalarOperand> Layer<F> for Sequential<F> {
    fn forward(&self, input: &Array<F, ndarray::IxDyn>) -> Result<Array<F, ndarray::IxDyn>> {
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.forward(&output)?;
        }
        Ok(output)
    }

    fn backward(
        &self,
        _input: &Array<F, ndarray::IxDyn>,
        grad_output: &Array<F, ndarray::IxDyn>,
    ) -> Result<Array<F, ndarray::IxDyn>> {
        // For simplicity, we'll just return the grad_output as-is
        // A real implementation would propagate through the layers in reverse
        Ok(grad_output.clone())
    }

    fn update(&mut self, learningrate: F) -> Result<()> {
        for layer in &mut self.layers {
            layer.update(learningrate)?;
        }
        Ok(())
    }

    fn params(&self) -> Vec<Array<F, ndarray::IxDyn>> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.params());
        }
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        for layer in &mut self.layers {
            layer.set_training(training);
        }
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn layer_type(&self) -> &str {
        "Sequential"
    }

    fn parameter_count(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.parameter_count())
            .sum()
    }
}

/// Configuration enum for different types of layers
#[derive(Debug, Clone)]
pub enum LayerConfig {
    /// Dense (fully connected) layer
    Dense {
        input_size: usize,
        output_size: usize,
        activation: Option<String>,
    },
    /// 2D Convolutional layer
    Conv2D {
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
    },
    /// Dropout layer
    Dropout { rate: f64 },
}

// Fixed modules
pub mod conv;
pub mod dense;
pub mod dropout;
pub mod normalization;
pub mod recurrent;

// Temporarily comment out layer modules that need fixing
// mod attention;
// mod embedding;
// mod regularization;

// Re-export fixed modules
pub use conv::Conv2D;
pub use dense::Dense;
pub use dropout::Dropout;
pub use normalization::{BatchNorm, LayerNorm};
pub use recurrent::LSTM;

// Re-export will be added as modules are fixed
// pub use attention::{AttentionConfig, AttentionMask, MultiHeadAttention, SelfAttention};
