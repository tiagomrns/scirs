//! Optimization strategies for transformer-based learned optimizers
//!
//! This module contains various optimization strategies that work in conjunction
//! with the transformer architecture to improve optimization performance.

pub mod gradient_processing;
pub mod learning_rate_adaptation;
pub mod momentum_integration;
pub mod regularization;

// Re-export key types for convenience
pub use gradient_processing::{
    GradientProcessor, GradientProcessingStrategy, GradientProcessingParams, GradientStatistics
};
pub use learning_rate_adaptation::{
    LearningRateAdapter, LearningRateAdaptationStrategy, LRAdaptationParams, ScheduleState
};
pub use momentum_integration::{
    MomentumIntegrator, MomentumStrategy, MomentumParams, MomentumState, MomentumStatistics
};
pub use regularization::{
    TransformerRegularizer, RegularizationStrategy, RegularizationParams, ParameterStatistics
};