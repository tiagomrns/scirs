//! Neural network building blocks module for SciRS2 - Core working version
//!
//! This version includes the core working modules that compile successfully.
//! Complex architectures with Send + Sync issues are temporarily disabled.

#![warn(missing_docs)]
#![recursion_limit = "4096"]
// Core working modules
pub mod activations;
pub mod error;
pub mod layers;
pub mod losses;
pub mod optimizers;
// Utility modules
pub mod callbacks;
pub mod utils;
// Core model functionality
pub mod models;
// Transformer architectures
pub mod transformer;
// Additional stable modules
pub mod data;
pub mod evaluation;
// Configuration and serialization
pub mod config;
// Transfer learning capabilities
pub mod transfer_learning;
// Model interpretation and explainability
pub mod interpretation_legacy;
// Performance optimizations
pub mod performance;
// Quantization and compression
pub mod compression;
pub mod distillation;
pub mod quantization;
// Hardware acceleration
pub mod gpu;
pub mod hardware;
// Training utilities
pub mod training;
// Visualization
pub mod visualization;
// Memory efficient implementations
pub mod memory_efficient;
// Model serving and deployment
pub mod serving;
// Interoperability
pub mod interop;
// Just-In-Time compilation
pub mod jit;
// TPU compatibility infrastructure
pub mod tpu;
// Unified performance integration
pub mod performance_integration;
// Advanced Mode Coordinator
pub mod advanced_coordinator;
// Re-export the error type
pub use error::{Error, NeuralError, Result};
// Optional prelude for convenience
pub mod prelude {
    //! Convenient re-exports for common neural network operations
    // Core functionality
    pub use crate::activations::{Activation, Mish, ReLU, Sigmoid, Softmax, Swish, Tanh, GELU};
    pub use crate::error::{Error, NeuralError, Result};
    pub use crate::layers::{Dense, Layer, Sequential};
    pub use crate::losses::{Loss, MeanSquaredError};
    pub use crate::optimizers::Optimizer;
    // Configuration and serialization
    pub use crate::config::{ConfigBuilder, ConfigSerializer};
    // Transfer learning
    pub use crate::transfer_learning::{
        LayerState, PretrainedWeightLoader, TransferLearningManager, TransferLearningOrchestrator,
        TransferStrategy,
    };
    // Model interpretation
    pub use crate::interpretation_legacy::{
        AttributionMethod, BaselineMethod, InterpretationReport, ModelInterpreter,
    };
    // Training utilities
    pub use crate::training::mixed_precision::MixedPrecisionManager;
    // Data utilities
    pub use crate::data::{DataLoader, Dataset};
    // Evaluation
    pub use crate::evaluation::Evaluator;
    // JIT compilation
    pub use crate::jit::{JITCompiler, JITOperation, TargetArchitecture};
    // TPU support
    pub use crate::tpu::{TPUDevice, TPUOperation, TPURuntime};
    // Hardware acceleration
    pub use crate::hardware::DeviceManager;
    // Visualization
    pub use crate::visualization::NetworkVisualizer;
    // Unified performance optimization
    pub use crate::performance_integration::{
        AutoOptimizationStrategy, OptimizationChoice, UnifiedPerformanceManager,
    };
    // Advanced Mode Coordinator
    pub use crate::num_coordinator::{
        AdaptiveConfig, DeviceType, MemoryStrategy, OptimizationConfig, PerformanceReport,
        AdvancedCoordinator,
    };
}
