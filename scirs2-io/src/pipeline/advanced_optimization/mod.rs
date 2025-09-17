//! Advanced pipeline optimization with machine learning, quantum, and neuromorphic algorithms
//!
//! This module provides sophisticated optimization techniques for data processing pipelines,
//! including machine learning-based auto-tuning, quantum-inspired optimization, and
//! neuromorphic computing approaches.

pub mod config;
pub mod monitoring;
pub mod performance;
pub mod quantum;
pub mod neuromorphic;

// Re-export commonly used types
pub use config::{
    OptimizedPipelineConfig, MemoryStrategy, CacheConfiguration, CacheReplacementPolicy,
    PrefetchStrategy, BatchProcessingMode, ExecutionRecord, PipelinePerformanceMetrics,
    SystemMetrics, MemoryUsage, CachePerformance, NumaTopology, NumaNode,
    AutoTuningParameters, RegressionDetector, QuantumOptimizationConfig,
    NeuromorphicConfig, ConsciousnessConfig,
};

pub use monitoring::{ResourceMonitor};

pub use performance::{
    PerformanceHistory, PipelineProfile, PerformanceTrend, TrendDirection,
    AutoTuner, PerformancePredictor,
};

pub use quantum::{
    QuantumOptimizer, QuantumState, QuantumAnnealer,
};

pub use neuromorphic::{
    NeuromorphicOptimizer, SpikingNeuralNetwork, SpikingNeuron, SynapticConnection,
    PlasticityRule, NeuromorphicMemory, SpikePattern, AdaptationEvent,
};