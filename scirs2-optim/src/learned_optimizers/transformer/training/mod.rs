//! Training components for transformer-based learned optimizers
//!
//! This module contains the training infrastructure for transformer optimizers,
//! including meta-learning, curriculum learning, and evaluation frameworks.

pub mod meta_learning;
pub mod curriculum;
pub mod evaluation;

// Re-export key types for convenience
pub use meta_learning::{
    TransformerMetaLearner, MetaLearningStrategy, MetaTrainingEvent, MetaEventType,
    TaskInfo, TaskCharacteristics, DomainInfo, DomainType, MetaPerformanceMetrics,
    DomainAdapter, FewShotLearner, ContinualLearningState
};
pub use curriculum::{
    CurriculumLearner, CurriculumStrategy, CurriculumParams, TaskDifficultyEstimator,
    LearningProgressTracker, CurriculumState, TaskScheduler, LearningPhase
};
pub use evaluation::{
    TransformerEvaluator, EvaluationStrategy, EvaluationResult, ConvergenceInfo,
    EfficiencyMetrics, StatisticalSignificance, RobustnessTestSuite, AggregationMethod
};