//! Orchestration for optimization coordination
//!
//! This module provides orchestration capabilities for complex optimization
//! workflows, including pipeline management, experiment lifecycle coordination,
//! and checkpoint/recovery systems.

#![allow(dead_code)]

pub mod pipeline_orchestrator;
pub mod experiment_manager;
pub mod checkpoint_manager;

// Re-export key types
pub use pipeline_orchestrator::{
    PipelineOrchestrator, OptimizationPipeline, PipelineStage, PipelineExecution,
    PipelineConfiguration, StageResult
};

pub use experiment_manager::{
    ExperimentManager, Experiment, ExperimentConfiguration, ExperimentExecution,
    ExperimentResult, ExperimentStatus
};

pub use checkpoint_manager::{
    CheckpointManager, Checkpoint, CheckpointConfiguration, CheckpointMetadata,
    RecoveryManager, RecoveryStrategy
};