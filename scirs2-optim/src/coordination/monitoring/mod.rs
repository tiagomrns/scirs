//! Monitoring for optimization coordination
//!
//! This module provides comprehensive monitoring capabilities for optimization
//! processes, including performance tracking, convergence detection, and
//! anomaly detection for optimization workflows.

#![allow(dead_code)]

pub mod performance_tracking;
pub mod convergence_detection;
pub mod anomaly_detection;

// Re-export key types
pub use performance_tracking::{
    PerformanceTracker, PerformanceMetrics, MetricCollector, MetricAggregator,
    PerformanceAlert, AlertManager
};

pub use convergence_detection::{
    ConvergenceDetector, ConvergenceAnalyzer, ConvergenceCriteria, ConvergenceResult,
    ConvergenceMonitor, ConvergenceIndicator
};

pub use anomaly_detection::{
    AnomalyDetector, AnomalyAnalyzer, AnomalyAlert, AnomalyClassifier,
    OutlierDetector, AnomalyReporter
};