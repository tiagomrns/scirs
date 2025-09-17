//! Task scheduling and resource management for optimization coordination
//!
//! This module provides task scheduling, resource allocation, and priority management
//! for coordinating multiple optimization processes.

#![allow(dead_code)]

pub mod task_scheduler;
pub mod resource_allocation;
pub mod priority_management;

// Re-export key types
pub use task_scheduler::{TaskScheduler, ScheduledTask, TaskPriority, SchedulingStrategy};
pub use resource_allocation::{ResourceManager, ResourcePool, ResourceAllocationTracker, 
                             ResourceOptimizationEngine, ResourceAllocationStrategy};
pub use priority_management::{PriorityManager, PriorityQueue, PriorityLevel, 
                             PriorityUpdateStrategy};