//! Production-grade distributed computing infrastructure
//!
//! This module provides comprehensive distributed computing capabilities
//! for SciRS2 Core 1.0, including distributed arrays, cluster management,
//! fault tolerance, and scalable computation orchestration.

pub mod array;
pub mod cluster;
pub mod communication;
pub mod fault_tolerance;
pub mod load_balancing;
pub mod orchestration;
pub mod scheduler;

// Array operations
pub use array::{DistributedArray, DistributedArrayManager};

// Cluster management
pub use cluster::{
    initialize_cluster_manager, BackoffStrategy, ClusterConfiguration, ClusterEventLog,
    ClusterHealth, ClusterManager, ClusterState, ComputeCapacity, DistributedTask,
    NodeCapabilities, NodeInfo as ClusterNodeInfo, NodeMetadata, NodeStatus, NodeType,
    ResourceRequirements, RetryPolicy, TaskId, TaskParameters, TaskPriority as ClusterTaskPriority,
    TaskType,
};

// Communication
pub use communication::{
    CommunicationEndpoint, CommunicationManager, DistributedMessage, HeartbeatHandler,
    MessageHandler,
};

// Fault tolerance
pub use fault_tolerance::{
    initialize_fault_tolerance, ClusterHealthSummary, FaultDetectionStrategy,
    FaultToleranceManager, NodeHealth as FaultNodeHealth, NodeInfo as FaultNodeInfo,
    RecoveryStrategy,
};

// Load balancing
pub use load_balancing::{
    LoadBalancer as DistributedLoadBalancer, LoadBalancingStats, LoadBalancingStrategy,
    NodeLoad as LoadBalancerNodeLoad, TaskAssignment as LoadBalancerTaskAssignment,
};

// Orchestration
pub use orchestration::{
    OrchestrationEngine, OrchestrationStats, OrchestratorNode, Task as OrchestrationTask,
    TaskPriority as OrchestrationTaskPriority, TaskStatus as OrchestrationTaskStatus, Workflow,
    WorkflowStatus,
};

// Scheduler
pub use scheduler::{
    initialize_distributed_scheduler, CompletedTask, DistributedScheduler, ExecutionTracker,
    FailedTask, LoadBalancer as SchedulerLoadBalancer,
    LoadBalancingStrategy as SchedulerLoadBalancingStrategy, NodeLoad as SchedulerNodeLoad,
    SchedulingAlgorithm, SchedulingPolicies, TaskAssignment as SchedulerTaskAssignment, TaskQueue,
};

/// Initialize distributed computing infrastructure
#[allow(dead_code)]
pub fn initialize_distributed_computing() -> crate::error::CoreResult<()> {
    cluster::initialize_cluster_manager()?;
    scheduler::initialize_distributed_scheduler()?;
    fault_tolerance::initialize_fault_tolerance()?;
    Ok(())
}

/// Get distributed system status
#[allow(dead_code)]
pub fn get_distributed_status() -> crate::error::CoreResult<DistributedSystemStatus> {
    let cluster_manager = cluster::ClusterManager::global()?;
    let scheduler = scheduler::DistributedScheduler::global()?;

    Ok(DistributedSystemStatus {
        cluster_health: cluster_manager.get_health()?,
        active_nodes: cluster_manager.get_active_nodes()?.len(),
        pending_tasks: scheduler.get_pending_task_count()?,
        total_capacity: cluster_manager.get_total_capacity()?,
    })
}

#[derive(Debug, Clone)]
pub struct DistributedSystemStatus {
    pub cluster_health: ClusterHealth,
    pub active_nodes: usize,
    pub pending_tasks: usize,
    pub total_capacity: ComputeCapacity,
}
