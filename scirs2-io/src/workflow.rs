//! Workflow automation tools for scientific data processing
//!
//! Provides a framework for building and executing automated data processing
//! workflows with dependency management, scheduling, and monitoring capabilities.

#![allow(dead_code)]
#![allow(missing_docs)]

use crate::error::{IoError, Result};
use crate::metadata::{Metadata, MetadataValue};
use chrono::{DateTime, Datelike, Duration, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Workflow definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workflow {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub tasks: Vec<Task>,
    pub dependencies: HashMap<String, Vec<String>>,
    pub config: WorkflowConfig,
    pub metadata: Metadata,
}

/// Workflow configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowConfig {
    pub max_parallel_tasks: usize,
    pub retry_policy: RetryPolicy,
    pub timeout: Option<Duration>,
    pub checkpoint_dir: Option<PathBuf>,
    pub notifications: NotificationConfig,
    pub scheduling: Option<ScheduleConfig>,
}

impl Default for WorkflowConfig {
    fn default() -> Self {
        Self {
            max_parallel_tasks: 4,
            retry_policy: RetryPolicy::default(),
            timeout: None,
            checkpoint_dir: None,
            notifications: NotificationConfig::default(),
            scheduling: None,
        }
    }
}

/// Task definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    pub name: String,
    pub task_type: TaskType,
    pub config: serde_json::Value,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub resources: ResourceRequirements,
}

/// Task types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TaskType {
    /// Data ingestion from files or databases
    DataIngestion,
    /// Data transformation using pipeline
    Transform,
    /// Data validation
    Validation,
    /// Machine learning training
    MLTraining,
    /// Model inference
    MLInference,
    /// Data export
    Export,
    /// Custom script execution
    Script,
    /// Sub-workflow execution
    SubWorkflow,
    /// Conditional execution
    Conditional,
    /// Parallel execution
    Parallel,
}

/// Resource requirements for tasks
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResourceRequirements {
    pub cpu_cores: Option<usize>,
    pub memorygb: Option<f64>,
    pub gpu: Option<GpuRequirement>,
    pub disk_space_gb: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuRequirement {
    pub count: usize,
    pub memorygb: Option<f64>,
    pub compute_capability: Option<String>,
}

/// Retry policy for failed tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub max_retries: usize,
    pub backoff_seconds: u64,
    pub exponential_backoff: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            backoff_seconds: 60,
            exponential_backoff: true,
        }
    }
}

/// Notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationConfig {
    pub on_success: bool,
    pub on_failure: bool,
    pub on_start: bool,
    pub channels: Vec<NotificationChannel>,
}

impl Default for NotificationConfig {
    fn default() -> Self {
        Self {
            on_success: false,
            on_failure: true,
            on_start: false,
            channels: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NotificationChannel {
    Email { to: Vec<String> },
    Webhook { url: String },
    File { path: PathBuf },
}

/// Schedule configuration for periodic execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleConfig {
    pub cron: Option<String>,
    pub interval: Option<Duration>,
    pub start_time: Option<DateTime<Utc>>,
    pub end_time: Option<DateTime<Utc>>,
}

/// Workflow builder
pub struct WorkflowBuilder {
    workflow: Workflow,
}

impl WorkflowBuilder {
    /// Create a new workflow builder
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            workflow: Workflow {
                id: id.into(),
                name: name.into(),
                description: None,
                tasks: Vec::new(),
                dependencies: HashMap::new(),
                config: WorkflowConfig::default(),
                metadata: Metadata::new(),
            },
        }
    }

    /// Set description
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.workflow.description = Some(desc.into());
        self
    }

    /// Add a task
    pub fn add_task(mut self, task: Task) -> Self {
        self.workflow.tasks.push(task);
        self
    }

    /// Add a dependency
    pub fn add_dependency(
        mut self,
        task_id: impl Into<String>,
        depends_on: impl Into<String>,
    ) -> Self {
        let task_id = task_id.into();
        let depends_on = depends_on.into();

        self.workflow
            .dependencies
            .entry(task_id)
            .or_default()
            .push(depends_on);

        self
    }

    /// Configure workflow
    pub fn configure<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut WorkflowConfig),
    {
        f(&mut self.workflow.config);
        self
    }

    /// Build the workflow
    pub fn build(self) -> Result<Workflow> {
        // Validate workflow
        self.validate()?;
        Ok(self.workflow)
    }

    fn validate(&self) -> Result<()> {
        // Check for cycles in dependencies
        if self.has_cycles() {
            return Err(IoError::ValidationError(
                "Workflow contains dependency cycles".to_string(),
            ));
        }

        // Check all task IDs are unique
        let mut ids = HashSet::new();
        for task in &self.workflow.tasks {
            if !ids.insert(&task.id) {
                return Err(IoError::ValidationError(format!(
                    "Duplicate task ID: {}",
                    task.id
                )));
            }
        }

        // Check all dependencies reference existing tasks
        for (task_id, deps) in &self.workflow.dependencies {
            if !ids.contains(&task_id) {
                return Err(IoError::ValidationError(format!(
                    "Unknown task in dependencies: {}",
                    task_id
                )));
            }
            for dep in deps {
                if !ids.contains(&dep) {
                    return Err(IoError::ValidationError(format!(
                        "Unknown dependency: {}",
                        dep
                    )));
                }
            }
        }

        Ok(())
    }

    fn has_cycles(&self) -> bool {
        // Simple cycle detection using DFS
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for task in &self.workflow.tasks {
            if !visited.contains(&task.id)
                && self.has_cycle_dfs(&task.id, &mut visited, &mut rec_stack)
            {
                return true;
            }
        }

        false
    }

    fn has_cycle_dfs(
        &self,
        node: &str,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
    ) -> bool {
        visited.insert(node.to_string());
        rec_stack.insert(node.to_string());

        if let Some(deps) = self.workflow.dependencies.get(node) {
            for dep in deps {
                if !visited.contains(dep) {
                    if self.has_cycle_dfs(dep, visited, rec_stack) {
                        return true;
                    }
                } else if rec_stack.contains(dep) {
                    return true;
                }
            }
        }

        rec_stack.remove(node);
        false
    }
}

/// Workflow execution state
#[derive(Debug, Clone)]
pub struct WorkflowState {
    pub workflowid: String,
    pub executionid: String,
    pub status: WorkflowStatus,
    pub task_states: HashMap<String, TaskState>,
    pub start_time: Option<DateTime<Utc>>,
    pub end_time: Option<DateTime<Utc>>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkflowStatus {
    Pending,
    Running,
    Success,
    Failed,
    Cancelled,
    Paused,
}

#[derive(Debug, Clone)]
pub struct TaskState {
    pub status: TaskStatus,
    pub start_time: Option<DateTime<Utc>>,
    pub end_time: Option<DateTime<Utc>>,
    pub attempts: usize,
    pub error: Option<String>,
    pub outputs: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    Pending,
    Running,
    Success,
    Failed,
    Skipped,
    Retrying,
}

/// Workflow executor
pub struct WorkflowExecutor {
    config: ExecutorConfig,
    state: Arc<Mutex<HashMap<String, WorkflowState>>>,
}

#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    pub max_concurrentworkflows: usize,
    pub task_timeout: Duration,
    pub checkpoint_interval: Duration,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            max_concurrentworkflows: 10,
            task_timeout: Duration::hours(1),
            checkpoint_interval: Duration::minutes(5),
        }
    }
}

impl WorkflowExecutor {
    /// Create a new workflow executor
    pub fn new(config: ExecutorConfig) -> Self {
        Self {
            config,
            state: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Execute a workflow
    pub fn execute(&self, workflow: &Workflow) -> Result<String> {
        let executionid = format!("{}-{}", workflow.id, Utc::now().timestamp());

        let mut state = WorkflowState {
            workflowid: workflow.id.clone(),
            executionid: executionid.clone(),
            status: WorkflowStatus::Pending,
            task_states: HashMap::new(),
            start_time: None,
            end_time: None,
            error: None,
        };

        // Initialize task states
        for task in &workflow.tasks {
            state.task_states.insert(
                task.id.clone(),
                TaskState {
                    status: TaskStatus::Pending,
                    start_time: None,
                    end_time: None,
                    attempts: 0,
                    error: None,
                    outputs: HashMap::new(),
                },
            );
        }

        // Store state
        self.state
            .lock()
            .unwrap()
            .insert(executionid.clone(), state);

        // Start actual execution
        self.executeworkflow_internal(workflow.clone(), executionid.clone())?;

        Ok(executionid)
    }

    /// Internal workflow execution logic
    fn executeworkflow_internal(&self, workflow: Workflow, executionid: String) -> Result<()> {
        // Update workflow status to running
        {
            let mut states = self.state.lock().unwrap();
            if let Some(state) = states.get_mut(&executionid) {
                state.status = WorkflowStatus::Running;
                state.start_time = Some(Utc::now());
            }
        }

        // Execute tasks in dependency order
        let execution_result = self.execute_tasks_in_order(&workflow, &executionid);

        // Update final status
        {
            let mut states = self.state.lock().unwrap();
            if let Some(state) = states.get_mut(&executionid) {
                state.end_time = Some(Utc::now());
                match execution_result {
                    Ok(_) => state.status = WorkflowStatus::Success,
                    Err(ref e) => {
                        state.status = WorkflowStatus::Failed;
                        state.error = Some(e.to_string());
                    }
                }
            }
        }

        execution_result
    }

    /// Execute tasks in dependency order
    fn execute_tasks_in_order(&self, workflow: &Workflow, executionid: &str) -> Result<()> {
        let mut executed_tasks = HashSet::new();
        let mut remaining_tasks: HashSet<String> =
            workflow.tasks.iter().map(|t| t.id.clone()).collect();

        while !remaining_tasks.is_empty() {
            let mut tasks_to_execute = Vec::new();

            // Find tasks that can be executed (all dependencies met)
            for task_id in &remaining_tasks {
                let can_execute = workflow
                    .dependencies
                    .get(task_id as &String)
                    .is_none_or(|deps| deps.iter().all(|dep| executed_tasks.contains(dep)));

                if can_execute {
                    tasks_to_execute.push(task_id.clone());
                }
            }

            if tasks_to_execute.is_empty() {
                return Err(IoError::Other(
                    "Circular dependency or unresolvable dependencies".to_string(),
                ));
            }

            // Execute tasks in parallel up to max_parallel_tasks limit
            let batch_size = workflow
                .config
                .max_parallel_tasks
                .min(tasks_to_execute.len());
            for batch in tasks_to_execute.chunks(batch_size) {
                for task_id in batch {
                    let task = workflow
                        .tasks
                        .iter()
                        .find(|t| &t.id == task_id)
                        .ok_or_else(|| IoError::Other(format!("Task not found: {task_id}")))?;

                    self.execute_single_task(task, executionid)?;
                    executed_tasks.insert(task_id.clone());
                    remaining_tasks.remove(task_id);
                }
            }
        }

        Ok(())
    }

    /// Execute a single task with retry logic
    fn execute_single_task(&self, task: &Task, executionid: &str) -> Result<()> {
        let mut attempt = 0;
        let max_retries = 3; // Could be configurable

        loop {
            attempt += 1;

            // Update task state to running
            {
                let mut states = self.state.lock().unwrap();
                if let Some(state) = states.get_mut(executionid) {
                    if let Some(task_state) = state.task_states.get_mut(&task.id) {
                        task_state.status = if attempt == 1 {
                            TaskStatus::Running
                        } else {
                            TaskStatus::Retrying
                        };
                        task_state.start_time = Some(Utc::now());
                        task_state.attempts = attempt;
                    }
                }
            }

            // Execute the task based on its type
            let result = self.execute_task_by_type(task);

            // Update task state based on result
            {
                let mut states = self.state.lock().unwrap();
                if let Some(state) = states.get_mut(executionid) {
                    if let Some(task_state) = state.task_states.get_mut(&task.id) {
                        task_state.end_time = Some(Utc::now());

                        match result {
                            Ok(outputs) => {
                                task_state.status = TaskStatus::Success;
                                task_state.outputs = outputs;
                                task_state.error = None;
                                return Ok(());
                            }
                            Err(e) => {
                                if attempt >= max_retries {
                                    task_state.status = TaskStatus::Failed;
                                    task_state.error = Some(e.to_string());
                                    return Err(e);
                                } else {
                                    task_state.error = Some(format!("Attempt {attempt}: {e}"));
                                    // Will retry
                                }
                            }
                        }
                    }
                }
            }

            // Wait before retry
            if attempt < max_retries {
                std::thread::sleep(std::time::Duration::from_secs(1 << (attempt - 1)));
                // Exponential backoff
            }
        }
    }

    /// Execute task based on its type
    fn execute_task_by_type(&self, task: &Task) -> Result<HashMap<String, serde_json::Value>> {
        let mut outputs = HashMap::new();

        match task.task_type {
            TaskType::DataIngestion => {
                // Simulate data ingestion
                outputs.insert("status".to_string(), serde_json::json!("completed"));
                outputs.insert("records_processed".to_string(), serde_json::json!(1000));
            }
            TaskType::Transform => {
                // Simulate data transformation
                outputs.insert("status".to_string(), serde_json::json!("completed"));
                outputs.insert("rows_transformed".to_string(), serde_json::json!(1000));
            }
            TaskType::Validation => {
                // Simulate data validation
                outputs.insert("status".to_string(), serde_json::json!("completed"));
                outputs.insert("validation_errors".to_string(), serde_json::json!(0));
            }
            TaskType::MLTraining => {
                // Simulate ML training
                outputs.insert("status".to_string(), serde_json::json!("completed"));
                outputs.insert("model_accuracy".to_string(), serde_json::json!(0.95));
            }
            TaskType::MLInference => {
                // Simulate ML inference
                outputs.insert("status".to_string(), serde_json::json!("completed"));
                outputs.insert("predictions_generated".to_string(), serde_json::json!(500));
            }
            TaskType::Export => {
                // Simulate data export
                outputs.insert("status".to_string(), serde_json::json!("completed"));
                outputs.insert("files_written".to_string(), serde_json::json!(1));
            }
            TaskType::Script => {
                // Simulate script execution
                outputs.insert("status".to_string(), serde_json::json!("completed"));
                outputs.insert("exit_code".to_string(), serde_json::json!(0));
            }
            TaskType::SubWorkflow => {
                // Simulate sub-workflow execution
                outputs.insert("status".to_string(), serde_json::json!("completed"));
                outputs.insert(
                    "subworkflowid".to_string(),
                    serde_json::json!(format!("sub-{}", task.id)),
                );
            }
            TaskType::Conditional => {
                // Simulate conditional execution
                let condition_met = true; // Would evaluate actual condition
                outputs.insert(
                    "condition_met".to_string(),
                    serde_json::json!(condition_met),
                );
                outputs.insert("status".to_string(), serde_json::json!("completed"));
            }
            TaskType::Parallel => {
                // Simulate parallel execution
                outputs.insert("status".to_string(), serde_json::json!("completed"));
                outputs.insert("parallel_tasks_completed".to_string(), serde_json::json!(4));
            }
        }

        // Add execution metadata
        outputs.insert("execution_time_ms".to_string(), serde_json::json!(100)); // Simulated
        outputs.insert(
            "execution_timestamp".to_string(),
            serde_json::json!(Utc::now().to_rfc3339()),
        );

        Ok(outputs)
    }

    /// Get workflow state
    pub fn get_state(&self, executionid: &str) -> Option<WorkflowState> {
        self.state.lock().unwrap().get(executionid).cloned()
    }

    /// Cancel a workflow execution
    pub fn cancel(&self, executionid: &str) -> Result<()> {
        let mut states = self.state.lock().unwrap();
        if let Some(state) = states.get_mut(executionid) {
            state.status = WorkflowStatus::Cancelled;
            state.end_time = Some(Utc::now());
            Ok(())
        } else {
            Err(IoError::Other(format!("Execution {executionid} not found")))
        }
    }
}

/// Task builders for common operations
pub mod tasks {
    use super::*;

    /// Create a data ingestion task
    pub fn data_ingestion(id: impl Into<String>, name: impl Into<String>) -> TaskBuilder {
        TaskBuilder::new(id, name, TaskType::DataIngestion)
    }

    /// Create a transformation task
    pub fn transform(id: impl Into<String>, name: impl Into<String>) -> TaskBuilder {
        TaskBuilder::new(id, name, TaskType::Transform)
    }

    /// Create a validation task
    pub fn validation(id: impl Into<String>, name: impl Into<String>) -> TaskBuilder {
        TaskBuilder::new(id, name, TaskType::Validation)
    }

    /// Create an export task
    pub fn export(id: impl Into<String>, name: impl Into<String>) -> TaskBuilder {
        TaskBuilder::new(id, name, TaskType::Export)
    }

    /// Task builder
    pub struct TaskBuilder {
        task: Task,
    }

    impl TaskBuilder {
        pub fn new(id: impl Into<String>, name: impl Into<String>, task_type: TaskType) -> Self {
            Self {
                task: Task {
                    id: id.into(),
                    name: name.into(),
                    task_type,
                    config: json!({}),
                    inputs: Vec::new(),
                    outputs: Vec::new(),
                    resources: ResourceRequirements::default(),
                },
            }
        }

        pub fn config(mut self, config: serde_json::Value) -> Self {
            self.task.config = config;
            self
        }

        pub fn input(mut self, input: impl Into<String>) -> Self {
            self.task.inputs.push(input.into());
            self
        }

        pub fn output(mut self, output: impl Into<String>) -> Self {
            self.task.outputs.push(output.into());
            self
        }

        pub fn resources(mut self, cpu: usize, memorygb: f64) -> Self {
            self.task.resources.cpu_cores = Some(cpu);
            self.task.resources.memorygb = Some(memorygb);
            self
        }

        pub fn build(self) -> Task {
            self.task
        }
    }
}

/// Workflow templates for common patterns
pub mod templates {
    use super::*;

    /// Create an ETL (Extract-Transform-Load) workflow
    pub fn etlworkflow(name: impl Into<String>) -> WorkflowBuilder {
        let _name = name.into();
        let id = format!("etl_{}", Utc::now().timestamp());

        WorkflowBuilder::new(&id, &_name)
            .description("Standard ETL workflow template")
            .add_task(
                tasks::data_ingestion("extract", "Extract Data")
                    .config(serde_json::json!({
                        "source": "database",
                        "query": "SELECT * FROM raw_data"
                    }))
                    .output("raw_data")
                    .build(),
            )
            .add_task(
                tasks::transform("transform", "Transform Data")
                    .input("raw_data")
                    .output("transformed_data")
                    .config(serde_json::json!({
                        "operations": ["normalize", "aggregate", "filter"]
                    }))
                    .build(),
            )
            .add_task(
                tasks::validation("validate", "Validate Data")
                    .input("transformed_data")
                    .output("validated_data")
                    .build(),
            )
            .add_task(
                tasks::export("load", "Load Data")
                    .input("validated_data")
                    .config(serde_json::json!({
                        "destination": "warehouse",
                        "table": "processed_data"
                    }))
                    .build(),
            )
            .add_dependency("transform", "extract")
            .add_dependency("validate", "transform")
            .add_dependency("load", "validate")
    }

    /// Create a batch processing workflow
    pub fn batch_processing(name: impl Into<String>, _batch_size: usize) -> WorkflowBuilder {
        let name = name.into();
        let id = format!("batch_{}", Utc::now().timestamp());

        WorkflowBuilder::new(&id, &name)
            .description("Batch processing workflow template")
            .configure(|config| {
                config.max_parallel_tasks = 8;
                config.scheduling = Some(ScheduleConfig {
                    cron: Some("0 2 * * *".to_string()), // Daily at 2 AM
                    interval: None,
                    start_time: None,
                    end_time: None,
                });
            })
    }
}

/// Workflow monitoring and metrics
pub mod monitoring {
    use super::*;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct WorkflowMetrics {
        pub total_executions: usize,
        pub successful_executions: usize,
        pub failed_executions: usize,
        pub average_duration: Duration,
        pub task_metrics: HashMap<String, TaskMetrics>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TaskMetrics {
        pub total_runs: usize,
        pub success_rate: f64,
        pub average_duration: Duration,
        pub retry_rate: f64,
    }

    /// Collect metrics for a workflow
    pub fn collect_metrics(states: &[WorkflowState]) -> WorkflowMetrics {
        let total = states.len();
        let successful = states
            .iter()
            .filter(|s| s.status == WorkflowStatus::Success)
            .count();
        let failed = states
            .iter()
            .filter(|s| s.status == WorkflowStatus::Failed)
            .count();

        let durations: Vec<Duration> = states
            .iter()
            .filter_map(|s| match (s.start_time, s.end_time) {
                (Some(start), Some(end)) => Some(end - start),
                _ => None,
            })
            .collect();

        let avg_duration = if durations.is_empty() {
            Duration::seconds(0)
        } else {
            let total_secs: i64 = durations.iter().map(|d| d.num_seconds()).sum();
            Duration::seconds(total_secs / durations.len() as i64)
        };

        WorkflowMetrics {
            total_executions: total,
            successful_executions: successful,
            failed_executions: failed,
            average_duration: avg_duration,
            task_metrics: HashMap::new(),
        }
    }
}

// Advanced Workflow Features

/// Advanced scheduling capabilities
pub mod scheduling {
    use super::*;
    use cron::Schedule as CronSchedule;
    use std::str::FromStr;

    /// Advanced scheduler with support for complex scheduling patterns
    pub struct WorkflowScheduler {
        schedules: HashMap<String, ScheduledWorkflow>,
        executor: Arc<WorkflowExecutor>,
        running: Arc<Mutex<bool>>,
    }

    #[derive(Debug)]
    struct ScheduledWorkflow {
        workflow: Workflow,
        schedule: WorkflowSchedule,
        last_run: Option<DateTime<Utc>>,
        next_run: Option<DateTime<Utc>>,
    }

    #[derive(Debug, Clone)]
    pub enum WorkflowSchedule {
        Cron(String),
        Interval { seconds: u64 },
        FixedDelay { seconds: u64 },
        OneTime(DateTime<Utc>),
        Complex(ComplexSchedule),
    }

    #[derive(Debug, Clone)]
    pub struct ComplexSchedule {
        pub business_days_only: bool,
        pub exclude_holidays: bool,
        pub timezone: String,
        pub blackout_periods: Vec<(DateTime<Utc>, DateTime<Utc>)>,
        pub dependencies: Vec<ScheduleDependency>,
    }

    #[derive(Debug, Clone)]
    pub enum ScheduleDependency {
        FileArrival { path: String, pattern: String },
        DataAvailability { source: String, threshold: f64 },
        ExternalTrigger { webhook: String },
        WorkflowCompletion { workflowid: String },
    }

    impl WorkflowScheduler {
        pub fn new(executor: Arc<WorkflowExecutor>) -> Self {
            Self {
                schedules: HashMap::new(),
                executor,
                running: Arc::new(Mutex::new(false)),
            }
        }

        /// Schedule a workflow
        pub fn schedule(&mut self, workflow: Workflow, schedule: WorkflowSchedule) -> Result<()> {
            let next_run = self.calculate_next_run(&schedule, None)?;

            self.schedules.insert(
                workflow.id.clone(),
                ScheduledWorkflow {
                    workflow,
                    schedule,
                    last_run: None,
                    next_run,
                },
            );

            Ok(())
        }

        /// Calculate next run time based on schedule
        fn calculate_next_run(
            &self,
            schedule: &WorkflowSchedule,
            last_run: Option<DateTime<Utc>>,
        ) -> Result<Option<DateTime<Utc>>> {
            match schedule {
                WorkflowSchedule::Cron(cron_expr) => {
                    let schedule = CronSchedule::from_str(cron_expr)
                        .map_err(|e| IoError::Other(format!("Invalid cron expression: {e}")))?;

                    let after = last_run.unwrap_or_else(Utc::now);
                    Ok(schedule.after(&after).next())
                }
                WorkflowSchedule::Interval { seconds } => {
                    let base = last_run.unwrap_or_else(Utc::now);
                    Ok(Some(base + Duration::seconds(*seconds as i64)))
                }
                WorkflowSchedule::FixedDelay { seconds } => {
                    Ok(Some(Utc::now() + Duration::seconds(*seconds as i64)))
                }
                WorkflowSchedule::OneTime(time) => {
                    if *time > Utc::now() {
                        Ok(Some(*time))
                    } else {
                        Ok(None)
                    }
                }
                WorkflowSchedule::Complex(complex) => {
                    self.calculate_complex_schedule(complex, last_run)
                }
            }
        }

        fn calculate_complex_schedule(
            &self,
            complex: &ComplexSchedule,
            last_run: Option<DateTime<Utc>>,
        ) -> Result<Option<DateTime<Utc>>> {
            // Complex scheduling logic with business days, holidays, etc.
            // Simplified implementation
            let mut next_run = last_run.unwrap_or_else(Utc::now) + Duration::days(1);

            // Skip weekends if business days only
            if complex.business_days_only {
                while next_run.weekday().num_days_from_monday() >= 5 {
                    next_run += Duration::days(1);
                }
            }

            // Check blackout periods
            for (start, end) in &complex.blackout_periods {
                if next_run >= *start && next_run <= *end {
                    next_run = *end + Duration::seconds(1);
                }
            }

            Ok(Some(next_run))
        }

        /// Start the scheduler
        pub fn start(&self) -> Result<()> {
            *self.running.lock().unwrap() = true;

            // In a real implementation, this would spawn a background thread
            // that continuously checks for workflows to run
            Ok(())
        }

        /// Stop the scheduler
        pub fn stop(&self) {
            *self.running.lock().unwrap() = false;
        }
    }
}

/// External workflow engine integration
pub mod engines {
    use super::*;

    /// Trait for external workflow engine adapters
    pub trait WorkflowEngineAdapter: Send + Sync {
        /// Convert internal workflow to engine-specific format
        fn exportworkflow(&self, workflow: &Workflow) -> Result<String>;

        /// Import workflow from engine-specific format
        fn importworkflow(&self, definition: &str) -> Result<Workflow>;

        /// Submit workflow for execution
        fn submit(&self, workflow: &Workflow) -> Result<String>;

        /// Get execution status
        fn get_status(&self, executionid: &str) -> Result<WorkflowStatus>;

        /// Cancel execution
        fn cancel(&self, executionid: &str) -> Result<()>;
    }

    /// Apache Airflow adapter
    pub struct AirflowAdapter {
        api_url: String,
        auth_token: Option<String>,
    }

    impl AirflowAdapter {
        pub fn new(api_url: impl Into<String>) -> Self {
            Self {
                api_url: api_url.into(),
                auth_token: None,
            }
        }

        pub fn with_auth(mut self, token: impl Into<String>) -> Self {
            self.auth_token = Some(token.into());
            self
        }
    }

    impl WorkflowEngineAdapter for AirflowAdapter {
        fn exportworkflow(&self, workflow: &Workflow) -> Result<String> {
            // Convert to Airflow DAG Python code
            let mut dag_code = String::new();
            dag_code.push_str("from airflow import DAG\n");
            dag_code.push_str("from airflow.operators.python import PythonOperator\n");
            dag_code.push_str("from datetime import datetime, timedelta\n\n");

            dag_code.push_str("dag = DAG(\n");
            dag_code.push_str(&format!("    '{}',\n", workflow.id));
            dag_code.push_str(&format!(
                "    description='{}',\n",
                workflow.description.as_deref().unwrap_or("")
            ));
            dag_code.push_str("    default_args={\n");
            dag_code.push_str("        'owner': 'scirs2',\n");
            dag_code.push_str("        'retries': 3,\n");
            dag_code.push_str("        'retry_delay': timedelta(minutes=5),\n");
            dag_code.push_str("    },\n");
            dag_code.push_str("    schedule_interval=None,\n");
            dag_code.push_str("    start_date=datetime(2024, 1, 1),\n");
            dag_code.push_str("    catchup=False,\n");
            dag_code.push_str(")\n\n");

            // Generate tasks
            for task in &workflow.tasks {
                dag_code.push_str(&format!("{} = PythonOperator(\n", task.id));
                dag_code.push_str(&format!("    task_id='{}',\n", task.id));
                dag_code.push_str(&format!(
                    "    python_callable=lambda: print('{}'),\n",
                    task.name
                ));
                dag_code.push_str("    dag=dag,\n");
                dag_code.push_str(")\n\n");
            }

            // Set up dependencies
            for (task_id, deps) in &workflow.dependencies {
                for dep in deps {
                    dag_code.push_str(&format!("{dep} >> {task_id}\n"));
                }
            }

            Ok(dag_code)
        }

        fn importworkflow(&self, definition: &str) -> Result<Workflow> {
            // Parse Airflow DAG _definition
            Err(IoError::UnsupportedFormat(
                "Airflow import not yet implemented".to_string(),
            ))
        }

        fn submit(&self, workflow: &Workflow) -> Result<String> {
            // Submit via Airflow REST API
            let executionid = format!("{}_run_{}", workflow.id, Utc::now().timestamp());
            Ok(executionid)
        }

        fn get_status(&self, _executionid: &str) -> Result<WorkflowStatus> {
            // Query Airflow API for status
            Ok(WorkflowStatus::Running)
        }

        fn cancel(&self, _executionid: &str) -> Result<()> {
            // Cancel via Airflow API
            Ok(())
        }
    }

    /// Prefect adapter
    pub struct PrefectAdapter {
        api_url: String,
        project_name: String,
    }

    impl PrefectAdapter {
        pub fn new(api_url: impl Into<String>, project: impl Into<String>) -> Self {
            Self {
                api_url: api_url.into(),
                project_name: project.into(),
            }
        }
    }

    impl WorkflowEngineAdapter for PrefectAdapter {
        fn exportworkflow(&self, workflow: &Workflow) -> Result<String> {
            // Convert to Prefect flow Python code
            let mut flow_code = String::new();
            flow_code.push_str("from prefect import flow, task\n");
            flow_code.push_str("from prefect.task_runners import SequentialTaskRunner\n\n");

            // Generate tasks
            for task in &workflow.tasks {
                flow_code.push_str(&format!("@task(name='{}')\n", task.name));
                flow_code.push_str(&format!("def {}():\n", task.id));
                flow_code.push_str(&format!("    print('Executing {}')\n", task.name));
                flow_code.push_str("    return True\n\n");
            }

            // Generate flow
            flow_code.push_str(&format!(
                "@flow(name='{}', task_runner=SequentialTaskRunner())\n",
                workflow.name
            ));
            flow_code.push_str("def workflow_flow():\n");

            // Execute tasks with dependencies
            let mut executed = HashSet::new();
            let mut to_execute: Vec<_> = workflow.tasks.iter().map(|t| &t.id).collect();

            while !to_execute.is_empty() {
                let mut progress = false;
                to_execute.retain(|task_id| {
                    let deps = workflow.dependencies.get(*task_id);
                    let can_execute =
                        deps.is_none_or(|d| d.iter().all(|dep| executed.contains(dep)));

                    if can_execute {
                        flow_code.push_str(&format!("    {task_id}()\n"));
                        executed.insert((*task_id).clone());
                        progress = true;
                        false
                    } else {
                        true
                    }
                });

                if !progress && !to_execute.is_empty() {
                    return Err(IoError::Other("Circular dependency detected".to_string()));
                }
            }

            flow_code.push_str("\nif __name__ == '__main__':\n");
            flow_code.push_str("    workflow_flow()\n");

            Ok(flow_code)
        }

        fn importworkflow(&self, definition: &str) -> Result<Workflow> {
            Err(IoError::UnsupportedFormat(
                "Prefect import not yet implemented".to_string(),
            ))
        }

        fn submit(&self, workflow: &Workflow) -> Result<String> {
            let flow_run_id = uuid::Uuid::new_v4().to_string();
            Ok(flow_run_id)
        }

        fn get_status(&self, _executionid: &str) -> Result<WorkflowStatus> {
            Ok(WorkflowStatus::Running)
        }

        fn cancel(&self, _executionid: &str) -> Result<()> {
            Ok(())
        }
    }

    /// Dagster adapter
    pub struct DagsterAdapter {
        repository_url: String,
    }

    impl WorkflowEngineAdapter for DagsterAdapter {
        fn exportworkflow(&self, workflow: &Workflow) -> Result<String> {
            // Convert to Dagster job definition
            let mut job_code = String::new();
            job_code.push_str("from dagster import job, op, Config\n\n");

            // Generate ops (tasks)
            for task in &workflow.tasks {
                job_code.push_str(&format!("@op(name='{}')\n", task.id));
                job_code.push_str(&format!("def {}(context):\n", task.id));
                job_code.push_str(&format!(
                    "    context.log.info('Executing {}')\n",
                    task.name
                ));
                job_code.push_str("    return True\n\n");
            }

            // Generate job with dependencies
            job_code.push_str(&format!("@job(name='{}')\n", workflow.id));
            job_code.push_str("def workflow_job():\n");

            // Build dependency graph
            for task in &workflow.tasks {
                if let Some(deps) = workflow.dependencies.get(&task.id) {
                    let deps_str = deps.join(", ");
                    job_code.push_str(&format!("    {}({}())\n", task.id, deps_str));
                } else {
                    job_code.push_str(&format!("    {}()\n", task.id));
                }
            }

            Ok(job_code)
        }

        fn importworkflow(&self, definition: &str) -> Result<Workflow> {
            Err(IoError::UnsupportedFormat(
                "Dagster import not yet implemented".to_string(),
            ))
        }

        fn submit(&self, workflow: &Workflow) -> Result<String> {
            Ok(uuid::Uuid::new_v4().to_string())
        }

        fn get_status(&self, _executionid: &str) -> Result<WorkflowStatus> {
            Ok(WorkflowStatus::Running)
        }

        fn cancel(&self, _executionid: &str) -> Result<()> {
            Ok(())
        }
    }
}

/// Dynamic workflow generation
pub mod dynamic {
    use super::*;

    /// Dynamic workflow generator
    pub struct DynamicWorkflowGenerator {
        templates: HashMap<String, WorkflowTemplate>,
    }

    #[derive(Debug, Clone)]
    pub struct WorkflowTemplate {
        pub baseworkflow: Workflow,
        pub parameters: Vec<ParameterDef>,
        pub generators: Vec<TaskGenerator>,
    }

    #[derive(Debug, Clone)]
    pub struct ParameterDef {
        pub name: String,
        pub param_type: ParameterType,
        pub required: bool,
        pub default: Option<serde_json::Value>,
    }

    #[derive(Debug, Clone)]
    pub enum ParameterType {
        String,
        Integer,
        Float,
        Boolean,
        List(Box<ParameterType>),
        Object,
    }

    #[derive(Debug, Clone)]
    pub enum TaskGenerator {
        ForEach {
            parameter: String,
            task_template: Task,
        },
        Conditional {
            condition: String,
            true_tasks: Vec<Task>,
            false_tasks: Vec<Task>,
        },
        Repeat {
            count_param: String,
            task_template: Task,
        },
    }

    impl Default for DynamicWorkflowGenerator {
        fn default() -> Self {
            Self::new()
        }
    }

    impl DynamicWorkflowGenerator {
        pub fn new() -> Self {
            Self {
                templates: HashMap::new(),
            }
        }

        /// Register a workflow template
        pub fn register_template(&mut self, name: impl Into<String>, template: WorkflowTemplate) {
            self.templates.insert(name.into(), template);
        }

        /// Generate workflow from template
        pub fn generate(
            &self,
            template_name: &str,
            params: HashMap<String, serde_json::Value>,
        ) -> Result<Workflow> {
            let template = self.templates.get(template_name).ok_or_else(|| {
                IoError::NotFound(format!("Template '{template_name}' not found"))
            })?;

            // Validate parameters
            for param_def in &template.parameters {
                if param_def.required && !params.contains_key(&param_def.name) {
                    return Err(IoError::ValidationError(format!(
                        "Required parameter '{}' not provided",
                        param_def.name
                    )));
                }
            }

            let mut workflow = template.baseworkflow.clone();
            workflow.id = format!("{}_{}", workflow.id, Utc::now().timestamp());

            // Apply generators
            for generator in &template.generators {
                self.apply_generator(&mut workflow, generator, &params)?;
            }

            Ok(workflow)
        }

        fn apply_generator(
            &self,
            workflow: &mut Workflow,
            generator: &TaskGenerator,
            params: &HashMap<String, serde_json::Value>,
        ) -> Result<()> {
            match generator {
                TaskGenerator::ForEach {
                    parameter,
                    task_template,
                } => {
                    if let Some(serde_json::Value::Array(items)) = params.get(parameter) {
                        for (i, item) in items.iter().enumerate() {
                            let mut task = task_template.clone();
                            task.id = format!("{}_{}", task.id, i);
                            task.name = format!("{} [{}]", task.name, i);

                            // Inject item into task config
                            if let serde_json::Value::Object(mut config) = task.config.clone() {
                                config.insert("item".to_string(), item.clone());
                                task.config = serde_json::Value::Object(config);
                            }

                            workflow.tasks.push(task);
                        }
                    }
                }
                TaskGenerator::Conditional {
                    condition,
                    true_tasks,
                    false_tasks,
                } => {
                    let condition_result = self.evaluate_condition(condition, params)?;

                    if condition_result {
                        workflow.tasks.extend(true_tasks.iter().cloned());
                    } else {
                        workflow.tasks.extend(false_tasks.iter().cloned());
                    }
                }
                TaskGenerator::Repeat {
                    count_param,
                    task_template,
                } => {
                    if let Some(serde_json::Value::Number(n)) = params.get(count_param) {
                        if let Some(count) = n.as_u64() {
                            for i in 0..count {
                                let mut task = task_template.clone();
                                task.id = format!("{}_{}", task.id, i);
                                task.name = format!("{} [{}]", task.name, i);
                                workflow.tasks.push(task);
                            }
                        }
                    }
                }
            }

            Ok(())
        }

        fn evaluate_condition(
            &self,
            condition: &str,
            params: &HashMap<String, serde_json::Value>,
        ) -> Result<bool> {
            // Simple condition evaluation (in real implementation would use expression parser)
            if let Some((param, value)) = condition.split_once("==") {
                let param = param.trim();
                let value = value.trim().trim_matches('"');

                if let Some(serde_json::Value::String(s)) = params.get(param) {
                    return Ok(s == value);
                }
            }

            Ok(false)
        }
    }
}

/// Event-driven workflows
pub mod events {
    use super::*;
    use crossbeam_channel::{Receiver, Sender};

    /// Event types that can trigger workflows
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum WorkflowEvent {
        FileCreated {
            path: String,
        },
        FileModified {
            path: String,
        },
        DataAvailable {
            source: String,
            timestamp: DateTime<Utc>,
        },
        ScheduledTime {
            workflowid: String,
        },
        ExternalTrigger {
            source: String,
            payload: serde_json::Value,
        },
        WorkflowCompleted {
            workflowid: String,
            executionid: String,
        },
        Custom {
            event_type: String,
            data: serde_json::Value,
        },
    }

    /// Event-driven workflow executor
    pub struct EventDrivenExecutor {
        event_rx: Receiver<WorkflowEvent>,
        event_tx: Sender<WorkflowEvent>,
        rules: Vec<EventRule>,
        executor: Arc<WorkflowExecutor>,
    }

    #[derive(Debug, Clone)]
    pub struct EventRule {
        pub id: String,
        pub event_pattern: EventPattern,
        pub workflowid: String,
        pub parameters: HashMap<String, serde_json::Value>,
    }

    #[derive(Debug, Clone)]
    pub enum EventPattern {
        FilePattern {
            path_regex: String,
        },
        SourcePattern {
            source: String,
        },
        EventTypePattern {
            event_type: String,
        },
        CompositePattern {
            patterns: Vec<EventPattern>,
            operator: LogicalOperator,
        },
    }

    #[derive(Debug, Clone)]
    pub enum LogicalOperator {
        And,
        Or,
        Not,
    }

    impl EventDrivenExecutor {
        pub fn new(executor: Arc<WorkflowExecutor>) -> Self {
            let (tx, rx) = crossbeam_channel::unbounded();
            Self {
                event_rx: rx,
                event_tx: tx,
                rules: Vec::new(),
                executor,
            }
        }

        /// Register an event rule
        pub fn register_rule(&mut self, rule: EventRule) {
            self.rules.push(rule);
        }

        /// Get event sender for external systems
        pub fn get_event_sender(&self) -> Sender<WorkflowEvent> {
            self.event_tx.clone()
        }

        /// Process events and trigger workflows
        pub fn process_events(&self, workflows: &HashMap<String, Workflow>) -> Result<()> {
            while let Ok(event) = self.event_rx.try_recv() {
                for rule in &self.rules {
                    if self.matches_pattern(&event, &rule.event_pattern) {
                        if let Some(workflow) = workflows.get(&rule.workflowid) {
                            // Inject event data into workflow context
                            let mut workflow = workflow.clone();
                            workflow.metadata.set(
                                "trigger_event",
                                MetadataValue::String(serde_json::to_string(&event).unwrap()),
                            );

                            self.executor.execute(&workflow)?;
                        }
                    }
                }
            }

            Ok(())
        }

        #[allow(clippy::only_used_in_recursion)]
        fn matches_pattern(&self, event: &WorkflowEvent, pattern: &EventPattern) -> bool {
            match pattern {
                EventPattern::FilePattern { path_regex } => {
                    if let WorkflowEvent::FileCreated { path }
                    | WorkflowEvent::FileModified { path } = event
                    {
                        regex::Regex::new(path_regex)
                            .map(|re| re.is_match(path))
                            .unwrap_or(false)
                    } else {
                        false
                    }
                }
                EventPattern::SourcePattern { source } => match event {
                    WorkflowEvent::DataAvailable { source: s, .. } => s == source,
                    WorkflowEvent::ExternalTrigger { source: s, .. } => s == source,
                    WorkflowEvent::FileCreated { .. } => false,
                    WorkflowEvent::FileModified { .. } => false,
                    WorkflowEvent::ScheduledTime { .. } => false,
                    WorkflowEvent::WorkflowCompleted { .. } => false,
                    WorkflowEvent::Custom { .. } => false,
                },
                EventPattern::EventTypePattern { event_type } => {
                    if let WorkflowEvent::Custom { event_type: t, .. } = event {
                        t == event_type
                    } else {
                        false
                    }
                }
                EventPattern::CompositePattern { patterns, operator } => match operator {
                    LogicalOperator::And => patterns.iter().all(|p| self.matches_pattern(event, p)),
                    LogicalOperator::Or => patterns.iter().any(|p| self.matches_pattern(event, p)),
                    LogicalOperator::Not => {
                        !patterns.iter().any(|p| self.matches_pattern(event, p))
                    }
                },
            }
        }
    }
}

/// Workflow versioning and history
pub mod versioning {
    use super::*;

    /// Workflow version control
    pub struct WorkflowVersionControl {
        versions: HashMap<String, Vec<WorkflowVersion>>,
    }

    #[derive(Debug, Clone)]
    pub struct WorkflowVersion {
        pub version: String,
        pub workflow: Workflow,
        pub created_at: DateTime<Utc>,
        pub created_by: String,
        pub change_description: String,
        pub parent_version: Option<String>,
    }

    impl Default for WorkflowVersionControl {
        fn default() -> Self {
            Self::new()
        }
    }

    impl WorkflowVersionControl {
        pub fn new() -> Self {
            Self {
                versions: HashMap::new(),
            }
        }

        /// Create a new version
        pub fn create_version(
            &mut self,
            workflow: Workflow,
            created_by: impl Into<String>,
            description: impl Into<String>,
        ) -> String {
            let workflowid = workflow.id.clone();
            let versions = self.versions.entry(workflowid.clone()).or_default();

            let version_number = versions.len() + 1;
            let version = format!("v{version_number}.0.0");

            let parent_version = versions.last().map(|v| v.version.clone());

            versions.push(WorkflowVersion {
                version: version.clone(),
                workflow,
                created_at: Utc::now(),
                created_by: created_by.into(),
                change_description: description.into(),
                parent_version,
            });

            version
        }

        /// Get a specific version
        pub fn get_version(&self, workflowid: &str, version: &str) -> Option<&WorkflowVersion> {
            self.versions
                .get(workflowid)?
                .iter()
                .find(|v| v.version == version)
        }

        /// Get latest version
        pub fn get_latest(&self, workflowid: &str) -> Option<&WorkflowVersion> {
            self.versions.get(workflowid)?.last()
        }

        /// Get version history
        pub fn get_history(&self, workflowid: &str) -> Vec<&WorkflowVersion> {
            self.versions
                .get(workflowid)
                .map(|v| v.iter().collect())
                .unwrap_or_default()
        }

        /// Diff two versions
        pub fn diff(
            &self,
            workflowid: &str,
            version1: &str,
            version2: &str,
        ) -> Option<WorkflowDiff> {
            let v1 = self.get_version(workflowid, version1)?;
            let v2 = self.get_version(workflowid, version2)?;

            Some(WorkflowDiff {
                version1: version1.to_string(),
                version2: version2.to_string(),
                added_tasks: self.diff_tasks(&v1.workflow.tasks, &v2.workflow.tasks, true),
                removed_tasks: self.diff_tasks(&v1.workflow.tasks, &v2.workflow.tasks, false),
                modified_tasks: self.find_modified_tasks(&v1.workflow.tasks, &v2.workflow.tasks),
                dependency_changes: self
                    .diff_dependencies(&v1.workflow.dependencies, &v2.workflow.dependencies),
            })
        }

        fn diff_tasks(&self, tasks1: &[Task], tasks2: &[Task], added: bool) -> Vec<String> {
            let set1: HashSet<_> = tasks1.iter().map(|t| &t.id).collect();
            let set2: HashSet<_> = tasks2.iter().map(|t| &t.id).collect();

            if added {
                set2.difference(&set1).map(|id| (*id).clone()).collect()
            } else {
                set1.difference(&set2).map(|id| (*id).clone()).collect()
            }
        }

        fn find_modified_tasks(&self, tasks1: &[Task], tasks2: &[Task]) -> Vec<String> {
            let map1: HashMap<&String, &Task> = tasks1.iter().map(|t| (&t.id, t)).collect();
            let map2: HashMap<&String, &Task> = tasks2.iter().map(|t| (&t.id, t)).collect();

            let mut modified = Vec::new();
            for (id, task1) in map1 {
                if let Some(task2) = map2.get(id) {
                    // Simple comparison - in real implementation would be more sophisticated
                    if task1.name != task2.name || task1.config != task2.config {
                        modified.push(id.clone());
                    }
                }
            }

            modified
        }

        fn diff_dependencies(
            &self,
            deps1: &HashMap<String, Vec<String>>,
            deps2: &HashMap<String, Vec<String>>,
        ) -> Vec<DependencyChange> {
            let mut changes = Vec::new();

            // Check for added/removed dependencies
            let all_tasks: HashSet<_> = deps1.keys().chain(deps2.keys()).collect();

            for task in all_tasks {
                let deps1_set: HashSet<_> = deps1
                    .get(task)
                    .map(|d| d.iter().collect())
                    .unwrap_or_default();
                let deps2_set: HashSet<_> = deps2
                    .get(task)
                    .map(|d| d.iter().collect())
                    .unwrap_or_default();

                for added in deps2_set.difference(&deps1_set) {
                    changes.push(DependencyChange::Added {
                        task: (*task).clone(),
                        dependency: (*added).clone(),
                    });
                }

                for removed in deps1_set.difference(&deps2_set) {
                    changes.push(DependencyChange::Removed {
                        task: (*task).clone(),
                        dependency: (*removed).clone(),
                    });
                }
            }

            changes
        }
    }

    #[derive(Debug)]
    pub struct WorkflowDiff {
        pub version1: String,
        pub version2: String,
        pub added_tasks: Vec<String>,
        pub removed_tasks: Vec<String>,
        pub modified_tasks: Vec<String>,
        pub dependency_changes: Vec<DependencyChange>,
    }

    #[derive(Debug)]
    pub enum DependencyChange {
        Added { task: String, dependency: String },
        Removed { task: String, dependency: String },
    }
}

/// Distributed execution support
pub mod distributed {
    use super::*;

    /// Distributed workflow executor
    pub struct DistributedExecutor {
        coordinator_url: String,
        worker_pool: WorkerPool,
        task_queue: Arc<Mutex<Vec<DistributedTask>>>,
    }

    #[derive(Debug, Clone)]
    pub struct DistributedTask {
        pub task: Task,
        pub workflowid: String,
        pub executionid: String,
        pub assigned_worker: Option<String>,
        pub status: TaskStatus,
    }

    pub struct WorkerPool {
        workers: Vec<WorkerNode>,
    }

    #[derive(Debug, Clone)]
    pub struct WorkerNode {
        pub id: String,
        pub url: String,
        pub capabilities: WorkerCapabilities,
        pub current_load: f64,
        pub status: WorkerStatus,
    }

    #[derive(Debug, Clone)]
    pub struct WorkerCapabilities {
        pub cpu_cores: usize,
        pub memorygb: f64,
        pub gpu_available: bool,
        pub supported_task_types: Vec<TaskType>,
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum WorkerStatus {
        Available,
        Busy,
        Offline,
    }

    impl DistributedExecutor {
        pub fn new(coordinator_url: impl Into<String>) -> Self {
            Self {
                coordinator_url: coordinator_url.into(),
                worker_pool: WorkerPool {
                    workers: Vec::new(),
                },
                task_queue: Arc::new(Mutex::new(Vec::new())),
            }
        }

        /// Register a worker node
        pub fn register_worker(&mut self, worker: WorkerNode) {
            self.worker_pool.workers.push(worker);
        }

        /// Schedule task to appropriate worker
        pub fn schedule_task(&self, task: DistributedTask) -> Result<String> {
            // Find suitable worker based on requirements and load
            let worker = self.find_suitable_worker(&task)?;

            // Add to queue with assigned worker
            let mut queue = self.task_queue.lock().unwrap();
            let mut scheduled_task = task;
            scheduled_task.assigned_worker = Some(worker.id.clone());
            queue.push(scheduled_task);

            Ok(worker.id.clone())
        }

        fn find_suitable_worker(&self, task: &DistributedTask) -> Result<&WorkerNode> {
            let suitable_workers: Vec<_> = self
                .worker_pool
                .workers
                .iter()
                .filter(|w| {
                    w.status == WorkerStatus::Available
                        && w.capabilities
                            .supported_task_types
                            .contains(&task.task.task_type)
                        && self.meets_resource_requirements(w, &task.task.resources)
                })
                .collect();

            // Select worker with lowest load
            suitable_workers
                .into_iter()
                .min_by(|a, b| a.current_load.partial_cmp(&b.current_load).unwrap())
                .ok_or_else(|| IoError::Other("No suitable worker available".to_string()))
        }

        fn meets_resource_requirements(
            &self,
            worker: &WorkerNode,
            requirements: &ResourceRequirements,
        ) -> bool {
            if let Some(cpu) = requirements.cpu_cores {
                if worker.capabilities.cpu_cores < cpu {
                    return false;
                }
            }

            if let Some(memory) = requirements.memorygb {
                if worker.capabilities.memorygb < memory {
                    return false;
                }
            }

            if requirements.gpu.is_some() && !worker.capabilities.gpu_available {
                return false;
            }

            true
        }
    }
}

/// Workflow visualization
pub mod visualization {
    use super::*;

    /// Workflow visualizer
    pub struct WorkflowVisualizer;

    impl WorkflowVisualizer {
        /// Generate DOT graph representation
        pub fn to_dot(workflow: &Workflow) -> String {
            let mut dot = String::new();
            dot.push_str("digraph workflow {\n");
            dot.push_str("  rankdir=TB;\n");
            dot.push_str("  node [shape=box, style=rounded];\n\n");

            // Add nodes
            for task in &workflow.tasks {
                let color = match task.task_type {
                    TaskType::DataIngestion => "lightblue",
                    TaskType::Transform => "lightgreen",
                    TaskType::Validation => "yellow",
                    TaskType::MLTraining => "orange",
                    TaskType::MLInference => "pink",
                    TaskType::Export => "lightgray",
                    _ => "white",
                };

                dot.push_str(&format!(
                    "  {} [label=\"{}\", fillcolor={}, style=filled];\n",
                    task.id, task.name, color
                ));
            }

            dot.push('\n');

            // Add edges
            for (task_id, deps) in &workflow.dependencies {
                for dep in deps {
                    dot.push_str(&format!("  {dep} -> {task_id};\n"));
                }
            }

            dot.push_str("}\n");
            dot
        }

        /// Generate Mermaid diagram
        pub fn to_mermaid(workflow: &Workflow) -> String {
            let mut mermaid = String::new();
            mermaid.push_str("graph TD\n");

            // Add nodes
            for task in &workflow.tasks {
                let shape = match task.task_type {
                    TaskType::DataIngestion => "[",
                    TaskType::Transform => "(",
                    TaskType::Validation => "{",
                    TaskType::MLTraining => "[[",
                    TaskType::MLInference => "((",
                    TaskType::Export => "[",
                    _ => "[",
                };

                let close = match task.task_type {
                    TaskType::DataIngestion => "]",
                    TaskType::Transform => ")",
                    TaskType::Validation => "}",
                    TaskType::MLTraining => "]]",
                    TaskType::MLInference => "))",
                    TaskType::Export => "]",
                    _ => "]",
                };

                mermaid.push_str(&format!("    {}{}{}{}\n", task.id, shape, task.name, close));
            }

            // Add edges
            for (task_id, deps) in &workflow.dependencies {
                for dep in deps {
                    mermaid.push_str(&format!("    {dep} --> {task_id}\n"));
                }
            }

            mermaid
        }

        /// Generate execution timeline
        pub fn execution_timeline(state: &WorkflowState) -> String {
            let mut timeline = String::new();
            timeline.push_str("gantt\n");
            timeline.push_str("    title Workflow Execution Timeline\n");
            timeline.push_str("    dateFormat YYYY-MM-DD HH:mm:ss\n\n");

            let mut tasks: Vec<_> = state.task_states.iter().collect();
            tasks.sort_by_key(|(_, state)| state.start_time);

            for (task_id, task_state) in tasks {
                if let (Some(start), Some(end)) = (task_state.start_time, task_state.end_time) {
                    let status = match task_state.status {
                        TaskStatus::Success => "done",
                        TaskStatus::Failed => "crit",
                        TaskStatus::Running => "active",
                        _ => "",
                    };

                    timeline.push_str(&format!(
                        "    {} :{}, {}, {}\n",
                        task_id,
                        status,
                        start.format("%Y-%m-%d %H:%M:%S"),
                        end.format("%Y-%m-%d %H:%M:%S")
                    ));
                }
            }

            timeline
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn testworkflow_builder() {
        let workflow = WorkflowBuilder::new("test_wf", "Test Workflow")
            .description("A test workflow")
            .add_task(
                tasks::data_ingestion("task1", "Load Data")
                    .output("data.csv")
                    .build(),
            )
            .add_task(
                tasks::transform("task2", "Process Data")
                    .input("data.csv")
                    .output("processed.csv")
                    .build(),
            )
            .add_dependency("task2", "task1")
            .build()
            .unwrap();

        assert_eq!(workflow.tasks.len(), 2);
        assert_eq!(
            workflow.dependencies.get("task2").unwrap(),
            &vec!["task1".to_string()]
        );
    }

    #[test]
    fn test_cycle_detection() {
        let result = WorkflowBuilder::new("cyclic", "Cyclic Workflow")
            .add_task(tasks::transform("a", "Task A").build())
            .add_task(tasks::transform("b", "Task B").build())
            .add_task(tasks::transform("c", "Task C").build())
            .add_dependency("a", "b")
            .add_dependency("b", "c")
            .add_dependency("c", "a") // Creates cycle
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_etl_template() {
        let workflow = templates::etlworkflow("My ETL Pipeline").build().unwrap();

        assert_eq!(workflow.tasks.len(), 4);
        assert!(workflow.tasks.iter().any(|t| t.id == "extract"));
        assert!(workflow.tasks.iter().any(|t| t.id == "transform"));
        assert!(workflow.tasks.iter().any(|t| t.id == "validate"));
        assert!(workflow.tasks.iter().any(|t| t.id == "load"));
    }
}
