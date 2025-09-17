//! CI/CD Automation for Performance Testing
//!
//! This module provides automated performance testing integration for CI/CD pipelines,
//! including GitHub Actions, GitLab CI, Jenkins, and other systems. It handles
//! automated benchmarking, regression detection, and report generation.

use crate::benchmarking::performance_regression_detector::{
    BaselineMetrics, ConfidenceInterval, EnvironmentInfo, MetricType, MetricValue,
    PerformanceMeasurement, PerformanceRegressionDetector, RegressionConfig, RegressionResult,
    TestConfiguration,
};
use crate::error::{OptimError, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime};

/// Main CI/CD automation engine
#[derive(Debug)]
pub struct CiCdAutomation {
    /// Performance regression detector
    regression_detector: PerformanceRegressionDetector,
    /// CI/CD configuration
    config: CiCdAutomationConfig,
    /// Current environment information
    environment: EnvironmentInfo,
    /// Performance test suite
    test_suite: PerformanceTestSuite,
    /// Report generator
    report_generator: ReportGenerator,
    /// Artifact manager
    artifact_manager: ArtifactManager,
}

/// CI/CD automation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiCdAutomationConfig {
    /// Enable automated performance testing
    pub enable_automation: bool,
    /// CI/CD platform type
    pub platform: CiCdPlatform,
    /// Test execution configuration
    pub test_execution: TestExecutionConfig,
    /// Baseline management settings
    pub baseline_management: BaselineManagementConfig,
    /// Report generation settings
    pub reporting: ReportingConfig,
    /// Artifact storage settings
    pub artifact_storage: ArtifactStorageConfig,
    /// Integration settings
    pub integrations: IntegrationConfig,
    /// Performance gates configuration
    pub performance_gates: PerformanceGatesConfig,
}

/// Supported CI/CD platforms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CiCdPlatform {
    GitHubActions,
    GitLabCI,
    Jenkins,
    AzureDevOps,
    CircleCI,
    TravisCI,
    TeamCity,
    Buildkite,
    Generic,
}

/// Test execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestExecutionConfig {
    /// Run tests on every commit
    pub run_on_commit: bool,
    /// Run tests on pull requests
    pub run_on_pr: bool,
    /// Run tests on releases
    pub run_on_release: bool,
    /// Run tests on schedule
    pub run_on_schedule: Option<CronSchedule>,
    /// Test timeout in seconds
    pub test_timeout: u64,
    /// Number of test iterations
    pub test_iterations: usize,
    /// Warmup iterations before measurement
    pub warmup_iterations: usize,
    /// Parallel test execution
    pub parallel_execution: bool,
    /// Test isolation level
    pub isolation_level: TestIsolationLevel,
}

/// Test isolation levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestIsolationLevel {
    /// Run tests in same process
    Process,
    /// Run each test in separate process
    ProcessPerTest,
    /// Run tests in containers
    Container,
    /// Run tests on separate machines
    Machine,
}

/// Cron schedule configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CronSchedule {
    /// Cron expression
    pub expression: String,
    /// Timezone
    pub timezone: String,
}

/// Baseline management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineManagementConfig {
    /// Automatically update baseline on main branch
    pub auto_update_main: bool,
    /// Update baseline on release tags
    pub update_on_release: bool,
    /// Minimum improvement required for baseline update
    pub min_improvement_threshold: f64,
    /// Require manual approval for baseline updates
    pub require_manual_approval: bool,
    /// Baseline retention policy
    pub retention_policy: BaselineRetentionPolicy,
}

/// Baseline retention policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineRetentionPolicy {
    /// Keep baselines for N days
    pub retention_days: u32,
    /// Maximum number of baselines to keep
    pub max_baselines: usize,
    /// Keep baselines for major releases
    pub keep_major_releases: bool,
}

/// Report generation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportingConfig {
    /// Generate HTML reports
    pub generate_html: bool,
    /// Generate JSON reports
    pub generate_json: bool,
    /// Generate JUnit XML reports
    pub generate_junit: bool,
    /// Include detailed performance data
    pub include_detailed_data: bool,
    /// Include performance trends
    pub include_trends: bool,
    /// Include comparison with baseline
    pub include_baseline_comparison: bool,
    /// Report template customization
    pub custom_templates: HashMap<String, PathBuf>,
}

/// Artifact storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactStorageConfig {
    /// Enable artifact storage
    pub enabled: bool,
    /// Storage provider
    pub provider: ArtifactStorageProvider,
    /// Storage configuration
    pub storage_config: HashMap<String, String>,
    /// Artifact retention policy
    pub retention_policy: ArtifactRetentionPolicy,
}

/// Artifact storage providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArtifactStorageProvider {
    Local(PathBuf),
    S3 { bucket: String, region: String },
    GCS { bucket: String },
    Azure { container: String },
    GitHub,
    GitLab,
}

/// Artifact retention policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactRetentionPolicy {
    /// Retention period in days
    pub retention_days: u32,
    /// Maximum artifacts per branch
    pub max_artifacts_per_branch: usize,
    /// Keep artifacts for releases
    pub keep_releases: bool,
}

/// Integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// GitHub integration settings
    pub github: Option<GitHubIntegration>,
    /// Slack integration settings
    pub slack: Option<SlackIntegration>,
    /// Email integration settings
    pub email: Option<EmailIntegration>,
    /// Custom webhook URLs
    pub webhooks: Vec<WebhookIntegration>,
}

/// GitHub integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitHubIntegration {
    /// GitHub API token
    pub token: String,
    /// Repository owner/organization
    pub owner: String,
    /// Repository name
    pub repo: String,
    /// Post results as PR comments
    pub post_pr_comments: bool,
    /// Create issues for regressions
    pub create_regression_issues: bool,
    /// Update commit status
    pub update_commit_status: bool,
}

/// Slack integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlackIntegration {
    /// Slack webhook URL
    pub webhook_url: String,
    /// Channel for notifications
    pub channel: String,
    /// Mention users on regressions
    pub mention_on_regression: Vec<String>,
}

/// Email integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailIntegration {
    /// SMTP server configuration
    pub smtp_server: String,
    /// SMTP port
    pub smtp_port: u16,
    /// Username
    pub username: String,
    /// Password
    pub password: String,
    /// From address
    pub from_address: String,
    /// Recipient addresses
    pub to_addresses: Vec<String>,
}

/// Webhook integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookIntegration {
    /// Webhook URL
    pub url: String,
    /// HTTP method
    pub method: HttpMethod,
    /// Headers
    pub headers: HashMap<String, String>,
    /// Authentication
    pub auth: Option<WebhookAuth>,
}

/// HTTP methods for webhooks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HttpMethod {
    GET,
    POST,
    PUT,
    PATCH,
}

/// Webhook authentication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WebhookAuth {
    Bearer(String),
    Basic { username: String, password: String },
    ApiKey { header: String, key: String },
}

/// Performance gates configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceGatesConfig {
    /// Enable performance gates
    pub enabled: bool,
    /// Fail build on performance regression
    pub fail_on_regression: bool,
    /// Maximum allowed regression percentage
    pub max_regression_percentage: f64,
    /// Minimum confidence threshold for failures
    pub min_confidence_threshold: f64,
    /// Gates by metric type
    pub metric_gates: HashMap<MetricType, MetricGate>,
}

/// Performance gate for specific metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricGate {
    /// Maximum allowed regression for this metric
    pub max_regression: f64,
    /// Enable gate for this metric
    pub enabled: bool,
    /// Gate severity (blocking or warning)
    pub severity: GateSeverity,
}

/// Gate severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GateSeverity {
    Blocking,
    Warning,
    Info,
}

/// Performance test suite
#[derive(Debug)]
#[allow(dead_code)]
pub struct PerformanceTestSuite {
    /// Available test cases
    test_cases: Vec<PerformanceTestCase>,
    /// Test configuration
    config: TestSuiteConfig,
}

/// Individual performance test case
#[derive(Debug, Clone)]
pub struct PerformanceTestCase {
    /// Test name
    pub name: String,
    /// Test description
    pub description: String,
    /// Test category
    pub category: TestCategory,
    /// Test executor
    pub executor: TestExecutor,
    /// Expected metrics
    pub expected_metrics: Vec<MetricType>,
    /// Test parameters
    pub parameters: HashMap<String, String>,
    /// Test timeout
    pub timeout: Duration,
}

/// Test categories
#[derive(Debug, Clone)]
pub enum TestCategory {
    UnitBenchmark,
    IntegrationBenchmark,
    StressBenchmark,
    MemoryBenchmark,
    ThroughputBenchmark,
    LatencyBenchmark,
    Custom(String),
}

/// Test execution methods
#[derive(Debug, Clone)]
pub enum TestExecutor {
    RustBench,
    Criterion,
    CustomCommand(String),
    InlineFunction(String),
}

/// Test suite configuration
#[derive(Debug, Clone)]
pub struct TestSuiteConfig {
    /// Include unit benchmarks
    pub include_unit: bool,
    /// Include integration benchmarks
    pub include_integration: bool,
    /// Include stress tests
    pub include_stress: bool,
    /// Custom test filters
    pub test_filters: Vec<String>,
}

/// Report generator
#[derive(Debug)]
#[allow(dead_code)]
pub struct ReportGenerator {
    /// Template engine
    template_engine: TemplateEngine,
    /// Report configuration
    config: ReportingConfig,
}

/// Template engine for report generation
#[derive(Debug)]
#[allow(dead_code)]
pub struct TemplateEngine {
    /// Available templates
    templates: HashMap<String, String>,
}

/// Artifact manager for storing test results
#[derive(Debug)]
#[allow(dead_code)]
pub struct ArtifactManager {
    /// Storage provider
    storage_provider: Box<dyn ArtifactStorage>,
    /// Configuration
    config: ArtifactStorageConfig,
}

/// Trait for artifact storage implementations
pub trait ArtifactStorage: std::fmt::Debug + Send + Sync {
    /// Upload an artifact
    fn upload(&self, path: &Path, key: &str) -> Result<String>;

    /// Download an artifact
    fn download(&self, key: &str, path: &Path) -> Result<()>;

    /// List artifacts
    fn list(&self, prefix: &str) -> Result<Vec<String>>;

    /// Delete an artifact
    fn delete(&self, key: &str) -> Result<()>;
}

/// CI/CD test execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiCdTestResult {
    /// Test execution status
    pub status: TestExecutionStatus,
    /// Performance measurements
    pub measurements: Vec<PerformanceMeasurement>,
    /// Regression analysis results
    pub regression_results: Vec<RegressionResult>,
    /// Generated reports
    pub reports: Vec<GeneratedReport>,
    /// Execution metadata
    pub metadata: TestExecutionMetadata,
}

/// Test execution status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestExecutionStatus {
    Success,
    Warning,
    Failed,
    Timeout,
    Error(String),
}

/// Generated report information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedReport {
    /// Report type
    pub report_type: ReportType,
    /// File path
    pub file_path: PathBuf,
    /// Artifact URL (if uploaded)
    pub artifact_url: Option<String>,
    /// Report metadata
    pub metadata: HashMap<String, String>,
}

/// Types of generated reports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportType {
    Html,
    Json,
    JUnit,
    CSV,
    Custom(String),
}

/// Test execution metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestExecutionMetadata {
    /// Execution start time
    pub start_time: SystemTime,
    /// Execution end time
    pub end_time: SystemTime,
    /// Total execution duration
    pub duration: Duration,
    /// Environment information
    pub environment: EnvironmentInfo,
    /// Git information
    pub gitinfo: GitInfo,
    /// CI/CD context
    pub ci_context: CiCdContext,
}

/// Git repository information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitInfo {
    /// Current commit hash
    pub commit_hash: String,
    /// Branch name
    pub branch: String,
    /// Commit message
    pub commit_message: String,
    /// Commit author
    pub author: String,
    /// Commit timestamp
    pub commit_timestamp: SystemTime,
    /// Remote URL
    pub remote_url: Option<String>,
}

/// CI/CD context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiCdContext {
    /// CI/CD platform
    pub platform: CiCdPlatform,
    /// Build number
    pub build_number: Option<String>,
    /// Job ID
    pub job_id: Option<String>,
    /// Pull request number
    pub pr_number: Option<u32>,
    /// Trigger event
    pub trigger_event: TriggerEvent,
    /// Environment variables
    pub env_vars: HashMap<String, String>,
}

/// CI/CD trigger events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TriggerEvent {
    Push,
    PullRequest,
    Release,
    Schedule,
    Manual,
    Other(String),
}

impl CiCdAutomation {
    /// Create a new CI/CD automation engine
    pub fn new(config: CiCdAutomationConfig) -> Result<Self> {
        let regression_detector = PerformanceRegressionDetector::new(RegressionConfig::default())?;

        let environment = Self::detect_environment()?;
        let test_suite = PerformanceTestSuite::new(TestSuiteConfig::default())?;
        let report_generator = ReportGenerator::new(config.reporting.clone())?;
        let artifact_manager = ArtifactManager::new(config.artifact_storage.clone())?;

        Ok(Self {
            regression_detector,
            config,
            environment,
            test_suite,
            report_generator,
            artifact_manager,
        })
    }

    /// Execute automated performance testing
    pub async fn execute_automated_testing(&mut self) -> Result<CiCdTestResult> {
        let start_time = SystemTime::now();

        // Detect CI/CD context
        let ci_context = self.detect_ci_context()?;
        let gitinfo = self.detect_git_info()?;

        // Determine if tests should run
        if !self.should_run_tests(&ci_context, &gitinfo)? {
            return Ok(CiCdTestResult {
                status: TestExecutionStatus::Success,
                measurements: vec![],
                regression_results: vec![],
                reports: vec![],
                metadata: TestExecutionMetadata {
                    start_time,
                    end_time: SystemTime::now(),
                    duration: SystemTime::now().duration_since(start_time)?,
                    environment: self.environment.clone(),
                    gitinfo,
                    ci_context,
                },
            });
        }

        // Execute performance tests
        let measurements = self.execute_performance_tests().await?;

        // Load historical baseline if available
        self.load_baseline_for_branch(&gitinfo.branch).await?;

        // Add measurements to regression detector
        for measurement in &measurements {
            self.regression_detector
                .add_measurement(measurement.clone())?;
        }

        // Detect regressions
        let regression_results = self.regression_detector.detect_regressions()?;

        // Apply performance gates
        let gate_results = self.apply_performance_gates(&regression_results)?;

        // Generate reports
        let reports = self
            .generate_reports(&measurements, &regression_results)
            .await?;

        // Upload artifacts
        self.upload_artifacts(&reports).await?;

        // Send notifications
        self.send_notifications(&regression_results, &gate_results)
            .await?;

        // Update baseline if appropriate
        if self.should_update_baseline(&gitinfo, &regression_results)? {
            self.update_baseline(&gitinfo).await?;
        }

        let end_time = SystemTime::now();
        let status = Self::determine_execution_status(&regression_results, &gate_results);

        Ok(CiCdTestResult {
            status,
            measurements,
            regression_results,
            reports,
            metadata: TestExecutionMetadata {
                start_time,
                end_time,
                duration: end_time.duration_since(start_time)?,
                environment: self.environment.clone(),
                gitinfo,
                ci_context,
            },
        })
    }

    /// Detect current environment information
    fn detect_environment() -> Result<EnvironmentInfo> {
        let os = std::env::consts::OS.to_string();
        let cpu_model = Self::get_cpu_model()?;
        let cpu_cores = num_cpus::get();
        let total_memory_mb = Self::get_total_memory_mb()?;
        let gpu_info = Self::get_gpu_info().ok();
        let compiler_version = Self::get_compiler_version()?;
        let rust_version = Self::get_rust_version()?;
        let env_vars = Self::get_relevant_env_vars();

        Ok(EnvironmentInfo {
            os,
            cpu_model,
            cpu_cores,
            total_memory_mb,
            gpu_info,
            compiler_version,
            rust_version,
            env_vars,
        })
    }

    /// Get CPU model information
    fn get_cpu_model() -> Result<String> {
        #[cfg(target_os = "linux")]
        {
            let output = Command::new("cat").arg("/proc/cpuinfo").output()?;

            let content = String::from_utf8_lossy(&output.stdout);
            for line in content.lines() {
                if line.starts_with("model name") {
                    if let Some(model) = line.split(':').nth(1) {
                        return Ok(model.trim().to_string());
                    }
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            let output = Command::new("sysctl")
                .arg("-n")
                .arg("machdep.cpu.brand_string")
                .output()?;

            if output.status.success() {
                return Ok(String::from_utf8_lossy(&output.stdout).trim().to_string());
            }
        }

        Ok("Unknown".to_string())
    }

    /// Get total memory in MB
    fn get_total_memory_mb() -> Result<usize> {
        #[cfg(target_os = "linux")]
        {
            let output = Command::new("cat").arg("/proc/meminfo").output()?;

            let content = String::from_utf8_lossy(&output.stdout);
            for line in content.lines() {
                if line.starts_with("MemTotal:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<usize>() {
                            return Ok(kb / 1024); // Convert KB to MB
                        }
                    }
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            let output = Command::new("sysctl")
                .arg("-n")
                .arg("hw.memsize")
                .output()?;

            if output.status.success() {
                let bytes_str = String::from_utf8_lossy(&output.stdout);
                let trimmed_str = bytes_str.trim();
                if let Ok(bytes) = trimmed_str.parse::<usize>() {
                    return Ok(bytes / (1024 * 1024)); // Convert bytes to MB
                }
            }
        }

        Ok(8192) // Default fallback
    }

    /// Get GPU information
    fn get_gpu_info() -> Result<String> {
        // Try nvidia-smi first
        if let Ok(output) = Command::new("nvidia-smi")
            .arg("--query-gpu=name")
            .arg("--format=csv,noheader,nounits")
            .output()
        {
            if output.status.success() {
                let gpu_name = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !gpu_name.is_empty() {
                    return Ok(format!("NVIDIA {}", gpu_name));
                }
            }
        }

        // Try lspci for other GPUs
        if let Ok(output) = Command::new("lspci").arg("-nn").output() {
            if output.status.success() {
                let content = String::from_utf8_lossy(&output.stdout);
                for line in content.lines() {
                    if line.contains("VGA compatible controller") || line.contains("3D controller")
                    {
                        return Ok(line.to_string());
                    }
                }
            }
        }

        Err(OptimError::Environment(
            "GPU information not available".to_string(),
        ))
    }

    /// Get compiler version
    fn get_compiler_version() -> Result<String> {
        let output = Command::new("rustc").arg("--version").output()?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
        } else {
            Ok("Unknown rustc version".to_string())
        }
    }

    /// Get Rust version
    fn get_rust_version() -> Result<String> {
        let output = Command::new("rustc").arg("--version").output()?;

        if output.status.success() {
            let version_str = String::from_utf8_lossy(&output.stdout);
            if let Some(version) = version_str.split_whitespace().nth(1) {
                return Ok(version.to_string());
            }
        }

        Ok("Unknown".to_string())
    }

    /// Get relevant environment variables
    fn get_relevant_env_vars() -> HashMap<String, String> {
        let mut env_vars = HashMap::new();

        let relevant_vars = [
            "CARGO_PROFILE",
            "RUSTFLAGS",
            "CARGO_TARGET_DIR",
            "RUST_BACKTRACE",
            "RUST_LOG",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "CI",
            "GITHUB_ACTIONS",
            "GITLAB_CI",
            "JENKINS_URL",
        ];

        for var in &relevant_vars {
            if let Ok(value) = std::env::var(var) {
                env_vars.insert(var.to_string(), value);
            }
        }

        env_vars
    }

    /// Detect CI/CD context
    fn detect_ci_context(&self) -> Result<CiCdContext> {
        let platform = self.detect_ci_platform();
        let build_number = std::env::var("BUILD_NUMBER")
            .ok()
            .or_else(|| std::env::var("GITHUB_RUN_NUMBER").ok())
            .or_else(|| std::env::var("CI_PIPELINE_ID").ok());

        let job_id = std::env::var("JOB_ID")
            .ok()
            .or_else(|| std::env::var("GITHUB_JOB").ok())
            .or_else(|| std::env::var("CI_JOB_ID").ok());

        let pr_number = std::env::var("PULL_REQUEST_NUMBER")
            .ok()
            .or_else(|| std::env::var("GITHUB_PR_NUMBER").ok())
            .or_else(|| std::env::var("CI_MERGE_REQUEST_IID").ok())
            .and_then(|s| s.parse().ok());

        let trigger_event = self.detect_trigger_event();

        let mut env_vars = HashMap::new();
        for (key, value) in std::env::vars() {
            if key.starts_with("CI") || key.starts_with("GITHUB") || key.starts_with("GITLAB") {
                env_vars.insert(key, value);
            }
        }

        Ok(CiCdContext {
            platform,
            build_number,
            job_id,
            pr_number,
            trigger_event,
            env_vars,
        })
    }

    /// Detect CI/CD platform
    fn detect_ci_platform(&self) -> CiCdPlatform {
        if std::env::var("GITHUB_ACTIONS").is_ok() {
            CiCdPlatform::GitHubActions
        } else if std::env::var("GITLAB_CI").is_ok() {
            CiCdPlatform::GitLabCI
        } else if std::env::var("JENKINS_URL").is_ok() {
            CiCdPlatform::Jenkins
        } else if std::env::var("SYSTEM_TEAMFOUNDATIONCOLLECTIONURI").is_ok() {
            CiCdPlatform::AzureDevOps
        } else if std::env::var("CIRCLECI").is_ok() {
            CiCdPlatform::CircleCI
        } else if std::env::var("TRAVIS").is_ok() {
            CiCdPlatform::TravisCI
        } else if std::env::var("TEAMCITY_VERSION").is_ok() {
            CiCdPlatform::TeamCity
        } else if std::env::var("BUILDKITE").is_ok() {
            CiCdPlatform::Buildkite
        } else {
            CiCdPlatform::Generic
        }
    }

    /// Detect trigger event
    fn detect_trigger_event(&self) -> TriggerEvent {
        if let Ok(event) = std::env::var("GITHUB_EVENT_NAME") {
            match event.as_str() {
                "push" => TriggerEvent::Push,
                "pull_request" => TriggerEvent::PullRequest,
                "release" => TriggerEvent::Release,
                "schedule" => TriggerEvent::Schedule,
                "workflow_dispatch" => TriggerEvent::Manual,
                other => TriggerEvent::Other(other.to_string()),
            }
        } else if std::env::var("CI_PIPELINE_SOURCE").is_ok() {
            // GitLab CI detection logic
            TriggerEvent::Push
        } else {
            TriggerEvent::Other("unknown".to_string())
        }
    }

    /// Detect Git information
    fn detect_git_info(&self) -> Result<GitInfo> {
        let commit_hash = self.get_git_output(&["rev-parse", "HEAD"])?;
        let branch = self.get_git_output(&["rev-parse", "--abbrev-ref", "HEAD"])?;
        let commit_message = self.get_git_output(&["log", "-1", "--pretty=%B"])?;
        let author = self.get_git_output(&["log", "-1", "--pretty=%an <%ae>"])?;

        let commit_timestamp_str = self.get_git_output(&["log", "-1", "--pretty=%ct"])?;
        let commit_timestamp = SystemTime::UNIX_EPOCH
            + Duration::from_secs(commit_timestamp_str.parse::<u64>().unwrap_or(0));

        let remote_url = self
            .get_git_output(&["config", "--get", "remote.origin.url"])
            .ok();

        Ok(GitInfo {
            commit_hash,
            branch,
            commit_message,
            author,
            commit_timestamp,
            remote_url,
        })
    }

    /// Execute git command and get output
    fn get_git_output(&self, args: &[&str]) -> Result<String> {
        let output = Command::new("git").args(args).output()?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
        } else {
            Err(OptimError::Environment(format!(
                "Git command failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )))
        }
    }

    /// Detect the build configuration (debug/release/custom)
    fn detect_build_config(&self) -> Result<String> {
        // Check environment variables first
        if let Ok(profile) = std::env::var("CARGO_CFG_DEBUG_ASSERTIONS") {
            return Ok(if profile == "true" {
                "debug".to_string()
            } else {
                "release".to_string()
            });
        }

        if let Ok(profile) = std::env::var("CARGO_PROFILE") {
            return Ok(profile);
        }

        if let Ok(profile) = std::env::var("PROFILE") {
            return Ok(profile);
        }

        // Check for common CI environment variables that indicate build type
        if let Ok(build_type) = std::env::var("BUILD_TYPE") {
            return Ok(build_type.to_lowercase());
        }

        if let Ok(config) = std::env::var("CONFIGURATION") {
            return Ok(config.to_lowercase());
        }

        // Check if running in debug mode based on optimization level
        if std::env::var("CARGO_CFG_OPT_LEVEL").unwrap_or_default() == "0" {
            return Ok("debug".to_string());
        }

        // Try to detect from Cargo.toml if available
        if let Ok(cargo_content) = std::fs::read_to_string("Cargo.toml") {
            if cargo_content.contains("[profile.release]")
                && !cargo_content.contains("debug = true")
            {
                return Ok("release".to_string());
            }
        }

        // Check common CI patterns
        match std::env::var("GITHUB_WORKFLOW").as_deref() {
            Ok(workflow) if workflow.to_lowercase().contains("release") => {
                return Ok("release".to_string());
            }
            Ok(workflow) if workflow.to_lowercase().contains("debug") => {
                return Ok("debug".to_string());
            }
            _ => {}
        }

        // Default to release for production CI environments, debug otherwise
        if std::env::var("CI").is_ok() {
            Ok("release".to_string())
        } else {
            Ok("debug".to_string())
        }
    }

    /// Determine if tests should run based on configuration and context
    fn should_run_tests(&self, ci_context: &CiCdContext, info: &GitInfo) -> Result<bool> {
        if !self.config.enable_automation {
            return Ok(false);
        }

        match &ci_context.trigger_event {
            TriggerEvent::Push => Ok(self.config.test_execution.run_on_commit),
            TriggerEvent::PullRequest => Ok(self.config.test_execution.run_on_pr),
            TriggerEvent::Release => Ok(self.config.test_execution.run_on_release),
            TriggerEvent::Schedule => Ok(self.config.test_execution.run_on_schedule.is_some()),
            TriggerEvent::Manual => Ok(true),
            _ => Ok(false),
        }
    }

    /// Execute performance tests
    async fn execute_performance_tests(&mut self) -> Result<Vec<PerformanceMeasurement>> {
        let mut measurements = Vec::new();

        for test_case in &self.test_suite.test_cases.clone() {
            if self.should_run_test_case(test_case) {
                let test_measurements = self.execute_test_case(test_case).await?;
                measurements.extend(test_measurements);
            }
        }

        Ok(measurements)
    }

    /// Check if a test case should be executed
    fn should_run_test_case(&self, _testcase: &PerformanceTestCase) -> bool {
        // Apply test filters here
        true
    }

    /// Execute a single test case
    async fn execute_test_case(
        &self,
        test_case: &PerformanceTestCase,
    ) -> Result<Vec<PerformanceMeasurement>> {
        let mut measurements = Vec::new();

        for iteration in 0..self.config.test_execution.test_iterations {
            let measurement = self
                .execute_single_test_iteration(test_case, iteration)
                .await?;
            measurements.push(measurement);
        }

        Ok(measurements)
    }

    /// Execute a single test iteration
    async fn execute_single_test_iteration(
        &self,
        test_case: &PerformanceTestCase,
        iteration: usize,
    ) -> Result<PerformanceMeasurement> {
        let start_time = SystemTime::now();

        // Execute the test based on executor type
        let metrics = match &test_case.executor {
            TestExecutor::RustBench => self.execute_rust_bench(test_case).await?,
            TestExecutor::Criterion => self.execute_criterion_bench(test_case).await?,
            TestExecutor::CustomCommand(cmd) => self.execute_custom_command(cmd, test_case).await?,
            TestExecutor::InlineFunction(func) => {
                self.execute_inline_function(func, test_case).await?
            }
        };

        let gitinfo = self.detect_git_info()?;

        Ok(PerformanceMeasurement {
            timestamp: start_time,
            commithash: gitinfo.commit_hash,
            branch: gitinfo.branch,
            build_config: self.detect_build_config()?,
            environment: self.environment.clone(),
            metrics,
            test_config: TestConfiguration {
                test_name: test_case.name.clone(),
                parameters: test_case.parameters.clone(),
                dataset_size: None,
                iterations: Some(iteration),
                batch_size: None,
                precision: "f64".to_string(),
            },
            metadata: HashMap::new(),
        })
    }

    /// Execute Rust benchmark
    async fn execute_rust_bench(
        &self,
        test_case: &PerformanceTestCase,
    ) -> Result<HashMap<MetricType, MetricValue>> {
        let mut metrics = HashMap::new();

        // Execute cargo bench for the specific test
        let mut cmd = Command::new("cargo");
        cmd.args(&["bench", "--bench", &test_case.name]);

        // Add any additional arguments from test case parameters
        if let Some(args) = test_case.parameters.get("args") {
            if let Ok(arg_list) = serde_json::from_str::<Vec<String>>(args) {
                cmd.args(&arg_list);
            }
        }

        let start_time = std::time::Instant::now();
        let output = cmd.output().map_err(|e| {
            crate::error::OptimError::ExecutionError(format!(
                "Failed to execute cargo bench: {}",
                e
            ))
        })?;
        let execution_time = start_time.elapsed().as_secs_f64();

        // Parse benchmark output for metrics
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        if !output.status.success() {
            return Err(crate::error::OptimError::ExecutionError(format!(
                "Benchmark execution failed: {}",
                stderr
            )));
        }

        // Parse execution time from benchmark output
        let parsed_time = self.parse_benchmark_time(&stdout).unwrap_or(execution_time);

        metrics.insert(
            MetricType::ExecutionTime,
            MetricValue {
                value: parsed_time,
                std_dev: Some(parsed_time * 0.05), // Estimated 5% variation
                sample_count: 1,
                min_value: parsed_time,
                max_value: parsed_time,
                percentiles: None,
            },
        );

        // Parse additional metrics if available
        if let Some(memory_usage) = self.parse_memory_usage(&stdout) {
            metrics.insert(
                MetricType::MemoryUsage,
                MetricValue {
                    value: memory_usage,
                    std_dev: Some(memory_usage * 0.1),
                    sample_count: 1,
                    min_value: memory_usage,
                    max_value: memory_usage,
                    percentiles: None,
                },
            );
        }

        Ok(metrics)
    }

    /// Execute Criterion benchmark
    async fn execute_criterion_bench(
        &self,
        test_case: &PerformanceTestCase,
    ) -> Result<HashMap<MetricType, MetricValue>> {
        let mut metrics = HashMap::new();

        // Execute criterion benchmark
        let mut cmd = Command::new("cargo");
        cmd.args(&[
            "bench",
            "--bench",
            &test_case.name,
            "--",
            "--output-format",
            "json",
        ]);

        // Add any additional arguments from test case parameters
        if let Some(args) = test_case.parameters.get("args") {
            if let Ok(arg_list) = serde_json::from_str::<Vec<String>>(args) {
                cmd.args(&arg_list);
            }
        }

        let start_time = std::time::Instant::now();
        let output = cmd.output().map_err(|e| {
            crate::error::OptimError::ExecutionError(format!(
                "Failed to execute criterion bench: {}",
                e
            ))
        })?;
        let execution_time = start_time.elapsed().as_secs_f64();

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        if !output.status.success() {
            return Err(crate::error::OptimError::ExecutionError(format!(
                "Criterion benchmark execution failed: {}",
                stderr
            )));
        }

        // Parse Criterion JSON output for detailed metrics
        if let Some(criterion_metrics) = self.parse_criterion_output(&stdout) {
            metrics.extend(criterion_metrics);
        } else {
            // Fallback to basic timing if JSON parsing fails
            let parsed_time = self.parse_benchmark_time(&stdout).unwrap_or(execution_time);
            metrics.insert(
                MetricType::ExecutionTime,
                MetricValue {
                    value: parsed_time,
                    std_dev: Some(parsed_time * 0.02), // Criterion typically has lower variance
                    sample_count: 1,
                    min_value: parsed_time,
                    max_value: parsed_time,
                    percentiles: None,
                },
            );
        }

        Ok(metrics)
    }

    /// Execute custom command
    async fn execute_custom_command(
        &self,
        command: &str,
        test_case: &PerformanceTestCase,
    ) -> Result<HashMap<MetricType, MetricValue>> {
        let mut metrics = HashMap::new();

        // Parse the command and arguments
        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.is_empty() {
            return Err(crate::error::OptimError::ExecutionError(
                "Empty command".to_string(),
            ));
        }

        let mut cmd = Command::new(parts[0]);
        if parts.len() > 1 {
            cmd.args(&parts[1..]);
        }

        // Add test case parameters as environment variables
        for (key, value) in &test_case.parameters {
            cmd.env(format!("TEST_{}", key.to_uppercase()), value);
        }

        // Set working directory if specified
        if let Some(workdir) = test_case.parameters.get("workdir") {
            cmd.current_dir(workdir);
        }

        let start_time = std::time::Instant::now();
        let output = cmd.output().map_err(|e| {
            crate::error::OptimError::ExecutionError(format!(
                "Failed to execute command '{}': {}",
                command, e
            ))
        })?;
        let execution_time = start_time.elapsed().as_secs_f64();

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        if !output.status.success() {
            return Err(crate::error::OptimError::ExecutionError(format!(
                "Custom command '{}' failed with exit code {}: {}",
                command,
                output.status.code().unwrap_or(-1),
                stderr
            )));
        }

        // Parse output for metrics
        let parsed_time = self.parse_benchmark_time(&stdout).unwrap_or(execution_time);
        metrics.insert(
            MetricType::ExecutionTime,
            MetricValue {
                value: parsed_time,
                std_dev: Some(parsed_time * 0.1),
                sample_count: 1,
                min_value: parsed_time,
                max_value: parsed_time,
                percentiles: None,
            },
        );

        // Parse custom metrics from output if available
        if let Some(custom_metrics) = self.parse_custom_metrics(&stdout) {
            metrics.extend(custom_metrics);
        }

        Ok(metrics)
    }

    /// Execute inline function
    async fn execute_inline_function(
        &self,
        function: &str,
        test_case: &PerformanceTestCase,
    ) -> Result<HashMap<MetricType, MetricValue>> {
        let mut metrics = HashMap::new();

        // For security reasons, we only support a limited set of predefined functions
        // This prevents arbitrary code execution
        let start_time = std::time::Instant::now();

        let execution_time = match function {
            "simple_loop" => {
                let iterations = test_case
                    .parameters
                    .get("iterations")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(1000);
                self.execute_simple_loop_benchmark(iterations)
            }
            "memory_allocation" => {
                let size = test_case
                    .parameters
                    .get("size")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(1024);
                self.execute_memory_allocation_benchmark(size)
            }
            "computation_intensive" => {
                let complexity = test_case
                    .parameters
                    .get("complexity")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(100);
                self.execute_computation_benchmark(complexity)
            }
            "sleep_benchmark" => {
                let durationms = test_case
                    .parameters
                    .get("durationms")
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(100);
                self.execute_sleep_benchmark(durationms).await
            }
            _ => {
                return Err(crate::error::OptimError::ExecutionError(format!(
                    "Unknown inline function: {}. Supported functions: simple_loop, memory_allocation, computation_intensive, sleep_benchmark", 
                    function
                )));
            }
        };

        let total_time = start_time.elapsed().as_secs_f64();

        metrics.insert(
            MetricType::ExecutionTime,
            MetricValue {
                value: execution_time,
                std_dev: Some(execution_time * 0.05),
                sample_count: 1,
                min_value: execution_time,
                max_value: execution_time,
                percentiles: None,
            },
        );

        // Add total overhead time as a separate metric
        if total_time > execution_time {
            metrics.insert(
                MetricType::Custom("overhead_time".to_string()),
                MetricValue {
                    value: total_time - execution_time,
                    std_dev: None,
                    sample_count: 1,
                    min_value: total_time - execution_time,
                    max_value: total_time - execution_time,
                    percentiles: None,
                },
            );
        }

        Ok(metrics)
    }

    /// Load baseline for specific branch
    async fn load_baseline_for_branch(&mut self, branch: &str) -> Result<()> {
        // Try to load baseline from different sources in order of preference

        // 1. Try to load from artifact storage
        if let Ok(baseline_measurements) = self.load_baseline_from_storage(branch).await {
            let baseline =
                self.convert_to_baseline_metrics(baseline_measurements, branch.to_string())?;
            self.regression_detector.set_baseline(baseline)?;
            return Ok(());
        }

        // 2. Try to load from local cache
        if let Ok(baseline_measurements) = self.load_baseline_from_cache(branch).await {
            let baseline =
                self.convert_to_baseline_metrics(baseline_measurements, branch.to_string())?;
            self.regression_detector.set_baseline(baseline)?;
            return Ok(());
        }

        // 3. Try to load from git history (run tests on previous commits)
        if let Ok(baseline_measurements) = self.generate_baseline_from_history(branch).await {
            let baseline = self
                .convert_to_baseline_metrics(baseline_measurements.clone(), branch.to_string())?;
            self.regression_detector.set_baseline(baseline)?;
            // Cache the generated baseline for future use
            let _ = self
                .save_baseline_to_cache(branch, &baseline_measurements)
                .await;
            return Ok(());
        }

        // 4. Create an empty baseline (first run)
        println!(
            "No baseline found for branch '{}', creating empty baseline",
            branch
        );
        Ok(())
    }

    /// Apply performance gates
    fn apply_performance_gates(
        &self,
        regression_results: &[RegressionResult],
    ) -> Result<Vec<GateResult>> {
        let mut gate_results = Vec::new();

        if !self.config.performance_gates.enabled {
            return Ok(gate_results);
        }

        for result in regression_results {
            if let Some(gate) = self
                .config
                .performance_gates
                .metric_gates
                .get(&result.metric)
            {
                if gate.enabled {
                    let gate_result = self.evaluate_gate(result, gate)?;
                    gate_results.push(gate_result);
                }
            }
        }

        Ok(gate_results)
    }

    /// Evaluate a single performance gate
    fn evaluate_gate(&self, result: &RegressionResult, gate: &MetricGate) -> Result<GateResult> {
        let regression_percentage = result.change_percentage.abs();
        let passed = regression_percentage <= gate.max_regression
            && result.confidence >= self.config.performance_gates.min_confidence_threshold;

        Ok(GateResult {
            metric: result.metric.clone(),
            passed,
            regression_percentage,
            threshold: gate.max_regression,
            confidence: result.confidence,
            severity: gate.severity.clone(),
            message: if passed {
                format!("Gate passed for {}", result.metric.to_string())
            } else {
                format!(
                    "Gate failed for {}: {:.2}% regression exceeds {:.2}% threshold",
                    result.metric.to_string(),
                    regression_percentage,
                    gate.max_regression
                )
            },
        })
    }

    /// Generate performance reports
    async fn generate_reports(
        &self,
        measurements: &[PerformanceMeasurement],
        regression_results: &[RegressionResult],
    ) -> Result<Vec<GeneratedReport>> {
        let mut reports = Vec::new();

        if self.config.reporting.generate_html {
            let html_report = self
                .report_generator
                .generate_html_report(measurements, regression_results)
                .await?;
            reports.push(html_report);
        }

        if self.config.reporting.generate_json {
            let json_report = self
                .report_generator
                .generate_json_report(measurements, regression_results)
                .await?;
            reports.push(json_report);
        }

        if self.config.reporting.generate_junit {
            let junit_report = self
                .report_generator
                .generate_junit_report(measurements, regression_results)
                .await?;
            reports.push(junit_report);
        }

        Ok(reports)
    }

    /// Upload artifacts to storage
    async fn upload_artifacts(&self, reports: &[GeneratedReport]) -> Result<()> {
        if !self.config.artifact_storage.enabled {
            return Ok(());
        }

        for report in reports {
            let key = format!(
                "reports/{}",
                report.file_path.file_name().unwrap().to_string_lossy()
            );
            self.artifact_manager
                .storage_provider
                .upload(&report.file_path, &key)?;
        }

        Ok(())
    }

    /// Send notifications about results
    async fn send_notifications(
        &self,
        regression_results: &[RegressionResult],
        gate_results: &[GateResult],
    ) -> Result<()> {
        if regression_results.is_empty() && gate_results.iter().all(|g| g.passed) {
            return Ok(());
        }

        // Send GitHub notifications
        if let Some(github_config) = &self.config.integrations.github {
            self.send_github_notifications(github_config, regression_results, gate_results)
                .await?;
        }

        // Send Slack notifications
        if let Some(slack_config) = &self.config.integrations.slack {
            self.send_slack_notifications(slack_config, regression_results, gate_results)
                .await?;
        }

        // Send webhook notifications
        for webhook in &self.config.integrations.webhooks {
            self.send_webhook_notification(webhook, regression_results, gate_results)
                .await?;
        }

        Ok(())
    }

    /// Send GitHub notifications
    async fn send_github_notifications(
        &self,
        config: &GitHubIntegration,
        _results: &[RegressionResult],
        _gate_results: &[GateResult],
    ) -> Result<()> {
        // TODO: Implement GitHub API integration
        Ok(())
    }

    /// Send Slack notifications
    async fn send_slack_notifications(
        &self,
        config: &SlackIntegration,
        _results: &[RegressionResult],
        _gate_results: &[GateResult],
    ) -> Result<()> {
        // TODO: Implement Slack webhook integration
        Ok(())
    }

    /// Send webhook notification
    async fn send_webhook_notification(
        &self,
        webhook: &WebhookIntegration,
        _results: &[RegressionResult],
        _gate_results: &[GateResult],
    ) -> Result<()> {
        // TODO: Implement webhook integration
        Ok(())
    }

    /// Check if baseline should be updated
    fn should_update_baseline(
        &self,
        gitinfo: &GitInfo,
        regression_results: &[RegressionResult],
    ) -> Result<bool> {
        // Don't update if there are regressions
        if !regression_results.is_empty() {
            return Ok(false);
        }

        // Check if we're on main branch and auto-update is enabled
        if self.config.baseline_management.auto_update_main && gitinfo.branch == "main" {
            return Ok(true);
        }

        // Check if this is a release and update-on-release is enabled
        if self.config.baseline_management.update_on_release {
            // TODO: Detect if this is a release
        }

        Ok(false)
    }

    /// Update baseline
    async fn update_baseline(&mut self, gitinfo: &GitInfo) -> Result<()> {
        self.regression_detector
            .update_baseline_from_recent(gitinfo.commit_hash.clone())?;
        // TODO: Store baseline in artifact storage
        Ok(())
    }

    /// Determine overall execution status
    fn determine_execution_status(
        regression_results: &[RegressionResult],
        gate_results: &[GateResult],
    ) -> TestExecutionStatus {
        let has_critical_regressions = regression_results.iter().any(|r| r.severity >= 0.9);
        let has_failed_gates = gate_results
            .iter()
            .any(|g| !g.passed && matches!(g.severity, GateSeverity::Blocking));

        if has_critical_regressions || has_failed_gates {
            TestExecutionStatus::Failed
        } else if !regression_results.is_empty() || gate_results.iter().any(|g| !g.passed) {
            TestExecutionStatus::Warning
        } else {
            TestExecutionStatus::Success
        }
    }

    // Helper methods for benchmark execution and parsing

    /// Parse benchmark execution time from output
    fn parse_benchmark_time(&self, output: &str) -> Option<f64> {
        // Try different patterns for extracting timing information
        for line in output.lines() {
            // Pattern 1: "time: [123.45 ms 124.56 ms 125.67 ms]"
            if let Some(captures) = Regex::new(r"time:\s*\[\s*([0-9.]+)\s*(ns|µs|ms|s)\s*")
                .ok()?
                .captures(line)
            {
                if let (Ok(time), Some(unit)) = (captures[1].parse::<f64>(), captures.get(2)) {
                    let multiplier = match unit.as_str() {
                        "ns" => 1e-9,
                        "µs" => 1e-6,
                        "ms" => 1e-3,
                        "s" => 1.0,
                        _ => 1.0, // Default to seconds for unknown units
                    };
                    return Some(time * multiplier);
                }
            }

            // Pattern 2: "Elapsed: 123.456 seconds"
            if let Some(captures) = Regex::new(r"(?i)elapsed:\s*([0-9.]+)\s*seconds?")
                .ok()?
                .captures(line)
            {
                if let Ok(time) = captures[1].parse::<f64>() {
                    return Some(time);
                }
            }

            // Pattern 3: "Duration: 123ms"
            if let Some(captures) = Regex::new(r"(?i)duration:\s*([0-9.]+)(ms|s)")
                .ok()?
                .captures(line)
            {
                if let (Ok(time), Some(unit)) = (captures[1].parse::<f64>(), captures.get(2)) {
                    let multiplier = if unit.as_str() == "ms" { 1e-3 } else { 1.0 };
                    return Some(time * multiplier);
                }
            }
        }
        None
    }

    /// Parse memory usage from output
    fn parse_memory_usage(&self, output: &str) -> Option<f64> {
        for line in output.lines() {
            if let Some(captures) = Regex::new(r"(?i)memory:\s*([0-9.]+)\s*(kb|mb|gb|bytes?)")
                .ok()?
                .captures(line)
            {
                if let (Ok(memory), Some(unit)) = (captures[1].parse::<f64>(), captures.get(2)) {
                    let multiplier = match unit.as_str().to_lowercase().as_str() {
                        "bytes" | "byte" => 1.0,
                        "kb" => 1024.0,
                        "mb" => 1024.0 * 1024.0,
                        "gb" => 1024.0 * 1024.0 * 1024.0,
                        _ => 1.0, // Default to bytes for unknown units
                    };
                    return Some(memory * multiplier);
                }
            }
        }
        None
    }

    /// Parse Criterion benchmark output
    fn parse_criterion_output(&self, output: &str) -> Option<HashMap<MetricType, MetricValue>> {
        let mut metrics = HashMap::new();

        for line in output.lines() {
            // Try to parse as JSON first
            if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(line) {
                if let Some(timing) = json_value.get("time") {
                    if let Some(mean) = timing.get("mean").and_then(|v| v.as_f64()) {
                        let std_dev = timing.get("std_dev").and_then(|v| v.as_f64());
                        let min_val = timing.get("min").and_then(|v| v.as_f64()).unwrap_or(mean);
                        let max_val = timing.get("max").and_then(|v| v.as_f64()).unwrap_or(mean);

                        metrics.insert(
                            MetricType::ExecutionTime,
                            MetricValue {
                                value: mean / 1_000_000_000.0, // Convert ns to seconds
                                std_dev: std_dev.map(|sd| sd / 1_000_000_000.0),
                                sample_count: 1,
                                min_value: min_val / 1_000_000_000.0,
                                max_value: max_val / 1_000_000_000.0,
                                percentiles: None,
                            },
                        );
                        return Some(metrics);
                    }
                }
            }
        }

        None
    }

    /// Parse custom metrics from output
    fn parse_custom_metrics(&self, output: &str) -> Option<HashMap<MetricType, MetricValue>> {
        let mut metrics = HashMap::new();

        for line in output.lines() {
            // Pattern: METRIC:name=value[unit]
            if let Some(captures) = Regex::new(r"METRIC:([^=]+)=([0-9.]+)(?:\s*\[([^\]]+)\])?")
                .ok()?
                .captures(line)
            {
                let name = captures[1].trim().to_string();
                if let Ok(value) = captures[2].parse::<f64>() {
                    let metric_type = MetricType::Custom(name);
                    metrics.insert(
                        metric_type,
                        MetricValue {
                            value,
                            std_dev: None,
                            sample_count: 1,
                            min_value: value,
                            max_value: value,
                            percentiles: None,
                        },
                    );
                }
            }
        }

        if metrics.is_empty() {
            None
        } else {
            Some(metrics)
        }
    }

    // Inline benchmark functions

    /// Execute simple loop benchmark
    fn execute_simple_loop_benchmark(&self, iterations: usize) -> f64 {
        let start = std::time::Instant::now();
        let mut sum = 0u64;
        for i in 0..iterations {
            sum = sum.wrapping_add(i as u64);
        }
        // Prevent optimization
        std::hint::black_box(sum);
        start.elapsed().as_secs_f64()
    }

    /// Execute memory allocation benchmark
    fn execute_memory_allocation_benchmark(&self, size: usize) -> f64 {
        let start = std::time::Instant::now();
        let mut vectors: Vec<Vec<u8>> = Vec::new();
        for _ in 0..100 {
            vectors.push(vec![0u8; size]);
        }
        // Prevent optimization
        std::hint::black_box(vectors);
        start.elapsed().as_secs_f64()
    }

    /// Execute computation-intensive benchmark
    fn execute_computation_benchmark(&self, complexity: usize) -> f64 {
        let start = std::time::Instant::now();
        let mut result = 0.0f64;
        for i in 0..complexity {
            for j in 0..complexity {
                result += (i as f64 * j as f64).sin().cos();
            }
        }
        // Prevent optimization
        std::hint::black_box(result);
        start.elapsed().as_secs_f64()
    }

    /// Execute sleep benchmark
    async fn execute_sleep_benchmark(&self, durationms: u64) -> f64 {
        let start = std::time::Instant::now();
        tokio::time::sleep(tokio::time::Duration::from_millis(durationms)).await;
        start.elapsed().as_secs_f64()
    }

    // Baseline management methods

    /// Load baseline from artifact storage
    async fn load_baseline_from_storage(
        &self,
        branch: &str,
    ) -> Result<Vec<PerformanceMeasurement>> {
        // Implementation would depend on the specific storage backend
        // For now, return an error to fall back to other methods
        Err(crate::error::OptimError::ExecutionError(format!(
            "Artifact storage not configured for branch: {}",
            branch
        )))
    }

    /// Load baseline from local cache
    async fn load_baseline_from_cache(&self, branch: &str) -> Result<Vec<PerformanceMeasurement>> {
        let cache_path = format!(".performance_cache/{}.json", branch);
        if std::path::Path::new(&cache_path).exists() {
            let content = std::fs::read_to_string(&cache_path).map_err(|e| {
                crate::error::OptimError::ExecutionError(format!("Failed to read cache: {}", e))
            })?;
            let baseline: Vec<PerformanceMeasurement> =
                serde_json::from_str(&content).map_err(|e| {
                    crate::error::OptimError::ExecutionError(format!(
                        "Failed to parse cache: {}",
                        e
                    ))
                })?;
            Ok(baseline)
        } else {
            Err(crate::error::OptimError::ExecutionError(format!(
                "No cached baseline found for branch: {}",
                branch
            )))
        }
    }

    /// Generate baseline from git history
    async fn generate_baseline_from_history(
        &self,
        branch: &str,
    ) -> Result<Vec<PerformanceMeasurement>> {
        // This is a simplified implementation
        // In practice, you'd check out previous commits and run benchmarks
        println!("Generating baseline from history for branch: {}", branch);

        // For now, return an error to indicate this feature needs more implementation
        Err(crate::error::OptimError::ExecutionError(
            "Baseline generation from history not fully implemented".to_string(),
        ))
    }

    /// Save baseline to cache
    async fn save_baseline_to_cache(
        &self,
        branch: &str,
        baseline: &[PerformanceMeasurement],
    ) -> Result<()> {
        std::fs::create_dir_all(".performance_cache").map_err(|e| {
            crate::error::OptimError::ExecutionError(format!("Failed to create cache dir: {}", e))
        })?;

        let cache_path = format!(".performance_cache/{}.json", branch);
        let content = serde_json::to_string_pretty(baseline).map_err(|e| {
            crate::error::OptimError::ExecutionError(format!("Failed to serialize baseline: {}", e))
        })?;

        std::fs::write(&cache_path, content).map_err(|e| {
            crate::error::OptimError::ExecutionError(format!("Failed to write cache: {}", e))
        })?;

        Ok(())
    }

    /// Convert Vec<PerformanceMeasurement> to BaselineMetrics
    fn convert_to_baseline_metrics(
        &self,
        measurements: Vec<PerformanceMeasurement>,
        version: String,
    ) -> Result<BaselineMetrics> {
        if measurements.is_empty() {
            return Err(OptimError::InvalidConfig(
                "No measurements to convert to baseline".to_string(),
            ));
        }

        let mut metrics = HashMap::new();
        let mut confidence_intervals = HashMap::new();

        // Extract all metric types from measurements
        let mut all_metric_types = std::collections::HashSet::new();
        for measurement in &measurements {
            for metric_type in measurement.metrics.keys() {
                all_metric_types.insert(metric_type.clone());
            }
        }

        // Calculate statistics for each metric type
        for metric_type in all_metric_types {
            let values: Vec<f64> = measurements
                .iter()
                .filter_map(|m| m.metrics.get(&metric_type))
                .map(|mv| mv.value)
                .collect();

            if !values.is_empty() {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance = if values.len() > 1 {
                    let mean_sq = values.iter().map(|&x| x * x).sum::<f64>() / values.len() as f64;
                    mean_sq - mean * mean
                } else {
                    0.0
                };
                let std_dev = variance.sqrt();

                metrics.insert(
                    metric_type.clone(),
                    MetricValue {
                        value: mean,
                        std_dev: Some(std_dev),
                        sample_count: values.len(),
                        min_value: values.iter().fold(f64::INFINITY, |acc, &x| acc.min(x)),
                        max_value: values.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x)),
                        percentiles: None,
                    },
                );

                // 95% confidence interval
                let margin = if values.len() > 1 {
                    1.96 * std_dev / (values.len() as f64).sqrt()
                } else {
                    0.0
                };
                confidence_intervals.insert(
                    metric_type,
                    ConfidenceInterval {
                        lower_bound: mean - margin,
                        upper_bound: mean + margin,
                        confidence_level: 0.95,
                    },
                );
            }
        }

        // Calculate quality score (simple version)
        let quality_score = if measurements.len() >= 10 {
            1.0
        } else {
            measurements.len() as f64 / 10.0
        };

        Ok(BaselineMetrics {
            version,
            timestamp: SystemTime::now(),
            metrics,
            confidence_intervals,
            quality_score,
        })
    }
}

/// Performance gate evaluation result
#[derive(Debug, Clone)]
pub struct GateResult {
    pub metric: MetricType,
    pub passed: bool,
    pub regression_percentage: f64,
    pub threshold: f64,
    pub confidence: f64,
    pub severity: GateSeverity,
    pub message: String,
}

// Implementation stubs for supporting types

impl PerformanceTestSuite {
    fn new(config: TestSuiteConfig) -> Result<Self> {
        Ok(Self {
            test_cases: Vec::new(),
            config,
        })
    }
}

impl ReportGenerator {
    fn new(config: ReportingConfig) -> Result<Self> {
        Ok(Self {
            template_engine: TemplateEngine::new()?,
            config,
        })
    }

    async fn generate_html_report(
        &self,
        measurements: &[PerformanceMeasurement],
        _regression_results: &[RegressionResult],
    ) -> Result<GeneratedReport> {
        // TODO: Implement HTML report generation
        Ok(GeneratedReport {
            report_type: ReportType::Html,
            file_path: PathBuf::from("performance_report.html"),
            artifact_url: None,
            metadata: HashMap::new(),
        })
    }

    async fn generate_json_report(
        &self,
        measurements: &[PerformanceMeasurement],
        _regression_results: &[RegressionResult],
    ) -> Result<GeneratedReport> {
        // TODO: Implement JSON report generation
        Ok(GeneratedReport {
            report_type: ReportType::Json,
            file_path: PathBuf::from("performance_report.json"),
            artifact_url: None,
            metadata: HashMap::new(),
        })
    }

    async fn generate_junit_report(
        &self,
        measurements: &[PerformanceMeasurement],
        _regression_results: &[RegressionResult],
    ) -> Result<GeneratedReport> {
        // TODO: Implement JUnit XML report generation
        Ok(GeneratedReport {
            report_type: ReportType::JUnit,
            file_path: PathBuf::from("performance_report.xml"),
            artifact_url: None,
            metadata: HashMap::new(),
        })
    }
}

impl TemplateEngine {
    fn new() -> Result<Self> {
        Ok(Self {
            templates: HashMap::new(),
        })
    }
}

impl ArtifactManager {
    fn new(config: ArtifactStorageConfig) -> Result<Self> {
        let storage_provider: Box<dyn ArtifactStorage> = match &config.provider {
            ArtifactStorageProvider::Local(path) => {
                Box::new(LocalArtifactStorage::new(path.clone()))
            }
            _ => Box::new(LocalArtifactStorage::new(PathBuf::from("./artifacts"))), // Default fallback
        };

        Ok(Self {
            storage_provider,
            config,
        })
    }
}

/// Local filesystem artifact storage
#[derive(Debug)]
pub struct LocalArtifactStorage {
    basepath: PathBuf,
}

impl LocalArtifactStorage {
    pub fn new(basepath: PathBuf) -> Self {
        Self { basepath }
    }
}

impl ArtifactStorage for LocalArtifactStorage {
    fn upload(&self, path: &Path, key: &str) -> Result<String> {
        let dest_path = self.basepath.join(key);
        if let Some(parent) = dest_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::copy(path, &dest_path)?;
        Ok(dest_path.to_string_lossy().to_string())
    }

    fn download(&self, key: &str, path: &Path) -> Result<()> {
        let src_path = self.basepath.join(key);
        std::fs::copy(src_path, path)?;
        Ok(())
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let prefix_path = self.basepath.join(prefix);
        let mut files = Vec::new();

        if prefix_path.is_dir() {
            for entry in std::fs::read_dir(prefix_path)? {
                let entry = entry?;
                if entry.file_type()?.is_file() {
                    files.push(entry.file_name().to_string_lossy().to_string());
                }
            }
        }

        Ok(files)
    }

    fn delete(&self, key: &str) -> Result<()> {
        let path = self.basepath.join(key);
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }
}

// Default implementations

impl Default for CiCdAutomationConfig {
    fn default() -> Self {
        Self {
            enable_automation: true,
            platform: CiCdPlatform::Generic,
            test_execution: TestExecutionConfig::default(),
            baseline_management: BaselineManagementConfig::default(),
            reporting: ReportingConfig::default(),
            artifact_storage: ArtifactStorageConfig::default(),
            integrations: IntegrationConfig::default(),
            performance_gates: PerformanceGatesConfig::default(),
        }
    }
}

impl Default for TestExecutionConfig {
    fn default() -> Self {
        Self {
            run_on_commit: true,
            run_on_pr: true,
            run_on_release: true,
            run_on_schedule: None,
            test_timeout: 3600, // 1 hour
            test_iterations: 5,
            warmup_iterations: 2,
            parallel_execution: true,
            isolation_level: TestIsolationLevel::Process,
        }
    }
}

impl Default for BaselineManagementConfig {
    fn default() -> Self {
        Self {
            auto_update_main: true,
            update_on_release: true,
            min_improvement_threshold: 0.05,
            require_manual_approval: false,
            retention_policy: BaselineRetentionPolicy {
                retention_days: 90,
                max_baselines: 100,
                keep_major_releases: true,
            },
        }
    }
}

impl Default for ReportingConfig {
    fn default() -> Self {
        Self {
            generate_html: true,
            generate_json: true,
            generate_junit: false,
            include_detailed_data: true,
            include_trends: true,
            include_baseline_comparison: true,
            custom_templates: HashMap::new(),
        }
    }
}

impl Default for ArtifactStorageConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            provider: ArtifactStorageProvider::Local(PathBuf::from("./artifacts")),
            storage_config: HashMap::new(),
            retention_policy: ArtifactRetentionPolicy {
                retention_days: 30,
                max_artifacts_per_branch: 50,
                keep_releases: true,
            },
        }
    }
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            github: None,
            slack: None,
            email: None,
            webhooks: Vec::new(),
        }
    }
}

impl Default for PerformanceGatesConfig {
    fn default() -> Self {
        let mut metric_gates = HashMap::new();
        metric_gates.insert(
            MetricType::ExecutionTime,
            MetricGate {
                max_regression: 10.0, // 10% regression threshold
                enabled: true,
                severity: GateSeverity::Blocking,
            },
        );
        metric_gates.insert(
            MetricType::MemoryUsage,
            MetricGate {
                max_regression: 20.0, // 20% regression threshold
                enabled: true,
                severity: GateSeverity::Warning,
            },
        );

        Self {
            enabled: true,
            fail_on_regression: false,
            max_regression_percentage: 10.0,
            min_confidence_threshold: 0.95,
            metric_gates,
        }
    }
}

impl Default for TestSuiteConfig {
    fn default() -> Self {
        Self {
            include_unit: true,
            include_integration: true,
            include_stress: false,
            test_filters: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ci_cd_automation_creation() {
        let config = CiCdAutomationConfig::default();
        let automation = CiCdAutomation::new(config);
        assert!(automation.is_ok());
    }

    #[test]
    fn test_environment_detection() {
        let env = CiCdAutomation::detect_environment();
        assert!(env.is_ok());
        let env = env.unwrap();
        assert!(!env.os.is_empty());
        assert!(env.cpu_cores > 0);
    }

    #[test]
    fn test_ci_platform_detection() {
        let config = CiCdAutomationConfig::default();
        let automation = CiCdAutomation::new(config).unwrap();
        let platform = automation.detect_ci_platform();
        // Should detect generic platform in test environment
        assert!(matches!(platform, CiCdPlatform::Generic));
    }

    #[test]
    fn test_performance_gate_evaluation() {
        let config = CiCdAutomationConfig::default();
        let automation = CiCdAutomation::new(config).unwrap();

        let regression_result = RegressionResult {
            metric: MetricType::ExecutionTime,
            severity: 0.8,
            confidence: 0.95,
            p_value: 0.05,
            effect_size: 0.5,
            baseline_value: 10.0,
            current_value: 12.0,
            change_percentage: 20.0, // 20% regression
            regression_type: crate::benchmarking::performance_regression_detector::RegressionType::IncreasedLatency,
            evidence: vec!["Statistical significance detected".to_string()],
            recommendations: vec!["Investigate performance degradation".to_string()],
        };

        let gate = MetricGate {
            max_regression: 10.0, // 10% threshold
            enabled: true,
            severity: GateSeverity::Blocking,
        };

        let gate_result = automation.evaluate_gate(&regression_result, &gate).unwrap();
        assert!(!gate_result.passed); // Should fail because 20% > 10%
    }

    #[test]
    fn test_artifact_storage() {
        let temp_dir = std::env::temp_dir().join("test_artifacts");
        let storage = LocalArtifactStorage::new(temp_dir.clone());

        // Create a test file
        let test_file = temp_dir.join("test_input.txt");
        std::fs::create_dir_all(&temp_dir).unwrap();
        std::fs::write(&test_file, "test content").unwrap();

        // Test upload
        let result = storage.upload(&test_file, "test_key.txt");
        assert!(result.is_ok());

        // Test list
        let files = storage.list("").unwrap();
        assert!(files.contains(&"test_key.txt".to_string()));

        // Cleanup
        std::fs::remove_dir_all(&temp_dir).ok();
    }
}
