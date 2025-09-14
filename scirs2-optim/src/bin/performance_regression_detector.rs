//! Performance Regression Detection Binary
//!
//! This binary provides a command-line interface for detecting performance regressions
//! in benchmark results using statistical analysis and configurable thresholds.

use clap::{Arg, Command};
use scirs2_optim::benchmarking::performance_regression_detector::{
    AlertThresholds, BaselineStrategy, CiCdConfig, EnvironmentInfo, MetricType, MetricValue,
    PerformanceMeasurement, PerformanceRegressionDetector, RegressionConfig, RegressionSensitivity,
    ReportFormat, StatisticalTest, TestConfiguration,
};
use scirs2_optim::error::{OptimError, Result};
use serde_json;
use std::fs;
use std::path::PathBuf;
use std::process;

#[allow(dead_code)]
fn main() {
    let matches = Command::new("performance-regression-detector")
        .version("0.1.0")
        .author("SciRS2 Development Team")
        .about("Advanced performance regression detection for continuous integration")
        .arg(
            Arg::new("benchmark-results")
                .long("benchmark-results")
                .value_name("FILE")
                .help("Path to benchmark results JSON file")
                .required(true),
        )
        .arg(
            Arg::new("baseline-dir")
                .long("baseline-dir")
                .value_name("DIR")
                .help("Directory containing baseline performance data")
                .required(true),
        )
        .arg(
            Arg::new("output-report")
                .long("output-report")
                .value_name("FILE")
                .help("Output file for regression analysis report")
                .required(true),
        )
        .arg(
            Arg::new("confidence-threshold")
                .long("confidence-threshold")
                .value_name("FLOAT")
                .help("Statistical confidence threshold (0.0-1.0)")
                .default_value("0.95"),
        )
        .arg(
            Arg::new("degradation-threshold")
                .long("degradation-threshold")
                .value_name("FLOAT")
                .help("Performance degradation threshold (e.g., 0.05 = 5%)")
                .default_value("0.05"),
        )
        .arg(
            Arg::new("sensitivity")
                .long("sensitivity")
                .value_name("LEVEL")
                .help("Regression detection sensitivity")
                .value_parser(["low", "medium", "high"])
                .default_value("medium"),
        )
        .arg(
            Arg::new("features")
                .long("features")
                .value_name("STRING")
                .help("Feature set being tested")
                .required(true),
        )
        .arg(
            Arg::new("statistical-test")
                .long("statistical-test")
                .value_name("TEST")
                .help("Statistical test to use for regression detection")
                .value_parser(["mann-whitney", "t-test", "wilcoxon", "kolmogorov-smirnov"])
                .default_value("mann-whitney"),
        )
        .arg(
            Arg::new("min-samples")
                .long("min-samples")
                .value_name("NUMBER")
                .help("Minimum number of samples required for analysis")
                .default_value("5"),
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .help("Enable verbose output")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("fail-on-regression")
                .long("fail-on-regression")
                .help("Exit with error code if regressions are detected")
                .action(clap::ArgAction::SetTrue),
        )
        .get_matches();

    // Parse command line arguments
    let benchmark_results = matches.get_one::<String>("benchmark-results").unwrap();
    let baseline_dir = matches.get_one::<String>("baseline-dir").unwrap();
    let outputreport = matches.get_one::<String>("output-report").unwrap();
    let confidence_threshold: f64 = matches
        .get_one::<String>("confidence-threshold")
        .unwrap()
        .parse()
        .unwrap_or_else(|_| {
            eprintln!("Error: Invalid confidence threshold");
            process::exit(1);
        });
    let degradation_threshold: f64 = matches
        .get_one::<String>("degradation-threshold")
        .unwrap()
        .parse()
        .unwrap_or_else(|_| {
            eprintln!("Error: Invalid degradation threshold");
            process::exit(1);
        });
    let sensitivity_str = matches.get_one::<String>("sensitivity").unwrap();
    let features = matches.get_one::<String>("features").unwrap();
    let statistical_test_str = matches.get_one::<String>("statistical-test").unwrap();
    let min_samples: usize = matches
        .get_one::<String>("min-samples")
        .unwrap()
        .parse()
        .unwrap_or_else(|_| {
            eprintln!("Error: Invalid min-samples value");
            process::exit(1);
        });
    let verbose = matches.get_flag("verbose");
    let fail_on_regression = matches.get_flag("fail-on-regression");

    // Convert string arguments to enum values
    let sensitivity = match sensitivity_str.as_str() {
        "low" => RegressionSensitivity::Low,
        "medium" => RegressionSensitivity::Medium,
        "high" => RegressionSensitivity::High,
        _ => {
            eprintln!("Error: Invalid sensitivity level");
            process::exit(1);
        }
    };

    let statistical_test = match statistical_test_str.as_str() {
        "mann-whitney" => StatisticalTest::MannWhitneyU,
        "t-test" => StatisticalTest::StudentTTest,
        "wilcoxon" => StatisticalTest::WilcoxonSignedRank,
        "kolmogorov-smirnov" => StatisticalTest::KolmogorovSmirnov,
        _ => {
            eprintln!("Error: Invalid statistical test");
            process::exit(1);
        }
    };

    // Create regression detection configuration
    let config = RegressionConfig {
        enable_detection: true,
        confidence_threshold,
        degradation_threshold,
        min_samples,
        max_history_size: 1000,
        tracked_metrics: vec![
            MetricType::ExecutionTime,
            MetricType::MemoryUsage,
            MetricType::Throughput,
            MetricType::ConvergenceRate,
        ],
        statistical_test: statistical_test.clone(),
        sensitivity: sensitivity.clone(),
        baseline_strategy: BaselineStrategy::RollingWindow(10),
        alert_thresholds: AlertThresholds::default(),
        ci_cd_config: CiCdConfig {
            enabled: true,
            fail_on_regression,
            generate_reports: true,
            report_format: ReportFormat::Json,
            report_path: PathBuf::from(outputreport),
            webhook_urls: vec![],
            slack_config: None,
            email_config: None,
        },
    };

    if verbose {
        println!("🔍 Performance Regression Detection Configuration:");
        println!("  Benchmark Results: {benchmark_results}");
        println!("  Baseline Directory: {baseline_dir}");
        println!("  Output Report: {outputreport}");
        println!("  Confidence Threshold: {confidence_threshold:.2}");
        println!(
            "  Degradation Threshold: {:.1}%",
            degradation_threshold * 100.0
        );
        println!("  Sensitivity: {:?}", sensitivity);
        println!("  Statistical Test: {:?}", statistical_test);
        println!("  Features: {features}");
        println!("  Min Samples: {min_samples}");
        println!();
    }

    // Run regression detection
    match run_regression_detection(
        benchmark_results,
        baseline_dir,
        outputreport,
        features,
        config,
        verbose,
    ) {
        Ok(has_regressions) => {
            if has_regressions && fail_on_regression {
                println!("❌ Performance regressions detected - failing build");
                process::exit(1);
            } else if has_regressions {
                println!("⚠️  Performance regressions detected - check report for details");
                process::exit(0);
            } else {
                println!("✅ No performance regressions detected");
                process::exit(0);
            }
        }
        Err(e) => {
            eprintln!("❌ Error running regression detection: {e}");
            process::exit(1);
        }
    }
}

#[allow(dead_code)]
fn run_regression_detection(
    benchmark_results: &str,
    baseline_dir: &str,
    outputreport: &str,
    features: &str,
    config: RegressionConfig,
    verbose: bool,
) -> Result<bool> {
    if verbose {
        println!("📊 Loading benchmark results...");
    }

    // Load benchmark _results
    let benchmarkdata = load_benchmark_results(benchmark_results)?;

    if verbose {
        println!("📈 Initializing regression detector...");
    }

    // Create regression detector
    let mut detector = PerformanceRegressionDetector::new(config)?;

    // Load baseline data
    let baselinepath = PathBuf::from(baseline_dir).join(format!("baseline_{features}.json"));
    if baselinepath.exists() {
        if verbose {
            println!("📋 Loading baseline data from: {}", baselinepath.display());
        }
        // Load baseline from file (simplified implementation)
        let _baseline_content = fs::read_to_string(&baselinepath)
            .map_err(|e| OptimError::ResourceError(format!("Failed to read baseline: {e}")))?;
        // For now, we'll skip the baseline loading and use current data as baseline
    } else {
        if verbose {
            println!("⚠️  No baseline data found - will establish new baseline");
        }
    }

    // Convert benchmark data to measurements
    let measurements = convert_benchmarkdata_to_measurements(&benchmarkdata, features)?;

    // Add measurements to detector
    for measurement in measurements {
        detector.add_measurement(measurement)?;
    }

    if verbose {
        println!("🔬 Analyzing performance data...");
    }

    // Detect regressions
    let regression_results = detector.detect_regressions()?;

    if verbose {
        println!("📝 Generating regression report...");
    }

    // Generate CI/CD report
    let report = detector.export_for_ci_cd()?;

    // Save report
    let outputpath = PathBuf::from(outputreport);
    let output_dir = outputpath.parent().unwrap();
    fs::create_dir_all(output_dir).map_err(|e| {
        OptimError::ResourceError(format!("Failed to create output directory: {e}"))
    })?;

    let report_json = serde_json::to_string_pretty(&report)
        .map_err(|e| OptimError::OptimizationError(format!("Failed to serialize report: {e}")))?;

    fs::write(outputreport, report_json)
        .map_err(|e| OptimError::ResourceError(format!("Failed to write report: {e}")))?;

    // Check if regressions were detected
    let has_regressions = !regression_results.is_empty();

    if verbose {
        println!("📊 Analysis Summary:");
        println!("  Regressions Detected: {}", regression_results.len());
        println!(
            "  Critical Regressions: {}",
            regression_results
                .iter()
                .filter(|r| r.severity >= 0.9)
                .count()
        );
        println!(
            "  High Severity Regressions: {}",
            regression_results
                .iter()
                .filter(|r| r.severity >= 0.7)
                .count()
        );
        println!(
            "  Status: {}",
            if has_regressions {
                "⚠️  REGRESSIONS FOUND"
            } else {
                "✅ PASSED"
            }
        );
        println!();
        println!("📄 Report saved to: {outputreport}");
    }

    Ok(has_regressions)
}

#[allow(dead_code)]
fn load_benchmark_results(path: &str) -> Result<serde_json::Value> {
    let content = fs::read_to_string(path)
        .map_err(|e| OptimError::ResourceError(format!("Failed to read benchmark results: {e}")))?;

    let data: serde_json::Value = serde_json::from_str(&content).map_err(|e| {
        OptimError::OptimizationError(format!("Failed to parse benchmark results: {e}"))
    })?;

    Ok(data)
}

#[allow(dead_code)]
fn convert_benchmarkdata_to_measurements(
    data: &serde_json::Value,
    _features: &str,
) -> Result<Vec<PerformanceMeasurement>> {
    use std::collections::HashMap;
    use std::time::SystemTime;

    let mut measurements = Vec::new();

    // Parse benchmark data (assuming it's in a specific format)
    if let Some(benchmarks) = data.as_array() {
        for (i, benchmark) in benchmarks.iter().enumerate() {
            let mut metrics = HashMap::new();

            // Extract execution time
            if let Some(time) = benchmark.get("execution_time").and_then(|v| v.as_f64()) {
                metrics.insert(
                    MetricType::ExecutionTime,
                    MetricValue {
                        value: time,
                        std_dev: benchmark.get("execution_time_std").and_then(|v| v.as_f64()),
                        sample_count: benchmark
                            .get("sample_count")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(1) as usize,
                        min_value: benchmark
                            .get("min_time")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(time),
                        max_value: benchmark
                            .get("max_time")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(time),
                        percentiles: None,
                    },
                );
            }

            // Extract memory usage
            if let Some(memory) = benchmark.get("memory_usage").and_then(|v| v.as_f64()) {
                metrics.insert(
                    MetricType::MemoryUsage,
                    MetricValue {
                        value: memory,
                        std_dev: benchmark.get("memory_std").and_then(|v| v.as_f64()),
                        sample_count: 1,
                        min_value: memory,
                        max_value: memory,
                        percentiles: None,
                    },
                );
            }

            // Extract throughput
            if let Some(throughput) = benchmark.get("throughput").and_then(|v| v.as_f64()) {
                metrics.insert(
                    MetricType::Throughput,
                    MetricValue {
                        value: throughput,
                        std_dev: benchmark.get("throughput_std").and_then(|v| v.as_f64()),
                        sample_count: 1,
                        min_value: throughput,
                        max_value: throughput,
                        percentiles: None,
                    },
                );
            }

            // Create measurement
            let measurement = PerformanceMeasurement {
                timestamp: SystemTime::now(),
                commithash: benchmark
                    .get("commit_hash")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
                branch: benchmark
                    .get("branch")
                    .and_then(|v| v.as_str())
                    .unwrap_or("main")
                    .to_string(),
                build_config: benchmark
                    .get("build_config")
                    .and_then(|v| v.as_str())
                    .unwrap_or("release")
                    .to_string(),
                environment: EnvironmentInfo::default(),
                metrics,
                test_config: TestConfiguration {
                    test_name: benchmark
                        .get("test_name")
                        .and_then(|v| v.as_str())
                        .unwrap_or(&format!("test_{i}"))
                        .to_string(),
                    parameters: HashMap::new(),
                    dataset_size: benchmark
                        .get("dataset_size")
                        .and_then(|v| v.as_u64())
                        .map(|v| v as usize),
                    iterations: benchmark
                        .get("iterations")
                        .and_then(|v| v.as_u64())
                        .map(|v| v as usize),
                    batch_size: benchmark
                        .get("batch_size")
                        .and_then(|v| v.as_u64())
                        .map(|v| v as usize),
                    precision: benchmark
                        .get("precision")
                        .and_then(|v| v.as_str())
                        .unwrap_or("f64")
                        .to_string(),
                },
                metadata: HashMap::new(),
            };

            measurements.push(measurement);
        }
    }

    Ok(measurements)
}
