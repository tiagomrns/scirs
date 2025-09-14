//! Comprehensive Stable Release Validation Script
//!
//! This script validates all aspects of the scirs2-interpolate library
//! for the 0.1.0 stable release, combining:
//! - API stabilization analysis
//! - Performance validation against SciPy
//! - Production hardening stress tests
//! - Documentation quality assessment
//! - SciPy parity completion
//!
//! Usage: cargo run --example stable_release_validation

use scirs2__interpolate::{
    // API Stabilization
    analyze_api_for_stable_release,
    // SciPy Parity
    enhance_scipy_parity_for_stable_release,
    // Documentation Enhancement
    polish_documentation_for_stable_release,
    // Production Hardening
    run_production_hardening,
    // Stress Testing
    run_production_stress_tests,
    // Performance Validation
    validate_stable_release_readiness,
    DocumentationReadiness,

    ParityReadiness,

    ProductionReadiness as HardeningReadiness,

    ProductionReadiness as StressReadiness,
    StableReadiness as PerfReadiness,

    StableReleaseReadiness as ApiReadiness,
};

/// Overall readiness assessment for stable release
#[derive(Debug, PartialEq)]
pub enum OverallReadiness {
    Ready,
    NearReady,
    NeedsWork,
    NotReady,
}

/// Comprehensive validation results
pub struct StableReleaseValidationReport {
    pub api_readiness: ApiReadiness,
    pub performance_readiness: PerfReadiness,
    pub production_readiness: HardeningReadiness,
    pub scipy_parity_readiness: ParityReadiness,
    pub documentation_readiness: DocumentationReadiness,
    pub stress_test_readiness: StressReadiness,
    pub overall_readiness: OverallReadiness,
    pub critical_blockers: Vec<String>,
    pub priority_tasks: Vec<String>,
    pub summary: String,
}

impl std::fmt::Display for StableReleaseValidationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "=== SCIRS2-INTERPOLATE 0.1.0 STABLE RELEASE VALIDATION ==="
        )?;
        writeln!(f)?;
        writeln!(f, "🎯 OVERALL READINESS: {:?}", self.overall_readiness)?;
        writeln!(f)?;

        writeln!(f, "📊 COMPONENT READINESS:")?;
        writeln!(f, "  API Stabilization:     {:?}", self.api_readiness)?;
        writeln!(
            f,
            "  Performance Validation: {:?}",
            self.performance_readiness
        )?;
        writeln!(
            f,
            "  Production Hardening:   {:?}",
            self.production_readiness
        )?;
        writeln!(
            f,
            "  SciPy Parity:          {:?}",
            self.scipy_parity_readiness
        )?;
        writeln!(
            f,
            "  Documentation:         {:?}",
            self.documentation_readiness
        )?;
        writeln!(
            f,
            "  Stress Testing:        {:?}",
            self.stress_test_readiness
        )?;
        writeln!(f)?;

        if !self.critical_blockers.is_empty() {
            writeln!(f, "🚨 CRITICAL BLOCKERS:")?;
            for blocker in &self.critical_blockers {
                writeln!(f, "  - {}", blocker)?;
            }
            writeln!(f)?;
        }

        if !self.priority_tasks.is_empty() {
            writeln!(f, "⚠️  PRIORITY TASKS:")?;
            for task in &self.priority_tasks {
                writeln!(f, "  - {}", task)?;
            }
            writeln!(f)?;
        }

        writeln!(f, "📝 SUMMARY:")?;
        writeln!(f, "{}", self.summary)?;

        Ok(())
    }
}

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Starting Comprehensive Stable Release Validation ===\n");

    let mut critical_blockers = Vec::new();
    let mut priority_tasks = Vec::new();

    // 1. API Stabilization Analysis
    println!("1. 🔍 Analyzing API Stabilization...");
    let api_report = analyze_api_for_stable_release()?;
    let api_readiness = api_report.overall_readiness.clone();

    match api_readiness {
        ApiReadiness::Ready => println!("   ✅ API is stable and ready"),
        ApiReadiness::NeedsWork => {
            println!("   ⚠️  API needs minor adjustments");
            priority_tasks.extend(api_report.recommendations.iter().take(3).cloned());
        }
        ApiReadiness::NotReady => {
            println!("   ❌ API has critical stability issues");
            critical_blockers.extend(
                api_report
                    .critical_issues
                    .iter()
                    .take(3)
                    .map(|issue| format!("API: {}", issue.description)),
            );
        }
    }
    println!("   Critical issues: {}", api_report.critical_issues.len());
    println!("   Recommendations: {}", api_report.recommendations.len());
    println!();

    // 2. Performance Validation
    println!("2. ⚡ Validating Performance...");
    let perf_report = validate_stable_release_readiness::<f64>()?;
    let performance_readiness = perf_report.overall_readiness.clone();

    match performance_readiness {
        PerfReadiness::Ready => println!("   ✅ Performance meets stable release criteria"),
        PerfReadiness::NearReady => {
            println!("   ⚠️  Performance is acceptable with minor concerns");
            priority_tasks.extend(perf_report.priority_items.iter().take(2).cloned());
        }
        PerfReadiness::NeedsWork => {
            println!("   ❌ Performance issues prevent stable release");
            critical_blockers.extend(
                perf_report
                    .priority_items
                    .iter()
                    .take(3)
                    .map(|item| format!("Performance: {}", item)),
            );
        }
    }
    println!(
        "   Benchmarks passed: {}/{}",
        perf_report
            .results
            .iter()
            .filter(|r| r.status.to_string().contains("Passed"))
            .count(),
        perf_report.results.len()
    );
    println!();

    // 3. Production Hardening
    println!("3. 🛡️  Validating Production Hardening...");
    let hardening_report = run_production_hardening::<f64>()?;
    let production_readiness = hardening_report.production_readiness.clone();

    match production_readiness {
        HardeningReadiness::Ready => println!("   ✅ Production hardening complete"),
        HardeningReadiness::NeedsWork => {
            println!("   ⚠️  Some production concerns remain");
            priority_tasks.extend(
                hardening_report
                    .priority_issues
                    .iter()
                    .take(2)
                    .map(|issue| issue.description.clone()),
            );
        }
        HardeningReadiness::NotReady => {
            println!("   ❌ Critical production issues found");
            critical_blockers.extend(
                hardening_report
                    .critical_issues
                    .iter()
                    .take(3)
                    .map(|issue| format!("Production: {}", issue.description)),
            );
        }
    }
    println!(
        "   Tests passed: {}/{}",
        hardening_report
            .test_results
            .iter()
            .filter(|r| r.status.to_string().contains("Passed"))
            .count(),
        hardening_report.test_results.len()
    );
    println!();

    // 4. SciPy Parity Enhancement
    println!("4. 🐍 Validating SciPy Parity...");
    let scipy_report = enhance_scipy_parity_for_stable_release::<f64>()?;
    let scipy_readiness = scipy_report.readiness.clone();

    match scipy_readiness {
        ParityReadiness::Ready => println!("   ✅ SciPy parity is complete"),
        ParityReadiness::NearReady => {
            println!("   ⚠️  Minor SciPy parity gaps remain");
            priority_tasks.push("Complete remaining SciPy compatibility features".to_string());
        }
        ParityReadiness::NeedsWork => {
            println!("   ⚠️  Several SciPy features missing");
            priority_tasks.push("Implement critical missing SciPy features".to_string());
        }
        ParityReadiness::NotReady => {
            println!("   ❌ Major SciPy parity gaps prevent release");
            critical_blockers.push("Critical SciPy features missing".to_string());
        }
    }
    println!(
        "   Feature coverage: {:.1}%",
        scipy_report.overall_coverage * 100.0
    );
    println!(
        "   Performance ratio: {:.2}x",
        scipy_report.performance_summary.overall_ratio
    );
    println!();

    // 5. Documentation Enhancement
    println!("5. 📚 Validating Documentation...");
    let doc_report = polish_documentation_for_stable_release();
    let doc_readiness = doc_report.readiness.clone();

    match doc_readiness {
        DocumentationReadiness::Ready => println!("   ✅ Documentation is complete and polished"),
        DocumentationReadiness::NearReady => {
            println!("   ⚠️  Documentation needs minor improvements");
            priority_tasks.extend(doc_report.priority_tasks.iter().take(2).cloned());
        }
        DocumentationReadiness::NeedsWork => {
            println!("   ❌ Documentation gaps prevent stable release");
            critical_blockers.extend(
                doc_report
                    .priority_tasks
                    .iter()
                    .take(3)
                    .map(|task| format!("Documentation: {}", task)),
            );
        }
    }
    println!("   Coverage: {:.1}%", doc_report.coverage_score * 100.0);
    println!("   Quality score: {:.1}/10", doc_report.quality_score);
    println!();

    // 6. Stress Testing
    println!("6. 💪 Running Production Stress Tests...");
    let stress_report = run_production_stress_tests::<f64>()?;
    let stress_readiness = stress_report.production_readiness.clone();

    match stress_readiness {
        StressReadiness::Ready => println!("   ✅ All stress tests passed"),
        StressReadiness::NeedsWork => {
            println!("   ⚠️  Some stress test concerns");
            if stress_report
                .test_results
                .iter()
                .any(|r| r.status.to_string().contains("Failed"))
            {
                priority_tasks.push("Address stress test failures".to_string());
            }
        }
        StressReadiness::NotReady => {
            println!("   ❌ Critical stress test failures");
            critical_blockers
                .push("Critical stress test failures prevent production use".to_string());
        }
    }
    println!(
        "   Stress tests passed: {}/{}",
        stress_report
            .test_results
            .iter()
            .filter(|r| r.status.to_string().contains("Passed"))
            .count(),
        stress_report.test_results.len()
    );
    println!();

    // Calculate overall readiness
    let overall_readiness = determine_overall_readiness(
        &api_readiness,
        &performance_readiness,
        &production_readiness,
        &scipy_readiness,
        &doc_readiness,
        &stress_readiness,
        &critical_blockers,
    );

    // Generate summary
    let summary = generate_summary(
        &overall_readiness,
        critical_blockers.len(),
        priority_tasks.len(),
    );

    // Create final report
    let final_report = StableReleaseValidationReport {
        api_readiness,
        performance_readiness,
        production_readiness,
        scipy_parity_readiness: scipy_readiness,
        documentation_readiness: doc_readiness,
        stress_test_readiness: stress_readiness,
        overall_readiness,
        critical_blockers,
        priority_tasks,
        summary,
    };

    // Display final results
    println!("{}", final_report);

    // Exit with appropriate code
    match final_report.overall_readiness {
        OverallReadiness::Ready => {
            println!("🎉 SCIRS2-INTERPOLATE IS READY FOR 0.1.0 STABLE RELEASE! 🎉");
            std::process::exit(0);
        }
        OverallReadiness::NearReady => {
            println!("⚠️  Minor issues remain before stable release");
            std::process::exit(1);
        }
        OverallReadiness::NeedsWork => {
            println!("⚠️  Moderate work needed before stable release");
            std::process::exit(2);
        }
        OverallReadiness::NotReady => {
            println!("❌ Critical issues prevent stable release");
            std::process::exit(3);
        }
    }
}

#[allow(dead_code)]
fn determine_overall_readiness(
    api: &ApiReadiness,
    perf: &PerfReadiness,
    prod: &HardeningReadiness,
    scipy: &ParityReadiness,
    doc: &DocumentationReadiness,
    stress: &StressReadiness,
    critical_blockers: &[String],
) -> OverallReadiness {
    // Any critical _blockers mean not ready
    if !critical_blockers.is_empty() {
        return OverallReadiness::NotReady;
    }

    // Check for any "NotReady" components
    if matches!(api, ApiReadiness::NotReady)
        || matches!(perf, PerfReadiness::NeedsWork)
        || matches!(prod, HardeningReadiness::NotReady)
        || matches!(scipy, ParityReadiness::NotReady)
        || matches!(doc, DocumentationReadiness::NeedsWork)
        || matches!(stress, StressReadiness::NotReady)
    {
        return OverallReadiness::NotReady;
    }

    // Count components that need work
    let needs_work_count = [
        matches!(api, ApiReadiness::NeedsWork),
        matches!(perf, PerfReadiness::NearReady),
        matches!(prod, HardeningReadiness::NeedsWork),
        matches!(scipy, ParityReadiness::NeedsWork),
        matches!(doc, DocumentationReadiness::NearReady),
        matches!(stress, StressReadiness::NeedsWork),
    ]
    .iter()
    .filter(|&&x| x)
    .count();

    match needs_work_count {
        0 => OverallReadiness::Ready,
        1..=2 => OverallReadiness::NearReady,
        _ => OverallReadiness::NeedsWork,
    }
}

#[allow(dead_code)]
fn generate_summary(
    readiness: &OverallReadiness,
    critical_count: usize,
    priority_count: usize,
) -> String {
    match readiness {
        OverallReadiness::Ready => {
            "All validation criteria have been met. The scirs2-interpolate library is ready for the 0.1.0 stable release with high confidence in API stability, performance, and production readiness.".to_string()
        }
        OverallReadiness::NearReady => {
            format!("The library is very close to stable release readiness with {} priority tasks remaining. These are minor issues that should be addressed but do not block the release.", priority_count)
        }
        OverallReadiness::NeedsWork => {
            format!("Moderate work is needed before stable release. {} priority tasks should be completed to ensure release quality and stability.", priority_count)
        }
        OverallReadiness::NotReady => {
            format!("Critical issues prevent stable release. {} critical blockers must be resolved before proceeding. Consider additional development iterations.", critical_count)
        }
    }
}
