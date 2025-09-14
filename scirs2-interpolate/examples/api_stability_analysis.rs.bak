//! API Stability Analysis for 0.1.0 Stable Release
//!
//! This example runs comprehensive API stabilization analysis to identify
//! issues that need to be addressed before the stable release.

use scirs2__interpolate::api_stabilization_enhanced::{
    analyze_api_for_stable_release, analyze_api_with_config, quick_api_analysis,
    StabilizationConfig, StrictnessLevel,
};
use scirs2__interpolate::InterpolateResult;

#[allow(dead_code)]
fn main() -> InterpolateResult<()> {
    println!("=== SciRS2 Interpolate API Stability Analysis ===\n");

    // Run strict analysis for stable release
    println!("Running strict API analysis for 0.1.0 stable release...\n");
    match analyze_api_for_stable_release() {
        Ok(report) => {
            println!("=== API STABILITY REPORT ===");
            println!("Overall Readiness: {:?}", report.overall_readiness);
            println!("Total API Items: {}", report.total_items);
            println!("Stable Items: {}", report.stable_items);
            println!("Unstable Items: {}", report.unstable_items);
            println!("Consistency Score: {:.2}", report.consistency_score);
            println!(
                "Documentation Coverage: {:.1}%",
                report.documentation_coverage
            );
            println!("Breaking Changes: {}", report.breaking_changes.len());
            println!("Critical Issues: {}", report.critical_issues.len());

            if !report.critical_issues.is_empty() {
                println!("\n=== CRITICAL ISSUES (BLOCKING STABLE RELEASE) ===");
                for (i, issue) in report.critical_issues.iter().enumerate() {
                    println!(
                        "{}. [{}] {} - {}",
                        i + 1,
                        issue.category,
                        issue.location,
                        issue.description
                    );
                    if let Some(resolution) = &issue.suggested_resolution {
                        println!("   Resolution: {}", resolution);
                    }
                }
            }

            if !report.breaking_changes.is_empty() {
                println!("\n=== BREAKING CHANGES ===");
                for (i, change) in report.breaking_changes.iter().enumerate() {
                    println!("{}. [{}] {}", i + 1, change.severity, change.description);
                    println!("   Affected: {}", change.affected_items.join(", "));
                    if let Some(migration) = &change.migration_path {
                        println!("   Migration: {}", migration);
                    }
                }
            }

            if !report.recommendations.is_empty() {
                println!("\n=== RECOMMENDATIONS ===");
                for (i, rec) in report.recommendations.iter().enumerate() {
                    println!("{}. {}", i + 1, rec);
                }
            }

            println!("\n=== ANALYSIS SUMMARY ===");
            match report.overall_readiness {
                scirs2_interpolate::api_stabilization_enhanced::StableReleaseReadiness::Ready => {
                    println!("✅ API is READY for 0.1.0 stable release!");
                }
                scirs2_interpolate::api_stabilization_enhanced::StableReleaseReadiness::NearlyReady => {
                    println!("⚠️  API is NEARLY READY - address remaining issues");
                }
                scirs2_interpolate::api_stabilization_enhanced::StableReleaseReadiness::NotReady => {
                    println!("❌ API is NOT READY - significant work required");
                }
            }
        }
        Err(e) => {
            println!("Failed to run API analysis: {}", e);

            // Try quick analysis as fallback
            println!("\nTrying quick analysis as fallback...");
            match quick_api_analysis() {
                Ok(report) => {
                    println!("Quick analysis completed successfully");
                    println!(
                        "Total Items: {}, Stable: {}, Unstable: {}",
                        report.total_items, report.stable_items, report.unstable_items
                    );
                }
                Err(e2) => {
                    println!("Quick analysis also failed: {}", e2);
                }
            }
        }
    }

    // Also try custom analysis with specific configuration
    println!("\n=== CUSTOM ANALYSIS ===");
    let custom_config = StabilizationConfig {
        min_documentation_coverage: 90.0,
        max_breaking_changes: 5,
        min_consistency_score: 0.85,
        allow_experimental_features: true,
        strictness_level: StrictnessLevel::Standard,
    };

    match analyze_api_with_config(custom_config) {
        Ok(report) => {
            println!("Custom analysis (relaxed constraints):");
            println!("  Overall Readiness: {:?}", report.overall_readiness);
            println!("  Critical Issues: {}", report.critical_issues.len());
            println!(
                "  Documentation Coverage: {:.1}%",
                report.documentation_coverage
            );
        }
        Err(e) => {
            println!("Custom analysis failed: {}", e);
        }
    }

    Ok(())
}
