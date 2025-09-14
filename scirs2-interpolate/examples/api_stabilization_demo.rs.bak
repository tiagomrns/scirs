//! API Stabilization Analysis Demo
//!
//! This example demonstrates how to use the API stabilization analyzer
//! to check readiness for the 0.1.0 stable release.

use scirs2__interpolate::{
    analyze_api_for_stable_release, analyze_api_with_config, quick_api_analysis,
    StabilizationConfig, StableReleaseReadiness, StrictnessLevel,
};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== API Stabilization Analysis Demo ===\n");

    // 1. Quick API analysis for development
    println!("1. Running quick API analysis...");
    match quick_api_analysis() {
        Ok(report) => {
            println!("{}", report);
            println!("Quick analysis readiness: {:?}\n", report.overall_readiness);
        }
        Err(e) => println!("Quick analysis failed: {}\n", e),
    }

    // 2. Comprehensive analysis for stable release
    println!("2. Running comprehensive analysis for stable release...");
    match analyze_api_for_stable_release() {
        Ok(report) => {
            println!("{}", report);

            match report.overall_readiness {
                StableReleaseReadiness::Ready => {
                    println!("✅ API is ready for stable release!");
                }
                StableReleaseReadiness::NeedsWork => {
                    println!("⚠️  API needs work before stable release");
                    if !report.recommendations.is_empty() {
                        println!("Priority actions:");
                        for rec in &report.recommendations[..3.min(report.recommendations.len())] {
                            println!("  - {}", rec);
                        }
                    }
                }
                StableReleaseReadiness::NotReady => {
                    println!("❌ API is not ready for stable release");
                    if !report.critical_issues.is_empty() {
                        println!("Critical blockers:");
                        for issue in &report.critical_issues[..3.min(report.critical_issues.len())]
                        {
                            println!("  - {}: {}", issue.location, issue.description);
                        }
                    }
                }
            }
            println!();
        }
        Err(e) => println!("Comprehensive analysis failed: {}\n", e),
    }

    // 3. Custom analysis with different strictness levels
    println!("3. Running analysis with different strictness levels...");

    let strictness_levels = [
        ("Relaxed", StrictnessLevel::Relaxed),
        ("Standard", StrictnessLevel::Standard),
        ("Strict", StrictnessLevel::Strict),
        ("Advanced-Strict", StrictnessLevel::AdvancedStrict),
    ];

    for (name, level) in &strictness_levels {
        let config = StabilizationConfig {
            min_documentation_coverage: match level {
                StrictnessLevel::Relaxed => 70.0,
                StrictnessLevel::Standard => 85.0,
                StrictnessLevel::Strict => 95.0,
                StrictnessLevel::AdvancedStrict => 98.0,
            },
            max_breaking_changes: match level {
                StrictnessLevel::Relaxed => 10,
                StrictnessLevel::Standard => 5,
                StrictnessLevel::Strict => 0,
                StrictnessLevel::AdvancedStrict => 0,
            },
            min_consistency_score: match level {
                StrictnessLevel::Relaxed => 0.7,
                StrictnessLevel::Standard => 0.8,
                StrictnessLevel::Strict => 0.9,
                StrictnessLevel::AdvancedStrict => 0.95,
            },
            allow_experimental_features: match level {
                StrictnessLevel::Relaxed => true,
                StrictnessLevel::Standard => true,
                StrictnessLevel::Strict => false,
                StrictnessLevel::AdvancedStrict => false,
            },
            strictness_level: level.clone(),
        };

        match analyze_api_with_config(config) {
            Ok(report) => {
                println!(
                    "  {} analysis: {:?} (consistency: {:.2}, coverage: {:.1}%)",
                    name,
                    report.overall_readiness,
                    report.consistency_score,
                    report.documentation_coverage
                );
            }
            Err(e) => println!("  {} analysis failed: {}", name, e),
        }
    }
    println!();

    // 4. Detailed API item analysis
    println!("4. Detailed API item analysis...");
    match analyze_api_for_stable_release() {
        Ok(report) => {
            println!("API Items by Stability Level:");

            // Group items by stability level
            let mut stable_items = Vec::new();
            let mut unstable_items = Vec::new();
            let mut experimental_items = Vec::new();

            for result in &report.analysis_results {
                match result.stability.level {
                    scirs2_interpolate::api_stabilization_enhanced::ApiStabilityLevel::Stable => {
                        stable_items.push(&result.item_name);
                    }
                    scirs2_interpolate::api_stabilization_enhanced::ApiStabilityLevel::MostlyStable => {
                        stable_items.push(&result.item_name);
                    }
                    scirs2_interpolate::api_stabilization_enhanced::ApiStabilityLevel::Unstable => {
                        unstable_items.push(&result.item_name);
                    }
                    scirs2_interpolate::api_stabilization_enhanced::ApiStabilityLevel::Experimental => {
                        experimental_items.push(&result.item_name);
                    }
                    scirs2_interpolate::api_stabilization_enhanced::ApiStabilityLevel::Deprecated => {
                        // Skip deprecated items
                    }
                }
            }

            println!(
                "  Stable items ({}): {}",
                stable_items.len(),
                stable_items
                    .iter()
                    .take(5)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            if stable_items.len() > 5 {
                println!("    ... and {} more", stable_items.len() - 5);
            }

            if !unstable_items.is_empty() {
                println!(
                    "  Unstable items ({}): {}",
                    unstable_items.len(),
                    unstable_items
                        .iter()
                        .take(3)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                if unstable_items.len() > 3 {
                    println!("    ... and {} more", unstable_items.len() - 3);
                }
            }

            if !experimental_items.is_empty() {
                println!(
                    "  Experimental items ({}): {}",
                    experimental_items.len(),
                    experimental_items
                        .iter()
                        .take(3)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                if experimental_items.len() > 3 {
                    println!("    ... and {} more", experimental_items.len() - 3);
                }
            }
            println!();
        }
        Err(e) => println!("Detailed analysis failed: {}\n", e),
    }

    // 5. Action plan for stable release
    println!("5. Action plan for stable release preparation:");
    match analyze_api_for_stable_release() {
        Ok(report) => {
            if report.overall_readiness == StableReleaseReadiness::Ready {
                println!("  ✅ No action needed - API is ready for stable release!");
            } else {
                println!("  📋 Priority actions:");
                for (i, recommendation) in report.recommendations.iter().enumerate() {
                    if i < 5 {
                        // Show top 5 recommendations
                        println!("    {}. {}", i + 1, recommendation);
                    }
                }

                if !report.critical_issues.is_empty() {
                    println!("  🚨 Critical blockers to resolve:");
                    for (i, issue) in report.critical_issues.iter().enumerate() {
                        if i < 3 {
                            // Show top 3 critical issues
                            println!("    - {} in {}", issue.description, issue.location);
                            if let Some(resolution) = &issue.suggested_resolution {
                                println!("      → {}", resolution);
                            }
                        }
                    }
                }

                if !report.breaking_changes.is_empty() {
                    println!("  ⚠️  Breaking changes to address:");
                    for (i, change) in report.breaking_changes.iter().enumerate() {
                        if i < 2 {
                            // Show top 2 breaking changes
                            println!("    - {}", change.description);
                            if let Some(mitigation) = &change.mitigation_strategy {
                                println!("      → Mitigation: {}", mitigation);
                            }
                        }
                    }
                }
            }
        }
        Err(e) => println!("  Failed to generate action plan: {}", e),
    }

    println!("\n=== Analysis Complete ===");
    println!("This analysis helps ensure the scirs2-interpolate crate is ready for a stable 0.1.0 release.");
    println!("Use the results to prioritize work and ensure API quality before release.");

    Ok(())
}
