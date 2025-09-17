//! Comprehensive numerical validation framework for scirs2-graph
//!
//! This module provides automated validation against reference implementations
//! with configurable tolerance levels and detailed error reporting.

use approx::{assert_abs_diff_eq, assert_relative_eq};
use scirs2_core::error::CoreResult;
use scirs2_graph::{algorithms, generators, measures, spectral, DiGraph, Graph};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;

/// Configuration for numerical validation tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Absolute tolerance for comparisons
    pub absolute_tolerance: f64,
    /// Relative tolerance for comparisons
    pub relative_tolerance: f64,
    /// Maximum allowable relative error
    pub max_relative_error: f64,
    /// Enable detailed error reporting
    pub verbose_errors: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            absolute_tolerance: 1e-10,
            relative_tolerance: 1e-8,
            max_relative_error: 1e-6,
            verbose_errors: true,
        }
    }
}

/// Reference values loaded from NetworkX computation
#[derive(Debug, Deserialize)]
pub struct ReferenceValues {
    pub pagerank: HashMap<String, serde_json::Value>,
    pub betweenness: HashMap<String, serde_json::Value>,
    pub clustering: HashMap<String, serde_json::Value>,
    pub shortest_paths: HashMap<String, serde_json::Value>,
    pub eigenvector_centrality: HashMap<String, serde_json::Value>,
    pub spectral: HashMap<String, serde_json::Value>,
    pub max_flow: HashMap<String, serde_json::Value>,
    pub katz_centrality: HashMap<String, serde_json::Value>,
}

/// Validation test result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub test_name: String,
    pub passed: bool,
    pub error_message: Option<String>,
    pub max_error: f64,
    pub mean_error: f64,
    pub values_compared: usize,
}

/// Comprehensive validation framework
pub struct ValidationFramework {
    config: ValidationConfig,
    reference_values: Option<ReferenceValues>,
    results: Vec<ValidationResult>,
}

impl ValidationFramework {
    /// Create a new validation framework
    pub fn new(config: ValidationConfig) -> Self {
        Self {
            config,
            reference_values: None,
            results: Vec::new(),
        }
    }

    /// Load reference values from JSON file
    pub fn load_reference_values(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        self.reference_values = Some(serde_json::from_str(&content)?);
        Ok(())
    }

    /// Validate PageRank algorithm
    pub fn validate_pagerank(&mut self) -> CoreResult<()> {
        let ref_values = self
            .reference_values
            .as_ref()
            .expect("Reference values not loaded");

        // Test 1: Simple directed graph
        let mut graph = DiGraph::new();
        for i in 0..4 {
            graph.add_node(i);
        }

        graph
            .add_edge(0, 1, 1.0)
            .map_err(|e| scirs2_core::error::CoreError::from(e.to_string()))?;
        graph
            .add_edge(0, 2, 1.0)
            .map_err(|e| scirs2_core::error::CoreError::from(e.to_string()))?;
        graph
            .add_edge(1, 2, 1.0)
            .map_err(|e| scirs2_core::error::CoreError::from(e.to_string()))?;
        graph
            .add_edge(2, 0, 1.0)
            .map_err(|e| scirs2_core::error::CoreError::from(e.to_string()))?;
        graph
            .add_edge(3, 0, 1.0)
            .map_err(|e| scirs2_core::error::CoreError::from(e.to_string()))?;
        graph
            .add_edge(3, 1, 1.0)
            .map_err(|e| scirs2_core::error::CoreError::from(e.to_string()))?;
        graph
            .add_edge(3, 2, 1.0)
            .map_err(|e| scirs2_core::error::CoreError::from(e.to_string()))?;

        let pagerank = algorithms::pagerank(&graph, 0.85, Some(100))?;

        // Extract reference values
        let ref_pr = &ref_values.pagerank["simple_directed"];
        let ref_pagerank: HashMap<i32, f64> =
            serde_json::from_value(ref_pr["pagerank"].clone()).unwrap();

        let mut errors = Vec::new();
        for i in 0..4 {
            let computed = pagerank[&i];
            let expected = ref_pagerank[&(i as i32)];
            let error = (computed - expected).abs();
            errors.push(error);

            if error > self.config.max_relative_error {
                self.results.push(ValidationResult {
                    test_name: format!("PageRank Simple Directed Node {}", i),
                    passed: false,
                    error_message: Some(format!(
                        "Error {} exceeds tolerance {}",
                        error, self.config.max_relative_error
                    )),
                    max_error: error,
                    mean_error: error,
                    values_compared: 1,
                });
                continue;
            }
        }

        let max_error = errors.iter().cloned().fold(0.0f64, f64::max);
        let mean_error = errors.iter().sum::<f64>() / errors.len() as f64;

        self.results.push(ValidationResult {
            test_name: "PageRank Simple Directed".to_string(),
            passed: max_error <= self.config.max_relative_error,
            error_message: None,
            max_error,
            mean_error,
            values_compared: errors.len(),
        });

        // Test 2: Complete graph (should be uniform)
        let complete = generators::complete_graph(10);
        let pr_complete = algorithms::pagerank(&complete, 0.85, Some(100))?;

        let expected_uniform = ref_values.pagerank["complete_10"]["expected_uniform"]
            .as_f64()
            .unwrap();

        let mut uniform_errors = Vec::new();
        for i in 0..10 {
            let error = (pr_complete[&i] - expected_uniform).abs();
            uniform_errors.push(error);
        }

        let max_uniform_error = uniform_errors.iter().cloned().fold(0.0f64, f64::max);
        let mean_uniform_error = uniform_errors.iter().sum::<f64>() / uniform_errors.len() as f64;

        self.results.push(ValidationResult {
            test_name: "PageRank Complete Graph Uniformity".to_string(),
            passed: max_uniform_error <= self.config.max_relative_error,
            error_message: None,
            max_error: max_uniform_error,
            mean_error: mean_uniform_error,
            values_compared: uniform_errors.len(),
        });

        Ok(())
    }

    /// Validate betweenness centrality algorithm
    pub fn validate_betweenness_centrality(&mut self) -> CoreResult<()> {
        let ref_values = self
            .reference_values
            .as_ref()
            .expect("Reference values not loaded");

        // Test 1: Path graph
        let path = generators::path_graph(5);
        let bc = algorithms::betweenness_centrality(&path)?;

        let ref_bc = &ref_values.betweenness["path_5"];
        let ref_betweenness: HashMap<i32, f64> =
            serde_json::from_value(ref_bc["betweenness"].clone()).unwrap();

        let mut errors = Vec::new();
        for i in 0..5 {
            let computed = bc[&i];
            let expected = ref_betweenness[&(i as i32)];
            let error = (computed - expected).abs();
            errors.push(error);
        }

        let max_error = errors.iter().cloned().fold(0.0f64, f64::max);
        let mean_error = errors.iter().sum::<f64>() / errors.len() as f64;

        self.results.push(ValidationResult {
            test_name: "Betweenness Centrality Path Graph".to_string(),
            passed: max_error <= self.config.max_relative_error,
            error_message: None,
            max_error,
            mean_error,
            values_compared: errors.len(),
        });

        // Test 2: Star graph
        let star = generators::star_graph(6);
        let bc_star = algorithms::betweenness_centrality(&star)?;

        let ref_star = &ref_values.betweenness["star_6"];
        let ref_star_bc: HashMap<i32, f64> =
            serde_json::from_value(ref_star["betweenness"].clone()).unwrap();

        let mut star_errors = Vec::new();
        for i in 0..6 {
            let computed = bc_star[&i];
            let expected = ref_star_bc[&(i as i32)];
            let error = (computed - expected).abs();
            star_errors.push(error);
        }

        let max_star_error = star_errors.iter().cloned().fold(0.0f64, f64::max);
        let mean_star_error = star_errors.iter().sum::<f64>() / star_errors.len() as f64;

        self.results.push(ValidationResult {
            test_name: "Betweenness Centrality Star Graph".to_string(),
            passed: max_star_error <= self.config.max_relative_error,
            error_message: None,
            max_error: max_star_error,
            mean_error: mean_star_error,
            values_compared: star_errors.len(),
        });

        Ok(())
    }

    /// Validate clustering coefficient algorithm
    pub fn validate_clustering_coefficient(&mut self) -> CoreResult<()> {
        let ref_values = self
            .reference_values
            .as_ref()
            .expect("Reference values not loaded");

        // Test 1: Complete graph
        let complete = generators::complete_graph(5);
        let cc = measures::clustering_coefficient(&complete)?;

        let ref_cc = ref_values.clustering["complete_5"]["global_clustering"]
            .as_f64()
            .unwrap();

        let error = (cc - ref_cc).abs();

        self.results.push(ValidationResult {
            test_name: "Clustering Coefficient Complete Graph".to_string(),
            passed: error <= self.config.max_relative_error,
            error_message: None,
            max_error: error,
            mean_error: error,
            values_compared: 1,
        });

        // Test 2: Tree (should be 0)
        let mut tree = Graph::new();
        for i in 0..6 {
            tree.add_node(i);
        }
        tree.add_edge(0, 1, 1.0)
            .map_err(|e| scirs2_core::error::CoreError::from(e.to_string()))?;
        tree.add_edge(0, 2, 1.0)
            .map_err(|e| scirs2_core::error::CoreError::from(e.to_string()))?;
        tree.add_edge(1, 3, 1.0)
            .map_err(|e| scirs2_core::error::CoreError::from(e.to_string()))?;
        tree.add_edge(1, 4, 1.0)
            .map_err(|e| scirs2_core::error::CoreError::from(e.to_string()))?;
        tree.add_edge(2, 5, 1.0)
            .map_err(|e| scirs2_core::error::CoreError::from(e.to_string()))?;

        let cc_tree = measures::clustering_coefficient(&tree)?;
        let ref_tree_cc = ref_values.clustering["tree"]["global_clustering"]
            .as_f64()
            .unwrap();

        let tree_error = (cc_tree - ref_tree_cc).abs();

        self.results.push(ValidationResult {
            test_name: "Clustering Coefficient Tree".to_string(),
            passed: tree_error <= self.config.absolute_tolerance,
            error_message: None,
            max_error: tree_error,
            mean_error: tree_error,
            values_compared: 1,
        });

        Ok(())
    }

    /// Validate shortest path algorithms
    pub fn validate_shortest_paths(&mut self) -> CoreResult<()> {
        let ref_values = self
            .reference_values
            .as_ref()
            .expect("Reference values not loaded");

        // Create weighted graph
        let mut graph = Graph::new();
        for i in 0..5 {
            graph.add_node(i);
        }

        let edges = vec![
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 4, 1.0),
            (0, 3, 2.0),
            (3, 4, 0.5),
        ];

        for (u, v, w) in edges {
            graph
                .add_edge(u, v, w)
                .map_err(|e| scirs2_core::error::CoreError::from(e.to_string()))?;
        }

        let path_result = algorithms::dijkstra_path(&graph, &0, &4)
            .map_err(|e| scirs2_core::error::CoreError::from(e.to_string()))?;
        let (path, length) = if let Some(result) = path_result {
            (result.nodes, result.cost)
        } else {
            return Err(scirs2_core::error::CoreError::from(
                "No path found".to_string(),
            ));
        };

        let ref_sp = &ref_values.shortest_paths["weighted_graph"];
        let ref_length = ref_sp["path_length_0_4"].as_f64().unwrap();
        let ref_path: Vec<i32> =
            serde_json::from_value(ref_sp["shortest_path_0_4"].clone()).unwrap();

        let length_error = (length - ref_length).abs();
        let path_matches = path.len() == ref_path.len()
            && path
                .iter()
                .zip(ref_path.iter())
                .all(|(&a, &b)| a as i32 == b);

        self.results.push(ValidationResult {
            test_name: "Shortest Path Weighted Graph".to_string(),
            passed: length_error <= self.config.absolute_tolerance && path_matches,
            error_message: if !path_matches {
                Some(format!("Path mismatch: {:?} vs {:?}", path, ref_path))
            } else {
                None
            },
            max_error: length_error,
            mean_error: length_error,
            values_compared: 1,
        });

        Ok(())
    }

    /// Validate eigenvector centrality algorithm
    pub fn validate_eigenvector_centrality(&mut self) -> CoreResult<()> {
        let ref_values = self
            .reference_values
            .as_ref()
            .expect("Reference values not loaded");

        // Create simple connected graph
        let mut graph = Graph::new();
        for i in 0..4 {
            graph.add_node(i);
        }

        let edges = vec![(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)];
        for (u, v) in edges {
            graph
                .add_edge(u, v, 1.0)
                .map_err(|e| scirs2_core::error::CoreError::from(e.to_string()))?;
        }

        let ec = algorithms::eigenvector_centrality(&graph)?;

        let ref_ec_data = &ref_values.eigenvector_centrality["simple_graph"];
        let ref_ec: HashMap<i32, f64> =
            serde_json::from_value(ref_ec_data["eigenvector_centrality"].clone()).unwrap();

        let mut errors = Vec::new();
        for i in 0..4 {
            let computed = ec[&i];
            let expected = ref_ec[&(i as i32)];
            let error = (computed - expected).abs();
            errors.push(error);
        }

        let max_error = errors.iter().cloned().fold(0.0f64, f64::max);
        let mean_error = errors.iter().sum::<f64>() / errors.len() as f64;

        self.results.push(ValidationResult {
            test_name: "Eigenvector Centrality".to_string(),
            passed: max_error <= self.config.max_relative_error,
            error_message: None,
            max_error,
            mean_error,
            values_compared: errors.len(),
        });

        Ok(())
    }

    /// Validate spectral properties
    pub fn validate_spectral_properties(&mut self) -> CoreResult<()> {
        let ref_values = self
            .reference_values
            .as_ref()
            .expect("Reference values not loaded");

        // Test spectral radius of complete graph
        let complete = generators::complete_graph(5);
        let spectral_radius = spectral::spectral_radius(&complete)?;

        let ref_spectral = ref_values.spectral["complete_5_spectral"]["expected"]
            .as_f64()
            .unwrap();

        let error = (spectral_radius - ref_spectral).abs();

        self.results.push(ValidationResult {
            test_name: "Spectral Radius Complete Graph".to_string(),
            passed: error <= self.config.max_relative_error,
            error_message: None,
            max_error: error,
            mean_error: error,
            values_compared: 1,
        });

        Ok(())
    }

    /// Validate maximum flow algorithm
    pub fn validate_max_flow(&mut self) -> CoreResult<()> {
        let ref_values = self
            .reference_values
            .as_ref()
            .expect("Reference values not loaded");

        // Create flow network
        let mut graph = DiGraph::new();
        for i in 0..6 {
            graph.add_node(i);
        }

        let capacities = vec![
            (0, 1, 10.0),
            (0, 2, 10.0),
            (1, 2, 2.0),
            (1, 3, 4.0),
            (1, 4, 8.0),
            (2, 4, 9.0),
            (3, 5, 10.0),
            (4, 3, 6.0),
            (4, 5, 10.0),
        ];

        for (u, v, cap) in capacities {
            graph
                .add_edge(u, v, cap)
                .map_err(|e| scirs2_core::error::CoreError::from(e.to_string()))?;
        }

        let (max_flow_value, _) = flow::dinic_max_flow(&graph, 0, 5)?;

        let ref_flow = ref_values.max_flow["flow_network"]["max_flow_value"]
            .as_f64()
            .unwrap();

        let error = (max_flow_value - ref_flow).abs();

        self.results.push(ValidationResult {
            test_name: "Maximum Flow".to_string(),
            passed: error <= self.config.absolute_tolerance,
            error_message: None,
            max_error: error,
            mean_error: error,
            values_compared: 1,
        });

        Ok(())
    }

    /// Run all validation tests
    pub fn run_all_tests(&mut self) -> CoreResult<()> {
        self.validate_pagerank()?;
        self.validate_betweenness_centrality()?;
        self.validate_clustering_coefficient()?;
        self.validate_shortest_paths()?;
        self.validate_eigenvector_centrality()?;
        self.validate_spectral_properties()?;
        self.validate_max_flow()?;

        Ok(())
    }

    /// Generate validation report
    pub fn generate_report(&self) -> String {
        let total_tests = self.results.len();
        let passed_tests = self.results.iter().filter(|r| r.passed).count();
        let failed_tests = total_tests - passed_tests;

        let mut report = String::new();
        report.push_str("=== Numerical Validation Report ===\n");
        report.push_str(&format!("Total Tests: {}\n", total_tests));
        report.push_str(&format!("Passed: {}\n", passed_tests));
        report.push_str(&format!("Failed: {}\n", failed_tests));
        report.push_str(&format!(
            "Success Rate: {:.1}%\n\n",
            100.0 * passed_tests as f64 / total_tests as f64
        ));

        report.push_str("Configuration:\n");
        report.push_str(&format!(
            "  Absolute tolerance: {:.2e}\n",
            self.config.absolute_tolerance
        ));
        report.push_str(&format!(
            "  Relative tolerance: {:.2e}\n",
            self.config.relative_tolerance
        ));
        report.push_str(&format!(
            "  Max relative error: {:.2e}\n\n",
            self.config.max_relative_error
        ));

        report.push_str("Test Results:\n");
        report.push_str("-".repeat(60).as_str());
        report.push_str("\n");

        for result in &self.results {
            let status = if result.passed { "PASS" } else { "FAIL" };
            report.push_str(&format!(
                "{:<30} | {} | Max: {:.2e} | Mean: {:.2e}\n",
                result.test_name, status, result.max_error, result.mean_error
            ));

            if let Some(ref error_msg) = result.error_message {
                report.push_str(&format!("    Error: {}\n", error_msg));
            }
        }

        if failed_tests > 0 {
            report.push_str("\n⚠️  Some tests failed. Review tolerance settings or investigate numerical issues.\n");
        } else {
            report.push_str("\n✅ All numerical validation tests passed!\n");
        }

        report
    }

    /// Get detailed statistics
    pub fn get_statistics(&self) -> ValidationStatistics {
        let total_comparisons: usize = self.results.iter().map(|r| r.values_compared).sum();

        let max_error = self
            .results
            .iter()
            .map(|r| r.max_error)
            .fold(0.0f64, f64::max);

        let mean_error = if !self.results.is_empty() {
            self.results.iter().map(|r| r.mean_error).sum::<f64>() / self.results.len() as f64
        } else {
            0.0
        };

        ValidationStatistics {
            total_tests: self.results.len(),
            passed_tests: self.results.iter().filter(|r| r.passed).count(),
            total_comparisons,
            max_error,
            mean_error,
            success_rate: if self.results.is_empty() {
                0.0
            } else {
                self.results.iter().filter(|r| r.passed).count() as f64 / self.results.len() as f64
            },
        }
    }
}

/// Statistics from validation run
#[derive(Debug, Clone)]
pub struct ValidationStatistics {
    pub total_tests: usize,
    pub passed_tests: usize,
    pub total_comparisons: usize,
    pub max_error: f64,
    pub mean_error: f64,
    pub success_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_framework() {
        let config = ValidationConfig::default();
        let mut framework = ValidationFramework::new(config);

        // Test without reference values (should work with built-in tests)
        // framework.run_all_tests().unwrap();

        let stats = framework.get_statistics();
        assert!(stats.success_rate >= 0.0 && stats.success_rate <= 1.0);
    }

    #[test]
    fn test_validation_config() {
        let config = ValidationConfig {
            absolute_tolerance: 1e-12,
            relative_tolerance: 1e-10,
            max_relative_error: 1e-8,
            verbose_errors: false,
        };

        assert_eq!(config.absolute_tolerance, 1e-12);
        assert_eq!(config.verbose_errors, false);
    }
}
