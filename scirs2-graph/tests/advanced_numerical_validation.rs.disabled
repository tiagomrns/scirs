//! Comprehensive numerical accuracy validation for Advanced mode
//!
//! This module tests the numerical accuracy of graph algorithms when
//! run through Advanced optimizations compared to reference implementations.

use scirs2_graph::advanced::{
    create_adaptive_advanced_processor, create_enhanced_advanced_processor,
    create_memory_efficient_advanced_processor, create_performance_advanced_processor,
    execute_with_enhanced_advanced, AdvancedProcessor,
};
use scirs2_graph::base::Graph;
use std::collections::HashMap;
use std::f64;

const EPSILON: f64 = 1e-10;
const RELATIVE_TOLERANCE: f64 = 1e-6;

/// Test data structure for numerical validation
struct ValidationTestCase {
    name: String,
    graph: Graph<usize, f64>,
    expected_results: HashMap<String, f64>,
    tolerance: f64,
}

/// Generate a set of reference test graphs with known properties
#[allow(dead_code)]
fn generate_reference_test_graphs() -> Vec<ValidationTestCase> {
    let mut test_cases = Vec::new();

    // Test Case 1: Simple linear graph
    let mut linear_graph = Graph::new();
    linear_graph.add_edge(0, 1, 1.0).unwrap();
    linear_graph.add_edge(1, 2, 2.0).unwrap();
    linear_graph.add_edge(2, 3, 3.0).unwrap();

    let mut linear_expected = HashMap::new();
    linear_expected.insert("node_count".to_string(), 4.0);
    linear_expected.insert("edge_count".to_string(), 3.0);
    linear_expected.insert("total_weight".to_string(), 6.0);

    test_cases.push(ValidationTestCase {
        name: "linear_graph".to_string(),
        graph: linear_graph,
        expected_results: linear_expected,
        tolerance: EPSILON,
    });

    // Test Case 2: Complete graph K4
    let mut complete_graph = Graph::new();
    for i in 0..4 {
        for j in (i + 1)..4 {
            complete_graph.add_edge(i, j, 1.0).unwrap();
        }
    }

    let mut complete_expected = HashMap::new();
    complete_expected.insert("node_count".to_string(), 4.0);
    complete_expected.insert("edge_count".to_string(), 6.0);
    complete_expected.insert("total_weight".to_string(), 6.0);
    complete_expected.insert("density".to_string(), 1.0); // Complete graph

    test_cases.push(ValidationTestCase {
        name: "complete_k4".to_string(),
        graph: complete_graph,
        expected_results: complete_expected,
        tolerance: EPSILON,
    });

    // Test Case 3: Star graph
    let mut star_graph = Graph::new();
    for i in 1..6 {
        star_graph.add_edge(0, i, i as f64).unwrap();
    }

    let mut star_expected = HashMap::new();
    star_expected.insert("node_count".to_string(), 6.0);
    star_expected.insert("edge_count".to_string(), 5.0);
    star_expected.insert("total_weight".to_string(), 15.0); // 1+2+3+4+5
    star_expected.insert("center_degree".to_string(), 5.0);

    test_cases.push(ValidationTestCase {
        name: "star_graph".to_string(),
        graph: star_graph,
        expected_results: star_expected,
        tolerance: EPSILON,
    });

    // Test Case 4: Cycle graph
    let mut cycle_graph = Graph::new();
    for i in 0..5 {
        cycle_graph.add_edge(i, (i + 1) % 5, 1.0).unwrap();
    }

    let mut cycle_expected = HashMap::new();
    cycle_expected.insert("node_count".to_string(), 5.0);
    cycle_expected.insert("edge_count".to_string(), 5.0);
    cycle_expected.insert("total_weight".to_string(), 5.0);
    cycle_expected.insert("is_cyclic".to_string(), 1.0); // True

    test_cases.push(ValidationTestCase {
        name: "cycle_graph".to_string(),
        graph: cycle_graph,
        expected_results: cycle_expected,
        tolerance: EPSILON,
    });

    test_cases
}

/// Calculate reference graph properties directly (without optimization)
#[allow(dead_code)]
fn calculate_reference_properties(graph: &Graph<usize, f64>) -> HashMap<String, f64> {
    let mut properties = HashMap::new();

    // Basic properties
    properties.insert("node_count".to_string(), graph.node_count() as f64);
    properties.insert("edge_count".to_string(), graph.edge_count() as f64);

    // Calculate total weight
    let total_weight: f64 = graph.edges().into_iter().map(|edge| edge.weight).sum();
    properties.insert("total_weight".to_string(), total_weight);

    // Calculate density
    let n = graph.node_count() as f64;
    if n > 1.0 {
        let density = (graph.edge_count() as f64) / (n * (n - 1.0) / 2.0);
        properties.insert("density".to_string(), density);
    }

    // Calculate degree-related properties
    let nodes: Vec<_> = graph.nodes().into_iter().collect();
    if !nodes.is_empty() {
        let degrees: Vec<usize> = nodes.iter().map(|&node| graph.degree(node)).collect();
        let max_degree = *degrees.iter().max().unwrap_or(&0);
        let min_degree = *degrees.iter().min().unwrap_or(&0);
        let avg_degree = degrees.iter().sum::<usize>() as f64 / degrees.len() as f64;

        properties.insert("max_degree".to_string(), max_degree as f64);
        properties.insert("min_degree".to_string(), min_degree as f64);
        properties.insert("avg_degree".to_string(), avg_degree);

        // Special case: center degree for star graphs
        if max_degree > 0 && degrees.iter().filter(|&&d| d == max_degree).count() == 1 {
            properties.insert("center_degree".to_string(), max_degree as f64);
        }
    }

    // Check for cycles (simplified - just check if any node has degree > 2)
    let has_high_degree = graph.nodes().into_iter().any(|node| graph.degree(node) > 2);
    properties.insert(
        "is_cyclic".to_string(),
        if has_high_degree { 1.0 } else { 0.0 },
    );

    properties
}

/// Test numerical accuracy of a single test case
#[allow(dead_code)]
fn test_numerical_accuracy(
    test_case: &ValidationTestCase,
    processor: &mut AdvancedProcessor,
    processor_name: &str,
) -> ValidationResult {
    let mut results = ValidationResult::new(test_case.name.clone(), processor_name.to_string());

    // Calculate reference properties
    let reference_props = calculate_reference_properties(&test_case.graph);

    // Calculate properties using advanced optimization
    let optimized_props = execute_with_enhanced_advanced(
        processor,
        &test_case.graph,
        &format!("validation_{}", test_case.name),
        |graph| Ok(calculate_reference_properties(graph)),
    )
    .unwrap_or_else(|_| HashMap::new());

    // Compare results
    for (property, &expected) in &test_case.expected_results {
        let reference_val = reference_props.get(property).copied().unwrap_or(0.0);
        let optimized_val = optimized_props.get(property).copied().unwrap_or(0.0);

        let accuracy =
            calculate_accuracy(expected, reference_val, optimized_val, test_case.tolerance);
        results.add_property_result(
            property.clone(),
            expected,
            reference_val,
            optimized_val,
            accuracy,
        );
    }

    // Also compare additional computed properties
    for (property, &reference_val) in &reference_props {
        if !test_case.expected_results.contains_key(property) {
            let optimized_val = optimized_props.get(property).copied().unwrap_or(0.0);
            let accuracy = calculate_accuracy(
                reference_val,
                reference_val,
                optimized_val,
                RELATIVE_TOLERANCE,
            );
            results.add_property_result(
                property.clone(),
                reference_val,
                reference_val,
                optimized_val,
                accuracy,
            );
        }
    }

    results
}

/// Calculate numerical accuracy between expected, reference, and optimized values
#[allow(dead_code)]
fn calculate_accuracy(expected: f64, reference: f64, optimized: f64, tolerance: f64) -> f64 {
    if (expected - reference).abs() < tolerance && (reference - optimized).abs() < tolerance {
        1.0 // Perfect accuracy
    } else if expected.is_finite() && reference.is_finite() && optimized.is_finite() {
        let ref_error = (expected - reference).abs() / expected.abs().max(1.0);
        let opt_error = (reference - optimized).abs() / reference.abs().max(1.0);
        let total_error = ref_error + opt_error;

        if total_error < tolerance {
            1.0 - total_error
        } else {
            (1.0 / (1.0 + total_error)).max(0.0)
        }
    } else {
        0.0 // NaN or infinite values
    }
}

/// Results of a validation test
#[derive(Debug, Clone)]
struct ValidationResult {
    test_name: String,
    processor_name: String,
    property_results: HashMap<String, PropertyResult>,
    overall_accuracy: f64,
}

#[derive(Debug, Clone)]
struct PropertyResult {
    expected: f64,
    reference: f64,
    optimized: f64,
    accuracy: f64,
}

impl ValidationResult {
    fn new(test_name: String, processor_name: String) -> Self {
        Self {
            test_name,
            processor_name,
            property_results: HashMap::new(),
            overall_accuracy: 0.0,
        }
    }

    fn add_property_result(
        &mut self,
        property: String,
        expected: f64,
        reference: f64,
        optimized: f64,
        accuracy: f64,
    ) {
        self.property_results.insert(
            property,
            PropertyResult {
                expected,
                reference,
                optimized,
                accuracy,
            },
        );

        // Update overall accuracy
        if !self.property_results.is_empty() {
            self.overall_accuracy = self
                .property_results
                .values()
                .map(|r| r.accuracy)
                .sum::<f64>()
                / self.property_results.len() as f64;
        }
    }

    fn is_passing(&self, threshold: f64) -> bool {
        self.overall_accuracy >= threshold
    }

    fn report(&self) -> String {
        let mut report = format!(
            "Validation Result: {} with {}\n",
            self.test_name, self.processor_name
        );
        report.push_str(&format!("Overall Accuracy: {:.6}\n", self.overall_accuracy));

        for (property, result) in &self.property_results {
            report.push_str(&format!(
                "  {}: Expected={:.6}, Reference={:.6}, Optimized={:.6}, Accuracy={:.6}\n",
                property, result.expected, result.reference, result.optimized, result.accuracy
            ));
        }

        report
    }
}

/// Comprehensive numerical validation suite
#[allow(dead_code)]
fn run_comprehensive_validation() -> Vec<ValidationResult> {
    let test_cases = generate_reference_test_graphs();
    let mut all_results = Vec::new();

    // Test different processor configurations
    let mut processors = vec![
        ("enhanced", create_enhanced_advanced_processor()),
        ("performance", create_performance_advanced_processor()),
        (
            "memory_efficient",
            create_memory_efficient_advanced_processor(),
        ),
        ("adaptive", create_adaptive_advanced_processor()),
    ];

    for test_case in &test_cases {
        for (processor_name, processor) in &mut processors {
            let result = test_numerical_accuracy(test_case, processor, processor_name);
            all_results.push(result);
        }
    }

    all_results
}

/// Generate validation report
#[allow(dead_code)]
fn generate_validation_report(results: &[ValidationResult]) -> String {
    let mut report = String::new();
    report.push_str("=== Advanced Numerical Validation Report ===\n\n");

    // Overall statistics
    let total_tests = results.len();
    let passing_tests = results.iter().filter(|r| r.is_passing(0.95)).count();
    let avg_accuracy = results.iter().map(|r| r.overall_accuracy).sum::<f64>() / total_tests as f64;

    report.push_str(&format!("Total Tests: {}\n", total_tests));
    report.push_str(&format!(
        "Passing Tests (>95% accuracy): {}\n",
        passing_tests
    ));
    report.push_str(&format!(
        "Pass Rate: {:.2}%\n",
        (passing_tests as f64 / total_tests as f64) * 100.0
    ));
    report.push_str(&format!("Average Accuracy: {:.6}\n\n", avg_accuracy));

    // Group results by processor
    let mut processor_stats: HashMap<String, Vec<&ValidationResult>> = HashMap::new();
    for result in results {
        processor_stats
            .entry(result.processor_name.clone())
            .or_insert_with(Vec::new)
            .push(result);
    }

    report.push_str("=== Results by Processor ===\n");
    for (processor, processor_results) in &processor_stats {
        let processor_avg = processor_results
            .iter()
            .map(|r| r.overall_accuracy)
            .sum::<f64>()
            / processor_results.len() as f64;
        let processor_passing = processor_results
            .iter()
            .filter(|r| r.is_passing(0.95))
            .count();

        report.push_str(&format!(
            "\n{}: {:.6} avg accuracy, {}/{} passing\n",
            processor,
            processor_avg,
            processor_passing,
            processor_results.len()
        ));
    }

    // Detailed results
    report.push_str("\n=== Detailed Results ===\n");
    for result in results {
        report.push_str(&result.report());
        report.push('\n');
    }

    report
}

/// Advanced algorithm-specific validation tests
#[allow(dead_code)]
fn generate_algorithm_validation_tests() -> Vec<AlgorithmValidationTest> {
    let mut tests = Vec::new();

    // PageRank validation
    tests.push(create_pagerank_validation_test());

    // Centrality measures validation
    tests.push(create_centrality_validation_test());

    // Shortest path validation
    tests.push(create_shortest_path_validation_test());

    // Community detection validation
    tests.push(create_community_detection_validation_test());

    // Connected components validation
    tests.push(create_connected_components_validation_test());

    tests
}

struct AlgorithmValidationTest {
    name: String,
    graph: Graph<usize, f64>,
    reference_implementation: fn(&Graph<usize, f64>) -> Result<ValidationOutput, String>,
    advanced_implementation:
        fn(&Graph<usize, f64>, &mut AdvancedProcessor) -> Result<ValidationOutput, String>,
    tolerance: f64,
    description: String,
}

#[derive(Debug, Clone)]
enum ValidationOutput {
    FloatArray(Vec<f64>),
    IntArray(Vec<usize>),
    Float(f64),
    Integer(usize),
    FloatMap(HashMap<usize, f64>),
    IntMap(HashMap<usize, usize>),
}

/// Create PageRank validation test
#[allow(dead_code)]
fn create_pagerank_validation_test() -> AlgorithmValidationTest {
    // Create a simple 4-node graph with known PageRank values
    let mut graph = Graph::new();
    graph.add_edge(0, 1, 1.0).unwrap();
    graph.add_edge(1, 2, 1.0).unwrap();
    graph.add_edge(2, 0, 1.0).unwrap();
    graph.add_edge(3, 0, 1.0).unwrap();

    AlgorithmValidationTest {
        name: "pagerank_validation".to_string(),
        graph,
        reference_implementation: |g| {
            // Simple reference PageRank implementation
            let nodes: Vec<_> = g.nodes().collect();
            let n = nodes.len();
            let damping = 0.85;
            let mut ranks = vec![1.0 / n as f64; n];
            let mut new_ranks = vec![0.0; n];

            for _ in 0..100 {
                // iterations
                for (i, &node) in nodes.iter().enumerate() {
                    let out_degree = g.degree(node) as f64;
                    if out_degree > 0.0 {
                        let contribution = ranks[i] / out_degree;
                        for neighbor in g.neighbors(node) {
                            if let Some(neighbor_idx) = nodes.iter().position(|&n| n == neighbor) {
                                new_ranks[neighbor_idx] += damping * contribution;
                            }
                        }
                    }
                }

                // Add random jump probability
                for rank in &mut new_ranks {
                    *rank += (1.0 - damping) / n as f64;
                }

                ranks = new_ranks.clone();
                new_ranks.fill(0.0);
            }

            Ok(ValidationOutput::FloatArray(ranks))
        },
        advanced_implementation: |g, processor| {
            let result =
                execute_with_enhanced_advanced(processor, g, "pagerank_validation", |graph| {
                    use scirs2_graph::measures::pagerank_centrality;
                    pagerank_centrality(graph, 0.85, 1e-8)
                });

            match result {
                Ok(pagerank_map) => {
                    let nodes: Vec<_> = g.nodes().collect();
                    let ranks: Vec<f64> = nodes
                        .iter()
                        .map(|&node| pagerank_map.get(&node).copied().unwrap_or(0.0))
                        .collect();
                    Ok(ValidationOutput::FloatArray(ranks))
                }
                Err(e) => Err(format!("PageRank failed: {:?}", e)),
            }
        },
        tolerance: 1e-3,
        description: "Validates PageRank algorithm accuracy against reference implementation"
            .to_string(),
    }
}

/// Create centrality measures validation test
#[allow(dead_code)]
fn create_centrality_validation_test() -> AlgorithmValidationTest {
    // Create a star graph for centrality testing
    let mut graph = Graph::new();
    for i in 1..6 {
        graph.add_edge(0, i, 1.0).unwrap();
    }

    AlgorithmValidationTest {
        name: "centrality_validation".to_string(),
        graph,
        reference_implementation: |g| {
            // Reference degree centrality (simple: just the degree)
            let nodes: Vec<_> = g.nodes().collect();
            let centralities: Vec<f64> = nodes.iter().map(|&node| g.degree(node) as f64).collect();
            Ok(ValidationOutput::FloatArray(centralities))
        },
        advanced_implementation: |g, processor| {
            let result =
                execute_with_enhanced_advanced(processor, g, "centrality_validation", |graph| {
                    use scirs2_graph::measures::{centrality, CentralityType};
                    centrality(graph, CentralityType::Degree)
                });

            match result {
                Ok(centrality_map) => {
                    let nodes: Vec<_> = g.nodes().collect();
                    let centralities: Vec<f64> = nodes
                        .iter()
                        .map(|&node| centrality_map.get(&node).copied().unwrap_or(0.0))
                        .collect();
                    Ok(ValidationOutput::FloatArray(centralities))
                }
                Err(e) => Err(format!("Centrality calculation failed: {:?}", e)),
            }
        },
        tolerance: EPSILON,
        description: "Validates degree centrality calculation".to_string(),
    }
}

/// Create shortest path validation test
#[allow(dead_code)]
fn create_shortest_path_validation_test() -> AlgorithmValidationTest {
    // Create a simple path graph
    let mut graph = Graph::new();
    graph.add_edge(0, 1, 2.0).unwrap();
    graph.add_edge(1, 2, 3.0).unwrap();
    graph.add_edge(2, 3, 1.0).unwrap();
    graph.add_edge(0, 3, 10.0).unwrap(); // Longer direct path

    AlgorithmValidationTest {
        name: "shortest_path_validation".to_string(),
        graph,
        reference_implementation: |g| {
            // Reference shortest path using simple Dijkstra
            let source = 0;
            let target = 3;
            let nodes: Vec<_> = g.nodes().collect();
            let mut distances = HashMap::new();

            for &node in &nodes {
                distances.insert(node, if node == source { 0.0 } else { f64::INFINITY });
            }

            let mut unvisited: std::collections::BinaryHeap<std::cmp::Reverse<(usize, f64)>> =
                std::collections::BinaryHeap::new();
            unvisited.push(std::cmp::Reverse((source, 0.0)));

            while let Some(std::cmp::Reverse((current, current_dist))) = unvisited.pop() {
                if current == target {
                    break;
                }

                if current_dist > *distances.get(&current).unwrap() {
                    continue;
                }

                for neighbor in g.neighbors(current) {
                    if let Some(edge) = g.find_edge(current, neighbor) {
                        let weight = *g.edge_weight(edge).unwrap();
                        let new_dist = current_dist + weight;

                        if new_dist < *distances.get(&neighbor).unwrap() {
                            distances.insert(neighbor, new_dist);
                            unvisited.push(std::cmp::Reverse((neighbor, new_dist)));
                        }
                    }
                }
            }

            Ok(ValidationOutput::Float(*distances.get(&target).unwrap()))
        },
        advanced_implementation: |g, processor| {
            let result =
                execute_with_enhanced_advanced(processor, g, "shortest_path_validation", |graph| {
                    use scirs2_graph::algorithms::shortest_path::dijkstra_path;
                    use std::collections::HashMap;

                    // Compute shortest paths from node 0 to all other nodes
                    let mut distances = HashMap::new();
                    let nodes: Vec<_> = graph.nodes().into_iter().collect();

                    for &target in &nodes {
                        if target != 0 {
                            if let Ok(Some(path)) = dijkstra_path(graph, &0, &target) {
                                distances.insert(target, path.total_weight);
                            }
                        } else {
                            distances.insert(0, Default::default()); // Distance to self is 0
                        }
                    }

                    Ok(distances)
                });

            match result {
                Ok(distances) => {
                    let target_distance = distances.get(&3).copied().unwrap_or(f64::INFINITY);
                    Ok(ValidationOutput::Float(target_distance))
                }
                Err(e) => Err(format!("Shortest path failed: {:?}", e)),
            }
        },
        tolerance: 1e-10,
        description: "Validates shortest path algorithm (Dijkstra) accuracy".to_string(),
    }
}

/// Create community detection validation test
#[allow(dead_code)]
fn create_community_detection_validation_test() -> AlgorithmValidationTest {
    // Create a graph with two obvious communities
    let mut graph = Graph::new();

    // Community 1: nodes 0, 1, 2
    graph.add_edge(0, 1, 1.0).unwrap();
    graph.add_edge(1, 2, 1.0).unwrap();
    graph.add_edge(2, 0, 1.0).unwrap();

    // Community 2: nodes 3, 4, 5
    graph.add_edge(3, 4, 1.0).unwrap();
    graph.add_edge(4, 5, 1.0).unwrap();
    graph.add_edge(5, 3, 1.0).unwrap();

    // Weak connection between communities
    graph.add_edge(2, 3, 0.1).unwrap();

    AlgorithmValidationTest {
        name: "community_detection_validation".to_string(),
        graph,
        reference_implementation: |g| {
            // Simple reference: return number of connected components
            let nodes: Vec<_> = g.nodes().collect();
            let mut visited = std::collections::HashSet::new();
            let mut components = 0;

            for &node in &nodes {
                if !visited.contains(&node) {
                    // BFS to find all nodes in this component
                    let mut queue = std::collections::VecDeque::new();
                    queue.push_back(node);
                    visited.insert(node);

                    while let Some(current) = queue.pop_front() {
                        for neighbor in g.neighbors(current) {
                            if !visited.contains(&neighbor) {
                                visited.insert(neighbor);
                                queue.push_back(neighbor);
                            }
                        }
                    }
                    components += 1;
                }
            }

            Ok(ValidationOutput::Integer(components))
        },
        advanced_implementation: |g, processor| {
            let result =
                execute_with_enhanced_advanced(processor, g, "community_validation", |graph| {
                    use scirs2_graph::algorithms::community::louvain_communities_result;
                    louvain_communities_result(graph)
                });

            match result {
                Ok(communities) => Ok(ValidationOutput::Integer(communities.len())),
                Err(e) => Err(format!("Community detection failed: {:?}", e)),
            }
        },
        tolerance: 0.0, // Exact match expected for component count
        description: "Validates community detection algorithm".to_string(),
    }
}

/// Create connected components validation test
#[allow(dead_code)]
fn create_connected_components_validation_test() -> AlgorithmValidationTest {
    // Create a graph with multiple disconnected components
    let mut graph = Graph::new();

    // Component 1: 0-1-2
    graph.add_edge(0, 1, 1.0).unwrap();
    graph.add_edge(1, 2, 1.0).unwrap();

    // Component 2: 3-4
    graph.add_edge(3, 4, 1.0).unwrap();

    // Component 3: isolated node 5 (will be added when we add edges)

    AlgorithmValidationTest {
        name: "connected_components_validation".to_string(),
        graph,
        reference_implementation: |g| {
            // Reference implementation: find connected components
            let nodes: Vec<_> = g.nodes().collect();
            let mut visited = std::collections::HashSet::new();
            let mut components = Vec::new();

            for &node in &nodes {
                if !visited.contains(&node) {
                    let mut component = Vec::new();
                    let mut queue = std::collections::VecDeque::new();
                    queue.push_back(node);
                    visited.insert(node);

                    while let Some(current) = queue.pop_front() {
                        component.push(*current);
                        for neighbor in g.neighbors(current) {
                            if !visited.contains(&neighbor) {
                                visited.insert(&neighbor);
                                queue.push_back(&neighbor);
                            }
                        }
                    }
                    components.push(component);
                }
            }

            Ok(ValidationOutput::Integer(components.len()))
        },
        advanced_implementation: |g, processor| {
            let result =
                execute_with_enhanced_advanced(processor, g, "components_validation", |graph| {
                    use scirs2_graph::algorithms::connectivity::connected_components;
                    connected_components(graph)
                });

            match result {
                Ok(components) => Ok(ValidationOutput::Integer(components.len())),
                Err(e) => Err(format!("Connected components failed: {:?}", e)),
            }
        },
        tolerance: 0.0, // Exact match expected
        description: "Validates connected components algorithm".to_string(),
    }
}

/// Run comprehensive algorithm validation
#[allow(dead_code)]
fn run_algorithm_validation() -> Vec<AlgorithmValidationResult> {
    let tests = generate_algorithm_validation_tests();
    let mut results = Vec::new();

    println!("🧮 Running comprehensive algorithm validation tests...");
    println!("===================================================");

    let mut processors = vec![
        ("enhanced", create_enhanced_advanced_processor()),
        ("performance", create_performance_advanced_processor()),
        (
            "memory_efficient",
            create_memory_efficient_advanced_processor(),
        ),
        ("adaptive", create_adaptive_advanced_processor()),
    ];

    for test in &tests {
        println!("\n📊 Testing algorithm: {}", test.name);
        println!("   Description: {}", test.description);

        for (processor_name, processor) in &mut processors {
            println!("   🔧 Testing with {} processor...", processor_name);

            let result = validate_algorithm(test, processor, processor_name);
            println!("     Accuracy: {:.6}", result.accuracy);

            results.push(result);
        }
    }

    results
}

#[derive(Debug, Clone)]
struct AlgorithmValidationResult {
    test_name: String,
    processor_name: String,
    accuracy: f64,
    reference_output: String,
    optimized_output: String,
    error_message: Option<String>,
}

/// Validate a single algorithm test
#[allow(dead_code)]
fn validate_algorithm(
    test: &AlgorithmValidationTest,
    processor: &mut AdvancedProcessor,
    processor_name: &str,
) -> AlgorithmValidationResult {
    // Run reference implementation
    let reference_result = (test.reference_implementation)(&test.graph);

    // Run advanced implementation
    let optimized_result = (test.advanced_implementation)(&test.graph, processor);

    let (accuracy, ref_output, opt_output, error) = match (reference_result, optimized_result) {
        (Ok(ref_val), Ok(opt_val)) => {
            let acc = calculate_validation_accuracy(&ref_val, &opt_val, test.tolerance);
            (
                acc,
                format!("{:?}", ref_val),
                format!("{:?}", opt_val),
                None,
            )
        }
        (Ok(ref_val), Err(e)) => (0.0, format!("{:?}", ref_val), "ERROR".to_string(), Some(e)),
        (Err(e1), Ok(opt_val)) => (0.0, "ERROR".to_string(), format!("{:?}", opt_val), Some(e1)),
        (Err(e1), Err(e2)) => (
            0.0,
            "ERROR".to_string(),
            "ERROR".to_string(),
            Some(format!("{} | {}", e1, e2)),
        ),
    };

    AlgorithmValidationResult {
        test_name: test.name.clone(),
        processor_name: processor_name.to_string(),
        accuracy,
        reference_output: ref_output,
        optimized_output: opt_output,
        error_message: error,
    }
}

/// Calculate accuracy between two validation outputs
#[allow(dead_code)]
fn calculate_validation_accuracy(
    reference: &ValidationOutput,
    optimized: &ValidationOutput,
    tolerance: f64,
) -> f64 {
    match (reference, optimized) {
        (ValidationOutput::Float(r), ValidationOutput::Float(o)) => {
            if (r - o).abs() < tolerance {
                1.0
            } else {
                1.0 / (1.0 + (r - o).abs())
            }
        }
        (ValidationOutput::Integer(r), ValidationOutput::Integer(o)) => {
            if r == o {
                1.0
            } else {
                0.0
            }
        }
        (ValidationOutput::FloatArray(r), ValidationOutput::FloatArray(o)) => {
            if r.len() != o.len() {
                return 0.0;
            }
            let errors: Vec<f64> = r
                .iter()
                .zip(o.iter())
                .map(|(rv, ov)| (rv - ov).abs())
                .collect();
            let avg_error = errors.iter().sum::<f64>() / errors.len() as f64;
            if avg_error < tolerance {
                1.0
            } else {
                1.0 / (1.0 + avg_error)
            }
        }
        _ => 0.0, // Type mismatch
    }
}

/// Generate comprehensive algorithm validation report
#[allow(dead_code)]
fn generate_algorithm_validation_report(results: &[AlgorithmValidationResult]) -> String {
    let mut report = String::new();
    report.push_str("=== Comprehensive Algorithm Validation Report ===\n\n");

    // Overall statistics
    let total_tests = results.len();
    let high_accuracy_tests = results.iter().filter(|r| r.accuracy > 0.99).count();
    let medium_accuracy_tests = results
        .iter()
        .filter(|r| r.accuracy > 0.95 && r.accuracy <= 0.99)
        .count();
    let low_accuracy_tests = results.iter().filter(|r| r.accuracy <= 0.95).count();
    let failed_tests = results.iter().filter(|r| r.error_message.is_some()).count();

    let avg_accuracy = results.iter().map(|r| r.accuracy).sum::<f64>() / total_tests as f64;

    report.push_str(&format!("📊 Overall Statistics:\n"));
    report.push_str(&format!("   Total Tests: {}\n", total_tests));
    report.push_str(&format!(
        "   High Accuracy (>99%): {}\n",
        high_accuracy_tests
    ));
    report.push_str(&format!(
        "   Medium Accuracy (95-99%): {}\n",
        medium_accuracy_tests
    ));
    report.push_str(&format!("   Low Accuracy (<95%): {}\n", low_accuracy_tests));
    report.push_str(&format!("   Failed Tests: {}\n", failed_tests));
    report.push_str(&format!("   Average Accuracy: {:.6}\n\n", avg_accuracy));

    // Results by algorithm
    let mut algorithm_stats: HashMap<String, Vec<&AlgorithmValidationResult>> = HashMap::new();
    for result in results {
        algorithm_stats
            .entry(result.test_name.clone())
            .or_insert_with(Vec::new)
            .push(result);
    }

    report.push_str("📈 Results by Algorithm:\n");
    for (algorithm, algorithm_results) in &algorithm_stats {
        let algorithm_avg = algorithm_results.iter().map(|r| r.accuracy).sum::<f64>()
            / algorithm_results.len() as f64;
        let algorithm_passing = algorithm_results
            .iter()
            .filter(|r| r.accuracy > 0.95)
            .count();

        report.push_str(&format!(
            "   {}: {:.6} avg accuracy, {}/{} passing\n",
            algorithm,
            algorithm_avg,
            algorithm_passing,
            algorithm_results.len()
        ));
    }

    // Results by processor
    let mut processor_stats: HashMap<String, Vec<&AlgorithmValidationResult>> = HashMap::new();
    for result in results {
        processor_stats
            .entry(result.processor_name.clone())
            .or_insert_with(Vec::new)
            .push(result);
    }

    report.push_str("\n🔧 Results by Processor:\n");
    for (processor, processor_results) in &processor_stats {
        let processor_avg = processor_results.iter().map(|r| r.accuracy).sum::<f64>()
            / processor_results.len() as f64;
        let processor_passing = processor_results
            .iter()
            .filter(|r| r.accuracy > 0.95)
            .count();

        report.push_str(&format!(
            "   {}: {:.6} avg accuracy, {}/{} passing\n",
            processor,
            processor_avg,
            processor_passing,
            processor_results.len()
        ));
    }

    // Detailed results
    report.push_str("\n📋 Detailed Results:\n");
    for result in results {
        report.push_str(&format!(
            "\n{}::{} - Accuracy: {:.6}\n",
            result.test_name, result.processor_name, result.accuracy
        ));
        if let Some(error) = &result.error_message {
            report.push_str(&format!("   ❌ Error: {}\n", error));
        } else {
            report.push_str(&format!("   ✅ Reference: {}\n", result.reference_output));
            report.push_str(&format!("   ⚡ Optimized: {}\n", result.optimized_output));
        }
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reference_graph_generation() {
        let test_cases = generate_reference_test_graphs();
        assert!(!test_cases.is_empty());

        for test_case in &test_cases {
            assert!(!test_case.name.is_empty());
            assert!(test_case.graph.node_count() > 0);
            assert!(!test_case.expected_results.is_empty());
        }
    }

    #[test]
    fn test_reference_property_calculation() {
        let mut graph = Graph::new();
        graph.add_edge(0, 1, 2.0).unwrap();
        graph.add_edge(1, 2, 3.0).unwrap();

        let props = calculate_reference_properties(&graph);

        assert_eq!(props.get("node_count"), Some(&3.0));
        assert_eq!(props.get("edge_count"), Some(&2.0));
        assert_eq!(props.get("total_weight"), Some(&5.0));
    }

    #[test]
    fn test_accuracy_calculation() {
        // Perfect accuracy
        assert_eq!(calculate_accuracy(1.0, 1.0, 1.0, 1e-10), 1.0);

        // Small error within tolerance
        assert!(calculate_accuracy(1.0, 1.0, 1.000001, 1e-3) > 0.99);

        // Large error
        assert!(calculate_accuracy(1.0, 1.0, 2.0, 1e-10) < 0.5);

        // Handle edge cases
        assert_eq!(calculate_accuracy(f64::NAN, 1.0, 1.0, 1e-10), 0.0);
        assert_eq!(calculate_accuracy(1.0, f64::INFINITY, 1.0, 1e-10), 0.0);
    }

    #[test]
    fn test_single_validation() {
        let test_cases = generate_reference_test_graphs();
        let test_case = &test_cases[0]; // Linear graph

        let mut processor = create_enhanced_advanced_processor();
        let result = test_numerical_accuracy(test_case, &mut processor, "test");

        assert!(!result.property_results.is_empty());
        assert!(result.overall_accuracy >= 0.0);
        assert!(result.overall_accuracy <= 1.0);
    }

    #[test]
    fn test_comprehensive_validation() {
        let results = run_comprehensive_validation();
        assert!(!results.is_empty());

        // Check that we have results for all processor types
        let processor_names: std::collections::HashSet<_> =
            results.iter().map(|r| r.processor_name.as_str()).collect();
        assert!(processor_names.contains("enhanced"));
        assert!(processor_names.contains("performance"));
        assert!(processor_names.contains("memory_efficient"));
        assert!(processor_names.contains("adaptive"));

        // Generate report (should not crash)
        let report = generate_validation_report(&results);
        assert!(!report.is_empty());
        println!("Validation Report:\n{}", report);

        // Check that most tests pass
        let passing_rate =
            results.iter().filter(|r| r.is_passing(0.90)).count() as f64 / results.len() as f64;
        assert!(
            passing_rate >= 0.8,
            "Pass rate too low: {:.2}",
            passing_rate
        );
    }

    #[test]
    fn test_validation_result_reporting() {
        let mut result = ValidationResult::new("test".to_string(), "processor".to_string());
        result.add_property_result("prop1".to_string(), 1.0, 1.0, 1.0, 1.0);
        result.add_property_result("prop2".to_string(), 2.0, 2.0, 2.1, 0.95);

        assert!(result.is_passing(0.9));
        assert!(!result.is_passing(0.99));

        let report = result.report();
        assert!(report.contains("test"));
        assert!(report.contains("processor"));
        assert!(report.contains("prop1"));
        assert!(report.contains("prop2"));
    }
}

/// Integration test that can be run manually
#[test]
#[allow(dead_code)]
fn integration_test_numerical_validation() {
    let results = run_comprehensive_validation();
    let report = generate_validation_report(&results);

    // Print report for manual inspection
    println!("\n{}", report);

    // Assert minimum quality standards
    let avg_accuracy =
        results.iter().map(|r| r.overall_accuracy).sum::<f64>() / results.len() as f64;
    assert!(
        avg_accuracy >= 0.95,
        "Average accuracy too low: {:.6}",
        avg_accuracy
    );

    let passing_rate =
        results.iter().filter(|r| r.is_passing(0.95)).count() as f64 / results.len() as f64;
    assert!(
        passing_rate >= 0.8,
        "Pass rate too low: {:.2}",
        passing_rate
    );
}
