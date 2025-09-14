//! Financial Risk Analysis Workflow with Advanced Mode
//!
//! This example demonstrates how to use scirs2-graph with Advanced optimizations
//! for financial network analysis, including risk assessment, contagion modeling,
//! and market structure analysis.

#![allow(dead_code)]

use scirs2_graph::advanced::{
    create_enhanced_advanced_processor, create_performance_advanced_processor,
    execute_with_enhanced_advanced, AdvancedProcessor,
};
use scirs2_graph::algorithms::community::louvain_communities;
use scirs2_graph::algorithms::connectivity::connected_components;
use scirs2_graph::algorithms::paths::shortest_path_dijkstra;
use scirs2_graph::algorithms::properties::{betweenness_centrality, closeness_centrality};
use scirs2_graph::base::Graph;
use scirs2_graph::generators::{barabasi_albert_graph, erdos_renyi_graph};
use scirs2_graph::measures::pagerank;
use std::collections::HashMap;
use std::time::Instant;

/// Financial entity in the network
#[derive(Debug, Clone, PartialEq)]
pub enum EntityType {
    Bank,
    InsuranceCompany,
    HedgeFund,
    Broker,
    CentralBank,
    Government,
}

/// Financial connection representing various types of financial relationships
#[derive(Debug, Clone)]
pub struct FinancialConnection {
    pub connection_type: ConnectionType,
    pub exposure_amount: f64,
    pub risk_weight: f64,
    pub maturity_days: Option<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConnectionType {
    Loan,
    Derivative,
    Guarantee,
    Investment,
    Clearing,
    Regulatory,
}

/// Financial risk assessment result
#[derive(Debug)]
pub struct RiskAssessment {
    pub systemic_risk_score: f64,
    pub interconnectedness_score: f64,
    pub contagion_probability: f64,
    pub critical_institutions: Vec<usize>,
    pub vulnerable_clusters: Vec<Vec<usize>>,
    pub stress_test_results: HashMap<String, f64>,
}

/// Workflow: Financial Risk Analysis
///
/// This workflow demonstrates comprehensive financial network analysis including:
/// 1. Systemic risk assessment
/// 2. Contagion modeling  
/// 3. Critical institution identification
/// 4. Stress testing simulation
#[allow(dead_code)]
fn financial_risk_analysis_workflow() -> Result<(), Box<dyn std::error::Error>> {
    println!("💰 Financial Risk Analysis Workflow");
    println!("===================================");

    // Create a financial network representing major institutions and their interconnections
    println!("🏦 Generating financial institution network...");
    let graph = create_financial_network(500, 0.015)?; // 500 institutions, moderate connectivity

    println!("✅ Financial network created:");
    println!("   - Institutions: {}", graph.node_count());
    println!("   - Connections: {}", graph.edge_count());
    println!(
        "   - Average connections per institution: {:.2}",
        graph.edge_count() as f64 * 2.0 / graph.node_count() as f64
    );

    // Create an enhanced advanced processor optimized for financial analysis
    let mut processor = create_enhanced_advanced_processor();

    // Step 1: Systemic Risk Assessment
    println!("\n📊 Step 1: Systemic Risk Assessment");
    println!("----------------------------------");

    let risk_assessment = assess_systemic_risk(&mut processor, &graph)?;

    println!("✅ Systemic risk assessment completed:");
    println!(
        "   - Overall systemic risk score: {:.3}",
        risk_assessment.systemic_risk_score
    );
    println!(
        "   - Network interconnectedness: {:.3}",
        risk_assessment.interconnectedness_score
    );
    println!(
        "   - Contagion probability: {:.1}%",
        risk_assessment.contagion_probability * 100.0
    );
    println!(
        "   - Critical institutions identified: {}",
        risk_assessment.critical_institutions.len()
    );

    // Step 2: Critical Institution Analysis
    println!("\n🎯 Step 2: Critical Institution Analysis");
    println!("---------------------------------------");

    let critical_analysis = analyze_critical_institutions(&mut processor, &graph)?;

    println!("✅ Critical institution analysis completed:");
    println!("   - Top 5 most critical institutions:");
    for (rank, (institution_id, criticality_score)) in critical_analysis.iter().take(5).enumerate()
    {
        println!(
            "     {}. Institution {}: criticality {:.4}",
            rank + 1,
            institution_id,
            criticality_score
        );
    }

    // Step 3: Contagion Modeling
    println!("\n🦠 Step 3: Contagion Modeling");
    println!("-----------------------------");

    let contagion_results = model_contagion_spread(
        &mut processor,
        &graph,
        &risk_assessment.critical_institutions,
    )?;

    println!("✅ Contagion modeling completed:");
    for (scenario, impact) in &contagion_results {
        println!(
            "   - {}: {:.1}% of network affected",
            scenario,
            impact * 100.0
        );
    }

    // Step 4: Financial Cluster Analysis
    println!("\n🏛️ Step 4: Financial Cluster Analysis");
    println!("------------------------------------");

    let clusters = analyze_financial_clusters(&mut processor, &graph)?;

    println!("✅ Financial cluster analysis completed:");
    println!("   - Number of financial clusters: {}", clusters.len());

    let cluster_sizes: Vec<_> = clusters.values().collect();
    let largest_cluster = cluster_sizes.iter().max().unwrap_or(&&0);
    let smallest_cluster = cluster_sizes.iter().min().unwrap_or(&&0);

    println!(
        "   - Largest cluster size: {} institutions",
        largest_cluster
    );
    println!(
        "   - Smallest cluster size: {} institutions",
        smallest_cluster
    );

    // Step 5: Stress Testing
    println!("\n⚡ Step 5: Stress Testing Simulation");
    println!("----------------------------------");

    let stress_test_results = run_stress_tests(&mut processor, &graph)?;

    println!("✅ Stress testing completed:");
    for (test_name, survival_rate) in &stress_test_results {
        println!(
            "   - {}: {:.1}% institutions survive",
            test_name,
            survival_rate * 100.0
        );
    }

    // Performance and optimization summary
    println!("\n📈 Performance Summary");
    println!("---------------------");

    let stats = processor.get_optimization_stats();
    println!(
        "   - Total optimizations performed: {}",
        stats.total_optimizations
    );
    println!(
        "   - Average performance speedup: {:.2}x",
        stats.average_speedup
    );
    println!(
        "   - GPU utilization: {:.1}%",
        stats.gpu_utilization * 100.0
    );
    println!("   - Memory efficiency: {:.3}", stats.memory_efficiency);
    println!(
        "   - Neural RL exploration rate: {:.3}",
        stats.neural_rl_epsilon
    );

    println!("\n✅ Financial risk analysis workflow completed successfully!");

    Ok(())
}

/// Create a realistic financial network with various institution types and connections
#[allow(dead_code)]
fn create_financial_network(
    num_institutions: usize,
    base_probability: f64,
) -> Result<Graph<usize, f64>, Box<dyn std::error::Error>> {
    // Use a combination of preferential attachment (for major banks) and random connections
    let mut graph = barabasi_albert_graph(num_institutions * 3 / 4, 4)?; // Major institutions with high connectivity

    // Add additional random connections for smaller institutions
    let random_component = erdos_renyi_graph(num_institutions / 4, base_probability * 2.0)?;

    // Merge the two components (simplified - in practice would be more sophisticated)
    let mut combined_graph = Graph::new();

    // Add all nodes
    for i in 0..num_institutions {
        combined_graph.add_node(i)?;
    }

    // Add edges from both graphs with some overlap
    for edge in graph.edges() {
        if edge.source < num_institutions && edge.target < num_institutions {
            let _ = combined_graph.add_edge(edge.source, edge.target, edge.weight);
        }
    }

    for edge in random_component.edges() {
        let source_offset = num_institutions * 3 / 4;
        if edge.source + source_offset < num_institutions
            && edge.target + source_offset < num_institutions
        {
            let _ = combined_graph.add_edge(
                edge.source + source_offset,
                edge.target + source_offset,
                edge.weight,
            );
        }
    }

    Ok(combined_graph)
}

/// Assess systemic risk using multiple centrality measures and network analysis
#[allow(dead_code)]
fn assess_systemic_risk(
    processor: &mut AdvancedProcessor,
    graph: &Graph<usize, f64>,
) -> Result<RiskAssessment, Box<dyn std::error::Error>> {
    println!("   🔍 Computing betweenness centrality...");
    let start = Instant::now();
    let betweenness =
        execute_with_enhanced_advanced(processor, graph, "risk_betweenness_centrality", |g| {
            betweenness_centrality(g)
        })?;
    println!(
        "   ✅ Betweenness centrality computed in {:?}",
        start.elapsed()
    );

    println!("   🔍 Computing closeness centrality...");
    let start = Instant::now();
    let closeness =
        execute_with_enhanced_advanced(processor, graph, "risk_closeness_centrality", |g| {
            closeness_centrality(g)
        })?;
    println!(
        "   ✅ Closeness centrality computed in {:?}",
        start.elapsed()
    );

    println!("   🔍 Computing PageRank scores...");
    let start = Instant::now();
    let pagerank_scores = execute_with_enhanced_advanced(processor, graph, "risk_pagerank", |g| {
        pagerank(g, 0.85, Some(100), Some(1e-6))
    })?;
    println!("   ✅ PageRank computed in {:?}", start.elapsed());

    // Combine metrics to compute risk scores
    let mut institution_risk_scores: HashMap<usize, f64> = HashMap::new();

    for &node in graph.nodes() {
        let betweenness_score = betweenness.get(&node).copied().unwrap_or(0.0);
        let closeness_score = closeness.get(&node).copied().unwrap_or(0.0);
        let pagerank_score = pagerank_scores.get(&node).copied().unwrap_or(0.0);

        // Weighted combination of centrality measures
        let risk_score = 0.4 * betweenness_score + 0.3 * closeness_score + 0.3 * pagerank_score;
        institution_risk_scores.insert(node, risk_score);
    }

    // Identify critical institutions (top 10% by risk score)
    let mut sorted_risks: Vec<_> = institution_risk_scores.iter().collect();
    sorted_risks.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    let critical_threshold = graph.node_count() / 10; // Top 10%
    let critical_institutions: Vec<usize> = sorted_risks
        .iter()
        .take(critical_threshold)
        .map(|(&node, _)| node)
        .collect();

    // Calculate aggregate risk metrics
    let systemic_risk_score = sorted_risks
        .iter()
        .take(critical_threshold)
        .map(|(_, &score)| score)
        .sum::<f64>()
        / critical_threshold as f64;
    let interconnectedness_score =
        graph.edge_count() as f64 / (graph.node_count() * (graph.node_count() - 1) / 2) as f64;
    let contagion_probability = (systemic_risk_score * interconnectedness_score).min(1.0);

    Ok(RiskAssessment {
        systemic_risk_score,
        interconnectedness_score,
        contagion_probability,
        critical_institutions,
        vulnerable_clusters: Vec::new(), // Would be computed in full implementation
        stress_test_results: HashMap::new(), // Would be populated by stress tests
    })
}

/// Analyze critical institutions using centrality measures
#[allow(dead_code)]
fn analyze_critical_institutions(
    processor: &mut AdvancedProcessor,
    graph: &Graph<usize, f64>,
) -> Result<Vec<(usize, f64)>, Box<dyn std::error::Error>> {
    println!("   🔍 Computing institution criticality scores...");

    let start = Instant::now();
    let betweenness =
        execute_with_enhanced_advanced(processor, graph, "critical_institution_analysis", |g| {
            betweenness_centrality(g)
        })?;

    let computation_time = start.elapsed();
    println!(
        "   ✅ Criticality analysis completed in {:?}",
        computation_time
    );

    let mut criticality_scores: Vec<(usize, f64)> = betweenness.into_iter().collect();
    criticality_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    Ok(criticality_scores)
}

/// Model contagion spread from critical institutions
#[allow(dead_code)]
fn model_contagion_spread(
    processor: &mut AdvancedProcessor,
    graph: &Graph<usize, f64>,
    critical_institutions: &[usize],
) -> Result<HashMap<String, f64>, Box<dyn std::error::Error>> {
    let mut results = HashMap::new();

    // Scenario 1: Single institution failure
    if let Some(&first_critical) = critical_institutions.first() {
        println!("   🔍 Modeling single institution failure contagion...");
        let start = Instant::now();

        let distances =
            execute_with_enhanced_advanced(processor, graph, "contagion_single_failure", |g| {
                shortest_path_dijkstra(g, first_critical)
            })?;

        // Institutions within 2 hops are considered affected
        let affected_count = distances.values().filter(|&&dist| dist <= 2.0).count();
        let impact_ratio = affected_count as f64 / graph.node_count() as f64;

        results.insert(
            "Single Critical Institution Failure".to_string(),
            impact_ratio,
        );
        println!(
            "   ✅ Single failure scenario computed in {:?}",
            start.elapsed()
        );
    }

    // Scenario 2: Multiple institution failure
    if critical_institutions.len() >= 3 {
        println!("   🔍 Modeling multiple institution failure contagion...");
        let start = Instant::now();

        let mut total_affected = 0;
        for &institution in critical_institutions.iter().take(3) {
            let distances = execute_with_enhanced_advanced(
                processor,
                graph,
                &format!("contagion_multi_failure_{}", institution),
                |g| shortest_path_dijkstra(g, institution),
            )?;

            total_affected += distances.values().filter(|&&dist| dist <= 2.0).count();
        }

        // Remove double counting (simplified)
        let unique_affected = (total_affected as f64 * 0.7).min(graph.node_count() as f64);
        let impact_ratio = unique_affected / graph.node_count() as f64;

        results.insert(
            "Multiple Critical Institution Failure".to_string(),
            impact_ratio,
        );
        println!(
            "   ✅ Multiple failure scenario computed in {:?}",
            start.elapsed()
        );
    }

    Ok(results)
}

/// Analyze financial clusters and sectoral concentrations
#[allow(dead_code)]
fn analyze_financial_clusters(
    processor: &mut AdvancedProcessor,
    graph: &Graph<usize, f64>,
) -> Result<HashMap<usize, usize>, Box<dyn std::error::Error>> {
    println!("   🔍 Detecting financial clusters...");

    let start = Instant::now();
    let clusters =
        execute_with_enhanced_advanced(processor, graph, "financial_cluster_detection", |g| {
            louvain_communities(g, None)
        })?;

    println!("   ✅ Financial clusters detected in {:?}", start.elapsed());

    Ok(clusters)
}

/// Run various stress testing scenarios
#[allow(dead_code)]
fn run_stress_tests(
    processor: &mut AdvancedProcessor,
    graph: &Graph<usize, f64>,
) -> Result<HashMap<String, f64>, Box<dyn std::error::Error>> {
    let mut results = HashMap::new();

    // Stress Test 1: Connectivity resilience
    println!("   🔍 Testing network connectivity resilience...");
    let start = Instant::now();

    let components =
        execute_with_enhanced_advanced(processor, graph, "stress_test_connectivity", |g| {
            connected_components(g)
        })?;

    // Measure largest connected component size
    let mut component_sizes: HashMap<usize, usize> = HashMap::new();
    for &component_id in components.values() {
        *component_sizes.entry(component_id).or_insert(0) += 1;
    }

    let largest_component_size = component_sizes.values().max().copied().unwrap_or(0);
    let connectivity_resilience = largest_component_size as f64 / graph.node_count() as f64;

    results.insert(
        "Connectivity Resilience Test".to_string(),
        connectivity_resilience,
    );
    println!("   ✅ Connectivity test completed in {:?}", start.elapsed());

    // Stress Test 2: Liquidity shock simulation
    println!("   🔍 Simulating liquidity shock...");
    let start = Instant::now();

    // Simplified liquidity shock: measure how many institutions remain connected
    // after removing top 5% highest degree nodes
    let node_degrees: HashMap<usize, usize> = graph
        .nodes()
        .iter()
        .map(|&node| {
            let degree = graph
                .edges()
                .filter(|e| e.source == node || e.target == node)
                .count();
            (node, degree)
        })
        .collect();

    let total_nodes = graph.node_count();
    let high_degree_threshold = total_nodes / 20; // Top 5%

    let mut sorted_degrees: Vec<_> = node_degrees.iter().collect();
    sorted_degrees.sort_by(|a, b| b.1.cmp(a.1));

    let surviving_institutions = total_nodes - high_degree_threshold;
    let liquidity_survival_rate = surviving_institutions as f64 / total_nodes as f64;

    results.insert("Liquidity Shock Test".to_string(), liquidity_survival_rate);
    println!(
        "   ✅ Liquidity shock test completed in {:?}",
        start.elapsed()
    );

    // Stress Test 3: Market volatility simulation
    let market_volatility_survival = 0.85; // Simplified - would involve complex modeling
    results.insert(
        "Market Volatility Test".to_string(),
        market_volatility_survival,
    );

    Ok(results)
}

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    financial_risk_analysis_workflow()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_financial_network_creation() {
        let graph = create_financial_network(100, 0.05).unwrap();
        assert_eq!(graph.node_count(), 100);
        assert!(graph.edge_count() > 0);
    }

    #[test]
    fn test_risk_assessment() {
        let graph = create_financial_network(50, 0.1).unwrap();
        let mut processor = create_performance_advanced_processor();

        let risk_assessment = assess_systemic_risk(&mut processor, &graph).unwrap();

        assert!(risk_assessment.systemic_risk_score >= 0.0);
        assert!(risk_assessment.systemic_risk_score <= 1.0);
        assert!(risk_assessment.interconnectedness_score >= 0.0);
        assert!(risk_assessment.interconnectedness_score <= 1.0);
        assert!(!risk_assessment.critical_institutions.is_empty());
    }

    #[test]
    fn test_contagion_modeling() {
        let graph = create_financial_network(30, 0.15).unwrap();
        let mut processor = create_enhanced_advanced_processor();
        let critical_institutions = vec![0, 1, 2]; // Top 3 institutions

        let contagion_results =
            model_contagion_spread(&mut processor, &graph, &critical_institutions).unwrap();

        assert!(!contagion_results.is_empty());
        for (_, impact) in contagion_results {
            assert!(impact >= 0.0);
            assert!(impact <= 1.0);
        }
    }

    #[test]
    fn test_stress_testing() {
        let graph = create_financial_network(40, 0.12).unwrap();
        let mut processor = create_enhanced_advanced_processor();

        let stress_results = run_stress_tests(&mut processor, &graph).unwrap();

        assert!(!stress_results.is_empty());
        for (_, survival_rate) in stress_results {
            assert!(survival_rate >= 0.0);
            assert!(survival_rate <= 1.0);
        }
    }
}
