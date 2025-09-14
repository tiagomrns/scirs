//! Comprehensive Advanced Mode Workflows
//!
//! This example demonstrates real-world usage patterns and workflows
//! for Advanced mode in various application domains.

#![allow(dead_code)]

use rand::{random, rng, Rng};
use scirs2_graph::advanced::{
    create_enhanced_advanced_processor, create_large_graph_advanced_processor,
    create_performance_advanced_processor, create_realtime_advanced_processor,
    execute_with_enhanced_advanced, AdvancedConfig, AdvancedProcessor, ExplorationStrategy,
};
use scirs2_graph::{
    barabasi_albert_graph, betweenness_centrality, closeness_centrality, connected_components,
    dijkstra_path, erdos_renyi_graph, floyd_warshall, label_propagation_result,
    louvain_communities_result, pagerank_centrality, strongly_connected_components, Graph,
};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Workflow 1: Social Network Analysis
///
/// This workflow demonstrates how to analyze social networks using Advanced mode,
/// including community detection, influence analysis, and path finding.
#[allow(dead_code)]
fn social_network_analysis_workflow() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌐 Social Network Analysis Workflow");
    println!("===================================");

    // Create a social network-like graph (scale-free with high clustering)
    println!("📊 Generating social network graph...");
    let mut rng = rand::rng();
    let graph = barabasi_albert_graph(5000, 10, &mut rng)?; // 5K users, preferential attachment

    println!("✅ Social network created:");
    println!("   - Users: {}", graph.node_count());
    println!("   - Connections: {}", graph.edge_count());
    println!(
        "   - Average degree: {:.2}",
        graph.edge_count() as f64 * 2.0 / graph.node_count() as f64
    );

    // Create an enhanced advanced processor optimized for social network analysis
    let mut processor = create_enhanced_advanced_processor();

    // Step 1: Community Detection
    println!("\n👥 Step 1: Community Detection");
    println!("------------------------------");

    let start = Instant::now();
    let communities = execute_with_enhanced_advanced(
        &mut processor,
        &graph,
        "social_community_detection",
        |g| louvain_communities_result(g, None, None),
    )?;
    let community_time = start.elapsed();

    println!(
        "✅ Found {} communities in {:?}",
        communities.len(),
        community_time
    );

    // Analyze community sizes
    let mut community_sizes: HashMap<usize, usize> = HashMap::new();
    for &community_id in communities.values() {
        *community_sizes.entry(community_id).or_insert(0) += 1;
    }

    let largest_community = community_sizes.values().max().unwrap_or(&0);
    let smallest_community = community_sizes.values().min().unwrap_or(&0);

    println!("   - Largest community: {} users", largest_community);
    println!("   - Smallest community: {} users", smallest_community);
    println!(
        "   - Average community size: {:.1}",
        graph.node_count() as f64 / communities.len() as f64
    );

    // Step 2: Influence Analysis (PageRank)
    println!("\n📈 Step 2: Influence Analysis");
    println!("----------------------------");

    let start = Instant::now();
    let influence_scores =
        execute_with_enhanced_advanced(&mut processor, &graph, "social_influence_analysis", |g| {
            pagerank_centrality(g, 0.85, 1e-6)
        })?;
    let influence_time = start.elapsed();

    println!("✅ Computed influence scores in {:?}", influence_time);

    // Find top influencers
    let mut sorted_influences: Vec<_> = influence_scores.iter().collect();
    sorted_influences.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    println!("   Top 5 influencers:");
    for (i, (node_id, score)) in sorted_influences.iter().take(5).enumerate() {
        println!(
            "     {}. User {}: influence score {:.6}",
            i + 1,
            node_id,
            score
        );
    }

    // Step 3: Path Analysis (connectivity between communities)
    println!("\n🔗 Step 3: Inter-Community Connectivity");
    println!("---------------------------------------");

    if let (Some(&user1), Some(&user2)) = (
        sorted_influences.get(0).map(|(id, _)| id),
        sorted_influences.get(1).map(|(id, _)| id),
    ) {
        let start = Instant::now();
        let path =
            execute_with_enhanced_advanced(&mut processor, &graph, "social_path_analysis", |g| {
                shortest_path_dijkstra(g, *user1)
            })?;
        let path_time = start.elapsed();

        if let Some(distance) = path.get(user2) {
            println!("✅ Path analysis completed in {:?}", path_time);
            println!("   - Distance between top influencers: {} hops", distance);
        }
    }

    // Performance summary
    let total_time = community_time + influence_time;
    println!("\n📊 Performance Summary:");
    println!("   - Total analysis time: {:?}", total_time);
    println!(
        "   - Community detection: {:.1}%",
        community_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
    );
    println!(
        "   - Influence analysis: {:.1}%",
        influence_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
    );

    let stats = processor.get_optimization_stats();
    println!("   - Advanced speedup: {:.2}x", stats.average_speedup);
    println!("   - Memory efficiency: {:.2}", stats.memory_efficiency);

    Ok(())
}

/// Workflow 2: Bioinformatics Network Analysis
///
/// Demonstrates protein interaction network analysis with specific
/// optimizations for biological data patterns.
#[allow(dead_code)]
fn bioinformatics_workflow() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧬 Bioinformatics Network Analysis Workflow");
    println!("===========================================");

    // Create a protein interaction network-like graph
    println!("🧪 Generating protein interaction network...");
    let graph = erdos_renyi_graph(2000, 0.008)?; // Sparse biological network

    println!("✅ Protein network created:");
    println!("   - Proteins: {}", graph.node_count());
    println!("   - Interactions: {}", graph.edge_count());
    println!(
        "   - Network density: {:.6}",
        graph.edge_count() as f64 / (graph.node_count() * (graph.node_count() - 1) / 2) as f64
    );

    // Use performance-optimized advanced for computational biology
    let mut processor = create_performance_advanced_processor();

    // Step 1: Identify Protein Complexes (Strong Components)
    println!("\n🔬 Step 1: Protein Complex Identification");
    println!("----------------------------------------");

    let start = Instant::now();
    let components =
        execute_with_enhanced_advanced(&mut processor, &graph, "protein_complex_detection", |g| {
            connected_components(g)
        })?;
    let complex_time = start.elapsed();

    println!(
        "✅ Identified {} protein complexes in {:?}",
        components.len(),
        complex_time
    );

    // Analyze complex sizes
    let mut complex_sizes: HashMap<usize, usize> = HashMap::new();
    for &complex_id in components.values() {
        *complex_sizes.entry(complex_id).or_insert(0) += 1;
    }

    let large_complexes = complex_sizes.values().filter(|&&size| size >= 10).count();
    println!("   - Large complexes (≥10 proteins): {}", large_complexes);

    // Step 2: Hub Protein Analysis (Centrality)
    println!("\n🎯 Step 2: Hub Protein Analysis");
    println!("------------------------------");

    let start = Instant::now();
    let centrality_scores =
        execute_with_enhanced_advanced(&mut processor, &graph, "hub_protein_analysis", |g| {
            betweenness_centrality(g)
        })?;
    let centrality_time = start.elapsed();

    println!("✅ Computed centrality scores in {:?}", centrality_time);

    // Find hub proteins
    let mut sorted_centrality: Vec<_> = centrality_scores.iter().collect();
    sorted_centrality.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    let hub_threshold = 0.1; // Top 10% are considered hubs
    let num_hubs = (sorted_centrality.len() as f64 * hub_threshold) as usize;

    println!("   Top {} hub proteins:", num_hubs.min(5));
    for (i, (protein_id, score)) in sorted_centrality.iter().take(num_hubs.min(5)).enumerate() {
        println!(
            "     {}. Protein {}: centrality {:.6}",
            i + 1,
            protein_id,
            score
        );
    }

    // Step 3: Functional Module Detection
    println!("\n🧩 Step 3: Functional Module Detection");
    println!("-------------------------------------");

    let start = Instant::now();
    let modules = execute_with_enhanced_advanced(
        &mut processor,
        &graph,
        "functional_module_detection",
        |g| label_propagation(g, Some(100)),
    )?;
    let module_time = start.elapsed();

    println!(
        "✅ Detected {} functional modules in {:?}",
        modules.len(),
        module_time
    );

    // Analyze module characteristics
    let mut module_sizes: HashMap<usize, usize> = HashMap::new();
    for &module_id in modules.values() {
        *module_sizes.entry(module_id).or_insert(0) += 1;
    }

    let avg_module_size = graph.node_count() as f64 / modules.len() as f64;
    println!("   - Average module size: {:.1} proteins", avg_module_size);

    // Performance and optimization analysis
    let total_time = complex_time + centrality_time + module_time;
    println!("\n🔧 Computational Analysis:");
    println!("   - Total computation time: {:?}", total_time);
    println!(
        "   - Complex detection: {:.1}%",
        complex_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
    );
    println!(
        "   - Centrality analysis: {:.1}%",
        centrality_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
    );
    println!(
        "   - Module detection: {:.1}%",
        module_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
    );

    let stats = processor.get_optimization_stats();
    println!(
        "   - Neural RL optimizations: {}",
        stats.total_optimizations
    );
    println!(
        "   - GPU utilization: {:.1}%",
        stats.gpu_utilization * 100.0
    );

    Ok(())
}

/// Workflow 3: Large-Scale Infrastructure Network Analysis
///
/// Demonstrates analysis of large infrastructure networks (roads, internet, etc.)
/// with memory-efficient processing.
#[allow(dead_code)]
fn infrastructure_network_workflow() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🏗️ Infrastructure Network Analysis Workflow");
    println!("===========================================");

    // Create a large infrastructure-like network
    println!("🌍 Generating infrastructure network...");
    let graph = random_graph(50000, 150000, false)?; // Large sparse network

    println!("✅ Infrastructure network created:");
    println!("   - Nodes (intersections): {}", graph.node_count());
    println!("   - Edges (roads/connections): {}", graph.edge_count());
    println!(
        "   - Average connectivity: {:.2}",
        graph.edge_count() as f64 * 2.0 / graph.node_count() as f64
    );

    // Use large-graph optimized advanced processor
    let mut processor = create_large_graph_advanced_processor();

    // Step 1: Network Resilience Analysis
    println!("\n🛡️ Step 1: Network Resilience Analysis");
    println!("-------------------------------------");

    let start = Instant::now();
    let components =
        execute_with_enhanced_advanced(&mut processor, &graph, "infrastructure_resilience", |g| {
            connected_components(g)
        })?;
    let resilience_time = start.elapsed();

    println!("✅ Resilience analysis completed in {:?}", resilience_time);

    // Calculate resilience metrics
    let largest_component_size = components
        .values()
        .fold(HashMap::new(), |mut acc, &comp_id| {
            *acc.entry(comp_id).or_insert(0) += 1;
            acc
        })
        .values()
        .max()
        .unwrap_or(&0);

    let connectivity_ratio = *largest_component_size as f64 / graph.node_count() as f64;

    println!(
        "   - Largest connected component: {} nodes ({:.1}%)",
        largest_component_size,
        connectivity_ratio * 100.0
    );
    println!(
        "   - Network fragmentation: {:.3}",
        1.0 - connectivity_ratio
    );

    // Step 2: Critical Node Identification
    println!("\n⚠️ Step 2: Critical Node Identification");
    println!("--------------------------------------");

    let start = Instant::now();
    let closeness_scores = execute_with_enhanced_advanced(
        &mut processor,
        &graph,
        "critical_node_identification",
        |g| closeness_centrality(g),
    )?;
    let critical_time = start.elapsed();

    println!("✅ Critical node analysis completed in {:?}", critical_time);

    // Find most critical nodes
    let mut sorted_closeness: Vec<_> = closeness_scores.iter().collect();
    sorted_closeness.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    println!("   Top 5 critical infrastructure nodes:");
    for (i, (node_id, score)) in sorted_closeness.iter().take(5).enumerate() {
        println!(
            "     {}. Node {}: criticality score {:.6}",
            i + 1,
            node_id,
            score
        );
    }

    // Step 3: Route Optimization Analysis
    println!("\n🗺️ Step 3: Route Optimization Analysis");
    println!("-------------------------------------");

    if let Some(&critical_node) = sorted_closeness.get(0).map(|(id, _)| id) {
        let start = Instant::now();
        let distances =
            execute_with_enhanced_advanced(&mut processor, &graph, "route_optimization", |g| {
                shortest_path_dijkstra(g, *critical_node)
            })?;
        let route_time = start.elapsed();

        println!("✅ Route optimization completed in {:?}", route_time);

        // Analyze reachability
        let reachable_nodes = distances.len();
        let max_distance = distances.values().max().unwrap_or(&0.0);
        let avg_distance = distances.values().sum::<f64>() / distances.len() as f64;

        println!(
            "   - Reachable nodes from critical point: {} ({:.1}%)",
            reachable_nodes,
            reachable_nodes as f64 / graph.node_count() as f64 * 100.0
        );
        println!("   - Maximum distance: {:.1}", max_distance);
        println!("   - Average distance: {:.2}", avg_distance);
    }

    // Memory and performance analysis
    let total_time = resilience_time + critical_time;
    println!("\n💾 Performance and Memory Analysis:");
    println!("   - Total analysis time: {:?}", total_time);
    println!(
        "   - Processing rate: {:.0} nodes/second",
        graph.node_count() as f64 / total_time.as_secs_f64()
    );

    let stats = processor.get_optimization_stats();
    println!(
        "   - Memory efficiency achieved: {:.2}x",
        stats.memory_efficiency
    );
    println!("   - Adaptive optimizations: {}", stats.total_optimizations);

    Ok(())
}

/// Workflow 4: Real-Time Network Monitoring
///
/// Demonstrates real-time analysis capabilities for dynamic networks
/// with continuous updates and monitoring.
#[allow(dead_code)]
fn real_time_monitoring_workflow() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚡ Real-Time Network Monitoring Workflow");
    println!("=======================================");

    // Create initial network state
    println!("📡 Initializing network monitoring...");
    let mut graph = random_graph(1000, 3000, false)?;

    println!("✅ Initial network state:");
    println!("   - Nodes: {}", graph.node_count());
    println!("   - Edges: {}", graph.edge_count());

    // Use real-time optimized advanced processor
    let mut processor = create_realtime_advanced_processor();

    // Configure for rapid adaptation
    // Note: In a real implementation, we'd need API to modify processor config

    println!("\n🔄 Starting real-time monitoring simulation...");

    // Simulate network changes over time
    let simulation_steps = 10;
    let mut performance_history = Vec::new();

    for step in 1..=simulation_steps {
        println!("\n📊 Monitoring Step {} of {}", step, simulation_steps);
        println!("------------------------");

        // Simulate network changes (add/remove edges)
        if step % 3 == 0 {
            // Add some edges (network growth)
            let new_edges = 50;
            for _ in 0..new_edges {
                if let (Ok(node1), Ok(node2)) = (
                    graph.add_node(step * 1000 + rand::random::<usize>() % 100),
                    graph.add_node(step * 1000 + rand::random::<usize>() % 100 + 100),
                ) {
                    let _ = graph.add_edge(node1, node2, 1.0);
                }
            }
            println!("   🔗 Added {} new connections", new_edges);
        }

        // Real-time analysis
        let start = Instant::now();

        // Quick connectivity check
        let components = execute_with_enhanced_advanced(
            &mut processor,
            &graph,
            &format!("realtime_connectivity_step_{}", step),
            |g| connected_components(g),
        )?;

        // Quick centrality update
        let pagerank_scores = execute_with_enhanced_advanced(
            &mut processor,
            &graph,
            &format!("realtime_centrality_step_{}", step),
            |g| pagerank(g, 0.85, Some(20), Some(1e-4)), // Reduced iterations for speed
        )?;

        let step_time = start.elapsed();
        performance_history.push(step_time);

        // Report step results
        let largest_component = components
            .values()
            .fold(HashMap::new(), |mut acc, &comp_id| {
                *acc.entry(comp_id).or_insert(0) += 1;
                acc
            })
            .values()
            .max()
            .unwrap_or(&0);

        let top_node = pagerank_scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(id, score)| (*id, *score));

        println!("   ✅ Analysis completed in {:?}", step_time);
        println!(
            "   📈 Network size: {} nodes, {} edges",
            graph.node_count(),
            graph.edge_count()
        );
        println!("   🔗 Largest component: {} nodes", largest_component);

        if let Some((node_id, score)) = top_node {
            println!("   ⭐ Top node: {} (score: {:.6})", node_id, score);
        }

        // Check if we're meeting real-time requirements (< 100ms target)
        let is_realtime = step_time < Duration::from_millis(100);
        println!(
            "   ⚡ Real-time target: {} (target: <100ms)",
            if is_realtime { "✅ MET" } else { "❌ MISSED" }
        );
    }

    // Analyze real-time performance
    println!("\n📈 Real-Time Performance Analysis:");
    println!("=================================");

    let avg_time = performance_history.iter().sum::<Duration>() / performance_history.len() as u32;
    let max_time = performance_history.iter().max().unwrap();
    let min_time = performance_history.iter().min().unwrap();

    let realtime_success_rate = performance_history
        .iter()
        .filter(|&&t| t < Duration::from_millis(100))
        .count() as f64
        / performance_history.len() as f64;

    println!("   - Average processing time: {:?}", avg_time);
    println!("   - Best case time: {:?}", min_time);
    println!("   - Worst case time: {:?}", max_time);
    println!(
        "   - Real-time success rate: {:.1}%",
        realtime_success_rate * 100.0
    );

    let stats = processor.get_optimization_stats();
    println!("   - Total adaptations: {}", stats.total_optimizations);
    println!("   - Final speedup achieved: {:.2}x", stats.average_speedup);
    println!(
        "   - Neural RL exploration rate: {:.3}",
        stats.neural_rl_epsilon
    );

    Ok(())
}

/// Workflow 5: Multi-Algorithm Benchmark Suite
///
/// Demonstrates comparative analysis of different algorithms
/// with automatic optimization selection.
#[allow(dead_code)]
fn benchmark_suite_workflow() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🏁 Multi-Algorithm Benchmark Suite");
    println!("=================================");

    // Create test graphs of different types
    let test_graphs = vec![
        ("Sparse Random", random_graph(2000, 4000, false)?),
        ("Dense Random", random_graph(1000, 50000, false)?),
        ("Scale-Free", barabasi_albert_graph(2000, 5)?),
        ("Erdős-Rényi", erdos_renyi_graph(2000, 0.005)?),
    ];

    // Create multiple processor configurations for comparison
    let mut processors = vec![
        ("Standard Advanced", create_enhanced_advanced_processor()),
        (
            "Performance Optimized",
            create_performance_advanced_processor(),
        ),
        ("Real-time Optimized", create_realtime_advanced_processor()),
        (
            "Large Graph Optimized",
            create_large_graph_advanced_processor(),
        ),
    ];

    // Define benchmark algorithms
    let algorithms = vec![
        ("connected_components", "Connected Components"),
        ("pagerank", "PageRank Centrality"),
        ("community_detection", "Community Detection"),
    ];

    println!("🧪 Running comprehensive benchmark suite...");
    println!("   - {} graph types", test_graphs.len());
    println!("   - {} processor configurations", processors.len());
    println!("   - {} algorithms", algorithms.len());

    let mut results = HashMap::new();

    for (graph_name, graph) in &test_graphs {
        println!("\n📊 Testing graph type: {}", graph_name);
        println!(
            "   - Nodes: {}, Edges: {}",
            graph.node_count(),
            graph.edge_count()
        );

        for (processor_name, processor) in &mut processors {
            println!("  🔧 Using processor: {}", processor_name);

            for (alg_id, alg_name) in &algorithms {
                let start = Instant::now();

                let result = match *alg_id {
                    "connected_components" => execute_with_enhanced_advanced(
                        processor,
                        graph,
                        &format!("{}_{}_{}", graph_name, processor_name, alg_id),
                        |g| connected_components(g).map(|components| components.len()),
                    )?,
                    "pagerank" => execute_with_enhanced_advanced(
                        processor,
                        graph,
                        &format!("{}_{}_{}", graph_name, processor_name, alg_id),
                        |g| pagerank(g, 0.85, Some(50), Some(1e-6)).map(|scores| scores.len()),
                    )?,
                    "community_detection" => execute_with_enhanced_advanced(
                        processor,
                        graph,
                        &format!("{}_{}_{}", graph_name, processor_name, alg_id),
                        |g| louvain_communities(g, None).map(|communities| communities.len()),
                    )?,
                    _ => 0,
                };

                let elapsed = start.elapsed();

                results.insert(
                    (
                        graph_name.to_string(),
                        processor_name.to_string(),
                        alg_id.to_string(),
                    ),
                    (elapsed, result),
                );

                println!("    ✅ {}: {:?} (result: {})", alg_name, elapsed, result);
            }
        }
    }

    // Analyze benchmark results
    println!("\n🏆 Benchmark Results Analysis:");
    println!("=============================");

    for (alg_id, alg_name) in &algorithms {
        println!("\n📈 Algorithm: {}", alg_name);
        println!("{}---{}", "".repeat(alg_name.len()), "".repeat(10));

        for (graph_name, _) in &test_graphs {
            println!("  Graph: {}", graph_name);

            let mut processor_times = Vec::new();
            for (processor_name, _) in &processors {
                if let Some((time, _)) = results.get(&(
                    graph_name.to_string(),
                    processor_name.to_string(),
                    alg_id.to_string(),
                )) {
                    processor_times.push((processor_name, *time));
                }
            }

            // Sort by performance
            processor_times.sort_by_key(|(_, time)| *time);

            if let Some((best_processor, best_time)) = processor_times.first() {
                println!("    🥇 Best: {} ({:?})", best_processor, best_time);

                for (processor_name, time) in &processor_times[1..] {
                    let slowdown = time.as_secs_f64() / best_time.as_secs_f64();
                    println!(
                        "       {}: {:?} ({:.2}x slower)",
                        processor_name, time, slowdown
                    );
                }
            }
        }
    }

    // Overall performance summary
    println!("\n🎯 Overall Performance Summary:");
    println!("==============================");

    let mut processor_wins = HashMap::new();
    for (alg_id, _) in &algorithms {
        for (graph_name, _) in &test_graphs {
            let mut best_time = Duration::from_secs(u64::MAX);
            let mut best_processor = "";

            for (processor_name, _) in &processors {
                if let Some((time, _)) = results.get(&(
                    graph_name.to_string(),
                    processor_name.to_string(),
                    alg_id.to_string(),
                )) {
                    if *time < best_time {
                        best_time = *time;
                        best_processor = processor_name;
                    }
                }
            }

            *processor_wins
                .entry(best_processor.to_string())
                .or_insert(0) += 1;
        }
    }

    println!("Processor Performance Rankings:");
    let mut sorted_wins: Vec<_> = processor_wins.iter().collect();
    sorted_wins.sort_by_key(|(_, wins)| std::cmp::Reverse(**wins));

    for (i, (processor, wins)) in sorted_wins.iter().enumerate() {
        println!("  {}. {}: {} wins", i + 1, processor, wins);
    }

    Ok(())
}

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 SciRS2 Advanced Comprehensive Workflows");
    println!("============================================");
    println!("This demo showcases real-world usage patterns for advanced mode.");
    println!();

    // Run all workflows
    social_network_analysis_workflow()?;
    bioinformatics_workflow()?;
    infrastructure_network_workflow()?;
    real_time_monitoring_workflow()?;
    benchmark_suite_workflow()?;

    println!("\n✨ All workflows completed successfully!");
    println!("======================================");
    println!();
    println!("💡 Key Insights from Comprehensive Testing:");
    println!("• Different advanced configurations excel in different scenarios");
    println!("• Neural RL adaptation improves performance over multiple runs");
    println!("• Memory optimization is crucial for large graphs");
    println!("• Real-time configurations trade accuracy for speed");
    println!("• Performance optimization achieves best overall results");
    println!();
    println!("📚 For detailed configuration guides, see the documentation");
    println!("🔬 For algorithm-specific optimizations, consult the API reference");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_social_network_workflow() {
        // Test basic social network analysis workflow
        let graph = barabasi_albert_graph(100, 3).unwrap();
        let mut processor = create_enhanced_advanced_processor();

        let communities =
            execute_with_enhanced_advanced(&mut processor, &graph, "test_social_community", |g| {
                louvain_communities(g, None)
            })
            .unwrap();

        assert!(!communities.is_empty());
        assert!(communities.len() <= graph.node_count());
    }

    #[test]
    fn test_bioinformatics_workflow() {
        // Test bioinformatics network analysis
        let graph = erdos_renyi_graph(200, 0.01).unwrap();
        let mut processor = create_performance_advanced_processor();

        let components =
            execute_with_enhanced_advanced(&mut processor, &graph, "test_bio_components", |g| {
                connected_components(g)
            })
            .unwrap();

        assert!(!components.is_empty());
        assert!(components.len() <= graph.node_count());
    }

    #[test]
    fn test_infrastructure_workflow() {
        // Test infrastructure network analysis
        let graph = random_graph(500, 1000, false).unwrap();
        let mut processor = create_large_graph_advanced_processor();

        let centrality = execute_with_enhanced_advanced(
            &mut processor,
            &graph,
            "test_infrastructure_centrality",
            |g| closeness_centrality(g),
        )
        .unwrap();

        assert!(!centrality.is_empty());
        assert!(centrality.len() <= graph.node_count());
    }

    #[test]
    fn test_realtime_workflow() {
        // Test real-time processing capabilities
        let graph = random_graph(100, 200, false).unwrap();
        let mut processor = create_realtime_advanced_processor();

        let start = Instant::now();
        let pagerank_result =
            execute_with_enhanced_advanced(&mut processor, &graph, "test_realtime_pagerank", |g| {
                pagerank(g, 0.85, Some(10), Some(1e-4))
            })
            .unwrap();
        let elapsed = start.elapsed();

        assert!(!pagerank_result.is_empty());
        // Real-time target: should complete quickly for small graphs
        assert!(elapsed < Duration::from_millis(500));
    }

    #[test]
    fn test_benchmark_suite() {
        // Test basic benchmark functionality
        let graph = random_graph(50, 100, false).unwrap();
        let mut processor = create_enhanced_advanced_processor();

        // Test multiple algorithms
        let components =
            execute_with_enhanced_advanced(&mut processor, &graph, "benchmark_components", |g| {
                connected_components(g)
            })
            .unwrap();

        let pagerank_scores =
            execute_with_enhanced_advanced(&mut processor, &graph, "benchmark_pagerank", |g| {
                pagerank(g, 0.85, Some(20), Some(1e-5))
            })
            .unwrap();

        assert!(!components.is_empty());
        assert!(!pagerank_scores.is_empty());

        // Verify optimization statistics
        let stats = processor.get_optimization_stats();
        assert!(stats.total_optimizations >= 2); // Should have run at least 2 algorithms
    }

    #[test]
    fn test_processor_configurations() {
        // Test different processor configurations
        let graph = random_graph(100, 200, false).unwrap();

        let processors = vec![
            create_enhanced_advanced_processor(),
            create_performance_advanced_processor(),
            create_realtime_advanced_processor(),
            create_large_graph_advanced_processor(),
        ];

        for mut processor in processors {
            let result = execute_with_enhanced_advanced(
                &mut processor,
                &graph,
                "test_processor_config",
                |g| connected_components(g),
            );

            assert!(result.is_ok());
            assert!(!result.unwrap().is_empty());
        }
    }
}
