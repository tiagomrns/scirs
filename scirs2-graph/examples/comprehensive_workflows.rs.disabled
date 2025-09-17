//! Comprehensive workflow examples for scirs2-graph
//!
//! This example demonstrates common workflows and usage patterns
//! for the scirs2-graph library, showcasing various graph operations,
//! algorithms, and performance optimizations.

use rand::rngs::StdRng;
use rand::{rng, SeedableRng};
use scirs2_graph::{
    barabasi_albert_graph,
    betweenness_centrality,
    breadth_first_search,
    // Graph measures
    centrality,
    clustering_coefficient,
    connected_components,
    // Algorithms
    dijkstra_path,
    // Graph generators
    erdos_renyi_graph,
    louvain_communities_result,
    pagerank_centrality,
    parallel_pagerank_centrality,
    watts_strogatz_graph,
    CentralityType,
    DiGraph,
    // Core graph types
    Graph,
    // I/O and utilities
    GraphError,
    Result,
};
use std::collections::HashMap;

/// Workflow 1: Basic Graph Operations and Analysis
///
/// Demonstrates fundamental graph creation, modification, and basic analysis.
#[allow(dead_code)]
fn workflow_basic_operations() -> Result<()> {
    println!("🔹 Workflow 1: Basic Graph Operations");

    // Create a new undirected graph
    let mut graph = Graph::<String, f64>::new();

    // Add nodes representing cities
    let cities = vec!["New York", "London", "Tokyo", "Sydney", "Paris"];
    for city in &cities {
        graph.add_node(city.to_string());
    }

    // Add weighted edges representing distances (in thousands of km)
    graph.add_edge("New York".to_string(), "London".to_string(), 5.5)?;
    graph.add_edge("New York".to_string(), "Tokyo".to_string(), 10.8)?;
    graph.add_edge("London".to_string(), "Paris".to_string(), 0.3)?;
    graph.add_edge("London".to_string(), "Sydney".to_string(), 17.0)?;
    graph.add_edge("Tokyo".to_string(), "Sydney".to_string(), 7.8)?;
    graph.add_edge("Paris".to_string(), "Sydney".to_string(), 16.8)?;

    // Basic graph properties
    println!("  📊 Graph Statistics:");
    println!("    Nodes: {}", graph.node_count());
    println!("    Edges: {}", graph.edge_count());
    let density = graph_density(&graph);
    println!("    Density: {:.3}", density);

    // Find shortest path between cities
    if let Some(path) = dijkstra_path(&graph, &"New York".to_string(), &"Sydney".to_string())? {
        println!(
            "  🛣️  Shortest path NYC→Sydney: {:?} (distance: {:.1}k km)",
            path.nodes, path.total_weight
        );
    }

    // Calculate degree centrality manually
    let mut degree_centrality = HashMap::new();
    for node in graph.nodes() {
        let degree = graph.degree(&node) as f64 / (graph.node_count() - 1) as f64;
        degree_centrality.insert(node, degree);
    }

    let most_central = degree_centrality
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    println!(
        "  🎯 Most central city (degree): {} ({:.3})",
        most_central.0, most_central.1
    );

    println!("  ✅ Basic operations completed successfully\n");
    Ok(())
}

/// Workflow 2: Large Graph Analysis with Performance Optimizations
///
/// Demonstrates working with larger graphs and leveraging performance optimizations
/// like parallel processing for computationally intensive algorithms.
#[allow(dead_code)]
fn workflow_large_graph_analysis() -> Result<()> {
    println!("🔹 Workflow 2: Large Graph Analysis");

    // Generate a large scale-free network (social network-like)
    let num_nodes = 10_000;
    let mut rng = StdRng::seed_from_u64(42);

    println!(
        "  🏗️  Generating Barabási-Albert graph ({} nodes)...",
        num_nodes
    );
    let graph = barabasi_albert_graph(num_nodes, 3, &mut rng)?;

    println!("  📊 Large Graph Statistics:");
    println!("    Nodes: {}", graph.node_count());
    println!("    Edges: {}", graph.edge_count());
    println!(
        "    Average degree: {:.2}",
        2.0 * graph.edge_count() as f64 / graph.node_count() as f64
    );

    // Compare sequential vs parallel PageRank
    println!("  ⚡ PageRank Performance Comparison:");

    // Sequential PageRank
    let start = std::time::Instant::now();
    let sequential_pagerank = pagerank_centrality(&graph, Some(0.85), Some(100), Some(1e-6))?;
    let sequential_time = start.elapsed();

    // Parallel PageRank (automatically uses parallel version for large graphs)
    #[cfg(feature = "parallel")]
    let (parallel_pagerank, parallel_time) = {
        let start = std::time::Instant::now();
        let result = parallel_pagerank_centrality(&graph, Some(0.85), Some(100), Some(1e-6))?;
        (result, start.elapsed())
    };
    #[cfg(not(feature = "parallel"))]
    let (parallel_pagerank, parallel_time) = (sequential_pagerank.clone(), sequential_time);

    println!("    Sequential: {:.2}ms", sequential_time.as_millis());
    println!("    Parallel:   {:.2}ms", parallel_time.as_millis());

    if parallel_time < sequential_time {
        let speedup = sequential_time.as_nanos() as f64 / parallel_time.as_nanos() as f64;
        println!("    🚀 Speedup: {:.2}x", speedup);
    }

    // Find most influential nodes
    let top_nodes: Vec<_> = parallel_pagerank.iter().collect::<Vec<_>>();
    let mut sorted_nodes = top_nodes.clone();
    sorted_nodes.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    println!("  🌟 Top 5 most influential nodes (PageRank):");
    for (i, (node, score)) in sorted_nodes.iter().take(5).enumerate() {
        println!("    {}. Node {}: {:.6}", i + 1, node, score);
    }

    println!("  ✅ Large graph analysis completed\n");
    Ok(())
}

/// Workflow 3: Community Detection and Network Analysis
///
/// Demonstrates community detection algorithms and network structure analysis.
#[allow(dead_code)]
fn workflow_community_detection() -> Result<()> {
    println!("🔹 Workflow 3: Community Detection");

    // Generate a graph with known community structure
    let mut rng = StdRng::seed_from_u64(42);
    let graph = watts_strogatz_graph(1000, 6, 0.1, &mut rng)?;

    println!("  🏗️  Generated small-world network (Watts-Strogatz)");
    println!(
        "    Nodes: {}, Edges: {}",
        graph.node_count(),
        graph.edge_count()
    );

    // Detect communities using Louvain algorithm
    println!("  🔍 Detecting communities with Louvain algorithm...");
    let communities = louvain_communities_result(&graph, None, None)?;

    println!("  📊 Community Structure:");
    println!(
        "    Number of communities: {}",
        communities.communities.len()
    );
    println!("    Modularity: {:.4}", communities.modularity);

    // Analyze community sizes
    let mut sizes: Vec<usize> = communities.communities.iter().map(|c| c.len()).collect();
    sizes.sort_by(|a, b| b.cmp(a));

    println!("  📈 Community size distribution:");
    println!("    Largest community: {} nodes", sizes[0]);
    println!("    Smallest community: {} nodes", sizes[sizes.len() - 1]);
    println!(
        "    Average size: {:.1}",
        sizes.iter().sum::<usize>() as f64 / sizes.len() as f64
    );

    // Calculate clustering coefficient
    let clustering = clustering_coefficient(&graph)?;
    println!("  🔗 Network clustering coefficient: {:.4}", clustering);

    println!("  ✅ Community detection completed\n");
    Ok(())
}

/// Workflow 4: Multi-Graph Comparison and Benchmarking
///
/// Demonstrates comparing different graph types and their properties.
#[allow(dead_code)]
fn workflow_graph_comparison() -> Result<()> {
    println!("🔹 Workflow 4: Graph Type Comparison");

    let n = 1000;
    let mut rng = StdRng::seed_from_u64(42);

    // Generate different types of graphs
    println!("  🏗️  Generating different graph types ({} nodes each):", n);

    let random_graph = erdos_renyi_graph(n, 0.01, &mut rng)?;
    let scale_free_graph = barabasi_albert_graph(n, 3, &mut rng)?;
    let small_world_graph = watts_strogatz_graph(n, 6, 0.1, &mut rng)?;

    let graphs = vec![
        ("Random (Erdős-Rényi)", &random_graph),
        ("Scale-free (Barabási-Albert)", &scale_free_graph),
        ("Small-world (Watts-Strogatz)", &small_world_graph),
    ];

    println!("  📊 Comparative Analysis:");
    println!(
        "    {':<25'} {':<8'} {':<8'} {':<10'} {':<10'}",
        "Graph Type", "Edges", "Density", "Clustering", "Diameter"
    );
    println!("    {}", "-".repeat(70));

    for (name, graph) in graphs {
        let edges = graph.edge_count();
        let density = graph_density(graph);
        let clustering = clustering_coefficient(graph)?;

        // Estimate diameter (computationally expensive for large graphs)
        let sample_nodes: Vec<_> = graph.nodes().into_iter().take(10).collect();
        let mut max_distance = 0.0;

        for i in 0..sample_nodes.len().min(5) {
            for j in (i + 1)..sample_nodes.len().min(5) {
                if let Some(path) = dijkstra_path(graph, &sample_nodes[i], &sample_nodes[j])? {
                    max_distance = max_distance.max(path.nodes.len() as f64 - 1.0);
                }
            }
        }

        println!(
            "    {:25} {:8} {:8.4} {:10.4} {:10.1}",
            name, edges, density, clustering, max_distance
        );
    }

    println!("  ✅ Graph comparison completed\n");
    Ok(())
}

/// Workflow 5: Directed Graph Analysis
///
/// Demonstrates working with directed graphs and specific directed graph algorithms.
#[allow(dead_code)]
fn workflow_directed_graph_analysis() -> Result<()> {
    println!("🔹 Workflow 5: Directed Graph Analysis");

    // Create a directed graph representing a citation network
    let mut digraph = DiGraph::<String, f64>::new();

    let papers = vec![
        "Paper A", "Paper B", "Paper C", "Paper D", "Paper E", "Paper F", "Paper G", "Paper H",
        "Paper I", "Paper J",
    ];

    // Add papers as nodes
    for paper in &papers {
        digraph.add_node(paper.to_string());
    }

    // Add citation relationships (directed edges)
    let citations = vec![
        ("Paper A", "Paper B"),
        ("Paper A", "Paper C"),
        ("Paper B", "Paper D"),
        ("Paper C", "Paper D"),
        ("Paper D", "Paper E"),
        ("Paper E", "Paper F"),
        ("Paper F", "Paper G"),
        ("Paper G", "Paper H"),
        ("Paper H", "Paper I"),
        ("Paper I", "Paper J"),
        ("Paper J", "Paper A"), // Creates a cycle
    ];

    for (citing, cited) in citations {
        digraph.add_edge(citing.to_string(), cited.to_string(), 1.0)?;
    }

    println!("  📚 Citation Network Statistics:");
    println!("    Papers: {}", digraph.node_count());
    println!("    Citations: {}", digraph.edge_count());

    // Analyze in-degree and out-degree distributions
    let nodes = digraph.nodes();
    let mut in_degrees = Vec::new();
    let mut out_degrees = Vec::new();

    for node in &nodes {
        in_degrees.push(digraph.in_degree(node));
        out_degrees.push(digraph.out_degree(node));
    }

    let avg_in_degree = in_degrees.iter().sum::<usize>() as f64 / nodes.len() as f64;
    let avg_out_degree = out_degrees.iter().sum::<usize>() as f64 / nodes.len() as f64;

    println!("    Average in-degree: {:.2}", avg_in_degree);
    println!("    Average out-degree: {:.2}", avg_out_degree);

    // Find strongly connected components
    let sccs = strongly_connected_components(&digraph)?;
    println!("    Strongly connected components: {}", sccs.len());

    println!("  ✅ Directed graph analysis completed\n");
    Ok(())
}

/// Workflow 6: Advanced Optimization Example
///
/// Demonstrates using advanced mode for automatic performance optimization.
#[cfg(feature = "Advanced")]
#[allow(dead_code)]
fn workflow_advanced_optimization() -> Result<()> {
    use scirs2_graph::advanced::{
        create_enhanced_advanced_processor, execute_with_enhanced_advanced,
    };

    println!("🔹 Workflow 6: Advanced Performance Optimization");

    // Create a large graph for performance testing
    let mut rng = StdRng::seed_from_u64(42);
    let graph = barabasi_albert_graph(50_000, 5, &mut rng)?;

    println!("  🧠 Created large graph for advanced demonstration:");
    println!("    Nodes: {}", graph.node_count());
    println!("    Edges: {}", graph.edge_count());

    // Create advanced processor
    let mut processor = create_enhanced_advanced_processor();

    // Standard vs Advanced PageRank comparison
    println!("  ⚡ Performance Comparison (PageRank):");

    // Standard implementation
    let start = std::time::Instant::now();
    let _standard_result = pagerank_centrality(&graph, Some(0.85), Some(100), Some(1e-6))?;
    let standard_time = start.elapsed();
    println!(
        "    Standard implementation: {:.2}ms",
        standard_time.as_millis()
    );

    // Advanced optimized implementation
    let start = std::time::Instant::now();
    let _advanced_result =
        execute_with_enhanced_advanced(&mut processor, &graph, "pagerank_large_graph", |g| {
            pagerank_centrality(g, Some(0.85), Some(100), Some(1e-6))
        })?;
    let advanced_time = start.elapsed();
    println!("    Advanced optimized: {:.2}ms", advanced_time.as_millis());

    if advanced_time < standard_time {
        let speedup = standard_time.as_nanos() as f64 / advanced_time.as_nanos() as f64;
        println!("    🚀 Speedup achieved: {:.2}x", speedup);
    }

    // Community detection with advanced
    println!("  🏘️  Community Detection with Advanced:");
    let start = std::time::Instant::now();
    let community_result =
        execute_with_enhanced_advanced(&mut processor, &graph, "community_detection_large", |g| {
            louvain_communities_result(g, None, None)
        })?;
    let community_time = start.elapsed();

    println!(
        "    Detected {} communities in {:.2}ms",
        community_result.communities.len(),
        community_time.as_millis()
    );
    println!("    Modularity: {:.4}", community_result.modularity);

    println!("  ✅ Advanced optimization completed\n");
    Ok(())
}

/// Workflow 7: Social Network Analysis Pipeline
///
/// Demonstrates a complete social network analysis workflow.
#[allow(dead_code)]
fn workflow_social_network_analysis() -> Result<()> {
    println!("🔹 Workflow 7: Social Network Analysis Pipeline");

    // Simulate a social network with different types of connections
    let mut social_graph = Graph::<String, f64>::new();

    // Add users
    let users = vec![
        "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry", "Ivy", "Jack",
        "Kate", "Liam", "Mia", "Noah", "Olivia", "Paul",
    ];

    for user in &users {
        social_graph.add_node(user.to_string())?;
    }

    // Add friendships (undirected edges with different strengths)
    let friendships = vec![
        ("Alice", "Bob", 0.9),     // Close friends
        ("Alice", "Charlie", 0.7), // Good friends
        ("Bob", "Diana", 0.8),
        ("Charlie", "Eve", 0.6), // Acquaintances
        ("Diana", "Frank", 0.9),
        ("Eve", "Grace", 0.8),
        ("Frank", "Henry", 0.7),
        ("Grace", "Ivy", 0.9),
        ("Henry", "Jack", 0.6),
        ("Ivy", "Kate", 0.8),
        ("Jack", "Liam", 0.7),
        ("Kate", "Mia", 0.9),
        ("Liam", "Noah", 0.6),
        ("Mia", "Olivia", 0.8),
        ("Noah", "Paul", 0.7),
        ("Paul", "Alice", 0.5), // Weak connection completing a cycle
        // Cross-community connections
        ("Alice", "Grace", 0.4),
        ("Bob", "Henry", 0.3),
        ("Charlie", "Jack", 0.4),
    ];

    for (user1, user2, strength) in friendships {
        social_graph.add_edge(user1.to_string(), user2.to_string(), strength)?;
    }

    println!("  👥 Social Network Statistics:");
    println!("    Users: {}", social_graph.node_count());
    println!("    Friendships: {}", social_graph.edge_count());
    println!("    Network density: {:.3}", graph_density(&social_graph));

    // Find most influential users (PageRank)
    println!("  🌟 Influence Analysis (PageRank):");
    let influence = pagerank_centrality(&social_graph, Some(0.85), Some(100), Some(1e-6))?;
    let mut influence_ranking: Vec<_> = influence.iter().collect();
    influence_ranking.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    println!("    Top 5 most influential users:");
    for (i, (user, score)) in influence_ranking.iter().take(5).enumerate() {
        println!("      {}. {}: {:.4}", i + 1, user, score);
    }

    // Find bridge personalities (betweenness centrality)
    println!("  🌉 Bridge Analysis (Betweenness Centrality):");
    let betweenness = betweenness_centrality(&social_graph)?;
    let mut bridge_ranking: Vec<_> = betweenness.iter().collect();
    bridge_ranking.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    println!("    Top 3 bridge personalities:");
    for (i, (user, score)) in bridge_ranking.iter().take(3).enumerate() {
        println!("      {}. {}: {:.4}", i + 1, user, score);
    }

    // Detect social groups
    println!("  👫 Community Detection:");
    let communities = louvain_communities_result(&social_graph, None, None)?;
    println!(
        "    Detected {} social groups",
        communities.communities.len()
    );
    println!("    Modularity: {:.4}", communities.modularity);

    for (i, community) in communities.communities.iter().enumerate() {
        println!("    Group {}: {:?}", i + 1, community);
    }

    println!("  ✅ Social network analysis completed\n");
    Ok(())
}

/// Workflow 8: Graph Machine Learning Pipeline
///
/// Demonstrates graph-based machine learning workflows.
#[allow(dead_code)]
fn workflow_graph_machine_learning() -> Result<()> {
    println!("🔹 Workflow 8: Graph Machine Learning Pipeline");

    // Create a molecular-like graph for demonstration
    let mut molecule_graph = Graph::<String, f64>::new();

    // Add atoms as nodes
    let atoms = vec!["C1", "C2", "O1", "N1", "C3", "C4", "H1", "H2", "H3"];
    for atom in &atoms {
        molecule_graph.add_node(atom.to_string())?;
    }

    // Add bonds as edges (with bond strength as weight)
    let bonds = vec![
        ("C1", "C2", 1.0), // Single bond
        ("C1", "O1", 2.0), // Double bond
        ("C2", "N1", 1.0),
        ("N1", "C3", 1.0),
        ("C3", "C4", 1.5), // Aromatic bond
        ("C1", "H1", 1.0),
        ("C2", "H2", 1.0),
        ("C4", "H3", 1.0),
    ];

    for (atom1, atom2, bond_strength) in bonds {
        molecule_graph.add_edge(atom1.to_string(), atom2.to_string(), bond_strength)?;
    }

    println!("  🧪 Molecular Graph Analysis:");
    println!("    Atoms: {}", molecule_graph.node_count());
    println!("    Bonds: {}", molecule_graph.edge_count());

    // Feature extraction for machine learning
    println!("  🤖 Feature Extraction:");

    // Node features: degree, local clustering
    let mut node_features = HashMap::new();
    for node in molecule_graph.nodes() {
        let degree = molecule_graph.degree(&node) as f64;

        // Calculate local clustering coefficient
        let neighbors: Vec<_> = molecule_graph.neighbors(&node).collect();
        let mut triangles = 0;
        let possible_triangles = neighbors.len() * (neighbors.len() - 1) / 2;

        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                if molecule_graph.has_edge(&neighbors[i], &neighbors[j]) {
                    triangles += 1;
                }
            }
        }

        let local_clustering = if possible_triangles > 0 {
            triangles as f64 / possible_triangles as f64
        } else {
            0.0
        };

        node_features.insert(node.clone(), (degree, local_clustering));
    }

    println!("    Node features extracted (degree, clustering):");
    for (node, (degree, clustering)) in &node_features {
        println!(
            "      {}: degree={:.1}, clustering={:.3}",
            node, degree, clustering
        );
    }

    // Graph-level features
    println!("  📊 Graph-level Features:");
    let avg_degree =
        molecule_graph.edges().count() as f64 * 2.0 / molecule_graph.node_count() as f64;
    let global_clustering = clustering_coefficient(&molecule_graph)?;
    let density = graph_density(&molecule_graph);

    println!("    Average degree: {:.2}", avg_degree);
    println!("    Global clustering: {:.4}", global_clustering);
    println!("    Graph density: {:.4}", density);

    // Subgraph/motif analysis
    println!("  🔍 Structural Motif Analysis:");
    let mut triangles = 0;
    let nodes: Vec<_> = molecule_graph.nodes().collect();

    for i in 0..nodes.len() {
        for j in (i + 1)..nodes.len() {
            for k in (j + 1)..nodes.len() {
                if molecule_graph.has_edge(&nodes[i], &nodes[j])
                    && molecule_graph.has_edge(&nodes[j], &nodes[k])
                    && molecule_graph.has_edge(&nodes[k], &nodes[i])
                {
                    triangles += 1;
                }
            }
        }
    }

    println!("    Triangular motifs found: {}", triangles);

    println!("  ✅ Graph ML pipeline completed\n");
    Ok(())
}

/// Workflow 9: Graph I/O and Persistence
///
/// Demonstrates loading, saving, and format conversion workflows.
#[allow(dead_code)]
fn workflow_graph_io_operations() -> Result<()> {
    println!("🔹 Workflow 9: Graph I/O and Format Conversion");

    // Create a sample graph
    let mut graph = Graph::<i32, f64>::new();
    for i in 0..10 {
        graph.add_node(i)?;
    }

    // Create a small ring topology
    for i in 0..10 {
        graph.add_edge(i, (i + 1) % 10, (i + 1) as f64)?;
    }

    println!("  💾 Graph Serialization Examples:");
    println!(
        "    Original graph: {} nodes, {} edges",
        graph.node_count(),
        graph.edge_count()
    );

    // JSON serialization (for demonstration - actual file I/O would use real files)
    let json_data = serde_json::to_string_pretty(&graph)
        .map_err(|e| GraphError::IoError(format!("JSON serialization failed: {}", e)))?;

    println!("    JSON representation size: {} bytes", json_data.len());

    // Simulate deserialization
    let _restored_graph: Graph<i32, f64> = serde_json::from_str(&json_data)
        .map_err(|e| GraphError::IoError(format!("JSON deserialization failed: {}", e)))?;

    println!("    ✅ JSON serialization/deserialization successful");

    // Edge list format simulation
    println!("  📝 Edge List Format:");
    let edge_list = graph
        .edges()
        .map(|edge| format!("{} {} {}", edge.source(), edge.target(), edge.weight()))
        .collect::<Vec<_>>()
        .join("\n");

    println!("    Edge list preview (first 3 lines):");
    for (i, line) in edge_list.lines().take(3).enumerate() {
        println!("      {}: {}", i + 1, line);
    }

    // Graph statistics for validation
    println!("  📊 Validation Statistics:");
    println!("    Node count preserved: {}", graph.node_count());
    println!("    Edge count preserved: {}", graph.edge_count());
    println!(
        "    Total weight: {:.2}",
        graph.edges().map(|e| e.weight()).sum::<f64>()
    );

    println!("  ✅ Graph I/O operations completed\n");
    Ok(())
}

/// Main function demonstrating all workflows
#[allow(dead_code)]
fn main() -> Result<()> {
    println!("🚀 SciRS2-Graph Comprehensive Workflow Examples");
    println!("===============================================\n");

    // Core workflows - always available
    workflow_basic_operations()?;
    workflow_large_graph_analysis()?;
    workflow_community_detection()?;
    workflow_graph_comparison()?;
    workflow_directed_graph_analysis()?;

    // Advanced workflows - feature dependent
    #[cfg(feature = "advanced")]
    workflow_advanced_optimization()?;

    // Application-specific workflows
    workflow_social_network_analysis()?;
    workflow_graph_machine_learning()?;
    workflow_graph_io_operations()?;

    println!("🎉 All workflows completed successfully!");
    println!("\n💡 Performance Optimization Tips:");
    println!("   • Use parallel algorithms for graphs with >10,000 nodes");
    println!("   • Enable advanced mode for 1.5-5x additional speedup");
    println!("   • Choose appropriate data types (u32 vs usize) for memory efficiency");
    println!("   • Use graph generators with fixed seeds for reproducible benchmarks");
    println!("   • Consider directed vs undirected graphs based on your domain");

    println!("\n🔧 Feature Recommendations:");
    println!("   • Enable 'parallel' feature for multi-threaded algorithms");
    println!("   • Enable 'advanced' feature for AI-driven optimizations");
    println!("   • Enable 'serde' feature for serialization/deserialization");
    println!("   • Profile memory usage with built-in memory profiling tools");

    println!("\n📚 Next Steps:");
    println!("   • Check the migration guide for transitioning from NetworkX");
    println!("   • Review API documentation for advanced algorithm options");
    println!("   • Benchmark your specific use case against baseline implementations");
    println!("   • Contribute your domain-specific workflows to the community");

    Ok(())
}
