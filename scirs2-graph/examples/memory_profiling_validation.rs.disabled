//! Memory profiling validation example
//!
//! This example demonstrates the memory profiling and optimization capabilities
//! of scirs2-graph by creating graphs of different sizes and analyzing their
//! memory usage patterns.

use scirs2_graph::{
    generators,
    memory::{
        suggest_optimizations, BitPackedGraph, CSRGraph, CompressedAdjacencyList, HybridGraph,
        MemoryProfiler, OptimizedGraphBuilder, RealTimeMemoryProfiler,
    },
    Graph,
};
use std::collections::HashMap;
use std::thread;
use std::time::Duration;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 SciRS2-Graph Memory Profiling Validation");
    println!("=".repeat(50));

    // Test 1: Basic Memory Profiling
    println!("\n📊 Test 1: Basic Memory Profiling");
    basic_memory_profiling()?;

    // Test 2: Representation Comparison
    println!("\n📈 Test 2: Memory Representation Comparison");
    representation_comparison()?;

    // Test 3: Fragmentation Analysis
    println!("\n🗂️  Test 3: Memory Fragmentation Analysis");
    fragmentation_analysis()?;

    // Test 4: Optimization Suggestions
    println!("\n💡 Test 4: Memory Optimization Suggestions");
    optimization_suggestions()?;

    // Test 5: Real-time Monitoring
    println!("\n⏱️  Test 5: Real-time Memory Monitoring");
    real_time_monitoring()?;

    // Test 6: Optimized Graph Builder
    println!("\n🏗️  Test 6: Optimized Graph Builder");
    optimized_builder_test()?;

    println!("\n✅ All memory profiling tests completed successfully!");
    Ok(())
}

#[allow(dead_code)]
fn basic_memory_profiling() -> Result<(), Box<dyn std::error::Error>> {
    let sizes = vec![1_000, 5_000, 10_000];
    let edge_probability = 0.01;

    println!("Graph Size | Total Memory | Node Memory | Edge Memory | Efficiency");
    println!("-".repeat(70));

    for size in sizes {
        let graph = generators::erdos_renyi_graph(size, edge_probability, None)?;
        let stats = MemoryProfiler::profile_graph(&graph);

        println!(
            "{:10} | {:11} | {:11} | {:11} | {:9.1}%",
            format!("{}", size),
            format_bytes(stats.total_bytes),
            format_bytes(stats.node_bytes),
            format_bytes(stats.edge_bytes),
            stats.efficiency * 100.0
        );
    }

    Ok(())
}

#[allow(dead_code)]
fn representation_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let n = 5_000;
    let edge_probability = 0.005; // Sparse graph

    // Create standard graph
    let graph = generators::erdos_renyi_graph(n, edge_probability, None)?;
    let standard_stats = MemoryProfiler::profile_graph(&graph);

    // Create CSR representation
    let edges: Vec<_> = (0..graph.node_count())
        .flat_map(|u| graph.neighbors(u).map(move |v| (u, v, 1.0)))
        .collect();
    let csr = CSRGraph::from_edges(n, edges.clone())?;

    // Create bit-packed representation (unweighted)
    let mut bitpacked = BitPackedGraph::new(n, false);
    for (u, v, _) in &edges {
        if u <= v {
            bitpacked.add_edge(*u, *v)?;
        }
    }

    // Create compressed adjacency list
    let adj_lists: Vec<Vec<_>> = (0..n).map(|u| graph.neighbors(u).collect()).collect();
    let compressed = CompressedAdjacencyList::from_adjacency(adj_lists);

    println!("Representation      | Memory Usage | vs Standard");
    println!("-".repeat(45));
    println!(
        "{:18} | {:11} | {:8}",
        "Standard Graph",
        format_bytes(standard_stats.total_bytes),
        "100.0%"
    );
    println!(
        "{:18} | {:11} | {:7.1}%",
        "CSR Graph",
        format_bytes(csr.memory_usage()),
        (csr.memory_usage() as f64 / standard_stats.total_bytes as f64) * 100.0
    );
    println!(
        "{:18} | {:11} | {:7.1}%",
        "Bit-packed Graph",
        format_bytes(bitpacked.memory_usage()),
        (bitpacked.memory_usage() as f64 / standard_stats.total_bytes as f64) * 100.0
    );
    println!(
        "{:18} | {:11} | {:7.1}%",
        "Compressed Adj.",
        format_bytes(compressed.memory_usage()),
        (compressed.memory_usage() as f64 / standard_stats.total_bytes as f64) * 100.0
    );

    Ok(())
}

#[allow(dead_code)]
fn fragmentation_analysis() -> Result<(), Box<dyn std::error::Error>> {
    // Create graphs with different degree distributions
    let graphs = vec![
        (
            "Erdős-Rényi (uniform)",
            generators::erdos_renyi_graph(2000, 0.01, None)?,
        ),
        (
            "Barabási-Albert (scale-free)",
            generators::barabasi_albert_graph(2000, 5, None)?,
        ),
        ("Complete", generators::complete_graph(100)?),
    ];

    println!("Graph Type           | Fragmentation | Wasted Memory");
    println!("-".repeat(55));

    for (name, graph) in graphs {
        let fragmentation = MemoryProfiler::analyze_fragmentation(&graph);
        println!(
            "{:19} | {:12.1}% | {:12}",
            name,
            fragmentation.fragmentation_ratio * 100.0,
            format_bytes(fragmentation.wasted_bytes)
        );
    }

    Ok(())
}

#[allow(dead_code)]
fn optimization_suggestions() -> Result<(), Box<dyn std::error::Error>> {
    // Create a graph with poor memory characteristics
    let mut graph = Graph::new();

    // Add nodes with uneven degree distribution
    for i in 0..1000 {
        graph.add_node(i)?;
    }

    // Add many edges to first few nodes (high degree)
    for i in 0..10 {
        for j in 10..1000 {
            graph.add_edge(i, j, 1.0)?;
        }
    }

    let stats = MemoryProfiler::profile_graph(&graph);
    let fragmentation = MemoryProfiler::analyze_fragmentation(&graph);
    let suggestions = suggest_optimizations(&stats, &fragmentation);

    println!(
        "Current memory efficiency: {:.1}%",
        stats.efficiency * 100.0
    );
    println!(
        "Fragmentation ratio: {:.1}%",
        fragmentation.fragmentation_ratio * 100.0
    );
    println!("\nOptimization suggestions:");
    for suggestion in &suggestions.suggestions {
        println!("  • {}", suggestion);
    }
    println!(
        "Potential savings: {}",
        format_bytes(suggestions.potential_savings)
    );

    Ok(())
}

#[allow(dead_code)]
fn real_time_monitoring() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting real-time memory monitoring...");

    let mut profiler = RealTimeMemoryProfiler::new(Duration::from_millis(100))?;
    profiler.start_monitoring()?;

    // Simulate memory-intensive operations
    for i in 0..5 {
        let size = 1000 * (i + 1);
        println!("  Creating graph with {} nodes...", size);
        let _graph = generators::erdos_renyi_graph(size, 0.01, None)?;
        thread::sleep(Duration::from_millis(200));
    }

    thread::sleep(Duration::from_millis(500));

    let metrics = profiler.get_current_metrics()?;
    profiler.stop_monitoring()?;

    println!("Monitoring results:");
    println!("  Peak memory: {}", format_bytes(metrics.peak_memory));
    println!("  Average memory: {}", format_bytes(metrics.average_memory));
    println!("  Growth rate: {:.2} MB/s", metrics.growth_rate_mb_per_sec);
    println!("  Sample count: {}", metrics.sample_count);

    Ok(())
}

#[allow(dead_code)]
fn optimized_builder_test() -> Result<(), Box<dyn std::error::Error>> {
    let n = 10_000;
    let expected_edges = 50_000;

    println!("Building graph with {} nodes, {} edges", n, expected_edges);

    // Time standard construction
    let start = std::time::Instant::now();
    let mut standard_graph = Graph::new();
    for i in 0..n {
        standard_graph.add_node(i)?;
    }
    for i in 0..expected_edges {
        let u = i % n;
        let v = (i + 1) % n;
        standard_graph.add_edge(u, v, 1.0)?;
    }
    let standard_time = start.elapsed();
    let standard_stats = MemoryProfiler::profile_graph(&standard_graph);

    // Time optimized construction
    let start = std::time::Instant::now();
    let mut builder = OptimizedGraphBuilder::new()
        .reserve_nodes(n)
        .reserve_edges(expected_edges)
        .with_estimated_edges_per_node(expected_edges / n);

    for i in 0..n {
        builder.add_node(i);
    }
    for i in 0..expected_edges {
        let u = i % n;
        let v = (i + 1) % n;
        builder.add_edge(u, v, 1.0);
    }
    let optimized_graph = builder.build()?;
    let optimized_time = start.elapsed();
    let optimized_stats = MemoryProfiler::profile_graph(&optimized_graph);

    println!("Construction Method | Time (ms) | Memory | Efficiency");
    println!("-".repeat(55));
    println!(
        "{:18} | {:8.1} | {:6} | {:9.1}%",
        "Standard",
        standard_time.as_millis() as f64,
        format_bytes(standard_stats.total_bytes),
        standard_stats.efficiency * 100.0
    );
    println!(
        "{:18} | {:8.1} | {:6} | {:9.1}%",
        "Optimized Builder",
        optimized_time.as_millis() as f64,
        format_bytes(optimized_stats.total_bytes),
        optimized_stats.efficiency * 100.0
    );

    let speedup = standard_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;
    println!("Speedup: {:.2}x", speedup);

    Ok(())
}

#[allow(dead_code)]
fn format_bytes(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.1} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}
