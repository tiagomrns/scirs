//! Graph measures and metrics
//!
//! This module provides functions for measuring various properties
//! of graphs, including centrality measures, clustering coefficients,
//! and other graph metrics.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use ndarray::{Array1, Array2};

use crate::algorithms::{dijkstra_path, dijkstra_path_digraph};
use crate::base::{DiGraph, EdgeWeight, Graph, Node};
use crate::error::{GraphError, Result};
use petgraph::graph::IndexType;

/// Centrality measure type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CentralityType {
    /// Degree centrality: number of connections
    Degree,
    /// Betweenness centrality: number of shortest paths going through a node
    Betweenness,
    /// Closeness centrality: inverse of the sum of shortest paths to all other nodes
    Closeness,
    /// Eigenvector centrality: nodes connected to important nodes are important
    Eigenvector,
    /// Katz centrality: influenced by all paths, with exponentially decaying weights
    Katz,
    /// PageRank centrality: Google's PageRank algorithm
    PageRank,
    /// Weighted degree centrality: sum of edge weights
    WeightedDegree,
    /// Weighted betweenness centrality: betweenness with edge weights
    WeightedBetweenness,
    /// Weighted closeness centrality: closeness with edge weights
    WeightedCloseness,
}

/// Calculates centrality measures for nodes in an undirected graph
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `centrality_type` - The type of centrality to calculate
///
/// # Returns
/// * `Result<HashMap<N, f64>>` - A map from nodes to their centrality values
///
/// # Time Complexity
/// - Degree: O(V + E)
/// - Betweenness: O(V * E) for unweighted, O(V² + VE) for weighted
/// - Closeness: O(V * (V + E)) using BFS for each node
/// - Eigenvector: O(V + E) per iteration, typically converges in O(log V) iterations
/// - Katz: O(V + E) per iteration, typically converges quickly
/// - PageRank: O(V + E) per iteration, typically converges in ~50 iterations
///
/// # Space Complexity
/// O(V) for storing centrality values plus algorithm-specific requirements
#[allow(dead_code)]
pub fn centrality<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    centrality_type: CentralityType,
) -> Result<HashMap<N, f64>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight
        + num_traits::Zero
        + num_traits::One
        + std::ops::Add<Output = E>
        + PartialOrd
        + Into<f64>
        + std::marker::Copy
        + std::fmt::Debug
        + std::default::Default,
    Ix: petgraph::graph::IndexType,
{
    let nodes = graph.nodes();
    let n = nodes.len();

    if n == 0 {
        return Err(GraphError::InvalidGraph("Empty graph".to_string()));
    }

    match centrality_type {
        CentralityType::Degree => degree_centrality(graph),
        CentralityType::Betweenness => betweenness_centrality(graph),
        CentralityType::Closeness => closeness_centrality(graph),
        CentralityType::Eigenvector => eigenvector_centrality(graph),
        CentralityType::Katz => katz_centrality(graph, 0.1, 1.0), // Default parameters
        CentralityType::PageRank => pagerank_centrality(graph, 0.85, 1e-6), // Default parameters
        CentralityType::WeightedDegree => weighted_degree_centrality(graph),
        CentralityType::WeightedBetweenness => weighted_betweenness_centrality(graph),
        CentralityType::WeightedCloseness => weighted_closeness_centrality(graph),
    }
}

/// Calculates centrality measures for nodes in a directed graph
///
/// # Arguments
/// * `graph` - The directed graph to analyze
/// * `centrality_type` - The type of centrality to calculate
///
/// # Returns
/// * `Result<HashMap<N, f64>>` - A map from nodes to their centrality values
#[allow(dead_code)]
pub fn centrality_digraph<N, E, Ix>(
    graph: &DiGraph<N, E, Ix>,
    centrality_type: CentralityType,
) -> Result<HashMap<N, f64>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight
        + num_traits::Zero
        + num_traits::One
        + std::ops::Add<Output = E>
        + PartialOrd
        + Into<f64>
        + std::marker::Copy
        + std::fmt::Debug
        + std::default::Default,
    Ix: petgraph::graph::IndexType,
{
    let nodes = graph.nodes();
    let n = nodes.len();

    if n == 0 {
        return Err(GraphError::InvalidGraph("Empty graph".to_string()));
    }

    match centrality_type {
        CentralityType::Degree => degree_centrality_digraph(graph),
        CentralityType::Betweenness => betweenness_centrality_digraph(graph),
        CentralityType::Closeness => closeness_centrality_digraph(graph),
        CentralityType::Eigenvector => eigenvector_centrality_digraph(graph),
        CentralityType::Katz => katz_centrality_digraph(graph, 0.1, 1.0), // Default parameters
        CentralityType::PageRank => pagerank_centrality_digraph(graph, 0.85, 1e-6), // Default parameters
        CentralityType::WeightedDegree => weighted_degree_centrality_digraph(graph),
        CentralityType::WeightedBetweenness => weighted_betweenness_centrality_digraph(graph),
        CentralityType::WeightedCloseness => weighted_closeness_centrality_digraph(graph),
    }
}

/// Calculates degree centrality for nodes in an undirected graph
///
/// # Time Complexity
/// O(V + E) where V is vertices and E is edges
///
/// # Space Complexity
/// O(V) for storing the centrality values
#[allow(dead_code)]
fn degree_centrality<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Result<HashMap<N, f64>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let nodes = graph.nodes();
    let n = nodes.len() as f64;
    let degrees = graph.degree_vector();

    let mut centrality = HashMap::new();
    let normalization = if n <= 1.0 { 1.0 } else { n - 1.0 };

    for (i, node) in nodes.iter().enumerate() {
        let degree = degrees[i] as f64;
        centrality.insert((*node).clone(), degree / normalization);
    }

    Ok(centrality)
}

/// Calculates degree centrality for nodes in a directed graph
#[allow(dead_code)]
fn degree_centrality_digraph<N, E, Ix>(graph: &DiGraph<N, E, Ix>) -> Result<HashMap<N, f64>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let nodes = graph.nodes();
    let n = nodes.len() as f64;
    let in_degrees = graph.in_degree_vector();
    let out_degrees = graph.out_degree_vector();

    let mut centrality = HashMap::new();
    let normalization = if n <= 1.0 { 1.0 } else { n - 1.0 };

    for (i, node) in nodes.iter().enumerate() {
        // We use the sum of in-degree and out-degree as the total degree
        let degree = (in_degrees[i] + out_degrees[i]) as f64;
        centrality.insert((*node).clone(), degree / normalization);
    }

    Ok(centrality)
}

/// Calculates betweenness centrality for nodes in an undirected graph
///
/// # Time Complexity
/// O(V * E) for unweighted graphs using Brandes' algorithm
/// O(V² + VE) for weighted graphs
///
/// # Space Complexity
/// O(V + E) for the algorithm's data structures
#[allow(dead_code)]
fn betweenness_centrality<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Result<HashMap<N, f64>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight
        + num_traits::Zero
        + num_traits::One
        + std::ops::Add<Output = E>
        + PartialOrd
        + Into<f64>
        + std::marker::Copy
        + std::fmt::Debug
        + std::default::Default,
    Ix: petgraph::graph::IndexType,
{
    let nodes = graph.nodes();
    let n = nodes.len();

    if n <= 1 {
        let mut result = HashMap::new();
        for node in nodes {
            result.insert(node.clone(), 0.0);
        }
        return Ok(result);
    }

    let mut betweenness = HashMap::new();
    for node in nodes.iter() {
        betweenness.insert((*node).clone(), 0.0);
    }

    // For each pair of nodes, find the shortest paths
    for (i, &s) in nodes.iter().enumerate() {
        for (j, &t) in nodes.iter().enumerate() {
            if i == j {
                continue;
            }

            // Find shortest path from s to t
            if let Ok(Some(path)) = dijkstra_path(graph, s, t) {
                // Skip source and target nodes
                for node in &path.nodes[1..path.nodes.len() - 1] {
                    *betweenness.entry(node.clone()).or_insert(0.0) += 1.0;
                }
            }
        }
    }

    // Normalize by the number of possible paths (excluding the node itself)
    let scale = 1.0 / ((n - 1) * (n - 2)) as f64;
    for val in betweenness.values_mut() {
        *val *= scale;
    }

    Ok(betweenness)
}

/// Calculates betweenness centrality for nodes in a directed graph
#[allow(dead_code)]
fn betweenness_centrality_digraph<N, E, Ix>(graph: &DiGraph<N, E, Ix>) -> Result<HashMap<N, f64>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight
        + num_traits::Zero
        + num_traits::One
        + std::ops::Add<Output = E>
        + PartialOrd
        + Into<f64>
        + std::marker::Copy
        + std::fmt::Debug
        + std::default::Default,
    Ix: petgraph::graph::IndexType,
{
    let nodes = graph.nodes();
    let n = nodes.len();

    if n <= 1 {
        let mut result = HashMap::new();
        for node in nodes {
            result.insert(node.clone(), 0.0);
        }
        return Ok(result);
    }

    let mut betweenness = HashMap::new();
    for node in nodes.iter() {
        betweenness.insert((*node).clone(), 0.0);
    }

    // For each pair of nodes, find the shortest paths
    for (i, &s) in nodes.iter().enumerate() {
        for (j, &t) in nodes.iter().enumerate() {
            if i == j {
                continue;
            }

            // Find shortest path from s to t
            if let Ok(Some(path)) = dijkstra_path_digraph(graph, s, t) {
                // Skip source and target nodes
                for node in &path.nodes[1..path.nodes.len() - 1] {
                    *betweenness.entry(node.clone()).or_insert(0.0) += 1.0;
                }
            }
        }
    }

    // Normalize by the number of possible paths (excluding the node itself)
    let scale = 1.0 / ((n - 1) * (n - 2)) as f64;
    for val in betweenness.values_mut() {
        *val *= scale;
    }

    Ok(betweenness)
}

/// Calculates closeness centrality for nodes in an undirected graph
#[allow(dead_code)]
fn closeness_centrality<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Result<HashMap<N, f64>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight
        + num_traits::Zero
        + num_traits::One
        + std::ops::Add<Output = E>
        + PartialOrd
        + Into<f64>
        + std::marker::Copy
        + std::fmt::Debug
        + std::default::Default,
    Ix: petgraph::graph::IndexType,
{
    let nodes = graph.nodes();
    let n = nodes.len();

    if n <= 1 {
        let mut result = HashMap::new();
        for node in nodes {
            result.insert(node.clone(), 1.0);
        }
        return Ok(result);
    }

    let mut closeness = HashMap::new();

    // For each node, calculate the sum of shortest paths to all other nodes
    for &node in nodes.iter() {
        let mut sum_distances = 0.0;
        let mut reachable_nodes = 0;

        for &other in nodes.iter() {
            if node == other {
                continue;
            }

            // Find shortest path from node to other
            if let Ok(Some(path)) = dijkstra_path(graph, node, other) {
                sum_distances += path.total_weight.into();
                reachable_nodes += 1;
            }
        }

        // If the node can reach other nodes
        if reachable_nodes > 0 {
            // Closeness is 1 / average path length
            let closeness_value = reachable_nodes as f64 / sum_distances;
            // Normalize by the fraction of nodes that are reachable
            let normalized = closeness_value * (reachable_nodes as f64 / (n - 1) as f64);
            closeness.insert(node.clone(), normalized);
        } else {
            closeness.insert(node.clone(), 0.0);
        }
    }

    Ok(closeness)
}

/// Calculates closeness centrality for nodes in a directed graph
#[allow(dead_code)]
fn closeness_centrality_digraph<N, E, Ix>(graph: &DiGraph<N, E, Ix>) -> Result<HashMap<N, f64>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight
        + num_traits::Zero
        + num_traits::One
        + std::ops::Add<Output = E>
        + PartialOrd
        + Into<f64>
        + std::marker::Copy
        + std::fmt::Debug
        + std::default::Default,
    Ix: petgraph::graph::IndexType,
{
    let nodes = graph.nodes();
    let n = nodes.len();

    if n <= 1 {
        let mut result = HashMap::new();
        for node in nodes {
            result.insert(node.clone(), 1.0);
        }
        return Ok(result);
    }

    let mut closeness = HashMap::new();

    // For each node, calculate the sum of shortest paths to all other nodes
    for &node in nodes.iter() {
        let mut sum_distances = 0.0;
        let mut reachable_nodes = 0;

        for &other in nodes.iter() {
            if node == other {
                continue;
            }

            // Find shortest path from node to other
            if let Ok(Some(path)) = dijkstra_path_digraph(graph, node, other) {
                sum_distances += path.total_weight.into();
                reachable_nodes += 1;
            }
        }

        // If the node can reach other nodes
        if reachable_nodes > 0 {
            // Closeness is 1 / average path length
            let closeness_value = reachable_nodes as f64 / sum_distances;
            // Normalize by the fraction of nodes that are reachable
            let normalized = closeness_value * (reachable_nodes as f64 / (n - 1) as f64);
            closeness.insert(node.clone(), normalized);
        } else {
            closeness.insert(node.clone(), 0.0);
        }
    }

    Ok(closeness)
}

/// Calculates eigenvector centrality for nodes in an undirected graph
///
/// Uses the power iteration method.
#[allow(dead_code)]
fn eigenvector_centrality<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Result<HashMap<N, f64>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight + num_traits::Zero + num_traits::One + PartialOrd + Into<f64> + std::marker::Copy,
    Ix: petgraph::graph::IndexType,
{
    let nodes = graph.nodes();
    let n = nodes.len();

    if n == 0 {
        return Err(GraphError::InvalidGraph("Empty _graph".to_string()));
    }

    // Get adjacency matrix with weights converted to f64
    let adj_mat = graph.adjacency_matrix();
    let mut adj_f64 = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            adj_f64[[i, j]] = adj_mat[[i, j]].into();
        }
    }

    // Start with uniform vector
    let mut x = Array1::<f64>::ones(n);
    x.mapv_inplace(|v| v / (n as f64).sqrt()); // Normalize

    // Power iteration
    let max_iter = 100;
    let tol = 1e-6;

    for _ in 0..max_iter {
        // y = A * x
        let y = adj_f64.dot(&x);

        // Compute the norm
        let norm = (y.iter().map(|v| v * v).sum::<f64>()).sqrt();
        if norm < 1e-10 {
            // If the norm is too small, we have a zero vector
            return Err(GraphError::AlgorithmError(
                "Eigenvector centrality computation failed: zero eigenvalue".to_string(),
            ));
        }

        // Normalize y
        let y_norm = y / norm;

        // Check for convergence
        let diff = (&y_norm - &x).iter().map(|v| v.abs()).sum::<f64>();
        if diff < tol {
            // We've converged, create the result
            let mut result = HashMap::new();
            for (i, &node) in nodes.iter().enumerate() {
                result.insert(node.clone(), y_norm[i]);
            }
            return Ok(result);
        }

        // Update x for next iteration
        x = y_norm;
    }

    // We've reached max iterations without converging
    Err(GraphError::AlgorithmError(
        "Eigenvector centrality computation did not converge".to_string(),
    ))
}

/// Calculates eigenvector centrality for nodes in a directed graph
///
/// Uses the power iteration method.
#[allow(dead_code)]
fn eigenvector_centrality_digraph<N, E, Ix>(graph: &DiGraph<N, E, Ix>) -> Result<HashMap<N, f64>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight + num_traits::Zero + num_traits::One + PartialOrd + Into<f64> + std::marker::Copy,
    Ix: petgraph::graph::IndexType,
{
    let nodes = graph.nodes();
    let n = nodes.len();

    if n == 0 {
        return Err(GraphError::InvalidGraph("Empty _graph".to_string()));
    }

    // Get adjacency matrix with weights converted to f64
    let adj_mat = graph.adjacency_matrix();
    let mut adj_f64 = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            adj_f64[[i, j]] = adj_mat[[i, j]].into();
        }
    }

    // Start with uniform vector
    let mut x = Array1::<f64>::ones(n);
    x.mapv_inplace(|v| v / (n as f64).sqrt()); // Normalize

    // Power iteration
    let max_iter = 100;
    let tol = 1e-6;

    for _ in 0..max_iter {
        // y = A * x
        let y = adj_f64.dot(&x);

        // Compute the norm
        let norm = (y.iter().map(|v| v * v).sum::<f64>()).sqrt();
        if norm < 1e-10 {
            // If the norm is too small, we have a zero vector
            return Err(GraphError::AlgorithmError(
                "Eigenvector centrality computation failed: zero eigenvalue".to_string(),
            ));
        }

        // Normalize y
        let y_norm = y / norm;

        // Check for convergence
        let diff = (&y_norm - &x).iter().map(|v| v.abs()).sum::<f64>();
        if diff < tol {
            // We've converged, create the result
            let mut result = HashMap::new();
            for (i, &node) in nodes.iter().enumerate() {
                result.insert(node.clone(), y_norm[i]);
            }
            return Ok(result);
        }

        // Update x for next iteration
        x = y_norm;
    }

    // We've reached max iterations without converging
    Err(GraphError::AlgorithmError(
        "Eigenvector centrality computation did not converge".to_string(),
    ))
}

/// Calculates the local clustering coefficient for each node in an undirected graph
///
/// The clustering coefficient measures how close a node's neighbors are to being a clique.
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// * `Result<HashMap<N, f64>>` - A map from nodes to their clustering coefficients
#[allow(dead_code)]
pub fn clustering_coefficient<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Result<HashMap<N, f64>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let mut coefficients = HashMap::new();

    for (idx, &node) in graph.nodes().iter().enumerate() {
        // Get the neighbors of this node
        let node_idx = petgraph::graph::NodeIndex::new(idx);
        let neighbors: HashSet<_> = graph.inner().neighbors(node_idx).collect();

        let k = neighbors.len();

        // If the node has less than 2 neighbors, its clustering coefficient is 0
        if k < 2 {
            coefficients.insert(node.clone(), 0.0);
            continue;
        }

        // Count the number of edges between neighbors
        let mut edge_count = 0;

        for &n1 in &neighbors {
            for &n2 in &neighbors {
                if n1 != n2 && graph.inner().contains_edge(n1, n2) {
                    edge_count += 1;
                }
            }
        }

        // Each edge is counted twice (once from each direction)
        edge_count /= 2;

        // Calculate the clustering coefficient
        let possible_edges = k * (k - 1) / 2;
        let coefficient = edge_count as f64 / possible_edges as f64;

        coefficients.insert(node.clone(), coefficient);
    }

    Ok(coefficients)
}

/// Calculates the global clustering coefficient (transitivity) of a graph
///
/// This is the ratio of the number of closed triplets to the total number of triplets.
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// * `Result<f64>` - The global clustering coefficient
#[allow(dead_code)]
pub fn graph_density<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Result<f64>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let n = graph.node_count();

    if n <= 1 {
        return Err(GraphError::InvalidGraph(
            "Graph density undefined for graphs with 0 or 1 nodes".to_string(),
        ));
    }

    let m = graph.edge_count();
    let possible_edges = n * (n - 1) / 2;

    Ok(m as f64 / possible_edges as f64)
}

/// Calculates the density of a directed graph
///
/// # Arguments
/// * `graph` - The directed graph to analyze
///
/// # Returns
/// * `Result<f64>` - The graph density
#[allow(dead_code)]
pub fn graph_density_digraph<N, E, Ix>(graph: &DiGraph<N, E, Ix>) -> Result<f64>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let n = graph.node_count();

    if n <= 1 {
        return Err(GraphError::InvalidGraph(
            "Graph density undefined for graphs with 0 or 1 nodes".to_string(),
        ));
    }

    let m = graph.edge_count();
    let possible_edges = n * (n - 1); // In directed graphs, there can be n(n-1) edges

    Ok(m as f64 / possible_edges as f64)
}

/// Calculates Katz centrality for nodes in an undirected graph
///
/// Katz centrality is a variant of eigenvector centrality that considers all paths between nodes,
/// with paths of longer length given exponentially decreasing weights.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `alpha` - Attenuation factor (must be smaller than the reciprocal of the largest eigenvalue)
/// * `beta` - Weight attributed to the immediate neighborhood
///
/// # Returns
/// * `Result<HashMap<N, f64>>` - Katz centrality values
#[allow(dead_code)]
pub fn katz_centrality<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    alpha: f64,
    beta: f64,
) -> Result<HashMap<N, f64>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight + num_traits::Zero + num_traits::One + Into<f64> + Copy,
    Ix: petgraph::graph::IndexType,
{
    let nodes = graph.nodes();
    let n = nodes.len();

    if n == 0 {
        return Err(GraphError::InvalidGraph("Empty graph".to_string()));
    }

    // Get adjacency matrix
    let adj_mat = graph.adjacency_matrix();
    let mut adj_f64 = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            adj_f64[[i, j]] = adj_mat[[i, j]].into();
        }
    }

    // Create identity matrix
    let mut identity = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        identity[[i, i]] = 1.0;
    }

    // Compute (I - α*A) - not used in iterative method but kept for reference
    let _factor_matrix = &identity - &(alpha * &adj_f64);

    // Create beta vector (all entries = beta)
    let beta_vec = Array1::<f64>::from_elem(n, beta);

    // We need to solve (I - α*A) * c = β*1
    // For simplicity, we'll use an iterative approach (power method variant)
    let mut centrality_vec = Array1::<f64>::ones(n);
    let max_iter = 100;
    let tol = 1e-6;

    for _ in 0..max_iter {
        // c_new = α*A*c + β*1
        let new_centrality = alpha * adj_f64.dot(&centrality_vec) + &beta_vec;

        // Check for convergence
        let diff = (&new_centrality - &centrality_vec)
            .iter()
            .map(|v| v.abs())
            .sum::<f64>();
        if diff < tol {
            centrality_vec = new_centrality;
            break;
        }

        centrality_vec = new_centrality;
    }

    // Convert to HashMap
    let mut result = HashMap::new();
    for (i, &node) in nodes.iter().enumerate() {
        result.insert(node.clone(), centrality_vec[i]);
    }

    Ok(result)
}

/// Calculates Katz centrality for nodes in a directed graph
///
/// # Arguments
/// * `graph` - The directed graph to analyze
/// * `alpha` - Attenuation factor
/// * `beta` - Weight attributed to the immediate neighborhood
///
/// # Returns
/// * `Result<HashMap<N, f64>>` - Katz centrality values
#[allow(dead_code)]
pub fn katz_centrality_digraph<N, E, Ix>(
    graph: &DiGraph<N, E, Ix>,
    alpha: f64,
    beta: f64,
) -> Result<HashMap<N, f64>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight + num_traits::Zero + num_traits::One + Into<f64> + Copy,
    Ix: petgraph::graph::IndexType,
{
    let nodes = graph.nodes();
    let n = nodes.len();

    if n == 0 {
        return Err(GraphError::InvalidGraph("Empty graph".to_string()));
    }

    // Get adjacency matrix
    let adj_mat = graph.adjacency_matrix();
    let mut adj_f64 = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            adj_f64[[i, j]] = adj_mat[[i, j]].into();
        }
    }

    // Create beta vector
    let beta_vec = Array1::<f64>::from_elem(n, beta);

    // Iterative approach
    let mut centrality_vec = Array1::<f64>::ones(n);
    let max_iter = 100;
    let tol = 1e-6;

    for _ in 0..max_iter {
        // c_new = α*A^T*c + β*1 (transpose for incoming links)
        let new_centrality = alpha * adj_f64.t().dot(&centrality_vec) + &beta_vec;

        // Check for convergence
        let diff = (&new_centrality - &centrality_vec)
            .iter()
            .map(|v| v.abs())
            .sum::<f64>();
        if diff < tol {
            centrality_vec = new_centrality;
            break;
        }

        centrality_vec = new_centrality;
    }

    // Convert to HashMap
    let mut result = HashMap::new();
    for (i, &node) in nodes.iter().enumerate() {
        result.insert(node.clone(), centrality_vec[i]);
    }

    Ok(result)
}

/// Calculates PageRank centrality for nodes in an undirected graph
///
/// PageRank is Google's famous algorithm for ranking web pages.
/// For undirected graphs, we treat each edge as bidirectional.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `damping` - Damping parameter (typically 0.85)
/// * `tolerance` - Convergence tolerance
///
/// # Returns
/// * `Result<HashMap<N, f64>>` - PageRank values
#[allow(dead_code)]
pub fn pagerank_centrality<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    damping: f64,
    tolerance: f64,
) -> Result<HashMap<N, f64>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let nodes = graph.nodes();
    let n = nodes.len();

    if n == 0 {
        return Err(GraphError::InvalidGraph("Empty graph".to_string()));
    }

    // Initialize PageRank values
    let mut pagerank = Array1::<f64>::from_elem(n, 1.0 / n as f64);
    let max_iter = 100;

    // Get degree of each node for normalization
    let degrees = graph.degree_vector();

    for _ in 0..max_iter {
        let mut new_pagerank = Array1::<f64>::from_elem(n, (1.0 - damping) / n as f64);

        // Calculate PageRank contribution from each node
        for (i, node_idx) in graph.inner().node_indices().enumerate() {
            let node_degree = degrees[i] as f64;
            if node_degree > 0.0 {
                let contribution = damping * pagerank[i] / node_degree;

                // Distribute to all neighbors
                for neighbor_idx in graph.inner().neighbors(node_idx) {
                    let neighbor_i = neighbor_idx.index();
                    new_pagerank[neighbor_i] += contribution;
                }
            }
        }

        // Check for convergence
        let diff = (&new_pagerank - &pagerank)
            .iter()
            .map(|v| v.abs())
            .sum::<f64>();
        if diff < tolerance {
            pagerank = new_pagerank;
            break;
        }

        pagerank = new_pagerank;
    }

    // Convert to HashMap
    let mut result = HashMap::new();
    for (i, &node) in nodes.iter().enumerate() {
        result.insert(node.clone(), pagerank[i]);
    }

    Ok(result)
}

/// Calculates PageRank centrality using parallel processing for large graphs
///
/// This parallel implementation of PageRank uses scirs2-core parallel operations
/// to accelerate computation on large graphs. It's particularly effective when
/// the graph has >10,000 nodes and sufficient CPU cores are available.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `damping` - Damping parameter (typically 0.85)
/// * `tolerance` - Convergence tolerance
/// * `max_iterations` - Maximum number of iterations (default: 100)
///
/// # Returns
/// * `Result<HashMap<N, f64>>` - PageRank values
///
/// # Time Complexity
/// O((V + E) * k / p) where V is nodes, E is edges, k is iterations,
/// and p is the number of parallel threads.
///
/// # Space Complexity
/// O(V + t) where t is the number of threads for thread-local storage.
///
/// # Performance Notes
/// - Best performance gain on graphs with >10,000 nodes
/// - Requires multiple CPU cores for meaningful speedup
/// - Memory access patterns optimized for cache efficiency
#[allow(dead_code)]
pub fn parallel_pagerank_centrality<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    damping: f64,
    tolerance: f64,
    max_iterations: Option<usize>,
) -> Result<HashMap<N, f64>>
where
    N: Node + Send + Sync + std::fmt::Debug,
    E: EdgeWeight + Send + Sync,
    Ix: petgraph::graph::IndexType + Send + Sync,
{
    use scirs2_core::parallel_ops::*;
    use std::sync::{Arc, Mutex};

    let nodes = graph.nodes();
    let n = nodes.len();

    if n == 0 {
        return Err(GraphError::InvalidGraph("Empty graph".to_string()));
    }

    // For small graphs, use sequential implementation
    if n < 1000 {
        return pagerank_centrality(graph, damping, tolerance);
    }

    let max_iter = max_iterations.unwrap_or(100);

    // Initialize PageRank values
    let mut pagerank = Array1::<f64>::from_elem(n, 1.0 / n as f64);

    // Get degree of each node for normalization
    let degrees = graph.degree_vector();

    // Pre-compute neighbor indices for efficient parallel access
    let neighbor_lists: Vec<Vec<usize>> = (0..n)
        .map(|i| {
            let node_idx = petgraph::graph::NodeIndex::new(i);
            graph
                .inner()
                .neighbors(node_idx)
                .map(|neighbor_idx| neighbor_idx.index())
                .collect()
        })
        .collect();

    for _iteration in 0..max_iter {
        // Parallel computation of new PageRank values
        let new_pagerank = Arc::new(Mutex::new(Array1::<f64>::from_elem(
            n,
            (1.0 - damping) / n as f64,
        )));

        // Use parallel iteration for PageRank calculation
        (0..n).into_par_iter().for_each(|i| {
            let node_degree = degrees[i] as f64;
            if node_degree > 0.0 {
                let contribution = damping * pagerank[i] / node_degree;

                // Distribute to all neighbors
                let mut local_updates = Vec::new();
                for &neighbor_i in &neighbor_lists[i] {
                    local_updates.push((neighbor_i, contribution));
                }

                // Apply updates atomically
                if !local_updates.is_empty() {
                    let mut new_pr = new_pagerank.lock().unwrap();
                    for (neighbor_i, contrib) in local_updates {
                        new_pr[neighbor_i] += contrib;
                    }
                }
            }
        });

        let new_pagerank = Arc::try_unwrap(new_pagerank).unwrap().into_inner().unwrap();

        // Convergence check
        let diff = pagerank
            .iter()
            .zip(new_pagerank.iter())
            .map(|(old, new)| (new - old).abs())
            .sum::<f64>();

        if diff < tolerance {
            pagerank = new_pagerank;
            break;
        }

        pagerank = new_pagerank;
    }

    // Conversion to HashMap
    let result: HashMap<N, f64> = nodes
        .iter()
        .enumerate()
        .map(|(i, node)| ((*node).clone(), pagerank[i]))
        .collect();

    Ok(result)
}

/// HITS (Hyperlink-Induced Topic Search) algorithm result
#[derive(Debug, Clone)]
pub struct HitsScores<N: Node> {
    /// Authority scores for each node
    pub authorities: HashMap<N, f64>,
    /// Hub scores for each node
    pub hubs: HashMap<N, f64>,
}

/// Compute HITS algorithm for a directed graph
///
/// The HITS algorithm computes two scores for each node:
/// - Authority score: nodes that are pointed to by many hubs
/// - Hub score: nodes that point to many authorities
///
/// # Arguments
/// * `graph` - The directed graph
/// * `max_iter` - Maximum number of iterations
/// * `tolerance` - Convergence tolerance
///
/// # Returns
/// * HitsScores containing authority and hub scores for each node
#[allow(dead_code)]
pub fn hits_algorithm<N, E, Ix>(
    graph: &DiGraph<N, E, Ix>,
    max_iter: usize,
    tolerance: f64,
) -> Result<HitsScores<N>>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight,
    Ix: IndexType,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();

    if n == 0 {
        return Ok(HitsScores {
            authorities: HashMap::new(),
            hubs: HashMap::new(),
        });
    }

    // Initialize scores
    let mut authorities = vec![1.0 / (n as f64).sqrt(); n];
    let mut hubs = vec![1.0 / (n as f64).sqrt(); n];
    let mut new_authorities = vec![0.0; n];
    let mut new_hubs = vec![0.0; n];

    // Create node index mapping
    let node_to_idx: HashMap<N, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();

    // Iterate until convergence
    for _ in 0..max_iter {
        // Update authority scores
        new_authorities.fill(0.0);
        for (i, node) in nodes.iter().enumerate() {
            // Authority score is sum of hub scores of nodes pointing to it
            if let Ok(predecessors) = graph.predecessors(node) {
                for pred in predecessors {
                    if let Some(&pred_idx) = node_to_idx.get(&pred) {
                        new_authorities[i] += hubs[pred_idx];
                    }
                }
            }
        }

        // Update hub scores
        new_hubs.fill(0.0);
        for (i, node) in nodes.iter().enumerate() {
            // Hub score is sum of authority scores of nodes it points to
            if let Ok(successors) = graph.successors(node) {
                for succ in successors {
                    if let Some(&succ_idx) = node_to_idx.get(&succ) {
                        new_hubs[i] += authorities[succ_idx];
                    }
                }
            }
        }

        // Normalize scores
        let auth_norm: f64 = new_authorities.iter().map(|x| x * x).sum::<f64>().sqrt();
        let hub_norm: f64 = new_hubs.iter().map(|x| x * x).sum::<f64>().sqrt();

        if auth_norm > 0.0 {
            for score in &mut new_authorities {
                *score /= auth_norm;
            }
        }

        if hub_norm > 0.0 {
            for score in &mut new_hubs {
                *score /= hub_norm;
            }
        }

        // Check convergence
        let auth_diff: f64 = authorities
            .iter()
            .zip(&new_authorities)
            .map(|(old, new)| (old - new).abs())
            .sum();
        let hub_diff: f64 = hubs
            .iter()
            .zip(&new_hubs)
            .map(|(old, new)| (old - new).abs())
            .sum();

        if auth_diff < tolerance && hub_diff < tolerance {
            break;
        }

        // Update scores
        authorities.copy_from_slice(&new_authorities);
        hubs.copy_from_slice(&new_hubs);
    }

    // Convert to HashMap
    let authority_map = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), authorities[i]))
        .collect();
    let hub_map = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), hubs[i]))
        .collect();

    Ok(HitsScores {
        authorities: authority_map,
        hubs: hub_map,
    })
}

/// Calculates PageRank centrality for nodes in a directed graph
///
/// # Arguments
/// * `graph` - The directed graph to analyze
/// * `damping` - Damping parameter (typically 0.85)
/// * `tolerance` - Convergence tolerance
///
/// # Returns
/// * `Result<HashMap<N, f64>>` - PageRank values
#[allow(dead_code)]
pub fn pagerank_centrality_digraph<N, E, Ix>(
    graph: &DiGraph<N, E, Ix>,
    damping: f64,
    tolerance: f64,
) -> Result<HashMap<N, f64>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let nodes = graph.nodes();
    let n = nodes.len();

    if n == 0 {
        return Err(GraphError::InvalidGraph("Empty graph".to_string()));
    }

    // Initialize PageRank values
    let mut pagerank = Array1::<f64>::from_elem(n, 1.0 / n as f64);
    let max_iter = 100;

    // Get out-degree of each node for normalization
    let out_degrees = graph.out_degree_vector();

    for _ in 0..max_iter {
        let mut new_pagerank = Array1::<f64>::from_elem(n, (1.0 - damping) / n as f64);

        // Calculate PageRank contribution from each node
        for (i, node_idx) in graph.inner().node_indices().enumerate() {
            let node_out_degree = out_degrees[i] as f64;
            if node_out_degree > 0.0 {
                let contribution = damping * pagerank[i] / node_out_degree;

                // Distribute to all outgoing neighbors
                for neighbor_idx in graph
                    .inner()
                    .neighbors_directed(node_idx, petgraph::Direction::Outgoing)
                {
                    let neighbor_i = neighbor_idx.index();
                    new_pagerank[neighbor_i] += contribution;
                }
            } else {
                // Handle dangling nodes by distributing equally to all nodes
                let contribution = damping * pagerank[i] / n as f64;
                for j in 0..n {
                    new_pagerank[j] += contribution;
                }
            }
        }

        // Check for convergence
        let diff = (&new_pagerank - &pagerank)
            .iter()
            .map(|v| v.abs())
            .sum::<f64>();
        if diff < tolerance {
            pagerank = new_pagerank;
            break;
        }

        pagerank = new_pagerank;
    }

    // Convert to HashMap
    let mut result = HashMap::new();
    for (i, &node) in nodes.iter().enumerate() {
        result.insert(node.clone(), pagerank[i]);
    }

    Ok(result)
}

/// Calculates weighted degree centrality for undirected graphs
///
/// Weighted degree centrality is the sum of the weights of all edges incident to a node.
#[allow(dead_code)]
fn weighted_degree_centrality<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Result<HashMap<N, f64>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Clone,
    Ix: IndexType,
{
    let mut centrality = HashMap::new();

    for node in graph.nodes() {
        let mut weight_sum = 0.0;

        if let Ok(neighbors) = graph.neighbors(node) {
            for neighbor in neighbors {
                if let Ok(weight) = graph.edge_weight(node, &neighbor) {
                    weight_sum += weight.into();
                }
            }
        }

        centrality.insert(node.clone(), weight_sum);
    }

    Ok(centrality)
}

/// Calculates weighted degree centrality for directed graphs
#[allow(dead_code)]
fn weighted_degree_centrality_digraph<N, E, Ix>(
    graph: &DiGraph<N, E, Ix>,
) -> Result<HashMap<N, f64>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Clone,
    Ix: IndexType,
{
    let mut centrality = HashMap::new();

    for node in graph.nodes() {
        let mut in_weight = 0.0;
        let mut out_weight = 0.0;

        // Sum incoming edge weights
        if let Ok(predecessors) = graph.predecessors(node) {
            for pred in predecessors {
                if let Ok(weight) = graph.edge_weight(&pred, node) {
                    in_weight += weight.into();
                }
            }
        }

        // Sum outgoing edge weights
        if let Ok(successors) = graph.successors(node) {
            for succ in successors {
                if let Ok(weight) = graph.edge_weight(node, &succ) {
                    out_weight += weight.into();
                }
            }
        }

        centrality.insert(node.clone(), in_weight + out_weight);
    }

    Ok(centrality)
}

/// Calculates weighted betweenness centrality for undirected graphs
///
/// Uses Dijkstra's algorithm to find shortest weighted paths between all pairs of nodes.
#[allow(dead_code)]
fn weighted_betweenness_centrality<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Result<HashMap<N, f64>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight
        + Into<f64>
        + num_traits::Zero
        + num_traits::One
        + std::ops::Add<Output = E>
        + PartialOrd
        + Copy
        + std::fmt::Debug
        + Default,
    Ix: IndexType,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();
    let mut centrality: HashMap<N, f64> = nodes.iter().map(|n| (n.clone(), 0.0)).collect();

    // For each pair of nodes, find shortest weighted paths
    for source in &nodes {
        for target in &nodes {
            if source != target {
                // Find shortest weighted path
                if let Ok(Some(path)) = dijkstra_path(graph, source, target) {
                    // Count how many times each intermediate node appears
                    for intermediate in &path.nodes[1..path.nodes.len() - 1] {
                        *centrality.get_mut(intermediate).unwrap() += 1.0;
                    }
                }
            }
        }
    }

    // Normalize by (n-1)(n-2) for undirected graphs
    if n > 2 {
        let normalization = ((n - 1) * (n - 2)) as f64;
        for value in centrality.values_mut() {
            *value /= normalization;
        }
    }

    Ok(centrality)
}

/// Calculates weighted betweenness centrality for directed graphs
#[allow(dead_code)]
fn weighted_betweenness_centrality_digraph<N, E, Ix>(
    graph: &DiGraph<N, E, Ix>,
) -> Result<HashMap<N, f64>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight
        + Into<f64>
        + num_traits::Zero
        + num_traits::One
        + std::ops::Add<Output = E>
        + PartialOrd
        + Copy
        + std::fmt::Debug
        + Default,
    Ix: IndexType,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();
    let mut centrality: HashMap<N, f64> = nodes.iter().map(|n| (n.clone(), 0.0)).collect();

    // For each pair of nodes, find shortest weighted paths
    for source in &nodes {
        for target in &nodes {
            if source != target {
                // Find shortest weighted path in directed graph
                if let Ok(Some(path)) = dijkstra_path_digraph(graph, source, target) {
                    // Count how many times each intermediate node appears
                    for intermediate in &path.nodes[1..path.nodes.len() - 1] {
                        *centrality.get_mut(intermediate).unwrap() += 1.0;
                    }
                }
            }
        }
    }

    // Normalize by (n-1)(n-2) for directed graphs
    if n > 2 {
        let normalization = ((n - 1) * (n - 2)) as f64;
        for value in centrality.values_mut() {
            *value /= normalization;
        }
    }

    Ok(centrality)
}

/// Calculates weighted closeness centrality for undirected graphs
///
/// Weighted closeness centrality uses the shortest weighted path distances.
#[allow(dead_code)]
fn weighted_closeness_centrality<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Result<HashMap<N, f64>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight
        + Into<f64>
        + num_traits::Zero
        + num_traits::One
        + std::ops::Add<Output = E>
        + PartialOrd
        + Copy
        + std::fmt::Debug
        + Default,
    Ix: IndexType,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();
    let mut centrality = HashMap::new();

    for node in &nodes {
        let mut total_distance = 0.0;
        let mut reachable_count = 0;

        // Calculate shortest weighted paths to all other nodes
        for other in &nodes {
            if node != other {
                if let Ok(Some(path)) = dijkstra_path(graph, node, other) {
                    let distance: f64 = path.total_weight.into();
                    total_distance += distance;
                    reachable_count += 1;
                }
            }
        }

        if reachable_count > 0 && total_distance > 0.0 {
            let closeness = reachable_count as f64 / total_distance;
            // Normalize by (n-1) to get values between 0 and 1
            let normalized_closeness = if n > 1 {
                closeness * (reachable_count as f64 / (n - 1) as f64)
            } else {
                closeness
            };
            centrality.insert(node.clone(), normalized_closeness);
        } else {
            centrality.insert(node.clone(), 0.0);
        }
    }

    Ok(centrality)
}

/// Calculates weighted closeness centrality for directed graphs
#[allow(dead_code)]
fn weighted_closeness_centrality_digraph<N, E, Ix>(
    graph: &DiGraph<N, E, Ix>,
) -> Result<HashMap<N, f64>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight
        + Into<f64>
        + num_traits::Zero
        + num_traits::One
        + std::ops::Add<Output = E>
        + PartialOrd
        + Copy
        + std::fmt::Debug
        + Default,
    Ix: IndexType,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();
    let mut centrality = HashMap::new();

    for node in &nodes {
        let mut total_distance = 0.0;
        let mut reachable_count = 0;

        // Calculate shortest weighted paths to all other nodes
        for other in &nodes {
            if node != other {
                if let Ok(Some(path)) = dijkstra_path_digraph(graph, node, other) {
                    let distance: f64 = path.total_weight.into();
                    total_distance += distance;
                    reachable_count += 1;
                }
            }
        }

        if reachable_count > 0 && total_distance > 0.0 {
            let closeness = reachable_count as f64 / total_distance;
            // Normalize by (n-1) to get values between 0 and 1
            let normalized_closeness = if n > 1 {
                closeness * (reachable_count as f64 / (n - 1) as f64)
            } else {
                closeness
            };
            centrality.insert(node.clone(), normalized_closeness);
        } else {
            centrality.insert(node.clone(), 0.0);
        }
    }

    Ok(centrality)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_degree_centrality() {
        let mut graph: Graph<char, f64> = Graph::new();

        // Create a star graph with A in the center
        // A -- B
        // |
        // |
        // C -- D

        graph.add_edge('A', 'B', 1.0).unwrap();
        graph.add_edge('A', 'C', 1.0).unwrap();
        graph.add_edge('C', 'D', 1.0).unwrap();

        let centrality = centrality(&graph, CentralityType::Degree).unwrap();

        // A has 2 connections out of 3 possible, so centrality = 2/3
        // B has 1 connection out of 3 possible, so centrality = 1/3
        // C has 2 connections out of 3 possible, so centrality = 2/3
        // D has 1 connection out of 3 possible, so centrality = 1/3

        assert_eq!(centrality[&'A'], 2.0 / 3.0);
        assert_eq!(centrality[&'B'], 1.0 / 3.0);
        assert_eq!(centrality[&'C'], 2.0 / 3.0);
        assert_eq!(centrality[&'D'], 1.0 / 3.0);
    }

    #[test]
    fn test_clustering_coefficient() {
        let mut graph: Graph<i32, f64> = Graph::new();

        // Create a graph:
        // 1 -- 2 -- 3
        // |         |
        // +----4----+
        //      |
        //      5

        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(2, 3, 1.0).unwrap();
        graph.add_edge(3, 4, 1.0).unwrap();
        graph.add_edge(4, 1, 1.0).unwrap();
        graph.add_edge(4, 5, 1.0).unwrap();

        let coefficients = clustering_coefficient(&graph).unwrap();

        // Node 1 has neighbors 2 and 4, and they're not connected, so coefficient = 0/1 = 0
        assert_eq!(coefficients[&1], 0.0);

        // Node 2 has neighbors 1 and 3, and they're not connected, so coefficient = 0/1 = 0
        assert_eq!(coefficients[&2], 0.0);

        // Node 3 has neighbors 2 and 4, and they're not connected, so coefficient = 0/1 = 0
        assert_eq!(coefficients[&3], 0.0);

        // Node 4 has neighbors 1, 3, and 5.
        // Neighbors 1 and 3 are not directly connected, and 5 is not connected to either 1 or 3
        // So coefficient = 0/3 = 0
        assert_eq!(coefficients[&4], 0.0);

        // Node 5 has only one neighbor (4), so coefficient = 0
        assert_eq!(coefficients[&5], 0.0);

        // Now let's add an edge to create a triangle
        graph.add_edge(1, 3, 1.0).unwrap();

        let coefficients = clustering_coefficient(&graph).unwrap();

        // Now nodes 1, 3, and 4 form a triangle
        // Node 4 has 3 neighbors (1, 3, 5) with 1 edge between them (1-3)
        // Possible edges: 3 choose 2 = 3
        // So coefficient = 1/3
        assert!((coefficients[&4] - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_graph_density() {
        let mut graph: Graph<i32, f64> = Graph::new();

        // Empty graph should error
        assert!(graph_density(&graph).is_err());

        // Add one node
        graph.add_node(1);

        // Graph with one node should error
        assert!(graph_density(&graph).is_err());

        // Create a graph with 4 nodes and 3 edges
        graph.add_node(2);
        graph.add_node(3);
        graph.add_node(4);

        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(2, 3, 1.0).unwrap();
        graph.add_edge(3, 4, 1.0).unwrap();

        // 3 edges out of 6 possible edges (4 choose 2 = 6)
        let density = graph_density(&graph).unwrap();
        assert_eq!(density, 0.5);

        // Add 3 more edges to make a complete graph
        graph.add_edge(1, 3, 1.0).unwrap();
        graph.add_edge(1, 4, 1.0).unwrap();
        graph.add_edge(2, 4, 1.0).unwrap();

        // 6 edges out of 6 possible edges
        let density = graph_density(&graph).unwrap();
        assert_eq!(density, 1.0);
    }

    #[test]
    fn test_katz_centrality() {
        let mut graph: Graph<char, f64> = Graph::new();

        // Create a simple star graph with A in the center
        graph.add_edge('A', 'B', 1.0).unwrap();
        graph.add_edge('A', 'C', 1.0).unwrap();
        graph.add_edge('A', 'D', 1.0).unwrap();

        let centrality = katz_centrality(&graph, 0.1, 1.0).unwrap();

        // The center node should have higher Katz centrality
        assert!(centrality[&'A'] > centrality[&'B']);
        assert!(centrality[&'A'] > centrality[&'C']);
        assert!(centrality[&'A'] > centrality[&'D']);

        // All leaf nodes should have similar centrality
        assert!((centrality[&'B'] - centrality[&'C']).abs() < 0.1);
        assert!((centrality[&'B'] - centrality[&'D']).abs() < 0.1);
    }

    #[test]
    fn test_pagerank_centrality() {
        let mut graph: Graph<char, f64> = Graph::new();

        // Create a simple triangle
        graph.add_edge('A', 'B', 1.0).unwrap();
        graph.add_edge('B', 'C', 1.0).unwrap();
        graph.add_edge('C', 'A', 1.0).unwrap();

        let centrality = pagerank_centrality(&graph, 0.85, 1e-6).unwrap();

        // All nodes should have equal PageRank in a symmetric triangle
        let values: Vec<f64> = centrality.values().cloned().collect();
        let expected = 1.0 / 3.0; // Should sum to 1, so each gets 1/3

        for &value in &values {
            assert!((value - expected).abs() < 0.1);
        }

        // Check that PageRank values sum to approximately 1
        let sum: f64 = values.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_pagerank_centrality_digraph() {
        let mut graph: DiGraph<char, f64> = DiGraph::new();

        // Create a directed path: A -> B -> C
        graph.add_edge('A', 'B', 1.0).unwrap();
        graph.add_edge('B', 'C', 1.0).unwrap();

        let centrality = pagerank_centrality_digraph(&graph, 0.85, 1e-6).unwrap();

        // C should have the highest PageRank (receives links but doesn't give any)
        // A should have the lowest (gives links but doesn't receive any except random jumps)
        assert!(centrality[&'C'] > centrality[&'B']);
        assert!(centrality[&'B'] > centrality[&'A']);
    }

    #[test]
    fn test_centrality_enum_katz_pagerank() {
        let mut graph: Graph<char, f64> = Graph::new();

        graph.add_edge('A', 'B', 1.0).unwrap();
        graph.add_edge('A', 'C', 1.0).unwrap();

        // Test that the enum-based centrality function works for new types
        let katz_result = centrality(&graph, CentralityType::Katz).unwrap();
        let pagerank_result = centrality(&graph, CentralityType::PageRank).unwrap();

        // Both should return valid results
        assert_eq!(katz_result.len(), 3);
        assert_eq!(pagerank_result.len(), 3);

        // All values should be positive
        for value in katz_result.values() {
            assert!(*value > 0.0);
        }
        for value in pagerank_result.values() {
            assert!(*value > 0.0);
        }
    }

    #[test]
    fn test_hits_algorithm() {
        let mut graph: DiGraph<char, f64> = DiGraph::new();

        // Create a small web graph
        // A and B are hubs (point to many pages)
        // C and D are authorities (pointed to by many pages)
        graph.add_edge('A', 'C', 1.0).unwrap();
        graph.add_edge('A', 'D', 1.0).unwrap();
        graph.add_edge('B', 'C', 1.0).unwrap();
        graph.add_edge('B', 'D', 1.0).unwrap();
        // E is both a hub and authority
        graph.add_edge('E', 'C', 1.0).unwrap();
        graph.add_edge('B', 'E', 1.0).unwrap();

        let hits = hits_algorithm(&graph, 100, 1e-6).unwrap();

        // Check that we have scores for all nodes
        assert_eq!(hits.authorities.len(), 5);
        assert_eq!(hits.hubs.len(), 5);

        // C and D should have high authority scores
        assert!(hits.authorities[&'C'] > hits.authorities[&'A']);
        assert!(hits.authorities[&'D'] > hits.authorities[&'A']);

        // A and B should have high hub scores
        assert!(hits.hubs[&'A'] > hits.hubs[&'C']);
        assert!(hits.hubs[&'B'] > hits.hubs[&'C']);

        // Check that scores are normalized (sum of squares = 1)
        let auth_norm: f64 = hits.authorities.values().map(|&x| x * x).sum::<f64>();
        let hub_norm: f64 = hits.hubs.values().map(|&x| x * x).sum::<f64>();
        assert!((auth_norm - 1.0).abs() < 0.01);
        assert!((hub_norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_weighted_degree_centrality() {
        let mut graph = crate::generators::create_graph::<&str, f64>();

        // Create a simple weighted graph
        graph.add_edge("A", "B", 2.0).unwrap();
        graph.add_edge("A", "C", 3.0).unwrap();
        graph.add_edge("B", "C", 1.0).unwrap();

        let centrality = centrality(&graph, CentralityType::WeightedDegree).unwrap();

        // A has edges with weights 2.0 + 3.0 = 5.0
        assert_eq!(centrality[&"A"], 5.0);

        // B has edges with weights 2.0 + 1.0 = 3.0
        assert_eq!(centrality[&"B"], 3.0);

        // C has edges with weights 3.0 + 1.0 = 4.0
        assert_eq!(centrality[&"C"], 4.0);
    }

    #[test]
    fn test_weighted_closeness_centrality() {
        let mut graph = crate::generators::create_graph::<&str, f64>();

        // Create a simple weighted graph
        graph.add_edge("A", "B", 1.0).unwrap();
        graph.add_edge("B", "C", 2.0).unwrap();

        let centrality = centrality(&graph, CentralityType::WeightedCloseness).unwrap();

        // All centrality values should be positive
        for value in centrality.values() {
            assert!(*value > 0.0);
        }

        // Node B should have highest closeness (shortest paths to others)
        assert!(centrality[&"B"] > centrality[&"A"]);
        assert!(centrality[&"B"] > centrality[&"C"]);
    }

    #[test]
    fn test_weighted_betweenness_centrality() {
        let mut graph = crate::generators::create_graph::<&str, f64>();

        // Create a path graph A-B-C
        graph.add_edge("A", "B", 1.0).unwrap();
        graph.add_edge("B", "C", 1.0).unwrap();

        let centrality = centrality(&graph, CentralityType::WeightedBetweenness).unwrap();

        // B should have positive betweenness (lies on path from A to C)
        assert!(centrality[&"B"] > 0.0);

        // A and C should have zero betweenness (no paths pass through them)
        assert_eq!(centrality[&"A"], 0.0);
        assert_eq!(centrality[&"C"], 0.0);
    }

    #[test]
    fn test_weighted_centrality_digraph() {
        let mut graph = crate::generators::create_digraph::<&str, f64>();

        // Create a simple directed weighted graph
        graph.add_edge("A", "B", 2.0).unwrap();
        graph.add_edge("B", "C", 3.0).unwrap();
        graph.add_edge("A", "C", 1.0).unwrap();

        let degree_centrality = centrality_digraph(&graph, CentralityType::WeightedDegree).unwrap();
        let closeness_centrality =
            centrality_digraph(&graph, CentralityType::WeightedCloseness).unwrap();

        // All centrality values should be non-negative
        for value in degree_centrality.values() {
            assert!(*value >= 0.0);
        }
        for value in closeness_centrality.values() {
            assert!(*value >= 0.0);
        }

        // A has weighted degree 2.0 + 1.0 = 3.0 (outgoing only)
        // B has weighted degree 2.0 (incoming) + 3.0 (outgoing) = 5.0 total
        // C has weighted degree 3.0 + 1.0 = 4.0 (incoming only)
        // So B should have the highest total weighted degree
        assert!(degree_centrality[&"B"] > degree_centrality[&"A"]);
        assert!(degree_centrality[&"B"] > degree_centrality[&"C"]);
    }
}
