//! Graph algorithms implementation
//!
//! This module provides common graph algorithms such as:
//! - Shortest path
//! - Connected components
//! - Minimum spanning tree
//! - Traversal algorithms (BFS, DFS)
//! - Topological sorting

use petgraph::algo::dijkstra;
use petgraph::visit::EdgeRef;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

use crate::base::{DiGraph, Edge, EdgeWeight, Graph, Node};
use crate::error::{GraphError, Result};

/// Path between two nodes in a graph
#[derive(Debug, Clone)]
pub struct Path<N: Node, E: EdgeWeight> {
    /// The nodes in the path, in order
    pub nodes: Vec<N>,
    /// The total weight of the path
    pub total_weight: E,
}

/// Finds the shortest path between source and target nodes using Dijkstra's algorithm
///
/// # Arguments
/// * `graph` - The graph to search in
/// * `source` - The source node
/// * `target` - The target node
///
/// # Returns
/// * `Ok(Some(Path))` - If a path exists
/// * `Ok(None)` - If no path exists
/// * `Err(GraphError)` - If the source or target node is not in the graph
pub fn shortest_path<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    source: &N,
    target: &N,
) -> Result<Option<Path<N, E>>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight
        + num_traits::Zero
        + num_traits::One
        + std::ops::Add<Output = E>
        + PartialOrd
        + std::marker::Copy
        + std::fmt::Debug
        + std::default::Default,
    Ix: petgraph::graph::IndexType,
{
    // Check if source and target are in the graph
    if !graph.has_node(source) {
        return Err(GraphError::InvalidGraph(format!(
            "Source node {:?} not found",
            source
        )));
    }
    if !graph.has_node(target) {
        return Err(GraphError::InvalidGraph(format!(
            "Target node {:?} not found",
            target
        )));
    }

    let source_idx = graph
        .inner()
        .node_indices()
        .find(|&idx| graph.inner()[idx] == *source)
        .unwrap();
    let target_idx = graph
        .inner()
        .node_indices()
        .find(|&idx| graph.inner()[idx] == *target)
        .unwrap();

    // Use petgraph's Dijkstra algorithm implementation
    let results = dijkstra(graph.inner(), source_idx, Some(target_idx), |e| *e.weight());

    // If target is not reachable, return None
    if !results.contains_key(&target_idx) {
        return Ok(None);
    }

    let total_weight = results[&target_idx];

    // Reconstruct the path
    let mut path = Vec::new();
    let mut current = target_idx;

    path.push(graph.inner()[current].clone());

    // Backtrack to find the path
    while current != source_idx {
        let min_prev = graph
            .inner()
            .edges_directed(current, petgraph::Direction::Incoming)
            .filter_map(|e| {
                let from = e.source();
                let edge_weight = *e.weight();

                // Check if this node is part of the shortest path
                if let Some(from_dist) = results.get(&from) {
                    // If this edge is part of the shortest path
                    if *from_dist + edge_weight == results[&current] {
                        return Some((from, *from_dist));
                    }
                }
                None
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        if let Some((prev, _)) = min_prev {
            current = prev;
            path.push(graph.inner()[current].clone());
        } else {
            // This shouldn't happen if Dijkstra's algorithm works correctly
            return Err(GraphError::AlgorithmError(
                "Failed to reconstruct path".to_string(),
            ));
        }
    }

    // Reverse the path to get it from source to target
    path.reverse();

    Ok(Some(Path {
        nodes: path,
        total_weight,
    }))
}

/// Finds the shortest path in a directed graph
///
/// # Arguments
/// * `graph` - The directed graph
/// * `source` - The source node
/// * `target` - The target node
///
/// # Returns
/// * `Ok(Some(Path))` - If a path exists
/// * `Ok(None)` - If no path exists
/// * `Err(GraphError)` - If the source or target node is not in the graph
pub fn shortest_path_digraph<N, E, Ix>(
    graph: &DiGraph<N, E, Ix>,
    source: &N,
    target: &N,
) -> Result<Option<Path<N, E>>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight
        + num_traits::Zero
        + num_traits::One
        + std::ops::Add<Output = E>
        + PartialOrd
        + std::marker::Copy
        + std::fmt::Debug
        + std::default::Default,
    Ix: petgraph::graph::IndexType,
{
    // Check if source and target are in the graph
    if !graph.has_node(source) {
        return Err(GraphError::InvalidGraph(format!(
            "Source node {:?} not found",
            source
        )));
    }
    if !graph.has_node(target) {
        return Err(GraphError::InvalidGraph(format!(
            "Target node {:?} not found",
            target
        )));
    }

    let source_idx = graph
        .inner()
        .node_indices()
        .find(|&idx| graph.inner()[idx] == *source)
        .unwrap();
    let target_idx = graph
        .inner()
        .node_indices()
        .find(|&idx| graph.inner()[idx] == *target)
        .unwrap();

    // Use petgraph's Dijkstra algorithm implementation
    let results = dijkstra(graph.inner(), source_idx, Some(target_idx), |e| *e.weight());

    // If target is not reachable, return None
    if !results.contains_key(&target_idx) {
        return Ok(None);
    }

    let total_weight = results[&target_idx];

    // Reconstruct the path
    let mut path = Vec::new();
    let mut current = target_idx;

    path.push(graph.inner()[current].clone());

    // Backtrack to find the path
    while current != source_idx {
        let min_prev = graph
            .inner()
            .edges_directed(current, petgraph::Direction::Incoming)
            .filter_map(|e| {
                let from = e.source();
                let edge_weight = *e.weight();

                // Check if this node is part of the shortest path
                if let Some(from_dist) = results.get(&from) {
                    // If this edge is part of the shortest path
                    if *from_dist + edge_weight == results[&current] {
                        return Some((from, *from_dist));
                    }
                }
                None
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        if let Some((prev, _)) = min_prev {
            current = prev;
            path.push(graph.inner()[current].clone());
        } else {
            // This shouldn't happen if Dijkstra's algorithm works correctly
            return Err(GraphError::AlgorithmError(
                "Failed to reconstruct path".to_string(),
            ));
        }
    }

    // Reverse the path to get it from source to target
    path.reverse();

    Ok(Some(Path {
        nodes: path,
        total_weight,
    }))
}

/// Each connected component is represented as a set of nodes
pub type Component<N> = HashSet<N>;

/// Finds all connected components in an undirected graph
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// * A vector of connected components, where each component is a set of nodes
pub fn connected_components<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Vec<Component<N>>
where
    N: Node,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let mut components: Vec<Component<N>> = Vec::new();
    let mut visited = HashSet::new();

    // For each node in the graph
    for node_idx in graph.inner().node_indices() {
        // Skip if already visited
        if visited.contains(&node_idx) {
            continue;
        }

        // New component
        let mut component = Component::new();
        let mut queue = VecDeque::new();
        queue.push_back(node_idx);
        visited.insert(node_idx);

        // BFS to find all nodes in this component
        while let Some(curr) = queue.pop_front() {
            component.insert(graph.inner()[curr].clone());

            // Check all neighbors
            for neighbor in graph.inner().neighbors(curr) {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }

        components.push(component);
    }

    components
}

/// Edge for minimum spanning tree
#[derive(Debug, Clone)]
struct MstEdge<N: Node, E: EdgeWeight> {
    source: N,
    target: N,
    weight: E,
}

/// Compare edges by weight
impl<N: Node, E: EdgeWeight> PartialOrd for MstEdge<N, E> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<N: Node, E: EdgeWeight> PartialEq for MstEdge<N, E> {
    fn eq(&self, other: &Self) -> bool {
        self.weight == other.weight
    }
}

impl<N: Node, E: EdgeWeight> Eq for MstEdge<N, E> {}

impl<N: Node, E: EdgeWeight> Ord for MstEdge<N, E> {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .weight
            .partial_cmp(&self.weight)
            .unwrap_or(Ordering::Equal)
    }
}

/// Computes a minimum spanning tree of an undirected graph using Kruskal's algorithm
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// * `Ok(Vec<Edge>)` - The edges in the minimum spanning tree
/// * `Err(GraphError)` - If the graph is empty or not connected
pub fn minimum_spanning_tree<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Result<Vec<Edge<N, E>>>
where
    N: Node,
    E: EdgeWeight + PartialOrd,
    Ix: petgraph::graph::IndexType,
{
    // Check if the graph is empty
    if graph.node_count() == 0 {
        return Err(GraphError::InvalidGraph("Graph is empty".to_string()));
    }

    // Check if the graph is connected
    let components = connected_components(graph);
    if components.len() > 1 {
        return Err(GraphError::InvalidGraph(
            "Graph is not connected".to_string(),
        ));
    }

    // Kruskal's algorithm

    // Create a disjoint-set data structure for tracking components
    let mut node_to_set: HashMap<N, usize> = HashMap::new();
    let mut sets: Vec<HashSet<N>> = Vec::new();

    // Initialize each node in its own set
    for i in 0..graph.node_count() {
        let node = graph
            .inner()
            .node_weight(petgraph::graph::NodeIndex::new(i))
            .unwrap()
            .clone();
        let set_idx = sets.len();
        let mut set = HashSet::new();
        set.insert(node.clone());
        sets.push(set);
        node_to_set.insert(node, set_idx);
    }

    // Sort edges by weight (min heap)
    let mut edges = BinaryHeap::new();
    for edge in graph.edges() {
        edges.push(MstEdge {
            source: edge.source,
            target: edge.target,
            weight: edge.weight,
        });
    }

    let mut mst_edges = Vec::new();

    // Process edges in order of increasing weight
    while let Some(edge) = edges.pop() {
        let source_set = node_to_set[&edge.source];
        let target_set = node_to_set[&edge.target];

        // If source and target are in different sets, merge them
        if source_set != target_set {
            // Add edge to MST
            mst_edges.push(Edge {
                source: edge.source.clone(),
                target: edge.target.clone(),
                weight: edge.weight.clone(),
            });

            // Merge sets
            let (smaller_idx, larger_idx) = if sets[source_set].len() < sets[target_set].len() {
                (source_set, target_set)
            } else {
                (target_set, source_set)
            };

            // Move all nodes from the smaller set to the larger set
            let smaller_set = std::mem::take(&mut sets[smaller_idx]);
            for node in &smaller_set {
                node_to_set.insert(node.clone(), larger_idx);
                sets[larger_idx].insert(node.clone());
            }

            // If we have n-1 edges, we're done
            if mst_edges.len() == graph.node_count() - 1 {
                break;
            }
        }
    }

    Ok(mst_edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shortest_path() {
        let mut graph: Graph<char, f64> = Graph::new();

        // Create a simple graph:
        // A -- 1.0 --> B -- 2.0 --> C
        // |              |
        // 3.0             4.0
        // |              |
        // D -- 5.0 --> E

        graph.add_edge('A', 'B', 1.0).unwrap();
        graph.add_edge('B', 'C', 2.0).unwrap();
        graph.add_edge('A', 'D', 3.0).unwrap();
        graph.add_edge('B', 'E', 4.0).unwrap();
        graph.add_edge('D', 'E', 5.0).unwrap();

        // Test A to C (should be A -> B -> C with weight 3.0)
        let path = shortest_path(&graph, &'A', &'C').unwrap().unwrap();
        assert_eq!(path.nodes, vec!['A', 'B', 'C']);
        assert_eq!(path.total_weight, 3.0);

        // Test A to E (should be A -> B -> E with weight 5.0)
        let path = shortest_path(&graph, &'A', &'E').unwrap().unwrap();
        assert_eq!(path.nodes, vec!['A', 'B', 'E']);
        assert_eq!(path.total_weight, 5.0);

        // Test D to C (should be D -> A -> B -> C with weight 6.0)
        let path = shortest_path(&graph, &'D', &'C').unwrap().unwrap();
        assert_eq!(path.nodes, vec!['D', 'A', 'B', 'C']);
        assert_eq!(path.total_weight, 6.0);
    }

    #[test]
    fn test_connected_components() {
        let mut graph: Graph<i32, f64> = Graph::new();

        // Component 1: 1 -- 2 -- 3
        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(2, 3, 1.0).unwrap();

        // Component 2: 4 -- 5
        graph.add_edge(4, 5, 1.0).unwrap();

        // Component 3: 6 (isolated node)
        graph.add_node(6);

        let components = connected_components(&graph);

        assert_eq!(components.len(), 3);

        // Check component 1
        let comp1 = components.iter().find(|comp| comp.contains(&1)).unwrap();
        assert!(comp1.contains(&1));
        assert!(comp1.contains(&2));
        assert!(comp1.contains(&3));
        assert_eq!(comp1.len(), 3);

        // Check component 2
        let comp2 = components.iter().find(|comp| comp.contains(&4)).unwrap();
        assert!(comp2.contains(&4));
        assert!(comp2.contains(&5));
        assert_eq!(comp2.len(), 2);

        // Check component 3
        let comp3 = components.iter().find(|comp| comp.contains(&6)).unwrap();
        assert!(comp3.contains(&6));
        assert_eq!(comp3.len(), 1);
    }

    #[test]
    fn test_minimum_spanning_tree() {
        let mut graph: Graph<char, f64> = Graph::new();

        // Create a graph:
        //     B
        //    /|\
        //  2/ | \3
        //  /  |  \
        // A   |1  C
        //  \  |  /
        //  4\ | /5
        //    \|/
        //     D

        graph.add_edge('A', 'B', 2.0).unwrap();
        graph.add_edge('A', 'D', 4.0).unwrap();
        graph.add_edge('B', 'C', 3.0).unwrap();
        graph.add_edge('B', 'D', 1.0).unwrap();
        graph.add_edge('C', 'D', 5.0).unwrap();

        let mst = minimum_spanning_tree(&graph).unwrap();

        // MST should have 3 edges (n-1 where n=4)
        assert_eq!(mst.len(), 3);

        // Extract edges as sets to make comparison easier
        let edge_sets: Vec<_> = mst
            .iter()
            .map(|e| {
                let mut set = HashSet::new();
                set.insert(e.source);
                set.insert(e.target);
                (set, e.weight)
            })
            .collect();

        // Check expected edges: B-D (1.0), A-B (2.0), B-C (3.0)
        let expected_edges = [
            (
                {
                    let mut s = HashSet::new();
                    s.insert('B');
                    s.insert('D');
                    s
                },
                1.0,
            ),
            (
                {
                    let mut s = HashSet::new();
                    s.insert('A');
                    s.insert('B');
                    s
                },
                2.0,
            ),
            (
                {
                    let mut s = HashSet::new();
                    s.insert('B');
                    s.insert('C');
                    s
                },
                3.0,
            ),
        ];

        for expected in &expected_edges {
            assert!(edge_sets
                .iter()
                .any(|(set, weight)| set == &expected.0 && (*weight - expected.1).abs() < 1e-10));
        }
    }

    #[test]
    fn test_shortest_path_digraph() {
        let mut graph: DiGraph<char, f64> = DiGraph::new();

        // Create a directed graph:
        // A -> B -> C
        // ^    |
        // |    v
        // D <- E

        graph.add_edge('A', 'B', 1.0).unwrap();
        graph.add_edge('B', 'C', 2.0).unwrap();
        graph.add_edge('B', 'E', 3.0).unwrap();
        graph.add_edge('E', 'D', 4.0).unwrap();
        graph.add_edge('D', 'A', 5.0).unwrap();

        // Test A to C
        let path = shortest_path_digraph(&graph, &'A', &'C').unwrap().unwrap();
        assert_eq!(path.nodes, vec!['A', 'B', 'C']);
        assert_eq!(path.total_weight, 3.0);

        // Test A to D
        let path = shortest_path_digraph(&graph, &'A', &'D').unwrap().unwrap();
        assert_eq!(path.nodes, vec!['A', 'B', 'E', 'D']);
        assert_eq!(path.total_weight, 8.0);

        // Test D to C
        let path = shortest_path_digraph(&graph, &'D', &'C').unwrap().unwrap();
        assert_eq!(path.nodes, vec!['D', 'A', 'B', 'C']);
        assert_eq!(path.total_weight, 8.0);

        // Test C to E (should be None as there's no path)
        let path = shortest_path_digraph(&graph, &'C', &'E').unwrap();
        assert!(path.is_none());
    }
}
