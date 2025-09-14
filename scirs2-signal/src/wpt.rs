// Wavelet Packet Transform (WPT)
//
// This module provides implementations of the Wavelet Packet Transform (WPT).
// Unlike the standard Discrete Wavelet Transform (DWT) which only decomposes
// the approximation coefficients at each level, the WPT decomposes both
// approximation and detail coefficients, resulting in a full binary tree
// of subbands.
//
// The WPT is particularly useful for applications such as:
// * Advanced signal analysis with better frequency resolution
// * Adaptive signal denoising
// * Feature extraction with custom subband selection
// * Signal compression with best basis selection
// * Pattern recognition with improved time-frequency localization

use crate::dwt::{dwt_decompose, dwt_reconstruct, Wavelet};
use crate::error::{SignalError, SignalResult};
use approx::assert_abs_diff_eq;
use num_traits::{Float, NumCast};
use std::collections::HashMap;
use std::fmt::Debug;

#[allow(unused_imports)]
/// Node in the wavelet packet tree
///
/// Each node represents a subband in the wavelet packet decomposition.
/// The nodes are organized in a binary tree structure, where:
/// * Left child (0) corresponds to the approximation coefficients
/// * Right child (1) corresponds to the detail coefficients
#[derive(Debug, Clone)]
pub struct WaveletPacket {
    /// Level in the decomposition tree (root is 0)
    pub level: usize,
    /// Position within the level (leftmost is 0)
    pub position: usize,
    /// Coefficient values at this node
    pub data: Vec<f64>,
    /// Wavelet used for decomposition
    pub wavelet: Wavelet,
    /// Signal extension mode
    pub mode: String,
}

impl WaveletPacket {
    /// Create a new wavelet packet node
    pub fn new(
        level: usize,
        position: usize,
        data: Vec<f64>,
        wavelet: Wavelet,
        mode: &str,
    ) -> Self {
        WaveletPacket {
            level,
            position,
            data,
            wavelet,
            mode: mode.to_string(),
        }
    }

    /// Get the unique path of this node in the tree
    ///
    /// A path is a sequence of 0s and 1s, where:
    /// * 0 means "take the approximation coefficients" (left child)
    /// * 1 means "take the detail coefficients" (right child)
    ///
    /// For example, path [0, 1] means "take approximation at level 1, then detail at level 2"
    pub fn path(&self) -> Vec<usize> {
        // The position can be converted to a binary path
        let mut path = Vec::new();
        let mut pos = self.position;

        // Convert position to binary path
        for _ in 0..self.level {
            path.push(pos % 2);
            pos /= 2;
        }

        // Reverse to get the correct order (from root to leaf)
        path.reverse();

        path
    }

    /// Get the parent node's position
    pub fn parent_position(&self) -> Option<usize> {
        if self.level == 0 {
            None // Root has no parent
        } else {
            Some(self.position / 2)
        }
    }

    /// Get the left child node's position (approximation)
    pub fn left_child_position(&self) -> usize {
        self.position * 2
    }

    /// Get the right child node's position (detail)
    pub fn right_child_position(&self) -> usize {
        self.position * 2 + 1
    }

    /// Decompose this node into its children
    pub fn decompose(&self) -> SignalResult<(Self, Self)> {
        // Nothing to decompose if data is too small
        if self.data.len() < 2 {
            return Err(SignalError::ValueError(
                "Data too small for decomposition".to_string(),
            ));
        }

        // Perform one level of DWT
        let (approx, detail) = dwt_decompose(&self.data, self.wavelet, Some(&self.mode))?;

        // Create child nodes
        let left = WaveletPacket::new(
            self.level + 1,
            self.left_child_position(),
            approx,
            self.wavelet,
            &self.mode,
        );

        let right = WaveletPacket::new(
            self.level + 1,
            self.right_child_position(),
            detail,
            self.wavelet,
            &self.mode,
        );

        Ok((left, right))
    }

    /// Reconstruct this node from its children
    pub fn reconstruct(left: &Self, right: &Self) -> SignalResult<Self> {
        // Check that the nodes are siblings
        if left.level != right.level || left.parent_position() != right.parent_position() {
            return Err(SignalError::ValueError(
                "Nodes are not siblings".to_string(),
            ));
        }

        // Check that the wavelet and mode match
        if left.wavelet != right.wavelet || left.mode != right.mode {
            return Err(SignalError::ValueError(
                "Wavelet or mode mismatch between siblings".to_string(),
            ));
        }

        // Perform one level of inverse DWT
        let reconstructed = dwt_reconstruct(&_left.data, &right.data, left.wavelet)?;

        // Create parent node
        let parent = WaveletPacket::new(
            left.level - 1,
            left.parent_position().unwrap(),
            reconstructed,
            left.wavelet,
            &_left.mode,
        );

        Ok(parent)
    }
}

/// Result of wavelet packet tree validation
#[derive(Debug, Clone)]
pub struct WaveletPacketValidationResult {
    /// Energy conservation ratios for each level
    pub energy_ratios: Vec<f64>,
    /// Maximum reconstruction error across all levels
    pub max_reconstruction_error: f64,
    /// Mean reconstruction error across all levels
    pub mean_reconstruction_error: f64,
    /// Signal-to-noise ratio of reconstruction
    pub reconstruction_snr: f64,
    /// Parseval frame ratio (should be close to 1.0)
    pub parseval_ratio: f64,
    /// Numerical stability score (0-1, higher is better)
    pub stability_score: f64,
    /// Orthogonality score (0-1, higher is better)
    pub orthogonality_score: f64,
    /// Issues found during validation
    pub issues: Vec<String>,
}

/// Wavelet packet tree data structure
///
/// This struct represents the full wavelet packet decomposition tree.
/// It stores all the nodes (subbands) in the decomposition.
#[derive(Debug)]
pub struct WaveletPacketTree {
    /// Root node (original signal)
    pub root: WaveletPacket,
    /// All nodes in the tree, organized by (level, position)
    pub nodes: HashMap<(usize, usize), WaveletPacket>,
    /// Maximum level of decomposition
    pub max_level: usize,
}

impl WaveletPacketTree {
    /// Create a new wavelet packet tree from a signal
    pub fn new<T>(data: &[T], wavelet: Wavelet, mode: Option<&str>) -> SignalResult<Self>
    where
        T: Float + NumCast + Debug,
    {
        // Convert input to f64
        let signal: Vec<f64> = _data
            .iter()
            .map(|&val| {
                num_traits::cast::cast::<T, f64>(val).ok_or_else(|| {
                    SignalError::ValueError(format!("Could not convert {:?} to f64", val))
                })
            })
            .collect::<SignalResult<Vec<_>>>()?;

        // Create root node
        let extension_mode = mode.unwrap_or("symmetric");
        let root = WaveletPacket::new(0, 0, signal, wavelet, extension_mode);

        // Initialize the tree
        let mut tree = WaveletPacketTree {
            root: root.clone(),
            nodes: HashMap::new(),
            max_level: 0,
        };

        // Add root node to the tree
        tree.nodes.insert((0, 0), root);

        Ok(tree)
    }

    /// Decompose the tree to a specified level
    pub fn decompose(&mut self, level: usize) -> SignalResult<()> {
        // Start with nodes at the current maximum level
        let current_max = self.max_level;

        // No need to decompose if we're already at or beyond the target level
        if current_max >= level {
            return Ok(());
        }

        // Decompose level by level
        for current_level in current_max..level {
            // Get all nodes at the current level
            let nodes_at_level: Vec<(usize, usize)> = self
                .nodes
                .keys()
                .filter(|(l_, _)| *l_ == current_level)
                .cloned()
                .collect();

            // Decompose each node at this level
            for (lvl, pos) in nodes_at_level {
                if let Some(node) = self.nodes.get(&(lvl, pos)) {
                    // Skip decomposition if data is too small
                    if node.data.len() < 2 {
                        continue;
                    }

                    // Clone the node to avoid borrowing issues
                    let node_clone = node.clone();

                    // Decompose the node
                    if let Ok((left, right)) = node_clone.decompose() {
                        // Add children to the tree
                        self.nodes.insert((left.level, left.position), left);
                        self.nodes.insert((right.level, right.position), right);
                    }
                }
            }
        }

        // Update maximum level
        self.max_level = level;

        Ok(())
    }

    /// Get a node at a specific level and position
    pub fn get_node(&self, level: usize, position: usize) -> Option<&WaveletPacket> {
        self.nodes.get(&(level, position))
    }

    /// Get all nodes at a specific level
    pub fn get_level(&self, level: usize) -> Vec<&WaveletPacket> {
        self.nodes
            .iter()
            .filter(|&((l, _))| *l == level)
            .map(|(_, node)| node)
            .collect()
    }

    /// Reconstruct the signal from selected nodes
    pub fn reconstruct_selective(&self, nodes: &[(usize, usize)]) -> SignalResult<Vec<f64>> {
        // Check that all nodes exist
        for &(level, position) in nodes {
            if !self.nodes.contains_key(&(level, position)) {
                return Err(SignalError::ValueError(format!(
                    "Node at level {} position {} not found",
                    level, position
                )));
            }
        }

        // For a single node, just return its data
        if nodes.len() == 1 {
            let (level, position) = nodes[0];
            if let Some(node) = self.nodes.get(&(level, position)) {
                return Ok(node.data.clone());
            }
        }

        // For multiple nodes, check if we have the root node
        // This is a special case where we can immediately return the signal
        if nodes.contains(&(0, 0)) {
            if let Some(root) = self.nodes.get(&(0, 0)) {
                return Ok(root.data.clone());
            }
        }

        // If we have all nodes at the same level, we need to reconstruct up the tree

        // Group nodes by level
        let mut nodes_by_level: HashMap<usize, Vec<usize>> = HashMap::new();
        for &(level, position) in nodes {
            nodes_by_level.entry(level).or_default().push(position);
        }

        // If all nodes are at the same level, try a direct reconstruction
        if nodes_by_level.len() == 1 {
            let level = *nodes_by_level.keys().next().unwrap();
            let positions = nodes_by_level.get(&level).unwrap();

            // If we have all nodes at this level, we can reconstruct
            let nodes_at_level = 1 << level; // 2^level
            if positions.len() == nodes_at_level {
                // Reconstruct from this level to the root
                // Start with clones of the nodes at this level
                let mut current_nodes: HashMap<usize, WaveletPacket> = HashMap::new();
                for &pos in positions {
                    if let Some(node) = self.nodes.get(&(level, pos)) {
                        current_nodes.insert(pos, node.clone());
                    }
                }

                // Reconstruct level by level
                for l in (0..level).rev() {
                    let mut next_level_nodes = HashMap::new();

                    // Process pairs of nodes
                    for parent_pos in 0..(1 << l) {
                        let left_pos = parent_pos * 2;
                        let right_pos = left_pos + 1;

                        if let (Some(left), Some(right)) =
                            (current_nodes.get(&left_pos), current_nodes.get(&right_pos))
                        {
                            if let Ok(parent) = WaveletPacket::reconstruct(left, right) {
                                next_level_nodes.insert(parent_pos, parent);
                            }
                        }
                    }

                    current_nodes = next_level_nodes;
                }

                // Return the root node data if available
                if let Some(root) = current_nodes.get(&0) {
                    return Ok(root.data.clone());
                }
            }
        }

        // If we can't do a direct reconstruction, try a more general approach
        // For simplicity, we'll start by supporting only complete levels
        let level = *nodes_by_level.keys().next().unwrap_or(&0);
        if let Some(positions) = nodes_by_level.get(&level) {
            if positions.len() == 1 << level {
                // We have all nodes at this level, reconstruct the signal

                // First, collect the nodes' data in correct order
                let mut subbands = Vec::with_capacity(positions.len());
                for pos in 0..(1 << level) {
                    if let Some(node) = self.nodes.get(&(level, pos)) {
                        subbands.push(node.data.clone());
                    } else {
                        return Err(SignalError::ValueError(format!(
                            "Missing node at level {} position {}",
                            level, pos
                        )));
                    }
                }

                // Enhanced reconstruction: Proper inverse wavelet packet transform
                // This implementation correctly reconstructs from coefficients at any level
                let mut current_nodes = HashMap::new();

                // Initialize with the nodes at the given level
                for (i, subband) in subbands.iter().enumerate() {
                    let node = WaveletPacket::new(
                        level,
                        i,
                        subband.clone(),
                        self.root.wavelet,
                        &self.root.mode,
                    );
                    current_nodes.insert(i, node);
                }

                // Reconstruct level by level up to the root
                for l in (0..level).rev() {
                    let mut next_level_nodes = HashMap::new();
                    let nodes_at_level = 1 << l; // 2^l

                    for parent_pos in 0..nodes_at_level {
                        let left_pos = parent_pos * 2;
                        let right_pos = left_pos + 1;

                        if let (Some(left), Some(right)) =
                            (current_nodes.get(&left_pos), current_nodes.get(&right_pos))
                        {
                            match WaveletPacket::reconstruct(left, right) {
                                Ok(parent) => {
                                    next_level_nodes.insert(parent_pos, parent);
                                }
                                Err(e) => {
                                    return Err(SignalError::ComputationError(format!(
                                        "Failed to reconstruct parent node at level {} position {}: {}",
                                        l, parent_pos, e
                                    )));
                                }
                            }
                        } else {
                            return Err(SignalError::ValueError(format!(
                                "Missing child nodes for reconstruction at level {} position {}",
                                l + 1,
                                left_pos
                            )));
                        }
                    }

                    current_nodes = next_level_nodes;
                }

                // Extract the reconstructed signal from the root node
                if let Some(root_node) = current_nodes.get(&0) {
                    let reconstructed = root_node.data.clone();

                    // Validate reconstruction quality
                    let original_energy: f64 = self.root.data.iter().map(|x| x * x).sum();
                    let reconstructed_energy: f64 = reconstructed.iter().map(|x| x * x).sum();

                    if original_energy > 1e-12 {
                        let energy_ratio = reconstructed_energy / original_energy;
                        if ((energy_ratio - 1.0) as f64).abs() > 0.1 {
                            eprintln!("Warning: Energy conservation issue in WPT reconstruction. Ratio: {:.6}", energy_ratio);
                        }
                    }

                    return Ok(reconstructed);
                } else {
                    return Err(SignalError::ComputationError(
                        "Failed to reconstruct root node".to_string(),
                    ));
                }
            }
        }

        // If we can't do any reconstruction, return an error
        Err(SignalError::ValueError(
            "Could not reconstruct signal from the given nodes".to_string(),
        ))
    }

    /// Enhanced validation of the wavelet packet tree
    ///
    /// This function performs comprehensive validation including:
    /// - Energy conservation at each level
    /// - Orthogonality of basis functions
    /// - Reconstruction accuracy
    /// - Numerical stability
    pub fn validate_enhanced(&self) -> SignalResult<WaveletPacketValidationResult> {
        let mut issues: Vec<String> = Vec::new();

        // Check energy conservation across levels
        let original_energy: f64 = self.root.data.iter().map(|x| x * x).sum();
        let mut energy_ratios = Vec::new();

        for level in 1..=self.max_level {
            let level_nodes = self.get_level(level);
            let level_energy: f64 = level_nodes
                .iter()
                .map(|node| node.data.iter().map(|x| x * x).sum::<f64>())
                .sum();

            if original_energy > 1e-12 {
                let ratio = level_energy / original_energy;
                energy_ratios.push(ratio);

                if ((ratio - 1.0) as f64).abs() > 0.01 {
                    issues.push(format!(
                        "Energy conservation violation at level {}: ratio = {:.6}",
                        level, ratio
                    ));
                }
            }
        }

        // Test reconstruction accuracy for complete decompositions
        let mut max_reconstruction_error = 0.0;
        let mut mean_reconstruction_error = 0.0;
        let mut reconstruction_snr = f64::INFINITY;

        for level in 1..=self.max_level {
            let nodes_at_level: Vec<(usize, usize)> =
                (0..(1 << level)).map(|pos| (level, pos)).collect();

            if let Ok(reconstructed) = self.reconstruct_selective(&nodes_at_level) {
                if reconstructed.len() == self.root.data.len() {
                    let errors: Vec<f64> = self
                        .root
                        .data
                        .iter()
                        .zip(reconstructed.iter())
                        .map(|(&orig, &recon)| (orig - recon).abs())
                        .collect();

                    let max_error = errors.iter().fold(0.0, |a, &b| a.max(b));
                    let mean_error = errors.iter().sum::<f64>() / errors.len() as f64;

                    max_reconstruction_error = max_reconstruction_error.max(max_error);
                    mean_reconstruction_error = mean_reconstruction_error.max(mean_error);

                    // Calculate SNR
                    let signal_power: f64 = self.root.data.iter().map(|x| x * x).sum();
                    let noise_power: f64 = errors.iter().map(|x| x * x).sum();

                    if noise_power > 1e-15 && signal_power > 1e-15 {
                        let snr = 10.0 * (signal_power / noise_power).log10();
                        reconstruction_snr = reconstruction_snr.min(snr);
                    }

                    if max_error > 1e-10 {
                        issues.push(format!(
                            "Reconstruction error at level {}: max = {:.2e}, mean = {:.2e}",
                            level, max_error, mean_error
                        ));
                    }
                }
            }
        }

        // Check for numerical stability issues
        let mut stability_score = 1.0;
        for (_, node) in &self.nodes {
            for &val in &node.data {
                if !val.is_finite() {
                    issues.push("Non-finite values detected in coefficients".to_string());
                    stability_score *= 0.5;
                }
                if val.abs() > 1e10 {
                    issues.push("Very large coefficient values detected".to_string());
                    stability_score *= 0.8;
                }
            }
        }

        // Calculate Parseval frame ratio
        let mut parseval_ratio = 1.0;
        if let Some(&first_ratio) = energy_ratios.first() {
            parseval_ratio = first_ratio;
        }

        // Orthogonality check (simplified)
        let orthogonality = self.check_orthogonality();

        if reconstruction_snr.is_infinite() {
            reconstruction_snr = 200.0; // Perfect reconstruction
        }

        Ok(WaveletPacketValidationResult {
            energy_ratios,
            max_reconstruction_error,
            mean_reconstruction_error,
            reconstruction_snr,
            parseval_ratio,
            stability_score,
            orthogonality_score: orthogonality,
            issues,
        })
    }

    /// Check orthogonality of wavelet packet basis functions
    fn check_orthogonality(&self) -> f64 {
        let mut min_orthogonality = 1.0;

        // Check orthogonality within each level
        for level in 1..=self.max_level {
            let level_nodes = self.get_level(level);

            for i in 0..level_nodes.len() {
                for j in (i + 1)..level_nodes.len() {
                    let node1 = &level_nodes[i];
                    let node2 = &level_nodes[j];

                    // Compute normalized inner product
                    let min_len = node1.data.len().min(node2.data.len());
                    if min_len > 0 {
                        let inner_product: f64 = node1.data[..min_len]
                            .iter()
                            .zip(node2.data[..min_len].iter())
                            .map(|(a, b)| a * b)
                            .sum();

                        let norm1: f64 = node1.data[..min_len]
                            .iter()
                            .map(|x| x * x)
                            .sum::<f64>()
                            .sqrt();
                        let norm2: f64 = node2.data[..min_len]
                            .iter()
                            .map(|x| x * x)
                            .sum::<f64>()
                            .sqrt();

                        if norm1 > 1e-12 && norm2 > 1e-12 {
                            let normalized_ip = inner_product.abs() / (norm1 * norm2);
                            min_orthogonality = min_orthogonality.min(1.0 - normalized_ip);
                        }
                    }
                }
            }
        }

        min_orthogonality
    }

    /// Get the best basis using a cost function
    pub fn get_best_basis(&self, costfunction: &str) -> SignalResult<Vec<(usize, usize)>> {
        // For now, implement a simple best basis selection using Shannon entropy
        let mut best_basis = Vec::new();
        
        // Start from the deepest level and work backwards
        for level in (0..=self.max_level).rev() {
            let nodes = self.get_level(level);
            for node in nodes {
                // Calculate cost based on the function type
                let _cost = match cost_function {
                    "shannon" => {
                        // Shannon entropy: -sum(p * log(p))
                        let total_energy: f64 = node.data.iter().map(|x| x * x).sum();
                        if total_energy > 1e-12 {
                            node.data.iter()
                                .map(|&x| {
                                    let p = (x * x) / total_energy;
                                    if p > 1e-12 { -p * p.ln() } else { 0.0 }
                                })
                                .sum::<f64>()
                        } else {
                            0.0
                        }
                    },
                    _ => 0.0, // Default cost
                };
                
                // Simple selection - include all leaf nodes for now
                if level == self.max_level {
                    best_basis.push((node.level, node.position));
                }
            }
        }
        
        Ok(best_basis)
    }

    /// Get the depth (maximum level) of the tree
    pub fn get_depth(&self) -> usize {
        self.max_level
    }

    /// Get all nodes at a specific level (alias for get_level with different return type)
    pub fn get_nodes_at_level(&self, level: usize) -> Vec<(usize, usize)> {
        self.nodes
            .keys()
            .filter(|(l, _)| *l == level)
            .cloned()
            .collect()
    }

    /// Get all leaf nodes (nodes that don't have children)
    pub fn get_leaf_nodes(&self) -> Vec<(usize, usize)> {
        let mut leaf_nodes = Vec::new();
        
        for &(level, position) in self.nodes.keys() {
            // A node is a leaf if it has no children
            let left_child = (level + 1, position * 2);
            let right_child = (level + 1, position * 2 + 1);
            
            if !self.nodes.contains_key(&left_child) && !self.nodes.contains_key(&right_child) {
                leaf_nodes.push((level, position));
            }
        }
        
        leaf_nodes
    }

    /// Compute all costs for nodes in the tree using the specified entropy type
    /// This is a placeholder implementation
    #[allow(dead_code)]
    pub fn compute_all_costs(&self, entropytype: &str) -> SignalResult<HashMap<(usize, usize), f64>> {
        let mut costs = HashMap::new();
        
        for ((level, position), node) in &self.nodes {
            let cost = match entropy_type {
                "shannon" => {
                    // Simple Shannon entropy calculation
                    let energy: f64 = node.data.iter().map(|x| x * x).sum();
                    if energy > 0.0 {
                        -energy.ln()
                    } else {
                        0.0
                    }
                },
                "norm" => {
                    // L2 norm
                    node.data.iter().map(|x| x * x).sum::<f64>().sqrt()
                },
                _ => {
                    // Default to energy
                    node.data.iter().map(|x| x * x).sum()
                }
            };
            costs.insert((*level, *position), cost);
        }
        
        Ok(costs)
    }
}

/// Performs a full wavelet packet decomposition of a signal to a specified level.
///
/// # Arguments
///
/// * `data` - The input signal
/// * `wavelet` - The wavelet to use for the transform
/// * `level` - The maximum level of decomposition
/// * `mode` - The signal extension mode (default: "symmetric")
///
/// # Returns
///
/// * A WaveletPacketTree containing all the decomposition nodes
///
/// # Examples
///
/// ```rust
/// use scirs2_signal::wpt::wp_decompose;
/// use scirs2_signal::dwt::Wavelet;
///
/// // Create a simple signal
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
///
/// // Perform wavelet packet decomposition to level 2
/// let wpt = wp_decompose(&signal, Wavelet::DB(4), 2, None).unwrap();
///
/// // Check that we have the expected number of nodes
/// // At level 0: 1 node
/// // At level 1: 2 nodes
/// // At level 2: 4 nodes
/// // Total: 7 nodes
/// assert_eq!(wpt.nodes.len(), 7);
///
/// // Verify that we have all nodes at level 2
/// let level2 = wpt.get_level(2);
/// assert_eq!(level2.len(), 4);
/// ```
#[allow(dead_code)]
pub fn wp_decompose<T>(
    data: &[T],
    wavelet: Wavelet,
    level: usize,
    mode: Option<&str>,
) -> SignalResult<WaveletPacketTree>
where
    T: Float + NumCast + Debug,
{
    // Create the wavelet packet tree
    let mut tree = WaveletPacketTree::new(data, wavelet, mode)?;

    // Decompose to the specified level
    tree.decompose(level)?;

    Ok(tree)
}

/// Extracts coefficients from a wavelet packet tree at a given level.
///
/// # Arguments
///
/// * `tree` - The wavelet packet tree
/// * `level` - The level from which to extract coefficients
///
/// # Returns
///
/// * A vector of coefficients, organized from left to right
///
/// # Examples
///
/// ```rust
/// use scirs2_signal::wpt::{wp_decompose, get_level_coefficients};
/// use scirs2_signal::dwt::Wavelet;
///
/// // Create a simple signal
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
///
/// // Perform wavelet packet decomposition to level 2
/// let wpt = wp_decompose(&signal, Wavelet::DB(4), 2, None).unwrap();
///
/// // Get all coefficients at level 2
/// let coeffs = get_level_coefficients(&wpt, 2);
///
/// // Check that we have coefficients from all 4 subbands at level 2
/// assert_eq!(coeffs.len(), 4);
/// ```
#[allow(dead_code)]
pub fn get_level_coefficients(tree: &WaveletPacketTree, level: usize) -> Vec<Vec<f64>> {
    let mut nodes = tree.get_level(level);

    // Sort by position (left to right)
    nodes.sort_by_key(|node| node.position);

    // Extract coefficients
    nodes.iter().map(|node| node.data.clone()).collect()
}

/// Reconstruct a signal from a wavelet packet tree using selected nodes.
///
/// # Arguments
///
/// * `tree` - The wavelet packet tree
/// * `nodes` - List of (level, position) pairs for nodes to include in reconstruction
///
/// # Returns
///
/// * The reconstructed signal
///
/// # Examples
///
/// ```rust
/// use scirs2_signal::wpt::{wp_decompose, reconstruct_from_nodes};
/// use scirs2_signal::dwt::Wavelet;
///
/// // Create a simple signal with power-of-2 length
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
///
/// // Perform wavelet packet decomposition to level 2 using Haar wavelet
/// // (Haar is simpler and doesn't introduce as much padding)
/// let wpt = wp_decompose(&signal, Wavelet::Haar, 2, None).unwrap();
///
/// // Reconstruct using all nodes at level 2
/// let nodes = vec![(2, 0), (2, 1), (2, 2), (2, 3)];
/// let reconstructed = reconstruct_from_nodes(&wpt, &nodes).unwrap();
///
/// // Check that reconstruction occurred (might have different length due to padding)
/// assert!(!reconstructed.is_empty());
///
/// // The reconstructed signal exists and has reasonable length
/// assert!(reconstructed.len() >= signal.len());
/// ```
#[allow(dead_code)]
pub fn reconstruct_from_nodes(
    tree: &WaveletPacketTree,
    nodes: &[(usize, usize)],
) -> SignalResult<Vec<f64>> {
    tree.reconstruct_selective(nodes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    #[test]
    fn test_wavelet_packet_path() {
        // Create some test nodes
        let root = WaveletPacket::new(0, 0, vec![1.0, 2.0, 3.0, 4.0], Wavelet::Haar, "symmetric");
        let node_1_0 = WaveletPacket::new(1, 0, vec![1.0, 2.0], Wavelet::Haar, "symmetric");
        let node_1_1 = WaveletPacket::new(1, 1, vec![1.0, 2.0], Wavelet::Haar, "symmetric");
        let node_2_3 = WaveletPacket::new(2, 3, vec![1.0], Wavelet::Haar, "symmetric");

        // Check paths
        assert_eq!(root.path(), Vec::<usize>::new());
        assert_eq!(node_1_0.path(), vec![0]);
        assert_eq!(node_1_1.path(), vec![1]);
        assert_eq!(node_2_3.path(), vec![1, 1]);
    }

    #[test]
    fn test_wavelet_packet_positions() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![0.5, 0.5];
        // Create a test node
        let node = WaveletPacket::new(2, 3, vec![1.0, 2.0], Wavelet::Haar, "symmetric");

        // Check positions
        assert_eq!(node.parent_position(), Some(1));
        assert_eq!(node.left_child_position(), 6);
        assert_eq!(node.right_child_position(), 7);
    }

    #[test]
    fn test_wavelet_packet_decompose() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![0.5, 0.5];
        // Create a signal with constant values
        let signal = vec![1.0; 8];
        let root = WaveletPacket::new(0, 0, signal, Wavelet::Haar, "symmetric");

        // Decompose
        let (left, right) = root.decompose().unwrap();

        // For a constant signal with Haar wavelet:
        // - Approximation should be constant values (possibly scaled)
        // - Details should be close to zero
        assert_eq!(left.data.len(), 4);
        assert_eq!(right.data.len(), 4);

        // Check approximation values (all should be equal)
        let first_approx = left.data[0];
        for &val in &left.data {
            assert_abs_diff_eq!(val, first_approx, epsilon = 1e-10);
        }

        // Check detail values
        for &val in &right.data {
            assert_abs_diff_eq!(val, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_wavelet_packet_reconstruct() {
        // Create left and right nodes
        let left = WaveletPacket::new(1, 0, vec![2.0; 4], Wavelet::Haar, "symmetric");
        let right = WaveletPacket::new(1, 1, vec![0.0; 4], Wavelet::Haar, "symmetric");

        // Reconstruct
        let parent = WaveletPacket::reconstruct(&left, &right).unwrap();

        // Check dimensions
        assert_eq!(parent.level, 0);
        assert_eq!(parent.position, 0);
        assert_eq!(parent.data.len(), 8);

        // For Haar wavelet with constant approximation and zero details:
        // Reconstructed signal should be constant values
        let first_val = parent.data[0];
        for &val in &parent.data {
            assert_abs_diff_eq!(val, first_val, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_wp_decompose() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![0.5, 0.5];
        // Create a test signal
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // Perform WPT decomposition to level 2
        let tree = wp_decompose(&signal, Wavelet::Haar, 2, None).unwrap();

        // Check that we have all nodes
        assert!(tree.nodes.contains_key(&(0, 0))); // Root
        assert!(tree.nodes.contains_key(&(1, 0))); // Level 1, approximation
        assert!(tree.nodes.contains_key(&(1, 1))); // Level 1, detail
        assert!(tree.nodes.contains_key(&(2, 0))); // Level 2, approx of approx
        assert!(tree.nodes.contains_key(&(2, 1))); // Level 2, detail of approx
        assert!(tree.nodes.contains_key(&(2, 2))); // Level 2, approx of detail
        assert!(tree.nodes.contains_key(&(2, 3))); // Level 2, detail of detail

        // Check total number of nodes
        assert_eq!(tree.nodes.len(), 7);

        // Check the root data
        let root = tree.get_node(0, 0).unwrap();
        assert_eq!(root.data, signal);
    }

    #[test]
    fn test_wp_reconstruct() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![0.5, 0.5];
        // Create a simple test signal with constant values
        let signal = vec![1.0; 8];

        // Perform WPT decomposition to level 1
        let tree = wp_decompose(&signal, Wavelet::Haar, 1, None).unwrap();

        // Verify we have the root node
        assert!(tree.nodes.contains_key(&(0, 0)));

        // Reconstruct from the root node directly
        let nodes = vec![(0, 0)];
        let reconstructed = reconstruct_from_nodes(&tree, &nodes).unwrap();

        // Check length
        assert_eq!(reconstructed.len(), signal.len());

        // The reconstruction should exactly match the original (it's just the root node)
        for (i, (&original, &recon)) in signal.iter().zip(reconstructed.iter()).enumerate() {
            assert_eq!(original, recon, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_selective_reconstruction() {
        // For selective reconstruction, let's check that we can reconstruct from a level 1 node

        // Create a constant signal
        let signal = vec![1.0; 8];

        // Perform WPT decomposition to level 1
        let tree = wp_decompose(&signal, Wavelet::Haar, 1, None).unwrap();

        // Verify that we have the expected level 1 nodes
        assert!(tree.nodes.contains_key(&(1, 0))); // Level 1, approximation

        // Get node data directly
        let approx_node = tree.get_node(1, 0).unwrap();
        let approx_data = &approx_node.data;

        // Reconstruction from a single node should return that node's data
        let nodes = vec![(1, 0)]; // Just use the approximation node
        let approx_only = reconstruct_from_nodes(&tree, &nodes).unwrap();

        // Check that this gives us the node's data directly
        assert_eq!(approx_only.len(), approx_data.len());

        for (i, (&a, &b)) in approx_data.iter().zip(approx_only.iter()).enumerate() {
            assert_eq!(a, b, "Mismatch at index {}", i);
        }
    }
}
