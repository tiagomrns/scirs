//! Advanced pattern recognition for memory access patterns.
//!
//! This module provides specialized algorithms for detecting complex access patterns
//! that are common in scientific computing workloads, such as:
//! - Diagonal traversals
//! - Block-based accesses
//! - Stencil operations
//! - Strided matrix operations
//! - Custom patterns defined by mathematical functions

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

use super::prefetch::AccessPattern;

/// The different types of complex patterns that can be recognized.
#[derive(Debug, Clone, PartialEq)]
pub enum ComplexPattern {
    /// Standard row-major traversal
    RowMajor,

    /// Standard column-major traversal
    ColumnMajor,

    /// Zigzag (alternating directions per row)
    Zigzag,

    /// Diagonal traversal (main diagonal)
    DiagonalMajor,

    /// Anti-diagonal traversal (other diagonal)
    DiagonalMinor,

    /// Block-based traversal (common in tiled algorithms)
    Block {
        block_height: usize,
        block_width: usize,
    },

    /// Strided access within blocks
    BlockStrided { block_size: usize, stride: usize },

    /// Stencil operation (center point with neighbors)
    Stencil { dimensions: usize, radius: usize },

    /// Rotating blocks (e.g., for matrix transposition)
    RotatingBlock { block_size: usize },

    /// Sparse access (e.g., for sparse matrices)
    Sparse { density: f64 },

    /// Hierarchical traversal (e.g., Z-order curve)
    Hierarchical { levels: usize },

    /// Custom pattern with a name
    Custom(String),
}

/// Confidence level for pattern detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Confidence {
    /// Pattern is definitely detected
    High,

    /// Pattern is probably detected
    Medium,

    /// Pattern might be detected
    Low,

    /// Pattern is only tentatively detected
    Tentative,
}

/// A recognized pattern with metadata.
#[derive(Debug, Clone)]
pub struct RecognizedPattern {
    /// The type of pattern
    pub pattern_type: ComplexPattern,

    /// Confidence in the pattern detection
    pub confidence: Confidence,

    /// Additional metadata about the pattern
    pub metadata: HashMap<String, String>,

    /// When the pattern was first detected
    pub first_detected: Instant,

    /// When the pattern was last confirmed
    pub last_confirmed: Instant,

    /// Number of times the pattern has been confirmed
    pub confirmation_count: usize,
}

impl RecognizedPattern {
    /// Create a new recognized pattern.
    pub fn new(patterntype: ComplexPattern, confidence: Confidence) -> Self {
        let now = Instant::now();
        Self {
            pattern_type: patterntype,
            confidence,
            metadata: HashMap::new(),
            first_detected: now,
            last_confirmed: now,
            confirmation_count: 1,
        }
    }

    /// Add metadata to the pattern.
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Confirm the pattern, increasing confidence.
    pub fn confirm(&mut self) {
        self.confirmation_count += 1;
        self.last_confirmed = Instant::now();

        // Increase confidence based on confirmation count
        if self.confirmation_count >= 10 {
            self.confidence = Confidence::High;
        } else if self.confirmation_count >= 5 {
            self.confidence = Confidence::Medium;
        } else if self.confirmation_count >= 2 {
            self.confidence = Confidence::Low;
        }
    }

    /// Check if the pattern is still valid.
    pub fn is_valid(&self, maxage: std::time::Duration) -> bool {
        self.last_confirmed.elapsed() <= maxage
    }
}

/// Configuration for pattern recognition.
#[derive(Debug, Clone)]
pub struct PatternRecognitionConfig {
    /// Minimum history size needed for pattern detection
    pub min_history_size: usize,

    /// Maximum time to consider a pattern valid without confirmation
    pub pattern_expiry: std::time::Duration,

    /// Whether to detect diagonal patterns
    pub detect_diagonal: bool,

    /// Whether to detect block patterns
    pub detect_block: bool,

    /// Whether to detect stencil patterns
    pub detect_stencil: bool,

    /// Whether to detect sparse patterns
    pub detect_sparse: bool,

    /// Whether to use machine learning for pattern detection
    pub use_machine_learning: bool,
}

impl Default for PatternRecognitionConfig {
    fn default() -> Self {
        Self {
            min_history_size: 20,
            pattern_expiry: std::time::Duration::from_secs(60),
            detect_diagonal: true,
            detect_block: true,
            detect_stencil: true,
            detect_sparse: true,
            use_machine_learning: false, // Disabled by default as it requires more dependencies
        }
    }
}

/// Pattern recognition engine for complex access patterns.
#[derive(Debug)]
pub struct PatternRecognizer {
    /// Configuration for the recognizer
    config: PatternRecognitionConfig,

    /// Dimensions of the array
    dimensions: Option<Vec<usize>>,

    /// History of accessed indices
    history: VecDeque<usize>,

    /// Recognized patterns
    patterns: Vec<RecognizedPattern>,

    /// Most recently detected basic pattern
    basic_pattern: AccessPattern,
}

impl PatternRecognizer {
    /// Create a new pattern recognizer.
    pub fn new(config: PatternRecognitionConfig) -> Self {
        Self {
            config,
            dimensions: None,
            history: VecDeque::with_capacity(100),
            patterns: Vec::new(),
            basic_pattern: AccessPattern::Random,
        }
    }

    /// Set the dimensions of the array.
    pub fn set_dimensions(&mut self, dimensions: Vec<usize>) {
        self.dimensions = Some(dimensions);
    }

    /// Add a new access to the history.
    pub fn record_access(&mut self, index: usize) {
        self.history.push_back(index);

        // Limit history size
        while self.history.len() > 100 {
            self.history.pop_front();
        }

        // Only try to detect patterns if we have enough history
        if self.history.len() >= self.config.min_history_size {
            self.detect_patterns();
        }
    }

    /// Detect patterns in the current history.
    fn detect_patterns(&mut self) {
        // Remove expired patterns
        self.patterns
            .retain(|pattern| pattern.is_valid(self.config.pattern_expiry));

        // Basic patterns
        self.detect_basic_patterns();

        // Complex patterns based on dimensions
        if let Some(dims) = self.dimensions.clone() {
            // Detect matrix traversal patterns
            if dims.len() >= 2 {
                self.detectmatrix_patterns(&dims);
            }

            // Detect block patterns
            if self.config.detect_block && dims.len() >= 2 {
                self.detect_block_patterns(&dims);
            }

            // Detect diagonal patterns
            if self.config.detect_diagonal && dims.len() == 2 {
                self.detect_diagonal_patterns(&dims);
            }

            // Detect stencil patterns
            if self.config.detect_stencil && dims.len() >= 2 {
                self.detect_stencil_patterns(&dims);
            }
        }

        // Detect sparse patterns
        if self.config.detect_sparse {
            self.detect_sparse_pattern();
        }
    }

    /// Detect basic sequential and strided patterns.
    fn detect_basic_patterns(&mut self) {
        let indices: Vec<_> = self.history.iter().cloned().collect();

        // Check for sequential access
        let mut sequential_count = 0;
        for i in 1..indices.len() {
            if indices[i] == indices[i.saturating_sub(1)] + 1 {
                sequential_count += 1;
            }
        }

        if sequential_count >= indices.len() * 3 / 4 {
            self.basic_pattern = AccessPattern::Sequential;

            // Check for row-major pattern if dimensions are known
            if let Some(ref dims) = self.dimensions {
                if dims.len() >= 2 {
                    let row_size = dims[1];
                    let pattern = ComplexPattern::RowMajor;

                    // Check if we already have this pattern
                    if let Some(existing) = self.find_pattern(&pattern) {
                        // Confirm existing pattern
                        existing.confirm();
                    } else {
                        // Add new pattern
                        let pattern = RecognizedPattern::new(pattern, Confidence::Medium)
                            .with_metadata("row_size", &row_size.to_string());
                        self.patterns.push(pattern);
                    }
                }
            }

            return;
        }

        // Check for strided access - try different strides
        let mut best_stride = 0;
        let mut best_stride_count = 0;

        for stride in 2..=20 {
            let mut stride_count = 0;
            for i in 1..indices.len() {
                if indices[i].saturating_sub(indices[i.saturating_sub(1)]) == stride {
                    stride_count += 1;
                }
            }

            if stride_count > best_stride_count {
                best_stride_count = stride_count;
                best_stride = stride;
            }
        }

        if best_stride_count >= indices.len() * 2 / 3 {
            self.basic_pattern = AccessPattern::Strided(best_stride);

            // Check for column-major pattern if dimensions are known
            if let Some(ref dims) = self.dimensions {
                if dims.len() >= 2 {
                    let num_rows = dims[0];

                    if best_stride == num_rows {
                        let pattern = ComplexPattern::ColumnMajor;

                        // Check if we already have this pattern
                        if let Some(existing) = self.find_pattern(&pattern) {
                            // Confirm existing pattern
                            existing.confirm();
                        } else {
                            // Add new pattern
                            let pattern = RecognizedPattern::new(pattern, Confidence::Medium)
                                .with_metadata("num_rows", &num_rows.to_string());
                            self.patterns.push(pattern);
                        }
                    }
                }
            }

            return;
        }

        // No simple pattern detected
        self.basic_pattern = AccessPattern::Random;
    }

    /// Detect matrix traversal patterns.
    fn detectmatrix_patterns(&mut self, dimensions: &[usize]) {
        if dimensions.len() < 2 {
            return;
        }

        let _rows = dimensions[0];
        let cols = dimensions[1];
        let indices: Vec<_> = self.history.iter().cloned().collect();

        // Check for zigzag pattern - alternating left-right traversal within rows
        let mut zigzag_evidence = 0;
        let mut last_row_direction = None;

        // Group indices by rows and preserve access order
        let mut rows: HashMap<usize, Vec<(usize, usize)>> = HashMap::new(); // (col_index, access_order)
        for (access_order, &idx) in indices.iter().enumerate() {
            let row = idx / cols;
            let col = idx % cols;
            rows.entry(row).or_default().push((col, access_order));
        }

        // Check if consecutive rows alternate direction based on access order
        let sorted_rows: Vec<_> = {
            let mut sorted = rows.keys().cloned().collect::<Vec<_>>();
            sorted.sort();
            sorted
        };

        for row_num in &sorted_rows {
            let mut cols_in_row = rows[row_num].clone();
            if cols_in_row.len() >= 2 {
                // Sort by access order to see the actual traversal pattern
                cols_in_row.sort_by_key(|(_, access_order)| *access_order);

                // Determine direction within this row based on column progression
                // Check if columns are accessed in increasing or decreasing order
                let mut increasing = 0;
                let mut decreasing = 0;
                for i in 1..cols_in_row.len() {
                    match cols_in_row[i].0.cmp(&cols_in_row[i.saturating_sub(1)].0) {
                        std::cmp::Ordering::Greater => increasing += 1,
                        std::cmp::Ordering::Less => decreasing += 1,
                        std::cmp::Ordering::Equal => {}
                    }
                }

                let current_direction = match increasing.cmp(&decreasing) {
                    std::cmp::Ordering::Greater => 1, // Left to right
                    std::cmp::Ordering::Less => -1,   // Right to left
                    std::cmp::Ordering::Equal => 0,   // No clear direction
                };

                // Check if direction alternates from previous row
                if current_direction != 0 {
                    if let Some(prev_direction) = last_row_direction {
                        if current_direction != prev_direction && prev_direction != 0 {
                            zigzag_evidence += 1;
                        }
                    }
                    last_row_direction = Some(current_direction);
                }
            }
        }

        // To confirm zigzag, we need at least 2 direction changes (3 rows minimum)
        // Also ensure we have seen enough rows to make this determination
        if zigzag_evidence >= 2 && sorted_rows.len() >= 3 {
            let pattern = ComplexPattern::Zigzag;

            // Check if we already have this pattern
            if let Some(existing) = self.find_pattern(&pattern) {
                // Confirm existing pattern
                existing.confirm();
            } else {
                // Add new pattern
                let pattern = RecognizedPattern::new(pattern, Confidence::Medium)
                    .with_metadata("zigzag_evidence", &zigzag_evidence.to_string());
                self.patterns.push(pattern);
            }
        }
    }

    /// Detect diagonal traversal patterns.
    fn detect_diagonal_patterns(&mut self, dimensions: &[usize]) {
        if dimensions.len() != 2 {
            return;
        }

        let _rows = dimensions[0];
        let cols = dimensions[1];
        let indices: Vec<_> = self.history.iter().cloned().collect();

        // Check for main diagonal traversal
        let mut diagonal_matches = 0;
        for i in 1..indices.len() {
            let prev_idx = indices[i.saturating_sub(1)];
            let curr_idx = indices[i];

            let prev_row = prev_idx / cols;
            let prev_col = prev_idx % cols;

            let curr_row = curr_idx / cols;
            let curr_col = curr_idx % cols;

            // Check if moving along the main diagonal (row+1, col+1)
            if curr_row == prev_row + 1 && curr_col == prev_col + 1 {
                diagonal_matches += 1;
            }
        }

        // Need a significant portion of transitions to be diagonal
        // For consecutive diagonal accesses, we expect (n-1) diagonal transitions
        let expected_transitions = indices.len().saturating_sub(1);
        // Lower threshold: at least 1/3 of transitions or at least 3 diagonal matches
        if (diagonal_matches >= expected_transitions / 3 || diagonal_matches >= 3)
            && diagonal_matches > 0
        {
            let pattern = ComplexPattern::DiagonalMajor;

            // Check if we already have this pattern
            if let Some(existing) = self.find_pattern(&pattern) {
                // Confirm existing pattern
                existing.confirm();
            } else {
                // Add new pattern
                let pattern = RecognizedPattern::new(pattern, Confidence::Medium)
                    .with_metadata("diagonal_matches", &diagonal_matches.to_string());
                self.patterns.push(pattern);
            }

            return;
        }

        // Check for anti-diagonal traversal
        let mut anti_diagonal_matches = 0;
        for i in 1..indices.len() {
            let prev_idx = indices[i.saturating_sub(1)];
            let curr_idx = indices[i];

            let prev_row = prev_idx / cols;
            let prev_col = prev_idx % cols;

            let curr_row = curr_idx / cols;
            let curr_col = curr_idx % cols;

            // Check if moving along the anti-diagonal (row+1, col-1)
            if curr_row == prev_row + 1 && curr_col + 1 == prev_col {
                anti_diagonal_matches += 1;
            }
        }

        // Need a significant portion of transitions to be anti-diagonal
        let expected_transitions = indices.len().saturating_sub(1);
        // Lower threshold: at least 1/3 of transitions or at least 3 anti-diagonal matches
        if (anti_diagonal_matches >= expected_transitions / 3 || anti_diagonal_matches >= 3)
            && anti_diagonal_matches > 0
        {
            let pattern = ComplexPattern::DiagonalMinor;

            // Check if we already have this pattern
            if let Some(existing) = self.find_pattern(&pattern) {
                // Confirm existing pattern
                existing.confirm();
            } else {
                // Add new pattern
                let pattern = RecognizedPattern::new(pattern, Confidence::Medium)
                    .with_metadata("anti_diagonal_matches", &anti_diagonal_matches.to_string());
                self.patterns.push(pattern);
            }
        }
    }

    /// Detect block-based access patterns.
    fn detect_block_patterns(&mut self, dimensions: &[usize]) {
        if dimensions.len() < 2 {
            return;
        }

        let rows = dimensions[0];
        let cols = dimensions[1];
        let indices: Vec<_> = self.history.iter().cloned().collect();

        // Try different block sizes
        let block_sizes_to_try = [
            (2, 2),
            (4, 4),
            (8, 8),
            (16, 16),
            (32, 32),
            (64, 64),
            (rows, 4),
            (4, cols),
        ];

        for &(block_height, block_width) in &block_sizes_to_try {
            // Skip invalid block sizes
            if block_height > rows || block_width > cols {
                continue;
            }

            let mut block_accesses = HashMap::new();

            // Group accesses by block
            for &idx in &indices {
                let row = idx / cols;
                let col = idx % cols;

                let block_row = row / block_height;
                let block_col = col / block_width;

                let block_id = (block_row, block_col);
                let entry: &mut Vec<usize> = block_accesses.entry(block_id).or_default();
                entry.push(idx);
            }

            // Check for complete blocks (where all elements in the block are accessed)
            let mut complete_blocks = 0;
            for accesses in block_accesses.values() {
                if accesses.len() == block_height * block_width {
                    complete_blocks += 1;
                }
            }

            // Check if we have evidence of block-based access
            if complete_blocks >= 2 && block_accesses.len() <= 10 {
                let pattern = ComplexPattern::Block {
                    block_height,
                    block_width,
                };

                // Check if we already have this pattern
                if let Some(existing) = self.find_pattern(&pattern) {
                    // Confirm existing pattern
                    existing.confirm();
                } else {
                    // Add new pattern
                    let pattern = RecognizedPattern::new(pattern, Confidence::Medium)
                        .with_metadata("complete_blocks", &complete_blocks.to_string())
                        .with_metadata("total_blocks", &block_accesses.len().to_string());
                    self.patterns.push(pattern);
                }
            }
        }

        // Check for strided access within blocks
        let mut block_strides = HashMap::new();

        // Group consecutive accesses by stride
        for i in 1..indices.len() {
            let stride = indices[i].saturating_sub(indices[i.saturating_sub(1)]);
            *block_strides.entry(stride).or_insert(0) += 1;
        }

        // Find the most common stride
        if let Some((&stride, &count)) = block_strides.iter().max_by_key(|(_, &count)| count) {
            if count >= indices.len() / 3 && stride > 1 {
                // Try to determine if this is within-block striding
                let possible_block_sizes = [8, 16, 32, 64, 128];

                for &block_size in &possible_block_sizes {
                    if stride < block_size && block_size % stride == 0 {
                        let pattern = ComplexPattern::BlockStrided { block_size, stride };

                        // Check if we already have this pattern
                        if let Some(existing) = self.find_pattern(&pattern) {
                            // Confirm existing pattern
                            existing.confirm();
                        } else {
                            // Add new pattern
                            let pattern = RecognizedPattern::new(pattern, Confidence::Low)
                                .with_metadata("stride_count", &count.to_string())
                                .with_metadata(
                                    "total_transitions",
                                    &(indices.len() - 1).to_string(),
                                );
                            self.patterns.push(pattern);
                        }

                        break; // Only add one block stride pattern
                    }
                }
            }
        }
    }

    /// Detect stencil operation patterns.
    fn detect_stencil_patterns(&mut self, dimensions: &[usize]) {
        if dimensions.len() < 2 {
            return;
        }

        let _rows = dimensions[0];
        let cols = dimensions[1];
        let indices: Vec<_> = self.history.iter().cloned().collect();

        // Look for classic stencil patterns (5-point stencil)
        // Pattern: center, then 4 neighbors (N, E, S, W)
        let mut stencil_groups = 0;

        // Look for groups of 5 consecutive accesses that form a stencil
        for window_start in 0..indices.len().saturating_sub(4) {
            if window_start + 4 >= indices.len() {
                break;
            }

            let center_idx = indices[window_start];
            let center_row = center_idx / cols;
            let center_col = center_idx % cols;

            // Check if the next 4 accesses are neighbors of the center
            let mut neighbors_found = 0;
            let expected_neighbors = [
                center_idx.saturating_sub(cols), // North
                center_idx + 1,                  // East
                center_idx + cols,               // South
                center_idx.saturating_sub(1),    // West
            ];

            for offset in 1..=4 {
                if window_start + offset < indices.len() {
                    let neighbor_idx = indices[window_start + offset];
                    if expected_neighbors.contains(&neighbor_idx) {
                        neighbors_found += 1;
                    }
                }
            }

            // If we found at least 3 of the 4 expected neighbors, count as stencil
            if neighbors_found >= 3 {
                stencil_groups += 1;
            }
        }

        // If we found enough stencil patterns, recognize it
        if stencil_groups >= 3 {
            let pattern = ComplexPattern::Stencil {
                dimensions: 2,
                radius: 1,
            };

            // Check if we already have this pattern
            if let Some(existing) = self.find_pattern(&pattern) {
                // Confirm existing pattern
                existing.confirm();
            } else {
                // Add new pattern
                let pattern = RecognizedPattern::new(pattern, Confidence::Medium)
                    .with_metadata("stencil_groups", &stencil_groups.to_string());
                self.patterns.push(pattern);
            }
        }
    }

    /// Detect sparse access patterns.
    fn detect_sparse_pattern(&mut self) {
        let indices: Vec<_> = self.history.iter().cloned().collect();

        // Skip if history is too small
        if indices.len() < 20 {
            return;
        }

        // Estimate the total space from the max index
        if let Some(&max_idx) = indices.iter().max() {
            let unique_indices = indices.iter().collect::<HashSet<_>>().len();

            // Calculate density: unique indices accessed / total space
            let density = unique_indices as f64 / (max_idx + 1) as f64;

            // If density is low, consider it sparse
            if density < 0.1 {
                let pattern = ComplexPattern::Sparse { density };

                // Check if we already have this pattern
                if let Some(existing) = self.find_pattern(&pattern) {
                    // Confirm existing pattern
                    existing.confirm();
                } else {
                    // Add new pattern
                    let confidence = if density < 0.01 {
                        Confidence::High
                    } else if density < 0.05 {
                        Confidence::Medium
                    } else {
                        Confidence::Low
                    };

                    let pattern = RecognizedPattern::new(pattern, confidence)
                        .with_metadata("unique_indices", &unique_indices.to_string())
                        .with_metadata("max_index", &max_idx.to_string())
                        .with_metadata("density", &format!("{density:.6}"));
                    self.patterns.push(pattern);
                }
            }
        }
    }

    /// Find an existing pattern by type.
    fn find_pattern(&mut self, patterntype: &ComplexPattern) -> Option<&mut RecognizedPattern> {
        self.patterns
            .iter_mut()
            .find(|p| &p.pattern_type == patterntype)
    }

    /// Get all recognized patterns ordered by confidence.
    pub fn get_patterns(&self) -> Vec<&RecognizedPattern> {
        let mut patterns: Vec<_> = self.patterns.iter().collect();
        patterns.sort_by(|a, b| b.confidence.cmp(&a.confidence));
        patterns
    }

    /// Get the best pattern for prefetching.
    pub fn get_best_pattern(&self) -> Option<&RecognizedPattern> {
        self.patterns
            .iter()
            .filter(|p| p.confidence >= Confidence::Medium)
            .max_by_key(|p| p.confidence)
    }

    /// Get the current basic access pattern.
    pub fn get_basic_pattern(&self) -> AccessPattern {
        self.basic_pattern
    }

    /// Clear all detected patterns and history.
    pub fn clear(&mut self) {
        self.history.clear();
        self.patterns.clear();
        self.basic_pattern = AccessPattern::Random;
    }
}

/// Factory for creating pattern recognizers.
#[allow(dead_code)]
pub struct PatternRecognizerFactory;

#[allow(dead_code)]
impl PatternRecognizerFactory {
    /// Create a new pattern recognizer with default configuration.
    pub fn create() -> PatternRecognizer {
        PatternRecognizer::new(PatternRecognitionConfig::default())
    }

    /// Create a new pattern recognizer with the specified configuration.
    pub fn create_with_config(config: PatternRecognitionConfig) -> PatternRecognizer {
        PatternRecognizer::new(config)
    }
}

/// Helper functions for converting between pattern types.
pub mod pattern_utils {
    use super::*;
    use crate::memory_efficient::prefetch::AccessPattern;

    /// Convert from complex pattern to basic pattern.
    #[allow(dead_code)]
    pub fn to_basic_pattern(pattern: &ComplexPattern) -> AccessPattern {
        match pattern {
            ComplexPattern::RowMajor => AccessPattern::Sequential,
            ComplexPattern::ColumnMajor => AccessPattern::Strided(0), // Stride depends on dimensions
            ComplexPattern::Zigzag => AccessPattern::Custom,
            ComplexPattern::DiagonalMajor => AccessPattern::Custom,
            ComplexPattern::DiagonalMinor => AccessPattern::Custom,
            ComplexPattern::Block { .. } => AccessPattern::Custom,
            ComplexPattern::BlockStrided { stride, .. } => AccessPattern::Strided(*stride),
            ComplexPattern::Stencil { .. } => AccessPattern::Custom,
            ComplexPattern::RotatingBlock { .. } => AccessPattern::Custom,
            ComplexPattern::Sparse { .. } => AccessPattern::Random,
            ComplexPattern::Hierarchical { .. } => AccessPattern::Custom,
            ComplexPattern::Custom(_) => AccessPattern::Custom,
        }
    }

    /// Get the prefetch pattern for a complex pattern.
    #[allow(dead_code)]
    pub fn get_prefetch_pattern(
        pattern: &ComplexPattern,
        dimensions: &[usize],
        current_idx: usize,
        prefetch_count: usize,
    ) -> Vec<usize> {
        match pattern {
            ComplexPattern::RowMajor => {
                // For row-major, prefetch the next sequential indices
                (1..=prefetch_count).map(|i| current_idx + i).collect()
            }
            ComplexPattern::ColumnMajor => {
                if dimensions.len() >= 2 {
                    let stride = dimensions[0];
                    // For column-major, prefetch with stride equal to number of rows
                    (1..=prefetch_count)
                        .map(|i| current_idx + stride * i)
                        .collect()
                } else {
                    // Default to sequential if dimensions unknown
                    (1..=prefetch_count).map(|i| current_idx + i).collect()
                }
            }
            ComplexPattern::Zigzag => {
                if dimensions.len() >= 2 {
                    let cols = dimensions[1];
                    let row = current_idx / cols;
                    let col = current_idx % cols;

                    // In zigzag, alternating rows go in opposite directions
                    let mut result = Vec::with_capacity(prefetch_count);

                    if row % 2 == 0 {
                        // Even rows go left to right
                        for i in 1..=prefetch_count {
                            if col + i < cols {
                                // Continue in this row
                                result.push(current_idx + i);
                            } else {
                                // Next row, right to left
                                let overflow = (col + i) - cols;
                                result.push(current_idx + (cols - col) + (cols - 1) - overflow);
                            }
                        }
                    } else {
                        // Odd rows go right to left
                        for i in 1..=prefetch_count {
                            if col >= i {
                                // Continue in this row
                                result.push(current_idx - i);
                            } else {
                                // Next row, left to right
                                let overflow = i - col;
                                result.push(current_idx + (col + 1) + overflow);
                            }
                        }
                    }

                    result
                } else {
                    // Default to sequential if dimensions unknown
                    (1..=prefetch_count).map(|i| current_idx + i).collect()
                }
            }
            ComplexPattern::DiagonalMajor => {
                if dimensions.len() >= 2 {
                    let cols = dimensions[1];
                    // For diagonal, move down and right
                    (1..=prefetch_count)
                        .map(|i| current_idx + cols * i + i)
                        .collect()
                } else {
                    // Default to sequential if dimensions unknown
                    (1..=prefetch_count).map(|i| current_idx + i).collect()
                }
            }
            ComplexPattern::DiagonalMinor => {
                if dimensions.len() >= 2 {
                    let cols = dimensions[1];
                    // For anti-diagonal, move down and left
                    (1..=prefetch_count)
                        .map(|i| current_idx + cols * i - i)
                        .collect()
                } else {
                    // Default to sequential if dimensions unknown
                    (1..=prefetch_count).map(|i| current_idx + i).collect()
                }
            }
            ComplexPattern::Block {
                block_height,
                block_width,
            } => {
                if dimensions.len() >= 2 {
                    let cols = dimensions[1];
                    let row = current_idx / cols;
                    let col = current_idx % cols;

                    // Calculate block coordinates
                    let block_row = row / *block_height;
                    let block_col = col / *block_width;

                    // Calculate position within block
                    let block_row_offset = row % *block_height;
                    let block_col_offset = col % *block_width;

                    // Predict next positions within the block (row-major within block)
                    let mut result = Vec::with_capacity(prefetch_count);
                    let mut remaining = prefetch_count;

                    // First, complete the current row in the block
                    for i in 1..=std::cmp::min(*block_width - block_col_offset, remaining) {
                        result.push(current_idx + i);
                        remaining -= 1;
                    }

                    // Then, continue with subsequent rows in the block
                    let mut next_row = block_row_offset + 1;
                    while remaining > 0 && next_row < *block_height {
                        for col_offset in 0..std::cmp::min(*block_width, remaining) {
                            let idx = (block_row * *block_height + next_row) * cols
                                + block_col * *block_width
                                + col_offset;
                            result.push(idx);
                            remaining -= 1;
                        }
                        next_row += 1;
                    }

                    // If still remaining, move to next block
                    if remaining > 0 {
                        let next_block_row = if block_col + 1 < cols / *block_width {
                            block_row // Same row, next column
                        } else {
                            block_row + 1 // Next row, first column
                        };

                        let next_block_col = if block_col + 1 < cols / *block_width {
                            block_col + 1 // Next column
                        } else {
                            0 // First column
                        };

                        // Add first few elements of next block
                        for i in 0..remaining {
                            let row_offset = i / *block_width;
                            let col_offset = i % *block_width;
                            let idx = (next_block_row * *block_height + row_offset) * cols
                                + next_block_col * *block_width
                                + col_offset;
                            result.push(idx);
                        }
                    }

                    result
                } else {
                    // Default to sequential if dimensions unknown
                    (1..=prefetch_count).map(|i| current_idx + i).collect()
                }
            }
            ComplexPattern::BlockStrided { block_size, stride } => {
                // Prefetch with the specified stride within the block
                (1..=prefetch_count)
                    .map(|i| {
                        let offset = i * stride;
                        let block_offset = offset % block_size;
                        let blocks_advanced = offset / block_size;

                        if blocks_advanced == 0 {
                            // Still in same block
                            current_idx + offset
                        } else {
                            // Advanced to next block(s)
                            current_idx + block_size * blocks_advanced + block_offset
                        }
                    })
                    .collect()
            }
            ComplexPattern::Stencil {
                dimensions: dim_count,
                radius,
            } => {
                if dimensions.len() >= *dim_count {
                    let cols = dimensions[1];
                    let row = current_idx / cols;
                    let col = current_idx % cols;

                    // In a stencil operation, predict accesses to neighboring cells
                    let mut result = Vec::new();

                    // Add cells in a radius around the current position
                    for r in -(*radius as isize)..=(*radius as isize) {
                        for c in -(*radius as isize)..=(*radius as isize) {
                            // Skip the center (current) position
                            if r == 0 && c == 0 {
                                continue;
                            }

                            let new_row = row as isize + r;
                            let new_col = col as isize + c;

                            // Check bounds
                            if new_row >= 0
                                && new_row < dimensions[0] as isize
                                && new_col >= 0
                                && new_col < cols as isize
                            {
                                let idx = (new_row as usize) * cols + (new_col as usize);
                                result.push(idx);
                            }
                        }
                    }

                    // Take only the requested number of predictions
                    result.into_iter().take(prefetch_count).collect()
                } else {
                    // Default to sequential if dimensions unknown
                    (1..=prefetch_count).map(|i| current_idx + i).collect()
                }
            }
            // For other patterns, default to nearby elements
            _ => {
                let mut result = Vec::with_capacity(prefetch_count);

                // Try to prefetch a mix of sequential and nearby indices
                for i in 1..=prefetch_count / 2 {
                    result.push(current_idx + i);
                }

                if dimensions.len() >= 2 {
                    let cols = dimensions[1];
                    // Add row above and below
                    result.push(current_idx.saturating_sub(cols));
                    result.push(current_idx + cols);
                }

                // Fill remaining slots with sequential prefetches
                while result.len() < prefetch_count {
                    result.push(current_idx + result.len() + 1);
                }

                // Deduplicate
                result
                    .into_iter()
                    .collect::<HashSet<_>>()
                    .into_iter()
                    .collect()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_row_major_detection() {
        let mut recognizer = PatternRecognizer::new(PatternRecognitionConfig::default());
        recognizer.set_dimensions(vec![8, 8]);

        // Record row-major traversal (sequential)
        for i in 0..64 {
            recognizer.record_access(i);
        }

        // Get detected patterns
        let patterns = recognizer.get_patterns();

        // Should detect row-major pattern
        assert!(patterns
            .iter()
            .any(|p| matches!(p.pattern_type, ComplexPattern::RowMajor)));

        // Check basic pattern
        assert_eq!(recognizer.get_basic_pattern(), AccessPattern::Sequential);
    }

    #[test]
    fn test_column_major_detection() {
        let mut recognizer = PatternRecognizer::new(PatternRecognitionConfig::default());
        recognizer.set_dimensions(vec![8, 8]);

        // Record column-major traversal
        for j in 0..8 {
            for i in 0..8 {
                recognizer.record_access(i * 8 + j);
            }
        }

        // Get detected patterns
        let patterns = recognizer.get_patterns();

        // Should detect column-major pattern
        assert!(patterns
            .iter()
            .any(|p| matches!(p.pattern_type, ComplexPattern::ColumnMajor)));

        // Check basic pattern - should be strided
        assert!(matches!(
            recognizer.get_basic_pattern(),
            AccessPattern::Strided(_)
        ));
    }

    #[test]
    fn test_zigzag_detection() {
        let config = PatternRecognitionConfig {
            min_history_size: 10, // Lower threshold for test
            ..Default::default()
        };
        let mut recognizer = PatternRecognizer::new(config);
        recognizer.set_dimensions(vec![8, 8]);

        // Record zigzag traversal - multiple complete rows to ensure enough data
        for row in 0..8 {
            if row % 2 == 0 {
                // Even rows: left to right
                for j in 0..8 {
                    recognizer.record_access(row * 8 + j);
                }
            } else {
                // Odd rows: right to left
                for j in (0..8).rev() {
                    recognizer.record_access(row * 8 + j);
                }
            }
        }

        // Get detected patterns
        let patterns = recognizer.get_patterns();

        // Should detect zigzag pattern
        assert!(patterns
            .iter()
            .any(|p| matches!(p.pattern_type, ComplexPattern::Zigzag)));
    }

    #[test]
    fn test_diagonal_detection() {
        let config = PatternRecognitionConfig {
            min_history_size: 10, // Lower threshold for test
            ..Default::default()
        };
        let mut recognizer = PatternRecognizer::new(config);
        recognizer.set_dimensions(vec![16, 16]);

        // Record diagonal traversal - longer diagonal to ensure enough data
        for i in 0..16 {
            recognizer.record_access(i * 16 + i);
        }

        // Add a few more diagonal elements to strengthen the pattern
        for i in 0..8 {
            recognizer.record_access(i * 16 + i);
        }

        // Get detected patterns
        let patterns = recognizer.get_patterns();

        // Should detect diagonal pattern
        assert!(patterns
            .iter()
            .any(|p| matches!(p.pattern_type, ComplexPattern::DiagonalMajor)));
    }

    #[test]
    fn test_block_detection() {
        let mut recognizer = PatternRecognizer::new(PatternRecognitionConfig::default());
        recognizer.set_dimensions(vec![8, 8]);

        // Record block traversal (4x4 blocks)
        // First block (top-left)
        for i in 0..4 {
            for j in 0..4 {
                recognizer.record_access(i * 8 + j);
            }
        }
        // Second block (top-right)
        for i in 0..4 {
            for j in 4..8 {
                recognizer.record_access(i * 8 + j);
            }
        }

        // Get detected patterns
        let patterns = recognizer.get_patterns();

        // Should detect block pattern
        assert!(patterns.iter().any(|p| {
            if let ComplexPattern::Block {
                block_height,
                block_width,
            } = p.pattern_type
            {
                block_height == 4 && block_width == 4
            } else {
                false
            }
        }));
    }

    #[test]
    fn test_stencil_detection() {
        let mut recognizer = PatternRecognizer::new(PatternRecognitionConfig::default());
        recognizer.set_dimensions(vec![10, 10]);

        // Record stencil operations (5-point stencil)
        for i in 1..9 {
            for j in 1..9 {
                // Center point
                let center = i * 10 + j;
                recognizer.record_access(center);

                // 4 neighbors (north, east, south, west)
                recognizer.record_access(center - 10); // North
                recognizer.record_access(center + 1); // East
                recognizer.record_access(center + 10); // South
                recognizer.record_access(center - 1); // West
            }
        }

        // Get detected patterns
        let patterns = recognizer.get_patterns();

        // Should detect stencil pattern
        assert!(patterns.iter().any(|p| {
            if let ComplexPattern::Stencil { dimensions, radius } = p.pattern_type {
                dimensions == 2 && radius == 1
            } else {
                false
            }
        }));
    }

    #[test]
    fn test_pattern_utils() {
        // Test row-major prefetching
        let pattern = ComplexPattern::RowMajor;
        let dimensions = vec![8, 8];
        let current_idx = 10;
        let prefetch_count = 3;

        let prefetches =
            pattern_utils::get_prefetch_pattern(&pattern, &dimensions, current_idx, prefetch_count);

        // Should prefetch the next 3 indices
        assert_eq!(prefetches, vec![11, 12, 13]);

        // Test diagonal prefetching
        let pattern = ComplexPattern::DiagonalMajor;

        let prefetches =
            pattern_utils::get_prefetch_pattern(&pattern, &dimensions, current_idx, prefetch_count);

        // Should prefetch along the diagonal (down-right)
        // Each step adds row_stride + 1
        assert_eq!(prefetches, vec![19, 28, 37]);
    }
}
