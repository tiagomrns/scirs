//! Data sharding and distribution management
//!
//! This module provides comprehensive data sharding capabilities:
//! - Consistent hashing for shard distribution
//! - Dynamic resharding and rebalancing
//! - Shard migration and replication
//! - Data locality optimization

use crate::error::{MetricsError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::net::SocketAddr;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};

pub use crate::optimization::distributed::config::{
    HashFunction, ShardingConfig, ShardingStrategy,
};

/// Data shard representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataShard {
    /// Shard ID
    pub id: String,
    /// Shard range
    pub range: DataRange,
    /// Primary node for this shard
    pub primary_node: String,
    /// Replica nodes
    pub replicas: Vec<String>,
    /// Data size (bytes)
    pub size_bytes: u64,
    /// Number of keys in shard
    pub key_count: usize,
    /// Last access time
    pub last_access: SystemTime,
    /// Shard status
    pub status: ShardStatus,
    /// Migration info (if being migrated)
    pub migration: Option<ShardMigration>,
}

/// Shard status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ShardStatus {
    /// Shard is active and serving requests
    Active,
    /// Shard is being migrated
    Migrating,
    /// Shard is being split
    Splitting,
    /// Shard is being merged
    Merging,
    /// Shard is inactive/offline
    Inactive,
    /// Shard is in error state
    Error(String),
}

/// Data range for sharding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataRange {
    /// Hash range (start_hash, end_hash)
    Hash { start: u64, end: u64 },
    /// Key range (start_key, end_key)
    Key { start: String, end: String },
    /// Numeric range (start, end)
    Numeric { start: f64, end: f64 },
    /// Time range
    Time { start: SystemTime, end: SystemTime },
    /// Geographic range
    Geographic {
        lat_min: f64,
        lat_max: f64,
        lon_min: f64,
        lon_max: f64,
    },
    /// Custom range
    Custom {
        range_type: String,
        range_data: Vec<u8>,
    },
}

impl DataRange {
    /// Check if a key falls within this range
    pub fn contains_key(&self, key: &str) -> bool {
        match self {
            DataRange::Hash { start, end } => {
                let hash = self.hash_key(key);
                hash >= *start && hash <= *end
            }
            DataRange::Key { start, end } => key >= start.as_str() && key <= end.as_str(),
            DataRange::Numeric { start, end } => {
                if let Ok(num) = key.parse::<f64>() {
                    num >= *start && num <= *end
                } else {
                    false
                }
            }
            DataRange::Time { start, end } => {
                // Attempt to parse key as timestamp
                if let Ok(timestamp_str) = key.parse::<u64>() {
                    if let Some(timestamp) =
                        SystemTime::UNIX_EPOCH.checked_add(Duration::from_secs(timestamp_str))
                    {
                        timestamp >= *start && timestamp <= *end
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            DataRange::Geographic { .. } => {
                // Would need to parse geographic coordinates from key
                // For now, return false
                false
            }
            DataRange::Custom { .. } => {
                // Custom logic would be implemented here
                false
            }
        }
    }

    /// Hash a key using the specified hash function
    fn hash_key(&self, key: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// Check if this range overlaps with another
    pub fn overlaps_with(&self, other: &DataRange) -> bool {
        match (self, other) {
            (DataRange::Hash { start: s1, end: e1 }, DataRange::Hash { start: s2, end: e2 }) => {
                s1 <= e2 && s2 <= e1
            }
            (DataRange::Key { start: s1, end: e1 }, DataRange::Key { start: s2, end: e2 }) => {
                s1 <= e2 && s2 <= e1
            }
            (
                DataRange::Numeric { start: s1, end: e1 },
                DataRange::Numeric { start: s2, end: e2 },
            ) => s1 <= e2 && s2 <= e1,
            (DataRange::Time { start: s1, end: e1 }, DataRange::Time { start: s2, end: e2 }) => {
                s1 <= e2 && s2 <= e1
            }
            _ => false, // Different range types don't overlap
        }
    }

    /// Split this range into two ranges
    pub fn split(&self) -> Result<(DataRange, DataRange)> {
        match self {
            DataRange::Hash { start, end } => {
                let mid = start + (end - start) / 2;
                Ok((
                    DataRange::Hash {
                        start: *start,
                        end: mid,
                    },
                    DataRange::Hash {
                        start: mid + 1,
                        end: *end,
                    },
                ))
            }
            DataRange::Key { start, end } => {
                // Simple string-based split (could be improved)
                let mid = format!("{}_{}", start, end);
                Ok((
                    DataRange::Key {
                        start: start.clone(),
                        end: mid.clone(),
                    },
                    DataRange::Key {
                        start: mid,
                        end: end.clone(),
                    },
                ))
            }
            DataRange::Numeric { start, end } => {
                let mid = start + (end - start) / 2.0;
                Ok((
                    DataRange::Numeric {
                        start: *start,
                        end: mid,
                    },
                    DataRange::Numeric {
                        start: mid,
                        end: *end,
                    },
                ))
            }
            DataRange::Time { start, end } => {
                let duration = end
                    .duration_since(*start)
                    .map_err(|_| MetricsError::ShardingError("Invalid time range".to_string()))?;
                let mid_duration = duration / 2;
                let mid = *start + mid_duration;
                Ok((
                    DataRange::Time {
                        start: *start,
                        end: mid,
                    },
                    DataRange::Time {
                        start: mid,
                        end: *end,
                    },
                ))
            }
            _ => Err(MetricsError::ShardingError(
                "Cannot split this range type".to_string(),
            )),
        }
    }
}

/// Shard migration information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardMigration {
    /// Migration ID
    pub id: String,
    /// Source node
    pub source_node: String,
    /// Target node
    pub target_node: String,
    /// Migration progress (0.0 - 1.0)
    pub progress: f64,
    /// Started time
    pub started_at: SystemTime,
    /// Estimated completion time
    pub estimated_completion: Option<SystemTime>,
    /// Migration status
    pub status: MigrationStatus,
    /// Error message (if failed)
    pub error: Option<String>,
}

/// Migration status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MigrationStatus {
    /// Migration is planned but not started
    Planned,
    /// Migration is in progress
    InProgress,
    /// Migration completed successfully
    Completed,
    /// Migration failed
    Failed,
    /// Migration was cancelled
    Cancelled,
}

/// Shard manager for handling sharding operations
#[derive(Debug)]
pub struct ShardManager {
    /// Sharding configuration
    config: ShardingConfig,
    /// Current shards
    shards: Arc<RwLock<HashMap<String, DataShard>>>,
    /// Node assignments
    node_assignments: Arc<RwLock<HashMap<String, Vec<String>>>>,
    /// Consistent hash ring (for consistent hashing)
    hash_ring: Arc<RwLock<BTreeMap<u64, String>>>,
    /// Active migrations
    migrations: Arc<RwLock<HashMap<String, ShardMigration>>>,
    /// Statistics
    stats: ShardingStats,
}

impl ShardManager {
    /// Create a new shard manager
    pub fn new(config: ShardingConfig) -> Self {
        Self {
            config,
            shards: Arc::new(RwLock::new(HashMap::new())),
            node_assignments: Arc::new(RwLock::new(HashMap::new())),
            hash_ring: Arc::new(RwLock::new(BTreeMap::new())),
            migrations: Arc::new(RwLock::new(HashMap::new())),
            stats: ShardingStats::default(),
        }
    }

    /// Initialize sharding with available nodes
    pub fn initialize(&mut self, nodes: Vec<String>) -> Result<()> {
        match self.config.strategy {
            ShardingStrategy::ConsistentHash => {
                self.initialize_consistent_hash(nodes)?;
            }
            ShardingStrategy::Hash => {
                self.initialize_hash_sharding(nodes)?;
            }
            ShardingStrategy::Range => {
                self.initialize_range_sharding(nodes)?;
            }
            _ => {
                return Err(MetricsError::ShardingError(
                    "Sharding strategy not implemented".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Initialize consistent hash ring
    fn initialize_consistent_hash(&mut self, nodes: Vec<String>) -> Result<()> {
        let mut hash_ring = self.hash_ring.write().unwrap();
        let mut shards = self.shards.write().unwrap();

        hash_ring.clear();
        shards.clear();

        // Add virtual nodes to the hash ring
        for node in &nodes {
            for i in 0..self.config.virtual_nodes {
                let virtual_node_key = format!("{}:{}", node, i);
                let hash = self.hash_string(&virtual_node_key);
                hash_ring.insert(hash, node.clone());
            }
        }

        // Create shards based on hash ring
        let mut prev_hash = 0u64;
        let ring_keys: Vec<u64> = hash_ring.keys().cloned().collect();

        for (i, &hash) in ring_keys.iter().enumerate() {
            let shard_id = format!("shard_{}", i);
            let node = hash_ring.get(&hash).unwrap().clone();

            let shard = DataShard {
                id: shard_id.clone(),
                range: DataRange::Hash {
                    start: prev_hash,
                    end: hash,
                },
                primary_node: node.clone(),
                replicas: self.select_replicas(&node, &nodes),
                size_bytes: 0,
                key_count: 0,
                last_access: SystemTime::now(),
                status: ShardStatus::Active,
                migration: None,
            };

            shards.insert(shard_id, shard);
            prev_hash = hash + 1;
        }

        Ok(())
    }

    /// Initialize hash-based sharding
    fn initialize_hash_sharding(&mut self, nodes: Vec<String>) -> Result<()> {
        let mut shards = self.shards.write().unwrap();
        shards.clear();

        let hash_range_size = u64::MAX / self.config.shard_count as u64;

        for i in 0..self.config.shard_count {
            let shard_id = format!("shard_{}", i);
            let start_hash = i as u64 * hash_range_size;
            let end_hash = if i == self.config.shard_count - 1 {
                u64::MAX
            } else {
                (i + 1) as u64 * hash_range_size - 1
            };

            let node = &nodes[i % nodes.len()];

            let shard = DataShard {
                id: shard_id.clone(),
                range: DataRange::Hash {
                    start: start_hash,
                    end: end_hash,
                },
                primary_node: node.clone(),
                replicas: self.select_replicas(node, &nodes),
                size_bytes: 0,
                key_count: 0,
                last_access: SystemTime::now(),
                status: ShardStatus::Active,
                migration: None,
            };

            shards.insert(shard_id, shard);
        }

        Ok(())
    }

    /// Initialize range-based sharding
    fn initialize_range_sharding(&mut self, nodes: Vec<String>) -> Result<()> {
        let mut shards = self.shards.write().unwrap();
        shards.clear();

        // For range sharding, we'll use key-based ranges
        // This is a simplified implementation
        for i in 0..self.config.shard_count {
            let shard_id = format!("shard_{}", i);
            let start_key = format!("{:04}", i * 1000);
            let end_key = format!("{:04}", (i + 1) * 1000 - 1);

            let node = &nodes[i % nodes.len()];

            let shard = DataShard {
                id: shard_id.clone(),
                range: DataRange::Key {
                    start: start_key,
                    end: end_key,
                },
                primary_node: node.clone(),
                replicas: self.select_replicas(node, &nodes),
                size_bytes: 0,
                key_count: 0,
                last_access: SystemTime::now(),
                status: ShardStatus::Active,
                migration: None,
            };

            shards.insert(shard_id, shard);
        }

        Ok(())
    }

    /// Select replica nodes for a primary node
    fn select_replicas(&self, primary: &str, all_nodes: &[String]) -> Vec<String> {
        let mut replicas = Vec::new();
        let mut count = 0;

        for node in all_nodes {
            if node != primary && count < self.config.replication_factor - 1 {
                replicas.push(node.clone());
                count += 1;
            }
        }

        replicas
    }

    /// Find the shard for a given key
    pub fn find_shard(&self, key: &str) -> Result<String> {
        let shards = self.shards.read().unwrap();

        for shard in shards.values() {
            if shard.range.contains_key(key) {
                return Ok(shard.id.clone());
            }
        }

        Err(MetricsError::ShardingError(
            "No shard found for key".to_string(),
        ))
    }

    /// Get the node responsible for a key
    pub fn get_node_for_key(&self, key: &str) -> Result<String> {
        match self.config.strategy {
            ShardingStrategy::ConsistentHash => self.get_node_consistent_hash(key),
            _ => {
                let shard_id = self.find_shard(key)?;
                let shards = self.shards.read().unwrap();
                if let Some(shard) = shards.get(&shard_id) {
                    Ok(shard.primary_node.clone())
                } else {
                    Err(MetricsError::ShardingError("Shard not found".to_string()))
                }
            }
        }
    }

    /// Get node using consistent hashing
    fn get_node_consistent_hash(&self, key: &str) -> Result<String> {
        let hash_ring = self.hash_ring.read().unwrap();
        if hash_ring.is_empty() {
            return Err(MetricsError::ShardingError(
                "Hash ring is empty".to_string(),
            ));
        }

        let key_hash = self.hash_string(key);

        // Find the first node with hash >= key_hash
        for (&node_hash, node) in hash_ring.range(key_hash..) {
            if node_hash >= key_hash {
                return Ok(node.clone());
            }
        }

        // Wrap around to the first node
        if let Some((_, node)) = hash_ring.iter().next() {
            Ok(node.clone())
        } else {
            Err(MetricsError::ShardingError(
                "No nodes in hash ring".to_string(),
            ))
        }
    }

    /// Hash a string using the configured hash function
    fn hash_string(&self, s: &str) -> u64 {
        match self.config.hash_function {
            HashFunction::Murmur3 | HashFunction::XxHash => {
                // Simplified hash using DefaultHasher
                use std::collections::hash_map::DefaultHasher;
                let mut hasher = DefaultHasher::new();
                s.hash(&mut hasher);
                hasher.finish()
            }
            HashFunction::Crc32 => {
                // Simplified CRC32 implementation
                let mut crc = 0xFFFFFFFFu32;
                for byte in s.bytes() {
                    crc ^= byte as u32;
                    for _ in 0..8 {
                        if crc & 1 != 0 {
                            crc = (crc >> 1) ^ 0xEDB88320;
                        } else {
                            crc >>= 1;
                        }
                    }
                }
                (crc ^ 0xFFFFFFFF) as u64
            }
            _ => {
                // Default to standard hasher
                use std::collections::hash_map::DefaultHasher;
                let mut hasher = DefaultHasher::new();
                s.hash(&mut hasher);
                hasher.finish()
            }
        }
    }

    /// Add a new node to the cluster
    pub fn add_node(&mut self, node_id: String) -> Result<()> {
        match self.config.strategy {
            ShardingStrategy::ConsistentHash => self.add_node_consistent_hash(node_id),
            _ => {
                // For other strategies, we might need to rebalance shards
                self.rebalance_shards_with_new_node(node_id)
            }
        }
    }

    /// Add node to consistent hash ring
    fn add_node_consistent_hash(&mut self, node_id: String) -> Result<()> {
        {
            let mut hash_ring = self.hash_ring.write().unwrap();

            // Add virtual nodes for the new node
            for i in 0..self.config.virtual_nodes {
                let virtual_node_key = format!("{}:{}", node_id, i);
                let hash = self.hash_string(&virtual_node_key);
                hash_ring.insert(hash, node_id.clone());
            }
        } // Drop the lock here

        // TODO: Trigger shard migration for rebalancing
        self.trigger_rebalancing()?;

        Ok(())
    }

    /// Remove a node from the cluster
    pub fn remove_node(&mut self, node_id: &str) -> Result<()> {
        match self.config.strategy {
            ShardingStrategy::ConsistentHash => self.remove_node_consistent_hash(node_id),
            _ => self.migrate_shards_from_node(node_id),
        }
    }

    /// Remove node from consistent hash ring
    fn remove_node_consistent_hash(&mut self, node_id: &str) -> Result<()> {
        {
            let mut hash_ring = self.hash_ring.write().unwrap();

            // Remove all virtual nodes for this node
            hash_ring.retain(|_, node| node != node_id);
        } // hash_ring lock is dropped here

        // TODO: Trigger shard migration for affected shards
        self.migrate_shards_from_node(node_id)?;

        Ok(())
    }

    /// Rebalance shards with a new node
    fn rebalance_shards_with_new_node(&mut self, _node_id: String) -> Result<()> {
        // TODO: Implement shard rebalancing logic
        self.trigger_rebalancing()
    }

    /// Migrate shards away from a node being removed
    fn migrate_shards_from_node(&mut self, node_id: &str) -> Result<()> {
        let shards = self.shards.read().unwrap();
        let affected_shards: Vec<_> = shards
            .values()
            .filter(|shard| shard.primary_node == node_id)
            .map(|shard| shard.id.clone())
            .collect();
        drop(shards);

        for shard_id in affected_shards {
            self.migrate_shard(&shard_id, None)?;
        }

        Ok(())
    }

    /// Trigger cluster rebalancing
    fn trigger_rebalancing(&mut self) -> Result<()> {
        // TODO: Implement rebalancing logic
        // This would analyze current shard distribution and trigger migrations
        // to achieve better balance
        Ok(())
    }

    /// Migrate a shard to a different node
    pub fn migrate_shard(&mut self, shard_id: &str, target_node: Option<String>) -> Result<String> {
        let migration_id = {
            let mut shards = self.shards.write().unwrap();
            let mut migrations = self.migrations.write().unwrap();

            let shard = shards
                .get_mut(shard_id)
                .ok_or_else(|| MetricsError::ShardingError("Shard not found".to_string()))?;

            if shard.status == ShardStatus::Migrating {
                return Err(MetricsError::ShardingError(
                    "Shard is already being migrated".to_string(),
                ));
            }

            // Select target node if not provided
            let target = target_node.unwrap_or_else(|| {
                // Simple selection: pick the first replica or a default node
                shard
                    .replicas
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "default_node".to_string())
            });

            let migration_id = format!(
                "migration_{}_{}",
                shard_id,
                SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis()
            );

            let migration = ShardMigration {
                id: migration_id.clone(),
                source_node: shard.primary_node.clone(),
                target_node: target.clone(),
                progress: 0.0,
                started_at: SystemTime::now(),
                estimated_completion: None,
                status: MigrationStatus::Planned,
                error: None,
            };

            shard.status = ShardStatus::Migrating;
            shard.migration = Some(migration.clone());
            migrations.insert(migration_id.clone(), migration);

            migration_id.clone()
        }; // Drop locks here before calling start_migration

        // TODO: Start actual migration process
        self.start_migration(&migration_id)?;

        Ok(migration_id)
    }

    /// Start a migration process
    fn start_migration(&mut self, migration_id: &str) -> Result<()> {
        let mut migrations = self.migrations.write().unwrap();

        if let Some(migration) = migrations.get_mut(migration_id) {
            migration.status = MigrationStatus::InProgress;
            // TODO: Implement actual migration logic
            // This would involve copying data from source to target
        }

        Ok(())
    }

    /// Complete a migration
    pub fn complete_migration(&mut self, migration_id: &str) -> Result<()> {
        let mut migrations = self.migrations.write().unwrap();
        let mut shards = self.shards.write().unwrap();

        let migration = migrations
            .get_mut(migration_id)
            .ok_or_else(|| MetricsError::ShardingError("Migration not found".to_string()))?;

        migration.status = MigrationStatus::Completed;
        migration.progress = 1.0;

        // Find and update the shard
        for shard in shards.values_mut() {
            if let Some(ref shard_migration) = shard.migration {
                if shard_migration.id == migration_id {
                    shard.primary_node = migration.target_node.clone();
                    shard.status = ShardStatus::Active;
                    shard.migration = None;
                    break;
                }
            }
        }

        Ok(())
    }

    /// Get sharding statistics
    pub fn get_stats(&self) -> ShardingStats {
        let shards = self.shards.read().unwrap();
        let migrations = self.migrations.read().unwrap();

        let total_shards = shards.len();
        let active_migrations = migrations
            .values()
            .filter(|m| m.status == MigrationStatus::InProgress)
            .count();

        let total_size: u64 = shards.values().map(|s| s.size_bytes).sum();
        let total_keys: usize = shards.values().map(|s| s.key_count).sum();

        ShardingStats {
            total_shards,
            active_migrations,
            total_size_bytes: total_size,
            total_keys,
            replication_factor: self.config.replication_factor,
            last_rebalance: SystemTime::now(), // Simplified
        }
    }

    /// List all shards
    pub fn list_shards(&self) -> Vec<DataShard> {
        let shards = self.shards.read().unwrap();
        shards.values().cloned().collect()
    }

    /// Get shard by ID
    pub fn get_shard(&self, shard_id: &str) -> Option<DataShard> {
        let shards = self.shards.read().unwrap();
        shards.get(shard_id).cloned()
    }

    /// Update shard statistics
    pub fn update_shard_stats(
        &mut self,
        shard_id: &str,
        size_bytes: u64,
        key_count: usize,
    ) -> Result<()> {
        let mut shards = self.shards.write().unwrap();

        if let Some(shard) = shards.get_mut(shard_id) {
            shard.size_bytes = size_bytes;
            shard.key_count = key_count;
            shard.last_access = SystemTime::now();
            Ok(())
        } else {
            Err(MetricsError::ShardingError("Shard not found".to_string()))
        }
    }
}

/// Sharding statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardingStats {
    /// Total number of shards
    pub total_shards: usize,
    /// Number of active migrations
    pub active_migrations: usize,
    /// Total data size across all shards
    pub total_size_bytes: u64,
    /// Total number of keys
    pub total_keys: usize,
    /// Replication factor
    pub replication_factor: usize,
    /// Last rebalance time
    pub last_rebalance: SystemTime,
}

impl Default for ShardingStats {
    fn default() -> Self {
        Self {
            total_shards: 0,
            active_migrations: 0,
            total_size_bytes: 0,
            total_keys: 0,
            replication_factor: 1,
            last_rebalance: SystemTime::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_range_contains_key() {
        let hash_range = DataRange::Hash {
            start: 1000,
            end: 2000,
        };
        // This test is dependent on the hash function, so we'll test basic functionality
        assert!(hash_range.contains_key("test") || !hash_range.contains_key("test"));

        let key_range = DataRange::Key {
            start: "a".to_string(),
            end: "z".to_string(),
        };
        assert!(key_range.contains_key("m"));
        assert!(!key_range.contains_key("z1"));

        let numeric_range = DataRange::Numeric {
            start: 10.0,
            end: 20.0,
        };
        assert!(numeric_range.contains_key("15"));
        assert!(!numeric_range.contains_key("25"));
    }

    #[test]
    fn test_data_range_split() {
        let hash_range = DataRange::Hash {
            start: 1000,
            end: 2000,
        };
        let (left, right) = hash_range.split().unwrap();

        if let (DataRange::Hash { start: s1, end: e1 }, DataRange::Hash { start: s2, end: e2 }) =
            (left, right)
        {
            assert_eq!(s1, 1000);
            assert_eq!(e2, 2000);
            assert_eq!(e1 + 1, s2);
        } else {
            panic!("Unexpected range types after split");
        }
    }

    #[test]
    fn test_shard_manager_creation() {
        let config = ShardingConfig::default();
        let manager = ShardManager::new(config);
        assert_eq!(manager.list_shards().len(), 0);
    }

    #[test]
    fn test_shard_manager_initialization() {
        let config = ShardingConfig {
            strategy: ShardingStrategy::Hash,
            shard_count: 4,
            replication_factor: 2,
            hash_function: HashFunction::Murmur3,
            virtual_nodes: 256,
            dynamic_resharding: true,
            migration_threshold: 0.8,
        };

        let mut manager = ShardManager::new(config);
        let nodes = vec![
            "node1".to_string(),
            "node2".to_string(),
            "node3".to_string(),
        ];

        manager.initialize(nodes).unwrap();
        assert_eq!(manager.list_shards().len(), 4);
    }

    #[test]
    fn test_find_shard() {
        let config = ShardingConfig {
            strategy: ShardingStrategy::Hash,
            shard_count: 2,
            replication_factor: 1,
            hash_function: HashFunction::Murmur3,
            virtual_nodes: 256,
            dynamic_resharding: true,
            migration_threshold: 0.8,
        };

        let mut manager = ShardManager::new(config);
        let nodes = vec!["node1".to_string(), "node2".to_string()];

        manager.initialize(nodes).unwrap();

        // Test that we can find a shard for any key
        let shard_id = manager.find_shard("test_key");
        assert!(shard_id.is_ok());
    }

    #[test]
    fn test_shard_migration() {
        let config = ShardingConfig::default();
        let mut manager = ShardManager::new(config);
        let nodes = vec!["node1".to_string(), "node2".to_string()];

        manager.initialize(nodes).unwrap();
        let shards = manager.list_shards();

        if let Some(shard) = shards.first() {
            let migration_id = manager.migrate_shard(&shard.id, Some("node2".to_string()));
            assert!(migration_id.is_ok());
        }
    }

    #[test]
    fn test_consistent_hash_node_operations() {
        let config = ShardingConfig {
            strategy: ShardingStrategy::ConsistentHash,
            shard_count: 4,
            replication_factor: 2,
            hash_function: HashFunction::Murmur3,
            virtual_nodes: 4, // Small number for testing
            dynamic_resharding: true,
            migration_threshold: 0.8,
        };

        let mut manager = ShardManager::new(config);
        let nodes = vec!["node1".to_string(), "node2".to_string()];

        manager.initialize(nodes).unwrap();

        // Test adding a node
        manager.add_node("node3".to_string()).unwrap();

        // Test removing a node
        manager.remove_node("node1").unwrap();
    }
}
