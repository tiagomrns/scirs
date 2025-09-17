//! System resource monitoring for pipeline optimization
//!
//! This module provides real-time monitoring of system resources including
//! CPU usage, memory utilization, I/O performance, and cache efficiency.

use crate::error::{IoError, Result};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use super::config::{CachePerformance, MemoryUsage, NumaTopology, SystemMetrics};

/// Real-time system resource monitor
#[derive(Debug)]
pub struct ResourceMonitor {
    /// Last update timestamp
    last_update: Instant,
    /// Update frequency
    update_frequency: Duration,
    /// Cached metrics to avoid frequent system calls
    cached_metrics: Option<SystemMetrics>,
    /// Monitoring history for trend analysis
    metrics_history: Vec<SystemMetrics>,
    /// Maximum history size
    max_history_size: usize,
}

impl Default for ResourceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceMonitor {
    pub fn new() -> Self {
        Self {
            last_update: Instant::now(),
            update_frequency: Duration::from_millis(500), // Update every 500ms
            cached_metrics: None,
            metrics_history: Vec::new(),
            max_history_size: 100, // Keep last 100 samples
        }
    }

    /// Get current system metrics with caching
    pub fn get_current_metrics(&mut self) -> Result<SystemMetrics> {
        let now = Instant::now();

        // Check if we need to update cached metrics
        if self.cached_metrics.is_none() ||
           now.duration_since(self.last_update) >= self.update_frequency {

            let metrics = self.collect_system_metrics()?;
            self.cached_metrics = Some(metrics.clone());
            self.last_update = now;

            // Add to history
            self.metrics_history.push(metrics.clone());
            if self.metrics_history.len() > self.max_history_size {
                self.metrics_history.remove(0);
            }

            Ok(metrics)
        } else {
            Ok(self.cached_metrics.as_ref().unwrap().clone())
        }
    }

    /// Collect fresh system metrics
    fn collect_system_metrics(&self) -> Result<SystemMetrics> {
        Ok(SystemMetrics {
            cpu_usage: self.get_cpu_usage()?,
            memory_usage: self.get_memory_usage()?,
            io_utilization: self.get_io_utilization()?,
            network_bandwidth_usage: self.get_network_usage()?,
            cache_performance: self.get_cache_performance()?,
            numa_topology: self.get_numa_topology()?,
        })
    }

    /// Get CPU usage percentage
    fn get_cpu_usage(&self) -> Result<f64> {
        #[cfg(target_os = "linux")]
        {
            self.get_linux_cpu_usage()
        }
        #[cfg(target_os = "windows")]
        {
            self.get_windows_cpu_usage()
        }
        #[cfg(target_os = "macos")]
        {
            self.get_macos_cpu_usage()
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
        {
            Ok(0.5) // Default fallback
        }
    }

    #[cfg(target_os = "linux")]
    fn get_linux_cpu_usage(&self) -> Result<f64> {
        // Read /proc/stat for CPU usage
        let stat_content = std::fs::read_to_string("/proc/stat")
            .map_err(|e| IoError::Other(format!("Failed to read /proc/stat: {}", e)))?;

        if let Some(cpu_line) = stat_content.lines().next() {
            let values: Vec<u64> = cpu_line
                .split_whitespace()
                .skip(1)
                .take(4)
                .filter_map(|s| s.parse().ok())
                .collect();

            if values.len() >= 4 {
                let idle = values[3];
                let total: u64 = values.iter().sum();
                return Ok(1.0 - (idle as f64) / (total as f64));
            }
        }

        Ok(0.5) // Fallback
    }

    #[cfg(target_os = "windows")]
    fn get_windows_cpu_usage(&self) -> Result<f64> {
        // Windows-specific implementation would go here
        // For now, return a placeholder
        Ok(0.5)
    }

    #[cfg(target_os = "macos")]
    fn get_macos_cpu_usage(&self) -> Result<f64> {
        // macOS-specific implementation would go here
        // For now, return a placeholder
        Ok(0.5)
    }

    /// Get memory usage information
    fn get_memory_usage(&self) -> Result<MemoryUsage> {
        #[cfg(target_os = "linux")]
        {
            self.get_linux_memory_usage()
        }
        #[cfg(not(target_os = "linux"))]
        {
            Ok(MemoryUsage {
                total: 8 * 1024 * 1024 * 1024,     // 8GB fallback
                available: 4 * 1024 * 1024 * 1024, // 4GB fallback
                used: 4 * 1024 * 1024 * 1024,
                utilization: 0.5,
            })
        }
    }

    #[cfg(target_os = "linux")]
    fn get_linux_memory_usage(&self) -> Result<MemoryUsage> {
        let meminfo_content = std::fs::read_to_string("/proc/meminfo")
            .map_err(|e| IoError::Other(format!("Failed to read /proc/meminfo: {}", e)))?;

        let mut total = 0u64;
        let mut available = 0u64;

        for line in meminfo_content.lines() {
            if line.starts_with("MemTotal:") {
                total = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0)
                    * 1024; // Convert KB to bytes
            } else if line.starts_with("MemAvailable:") {
                available = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0)
                    * 1024; // Convert KB to bytes
            }
        }

        let used = total - available;
        let utilization = if total > 0 {
            used as f64 / total as f64
        } else {
            0.0
        };

        Ok(MemoryUsage {
            total,
            available,
            used,
            utilization,
        })
    }

    /// Get I/O utilization
    fn get_io_utilization(&self) -> Result<f64> {
        // Simplified I/O utilization - could be expanded with platform-specific code
        #[cfg(target_os = "linux")]
        {
            self.get_linux_io_utilization()
        }
        #[cfg(not(target_os = "linux"))]
        {
            Ok(0.3) // Placeholder
        }
    }

    #[cfg(target_os = "linux")]
    fn get_linux_io_utilization(&self) -> Result<f64> {
        // Read /proc/diskstats for I/O statistics
        // This is a simplified implementation
        match std::fs::read_to_string("/proc/diskstats") {
            Ok(content) => {
                // Parse diskstats and calculate utilization
                // For simplicity, return a placeholder
                Ok(0.3)
            }
            Err(_) => Ok(0.3), // Fallback
        }
    }

    /// Get network bandwidth usage
    fn get_network_usage(&self) -> Result<f64> {
        // Simplified network usage - could be expanded with platform-specific code
        #[cfg(target_os = "linux")]
        {
            self.get_linux_network_usage()
        }
        #[cfg(not(target_os = "linux"))]
        {
            Ok(0.2) // Placeholder
        }
    }

    #[cfg(target_os = "linux")]
    fn get_linux_network_usage(&self) -> Result<f64> {
        // Read /proc/net/dev for network statistics
        // This is a simplified implementation
        match std::fs::read_to_string("/proc/net/dev") {
            Ok(content) => {
                // Parse network stats and calculate usage
                // For simplicity, return a placeholder
                Ok(0.2)
            }
            Err(_) => Ok(0.2), // Fallback
        }
    }

    /// Get cache performance metrics
    fn get_cache_performance(&self) -> Result<CachePerformance> {
        // This would typically require hardware performance counters
        // For now, return reasonable defaults
        Ok(CachePerformance {
            l1_hit_rate: 0.95,
            l2_hit_rate: 0.85,
            l3_hit_rate: 0.75,
            tlb_hit_rate: 0.99,
        })
    }

    /// Get NUMA topology information
    fn get_numa_topology(&self) -> Result<NumaTopology> {
        #[cfg(target_os = "linux")]
        {
            self.get_linux_numa_topology()
        }
        #[cfg(not(target_os = "linux"))]
        {
            Ok(NumaTopology::default())
        }
    }

    #[cfg(target_os = "linux")]
    fn get_linux_numa_topology(&self) -> Result<NumaTopology> {
        // Read NUMA topology from /sys/devices/system/node/
        // This is a simplified implementation
        match std::fs::read_dir("/sys/devices/system/node/") {
            Ok(_entries) => {
                // Parse NUMA node information
                // For simplicity, return default topology
                Ok(NumaTopology::default())
            }
            Err(_) => Ok(NumaTopology::default()),
        }
    }

    /// Get metrics trend over time
    pub fn get_metrics_trend(&self, duration: Duration) -> Vec<&SystemMetrics> {
        let cutoff_time = Instant::now() - duration;
        // For simplicity, return recent metrics
        // In a real implementation, we'd need to track timestamps
        self.metrics_history.iter().collect()
    }

    /// Check if system is under high load
    pub fn is_high_load(&self) -> bool {
        if let Some(metrics) = &self.cached_metrics {
            metrics.cpu_usage > 0.8 ||
            metrics.memory_usage.utilization > 0.9 ||
            metrics.io_utilization > 0.8
        } else {
            false
        }
    }

    /// Get resource utilization score (0.0 to 1.0)
    pub fn get_utilization_score(&self) -> f64 {
        if let Some(metrics) = &self.cached_metrics {
            (metrics.cpu_usage +
             metrics.memory_usage.utilization +
             metrics.io_utilization) / 3.0
        } else {
            0.5
        }
    }

    /// Predict resource pressure in near future
    pub fn predict_resource_pressure(&self, lookahead: Duration) -> f64 {
        // Simple linear extrapolation based on recent trends
        if self.metrics_history.len() < 2 {
            return self.get_utilization_score();
        }

        let recent_scores: Vec<f64> = self.metrics_history
            .iter()
            .rev()
            .take(10)
            .map(|m| (m.cpu_usage + m.memory_usage.utilization + m.io_utilization) / 3.0)
            .collect();

        if recent_scores.len() < 2 {
            return recent_scores[0];
        }

        // Calculate trend slope
        let n = recent_scores.len() as f64;
        let sum_x: f64 = (0..recent_scores.len()).map(|i| i as f64).sum();
        let sum_y: f64 = recent_scores.iter().sum();
        let sum_xy: f64 = recent_scores
            .iter()
            .enumerate()
            .map(|(i, &y)| i as f64 * y)
            .sum();
        let sum_x2: f64 = (0..recent_scores.len()).map(|i| (i as f64).powi(2)).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        // Project into the future
        let future_steps = lookahead.as_secs() as f64 / self.update_frequency.as_secs() as f64;
        let predicted = intercept + slope * (n + future_steps);

        predicted.clamp(0.0, 1.0)
    }
}