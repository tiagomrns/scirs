//! # Profiling
//!
//! This module provides utilities for profiling computational performance in scientific applications.
//!
//! ## Features
//!
//! * Function-level timing instrumentation
//! * Memory allocation tracking
//! * Hierarchical profiling for nested operations
//! * Easy-to-use macros for profiling sections of code
//!
//! ## Usage
//!
//! ```rust,no_run
//! use scirs2_core::profiling::{Profiler, Timer, MemoryTracker};
//!
//! // Start the global profiler
//! Profiler::global().lock().unwrap().start();
//!
//! // Time a function call
//! let result = Timer::time_function("matrix_multiplication", || {
//!     // Perform matrix multiplication
//!     // ...
//!     42 // Return some result
//! });
//!
//! // Time a code block with more control
//! let timer = Timer::start("data_processing");
//! // Perform data processing
//! // ...
//! timer.stop();
//!
//! // Track memory allocations
//! let tracker = MemoryTracker::start("large_array_operation");
//! let large_array = vec![0; 1_000_000];
//! // ...
//! tracker.stop();
//!
//! // Print profiling report
//! Profiler::global().lock().unwrap().print_report();
//!
//! // Stop profiling
//! Profiler::global().lock().unwrap().stop();
//! ```

use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Timer for measuring code execution time
pub struct Timer {
    /// Name of the operation being timed
    name: String,
    /// Start time
    start_time: Instant,
    /// Whether the timer is currently running
    running: bool,
    /// Whether to automatically report the timing when dropped
    auto_report: bool,
    /// Parent timer name for hierarchical profiling
    parent: Option<String>,
}

impl Timer {
    /// Start a new timer with the given name
    pub fn start(name: &str) -> Self {
        let timer = Self {
            name: name.to_string(),
            start_time: Instant::now(),
            running: true,
            auto_report: true,
            parent: None,
        };
        if let Ok(mut profiler) = Profiler::global().lock() {
            profiler.register_timer_start(&timer);
        }
        timer
    }

    /// Start a new hierarchical timer with a parent
    pub fn start_with_parent(name: &str, parent: &str) -> Self {
        let timer = Self {
            name: name.to_string(),
            start_time: Instant::now(),
            running: true,
            auto_report: true,
            parent: Some(parent.to_string()),
        };
        if let Ok(mut profiler) = Profiler::global().lock() {
            profiler.register_timer_start(&timer);
        }
        timer
    }

    /// Time a function call and return its result
    pub fn time_function<F, R>(name: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let timer = Self::start(name);
        let result = f();
        timer.stop();
        result
    }

    /// Time a function call with a parent timer and return its result
    pub fn time_function_with_parent<F, R>(name: &str, parent: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let timer = Self::start_with_parent(name, parent);
        let result = f();
        timer.stop();
        result
    }

    /// Stop the timer and record the elapsed time
    pub fn stop(&self) {
        if !self.running {
            return;
        }

        let elapsed = self.start_time.elapsed();
        if let Ok(mut profiler) = Profiler::global().lock() {
            profiler.register_timer_stop(&self.name, elapsed, self.parent.as_deref());
        }
    }

    /// Get the elapsed time without stopping the timer
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Disable auto-reporting when dropped
    pub fn without_auto_report(mut self) -> Self {
        self.auto_report = false;
        self
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        if self.running && self.auto_report {
            let elapsed = self.start_time.elapsed();
            if let Ok(mut profiler) = Profiler::global().lock() {
                profiler.register_timer_stop(&self.name, elapsed, self.parent.as_deref());
            }
        }
    }
}

/// Memory allocation tracker
pub struct MemoryTracker {
    /// Name of the operation being tracked
    name: String,
    /// Start memory usage
    start_memory: usize,
    /// Whether the tracker is currently running
    running: bool,
    /// Whether to automatically report when dropped
    auto_report: bool,
}

impl MemoryTracker {
    /// Start a new memory tracker with the given name
    pub fn start(name: &str) -> Self {
        let current_memory = Self::current_memory_usage();
        let tracker = Self {
            name: name.to_string(),
            start_memory: current_memory,
            running: true,
            auto_report: true,
        };
        if let Ok(mut profiler) = Profiler::global().lock() {
            profiler.register_memory_tracker_start(&tracker);
        }
        tracker
    }

    /// Stop the tracker and record the memory usage
    pub fn stop(&self) {
        if !self.running {
            return;
        }

        let current_memory = Self::current_memory_usage();
        let memory_delta = current_memory.saturating_sub(self.start_memory);
        if let Ok(mut profiler) = Profiler::global().lock() {
            profiler.register_memory_tracker_stop(&self.name, memory_delta);
        }
    }

    /// Track memory usage for a function call and return its result
    pub fn track_function<F, R>(name: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let tracker = Self::start(name);
        let result = f();
        tracker.stop();
        result
    }

    /// Get the current memory delta without stopping the tracker
    pub fn memory_delta(&self) -> isize {
        let current_memory = Self::current_memory_usage();
        current_memory as isize - self.start_memory as isize
    }

    /// Disable auto-reporting when dropped
    pub fn without_auto_report(mut self) -> Self {
        self.auto_report = false;
        self
    }

    /// Get the current memory usage (platform-dependent implementation)
    fn current_memory_usage() -> usize {
        // This is a simplified implementation that doesn't actually track real memory
        // A real implementation would use platform-specific APIs to get memory usage
        #[cfg(target_os = "linux")]
        {
            // On Linux, we would read /proc/self/statm
            0
        }

        #[cfg(target_os = "macos")]
        {
            // On macOS, we would use task_info
            0
        }

        #[cfg(target_os = "windows")]
        {
            // On Windows, we would use GetProcessMemoryInfo
            0
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        {
            // Fallback for other platforms
            0
        }
    }
}

impl Drop for MemoryTracker {
    fn drop(&mut self) {
        if self.running && self.auto_report {
            let current_memory = Self::current_memory_usage();
            let memory_delta = current_memory.saturating_sub(self.start_memory);
            if let Ok(mut profiler) = Profiler::global().lock() {
                profiler.register_memory_tracker_stop(&self.name, memory_delta);
            }
        }
    }
}

/// Timing entry for the profiler
#[derive(Debug, Clone)]
struct TimingEntry {
    /// Number of calls
    calls: usize,
    /// Total duration
    total_duration: Duration,
    /// Minimum duration
    min_duration: Duration,
    /// Maximum duration
    max_duration: Duration,
    /// Parent operation (used for hierarchical profiling structure)
    #[allow(dead_code)]
    parent: Option<String>,
    /// Child operations
    children: Vec<String>,
}

impl TimingEntry {
    /// Create a new timing entry
    fn new(duration: Duration, parent: Option<&str>) -> Self {
        Self {
            calls: 1,
            total_duration: duration,
            min_duration: duration,
            max_duration: duration,
            parent: parent.map(String::from),
            children: Vec::new(),
        }
    }

    /// Add a new timing measurement
    fn add_measurement(&mut self, duration: Duration) {
        self.calls += 1;
        self.total_duration += duration;
        self.min_duration = std::cmp::min(self.min_duration, duration);
        self.max_duration = std::cmp::max(self.max_duration, duration);
    }

    /// Add a child operation
    fn add_child(&mut self, child: &str) {
        if !self.children.contains(&child.to_string()) {
            self.children.push(child.to_string());
        }
    }

    /// Get the average duration
    fn average_duration(&self) -> Duration {
        if self.calls == 0 {
            Duration::from_secs(0)
        } else {
            self.total_duration / self.calls as u32
        }
    }
}

/// Memory tracking entry for the profiler
#[derive(Debug, Clone)]
struct MemoryEntry {
    /// Number of allocations
    allocations: usize,
    /// Total memory delta (can be negative for memory releases)
    total_delta: isize,
    /// Maximum memory delta in a single allocation
    max_delta: usize,
}

impl MemoryEntry {
    /// Create a new memory entry
    fn new(delta: usize) -> Self {
        Self {
            allocations: 1,
            total_delta: delta as isize,
            max_delta: delta,
        }
    }

    /// Add a new memory measurement
    fn add_measurement(&mut self, delta: usize) {
        self.allocations += 1;
        self.total_delta += delta as isize;
        self.max_delta = std::cmp::max(self.max_delta, delta);
    }

    /// Get the average memory delta
    #[allow(dead_code)]
    fn average_delta(&self) -> f64 {
        if self.allocations == 0 {
            0.0
        } else {
            self.total_delta as f64 / self.allocations as f64
        }
    }
}

/// Profiler for collecting performance metrics
pub struct Profiler {
    /// Timing measurements
    timings: HashMap<String, TimingEntry>,
    /// Memory measurements
    memory: HashMap<String, MemoryEntry>,
    /// Currently active timers
    active_timers: HashMap<String, Instant>,
    /// Whether the profiler is currently running
    running: bool,
}

impl Profiler {
    /// Create a new profiler
    pub fn new() -> Self {
        Self {
            timings: HashMap::new(),
            memory: HashMap::new(),
            active_timers: HashMap::new(),
            running: false,
        }
    }

    /// Get the global profiler instance
    pub fn global() -> &'static Mutex<Profiler> {
        static GLOBAL_PROFILER: Lazy<Mutex<Profiler>> = Lazy::new(|| Mutex::new(Profiler::new()));
        &GLOBAL_PROFILER
    }

    /// Start the profiler
    pub fn start(&mut self) {
        self.running = true;
        self.timings.clear();
        self.memory.clear();
        self.active_timers.clear();
    }

    /// Stop the profiler
    pub fn stop(&mut self) {
        self.running = false;
    }

    /// Reset the profiler
    pub fn reset(&mut self) {
        self.timings.clear();
        self.memory.clear();
        self.active_timers.clear();
    }

    /// Register the start of a timer
    pub fn register_timer_start(&mut self, timer: &Timer) {
        if !self.running {
            return;
        }

        self.active_timers
            .insert(timer.name.clone(), timer.start_time);

        // Register the parent-child relationship
        if let Some(parent) = &timer.parent {
            if let Some(entry) = self.timings.get_mut(parent) {
                entry.add_child(&timer.name);
            }
        }
    }

    /// Register the stop of a timer
    pub fn register_timer_stop(&mut self, name: &str, duration: Duration, parent: Option<&str>) {
        if !self.running {
            return;
        }

        // Remove from active timers
        self.active_timers.remove(name);

        // Update the timing entry
        match self.timings.get_mut(name) {
            Some(entry) => {
                entry.add_measurement(duration);
            }
            None => {
                let entry = TimingEntry::new(duration, parent);
                self.timings.insert(name.to_string(), entry);
            }
        }

        // Register the parent-child relationship
        if let Some(parent) = parent {
            if let Some(entry) = self.timings.get_mut(parent) {
                entry.add_child(name);
            }
        }
    }

    /// Register the start of a memory tracker
    pub fn register_memory_tracker_start(&mut self, _tracker: &MemoryTracker) {
        if !self.running {
            return;
        }

        // Nothing to do at start, just ensure the method exists for symmetry
    }

    /// Register the stop of a memory tracker
    pub fn register_memory_tracker_stop(&mut self, name: &str, delta: usize) {
        if !self.running {
            return;
        }

        // Update the memory entry
        match self.memory.get_mut(name) {
            Some(entry) => {
                entry.add_measurement(delta);
            }
            None => {
                let entry = MemoryEntry::new(delta);
                self.memory.insert(name.to_string(), entry);
            }
        }
    }

    /// Print a report of the profiling results
    pub fn print_report(&self) {
        if self.timings.is_empty() && self.memory.is_empty() {
            println!("No profiling data collected.");
            return;
        }

        if !self.timings.is_empty() {
            println!("\n=== Timing Report ===");
            println!(
                "{:<30} {:<10} {:<15} {:<15} {:<15}",
                "Operation", "Calls", "Total (ms)", "Average (ms)", "Max (ms)"
            );
            println!("{}", "-".repeat(90));

            // Sort by total duration
            let mut entries: Vec<(&String, &TimingEntry)> = self.timings.iter().collect();
            entries.sort_by(|a, b| b.1.total_duration.cmp(&a.1.total_duration));

            for (name, entry) in entries {
                println!(
                    "{:<30} {:<10} {:<15.2} {:<15.2} {:<15.2}",
                    name,
                    entry.calls,
                    entry.total_duration.as_secs_f64() * 1000.0,
                    entry.average_duration().as_secs_f64() * 1000.0,
                    entry.max_duration.as_secs_f64() * 1000.0
                );
            }
        }

        if !self.memory.is_empty() {
            println!("\n=== Memory Report ===");
            println!(
                "{:<30} {:<10} {:<15} {:<15}",
                "Operation", "Counts", "Total (KB)", "Max (KB)"
            );
            println!("{}", "-".repeat(75));

            // Sort by total memory delta
            let mut entries: Vec<(&String, &MemoryEntry)> = self.memory.iter().collect();
            entries.sort_by(|a, b| b.1.total_delta.abs().cmp(&a.1.total_delta.abs()));

            for (name, entry) in entries {
                println!(
                    "{:<30} {:<10} {:<15.2} {:<15.2}",
                    name,
                    entry.allocations,
                    entry.total_delta as f64 / 1024.0,
                    entry.max_delta as f64 / 1024.0
                );
            }
        }
    }

    /// Get a report of the profiling results as a string
    pub fn get_report(&self) -> String {
        use std::fmt::Write;
        let mut report = String::new();

        if self.timings.is_empty() && self.memory.is_empty() {
            writeln!(report, "No profiling data collected.").unwrap();
            return report;
        }

        if !self.timings.is_empty() {
            writeln!(report, "\n=== Timing Report ===").unwrap();
            writeln!(
                report,
                "{:<30} {:<10} {:<15} {:<15} {:<15}",
                "Operation", "Calls", "Total (ms)", "Average (ms)", "Max (ms)"
            )
            .unwrap();
            writeln!(report, "{}", "-".repeat(90)).unwrap();

            // Sort by total duration
            let mut entries: Vec<(&String, &TimingEntry)> = self.timings.iter().collect();
            entries.sort_by(|a, b| b.1.total_duration.cmp(&a.1.total_duration));

            for (name, entry) in entries {
                writeln!(
                    report,
                    "{:<30} {:<10} {:<15.2} {:<15.2} {:<15.2}",
                    name,
                    entry.calls,
                    entry.total_duration.as_secs_f64() * 1000.0,
                    entry.average_duration().as_secs_f64() * 1000.0,
                    entry.max_duration.as_secs_f64() * 1000.0
                )
                .unwrap();
            }
        }

        if !self.memory.is_empty() {
            writeln!(report, "\n=== Memory Report ===").unwrap();
            writeln!(
                report,
                "{:<30} {:<10} {:<15} {:<15}",
                "Operation", "Counts", "Total (KB)", "Max (KB)"
            )
            .unwrap();
            writeln!(report, "{}", "-".repeat(75)).unwrap();

            // Sort by total memory delta
            let mut entries: Vec<(&String, &MemoryEntry)> = self.memory.iter().collect();
            entries.sort_by(|a, b| b.1.total_delta.abs().cmp(&a.1.total_delta.abs()));

            for (name, entry) in entries {
                writeln!(
                    report,
                    "{:<30} {:<10} {:<15.2} {:<15.2}",
                    name,
                    entry.allocations,
                    entry.total_delta as f64 / 1024.0,
                    entry.max_delta as f64 / 1024.0
                )
                .unwrap();
            }
        }

        report
    }

    /// Get timing statistics for a specific operation
    pub fn get_timing_stats(&self, name: &str) -> Option<(usize, Duration, Duration, Duration)> {
        self.timings.get(name).map(|entry| {
            (
                entry.calls,
                entry.total_duration,
                entry.average_duration(),
                entry.max_duration,
            )
        })
    }

    /// Get memory statistics for a specific operation
    pub fn get_memory_stats(&self, name: &str) -> Option<(usize, isize, usize)> {
        self.memory
            .get(name)
            .map(|entry| (entry.allocations, entry.total_delta, entry.max_delta))
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Access a memory tracker from the profiling module to avoid name conflicts
pub fn profiling_memory_tracker() -> &'static MemoryTracker {
    // Create a dummy memory tracker for static access
    static MEMORY_TRACKER: once_cell::sync::Lazy<MemoryTracker> =
        once_cell::sync::Lazy::new(|| MemoryTracker {
            name: "global".to_string(),
            start_memory: 0,
            running: false,
            auto_report: false,
        });
    &MEMORY_TRACKER
}

/// Macro for timing a block of code
#[macro_export]
macro_rules! profile_time {
    ($name:expr, $body:block) => {{
        let timer = $crate::profiling::Timer::start($name);
        let result = $body;
        timer.stop();
        result
    }};
}

/// Macro for tracking memory usage in a block of code
#[macro_export]
macro_rules! profile_memory {
    ($name:expr, $body:block) => {{
        let tracker = $crate::profiling::MemoryTracker::start($name);
        let result = $body;
        tracker.stop();
        result
    }};
}

/// Macro for timing a block of code with a parent operation
#[macro_export]
macro_rules! profile_time_with_parent {
    ($name:expr, $parent:expr, $body:block) => {{
        let timer = $crate::profiling::Timer::start_with_parent($name, $parent);
        let result = $body;
        timer.stop();
        result
    }};
}
