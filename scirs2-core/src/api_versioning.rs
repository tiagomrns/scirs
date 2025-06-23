//! API versioning system for backward compatibility
//!
//! This module provides version management for scirs2-core APIs,
//! ensuring smooth transitions between versions and maintaining
//! backward compatibility.

use std::fmt;
use std::sync::{Mutex, OnceLock};

/// Represents a semantic version number
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Version {
    /// Major version (breaking changes)
    pub major: u32,
    /// Minor version (new features, backward compatible)
    pub minor: u32,
    /// Patch version (bug fixes)
    pub patch: u32,
}

impl Version {
    /// Create a new version
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Current version of scirs2-core
    pub const CURRENT: Self = Self::new(0, 1, 0);

    /// Check if this version is compatible with another
    pub fn is_compatible_with(&self, other: &Version) -> bool {
        // Same major version and greater or equal minor/patch
        self.major == other.major
            && (self.minor > other.minor
                || (self.minor == other.minor && self.patch >= other.patch))
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Trait for versioned APIs
pub trait Versioned {
    /// Get the version this API was introduced in
    fn since_version() -> Version;

    /// Get the version this API was deprecated in (if any)
    fn deprecated_version() -> Option<Version> {
        None
    }

    /// Check if this API is available in a given version
    fn is_available_in(version: &Version) -> bool {
        version.is_compatible_with(&Self::since_version())
            && Self::deprecated_version()
                .map(|dep| version < &dep)
                .unwrap_or(true)
    }
}

/// Macro to mark APIs with version information
#[macro_export]
macro_rules! since_version {
    ($major:expr, $minor:expr, $patch:expr) => {
        fn since_version() -> $crate::api_versioning::Version {
            $crate::api_versioning::Version::new($major, $minor, $patch)
        }
    };
}

/// Macro to mark deprecated APIs
#[macro_export]
macro_rules! deprecated_in {
    ($major:expr, $minor:expr, $patch:expr) => {
        fn deprecated_version() -> Option<$crate::api_versioning::Version> {
            Some($crate::api_versioning::Version::new($major, $minor, $patch))
        }
    };
}

/// Version registry for tracking API changes
pub struct VersionRegistry {
    entries: Vec<ApiEntry>,
}

#[derive(Debug, Clone)]
pub struct ApiEntry {
    pub name: String,
    pub module: String,
    pub since: Version,
    pub deprecated: Option<Version>,
    pub replacement: Option<String>,
}

impl VersionRegistry {
    /// Create a new version registry
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Register a new API
    pub fn register_api(
        &mut self,
        name: impl Into<String>,
        module: impl Into<String>,
        since: Version,
    ) -> &mut Self {
        self.entries.push(ApiEntry {
            name: name.into(),
            module: module.into(),
            since,
            deprecated: None,
            replacement: None,
        });
        self
    }

    /// Mark an API as deprecated
    pub fn deprecate_api(
        &mut self,
        name: &str,
        version: Version,
        replacement: Option<String>,
    ) -> Result<(), String> {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.name == name) {
            entry.deprecated = Some(version);
            entry.replacement = replacement;
            Ok(())
        } else {
            Err(format!("API '{}' not found in registry", name))
        }
    }

    /// Get all APIs available in a specific version
    pub fn apis_in_version(&self, version: &Version) -> Vec<&ApiEntry> {
        self.entries
            .iter()
            .filter(|entry| {
                version.is_compatible_with(&entry.since)
                    && entry.deprecated.map(|dep| version < &dep).unwrap_or(true)
            })
            .collect()
    }

    /// Get all deprecated APIs in a version
    pub fn deprecated_apis(&self, version: &Version) -> Vec<&ApiEntry> {
        self.entries
            .iter()
            .filter(|entry| entry.deprecated.map(|dep| version >= &dep).unwrap_or(false))
            .collect()
    }

    /// Generate migration guide between versions
    pub fn migration_guide(&self, from: &Version, to: &Version) -> String {
        let mut guide = format!("Migration Guide: {} → {}\n\n", from, to);

        // Find removed APIs
        let removed: Vec<_> = self
            .entries
            .iter()
            .filter(|e| {
                from.is_compatible_with(&e.since)
                    && e.deprecated.map(|d| to >= &d && from < &d).unwrap_or(false)
            })
            .collect();

        if !removed.is_empty() {
            guide.push_str("## Removed APIs\n\n");
            for api in removed {
                guide.push_str(&format!("- **{}** ({})\n", api.name, api.module));
                if let Some(ref replacement) = api.replacement {
                    guide.push_str(&format!("  Replace with: {}\n", replacement));
                }
                guide.push('\n');
            }
        }

        // Find new APIs
        let new_apis: Vec<_> = self
            .entries
            .iter()
            .filter(|e| to.is_compatible_with(&e.since) && !from.is_compatible_with(&e.since))
            .collect();

        if !new_apis.is_empty() {
            guide.push_str("## New APIs\n\n");
            for api in new_apis {
                guide.push_str(&format!(
                    "- **{}** ({}) - Since {}\n",
                    api.name, api.module, api.since
                ));
            }
        }

        guide
    }
}

impl Default for VersionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Global version registry
static REGISTRY: OnceLock<Mutex<VersionRegistry>> = OnceLock::new();

/// Get the global registry for modification
pub fn global_registry_mut() -> std::sync::MutexGuard<'static, VersionRegistry> {
    REGISTRY
        .get_or_init(|| Mutex::new(VersionRegistry::new()))
        .lock()
        .unwrap()
}

/// Get the global registry for reading
pub fn global_registry() -> &'static Mutex<VersionRegistry> {
    REGISTRY.get_or_init(|| Mutex::new(VersionRegistry::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_compatibility() {
        let v1_0_0 = Version::new(1, 0, 0);
        let v1_1_0 = Version::new(1, 1, 0);
        let v1_0_1 = Version::new(1, 0, 1);
        let v2_0_0 = Version::new(2, 0, 0);

        assert!(v1_1_0.is_compatible_with(&v1_0_0));
        assert!(v1_0_1.is_compatible_with(&v1_0_0));
        assert!(!v2_0_0.is_compatible_with(&v1_0_0));
        assert!(!v1_0_0.is_compatible_with(&v1_1_0));
    }

    #[test]
    fn test_version_registry() {
        let mut registry = VersionRegistry::new();

        registry
            .register_api("Array", "core", Version::new(0, 1, 0))
            .register_api("Matrix", "linalg", Version::new(0, 1, 0))
            .register_api("OldArray", "core", Version::new(0, 1, 0));

        registry
            .deprecate_api("OldArray", Version::new(0, 2, 0), Some("Array".to_string()))
            .unwrap();

        let v0_1_0 = Version::new(0, 1, 0);
        let v0_2_0 = Version::new(0, 2, 0);

        // Check APIs in v0.1.0
        let apis_v1 = registry.apis_in_version(&v0_1_0);
        assert_eq!(apis_v1.len(), 3);

        // Check APIs in v0.2.0
        let apis_v2 = registry.apis_in_version(&v0_2_0);
        assert_eq!(apis_v2.len(), 2); // OldArray is deprecated

        // Check deprecated APIs
        let deprecated = registry.deprecated_apis(&v0_2_0);
        assert_eq!(deprecated.len(), 1);
        assert_eq!(deprecated[0].name, "OldArray");
    }

    #[test]
    fn test_migration_guide() {
        let mut registry = VersionRegistry::new();

        registry
            .register_api("Feature1", "module1", Version::new(0, 1, 0))
            .register_api("Feature2", "module2", Version::new(0, 2, 0))
            .register_api("OldFeature", "module1", Version::new(0, 1, 0));

        registry
            .deprecate_api(
                "OldFeature",
                Version::new(0, 2, 0),
                Some("Feature2".to_string()),
            )
            .unwrap();

        let guide = registry.migration_guide(&Version::new(0, 1, 0), &Version::new(0, 2, 0));

        assert!(guide.contains("Removed APIs"));
        assert!(guide.contains("OldFeature"));
        assert!(guide.contains("Replace with: Feature2"));
        assert!(guide.contains("New APIs"));
        assert!(guide.contains("Feature2"));
    }
}
