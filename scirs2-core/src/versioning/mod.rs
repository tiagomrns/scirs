//! # API Versioning Infrastructure
//!
//! Production-grade API versioning system for `SciRS2` Core providing semantic
//! versioning, backward compatibility guarantees, and version negotiation
//! capabilities for enterprise deployments and long-term API stability.
//!
//! ## Features
//!
//! - Semantic versioning (`SemVer`) compliance with custom extensions
//! - Backward compatibility enforcement and validation
//! - API version negotiation and client-server compatibility
//! - Breaking change detection and migration assistance
//! - Version deprecation management with transition periods
//! - API evolution tracking and documentation generation
//! - Enterprise-grade stability guarantees
//! - Integration with CI/CD pipelines for automated compatibility testing
//!
//! ## Modules
//!
//! - `semantic`: Semantic versioning implementation with `SciRS2` extensions
//! - `compatibility`: Backward compatibility checking and enforcement
//! - `negotiation`: Version negotiation between clients and servers
//! - `migration`: Migration assistance for API upgrades
//! - `deprecation`: Deprecation management and transition planning
//!
//! ## Example
//!
//! ```rust
//! use scirs2_core::versioning::{Version, VersionManager, CompatibilityLevel, ApiVersionBuilder, ClientCapabilities};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create version manager
//! let mut version_manager = VersionManager::new();
//!
//! // Register API versions using the builder pattern
//! let v1_0_0 = ApiVersionBuilder::new(Version::parse("1.0.0")?)
//!     .new_feature("Initial API release")
//!     .build()?;
//! let v1_1_0 = ApiVersionBuilder::new(Version::parse("1.1.0")?)
//!     .new_feature("Added new computation methods")
//!     .build()?;
//! let v2_0_0 = ApiVersionBuilder::new(Version::parse("2.0.0")?)
//!     .breaking_change("Changed function signatures")
//!     .build()?;
//!
//! version_manager.registerversion(v1_0_0.clone())?;
//! version_manager.registerversion(v1_1_0.clone())?;
//! version_manager.registerversion(v2_0_0.clone())?;
//!
//! // Check compatibility
//! let compat = version_manager.check_compatibility(&v1_0_0.version, &v1_1_0.version)?;
//! assert_eq!(compat, CompatibilityLevel::BackwardCompatible);
//!
//! // Negotiate version with client capabilities
//! let client_caps = ClientCapabilities::new("test_client".to_string(), Version::parse("1.0.5")?);
//! let negotiated = version_manager.negotiateversion(&client_caps)?;
//! assert!(negotiated.negotiated_version.major() >= 1);
//! # Ok(())
//! # }
//! ```

pub mod compatibility;
pub mod deprecation;
pub mod migration;
pub mod negotiation;
pub mod semantic;

use crate::error::CoreError;
use std::collections::{BTreeSet, HashMap};

use serde::{Deserialize, Serialize};

// Re-export main types
pub use compatibility::{CompatibilityChecker, CompatibilityLevel, CompatibilityReport};
pub use deprecation::{DeprecationManager, DeprecationPolicy, DeprecationStatus};
pub use migration::{MigrationManager, MigrationPlan, MigrationStep};
pub use negotiation::{ClientCapabilities, NegotiationResult, VersionNegotiator};
pub use semantic::{Version, VersionBuilder, VersionConstraint, VersionRange};

/// API version information with metadata
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApiVersion {
    /// The semantic version
    pub version: Version,
    /// Release date
    pub release_date: chrono::DateTime<chrono::Utc>,
    /// Stability level
    pub stability: StabilityLevel,
    /// Support status
    pub support_status: SupportStatus,
    /// End of life date (if applicable)
    pub end_of_life: Option<chrono::DateTime<chrono::Utc>>,
    /// Feature flags supported in this version
    pub features: BTreeSet<String>,
    /// Breaking changes from previous version
    pub breakingchanges: Vec<String>,
    /// New features in this version
    pub new_features: Vec<String>,
    /// Bug fixes in this version
    pub bug_fixes: Vec<String>,
    /// Deprecated features in this version
    pub deprecated_features: Vec<String>,
    /// Minimum client version required
    pub min_clientversion: Option<Version>,
    /// Maximum client version supported
    pub max_clientversion: Option<Version>,
}

/// API stability levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum StabilityLevel {
    /// Experimental - subject to breaking changes
    Experimental,
    /// Alpha - feature complete but may have breaking changes
    Alpha,
    /// Beta - stable API but may have minor breaking changes
    Beta,
    /// Stable - backward compatible changes only
    Stable,
    /// Mature - minimal changes, long-term support
    Mature,
    /// Legacy - deprecated but still supported
    Legacy,
}

impl StabilityLevel {
    /// Get the string representation
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Experimental => "experimental",
            Self::Alpha => "alpha",
            Self::Beta => "beta",
            Self::Stable => "stable",
            Self::Mature => "mature",
            Self::Legacy => "legacy",
        }
    }
}

/// Support status for API versions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SupportStatus {
    /// Active development and support
    Active,
    /// Maintenance mode - bug fixes only
    Maintenance,
    /// Deprecated - migration encouraged
    Deprecated,
    /// End of life - no longer supported
    EndOfLife,
    /// Security updates only
    SecurityOnly,
}

impl SupportStatus {
    /// Get the string representation
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Maintenance => "maintenance",
            Self::Deprecated => "deprecated",
            Self::EndOfLife => "end_of_life",
            Self::SecurityOnly => "security_only",
        }
    }
}

/// Version manager for coordinating all versioning operations
pub struct VersionManager {
    /// Registered API versions
    versions: HashMap<Version, ApiVersion>,
    /// Current active version
    currentversion: Option<Version>,
    /// Compatibility checker
    compatibility_checker: CompatibilityChecker,
    /// Version negotiator
    negotiator: VersionNegotiator,
    /// Migration manager
    migration_manager: MigrationManager,
    /// Deprecation manager
    deprecation_manager: DeprecationManager,
}

impl VersionManager {
    /// Create a new version manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            versions: HashMap::new(),
            currentversion: None,
            compatibility_checker: CompatibilityChecker::new(),
            negotiator: VersionNegotiator::new(),
            migration_manager: MigrationManager::new(),
            deprecation_manager: DeprecationManager::new(),
        }
    }

    /// Register an API version
    ///
    /// # Errors
    ///
    /// Returns an error if the version is already registered.
    pub fn registerversion(&mut self, apiversion: ApiVersion) -> Result<(), CoreError> {
        let version = apiversion.version.clone();

        // Validate version is not already registered
        if self.versions.contains_key(&version) {
            return Err(CoreError::ComputationError(
                crate::error::ErrorContext::new(format!("Version {version} is already registered")),
            ));
        }

        // Register with compatibility checker
        self.compatibility_checker.register_version(&apiversion)?;

        // Register with migration manager
        self.migration_manager.register_version(&apiversion)?;

        // Register with deprecation manager
        self.deprecation_manager.register_version(&apiversion)?;

        self.versions.insert(version, apiversion);
        Ok(())
    }

    /// Set the current active version
    ///
    /// # Errors
    ///
    /// Returns an error if the version is not registered.
    pub fn set_currentversion(&mut self, version: Version) -> Result<(), CoreError> {
        if !self.versions.contains_key(&version) {
            return Err(CoreError::ComputationError(
                crate::error::ErrorContext::new(format!("Version {version} is not registered")),
            ));
        }

        self.currentversion = Some(version);
        Ok(())
    }

    /// Get the current active version
    #[must_use]
    pub fn currentversion(&self) -> Option<&Version> {
        self.currentversion.as_ref()
    }

    /// Get all registered versions
    #[must_use]
    pub fn getversions(&self) -> Vec<&ApiVersion> {
        let mut versions: Vec<_> = self.versions.values().collect();
        versions.sort_by(|a, b| a.version.cmp(&b.version));
        versions
    }

    /// Get supported versions (active and maintenance)
    #[must_use]
    pub fn get_supportedversions(&self) -> Vec<&ApiVersion> {
        self.versions
            .values()
            .filter(|v| {
                matches!(
                    v.support_status,
                    SupportStatus::Active | SupportStatus::Maintenance
                )
            })
            .collect()
    }

    /// Get version by version number
    #[must_use]
    pub fn getversion(&self, version: &Version) -> Option<&ApiVersion> {
        self.versions.get(version)
    }

    /// Check compatibility between two versions
    ///
    /// # Errors
    ///
    /// Returns an error if compatibility checking fails.
    pub fn check_compatibility(
        &self,
        fromversion: &Version,
        toversion: &Version,
    ) -> Result<CompatibilityLevel, CoreError> {
        self.compatibility_checker
            .check_compatibility(fromversion, toversion)
    }

    /// Get detailed compatibility report
    ///
    /// # Errors
    ///
    /// Returns an error if compatibility report generation fails.
    pub fn get_compatibility_report(
        &self,
        fromversion: &Version,
        toversion: &Version,
    ) -> Result<CompatibilityReport, CoreError> {
        self.compatibility_checker
            .get_compatibility_report(fromversion, toversion)
    }

    /// Negotiate version with client
    ///
    /// # Errors
    ///
    /// Returns an error if version negotiation fails.
    pub fn negotiateversion(
        &self,
        client_capabilities: &ClientCapabilities,
    ) -> Result<NegotiationResult, CoreError> {
        let supportedversions: Vec<_> = self
            .get_supportedversions()
            .into_iter()
            .map(|v| &v.version)
            .collect();

        self.negotiator
            .negotiate(client_capabilities, &supportedversions)
    }

    /// Get migration plan between versions
    ///
    /// # Errors
    ///
    /// Returns an error if migration plan generation fails.
    pub fn get_migration_plan(
        &self,
        fromversion: &Version,
        toversion: &Version,
    ) -> Result<MigrationPlan, CoreError> {
        self.migration_manager
            .create_migration_plan(fromversion, toversion)
    }

    /// Check if a version is deprecated
    #[must_use]
    pub fn isversion_deprecated(&self, version: &Version) -> bool {
        if let Some(apiversion) = self.versions.get(version) {
            matches!(
                apiversion.support_status,
                SupportStatus::Deprecated | SupportStatus::EndOfLife
            )
        } else {
            false
        }
    }

    /// Get deprecation information for a version
    #[must_use]
    pub fn get_deprecation_status(&self, version: &Version) -> Option<DeprecationStatus> {
        self.deprecation_manager.get_deprecation_status(version)
    }

    /// Update deprecation status
    ///
    /// # Errors
    ///
    /// Returns an error if the deprecation status update fails.
    pub fn update_deprecation_status(
        &mut self,
        version: &Version,
        status: DeprecationStatus,
    ) -> Result<(), CoreError> {
        self.deprecation_manager.update_status(version, status)
    }

    /// Get the latest version in a major version line
    #[must_use]
    pub fn get_latest_in_major(&self, major: u64) -> Option<&ApiVersion> {
        self.versions
            .values()
            .filter(|v| v.version.major() == major)
            .max_by(|a, b| a.version.cmp(&b.version))
    }

    /// Get the latest stable version
    #[must_use]
    pub fn get_latest_stable(&self) -> Option<&ApiVersion> {
        self.versions
            .values()
            .filter(|v| {
                v.stability == StabilityLevel::Stable || v.stability == StabilityLevel::Mature
            })
            .filter(|v| v.support_status == SupportStatus::Active)
            .max_by(|a, b| a.version.cmp(&b.version))
    }

    /// Check if an upgrade path exists
    #[must_use]
    pub fn has_upgrade_path(&self, fromversion: &Version, toversion: &Version) -> bool {
        self.migration_manager
            .has_migration_path(fromversion, toversion)
    }

    /// Validate version constraints
    ///
    /// # Errors
    ///
    /// Returns an error if validation fails.
    pub fn validate_constraint(
        &self,
        constraint: &VersionConstraint,
    ) -> Result<Vec<&Version>, CoreError> {
        let matchingversions: Vec<_> = self
            .versions
            .keys()
            .filter(|v| constraint.matches(v))
            .collect();

        Ok(matchingversions)
    }

    /// Get version statistics
    #[must_use]
    pub fn getversion_statistics(&self) -> VersionStatistics {
        let mut stats = VersionStatistics::default();

        for apiversion in self.versions.values() {
            stats.totalversions += 1;

            match apiversion.stability {
                StabilityLevel::Experimental => stats.experimentalversions += 1,
                StabilityLevel::Alpha => stats.alphaversions += 1,
                StabilityLevel::Beta => stats.betaversions += 1,
                StabilityLevel::Stable => stats.stableversions += 1,
                StabilityLevel::Mature => stats.matureversions += 1,
                StabilityLevel::Legacy => stats.legacyversions += 1,
            }

            match apiversion.support_status {
                SupportStatus::Active => stats.activeversions += 1,
                SupportStatus::Maintenance => stats.maintenanceversions += 1,
                SupportStatus::Deprecated => stats.deprecatedversions += 1,
                SupportStatus::EndOfLife => stats.end_of_lifeversions += 1,
                SupportStatus::SecurityOnly => stats.security_onlyversions += 1,
            }
        }

        stats
    }

    /// Perform version maintenance tasks
    ///
    /// # Errors
    ///
    /// Returns an error if maintenance tasks fail.
    pub fn perform_maintenance(&mut self) -> Result<MaintenanceReport, CoreError> {
        let mut report = MaintenanceReport::default();
        let now = chrono::Utc::now();

        // Check for expired versions
        for (version, apiversion) in &mut self.versions {
            if let Some(eol_date) = apiversion.end_of_life {
                if now > eol_date && apiversion.support_status != SupportStatus::EndOfLife {
                    apiversion.support_status = SupportStatus::EndOfLife;
                    report.versions_marked_eol.push(version.clone());
                }
            }
        }

        // Update deprecation statuses
        let deprecation_updates = self.deprecation_manager.perform_maintenance()?;
        report.deprecation_updates = deprecation_updates.len();

        // Clean up old migration plans
        let migration_cleanup = self.migration_manager.cleanup_old_plans()?;
        report.migration_plans_cleaned = migration_cleanup;

        Ok(report)
    }
}

impl Default for VersionManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Version statistics for monitoring and reporting
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VersionStatistics {
    /// Total number of registered versions
    pub totalversions: usize,
    /// Experimental versions
    pub experimentalversions: usize,
    /// Alpha versions
    pub alphaversions: usize,
    /// Beta versions
    pub betaversions: usize,
    /// Stable versions
    pub stableversions: usize,
    /// Mature versions
    pub matureversions: usize,
    /// Legacy versions
    pub legacyversions: usize,
    /// Active versions
    pub activeversions: usize,
    /// Maintenance versions
    pub maintenanceversions: usize,
    /// Deprecated versions
    pub deprecatedversions: usize,
    /// End of life versions
    pub end_of_lifeversions: usize,
    /// Security only versions
    pub security_onlyversions: usize,
}

/// Maintenance report for version management operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MaintenanceReport {
    /// Versions marked as end of life
    pub versions_marked_eol: Vec<Version>,
    /// Number of deprecation status updates
    pub deprecation_updates: usize,
    /// Number of migration plans cleaned up
    pub migration_plans_cleaned: usize,
}

/// Builder for creating API versions
pub struct ApiVersionBuilder {
    version: Option<Version>,
    release_date: chrono::DateTime<chrono::Utc>,
    stability: StabilityLevel,
    support_status: SupportStatus,
    end_of_life: Option<chrono::DateTime<chrono::Utc>>,
    features: BTreeSet<String>,
    breakingchanges: Vec<String>,
    new_features: Vec<String>,
    bug_fixes: Vec<String>,
    deprecated_features: Vec<String>,
    min_clientversion: Option<Version>,
    max_clientversion: Option<Version>,
}

impl ApiVersionBuilder {
    /// Create a new API version builder
    #[must_use]
    pub fn new(version: Version) -> Self {
        Self {
            version: Some(version),
            release_date: chrono::Utc::now(),
            stability: StabilityLevel::Stable,
            support_status: SupportStatus::Active,
            end_of_life: None,
            features: BTreeSet::new(),
            breakingchanges: Vec::new(),
            new_features: Vec::new(),
            bug_fixes: Vec::new(),
            deprecated_features: Vec::new(),
            min_clientversion: None,
            max_clientversion: None,
        }
    }

    /// Set release date
    #[must_use]
    pub fn release_date(mut self, date: chrono::DateTime<chrono::Utc>) -> Self {
        self.release_date = date;
        self
    }

    /// Set stability level
    #[must_use]
    pub fn stability(mut self, stability: StabilityLevel) -> Self {
        self.stability = stability;
        self
    }

    /// Set support status
    #[must_use]
    pub fn support_status(mut self, status: SupportStatus) -> Self {
        self.support_status = status;
        self
    }

    /// Set end of life date
    #[must_use]
    pub fn end_of_life(mut self, date: chrono::DateTime<chrono::Utc>) -> Self {
        self.end_of_life = Some(date);
        self
    }

    /// Add a feature
    #[must_use]
    pub fn feature(mut self, feature: &str) -> Self {
        self.features.insert(feature.to_string());
        self
    }

    /// Add a breaking change
    #[must_use]
    pub fn breaking_change(mut self, change: &str) -> Self {
        self.breakingchanges.push(change.to_string());
        self
    }

    /// Add a new feature
    #[must_use]
    pub fn new_feature(mut self, feature: &str) -> Self {
        self.new_features.push(feature.to_string());
        self
    }

    /// Add a bug fix
    #[must_use]
    pub fn bug_fix(mut self, fix: &str) -> Self {
        self.bug_fixes.push(fix.to_string());
        self
    }

    /// Add a deprecated feature
    #[must_use]
    pub fn deprecated_feature(mut self, feature: &str) -> Self {
        self.deprecated_features.push(feature.to_string());
        self
    }

    /// Set minimum client version
    #[must_use]
    pub fn min_clientversion(mut self, version: Version) -> Self {
        self.min_clientversion = Some(version);
        self
    }

    /// Set maximum client version
    #[must_use]
    pub fn max_clientversion(mut self, version: Version) -> Self {
        self.max_clientversion = Some(version);
        self
    }

    /// Build the API version
    ///
    /// # Errors
    ///
    /// Returns an error if the version is not set.
    pub fn build(self) -> Result<ApiVersion, CoreError> {
        let version = self.version.ok_or_else(|| {
            CoreError::ComputationError(crate::error::ErrorContext::new(
                "Version is required".to_string(),
            ))
        })?;

        Ok(ApiVersion {
            version,
            release_date: self.release_date,
            stability: self.stability,
            support_status: self.support_status,
            end_of_life: self.end_of_life,
            features: self.features,
            breakingchanges: self.breakingchanges,
            new_features: self.new_features,
            bug_fixes: self.bug_fixes,
            deprecated_features: self.deprecated_features,
            min_clientversion: self.min_clientversion,
            max_clientversion: self.max_clientversion,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn testversion_manager_creation() {
        let manager = VersionManager::new();
        assert!(manager.currentversion().is_none());
        assert_eq!(manager.getversions().len(), 0);
    }

    #[test]
    fn test_apiversion_builder() {
        let version = Version::parse("1.0.0").unwrap();
        let apiversion = ApiVersionBuilder::new(version)
            .stability(StabilityLevel::Stable)
            .feature("feature1")
            .new_feature("New awesome feature")
            .build()
            .unwrap();

        assert_eq!(apiversion.version.to_string(), "1.0.0");
        assert_eq!(apiversion.stability, StabilityLevel::Stable);
        assert!(apiversion.features.contains("feature1"));
        assert_eq!(apiversion.new_features.len(), 1);
    }

    #[test]
    fn testversion_registration() {
        let mut manager = VersionManager::new();
        let version = Version::parse("1.0.0").unwrap();
        let apiversion = ApiVersionBuilder::new(version.clone()).build().unwrap();

        manager.registerversion(apiversion).unwrap();
        assert_eq!(manager.getversions().len(), 1);
        assert!(manager.getversion(&version).is_some());
    }

    #[test]
    fn test_currentversion_setting() {
        let mut manager = VersionManager::new();
        let version = Version::parse("1.0.0").unwrap();
        let apiversion = ApiVersionBuilder::new(version.clone()).build().unwrap();

        manager.registerversion(apiversion).unwrap();
        manager.set_currentversion(version.clone()).unwrap();
        assert_eq!(manager.currentversion(), Some(&version));
    }

    #[test]
    fn test_stability_levels() {
        assert_eq!(StabilityLevel::Experimental.as_str(), "experimental");
        assert_eq!(StabilityLevel::Stable.as_str(), "stable");
        assert_eq!(StabilityLevel::Mature.as_str(), "mature");

        assert!(StabilityLevel::Experimental < StabilityLevel::Alpha);
        assert!(StabilityLevel::Stable > StabilityLevel::Beta);
    }

    #[test]
    fn test_support_status() {
        assert_eq!(SupportStatus::Active.as_str(), "active");
        assert_eq!(SupportStatus::Deprecated.as_str(), "deprecated");
        assert_eq!(SupportStatus::EndOfLife.as_str(), "end_of_life");
    }

    #[test]
    fn testversion_statistics() {
        let mut manager = VersionManager::new();

        // Add some versions
        let v1 = ApiVersionBuilder::new(Version::parse("1.0.0").unwrap())
            .stability(StabilityLevel::Stable)
            .build()
            .unwrap();
        let v2 = ApiVersionBuilder::new(Version::parse("2.0.0").unwrap())
            .stability(StabilityLevel::Beta)
            .support_status(SupportStatus::Maintenance)
            .build()
            .unwrap();

        manager.registerversion(v1).unwrap();
        manager.registerversion(v2).unwrap();

        let stats = manager.getversion_statistics();
        assert_eq!(stats.totalversions, 2);
        assert_eq!(stats.stableversions, 1);
        assert_eq!(stats.betaversions, 1);
        assert_eq!(stats.activeversions, 1);
        assert_eq!(stats.maintenanceversions, 1);
    }
}
