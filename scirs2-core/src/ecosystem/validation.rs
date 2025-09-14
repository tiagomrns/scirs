//! Ecosystem validation and compatibility checking
//!
//! This module provides comprehensive validation for the SciRS2 ecosystem,
//! ensuring compatibility between modules, API stability, and proper
//! integration for production 1.0 deployments.

use crate::apiversioning::Version;
use crate::error::{CoreError, CoreResult, ErrorContext};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Global ecosystem validator instance
static GLOBAL_VALIDATOR: std::sync::OnceLock<Arc<EcosystemValidator>> = std::sync::OnceLock::new();

/// Comprehensive ecosystem validator for production environments
#[derive(Debug)]
pub struct EcosystemValidator {
    registry: Arc<RwLock<ModuleRegistry>>,
    compatibilitymatrix: Arc<RwLock<CompatibilityMatrix>>,
    validation_cache: Arc<RwLock<ValidationCache>>,
    policies: Arc<RwLock<ValidationPolicies>>,
}

#[allow(dead_code)]
impl EcosystemValidator {
    /// Create new ecosystem validator
    pub fn new() -> CoreResult<Self> {
        Ok(Self {
            registry: Arc::new(RwLock::new(ModuleRegistry::new())),
            compatibilitymatrix: Arc::new(RwLock::new(CompatibilityMatrix::new())),
            validation_cache: Arc::new(RwLock::new(ValidationCache::new())),
            policies: Arc::new(RwLock::new(ValidationPolicies::default())),
        })
    }

    /// Get global validator instance
    pub fn global() -> CoreResult<Arc<Self>> {
        Ok(GLOBAL_VALIDATOR
            .get_or_init(|| Arc::new(Self::new().unwrap()))
            .clone())
    }

    /// Register a module in the ecosystem
    pub fn register_module(&self, module: ModuleInfo) -> CoreResult<()> {
        let mut registry = self.registry.write().map_err(|_| {
            CoreError::InvalidState(crate::error::ErrorContext {
                message: "Failed to acquire registry lock".to_string(),
                location: None,
                cause: None,
            })
        })?;

        registry.register(module)?;

        // Invalidate relevant caches
        let mut cache = self.validation_cache.write().map_err(|_| {
            CoreError::InvalidState(ErrorContext {
                message: "Failed to acquire cache lock".to_string(),
                location: Some(crate::error::ErrorLocation::new(file!(), line!())),
                cause: None,
            })
        })?;
        cache.invalidate_module_related_cache();

        Ok(())
    }

    /// Validate entire ecosystem compatibility
    pub fn validate_ecosystem(&self) -> CoreResult<EcosystemValidationResult> {
        let start_time = Instant::now();

        // Check cache first
        {
            let cache = self.validation_cache.read().map_err(|_| {
                CoreError::InvalidState(ErrorContext {
                    message: "Failed to acquire cache lock".to_string(),
                    location: Some(crate::error::ErrorLocation::new(file!(), line!())),
                    cause: None,
                })
            })?;
            if let Some(cachedresult) = cache.get_ecosystem_validation() {
                if cachedresult.is_recent(Duration::from_secs(300)) {
                    // 5 minutes
                    return Ok(cachedresult.result.clone());
                }
            }
        }

        let registry = self.registry.read().map_err(|_| {
            CoreError::InvalidState(crate::error::ErrorContext {
                message: "Failed to acquire registry lock".to_string(),
                location: None,
                cause: None,
            })
        })?;
        let policies = self.policies.read().map_err(|_| {
            CoreError::InvalidState(ErrorContext {
                message: "Failed to acquire policies lock".to_string(),
                location: Some(crate::error::ErrorLocation::new(file!(), line!())),
                cause: None,
            })
        })?;

        let mut result = EcosystemValidationResult::new();

        // Validate individual modules
        for module in registry.all_modules() {
            let moduleresult = self.validate_module_internal(module, &policies)?;
            result.add_moduleresult(module.name.clone(), moduleresult);
        }

        // Validate inter-module compatibility
        let compatibilityresult = self.validate_inter_module_compatibility(&registry, &policies)?;
        result.add_compatibilityresult(compatibilityresult);

        // Validate API stability
        let api_stabilityresult = self.validate_api_stability(&registry, &policies)?;
        result.add_api_stabilityresult(api_stabilityresult);

        // Validate version consistency
        let version_consistencyresult = self.validate_version_consistency(&registry)?;
        result.add_version_consistencyresult(version_consistencyresult);

        result.validation_time = start_time.elapsed();
        result.timestamp = Instant::now();

        // Cache the result
        {
            let mut cache = self.validation_cache.write().map_err(|_| {
                CoreError::InvalidState(ErrorContext {
                    message: "Failed to acquire cache lock".to_string(),
                    location: Some(crate::error::ErrorLocation::new(file!(), line!())),
                    cause: None,
                })
            })?;
            cache.cache_ecosystem_validation(result.clone());
        }

        Ok(result)
    }

    /// Validate specific module compatibility with ecosystem
    pub fn validate_module(&self, modulename: &str) -> CoreResult<ModuleValidationResult> {
        let registry = self.registry.read().map_err(|_| {
            CoreError::InvalidState(crate::error::ErrorContext {
                message: "Failed to acquire registry lock".to_string(),
                location: None,
                cause: None,
            })
        })?;
        let policies = self.policies.read().map_err(|_| {
            CoreError::InvalidState(ErrorContext {
                message: "Failed to acquire policies lock".to_string(),
                location: Some(crate::error::ErrorLocation::new(file!(), line!())),
                cause: None,
            })
        })?;

        let module = registry.get_module(modulename).ok_or_else(|| {
            CoreError::ValidationError(ErrorContext {
                message: format!("Module '{modulename}' not found in registry"),
                location: None,
                cause: None,
            })
        })?;

        self.validate_module_internal(module, &policies)
    }

    fn validate_module_internal(
        &self,
        module: &ModuleInfo,
        policies: &ValidationPolicies,
    ) -> CoreResult<ModuleValidationResult> {
        let mut result = ModuleValidationResult::new(module.name.clone());

        // Validate version format
        if let Err(e) = Version::parse(&module.version) {
            result.adderror(ValidationError::new(
                ValidationErrorType::InvalidVersion,
                format!("Invalid _version format '{}': {}", module.version, e),
            ));
        }

        // Validate dependencies
        for dep in &module.dependencies {
            let depresult = self.validate_dependencypolicies(module, dep, policies)?;
            if !depresult.is_valid() {
                result.adderror(ValidationError::new(
                    ValidationErrorType::DependencyError,
                    format!("Dependency validation failed for '{}'", dep.name),
                ));
            }
        }

        // Validate API surface
        // TODO: Implement proper API surface validation
        // For now, create a successful result
        let apiresult = ApiStabilityCheck {
            is_stable: true,
            breakingchanges: Vec::new(),
        };
        if !apiresult.is_valid() {
            result.adderror(ValidationError::new(
                ValidationErrorType::ApiCompatibility,
                "API surface validation failed".to_string(),
            ));
        }

        // Validate feature flags
        for feature in &module.features {
            if !self.is_feature_compatible(feature, policies)? {
                result.add_warning(ValidationWarning::new(
                    ValidationWarningType::FeatureCompatibility,
                    format!("Feature '{feature}' may have compatibility issues"),
                ));
            }
        }

        // Validate security requirements
        if policies.enforce_security_checks {
            let securityresult = self.validate_module_security(module)?;
            if !securityresult.is_secure() {
                result.adderror(ValidationError::new(
                    ValidationErrorType::SecurityViolation,
                    "Module failed security validation".to_string(),
                ));
            }
        }

        Ok(result)
    }

    fn validate_dependencypolicies(
        &self,
        module: &ModuleInfo,
        dep: &DependencyInfo,
        policies: &ValidationPolicies,
    ) -> CoreResult<DependencyValidationResult> {
        let mut result = DependencyValidationResult::new(dep.name.clone());

        // Check if dependency exists in registry
        let registry = self.registry.read().map_err(|_| {
            CoreError::InvalidState(crate::error::ErrorContext {
                message: "Failed to acquire registry lock".to_string(),
                location: None,
                cause: None,
            })
        })?;

        if let Some(dep_module) = registry.get_module(&dep.name) {
            // Validate version compatibility
            let dep_version = Version::parse(&dep_module.version).map_err(|e| {
                CoreError::ValidationError(ErrorContext {
                    message: format!("Invalid dependency version: {e}"),
                    location: None,
                    cause: None,
                })
            })?;

            if !dep.version_requirement.version(&dep_version) {
                result.add_incompatibility(format!(
                    "Version mismatch: required {}, found {}",
                    dep.version_requirement, dep_version
                ));
            }

            // Check circular dependencies
            if self.has_circular_dependency(&module.name, &dep.name) {
                result.add_incompatibility("Circular dependency detected".to_string());
            }
        } else {
            result.add_incompatibility("Dependency not found in ecosystem".to_string());
        }

        Ok(result)
    }

    fn validate_inter_module_compatibility(
        &self,
        registry: &ModuleRegistry,
        policies: &ValidationPolicies,
    ) -> CoreResult<CompatibilityValidationResult> {
        let mut result = CompatibilityValidationResult::new();
        let modules = registry.all_modules();

        // Build compatibility matrix
        let mut matrix = self.compatibilitymatrix.write().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire matrix lock".to_string(),
            ))
        })?;

        for module_a in &modules {
            for module_b in &modules {
                if module_a.name != module_b.name {
                    let compatibility =
                        self.check_module_compatibility(module_a, module_b, policies)?;
                    (*matrix).b(&module_a.name, &module_b.name, compatibility.clone());

                    if !compatibility.is_compatible() {
                        result.add_incompatibility(format!(
                            "Modules '{}' and '{}' are incompatible: {}",
                            module_a.name,
                            module_b.name,
                            compatibility.reason_2()
                        ));
                    }
                }
            }
        }

        Ok(result)
    }

    fn check_module_compatibility(
        &self,
        module_a: &ModuleInfo,
        module_b: &ModuleInfo,
        policies: &ValidationPolicies,
    ) -> CoreResult<ModuleCompatibility> {
        // Check version compatibility
        let version_a = Version::parse(&module_a.version).map_err(|e| {
            CoreError::ValidationError(ErrorContext {
                message: format!("Invalid _version for module '{}': {}", module_a.name, e),
                location: Some(crate::error::ErrorLocation::new(file!(), line!())),
                cause: None,
            })
        })?;
        let version_b = Version::parse(&module_b.version).map_err(|e| {
            CoreError::ValidationError(ErrorContext {
                message: format!("Invalid _version for module '{}': {}", module_b.name, e),
                location: Some(crate::error::ErrorLocation::new(file!(), line!())),
                cause: None,
            })
        })?;

        if !self.areversions_compatible(&version_a.to_string(), &version_b.to_string()) {
            return Ok(ModuleCompatibility::incompatible(format!(
                "Version incompatibility: {version_a} vs {version_b}"
            )));
        }

        // Check API compatibility
        if !self.are_apis_compatible(&module_a.apisurface, &module_b.apisurface) {
            return Ok(ModuleCompatibility::incompatible(
                "API incompatibility".to_string(),
            ));
        }

        // Check feature compatibility
        if !self.are_features_compatible(&module_a.features, &module_b.features) {
            return Ok(ModuleCompatibility::incompatible(
                "Feature incompatibility".to_string(),
            ));
        }

        Ok(ModuleCompatibility::compatible())
    }

    fn validate_api_stability(
        &self,
        registry: &ModuleRegistry,
        policies: &ValidationPolicies,
    ) -> CoreResult<ApiStabilityResult> {
        let mut result = ApiStabilityResult::new();

        for module in registry.all_modules() {
            // Check for breaking changes in API
            if let Some(_previous_version) = registry.get_previous_version(&module.name) {
                let stability_check = self.check_api_stability(&module.name);
                if !stability_check.is_stable() {
                    result.add_breaking_change(
                        module.name.clone(),
                        stability_check.breakingchanges().to_vec(),
                    );
                }
            }

            // Validate API versioning compliance
            if !self.is_api_properly_versioned(&module.apisurface) {
                result.add_versioning_violation(
                    module.name.clone(),
                    "API not properly versioned".to_string(),
                );
            }
        }

        Ok(result)
    }

    fn validate_version_consistency(
        &self,
        registry: &ModuleRegistry,
    ) -> CoreResult<VersionConsistencyResult> {
        let mut result = VersionConsistencyResult::new();
        let modules = registry.all_modules();

        // Check for version conflicts
        let mut version_map: HashMap<String, Vec<Version>> = HashMap::new();

        for module in &modules {
            let version = Version::parse(&module.version).map_err(|e| {
                CoreError::ValidationError(ErrorContext {
                    message: format!("Invalid version for module '{}': {}", module.name, e),
                    location: Some(crate::error::ErrorLocation::new(file!(), line!())),
                    cause: None,
                })
            })?;
            version_map
                .entry(module.name.clone())
                .or_default()
                .push(version);
        }

        for (modulename, versions) in version_map {
            if versions.len() > 1 {
                result.add_conflict(modulename, versions);
            }
        }

        // Validate dependency version consistency
        for module in &modules {
            for dep in &module.dependencies {
                if let Some(dep_module) = registry.get_module(&dep.name) {
                    let dep_version = Version::parse(&dep_module.version).map_err(|e| {
                        CoreError::ValidationError(ErrorContext {
                            message: format!(
                                "Invalid _version format for dependency {}: {}",
                                dep.name, e
                            ),
                            location: Some(crate::error::ErrorLocation::new(file!(), line!())),
                            cause: None,
                        })
                    })?;
                    if !dep.version_requirement.version(&dep_version) {
                        result.add_dependency_mismatch(
                            module.name.clone(),
                            dep.name.clone(),
                            dep.version_requirement.clone(),
                            dep_version,
                        );
                    }
                }
            }
        }

        Ok(result)
    }

    #[allow(dead_code)]
    fn surface(
        &self,
        apisurface: &ApiSurface,
        policies: &ValidationPolicies,
    ) -> CoreResult<ApiValidationResult> {
        let mut result = ApiValidationResult::new();

        // Validate public APIs
        for api in &apisurface.public_apis {
            if !self.is_api_properly_documented(api)? {
                result.add_documentation_issue(api.name.clone());
            }

            if policies.enforce_semver && !self.is_api_semver_compliant(api)? {
                result.add_semver_violation(api.name.clone());
            }
        }

        // Validate deprecated APIs
        for api in &apisurface.deprecated_apis {
            if !api.has_migration_path() {
                result.add_deprecation_issue(
                    api.name.clone(),
                    "No migration path provided".to_string(),
                );
            }
        }

        Ok(result)
    }

    fn validate_module_security(
        &self,
        module: &ModuleInfo,
    ) -> CoreResult<SecurityValidationResult> {
        let mut result = SecurityValidationResult::new(module.name.clone());

        // Check for known vulnerabilities
        for dep in &module.dependencies {
            if self.has_known_vulnerabilities(&dep.name) {
                result.add_vulnerability(format!(
                    "Dependency '{}' has known vulnerabilities",
                    dep.name
                ));
            }
        }

        // Validate security features
        if !module.features.contains(&"security".to_string())
            && self.requires_security_features(module)?
        {
            result.add_security_issue("Module should enable security features".to_string());
        }

        Ok(result)
    }

    // Helper methods
    #[allow(dead_code)]
    fn check_circular_dependencies(
        &self,
        modulename: &str,
        dependencies: &[DependencyInfo],
    ) -> CoreResult<bool> {
        // Simple circular dependency detection
        for dep in dependencies {
            if dep.name == modulename {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn check_version_compatibility(
        &self,
        version_a: &Version,
        version_b: &Version,
        policies: &ValidationPolicies,
    ) -> bool {
        if policies.strict_version_matching {
            version_a == version_b
        } else {
            // Allow compatible versions (same major version)
            version_a.major == version_b.major
        }
    }

    fn check_api_compatibility(&self, api_a: &ApiSurface, apib: &ApiSurface) -> CoreResult<bool> {
        // Simple API compatibility check
        // In _a real implementation, this would do deep API analysis
        Ok(api_a.public_apis.len() == apib.public_apis.len())
    }

    fn check_feature_compatibility(
        &self,
        _features_a: &[String],
        _features_b: &[String],
    ) -> CoreResult<bool> {
        // Check for conflicting features
        // In _a real implementation, would check for conflicts
        // No conflicting features for now
        Ok(true)
    }

    fn validate_apipolicies(
        &self,
        previous: &ApiSurface,
        current: &ApiSurface,
        policies: &ValidationPolicies,
    ) -> CoreResult<ApiStabilityCheck> {
        let mut breakingchanges = Vec::new();

        // Check for removed APIs
        for prev_api in &previous.public_apis {
            if !current
                .public_apis
                .iter()
                .any(|api| api.name == prev_api.name)
            {
                breakingchanges.push(format!("API '{}' was removed", prev_api.name));
            }
        }

        // Check for signature changes
        for current_api in &current.public_apis {
            if let Some(prev_api) = previous
                .public_apis
                .iter()
                .find(|api| api.name == current_api.name)
            {
                if current_api.signature != prev_api.signature {
                    breakingchanges.push(format!("API '{}' signature changed", current_api.name));
                }
            }
        }

        Ok(ApiStabilityCheck::new(
            breakingchanges.is_empty(),
            breakingchanges,
        ))
    }

    fn is_apisurface_versioned(&self, apisurface: &ApiSurface) -> CoreResult<bool> {
        // Check if all APIs have version information
        for api in &apisurface.public_apis {
            if api.since_version.is_none() {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn is_feature_compatible(
        &self,
        feature: &str,
        policies: &ValidationPolicies,
    ) -> CoreResult<bool> {
        // Check feature compatibility rules
        Ok(!policies.incompatible_features.contains(feature))
    }

    fn is_api_properly_documented(&self, api: &ApiInfo) -> CoreResult<bool> {
        Ok(!api.documentation.is_empty())
    }

    fn is_api_semver_compliant(&self, api: &ApiInfo) -> CoreResult<bool> {
        // Check if API follows semantic versioning principles
        Ok(api.since_version.is_some())
    }

    fn req(&self, _module: &ModuleInfo, _versionreq: &VersionRequirement) -> CoreResult<bool> {
        // In a real implementation, this would check against vulnerability databases
        Ok(false)
    }

    fn requires_security_features(&self, module: &ModuleInfo) -> CoreResult<bool> {
        // Check if module handles sensitive data or network operations
        Ok(module.name.contains("network") || module.name.contains("auth"))
    }

    /// Update validation policies
    pub fn updatepolicies(&self, newpolicies: ValidationPolicies) -> CoreResult<()> {
        let mut policies = self.policies.write().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire policies lock".to_string(),
            ))
        })?;
        *policies = newpolicies;

        // Clear cache since policies changed
        let mut cache = self.validation_cache.write().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire cache lock".to_string(),
            ))
        })?;
        cache.clear();

        Ok(())
    }

    /// Get ecosystem health summary
    pub fn get_ecosystem_health(&self) -> CoreResult<EcosystemHealth> {
        let validationresult = self.validate_ecosystem()?;
        Ok(EcosystemHealth::from_validationresult(&validationresult))
    }

    pub fn has_circular_dependency(&self, _module: &str, dependency: &str) -> bool {
        // Placeholder implementation
        false
    }

    pub fn areversions_compatible(&self, _version_a: &str, _versionb: &str) -> bool {
        // Placeholder implementation
        true
    }

    pub fn are_apis_compatible(&self, _api_a: &ApiSurface, _apib: &ApiSurface) -> bool {
        // Placeholder implementation
        true
    }

    pub fn are_features_compatible(&self, _features_a: &[String], _featuresb: &[String]) -> bool {
        // Placeholder implementation
        true
    }

    pub fn check_api_stability(&self, module: &str) -> ApiStabilityCheck {
        // Placeholder implementation
        ApiStabilityCheck::new(true, vec![])
    }

    pub fn is_api_properly_versioned(&self, _apisurface: &ApiSurface) -> bool {
        // Placeholder implementation
        true
    }

    pub fn has_known_vulnerabilities(&self, module: &str) -> bool {
        // Placeholder implementation
        false
    }
}

/// Module registry for tracking ecosystem components
#[derive(Debug)]
pub struct ModuleRegistry {
    modules: HashMap<String, ModuleInfo>,
    previousversions: HashMap<String, ModuleInfo>,
}

impl ModuleRegistry {
    pub fn new() -> Self {
        Self {
            modules: HashMap::new(),
            previousversions: HashMap::new(),
        }
    }

    pub fn register(&mut self, module: ModuleInfo) -> CoreResult<()> {
        // Store previous version if updating
        if let Some(existing) = self.modules.get(&module.name) {
            self.previousversions
                .insert(module.name.clone(), existing.clone());
        }

        self.modules.insert(module.name.clone(), module);
        Ok(())
    }

    pub fn get_module(&self, name: &str) -> Option<&ModuleInfo> {
        self.modules.get(name)
    }

    pub fn get_previous_version(&self, name: &str) -> Option<&ModuleInfo> {
        self.previousversions.get(name)
    }

    pub fn all_modules(&self) -> Vec<&ModuleInfo> {
        self.modules.values().collect()
    }

    pub fn module_count(&self) -> usize {
        self.modules.len()
    }
}

impl Default for ModuleRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Information about a module in the ecosystem
#[derive(Debug, Clone)]
pub struct ModuleInfo {
    pub name: String,
    pub version: String,
    pub dependencies: Vec<DependencyInfo>,
    pub apisurface: ApiSurface,
    pub features: Vec<String>,
    pub metadata: ModuleMetadata,
}

#[derive(Debug, Clone)]
pub struct DependencyInfo {
    pub name: String,
    pub version_requirement: VersionRequirement,
    pub optional: bool,
}

#[derive(Debug, Clone)]
pub struct VersionRequirement {
    pub requirement: String,
}

impl VersionRequirement {
    pub fn new(requirement: &str) -> Self {
        Self {
            requirement: requirement.to_string(),
        }
    }

    pub fn version(&self, version: &Version) -> bool {
        // Simple version matching for now
        // In a real implementation, this would parse semver requirements
        self.requirement == version.to_string()
    }
}

impl std::fmt::Display for VersionRequirement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.requirement)
    }
}

#[derive(Debug, Clone)]
pub struct ApiSurface {
    pub public_apis: Vec<ApiInfo>,
    pub deprecated_apis: Vec<DeprecatedApiInfo>,
}

#[derive(Debug, Clone)]
pub struct ApiInfo {
    pub name: String,
    pub signature: String,
    pub documentation: String,
    pub since_version: Option<Version>,
    pub stability: ApiStability,
}

#[derive(Debug, Clone)]
pub struct DeprecatedApiInfo {
    pub name: String,
    pub deprecated_since: Version,
    pub removal_version: Option<Version>,
    pub migration_path: Option<String>,
}

impl DeprecatedApiInfo {
    pub fn has_migration_path(&self) -> bool {
        self.migration_path.is_some()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiStability {
    Stable,
    Unstable,
    Experimental,
}

#[derive(Debug, Clone)]
pub struct ModuleMetadata {
    pub author: String,
    pub description: String,
    pub license: String,
    pub repository: Option<String>,
    pub build_time: Option<String>,
}

/// Compatibility matrix for module interactions
#[derive(Debug)]
pub struct CompatibilityMatrix {
    matrix: HashMap<(String, String), ModuleCompatibility>,
}

impl CompatibilityMatrix {
    pub fn new() -> Self {
        Self {
            matrix: HashMap::new(),
        }
    }

    pub fn b(&mut self, module_a: &str, moduleb: &str, compatibility: ModuleCompatibility) {
        self.matrix
            .insert((module_a.to_string(), moduleb.to_string()), compatibility);
    }

    pub fn b_2(&self, module_a: &str, moduleb: &str) -> Option<&ModuleCompatibility> {
        self.matrix
            .get(&(module_a.to_string(), moduleb.to_string()))
    }
}

impl Default for CompatibilityMatrix {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct ModuleCompatibility {
    compatible: bool,
    reason: String,
}

impl ModuleCompatibility {
    pub fn compatible() -> Self {
        Self {
            compatible: true,
            reason: String::new(),
        }
    }

    pub fn incompatible(reason: String) -> Self {
        Self {
            compatible: false,
            reason,
        }
    }
}

impl ModuleCompatibility {
    pub fn is_compatible(&self) -> bool {
        self.compatible
    }

    pub fn reason_2(&self) -> &str {
        &self.reason
    }
}

/// Validation cache for performance optimization
#[derive(Debug)]
pub struct ValidationCache {
    ecosystem_validation: Option<CachedValidationResult>,
    module_validations: HashMap<String, CachedModuleValidation>,
}

impl ValidationCache {
    pub fn new() -> Self {
        Self {
            ecosystem_validation: None,
            module_validations: HashMap::new(),
        }
    }

    pub fn cache_ecosystem_validation(&mut self, result: EcosystemValidationResult) {
        self.ecosystem_validation = Some(CachedValidationResult {
            result,
            timestamp: Instant::now(),
        });
    }

    pub fn get_ecosystem_validation(&self) -> Option<&CachedValidationResult> {
        self.ecosystem_validation.as_ref()
    }

    pub fn invalidate_module_related_cache(&mut self) {
        self.ecosystem_validation = None;
        self.module_validations.clear();
    }

    pub fn clear(&mut self) {
        self.ecosystem_validation = None;
        self.module_validations.clear();
    }
}

impl Default for ValidationCache {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct CachedValidationResult {
    pub result: EcosystemValidationResult,
    pub timestamp: Instant,
}

impl CachedValidationResult {
    pub fn age(&self, maxage: Duration) -> bool {
        self.timestamp.elapsed() < maxage
    }

    pub fn is_recent(&self, maxage: Duration) -> bool {
        self.age(maxage)
    }
}

#[derive(Debug)]
pub struct CachedModuleValidation {
    pub result: ModuleValidationResult,
    pub timestamp: Instant,
}

/// Validation policies and configuration
#[derive(Debug, Clone)]
pub struct ValidationPolicies {
    pub enforce_semver: bool,
    pub strict_version_matching: bool,
    pub enforce_security_checks: bool,
    pub allow_deprecated_apis: bool,
    pub max_dependency_depth: usize,
    pub incompatible_features: HashSet<String>,
    pub required_features: HashSet<String>,
}

impl Default for ValidationPolicies {
    fn default() -> Self {
        Self {
            enforce_semver: true,
            strict_version_matching: false,
            enforce_security_checks: true,
            allow_deprecated_apis: true,
            max_dependency_depth: 10,
            incompatible_features: HashSet::new(),
            required_features: HashSet::new(),
        }
    }
}

/// Comprehensive validation results
#[derive(Debug, Clone)]
pub struct EcosystemValidationResult {
    pub timestamp: Instant,
    pub validation_time: Duration,
    pub moduleresults: HashMap<String, ModuleValidationResult>,
    pub compatibilityresult: CompatibilityValidationResult,
    pub api_stabilityresult: ApiStabilityResult,
    pub version_consistencyresult: VersionConsistencyResult,
    pub overall_status: ValidationStatus,
}

impl EcosystemValidationResult {
    pub fn new() -> Self {
        Self {
            timestamp: Instant::now(),
            validation_time: Duration::ZERO,
            moduleresults: HashMap::new(),
            compatibilityresult: CompatibilityValidationResult::new(),
            api_stabilityresult: ApiStabilityResult::new(),
            version_consistencyresult: VersionConsistencyResult::new(),
            overall_status: ValidationStatus::Unknown,
        }
    }

    pub fn name(&mut self, modulename: String, result: ModuleValidationResult) {
        self.moduleresults.insert(modulename, result);
        self.update_overall_status();
    }

    pub fn add_moduleresult(&mut self, modulename: String, result: ModuleValidationResult) {
        self.moduleresults.insert(modulename, result);
        self.update_overall_status();
    }

    pub fn add_compatibilityresult(&mut self, result: CompatibilityValidationResult) {
        self.compatibilityresult = result;
        self.update_overall_status();
    }

    pub fn add_api_stabilityresult(&mut self, result: ApiStabilityResult) {
        self.api_stabilityresult = result;
        self.update_overall_status();
    }

    pub fn add_version_consistencyresult(&mut self, result: VersionConsistencyResult) {
        self.version_consistencyresult = result;
        self.update_overall_status();
    }

    fn update_overall_status(&mut self) {
        let haserrors = self.moduleresults.values().any(|r| !r.errors.is_empty())
            || !self.compatibilityresult.incompatibilities.is_empty()
            || !self.api_stabilityresult.breakingchanges.is_empty()
            || !self.version_consistencyresult.conflicts.is_empty();

        let has_warnings = self.moduleresults.values().any(|r| !r.warnings.is_empty());

        self.overall_status = if haserrors {
            ValidationStatus::Failed
        } else if has_warnings {
            ValidationStatus::Warning
        } else {
            ValidationStatus::Passed
        };
    }

    pub fn is_valid(&self) -> bool {
        matches!(
            self.overall_status,
            ValidationStatus::Passed | ValidationStatus::Warning
        )
    }

    pub fn error_count(&self) -> usize {
        self.moduleresults
            .values()
            .map(|r| r.errors.len())
            .sum::<usize>()
            + self.compatibilityresult.incompatibilities.len()
            + self.api_stabilityresult.breakingchanges.len()
            + self.version_consistencyresult.conflicts.len()
    }

    pub fn warning_count(&self) -> usize {
        self.moduleresults.values().map(|r| r.warnings.len()).sum()
    }
}

impl Default for EcosystemValidationResult {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationStatus {
    Unknown,
    Passed,
    Warning,
    Failed,
}

/// Individual module validation result
#[derive(Debug, Clone)]
pub struct ModuleValidationResult {
    pub modulename: String,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
    pub status: ValidationStatus,
}

impl ModuleValidationResult {
    pub fn new(modulename: String) -> Self {
        Self {
            modulename,
            errors: Vec::new(),
            warnings: Vec::new(),
            status: ValidationStatus::Unknown,
        }
    }

    pub fn adderror(&mut self, error: ValidationError) {
        self.errors.push(error);
        self.status = ValidationStatus::Failed;
    }

    pub fn add_warning(&mut self, warning: ValidationWarning) {
        self.warnings.push(warning);
        if self.status == ValidationStatus::Unknown {
            self.status = ValidationStatus::Warning;
        }
    }

    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct ValidationError {
    pub errortype: ValidationErrorType,
    pub message: String,
    pub context: Option<String>,
}

impl ValidationError {
    pub fn new(errortype: ValidationErrorType, message: String) -> Self {
        Self {
            errortype,
            message,
            context: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationErrorType {
    InvalidVersion,
    DependencyError,
    ApiCompatibility,
    SecurityViolation,
    FeatureConflict,
}

#[derive(Debug, Clone)]
pub struct ValidationWarning {
    pub warningtype: ValidationWarningType,
    pub message: String,
}

impl ValidationWarning {
    pub fn new(warningtype: ValidationWarningType, message: String) -> Self {
        Self {
            warningtype,
            message,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationWarningType {
    FeatureCompatibility,
    PerformanceImpact,
    DeprecationWarning,
}

/// Additional validation result types
#[derive(Debug, Clone)]
pub struct CompatibilityValidationResult {
    pub incompatibilities: Vec<String>,
}

impl CompatibilityValidationResult {
    pub fn new() -> Self {
        Self {
            incompatibilities: Vec::new(),
        }
    }

    pub fn add_incompatibility(&mut self, incompatibility: String) {
        self.incompatibilities.push(incompatibility);
    }
}

impl Default for CompatibilityValidationResult {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct ApiStabilityResult {
    pub breakingchanges: HashMap<String, Vec<String>>,
    pub versioning_violations: HashMap<String, String>,
}

impl ApiStabilityResult {
    pub fn new() -> Self {
        Self {
            breakingchanges: HashMap::new(),
            versioning_violations: HashMap::new(),
        }
    }

    pub fn add_breaking_change(&mut self, module: String, changes: Vec<String>) {
        self.breakingchanges.insert(module, changes);
    }

    pub fn add_versioning_violation(&mut self, module: String, violation: String) {
        self.versioning_violations.insert(module, violation);
    }
}

impl Default for ApiStabilityResult {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct VersionConsistencyResult {
    pub conflicts: HashMap<String, Vec<Version>>,
    pub dependency_mismatches: Vec<DependencyMismatch>,
}

impl VersionConsistencyResult {
    pub fn new() -> Self {
        Self {
            conflicts: HashMap::new(),
            dependency_mismatches: Vec::new(),
        }
    }

    pub fn add_conflict(&mut self, module: String, versions: Vec<Version>) {
        self.conflicts.insert(module, versions);
    }

    pub fn add_dependency_mismatch(
        &mut self,
        module: String,
        dependency: String,
        required: VersionRequirement,
        found: Version,
    ) {
        self.dependency_mismatches.push(DependencyMismatch {
            module,
            dependency,
            required,
            found,
        });
    }
}

impl Default for VersionConsistencyResult {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct DependencyMismatch {
    pub module: String,
    pub dependency: String,
    pub required: VersionRequirement,
    pub found: Version,
}

/// Additional helper types
#[derive(Debug, Clone)]
pub struct DependencyValidationResult {
    pub dependency_name: String,
    pub incompatibilities: Vec<String>,
}

impl DependencyValidationResult {
    pub fn new(modulename: String) -> Self {
        Self {
            dependency_name: modulename,
            incompatibilities: Vec::new(),
        }
    }

    pub fn add_incompatibility(&mut self, incompatibility: String) {
        self.incompatibilities.push(incompatibility);
    }

    pub fn is_valid(&self) -> bool {
        self.incompatibilities.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct ApiValidationResult {
    pub documentation_issues: Vec<String>,
    pub semver_violations: Vec<String>,
    pub deprecation_issues: HashMap<String, String>,
}

impl ApiValidationResult {
    pub fn new() -> Self {
        Self {
            documentation_issues: Vec::new(),
            semver_violations: Vec::new(),
            deprecation_issues: HashMap::new(),
        }
    }

    pub fn name(&mut self, apiname: String) {
        self.documentation_issues.push(apiname);
    }

    pub fn add_documentation_issue(&mut self, apiname: String) {
        self.documentation_issues.push(apiname);
    }

    pub fn name_2(&mut self, apiname: String) {
        self.semver_violations.push(apiname);
    }

    pub fn add_semver_violation(&mut self, apiname: String) {
        self.semver_violations.push(apiname);
    }

    pub fn name_3(&mut self, apiname: String, issue: String) {
        self.deprecation_issues.insert(apiname, issue);
    }

    pub fn add_deprecation_issue(&mut self, apiname: String, issue: String) {
        self.deprecation_issues.insert(apiname, issue);
    }

    pub fn is_valid(&self) -> bool {
        self.documentation_issues.is_empty()
            && self.semver_violations.is_empty()
            && self.deprecation_issues.is_empty()
    }
}

impl Default for ApiValidationResult {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct SecurityValidationResult {
    pub modulename: String,
    pub vulnerabilities: Vec<String>,
    pub security_issues: Vec<String>,
}

impl SecurityValidationResult {
    pub fn new(modulename: String) -> Self {
        Self {
            modulename,
            vulnerabilities: Vec::new(),
            security_issues: Vec::new(),
        }
    }

    pub fn add_vulnerability(&mut self, vulnerability: String) {
        self.vulnerabilities.push(vulnerability);
    }

    pub fn add_security_issue(&mut self, issue: String) {
        self.security_issues.push(issue);
    }

    pub fn is_secure(&self) -> bool {
        self.vulnerabilities.is_empty() && self.security_issues.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct ApiStabilityCheck {
    pub is_stable: bool,
    pub breakingchanges: Vec<String>,
}

impl ApiStabilityCheck {
    pub fn new(is_stable: bool, breakingchanges: Vec<String>) -> Self {
        Self {
            is_stable,
            breakingchanges,
        }
    }

    pub fn is_stable(&self) -> bool {
        self.is_stable
    }

    pub fn breakingchanges(&self) -> &[String] {
        &self.breakingchanges
    }

    pub fn is_valid(&self) -> bool {
        self.is_stable && self.breakingchanges.is_empty()
    }
}

/// Ecosystem health summary
#[derive(Debug, Clone)]
pub struct EcosystemHealth {
    pub overall_status: HealthStatus,
    pub module_count: usize,
    pub error_count: usize,
    pub warning_count: usize,
    pub compatibility_score: f64,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    Excellent,
    Good,
    Fair,
    Poor,
    Critical,
}

impl EcosystemHealth {
    pub fn from_validationresult(result: &EcosystemValidationResult) -> Self {
        let module_count = result.moduleresults.len();
        let error_count = result.error_count();
        let warning_count = result.warning_count();

        let compatibility_score = if module_count == 0 {
            1.0
        } else {
            1.0 - (error_count as f64 / (module_count as f64 * 10.0))
        };

        let overall_status = match error_count {
            0 => {
                if warning_count == 0 {
                    HealthStatus::Excellent
                } else {
                    HealthStatus::Good
                }
            }
            1..=5 => HealthStatus::Fair,
            6..=15 => HealthStatus::Poor,
            _ => HealthStatus::Critical,
        };

        let recommendations = Self::generate_recommendations(result);

        Self {
            overall_status,
            module_count,
            error_count,
            warning_count,
            compatibility_score: compatibility_score.clamp(0.0, 1.0),
            recommendations,
        }
    }

    fn generate_recommendations(result: &EcosystemValidationResult) -> Vec<String> {
        let mut recommendations = Vec::new();

        if result.error_count() > 0 {
            recommendations
                .push("Address validation errors before production deployment".to_string());
        }

        if !result.api_stabilityresult.breakingchanges.is_empty() {
            recommendations
                .push("Review API breaking changes and update version numbers".to_string());
        }

        if !result.version_consistencyresult.conflicts.is_empty() {
            recommendations.push("Resolve version conflicts between modules".to_string());
        }

        if result.warning_count() > 10 {
            recommendations
                .push("Consider addressing warnings to improve ecosystem stability".to_string());
        }

        recommendations
    }
}

/// Initialize ecosystem validation with detected modules
#[allow(dead_code)]
pub fn initialize_ecosystem_validation() -> CoreResult<()> {
    let validator = EcosystemValidator::global()?;

    // Register core modules
    validator.register_module(create_core_module_info())?;

    // Additional modules would be detected and registered here

    Ok(())
}

#[allow(dead_code)]
fn create_core_module_info() -> ModuleInfo {
    ModuleInfo {
        name: "scirs2-core".to_string(),
        version: "1.0.0".to_string(),
        dependencies: Vec::new(),
        apisurface: ApiSurface {
            public_apis: vec![ApiInfo {
                name: "validate_ecosystem".to_string(),
                signature: "fn validate_ecosystem() -> CoreResult<EcosystemValidationResult>"
                    .to_string(),
                documentation: "Validates the entire ecosystem compatibility".to_string(),
                since_version: Some(Version::new(1, 0, 0)),
                stability: ApiStability::Stable,
            }],
            deprecated_apis: Vec::new(),
        },
        features: vec!["validation".to_string(), "ecosystem".to_string()],
        metadata: ModuleMetadata {
            author: "SciRS2 Team".to_string(),
            description: "Core utilities for SciRS2 ecosystem".to_string(),
            license: "MIT".to_string(),
            repository: Some("https://github.com/cool-japan/scirs".to_string()),
            build_time: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_creation() {
        let validator = EcosystemValidator::new().unwrap();
        // Basic functionality test
    }

    #[test]
    fn test_module_registration() {
        let validator = EcosystemValidator::new().unwrap();
        let module = create_core_module_info();

        validator.register_module(module).unwrap();

        let result = validator.validate_module("scirs2-core").unwrap();
        assert!(result.is_valid());
    }

    #[test]
    fn test_ecosystem_validation() {
        let validator = EcosystemValidator::new().unwrap();
        validator
            .register_module(create_core_module_info())
            .unwrap();

        let result = validator.validate_ecosystem().unwrap();
        assert!(result.is_valid());
    }

    #[test]
    fn test_version_requirement() {
        let req = VersionRequirement::new("1.0.0");
        let version = Version::new(1, 0, 0);

        assert!(req.version(&version));
    }

    #[test]
    fn test_ecosystem_health() {
        let mut result = EcosystemValidationResult::new();
        result.add_moduleresult(
            "test".to_string(),
            ModuleValidationResult::new("test".to_string()),
        );

        let health = EcosystemHealth::from_validationresult(&result);
        assert_eq!(health.overall_status, HealthStatus::Excellent);
    }
}
