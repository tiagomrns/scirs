# API Consistency Analysis Results

## Summary
Analysis of scirs2-stats public APIs reveals several areas for improvement to achieve v1.0.0 consistency standards.

## Major Consistency Issues Found

### 1. Parameter Naming Inconsistencies
- **Correlation functions**: Use `x, y` parameters consistently
- **Statistical tests**: Some use `a, b` while others use `x, y` 
- **Descriptive stats**: Mix of `x`, `data`, `a` parameter names
- **Recommendation**: Standardize on `x, y` for bivariate functions, `data` for univariate

### 2. Return Type Variations
- **Simple correlation functions**: Return `StatsResult<F>` (single value)
- **Enhanced correlation functions**: Return `StatsResult<(F, F)>` (value + p-value)
- **Statistical tests**: Return `StatsResult<TTestResult<F>>` (structured result)
- **Recommendation**: Create consistent result types for similar operation classes

### 3. Error Handling Inconsistencies
- **Validation patterns**: Mix of manual checks and `ErrorMessages::` calls
- **NaN handling**: Some functions have `nan_policy` parameter, others don't
- **Empty array handling**: Inconsistent error messages
- **Recommendation**: Standardize validation using `ErrorMessages` throughout

### 4. Alternative Hypothesis Handling  
- **T-tests**: Use `Alternative` enum
- **Correlation p-values**: Use string parameters like "two-sided"
- **Other tests**: Mix of approaches
- **Recommendation**: Standardize on `Alternative` enum across all statistical tests

### 5. Generic Type Constraints
- **Inconsistent bounds**: Some functions require `std::fmt::Debug`, others don't
- **NumCast requirements**: Applied inconsistently
- **Float bounds**: Varying precision requirements
- **Recommendation**: Standardize generic constraints for similar function categories

## Specific Recommendations

### 1. Create Standardized Result Types
```rust
pub struct CorrelationResult<F: Float> {
    pub coefficient: F,
    pub pvalue: Option<F>,
    pub confidence_interval: Option<(F, F)>,
}

pub struct TestResult<F: Float> {
    pub statistic: F,
    pub pvalue: F,
    pub effect_size: Option<F>,
    pub confidence_interval: Option<(F, F)>,
}
```

### 2. Standardize Parameter Names
- Univariate functions: `data: &ArrayView1<F>`
- Bivariate functions: `x: &ArrayView1<F>, y: &ArrayView1<F>`
- Statistical tests: `sample1, sample2` or `treatment, control`

### 3. Consistent NaN Handling
- Add `nan_policy: NanPolicy` parameter to all functions that process data
- Default to `NanPolicy::Propagate` for consistency with SciPy

### 4. Unified Alternative Hypothesis System
- Extend `Alternative` enum for all statistical tests
- Deprecate string-based alternative parameters

### 5. Standardized Generic Constraints
```rust
// For basic statistical functions
where F: Float + std::iter::Sum<F> + NumCast

// For advanced statistical functions  
where F: Float + std::iter::Sum<F> + NumCast + std::fmt::Debug
```

## Implementation Priority
1. **High**: Error handling and validation standardization
2. **High**: Parameter naming consistency  
3. **Medium**: Return type unification
4. **Medium**: Alternative hypothesis standardization
5. **Low**: Generic constraint optimization

## Breaking Changes Assessment
- Parameter name changes: **Breaking**
- Return type changes: **Breaking** 
- Alternative enum expansion: **Non-breaking** (with deprecation)
- Error message standardization: **Non-breaking**

## Recommendation
Address these issues before v1.0.0 release to establish stable API patterns.