# API Standardization Plan for scirs2-stats

## Distribution Convenience Functions

We'll follow SciPy's naming convention which uses abbreviated names for commonly used distributions:

### Current → Standardized Mapping

**Keep as-is (already following SciPy):**
- `norm` - Normal distribution
- `uniform` - Uniform distribution  
- `t` - Student's t distribution
- `chi2` - Chi-square distribution
- `f` - F distribution
- `gamma` - Gamma distribution
- `beta` - Beta distribution
- `expon` - Exponential distribution
- `poisson` - Poisson distribution
- `binom` - Binomial distribution
- `geom` - Geometric distribution
- `nbinom` - Negative binomial distribution

**Need to add aliases for consistency:**
- `bernoulli` → Add alias `bern` (keep both)
- `cauchy` → Already correct (no abbreviation in SciPy)
- `laplace` → Already correct (no abbreviation in SciPy)  
- `logistic` → Already correct (no abbreviation in SciPy)
- `lognorm` → Already correct (standard abbreviation)
- `weibull` → Add alias `weibull_min` (SciPy convention)
- `pareto` → Already correct (no abbreviation in SciPy)
- `hypergeom` → Already correct (standard abbreviation)
- `vonmises` → Already correct (no abbreviation in SciPy)
- `wrapcauchy` → Already correct (no abbreviation in SciPy)

## Statistical Test Functions

### Consolidate duplicate functions:
- Remove `mannwhitneyu` alias, keep only `mann_whitney`
- Remove legacy `ttest_1samp`, `ttest_ind`, `ttest_rel` - use enhanced versions only
- Rename enhanced versions to remove "enhanced_" prefix

### Correlation Functions
- Keep both versions as they serve different purposes:
  - `pearson_r`, `spearman_r`, `kendall_tau` - return only correlation coefficient (fast, no p-value)
  - `pearsonr`, `spearmanr`, `kendallr` - return (correlation, p-value) tuple matching SciPy
- Consider renaming `kendallr` to `kendalltau` to match SciPy exactly
- Document clearly which function to use when

## Parameter Ordering

### Standard parameter order for distributions:
1. Shape parameters (if any)
2. Location parameter (`loc`)
3. Scale parameter (`scale`)

### Exceptions (matching SciPy):
- Normal: `loc` (mean), `scale` (std)
- Uniform: `loc` (lower), `scale` (upper - lower)
- Exponential: Uses `scale` parameter (1/rate), not `rate`

## Return Types

### Statistical tests should return structured results:
- All test functions return a result object with `statistic` and `pvalue` fields
- Additional fields as needed (e.g., `df` for t-tests)
- Remove functions that return tuples

## Implementation Priority

1. **Phase 1**: Add new aliases without breaking existing API
2. **Phase 2**: Deprecate old names with warnings
3. **Phase 3**: Remove deprecated functions in next major version