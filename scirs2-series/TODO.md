# scirs2-series TODO

This module provides time series analysis functionality similar to the time series components in pandas and statsmodels.

## Current Status

- [x] Set up module structure
- [x] Error handling
- [x] Decomposition module
  - [x] Classical decomposition (additive and multiplicative)
  - [x] STL decomposition
  - [x] Seasonal decomposition using moving averages
  - [x] Trend extraction

- [x] Forecasting module
  - [x] Exponential smoothing models
  - [x] ARIMA/SARIMA models
  - [x] Simple forecasting methods (naive, mean, drift)
  - [x] Prophet-like API for complex time series

- [x] Features module
  - [x] Time series feature extraction
  - [x] Autocorrelation features
  - [x] Statistical features
  - [x] Frequency domain features

- [x] Utilities
  - [x] Resampling and frequency conversion
  - [x] Missing value interpolation
  - [x] Outlier detection
  - [x] Date manipulation helpers

## Future Tasks

- [ ] Enhance decomposition methods
  - [ ] Robust decomposition methods
  - [ ] SSA (Singular Spectrum Analysis)
  - [ ] TBATS decomposition

- [ ] Expand forecasting capabilities
  - [ ] Neural network based forecasting
  - [ ] VAR/VARIMA models
  - [ ] Multivariate forecasting
  - [ ] Uncertainty estimation and prediction intervals

- [ ] Add advanced time series features
  - [ ] Advanced seasonality detection
  - [ ] Change point detection
  - [ ] Anomaly detection
  - [ ] Granger causality testing

- [ ] Improve performance
  - [ ] Parallel implementation of computationally intensive operations
  - [ ] Optimize algorithms for large time series
  - [ ] Implement streaming support for online processing

## Testing and Quality Assurance

- [x] Unit tests for basic functionality
- [ ] Comprehensive test coverage
- [ ] Benchmarks against pandas/statsmodels implementations
- [ ] Edge case handling (missing data, irregular time series)

## Documentation

- [x] Basic API documentation
- [ ] Tutorial notebooks demonstrating time series analysis workflow
- [ ] Detailed examples for different domains (finance, IoT, etc.)
- [ ] Performance comparison with Python equivalents