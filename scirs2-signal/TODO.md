# scirs2-signal TODO

This module provides signal processing functionality similar to SciPy's signal module.

## Current Status

- [x] Set up module structure
- [x] Error handling
- [x] Filtering
  - [x] FIR and IIR filters
  - [x] Filter design (Butterworth, Chebyshev, Bessel, etc.)
  - [x] Zero-phase filtering (filtfilt)
- [x] Convolution and correlation
  - [x] 1D convolution with different modes
  - [x] Cross-correlation
  - [x] Deconvolution
- [x] Spectral analysis
  - [x] Periodogram
  - [x] Type aliases for complex return types
- [x] Peak finding and analysis
  - [x] Peak detection with various criteria
  - [x] Peak properties (prominence, width)
- [x] Waveform generation
  - [x] Basic waveforms (sine, square, sawtooth)
  - [x] Specialized signals (chirp, Gaussian pulse)
- [x] Signal measurements
  - [x] RMS, SNR, THD
  - [x] Peak-to-peak and peak-to-RMS
- [x] Resampling
  - [x] Up/down sampling
  - [x] Arbitrary rate resampling
- [x] Fixed Clippy warnings and style issues
  - [x] Implemented FromStr trait for FilterType
  - [x] Replaced needless_range_loop with iterator patterns
  - [x] Fixed comparison_chain warnings

## Future Tasks

- [ ] Implement additional functionality
  - [ ] Implement remaining STFT and spectrogram functions
  - [ ] Complete Welch's method implementation
  - [ ] Add wavelet transforms
  - [ ] Implement filter bank design
- [ ] Add more filter types and design methods
  - [ ] Parks-McClellan optimal FIR filters
  - [ ] Savitzky-Golay filters
  - [ ] Adaptive filters
- [ ] Enhance spectral analysis
  - [ ] Multitaper methods
  - [ ] Higher-order spectral analysis
  - [ ] Time-frequency representations
- [ ] Add more examples and documentation
  - [ ] Tutorial for common signal processing tasks
  - [ ] Visual examples for different methods
- [ ] Performance optimization
  - [ ] Parallelization of computationally intensive operations
  - [ ] Memory optimization for large signals

## Long-term Goals

- [ ] Performance comparable to or better than SciPy's signal
- [ ] Integration with FFT and interpolation modules
- [ ] Real-time signal processing capabilities
- [ ] Support for specialized domains (audio, biomedical, communications)
- [ ] GPU-accelerated implementations for large datasets
- [ ] Advanced visualization tools