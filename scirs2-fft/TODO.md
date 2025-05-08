# scirs2-fft TODO

This module provides Fast Fourier Transform functionality similar to SciPy's fft module.

## Current Status

- [x] Set up module structure
- [x] Error handling
- [x] FFT and inverse FFT (1D, 2D, and N-dimensional)
- [x] Real FFT and inverse Real FFT (optimized for real input)
- [x] Discrete Cosine Transform (DCT) types I-IV
- [x] Discrete Sine Transform (DST) types I-IV
- [x] Helper functions (fftshift, ifftshift, fftfreq, rfftfreq)
- [x] Window functions (Hann, Hamming, Blackman, etc.)
- [x] Integration with ndarray for multi-dimensional arrays

## Future Tasks

- [x] Fix remaining Clippy warnings
  - [x] Address needless_range_loop warnings
  - [x] Fix comparison_chain warnings
  - [x] Address only_used_in_recursion warnings
  - [x] Fix doc_overindented_list_items warning
- [x] Performance optimizations
  - [x] Parallelization of larger transforms
  - [x] More efficient memory usage for large arrays
    - [x] Implemented memory-efficient 2D FFT
    - [x] Added streaming FFT for processing large arrays in chunks
    - [x] Created in-place FFT operations to minimize allocations
- [x] Add more examples and documentation
  - [x] Tutorial for common FFT operations (fft_tutorial.rs)
  - [x] Examples for spectral analysis (spectral_analysis.rs)
  - [x] Memory-efficient FFT examples (memory_efficient_fft.rs)
- [x] Additional functionality
  - [x] Short-time Fourier transform (STFT) interface
  - [x] Non-uniform FFT
  - [x] Hilbert transform
  - [x] Fractional Fourier transform (implementation with complex number handling needs improvement)
- [x] Add visualization utilities
  - [x] Spectrograms
  - [x] Waterfall plots

## Enhanced FFT API and Interoperability

- [ ] Implement array interoperability features
  - [ ] Support for various array-like objects
  - [ ] Backend system similar to SciPy's backend model
  - [ ] Pluggable FFT implementations
- [ ] Enhance worker management for parallelization
  - [ ] Thread pool configuration
  - [ ] Worker count control similar to SciPy's `set_workers`/`get_workers`
  - [ ] Thread safety guarantees for all operations
- [ ] Add context managers for FFT settings
  - [ ] Backend selection context
  - [ ] Worker count context
  - [ ] Plan caching control

## Fast Hankel Transform

- [ ] Implement Fast Hankel Transform (FHT)
  - [ ] Forward transform (fht)
  - [ ] Inverse transform (ifht)
  - [ ] Optimal offset calculation (fhtoffset)
  - [ ] Support for biased transforms
  - [ ] Comprehensive examples with visualizations

## Multidimensional Transform Enhancements

- [ ] Improve N-dimensional transforms
  - [ ] Optimized memory access patterns
  - [ ] Advanced chunking strategies for large arrays
  - [ ] Axis-specific operations with optional normalization
  - [ ] Advanced striding support

## Plan Caching and Optimization

- [ ] Implement advanced planning strategies
  - [ ] Plan caching mechanism for repeated transforms
  - [ ] Auto-tuning for hardware-specific optimizations
  - [ ] Plan serialization for reuse across runs
  - [ ] Plan sharing across threads
- [ ] Add `next_fast_len` and `prev_fast_len` helpers
  - [ ] Optimal sizing for FFT speed
  - [ ] Support for SIMD-friendly sizes
  - [ ] Automatic padding strategies

## Extended Transform Types

- [ ] Implement additional transform variants
  - [ ] Higher-order DCT types (V-VIII)
  - [ ] Higher-order DST types (V-VIII)
  - [ ] Hartley transform
  - [ ] Modified DCT/DST (MDCT/MDST)
  - [ ] Z-transform for non-uniform frequency spacing

## Custom Window Functions

- [ ] Extend window function support
  - [ ] Comprehensive window catalog matching SciPy
  - [ ] Window design tools and generators
  - [ ] Window visualization utilities
  - [ ] Window properties analysis (energy, bandwidth)

## Long-term Goals

- [ ] Performance comparable to or better than FFTW
  - [ ] Benchmark suite for comparison
  - [ ] Performance optimization database
  - [ ] Auto-tuning for specific hardware
- [ ] GPU-accelerated implementations
  - [ ] CUDA/HIP/SYCL support
  - [ ] Memory management for large transforms
  - [ ] Hybrid CPU/GPU execution strategies
- [ ] Support for distributed FFT computations
  - [ ] MPI-based distributed transforms
  - [ ] Domain decomposition strategies
  - [ ] Network efficiency optimizations
- [ ] Integration with signal processing and filtering
  - [ ] Filter design and application in frequency domain
  - [ ] Convolution optimizations
  - [ ] Signal analysis toolkit
- [ ] Advanced time-frequency analysis tools
  - [ ] Enhanced spectrogram tools
  - [ ] Wavelet transform integration
  - [ ] Reassignment methods
  - [ ] Synchrosqueezing transforms
- [ ] Support for specialized hardware (FPGA, custom accelerators)
  - [ ] Hardware-specific optimizations
  - [ ] Offloading strategies
  - [ ] Custom kernels for different architectures