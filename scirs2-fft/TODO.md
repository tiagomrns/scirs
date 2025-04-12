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

## Long-term Goals

- [ ] Performance comparable to or better than FFTW
- [ ] GPU-accelerated implementations
- [ ] Support for distributed FFT computations
- [ ] Integration with signal processing and filtering
- [ ] Advanced time-frequency analysis tools
- [ ] Support for specialized hardware (FPGA, custom accelerators)