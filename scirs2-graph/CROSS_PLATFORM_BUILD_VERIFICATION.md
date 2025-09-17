# Cross-Platform Build Verification for scirs2-graph

## Overview

This document outlines the cross-platform build verification strategy for scirs2-graph to ensure the library works correctly across different operating systems, architectures, and Rust toolchains.

## Supported Platforms

### Tier 1 Platforms (Full Support)
- **Linux x86_64** (Ubuntu 20.04+, RHEL 8+, Debian 11+)
- **macOS x86_64** (macOS 10.15+)
- **Windows x86_64** (Windows 10+)

### Tier 2 Platforms (Best Effort)
- **Linux ARM64** (aarch64)
- **macOS ARM64** (Apple Silicon)
- **Linux i686** (32-bit Intel)
- **Windows i686** (32-bit Windows)

### Tier 3 Platforms (Community Support)
- **FreeBSD x86_64**
- **OpenBSD x86_64**
- **NetBSD x86_64**

## Build Verification Matrix

### Rust Toolchain Versions
- **Stable**: Latest stable Rust release
- **Beta**: Latest beta Rust release
- **MSRV**: Minimum Supported Rust Version (1.70.0)

### Build Configurations
- **Debug**: Development builds with debug symbols
- **Release**: Optimized production builds
- **Feature Combinations**:
  - Default features
  - All features enabled
  - Minimal features (core only)
  - SIMD features
  - Parallel features
  - GPU features (where supported)

## Platform-Specific Considerations

### Linux
- **Dependencies**: OpenBLAS, LAPACK, Intel MKL (optional)
- **Package Managers**: apt, yum, dnf, pacman
- **Architectures**: x86_64, aarch64, i686
- **Distributions**: Ubuntu, RHEL, CentOS, Debian, Arch, openSUSE

### macOS
- **Dependencies**: Accelerate framework, Homebrew packages
- **Architectures**: x86_64 (Intel), aarch64 (Apple Silicon)
- **Xcode**: Latest stable Xcode version
- **Homebrew**: Latest packages

### Windows
- **Dependencies**: Microsoft Visual C++ Build Tools
- **Architectures**: x86_64, i686
- **MSVC**: Visual Studio 2019/2022
- **Package Managers**: vcpkg, Chocolatey

## Verification Checklist

### Basic Build Verification
- [ ] `cargo build` succeeds
- [ ] `cargo build --release` succeeds
- [ ] `cargo test` passes
- [ ] `cargo bench` runs without errors
- [ ] `cargo doc` generates documentation

### Feature Verification
- [ ] Default features build and test
- [ ] `--no-default-features` builds
- [ ] `--all-features` builds and tests
- [ ] Individual feature flags work correctly

### Performance Verification
- [ ] SIMD instructions are available and used
- [ ] Parallel processing works correctly
- [ ] Memory usage is reasonable
- [ ] Performance meets baseline requirements

### Integration Verification
- [ ] Examples compile and run
- [ ] Benchmarks execute successfully
- [ ] C FFI bindings work (if applicable)
- [ ] Python bindings work (if applicable)

## Automated Testing Infrastructure

### GitHub Actions Workflow

```yaml
name: Cross-Platform Build Verification

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  build-linux:
    name: Linux
    runs-on: ubuntu-latest
    strategy:
      matrix:
        rust: [stable, beta, 1.70.0]
        features: [default, all, minimal]
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: ${{ matrix.rust }}
          override: true
      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y libopenblas-dev liblapack-dev
      - name: Build
        run: cargo build --verbose
      - name: Test
        run: cargo test --verbose
      - name: Benchmark
        run: cargo bench --verbose

  build-macos:
    name: macOS
    runs-on: macos-latest
    strategy:
      matrix:
        rust: [stable, beta]
        target: [x86_64-apple-darwin, aarch64-apple-darwin]
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: ${{ matrix.rust }}
          target: ${{ matrix.target }}
          override: true
      - name: Install dependencies
        run: brew install openblas lapack
      - name: Build
        run: cargo build --target ${{ matrix.target }} --verbose
      - name: Test
        if: matrix.target != 'aarch64-apple-darwin'
        run: cargo test --target ${{ matrix.target }} --verbose

  build-windows:
    name: Windows
    runs-on: windows-latest
    strategy:
      matrix:
        rust: [stable, beta]
        target: [x86_64-pc-windows-msvc, i686-pc-windows-msvc]
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: ${{ matrix.rust }}
          target: ${{ matrix.target }}
          override: true
      - name: Build
        run: cargo build --target ${{ matrix.target }} --verbose
      - name: Test
        run: cargo test --target ${{ matrix.target }} --verbose

  build-arm:
    name: ARM64
    runs-on: ubuntu-latest
    strategy:
      matrix:
        target: [aarch64-unknown-linux-gnu]
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          target: ${{ matrix.target }}
          override: true
      - name: Install cross-compilation tools
        run: |
          sudo apt-get update
          sudo apt-get install -y gcc-aarch64-linux-gnu
      - name: Build
        run: cargo build --target ${{ matrix.target }} --verbose
        env:
          CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER: aarch64-linux-gnu-gcc
```

### Manual Testing Procedures

#### Linux Testing
```bash
# Install dependencies
sudo apt-get install libopenblas-dev liblapack-dev

# Test different feature combinations
cargo test --no-default-features
cargo test --features="parallel"
cargo test --features="simd"
cargo test --all-features

# Performance testing
cargo bench --bench graph_benchmarks
cargo bench --bench large_graph_stress

# Memory testing
valgrind --tool=memcheck cargo test --release
```

#### macOS Testing
```bash
# Install dependencies
brew install openblas lapack

# Test on Intel Macs
cargo test --target x86_64-apple-darwin

# Test on Apple Silicon (if available)
cargo test --target aarch64-apple-darwin

# Performance testing with Accelerate framework
export CARGO_TARGET_X86_64_APPLE_DARWIN_RUSTFLAGS="-C target-feature=+avx2"
cargo bench --release
```

#### Windows Testing
```powershell
# Install Visual Studio Build Tools
# Test MSVC toolchain
cargo test --target x86_64-pc-windows-msvc

# Test different architectures
cargo test --target i686-pc-windows-msvc

# Performance testing
cargo bench --release
```

## Platform-Specific Issues and Solutions

### Linux Issues
1. **OpenBLAS conflicts**: Use system package manager versions
2. **GLIBC compatibility**: Target appropriate minimum versions
3. **Missing dependencies**: Document required system packages

### macOS Issues
1. **Apple Silicon compatibility**: Ensure universal binaries
2. **Accelerate framework**: Proper linking and feature detection
3. **macOS version compatibility**: Test on supported versions

### Windows Issues
1. **MSVC runtime**: Ensure correct Visual C++ redistributables
2. **Path length limits**: Handle long path names correctly
3. **DLL dependencies**: Bundle or document required libraries

## Performance Baseline Requirements

### Minimum Performance Standards
- **Build time**: < 5 minutes on standard CI machines
- **Test execution**: < 10 minutes for full test suite
- **Memory usage**: < 2GB for standard test suite
- **Binary size**: < 50MB for release builds

### Performance Regression Detection
- Compare against previous release benchmarks
- Track performance metrics over time
- Alert on >20% performance degradation

## Quality Assurance Checklist

### Pre-Release Verification
- [ ] All Tier 1 platforms build successfully
- [ ] Core functionality tested on all platforms
- [ ] Performance benchmarks pass
- [ ] Memory usage within limits
- [ ] Documentation builds correctly
- [ ] Examples work on all platforms

### Release Verification
- [ ] Release artifacts created for all platforms
- [ ] Installation packages tested
- [ ] Distribution packages validated
- [ ] Performance metrics documented
- [ ] Known issues documented

## Troubleshooting Common Issues

### Build Failures
1. **Missing dependencies**: Install required system libraries
2. **Toolchain issues**: Update Rust toolchain
3. **Feature conflicts**: Check feature flag combinations
4. **Linker errors**: Verify system linker configuration

### Test Failures
1. **Numerical precision**: Platform-specific floating point behavior
2. **Timing issues**: Adjust test timeouts for slower systems
3. **Resource limits**: Check system memory and disk space
4. **Parallel execution**: Verify thread safety on all platforms

### Performance Issues
1. **SIMD not working**: Check CPU feature detection
2. **Parallel speedup poor**: Verify thread scheduling
3. **Memory usage high**: Check for platform-specific allocator issues
4. **Binary size large**: Optimize compilation flags

## Future Platform Support

### Planned Additions
- **WebAssembly** (WASM): For browser compatibility
- **Android**: For mobile applications
- **iOS**: For mobile applications
- **RISC-V**: For emerging architectures

### Evaluation Criteria
- Community demand
- Maintenance burden
- Technical feasibility
- Resource availability

## Documentation Requirements

### Platform-Specific Documentation
- Installation instructions for each platform
- Platform-specific dependencies
- Known limitations and workarounds
- Performance characteristics by platform

### User Guidance
- Platform selection recommendations
- Performance optimization tips
- Troubleshooting common issues
- Migration between platforms

## Conclusion

Cross-platform build verification is essential for the reliability and adoption of scirs2-graph. This comprehensive strategy ensures that the library works correctly across diverse environments while maintaining high performance and quality standards.

The verification process includes automated testing, manual validation, performance monitoring, and continuous quality assurance to deliver a robust cross-platform graph processing library.