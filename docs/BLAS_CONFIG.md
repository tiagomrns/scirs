# BLAS Backend Configuration

This document explains how to configure BLAS (Basic Linear Algebra Subprograms) backends in SciRS2 to resolve OpenBLAS linking issues on macOS.

## Problem Solved

The previous configuration caused OpenBLAS linking errors on macOS:
```
ld: library 'openblas' not found
ld: warning: search path '/usr/lib/x86_64-linux-gnu' not found
```

**Solution**: SciRS2 now uses platform-appropriate defaults and doesn't force OpenBLAS on macOS.

## Platform Behavior

**Default behavior** (recommended):
- **macOS**: Uses system Accelerate framework (no OpenBLAS dependencies)
- **Linux**: Platform will determine appropriate BLAS (typically system BLAS)
- **Windows**: Platform will determine appropriate BLAS

## Override Backend When Needed

You can force a specific backend by using features:

```bash
# Force use of macOS Accelerate framework (macOS only)
cargo build --features accelerate

# Force use of OpenBLAS (Linux/Windows)
cargo build --features openblas

# Force use of Intel MKL (if available)
cargo build --features intel-mkl

# Force use of reference Netlib
cargo build --features netlib
```

## Usage Examples

### Default (Recommended)
```bash
# Uses platform-appropriate backend automatically
cargo build
cargo test
```

### macOS with Explicit Accelerate
```bash
# Explicitly use Accelerate framework on macOS
cargo build --features accelerate
```

### Linux with Intel MKL
```bash
# Use Intel MKL instead of default OpenBLAS on Linux
cargo build --features intel-mkl
```

## Troubleshooting

### macOS OpenBLAS Linking Issues - FIXED

The previous OpenBLAS linking errors on macOS are now resolved:
- ✅ No more `ld: library 'openblas' not found`
- ✅ No more Linux-specific path warnings
- ✅ Uses system Accelerate framework by default

**If you still encounter issues**:

1. **Clean and rebuild** (recommended):
   ```bash
   cargo clean
   cargo build
   ```

2. **Verify no conflicting features**:
   Make sure you're not accidentally using `--features openblas` on macOS

3. **Explicitly use Accelerate** (if needed):
   ```bash
   cargo build --features accelerate
   ```

### Performance Considerations

- **macOS**: Accelerate framework is highly optimized for Apple hardware
- **Linux**: OpenBLAS provides good general performance; Intel MKL may be faster on Intel CPUs
- **Windows**: OpenBLAS is the most compatible option

### Dependencies Required

**Default configuration** (no additional dependencies required):
- **macOS**: Uses system Accelerate framework (no external BLAS needed)
- **Linux**: Uses system BLAS libraries or ndarray-linalg defaults
- **Windows**: Uses ndarray-linalg defaults

**When using specific backends**:
- **`--features openblas`**: Includes OpenBLAS source (compiled automatically)
- **`--features intel-mkl`**: Requires Intel MKL installation
- **`--features netlib`**: Uses reference BLAS implementation

## Backend-Specific Notes

### Accelerate Framework (macOS)
- System-provided, no external dependencies
- Highly optimized for Apple Silicon and Intel Macs
- Supports both single and double precision
- Unified memory on Apple Silicon provides additional benefits

### OpenBLAS
- Cross-platform, open source
- Good performance on various architectures
- Requires compilation or system package installation
- Multithreaded by default

### Intel MKL
- Highest performance on Intel CPUs
- Commercial license (free for some uses)
- Requires Intel MKL installation
- Supports advanced features like sparse BLAS

### Netlib Reference
- Reference implementation
- Portable but not optimized
- Mainly useful for testing and debugging
- Single-threaded