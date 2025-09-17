#!/bin/bash
# Setup script for LAPACK linking on macOS ARM64
# Run with: source setup_lapack.sh

export RUSTFLAGS="-L /opt/homebrew/opt/lapack/lib -L /opt/homebrew/opt/openblas/lib -l lapack -l blas"
export DEP_LAPACK_SRC="accelerate"

echo "✅ LAPACK environment variables set successfully"
echo "📦 RUSTFLAGS: $RUSTFLAGS"
echo "🔧 DEP_LAPACK_SRC: $DEP_LAPACK_SRC"
echo ""
echo "Now you can run: cargo test, cargo build, etc."