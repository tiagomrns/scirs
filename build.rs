// Build script for scirs workspace
// Optimizes build for CI environments with limited disk space

use std::env;

#[allow(dead_code)]
fn main() {
    // Check if we're in a CI environment
    let is_ci = env::var("CI").is_ok() || env::var("GITHUB_ACTIONS").is_ok();
    
    if is_ci {
        println!("cargo:rustc-env=SCIRS_CI_BUILD=1");
        
        // Enable optimizations for CI builds
        println!("cargo:rustc-link-arg=-Wl,--compress-debug-sections=zlib");
        println!("cargo:rustc-link-arg=-Wl,--gc-sections");
        
        // Reduce debug info in CI
        if env::var("CARGO_CFG_DEBUG_ASSERTIONS").is_ok() {
            println!("cargo:rustc-env=CARGO_PROFILE_DEV_DEBUG=1");
        }
    }
    
    // Platform-specific BLAS/LAPACK configuration
    #[cfg(target_os = "macos")]
    {
        env::set_var("DEP_LAPACK_SRC", "accelerate");
        env::set_var("LAPACK_SRC", "accelerate");
        env::set_var("BLAS_SRC", "accelerate");
    }
    
    #[cfg(target_os = "linux")]
    {
        env::set_var("DEP_LAPACK_SRC", "openblas");
        env::set_var("LAPACK_SRC", "openblas");
        env::set_var("BLAS_SRC", "openblas");
    }
    
    #[cfg(target_os = "windows")]
    {
        env::set_var("DEP_LAPACK_SRC", "intel-mkl");
        env::set_var("LAPACK_SRC", "intel-mkl");
        env::set_var("BLAS_SRC", "intel-mkl");
    }
    
    #[cfg(target_os = "freebsd")]
    {
        env::set_var("DEP_LAPACK_SRC", "openblas");
        env::set_var("LAPACK_SRC", "openblas");
        env::set_var("BLAS_SRC", "openblas");
    }
    
    #[cfg(target_os = "netbsd")]
    {
        env::set_var("DEP_LAPACK_SRC", "netlib");
        env::set_var("LAPACK_SRC", "netlib");
        env::set_var("BLAS_SRC", "netlib");
    }
    
    #[cfg(target_os = "openbsd")]
    {
        env::set_var("DEP_LAPACK_SRC", "openblas");
        env::set_var("LAPACK_SRC", "openblas");
        env::set_var("BLAS_SRC", "openblas");
    }
    
    // Only rebuild if build.rs changes
    println!("cargo:rerun-if-changed=build.rs");
}
