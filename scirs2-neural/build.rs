#[allow(dead_code)]
fn main() {
    // Link pthread library for OpenBLAS static linking
    // This must be BEFORE openblas is linked
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=gfortran");
        println!("cargo:rustc-link-lib=gomp");
    }

    // Force static linking of pthread
    // println!("cargo:rustc-link-arg=-Wl,--start-group");
    // Ensure proper link order - use architecture-agnostic paths
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    println!("cargo:rustc-link-search=native=/lib/x86_64-linux-gnu");
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    println!("cargo:rustc-link-search=native=/usr/lib/aarch64-linux-gnu");
    println!("cargo:rustc-link-search=native=/lib/aarch64-linux-gnu");
    // Add end-group to balance start-group
    // println!("cargo:rustc-link-arg=-Wl,--end-group");
}
