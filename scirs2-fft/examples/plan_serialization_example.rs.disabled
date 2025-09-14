use num_complex::Complex64;
use scirs2_fft::plan_serialization::{create_and_time_plan, PlanSerializationManager};
use std::time::{Duration, Instant};

#[allow(dead_code)]
fn main() {
    println!("FFT Plan Serialization Example");
    println!("------------------------------");

    // Create a temporary file for plan database
    let db_path = std::env::temp_dir().join("fft_plan_db.json");
    println!("Using plan database at: {db_path:?}");

    // Create plan serialization manager
    let manager = PlanSerializationManager::new(&db_path);

    // Detect and display architecture information
    let arch_id = PlanSerializationManager::detect_arch_id();
    println!("Detected architecture: {arch_id}");

    // Benchmark creation of FFT plans for various sizes
    benchmark_plan_creation(&manager);

    // Compare performance with and without plan serialization
    compare_performance(&manager);

    // Clean up
    println!("\nNote: The plan database remains at {db_path:?} for future use.");
    println!("You can remove it if no longer needed.");
}

#[allow(dead_code)]
fn benchmark_plan_creation(manager: &PlanSerializationManager) {
    println!("\nBenchmarking FFT plan creation for different sizes:");
    println!("{:>8} | {:>12} | {:>10}", "Size", "Time (us)", "Exists?");
    println!("---------------------------------");

    for size in [128, 256, 512, 1024, 2048, 4096, 8192].iter() {
        // Check if plan already exists
        let exists = manager.plan_exists(*size, true);

        // Time plan creation
        let start = Instant::now();
        let (_, creation_time_ns) = create_and_time_plan(*size, true);
        let total_time = start.elapsed();

        // Create plan info
        let plan_info = manager.create_plan_info(*size, true);

        // Record this plan usage
        if let Err(e) = manager.record_plan_usage(&plan_info, creation_time_ns) {
            eprintln!("Error recording plan usage: {e}");
        }

        println!(
            "{:>8} | {:>12.2} | {:>10}",
            size,
            total_time.as_micros(),
            if exists { "Yes" } else { "No" }
        );
    }

    // Save database for future runs
    if let Err(e) = manager.save_database() {
        eprintln!("Error saving plan database: {e}");
    }
}

#[allow(dead_code)]
fn compare_performance(manager: &PlanSerializationManager) {
    println!("\nComparing FFT performance with cached vs. non-cached plans:");

    // Create test data
    let size = 2048;
    let mut input = vec![Complex64::default(); size];
    let mut output = vec![Complex64::default(); size];

    for i in 0..size {
        input[i] = Complex64::new(i as f64, (i * 2) as f64);
    }

    // First run (non-cached plan, but will be cached for second run)
    println!("\nFirst run (plan will be created):");
    let plan_info = manager.create_plan_info(size, true);

    let start = Instant::now();
    let (plan, creation_time) = create_and_time_plan(size, true);

    // Process data with planned FFT
    let mut buffer = input.clone();
    plan.process(&mut buffer);
    output.copy_from_slice(&buffer);

    let total_time = start.elapsed();

    // Record plan usage
    _manager
        .record_plan_usage(&plan_info, creation_time)
        .unwrap();

    println!(
        "Plan creation time: {:?}",
        Duration::from_nanos(creation_time)
    );
    println!("Total execution time: {total_time:?}");

    // Second run (should use cached plan info)
    println!("\nSecond run (should use cached plan knowledge):");
    let start = Instant::now();
    let (plan, creation_time) = create_and_time_plan(size, true);

    // Process data with planned FFT
    let mut buffer = input.clone();
    plan.process(&mut buffer);
    output.copy_from_slice(&buffer);

    let total_time = start.elapsed();

    println!(
        "Plan creation time: {:?}",
        Duration::from_nanos(creation_time)
    );
    println!("Total execution time: {total_time:?}");

    // Save database for future runs
    if let Err(e) = manager.save_database() {
        eprintln!("Error saving plan database: {e}");
    }
}
