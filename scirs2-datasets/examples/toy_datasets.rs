use scirs2_datasets::{load_boston, load_iris};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let iris = load_iris()?;
    println!("Iris dataset loaded:");
    println!("  Samples: {}", iris.n_samples());
    println!("  Features: {}", iris.n_features());
    println!(
        "  Target classes: {}",
        iris.target_names.as_ref().map_or(0, |v| v.len())
    );

    let boston = load_boston()?;
    println!("\nBoston Housing dataset loaded:");
    println!("  Samples: {}", boston.n_samples());
    println!("  Features: {}", boston.n_features());

    Ok(())
}
