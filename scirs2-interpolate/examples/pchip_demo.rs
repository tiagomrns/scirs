use ndarray::{array, Array1};
use scirs2_interpolate::{
    cubic_interpolate, linear_interpolate, pchip_interpolate, PchipInterpolator,
};

fn main() {
    println!("PCHIP Interpolation Demonstration");
    println!("================================");

    // Example 1: Basic usage
    basic_pchip_example();

    // Example 2: Shape preservation with different interpolation methods
    shape_preservation_example();

    // Example 3: Monotonicity preservation
    monotonicity_preservation_example();
}

fn basic_pchip_example() {
    println!("\n1. Basic PCHIP Usage");
    println!("-------------------");

    // Create sample data
    let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
    let y = array![0.0, 1.0, 4.0, 9.0, 16.0]; // y = x²

    // Create PCHIP interpolator
    let interp = PchipInterpolator::new(&x.view(), &y.view(), false).unwrap();

    // Evaluate at specific points
    let x_eval = array![0.5, 1.5, 2.5, 3.5];
    let mut y_pchip = Array1::zeros(x_eval.len());

    for (i, &x_val) in x_eval.iter().enumerate() {
        y_pchip[i] = interp.evaluate(x_val).unwrap();
    }

    // Print results
    println!("Sample points: [0, 1, 2, 3, 4] → [0, 1, 4, 9, 16]  (y = x²)");
    println!("\nPCHIP Interpolation Results:");
    println!("  x   |  PCHIP  | Actual (x²)");
    println!("------|---------|------------");
    for i in 0..x_eval.len() {
        println!(
            " {:.1} |  {:.3}  |    {:.1}",
            x_eval[i],
            y_pchip[i],
            x_eval[i] * x_eval[i]
        );
    }

    // Direct function usage
    println!("\nUsing the pchip_interpolate function:");
    let y_func = pchip_interpolate(&x.view(), &y.view(), &x_eval.view(), false).unwrap();
    for i in 0..x_eval.len() {
        println!("  At x = {:.1}, y = {:.3}", x_eval[i], y_func[i]);
    }
}

fn shape_preservation_example() {
    println!("\n2. Shape Preservation Comparison");
    println!("-------------------------------");

    // Create sample data with a rapid change
    let x = array![0.0, 2.0, 3.0, 5.0, 6.0, 8.0, 9.0, 11.0, 12.0];
    let y = array![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];

    // Points to evaluate
    let mut x_eval = Array1::zeros(101);
    for i in 0..101 {
        x_eval[i] = 0.12 * i as f64;
    }

    // Interpolate using different methods
    let y_linear = linear_interpolate(&x.view(), &y.view(), &x_eval.view()).unwrap();
    let y_cubic = cubic_interpolate(&x.view(), &y.view(), &x_eval.view()).unwrap();
    let y_pchip = pchip_interpolate(&x.view(), &y.view(), &x_eval.view(), false).unwrap();

    // Print sample values
    println!("Original data points:");
    for i in 0..x.len() {
        println!("  ({:.1}, {:.1})", x[i], y[i]);
    }

    println!("\nInterpolation comparison at selected points:");
    println!("  x   | Linear |  Cubic  |  PCHIP  ");
    println!("------|--------|---------|--------");

    // Choose a few interesting points to display
    let indices = [20, 30, 40, 50, 60, 70, 80];
    for &i in indices.iter() {
        println!(
            " {:.1} |  {:.3} |  {:.3} |  {:.3} ",
            x_eval[i], y_linear[i], y_cubic[i], y_pchip[i]
        );
    }

    // Analyze overshoot
    let max_linear = y_linear.iter().fold(0.0f64, |a, &b| f64::max(a, b));
    let min_linear = y_linear.iter().fold(1.0f64, |a, &b| f64::min(a, b));

    let max_cubic = y_cubic.iter().fold(0.0f64, |a, &b| f64::max(a, b));
    let min_cubic = y_cubic.iter().fold(1.0f64, |a, &b| f64::min(a, b));

    let max_pchip = y_pchip.iter().fold(0.0f64, |a, &b| f64::max(a, b));
    let min_pchip = y_pchip.iter().fold(1.0f64, |a, &b| f64::min(a, b));

    println!("\nOvershooting analysis:");
    println!("  Method | Min Value | Max Value | Overshoots?");
    println!("---------|-----------|-----------|------------");
    println!(
        "  Linear |    {:.3}   |    {:.3}   | {}",
        min_linear,
        max_linear,
        (min_linear < 0.0 || max_linear > 1.0)
    );
    println!(
        "   Cubic |    {:.3}   |    {:.3}   | {}",
        min_cubic,
        max_cubic,
        (min_cubic < 0.0 || max_cubic > 1.0)
    );
    println!(
        "   PCHIP |    {:.3}   |    {:.3}   | {}",
        min_pchip,
        max_pchip,
        (min_pchip < 0.0 || max_pchip > 1.0)
    );

    println!("\nNote: PCHIP is designed to preserve shape and avoid overshooting the data,");
    println!("      which is especially useful for physical data where negative values or");
    println!("      oscillations might be physically unrealistic.");
}

fn monotonicity_preservation_example() {
    println!("\n3. Monotonicity Preservation");
    println!("--------------------------");

    // Create monotonically increasing data
    let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y = array![0.0, 0.5, 1.0, 2.0, 4.5, 8.0];

    // Points to evaluate
    let mut x_eval = Array1::zeros(51);
    for i in 0..51 {
        x_eval[i] = 0.1 * i as f64;
    }

    // Interpolate using different methods
    let y_cubic = cubic_interpolate(&x.view(), &y.view(), &x_eval.view()).unwrap();
    let y_pchip = pchip_interpolate(&x.view(), &y.view(), &x_eval.view(), false).unwrap();

    // Check for monotonicity violations
    let mut cubic_violations = 0;
    let mut pchip_violations = 0;

    for i in 1..y_cubic.len() {
        if (y_cubic[i] - y_cubic[i - 1]) < 0.0 {
            cubic_violations += 1;
        }

        if (y_pchip[i] - y_pchip[i - 1]) < 0.0 {
            pchip_violations += 1;
        }
    }

    println!("Original monotonically increasing data points:");
    for i in 0..x.len() {
        println!("  ({:.1}, {:.1})", x[i], y[i]);
    }

    println!(
        "\nMonotonicity analysis on {} interpolated points:",
        x_eval.len()
    );
    println!("  Method | Monotonicity Violations | Preserves Monotonicity?");
    println!("---------|--------------------------|-------------------------");
    println!(
        "   Cubic | {} violations            | {}",
        cubic_violations,
        if cubic_violations == 0 { "Yes" } else { "No" }
    );
    println!(
        "   PCHIP | {} violations            | {}",
        pchip_violations,
        if pchip_violations == 0 { "Yes" } else { "No" }
    );

    println!("\nNote: PCHIP is specifically designed to preserve monotonicity in the data,");
    println!("      which is important for many physical systems and when interpolating");
    println!("      cumulative distribution functions or similar monotonic relationships.");
}
