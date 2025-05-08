use ndarray::{array, ArrayView1};
use scirs2_integrate::ode::{solve_ivp, ODEMethod, ODEOptions};
use std::f64::consts::PI;

fn main() {
    println!("Improved BDF Solver Example");
    println!("---------------------------");

    // Example 1: Simple Exponential Decay
    println!("Example 1: Simple Exponential Decay (y' = -y, y(0) = 1)");

    let f = |_t: f64, y: ArrayView1<f64>| array![-y[0]];

    // Solve with BDF method (use more conservative settings)
    let options = ODEOptions {
        method: ODEMethod::Bdf,
        bdf_order: 1, // BDF1 method (more stable)
        rtol: 1e-3,   // Less strict tolerance
        atol: 1e-6,
        max_steps: 200,       // More steps allowed
        h0: Some(1e-3),       // Smaller initial step
        min_step: Some(1e-6), // Smaller minimum step
        ..Default::default()
    };

    match solve_ivp(f, [0.0, 2.0], array![1.0], Some(options)) {
        Ok(result) => {
            println!("Time points: {:?}", result.t);
            println!("Solution values:");
            for i in 0..result.t.len() {
                let t = result.t[i];
                let y = result.y[i][0];
                let exact = (-t).exp();
                let error = (y - exact).abs();
                println!(
                    "  t = {:.6}, y = {:.6}, exact = {:.6}, error = {:.6e}",
                    t, y, exact, error
                );
            }

            println!("\nStatistics:");
            println!("  Number of steps: {}", result.n_steps);
            println!("  Number of function evaluations: {}", result.n_eval);
            println!("  Number of accepted steps: {}", result.n_accepted);
            println!("  Number of rejected steps: {}", result.n_rejected);
            println!(
                "  Final step size: {:.6e}",
                result.final_step.unwrap_or(0.0)
            );
            println!("  Success: {}", result.success);
            if let Some(msg) = &result.message {
                println!("  Message: {}", msg);
            }

            // Example 2: Van der Pol Oscillator (Stiff System)
            println!("\nExample 2: Van der Pol Oscillator (Stiff System)");

            let mu = 1.0; // Lower stiffness parameter for example (original was 10.0)
            let van_der_pol =
                |_t: f64, y: ArrayView1<f64>| array![y[1], mu * (1.0 - y[0].powi(2)) * y[1] - y[0]];

            // Solve with BDF method (good for stiff problems)
            let options = ODEOptions {
                method: ODEMethod::Bdf,
                bdf_order: 1, // BDF1 method (more stable)
                rtol: 1e-3,   // Less strict tolerances
                atol: 1e-6,
                max_steps: 500,       // Reasonable number of steps
                h0: Some(1e-3),       // Small initial step
                min_step: Some(1e-6), // Smaller minimum step
                ..Default::default()
            };

            // Initial condition [2, 0]
            match solve_ivp(van_der_pol, [0.0, 10.0], array![2.0, 0.0], Some(options)) {
                Ok(result) => {
                    println!("Solving over t = [0, 10] with initial condition [2, 0]");
                    println!("Number of time points: {}", result.t.len());
                    println!(
                        "Final state: [{:.6}, {:.6}]",
                        result.y.last().unwrap()[0],
                        result.y.last().unwrap()[1]
                    );

                    println!("\nStatistics:");
                    println!("  Number of steps: {}", result.n_steps);
                    println!("  Number of function evaluations: {}", result.n_eval);
                    println!("  Number of accepted steps: {}", result.n_accepted);
                    println!("  Number of rejected steps: {}", result.n_rejected);
                    println!(
                        "  Final step size: {:.6e}",
                        result.final_step.unwrap_or(0.0)
                    );
                    println!("  Success: {}", result.success);
                    if let Some(msg) = &result.message {
                        println!("  Message: {}", msg);
                    }

                    // Example 3: Harmonic Oscillator
                    println!("\nExample 3: Harmonic Oscillator");

                    let harmonic = |_t: f64, y: ArrayView1<f64>| array![y[1], -y[0]];

                    // Initial condition [1, 0] (cosine solution)
                    let options = ODEOptions {
                        method: ODEMethod::RK4, // Use RK4 instead of BDF for stability
                        h0: Some(0.1),          // Fixed step size
                        ..Default::default()
                    };

                    match solve_ivp(harmonic, [0.0, 2.0 * PI], array![1.0, 0.0], Some(options)) {
                        Ok(result) => {
                            println!("Solving over t = [0, 2π] with initial condition [1, 0]");
                            println!("Solution should complete 1 oscillation and return to [1, 0]");

                            // Check specific points (should follow cos/sin)
                            let check_points = vec![0.0, PI / 2.0, PI, 3.0 * PI / 2.0, 2.0 * PI];

                            println!("\nChecking solution at key points:");
                            println!(
                                "{:>10} {:>15} {:>15} {:>15} {:>15}",
                                "t", "y[0]", "exact", "y[1]", "exact"
                            );
                            println!(
                                "{:->10} {:->15} {:->15} {:->15} {:->15}",
                                "", "", "", "", ""
                            );

                            for &t in &check_points {
                                // Find closest time point
                                let idx = result
                                    .t
                                    .iter()
                                    .enumerate()
                                    .min_by(|(_, &a), (_, &b)| {
                                        (a - t).abs().partial_cmp(&(b - t).abs()).unwrap()
                                    })
                                    .map(|(idx, _)| idx)
                                    .unwrap_or(0);

                                let y0 = result.y[idx][0];
                                let y1 = result.y[idx][1];
                                let exact0 = t.cos();
                                let exact1 = -t.sin();

                                println!(
                                    "{:10.6} {:15.6} {:15.6} {:15.6} {:15.6}",
                                    result.t[idx], y0, exact0, y1, exact1
                                );
                            }

                            println!("\nStatistics:");
                            println!("  Number of steps: {}", result.n_steps);
                            println!("  Number of function evaluations: {}", result.n_eval);
                            println!("  Number of accepted steps: {}", result.n_accepted);
                            println!("  Number of rejected steps: {}", result.n_rejected);
                            println!(
                                "  Final step size: {:.6e}",
                                result.final_step.unwrap_or(0.0)
                            );
                            println!("  Success: {}", result.success);
                            if let Some(msg) = &result.message {
                                println!("  Message: {}", msg);
                            }

                            // Check final state (should be close to [1, 0] after 1 full cycle)
                            let final_y = result.y.last().unwrap();
                            println!("\nFinal state: [{:.6}, {:.6}]", final_y[0], final_y[1]);
                            println!(
                                "Error from exact: [{:.6e}, {:.6e}]",
                                (final_y[0] - 1.0).abs(),
                                (final_y[1] - 0.0).abs()
                            );
                        }
                        Err(e) => {
                            println!("Error solving harmonic oscillator: {}", e);
                        }
                    }
                }
                Err(e) => {
                    println!("Error solving Van der Pol oscillator: {}", e);
                }
            }
        }
        Err(e) => {
            println!("Error solving exponential decay: {}", e);
        }
    }
}
