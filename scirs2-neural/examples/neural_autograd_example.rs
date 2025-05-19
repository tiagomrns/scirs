use scirs2_neural::autograd;

fn main() {
    println!("# Autograd Example");
    println!();

    // Show the autograd example information
    autograd::autograd_example();

    println!();
    println!("For a complete example of using autograd for neural networks,");
    println!("you can refer to the autograd crate documentation and examples:");
    println!("https://github.com/raskr/rust-autograd/tree/master/examples");

    println!();
    println!("The following features are available through autograd:");
    println!("- Automatic differentiation for neural networks");
    println!("- Tensor operations with gradients");
    println!("- Forward and backward passes");
    println!("- Common neural network operations (convolution, pooling, etc.)");
    println!("- Optimizers like SGD, Adam, etc.");

    println!();
    println!("To use autograd in your code, add it as a dependency:");
    println!("```");
    println!("autograd = \"1.1.1\"");
    println!("```");

    println!();
    println!("Then import it in your code:");
    println!("```");
    println!("use autograd::prelude::*;");
    println!("```");

    println!();
    println!("Example of a simple neural network with autograd:");
    println!("```rust");
    println!("use autograd as ag;");
    println!("use ag::ndarray;");
    println!();
    println!("ag::with(|g| {{");
    println!("    // Create input data");
    println!("    let x = g.ones(&[2, 3]);");
    println!();
    println!("    // Create weights and biases");
    println!("    let w = g.variable(ndarray::Array2::ones((3, 1)));");
    println!("    let b = g.variable(ndarray::Array1::ones(1));");
    println!();
    println!("    // Forward pass");
    println!("    let y = x.dot(&w) + b;");
    println!();
    println!("    // Apply activation");
    println!("    let activation = g.sigmoid(&y);");
    println!();
    println!("    // Compute loss");
    println!("    let loss = activation.mean();");
    println!();
    println!("    // Compute gradients");
    println!("    let grads = g.grad(&[loss], &[w.clone(), b.clone()]);");
    println!();
    println!("    // Update parameters");
    println!("    // ...");
    println!("}});");
    println!("```");
}
