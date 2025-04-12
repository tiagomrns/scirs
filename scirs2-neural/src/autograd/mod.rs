//! Automatic differentiation module for neural networks.
//!
//! This module provides automatic differentiation capabilities for neural networks,
//! using the autograd crate. This is a simple wrapper for demonstration, showing how
//! we can integrate with autograd.

// We re-export the autograd crate
pub extern crate autograd;

/// Example function showing how autograd can be used
pub fn autograd_example() {
    // This demonstrates basic usage of autograd for a simple neural network operation
    println!("To use autograd in your code, import it directly:");
    println!("use autograd::prelude::*;");
    println!();
    println!("For more examples, see:");
    println!("https://docs.rs/autograd/latest/autograd/");
    println!();
    println!("This integration is provided as a simple demonstration of how to use");
    println!("autograd with SciRS2-neural.");
}
