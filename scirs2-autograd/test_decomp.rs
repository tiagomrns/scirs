use scirs2_autograd as ag;
use ag::tensor_ops::{qr, svd};
use ag::prelude::*;
use ndarray::array;

fn main() {
    let ctx = ag::Context::new();
    
    // Test 2x2 matrix
    let matrix = array![[4.0, 3.0], 
                       [0.0, -1.0]];
    let matrix_tensor = ag::tensor(matrix.into_dyn(), &ctx);
    
    // Test QR decomposition
    println!("Testing QR decomposition...");
    let (q, r) = qr(&matrix_tensor);
    println!("Q shape: {:?}", q.eval(&ctx).unwrap().shape());
    println!("R shape: {:?}", r.eval(&ctx).unwrap().shape());
    
    // Test SVD decomposition
    println!("\nTesting SVD decomposition...");
    let (u, s, v) = svd(&matrix_tensor);
    println!("U shape: {:?}", u.eval(&ctx).unwrap().shape());
    println!("S shape: {:?}", s.eval(&ctx).unwrap().shape());
    println!("V shape: {:?}", v.eval(&ctx).unwrap().shape());
    
    println!("\nDecomposition tests completed successfully!");
}