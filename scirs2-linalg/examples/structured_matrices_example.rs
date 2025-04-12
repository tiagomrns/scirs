use ndarray::array;
use scirs2_linalg::structured::{
    structured_to_operator, CirculantMatrix, HankelMatrix, StructuredMatrix, ToeplitzMatrix,
};
use scirs2_linalg::MatrixFreeOp;

fn main() {
    println!("Structured Matrices Example\n");

    // ----- Toeplitz Matrix Example -----
    println!("=== Toeplitz Matrix ===");

    // Create a Toeplitz matrix
    let first_row = array![1.0, 2.0, 3.0, 4.0];
    let first_col = array![1.0, 5.0, 6.0, 7.0];

    let toeplitz = ToeplitzMatrix::new(first_row.view(), first_col.view()).unwrap();

    // Print the matrix
    let dense_toeplitz = toeplitz.to_dense().unwrap();
    println!("Toeplitz matrix:");
    println!("{}", dense_toeplitz);

    // Matrix-vector multiplication
    let x = array![1.0, 2.0, 3.0, 4.0];
    let y = toeplitz.matvec(&x.view()).unwrap();

    println!("\nMatrix-vector product:");
    println!("x = {:?}", x);
    println!("Tx = {:?}", y);

    // Create a symmetric Toeplitz matrix
    let first_row_sym = array![1.0, 2.0, 3.0];
    let toeplitz_sym = ToeplitzMatrix::new_symmetric(first_row_sym.view()).unwrap();

    println!("\nSymmetric Toeplitz matrix:");
    println!("{}", toeplitz_sym.to_dense().unwrap());

    // ----- Circulant Matrix Example -----
    println!("\n=== Circulant Matrix ===");

    // Create a circulant matrix
    let first_row = array![1.0, 2.0, 3.0, 4.0];
    let circulant = CirculantMatrix::new(first_row.view()).unwrap();

    // Print the matrix
    let dense_circulant = circulant.to_dense().unwrap();
    println!("Circulant matrix:");
    println!("{}", dense_circulant);

    // Matrix-vector multiplication
    let y = circulant.matvec(&x.view()).unwrap();

    println!("\nMatrix-vector product:");
    println!("x = {:?}", x);
    println!("Cx = {:?}", y);

    // ----- Hankel Matrix Example -----
    println!("\n=== Hankel Matrix ===");

    // Create a Hankel matrix
    let first_col = array![1.0, 2.0, 3.0];
    let last_row = array![3.0, 4.0, 5.0];

    let hankel = HankelMatrix::new(first_col.view(), last_row.view()).unwrap();

    // Print the matrix
    let dense_hankel = hankel.to_dense().unwrap();
    println!("Hankel matrix:");
    println!("{}", dense_hankel);

    // Matrix-vector multiplication
    let x_hankel = array![1.0, 2.0, 3.0];
    let y = hankel.matvec(&x_hankel.view()).unwrap();

    println!("\nMatrix-vector product:");
    println!("x = {:?}", x_hankel);
    println!("Hx = {:?}", y);

    // Create a Hankel matrix from a sequence
    let sequence = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let hankel_seq = HankelMatrix::from_sequence(sequence.view(), 3, 3).unwrap();

    println!("\nHankel matrix from sequence:");
    println!("{}", hankel_seq.to_dense().unwrap());

    // ----- Matrix-free operations -----
    println!("\n=== Matrix-free operations ===");

    // Convert to matrix-free operator
    let toeplitz_op = structured_to_operator(&toeplitz);

    // Apply the operator
    let y_op = toeplitz_op.apply(&x.view()).unwrap();

    println!("Matrix-free operator result:");
    println!("x = {:?}", x);
    println!("Op(x) = {:?}", y_op);

    println!("\nCompare with direct matrix-vector product:");
    println!("Tx = {:?}", y);

    // Notice that the results are identical, but the matrix-free approach
    // doesn't require storing the full matrix
}
