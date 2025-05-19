use ag::tensor_ops::*;
use scirs2_autograd as ag;

fn test_frobenius_gradient() {
    ag::run::<f64, _, _>(|ctx| {
        // Create test matrices
        let a = ag::tensor_ops::random_normal(&[3, 3], 0.0, 1.0, ctx);
        let b = ag::tensor_ops::random_normal(&[3, 3], 0.0, 1.0, ctx);

        // Compute matrix product and norm
        let c = ag::tensor_ops::matmul(&a, &b);
        let norm = frobenius_norm(&c);

        // Compute gradients - these should be non-zero
        let grads = ag::tensor_ops::grad(&[norm], &[&a, &b]);

        println!("Gradient w.r.t. A:\n{:?}", grads[0].eval(ctx));
        println!("Gradient w.r.t. B:\n{:?}", grads[1].eval(ctx));

        // Test trace gradient
        let tr = trace(&c);
        let trace_grads = ag::tensor_ops::grad(&[tr], &[&a, &b]);

        println!("Trace gradient w.r.t. A:\n{:?}", trace_grads[0].eval(ctx));
        println!("Trace gradient w.r.t. B:\n{:?}", trace_grads[1].eval(ctx));
    });
}

fn test_decomposition_gradients() {
    ag::run::<f64, _, _>(|ctx| {
        let a = ag::tensor_ops::random_normal(&[4, 4], 0.0, 1.0, ctx);

        // Test QR decomposition
        let (q, r) = qr(&a);
        let loss_qr = frobenius_norm(&q) + frobenius_norm(&r);
        let grad_qr = ag::tensor_ops::grad(&[loss_qr], &[&a]);
        println!("QR gradient:\n{:?}", grad_qr[0].eval(ctx));

        // Test LU decomposition
        // let (_p, l, u) = lu(&a);
        // let loss_lu = frobenius_norm(&l) + frobenius_norm(&u);
        // let grad_lu = ag::tensor_ops::grad(&[loss_lu], &[&a]);
        // println!("LU gradient:\n{:?}", grad_lu[0].eval(ctx));

        // Test SVD
        let (_u, s, _v) = svd(&a);
        let loss_svd = ag::tensor_ops::reduce_sum(&s, &[0], false);
        let grad_svd = ag::tensor_ops::grad(&[loss_svd], &[&a]);
        println!("SVD gradient:\n{:?}", grad_svd[0].eval(ctx));
    });
}

fn test_linear_algebra_operations() {
    ag::run::<f64, _, _>(|ctx| {
        // Test eye matrix
        let eye_matrix = eye(4, ctx);
        println!("Identity matrix:\n{:?}", eye_matrix.eval(ctx));

        // Test trace
        let a = ag::tensor_ops::convert_to_tensor(
            ndarray::array![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]].into_dyn(),
            ctx,
        );
        let tr = a.trace();
        println!("Trace of matrix: {:?}", tr.eval(ctx));

        // Test diagonal operations
        let diag_values =
            ag::tensor_ops::convert_to_tensor(ndarray::array![10., 20., 30.].into_dyn(), ctx);
        let diag_matrix = diag(&diag_values);
        println!("Diagonal matrix:\n{:?}", diag_matrix.eval(ctx));

        let extracted = diag_matrix.diag();
        println!("Extracted diagonal: {:?}", extracted.eval(ctx));

        // Test scalar multiplication
        let scaled = a.scalar_mul(2.0);
        println!("Scaled matrix:\n{:?}", scaled.eval(ctx));
    });
}

fn main() {
    println!("Testing linear algebra operations:");
    test_linear_algebra_operations();

    println!("\nTesting Frobenius norm gradient:");
    test_frobenius_gradient();

    println!("\nTesting decomposition gradients:");
    test_decomposition_gradients();
}
