use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::Float;

/// Frobenius norm operation
pub struct FrobeniusNormOp;

impl<F: Float> Op<F> for FrobeniusNormOp {
    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let input_array = input.view();

        let mut sum_squared = F::zero();
        for &elem in input_array.iter() {
            sum_squared += elem * elem;
        }

        let norm = sum_squared.sqrt();
        ctx.append_output(ndarray::arr0(norm).into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let grad_output = ctx.output_grad();
        let input = ctx.input(0);
        let output = ctx.output();
        let g = ctx.graph();

        // Evaluate tensors to arrays
        let grad_output_array = match grad_output.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let output_array = match output.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let input_array = match input.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        // Get scalar values
        let grad_scalar = grad_output_array[[0]];
        let norm = output_array[[0]];

        // Gradient is input / norm * grad_output
        let grad_input = if norm > F::epsilon() {
            input_array.mapv(|x| x / norm * grad_scalar)
        } else {
            // Create zeros array with same shape as input
            ndarray::Array::zeros(input_array.raw_dim())
        };

        // Convert to tensor and append
        let grad_tensor = crate::tensor_ops::convert_to_tensor(grad_input, g);
        ctx.append_input_grad(0, Some(grad_tensor));
    }
}

pub fn frobenius_norm<'g, F: Float>(matrix: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = matrix.graph();
    Tensor::builder(g)
        .append_input(matrix, false)
        .build(FrobeniusNormOp)
}
