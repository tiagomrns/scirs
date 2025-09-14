use crate::ndarray_ext::NdArray;
use crate::op::OpError;
use crate::Float;

pub(crate) struct AdamOp<F: Float> {
    pub(crate) alpha: F,
    pub(crate) eps: F,
    pub(crate) b1: F,
    pub(crate) b2: F,
}

impl<F: Float> crate::op::Op<F> for AdamOp<F> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<F>) -> Result<(), OpError> {
        // Since we can't modify inputs directly with input_mut, we need to
        // create new arrays for all our outputs and return them

        // Debug info
        eprintln!("AdamOp::compute - Number of inputs: {}", ctx.inputs().len());
        for (i, input) in ctx.inputs().iter().enumerate() {
            eprintln!("Input {}: shape {:?}", i, input.shape());
        }

        // Check if we have all the inputs we need
        if ctx.inputs().len() < 5 {
            return Err(OpError::IncompatibleShape(format!(
                "AdamOp requires 5 inputs, but got {}",
                ctx.inputs().len()
            )));
        }

        // Get all the inputs we need (clone them to avoid borrowing issues)
        let param = ctx.input(0).to_owned(); // The parameter to update
        let grad = ctx.input(1).to_owned(); // The gradient
        let m = ctx.input(2).to_owned(); // First moment estimate
        let v = ctx.input(3).to_owned(); // Second moment estimate
        let t_array = ctx.input(4).to_owned(); // Timestep

        // Handle shape mismatches: ensure arrays have compatible shapes
        // We need to create arrays of matching shapes for operations to work
        let gradshape = grad.shape().to_vec();

        // Get the current timestep value and increment it
        let t_val = t_array[ndarray::IxDyn(&[])];
        let new_t = t_val + F::one();
        let new_t_array = NdArray::from_elem(ndarray::IxDyn(&[]), new_t);

        // Create new momentum and velocity arrays with the same shape as grad
        // If original arrays are scalar and grad is not, we need to broadcast
        let mut new_m: NdArray<F>;
        let mut new_v: NdArray<F>;

        // Check if we need to broadcast scalar arrays to match grad shape
        if m.shape().is_empty() && !gradshape.is_empty() {
            // If m is scalar but grad is not, create a new array with m's value broadcast to grad's shape
            let m_val = m[ndarray::IxDyn(&[])];
            new_m = NdArray::from_elem(ndarray::IxDyn(&gradshape), m_val);
        } else {
            new_m = m.to_owned();
        }

        if v.shape().is_empty() && !gradshape.is_empty() {
            // If v is scalar but grad is not, create a new array with v's value broadcast to grad's shape
            let v_val = v[ndarray::IxDyn(&[])];
            new_v = NdArray::from_elem(ndarray::IxDyn(&gradshape), v_val);
        } else {
            new_v = v.to_owned();
        }

        // Also handle param broadcasting if needed
        let mut new_param: NdArray<F>;
        if param.shape().is_empty() && !gradshape.is_empty() {
            let param_val = param[ndarray::IxDyn(&[])];
            new_param = NdArray::from_elem(ndarray::IxDyn(&gradshape), param_val);
        } else {
            new_param = param.to_owned();
        }

        // Compute new first moment estimate
        let tmp_b1 = F::one() - self.b1;
        new_m.zip_mut_with(&grad, move |m_val, g_val| {
            *m_val = *m_val * self.b1 + tmp_b1 * *g_val
        });

        // Compute new second moment estimate
        let tmp_b2 = F::one() - self.b2;
        new_v.zip_mut_with(&grad, move |v_val, g_val| {
            *v_val = *v_val * self.b2 + tmp_b2 * *g_val * *g_val
        });

        // Compute bias-corrected estimates
        let m_correction = F::one() / (F::one() - self.b1.powf(new_t));
        let v_correction = F::one() / (F::one() - self.b2.powf(new_t));

        let m_hat = new_m.mapv(move |m_val| m_val * m_correction);
        let v_hat = new_v.mapv(move |v_val| v_val * v_correction);

        // Compute the parameter update
        let mut update = m_hat.to_owned();
        update.zip_mut_with(&v_hat, move |m_hat_val, v_hat_val| {
            *m_hat_val /= v_hat_val.sqrt() + self.eps;
        });

        // Apply updates to parameters
        new_param.zip_mut_with(&update, move |param_val, update_val| {
            *param_val -= self.alpha * *update_val
        });

        // Append all outputs to the context
        ctx.append_output(new_param); // Updated parameter
        ctx.append_output(grad); // Gradient (unchanged)
        ctx.append_output(new_m); // Updated first moment
        ctx.append_output(new_v); // Updated second moment
        ctx.append_output(new_t_array); // Updated timestep

        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<F>) {
        // Since this is an optimizer operation, we don't propagate gradients
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
        ctx.append_input_grad(2, None);
        ctx.append_input_grad(3, None);
        ctx.append_input_grad(4, None);
    }
}
