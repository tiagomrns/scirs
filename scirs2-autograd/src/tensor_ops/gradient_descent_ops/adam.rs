use crate::Float;

pub(crate) struct AdamOp<F: Float> {
    pub(crate) alpha: F,
    pub(crate) eps: F,
    pub(crate) b1: F,
    pub(crate) b2: F,
}

impl<F: Float> crate::op::Op<F> for AdamOp<F> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<F>) -> Result<(), crate::op::OpError> {
        // Get the gradient and clone to avoid borrow issues
        let input1 = ctx.input(1).to_owned();

        // Get the current timestep value
        let t_val = {
            let t = ctx.input(4);
            // t is not null
            t[ndarray::IxDyn(&[])]
        };

        // Update t for next iteration
        {
            let mut t = ctx.input_mut(4);
            unsafe {
                *t.as_mut_ptr() += F::one();
            }
        }

        // Make new m
        {
            let tmp = F::one() - self.b1;
            let mut input2 = ctx.input_mut(2);
            input2.zip_mut_with(&input1, move |x2_elem, &g| {
                *x2_elem = *x2_elem * self.b1 + tmp * g
            });
        }

        // Get the updated m
        let new_m = ctx.input(2).to_owned();

        // Make new v
        {
            let tmp = F::one() - self.b2;
            let mut input3 = ctx.input_mut(3);
            input3.zip_mut_with(&input1, move |x3_elem, &g| {
                *x3_elem = *x3_elem * self.b2 + tmp * g * g
            });
        }

        // Get the updated v
        let new_v = ctx.input(3).to_owned();

        // Compute bias-corrected estimates
        let m_hat = {
            let rhs = F::one() / (F::one() - self.b1.powf(t_val));
            new_m.mapv(move |new_m_elem| new_m_elem * rhs)
        };

        let v_hat = {
            let rhs = F::one() / (F::one() - self.b2.powf(t_val));
            new_v.mapv(move |new_v_elem| new_v_elem * rhs)
        };

        // Compute update
        let mut update = m_hat.to_owned();
        update.zip_mut_with(&v_hat, move |a, &b| (*a) /= b.sqrt() + self.eps);

        // Update parameters
        let mut param = ctx.input_mut(0);
        param.zip_mut_with(&update, move |l, &r| *l -= self.alpha * r);

        ctx.append_output(ndarray::Array::zeros(vec![]).into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<F>) {
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
        ctx.append_input_grad(2, None);
        ctx.append_input_grad(3, None);
        ctx.append_input_grad(4, None);
    }
}
