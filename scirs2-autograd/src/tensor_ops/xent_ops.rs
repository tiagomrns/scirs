use crate::ndarray_ext::{NdArray, NdArrayView};
use crate::op;
use crate::tensor::Tensor;
use crate::tensor_ops;
use crate::tensor_ops::*;
use crate::Float;
use ndarray;

pub struct SoftmaxCrossEntropy;
pub struct SparseSoftmaxCrossEntropy;
pub struct SparseSoftmaxCrossEntropyGrad;
pub struct SigmoidCrossEntropy;
pub struct LogSoftmax {
    pub axis: isize,
}

impl<T: Float> op::Op<T> for LogSoftmax {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = &ctx.input(0);
        ctx.append_output(x - &crate::tensor_ops::math_ops::logsumexp_forward(x, self.axis, true));
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let gy = ctx.output_grad();
        let sm = exp(ctx.output());
        let sum = reduce_sum(gy, &[1], true);
        let mul = sm * sum;
        ctx.append_input_grad(0, Some(gy - mul));
    }
}

impl<T: Float> op::Op<T> for SigmoidCrossEntropy {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x: &NdArrayView<T> = &ctx.input(0);
        let t: &NdArrayView<T> = &ctx.input(1);

        assert_eq!(x.shape(), t.shape(), "x.shape must match t.shape");

        let max_fn = T::max;
        let mut tmp: NdArray<T> =
            x.mapv(move |a| ((-a.abs()).exp() + T::one()).ln() + max_fn(T::zero(), a));
        tmp -= &(t * x);
        ctx.append_output(tmp);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let s = ctx.graph();
        let x = ctx.input(0);
        let t = ctx.input(1);
        let gy = ctx.output_grad();
        let gx1 = {
            let exp = exp(x);
            ((exp / (scalar(T::one(), s) + exp)) - t) * gy
        };
        let gx2 = neg(gy * t);
        ctx.append_input_grad(0, Some(gx1));
        ctx.append_input_grad(1, Some(gx2));
    }
}

impl<T: Float> op::Op<T> for SparseSoftmaxCrossEntropy {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let (x, t) = (&ctx.input(0), &ctx.input(1));
        let log_x: NdArray<T> = x - &tensor_ops::math_ops::logsumexp_forward(x, 1, true);

        // validation
        {
            let tshape = t.shape();
            if log_x.ndim() != 2 {
                return Err(op::OpError::IncompatibleShape(format!(
                    "SparseSoftmaxCrossEntropy: given first argument's ndim is not 2: shape={:?}",
                    log_x.shape()
                )));
            }
            let t_rank = tshape.len();
            if t_rank == 2 {
                // example label shape: [batch_size, 1]
                if tshape[1] != 1 {
                    return Err(op::OpError::IncompatibleShape(
                        format!("SparseSoftmaxCrossEntropy: second argument's shape must be (batch_size, 1) or (batch_size,). given shape={tshape:?}")
                    ));
                }
            } else if t_rank != 1 {
                // example label shape: [batch_size]
                return Err(op::OpError::IncompatibleShape(
                    format!("SparseSoftmaxCrossEntropy: second argument's shape must be (batch_size, 1) or (batch_size,). given shape={tshape:?}")
                ));
            }
        }

        let mut t_iter = t.iter();
        // loops batch size times.
        let ret = log_x
            .map_axis(ndarray::Axis(1), move |row| {
                -*row
                    .get(
                        t_iter
                            .next()
                            .expect("Batch size mismatch: inputs vs labels")
                            .to_usize()
                            .expect("Invalid label value: can't cast to usize"),
                    )
                    .expect("Wrong label value")
            })
            .into_shape_with_order(ndarray::IxDyn(&[log_x.shape()[0], 1]))
            .unwrap();

        ctx.append_output(ret);
        ctx.append_output(log_x);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let s = ctx.graph();
        let t = ctx.input(1);
        let gy = ctx.output_grad();
        let log_x = nth_tensor(ctx.output(), 1);

        let gx1 = Tensor::builder(s)
            .append_input(log_x, false)
            .append_input(t, false)
            .append_input(gy, false)
            .build(SparseSoftmaxCrossEntropyGrad);

        // gx2 won't be used in most cases.
        let gx2 = {
            let x = exp(log_x);
            let sum = reduce_sum(x * log_x, &[1], true);
            x * gy * (sum - log_x)
        };

        ctx.append_input_grad(0, Some(gx1));
        ctx.append_input_grad(1, Some(gx2));
    }
}

impl<T: Float> op::Op<T> for SparseSoftmaxCrossEntropyGrad {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let log_x = &ctx.input(0); // x is softmax
        let mut x = log_x.map(|a| a.exp());
        let t = &ctx.input(1);
        for (mut row, &t_) in x.axis_iter_mut(ndarray::Axis(0)).zip(t) {
            row[t_.to_usize().unwrap()] -= T::one();
        }

        let gy = &ctx.input(2);
        x *= gy;
        ctx.append_output(x);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
    }
}

impl<T: Float> op::Op<T> for SoftmaxCrossEntropy {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = &ctx.input(0);
        let log_x: NdArray<T> = x - &tensor_ops::math_ops::logsumexp_forward(x, 1, true);
        // `t` must be one-hot
        let t = &ctx.input(1);
        assert_eq!(log_x.ndim(), 2, "x must be 2-ranked tensor");
        assert_eq!(t.ndim(), 2, "t must be 2-ranked tensor");
        // - t log x ( =(batch, num_classes))
        let minus_one = T::one().neg();
        ctx.append_output(
            (t * &log_x)
                .sum_axis(ndarray::Axis(1))
                .mapv(move |elem| elem * minus_one),
        );
        ctx.append_output(log_x);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let output = ctx.output();
        let log_x = nth_tensor(output, 1);
        let gy = ctx.output_grad();
        let x = exp(log_x);
        let t = ctx.input(1);

        // x = softmax, gy = dy/dx
        // = {gy - Σ(x * gy)} * x
        // = {-t/x - Σ(x * -t/x)} * x
        // = {-t/x + Σt} * x
        // = -t + x
        let gx1 = (x - t) * gy;

        // gx2 won't be used in most cases
        let gx2 = {
            let sum = reduce_sum(x * log_x, &[-1], true);
            gy * (sum - log_x) * output
        };

        ctx.append_input_grad(0, Some(gx1));
        ctx.append_input_grad(1, Some(gx2));
    }
}
