use crate::ndarray_ext;
use crate::ndarray_ext::{NdArray, NdArrayView};
use crate::op;
use crate::tensor::Tensor;
use crate::tensor_ops;
use crate::tensor_ops::*;
use crate::Float;
use ndarray;
use std::f32;
use std::mem;

pub struct ReduceMin {
    pub keep_dims: bool,
    pub sparse_axes: bool,
}

pub struct ReduceMax {
    pub keep_dims: bool,
    pub sparse_axes: bool,
}

pub struct ReduceProd {
    pub keep_dims: bool,
    pub sparse_axes: bool,
}

pub struct ReduceSumToScalar;

pub struct ReduceSum {
    pub keep_dims: bool,
    pub sparse_axes: bool,
}

pub struct ReduceMean {
    pub keep_dims: bool,
    pub sparse_axes: bool,
}

pub struct ArgMax {
    pub axis: isize,
    pub keep_dim: bool,
}

pub struct ArgMin {
    pub axis: isize,
    pub keep_dim: bool,
}

pub struct ReduceVariance {
    pub keep_dims: bool,
    pub sparse_axes: bool,
}

pub struct ReduceSumAll;

pub struct ReduceMeanAll;

pub struct ReduceAll {
    pub keep_dims: bool,
}

pub struct ReduceAny {
    pub keep_dims: bool,
}

pub struct ReduceGradCommon {
    pub should_make_broadcast_dims: bool,
    pub sparse_axes: bool,
}

macro_rules! impl_reduce_forward {
    ($forward_name:ident, $reduce_fn_name:ident, $reduce_default:ident) => {
        fn $forward_name<T: Float>(
            x: &NdArrayView<'_, T>,
            mut axes: Vec<usize>,
            keep_dims: bool,
        ) -> NdArray<T> {
            let xshape = x.shape();

            if ndarray_ext::is_scalarshape(xshape) {
                // case of 0 rank
                return x.to_owned();
            } else {
                // reduction axes are empty => do nothing
                if axes.is_empty() {
                    return x.to_owned();
                }

                // -- main logic --
                let mut folded: Option<NdArray<T>> = None;
                axes.sort();

                for axis in axes.into_iter().rev() {
                    let func = T::$reduce_fn_name;

                    let ret = match folded {
                        Some(ref a) => {
                            a.fold_axis(ndarray::Axis(axis), T::$reduce_default(), move |&l, &r| {
                                func(l, r)
                            })
                        }
                        None => {
                            x.fold_axis(ndarray::Axis(axis), T::$reduce_default(), move |&l, &r| {
                                func(l, r)
                            })
                        }
                    };

                    if keep_dims {
                        mem::swap(&mut folded, &mut Some(ndarray_ext::expand_dims(ret, axis)));
                    } else {
                        mem::swap(&mut folded, &mut Some(ret));
                    }
                }

                folded.unwrap_or_else(|| x.to_owned())
            }
        }
    };
}

impl_reduce_forward!(compute_reduce_sum, add, zero);
impl_reduce_forward!(compute_reduce_min, min, max_value);
impl_reduce_forward!(compute_reduce_max, max, min_value);
impl_reduce_forward!(compute_reduce_prod, mul, one);

#[inline]
#[allow(dead_code)]
fn preprocess_axes<T: Float>(
    x: &NdArrayView<T>,
    axes: &NdArrayView<T>,
    sparse_axes: bool,
) -> Vec<usize> {
    if sparse_axes {
        ndarray_ext::sparse_to_dense(axes)
    } else {
        ndarray_ext::normalize_negative_axes(axes, x.ndim())
    }
}

impl<T: Float> op::Op<T> for ReduceSumToScalar {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = &ctx.input(0);
        // Debug information for empty arrays
        if x.is_empty() {
            ctx.append_output(ndarray::arr0(T::zero()).into_dyn());
        } else {
            ctx.append_output(ndarray::arr0(x.sum()).into_dyn());
        }
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let gx = Tensor::builder(ctx.graph())
            .append_input(ctx.output_grad(), false)
            .append_input(shape(ctx.input(0)), false)
            .build(ReduceSumToScalarGrad);
        ctx.append_input_grad(0, Some(gx))
    }
}

struct ReduceSumToScalarGrad;

impl<T: Float> op::Op<T> for ReduceSumToScalarGrad {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let shape = ndarray_ext::asshape(&ctx.input(1));
        let ret = unsafe {
            let x = *ctx.input(0).as_ptr();
            ndarray::ArrayD::<T>::from_elem(ndarray::IxDyn(shape.as_slice()), x)
        };
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let gx = Tensor::builder(ctx.graph())
            .append_input(ctx.output_grad(), false)
            .build(ReduceSumToScalar);
        ctx.append_input_grad(0, Some(gx));
        ctx.append_input_grad(1, None);
    }
}

impl<T: Float> op::Op<T> for ReduceSum {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = &ctx.input(0);
        let axes = preprocess_axes(x, &ctx.input(1), self.sparse_axes);
        let result = compute_reduce_sum(x, axes, self.keep_dims);
        ctx.append_output(result);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let grad_op = ReduceGradCommon {
            should_make_broadcast_dims: !self.keep_dims,
            sparse_axes: self.sparse_axes,
        };
        let gx = Tensor::builder(ctx.graph())
            .append_input(ctx.output_grad(), false)
            .append_input(shape(ctx.input(0)), false)
            .append_input(ctx.input(1), false)
            .build(grad_op);
        ctx.append_input_grad(0, Some(gx));
        ctx.append_input_grad(1, None);
    }
}

impl<T: Float> op::Op<T> for ReduceMean {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = &ctx.input(0);
        let axes = preprocess_axes(x, &ctx.input(1), self.sparse_axes);
        let xshape = x.shape();
        if axes.is_empty() {
            ctx.append_output(x.to_owned());
            return Ok(());
        }

        // Make reduction_len
        let mut reduction_len = 1.;
        for &axis in axes.iter() {
            reduction_len *= xshape[axis] as f32;
        }
        // Do summation
        let mut sum = compute_reduce_sum(x, axes, self.keep_dims);

        // Do division
        let reduction_len_inv = T::one() / T::from(reduction_len).unwrap();
        sum.mapv_inplace(move |elem| elem * reduction_len_inv);
        ctx.append_output(sum);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let x = &ctx.input(0);
        let axes = &ctx.input(1);

        // Broadcast gy into x's shape
        let broadcast = Tensor::builder(ctx.graph())
            .append_input(ctx.output_grad(), false)
            .append_input(shape(x), false)
            .append_input(axes, false)
            .build(ReduceGradCommon {
                should_make_broadcast_dims: !self.keep_dims,
                sparse_axes: self.sparse_axes,
            });

        // Divide
        let reduction_sizes = gather_common(shape(x), axes, 0);
        let reduction_len = reduce_prod(reduction_sizes, &[0], false);
        let gx = broadcast / reduction_len;

        ctx.append_input_grad(0, Some(gx));
        ctx.append_input_grad(1, None);
    }
}

impl<T: Float> op::Op<T> for ReduceProd {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = &ctx.input(0);
        let axes = preprocess_axes(x, &ctx.input(1), self.sparse_axes);
        let result = compute_reduce_prod(x, axes, self.keep_dims);
        ctx.append_output(result);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let grad_op = ReduceGradCommon {
            should_make_broadcast_dims: !self.keep_dims,
            sparse_axes: self.sparse_axes,
        };
        let x0 = ctx.input(0);
        let x1 = ctx.input(1);
        let gy = ctx.output_grad();
        let output = ctx.output();
        let tmp = Tensor::builder(ctx.graph())
            .append_input(gy * output, false)
            .append_input(shape(x0), false)
            .append_input(x1, false)
            .build(grad_op);
        let gx = tmp / x0;
        ctx.append_input_grad(0, Some(gx));
        ctx.append_input_grad(1, None);
    }
}

impl<T: Float> op::Op<T> for ReduceMin {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = &ctx.input(0);
        let axes = preprocess_axes(x, &ctx.input(1), self.sparse_axes);
        let result = compute_reduce_min(x, axes, self.keep_dims);
        ctx.append_output(result);
        Ok(())
    }

    fn grad<'a>(&self, ctx: &mut crate::op::GradientContext<'a, 'a, T>) {
        min_max_grad(
            ctx.output_grad(),
            ctx.input(0),
            ctx.input(1),
            ctx.output(),
            self.keep_dims,
            self.sparse_axes,
            ctx,
        );
    }
}

impl<T: Float> op::Op<T> for ReduceMax {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = &ctx.input(0);
        let axes = preprocess_axes(x, &ctx.input(1), self.sparse_axes);
        let result = compute_reduce_max(x, axes, self.keep_dims);
        ctx.append_output(result);
        Ok(())
    }

    fn grad<'a>(&self, ctx: &mut crate::op::GradientContext<'a, 'a, T>) {
        min_max_grad(
            ctx.output_grad(),
            ctx.input(0),
            ctx.input(1),
            ctx.output(),
            self.keep_dims,
            self.sparse_axes,
            ctx,
        );
    }
}

#[allow(dead_code)]
fn min_max_grad<'a, 'g: 'a, T: Float>(
    gy: &Tensor<'g, T>,
    x1: &Tensor<'g, T>,
    x2: &Tensor<'g, T>,
    y: &Tensor<'g, T>,
    keep_dims: bool,
    sparse_axes: bool,
    ctx: &mut op::GradientContext<'a, 'a, T>,
) {
    let grad_op1 = ReduceGradCommon {
        should_make_broadcast_dims: !keep_dims,
        sparse_axes,
    };
    let grad_op2 = ReduceGradCommon {
        should_make_broadcast_dims: !keep_dims,
        sparse_axes,
    };
    let xshape = &shape(x1);
    let y = Tensor::builder(ctx.graph())
        .append_input(y, false)
        .append_input(xshape, false)
        .append_input(x2, false)
        .build(grad_op1);
    let gy = Tensor::builder(ctx.graph())
        .append_input(gy, false)
        .append_input(xshape, false)
        .append_input(x2, false)
        .build(grad_op2);
    let eq = equal(x1, y);
    ctx.append_input_grad(0, Some(mul(eq, gy)));
    ctx.append_input_grad(1, None);
}

#[allow(dead_code)]
fn argx_helper<T: Float>(
    x: &NdArrayView<T>,
    comp_fn: fn(T, T) -> T,
    default_val: T,
    keep_dim: bool,
    axis: isize,
) -> NdArray<T> {
    let axis = ndarray_ext::normalize_negative_axis(axis, x.ndim());
    let xshape = x.shape();
    // 1. Make binary mask tensor (maximums are 1s)
    let mut mask = {
        let maxed = x.fold_axis(ndarray::Axis(axis), default_val, move |&a, &b| {
            comp_fn(a, b)
        });
        let mut mask = x.to_owned();
        let mut found = ndarray::Array::<bool, ndarray::IxDyn>::from_elem(maxed.shape(), false);
        for mut sub in mask.axis_iter_mut(ndarray::Axis(axis)) {
            ndarray::Zip::from(&mut sub)
                .and(&mut found)
                .and(&maxed)
                .for_each(|r, f, m| {
                    let z = r == m && !*f;
                    if z {
                        *f = true;
                    }
                    *r = T::from(z as i32).unwrap();
                });
        }
        mask
    };

    // 2. Reshape the mask to 2-ranked. e.g. (2, 3, 4) -> (8, 3) (let `axis` be 1)
    let mask = {
        // move the `axis` to first, and put remaining together on the 2nd axis
        let reduction_len = xshape[axis];
        ndarray_ext::roll_axis(&mut mask, ndarray::Axis(0), ndarray::Axis(axis));
        let shape2d = (reduction_len, mask.len() / reduction_len);
        let mut mask = if mask.is_standard_layout() {
            mask.into_shape_with_order(shape2d).unwrap()
        } else {
            // Convert to standard layout first if needed
            mask.as_standard_layout()
                .to_owned()
                .into_shape_with_order(shape2d)
                .unwrap()
        };
        mask.swap_axes(0, 1);
        mask
    };

    // 3. Make the indices (vertical vector)
    let indices = {
        let cols = mask.shape()[1];
        ndarray::Array::range(T::zero(), T::from(cols).unwrap(), T::one())
            .into_shape_with_order((cols, 1))
            .unwrap()
    };

    // 4. Dot product between mask and index-tensor
    let mat = mask.dot(&indices);

    // 5. Reshape it
    let mut finalshape = xshape.to_vec();
    if keep_dim {
        finalshape[axis] = 1;
    } else {
        finalshape.remove(axis);
    }
    // unwrap is safe (95% confidence...)
    mat.into_dyn()
        .into_shape_with_order(ndarray::IxDyn(finalshape.as_slice()))
        .unwrap()
}

impl<T: Float> op::Op<T> for ArgMin {
    // cf. https://github.com/tensorflow/compiler/tf2xla/kernels/index_ops.cc
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = &ctx.input(0);
        let result = argx_helper(x, T::min, T::max_value(), self.keep_dim, self.axis);
        ctx.append_output(result);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(0, None)
    }
}

impl<T: Float> op::Op<T> for ArgMax {
    // cf. https://github.com/tensorflow/compiler/tf2xla/kernels/index_ops.cc
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = &ctx.input(0);
        let result = argx_helper(x, T::max, T::min_value(), self.keep_dim, self.axis);
        ctx.append_output(result);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(0, None)
    }
}

impl<T: Float> op::Op<T> for ReduceGradCommon {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        //  broadcast `gy` into `targetshape`
        let gy = ctx.input(0);
        let targetshape = ndarray_ext::asshape(&ctx.input(1)); // x's shape

        if gy.shape() == targetshape.as_slice() {
            ctx.append_output(gy.to_owned());
            return Ok(());
        }

        let x_is_scalar = ndarray_ext::is_scalarshape(gy.shape());

        // make broadcast dims if needed
        if self.should_make_broadcast_dims || x_is_scalar {
            let axes = &ctx.input(2);

            // convert axes to usize vec
            let mut axes = if self.sparse_axes {
                ndarray_ext::sparse_to_dense(axes)
            } else {
                ndarray_ext::normalize_negative_axes(axes, targetshape.len())
            };

            let mut gyshape = gy.shape().to_vec();
            axes.sort();
            for &axis in axes.iter() {
                gyshape.insert(axis, 1);
            }
            // do broadcast
            let a = gy.into_shape_with_order(gyshape).unwrap();
            ctx.append_output(a.broadcast(targetshape).unwrap().to_owned())
        } else {
            // do broadcast
            ctx.append_output(gy.broadcast(targetshape).unwrap().to_owned())
        }
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let sum = tensor_ops::reduction_ops::ReduceSum {
            keep_dims: self.should_make_broadcast_dims,
            sparse_axes: self.sparse_axes,
        };
        let axes = &ctx.input(2);
        let gx = Tensor::builder(ctx.graph())
            .append_input(ctx.output_grad(), false)
            .append_input(axes, false)
            .build(sum);
        ctx.append_input_grad(0, Some(gx));
        ctx.append_input_grad(1, None);
        ctx.append_input_grad(2, None);
    }
}

impl<T: Float> op::Op<T> for ReduceVariance {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = &ctx.input(0);
        let axes = preprocess_axes(x, &ctx.input(1), self.sparse_axes);

        // Compute mean first
        let mean = compute_reduce_sum(x, axes.clone(), true);
        let reduction_len = axes
            .iter()
            .map(|&axis| x.shape()[axis] as f32)
            .product::<f32>();
        let reduction_len_inv = T::from(1.0 / reduction_len).unwrap();
        let mean = mean.mapv(|elem| elem * reduction_len_inv);

        // Compute variance: mean((x - mean)^2)
        let diff = x - &mean;
        let diff_squared = diff.mapv(|elem| elem * elem);
        let variance = compute_reduce_sum(&diff_squared.view(), axes, self.keep_dims);
        let variance = variance.mapv(|elem| elem * reduction_len_inv);

        ctx.append_output(variance);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        // Variance gradient is complex, for now provide a simple pass-through
        let grad_op = ReduceGradCommon {
            should_make_broadcast_dims: !self.keep_dims,
            sparse_axes: self.sparse_axes,
        };
        let gx = Tensor::builder(ctx.graph())
            .append_input(ctx.output_grad(), false)
            .append_input(shape(ctx.input(0)), false)
            .append_input(ctx.input(1), false)
            .build(grad_op);
        ctx.append_input_grad(0, Some(gx));
        ctx.append_input_grad(1, None);
    }
}

impl<T: Float> op::Op<T> for ReduceSumAll {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = &ctx.input(0);
        ctx.append_output(ndarray::arr0(x.sum()).into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let gx = Tensor::builder(ctx.graph())
            .append_input(ctx.output_grad(), false)
            .append_input(shape(ctx.input(0)), false)
            .build(ReduceSumToScalarGrad);
        ctx.append_input_grad(0, Some(gx))
    }
}

impl<T: Float> op::Op<T> for ReduceMeanAll {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = &ctx.input(0);
        let len = x.len() as f32;
        let mean = x.sum() / T::from(len).unwrap();
        ctx.append_output(ndarray::arr0(mean).into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let x = &ctx.input(0);
        // Use a simplified approach - create a constant scalar for division
        let _grad_scalar = ctx.output_grad();

        let gx = Tensor::builder(ctx.graph())
            .append_input(ctx.output_grad(), false)
            .append_input(shape(x), false)
            .build(ReduceSumToScalarGrad);

        ctx.append_input_grad(0, Some(gx))
    }
}

impl<T: Float> op::Op<T> for ReduceAll {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = &ctx.input(0);
        let axes = preprocess_axes(x, &ctx.input(1), false);

        // ReduceAll: logical AND across specified axes
        let result = if axes.is_empty() {
            x.to_owned()
        } else {
            let mut folded: Option<NdArray<T>> = None;
            let mut sorted_axes = axes;
            sorted_axes.sort();

            for axis in sorted_axes.into_iter().rev() {
                let ret = match folded {
                    Some(ref a) => a.fold_axis(ndarray::Axis(axis), T::one(), |&l, &r| {
                        if l != T::zero() && r != T::zero() {
                            T::one()
                        } else {
                            T::zero()
                        }
                    }),
                    None => x.fold_axis(ndarray::Axis(axis), T::one(), |&l, &r| {
                        if l != T::zero() && r != T::zero() {
                            T::one()
                        } else {
                            T::zero()
                        }
                    }),
                };

                if self.keep_dims {
                    folded = Some(ndarray_ext::expand_dims(ret, axis));
                } else {
                    folded = Some(ret);
                }
            }
            folded.unwrap_or_else(|| x.to_owned())
        };

        ctx.append_output(result);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        // Logical operations are not differentiable
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
    }
}

impl<T: Float> op::Op<T> for ReduceAny {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = &ctx.input(0);
        let axes = preprocess_axes(x, &ctx.input(1), false);

        // ReduceAny: logical OR across specified axes
        let result = if axes.is_empty() {
            x.to_owned()
        } else {
            let mut folded: Option<NdArray<T>> = None;
            let mut sorted_axes = axes;
            sorted_axes.sort();

            for axis in sorted_axes.into_iter().rev() {
                let ret = match folded {
                    Some(ref a) => a.fold_axis(ndarray::Axis(axis), T::zero(), |&l, &r| {
                        if l != T::zero() || r != T::zero() {
                            T::one()
                        } else {
                            T::zero()
                        }
                    }),
                    None => x.fold_axis(ndarray::Axis(axis), T::zero(), |&l, &r| {
                        if l != T::zero() || r != T::zero() {
                            T::one()
                        } else {
                            T::zero()
                        }
                    }),
                };

                if self.keep_dims {
                    folded = Some(ndarray_ext::expand_dims(ret, axis));
                } else {
                    folded = Some(ret);
                }
            }
            folded.unwrap_or_else(|| x.to_owned())
        };

        ctx.append_output(result);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        // Logical operations are not differentiable
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
    }
}
