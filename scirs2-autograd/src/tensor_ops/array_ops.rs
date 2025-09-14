use crate::ndarray;
use crate::ndarray_ext;
#[cfg(feature = "blas")]
use crate::ndarray_ext::NdArrayViewMut;
use crate::ndarray_ext::{NdArray, NdArrayView};
use crate::op;
use crate::tensor::Tensor;
use crate::tensor_ops::*;
use crate::Float;
use ndarray::SliceInfoElem;
use std::iter::FromIterator;

pub struct ExpandDims;

pub struct Squeeze;

pub struct Slice {
    pub indices: Vec<SliceInfoElem>,
}

pub struct SliceGrad {
    pub indices: Vec<SliceInfoElem>,
}

pub struct Split {
    pub axis: isize,
    pub start_index: isize,
    pub end_index: isize,
}

pub struct SplitGrad {
    pub axis: isize,
    pub start_index: isize,
    pub end_index: isize,
}

pub struct Tile {
    pub axis: isize,
    pub num: usize,
}

pub struct Concat {
    pub axis: isize,
}

pub struct ConcatGrad {
    pub axis: isize,
    pub index: usize,
}

pub struct Clip<T: Float> {
    pub min: T,
    pub max: T,
}

pub struct ClipGrad<T: Float> {
    pub min: T,
    pub max: T,
}

pub struct AddN;

pub struct Gather {
    pub axis: isize,
    pub should_normalize_negative_indices: bool,
}

pub struct GatherGrad {
    pub axis: isize,
}

pub struct IndexOp {
    pub index: isize,
}

pub struct IndexOpGrad {
    pub index: isize,
}

pub struct SetDiff1D;

pub struct Shape;

pub struct Rank;

pub struct Size;

pub struct Reshape;

pub struct InferBinOpShape;

pub struct Assign;

impl<T: Float> op::Op<T> for Assign {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let input1 = ctx.input(1).to_owned();
        ctx.input_mut(0).assign(&input1);
        ctx.append_output(ndarray::Array::zeros(vec![]).into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
    }
}

impl<T: Float> op::Op<T> for InferBinOpShape {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let ashape_float = ctx.input(0);
        let bshape_float = ctx.input(1);
        let ashape: Vec<usize> = ashape_float
            .map(|x| x.to_usize().unwrap())
            .iter()
            .cloned()
            .collect();
        let bshape: Vec<usize> = bshape_float
            .map(|x| x.to_usize().unwrap())
            .iter()
            .cloned()
            .collect();
        let a_is_scalar = ndarray_ext::is_scalarshape(ashape.as_slice());
        let b_is_scalar = ndarray_ext::is_scalarshape(bshape.as_slice());

        if !a_is_scalar && !b_is_scalar {
            let a_rank = ashape.len();
            let b_rank = bshape.len();
            if a_rank != b_rank {
                return Err(op::OpError::IncompatibleShape(
                    "InferBinOpShape: rank of lhs and rhs must match.".to_string(),
                ));
            }
            let max = ashape
                .iter()
                .zip(bshape)
                .map(|(a, b)| T::from((*a).max(b)).unwrap())
                .collect::<Vec<T>>();
            ctx.append_output(NdArray::from_shape_vec(ndarray::IxDyn(&[a_rank]), max).unwrap())
        } else if !a_is_scalar {
            ctx.append_output(ashape_float.to_owned());
        } else {
            ctx.append_output(bshape_float.to_owned());
        }
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
    }
}

impl<T: Float> op::Op<T> for Shape {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let x = &ctx.input(0);
        let shape_vec = ndarray_ext::shape_of_view(x);
        let shape_t: Vec<T> = shape_vec.iter().map(|&s| T::from(s).unwrap()).collect();
        let ret = NdArray::from_shape_vec(ndarray::IxDyn(&[shape_vec.len()]), shape_t).unwrap();
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, None);
    }
}

impl<T: Float> op::Op<T> for Rank {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let x = ctx.input(0);
        let ret = NdArray::from_elem(ndarray::IxDyn(&[]), T::from(x.ndim()).unwrap());
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, None);
    }
}

impl<T: Float> op::Op<T> for Size {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let x = ctx.input(0);
        let ret = NdArray::from_elem(ndarray::IxDyn(&[]), T::from(x.len()).unwrap());
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, None);
    }
}

impl<T: Float> op::Op<T> for Reshape {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let x = &ctx.input(0);
        let shape_arr = &ctx.input(1);
        let target = shape_arr
            .iter()
            .map(|&dim_size| {
                if dim_size != -T::one() {
                    dim_size.to_usize().unwrap()
                } else {
                    let product: T = shape_arr.iter().fold(T::one(), |acc, &x| acc * x);
                    x.len() / product.neg().to_usize().unwrap()
                }
            })
            .collect::<Vec<_>>();
        // If x is *not* a c-contiguous, just copying it for now
        // due to current state of ndarray: https://github.com/rust-ndarray/ndarray/issues/390
        if x.is_standard_layout() {
            if let Ok(a) = x
                .clone()
                .into_shape_with_order(ndarray::IxDyn(target.as_slice()))
            {
                ctx.append_output(a.to_owned());
            } else {
                let copy = ndarray_ext::deep_copy(x);
                if let Ok(a) = copy.into_shape_with_order(ndarray::IxDyn(target.as_slice())) {
                    ctx.append_output(a);
                } else {
                    return Err(op::OpError::IncompatibleShape(format!(
                        "reshape failed: {:?} vs {:?}",
                        x.shape(),
                        target
                    )));
                }
            }
        } else if let Ok(a) =
            ndarray_ext::deep_copy(x).into_shape_with_order(ndarray::IxDyn(target.as_slice()))
        {
            ctx.append_output(a)
        } else {
            return Err(op::OpError::IncompatibleShape(format!(
                "reshape failed: {:?} vs {:?}",
                x.shape(),
                target
            )));
        }
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let gy = ctx.output_grad();
        let x = ctx.input(0);
        let gx = Tensor::builder(ctx.graph())
            .append_input(gy, false)
            .append_input(shape(x), false)
            .build(Reshape);
        ctx.append_input_grad(0, Some(gx));
        ctx.append_input_grad(1, None);
    }
}

impl<T: Float> op::Op<T> for SetDiff1D {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let x0 = ctx.input(0);
        let x1 = &ctx.input(1);

        let set_a: crate::FxHashSet<isize> = crate::FxHashSet::from_iter(
            x0.as_slice()
                .unwrap()
                .iter()
                .map(|&a| a.to_isize().unwrap()),
        );

        let set_b: crate::FxHashSet<isize> = crate::FxHashSet::from_iter(
            x1.as_slice()
                .unwrap()
                .iter()
                .map(|&a| a.to_isize().unwrap()),
        );

        let diff = set_a.difference(&set_b);

        let mut vec = diff.collect::<Vec<&isize>>();
        vec.sort();
        let vec = vec
            .into_iter()
            .map(|&a| T::from(a).unwrap())
            .collect::<Vec<T>>();
        let len = vec.len();
        // safe unwrap
        let ret = NdArray::from_shape_vec(ndarray::IxDyn(&[len]), vec).unwrap();
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
    }
}

impl<T: Float> op::Op<T> for IndexOp {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let x = ctx.input(0);
        let i = if self.index < 0 {
            ((x.len() as isize) + self.index) as usize
        } else {
            self.index as usize
        };
        // unwrap is safe
        let flat_x = x.view().into_shape_with_order(x.len()).unwrap();
        if let Some(ret) = flat_x.get(i) {
            ctx.append_output(ndarray::arr0(*ret).into_dyn());
            Ok(())
        } else {
            Err(op::OpError::OutOfBounds(format!(
                "access_elem: tried to access index {} in tensor of length {} (shape: {:?})",
                i,
                x.len(),
                x.shape(),
            )))
        }
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let op = IndexOpGrad { index: self.index };
        let x = ctx.input(0);
        let gy = ctx.output_grad();
        let gx = Tensor::builder(ctx.graph())
            .setshape(&shape(x))
            .append_input(x, false)
            .append_input(gy, false)
            .build(op);
        ctx.append_input_grad(0, Some(gx));
    }
}

impl<T: Float> op::Op<T> for IndexOpGrad {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let x = ctx.input(0);
        let gy = &ctx.input(1);
        let mut result = NdArray::zeros(x.shape());
        let i = if self.index < 0 {
            ((x.len() as isize) + self.index) as usize
        } else {
            self.index as usize
        };
        // unwrap is safe
        let len = result.len();
        if let Some(a) = result
            .view_mut()
            .into_shape_with_order(len)
            .unwrap() // safe unwrap
            .get_mut(i)
        {
            *a = gy[ndarray::IxDyn(&[])];
        } else {
            return Err(op::OpError::OutOfBounds(format!(
                "access_elem: tried to access index {} in tensor of length {} (shape: {:?})",
                i,
                x.len(),
                x.shape(),
            )));
        }
        ctx.append_output(result);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, None);
    }
}

impl<T: Float> op::Op<T> for Gather {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let param = &ctx.input(1);
        let indices = &ctx.input(0);
        let indicesshape = indices.shape();
        let paramshape = param.shape();
        let axis = ndarray_ext::normalize_negative_axis(self.axis, param.ndim());

        let outputshape: Vec<usize> = {
            let former: &[usize] = &paramshape[..axis];
            let latter: &[usize] = &paramshape[axis + 1..];
            // doing former + indices.shape() + latter
            former
                .iter()
                .chain(indicesshape)
                .chain(latter)
                .cloned()
                .collect()
        };

        let flat_indices = if self.should_normalize_negative_indices {
            ndarray_ext::normalize_negative_axes(indices, paramshape[axis])
        } else {
            indices
                .map(|a| a.to_usize().expect("Invalid index value"))
                .iter()
                .cloned()
                .collect::<Vec<_>>()
        };
        let selected = ndarray_ext::select(param, ndarray::Axis(axis), flat_indices.as_slice());
        let ret = selected
            .into_shape_with_order(outputshape.as_slice())
            .unwrap();
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let x = ctx.input(0);
        let x1 = ctx.input(1);
        let gy = ctx.output_grad();
        let gx = Tensor::builder(ctx.graph())
            .append_input(x, false)
            .append_input(x1, false)
            .append_input(gy, false)
            .setshape(&shape(x))
            .build(GatherGrad { axis: self.axis });
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, Some(gx));
    }
}

impl<T: Float> op::Op<T> for GatherGrad {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let indices = ctx.input(0);
        let param = &ctx.input(1);
        let paramshape = param.shape();
        let gy = &ctx.input(2);
        let axis = if self.axis == -1 {
            param.ndim()
        } else {
            self.axis as usize
        };

        // get read-only view of gy and reshape it
        let gy = {
            let former = &paramshape[..axis];
            let latter = &paramshape[axis + 1..];
            let shape: Vec<usize> = former
                .iter()
                .chain(&[indices.len()])
                .chain(latter)
                .cloned()
                .collect();
            gy.view().into_shape_with_order(shape).unwrap()
        };

        let mut gx = NdArray::zeros(param.shape());

        for (gy_sub, &i) in gy.axis_iter(ndarray::Axis(axis)).zip(indices) {
            let i = i.to_isize().unwrap();
            // get gx's sub view
            let gx_sliced = unsafe {
                gx.slice_mut(
                    ndarray::SliceInfo::<_, ndarray::IxDyn, ndarray::IxDyn>::new(
                        (0..param.ndim())
                            .map(|dim| {
                                if dim == axis {
                                    SliceInfoElem::Slice {
                                        start: i,
                                        end: Some(i + 1),
                                        step: 1,
                                    }
                                } else {
                                    SliceInfoElem::Slice {
                                        start: 0,
                                        end: None,
                                        step: 1,
                                    }
                                }
                            })
                            .collect::<Vec<_>>(),
                    )
                    .unwrap()
                    .as_ref(),
                )
            };

            // squeeze
            let mut gx_sliced = gx_sliced.index_axis_move(ndarray::Axis(axis), 0);
            // assign gy to sliced view
            gx_sliced.zip_mut_with(&gy_sub, |gx, &gy| {
                *gx += gy;
            });
        }

        ctx.append_output(gx);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
        ctx.append_input_grad(2, None);
    }
}

#[cfg(feature = "blas")]
pub(crate) fn inplace_add_impl<F: Float>(mut a: NdArrayViewMut<F>, b: &NdArrayView<F>) {
    use crate::same_type;
    use crate::tensor_ops::blas_ffi::{vdAdd, vsAdd, MklInt};
    unsafe {
        if same_type::<F, f32>() {
            vsAdd(
                a.len() as MklInt,
                a.as_ptr() as *const f32,
                b.as_ptr() as *const f32,
                a.as_mut_ptr() as *mut f32,
            );
        } else if same_type::<F, f64>() {
            vdAdd(
                a.len() as MklInt,
                a.as_ptr() as *const f64,
                b.as_ptr() as *const f64,
                a.as_mut_ptr() as *mut f64,
            );
            return;
        } else {
            a += b;
        }
    }
}

impl<T: Float> op::Op<T> for AddN {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let inputs_len = ctx.inputs().len();
        if 0 == inputs_len {
            unreachable!()
        } else if 1 == inputs_len {
            let ret = ctx.input(0);
            ctx.append_output(ret.to_owned());
        } else if 2 == inputs_len {
            let ret = &ctx.input(0) + &ctx.input(1);
            ctx.append_output(ret);
        } else {
            let mut base = &ctx.input(0) + &ctx.input(1);
            for i in 2..inputs_len {
                base += &ctx.input(i);
            }
            ctx.append_output(base);
        }
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let gy = ctx.output_grad().to_owned();
        for i in 0..ctx.inputs().len() {
            ctx.append_input_grad(i, Some(gy));
        }
    }
}

impl<T: Float> op::Op<T> for Clip<T> {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let ret = ctx.input(0).map(move |a| a.min(self.max).max(self.min));
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let gy = ctx.output_grad();
        let x0 = ctx.input(0);
        let gx = Tensor::builder(ctx.graph())
            .setshape(&shape(gy))
            .append_input(x0, false)
            .append_input(gy, false)
            .build(ClipGrad {
                min: self.min,
                max: self.max,
            });
        ctx.append_input_grad(0, Some(gx));
    }
}

impl<T: Float> op::Op<T> for ClipGrad<T> {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let mut ret = ctx.input(0).mapv(move |x| {
            // x > min && x < max
            T::from((((x > self.min) as i32) as f32) * (((x < self.max) as i32) as f32)).unwrap()
        });
        ret *= &ctx.input(1);
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
    }
}

impl<T: Float> op::Op<T> for Concat {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let mut views = Vec::with_capacity(ctx.inputs().len());
        for i in 0..ctx.inputs().len() {
            views.push(ctx.input(i));
        }

        let axis = if self.axis < 0 {
            (ctx.input(0).ndim() as isize + self.axis) as usize
        } else {
            self.axis as usize
        };

        match ndarray::concatenate(ndarray::Axis(axis), views.as_slice()) {
            Ok(y) => {
                ctx.append_output(y);
                Ok(())
            }
            Err(e) => Err(op::OpError::NdArrayError("concat".to_string(), e)),
        }
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        // [x1, x2, x3, ..., gy]
        let num_inputs = ctx.inputs().len();
        let output_grad = ctx.output_grad();
        let graph = ctx.graph();
        let input0shape = shape(ctx.input(0));

        // Clone all inputs to avoid borrow issues
        let inputs: Vec<&Tensor<T>> = (0..num_inputs).map(|i| ctx.input(i)).collect();

        for i in 0..num_inputs {
            let mut builder = Tensor::builder(graph)
                .setshape(&input0shape)
                .append_input(output_grad, false);

            for input in &inputs {
                builder = builder.append_input(input, false);
            }

            let gx = builder.build(ConcatGrad {
                index: i,
                axis: self.axis,
            });
            ctx.append_input_grad(i, Some(gx));
        }
    }
}

impl<T: Float> op::Op<T> for ConcatGrad {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let gy = ctx.input(0);

        let axis = if self.axis < 0 {
            (ctx.input(0).ndim() as isize + self.axis) as usize
        } else {
            self.axis as usize
        };

        // make slice indices
        let mut start_idx = 0;
        for i in 1..self.index {
            start_idx += ctx.input(i).shape()[axis];
        }
        let region_len = ctx.input(self.index + 1).shape()[axis] as isize;
        let indices = (0..gy.ndim())
            .map(move |_axis| {
                if _axis == axis {
                    // partial region
                    SliceInfoElem::Slice {
                        start: start_idx as isize,
                        end: Some(region_len),
                        step: 1,
                    }
                } else {
                    // full slice
                    SliceInfoElem::Slice {
                        start: 0,
                        end: None,
                        step: 1,
                    }
                }
            })
            .collect::<Vec<_>>();

        // Clone the *view*
        unsafe {
            match ndarray::SliceInfo::<_, ndarray::IxDyn, ndarray::IxDyn>::new(indices) {
                Ok(ok) => {
                    // do slice
                    let ret = gy.clone().slice_move(ok.as_ref());
                    ctx.append_output(ret.to_owned());
                    Ok(())
                }
                Err(e) => Err(op::OpError::NdArrayError("ConcatGrad: ".to_string(), e)),
            }
        }
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let inputs = ctx.inputs();
        for i in 0..inputs.len() {
            ctx.append_input_grad(i, None);
        }
    }
}

impl<T: Float> op::Op<T> for Tile {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let x = ctx.input(0);
        let axis = ndarray_ext::normalize_negative_axis(self.axis, x.ndim());
        let views = vec![x.clone(); self.num];
        match ndarray::concatenate(ndarray::Axis(axis), views.as_slice()) {
            Ok(ret) => {
                ctx.append_output(ret);
                Ok(())
            }
            Err(e) => Err(op::OpError::NdArrayError("tile: ".to_string(), e)),
        }
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, Some(reduce_sum(ctx.output_grad(), &[self.axis], true)));
    }
}

impl<T: Float> op::Op<T> for Split {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let x = &ctx.input(0);
        let axis = ndarray_ext::normalize_negative_axis(self.axis, x.ndim());
        let mut ret = x.clone();
        let indices = make_indices_for_split(x, self.start_index, self.end_index, axis);
        ret.slice_collapse(indices.as_slice());
        ctx.append_output(ret.to_owned());
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let op = SplitGrad {
            axis: self.axis,
            start_index: self.start_index,
            end_index: self.end_index,
        };
        let x = ctx.input(0);
        let gy = ctx.output_grad();
        let gx = Tensor::builder(ctx.graph())
            .append_input(x, false)
            .append_input(gy, false)
            .setshape(&shape(x))
            .build(op);
        ctx.append_input_grad(0, Some(gx));
    }
}

impl<T: Float> op::Op<T> for SplitGrad {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let x = ctx.input(0);
        let mut gx = NdArray::zeros(x.shape());

        let axis = ndarray_ext::normalize_negative_axis(self.axis, x.ndim());
        let indices = make_indices_for_split(&x, self.start_index, self.end_index, axis);

        unsafe {
            gx.slice_mut(
                ndarray::SliceInfo::<_, ndarray::IxDyn, ndarray::IxDyn>::new(indices)
                    .unwrap()
                    .as_ref(),
            )
            .zip_mut_with(&ctx.input(1), |a, &g| *a = g);
        }
        ctx.append_output(gx);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, None);
    }
}

#[inline]
#[allow(dead_code)]
fn make_indices_for_split<T: Float>(
    x: &NdArrayView<T>,
    start_index: isize,
    end_index: isize,
    axis: usize,
) -> Vec<SliceInfoElem> {
    let ndim = x.ndim();
    assert!(ndim > axis, "Wrong split axis");
    (0..ndim)
        .map(|i| {
            if i == axis {
                SliceInfoElem::Slice {
                    start: start_index,
                    end: Some(end_index),
                    step: 1,
                }
            } else {
                // full slice
                SliceInfoElem::Slice {
                    start: 0,
                    end: None,
                    step: 1,
                }
            }
        })
        .collect::<Vec<_>>()
}

impl<T: Float> op::Op<T> for Slice {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let mut y = ctx.input(0);
        y.slice_collapse(self.indices.as_slice());
        ctx.append_output(y.to_owned());
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let op = SliceGrad {
            indices: self.indices.clone(),
        };
        let x = ctx.input(0);
        let gy = ctx.output_grad();
        let gx = Tensor::builder(ctx.graph())
            .append_input(x, false)
            .append_input(gy, false)
            .setshape(&shape(x))
            .build(op);
        ctx.append_input_grad(0, Some(gx));
    }
}

impl<T: Float> op::Op<T> for SliceGrad {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let x = ctx.input(0);
        let mut gx = NdArray::zeros(x.shape());
        // sliced view
        unsafe {
            gx.slice_mut(
                ndarray::SliceInfo::<&[SliceInfoElem], ndarray::IxDyn, ndarray::IxDyn>::new(
                    &self.indices,
                )
                .unwrap()
                .as_ref(),
            )
            .zip_mut_with(&ctx.input(1), |a, &g| *a = g);
        }
        ctx.append_output(gx);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        // is this ok?
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
    }
}
impl<T: Float> op::Op<T> for Squeeze {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let mut x = ctx.input(0).clone();
        let mut axes = ctx
            .input(1)
            .iter()
            .map(|a| a.to_isize().unwrap())
            .collect::<Vec<_>>();
        axes.sort();
        for (adjust, &i) in axes.iter().enumerate() {
            let axis = if i < 0 {
                (x.ndim() as isize + i) as usize
            } else {
                i as usize
            };
            let axis = axis - adjust;
            assert_eq!(1, x.shape()[axis], "Can't squeeze a dim whose size != 1");
            // axis making ok
            x = x.index_axis_move(ndarray::Axis(axis), 0);
        }
        ctx.append_output(x.to_owned());
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, Some(expand_dims(ctx.output_grad(), ctx.input(1))));
        ctx.append_input_grad(1, None);
    }
}

impl<T: Float> op::Op<T> for ExpandDims {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let ret = ctx.input(0);
        let mut axes = ctx
            .input(1)
            .iter()
            .map(|a| a.to_isize().unwrap())
            .collect::<Vec<_>>();
        axes.sort();
        let mut outputshape = ret.shape().to_vec();
        for &i in axes.iter() {
            let axis = if i < 0 {
                (ret.ndim() as isize + i) as usize
            } else {
                i as usize
            };
            outputshape.insert(axis, 1);
        }
        ctx.append_output(ret.into_shape_with_order(outputshape).unwrap().to_owned());
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, Some(squeeze(ctx.output_grad(), ctx.input(1))));
        ctx.append_input_grad(1, None);
    }
}
