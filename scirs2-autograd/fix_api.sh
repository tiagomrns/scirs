#!/bin/bash

# Fix all the linear algebra ops files
for file in decomposition_ops.rs eigen_ops.rs linalg_ops.rs matrix_functions.rs matrix_ops.rs norm_ops.rs scalar_ops.rs solver_ops.rs special_matrices.rs; do
    echo "Fixing $file..."
    sed -i "s/use crate::ops::\*/use crate::op::*/g" src/tensor_ops/$file
    sed -i "s/use crate::tensor::{Tensor, TensorBuilder}/use crate::tensor::Tensor/g" src/tensor_ops/$file
    sed -i "s/fn name(&self) -> &str {/fn name(\&self) -> \&'static str {/g" src/tensor_ops/$file
    sed -i "s/fn compute(&self, ctx: &mut crate::op::ComputeContext<F>) -> crate::Result<()> {/fn compute(\&self, ctx: \&mut ComputeContext<F>) -> Result<(), OpError> {/g" src/tensor_ops/$file
    sed -i "s/fn grad(&self, ctx: &mut crate::op::GradientContext<F>) -> crate::Result<()> {/fn grad(\&self, ctx: \&mut GradientContext<F>) {/g" src/tensor_ops/$file
    sed -i "s/return Err(crate::error::AutogradError::Op(/return Err(OpError::Other(/g" src/tensor_ops/$file
    sed -i "s/.map_err(|_| crate::error::AutogradError::Op(/.map_err(|_| OpError::Other(/g" src/tensor_ops/$file
    sed -i "s/as_array_view()/as_array().view()/g" src/tensor_ops/$file
    sed -i "s/let input = &ctx.input(/let input = ctx.input(/g" src/tensor_ops/$file
    sed -i "s/let x = &ctx.input(/let x = ctx.input(/g" src/tensor_ops/$file
    sed -i "s/let y = &ctx.input(/let y = ctx.input(/g" src/tensor_ops/$file
    sed -i "s/let gy = ctx.output_grad()/let gy = ctx.output_grad().as_ref()/g" src/tensor_ops/$file
    sed -i "s/let gx = ctx.output_grad()/let gx = ctx.output_grad().as_ref()/g" src/tensor_ops/$file
    sed -i "s/let q = ctx.output(/let q = ctx.output(/g" src/tensor_ops/$file
    sed -i "s/let r = ctx.output(/let r = ctx.output(/g" src/tensor_ops/$file
    sed -i "s/let x = ctx.input(/let x = ctx.input(/g" src/tensor_ops/$file
    sed -i "s/let a = ctx.input(/let a = ctx.input(/g" src/tensor_ops/$file
    sed -i "s/let b = ctx.input(/let b = ctx.input(/g" src/tensor_ops/$file
    sed -i "s/let shape = input.shape()/let shape = ctx.input_shape(0)/g" src/tensor_ops/$file
    sed -i "s/Ok(())//" src/tensor_ops/$file
    sed -i "s/import Axis//" src/tensor_ops/$file
done

echo "Done!">