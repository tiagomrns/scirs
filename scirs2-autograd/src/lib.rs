#![allow(deprecated)]
//! Differentiable operations and tensors backed by [ndarray](https://github.com/rust-ndarray/ndarray).
#![recursion_limit = "1024"]
//!
//! ## Enabling blas
//! If you use basic linalg operations, especially matrix multiplications, `blas` feature would be important to speed them up.
//!
//! ```toml
//! [dependencies]
//! scirs2-autograd = {"<version>", features = ["blas", "<blas-implementation-choice>"] }
//! ```
//! `<blas-implementation-choice>` must be one of the following:
//! - `openblas`
//! - `netlib`
//!
//! ## Features
//! ### Reverse-mode automatic differentiation using lazy tensors
//! Here we are just computing partial derivatives of `z = 2x^2 + 3y + 1`.
//!
//! ```
//! use scirs2_autograd as ag;
//! use ag::tensor_ops as T;
//!
//! # fn main() {
//! ag::run(|ctx: &mut ag::Context<_>| {
//!     let x = ctx.placeholder("x", &[]);
//!     let y = ctx.placeholder("y", &[]);
//!     let z = 2.*x*x + 3.*y + 1.;
//!
//!     // dz/dy
//!     let gy = &T::grad(&[z], &[y])[0];
//!     println!("{:?}", gy.eval(ctx));   // => Ok(3.)
//!
//!     // dz/dx (requires to fill the placeholder `x`)
//!     let gx = &T::grad(&[z], &[x])[0];
//!     let feed = ag::ndarray::arr0(2.);
//!     println!("{:?}", ctx.evaluator().push(gx).feed(x, feed.view().into_dyn()).run()[0]);  // => Ok(8.)
//!
//!     // ddz/dx (differentiates `z` again)
//!     let ggx = &T::grad(&[gx], &[x])[0];
//!     println!("{:?}", ggx.eval(ctx));  // => Ok(4.)
//! });
//! # }
//! ```
//!
//! ### Neural networks
//! This crate has various low-level features inspired by tensorflow/theano to train neural networks.
//! Since computation graphs require only bare minimum of heap allocations, the overhead is small, even for complex networks.
//! ```
//! // MNIST digits classification model with multi-layer-perceptron
//! use scirs2_autograd as ag;
//! use ag::optimizers::adam::Adam;
//! use ag::tensor_ops::*;
//! use ag::prelude::*;
//!
//! let mut env = ag::VariableEnvironment::new();
//!
//! let mut rng = ag::ndarray_ext::ArrayRng::<f32>::default();
//!
//! // Register variables in the default namespace
//! env.name("w").set(rng.glorot_uniform(&[28 * 28, 10]));
//! env.name("b").set(ag::ndarray_ext::zeros(&[1, 10]));
//!
//! let adam = Adam::default("my_adam", env.default_namespace().current_var_ids(), &mut env);
//!
//! for epoch in 0..3 { // 0.11 sec/epoch on 2.7GHz Intel Core i5
//!     env.run(|ctx| {
//!         let x = ctx.placeholder("x", &[-1, 28*28]);
//!         let y = ctx.placeholder("y", &[-1]);
//!         let w = ctx.variable("w");
//!         let b = ctx.variable("b");
//!         let z = matmul(x, w) + b;
//!         let mean_loss = reduce_mean(sparse_softmax_cross_entropy(z, &y), &[0], false);
//!         let grads = &grad(&[mean_loss], &[w, b]);
//!
//!         // let mut feeder = ag::Feeder::new();
//!         // feeder.push(x, x_batch).push(y, y_batch);
//!         // adam.update(&[w, b], grads, ctx, feeder);
//!     });
//! }
//! ```
//!
//! ### Abstractions
//! ```
//! use scirs2_autograd as ag;
//! use ag::tensor_ops::*;
//! use ag::ndarray;
//!
//! // Use `Tensor::map()` to create a new ndarray
//! ag::run(|ctx| {
//!     let x = ones(&[2, 3], ctx);
//!     // apply ndarray's methods
//!     let y = x.map(|x| x.fold_axis(ndarray::Axis(0), 0.0, |acc, x| acc + x));
//!     let z = x.map(|x| ag::ndarray_ext::zeros(x.shape()));
//! });
//!
//! // Hooks
//! ag::run(|ctx| {
//!     let x: ag::Tensor<f32> = ones(&[2, 3], ctx).showshape();
//!     let y: ag::Tensor<f32> = ones(&[2, 3], ctx).raw_hook(|x| println!("{}", x));
//! });
//! ```
//!
//! ### Other useful features
//! - [Model persistence](variable#model-persistence)
//! - [Variable namespace](variable#variable-and-namespace)

#[allow(unused_imports)]
// Expose to prevent version conflict
#[macro_use(s)]
/// re-exported for convenience and version-compatibility
pub extern crate ndarray;

// BLAS dependencies now handled through scirs2-core

extern crate approx;
extern crate libc;
extern crate matrixmultiply;
extern crate num;
extern crate num_traits;
/// re-exported for convenience and version-compatibility
pub extern crate rand;
extern crate rand_distr;
// extern crate rayon;  // Now use scirs2-core parallel abstractions
extern crate rustc_hash;
extern crate serde;
extern crate serde_json;
pub(crate) extern crate smallvec;
extern crate special;
extern crate uuid;

pub mod error;
pub mod evaluation;
mod gradient;
pub mod gradient_clipping;
pub mod graph;
pub mod hooks;
pub mod integration;
pub mod ndarray_ext;
pub mod op;
pub mod optimization;
pub mod optimizers;
pub mod parallel;
pub mod prelude;
pub mod schedulers;
pub mod tensor;
pub mod tensor_ops;
pub mod test_helper;
pub mod testing;
pub mod tracing;
pub mod variable;
pub mod visualization;

use rustc_hash::{FxHashMap, FxHashSet};
use std::any::TypeId;
use std::fmt;

/// A primitive type in this crate, which is actually a decorated `num_traits::Float`.
pub trait Float:
    num_traits::Float
    + num_traits::NumAssignOps
    + Copy
    + Send
    + Sync
    + fmt::Display
    + fmt::Debug
    + Sized
    + serde::Serialize
    + serde::de::DeserializeOwned
    + 'static
{
}

#[doc(hidden)]
/// Internal trait.
pub trait Int:
    num::Integer
    + num_traits::NumAssignOps
    + num_traits::ToPrimitive
    + Copy
    + Send
    + fmt::Display
    + Sized
    + serde::Serialize
    + serde::de::DeserializeOwned
    + 'static
{
}

impl<T> Float for T where
    T: num::Float
        + num_traits::NumAssignOps
        + Copy
        + Send
        + Sync
        + fmt::Display
        + fmt::Debug
        + Sized
        + serde::Serialize
        + serde::de::DeserializeOwned
        + 'static
{
}

impl<T> Int for T where
    T: num::Integer
        + num_traits::NumAssignOps
        + num_traits::ToPrimitive
        + Copy
        + Send
        + Sync
        + fmt::Display
        + Sized
        + serde::Serialize
        + serde::de::DeserializeOwned
        + 'static
{
}

#[inline(always)]
/// Return `true` if `A` and `B` are the same type
pub(crate) fn same_type<A: 'static, B: 'static>() -> bool {
    TypeId::of::<A>() == TypeId::of::<B>()
}

pub use crate::ndarray_ext::array_gen;

pub use crate::ndarray_ext::{NdArray, NdArrayView, NdArrayViewMut};

pub use crate::evaluation::{Evaluator, Feeder};

pub use crate::tensor::Tensor;

pub(crate) use graph::Graph;

pub use crate::error::{AutogradError, EvalError, OpError, Result};
pub use crate::graph::{run, Context};
pub use crate::variable::VariableEnvironment;
