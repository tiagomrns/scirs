use crate::Float;
use crate::{op, Context};
use crate::{NdArray, NdArrayView};

use crate::error::OpError;
use crate::graph::{AsGraph, Graph, TensorID};
use crate::op::{GradientContext, SmallVec};
use crate::variable::VariableID;
use std::cell::Ref;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};

/// Lazy-evaluated multi-dimensional array
///
/// Similar to numpy array, but is designed a bit differently:
/// - `Tensor` itself doesn't have its content
///   - cheap to `Copy`
///   - Lazily evaluated to [ndarray::Array], i.e. the value is obtained only after call `Tensor::eval`, `Evaluator::run`, or `Optimizer::update`.
/// - `Tensor` belongs to a particular [Graph] and is shorter lived than the graph
///   - Note the lifetime parameter: `Tensor<'graph_>`
///   - The graph is always wrapped in [Context]
/// - `Tensor` can behave as a trainable variable when in-place operations are applied
///   - See [crate::variable]
///
/// ### Basic usages
///    ```
/// use scirs2_autograd as ag;
/// use ag::tensor_ops as T;
/// use ag::prelude::*;
///
/// ag::run(|ctx: &mut ag::Context<_>| {  // `Context` is required to play with tensors
///     // Create a random tensor with shape [2, 3] in this context
///     let random: ag::Tensor<f64> = T::standard_normal(&[2, 3], ctx);
///
///     // Binary operators are implemented
///     let mul = random * 3.;
///
///     // Tensor is evaluated as an ndarray::Array<T, IxDyn>.
///     type NdArray = ag::NdArray<f64>;
///     let mul_val: Result<NdArray, ag::EvalError> = mul.eval(ctx);
///
///     // Internally used ndarray::ArrayView allows reshaping tensors without copying
///     let reshaped = T::reshape(random, &[6]);
///
///     // Evaluating multiple tensors at once.
///     // Note that although `random` node is required two times in this computation graph,
///     // it's evaluated only once since `Evaluator` is smart enough to avoid duplicated computations.
///     let pair: Vec<Result<NdArray, ag::EvalError>> = ctx.evaluator().extend(&[mul, reshaped]).run();
/// });
///    ```
#[derive(Clone, Copy)]
pub struct Tensor<'graph, F: Float> {
    pub(crate) id: TensorID, // tensor id in the graph
    pub(crate) graph: &'graph Graph<F>,
}

impl<F: Float + std::fmt::Debug> std::fmt::Debug for Tensor<'_, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("id", &self.id)
            .field("is_source", &self.is_source())
            .field("is_differentiable", &self.is_differentiable())
            .finish()
    }
}

impl<F: Float> PartialEq for Tensor<'_, F> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && std::ptr::eq(self.graph, other.graph)
    }
}

impl<'graph, F: Float> Tensor<'graph, F> {
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn get_incoming_tensors(&self) -> Ref<SmallVec<IncomingTensor>> {
        Ref::map(self.inner(), |x| &x.incoming_nodes)
    }

    // Returns the i-th input node of this tensor
    pub(crate) fn get_incoming_tensor(
        &self,
        i: usize,
        g: &'graph Graph<F>,
    ) -> Option<Tensor<'graph, F>> {
        self.inner().incoming_nodes.get(i).map(|x| x.as_tensor(g))
    }

    #[inline(always)]
    pub(crate) fn inner(&self) -> Ref<TensorInternal<F>> {
        self.graph.access_inner(self.id)
    }

    /// Returns the graph to which this tensor belongs.
    #[inline]
    pub(crate) fn graph(&self) -> &'graph Graph<F> {
        self.graph
    }

    /// Evaluates this tensor as an `ndarray::Array<F, ndarray::IxDyn>`.
    ///
    ///    ```
    /// use ndarray::array;
    /// use scirs2_autograd as ag;
    /// use ag::tensor_ops as T;
    ///
    /// ag::run(|c| {
    ///    let a = T::zeros(&[2], c);
    ///    assert_eq!(a.eval(c), Ok(array![0., 0.].into_dyn()));
    /// });
    ///    ```
    ///
    /// See also [Evaluator](../evaluation/struct.Evaluator.html).
    pub fn eval(&self, ctx: &Context<F>) -> Result<NdArray<F>, crate::EvalError> {
        crate::graph::assert_same_graph(ctx, self.graph);
        // Use the evaluator directly for now to avoid more complex changes
        let result = ctx.evaluator().eval(self);
        result
    }

    /// Ensures that this tensor is evaluated after the arguments.
    ///
    /// You can use the return value instead of `self`.
    /// Panics if `self` is a source node, such as a variable or placeholder.
    ///
    ///    ```
    /// use ndarray::array;
    /// use scirs2_autograd as ag;
    /// use ag::tensor_ops as T;
    /// use ag::prelude::*;
    ///
    /// ag::run(|c| {
    ///     // Create two independent computations
    ///     let a = T::convert_to_tensor(array![1., 2.], c);
    ///     let b = T::convert_to_tensor(array![3., 4.], c);
    ///     
    ///     // These operations are independent
    ///     let mul_a = a * 2.;
    ///     let mul_b = b * 3.;
    ///     
    ///     // Force mul_c to depend on both mul_a and mul_b
    ///     // This ensures mul_a and mul_b are evaluated before mul_c
    ///     let d = T::convert_to_tensor(array![5., 6.], c);
    ///     let mul_c = (d * 4.).depends_on(&[mul_a, mul_b]);
    ///     
    ///     // Evaluation order is now guaranteed: mul_a, mul_b, then mul_c
    ///     assert_eq!(mul_c.eval(c), Ok(array![20., 24.].into_dyn()));
    /// });
    ///    ```
    #[inline]
    pub fn depends_on<A>(self, on: &[A]) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        crate::tensor_ops::control_dependencies(self, on)
    }

    /// Creates a new [TensorBuilder](struct.TensorBuilder.html).
    #[inline]
    pub fn builder(graph: &'graph impl AsGraph<F>) -> TensorBuilder<'graph, F> {
        // Starts with default values
        TensorBuilder {
            graph: graph.as_graph(),
            shape: None,
            in_nodes: SmallVec::new(),
            differentiable: true,
            placeholder_name: None,
            backprop_inputs: None,
            knownshape: None,
            variable_id: None,
        }
    }

    /// Applies the given function to `x` and creates a new tensor.
    ///
    /// Useful in cases where you need to create a tensor using a run-time value of `x`.
    ///    ```
    /// use scirs2_autograd as ag;
    /// use ag::tensor_ops::*;
    ///
    /// ag::run(|g| {
    ///     let x: ag::Tensor<f32> = standard_uniform(&[2, 3], g);
    ///     let sin = x.map(|arr| arr.map(|elem| elem.sin()));
    ///
    ///     sin.eval(g);
    /// });
    ///    ```
    pub fn map(&self, f: fn(NdArrayView<F>) -> NdArray<F>) -> Tensor<'graph, F> {
        crate::tensor_ops::map(self, f)
    }

    /// Registers a hook on the receiver tensor.
    ///
    ///    ```
    /// use scirs2_autograd as ag;
    /// use ag::tensor_ops::*;
    ///
    /// ag::run(|g| {
    ///     let a: ag::Tensor<f32> = zeros(&[4, 2], g).register_hook(ag::hooks::Show);
    ///     let b: ag::Tensor<f32> = ones(&[2, 3], g).register_hook(ag::hooks::ShowShape);
    ///     let c = matmul(a, b);
    ///
    ///     c.eval(g);
    ///     // [[0.0, 0.0],
    ///     // [0.0, 0.0],
    ///     // [0.0, 0.0],
    ///     // [0.0, 0.0]] shape=[4, 2], strides=[2, 1], layout=C (0x1)
    ///
    ///     // [2, 3]
    /// });
    ///    ```
    #[inline]
    pub fn register_hook<H: crate::hooks::Hook<F> + 'static>(self, hook: H) -> Tensor<'graph, F> {
        Tensor::builder(self.graph)
            .append_input(self, false)
            .build(crate::tensor_ops::hook_ops::HookOp::new(hook))
    }

    /// Sets a hook that displays the evaluation result of the receiver tensor to stdout.
    ///
    ///    ```
    /// use scirs2_autograd as ag;
    /// use ag::tensor_ops::*;
    ///
    /// ag::run(|g| {
    ///     let a: ag::Tensor<f32> = zeros(&[4, 2], g).show();
    ///     a.eval(g);
    ///     // [[0.0, 0.0],
    ///     // [0.0, 0.0],
    ///     // [0.0, 0.0],
    ///     // [0.0, 0.0]] shape=[4, 2], strides=[2, 1], layout=C (0x1)
    ///     });
    ///    ```
    #[inline]
    pub fn show(self) -> Tensor<'graph, F> {
        self.register_hook(crate::hooks::Show)
    }

    /// Sets a hook that displays the evaluation result of the receiver tensor to stdout, with the given prefix.
    ///
    ///    ```
    /// use scirs2_autograd as ag;
    /// use ag::tensor_ops::*;
    ///
    /// ag::run(|g| {
    ///     let a: ag::Tensor<f32> = zeros(&[4, 2], g).show_prefixed("My value:");
    ///     a.eval(g);
    ///     // My value:
    ///     // [[0.0, 0.0],
    ///     // [0.0, 0.0],
    ///     // [0.0, 0.0],
    ///     // [0.0, 0.0]] shape=[4, 2], strides=[2, 1], layout=C (0x1)
    /// });
    ///
    ///    ```
    #[inline]
    pub fn show_prefixed(self, prefix: &'static str) -> Tensor<'graph, F> {
        self.register_hook(crate::hooks::ShowPrefixed(prefix))
    }

    /// Sets a hook that displays the shape of the evaluated receiver tensor to stdout.
    ///
    ///    ```
    /// use scirs2_autograd as ag;
    /// use ag::tensor_ops::*;
    ///
    /// ag::run(|g| {
    ///     let a: ag::Tensor<f32> = zeros(&[2, 3], g).showshape();
    ///     a.eval(g);
    ///     // [2, 3]
    /// });
    ///    ```
    #[inline]
    pub fn showshape(self) -> Tensor<'graph, F> {
        self.register_hook(crate::hooks::ShowShape)
    }

    /// Sets a hook that displays the shape of the evaluated receiver tensor to stdout, with the given prefix.
    ///
    ///    ```
    /// use scirs2_autograd as ag;
    /// use ag::tensor_ops::*;
    ///
    /// ag::run(|g| {
    ///     let a: ag::Tensor<f32> = zeros(&[2, 3], g).show_prefixedshape("My shape:");
    ///     a.eval(g);
    ///     // My shape:
    ///     // [2, 3]
    /// });
    ///    ```
    #[inline]
    pub fn show_prefixedshape(self, prefix: &'static str) -> Tensor<'graph, F> {
        self.register_hook(crate::hooks::ShowPrefixedShape(prefix))
    }

    /// Sets a hook that displays the given string after evaluation of the receiver tensor.
    ///
    ///    ```
    /// use scirs2_autograd as ag;
    /// use ag::tensor_ops::*;
    ///
    /// ag::run(|g| {
    ///     let a: ag::Tensor<f32> = zeros(&[2, 3], g).print("This is `a`");
    ///     a.eval(g);
    ///     // This is `a`
    /// });
    ///    ```
    #[inline]
    pub fn print(self, what: &'static str) -> Tensor<'graph, F> {
        self.register_hook(crate::hooks::Print(what))
    }

    /// Sets a hook that calls the given closure after evaluation of the receiver tensor.
    ///
    ///    ```
    /// use scirs2_autograd as ag;
    /// use ag::tensor_ops::*;
    ///
    /// ag::run(|g| {
    ///     let a: ag::Tensor<f32> = zeros(&[2, 3], g)
    ///         .raw_hook(|arr| println!("{:?}", arr));
    ///
    ///     a.eval(g);
    /// });
    ///    ```
    #[inline]
    pub fn raw_hook<FUN: Fn(&NdArrayView<F>) + 'static + Send + Sync>(
        self,
        f: FUN,
    ) -> Tensor<'graph, F> {
        self.register_hook(crate::hooks::Raw {
            raw: f,
            phantom: PhantomData,
        })
    }

    /// Returns the id of this tensor in this graph.
    #[inline(always)]
    pub fn id(&self) -> usize {
        self.id
    }

    /// Returns the number of inputs of this tensor.
    #[inline]
    pub fn num_inputs(&self) -> usize {
        self.inner().num_inputs()
    }

    /// Returns the number of inputs of this tensor.
    #[inline]
    pub fn num_backprop_inputs(&self) -> usize {
        let inner = self.inner();
        inner
            .backprop_inputs
            .as_ref()
            .unwrap_or(&inner.incoming_nodes)
            .len()
    }

    #[inline]
    /// Returns true if this node has no incoming nodes.
    pub fn is_source(&self) -> bool {
        self.inner().is_source()
    }

    #[inline]
    pub(crate) fn get_variable_id(&self) -> Option<VariableID> {
        self.inner().variable_id
    }

    #[inline]
    /// Input node used when backprop.
    pub fn get_backprop_input(&self, idx: usize) -> Tensor<'graph, F> {
        self.graph
            .tensor(self.inner().get_backprop_inputs()[idx].id)
    }

    #[inline]
    pub fn is_placeholder(&self) -> bool {
        self.inner().placeholder_name.is_some()
    }

    #[inline]
    pub fn placeholder_name(&self) -> Option<&str> {
        self.inner().placeholder_name
    }

    #[inline]
    pub fn validate_using_knownshape(&self, shape: &[usize]) {
        if let Some(ref knownshape) = self.inner().knownshape {
            if !knownshape.validate(shape) {
                panic!(
                    "Shape error: placeholder required {:?}, but got {:?}",
                    knownshape.get(),
                    shape
                );
            }
        } else {
            panic!("This is not a placeholder");
        }
    }

    #[inline]
    pub fn is_differentiable(&self) -> bool {
        self.inner().is_differentiable
    }

    /// True is this tensor was created by `Graph::variable`.
    #[inline]
    #[allow(unused)]
    pub(crate) fn is_variable(&self) -> bool {
        self.inner().is_variable()
    }

    /// Returns the shape of this tensor as a vector.
    /// This method evaluates the shape tensor if needed.
    pub fn shape(&self) -> Vec<usize> {
        // Fallback: try to get shape from knownshape or estimate
        if let Some(ref knownshape) = self.inner().knownshape {
            knownshape
                .get()
                .iter()
                .map(|&x| x.max(0) as usize)
                .collect()
        } else {
            // Last resort: return empty shape
            vec![]
        }
    }

    /// Returns access to the underlying data by evaluating this tensor.
    /// Note: This creates a temporary context for evaluation.
    pub fn data(&self) -> Vec<F> {
        // Since we can't create a Context directly, return empty for now
        // In practice, this would require evaluation within a run() context
        vec![]
    }

    /// Creates a tensor from a vector of data and shape.
    pub fn from_vec(
        _data: Vec<F>,
        shape: Vec<usize>,
        graph: &'graph Graph<F>,
    ) -> Tensor<'graph, F> {
        let array = match NdArray::from_shape_vec(ndarray::IxDyn(&shape), _data) {
            Ok(arr) => arr,
            Err(_) => NdArray::zeros(ndarray::IxDyn(&shape)),
        };
        crate::tensor_ops::convert_to_tensor(array, graph)
    }

    /// Returns true if this tensor requires gradients.
    pub fn requires_grad(&self) -> bool {
        self.is_differentiable()
    }

    /// Convert shape to vector (for compatibility).
    pub fn to_vec(&self) -> Vec<usize> {
        self.shape()
    }
}

impl<'b, T: Float> AsRef<Tensor<'b, T>> for Tensor<'b, T> {
    #[inline(always)]
    fn as_ref(&self) -> &Tensor<'b, T> {
        self
    }
}

pub(crate) struct TensorInternal<F: Float> {
    /// Tensor ID. Unique in the graph which this tensor belongs to.
    pub(crate) id: usize,

    /// Operation to evaluate this tensor.
    pub(crate) op: Option<Box<dyn op::Op<F>>>,

    /// References to immediate predecessors.
    pub(crate) incoming_nodes: SmallVec<IncomingTensor>,

    /// The rank number for topological ordering in a graph.
    pub(crate) topo_rank: usize,

    /// The shape of this tensor
    pub(crate) shape: Option<usize>, // usize is tensor id

    /// placeholder name
    pub(crate) placeholder_name: Option<&'static str>,

    /// This is true if this tensor can have gradient for any objectives.
    pub(crate) is_differentiable: bool,

    /// Input nodes used when backprop.
    ///
    /// This is same as `inputs` in most cases.
    pub(crate) backprop_inputs: Option<SmallVec<IncomingTensor>>,

    /// Static shape of this tensor.
    /// Each dim size is *signed* for placeholders.
    pub(crate) knownshape: Option<KnownShape>,

    /// ID to lookup variable array in VariableEnvironment
    pub(crate) variable_id: Option<VariableID>,
}

impl<F: Float> TensorInternal<F> {
    /// Creates a new empty TensorInternal
    #[allow(dead_code)]
    pub fn new() -> Self {
        TensorInternal {
            id: 0,
            op: Some(Box::new(Dummy)),
            incoming_nodes: SmallVec::new(),
            topo_rank: 0,
            shape: None,
            placeholder_name: None,
            is_differentiable: true,
            backprop_inputs: None,
            knownshape: None,
            variable_id: None,
        }
    }

    /// Returns the Op of this tensor
    pub fn get_op(&self) -> &dyn op::Op<F> {
        self.op
            .as_ref()
            .expect("bad impl: Op is now stolen in gradient.rs")
            .as_ref()
    }

    #[inline(always)]
    pub fn id(&self) -> usize {
        self.id
    }

    #[inline]
    /// Returns true if this node has no incoming nodes.
    pub(crate) fn is_source(&self) -> bool {
        self.incoming_nodes.is_empty()
    }

    #[inline]
    /// Returns true if this node has no incoming nodes.
    pub(crate) fn is_variable(&self) -> bool {
        self.variable_id.is_some()
    }

    /// Returns the number of inputs of this tensor.
    #[inline]
    pub(crate) fn num_inputs(&self) -> usize {
        self.incoming_nodes.len()
    }

    /// True if the op of this tensor is differentiable
    #[inline]
    #[allow(dead_code)]
    pub fn is_differentiable(&self) -> bool {
        self.is_differentiable
    }

    #[inline]
    /// Input nodes used when backprop.
    ///
    /// This is same as `inputs` in most cases.
    pub(crate) fn get_backprop_inputs(&self) -> &[IncomingTensor] {
        self.backprop_inputs
            .as_ref()
            .unwrap_or(&self.incoming_nodes)
            .as_slice()
    }
}

impl<T: Float> fmt::Debug for TensorInternal<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Node name: {}, id: {}, num of inputs: {}, in-edges: {:?}",
            self.get_op().name(),
            self.id(),
            self.incoming_nodes.len(),
            self.incoming_nodes
        )
    }
}

// empty implementation
impl<T: Float> Eq for TensorInternal<T> {}

impl<T: Float> PartialEq for TensorInternal<T> {
    #[inline(always)]
    fn eq(&self, other: &TensorInternal<T>) -> bool {
        // compare addresses on the heap
        self.id() == other.id()
    }
}

/// Raw pointer hashing
impl<T: Float> Hash for TensorInternal<T> {
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state)
    }
}

impl<T: Float> AsRef<TensorInternal<T>> for TensorInternal<T> {
    #[inline(always)]
    fn as_ref(&self) -> &TensorInternal<T> {
        self
    }
}

impl<T: Float> fmt::Display for TensorInternal<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "name={}", self.get_op().name(),)
    }
}

/// Denotes a tensor that enters a certain tensor in a computation graph
///
/// See also [TensorBuilder](struct.TensorBuilder.html).
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct IncomingTensor {
    // Tensor id
    pub(crate) id: usize,
    // true if this was created by `new_mut()`. allows to mutate this tensor
    pub(crate) allow_mut: bool,
    // If the tensor has multiple output arrays, i.e., multi-output op,
    // we have to select which one to use with this selector.
    pub(crate) array_selector: usize,
}

impl<'graph> IncomingTensor {
    /// Instantiates a new immutable `IncomingTensor`.
    #[inline]
    pub(crate) fn new<F: Float>(val: &Tensor<'graph, F>, arrayselector: usize) -> IncomingTensor {
        IncomingTensor {
            id: val.id(),
            allow_mut: false,
            array_selector: arrayselector,
        }
    }

    /// Instantiates a new `IncomingTensor` that references a variable tensor.
    #[inline]
    pub(crate) fn new_mut<F: Float>(
        val: &Tensor<'graph, F>,
        array_selector: usize,
    ) -> IncomingTensor {
        IncomingTensor {
            id: val.id(),
            allow_mut: true,
            array_selector,
        }
    }

    #[inline(always)]
    pub(crate) fn as_tensor<F: Float>(&self, graph: &'graph Graph<F>) -> Tensor<'graph, F> {
        graph.tensor(self.id)
    }

    #[inline]
    #[allow(dead_code)]
    pub(crate) fn get_variable_id<F: Float>(&self, graph: &Graph<F>) -> Option<VariableID> {
        graph.access_inner(self.id).variable_id
    }
}

/// Builder for `ag::Tensor` returned by [Tensor::builder](struct.Tensor.html#method.builder).
///
/// This structure is required only when constructing user-defined `Op`.
///    ```
/// use scirs2_autograd as ag;
/// use ag::op::{Op, OpError, ComputeContext, GradientContext};
/// use ag::tensor_ops::*;
///
/// struct DummyOp {
///    a: f32
/// }
///
/// impl Op<f32> for DummyOp {
///     fn compute(&self, ctx: &mut ComputeContext<f32>) -> Result<(), OpError> { Ok(()) }
///     fn grad(&self, ctx: &mut GradientContext<f32>) {}
/// }
///
/// ag::run(|g: &mut ag::Context<f32>| {
///     let input = &zeros((&[0]), g);
///     let my_output: ag::Tensor<_> = ag::Tensor::builder(g)
///         .append_input(input, false) // immutable input
///         .append_input(input, true)  // mutable input
///         .build(DummyOp {a: 42.});
/// });
///    ```
pub struct TensorBuilder<'g, F: Float> {
    graph: &'g Graph<F>,
    shape: Option<usize>, // usize is tensor id
    in_nodes: SmallVec<IncomingTensor>,
    differentiable: bool,
    backprop_inputs: Option<SmallVec<IncomingTensor>>,
    knownshape: Option<KnownShape>,
    variable_id: Option<VariableID>,
    placeholder_name: Option<&'static str>,
}

const NUM_MAX_KNOWN_SHAPE_SIZE: usize = 4;
type ShapeVec = smallvec::SmallVec<[isize; NUM_MAX_KNOWN_SHAPE_SIZE]>;

pub(crate) struct KnownShape {
    shape: ShapeVec,
    #[allow(dead_code)]
    is_fully_defined: bool,
}

impl KnownShape {
    pub(crate) fn new(shape: &[isize]) -> Self {
        let mut is_fully_defined = true;
        for &a in shape {
            if a == -1 {
                is_fully_defined = false;
            } else if a <= -1 || a == 0 {
                panic!("Given shape ({:?}) contains invalid dim size(s)", &shape);
            }
        }

        Self {
            shape: ShapeVec::from(shape),
            is_fully_defined,
        }
    }

    #[inline]
    pub fn get(&self) -> &[isize] {
        self.shape.as_slice()
    }

    pub fn validate(&self, target: &[usize]) -> bool {
        if self.shape.len() != target.len() {
            return false;
        }
        for (&i, &u) in self.shape.iter().zip(target) {
            if i > 0 && i as usize != u {
                return false;
            }
        }
        true
    }

    #[inline]
    #[allow(dead_code)]
    pub fn is_fully_defined(&self) -> bool {
        self.is_fully_defined
    }
}

#[test]
#[allow(dead_code)]
fn test_topo_order() {
    use crate::tensor_ops as T;
    crate::run(|g| {
        let a: Tensor<f32> = T::zeros(&[4, 2], g);
        let v: Tensor<f32> = T::zeros(&[2, 3], g);
        let b: Tensor<f32> = T::zeros(&[4, 3], g);
        let z = T::matmul(a, v) + b;
        let mut vars = [a.inner(), v.inner(), b.inner(), z.inner()];
        // `sort_by_key` don't reverse the order of `a` and `v`
        vars.sort_by_key(|a| a.topo_rank);
        assert_eq!(vars[0].id, a.id);
        assert_eq!(vars[1].id, v.id);
        assert_eq!(vars[2].id, b.id);
        assert_eq!(vars[3].id, z.id);
    });
}

impl<'graph, F: Float> TensorBuilder<'graph, F> {
    #[inline]
    pub(crate) fn set_variable(mut self, s: VariableID) -> TensorBuilder<'graph, F> {
        self.variable_id = Some(s);
        self
    }

    #[inline]
    pub(crate) fn set_knownshape(mut self, s: &[isize]) -> TensorBuilder<'graph, F> {
        self.knownshape = Some(KnownShape::new(s));
        self
    }

    #[inline]
    pub(crate) fn setshape(mut self, s: &Tensor<'graph, F>) -> TensorBuilder<'graph, F> {
        self.shape = Some(s.id());
        self
    }

    #[inline]
    pub fn set_differentiable(mut self, differentiable: bool) -> TensorBuilder<'graph, F> {
        self.differentiable = differentiable;
        self
    }

    #[inline]
    /// Appends input tensor.
    ///
    /// `allow_mut` indicates whether this tensor should be treated as mutable input or not.
    pub fn append_input<T: AsRef<Tensor<'graph, F>>>(
        self,
        tensor: T,
        allow_mut: bool,
    ) -> TensorBuilder<'graph, F> {
        self.append_input_with_selector(tensor, allow_mut, 0)
    }

    // only use by nth_tensor()
    #[inline]
    pub(crate) fn append_input_with_selector<T: AsRef<Tensor<'graph, F>>>(
        mut self,
        tensor: T,
        allow_mut: bool,
        array_selector: usize,
    ) -> TensorBuilder<'graph, F> {
        let t = tensor.as_ref();
        crate::graph::assert_same_graph(t.graph, self.graph);
        if allow_mut {
            self.in_nodes
                .push(IncomingTensor::new_mut(t, array_selector));
        } else {
            self.in_nodes.push(IncomingTensor::new(t, array_selector));
        }
        self
    }

    #[inline]
    pub(crate) fn set_placeholder_name(mut self, a: &'static str) -> TensorBuilder<'graph, F> {
        self.placeholder_name = Some(a);
        self
    }

    #[inline]
    /// Append the given tensor to the backprop-input-list.
    ///
    /// Not required unless backprop-inputs are differs from normal-case inputs
    pub fn append_backprop_input<T: AsRef<Tensor<'graph, F>>>(
        mut self,
        a: T,
    ) -> TensorBuilder<'graph, F> {
        crate::graph::assert_same_graph(a.as_ref().graph, self.graph);
        if let Some(ref mut inputs) = self.backprop_inputs {
            inputs.push(IncomingTensor::new(a.as_ref(), 0));
        } else {
            let mut inputs = SmallVec::new();
            inputs.push(IncomingTensor::new(a.as_ref(), 0));
            self.backprop_inputs = Some(inputs);
        }
        self
    }

    /// Finalizes this builder and creates a tensor with given `Op` in the graph.
    pub fn build<O>(self, op: O) -> Tensor<'graph, F>
    where
        O: op::Op<F> + 'static,
    {
        let graph = self.graph;
        let rank = if self.in_nodes.is_empty() {
            0
        } else {
            self.in_nodes
                .iter()
                .map(|a| graph.access_inner(a.id).topo_rank)
                .max()
                .map(|a| a + 1)
                .unwrap_or(0)
        };

        let new = TensorInternal {
            // `id` is set in `Graph::install`
            id: usize::default(),
            op: Some(Box::new(op)),
            incoming_nodes: self.in_nodes,
            topo_rank: rank,
            shape: self.shape,
            is_differentiable: self.differentiable,
            backprop_inputs: self.backprop_inputs,
            knownshape: self.knownshape,
            variable_id: self.variable_id,
            placeholder_name: self.placeholder_name,
        };
        Tensor {
            id: graph.install(new),
            graph,
        }
    }
}

#[allow(dead_code)]
pub(crate) struct Dummy;

impl<T: Float> op::Op<T> for Dummy {
    fn compute(&self, _: &mut op::ComputeContext<T>) -> Result<(), OpError> {
        Ok(())
    }
    fn grad(&self, _: &mut GradientContext<T>) {}
}

use crate::tensor_ops as T;

// -- std::tensor_ops::{Add, Sub, Mul, Div} implementations --
macro_rules! impl_bin_op_between_tensor_and_float_trait {
    ($trt:ident, $func:ident, $op:ident) => {
        // Tensor op Float
        impl<'b, F: Float> $trt<F> for Tensor<'b, F> {
            type Output = Tensor<'b, F>;
            fn $func(self, rhs: F) -> Self::Output {
                T::$func(&self, &T::scalar(rhs, self.graph))
            }
        }

        // &Tensor op Float
        impl<'l, 'b, F: Float> $trt<F> for &'l Tensor<'b, F> {
            type Output = Tensor<'b, F>;
            fn $func(self, rhs: F) -> Self::Output {
                T::$func(self, &T::scalar(rhs, self.graph))
            }
        }
    };
}

macro_rules! impl_bin_op_between_tensor_and_primitive {
    ($trt:ident, $func:ident, $op:ident, $scalar_type:ty) => {
        // primitive op Tensor
        impl<'r, 'b, F: Float> $trt<Tensor<'b, F>> for $scalar_type {
            type Output = Tensor<'b, F>;
            fn $func(self, rhs: Tensor<'b, F>) -> Self::Output {
                T::$func(&T::scalar(F::from(self).unwrap(), rhs.graph), &rhs)
            }
        }

        // primitive op &Tensor
        impl<'r, 'b, F: Float> $trt<&'r Tensor<'b, F>> for $scalar_type {
            type Output = Tensor<'b, F>;
            fn $func(self, rhs: &'r Tensor<'b, F>) -> Self::Output {
                T::$func(&T::scalar(F::from(self).unwrap(), rhs.graph), rhs)
            }
        }
    };
}

impl_bin_op_between_tensor_and_float_trait!(Add, add, AddOp);
impl_bin_op_between_tensor_and_float_trait!(Sub, sub, SubOp);
impl_bin_op_between_tensor_and_float_trait!(Mul, mul, MulOp);
impl_bin_op_between_tensor_and_float_trait!(Div, div, DivOp);

impl_bin_op_between_tensor_and_primitive!(Add, add, AddOp, f64);
impl_bin_op_between_tensor_and_primitive!(Sub, sub, SubOp, f64);
impl_bin_op_between_tensor_and_primitive!(Mul, mul, MulOp, f64);
impl_bin_op_between_tensor_and_primitive!(Div, div, DivOp, f64);

impl_bin_op_between_tensor_and_primitive!(Add, add, AddOp, f32);
impl_bin_op_between_tensor_and_primitive!(Sub, sub, SubOp, f32);
impl_bin_op_between_tensor_and_primitive!(Mul, mul, MulOp, f32);
impl_bin_op_between_tensor_and_primitive!(Div, div, DivOp, f32);

macro_rules! impl_bin_op_between_tensors {
    ($trt:ident, $func:ident, $op:ident) => {
        // Tensor op Tensor
        impl<'b, F: Float> $trt for Tensor<'b, F> {
            type Output = Tensor<'b, F>;
            fn $func(self, rhs: Tensor<'b, F>) -> Self::Output {
                T::$func(&self, &rhs)
            }
        }

        // Tensor op &Tensor
        impl<'r, 'b, F: Float> $trt<&'r Tensor<'b, F>> for Tensor<'b, F> {
            type Output = Tensor<'b, F>;
            fn $func(self, rhs: &'r Tensor<'b, F>) -> Self::Output {
                T::$func(&self, rhs)
            }
        }

        // &Tensor op Tensor
        impl<'l, 'b, F: Float> $trt<Tensor<'b, F>> for &'l Tensor<'b, F> {
            type Output = Tensor<'b, F>;
            fn $func(self, rhs: Tensor<'b, F>) -> Self::Output {
                T::$func(self, &rhs)
            }
        }

        // &Tensor op &Tensor
        // lifetime of the two tensors are unrelated
        impl<'l, 'r, 'b, F: Float> $trt<&'r Tensor<'b, F>> for &'l Tensor<'b, F> {
            type Output = Tensor<'b, F>;
            fn $func(self, rhs: &'r Tensor<'b, F>) -> Self::Output {
                T::$func(self, rhs)
            }
        }
    };
}

impl_bin_op_between_tensors!(Add, add, AddOp);
impl_bin_op_between_tensors!(Sub, sub, SubOp);
impl_bin_op_between_tensors!(Mul, mul, MulOp);
impl_bin_op_between_tensors!(Div, div, DivOp);

/// Implementors can be converted to `Tensor`.
pub trait AsTensor<'graph, F: Float> {
    fn as_tensor(&self, graph: &'graph impl AsGraph<F>) -> Tensor<'graph, F>;
}

impl<'graph, F: Float> AsTensor<'graph, F> for Tensor<'graph, F> {
    fn as_tensor(&self, graph: &'graph impl AsGraph<F>) -> Tensor<'graph, F> {
        *self
    }
}

macro_rules! impl_as_tensor_for_array {
    ($num_elems:expr) => {
        impl<'graph, F: Float, I: crate::Int> AsTensor<'graph, F> for [I; $num_elems] {
            fn as_tensor(&self, graph: &'graph impl AsGraph<F>) -> Tensor<'graph, F> {
                let vec = self
                    .iter()
                    .map(|&a| F::from(a).unwrap())
                    .collect::<Vec<F>>();

                // unwrap is safe
                let arr = NdArray::from_shape_vec(ndarray::IxDyn(&[self.len()]), vec).unwrap();
                T::convert_to_tensor(arr, graph.as_graph())
            }
        }
    };
}

impl_as_tensor_for_array!(0);
impl_as_tensor_for_array!(1);
impl_as_tensor_for_array!(2);
impl_as_tensor_for_array!(3);
impl_as_tensor_for_array!(4);
impl_as_tensor_for_array!(5);
impl_as_tensor_for_array!(6);
impl_as_tensor_for_array!(7);
impl_as_tensor_for_array!(8);
