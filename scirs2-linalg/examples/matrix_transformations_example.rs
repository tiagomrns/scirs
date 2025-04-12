#[cfg(feature = "autograd")]
mod example {
    use ndarray::{array, Array1, Array2, IxDyn};
    use num_traits::Float;
    use scirs2_autograd::error::Result as AutogradResult;
    use scirs2_autograd::tensor::Tensor;
    use scirs2_autograd::variable::Variable;
    use scirs2_linalg::autograd::variable::{
        var_dot as dot, var_matvec as matvec, var_norm as norm, var_project as project,
        var_reflection_matrix as reflection_matrix, var_rotation_matrix_2d as rotation_matrix_2d,
        var_scaling_matrix as scaling_matrix, var_shear_matrix as shear_matrix,
    };

    pub fn run() -> AutogradResult<()> {
        println!("Matrix Transformations with Autograd Example");
        println!("--------------------------------------------\n");

        // Example 1: 2D Rotation Matrix
        println!("Example 1: 2D Rotation Matrix");

        // Create scalar tensor for angle (45 degrees)
        let angle_data = array![std::f64::consts::PI / 4.0].into_dyn();
        let angle = Variable::new(angle_data, true);

        let rotation = rotation_matrix_2d(&angle)?;

        println!("Rotation matrix for 45 degrees:");
        println!("{:?}\n", rotation.data());

        // Apply the rotation to a vector and calculate gradient
        // Create unit vector along x-axis
        let v_data = array![1.0, 0.0].into_dyn();
        let v = Variable::new(v_data, false);

        // Apply rotation to vector (using matmul)
        let rotated_v = matvec(&rotation, &v)?;
        println!("Rotated vector [1, 0]:");
        println!("{:?}\n", rotated_v.data());

        // Compute the gradient of the y-coordinate with respect to the angle
        let y_axis_data = array![0.0, 1.0].into_dyn();
        let y_axis = Variable::new(y_axis_data, false);

        let mut y_coord = dot(&rotated_v, &y_axis)?;

        // Create gradient for backward pass (typically filled with ones)
        let grad_data = array![1.0].into_dyn();
        y_coord.backward(Some(grad_data))?;

        println!("Gradient of y-coordinate with respect to angle:");
        println!("{:?}\n", angle.grad());

        // Example 2: Scaling Matrix
        println!("Example 2: Scaling Matrix");
        let scales_data = array![2.0, 0.5].into_dyn(); // Scale x by 2, y by 0.5
        let scales = Variable::new(scales_data, true);
        let scaling = scaling_matrix(&scales)?;

        println!("Scaling matrix for [2.0, 0.5]:");
        println!("{:?}\n", scaling.data());

        // Apply scaling to a vector and calculate gradient
        let v_data = array![1.0, 1.0].into_dyn(); // Vector [1, 1]
        let v = Variable::new(v_data, false);
        let scaled_v = matvec(&scaling, &v)?;
        println!("Scaled vector [1, 1]:");
        println!("{:?}\n", scaled_v.data());

        // Example 3: Reflection Matrix
        println!("Example 3: Reflection Matrix");
        let normal_data = array![1.0, 1.0].into_dyn(); // Normal vector for reflection plane
        let normal = Variable::new(normal_data, true);
        let reflection = reflection_matrix(&normal)?;

        println!("Reflection matrix for normal [1, 1]:");
        println!("{:?}\n", reflection.data());

        // Apply reflection to a vector
        let v_data = array![1.0, 0.0].into_dyn(); // Vector [1, 0]
        let v = Variable::new(v_data, false);
        let reflected_v = matvec(&reflection, &v)?;
        println!("Reflected vector [1, 0]:");
        println!("{:?}\n", reflected_v.data());

        // Example 4: Shear Matrix
        println!("Example 4: Shear Matrix");
        let shear_factor_data = array![0.5].into_dyn();
        let shear_factor = Variable::new(shear_factor_data, true);
        let shear = shear_matrix(&shear_factor, 0, 1, 2)?; // Shear in x direction

        println!("Shear matrix for factor 0.5:");
        println!("{:?}\n", shear.data());

        // Apply shear to a vector
        let v_data = array![1.0, 1.0].into_dyn(); // Vector [1, 1]
        let v = Variable::new(v_data, false);
        let sheared_v = matvec(&shear, &v)?;
        println!("Sheared vector [1, 1]:");
        println!("{:?}\n", sheared_v.data());

        // Example 5: Orthogonal Projection
        println!("Example 5: Orthogonal Projection");
        // Create a matrix whose columns span a subspace
        let a_data = array![[1.0, 0.0], [1.0, 1.0]].into_dyn();
        let a = Variable::new(a_data, true);

        // Create a vector to project
        let x_data = array![2.0, 3.0].into_dyn();
        let x = Variable::new(x_data, false);

        // Project the vector onto the column space of A
        let projected = project(&a, &x)?;

        println!("Matrix A (columns span the subspace):");
        println!("{:?}", a.data());
        println!("\nVector x to project:");
        println!("{:?}", x.data());
        println!("\nProjection of x onto the column space of A:");
        println!("{:?}\n", projected.data());

        // Instead of the norm, let's compute the sum of the projection
        // to avoid the shape issue with the norm function
        let mut sum_val = projected.sum(None)?;

        // Create gradient for backward pass (typically filled with ones)
        let grad_data = array![1.0].into_dyn();
        sum_val.backward(Some(grad_data))?;

        println!("Gradient of projection sum with respect to A:");
        println!("{:?}", a.grad());

        Ok(())
    }
}

fn main() {
    #[cfg(feature = "autograd")]
    {
        example::run().unwrap();
    }

    #[cfg(not(feature = "autograd"))]
    {
        println!("This example requires the 'autograd' feature.");
        println!(
            "Run with: cargo run --example matrix_transformations_example --features autograd"
        );
    }
}
