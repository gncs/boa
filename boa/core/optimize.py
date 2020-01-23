import tensorflow as tf
import tensorflow_probability as tfp

from numpy import finfo, float64


def bounded_minimize(function,
                     vs,
                     num_correction_pairs=10,
                     tolerance=1e-05,
                     x_tolerance=0,
                     f_relative_tolerance=1e7,
                     initial_inverse_hessian_estimate=None,
                     max_iterations=2000,
                     parallel_iterations=1):
    """
    Takes a function whose arguments are boa.core.BoundedVariables,
    and performs L-BFGS-B on it.

    :param function:
    :param vs: BoundedVariables or iterables of BoundedVariables
    :param num_correction_pairs:
    :param tolerance:
    :param x_tolerance:
    :param f_relative_tolerance:
    :param initial_inverse_hessian_estimate:
    :param max_iterations:
    :param parallel_iterations:
    :return:
    """

    float64_machine_eps = finfo(float64).eps

    # These are chosen to match the parameters of
    # scipy.optimizer.fmin_l_bfgs_b
    optimizer_args = {"num_correction_pairs": num_correction_pairs,
                      "tolerance": tolerance,  # This is pgtol in scipy
                      "x_tolerance": x_tolerance,

                      # This is eps * factr in scipy
                      "f_relative_tolerance": float64_machine_eps * f_relative_tolerance,
                      "initial_inverse_hessian_estimate": initial_inverse_hessian_estimate,
                      "max_iterations": max_iterations,
                      "parallel_iterations": parallel_iterations}

    # Get the reparameterization of the BoundedVariables
    reparameterizations = [v.reparameterization for v in vs]

    # Get the shapes for the reparameterizations
    reparam_shapes = [r.shape for r in reparameterizations]

    # Get the starting indices of the reparameterizations in the flattened representation
    reparam_starts = [0]
    for reparam in reparameterizations:
        reparam_starts.append(reparam_starts[-1] + tf.size(reparam).numpy())

    # Pull-back of the function to the unconstrained domain:
    # Reparameterize the function such that instead of taking its original bounded
    # arguments, it takes the unconstrained ones, and they get forward transformed.
    def reparameterized_function(*args):
        return function(*[v.forward_transform(arg) for v, arg in zip(vs, args)])

    # Takes a flattened vector of the original function arguments, and returns them in
    # their original reshaped form as a list
    def flattened_vector_to_reshaped_args(x):
        args = []

        for k in range(len(reparameterizations)):
            # Split off variable
            var = x[reparam_starts[k]:reparam_starts[k + 1]]

            # Reshape variable
            var = tf.reshape(var, reparam_shapes[k])

            args.append(var)

        return args

    def fn_with_grads(x):

        # Get back the original arguments
        args = flattened_vector_to_reshaped_args(x)

        value, gradients = tfp.math.value_and_gradient(f=reparameterized_function,
                                                       xs=args)

        # We must concatenate the gradients because lbfgs_minimize expects a single vector
        gradients = tf.concat(gradients, axis=0)

        return value, gradients

    flattened_reparams = [tf.reshape(r, [-1]) for r in reparameterizations]
    initial_position = tf.concat(flattened_reparams, axis=0)

    optimizer_results = tfp.optimizer.lbfgs_minimize(fn_with_grads,
                                                     initial_position=initial_position,
                                                     **optimizer_args)

    optimum = flattened_vector_to_reshaped_args(optimizer_results.position)

    # Assign the results to the variables
    for r, opt in zip(reparameterizations, optimum):
        r.assign(opt)

    # Return the loss
    return optimizer_results.objective_value
