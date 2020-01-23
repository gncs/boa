import tensorflow as tf
import tensorflow_probability as tfp

from numpy import finfo, float64


def bounded_minimize(function,
                     vs,
                     optimizer_args):
    """
    Takes a function whose arguments are boa.core.BoundedVariables,
    and performs L-BFGS-B on it.
    :param function: Function to be minimized
    :param vs: Iterable of BoundedVariables containing the initial guess
    :param optimizer_args: arguments to be passed to tfp.optimize.lbfgs_minimize
    :return: loss of the optimizer
    """

    float64_machine_eps = finfo(float64).eps

    # These are chosen to match the parameters of
    # scipy.optimizer.fmin_l_bfgs_b
    default_optimizer_args = {"num_correction_pairs": 10,
                              "tolerance": 1e-05,  # This is pgtol in scipy
                              "x_tolerance": 0,
                              "f_relative_tolerance": float64_machine_eps * 1e7,  # This is eps * factr in scipy
                              "initial_inverse_hessian_estimate": None,
                              "max_iterations": 50,
                              "parallel_iterations": 1}

    # Add missing optimizer arguments from the default ones
    for arg, arg_val in default_optimizer_args.items():
        if arg not in optimizer_args:
            optimizer_args[arg] = arg_val

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
