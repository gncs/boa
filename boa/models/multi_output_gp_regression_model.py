from typing import List

import abc
import logging
import json

from tqdm import trange

import tensorflow as tf
import numpy as np

from stheno.tensorflow import dense

from boa.core.gp import GaussianProcess
from boa.core.utils import tensor_hash, standardize
from boa.core.utils import calculate_euclidean_distance_percentiles, calculate_per_dimension_distance_percentiles, \
    setup_logger

from not_tf_opt import AbstractVariable, BoundedVariable, UnconstrainedVariable, \
    minimize, get_reparametrizations

logger = setup_logger(__name__, level=logging.DEBUG, to_console=True)


class ModelError(Exception):
    """Base error thrown by models"""


class MultiOutputGPRegressionModel(tf.keras.Model, abc.ABC):
    AVAILABLE_LENGTHSCALE_INITIALIZATIONS = ["random", "l2_median", "marginal_median"]

    def __init__(self,
                 kernel: str,
                 input_dim: int,
                 output_dim: int,
                 kernel_args={},
                 parallel: bool = False,
                 verbose: bool = False,
                 name: str = "abstract_model",
                 **kwargs):
        """

        :param kernel:
        :param input_dim:
        :param output_dim:
        :param parallel:
        :param verbose:
        :param _num_starting_data_points: Should not be set by the user. Only used to restore models.
        :param name:
        :param kwargs:
        """

        super(MultiOutputGPRegressionModel, self).__init__(name=name,
                                                           dtype=tf.float64,
                                                           **kwargs)

        # Check if the specified kernel is available
        if kernel in GaussianProcess.AVAILABLE_KERNELS:
            self.kernel_name = kernel
            self.kernel_args = kernel_args
        else:
            raise ModelError("Specified kernel {} not available!".format(kernel))

        self.parallel = parallel

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.verbose = verbose

        self.xs = tf.Variable(tf.zeros((0, input_dim), dtype=tf.float64),
                              name="inputs",
                              trainable=False,
                              shape=(None, input_dim))

        self.ys = tf.Variable(tf.zeros((0, output_dim), dtype=tf.float64),
                              name="outputs",
                              trainable=False,
                              shape=(None, output_dim))

        # ---------------------------------------------------------------------
        # Model hyperparameters
        # ---------------------------------------------------------------------
        self._signal_amplitudes: List[BoundedVariable] = [BoundedVariable([0.],
                                                                          lower=-np.inf,
                                                                          upper=np.inf,
                                                                          name=f"signal_amplitude_{i}")
                                                          for i in range(self.output_dim)]

        self._noise_amplitudes: List[BoundedVariable] = [BoundedVariable([0.],
                                                                         lower=-np.inf,
                                                                         upper=np.inf,
                                                                         name=f"noise_amplitude_{i}")
                                                         for i in range(self.output_dim)]

        # Create hyperparameters
        if self.has_explicit_length_scales():
            self._length_scales: List[BoundedVariable] = [BoundedVariable(tf.zeros([self.gp_input_dim(i)]),
                                                                          lower=-np.inf,
                                                                          upper=np.inf,
                                                                          name=f"length_scales_{i}")
                                                          for i in range(self.output_dim)]

        # ---------------------------------------------------------------------
        # Flags
        # ---------------------------------------------------------------------
        self.trained = tf.Variable(False, name="trained", trainable=False)

        # ---------------------------------------------------------------------
        # Statistics for initialization
        # ---------------------------------------------------------------------

        # Statistics on the inputs of each GP in our model
        self._gp_input_statistics = [None] * self.output_dim
        self._gp_input_statistics_hashes = [None] * self.output_dim

        # Models
        self._gp_hyperparameter_hashes = [None] * self.output_dim
        self.models = [None] * self.output_dim

    def length_scales(self, index):
        return self._length_scales[index]

    def signal_amplitude(self, index):
        return self._signal_amplitudes[index]

    def noise_amplitude(self, index):
        return self._noise_amplitudes[index]

    @abc.abstractmethod
    def has_explicit_length_scales(self):
        pass

    def gp_variables_to_train(self, index, transformed):
        ls = self.length_scales(index)
        sa = self.signal_amplitude(index)
        na = self.noise_amplitude(index)

        if transformed:
            ls = ls()
            sa = sa()
            na = na()

        return ls, sa, na

    def gp_assign_variables(self, index, values):
        variables = self.gp_variables_to_train(index, transformed=False)

        if len(variables) != len(values):
            raise ModelError(f"Number of variables ({len(variables)}), "
                             f"and given values ({len(values)}) must be the same!")

        for variable, value in zip(variables, values):
            variable.assign(value)

    def variables_to_train(self, transformed):
        variables = [self.gp_variables_to_train(i, transformed=transformed)
                     for i in range(self.output_dim)]

        return variables

    def assign_variables(self, values):
        if self.output_dim != len(values):
            raise ModelError(f"Length of variable list ({len(values)}) "
                             f"must match the number of outputs ({self.output_dim})!")

        for i, value in enumerate(values):
            self.gp_assign_variables(index=i, values=value)

    def copy(self, name=None):

        # Reflect the class of the current instance
        constructor = self.__class__

        # Get the config of the instance
        config = self.get_config()

        # Instantiate the model
        model = constructor(**config)

        # Create dictionaries of model variables
        self_dict = {v.name: v for v in self.variables}
        model_dict = {v.name: v for v in model.variables}

        # Copy variables over
        for k, v in self_dict.items():
            model_dict[k].assign(v)

        self_submodule_dict = {v.name: v for v in self.submodules}
        model_submodule_dict = {v.name: v for v in model.submodules}

        # Copy BoundedVariables
        for k, v in self_submodule_dict.items():
            if issubclass(v.__class__, AbstractVariable):
                model_submodule_dict[k].assign_var(v)

        return model

    def condition_on(self, xs, ys, keep_previous=True):
        """
        the conditioning operation:

        posterior = prior | (xs, ys)

        :param xs: rank-2 tensor: N x I where N is the number of training examples,
        I is the dimension of the input.
        :param ys: rank-2 tensor: N x O, where N is the number of training examples,
        O is the dimension of the output.
        :param keep_previous: if True, the data on which we conditioned before is retained as well.

        :return: Reference to the conditioned model
        """

        xs, ys = self._validate_and_convert_input_output(xs, ys)

        model = self.copy()

        if keep_previous:
            xs = tf.concat((self.xs, xs), axis=0)
            ys = tf.concat((self.ys, ys), axis=0)

        model.xs.assign(xs)
        model.ys.assign(ys)

        return model

    def create_hyperparameter_initializers(self,
                                           index: int,
                                           length_scale_init_mode: str,
                                           random_init_lower_bound: float = 0.5,
                                           random_init_upper_bound: float = 2.0,
                                           length_scale_base_lower_bound: float = 1e-2,
                                           length_scale_base_upper_bound: float = 1e2,
                                           signal_lower_bound=1e-2,
                                           signal_upper_bound=1e1,
                                           noise_scale_factor=0.1,
                                           percentiles=(0, 20, 50, 80, 100)):
        """
        Creates the initializers for the length scales, signal amplitudes and noise variances.
        :param length_scale_init_mode:
        :param index:
        :return:
        """

        if length_scale_init_mode not in self.AVAILABLE_LENGTHSCALE_INITIALIZATIONS:
            raise ModelError(f"Length scale initialization mode must be one of "
                             f"{self.AVAILABLE_LENGTHSCALE_INITIALIZATIONS}! ({length_scale_init_mode} was given)")

        # Get the input data for the current GP
        gp_input = self.gp_train_input(index=index)

        # Dimension of a single training example
        gp_input_dim = self.gp_input_dim(index=index)

        # Check if we have already calculated the statistics for this input
        gp_input_hash = tensor_hash(gp_input)

        # If the data changed, we calculate the statistics again
        if gp_input_hash != self._gp_input_statistics_hashes[index]:
            logger.info(f"Calculating statistics for GP {index}")
            # Set new hash
            self._gp_input_statistics_hashes[index] = gp_input_hash

            # Calculate new statistics
            l2_percentiles = calculate_euclidean_distance_percentiles(gp_input, percentiles)
            marginal_percentiles = calculate_per_dimension_distance_percentiles(gp_input, percentiles)

            # Cast them to the data type of the model
            l2_percentiles = tf.cast(l2_percentiles, self.dtype)
            marginal_percentiles = tf.cast(marginal_percentiles, self.dtype)

            # Set the new statistics
            self._gp_input_statistics[index] = {"l2": l2_percentiles,
                                                "marginal": marginal_percentiles}

        # ---------------------------------------------------------------------
        # Random initialization
        # ---------------------------------------------------------------------
        if length_scale_init_mode == "random":
            ls_init = tf.random.uniform(shape=(gp_input_dim,),
                                        minval=random_init_lower_bound,
                                        maxval=random_init_upper_bound,
                                        dtype=tf.float64)

            ls_lower_bound = length_scale_base_lower_bound
            ls_upper_bound = length_scale_base_upper_bound

        # ---------------------------------------------------------------------
        # Initialization using the median of the non-zero pairwise Euclidean
        # distances between training inputs
        # ---------------------------------------------------------------------
        elif length_scale_init_mode == "l2_median":
            l2_percentiles = self._gp_input_statistics[index]["l2"]

            # Center on the medians
            ls_init = l2_percentiles[2]

            ls_rand_range = tf.minimum(l2_percentiles[2] - l2_percentiles[1],
                                       l2_percentiles[3] - l2_percentiles[2])

            ls_init += tf.random.uniform(shape=(gp_input_dim,),
                                         minval=-ls_rand_range,
                                         maxval=ls_rand_range,
                                         dtype=self.dtype)

            sqrt_gp_input_dim = tf.sqrt(tf.cast(gp_input_dim, self.dtype))

            ls_lower_bound = tf.ones(shape=(gp_input_dim,), dtype=self.dtype)
            ls_lower_bound = ls_lower_bound * l2_percentiles[0] / (4. * sqrt_gp_input_dim)

            ls_upper_bound = tf.ones(shape=(gp_input_dim,), dtype=self.dtype)
            ls_upper_bound = ls_upper_bound * l2_percentiles[-1] * 64. / sqrt_gp_input_dim

        # ---------------------------------------------------------------------
        # Initialization using the marginal median of pairwise distances
        # between training input dimensions
        # ---------------------------------------------------------------------
        elif length_scale_init_mode == "marginal_median":
            marginal_percentiles = self._gp_input_statistics[index]["marginal"]

            # Center on the medians
            ls_init = marginal_percentiles[2, :]

            ls_rand_range = tf.minimum(marginal_percentiles[2, :] - marginal_percentiles[1, :],
                                       marginal_percentiles[3, :] - marginal_percentiles[2, :])

            ls_init += tf.random.uniform(shape=(gp_input_dim,),
                                         minval=-ls_rand_range,
                                         maxval=ls_rand_range,
                                         dtype=self.dtype)

            # We need to multiply the lengthscales by sqrt(N) to correct for the number of dimensions
            dim_coeff = tf.sqrt(tf.cast(gp_input_dim, tf.float64))
            ls_init = ls_init * dim_coeff

            ls_lower_bound = tf.ones(shape=(gp_input_dim,), dtype=self.dtype)
            ls_lower_bound = ls_lower_bound * marginal_percentiles[0, :] / 4.

            ls_upper_bound = tf.ones(shape=(gp_input_dim,), dtype=self.dtype)
            ls_upper_bound = ls_upper_bound * marginal_percentiles[-1, :] * 64.

        else:
            raise NotImplementedError

        # Create bounded variables
        length_scales = BoundedVariable(ls_init,
                                        lower=tf.maximum(ls_lower_bound, length_scale_base_lower_bound),
                                        upper=tf.minimum(ls_upper_bound, length_scale_base_upper_bound),
                                        dtype=self.dtype)

        signal_amplitude = BoundedVariable(tf.random.uniform(shape=(1,),
                                                             minval=random_init_lower_bound,
                                                             maxval=random_init_upper_bound,
                                                             dtype=self.dtype),
                                           lower=signal_lower_bound,
                                           upper=signal_upper_bound,
                                           dtype=self.dtype)

        noise_amplitude = BoundedVariable(tf.random.uniform(shape=(1,),
                                                            minval=noise_scale_factor * random_init_lower_bound,
                                                            maxval=noise_scale_factor * random_init_upper_bound,
                                                            dtype=self.dtype),
                                          lower=noise_scale_factor * signal_lower_bound,
                                          upper=noise_scale_factor * signal_upper_bound,
                                          dtype=self.dtype)

        return length_scales, signal_amplitude, noise_amplitude

    def create_all_hyperparameter_initializers(self,
                                               length_scale_init_mode: str,
                                               **kwargs):
        """
        Initializes the hyperparameters for every GP in the joint model
        :param length_scale_init_mode:
        :param kwargs:
        :return:
        """

        length_scales = []
        signal_amplitudes = []
        noise_amplitudes = []

        # Iterate through all the GPs
        for i in range(self.output_dim):
            hyperparams = self.create_hyperparameter_initializers(index=i,
                                                                  length_scale_init_mode=length_scale_init_mode,
                                                                  **kwargs)

            ls, signal_amplitude, noise_amplitude = hyperparams

            length_scales.append(ls)
            signal_amplitudes.append(signal_amplitude)
            noise_amplitudes.append(noise_amplitude)

        return length_scales, signal_amplitudes, noise_amplitudes

    def initialize_gp_hyperparameters(self,
                                      index,
                                      length_scale_init_mode,
                                      **kwargs):

        hyperparams = self.create_hyperparameter_initializers(index=index,
                                                              length_scale_init_mode=length_scale_init_mode,
                                                              **kwargs)

        ls, signal_amplitude, noise_amplitude = hyperparams

        self._length_scales[index].assign_var(ls)
        self._signal_amplitudes[index].assign_var(signal_amplitude)
        self._noise_amplitudes[index].assign_var(noise_amplitude)

        return self._length_scales[index], self._signal_amplitudes[index], self._noise_amplitudes[index]

    def initialize_hyperparameters(self,
                                   length_scale_init_mode,
                                   **kwargs):

        hyperparam_list = []

        for i in range(self.output_dim):
            hyperparams = self.initialize_gp_hyperparameters(index=i,
                                                             length_scale_init_mode=length_scale_init_mode,
                                                             **kwargs)

            hyperparam_list.append(hyperparams)

        return hyperparam_list

    def fit(self,
            fit_joint=False,
            optimizer="l-bfgs-b",
            optimizer_restarts=1,
            length_scale_init_mode="l2_median",
            iters=1000,
            rate=1e-1,
            tolerance=1e-5,
            trace=False,
            debugging_trace=False,
            seed=None,
            err_level="catch",
            **kwargs) -> None:
        """
        :param fit_joint: If True, fits the log likelihood of the whole model instead of fitting each, separately
        :param optimizer:
        :param optimizer_restarts:
        :param length_scale_init_mode:
        :param iters:
        :param tolerance:
        :param trace:
        :param debugging_trace:
        :param seed:
        :param kwargs:
        :return:
        """

        if self.xs.value().shape[0] == 0:
            raise ModelError("No data to fit to!")

        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)

        # ---------------------------------------------------------------------
        # Fitting the GPs separately
        # ---------------------------------------------------------------------
        if not fit_joint:

            # Iterate the optimization for each dimension
            for i in range(self.output_dim):

                best_fit_values = None

                best_loss = np.inf

                # Robust optimization
                restart_index = 0

                while restart_index < optimizer_restarts:

                    # Increase step
                    restart_index = restart_index + 1

                    hyperparams = self.initialize_gp_hyperparameters(index=i,
                                                                     length_scale_init_mode=length_scale_init_mode)

                    length_scales, signal_amplitude, noise_amplitude = hyperparams
                    # =================================================================
                    # Debugging stuff
                    # =================================================================
                    if debugging_trace:
                        gp = GaussianProcess(kernel=self.kernel_name,
                                             input_dim=self.gp_input_dim(index=i),
                                             signal_amplitude=signal_amplitude(),
                                             length_scales=length_scales(),
                                             noise_amplitude=noise_amplitude())

                        gp_input = standardize(self.gp_train_input(index=i))

                        K = dense((gp.signal + gp.noise + gp.jitter).kernel(gp_input))
                        print(f"Kernel matrix: {K}")

                        eigvals, _ = tf.linalg.eig(K)
                        eigvals = tf.cast(eigvals, tf.float64)
                        print(f"Eigenvalues: {eigvals.numpy()}")

                        # Largest eigenvalue divided by the smallest
                        condition_number = eigvals[-1] / eigvals[0]

                        # Effective degrees of freedom
                        edof = tf.reduce_sum(eigvals / (eigvals + noise_amplitude()))

                        print(f"Condition number before opt: {condition_number}")
                        print(f"Effective degrees of freedom before opt: {edof}")
                        print(f"Length Scales: {length_scales().numpy()}")
                        print(f"Noise coeff: {noise_amplitude()}")
                        print(f"Signal coeff: {signal_amplitude()}")
                    # =================================================================
                    # End of Debugging stuff
                    # =================================================================

                    loss = np.inf

                    negative_gp_log_prob = lambda: -self.gp_log_prob(xs=self.xs,
                                                                     ys=self.ys,
                                                                     index=i,
                                                                     predictive=False,
                                                                     average=True)
                    try:
                        if optimizer == "l-bfgs-b":
                            # Perform L-BFGS-B optimization
                            loss, converged, diverged = minimize(function=negative_gp_log_prob,
                                                                 vs=hyperparams,
                                                                 explicit=False,
                                                                 parallel_iterations=10,
                                                                 max_iterations=iters,
                                                                 trace=False)

                            # =================================================================
                            # Debugging stuff
                            # =================================================================
                            if debugging_trace:
                                gp = GaussianProcess(kernel=self.kernel_name,
                                                     input_dim=self.gp_input_dim(index=i),
                                                     signal_amplitude=signal_amplitude(),
                                                     length_scales=length_scales(),
                                                     noise_amplitude=noise_amplitude())

                                gp_input = standardize(self.gp_train_input(index=i))

                                K = dense((gp.signal + gp.noise + gp.jitter).kernel(gp_input))

                                print(f"Kernel matrix: {K}")

                                eigvals, _ = tf.linalg.eig(K)
                                eigvals = tf.cast(eigvals, tf.float64)

                                # Largest eigenvalue divided by the smallest
                                condition_number = eigvals[-1] / eigvals[0]

                                # Effective degrees of freedom
                                edof = tf.reduce_sum(eigvals / (eigvals + noise_amplitude()))

                                print("-" * 40)
                                print(f"Eigenvalues after opt: {eigvals.numpy()}")
                                print(f"Condition number after opt: {condition_number}")
                                print(f"Effective degrees of freedom after opt: {edof}")
                                print(f"Length Scales: {length_scales().numpy()}")
                                print(f"Noise coeff: {noise_amplitude()}")
                                print(f"Signal coeff: {signal_amplitude()}")
                                print("=" * 40)
                            # =================================================================
                            # End of Debugging stuff
                            # =================================================================
                            if diverged:
                                logger.error(f"Model diverged, restarting iteration {restart_index}! "
                                             f"(loss was {loss:.3f})")
                                restart_index -= 1
                                continue

                        else:

                            # Get the list of reparametrizations for the hyperparameters
                            reparams = get_reparametrizations(hyperparams, flatten=True)

                            optimizer = tf.optimizers.Adam(rate, epsilon=1e-8)

                            prev_loss = np.inf

                            with trange(iters) as t:
                                for iteration in t:
                                    with tf.GradientTape(watch_accessed_variables=False) as tape:
                                        tape.watch(reparams)

                                        loss = negative_gp_log_prob()

                                    if tf.abs(prev_loss - loss) < tolerance:
                                        logger.info(f"Loss decreased less than {tolerance}, "
                                                    f"optimisation terminated at iteration {iteration}.")
                                        break

                                    prev_loss = loss

                                    gradients = tape.gradient(loss, reparams)
                                    optimizer.apply_gradients(zip(gradients, reparams))

                                    t.set_description(f"Loss at iteration {iteration}: {loss:.3f}.")

                    except tf.errors.InvalidArgumentError as e:
                        logger.error(str(e))
                        restart_index = restart_index - 1

                        if err_level == "raise":
                            raise e

                        elif err_level == "catch":
                            continue

                    if loss < best_loss:

                        logger.info(f"Output {i}, Iteration {restart_index}: New best loss: {loss:.3f}")

                        best_loss = loss

                        best_fit_values = self.gp_variables_to_train(index=i, transformed=True)

                    else:
                        logger.info(f"Output {i}, Iteration {restart_index}: Loss: {loss:.3f}")

                    if np.isnan(loss) or np.isinf(loss):
                        logger.error(f"Output {i}, Iteration {restart_index}: Loss was {loss}, "
                                     f"restarting training iteration!")
                        restart_index = restart_index - 1
                        continue

                self.gp_assign_variables(i, best_fit_values)

        # ---------------------------------------------------------------------
        # Fitting the GPs together
        # ---------------------------------------------------------------------
        else:
            best_fit_values = None
            best_loss = np.inf

            restart_index = 0

            while restart_index < optimizer_restarts:

                restart_index = restart_index + 1

                hyperparams = self.initialize_hyperparameters(length_scale_init_mode=length_scale_init_mode)

                loss = np.inf

                negative_model_log_likelihood = lambda: -self.log_prob(xs=self.xs,
                                                                       ys=self.ys,
                                                                       predictive=False,
                                                                       average=True)
                try:
                    if optimizer == "l-bfgs-b":
                        # Perform L-BFGS-B optimization
                        loss, converged, diverged = minimize(function=negative_model_log_likelihood,
                                                             vs=hyperparams,
                                                             explicit=False,
                                                             parallel_iterations=10,
                                                             max_iterations=iters,
                                                             trace=False)
                    else:

                        # Get the list of reparametrizations for the hyperparameters
                        reparams = get_reparametrizations(hyperparams, flatten=True)

                        optimizer = tf.optimizers.Adam(rate, epsilon=1e-8)

                        prev_loss = np.inf

                        with trange(iters) as t:
                            for iteration in t:
                                with tf.GradientTape(watch_accessed_variables=False) as tape:
                                    tape.watch(reparams)

                                    loss = negative_model_log_likelihood()

                                    # Average the NLL over the training data to keep learning rate consistent
                                    loss = loss

                                if tf.abs(prev_loss - loss) < tolerance:
                                    logger.info(f"Loss decreased less than {tolerance}, "
                                                f"optimisation terminated at iteration {iteration}.")
                                    break

                                prev_loss = loss

                                gradients = tape.gradient(loss, reparams)
                                optimizer.apply_gradients(zip(gradients, reparams))

                                t.set_description(f"Loss at iteration {iteration}: {loss:.3f}.")

                except Exception as e:
                    for i in range(self.output_dim):
                        print(f"{i}: {self.length_scales(i)()}")
                    raise e

                except tf.errors.InvalidArgumentError as e:
                    logger.error(str(e))
                    restart_index = restart_index - 1

                    if err_level == "raise":
                        raise e

                    elif err_level == "catch":
                        continue

                if loss < best_loss:

                    logger.info(f"Iteration {restart_index}, New best loss: {loss:.3f}")

                    best_loss = loss

                    best_fit_values = self.variables_to_train(transformed=True)

                else:
                    logger.info(f"Iteration {restart_index}: Loss: {loss:.3f}")

                if np.isnan(loss) or np.isinf(loss):
                    logger.error(f"Iteration {restart_index}: Loss was {loss}, "
                                 f"restarting training iteration!")
                    restart_index = restart_index - 1
                    continue

            self.assign_variables(best_fit_values)

        self.trained.assign(True)

    @abc.abstractmethod
    def gp_input(self, index, xs, ys):
        pass

    @abc.abstractmethod
    def gp_output(self, index, ys):
        pass

    @abc.abstractmethod
    def gp_predictive_input(self, xs, means):
        pass

    @abc.abstractmethod
    def gp_input_dim(self, index):
        """
        Dimension of a single training example
        :param index:
        :return:
        """

    def gp_train_input(self, index):
        """
        Gets all the training data for the i-th GP in the joint model
        :param index:
        :return:
        """
        return self.gp_input(index,
                             xs=self.xs,
                             ys=self.ys)

    def gp_train_output(self, index):
        """
        Gets the training outputs for the i-th GP in the joint model
        :param index:
        :return:
        """
        return self.gp_output(index,
                              ys=self.ys)

    def gp_predict(self, xs, index):

        if not self.trained:
            logger.warning("Using untrained model for prediction!")
        if xs.shape[1] != self.gp_input_dim(index):
            raise ModelError(f"GP {index} requires an input with shape "
                             f"(None, {self.gp_input_dim(index)}), "
                             f"but got input with shape {xs.shape}!")

        self.create_gp(index=index)

        # Condition the model on the training data
        model = self.models[index] | (self.gp_train_input(index=index),
                                      self.gp_train_output(index=index))

        mean, var = model.predict(xs, latent=False)

        return mean, var

    def gp_log_prob(self,
                    xs,
                    ys,
                    index,
                    predictive=True,
                    average=False):

        self.create_gp(index=index)

        # Select appropriate slices of data
        gp_input = self.gp_input(index=index, xs=xs, ys=ys)
        gp_output = self.gp_output(index=index, ys=ys)

        # Get the model
        model = self.models[index]

        if predictive:
            gp_train_input = self.gp_train_input(index=index)
            gp_train_output = self.gp_train_output(index=index)

            # Condition the model
            model = model | (gp_train_input, gp_train_output)

        log_prob = model.log_pdf(gp_input,
                                 gp_output,
                                 predictive=predictive)

        if average:
            log_prob = log_prob / tf.cast(xs.shape[0], xs.dtype)

        return log_prob

    def predict(self, xs, numpy=False, **kwargs):

        xs = self._validate_and_convert(xs, output=False)

        means = []
        variances = []

        for i in range(self.output_dim):
            mean, var = self.gp_predict(self.gp_predictive_input(xs=xs,
                                                                 means=means),
                                        index=i)

            means.append(mean)
            variances.append(var)

        means = tf.concat(means, axis=1)
        variances = tf.concat(variances, axis=1)

        if numpy:
            means = means.numpy()
            variances = variances.numpy()

        return means, variances

    def log_prob(self, xs, ys, predictive=True, numpy=False, average=False):

        xs, ys = self._validate_and_convert_input_output(xs, ys)

        log_prob = 0.

        for i in range(self.output_dim):
            current_log_prob = self.gp_log_prob(xs,
                                                ys,
                                                index=i,
                                                predictive=predictive,
                                                average=average)

            log_prob = log_prob + current_log_prob

        if numpy:
            log_prob = log_prob.numpy()

        return log_prob

    @abc.abstractmethod
    def get_config(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def from_config(config, **kwargs):
        pass

    def create_gp(self, index):
        # Hash the hyperparameters for the i-th GP
        # param_hash = tensor_hash(self.noise_amplitude(index)()) + \
        #              tensor_hash(self.signal_amplitude(index)()) + \
        #              tensor_hash(self.length_scales(index)())
        #
        # # If the hashes match, do nothing
        # if param_hash == self._gp_hyperparameter_hashes[index]:
        #     return
        #
        # # Store the new hash
        # self._gp_hyperparameter_hashes[index] = param_hash

        # Create GP
        gp = GaussianProcess(kernel=self.kernel_name,
                             input_dim=self.gp_input_dim(index=index),
                             signal_amplitude=self.signal_amplitude(index)(),
                             length_scales=self.length_scales(index)(),
                             noise_amplitude=self.noise_amplitude(index)())

        self.models[index] = gp

    def create_gps(self):
        for i in range(self.output_dim):
            self.create_gp(i)

    def save(self, save_path, **kwargs):

        if not self.trained:
            logger.warning("Saved model has not been trained yet!")

        self.save_weights(save_path)

        config = self.get_config()

        with open(save_path + ".json", "w") as config_file:
            json.dump(config, config_file, indent=4, sort_keys=True)

    @staticmethod
    @abc.abstractmethod
    def restore(save_path):
        pass

    def _validate_and_convert(self, xs, output=False):

        xs = tf.convert_to_tensor(xs)
        xs = tf.cast(xs, tf.float64)

        if len(xs.shape) == 1:
            second_dim = self.output_dim if output else self.input_dim

            # Attempt to convert the xs to the right shape
            xs = tf.reshape(xs, (-1, second_dim))

        # Check if the shapes are correct
        if not len(xs.shape) == 2:
            raise ModelError("The input must be of rank 2!")

        if (not output and xs.shape[1] != self.input_dim) or \
                (output and xs.shape[1] != self.output_dim):
            out_text = 'output' if output else 'input'
            raise ModelError(f"The second dimension of the {out_text} "
                             f"is incorrect: {xs.shape[1]} (expected {self.output_dim if output else self.input_dim})!")

        return xs

    def _validate_and_convert_input_output(self, xs, ys):

        xs = self._validate_and_convert(xs, output=False)
        ys = self._validate_and_convert(ys, output=True)

        # Ensure the user provided the same number of input and output points
        if not xs.shape[0] == ys.shape[0]:
            raise ModelError(f"The first dimension of the input ({xs.shape[0]}) and the output ({ys.shape[0]}) must "
                             f"be equal! (the data needs to form valid input-output pairs)")

        return xs, ys
