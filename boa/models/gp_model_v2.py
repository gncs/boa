from typing import Tuple, List

import tensorflow as tf
from stheno.tensorflow import GP, EQ, Matern52, Delta
from varz.tensorflow import Vars, minimise_l_bfgs_b

import numpy as np

from .abstract_model_v2 import AbstractModel, ModelError


class GPModel(AbstractModel):
    AVAILABLE_KERNELS = {"rbf": EQ,
                         "matern52": Matern52, }

    def __init__(self,
                 kernel: str,
                 num_optimizer_restarts: int,
                 parallel: bool = False,
                 name: str = "gp_model",
                 verbose=True,
                 **kwargs):
        """
        Constructor of GP model.

        :param kernel: name of kernel
        :param num_optimizer_restarts: number of times the optimization of the hyperparameters is restarted
        :param parallel: run optimizations in parallel
        :param name: name of the GP model
        """

        super(GPModel, self).__init__(name=name, **kwargs)

        # Independent GPs for each output dimension
        self.models: List = []

        # Check if the specified kernel is available
        if kernel in self.AVAILABLE_KERNELS:
            self.kernel_name = kernel
        else:
            raise ModelError("Specified kernel {} not available!".format(kernel))

        self.num_optimizer_restarts = num_optimizer_restarts
        self.parallel = parallel

        self.input_dim = 0
        self.output_dim = 0

        self.verbose = verbose

        self.xs_mean = tf.constant([[]])
        self.xs_std = tf.constant([[]])

        self.ys_mean = tf.constant([[]])
        self.ys_std = tf.constant([[]])

        self.xs = tf.constant([[]])
        self.ys = tf.constant([[]])

        self.num_pseudo_points = 0
        self.num_true_points = 0

        # Range for the initialisation of GP hyperparameters
        self.init_minval = tf.constant(0.5, dtype=tf.float64)
        self.init_maxval = tf.constant(2.0, dtype=tf.float64)

        # Create model variables container
        self.vars = Vars(tf.float64)

    def __or__(self, inputs: Tuple) -> tf.keras.Model:
        """
        This override of | is implements the inference of the posterior GP model from the inputs (xs, ys)

        :param inputs: tuple of array-like xs and ys, i.e. input points and their corresponding function values
        :return: None
        """

        # Validate inputs
        super(GPModel, self).__or__(inputs)

        xs, ys = inputs

        xs = tf.convert_to_tensor(xs, dtype=tf.float64)
        ys = tf.convert_to_tensor(ys, dtype=tf.float64)

        # Set stuff
        self.input_dim = xs.shape[1]
        self.output_dim = ys.shape[1]

        self.xs = xs
        self.ys = ys
        self.num_true_points = xs.shape[0]

        # Create GP hyperparameter variables
        for i in range(self.output_dim):
            # Length scales
            self.vars.bnd(init=tf.random.uniform(shape=(self.input_dim,),
                                                 minval=self.init_minval,
                                                 maxval=self.init_maxval,
                                                 dtype=tf.float64),
                          lower=1e-4,
                          upper=1e4,
                          name="length_scales_dim_{}".format(i))

            # GP variance
            self.vars.pos(init=tf.random.uniform(shape=(1,),
                                                 minval=self.init_minval,
                                                 maxval=self.init_maxval,
                                                 dtype=tf.float64),
                          name="gp_variance_dim_{}".format(i))

            # Noise variance: bound between 1e-4 and 1e4
            self.vars.bnd(init=tf.random.uniform(shape=(1,),
                                                 minval=self.init_minval,
                                                 maxval=self.init_maxval,
                                                 dtype=tf.float64),
                          lower=1e-4,
                          upper=1e4,
                          name="noise_variance_dim_{}".format(i))

        # Perform updates
        self._update_mean_std()
        self._update_models()

        return self

    @staticmethod
    def normalize(a: tf.Tensor, mean: tf.Tensor, std: tf.Tensor) -> tf.Tensor:
        return (a - mean) / std

    def _update_mean_std(self, min_std=1e-10) -> None:

        xs_mean, xs_var = tf.nn.moments(self.xs, axes=[0])

        self.xs_mean = xs_mean
        self.xs_std = tf.maximum(tf.sqrt(xs_var), min_std)

        ys_mean, ys_var = tf.nn.moments(self.ys, axes=[0])

        self.ys_mean = ys_mean
        self.ys_std = tf.maximum(tf.sqrt(ys_var), min_std)

    def get_prior_gp_model(self, length_scale, gp_variance, noise_variance):

        # Construct parameterized kernel
        kernel = self.AVAILABLE_KERNELS[self.kernel_name]()
        prior_gp = gp_variance * (GP(kernel) > length_scale) + noise_variance * GP(Delta())

        return prior_gp

    def train(self, init_minval=0.5, init_maxval=2.0) -> None:
        self._update_mean_std()
        x_normalized = self.normalize(self.xs, mean=self.xs_mean, std=self.xs_std)
        y_normalized = self.normalize(self.ys, mean=self.ys_mean, std=self.ys_std)

        self.models.clear()

        # Build model for each output
        for i in range(self.output_dim):

            best_loss = np.inf
            best_model = None

            ls_name = "length_scales_dim_{}".format(i)
            gp_var_name = "gp_variance_dim_{}".format(i)
            noise_var_name = "noise_variance_dim_{}".format(i)

            dummy_vars = Vars(tf.float64)

            for j in range(self.num_optimizer_restarts):

                if self.verbose:
                    print("Optimization round: {} / {}".format(j + 1, self.num_optimizer_restarts))

                # Re-initialize the current GP's hyperparameters

                # Length scales
                dummy_vars.bnd(init=tf.random.uniform(shape=(self.input_dim,),
                                                      minval=self.init_minval,
                                                      maxval=self.init_maxval,
                                                      dtype=tf.float64),
                               lower=1e-4,
                               upper=1e4,
                               name=ls_name)

                # GP variance
                dummy_vars.pos(init=tf.random.uniform(shape=(1,),
                                                      minval=self.init_minval,
                                                      maxval=self.init_maxval,
                                                      dtype=tf.float64),
                               name=gp_var_name)

                # Noise variance: bound between 1e-4 and 1e4
                dummy_vars.bnd(init=tf.random.uniform(shape=(1,),
                                                      minval=self.init_minval,
                                                      maxval=self.init_maxval,
                                                      dtype=tf.float64),
                               lower=1e-4,
                               upper=1e4,
                               name=noise_var_name)

                # Training objective
                def negative_gp_log_likelihood(gp_variance, length_scale, noise_variance):

                    prior_gp_ = self.get_prior_gp_model(length_scale, gp_variance, noise_variance)

                    return -prior_gp_(x_normalized).logpdf(y_normalized[:, i:i + 1])

                # Perform L-BFGS-B optimization
                loss = minimise_l_bfgs_b(lambda v: negative_gp_log_likelihood(v[gp_var_name],
                                                                              v[ls_name],
                                                                              v[noise_var_name]),
                                         dummy_vars,
                                         names=[ls_name,
                                                gp_var_name,
                                                noise_var_name])

                if loss < best_loss:

                    if self.verbose:
                        print("New best objective value: {:.4f}".format(loss))

                    best_loss = loss

                    # Reassign variables
                    self.vars.assign(ls_name, dummy_vars[ls_name])
                    self.vars.assign(gp_var_name, dummy_vars[gp_var_name])
                    self.vars.assign(noise_var_name, dummy_vars[noise_var_name])

                    # Construct parameterized kernel
                    kernel = self.AVAILABLE_KERNELS[self.kernel_name]()
                    prior_gp = self.vars[gp_var_name] * (GP(kernel) > self.vars[ls_name]) + self.vars[
                        noise_var_name] * GP(Delta())

                    # Infer the model
                    best_model = prior_gp | (x_normalized, y_normalized[:, i:i + 1])

            self.models.append(best_model)

    def predict_batch(self, xs):

        if len(self.models) == 0:
            raise ModelError("The model has not been trained yet!")

        xs = tf.convert_to_tensor(xs, dtype=tf.float64)

        if xs.shape[1] != self.input_dim:
            raise ModelError("xs with shape {} must have 1st dimension (0 indexed) {}!".format(xs.shape,
                                                                                               self.input_dim))

        self._update_models()

        means = []
        var = []

        for i, model in enumerate(self.models):
            normal = model(self.normalize(xs,
                                          mean=self.xs_mean,
                                          std=self.xs_std))
            means.append(normal.mean)
            var.append(tf.reshape(tf.linalg.diag_part(normal.var), [-1, 1]))

        means = tf.concat(means, axis=1)
        var = tf.concat(var, axis=1)

        return (means * self.ys_std + self.ys_mean), (var * self.ys_std ** 2)

    def add_pseudo_point(self, x):

        x = tf.convert_to_tensor(x, dtype=tf.float64)

        if x.shape != (1, self.input_dim):
            raise ModelError("point with shape {} must have shape {}!".format(x.shape, (1, self.input_dim)))

        mean, var = self.predict_batch(x)

        self._append_data_point(x, mean)
        self.num_pseudo_points += 1

    def add_true_point(self, x, y) -> None:

        x = tf.convert_to_tensor(x, dtype=tf.float64)
        y = tf.convert_to_tensor(y, dtype=tf.float64)

        if x.shape != (1, self.input_dim):
            raise ModelError("x with shape {} must have shape {}!".format(x.shape, (1, self.input_dim)))
        if y.shape != (1, self.output_dim):
            raise ModelError("y with shape {} must have shape {}!".format(y.shape, (1, self.output_dim)))

        assert self.num_pseudo_points == 0

        self._append_data_point(x, y)
        self.num_true_points += 1

    def remove_pseudo_points(self) -> None:
        self.xs = self.xs[:-self.num_pseudo_points, :]
        self.ys = self.ys[:-self.num_pseudo_points, :]
        self.num_pseudo_points = 0

    def _append_data_point(self, x, y) -> None:
        self.xs = tf.concat((self.xs, x), axis=0)
        self.ys = tf.concat((self.ys, y), axis=0)

    def _update_models(self) -> None:
        x_normalized = self.normalize(self.xs, mean=self.xs_mean, std=self.xs_std)
        y_normalized = self.normalize(self.ys, mean=self.ys_mean, std=self.ys_std)

        for i in range(len(self.models)):
            model = self.get_prior_gp_model(length_scale=self.vars["length_scales_dim_{}".format(i)],
                                            gp_variance=self.vars["gp_variance_dim_{}".format(i)],
                                            noise_variance=self.vars["noise_variance_dim_{}".format(i)])

            self.models[i] = model | (x_normalized, y_normalized[:, i:i + 1])
