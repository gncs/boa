from typing import Tuple, List

import tensorflow as tf
from stheno.tensorflow import GP, EQ, Matern52, Delta
import stheno as stf

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

    def __or__(self, inputs: Tuple) -> None:
        """
        This override of | is implements the inference of the posterior GP model from the inputs (xs, ys)

        :param inputs: tuple of array-like xs and ys, i.e. input points and their corresponding function values
        :return: None
        """

        if not isinstance(inputs, tuple) or len(inputs) != 2:
            raise ModelError("Input must be a tuple of (xs, ys)!")

        xs, ys = inputs

        # Reasonable test of whether the inputs are array-like
        if not hasattr(xs, "__len__") or not hasattr(ys, "__len__"):
            raise ModelError("xs and ys must be array-like!")

        xs = tf.convert_to_tensor(xs, dtype=tf.float64)
        ys = tf.convert_to_tensor(ys, dtype=tf.float64)

        # Check if the shapes are correct
        if not len(xs.shape) == 2 or not len(ys.shape) == 2:
            raise ModelError("xs and ys must be of rank 2!")

        # Ensure the user provided the same number of input and output points
        if not xs.shape[0] == ys.shape[0]:
            raise ModelError("The first dimension of xs and ys must be equal!")

        # Set stuff
        self.input_dim = xs.shape[1]
        self.output_dim = ys.shape[1]

        self.xs = xs
        self.ys = ys
        self.num_true_points = xs.shape[0]

        log_init_minval = tf.math.log(self.init_minval)
        log_init_maxval = tf.math.log(self.init_maxval)

        # Create GP hyperparameter variables
        self.log_length_scales = [tf.Variable(tf.random.uniform(shape=(self.input_dim, ),
                                                                minval=log_init_minval,
                                                                maxval=log_init_maxval,
                                                                dtype=tf.float64),
                                              name="log_length_scales_dim_{}".format(i))
                                  for i in range(self.output_dim)]

        self.log_gp_variances = [tf.Variable(tf.random.uniform(shape=(1,),
                                                               minval=log_init_minval,
                                                               maxval=log_init_maxval,
                                                               dtype=tf.float64),
                                             name="log_gp_variances_dim_{}".format(i))
                                 for i in range(self.output_dim)]

        self.log_noise_variances = [tf.Variable(tf.random.uniform(shape=(1,),
                                                                  minval=log_init_minval,
                                                                  maxval=log_init_maxval,
                                                                  dtype=tf.float64),
                                                name="log_noise_variance_dim_{}".format(i))
                                    for i in range(self.output_dim)]

        # Perform updates
        self._update_mean_std()
        self._update_models()

    @staticmethod
    def normalize(a: tf.Tensor, mean: tf.Tensor, std: tf.Tensor) -> tf.Tensor:
        return (a - mean) / std

    def _update_mean_std(self) -> None:
        min_std = 1e-10

        xs_mean, xs_std = tf.nn.moments(self.xs, axes=[0])

        self.xs_mean = xs_mean
        self.xs_std = tf.maximum(xs_std, min_std)

        ys_mean, ys_std = tf.nn.moments(self.ys, axes=[0])

        self.ys_mean = ys_mean
        self.ys_std = tf.maximum(ys_std, min_std)

    def train(self, init_minval=0.5, init_maxval=2.0) -> None:
        self._update_mean_std()
        x_normalized = self.normalize(self.xs, mean=self.xs_mean, std=self.xs_std)
        y_normalized = self.normalize(self.ys, mean=self.ys_mean, std=self.ys_std)

        self.models.clear()

        # Build model for each output
        for i in range(self.output_dim):

            kernel = self.AVAILABLE_KERNELS[self.kernel_name]()

            old_loss = np.inf

            for j in range(self.num_optimizer_restarts):

                # Re-initialize the current GP's hyperparameters
                log_length_scale = tf.Variable(tf.random.uniform(shape=(self.input_dim,),
                                                                 minval=tf.math.log(self.init_minval),
                                                                 maxval=tf.math.log(self.init_maxval),
                                                                 dtype=tf.float64),
                                               name="dummy_log_length_scales_dim_{}".format(i))

                log_gp_variance = tf.Variable(tf.random.uniform(shape=(1,),
                                                                minval=tf.math.log(self.init_minval),
                                                                maxval=tf.math.log(self.init_maxval),
                                                                dtype=tf.float64),
                                              name="dummy_log_gp_variance_dim_{}".format(i))

                log_noise_variance = tf.Variable(tf.random.uniform(shape=(1,),
                                                                   minval=tf.math.log(self.init_minval),
                                                                   maxval=tf.math.log(self.init_maxval),
                                                                   dtype=tf.float64),
                                                 name="dummy_log_noise_variance_dim_{}".format(i))

                trained_variables = (log_length_scale, log_gp_variance, log_noise_variance)

                length_scale = tf.exp(log_length_scale)
                gp_variance = tf.exp(log_gp_variance)
                noise_variance = tf.exp(log_noise_variance)

                # Construct parameterized kernel
                kernel = gp_variance * (kernel > length_scale) + noise_variance * Delta()

                # Infer the model
                model = GP(kernel) | (x_normalized, y_normalized[:, i:i + 1])

                # Set up the optimizer for gradient descent
                optimizer = tf.optimizers.Adam(1e-3)

                # Optimize the hyperparameters using the negative log-likelihood of the GP
                with tf.GradientTape() as tape:

                    loss = -model(x_normalized).logpdf(y_normalized[:, i:i + 1])

                gradients = tape.gradient(loss, trained_variables)
                optimizer.apply_gradients(zip(gradients, trained_variables))

            self.models.append(model)

    #         model.optimize_restarts(
    #             num_restarts=self.num_optimizer_restarts,
    #             parallel=self.parallel,
    #             robust=True,
    #             verbose=False,
    #             messages=False,
    #         )
    #
    #         self.models.append(model)

    def predict_batch(self, xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert xs.shape[1] == self.input_dim
        self._update_models()

        means = np.zeros((xs.shape[0], self.output_dim))
        var = np.zeros((xs.shape[0], self.output_dim))

        for i, model in enumerate(self.models):
            means[:, i:i + 1], var[:, i:i + 1] = model.predict(Xnew=self.normalize(xs,
                                                                                   mean=self.xs_mean,
                                                                                   std=self.xs_std),
                                                               full_cov=False)

        return (means * self.ys_std + self.ys_mean), (var * self.ys_std**2)

    def add_pseudo_point(self, x: np.ndarray) -> None:
        assert x.shape[0] == self.input_dim

        mean, var = self.predict_batch(x.reshape(1, -1))

        self._append_data_point(x, mean)
        self.num_pseudo_points += 1

    def add_true_point(self, x: np.ndarray, y: np.ndarray) -> None:
        assert self.num_pseudo_points == 0
        assert x.shape[0] == self.input_dim
        assert y.shape[0] == self.output_dim

        self._append_data_point(x, y)
        self.num_true_points += 1

    def remove_pseudo_points(self) -> None:
        self.xs = self.xs[:-self.num_pseudo_points, :]
        self.ys = self.ys[:-self.num_pseudo_points, :]
        self.num_pseudo_points = 0

    def _append_data_point(self, x: np.ndarray, y: np.ndarray) -> None:
        self.xs = np.vstack((self.xs, x))
        self.ys = np.vstack((self.ys, y))

    def _update_models(self) -> None:
        x_normalized = self.normalize(self.xs, mean=self.xs_mean, std=self.xs_std)
        y_normalized = self.normalize(self.ys, mean=self.ys_mean, std=self.ys_std)

        for i, model in enumerate(self.models):
            self.models[i] = model | (x_normalized, y_normalized[:, i:i + 1])
