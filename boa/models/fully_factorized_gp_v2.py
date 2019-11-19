from typing import Tuple, List

import tensorflow as tf
from stheno.tensorflow import GP, EQ, Matern52, Delta
from varz.tensorflow import Vars, minimise_l_bfgs_b

import numpy as np

from .abstract_model_v2 import AbstractModel, ModelError


class FullyFactorizedGPModel(AbstractModel):

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

        super(FullyFactorizedGPModel, self).__init__(kernel=kernel,
                                                     num_optimizer_restarts=num_optimizer_restarts,
                                                     parallel=parallel,
                                                     verbose=verbose,
                                                     name=name,
                                                     **kwargs)

        self.length_scales: List[tf.Variable] = []
        self.gp_variances: List[tf.Variable] = []
        self.noise_variances: List[tf.Variable] = []

    def __or__(self, inputs: Tuple) -> tf.keras.Model:
        """
        This override of | is implements the inference of the posterior GP model from the inputs (xs, ys)

        :param inputs: tuple of array-like xs and ys, i.e. input points and their corresponding function values
        :return: None
        """

        # Validate inputs and set data
        super(FullyFactorizedGPModel, self).__or__(inputs)

        # Create GP hyperparameter variables
        for i in range(self.output_dim):
            self.length_scales.append(tf.Variable(tf.ones(self.input_dim,
                                                          dtype=tf.float64),
                                                  name="length_scales_dim_{}".format(i),
                                                  trainable=False))

            self.gp_variances.append(tf.Variable((1.0,),
                                                 dtype=tf.float64,
                                                 name="gp_variance_dim_{}".format(i),
                                                 trainable=False))

            self.noise_variances.append(tf.Variable((1.0,),
                                                    dtype=tf.float64,
                                                    name="noise_variance_dim_{}".format(i),
                                                    trainable=False))

        # Perform updates
        self._update_mean_std()

        return self

    def train(self, init_minval=0.5, init_maxval=2.0) -> None:
        self._update_mean_std()
        x_normalized = self.normalize(self.xs, mean=self.xs_mean, std=self.xs_std)
        y_normalized = self.normalize(self.ys, mean=self.ys_mean, std=self.ys_std)

        # Build model for each output
        for i in range(self.output_dim):

            best_loss = np.inf
            best_signal_gp = None
            best_noise_gp = None

            ls_name = "length_scales_dim_{}".format(i)
            gp_var_name = "gp_variance_dim_{}".format(i)
            noise_var_name = "noise_variance_dim_{}".format(i)

            vs = Vars(tf.float64)

            # Length scales
            vs.bnd(init=tf.ones(self.input_dim, dtype=tf.float64),
                   lower=1e-4,
                   upper=1e4,
                   name=ls_name)

            # GP variance
            vs.pos(init=tf.ones(1, dtype=tf.float64),
                   name=gp_var_name)

            # Noise variance: bound between 1e-4 and 1e4
            vs.bnd(init=tf.ones(1, dtype=tf.float64),
                   lower=1e-4,
                   upper=1e4,
                   name=noise_var_name)

            for j in range(self.num_optimizer_restarts):

                if self.verbose:
                    print("Optimization round: {} / {}".format(j + 1, self.num_optimizer_restarts))

                # Re-initialize the current GP's hyperparameters
                vs.assign(ls_name, tf.random.uniform(shape=(self.input_dim,),
                                                     minval=self.init_minval,
                                                     maxval=self.init_maxval,
                                                     dtype=tf.float64))

                vs.assign(gp_var_name, tf.random.uniform(shape=(1,),
                                                         minval=self.init_minval,
                                                         maxval=self.init_maxval,
                                                         dtype=tf.float64))

                vs.assign(noise_var_name, tf.random.uniform(shape=(1,),
                                                            minval=self.init_minval,
                                                            maxval=self.init_maxval,
                                                            dtype=tf.float64))

                # Training objective
                def negative_gp_log_likelihood(gp_variance, length_scale, noise_variance):

                    signal_gp_, noise_gp_ = self.get_prior_gp_model(length_scale,
                                                                    gp_variance,
                                                                    noise_variance)

                    prior_gp_ = signal_gp_ + noise_gp_

                    return -prior_gp_(x_normalized).logpdf(y_normalized[:, i:i + 1])

                loss = np.inf
                try:
                    pass
                    # Perform L-BFGS-B optimization
                    loss = minimise_l_bfgs_b(lambda v: negative_gp_log_likelihood(v[gp_var_name],
                                                                                  v[ls_name],
                                                                                  v[noise_var_name]),
                                             vs,
                                             names=[ls_name,
                                                    gp_var_name,
                                                    noise_var_name])
                except Exception as e:
                    print("Iteration {} failed: {}".format(i + 1, str(e)))

                if loss < best_loss:

                    self.signal_gps.clear()
                    self.noise_gps.clear()

                    if self.verbose:
                        print("New best objective value: {:.4f}".format(loss))

                    best_loss = loss

                    # Reassign variables
                    self.length_scales[i].assign(vs[ls_name])
                    self.gp_variances[i].assign(vs[gp_var_name])
                    self.noise_variances[i].assign(vs[noise_var_name])

                    signal_gp, noise_gp = self.get_prior_gp_model(self.length_scales[i],
                                                                  self.gp_variances[i],
                                                                  self.noise_variances[i])

                    self.signal_gps.append(signal_gp)
                    self.noise_gps.append(noise_gp)

    def predict_batch(self, xs):

        if len(self.signal_gps) == 0:
            raise ModelError("The model has not been trained yet!")

        xs = tf.convert_to_tensor(xs, dtype=tf.float64)

        if xs.shape[1] != self.input_dim:
            raise ModelError("xs with shape {} must have 1st dimension (0 indexed) {}!".format(xs.shape,
                                                                                               self.input_dim))

        means = []
        var = []

        xs_normalized = self.normalize(xs, mean=self.xs_mean, std=self.xs_std)

        train_xs = self.normalize(self.xs, mean=self.xs_mean, std=self.xs_std)
        train_ys = self.normalize(self.ys, mean=self.ys_mean, std=self.ys_std)

        for i, (signal_gp, noise_gp) in enumerate(zip(self.signal_gps, self.noise_gps)):

            model = (signal_gp + noise_gp) | (train_xs, train_ys[:, i:i+1])

            means.append(model.mean(xs_normalized))
            var.append(model.kernel.elwise(xs_normalized))

        means = tf.concat(means, axis=1)
        var = tf.concat(var, axis=1)

        return (means * self.ys_std + self.ys_mean), (var * self.ys_std ** 2)
