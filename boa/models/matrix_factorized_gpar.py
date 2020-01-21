import logging
import json

import numpy as np
import tensorflow as tf
from varz.tensorflow import Vars, minimise_l_bfgs_b, minimise_adam

from boa.models.abstract_model import ModelError
from .gpar import GPARModel

from boa.core.utils import setup_logger
from boa.core.gp import GaussianProcess

logger = setup_logger(__name__, level=logging.DEBUG, to_console=True, log_file="logs/mf_gpar.log")


class MatrixFactorizedGPARModel(GPARModel):
    LLS_MAT = "left_length_scale_matrix"
    RLS_MAT = "right_length_scale_matrix"

    def __init__(self,
                 kernel: str,
                 input_dim: int,
                 output_dim: int,
                 latent_dim: int,
                 initialization_heuristic: str = "median",
                 verbose: bool = False,
                 name="matrix_factorized_gpar_model",
                 **kwargs):

        super(MatrixFactorizedGPARModel, self).__init__(
            kernel=kernel,
            input_dim=input_dim,
            output_dim=output_dim,
            initialization_heuristic=initialization_heuristic,
            _create_length_scales=False,  # Never create the length scales in the parent class
            verbose=verbose,
            name=name,
            **kwargs)

        self.latent_dim = latent_dim

        self.output_length_scales = []

        # Create TF variables for the hyperparameters

        # Create low rank representation for the input length scales
        self.left_length_scale_matrix = tf.Variable(tf.ones((self.output_dim, self.latent_dim), dtype=tf.float64),
                                                    trainable=False,
                                                    name=self.LLS_MAT)

        self.right_length_scale_matrix = tf.Variable(tf.ones((self.latent_dim, self.input_dim), dtype=tf.float64),
                                                     trainable=False,
                                                     name=self.RLS_MAT)

        # # dimensions O x I
        input_length_scales = tf.matmul(self.left_length_scale_matrix, self.right_length_scale_matrix)

        for i in range(self.output_dim):
            # Length scales for the output dimensions only
            out_length_scales = tf.Variable(tf.ones(i, dtype=tf.float64),
                                            name=f"{i}/output_length_scale",
                                            trainable=False)

            self.output_length_scales.append(out_length_scales)

            # # i-th length scales
            length_scales = tf.concat((input_length_scales[i, :], out_length_scales), axis=0)

            self.length_scales.append(length_scales)

    def copy(self, name=None):

        mf_gpar = super(MatrixFactorizedGPARModel, self).copy(name=name)

        input_length_scales = tf.matmul(mf_gpar.left_length_scale_matrix, mf_gpar.right_length_scale_matrix)

        for i in range(self.output_dim):
            mf_gpar.length_scales[i] = tf.concat((input_length_scales[i, :], mf_gpar.output_length_scales[i]), axis=0)

        return mf_gpar

    def create_hyperparameters(self) -> Vars:

        vs = Vars(tf.float64)

        # Create container for matrix factors
        vs.bnd(init=tf.ones((self.output_dim, self.latent_dim), dtype=tf.float64),
               lower=1e-10,
               upper=1e2,
               name=self.LLS_MAT)

        vs.bnd(init=tf.ones((self.latent_dim, self.input_dim), dtype=tf.float64),
               lower=1e-10,
               upper=1e2,
               name=self.RLS_MAT)

        # Create the rest of the hyperparameters

        for i in range(self.output_dim):
            # Note the scaling in dimension with the index
            vs.bnd(init=tf.ones(i, dtype=tf.float64), lower=1e-3, upper=1e2, name=f"{i}/output_length_scales")

            # GP variance
            vs.bnd(init=tf.ones(1, dtype=tf.float64), lower=1e-4, upper=1e4, name=f"{i}/signal_amplitude")

            # Noise variance: bound between 1e-4 and 1e4
            vs.bnd(init=tf.ones(1, dtype=tf.float64), lower=1e-6, upper=1e2, name=f"{i}/noise_amplitude")

        return vs

    def initialize_hyperparameters(self, vs: Vars, length_scale_init="random", init_minval=0.5, init_maxval=2.0):

        if length_scale_init == "median":
            ls_init = tf.sqrt(self.xs_euclidean_percentiles[2] / self.latent_dim)

            ls_rand_range = tf.minimum(self.xs_euclidean_percentiles[2] - self.xs_euclidean_percentiles[0],
                                       self.xs_euclidean_percentiles[4] - self.xs_euclidean_percentiles[2])

            ls_rand_range = tf.sqrt(ls_rand_range / self.latent_dim)

            # left length scale
            lls_init = ls_init + tf.random.uniform(
                shape=(self.output_dim, self.latent_dim), dtype=tf.float64, minval=-ls_rand_range, maxval=ls_rand_range)

            # right length scale
            rls_init = ls_init + tf.random.uniform(
                shape=(self.latent_dim, self.input_dim), dtype=tf.float64, minval=-ls_rand_range, maxval=ls_rand_range)

            vs.assign(self.LLS_MAT, lls_init)
            vs.assign(self.RLS_MAT, rls_init)

        else:
            vs.assign(
                self.LLS_MAT,
                tf.random.normal(shape=(self.output_dim, self.latent_dim),
                                 dtype=tf.float64,
                                 mean=tf.cast(np.log(0.5 / self.latent_dim), dtype=tf.float64),
                                 stddev=tf.cast(tf.sqrt(2 / (self.output_dim + self.latent_dim)), dtype=tf.float64)))

            vs.assign(
                self.RLS_MAT,
                tf.random.normal(shape=(self.latent_dim, self.input_dim),
                                 dtype=tf.float64,
                                 mean=tf.cast(np.log(0.5 / self.latent_dim), dtype=tf.float64),
                                 stddev=tf.cast(tf.sqrt(2 / (self.latent_dim + self.input_dim)), dtype=tf.float64)))

        for i in range(self.output_dim):

            if length_scale_init == "median":
                ls_init = self.ys_euclidean_percentiles[2]

                ls_rand_range = tf.minimum(self.ys_euclidean_percentiles[2] - self.ys_euclidean_percentiles[0],
                                           self.ys_euclidean_percentiles[4] - self.ys_euclidean_percentiles[2])

                ls_init += tf.random.uniform(shape=(i, ), minval=-ls_rand_range, maxval=ls_rand_range, dtype=tf.float64)

                vs.assign(f"{i}/output_length_scales", ls_init)

            else:
                vs.assign(f"{i}/output_length_scales",
                          tf.random.uniform(shape=(i, ), minval=init_minval, maxval=init_maxval, dtype=tf.float64))

            vs.assign(f"{i}/signal_amplitude",
                      tf.random.uniform(shape=(1, ), minval=init_minval, maxval=init_maxval, dtype=tf.float64))

            vs.assign(f"{i}/noise_amplitude",
                      tf.random.uniform(shape=(1, ), minval=init_minval, maxval=init_maxval, dtype=tf.float64))

    def fit(self, xs, ys, optimizer="adam", optimizer_restarts=1, trace=True, iters=200, rate=1e-2) -> None:

        xs, ys = self._validate_and_convert_input_output(xs, ys)

        logger.info(f"Training data supplied with xs shape {xs.shape} and ys shape {ys.shape}, training!")

        self._calculate_statistics_for_median_initialization_heuristic(xs, ys)

        # Create dummy variables for training
        vs = self.create_hyperparameters()

        # Define i-th GP training loss
        def negative_gp_log_likelihood(idx, signal_amplitude, length_scales, noise_amplitude):

            gp = GaussianProcess(kernel=self.kernel_name,
                                 signal_amplitude=signal_amplitude,
                                 length_scales=length_scales,
                                 noise_amplitude=noise_amplitude)

            gp_input = tf.concat((xs, ys[:, :idx]), axis=1)

            return -gp.log_pdf(gp_input, ys[:, idx:idx + 1], normalize=True)

        # Define the GPAR training loss
        def negative_mf_gpar_log_likelihood(vs):
            losses = []

            # Create length scale matrix from its left and right factors
            length_scales = tf.matmul(vs[self.LLS_MAT], vs[self.RLS_MAT])

            for k in range(self.output_dim):
                # Create a single length scale vector by concatenating
                # the input and the output length scales
                gp_length_scales = tf.concat((length_scales[k, :], vs[f"{k}/output_length_scales"]), axis=0)

                losses.append(
                    negative_gp_log_likelihood(idx=k,
                                               signal_amplitude=vs[f"{k}/signal_amplitude"],
                                               length_scales=gp_length_scales,
                                               noise_amplitude=vs[f"{k}/noise_amplitude"]))

            return tf.add_n(losses)

        best_loss = np.inf

        i = 0
        # Train N MF-GPAR models and select the best one
        while i < optimizer_restarts:

            i += 1

            if self.verbose:
                print("-------------------------------")
                print(f"Training iteration {i}")
                print("-------------------------------")

            # Re-initialize to a random configuration
            self.initialize_hyperparameters(vs, length_scale_init="median")

            loss = np.inf

            try:
                if optimizer == "l-bfgs-b":
                    # Perform L-BFGS-B optimization
                    loss = minimise_l_bfgs_b(negative_mf_gpar_log_likelihood,
                                             vs,
                                             err_level="raise",
                                             trace=trace,
                                             iters=iters)

                elif optimizer == "adam":

                    loss = minimise_adam(negative_mf_gpar_log_likelihood, vs)
                else:
                    ModelError("unrecognized loss!")

            except tf.errors.InvalidArgumentError as e:
                logger.error(str(e))
                loss = np.nan
            except Exception as e:
                print("Iteration {} failed: {}".format(i, str(e)))

            if loss < best_loss:

                if self.verbose:
                    print("New best loss: {:.3f}".format(loss))

                best_loss = loss
                self.models.clear()

                # Assign the hyperparameters to the model variables
                self.left_length_scale_matrix.assign(vs[self.LLS_MAT])
                self.right_length_scale_matrix.assign(vs[self.RLS_MAT])

                input_length_scales = tf.matmul(self.left_length_scale_matrix, self.right_length_scale_matrix)

                for j in range(self.output_dim):
                    self.output_length_scales[j].assign(vs[f"{j}/output_length_scales"])
                    self.signal_amplitudes[j].assign(vs[f"{j}/signal_amplitude"])
                    self.noise_amplitudes[j].assign(vs[f"{j}/noise_amplitude"])

                    self.length_scales[j] = tf.concat((input_length_scales[j, :], self.output_length_scales[j]), axis=0)

            elif self.verbose:
                print("Loss: {:.3f}".format(loss))

            if np.isnan(loss):
                print("Loss was NaN, restarting training iteration!")
                i -= 1

        self.trained.assign(True)

    def create_gps(self):
        self.models.clear()

        self.length_scales.clear()

        # dimensions O x I
        input_length_scales = tf.matmul(self.left_length_scale_matrix, self.right_length_scale_matrix)

        for i in range(self.output_dim):
            # i-th length scales
            length_scales = tf.concat((input_length_scales[i, :], self.output_length_scales[i]), axis=0)

            self.length_scales.append(length_scales)

        for i in range(self.output_dim):
            gp = GaussianProcess(kernel=self.kernel_name,
                                 signal_amplitude=self.signal_amplitudes[i],
                                 length_scales=self.length_scales[i],
                                 noise_amplitude=self.noise_amplitudes[i])

            self.models.append(gp)

    @staticmethod
    def restore(save_path):

        with open(save_path + ".json", "r") as config_file:
            config = json.load(config_file)

        model = GPARModel.from_config(config)

        model.load_weights(save_path)
        model.create_gps()

        return model

    def get_config(self):

        return {
            "name": self.name,
            "kernel": self.kernel_name,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "latent_dim": self.latent_dim,
            "learn_permutation": self.learn_permutation,
            "denoising": self.denoising,
            "initialization_heuristic": self.initialization_heuristic,
            "verbose": self.verbose,
        }

    @staticmethod
    def from_config(config, restore_num_data_points=False):
        return MatrixFactorizedGPARModel(**config)
