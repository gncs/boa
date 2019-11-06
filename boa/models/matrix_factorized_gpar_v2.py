from typing import Tuple, List

import numpy as np
import tensorflow as tf
from varz.tensorflow import Vars, minimise_l_bfgs_b

from .abstract_model_v2 import ModelError
from .gpar_v2 import GPARModel


class MatrixFactorizedGPARModel(GPARModel):

    LLLS_MAT = "left_log_length_scale_matrix"
    RLLS_MAT = "right_log_length_scale_matrix"

    def __init__(self,
                 kernel: str,
                 num_optimizer_restarts: int,
                 latent_dim: int,
                 verbose: bool = False,
                 name="matrix_factorized_gpar_model",
                 **kwargs):
        super(MatrixFactorizedGPARModel, self).__init__(kernel=kernel,
                                                        num_optimizer_restarts=num_optimizer_restarts,
                                                        verbose=verbose,
                                                        name=name,
                                                        **kwargs)

        self.latent_dim = latent_dim

        self.output_length_scales = []

    def __or__(self, inputs: Tuple) -> tf.keras.Model:
        # Validate and assign inputs
        super(MatrixFactorizedGPARModel, self).__or__(inputs)

        # Create TF variables for the hyperparameters

        # Create low rank representation for the input length scales
        self.left_log_length_scale_matrix = tf.Variable(
            tf.ones((self.output_dim, self.latent_dim), dtype=tf.float64),
            trainable=False,
            name=self.LLLS_MAT)

        self.right_log_length_scale_matrix = tf.Variable(
            tf.ones((self.latent_dim, self.input_dim), dtype=tf.float64),
            trainable=False,
            name=self.RLLS_MAT)

        # dimensions O x I
        input_log_length_scales = tf.matmul(self.left_log_length_scale_matrix,
                                            self.right_log_length_scale_matrix)

        input_length_scales = tf.exp(input_log_length_scales)

        for i in range(self.output_dim):

            # Length scales for the output dimensions only
            out_length_scales = tf.Variable(tf.ones(i, dtype=tf.float64),
                                            name=f"output_length_scale_dim_{i}")

            self.output_length_scales.append(out_length_scales)

            # i-th length scales
            length_scales = tf.concat((input_length_scales[i, :],
                                       out_length_scales),
                                      axis=0)

            self.length_scales.append(length_scales)

        self._update_mean_std()

        return self

    def create_hyperparameters(self,
                               log_lower_bound: float = -7,
                               log_upper_bound: float = 7) -> Vars:

        vs = Vars(tf.float64)

        # Create container for matrix factors
        vs.bnd(init=tf.ones((self.output_dim, self.latent_dim),
                            dtype=tf.float64),
               lower=log_lower_bound,
               upper=log_upper_bound,
               name=self.LLLS_MAT)

        vs.bnd(init=tf.ones((self.latent_dim, self.input_dim),
                            dtype=tf.float64),
               lower=log_lower_bound,
               upper=log_upper_bound,
               name=self.RLLS_MAT)

        # Create the rest of the hyperparameters

        for i in range(self.output_dim):
            # Note the scaling in dimension with the index
            vs.bnd(init=tf.ones(i, dtype=tf.float64),
                   lower=1e-4,
                   upper=1e4,
                   name="output_length_scales_dim_{}".format(i))

            # GP variance
            vs.pos(init=tf.ones(1, dtype=tf.float64),
                   name="gp_variance_dim_{}".format(i))

            # Noise variance: bound between 1e-4 and 1e4
            vs.bnd(init=tf.ones(1, dtype=tf.float64),
                   lower=1e-4,
                   upper=1e4,
                   name="noise_variance_dim_{}".format(i))

        return vs

    def initialize_hyperparameters(self,
                                   vs: Vars,
                                   log_lower_bound: float = -7,
                                   log_upper_bound: float = 7 ) -> None:

        vs.assign(self.LLLS_MAT,
                  tf.random.uniform(shape=(self.output_dim, self.latent_dim),
                                    dtype=tf.float64,
                                    minval=log_lower_bound,
                                    maxval=log_upper_bound))

        vs.assign(self.RLLS_MAT,
                  tf.random.uniform(shape=(self.latent_dim, self.input_dim),
                                    dtype=tf.float64,
                                    minval=log_lower_bound,
                                    maxval=log_upper_bound))

        for i in range(self.output_dim):

            vs.assign("output_length_scales_dim_{}".format(i),
                      tf.random.uniform(shape=(i,),
                                        minval=self.init_minval,
                                        maxval=self.init_maxval,
                                        dtype=tf.float64))

            vs.assign("gp_variance_dim_{}".format(i),
                      tf.random.uniform(shape=(1,),
                                        minval=self.init_minval,
                                        maxval=self.init_maxval,
                                        dtype=tf.float64))

            vs.assign("noise_variance_dim_{}".format(i),
                      tf.random.uniform(shape=(1,),
                                        minval=self.init_minval,
                                        maxval=self.init_maxval,
                                        dtype=tf.float64))

    def train(self) -> None:

        self._update_mean_std()
        x_normalized = self.normalize(self.xs, mean=self.xs_mean, std=self.xs_std)
        y_normalized = self.normalize(self.ys, mean=self.ys_mean, std=self.ys_std)

        self.models.clear()

        # Create dummy variables for training
        vs = self.create_hyperparameters()

        # Define i-th GP training loss
        def negative_gp_log_likelihood(idx, gp_var, length_scale, noise_var):
            prior_gp_ = self.get_prior_gp_model(length_scale,
                                                gp_var,
                                                noise_var)

            gp_input = tf.concat((x_normalized, y_normalized[:, :idx]), axis=1)

            return -prior_gp_(gp_input).logpdf(y_normalized[:, idx:idx + 1])

        # Define the GPAR training loss
        def negative_mf_gpar_log_likelihood(vs):
            losses = []

            # Create length scale matrix from its left and right factors
            log_length_scales = tf.matmul(vs[self.LLLS_MAT],
                                          vs[self.RLLS_MAT])

            length_scales = tf.exp(log_length_scales)

            for i in range(self.output_dim):

                # Create a single length scale vector by concatenating
                # the input and the output length scales
                gp_length_scales = tf.concat((length_scales[i, :],
                                              vs[f"output_length_scales_dim_{i}"]),
                                             axis=0)

                losses.append(
                    negative_gp_log_likelihood(idx=i,
                                               gp_var=vs[f"gp_variance_dim_{i}"],
                                               length_scale=gp_length_scales,
                                               noise_var=vs[f"noise_variance_dim_{i}"]))

            return tf.add_n(losses)

        best_loss = np.inf

        # Train N MF-GPAR models and select the best one
        for i in range(self.num_optimizer_restarts):

            # Re-initialize to a random configuration
            self.initialize_hyperparameters(vs)

            loss = np.inf

            try:
                # Perform L-BFGS-B optimization
                loss = minimise_l_bfgs_b(negative_mf_gpar_log_likelihood, vs)

            except Exception as e:
                print("Iteration {} failed: {}".format(i + 1, str(e)))

            if loss < best_loss:

                best_loss = loss
                self.models.clear()

                # Assign the hyperparameters to the model variables
                self.left_log_length_scale_matrix.assign(vs[self.LLLS_MAT])
                self.right_log_length_scale_matrix.assign(vs[self.RLLS_MAT])

                input_log_length_scales = tf.matmul(self.left_log_length_scale_matrix,
                                                    self.right_log_length_scale_matrix)

                input_length_scales = tf.exp(input_log_length_scales)

                for j in range(self.output_dim):

                    self.output_length_scales[j].assign(vs[f"output_length_scales_dim_{j}"])
                    self.gp_variances[j].assign(vs[f"gp_variance_dim_{j}"])
                    self.noise_variances[j].assign(vs[f"noise_variance_dim_{j}"])

                    length_scales = tf.concat((input_length_scales[j, :],
                                               self.output_length_scales[j]),
                                              axis=0)

                    prior_gp = self.get_prior_gp_model(length_scales,
                                                       self.gp_variances[j],
                                                       self.noise_variances[j])

                    gp_input = tf.concat((x_normalized, y_normalized[:, :j]),
                                         axis=1)

                    # Condition the model
                    best_model = prior_gp | (gp_input, y_normalized[:, j:j + 1])

                    self.models.append(best_model)

