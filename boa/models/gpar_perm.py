import logging

from tqdm import trange

import numpy as np
import tensorflow as tf

from boa.core import GaussianProcess, setup_logger, tf_bounded_variable
from .abstract_model_v2 import AbstractModel, ModelError
from .gpar_v2 import GPARModel

logger = setup_logger(__name__, level=logging.DEBUG, to_console=True, log_file="logs/perm_gpar.log")


class PermutedGPARModel(GPARModel):
    PERM_NAME = "permutation"

    def __init__(self,
                 kernel: str,
                 input_dim: int,
                 output_dim: int,
                 initialization_heuristic: str = "median",
                 denoising: bool = False,
                 verbose: bool = False,
                 name: str = "permuted_gpar_model",
                 **kwargs):

        super(PermutedGPARModel, self).__init__(kernel=kernel,
                                                input_dim=input_dim,
                                                output_dim=output_dim,
                                                initialization_heuristic=initialization_heuristic,
                                                denoising=denoising,
                                                verbose=verbose,
                                                name=name,
                                                **kwargs)

        self.permutation = tf.Variable(tf.eye(output_dim, dtype=tf.float64), name=self.PERM_NAME)
        self.soft_perm = tf.Variable(tf.eye(output_dim, dtype=tf.float64), name=self.PERM_NAME + "_soft")

        # Empty previous hyperparam lists
        self.length_scales = []
        self.signal_amplitudes = []
        self.noise_amplitudes = []

        # Create TF variables for each of the hyperparameters, so that
        # we can use Keras' serialization features
        for i in range(self.output_dim):
            # Note the scaling in dimension
            self.length_scales.append(tf.Variable(tf.ones(self.input_dim + i,
                                                          dtype=tf.float64),
                                                  name=f"{i}/length_scales"))

            self.signal_amplitudes.append(tf.Variable((1,),
                                                      dtype=tf.float64,
                                                      name=f"{i}/signal_amplitude"))

            self.noise_amplitudes.append(tf.Variable((1,),
                                                     dtype=tf.float64,
                                                     name=f"{i}/noise_amplitude"))

    def permutation_matrix(self, log_mat, temperature, sinkhorn_iterations=20, soft=True):

        temperature = tf.cast(temperature, tf.float64)

        # Add Gumbel noise for robustness
        log_perm_mat = log_mat - tf.math.log(-tf.math.log(tf.random.uniform(shape=log_mat.shape,
                                                                            dtype=tf.float64) + 1e-20))
        log_perm_mat = log_perm_mat - tf.math.log(temperature)

        # Perform Sinkhorn normalization
        for _ in range(sinkhorn_iterations):
            # Column-wise normalization in log domain
            log_perm_mat = log_perm_mat - tf.reduce_logsumexp(log_perm_mat, axis=1, keepdims=True)

            # Row-wise normalization in log domain
            log_perm_mat = log_perm_mat - tf.reduce_logsumexp(log_perm_mat, axis=0, keepdims=True)

        soft_perm_mat = tf.exp(log_perm_mat)

        hard_perm_mat = tf.one_hot(tf.argmax(soft_perm_mat, axis=0), self.output_dim, dtype=tf.float64)

        return soft_perm_mat, hard_perm_mat

    def get_trainable_variables(self):
        return [] + \
               list(map(lambda x: x[0], self.length_scales)) + \
               list(map(lambda x: x[0], self.signal_amplitudes)) + \
               list(map(lambda x: x[0], self.noise_amplitudes))

    # [self.permutation] + \

    def initialize_hyperparameters(self,
                                   length_scale_init="random",
                                   init_minval=0.5,
                                   init_maxval=2.0):

        permutation = tf.Variable(tf.random.uniform(shape=(self.output_dim, self.output_dim), dtype=tf.float64))

        length_scales = []
        signal_amplitudes = []
        noise_amplitudes = []

        for index in range(self.output_dim):
            ls_name = f"{index}/length_scales"
            gp_var_name = f"{index}/signal_amplitude"
            noise_var_name = f"{index}/noise_amplitude"

            if length_scale_init == "median":

                # Center on the medians, treat the inputs and the outputs separately
                xs_ls_init = self.xs_euclidean_percentiles[2]
                ys_ls_init = self.ys_euclidean_percentiles[2]

                xs_ls_rand_range = tf.minimum(self.xs_euclidean_percentiles[2] - self.xs_euclidean_percentiles[0],
                                              self.xs_euclidean_percentiles[4] - self.xs_euclidean_percentiles[2])

                ys_ls_rand_range = tf.minimum(self.ys_euclidean_percentiles[2] - self.ys_euclidean_percentiles[0],
                                              self.ys_euclidean_percentiles[4] - self.ys_euclidean_percentiles[2])

                xs_ls_init += tf.random.uniform(shape=(self.input_dim,),
                                                minval=-xs_ls_rand_range,
                                                maxval=xs_ls_rand_range,
                                                dtype=tf.float64)

                ys_ls_init += tf.random.uniform(shape=(index,),
                                                minval=-ys_ls_rand_range,
                                                maxval=ys_ls_rand_range,
                                                dtype=tf.float64)

                # Once the inputs and outputs have been initialized separately, concatenate them
                ls_init = tf.concat((xs_ls_init, ys_ls_init), axis=0)

            else:
                ls_init = tf.random.uniform(shape=(self.input_dim + index,),
                                            minval=init_minval,
                                            maxval=init_maxval,
                                            dtype=tf.float64)

            # Note the scaling in dimension
            length_scales.append(tf_bounded_variable(ls_init,
                                                     lower=1e-3,
                                                     upper=1e2))

            signal_amplitudes.append(tf_bounded_variable(tf.random.uniform(shape=(1,),
                                                                           minval=init_minval,
                                                                           maxval=init_maxval,
                                                                           dtype=tf.float64),
                                                         lower=1e-4,
                                                         upper=1e4))

            noise_amplitudes.append(tf_bounded_variable(tf.random.uniform(shape=(1,),
                                                                          minval=init_minval,
                                                                          maxval=init_maxval,
                                                                          dtype=tf.float64),
                                                        lower=1e-6,
                                                        upper=1e2))

        return permutation, length_scales, signal_amplitudes, noise_amplitudes

    def fit(self, xs, ys, optimizer_restarts=1, learn_rate=1e-1, tol=1e-6, iters=1000, start_temp=2., end_temp=1e-10) -> None:

        xs, ys = self._validate_and_convert_input_output(xs, ys)

        logger.info(f"Training data supplied with xs shape {xs.shape} and ys shape {ys.shape}, training!")

        self._calculate_statistics_for_median_initialization_heuristic(xs, ys)

        # Robust optimization
        j = 0

        best_loss = np.inf

        while j < optimizer_restarts:

            j += 1

            # Create dummy variables for optimization
            hps = self.initialize_hyperparameters(length_scale_init=self.initialization_heuristic)

            permutation, length_scales, signal_amplitudes, noise_amplitudes = hps

            hps = [permutation] + \
                  list(map(lambda x: x[0], length_scales)) + \
                  list(map(lambda x: x[0], signal_amplitudes)) + \
                  list(map(lambda x: x[0], noise_amplitudes))

            # Epsilon set to 1e-8 to match Wessel's Varz Adam settings.
            optimizer = tf.optimizers.Adam(learn_rate,
                                           epsilon=1e-8)
            prev_loss = np.inf

            with trange(iters) as t:

                for iteration in t:
                    with tf.GradientTape(watch_accessed_variables=True) as tape:

                        loss = 0

                        tape.watch(hps)

                        temperature = tf.maximum((end_temp - start_temp) / (iters / 2) * iteration + start_temp, end_temp)

                        soft_perm, hard_perm = self.permutation_matrix(permutation,
                                                                       temperature=temperature,
                                                                       sinkhorn_iterations=20)

                        soft_permuted_ys = tf.matmul(ys, soft_perm)
                        hard_permuted_ys = tf.matmul(ys, hard_perm)

                        # Forward pass: use hard permutation
                        # Backward pass: pretend we used the soft permutation all along
                        # permuted_output = soft_permuted_ys + tf.stop_gradient(hard_permuted_ys - soft_permuted_ys)
                        permuted_output = soft_permuted_ys

                        for i in range(self.output_dim):
                            # Define i-th GP training loss
                            # Create i-th GP
                            gp = GaussianProcess(kernel=self.kernel_name,
                                                 signal_amplitude=signal_amplitudes[i][1](signal_amplitudes[i][0]),
                                                 length_scales=length_scales[i][1](length_scales[i][0]),
                                                 noise_amplitude=noise_amplitudes[i][1](noise_amplitudes[i][0]))

                            # Create input to the i-th GP
                            ys_to_append = permuted_output[:, :i]
                            gp_input = tf.concat((xs, ys_to_append), axis=1)

                            gp_nll = -gp.log_pdf(gp_input, ys[:, i:i + 1], normalize=True)

                            loss += gp_nll

                    if tf.abs(prev_loss - loss) < tol:
                        logger.info(
                            f"Loss decreased less than {tol}, optimisation terminated at iteration {iteration}.")
                        break

                    prev_loss = loss

                    gradients = tape.gradient(loss, hps)
                    optimizer.apply_gradients(zip(gradients, hps))

                    t.set_description(f"Loss at iteration {iteration}: {loss:.3f}, temperature: {temperature:.5f}")

            if loss < best_loss:

                logger.info(f"Output {i}, Iteration {j}: New best loss: {loss:.3f}")

                best_loss = loss

                for i in range(self.output_dim):
                    # Assign the hyperparameters for each input to the model variables
                    self.length_scales[i].assign(length_scales[i][1](length_scales[i][0]))
                    self.signal_amplitudes[i].assign(signal_amplitudes[i][1](signal_amplitudes[i][0]))
                    self.noise_amplitudes[i].assign(noise_amplitudes[i][1](noise_amplitudes[i][0]))

                    soft_perm, perm = self.permutation_matrix(permutation, 1e-10, sinkhorn_iterations=100)
                    self.permutation.assign(perm)
                    self.soft_perm.assign(soft_perm)

            else:
                logger.info(f"Output {i}, Iteration {j}: Loss: {loss:.3f}")

            if np.isnan(loss) or np.isinf(loss):
                logger.error(f"Output {i}, Iteration {j}: Loss was {loss}, restarting training iteration!")
                j = j - 1

        self.trained.assign(True)

    def predict(self, xs, numpy=False):

        if not self.trained:
            logger.warning("Using untrained model for prediction!")

        if len(self.models) < self.output_dim:
            logger.info("GPs haven't been cached yet, creating them now.")
            self.create_gps()

        xs = self._validate_and_convert(xs, output=False)

        means = []
        variances = []

        permuted_ys = tf.matmul(self.ys, self.permutation)

        for i, model in enumerate(self.models):
            gp_input = tf.concat([xs] + means, axis=1)
            gp_train_input = tf.concat([self.xs, permuted_ys[:, :i]], axis=1)

            model = model | (gp_train_input, permuted_ys[:, i: i + 1])

            mean, var = model.predict(gp_input, latent=False)

            means.append(mean)
            variances.append(var)

        means = tf.concat(means, axis=1)
        variances = tf.concat(variances, axis=1)

        # Permute back the output
        perm_inverse = tf.transpose(self.permutation)

        means = tf.matmul(means, perm_inverse)
        variances = tf.matmul(variances, perm_inverse)

        if numpy:
            means = means.numpy()
            variances = variances.numpy()

        return means, variances
