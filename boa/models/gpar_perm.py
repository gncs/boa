import logging

from tqdm import trange
from scipy.optimize import linear_sum_assignment

import numpy as np
import tensorflow as tf

from boa.core import GaussianProcess, setup_logger
from .gpar import GPARModel

from not_tf_opt import minimize, BoundedVariable

from boa import ROOT_DIR

logger = setup_logger(__name__, level=logging.DEBUG, to_console=True, log_file=f"{ROOT_DIR}/../logs/perm_gpar.log")


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
            self.length_scales.append(
                tf.Variable(tf.ones(self.input_dim + i, dtype=tf.float64), name=f"{i}/length_scales"))

            self.signal_amplitudes.append(tf.Variable((1, ), dtype=tf.float64, name=f"{i}/signal_amplitude"))

            self.noise_amplitudes.append(tf.Variable((1, ), dtype=tf.float64, name=f"{i}/noise_amplitude"))

    def permutation_matrix(self, log_mat, temperature, sinkhorn_iterations=20, soft=True):

        temperature = tf.cast(temperature, tf.float64)

        # Add Gumbel noise for robustness
        # log_perm_mat = log_mat - tf.math.log(
        #     -tf.math.log(tf.random.uniform(shape=log_mat.shape, dtype=tf.float64) + 1e-20))
        log_perm_mat = log_mat - tf.math.log(temperature)

        # Perform Sinkhorn normalization
        for _ in range(sinkhorn_iterations):
            # Column-wise normalization in log domain
            log_perm_mat = log_perm_mat - tf.reduce_logsumexp(log_perm_mat, axis=1, keepdims=True)

            # Row-wise normalization in log domain
            log_perm_mat = log_perm_mat - tf.reduce_logsumexp(log_perm_mat, axis=0, keepdims=True)

        soft_perm_mat = tf.exp(log_perm_mat)

        hard_perm_mat = tf.one_hot(tf.argmax(soft_perm_mat, axis=0), self.output_dim, dtype=tf.float64)

        return soft_perm_mat, hard_perm_mat

    def create_hyperparameter_initializers(self, length_scale_init="random", init_minval=0.5, init_maxval=2.0):

        permutation = tf.Variable(tf.random.uniform(shape=(self.output_dim, self.output_dim), dtype=tf.float64))

        length_scales = []
        signal_amplitudes = []
        noise_amplitudes = []

        for index in range(self.output_dim):

            if length_scale_init == "median":

                # Center on the medians, treat the inputs and the outputs separately
                xs_ls_init = self.xs_euclidean_percentiles[2]
                ys_ls_init = self.ys_euclidean_percentiles[2]

                xs_ls_rand_range = tf.minimum(self.xs_euclidean_percentiles[2] - self.xs_euclidean_percentiles[0],
                                              self.xs_euclidean_percentiles[4] - self.xs_euclidean_percentiles[2])

                ys_ls_rand_range = tf.minimum(self.ys_euclidean_percentiles[2] - self.ys_euclidean_percentiles[0],
                                              self.ys_euclidean_percentiles[4] - self.ys_euclidean_percentiles[2])

                xs_ls_init += tf.random.uniform(shape=(self.input_dim, ),
                                                minval=-xs_ls_rand_range,
                                                maxval=xs_ls_rand_range,
                                                dtype=tf.float64)

                ys_ls_init += tf.random.uniform(shape=(index, ),
                                                minval=-ys_ls_rand_range,
                                                maxval=ys_ls_rand_range,
                                                dtype=tf.float64)

                # Once the inputs and outputs have been initialized separately, concatenate them
                ls_init = tf.concat((xs_ls_init, ys_ls_init), axis=0)

            else:
                ls_init = tf.random.uniform(shape=(self.input_dim + index, ),
                                            minval=init_minval,
                                            maxval=init_maxval,
                                            dtype=tf.float64)

            # Note the scaling in dimension
            length_scales.append(BoundedVariable(ls_init, lower=3e-2, upper=1e2))

            signal_amplitudes.append(
                BoundedVariable(tf.random.uniform(shape=(1, ), minval=init_minval, maxval=init_maxval,
                                                  dtype=tf.float64),
                                lower=1e-2,
                                upper=1e2))

            noise_amplitudes.append(
                BoundedVariable(tf.random.uniform(shape=(1, ), minval=init_minval, maxval=init_maxval,
                                                  dtype=tf.float64),
                                lower=1e-6,
                                upper=1e2))

        return permutation, length_scales, signal_amplitudes, noise_amplitudes

    def fit(self,
            xs,
            ys,
            optimizer_restarts=1,
            learn_rate=1e-1,
            tolerance=1e-6,
            iters=1000,
            start_temp=2.,
            end_temp=1e-10,
            beta=1.,
            use_bfgs=False,
            hard_forward_permutation=False) -> None:

        xs, ys = self._validate_and_convert_input_output(xs, ys)

        logger.info(f"Training data supplied with xs shape {xs.shape} and ys shape {ys.shape}, training!")

        self._calculate_statistics_for_median_initialization_heuristic(xs, ys)

        # Robust optimization
        j = 0

        best_loss = np.inf

        while j < optimizer_restarts:

            j += 1

            # Create dummy variables for optimization
            hps = self.create_hyperparameter_initializers(length_scale_init=self.initialization_heuristic)

            permutation, length_scales, signal_amplitudes, noise_amplitudes = hps

            hps = [permutation] + \
                  list(map(lambda x: x.reparameterization, signal_amplitudes)) + \
                  list(map(lambda x: x.reparameterization, noise_amplitudes))

            if use_bfgs:

                for i in range(self.output_dim):
                    # Define i-th GP training loss
                    def negative_gp_log_likelihood(signal_amplitude, length_scales, noise_amplitude):
                        gp = GaussianProcess(kernel=self.kernel_name,
                                             input_dim=self.input_dim + 1,
                                             signal_amplitude=signal_amplitude,
                                             length_scales=length_scales,
                                             noise_amplitude=noise_amplitude)

                        # ys_to_append = pred_ys if self.denoising else ys[:, :i]

                        # Permute the output
                        ys_to_append = ys[:, :i]
                        gp_input = tf.concat((xs, ys_to_append), axis=1)

                        return -gp.log_pdf(gp_input, ys[:, i:i + 1], normalize_with_input=True)

                    loss, converged, diverged = minimize(
                        negative_gp_log_likelihood,
                        vs=(signal_amplitudes[i], length_scales[i], noise_amplitudes[i]),
                        parallel_iterations=10)

                    if diverged:
                        logger.error(f"Model diverged, restarting iteration {j} (loss was {loss:.3f})!")
                        j -= 1
                        continue

            else:
                lr = tf.Variable(learn_rate, dtype=tf.float64)

                # Epsilon set to 1e-8 to match Wessel's Varz Adam settings.
                optimizer = tf.optimizers.Adam(lr, epsilon=1e-8)
                prev_loss = np.inf

                target_beta = tf.cast(beta, tf.float64)

                with trange(iters) as t:

                    for iteration in t:
                        with tf.GradientTape(watch_accessed_variables=False) as tape:

                            loss = 0

                            tape.watch(hps)

                            temperature = tf.maximum((end_temp - start_temp) / (iters / 2) * iteration + start_temp,
                                                     end_temp)

                            # Warm-up for beta
                            beta = tf.minimum(iteration * target_beta / (iters / 2), target_beta)

                            # Learning rate schedule
                            if iteration == iters // 3:
                                lr.assign(learn_rate / 3)

                            if iteration == 2 * iters // 3:
                                lr.assign(learn_rate / 10)

                            soft_perm, hard_perm = self.permutation_matrix(permutation,
                                                                           temperature=temperature,
                                                                           sinkhorn_iterations=20)

                            x_ind, y_ind = linear_sum_assignment(soft_perm.numpy())
                            hard_perm_ = np.zeros(soft_perm.shape)
                            hard_perm_[x_ind, y_ind] = 1
                            hard_perm_ = tf.convert_to_tensor(hard_perm_, dtype=tf.float64)

                            soft_permuted_ys = tf.matmul(ys, soft_perm)
                            hard_permuted_ys = tf.matmul(ys, hard_perm)
                            hard_permuted_ys_ = tf.matmul(ys, hard_perm_)

                            if hard_forward_permutation:
                                # Forward pass: use hard permutation
                                # Backward pass: pretend we used the soft permutation all along
                                permuted_output = soft_permuted_ys + tf.stop_gradient(hard_permuted_ys_ -
                                                                                      soft_permuted_ys)
                            else:
                                permuted_output = soft_permuted_ys

                            for i in range(self.output_dim):
                                # Define i-th GP training loss
                                # Create i-th GP
                                gp = GaussianProcess(kernel=self.kernel_name,
                                                     input_dim=self.input_dim + i,
                                                     signal_amplitude=signal_amplitudes[i](),
                                                     length_scales=length_scales[i](),
                                                     noise_amplitude=noise_amplitudes[i]())

                                # Create input to the i-th GP
                                ys_to_append = permuted_output[:, :i]
                                gp_input = tf.concat((xs, ys_to_append), axis=1)

                                gp_nll = -gp.log_pdf(gp_input, ys[:, i:i + 1], normalize_with_input=True)

                                # Frobenius entropy of the soft permutation
                                perm_entropy = soft_perm * tf.math.log(soft_perm)
                                perm_entropy = -tf.reduce_sum(perm_entropy)

                                loss += gp_nll + beta * perm_entropy

                        if tf.abs(prev_loss - loss) < tolerance:
                            logger.info(
                                f"Loss decreased less than {tolerance}, optimisation terminated at iteration {iteration}."
                            )
                            break

                        prev_loss = loss

                        gradients = tape.gradient(loss, hps)
                        optimizer.apply_gradients(zip(gradients, hps))

                        t.set_description(f"Loss at iteration {iteration}: {loss:.3f}, temperature: {temperature:.5f}, "
                                          f"NLL: {gp_nll:.3f}, Entropy: {perm_entropy:.3f}, Beta: {beta:.3f}, "
                                          f"Learning Rate: {lr.value().numpy():.3f}")

            if loss < best_loss:

                logger.info(f"Output {i}, Iteration {j}: New best loss: {loss:.3f}")

                best_loss = loss

                for i in range(self.output_dim):
                    # Assign the hyperparameters for each input to the model variables
                    self.length_scales[i].assign(length_scales[i]())
                    self.signal_amplitudes[i].assign(signal_amplitudes[i]())
                    self.noise_amplitudes[i].assign(noise_amplitudes[i]())

                    soft_perm, perm = self.permutation_matrix(permutation, end_temp, sinkhorn_iterations=100)
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

            model = model | (gp_train_input, permuted_ys[:, i:i + 1])

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
