import logging
import json
from tqdm import trange

import numpy as np
import tensorflow as tf

from stheno.tensorflow import dense

from .multi_output_gp_regression_model import MultiOutputGPRegressionModel, ModelError
from boa.core import GaussianProcess, setup_logger, inv_perm

from not_tf_opt import minimize, BoundedVariable

from boa import ROOT_DIR

logger = setup_logger(__name__, level=logging.DEBUG, to_console=True, log_file=f"{ROOT_DIR}/../logs/gpar.log")


class GPARModel(MultiOutputGPRegressionModel):
    def __init__(self,
                 kernel: str,
                 input_dim: int,
                 output_dim: int,
                 denoising: bool = False,
                 verbose: bool = False,
                 name: str = "gpar_model",
                 **kwargs):
        """
        Constructor of GPAR model.

        :param kernel: name of kernel
        :param num_optimizer_restarts: number of times the optimization of the hyperparameters is restarted
        :param verbose: log optimization of hyperparameters
        """

        super(GPARModel, self).__init__(kernel=kernel,
                                        input_dim=input_dim,
                                        output_dim=output_dim,
                                        verbose=verbose,
                                        name=name,
                                        **kwargs)

        self.denoising = denoising

        self.permutation = tf.Variable(tf.range(output_dim, dtype=tf.int32))

    def gp_input(self, index, xs, ys):
        return tf.concat([xs, ys[:, :index]], axis=1)

    def gp_input_dim(self, index):
        return self.input_dim + index

    def gp_output(self, index, ys):
        return ys[:, index:index + 1]

    def has_explicit_length_scales(self):
        return True

    def gp_predictive_input(self, xs, means):
        return tf.concat([xs] + means, axis=1)

    def fit_greedy_ordering(self,
                            train_xs,
                            train_ys,
                            validation_xs,
                            validation_ys,
                            optimizer="l-bfgs-b",
                            optimizer_restarts=1,
                            trace=False,
                            num_target_dimensions=0,
                            iters=1000,
                            tolerance=1e-5,
                            seed=None,
                            rate=1e-2,
                            error_level="catch"):
        """
        Perform the greedy search for the optimal output ordering described in the GPAR paper.
        """

        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)

        # Pre fitting data perparation
        train_xs, train_ys = self._validate_and_convert_input_output(train_xs, train_ys)
        validation_xs, validation_ys = self._validate_and_convert_input_output(validation_xs, validation_ys)

        logger.info(f"Training data supplied with xs shape {train_xs.shape} and ys shape {train_ys.shape}, training!")

        self._calculate_statistics_for_median_initialization_heuristic(train_xs, train_ys)

        # Permutation to use for prediction
        permutation = []

        # Fit "auxiliary" dimensions one-by-one
        # We omit the "target" dimensions at the end of the column-space of ys
        for i in range(self.output_dim - num_target_dimensions):

            best_validation_log_prob = np.inf
            best_candidate = 0

            logger.info(f"Selecting output {i}!")

            # Search through the remaining output dimensions
            remaining_dimensions = [
                dim for dim in range(self.output_dim - num_target_dimensions) if dim not in permutation
            ]

            # Perform robust optimization for each candidate loss
            for candidate_dim in remaining_dimensions:

                logger.info(f"Training candidate dimension {candidate_dim} for output {i}!")

                # Define i-th GP training loss with output candidate_dim
                # Note the permutation of the dimensions of y
                def negative_gp_log_likelihood(xs, ys, signal_amplitude, length_scales, noise_amplitude, train=True):

                    gp = GaussianProcess(kernel=self.kernel_name,
                                         input_dim=self.input_dim + i,
                                         signal_amplitude=signal_amplitude,
                                         length_scales=length_scales,
                                         noise_amplitude=noise_amplitude)

                    # Note the permutation of the dimensions of y
                    ys_to_append = tf.gather(ys, indices=tf.convert_to_tensor(permutation, dtype=tf.int32), axis=1)

                    gp_input = tf.concat((xs, ys_to_append), axis=1)

                    # If we're not training, we condition on the training data
                    if not train:
                        train_ys_to_append = tf.gather(train_ys,
                                                       indices=tf.convert_to_tensor(permutation, dtype=tf.int32),
                                                       axis=1)
                        train_gp_input = tf.concat((train_xs, train_ys_to_append), axis=1)

                        gp = gp | (train_gp_input, train_ys[:, candidate_dim:candidate_dim + 1])

                    return -gp.log_pdf(gp_input,
                                       ys[:, candidate_dim:candidate_dim + 1],
                                       normalize_with_input=train,
                                       normalize_with_training_data=not train)

                # Robust optimization
                j = 0

                while j < optimizer_restarts:
                    j += 1

                    hyperparams = self.create_hyperparameter_initializers(
                        index=i, length_scale_init=self.initialization_heuristic)

                    length_scales, signal_amplitude, noise_amplitude = hyperparams

                    valid_log_prob = np.inf

                    try:
                        if optimizer == "l-bfgs-b":
                            # Perform L-BFGS-B optimization
                            res = minimize(function=lambda s, l, n: negative_gp_log_likelihood(
                                train_xs, train_ys, s, l, n, train=True),
                                           vs=(signal_amplitude, length_scales, noise_amplitude),
                                           parallel_iterations=10,
                                           max_iterations=iters)

                            loss, _, diverged = res

                            if diverged:
                                logger.error(f"Optimization diverged, restarting iteration {j}! (loss was {loss:.3f})")
                                j -= 1
                                continue

                            valid_log_prob = negative_gp_log_likelihood(validation_xs,
                                                                        validation_ys,
                                                                        signal_amplitude(),
                                                                        length_scales(),
                                                                        noise_amplitude(),
                                                                        train=False)

                        else:

                            # Get the list of reparametrizations for the hyperparameters
                            reparams = BoundedVariable.get_reparametrizations(hyperparams)

                            optimizer = tf.optimizers.Adam(rate, epsilon=1e-8)

                            prev_loss = np.inf

                            with trange(iters) as t:
                                for iteration in t:
                                    with tf.GradientTape(watch_accessed_variables=False) as tape:
                                        tape.watch(reparams)

                                        loss = negative_gp_log_likelihood(train_xs,
                                                                          train_ys,
                                                                          signal_amplitude(),
                                                                          length_scales(),
                                                                          noise_amplitude(),
                                                                          train=True)

                                    if tf.abs(prev_loss - loss) < tolerance:
                                        logger.info(f"Loss decreased less than {tolerance}, "
                                                    f"optimisation terminated at iteration {iteration}.")
                                        break

                                    prev_loss = loss

                                    gradients = tape.gradient(loss, reparams)
                                    optimizer.apply_gradients(zip(gradients, reparams))

                                    t.set_description(f"Loss at iteration {iteration}: {loss:.3f}.")

                            valid_log_prob = negative_gp_log_likelihood(validation_xs,
                                                                        validation_ys,
                                                                        signal_amplitude(),
                                                                        length_scales(),
                                                                        noise_amplitude(),
                                                                        train=False)

                    except tf.errors.InvalidArgumentError as e:
                        logger.error(str(e))
                        j = j - 1

                        if error_level == "raise":
                            raise e
                        elif error_level == "catch":
                            continue

                    except Exception as e:

                        logger.error("Iteration {} failed: {}".format(i, str(e)))
                        j = j - 1

                        if error_level == "raise":
                            raise e
                        elif error_level == "catch":
                            continue

                    if valid_log_prob < best_validation_log_prob:

                        logger.info(f"Output {i}, candidate dimension {candidate_dim}, "
                                    f"Iteration {j}: New best negative log likelihood: {valid_log_prob:.3f}")

                        best_validation_log_prob = valid_log_prob

                        # Assign the hyperparameters for each input to the model variables
                        self.gp_length_scales[i].assign(length_scales())
                        self.signal_amplitudes[i].assign(signal_amplitude())
                        self.noise_amplitudes[i].assign(noise_amplitude())

                        # Update permutation
                        best_candidate = candidate_dim

                    else:
                        logger.info(f"Output {i}, Iteration {j}: Loss: {loss:.3f}")

                    if np.isnan(loss) or np.isinf(loss):
                        logger.error(f"Output {i}, Iteration {j}: Loss was {loss}, restarting training iteration!")
                        j = j - 1
                        continue

            logger.info(f"Selected dimension {best_candidate} for output {i}!")
            permutation.append(best_candidate)
            logger.info(f"Permutation so far: {permutation}!")

        # Add on the target dimensions to the permutation to pass to fit()
        permutation = permutation + list(range(self.output_dim - num_target_dimensions, self.output_dim))

        logger.info(f"Permutation discovered: {permutation}. Fitting everything now!")

        # Fit on the joint train and validation set
        self.fit(xs=tf.concat([train_xs, validation_xs], axis=0),
                 ys=tf.concat([train_ys, validation_ys], axis=0),
                 optimizer=optimizer,
                 optimizer_restarts=optimizer_restarts,
                 permutation=permutation,
                 trace=trace,
                 iters=iters,
                 rate=rate)

    @staticmethod
    def restore(save_path):

        with open(save_path + ".json", "r") as config_file:
            config = json.load(config_file)

        model = GPARModel.from_config(config, )

        model.load_weights(save_path)
        model.create_gps()

        return model

    def get_config(self):

        return {
            "name": self.name,
            "kernel": self.kernel_name,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "denoising": self.denoising,
            "verbose": self.verbose,
        }

    @staticmethod
    def from_config(config, **kwargs):
        return GPARModel(**config)
