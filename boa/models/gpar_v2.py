import logging
import os
import json

from typing import Tuple, List

import numpy as np
import tensorflow as tf
from varz.tensorflow import Vars, minimise_l_bfgs_b, minimise_adam

from .abstract_model_v2 import AbstractModel, ModelError
from boa.core import GaussianProcess, PermutationVariable, setup_logger, inv_perm

logger = setup_logger(__name__, level=logging.DEBUG, to_console=True, log_file="logs/gpar.log")


class GPARModel(AbstractModel):

    def __init__(self,
                 kernel: str,
                 input_dim: int,
                 output_dim: int,
                 initialization_heuristic: str = "median",
                 denoising: bool = False,
                 verbose: bool = False,
                 _create_length_scales: bool = True,
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
        self.initialization_heuristic = initialization_heuristic

        self.permutation = []

        self.length_scales = []
        self.signal_amplitudes = []
        self.noise_amplitudes = []

        # Create TF variables for each of the hyperparameters, so that
        # we can use Keras' serialization features
        for i in range(self.output_dim):
            # Note the scaling in dimension
            if _create_length_scales:
                self.length_scales.append(tf.Variable(tf.ones(self.input_dim + i,
                                                              dtype=tf.float64),
                                                      name=f"{i}/length_scales",
                                                      trainable=False))

            self.signal_amplitudes.append(tf.Variable((1,),
                                                      dtype=tf.float64,
                                                      name=f"{i}/signal_amplitude",
                                                      trainable=False))

            self.noise_amplitudes.append(tf.Variable((1,),
                                                     dtype=tf.float64,
                                                     name=f"{i}/noise_amplitude",
                                                     trainable=False))

    def create_hyperparameters(self) -> Vars:
        """
        Creates the hyperparameter container that the model uses
        and creates the constrained hyperparameter variables in it,
        initialized to some dummy values.

        *Note*: It is not safe to use the initialized values for training,
        always call initialize_hyperparameters first!

        :return: Varz variable container
        """

        logger.debug("Creating hyperparameters!")

        vs = Vars(tf.float64)

        for i in range(self.output_dim):
            ls_name = f"{i}/length_scales"
            gp_var_name = f"{i}/signal_amplitude"
            noise_var_name = f"{i}/noise_amplitude"

            # Note the scaling in dimension with the index
            vs.bnd(init=tf.ones(self.input_dim + i, dtype=tf.float64),
                   lower=1e-3,
                   upper=1e2,
                   name=ls_name)

            # GP variance
            vs.bnd(init=tf.ones(1, dtype=tf.float64),
                   lower=1e-4,
                   upper=1e4,
                   name=gp_var_name)

            # Noise variance: bound between 1e-4 and 1e4
            vs.bnd(init=tf.ones(1, dtype=tf.float64),
                   lower=1e-6,
                   upper=1e2,
                   name=noise_var_name)

        return vs

    def initialize_hyperparameters(self,
                                   vs,
                                   index,
                                   length_scale_init="random",
                                   init_minval=0.5,
                                   init_maxval=2.0) -> None:

        # logger.debug(f"Reinitializing hyperparameters with length scale init mode: {length_scale_init}.")

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

        elif length_scale_init == "dim_median":
            xs_ls_init = self.xs_per_dim_percentiles[:, 2]
            xs_ls_rand_range = tf.minimum(self.xs_per_dim_percentiles[:, 2] - self.xs_per_dim_percentiles[:, 0],
                                          self.xs_per_dim_percentiles[:, 4] - self.xs_per_dim_percentiles[:, 2])

            xs_ls_init += tf.random.uniform(shape=(self.input_dim,),
                                            minval=-xs_ls_rand_range,
                                            maxval=xs_ls_rand_range,
                                            dtype=tf.float64)

            ys_ls_init = self.ys_per_dim_percentiles[:index, 2]
            ys_ls_rand_range = tf.minimum(
                self.ys_per_dim_percentiles[:index, 2] - self.ys_per_dim_percentiles[:index, 0],
                self.ys_per_dim_percentiles[:index, 4] - self.ys_per_dim_percentiles[:index, 2])

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
        vs.assign(ls_name, ls_init)

        vs.assign(gp_var_name,
                  tf.random.uniform(shape=(1,),
                                    minval=init_minval,
                                    maxval=init_maxval,
                                    dtype=tf.float64))

        vs.assign(noise_var_name,
                  tf.random.uniform(shape=(1,),
                                    minval=init_minval,
                                    maxval=init_maxval,
                                    dtype=tf.float64))

    def fit(self,
            xs,
            ys,
            optimizer="l-bfgs-b",
            optimizer_restarts=1,
            permutation=None,
            trace=False,
            iters=1000,
            seed=None,
            rate=1e-2) -> None:

        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)

        xs, ys = self._validate_and_convert_input_output(xs, ys)

        if self.denoising:
            # This tensor will store the predictive means for the previous dimensions
            pred_ys = tf.zeros(shape=(ys.shape[0], 0), dtype=tf.float64)

        # If no permutation is given, use the regular order of the ys
        if permutation is None:
            permutation = list(range(self.output_dim))

        if len(permutation) != self.output_dim:
            raise ModelError("Length of permutation must match the number of outputs!")

        # Check if the permutation has every output in it
        if len([x for x in range(self.output_dim) if x not in permutation]) != 0:
            raise ModelError("Permutation must contain every output dimension!")

        # We're learning the hyperparameters for this permutation
        self.permutation = permutation

        ys = tf.gather(ys, indices=tf.convert_to_tensor(permutation, dtype=tf.int32), axis=1)

        logger.info(f"Training data supplied with xs shape {xs.shape} and ys shape {ys.shape}, training!")

        self._calculate_statistics_for_median_initialization_heuristic(xs, ys)

        vs = self.create_hyperparameters()

        cumulative_loss = 0

        # Optimize dimensions individually
        for i in range(self.output_dim):
            length_scales_name = f"{i}/length_scales"
            sig_amp_name = f"{i}/signal_amplitude"
            noise_amp = f"{i}/noise_amplitude"

            best_loss = np.inf

            # Define i-th GP training loss
            def negative_gp_log_likelihood(signal_amplitude, length_scales, noise_amplitude):

                gp = GaussianProcess(kernel=self.kernel_name,
                                     signal_amplitude=signal_amplitude,
                                     length_scales=length_scales,
                                     noise_amplitude=noise_amplitude)

                #ys_to_append = pred_ys if self.denoising else ys[:, :i]

                # Permute the output
                ys_to_append = ys[:, :i]
                gp_input = tf.concat((xs, ys_to_append), axis=1)

                return -gp.log_pdf(gp_input, ys[:, i:i + 1], normalize=True)

            # Robust optimization
            j = 0

            while j < optimizer_restarts:
                j += 1

                self.initialize_hyperparameters(vs, index=i, length_scale_init=self.initialization_heuristic)

                loss = np.inf

                try:
                    if optimizer == "l-bfgs-b":
                        # Perform L-BFGS-B optimization
                        loss = minimise_l_bfgs_b(lambda v: negative_gp_log_likelihood(signal_amplitude=v[sig_amp_name],
                                                                                      length_scales=v[
                                                                                          length_scales_name],
                                                                                      noise_amplitude=v[noise_amp]),
                                                 vs,
                                                 names=[sig_amp_name,
                                                        length_scales_name,
                                                        noise_amp],
                                                 trace=trace,
                                                 iters=iters,
                                                 err_level="raise")
                    else:
                        # Perform Adam optimization
                        loss = minimise_adam(lambda v: negative_gp_log_likelihood(signal_amplitude=v[sig_amp_name],
                                                                                  length_scales=v[length_scales_name],
                                                                                  noise_amplitude=v[noise_amp]),
                                             vs,
                                             names=[sig_amp_name,
                                                    length_scales_name,
                                                    noise_amp],
                                             iters=iters,
                                             rate=rate,
                                             trace=trace)

                except tf.errors.InvalidArgumentError as e:
                    logger.error(str(e))
                    loss = np.nan

                except Exception as e:

                    logger.error(f"Saving: {vs[sig_amp_name]}, --- {vs[noise_amp]}")

                    if not os.path.exists(os.path.dirname("logs/" + length_scales_name)):
                        os.makedirs(os.path.dirname("logs/" + length_scales_name))

                    np.save("logs/" + length_scales_name, vs[length_scales_name].numpy())
                    np.save("logs/" + sig_amp_name, vs[sig_amp_name].numpy())
                    np.save("logs/" + noise_amp, vs[noise_amp].numpy())

                    logger.error("Iteration {} failed: {}".format(i, str(e)))

                    j = j - 1
                    continue

                if loss < best_loss:

                    logger.info(f"Output {i}, Iteration {j}: New best loss: {loss:.3f}")

                    best_loss = loss

                    # Assign the hyperparameters for each input to the model variables
                    self.length_scales[i].assign(vs[length_scales_name])
                    self.signal_amplitudes[i].assign(vs[sig_amp_name])
                    self.noise_amplitudes[i].assign(vs[noise_amp])

                    # If we are using "denoising GPAR", then we now need to get the predictive means
                    # for the current output
                    if self.denoising:
                        gp = GaussianProcess(kernel=self.kernel_name,
                                             signal_amplitude=self.signal_amplitudes[i],
                                             length_scales=self.length_scales[i],
                                             noise_amplitude=self.noise_amplitudes[i])

                        gp_input = tf.concat((xs, ys[:, :i]), axis=1)

                        gp = gp | (gp_input, ys[:, i:i + 1])

                        predictive_mean, _ = gp.predict(gp_input)

                else:
                    logger.info(f"Output {i}, Iteration {j}: Loss: {loss:.3f}")

                if np.isnan(loss) or np.isinf(loss):
                    logger.error(f"Output {i}, Iteration {j}: Loss was {loss}, restarting training iteration!")
                    j = j - 1
                    continue

                if self.denoising:
                    pred_ys = tf.concat([pred_ys, predictive_mean], axis=1)

            cumulative_loss += best_loss

        self.trained.assign(True)

        return cumulative_loss

    def fit_greedy_ordering(self,
                            xs,
                            ys,
                            optimizer="l-bfgs-b",
                            optimizer_restarts=1,
                            trace=False,
                            num_target_dimensions=0,
                            iters=1000,
                            seed=None,
                            rate=1e-2):
        """
        Perform the greedy search for the optimal output ordering described in the GPAR paper.
        """

        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)

        # Pre fitting data perparation
        xs, ys = self._validate_and_convert_input_output(xs, ys)

        logger.info(f"Training data supplied with xs shape {xs.shape} and ys shape {ys.shape}, training!")

        self._calculate_statistics_for_median_initialization_heuristic(xs, ys)

        vs = self.create_hyperparameters()

        # Permutation to use for prediction
        permutation = []

        # Fit "auxiliary" dimensions one-by-one
        # We omit the "target" dimensions at the end of the column-space of ys
        for i in range(self.output_dim - num_target_dimensions):
            length_scales_name = f"{i}/length_scales"
            sig_amp_name = f"{i}/signal_amplitude"
            noise_amp = f"{i}/noise_amplitude"

            best_loss = np.inf
            best_candidate = 0

            logger.info(f"Selecting output {i}!")

            # Search through the remaining output dimensions
            remaining_dimensions = [dim for dim in range(self.output_dim - num_target_dimensions)
                                    if dim not in permutation]

            # Perform robust optimization for each candidate loss
            for candidate_dim in remaining_dimensions:

                logger.info(f"Training candidate dimension {candidate_dim} for output {i}!")

                # Define i-th GP training loss with output candidate_dim
                # Note the permutation of the dimensions of y
                def negative_gp_log_likelihood(signal_amplitude, length_scales, noise_amplitude):

                    gp = GaussianProcess(kernel=self.kernel_name,
                                         signal_amplitude=signal_amplitude,
                                         length_scales=length_scales,
                                         noise_amplitude=noise_amplitude)

                    # Note the permutation of the dimensions of y
                    ys_to_append = tf.gather(ys, indices=tf.convert_to_tensor(permutation, dtype=tf.int32), axis=1)

                    gp_input = tf.concat((xs, ys_to_append), axis=1)

                    return -gp.log_pdf(gp_input, ys[:, candidate_dim:candidate_dim + 1], normalize=True)

                # Robust optimization
                j = 0

                while j < optimizer_restarts:
                    j += 1

                    self.initialize_hyperparameters(vs, index=i, length_scale_init=self.initialization_heuristic)

                    loss = np.inf

                    try:
                        if optimizer == "l-bfgs-b":
                            # Perform L-BFGS-B optimization
                            loss = minimise_l_bfgs_b(lambda v: negative_gp_log_likelihood(signal_amplitude=v[sig_amp_name],
                                                                                          length_scales=v[
                                                                                              length_scales_name],
                                                                                          noise_amplitude=v[noise_amp]),
                                                     vs,
                                                     names=[sig_amp_name,
                                                            length_scales_name,
                                                            noise_amp],
                                                     trace=trace,
                                                     iters=iters,
                                                     err_level="raise")
                        else:
                            # Perform Adam optimization
                            loss = minimise_adam(lambda v: negative_gp_log_likelihood(signal_amplitude=v[sig_amp_name],
                                                                                      length_scales=v[length_scales_name],
                                                                                      noise_amplitude=v[noise_amp]),
                                                 vs,
                                                 names=[sig_amp_name,
                                                        length_scales_name,
                                                        noise_amp],
                                                 iters=iters,
                                                 rate=rate,
                                                 trace=trace)

                    except tf.errors.InvalidArgumentError as e:
                        logger.error(str(e))
                        loss = np.nan
                        raise e

                    except Exception as e:

                        logger.error(f"Saving: {vs[sig_amp_name]}, --- {vs[noise_amp]}")

                        if not os.path.exists(os.path.dirname("logs/" + length_scales_name)):
                            os.makedirs(os.path.dirname("logs/" + length_scales_name))

                        np.save("logs/" + length_scales_name, vs[length_scales_name].numpy())
                        np.save("logs/" + sig_amp_name, vs[sig_amp_name].numpy())
                        np.save("logs/" + noise_amp, vs[noise_amp].numpy())

                        logger.error("Iteration {} failed: {}".format(i, str(e)))

                        j = j - 1
                        raise e

                    if loss < best_loss:

                        logger.info(f"Output {i}, candidate dimension {candidate_dim}, "
                                    f"Iteration {j}: New best loss: {loss:.3f}")

                        best_loss = loss

                        # Assign the hyperparameters for each input to the model variables
                        self.length_scales[i].assign(vs[length_scales_name])
                        self.signal_amplitudes[i].assign(vs[sig_amp_name])
                        self.noise_amplitudes[i].assign(vs[noise_amp])

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

        self.fit(xs=xs,
                 ys=ys,
                 optimizer=optimizer,
                 optimizer_restarts=optimizer_restarts,
                 permutation=permutation,
                 trace=trace,
                 iters=iters,
                 rate=rate)

    def predict(self, xs, numpy=False):

        if not self.trained:
            raise ModelError("Using untrained model for prediction!")

        if len(self.models) < self.output_dim:
            logger.info("GPs haven't been cached yet, creating them now.")
            self.create_gps()

        xs = self._validate_and_convert(xs, output=False)

        train_ys = tf.gather(self.ys, indices=tf.convert_to_tensor(self.permutation, dtype=tf.int32), axis=1)

        inverse_permutation = tf.convert_to_tensor(inv_perm(self.permutation), dtype=tf.int32)

        means = []
        variances = []

        for i, model in enumerate(self.models):
            gp_input = tf.concat([xs] + means, axis=1)
            gp_train_input = tf.concat([self.xs, train_ys[:, :i]], axis=1)

            model = model | (gp_train_input, train_ys[:, i: i + 1])

            mean, var = model.predict(gp_input, latent=False)

            means.append(mean)
            variances.append(var)

        means = tf.concat(means, axis=1)
        variances = tf.concat(variances, axis=1)

        # Permute stuff back
        means = tf.gather(means, inverse_permutation, axis=1)
        variances = tf.gather(variances, inverse_permutation, axis=1)

        if numpy:
            means = means.numpy()
            variances = variances.numpy()

        return means, variances

    def create_gps(self):
        self.models.clear()

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
            "denoising": self.denoising,
            "initialization_heuristic": self.initialization_heuristic,
            "verbose": self.verbose,
        }

    @staticmethod
    def from_config(config):
        return GPARModel(**config)
