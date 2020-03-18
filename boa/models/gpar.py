import logging
import json
from tqdm import trange

import numpy as np
import tensorflow as tf

from stheno.tensorflow import dense

from .abstract_model import AbstractModel, ModelError
from boa.core import GaussianProcess, setup_logger, inv_perm

from not_tf_opt import minimize, BoundedVariable

from boa import ROOT_DIR

logger = setup_logger(__name__, level=logging.DEBUG, to_console=True, log_file=f"{ROOT_DIR}/../logs/gpar.log")


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

        self.permutation = tf.Variable(tf.range(output_dim, dtype=tf.int32))

        self.length_scales = []
        self.signal_amplitudes = []
        self.noise_amplitudes = []

        # Create TF variables for each of the hyperparameters, so that
        # we can use Keras' serialization features
        for i in range(self.output_dim):
            # Note the scaling in dimension
            if _create_length_scales:
                self.length_scales.append(
                    tf.Variable(tf.ones(self.input_dim + i, dtype=tf.float64),
                                name=f"{i}/length_scales",
                                trainable=False))

            self.signal_amplitudes.append(
                tf.Variable((1,), dtype=tf.float64, name=f"{i}/signal_amplitude", trainable=False))

            self.noise_amplitudes.append(
                tf.Variable((1,), dtype=tf.float64, name=f"{i}/noise_amplitude", trainable=False))

    def initialize_hyperparameters(self, index,
                                   length_scale_init="random",
                                   init_minval=0.5, init_maxval=2.0):

        if length_scale_init == "median":

            # Center on the medians, treat the inputs and the outputs separately
            xs_ls_init = self.xs_euclidean_percentiles[4]
            ys_ls_init = self.ys_euclidean_percentiles[4]

            xs_ls_rand_range = tf.minimum(self.xs_euclidean_percentiles[3] - self.xs_euclidean_percentiles[1],
                                          self.xs_euclidean_percentiles[5] - self.xs_euclidean_percentiles[3])

            ys_ls_rand_range = tf.minimum(self.ys_euclidean_percentiles[3] - self.ys_euclidean_percentiles[1],
                                          self.ys_euclidean_percentiles[5] - self.ys_euclidean_percentiles[3])

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

            ls_lower_bound = tf.concat(
                [tf.ones(shape=xs_ls_init.shape, dtype=self.dtype) * self.xs_euclidean_percentiles[0] / 4.,
                 tf.ones(shape=ys_ls_init.shape, dtype=self.dtype) * self.ys_euclidean_percentiles[0] / 4.],
                axis=0)

            ls_upper_bound = tf.concat(
                [tf.ones(shape=xs_ls_init.shape, dtype=self.dtype) * self.xs_euclidean_percentiles[-1] * 32.,
                 tf.ones(shape=ys_ls_init.shape, dtype=self.dtype) * self.ys_euclidean_percentiles[-1] * 32.],
                axis=0)

        elif length_scale_init == "dim_median":
            # Center on the medians, treat the inputs and the outputs separately
            xs_ls_init = self.xs_per_dim_percentiles[4, :]
            ys_ls_init = self.ys_per_dim_percentiles[4, :index]

            xs_ls_rand_range = tf.minimum(self.xs_per_dim_percentiles[3, :] - self.xs_per_dim_percentiles[1, :],
                                          self.xs_per_dim_percentiles[5, :] - self.xs_per_dim_percentiles[3, :])

            ys_ls_rand_range = tf.minimum(self.ys_per_dim_percentiles[3, :] - self.ys_per_dim_percentiles[1, :],
                                          self.ys_per_dim_percentiles[5, :] - self.ys_per_dim_percentiles[3, :])

            xs_ls_init += tf.random.uniform(shape=(self.input_dim,),
                                            minval=-xs_ls_rand_range,
                                            maxval=xs_ls_rand_range,
                                            dtype=tf.float64)

            ys_ls_init += tf.random.uniform(shape=(index,),
                                            minval=-ys_ls_rand_range[:index],
                                            maxval=ys_ls_rand_range[:index],
                                            dtype=tf.float64)

            # Once the inputs and outputs have been initialized separately, concatenate them
            ls_init = tf.concat((xs_ls_init, ys_ls_init), axis=0)

            ls_lower_bound = tf.concat(
                [tf.ones(shape=xs_ls_init.shape, dtype=self.dtype) * self.xs_per_dim_percentiles[0, :] / 4.,
                 tf.ones(shape=ys_ls_init.shape, dtype=self.dtype) * self.ys_per_dim_percentiles[0, :index] / 4.],
                axis=0)

            ls_upper_bound = tf.concat(
                [tf.ones(shape=xs_ls_init.shape, dtype=self.dtype) * self.xs_per_dim_percentiles[-1, :] * 32.,
                 tf.ones(shape=ys_ls_init.shape, dtype=self.dtype) * self.ys_per_dim_percentiles[-1, :index] * 32.],
                axis=0)

        else:
            ls_init = tf.random.uniform(shape=(self.input_dim + index,),
                                        minval=init_minval,
                                        maxval=init_maxval,
                                        dtype=tf.float64)

            ls_lower_bound = 1e-2
            ls_upper_bound = 1e2

        # Create bounded variables
        length_scales = BoundedVariable(ls_init,
                                        lower=ls_lower_bound,
                                        upper=ls_upper_bound,
                                        dtype=tf.float64)

        signal_amplitude = BoundedVariable(tf.random.uniform(shape=(1,),
                                                             minval=init_minval,
                                                             maxval=init_maxval,
                                                             dtype=tf.float64),
                                           lower=1e-1,
                                           upper=1e2,
                                           dtype=tf.float64)

        noise_amplitude = BoundedVariable(tf.random.uniform(shape=(1,),
                                                            minval=0.1 * init_minval,
                                                            maxval=0.1 * init_maxval,
                                                            dtype=tf.float64),
                                          lower=1e-3,
                                          upper=1e1)

        return length_scales, signal_amplitude, noise_amplitude

    def fit(self,
            xs,
            ys,
            optimizer="l-bfgs-b",
            optimizer_restarts=1,
            permutation=None,
            trace=False,
            tolerance=1e-5,
            iters=1000,
            seed=None,
            rate=1e-2,
            debugging_trace=False,
            err_level="catch", ) -> None:

        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)

        xs, ys = self._validate_and_convert_input_output(xs, ys)

        if self.denoising:
            # This tensor will store the predictive means for the previous dimensions
            pred_ys = tf.zeros(shape=(ys.shape[0], 0), dtype=tf.float64)

        # If no permutation is given, use the regular order of the ys
        if permutation is None:
            permutation = tf.range(self.output_dim, dtype=tf.int32)

        if len(permutation) != self.output_dim:
            raise ModelError("Length of permutation must match the number of outputs!")

        # Check if the permutation has every output in it
        if len([x for x in range(self.output_dim) if x not in permutation]) != 0:
            raise ModelError("Permutation must contain every output dimension!")

        # We're learning the hyperparameters for this permutation
        self.permutation.assign(permutation)

        ys = tf.gather(ys, indices=tf.convert_to_tensor(permutation, dtype=tf.int32), axis=1)

        logger.info(f"Training data supplied with xs shape {xs.shape} and ys shape {ys.shape}, training!")

        self._calculate_statistics_for_median_initialization_heuristic(xs, ys)

        cumulative_loss = 0

        # Optimize dimensions individually
        for i in range(self.output_dim):

            best_loss = np.inf

            # Define i-th GP training loss
            def negative_gp_log_likelihood(length_scales, signal_amplitude, noise_amplitude):

                gp = GaussianProcess(kernel=self.kernel_name,
                                     input_dim=self.input_dim + i,
                                     signal_amplitude=signal_amplitude,
                                     length_scales=length_scales,
                                     noise_amplitude=noise_amplitude)

                # ys_to_append = pred_ys if self.denoising else ys[:, :i]

                # Permute the output
                ys_to_append = ys[:, :i]
                gp_input = tf.concat((xs, ys_to_append), axis=1)

                return -gp.log_pdf(gp_input, ys[:, i:i + 1], normalize_with_input=True)

            # Robust optimization
            j = 0

            while j < optimizer_restarts:
                j += 1

                hyperparams = self.initialize_hyperparameters(index=i, length_scale_init=self.initialization_heuristic)

                length_scales, signal_amplitude, noise_amplitude = hyperparams

                # =================================================================
                # Debugging stuff
                # =================================================================
                if debugging_trace:
                    gp = GaussianProcess(kernel=self.kernel_name,
                                         input_dim=self.input_dim + i,
                                         signal_amplitude=signal_amplitude(),
                                         length_scales=length_scales(),
                                         noise_amplitude=noise_amplitude())

                    # ys_to_append = pred_ys if self.denoising else ys[:, :i]

                    # Permute the output
                    ys_to_append = ys[:, :i]
                    gp_input = tf.concat((xs, ys_to_append), axis=1)

                    fwd, _ = gp._create_transforms(gp_input)

                    K = dense((gp.signal + gp.noise + gp.jitter).kernel(fwd(gp_input)))
                    # print(f"x percentiles: {self.xs_euclidean_percentiles}")
                    # print(f"y percentiles: {self.ys_euclidean_percentiles}")
                    #
                    # print(f"x dim percentiles: {self.xs_per_dim_percentiles}")
                    # print(f"y dim percentiles: {self.ys_per_dim_percentiles}")

                    # print(f"Kernel matrix: {K}")

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

                try:
                    if optimizer == "l-bfgs-b":
                        # Perform L-BFGS-B optimization
                        loss, converged, diverged = minimize(function=negative_gp_log_likelihood,
                                                             vs=hyperparams,
                                                             parallel_iterations=10,
                                                             max_iterations=iters,
                                                             trace=False)

                        # =================================================================
                        # Debugging stuff
                        # =================================================================
                        if debugging_trace:
                            gp = GaussianProcess(kernel=self.kernel_name,
                                                 input_dim=self.input_dim + i,
                                                 signal_amplitude=signal_amplitude(),
                                                 length_scales=length_scales(),
                                                 noise_amplitude=noise_amplitude())

                            # ys_to_append = pred_ys if self.denoising else ys[:, :i]

                            # Permute the output
                            ys_to_append = ys[:, :i]
                            gp_input = tf.concat((xs, ys_to_append), axis=1)

                            fwd, _ = gp._create_transforms(gp_input)

                            K = dense((gp.signal + gp.noise + gp.jitter).kernel(fwd(gp_input)))

                            # print(f"Kernel matrix: {K}")

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
                            logger.error(f"Model diverged, restarting iteration {j}! (loss was {loss:.3f})")
                            j -= 1
                            continue

                    else:

                        # Get the list of reparametrizations for the hyperparameters
                        reparams = BoundedVariable.get_reparametrizations(hyperparams)

                        optimizer = tf.optimizers.Adam(rate, epsilon=1e-8)

                        prev_loss = np.inf

                        with trange(iters) as t:
                            for iteration in t:
                                with tf.GradientTape(watch_accessed_variables=False) as tape:
                                    tape.watch(reparams)

                                    loss = negative_gp_log_likelihood(signal_amplitude=signal_amplitude(),
                                                                      length_scales=length_scales(),
                                                                      noise_amplitude=noise_amplitude())

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
                    j = j - 1

                    if err_level == "raise":
                        raise e

                    elif err_level == "catch":
                        continue

                except Exception as e:
                    logger.error("Iteration {} failed: {}".format(i, str(e)))
                    j = j - 1

                    if err_level == "raise":
                        raise e

                    elif err_level == "catch":
                        continue

                if loss < best_loss:

                    logger.info(f"Output {i}, Iteration {j}: New best loss: {loss:.3f}")

                    best_loss = loss

                    # Assign the hyperparameters for each input to the model variables
                    self.length_scales[i].assign(length_scales())
                    self.signal_amplitudes[i].assign(signal_amplitude())
                    self.noise_amplitudes[i].assign(noise_amplitude())

                    # If we are using "denoising GPAR", then we now need to get the predictive means
                    # for the current output
                    if self.denoising:
                        gp = GaussianProcess(kernel=self.kernel_name,
                                             input_dim=self.input_dim + i,
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

                    hyperparams = self.initialize_hyperparameters(index=i,
                                                                  length_scale_init=self.initialization_heuristic)

                    length_scales, signal_amplitude, noise_amplitude = hyperparams

                    valid_log_prob = np.inf

                    try:
                        if optimizer == "l-bfgs-b":
                            # Perform L-BFGS-B optimization
                            res = minimize(function=lambda s, l, n: negative_gp_log_likelihood(train_xs,
                                                                                               train_ys,
                                                                                               s, l, n,
                                                                                               train=True),
                                           vs=(signal_amplitude,
                                               length_scales,
                                               noise_amplitude),
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
                        self.length_scales[i].assign(length_scales())
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

    def predict(self, xs, numpy=False):

        if not self.trained:
            raise ModelError("Using untrained model for prediction!")

        if len(self.models) < self.output_dim:
            logger.info("GPs haven't been cached yet, creating them now.")
            self.create_gps()

        xs = self._validate_and_convert(xs, output=False)

        train_ys = self.permuted_ys

        means = []
        variances = []

        for i, model in enumerate(self.models):
            gp_input = tf.concat([xs] + means, axis=1)
            gp_train_input = tf.concat([self.xs, train_ys[:, :i]], axis=1)

            model = model | (gp_train_input, train_ys[:, i:i + 1])

            mean, var = model.predict(gp_input, latent=False)

            means.append(mean)
            variances.append(var)

        means = tf.concat(means, axis=1)
        variances = tf.concat(variances, axis=1)

        # Permute stuff back
        means = self.inverse_permute_output(means)
        variances = self.inverse_permute_output(variances)

        if numpy:
            means = means.numpy()
            variances = variances.numpy()

        return means, variances

    def log_prob(self, xs, ys, use_conditioning_data=True, latent=False, numpy=False, target_dims=None):

        if target_dims is not None and not isinstance(target_dims, (tuple, list)):
            raise ModelError("target_dims must be a list or a tuple!")

        if len(self.models) < self.output_dim:
            logger.info("GPs haven't been cached yet, creating them now.")
            self.create_gps()

        xs, ys = self._validate_and_convert_input_output(xs, ys)

        # Put the ys in the "correct" order
        ys = self.permute_output(ys)

        # Permute the training data according to the set permutation
        train_ys = self.permuted_ys

        log_prob = 0.

        for i, model in enumerate(self.models):

            if i not in target_dims:
                continue

            gp_input = tf.concat([xs, ys[:, :i]], axis=1)
            gp_train_input = tf.concat([self.xs, train_ys[:, :i]], axis=1)

            cond_model = model | (gp_train_input, train_ys[:, i:i + 1])

            if use_conditioning_data:
                model_log_prob = cond_model.log_pdf(gp_input, ys[:, i:i + 1],
                                                    latent=latent,
                                                    with_jitter=False,
                                                    normalize_with_training_data=True)
            else:
                # Normalize model to the regime on which the models were trained
                norm_xs = cond_model.normalize_with_training_data(gp_input, output=False)
                norm_ys = cond_model.normalize_with_training_data(ys[:, i:i + 1], output=True)

                model_log_prob = model.log_pdf(norm_xs,
                                               norm_ys,
                                               latent=latent,
                                               with_jitter=False)

            log_prob = log_prob + model_log_prob

        if numpy:
            log_prob = log_prob.numpy()

        return log_prob

    def create_gps(self):
        self.models.clear()

        for i in range(self.output_dim):
            gp = GaussianProcess(kernel=self.kernel_name,
                                 input_dim=self.input_dim + i,
                                 signal_amplitude=self.signal_amplitudes[i],
                                 length_scales=self.length_scales[i],
                                 noise_amplitude=self.noise_amplitudes[i])

            self.models.append(gp)

    def permute_output(self, ys):
        return tf.gather(ys, indices=tf.convert_to_tensor(self.permutation, dtype=tf.int32), axis=1)

    def inverse_permute_output(self, ys):
        return tf.gather(ys, indices=self.inverse_permutation, axis=1)

    @property
    def permuted_ys(self):
        return self.permute_output(self.ys)

    @property
    def inverse_permutation(self):
        return tf.convert_to_tensor(inv_perm(self.permutation), dtype=tf.int32)

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
