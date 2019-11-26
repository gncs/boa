from typing import Tuple, List

import numpy as np
import tensorflow as tf
from varz.tensorflow import Vars, minimise_l_bfgs_b

from .abstract_model_v2 import AbstractModel, ModelError
from boa.core.permutation import PermutationVariable


class GPARModel(AbstractModel):

    def __init__(self,
                 kernel: str,
                 num_optimizer_restarts: int,
                 learn_permutation: bool = False,
                 denoising: bool = True,
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
                                        num_optimizer_restarts=num_optimizer_restarts,
                                        verbose=verbose,
                                        name=name,
                                        **kwargs)

        self.denoising = denoising

        self.length_scales: List[tf.Variable] = []
        self.gp_variances: List[tf.Variable] = []
        self.noise_variances: List[tf.Variable] = []

        self.learn_permutation = learn_permutation

        self.output_perm = None

    def _set_data(self, inputs: Tuple) -> tf.keras.Model:
        """
        This override of | is implements the inference of the posterior GP model from the inputs (xs, ys)

        :param inputs: tuple of array-like xs and ys, i.e. input points and their corresponding function values
        :return: None
        """

        # Validate inputs
        super(GPARModel, self)._set_data(inputs)

        self.output_perm = tf.eye(self.output_dim, dtype=tf.float64)

        # Create TF variables for each of the hyperparameters, so that
        # we can use Keras' serialization features
        for i in range(self.output_dim):

            # Note the scaling in dimension
            self.length_scales.append(tf.Variable(tf.ones(self.input_dim + i,
                                                          dtype=tf.float64),
                                                  name="length_scales_dim_{}".format(i),
                                                  trainable=False))

            self.gp_variances.append(tf.Variable((1,),
                                                 dtype=tf.float64,
                                                 name="gp_variance_dim_{}".format(i),
                                                 trainable=False))

            self.noise_variances.append(tf.Variable((1,),
                                                    dtype=tf.float64,
                                                    name="noise_variance_dim_{}".format(i),
                                                    trainable=False))

        self._update_mean_std()

        return self

    def create_hyperparameters(self) -> Vars:
        """
        Creates the hyperparameter container that the model uses
        and creates the constrained hyperparameter variables in it,
        initialized to some dummy values.

        *Note*: It is not safe to use the initialized values for training,
        always call initialize_hyperparameters first!

        :return: Varz variable container
        """

        self._update_mean_std()

        vs = Vars(tf.float64)

        if self.learn_permutation:
            self.output_perm = PermutationVariable(n_items=self.output_dim,
                                                   vs=vs,
                                                   temperature=0.3,
                                                   name="output_perm")

        for i in range(self.output_dim):

            # Note the scaling in dimension with the index
            vs.bnd(init=tf.ones(self.input_dim + i, dtype=tf.float64),
                   lower=1e-4,
                   upper=1e4,
                   name="length_scales_dim_{}".format(i))

            # GP variance
            vs.bnd(init=tf.ones(1, dtype=tf.float64),
                   lower=1e-5,
                   upper=1e5,
                   name="gp_variance_dim_{}".format(i))

            # Noise variance: bound between 1e-4 and 1e4
            vs.bnd(init=tf.ones(1, dtype=tf.float64),
                   lower=1e-5,
                   upper=1e5,
                   name="noise_variance_dim_{}".format(i))

        return vs

    def initialize_hyperparameters(self, vs) -> None:

        for i in range(self.output_dim):
            vs.assign("length_scales_dim_{}".format(i),
                      tf.random.uniform(shape=(self.input_dim + i,),
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

    def fit(self) -> None:
        self._update_mean_std()
        x_normalized = self.normalize(self.xs, mean=self.xs_mean, std=self.xs_std)
        y_normalized = self.normalize(self.ys, mean=self.ys_mean, std=self.ys_std)

        self.signal_gps.clear()
        self.noise_gps.clear()

        # Create dummy variables for training
        vs = self.create_hyperparameters()

        if self.learn_permutation:
            y_normalized = self.output_perm.permute(tf.transpose(y_normalized))
            y_normalized = tf.transpose(y_normalized)

        # Define the GPAR training loss
        def negative_gpar_log_likelihood(vs):

            posterior_predictive_means = []

            # Define i-th GP training loss
            def negative_gp_log_likelihood(idx, gp_var, length_scale, noise_var):
                signal_gp_, noise_gp_ = self.get_prior_gp_model(length_scale,
                                                                gp_var,
                                                                noise_var)

                prior_gp_ = signal_gp_ + noise_gp_

                gp_input = tf.concat((x_normalized, y_normalized[:, :idx]), axis=1)

                return -prior_gp_(gp_input).logpdf(y_normalized[:, idx:idx + 1])

            losses = []

            for i in range(self.output_dim):
                losses.append(
                    negative_gp_log_likelihood(idx=i,
                                               gp_var=vs["gp_variance_dim_{}".format(i)],
                                               length_scale=vs["length_scales_dim_{}".format(i)],
                                               noise_var=vs["noise_variance_dim_{}".format(i)]))

            return tf.add_n(losses)

        best_loss = np.inf

        i = 0

        # Train N GPAR models and select the best one
        while i < self.num_optimizer_restarts:

            i += 1

            if self.verbose:
                print("-------------------------------")
                print(f"Training iteration {i}")
                print("-------------------------------")

            # Re-initialize to a random configuration
            self.initialize_hyperparameters(vs)

            for j in range(self.output_dim):

                print(f'{j} - gv: {vs[f"gp_variance_dim_{j}"].numpy()}, nv: {vs[f"noise_variance_dim_{j}"].numpy()}')

            loss = np.inf

            try:
                # Perform L-BFGS-B optimization
                loss = minimise_l_bfgs_b(negative_gpar_log_likelihood, vs, trace=True, err_level="raise")

            except Exception as e:

                for j in range(self.output_dim):
                    ls_name = "length_scales_dim_{}".format(j)
                    gp_var = "gp_variance_dim_{}".format(j)
                    noise_var = "noise_variance_dim_{}".format(j)
                    print(f"Saving: {vs[gp_var]}, --- {vs[noise_var]}")
                    np.save("logs/" + ls_name, vs[ls_name].numpy())
                    np.save("logs/" + gp_var, vs[gp_var].numpy())
                    np.save("logs/" + noise_var, vs[noise_var].numpy())

                print("Iteration {} failed: {}".format(i, str(e)))
                raise e

            if loss < best_loss:

                if self.verbose:
                    print("New best loss: {:.3f}".format(loss))

                best_loss = loss

                self.signal_gps.clear()
                self.noise_gps.clear()

                # Assign the hyperparameters for each input to the model variables
                for j in range(self.output_dim):

                    self.length_scales[j].assign(vs["length_scales_dim_{}".format(j)])
                    self.gp_variances[j].assign(vs["gp_variance_dim_{}".format(j)])
                    self.noise_variances[j].assign(vs["noise_variance_dim_{}".format(j)])

                    signal_gp, noise_gp = self.get_prior_gp_model(self.length_scales[j],
                                                                  self.gp_variances[j],
                                                                  self.noise_variances[j])

                    self.signal_gps.append(signal_gp)
                    self.noise_gps.append(noise_gp)

            elif self.verbose:
                print("Loss: {:.3f}".format(loss))

            if np.isnan(loss):
                print("Loss was NaN, restarting training iteration!")

                i -= 1

            # print("GP variances:")
            # for j in range(self.output_dim):
            #
            #     print(vs[f"gp_variance_dim_{j}"].numpy())
            #     print(vs[f"noise_variance_dim_{j}"].numpy())
            #
            #     ls = vs[f"length_scales_dim_{j}"].numpy()
            #
            #     print("length scale quantiles")
            #     print(np.min(ls))
            #     print(np.percentile(ls, 25))
            #     print(np.percentile(ls, 50))
            #     print(np.percentile(ls, 75))
            #     print(np.max(ls))
            #     print("===========")

    def predict_batch(self, xs) -> Tuple[tf.Tensor, tf.Tensor]:

        if len(self.signal_gps) == 0:
            raise ModelError("The model has not been trained yet!")

        xs = tf.convert_to_tensor(xs, dtype=tf.float64)

        if xs.shape[1] != self.input_dim:
            raise ModelError("xs with shape {} must have 1st dimension (0 indexed) {}!".format(xs.shape,
                                                                                               self.input_dim))
        xs_normalized = self.normalize(xs, mean=self.xs_mean, std=self.xs_std)

        train_xs = self.normalize(self.xs, mean=self.xs_mean, std=self.xs_std)
        train_ys = self.normalize(self.ys, mean=self.ys_mean, std=self.ys_std)

        means = []
        var = []

        for i, (signal_gp, noise_gp) in enumerate(zip(self.signal_gps, self.noise_gps)):

            gp_input = tf.concat([xs_normalized] + means, axis=1)

            gp_train_input = tf.concat([train_xs, train_ys[:, :i]], axis=1)

            posterior_gp = (signal_gp + noise_gp) | (gp_train_input, train_ys[:, i: i + 1])

            means.append(posterior_gp.mean(gp_input))
            var.append(posterior_gp.kernel.elwise(gp_input))

        means = tf.concat(means, axis=1)
        var = tf.concat(var, axis=1)

        return (means * self.ys_std + self.ys_mean), (var * self.ys_std ** 2)

