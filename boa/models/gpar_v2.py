from typing import Tuple, List

import numpy as np
import tensorflow as tf
from varz.tensorflow import Vars, minimise_l_bfgs_b

from .abstract_model_v2 import AbstractModel, ModelError


class GPARModel(AbstractModel):

    def __init__(self,
                 kernel: str,
                 num_optimizer_restarts: int,
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

        self.model_post_means: List[tf.Tensor] = []
        self.model_post_vars: List[tf.Tensor] = []
        self.model_post_phs: List[Tuple[tf.Variable, tf.Variable, tf.Variable]] = []

        self.length_scales: List[tf.Variable] = []
        self.gp_variances: List[tf.Variable] = []
        self.noise_variances: List[tf.Variable] = []

    def __or__(self, inputs: Tuple) -> tf.keras.Model:
        """
        This override of | is implements the inference of the posterior GP model from the inputs (xs, ys)

        :param inputs: tuple of array-like xs and ys, i.e. input points and their corresponding function values
        :return: None
        """

        # Validate inputs
        super(GPARModel, self).__or__(inputs)

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
        vs = Vars(tf.float64)

        for i in range(self.output_dim):
            # Note the scaling in dimension with the index
            vs.bnd(init=tf.ones(self.input_dim + i, dtype=tf.float64),
                   lower=1e-4,
                   upper=1e4,
                   name="length_scales_dim_{}".format(i))

            # GP variance
            vs.pos(init=tf.ones(1, dtype=tf.float64),
                   name="gp_variance_dim_{}".format(i))

            # Noise variance: bound between 1e-4 and 1e4
            vs.bnd(init=tf.ones(1, dtype=tf.float64),
                   lower=1e-4,
                   upper=1e4,
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
        def negative_gpar_log_likelihood(vs):

            losses = []

            for i in range(self.output_dim):
                losses.append(
                    negative_gp_log_likelihood(idx=i,
                                               gp_var=vs["gp_variance_dim_{}".format(i)],
                                               length_scale=vs["length_scales_dim_{}".format(i)],
                                               noise_var=vs["noise_variance_dim_{}".format(i)]))

            return tf.add_n(losses)

        best_loss = np.inf

        # Train N GPAR models and select the best one
        for i in range(self.num_optimizer_restarts):

            # Re-initialize to a random configuration
            self.initialize_hyperparameters(vs)

            loss = np.inf

            try:
                # Perform L-BFGS-B optimization
                loss = minimise_l_bfgs_b(negative_gpar_log_likelihood, vs)

            except Exception as e:
                print("Iteration {} failed: {}".format(i + 1, str(e)))

            if loss < best_loss:

                best_loss = loss

                self.models.clear()

                # Assign the hyperparameters for each input to the model variables
                for j in range(self.output_dim):

                    self.length_scales[j].assign(vs["length_scales_dim_{}".format(j)])
                    self.gp_variances[j].assign(vs["gp_variance_dim_{}".format(j)])
                    self.noise_variances[j].assign(vs["noise_variance_dim_{}".format(j)])

                    prior_gp = self.get_prior_gp_model(self.length_scales[j],
                                                       self.gp_variances[j],
                                                       self.noise_variances[j])

                    gp_input = tf.concat((x_normalized, y_normalized[:, :j]), axis=1)

                    # Condition the model
                    best_model = prior_gp | (gp_input, y_normalized[:, j:j + 1])

                    self.models.append(best_model)

    def predict_batch(self, xs) -> Tuple[tf.Tensor, tf.Tensor]:

        if len(self.models) == 0:
            raise ModelError("The model has not been trained yet!")

        xs = tf.convert_to_tensor(xs, dtype=tf.float64)

        if xs.shape[1] != self.input_dim:
            raise ModelError("xs with shape {} must have 1st dimension (0 indexed) {}!".format(xs.shape,
                                                                                               self.input_dim))

        xs_normalized = self.normalize(xs, mean=self.xs_mean, std=self.xs_std)

        means = []
        var = []

        for i, model in enumerate(self.models):

            gp_input = tf.concat([xs_normalized] + means, axis=1)

            means.append(model.mean(gp_input))
            var.append(model.kernel.elwise(gp_input))

        means = tf.concat(means, axis=1)
        var = tf.concat(var, axis=1)

        return (means * self.ys_std + self.ys_mean), (var * self.ys_std ** 2)

