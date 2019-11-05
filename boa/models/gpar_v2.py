from typing import Tuple, List, Optional, Callable

import numpy as np
import tensorflow as tf
from stheno.tensorflow import GP, EQ, Matern52, Delta
from varz.tensorflow import Vars, minimise_l_bfgs_b

from .abstract_model_v2 import AbstractModel, ModelError


class GPARModel(AbstractModel):
    AVAILABLE_KERNELS = {"rbf": EQ,
                         "matern52": Matern52, }

    # Ensures that covariance matrix stays positive semidefinite
    VARIABLE_LOG_BOUNDS = (-6, 7)
    CHECKPOINT_NAME = "gpar_v2.ckpt"

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

            self.gp_variances.append(tf.Variable(1.0,
                                                 dtype=tf.float64,
                                                 name="gp_variance_dim_{}".format(i),
                                                 trainable=False))

            self.noise_variances.append(tf.Variable(1.0,
                                                    dtype=tf.float64,
                                                    name="noise_variance_dim_{}".format(i),
                                                    trainable=False))

        self._update_mean_std()

        return self

    def initialize_training_variables(self, vs: Vars) -> None:

        for i in range(self.output_dim):
            # Note the scaling in dimension with the index
            vs.bnd(init=tf.random.uniform(shape=(self.input_dim + i,),
                                          minval=self.init_minval,
                                          maxval=self.init_maxval,
                                          dtype=tf.float64),
                   lower=1e-4,
                   upper=1e4,
                   name="length_scales_dim_{}".format(i))

            # GP variance
            vs.pos(init=tf.random.uniform(shape=(1,),
                                          minval=self.init_minval,
                                          maxval=self.init_maxval,
                                          dtype=tf.float64),
                   name="gp_variance_dim_{}".format(i))

            # Noise variance: bound between 1e-4 and 1e4
            vs.bnd(init=tf.random.uniform(shape=(1,),
                                          minval=self.init_minval,
                                          maxval=self.init_maxval,
                                          dtype=tf.float64),
                   lower=1e-4,
                   upper=1e4,
                   name="noise_variance_dim_{}".format(i))

    def train(self) -> None:
        self._update_mean_std()
        x_normalized = self.normalize(self.xs, mean=self.xs_mean, std=self.xs_std)
        y_normalized = self.normalize(self.ys, mean=self.ys_mean, std=self.ys_std)

        self.models.clear()

        # Create dummy variables for training
        vs = Vars(tf.float64)

        # Define i-th GP training loss
        def negative_gp_log_likelihood(idx, gp_var, length_scale, noise_var):

            prior_gp_ = self.get_prior_gp_model(self.kernel_name,
                                                length_scale,
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
            self.initialize_training_variables(vs)

            try:
                # Perform L-BFGS-B optimization
                loss = minimise_l_bfgs_b(negative_gpar_log_likelihood,
                                         vs)

            except Exception as e:
                print("Iteration {} failed: {}".format(i + 1, str(e)))

            if loss < best_loss:

                best_loss = loss

    def predict_batch(self, xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert xs.shape[1] == self.input_dim
        xs_test_normalized = self.normalize(xs, mean=self.xs_mean, std=self.xs_std)

        mean_list: List[np.ndarray] = []
        var_list: List[np.ndarray] = []

        with self._get_session() as session:
            session.run(tf.global_variables_initializer())
            self.load_model(session)

            feed_dict = {}
            for i, (x_ph, y_ph, x_test_ph) in enumerate(self.model_post_phs):
                feed_dict[x_ph] = np.concatenate((self.xs_normalized, self.ys_normalized[:, :i]), axis=1)
                feed_dict[y_ph] = self.ys_normalized[:, i:i + 1]
                feed_dict[x_test_ph] = np.concatenate([xs_test_normalized] + mean_list, axis=1)

                mean_list.append(session.run(self.model_post_means[i], feed_dict=feed_dict))
                var_list.append(session.run(self.model_post_vars[i], feed_dict=feed_dict))

        mean = np.concatenate(mean_list, axis=1)
        variance = np.concatenate(var_list, axis=1)

        return (mean * self.ys_std + self.ys_mean), (variance * self.ys_std ** 2)

    def save_model(self, session):
        saver = tf.train.Saver()
        saver.save(sess=session, save_path=self.CHECKPOINT_NAME)

    def load_model(self, session):
        saver = tf.train.Saver()
        saver.restore(sess=session, save_path=self.CHECKPOINT_NAME)

    def _print(self, message: str):
        if self.verbose:
            print(message)
