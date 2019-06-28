from typing import List

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfpd

from boa.models.gpar import GPARModel


class ParameterManager:
    def __init__(self, session, variables: List):
        self.session = session
        self.variables = variables

    def get_values(self) -> List:
        return [list(self.session.run(variable) for variable in variable_tuple) for variable_tuple in self.variables]

    def set_values(self, values) -> None:
        operations = []
        for variable_tuple, value_tuple in zip(self.variables, values):
            for variable, value in zip(variable_tuple, value_tuple):
                operations.append(variable.assign(value))

        self.session.run(operations)

    def init_values(self, random_seed=None):
        np.random.seed(random_seed)

        values = []
        for variable_tuple in self.variables:
            value_tuple = []
            for variable in variable_tuple:
                shape = variable.get_shape()

                # Scalar value
                if not shape:
                    value = np.log(np.random.uniform(low=0.5, high=2, size=1)[0])

                # Lengthscales
                else:
                    value = np.log(np.random.uniform(low=0.5, high=2, size=shape))
                value_tuple.append(value)
            values.append(value_tuple)

        self.set_values(values)


class HyperGPARModel(GPARModel):
    INIT_MU = 1.0
    INIT_STD = 2.5

    def __init__(self, num_reg_ls: int = None, *args, **kwargs):
        """Constructor of HyperGPAR model."""

        super().__init__(*args, **kwargs)

        self.reg_parameter_manager = None
        self.init_reg_params = None
        self.num_reg_ls = num_reg_ls

    def _setup_loss(self) -> None:
        # Log PDFs
        model_logpdfs = []

        for i, model in enumerate(self.models):
            x_ph = tf.placeholder(tf.float64, [None, self.input_dim + i], name='x_train')
            y_ph = tf.placeholder(tf.float64, [None, 1], name='y_train')

            model_logpdfs.append(model(x_ph).logpdf(y_ph))
            self.model_logpdf_phs.append((x_ph, y_ph))

        # Set up regularization parameters for variance and noise
        reg_params = [
            tf.Variable(
                initial_value=self.INIT_MU,
                dtype=tf.float64,
                name='mu_variance',
            ),
            tf.Variable(
                initial_value=self.INIT_STD,
                dtype=tf.float64,
                name='std_variance',
            ),
            tf.Variable(
                initial_value=self.INIT_MU,
                dtype=tf.float64,
                name='mu_noise',
            ),
            tf.Variable(
                initial_value=self.INIT_STD,
                dtype=tf.float64,
                name='std_noise',
            ),
        ]

        # Set up regularization parameters for lengthscales
        num_reg_ls = self.num_reg_ls if not (self.num_reg_ls is None) else (self.input_dim + self.output_dim - 1)
        for i in range(num_reg_ls):
            reg_params += [
                tf.Variable(
                    initial_value=self.INIT_MU,
                    dtype=tf.float64,
                    name='mu_lengthscale_' + str(i),
                ),
                tf.Variable(
                    initial_value=self.INIT_STD,
                    dtype=tf.float64,
                    name='std_lengthscale_' + str(i),
                ),
            ]

        self.reg_parameter_manager = ParameterManager(self.session, [reg_params])

        # Iterate over models
        regularizers = []
        for log_variance, log_noise, log_lengthscales in self.log_hps:
            regularizers += [
                tf.log(tfpd.Normal(loc=reg_params[0], scale=reg_params[1]).prob(tf.exp(log_variance))),
                tf.log(tfpd.Normal(loc=reg_params[2], scale=reg_params[3]).prob(tf.exp(log_noise))),
            ]

            for i in range(min(num_reg_ls, log_lengthscales.shape[0])):
                regularizers.append(
                    tf.log(
                        tfpd.Normal(loc=reg_params[4 + i], scale=reg_params[4 + i + 1]).prob(
                            tf.exp(log_lengthscales[i]))))

        self.loss = -tf.add_n(model_logpdfs + regularizers)

    def _post_tf_init(self):
        self.init_reg_params = self.reg_parameter_manager.get_values()

    def train(self):
        self._update_mean_std()

        feed_dict = {}
        for i, (x_placeholder, y_placeholder) in enumerate(self.model_logpdf_phs):
            feed_dict[x_placeholder] = np.concatenate((self.xs_normalized, self.ys_normalized[:, :i]), axis=1)
            feed_dict[y_placeholder] = self.ys_normalized[:, i:i + 1]

        lowest_loss = self.session.run(self.loss, feed_dict=feed_dict)
        best_params = self.log_hp_manager.get_values()
        best_reg_params = self.reg_parameter_manager.get_values()

        for i in range(self.num_optimizer_restarts):
            self.log_hp_manager.init_values(random_seed=i)
            self.reg_parameter_manager.set_values(self.init_reg_params)

            self.optimizer.minimize(self.session, feed_dict=feed_dict)
            loss = self.session.run(self.loss, feed_dict=feed_dict)
            self._print(f'Iteration {i}\tLoss: {loss}')

            if loss < lowest_loss:
                lowest_loss = loss
                best_params = self.log_hp_manager.get_values()
                best_reg_params = self.reg_parameter_manager.get_values()

        self.log_hp_manager.set_values(best_params)
        self.reg_parameter_manager.set_values(best_reg_params)
        loss = self.session.run(self.loss, feed_dict=feed_dict)
        self._print(f'Final loss: {loss}')
