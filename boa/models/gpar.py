from typing import Tuple, List

import numpy as np
import stheno.tf as stf
import tensorflow as tf

from .abstract import AbstractModel


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
                    value = np.exp(np.random.uniform(low=0.1, high=1, size=1)[0])

                # Lengthscales
                else:
                    value = np.exp(np.random.uniform(low=0.01, high=0.5, size=shape))
                value_tuple.append(value)
            values.append(value_tuple)

        self.set_values(values)


class GPARModel(AbstractModel):
    VARIABLE_LOG_BOUNDS = (-8, 9)

    def __init__(self, kernel: str, num_optimizer_restarts: int):
        """
        Constructor of GPAR model.

        :param kernel: name of kernel
        :param num_optimizer_restarts: number of times the optimization of the hyperparameters is restarted
        """

        super().__init__()

        self.kernel_name = kernel
        self.num_optimizer_restarts = num_optimizer_restarts

        self.input_dim = 0
        self.output_dim = 0

        self.xs_mean = np.array([[]])
        self.xs_std = np.array([[]])

        self.ys_mean = np.array([[]])
        self.ys_std = np.array([[]])

        self.xs = np.array([[]])
        self.ys = np.array([[]])

        self.xs_normalized = np.array([[]])
        self.ys_normalized = np.array([[]])

        self.num_pseudo_points = 0
        self.num_true_points = 0

        # TF objects
        self.session = None

        self.models = []
        self.hyperparameters = []
        self.parameter_manager = None

        self.model_logpdfs = []
        self.model_logpdf_phs = []

        self.model_post_means = []
        self.model_post_vars = []
        self.model_post_phs = []

        self.loss = None
        self.optimizer = None

    def set_data(self, xs: np.ndarray, ys: np.ndarray):
        """
        Set data for GPAR.

        :param xs: dimensions N x D_input
        :param ys: dimensions N x D_output
        """
        if not self.models:
            self.input_dim = xs.shape[1]
            self.output_dim = ys.shape[1]
        else:
            assert self.input_dim == xs.shape[1]
            assert self.output_dim == ys.shape[1]

        self.xs = xs
        self.ys = ys
        self.num_true_points = xs.shape[0]

        self._update_mean_std()

        if not self.models:
            self._setup()

    @staticmethod
    def normalize(a: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        return (a - mean) / std

    def _update_mean_std(self) -> None:
        min_std = 1e-10

        self.xs_mean = np.mean(self.xs, axis=0)
        self.xs_std = np.maximum(np.std(self.xs, axis=0), min_std)

        self.ys_mean = np.mean(self.ys, axis=0)
        self.ys_std = np.maximum(np.std(self.ys, axis=0), min_std)

        self.xs_normalized = self.normalize(self.xs, mean=self.xs_mean, std=self.xs_std)
        self.ys_normalized = self.normalize(self.ys, mean=self.ys_mean, std=self.ys_std)

    def _setup_gp(self, num_dims):
        log_variance = tf.Variable(
            initial_value=0.0,
            dtype=tf.float64,
            name='log_variance',
        )

        log_noise = tf.Variable(
            initial_value=0.0,
            dtype=tf.float64,
            name='log_noise',
        )

        log_lengthscale = tf.Variable(
            initial_value=tf.fill(dims=[num_dims], value=tf.dtypes.cast(x=0.0, dtype=tf.float64)),
            name='log_lengthscale',
        )

        self.hyperparameters.append((log_variance, log_noise, log_lengthscale))

        kernel = self._get_kernel()
        return tf.exp(log_variance) * stf.GP(kernel()).stretch(tf.exp(log_lengthscale)) \
               + tf.exp(log_noise) * stf.GP(stf.Delta())

    def _setup(self) -> None:
        if self.session:
            self.session.close()

        self.session = tf.Session()

        # Models
        for i in range(self.output_dim):
            self.models.append(self._setup_gp(self.input_dim + i))

        # Log PDFs
        for i, model in enumerate(self.models):
            x_placeholder = tf.placeholder(tf.float64, [None, self.input_dim + i], name='x_train')
            y_placeholder = tf.placeholder(tf.float64, [None, 1], name='y_train')

            self.model_logpdfs.append(model(x_placeholder).logpdf(y_placeholder))
            self.model_logpdf_phs.append((x_placeholder, y_placeholder))

        # Posteriors
        for i, model in enumerate(self.models):
            x_placeholder = tf.placeholder(tf.float64, [None, self.input_dim + i], name='x_train')
            y_placeholder = tf.placeholder(tf.float64, [None, 1], name='y_train')
            x_test_placeholder = tf.placeholder(tf.float64, [None, self.input_dim + i], name='x_test')

            model_post = model | (model(x_placeholder), y_placeholder)

            self.model_post_means.append(model_post.mean(x_test_placeholder))
            self.model_post_vars.append(stf.dense(model_post.kernel.elwise(x_test_placeholder)))
            self.model_post_phs.append((x_placeholder, y_placeholder, x_test_placeholder))

        self.loss = -tf.add_n(self.model_logpdfs)

        self.parameter_manager = ParameterManager(session=self.session, variables=self.hyperparameters)

        bounds = {}
        for variables in self.hyperparameters:
            for variable in variables:
                bounds[variable] = self.VARIABLE_LOG_BOUNDS

        options = {
            'disp': None,
            'iprint': -1,
            'maxcor': 10,
            'ftol': 1E-10,
            'gtol': 1e-08,
            'eps': 1e-09,
            'maxfun': 15000,
            'maxiter': 15000,
            'maxls': 30
        }

        # Optimizers supporting bounds: L-BFGS-B, TNC, SLSQP,...
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            self.loss,
            var_to_bounds=bounds,
            method='L-BFGS-B',
            options=options,
        )

        self.session.run(tf.global_variables_initializer())

    def _get_kernel(self):
        if self.kernel_name == 'matern' or self.kernel_name == 'matern52':
            return stf.Matern52
        elif self.kernel_name == 'rbf':
            return stf.EQ

        raise Exception("Unknown kernel '" + str(self.kernel_name) + "'")

    def train(self):
        self._update_mean_std()

        feed_dict = {}
        for i, (x_placeholder, y_placeholder) in enumerate(self.model_logpdf_phs):
            feed_dict[x_placeholder] = np.concatenate((self.xs_normalized, self.ys_normalized[:, :i]), axis=1)
            feed_dict[y_placeholder] = self.ys_normalized[:, i:i + 1]

        lowest_loss = self.session.run(self.loss, feed_dict=feed_dict)
        best_params = self.parameter_manager.get_values()

        for i in range(self.num_optimizer_restarts):
            self.parameter_manager.init_values(random_seed=i)

            self.optimizer.minimize(self.session, feed_dict=feed_dict)
            loss = self.session.run(self.loss, feed_dict=feed_dict)
            print(f'Iteration {i}\tLoss: {loss}')

            if loss < lowest_loss:
                lowest_loss = loss
                best_params = self.parameter_manager.get_values()

        self.parameter_manager.set_values(best_params)
        loss = self.session.run(self.loss, feed_dict=feed_dict)
        print(f'Final loss: {loss}')

    def predict_batch(self, xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert xs.shape[1] == self.input_dim
        xs_test_normalized = self.normalize(xs, mean=self.xs_mean, std=self.xs_std)

        mean_list = []
        var_list = []

        feed_dict = {}
        for i, (x_placeholder, y_placeholder, x_test_placeholder) in enumerate(self.model_post_phs):
            feed_dict[x_placeholder] = np.concatenate((self.xs_normalized, self.ys_normalized[:, :i]), axis=1)
            feed_dict[y_placeholder] = self.ys_normalized[:, i:i + 1]
            feed_dict[x_test_placeholder] = np.concatenate([xs_test_normalized] + mean_list, axis=1)

            mean_list.append(self.session.run(self.model_post_means[i], feed_dict=feed_dict))
            var_list.append(self.session.run(self.model_post_vars[i], feed_dict=feed_dict))

        mean = np.concatenate(mean_list, axis=1)
        variance = np.concatenate(var_list, axis=1)

        return (mean * self.ys_std + self.ys_mean), (variance * self.ys_std**2)

    def add_pseudo_point(self, x: np.ndarray) -> None:
        assert x.shape[1] == self.input_dim

        mean, var = self.predict_batch(xs=self.normalize(x, mean=self.xs_mean, std=self.xs_std))

        # Renormalize y and add data point to models
        self._append_data_point(x, mean * self.ys_std + self.ys_mean)
        self.num_pseudo_points += 1

    def add_true_point(self, x: np.ndarray, y: np.ndarray) -> None:
        assert self.num_pseudo_points == 0
        assert x.shape[1] == self.input_dim
        assert y.shape[1] == self.output_dim

        self._append_data_point(x, y)
        self.num_true_points += 1

    def remove_pseudo_points(self) -> None:
        self.xs = self.xs[:-self.num_pseudo_points, :]
        self.ys = self.ys[:-self.num_pseudo_points, :]

        self.xs_normalized = self.normalize(self.xs, mean=self.xs_mean, std=self.xs_std)
        self.ys_normalized = self.normalize(self.ys, mean=self.ys_mean, std=self.ys_std)

        self.num_pseudo_points = 0

    def _append_data_point(self, x: np.ndarray, y: np.ndarray) -> None:
        self.xs = np.vstack((self.xs, x))
        self.ys = np.vstack((self.ys, y))

        self.xs_normalized = self.normalize(self.xs, mean=self.xs_mean, std=self.xs_std)
        self.ys_normalized = self.normalize(self.ys, mean=self.ys_mean, std=self.ys_std)