from typing import Tuple, List, Optional, Callable

import numpy as np
import stheno.tensorflow as stf
import tensorflow as tf

from .abstract import AbstractModel


class GPARModel(AbstractModel):

    # Ensures that covariance matrix stays positive semidefinite
    VARIABLE_LOG_BOUNDS = (-6, 7)
    CHECKPOINT_NAME = "gpar_v2.ckpt"

    def __init__(self, kernel: str, num_optimizer_restarts: int, verbose: bool = False):
        """
        Constructor of GPAR model.

        :param kernel: name of kernel
        :param num_optimizer_restarts: number of times the optimization of the hyperparameters is restarted
        :param verbose: log optimization of hyperparameters
        """

        super().__init__()

        self.kernel_name = kernel
        self.num_optimizer_restarts = num_optimizer_restarts
        self.verbose = verbose

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

        self.models: List = []
        self.log_hps: List[tf.Variable] = []

        self.model_logpdf_phs: List[Tuple[tf.Variable, tf.Variable]] = []

        self.model_post_means: List[tf.Tensor] = []
        self.model_post_vars: List[tf.Tensor] = []
        self.model_post_phs: List[Tuple[tf.Variable, tf.Variable, tf.Variable]] = []

        self.loss: Optional[tf.Tensor] = None
        self.optimizer = None

    def set_data(self, xs: np.ndarray, ys: np.ndarray) -> None:
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
            self._setup_models()
            self.loss = self._setup_loss()
            self.optimizer = self._setup_optimizer()

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

    def _setup_gp(self, num_dims, init_hp_minval=0.5, init_hp_maxval=2.0) -> tf.Tensor:

        gp_number = len(self.log_hps)

        log_variance = tf.Variable(tf.random.uniform(shape=(1, ),
                                                     minval=np.log(init_hp_minval),
                                                     maxval=np.log(init_hp_maxval)),
                                   dtype=tf.float64,
                                   name='log_variance_dim_{}'.format(gp_number))

        log_noise = tf.Variable(tf.random.uniform(shape=(1, ),
                                                  minval=np.log(init_hp_minval),
                                                  maxval=np.log(init_hp_maxval)),
                                dtype=tf.float64,
                                shape=(1, ),
                                name='log_noise_dim_{}'.format(gp_number))

        log_length_scales = tf.Variable(tf.random.uniform(shape=(num_dims, ),
                                                          minval=np.log(init_hp_minval),
                                                          maxval=np.log(init_hp_maxval)),
                                        dtype=tf.float64,
                                        shape=(1, ),
                                        name='log_length_scales_dim_{}'.format(gp_number))

        self.log_hps.append((log_variance, log_noise, log_length_scales))

        kernel = self._get_kernel()
        return (tf.exp(log_variance) * stf.GP(kernel()).stretch(tf.exp(log_length_scales)) +
                tf.exp(log_noise) * stf.GP(stf.Delta()))

    @staticmethod
    def _set_tf_config() -> None:

        tf.config.threading.set_inter_op_parallelism_threads(0)
        tf.config.threading.set_intra_op_parallelism_threads(0)
        tf.config.set_soft_device_placement(True)

    def _setup_models(self) -> None:
        # Models
        for i in range(self.output_dim):
            self.models.append(self._setup_gp(self.input_dim + i))

        # Model posteriors
        for i, model in enumerate(self.models):
            x_ph = tf.placeholder(tf.float64, [None, self.input_dim + i], name='x_train')
            y_ph = tf.placeholder(tf.float64, [None, 1], name='y_train')
            x_test_ph = tf.placeholder(tf.float64, [None, self.input_dim + i], name='x_test')

            model_post = model | (model(x_ph), y_ph)

            self.model_post_means.append(model_post.mean(x_test_ph))
            self.model_post_vars.append(stf.dense(model_post.kernel.elwise(x_test_ph)))
            self.model_post_phs.append((x_ph, y_ph, x_test_ph))

    def _setup_loss(self) -> tf.Tensor:
        # Log PDFs
        model_logpdfs = []

        for i, model in enumerate(self.models):
            x_ph = tf.placeholder(tf.float64, [None, self.input_dim + i], name='x_train')
            y_ph = tf.placeholder(tf.float64, [None, 1], name='y_train')

            model_logpdfs.append(model(x_ph).logpdf(y_ph))
            self.model_logpdf_phs.append((x_ph, y_ph))

        return -tf.add_n(model_logpdfs)

    def _setup_optimizer(self) -> tf.contrib.opt.ScipyOptimizerInterface:
        bounds = {}
        for variables in self.log_hps:
            for variable in variables:
                bounds[variable] = self.VARIABLE_LOG_BOUNDS

        options = {
            'disp': None,
            'iprint': -1,
        }

        # Optimizers supporting bounds: L-BFGS-B, TNC, SLSQP,...
        return tf.contrib.opt.ScipyOptimizerInterface(
            self.loss,
            var_to_bounds=bounds,
            method='L-BFGS-B',
            options=options,
        )

    def _get_kernel(self) -> Callable:
        if self.kernel_name == 'matern' or self.kernel_name == 'matern52':
            return stf.Matern52
        elif self.kernel_name == 'rbf':
            return stf.EQ

        raise Exception("Unknown kernel '" + str(self.kernel_name) + "'")

    def train(self) -> None:
        self._update_mean_std()

        with self._get_session() as session:
            feed_dict = {}
            for i, (x_ph, y_ph) in enumerate(self.model_logpdf_phs):
                feed_dict[x_ph] = np.concatenate((self.xs_normalized, self.ys_normalized[:, :i]), axis=1)
                feed_dict[y_ph] = self.ys_normalized[:, i:i + 1]

            lowest_loss = float('inf')
            success = False

            for i in range(self.num_optimizer_restarts):
                # Re-initialize variables
                session.run(tf.global_variables_initializer())

                try:
                    self.optimizer.minimize(session, feed_dict=feed_dict)
                    loss = session.run(self.loss, feed_dict=feed_dict)
                    self._print(f'Iteration {i},\tLoss: {loss:.4f}')
                    success = True

                    if loss < lowest_loss:
                        lowest_loss = loss
                        self.save_model(session)

                # Exception thrown if Cholesky decomposition was not successful
                except Exception as e:
                    self._print(f'Iteration {i} failed: ' + str(e))

            if (not success) and (self.num_optimizer_restarts > 0):
                raise RuntimeError(f'Failed to optimize model with {self.num_optimizer_restarts} attempts')

            self.load_model(session)
            loss = session.run(self.loss, feed_dict=feed_dict)
            self._print(f'Final loss: {loss:.4f}')

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

    def add_pseudo_point(self, x: np.ndarray) -> None:
        assert x.shape[0] == self.input_dim

        mean, var = self.predict_batch(xs=x.reshape(1, -1))

        self._append_data_point(x, mean)
        self.num_pseudo_points += 1

    def add_true_point(self, x: np.ndarray, y: np.ndarray) -> None:
        assert self.num_pseudo_points == 0
        assert x.shape[0] == self.input_dim
        assert y.shape[0] == self.output_dim

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

    def save_model(self, session: tf.Session):
        saver = tf.train.Saver()
        saver.save(sess=session, save_path=self.CHECKPOINT_NAME)

    def load_model(self, session: tf.Session):
        saver = tf.train.Saver()
        saver.restore(sess=session, save_path=self.CHECKPOINT_NAME)

    def _print(self, message: str):
        if self.verbose:
            print(message)