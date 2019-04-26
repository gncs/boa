import numpy as np
import stheno.tf as stf
import tensorflow as tf
from tensorflow_probability import distributions as tfpd

from .gpar import GPARModel, ParameterManager


class HyperGPARModel(GPARModel):
    INIT_STD = 2.5

    def __init__(self, kernel: str, num_optimizer_restarts: int, verbose: bool = False):
        """
        Constructor of HyperGPAR model.

        :param kernel: name of kernel
        :param num_optimizer_restarts: number of times the optimization of the hyperparameters is restarted
        :param verbose: log optimization of hyperparameters
        """

        super().__init__(kernel=kernel, num_optimizer_restarts=num_optimizer_restarts, verbose=verbose)

        self.reg_parameter_manager = None
        self.init_reg_params = None

    def _setup(self) -> None:
        if self.session:
            self.session.close()

        config = tf.ConfigProto(
            intra_op_parallelism_threads=0, inter_op_parallelism_threads=0, allow_soft_placement=True)
        self.session = tf.Session(config=config)

        # Models
        for i in range(self.output_dim):
            self.models.append(self._setup_gp(self.input_dim + i))

        # Log PDFs
        for i, model in enumerate(self.models):
            x_placeholder = tf.placeholder(tf.float64, [None, self.input_dim + i], name='x_train')
            y_placeholder = tf.placeholder(tf.float64, [None, 1], name='y_train')

            self.model_logpdfs.append(model(x_placeholder).logpdf(y_placeholder))
            self.model_logpdf_phs.append((x_placeholder, y_placeholder))

        # Regularizers
        reg_params = [
            tf.Variable(
                initial_value=1.0,
                dtype=tf.float64,
                name='variance_mu',
            ),
            tf.Variable(
                initial_value=self.INIT_STD,
                dtype=tf.float64,
                name='variance_std',
            ),
            tf.Variable(
                initial_value=1.0,
                dtype=tf.float64,
                name='noise_mu',
            ),
            tf.Variable(
                initial_value=self.INIT_STD,
                dtype=tf.float64,
                name='noise_std',
            ),
            tf.Variable(
                initial_value=tf.fill(dims=[self.input_dim], value=tf.dtypes.cast(x=1.0, dtype=tf.float64)),
                name='lengthscales_mu',
            ),
            tf.Variable(
                initial_value=tf.fill(dims=[self.input_dim], value=tf.dtypes.cast(x=self.INIT_STD, dtype=tf.float64)),
                name='lengthscales_std',
            ),
        ]

        self.reg_parameter_manager = ParameterManager(self.session, [reg_params])

        regularizers = []
        for i, (log_variance, log_noise, log_lengthscales) in enumerate(self.log_hyperparameters):
            regularizers += [
                tf.log(tfpd.Normal(loc=reg_params[0], scale=reg_params[1]).prob(tf.exp(log_variance))),
                tf.log(tfpd.Normal(loc=reg_params[2], scale=reg_params[3]).prob(tf.exp(log_noise))),
                tf.log(
                    tfpd.MultivariateNormalDiag(
                        loc=tf.slice(reg_params[4], [0], [self.input_dim + i]),
                        scale_diag=tf.slice(reg_params[5], [0], [self.input_dim + i]),
                    ).prob(tf.exp(log_lengthscales))),
            ]

        # Posteriors
        for i, model in enumerate(self.models):
            x_placeholder = tf.placeholder(tf.float64, [None, self.input_dim + i], name='x_train')
            y_placeholder = tf.placeholder(tf.float64, [None, 1], name='y_train')
            x_test_placeholder = tf.placeholder(tf.float64, [None, self.input_dim + i], name='x_test')

            model_post = model | (model(x_placeholder), y_placeholder)

            self.model_post_means.append(model_post.mean(x_test_placeholder))
            self.model_post_vars.append(stf.dense(model_post.kernel.elwise(x_test_placeholder)))
            self.model_post_phs.append((x_placeholder, y_placeholder, x_test_placeholder))

        self.loss = -tf.add_n(self.model_logpdfs + regularizers)

        self.parameter_manager = ParameterManager(session=self.session, variables=self.log_hyperparameters)

        bounds = {}
        for variables in self.log_hyperparameters:
            for variable in variables:
                bounds[variable] = self.VARIABLE_LOG_BOUNDS

        options = {
            'disp': None,
            'iprint': -1,
        }

        # Optimizers supporting bounds: L-BFGS-B, TNC, SLSQP,...
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            self.loss,
            var_to_bounds=bounds,
            method='L-BFGS-B',
            options=options,
        )

        self.session.run(tf.global_variables_initializer())
        self.init_reg_params = self.reg_parameter_manager.get_values()

    def train(self):
        self._update_mean_std()

        feed_dict = {}
        for i, (x_placeholder, y_placeholder) in enumerate(self.model_logpdf_phs):
            feed_dict[x_placeholder] = np.concatenate((self.xs_normalized, self.ys_normalized[:, :i]), axis=1)
            feed_dict[y_placeholder] = self.ys_normalized[:, i:i + 1]

        lowest_loss = self.session.run(self.loss, feed_dict=feed_dict)
        best_params = self.parameter_manager.get_values()
        best_reg_params = self.reg_parameter_manager.get_values()

        for i in range(self.num_optimizer_restarts):
            self.parameter_manager.init_values(random_seed=i)
            self.reg_parameter_manager.set_values(self.init_reg_params)

            self.optimizer.minimize(self.session, feed_dict=feed_dict)
            loss = self.session.run(self.loss, feed_dict=feed_dict)
            self._print(f'Iteration {i}\tLoss: {loss}')

            if loss < lowest_loss:
                lowest_loss = loss
                best_params = self.parameter_manager.get_values()
                best_reg_params = self.reg_parameter_manager.get_values()

        self.parameter_manager.set_values(best_params)
        self.reg_parameter_manager.set_values(best_reg_params)
        loss = self.session.run(self.loss, feed_dict=feed_dict)
        self._print(f'Final loss: {loss}')
