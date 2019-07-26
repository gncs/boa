import numpy as np
import stheno.tf as stf
import tensorflow as tf

from .gpar import GPARModel


class MFGPARModel(GPARModel):
    # Ensures that covariance matrix stays positive semidefinite
    VARIABLE_LOG_BOUNDS = (-7, 7)
    CHECKPOINT_NAME = 'mf_gpar.ckpt'

    def __init__(self, latent_size: int, max_opt_steps=1000, learning_rate=0.1, *args, **kwargs):
        """Initializer of MatrixFactorization-GPAR model."""
        super().__init__(*args, **kwargs)

        self.latent_size = latent_size
        self.learning_rate = learning_rate
        self.max_opt_steps = max_opt_steps

    def _setup_pca_gp(self, log_ls_input: tf.Tensor, outputs: int) -> tf.Tensor:
        log_variance = tf.get_variable(
            shape=(1, ),
            initializer=tf.random_uniform_initializer(minval=np.log(0.5), maxval=np.log(2.0), dtype=tf.float64),
            dtype=tf.float64,
            name='log_variance',
            constraint=lambda x: tf.clip_by_value(x, self.VARIABLE_LOG_BOUNDS[0], self.VARIABLE_LOG_BOUNDS[1]),
        )

        log_noise = tf.get_variable(
            shape=(1, ),
            initializer=tf.random_uniform_initializer(minval=np.log(0.5), maxval=np.log(2.0), dtype=tf.float64),
            dtype=tf.float64,
            name='log_noise',
            constraint=lambda x: tf.clip_by_value(x, -5, 5),
        )

        output_log_lengthscales = tf.get_variable(
            shape=(outputs, ),
            initializer=tf.random_uniform_initializer(minval=np.log(0.5), maxval=np.log(2.0), dtype=tf.float64),
            dtype=tf.float64,
            name='output_log_ls',
            constraint=lambda x: tf.clip_by_value(x, self.VARIABLE_LOG_BOUNDS[0], self.VARIABLE_LOG_BOUNDS[1]),
        )

        self.log_hps.append((log_variance, log_noise, output_log_lengthscales))

        log_lengthscales = tf.concat([log_ls_input[outputs], output_log_lengthscales], axis=0)

        kernel = self._get_kernel()
        return (tf.exp(log_variance) * stf.GP(kernel()).stretch(tf.exp(log_lengthscales)) +
                tf.exp(log_noise) * stf.GP(stf.Delta()))

    def _setup_models(self) -> None:
        print('Setting up model')

        # Models
        self.left_log_ls = tf.get_variable(
            shape=(self.output_dim, self.latent_size),
            initializer=None,
            dtype=tf.float64,
            name='left_log_ls',
        )

        self.right_log_ls = tf.get_variable(
            shape=(self.latent_size, self.input_dim),
            initializer=None,
            dtype=tf.float64,
            name='right_log_ls',
        )

        log_ls_input = tf.matmul(self.left_log_ls, self.right_log_ls)

        for i in range(self.output_dim):
            with tf.variable_scope(str(i)):
                self.models.append(self._setup_pca_gp(log_ls_input, i))

        # Model posteriors
        for i, model in enumerate(self.models):
            x_ph = tf.placeholder(tf.float64, [None, self.input_dim + i], name='x_train')
            y_ph = tf.placeholder(tf.float64, [None, 1], name='y_train')
            x_test_ph = tf.placeholder(tf.float64, [None, self.input_dim + i], name='x_test')

            model_post = model | (model(x_ph), y_ph)

            self.model_post_means.append(model_post.mean(x_test_ph))
            self.model_post_vars.append(stf.dense(model_post.kernel.elwise(x_test_ph)))
            self.model_post_phs.append((x_ph, y_ph, x_test_ph))

    def _setup_optimizer(self) -> tf.contrib.opt.ScipyOptimizerInterface:
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        return optimizer.minimize(self.loss)

    def train(self) -> None:
        self._update_mean_std()

        with self._get_session() as session:
            feed_dict = {}
            for i, (x_ph, y_ph) in enumerate(self.model_logpdf_phs):
                feed_dict[x_ph] = np.concatenate((self.xs_normalized, self.ys_normalized[:, :i]), axis=1)
                feed_dict[y_ph] = self.ys_normalized[:, i:i + 1]

            lowest_loss = float('inf')
            one_success = False

            for i in range(self.num_optimizer_restarts):
                # Re-initialize variables
                session.run(tf.global_variables_initializer())

                loss = last_loss = float('inf')
                failed = False

                for j in range(self.max_opt_steps):
                    try:
                        session.run(self.optimizer, feed_dict=feed_dict)
                        loss = session.run(self.loss, feed_dict=feed_dict)
                    except Exception:
                        failed = True
                        break

                    rel_diff = abs(last_loss - loss) / abs(last_loss)
                    if rel_diff < 1E-6:
                        break
                    else:
                        last_loss = loss

                if not failed:
                    self._print(f'Iteration {i},\tLoss: {loss:.4f}')
                    one_success = True

                    if loss < lowest_loss:
                        lowest_loss = loss
                        self.save_model(session)
                else:
                    self._print(f'Iteration {i} failed')

            if not one_success:
                raise RuntimeError(f'Failed to optimize model with {self.num_optimizer_restarts} attempts')

            self.load_model(session)
            loss = session.run(self.loss, feed_dict=feed_dict)
            self._print(f'Final loss: {loss:.4f}')
