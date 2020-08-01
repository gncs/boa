import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class GaussianDenseWithGammaPrior(tf.keras.layers.Dense):
    _AVAILABLE_PRIOR_MODES = [
        "per_param",
        "weight_and_bias",
        "per_unit"
    ]

    def __init__(self,
                 units,
                 prior_mode,
                 alpha0=1.,
                 beta0=1.,
                 activation=None,
                 use_bias=True,
                 name="gaussian_dense_with_prior",
                 **kwargs):

        super().__init__(units=units,
                         activation=activation,
                         use_bias=use_bias,
                         name=name,
                         **kwargs)

        if prior_mode not in self._AVAILABLE_PRIOR_MODES:
            raise ValueError(f"Prior mode must be one of {self._AVAILABLE_PRIOR_MODES}, but '{prior_mode}' was given!")

        self.prior_mode = prior_mode
        self.alpha0 = alpha0
        self.beta0 = beta0

        self.eps = 1e-6
        self.eps_squared = self.eps ** 2.

        self.vars = []

    @property
    def num_params(self):
        return tf.size(self.kernel) + (tf.size(self.bias) if self.use_bias else 0)

    def weight_log_prob(self):
        log_prob = tf.reduce_sum(self.kernel_prior.log_prob(self.kernel))

        if self.use_bias:
            log_prob += tf.reduce_sum(self.bias_prior.log_prob(self.bias))

        return log_prob

    def hyper_prior_log_prob(self):

        log_prob = tf.reduce_sum(self.kernel_prec_hyper_prior.log_prob(self.scale_to_prec(self.kernel_scale)))

        if self.use_bias:
            if self.prior_mode == "per_unit":
                bias_hyperprior = self.kernel_prec_hyper_prior
            else:
                bias_hyperprior = self.bias_prec_hyper_prior

            log_prob += tf.reduce_sum(bias_hyperprior.log_prob(self.scale_to_prec(self.bias_scale)))

        return log_prob

    def get_weights(self):
        return [v.numpy() for v in self.vars]

    def set_weights(self, weights):
        for v, weight in zip(self.vars, weights):
            v.assign(weight)

    def scale_to_prec(self, scale):
        return 1. / (tf.square(scale) + self.eps)

    def prec_to_scale(self, prec):
        return 1. / tf.sqrt(prec + self.eps_squared)

    def resample_precisions(self):

        # Perform Gibbs step for appropriate prior mode
        if self.prior_mode == "per_param":

            kernel_conc = self.alpha0 + 0.5
            kernel_conc = tf.ones_like(self.kernel) * kernel_conc
            kernel_rate = self.beta0 + tf.square(self.kernel) / 2.

            if self.use_bias:
                bias_conc = self.alpha0 + 0.5
                bias_conc = tf.ones_like(self.bias) * bias_conc
                bias_rate = self.beta0 + tf.square(self.bias) / 2.

        elif self.prior_mode == "weight_and_bias":

            kernel_conc = self.alpha0 + tf.cast(tf.size(self.kernel), self.kernel.dtype) / 2.
            kernel_rate = self.beta0 + tf.reduce_sum(tf.square(self.kernel)) / 2.

            if self.use_bias:
                bias_conc = self.alpha0 + tf.cast(tf.size(self.bias), self.bias.dtype) / 2.
                bias_rate = self.beta0 + tf.reduce_sum(tf.square(self.bias)) / 2.

        elif self.prior_mode == "per_unit":

            conc = self.alpha0 + tf.cast(self.kernel.shape[0], self.kernel.dtype) / 2.
            rate = self.beta0 + tf.reduce_sum(tf.square(self.kernel), axis=0) / 2.

            if self.use_bias:
                conc = conc + 0.5
                rate = rate + tf.square(self.bias) / 2.

                bias_conc = tf.ones_like(self.bias) * conc
                bias_rate = rate

            kernel_conc = tf.ones(self.units, dtype=self.dtype) * conc
            kernel_rate = rate

        else:
            raise NotImplementedError

        self.kernel_conc.assign(kernel_conc)
        self.kernel_rate.assign(kernel_rate)

        if self.use_bias and self.prior_mode != "per_unit":
            self.bias_conc.assign(bias_conc)
            self.bias_rate.assign(bias_rate)

        # Sample from the posteriors
        new_kernel_scale = self.prec_to_scale(self.kernel_prec_hyper_prior.sample())

        if self.prior_mode == "per_param":
            assigned_new_kernel_scale = new_kernel_scale
        elif self.prior_mode == "per_unit":
            assigned_new_kernel_scale = tf.ones([self.kernel.shape[0], 1], dtype=self.dtype) * new_kernel_scale[None, :]
        elif self.prior_mode == "weight_and_bias":
            assigned_new_kernel_scale = tf.ones_like(self.kernel) * new_kernel_scale
        else:
            raise NotImplementedError

        self.kernel_scale.assign(assigned_new_kernel_scale)

        if self.use_bias:

            if self.prior_mode != "per_unit":
                new_bias_scale = self.prec_to_scale(self.bias_prec_hyper_prior.sample())

            if self.prior_mode == "per_param":
                pass
            elif self.prior_mode == "per_unit":
                new_bias_scale = new_kernel_scale
            elif self.prior_mode == "weight_and_bias":
                new_bias_scale = tf.ones(self.units, dtype=self.dtype) * new_bias_scale
            else:
                raise NotImplementedError

            self.bias_scale.assign(new_bias_scale)

    def build(self, input_shape, scale_init=0.1):

        self.vars = []

        super().build(input_shape)

        kernel_hyperprior_shape = {
            "per_param": self.kernel.shape,
            "per_unit": (self.units, ),
            "weight_and_bias": (),
        }[self.prior_mode]

        self.kernel_conc = self.add_weight(
            'kernel_conc',
            shape=kernel_hyperprior_shape,
            initializer=tf.constant_initializer(value=self.alpha0),
            regularizer=None,
            constraint=None,
            dtype=self.dtype,
            trainable=False)

        self.kernel_rate = self.add_weight(
            'kernel_rate',
            shape=kernel_hyperprior_shape,
            initializer=tf.constant_initializer(value=self.beta0),
            regularizer=None,
            constraint=None,
            dtype=self.dtype,
            trainable=False)

        self.kernel_prec_hyper_prior = tfd.Gamma(concentration=self.kernel_conc,
                                                 rate=self.kernel_rate)

        self.kernel_scale = self.add_weight(
            'kernel_scale',
            shape=self.kernel.shape,
            initializer=tf.constant_initializer(value=scale_init),
            regularizer=None,
            constraint=None,
            dtype=self.dtype,
            trainable=False)

        self.kernel_prior = tfd.Normal(loc=tf.zeros_like(self.kernel),
                                       scale=self.kernel_scale)

        self.vars += [
            self.kernel,
            self.kernel_conc,
            self.kernel_rate,
            self.kernel_scale
        ]

        if self.use_bias:

            self.bias_scale = self.add_weight(
                'bias_scale',
                shape=[self.units, ],
                initializer=tf.constant_initializer(value=scale_init),
                regularizer=None,
                constraint=None,
                dtype=self.dtype,
                trainable=False)

            if self.prior_mode != "per_unit":

                bias_hyperprior_shape = {
                    "per_param": (self.units, ),
                    "weight_and_bias": (),
                }[self.prior_mode]

                self.bias_conc = self.add_weight(
                    'bias_conc',
                    shape=bias_hyperprior_shape,
                    initializer=tf.constant_initializer(value=self.alpha0),
                    regularizer=None,
                    constraint=None,
                    dtype=self.dtype,
                    trainable=False)

                self.bias_rate = self.add_weight(
                    'bias_rate',
                    shape=bias_hyperprior_shape,
                    initializer=tf.constant_initializer(value=self.beta0),
                    regularizer=None,
                    constraint=None,
                    dtype=self.dtype,
                    trainable=False)

                self.bias_prec_hyper_prior = tfd.Gamma(concentration=self.bias_conc,
                                                       rate=self.bias_rate)

            self.bias_prior = tfd.Normal(loc=tf.zeros_like(self.bias),
                                         scale=self.bias_scale)

            self.vars += [
                self.bias,
                self.bias_conc,
                self.bias_rate,
                self.bias_scale
            ]
