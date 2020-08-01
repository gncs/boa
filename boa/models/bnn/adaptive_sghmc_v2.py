import tensorflow as tf
from functools import partial


class AdaptiveSGHMC(tf.optimizers.Optimizer):
    """
    In this implementation we assume that the scaled gradient noise variance
    (beta_hat in the original paper) is set to 0.
    """

    _VELOCITY_NAME = "velocity"
    _SQUARED_GRAD_NAME = "squared_grad"
    _SMOOTH_GRAD_NAME = "smooth_grad"
    _TAU_NAME = "tau"

    def __init__(self,
                 learning_rate,
                 burnin,
                 initialization_rounds=0,
                 overestimation_rate=1.,
                 data_size=1,
                 friction=0.01,
                 name="AdaptiveSGHMC",
                 eps=1e-6,
                 **kwargs):

        with tf.name_scope(name):

            super().__init__(name=name, **kwargs)

            self._set_hyper("learning_rate", learning_rate)
            self._set_hyper("burnin", burnin)
            self._set_hyper("data_size", data_size)
            self._set_hyper("friction", friction)

            self._set_hyper("initialization_rounds", initialization_rounds)
            self._set_hyper("overestimation_rate", overestimation_rate)

            self.eps = eps
            self.eps_squared = self.eps ** 2.

            self.use_initialization = initialization_rounds > 0

    def get_config(self):
        config = super().get_config()

        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "burnin": self._serialize_hyperparameter("burnin"),
            "data_size": self._serialize_hyperparameter("data_size"),
            "momentum_decay": self._serialize_hyperparameter("momentum_decay"),
            "initialization_rounds": self._serialize_hyperparameter("initialization_rounds"),
            "overestimation_rate": self._serialize_hyperparameter("overestimation_rate"),
        })

        return config

    def _create_slots(self, var_list):

        # If we estimate the auxiliary parameters before the burn-in, we initialize them to zeros,
        # if there is not pre-burn-in initialization, ones are usually reasonable
        auxiliary_slot_initializer = "zeros" if self.use_initialization > 0 else "ones"

        for var in var_list:
            self.add_slot(var, self._VELOCITY_NAME, initializer="zeros")

            # V_hat
            self.add_slot(var, self._SQUARED_GRAD_NAME, initializer=auxiliary_slot_initializer)

            # g
            self.add_slot(var, self._SMOOTH_GRAD_NAME, initializer=auxiliary_slot_initializer)

            # exponential average coefficient
            self.add_slot(var, self._TAU_NAME, initializer="ones")

    def _resource_apply_dense(self, grad, var, apply_state=None):

        velocity = self.get_slot(var, self._VELOCITY_NAME)
        squared_grad = self.get_slot(var, self._SQUARED_GRAD_NAME)
        smooth_grad = self.get_slot(var, self._SMOOTH_GRAD_NAME)
        tau = self.get_slot(var, self._TAU_NAME)

        return self._sghmc_step(batch_grad=grad,
                                variable=var,
                                velocity=velocity,
                                squared_grad=squared_grad,
                                smooth_grad=smooth_grad,
                                tau=tau)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        raise NotImplementedError

    def _sghmc_step(self,
                    batch_grad,
                    variable,
                    velocity,
                    squared_grad,
                    smooth_grad,
                    tau):

        init_rounds = self._get_hyper("initialization_rounds", dtype=self.iterations.dtype)
        data_size = self._get_hyper("data_size", dtype=batch_grad.dtype)

        # Scale the gradient according to the data size
        batch_grad = batch_grad * data_size

        return tf.cond(self.iterations < init_rounds,

                       true_fn=partial(self._sghmc_initialization_step,
                                       batch_grad=batch_grad,
                                       smooth_grad=smooth_grad,
                                       squared_grad=squared_grad,
                                       tau=tau),

                       false_fn=partial(self._sghmc_initialized_step,
                                        batch_grad=batch_grad,
                                        variable=variable,
                                        velocity=velocity,
                                        squared_grad=squared_grad,
                                        smooth_grad=smooth_grad,
                                        tau=tau,
                                        ))

    def _sghmc_initialization_step(self,
                                   batch_grad,
                                   smooth_grad,
                                   squared_grad,
                                   tau):

        # ---------------------------------------------------------------------
        # Hyper-parameter adaptation based on the paper "No more pesky learning rates"
        # ---------------------------------------------------------------------
        init_rounds = self._get_hyper("initialization_rounds", dtype=self.iterations.dtype)
        overestimation_rate = self._get_hyper("overestimation_rate", dtype=batch_grad.dtype)

        smooth_grad.assign_add(batch_grad)
        squared_grad.assign_add(tf.square(batch_grad))

        def average_step():
            n0 = tf.cast(init_rounds, batch_grad.dtype)

            smooth_grad.assign(smooth_grad / n0)
            squared_grad.assign((squared_grad / n0) * overestimation_rate)
            tau.assign(n0 * tf.ones_like(tau))

            tf.print("SGHMC hyperparameters initialized!")

            return []

        # During the initialization rounds, there are no updates to the actual
        # parameters and momentum
        return tf.cond(self.iterations == init_rounds - 1,
                       true_fn=average_step,
                       false_fn=lambda: [])

    def _sghmc_burnin_update(self,
                             batch_grad,
                             smooth_grad,
                             squared_grad,
                             tau):
        # ---------------------------------------------------------------------
        # Optimizer hyper-parameter adaptation during burn-in
        # ---------------------------------------------------------------------
        # If we are still in the burn-in phase, adapt:
        # the preconditioner,
        # the the exponentail averaging coefficient
        # the gradient noise

        noise_variance_ratio = tf.square(smooth_grad) / (squared_grad + self.eps)

        # Tau delta
        delta_tau = -tau * noise_variance_ratio + 1.

        tau_inv = 1. / (tau + self.eps)

        # g delta
        delta_smooth_grad = tau_inv * (-smooth_grad + batch_grad)

        # V_theta delta
        delta_squared_grad = tau_inv * (-squared_grad + tf.square(batch_grad))

        new_tau = tau + delta_tau
        new_smooth_grad = smooth_grad + delta_smooth_grad
        new_squared_grad = squared_grad + delta_squared_grad

        # Simultaneous update to optimizer hyper-parameters
        tau.assign(new_tau)
        smooth_grad.assign(new_smooth_grad)
        squared_grad.assign(new_squared_grad)

        return []

    def _sghmc_initialized_step(self,
                                batch_grad,
                                variable,
                                velocity,
                                squared_grad,
                                smooth_grad,
                                tau):

        init_rounds = self._get_hyper("initialization_rounds", dtype=self.iterations.dtype)
        burnin = self._get_hyper("burnin", dtype=self.iterations.dtype)
        friction = self._get_hyper("friction", dtype=batch_grad.dtype)
        learning_rate = self._get_hyper("learning_rate", dtype=batch_grad.dtype)
        learning_rate_squared = learning_rate ** 2.

        tf.cond(tf.logical_and(init_rounds < self.iterations,
                               self.iterations < burnin),

                true_fn=partial(self._sghmc_burnin_update,
                                batch_grad=batch_grad,
                                smooth_grad=smooth_grad,
                                squared_grad=squared_grad,
                                tau=tau),

                false_fn=lambda: [])

        # ---------------------------------------------------------------------
        # Actual SGHMC step
        # ---------------------------------------------------------------------
        inverse_mass = 1. / tf.sqrt(squared_grad + self.eps_squared)

        # Note the assumption that momentum_decay = learning_rate * inverse_mass * C
        noise_variance = 2. * inverse_mass * friction - learning_rate_squared
        noise_variance = learning_rate_squared * tf.maximum(noise_variance, self.eps_squared)

        velocity_noise = tf.random.normal(shape=batch_grad.shape, dtype=batch_grad.dtype)
        velocity_noise = tf.sqrt(noise_variance) * velocity_noise

        velocity_delta = -friction * velocity + \
                         -learning_rate_squared * inverse_mass * batch_grad + \
                         velocity_noise

        new_velocity = velocity + velocity_delta
        new_variable = variable + new_velocity

        variable.assign(new_variable)
        velocity.assign(new_velocity)

        # https://github.com/tensorflow/tensorflow/issues/30711#issuecomment-512921409
        return []


