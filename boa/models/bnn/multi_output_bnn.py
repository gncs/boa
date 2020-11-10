from typing import NamedTuple

import abc
import json

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from boa.core.utils import get_mean_and_std
from boa.models.bnn.dense_with_prior import GaussianDenseWithGammaPrior
from boa.models.bnn.adaptive_sghmc_v2 import AdaptiveSGHMC

from tqdm import trange

tfd = tfp.distributions
tfb = tfp.bijectors
tfl = tf.keras.layers


class SufficientStatistics(NamedTuple):
    mean: tf.Tensor
    std: tf.Tensor


class ModelError(Exception):
    """Base error thrown by models"""


class BNN(tf.keras.Model, abc.ABC):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 **kwargs):
        """

        :param kernel:
        :param input_dim:
        :param output_dim:
        :param parallel:
        :param verbose:
        :param _num_starting_data_points: Should not be set by the user. Only used to restore models.
        :param name:
        :param kwargs:
        """

        super().__init__(dtype=tf.float64, **kwargs)

        # Check if the specified kernel is available
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.xs = tf.Variable(tf.zeros((0, input_dim), dtype=tf.float64),
                              name="inputs",
                              trainable=False,
                              shape=(None, input_dim))

        self.ys = tf.Variable(tf.zeros((0, output_dim), dtype=tf.float64),
                              name="outputs",
                              trainable=False,
                              shape=(None, output_dim))

        # Every element of this list will contain a complete specification of a
        # model sampled using SGHMC
        self.model_samples = []

        self.xs_stats = None
        self.ys_stats = None

        # ---------------------------------------------------------------------
        # Flags
        # ---------------------------------------------------------------------
        self.trained = tf.Variable(False, name="trained", trainable=False)

    @abc.abstractmethod
    def resample_hyperparameters(self):
        pass

    @abc.abstractmethod
    def weight_log_prob(self):
        pass

    @abc.abstractmethod
    def hyper_log_prob(self):
        pass

    @abc.abstractmethod
    def get_weights(self):
        pass

    @abc.abstractmethod
    def set_weights(self, weights):
        pass

    @abc.abstractmethod
    def log_prob(self, xs, ys, **kwargs):
        pass

    def copy(self, name=None):

        # Reflect the class of the current instance
        constructor = self.__class__

        # Get the config of the instance
        config = self.get_config()

        # Instantiate the model
        model = constructor(**config)

        # # Create dictionaries of model variables
        # self_dict = {v.name: v for v in self.variables}
        # model_dict = {v.name: v for v in model.variables}
        #
        # # Copy variables over
        # for k, v in self_dict.items():
        #     model_dict[k].assign(v)

        return model

    def condition_on(self, xs, ys, keep_previous=True):
        """
        the conditioning operation:

        posterior = prior | (xs, ys)

        :param xs: rank-2 tensor: N x I where N is the number of training examples,
        I is the dimension of the input.
        :param ys: rank-2 tensor: N x O, where N is the number of training examples,
        O is the dimension of the output.
        :param keep_previous: if True, the data on which we conditioned before is retained as well.

        :return: Reference to the conditioned model
        """

        xs, ys = self._validate_and_convert_input_output(xs, ys)

        model = self.copy()

        if keep_previous:
            xs = tf.concat((self.xs, xs), axis=0)
            ys = tf.concat((self.ys, ys), axis=0)

        model.xs.assign(xs)
        model.ys.assign(ys)

        return model

    def standardize_input(self, xs):
        if self.xs_stats is None:
            raise ValueError("self.xs_stats hasn't been initialized yet and was None!")

        return (xs - self.xs_stats.mean) / self.xs_stats.std

    def standardize_output(self, ys):
        if self.ys_stats is None:
            raise ValueError("self.ys_stats hasn't been initialized yet and was None!")

        return (ys - self.ys_stats.mean) / self.ys_stats.std

    def unstandardize_output(self, ys):
        if self.ys_stats is None:
            raise ValueError("self.ys_stats hasn't been initialized yet and was None!")

        return ys * self.ys_stats.std + self.ys_stats.mean

    def fit(self,
            num_samples,
            burnin,
            keep_every=50,
            resample_hypers_every=100,
            learning_rate=1e-2,
            friction=0.05,
            initialization_rounds=10,
            overestimation_rate=3.,
            seed=None,
            ) -> None:
        """
        :param seed:
        :param kwargs:
        :return:
        """

        num_epochs = initialization_rounds + burnin + num_samples * keep_every

        if self.xs.value().shape[0] == 0:
            raise ModelError("No data to fit to!")

        self.build(input_shape=(None, self.input_dim))

        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)

        xs = self.xs.value()
        ys = self.ys.value()

        xs_mean, xs_std = get_mean_and_std(xs)
        ys_mean, ys_std = get_mean_and_std(ys)

        self.xs_stats = SufficientStatistics(mean=xs_mean,
                                             std=xs_std)
        self.ys_stats = SufficientStatistics(mean=ys_mean,
                                             std=ys_std)

        xs = self.standardize_input(xs)
        ys = self.standardize_output(ys)

        data_size = tf.cast(xs.shape[0], xs.dtype)

        sampler = AdaptiveSGHMC(learning_rate=learning_rate,
                                friction=friction,
                                burnin=burnin,
                                data_size=data_size,
                                initialization_rounds=initialization_rounds,
                                overestimation_rate=overestimation_rate)

        self.model_samples.clear()

        @tf.function
        def train_step(step):

            if step % resample_hypers_every == 0:
                self.resample_hyperparameters()

            with tf.GradientTape() as tape:
                data_log_prob = tf.reduce_mean(self.log_prob(xs, ys))

                weight_log_prob = self.weight_log_prob() / data_size
                hyper_log_prob = self.hyper_log_prob() / data_size

                loss = -(data_log_prob + weight_log_prob + hyper_log_prob)

            gradients = tape.gradient(loss, self.trainable_variables)
            sampler.apply_gradients(zip(gradients, self.trainable_variables))

            return loss, data_log_prob, weight_log_prob, hyper_log_prob

        # We start at 1, so that the hyperparameters are not resampled at the start
        # step_bar = trange(1, num_epochs + 1)
        # for step in step_bar:

        #     if step > burnin and step % keep_every == 0:
        #         self.model_samples.append(self.get_weights())

        #     loss, data_log_prob, weight_log_prob, hyper_log_prob = train_step(tf.convert_to_tensor(step))

        #     step_bar.set_description(f"Loss: {loss.numpy():.3f}, "
        #                              f"Data log prob: {data_log_prob.numpy():.3f}, "
        #                              f"Weight log prob: {weight_log_prob.numpy():.3f}, "
        #                              f"Hyper log prob: {hyper_log_prob.numpy():.3f}")


        for step in range(1, num_epochs + 1):

            if step > burnin and step % keep_every == 0:
                self.model_samples.append(self.get_weights())

            loss, data_log_prob, weight_log_prob, hyper_log_prob = train_step(tf.convert_to_tensor(step))

        print(f"Optimization finised with:\nLoss: {loss.numpy():.3f}, "
                f"Data log prob: {data_log_prob.numpy():.3f}, "
                f"Weight log prob: {weight_log_prob.numpy():.3f}, "
                f"Hyper log prob: {hyper_log_prob.numpy():.3f}")

        self.trained.assign(True)

    def predict(self, xs, numpy=False, **kwargs):

        xs = self._validate_and_convert(xs, output=False)
        xs = self.standardize_input(xs)

        predictions = []

        for weights in self.model_samples:
            self.set_weights(weights)
            prediction = self(xs)
            prediction = self.unstandardize_output(prediction)

            predictions.append(prediction)

        predictions = tf.stack(predictions, axis=0)

        means, variances = tf.nn.moments(predictions, axes=[0])

        if numpy:
            means = means.numpy()
            variances = variances.numpy()

        return means, variances

    @abc.abstractmethod
    def get_config(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def from_config(config, **kwargs):
        pass

    def _validate_and_convert(self, xs, output=False):

        xs = tf.convert_to_tensor(xs)
        xs = tf.cast(xs, tf.float64)

        if len(xs.shape) == 1:
            second_dim = self.output_dim if output else self.input_dim

            # Attempt to convert the xs to the right shape
            xs = tf.reshape(xs, (-1, second_dim))

        # Check if the shapes are correct
        if not len(xs.shape) == 2:
            raise ModelError("The input must be of rank 2!")

        if (not output and xs.shape[1] != self.input_dim) or (output and xs.shape[1] != self.output_dim):
            out_text = 'output' if output else 'input'
            raise ModelError(f"The second dimension of the {out_text} "
                             f"is incorrect: {xs.shape[1]} (expected {self.output_dim if output else self.input_dim})!")

        return xs

    def _validate_and_convert_input_output(self, xs, ys):

        xs = self._validate_and_convert(xs, output=False)
        ys = self._validate_and_convert(ys, output=True)

        # Ensure the user provided the same number of input and output points
        if not xs.shape[0] == ys.shape[0]:
            raise ModelError(f"The first dimension of the input ({xs.shape[0]}) and the output ({ys.shape[0]}) must "
                             f"be equal! (the data needs to form valid input-output pairs)")

        return xs, ys

    def save(self, save_path, **kwargs):

        if not self.trained:
            print("Saved model has not been trained yet!")

        self.save_weights(save_path)

        config = self.get_config()

        with open(save_path + ".json", "w") as config_file:
            json.dump(config, config_file, indent=4, sort_keys=True)

        np.save(save_path + "_model_samples.npy", self.model_samples)

    @staticmethod
    @abc.abstractmethod
    def restore(save_path):
        pass


class BasicBNN(BNN):

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_units=50,
                 log_var_init=1e-3,
                 log_var_prior_loc=1e-4,
                 alpha0=10.,
                 beta0=10.,
                 **kwargs):

        super().__init__(input_dim=input_dim,
                         output_dim=output_dim,
                         **kwargs)

        self.hidden_units = hidden_units

        self.transforms = [
            GaussianDenseWithGammaPrior(units=self.hidden_units,
                                        prior_mode="weight_and_bias",
                                        activation=tf.nn.tanh,
                                        alpha0=alpha0,
                                        beta0=beta0,
                                        dtype=self.dtype),

            GaussianDenseWithGammaPrior(units=self.hidden_units,
                                        prior_mode="weight_and_bias",
                                        activation=tf.nn.tanh,
                                        alpha0=alpha0,
                                        beta0=beta0,
                                        dtype=self.dtype),

            GaussianDenseWithGammaPrior(units=self.hidden_units,
                                        prior_mode="weight_and_bias",
                                        activation=tf.nn.tanh,
                                        alpha0=alpha0,
                                        beta0=beta0,
                                        dtype=self.dtype),

            GaussianDenseWithGammaPrior(units=self.output_dim,
                                        prior_mode="weight_and_bias",
                                        alpha0=alpha0,
                                        beta0=beta0,
                                        dtype=self.dtype)
        ]

        log_var_init = tf.cast(tf.math.log(log_var_init), self.dtype)
        self.likelihood_log_var = tf.Variable(log_var_init * tf.zeros(self.output_dim, dtype=self.dtype),
                                              name="likelihood_log_variance")

        loc_init = tf.cast(tf.math.log(log_var_prior_loc), self.dtype)
        self.var_prior = tfd.Independent(
            distribution=tfd.LogNormal(loc=loc_init * tf.ones(self.output_dim, dtype=self.dtype),
                                       scale=0.1),
            reinterpreted_batch_ndims=1)

    def resample_hyperparameters(self):
        for layer in self.transforms:
            layer.resample_precisions()

    def weight_log_prob(self):

        log_prob = 0.

        for layer in self.transforms:
            log_prob += layer.weight_log_prob()

        return log_prob

    def hyper_log_prob(self):
        log_prob = self.var_prior.log_prob(tf.exp(self.likelihood_log_var))

        for layer in self.transforms:
            log_prob += layer.hyper_prior_log_prob()

        return log_prob

    def get_weights(self):
        weights = [layer.get_weights() for layer in self.transforms]
        weights.append(self.likelihood_log_var.numpy())

        return weights

    def set_weights(self, weights):
        for layer, ws in zip(self.transforms, weights[:-1]):
            layer.set_weights(ws)

        self.likelihood_log_var.assign(weights[-1])

    def get_config(self):
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_units": self.hidden_units
        }

    @staticmethod
    def from_config(config, **kwargs):
        return BasicBNN(**config)

    def log_prob(self, xs, ys, **kwargs):

        loc = self(xs)
        scale = tf.ones_like(loc) * tf.exp(0.5 * self.likelihood_log_var)

        likelihood = tfd.Normal(loc=loc, scale=scale)

        return tf.reduce_sum(likelihood.log_prob(ys), axis=1)

    def call(self, inputs, training=None, mask=None):

        tensor = inputs

        for layer in self.transforms:
            tensor = layer(tensor)

        return tensor

    @staticmethod
    def restore(save_path):

        with open(save_path + ".json", "r") as config_file:
            config = json.load(config_file)

        model = BasicBNN.from_config(config, )

        model.load_weights(save_path)

        model.model_samples = np.load(save_path + "_model_samples.npy")

        return model
