import tensorflow as tf
from stheno.tensorflow import EQ, Delta, Matern52, GP, Graph, dense

from .utils import CoreError, tensor_hash, standardize
from .kernel import DiscreteMatern52, DiscreteEQ, PermutationEQ, PermutationMatern52

__all__ = ["GaussianProcess", "CoreError"]


class GaussianProcess(object):
    """
    Base GP class for the use of more complicated GP models in BOA.
    It is a wrapper around Stheno GPs, and provides additional methods that we need in BOA, such as optimizing
    the hyperparameters of models.
    """

    AVAILABLE_KERNELS = {
        "rbf": EQ,
        "matern52": Matern52,
        "discrete_matern52": DiscreteMatern52,
        "discrete_rbf": DiscreteEQ,
        "perm_eq": PermutationEQ,
        "perm_matern52": PermutationMatern52
    }

    SIG_AMP = "signal_amplitude"
    LEN_SCALE = "length_scales"
    NOISE_AMP = "noise_amplitude"

    def __init__(self,
                 kernel: str,
                 input_dim: int,
                 signal_amplitude,
                 length_scales,
                 noise_amplitude,
                 kernel_args={},
                 jitter: tf.float64 = 1e-10,
                 name: str = "gaussian_process"):

        self.name = name

        # Check if the specified kernel is available
        if kernel in self.AVAILABLE_KERNELS:
            self.kernel_name = kernel
            self.kernel_args = kernel_args
            self.kernel = self.AVAILABLE_KERNELS[kernel](**kernel_args)
        else:
            raise CoreError(f"Specified kernel '{kernel}' not available!")

        # Convert the amplitudes
        self.signal_amplitude = tf.convert_to_tensor(signal_amplitude, tf.float64)
        self.noise_amplitude = tf.convert_to_tensor(noise_amplitude, tf.float64)
        self.jitter_amplitude = tf.convert_to_tensor(jitter, tf.float64)

        if signal_amplitude <= 0.:
            raise CoreError(f"Signal amplitude must be strictly positive! {self.signal_amplitude} was given.")

        if noise_amplitude <= 0.:
            raise CoreError(f"Noise amplitude must be strictly positive! {self.noise_amplitude} was given.")

        if jitter <= 0.:
            raise CoreError(f"Jitter amplitude must be strictly positive! {self.jitter_amplitude} was given.")
        # Convert and reshape the length scales
        self.length_scales = tf.convert_to_tensor(length_scales, tf.float64)

        if tf.rank(self.length_scales) > 1:
            raise CoreError(f"Length scales rank must be at most 1!")

        self.length_scales = tf.reshape(self.length_scales, [-1])

        if tf.reduce_any(self.length_scales <= 0):
            raise CoreError(f"All length scale amplitudes must be strictly positive! {self.length_scales} was given.")

        self.input_dim = input_dim

        # Create model parts
        self.graph = Graph()

        # Need special handling for the kernel length scale, because, the default
        # Stheno stretch behaviour divides the inputs to the kernel: k(x / l, y / l)
        signal_kernel = self.signal_amplitude * self.kernel.stretch(self.length_scales)

        noise_kernel = self.noise_amplitude * Delta()

        jitter_kernel = self.jitter_amplitude * self.signal_amplitude * Delta()

        self.signal = GP(signal_kernel, graph=self.graph)
        self.noise = GP(noise_kernel, graph=self.graph)
        self.jitter = GP(jitter_kernel, graph=self.graph)

        # Data stuff
        self.xs = tf.zeros((0, self.input_dim), dtype=tf.float64)
        self.ys = tf.zeros((0, 1), dtype=tf.float64)

        # ---------------------------------------------------------------------
        # Stuff used in property methods
        # ---------------------------------------------------------------------
        self._xs_stat_hash = None
        self._xs_mean = None
        self._xs_std = None

        self._ys_stat_hash = None
        self._ys_mean = None
        self._ys_std = None

    @property
    def xs_mean_and_std(self, eps=1e-7, min_std=1e-8):
        """
        Returns the means and standard deviations of the training inputs.
        If we have calculated them already, then we do not perform the calculations again.
        :param eps: small positive value to improve the stability of taking the square root
        :param min_std: lower bound on the standard deviation
        :return:
        """
        xs_stat_hash = tensor_hash(self.xs)

        # If the data does not match what we had before, recalculate the statistics
        if xs_stat_hash != self._xs_stat_hash:
            self._xs_stat_hash = xs_stat_hash

            if self.xs.shape[0] > 0:
                mean, var = tf.nn.moments(self.xs, axes=[0], keepdims=True)
            else:
                mean, var = tf.cast(0., tf.float64), tf.cast(1., tf.float64)

            self._xs_mean = mean
            self._xs_std = tf.maximum(tf.sqrt(var + eps), min_std)

        return self._xs_mean, self._xs_std

    @property
    def ys_mean_and_std(self, eps=1e-7, min_std=1e-8):
        ys_stat_hash = tensor_hash(self.ys)

        # If the data does not match what we had before, recalculate the statistics
        if ys_stat_hash != self._ys_stat_hash:
            self._ys_stat_hash = ys_stat_hash

            if self.ys.shape[0] > 0:
                mean, var = tf.nn.moments(self.ys, axes=[0], keepdims=True)
            else:
                mean, var = tf.cast(0., tf.float64), tf.cast(1., tf.float64)

            self._ys_mean = mean
            self._ys_std = tf.maximum(tf.sqrt(var + eps), min_std)

        return self._ys_mean, self._ys_std

    @property
    def xs_standardized(self):
        mean, std = self.xs_mean_and_std
        return (self.xs - mean) / std

    @property
    def ys_standardized(self):
        mean, std = self.ys_mean_and_std
        return (self.ys - mean) / std

    def standardize_predictive_input(self, xs):
        """
        Standardizes the new input with the statistics of the training inputs
        :param xs:
        :return:
        """
        mean, std = self.xs_mean_and_std

        return (xs - mean) / std

    def standardize_predictive_output(self, ys):
        """
        Standardizes the new output with the statistics of the training outputs
        :param ys:
        :return:
        """
        mean, std = self.ys_mean_and_std

        return (ys - mean) / std

    def unstandardize_predictive_output(self, ys):
        """
        Back-transform a standardized output with the statistics of the training outputs
        :param ys:
        :return:
        """
        mean, std = self.ys_mean_and_std

        return ys * std + mean

    def copy(self):
        """
        Copy the GP

        :return: Reference to the copy of the GP
        """

        gp = GaussianProcess(kernel=self.kernel_name,
                             input_dim=self.input_dim,
                             signal_amplitude=self.signal_amplitude,
                             length_scales=self.length_scales,
                             noise_amplitude=self.noise_amplitude,
                             jitter=self.jitter_amplitude)

        # Copy data
        gp.xs = self.xs
        gp.ys = self.ys

        return gp

    def __or__(self, inputs, min_std=1e-10) -> 'GaussianProcess':
        """
        Adds data to the model. The notation is supposed to imitate
        the conditioning operation:

        posterior = prior | (xs, ys)

        :param inputs: Tuple of a rank-2 tensor and a rank-1 tensor: the first
        N x I and the second N, where N is the number of training examples,
        I is the dimension of the input.

        :return: Reference to the conditioned model
        """

        gp = self.copy()

        xs, ys = inputs

        xs = tf.reshape(xs, (-1, self.input_dim))
        ys = tf.reshape(ys, (-1, 1))

        # aggregate data
        xs = tf.concat((self.xs, xs), axis=0)
        ys = tf.concat((self.ys, ys), axis=0)

        # Set GP data
        gp.xs = xs
        gp.ys = ys

        return gp

    def log_pdf(self, xs, ys, predictive=False, log_normal=False):

        if predictive:
            xs = self.standardize_predictive_input(xs)
            ys = self.standardize_predictive_output(ys)

        else:
            xs = standardize(xs)
            ys = standardize(ys)

        gp = self.signal

        if not predictive:
            gp = gp + self.noise + self.jitter

        # Condition on the training data
        else:
            if self.xs.shape[0] > 0:
                gp = gp | (self.xs_standardized, self.ys_standardized)

        log_pdf = gp(xs).logpdf(ys)

        if log_normal:
            log_pdf = log_pdf - tf.reduce_sum(ys)

        return log_pdf

    def sample(self, xs, num=1, latent=False):

        xs = self.standardize_predictive_input(xs)

        if latent:
            sample = self.signal(xs).sample(num=num)

        else:
            sample = (self.signal + self.noise)(xs).sample(num=num)

        sample = tf.reshape(sample, (-1, 1))
        sample = self.unstandardize_predictive_output(sample)
        sample = tf.reshape(sample, (-1, num))

        return sample

    def predict(self, xs, latent=True, with_jitter=False, denoising=False):
        """
        :param xs: Input points for which we are predicting the output
        :param latent: if True, we will _NOT_ add on the learned noise process for prediction
        :param with_jitter: If True, we predict with a constant jittery noise process added on
        :return: Tensor of predictions given by the model
        """

        xs = self.standardize_predictive_input(xs)

        gp = self.signal

        if not latent:
            gp = gp + self.noise

        if with_jitter:
            gp = gp + self.jitter

        if denoising:
            K = dense(gp.kernel(self.xs_standardized))
            K_star = dense(self.signal.kernel(self.xs_standardized, xs))
            K_star_star = dense(self.signal.kernel(xs, xs))

            K_inv_times_K_star = tf.linalg.solve(K, K_star)

            prediction = tf.matmul(K_inv_times_K_star, self.ys_standardized, transpose_a=True)

            pred_var = K_star_star - tf.matmul(K_star, K_inv_times_K_star, transpose_a=True)

            # We care about the variances at the points, not the full covariance matrix
            pred_var = tf.linalg.diag_part(pred_var)

        else:
            gp = gp | (self.xs_standardized, self.ys_standardized)

            # dense(X) is a no-op if X is a tensorflow op, or it is X.mat if it is a stheno.Dense
            prediction = dense(gp.mean(xs))
            pred_var = dense(gp.kernel.elwise(xs))

        prediction = tf.reshape(prediction, (-1, 1))
        prediction = self.unstandardize_predictive_output(prediction)

        pred_var = tf.reshape(pred_var, (-1, 1))
        pred_var = pred_var * (self.ys_mean_and_std[1]**2)

        return prediction, pred_var


class DiscreteGaussianProcess(GaussianProcess):
    def __init__(self,
                 kernel: str,
                 input_dim: int,
                 signal_amplitude,
                 kernel_args={},
                 jitter: tf.float64 = 1e-10,
                 verbose: bool = False,
                 name: str = "discrete_gaussian_process",
                 **kwargs):

        super().__init__(kernel=kernel,
                         kernel_args=kernel_args,
                         input_dim=input_dim,
                         name=name,
                         signal_amplitude=signal_amplitude,
                         jitter=jitter,
                         verbose=verbose,
                         **kwargs)
