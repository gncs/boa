import logging
import tensorflow as tf
from stheno.tensorflow import EQ, Delta, Matern52, GP, Graph, dense

from boa import ROOT_DIR

from .utils import CoreError, setup_logger
from .kernel import DiscreteMatern52, DiscreteEQ, KendallTauEQ, KendallTauMatern52

__all__ = ["GaussianProcess", "CoreError"]

logger = setup_logger(__name__, level=logging.DEBUG, log_file=f"{ROOT_DIR}/../logs/gp.log", to_console=True)


class GaussianProcess(tf.Module):
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
        "tau_rbf": KendallTauEQ,
        "tau_matern52": KendallTauMatern52
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
                 jitter: tf.float64 = 1e-10,
                 verbose: bool = False,
                 name: str = "gaussian_process",
                 **kwargs):

        super(GaussianProcess, self).__init__(name=name, **kwargs)

        # Check if the specified kernel is available
        if kernel in self.AVAILABLE_KERNELS:
            self.kernel_name = kernel
            self.kernel = self.AVAILABLE_KERNELS[kernel]()
        else:
            raise CoreError(f"Specified kernel '{kernel}' not available!")

        # Convert the amplitudes
        self.signal_amplitude = tf.convert_to_tensor(signal_amplitude, tf.float64)
        self.noise_amplitude = tf.convert_to_tensor(noise_amplitude, tf.float64)
        self.jitter_amplitude = tf.convert_to_tensor(jitter, tf.float64)

        if self.signal_amplitude <= 0:
            raise CoreError(f"Signal amplitude must be strictly positive! {self.signal_amplitude} was given.")

        if self.noise_amplitude <= 0:
            raise CoreError(f"Noise amplitude must be strictly positive! {self.noise_amplitude} was given.")

        if self.jitter_amplitude <= 0:
            raise CoreError(f"Jitter amplitude must be strictly positive! {self.jitter_amplitude} was given.")

        # Convert and reshape the length scales
        self.length_scales = tf.convert_to_tensor(length_scales, tf.float64)

        if tf.rank(self.length_scales) > 1:
            raise CoreError(f"Length scales rank must be at most 1!")

        self.length_scales = tf.reshape(self.length_scales, [-1])

        if tf.reduce_any(self.length_scales <= 0):
            raise CoreError(f"All length scale amplitudes must be strictly positive! {self.length_scales} was given.")

        self.verbose = verbose

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

        self.xs_forward_transform = lambda x, var=False: x
        self.ys_forward_transform = lambda y, var=False: y

        self.xs_backward_transform = lambda x, var=False: x
        self.ys_backward_transform = lambda y, var=False: y

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
                             jitter=self.jitter_amplitude,
                             verbose=self.verbose)

        # Copy data
        gp.xs = self.xs
        gp.ys = self.ys

        gp.xs_forward_transform, gp.xs_backward_transform = self._create_transforms(gp.xs)
        gp.ys_forward_transform, gp.ys_backward_transform = self._create_transforms(gp.ys)

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

        # Calculate forward and backward transforms
        gp.xs_forward_transform, gp.xs_backward_transform = self._create_transforms(xs)
        gp.ys_forward_transform, gp.ys_backward_transform = self._create_transforms(ys)

        return gp

    def log_pdf(self,
                xs,
                ys,
                normalize_with_input=False,
                normalize_with_training_data=False,
                latent=False,
                with_jitter=True):

        if normalize_with_input and normalize_with_training_data:
            raise CoreError("Data in log_pdf can only be normalized with one scheme only (both were specified True)!")

        if normalize_with_input:
            xs_forward, _ = self._create_transforms(xs)
            ys_forward, _ = self._create_transforms(ys)

        elif normalize_with_training_data:
            xs_forward, _ = self._create_transforms(self.xs)
            ys_forward, _ = self._create_transforms(self.ys)

        else:
            xs_forward = lambda x: x
            ys_forward = lambda y: y

        xs = xs_forward(xs)
        ys = ys_forward(ys)

        # print("using xs ys")
        # print(xs, ys)

        gp = self.signal

        if not latent:
            gp = gp + self.noise

        if with_jitter:
            gp = gp + self.jitter

        # Condition on the training data
        if self.xs.shape[0] > 0:
            gp = gp | (xs_forward(self.xs), ys_forward(self.ys))

        return gp(xs).logpdf(ys)

    def sample(self, xs, num=1, latent=False):

        xs = self.xs_forward_transform(xs)

        if latent:
            sample = self.signal(xs).sample(num=num)

        else:
            sample = (self.signal + self.noise)(xs).sample(num=num)

        sample = tf.reshape(sample, (-1, 1))
        sample = self.ys_backward_transform(sample)
        sample = tf.reshape(sample, (-1, num))

        return sample

    def predict(self, xs, latent=True, with_jitter=True):
        """
        :param xs: Input points for which we are predicting the output
        :param latent: if True, we will _NOT_ add on the learned noise process for prediction
        :param with_jitter: If True, we predict with a constant jittery noise process added on
        :return: Tensor of predictions given by the model
        """

        xs = self.xs_forward_transform(xs)

        gp = self.signal

        if not latent:
            gp = gp + self.noise

        if with_jitter:
            gp = gp + self.jitter

        gp = gp | (self.xs_forward_transform(self.xs), self.ys_forward_transform(self.ys))

        # dense(X) is a no-op if X is a tensorflow op, or it is X.mat if it is a stheno.Dense
        prediction = dense(gp.mean(xs))
        pred_var = dense(gp.kernel.elwise(xs))

        prediction = tf.reshape(prediction, (-1, 1))
        prediction = self.ys_backward_transform(prediction)

        pred_var = tf.reshape(pred_var, (-1, 1))
        pred_var = self.ys_backward_transform(pred_var, var=True)

        return prediction, pred_var

    def normalize_with_training_data(self, data, output=False):

        # Create the normalizing transforms
        if output:
            forward_transform, _ = self._create_transforms(self.ys)
        else:
            forward_transform, _ = self._create_transforms(self.xs)

        return forward_transform(data)

    @staticmethod
    def _create_transforms(input, min_std=1e-6):
        """

        :param input: rank-2 tensor with rows of inputs
        :param min_std:
        :return:
        """

        if input.shape[0] == 0:
            return lambda x, var=False: x, lambda x, var=False: x

        # Calculate data statistics
        mean, var = tf.nn.moments(input, axes=[0], keepdims=True)
        std = tf.maximum(tf.sqrt(var), min_std)

        def forward(x, var=False):
            if var:
                return x / (std**2)
            else:
                return (x - mean) / std

        def backward(x, var=False):
            if var:
                return x * (std**2)
            else:
                return (x * std) + mean

        return forward, backward


class DiscreteGaussianProcess(GaussianProcess):

    def __init__(self,
                 kernel: str,
                 input_dim: int,
                 signal_amplitude,
                 jitter: tf.float64 = 1e-10,
                 verbose: bool = False,
                 name: str = "discrete_gaussian_process",
                 **kwargs):

        super().__init__(kernel=kernel,
                         input_dim=input_dim,
                         name=name,
                         signal_amplitude=signal_amplitude,
                         jitter=jitter,
                         verbose=verbose,
                         **kwargs)

    @staticmethod
    def _create_transforms(input, min_std=1e-6, input_normalizable=False):
        if input.shape[0] == 0:
            return lambda x, var=False: x, lambda x, var=False: x

        # Calculate data statistics
        mean, var = tf.nn.moments(input, axes=[0], keepdims=True)
        std = tf.maximum(tf.sqrt(var), min_std)

        if input_normalizable:
            def forward(x, var=False):
                if var:
                    return x / (std**2)
                else:
                    return (x - mean) / std

            def backward(x, var=False):
                if var:
                    return x * (std ** 2)
                else:
                    return (x * std) + mean

        else:
            forward = lambda x, var=False: x
            backward = lambda x, var=False: x

        return forward, backward
