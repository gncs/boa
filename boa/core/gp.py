import logging
import tensorflow as tf
from stheno.tensorflow import EQ, Delta, Matern52, GP, Graph

__all__ = ["GaussianProcess", "CoreError"]

logger = logging.getLogger(__name__)


class CoreError(Exception):
    """
    Base error thrown by modules in the core
    """


class GaussianProcess(tf.Module):
    """
    Base GP class for the use of more complicated GP models in BOA.
    It is a wrapper around Stheno GPs, and provides additional methods that we need in BOA, such as optimizing
    the hyperparameters of models.
    """

    AVAILABLE_KERNELS = {"rbf": EQ,
                         "matern52": Matern52, }

    SIG_AMP = "signal_amplitude"
    LEN_SCALE = "length_scales"
    NOISE_AMP = "noise_amplitude"

    def __init__(self,
                 kernel: str,
                 signal_amplitude,
                 length_scales,
                 noise_amplitude,
                 jitter: tf.float64 = 1e-6,
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

        self.signal_amplitude = signal_amplitude
        self.length_scales = length_scales
        self.noise_amplitude = noise_amplitude
        self.jitter = jitter

        self.verbose = verbose

        self.input_dim = self.length_scales.shape[0]

        # Create model parts
        self.graph = Graph()

        signal_kernel = self.signal_amplitude * self.kernel.stretch(self.length_scales)

        noise_kernel = self.noise_amplitude * Delta()

        self.signal = GP(signal_kernel, graph=self.graph)
        self.noise = GP(noise_kernel, graph=self.graph)

        # Data stuff
        self.xs = tf.zeros((0, self.input_dim), dtype=tf.float64)
        self.ys = tf.zeros((0, self.input_dim), dtype=tf.float64)

        self.xs_forward_transform = lambda x: x
        self.ys_forward_transform = lambda y: y

        self.xs_backward_transform = lambda x: x
        self.ys_backward_transform = lambda y: y

    def copy(self):
        """
        Copy the GP

        :return: Reference to the copy of the GP
        """

        gp = GaussianProcess(kernel=self.kernel_name,
                             signal_amplitude=self.signal_amplitude,
                             length_scales=self.length_scales,
                             noise_amplitude=self.noise_amplitude,
                             jitter=self.jitter,
                             verbose=self.verbose)

        return gp

    def __or__(self, inputs, min_std=1e-10) -> tf.Module:
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

        gp.signal = gp.signal | (gp.xs_forward_transform(xs), gp.ys_forward_transform(ys))

        new_noise = GP(gp.noise.kernel, gp.noise.mean, graph=gp.signal.graph)
        gp.noise = new_noise

        return gp

    def log_pdf(self, xs, ys, normalize=False):

        if normalize:
            xs_forward, _ = self._create_transforms(xs)
            ys_forward, _ = self._create_transforms(ys)

        else:
            xs_forward = self.xs_forward_transform
            ys_forward = self.ys_forward_transform

        xs = xs_forward(xs)
        ys = ys_forward(ys)

        return self.signal(xs).logpdf(ys)

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

    def predict(self, xs, latent=True):

        xs = self.xs_forward_transform(xs)

        if latent:
            prediction = self.signal.mean(xs)
            pred_var = self.signal.kernel.elwise(xs)

        else:
            prediction = (self.signal + self.noise).mean(xs)
            pred_var = (self.signal + self.noise).kernel.elwise(xs)

        prediction = tf.reshape(prediction, (-1, 1))
        prediction = self.ys_backward_transform(prediction)

        pred_var = tf.reshape(pred_var, (-1, 1))
        pred_var = self.ys_backward_transform(pred_var, var=True)

        return prediction, pred_var

    @staticmethod
    def _create_transforms(input, min_std=1e-6):

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

