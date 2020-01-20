import tensorflow as tf

from .utils import CoreError, sigmoid_inverse


class BoundedVariable(tf.Module):
    """
    Wrapper class for a Tensorflow variable. It enables solving constrained optimization
    problems using gradient-based methods by smoothly reparameterizing an unconstrained varaible through a
    sigmoid transformation.
    """

    def __init__(self, init, lower, upper, dtype=tf.float64, name="bounded_variable", **kwargs):

        super(BoundedVariable, self).__init__(name=name, **kwargs)

        self.dtype = dtype

        self.lower = tf.convert_to_tensor(lower, dtype=self.dtype)
        self.upper = tf.convert_to_tensor(upper, dtype=self.dtype)

        self.reparameterization = tf.Variable(self.backward_transform(init))

    @tf.Module.with_name_scope
    def forward_transform(self, x):
        """
        Go from unconstrained domain to constrained domain
        :param x: tensor to be transformed in the unconstrained domain
        :return: tensor in the constrained domain
        """
        x = tf.convert_to_tensor(x, dtype=self.dtype)
        return (self.upper - self.lower) * tf.nn.sigmoid(x) + self.lower

    @tf.Module.with_name_scope
    def backward_transform(self, x, eps=1e-12):
        """
        Go from constrained domain to unconstrained domain
        :param x: tensor to be transformed in the constrained domain
        :return: tensor in the unconstrained domain
        """
        x = tf.convert_to_tensor(x, dtype=self.dtype)
        return sigmoid_inverse((x - self.lower) / (self.upper - self.lower + eps))

    @tf.Module.with_name_scope
    def assign(self, x):
        self.reparameterization.assign(self.backward_transform(x))

    @tf.Module.with_name_scope
    def __call__(self):
        return self.forward_transform(self.reparameterization)


class PermutationVariable(tf.Module):

    def __init__(self, name="permutation_variable", **kwargs):

        super(PermutationVariable, self).__init__(name=name, **kwargs)
