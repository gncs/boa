from collections.abc import Iterable

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

    @staticmethod
    def get_all(bounded_vars):
        """
        Get the forward transforms of all given bounded variables
        :param bounded_vars:
        :return:
        """

        res = []
        for bv in bounded_vars:
            if isinstance(bv, BoundedVariable):
                res.append(bv())

            elif isinstance(bv, Iterable):
                res.append(BoundedVariable.get_all(bv))

        return res

    @staticmethod
    def get_reparametrizations(bounded_vars, flatten=False):
        """
        Returns the list of reparameterizations for a list of BoundedVariables. Useful to pass to
        tf.GradientTape.watch

        :param bounded_vars:
        :return:
        """

        res = []
        for bv in bounded_vars:
            if isinstance(bv, BoundedVariable):
                res.append(bv.reparameterization)

            elif isinstance(bv, Iterable):

                reparams = BoundedVariable.get_reparametrizations(bv)

                if flatten:
                    res += reparams
                else:
                    res.append(reparams)

        return res


class PermutationVariable(tf.Module):
    def __init__(self, name="permutation_variable", **kwargs):
        super(PermutationVariable, self).__init__(name=name, **kwargs)
