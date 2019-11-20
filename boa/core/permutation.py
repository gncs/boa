import tensorflow as tf
from varz.tensorflow import Vars

class PermutationVariable(tf.Module):

    def __init__(self,
                 n_items: int,
                 vs: Vars = None,
                 temperature: float = 1.0,
                 sinkhorn_iterations: int = 20,
                 dtype: tf.DType = tf.float64,
                 name: str = "permutation_variable",
                 **kwargs):
        """
        :param n_items:
        :param vs: Vars container for the permutation parameters. If given,
        the parameters are assigned to the container.
        :param temperature:
        :param sinkhorn_iterations:
        :param dtype:
        :param name:
        :param kwargs:
        """

        super(PermutationVariable, self).__init__(name=name, **kwargs)

        self.n_items = n_items
        self.dtype = dtype

        init = tf.random.uniform(minval=tf.cast(0, dtype=self.dtype),
                                 maxval=tf.cast(1, dtype=self.dtype),
                                 shape=(self.n_items,),
                                 dtype=self.dtype)
        if vs is None:
            self._permutation_params = tf.Variable(init,
                                                   name=name)
        else:
            self._permutation_params = vs.get(init,
                                              name=name)

        self.vs = vs

        self.perm_map = tf.ones((1, self.n_items), dtype=self.dtype)
        self.temperature = tf.cast(temperature, self.dtype)
        self.sinkhorn_iterations = sinkhorn_iterations

    @property
    @tf.Module.with_name_scope
    def permutation_params(self):
        return self._permutation_params if self.vs is None else self.vs[self.name]

    @permutation_params.setter
    @tf.Module.with_name_scope
    def permutation_params(self, x):

        x = tf.convert_to_tensor(x, dtype=self.dtype)

        assert len(x.shape) == 1
        assert x.shape[0] == self.n_items

        if self.vs is None:
            self._permutation_params.assign(x)
        else:
            self.vs.assign(self.name, x)

    @tf.Module.with_name_scope
    def permutation_matrix(self, soft=True):

        perm_params = tf.reshape(self.permutation_params, (-1, 1))

        perm_mat = tf.matmul(perm_params, self.perm_map)

        # Add Gumbel noise for robustness
        perm_mat = perm_mat - tf.math.log(-tf.math.log(tf.random.uniform(shape=perm_mat.shape,
                                                                         dtype=self.dtype) + 1e-20))

        perm_mat = perm_mat / self.temperature

        # Perform Sinkhorn normalization
        for _ in range(self.sinkhorn_iterations):

            # Column-wise normalization in log domain
            perm_mat = perm_mat - tf.reduce_logsumexp(perm_mat, axis=1, keepdims=True)

            # Row-wise normalization in log domain
            perm_mat = perm_mat - tf.reduce_logsumexp(perm_mat, axis=0, keepdims=True)

        perm_mat = tf.exp(perm_mat)

        if not soft:
            perm_mat = tf.one_hot(tf.argmax(perm_mat, axis=0), self.n_items, dtype=self.dtype)

        return perm_mat

    @tf.Module.with_name_scope
    def __add__(self, x):

        return self.permutation_matrix() + x

    @tf.Module.with_name_scope
    def __mul__(self, x):

        return self.permutation_matrix() * x

    @tf.Module.with_name_scope
    def __div__(self, x):

        return self.permutation_matrix() / x

    @tf.Module.with_name_scope
    def permute(self, x, soft=True):

        x = tf.convert_to_tensor(x, dtype=self.dtype)

        if len(x.shape) == 1:
            x = tf.reshape(x, [-1, 1])

        return tf.matmul(self.permutation_matrix(soft=soft), x)
