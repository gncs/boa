from plum import Dispatcher, Self
from stheno.util import uprank
from stheno.kernel import Kernel
from stheno.matrix import Dense
from lab import B
from tensorflow import stop_gradient, round
import tensorflow_probability as tfp
import tensorflow as tf

from scipy.stats import kendalltau

log_add_exp = tfp.math.log_add_exp


def _round_with_straight_through(x, precision=1.):
    return x + stop_gradient(round(x * precision) / precision - x)


def kendall_corr_distance(tup):
    x, y = tup

    return tf.convert_to_tensor(2 * x.shape[-1] * (1 - kendalltau(x, y).correlation), dtype=tf.float64)


def kendall_elwise_distance(x, y):
    res = tf.map_fn(kendall_corr_distance, (x, y), dtype=tf.float64)
    res = tf.reshape(res, [-1, 1])
    return res


def kendall_pointwise_distance(x, y):
    tile_x = tf.tile(x[:, None, :], [1, y.shape[0], 1])[None, :, :, :]
    tile_y = tf.tile(y[None, :, :], [x.shape[0], 1, 1])[None, :, :, :]

    num_x_points = x.shape[0]
    dim = x.shape[-1]

    # 2 x (N x M) x D
    cartesian_product = tf.reshape(tf.concat([tile_x, tile_y], axis=0), [2, -1, dim])

    dist_vector = tf.map_fn(kendall_corr_distance,
                            (cartesian_product[0], cartesian_product[1]),
                            dtype=tf.float64)

    return tf.reshape(dist_vector, [num_x_points, -1])


class DiscreteEQ(Kernel):
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, precision=1., **kwargs):

        super().__init__(**kwargs)

        self.precision = precision

    @_dispatch(B.Numeric, B.Numeric)
    @uprank
    def __call__(self, x, y):
        return Dense(self._compute(B.pw_dists2(_round_with_straight_through(x, precision=self.precision),
                                               _round_with_straight_through(y, precision=self.precision))))

    @_dispatch(B.Numeric, B.Numeric)
    @uprank
    def elwise(self, x, y):
        return self._compute(B.ew_dists2(_round_with_straight_through(x, precision=self.precision),
                                         _round_with_straight_through(y, precision=self.precision)))

    def _compute(self, dists2):
        return B.exp(-0.5 * dists2)

    @property
    def _stationary(self):
        return True

    @_dispatch(Self)
    def __eq__(self, other):
        return True


class DiscreteMatern52(Kernel):
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, precision=1., **kwargs):
        super().__init__(**kwargs)

        self.precision = precision

    @_dispatch(B.Numeric, B.Numeric)
    @uprank
    def __call__(self, x, y):
        res = Dense(self._compute(B.pw_dists(_round_with_straight_through(x, precision=self.precision),
                                             _round_with_straight_through(y, precision=self.precision))))

        return res

    @_dispatch(B.Numeric, B.Numeric)
    @uprank
    def elwise(self, x, y):
        return self._compute(B.ew_dists(_round_with_straight_through(x, precision=self.precision),
                                        _round_with_straight_through(y, precision=self.precision)))

    def _compute(self, dists):
        r1 = 5 ** .5 * dists
        r2 = 5 * dists ** 2 / 3

        return (1 + r1 + r2) * B.exp(-r1)

    @property
    def _stationary(self):
        return True

    @_dispatch(Self)
    def __eq__(self, other):
        return True


class KendallTauEQ(Kernel):
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ls = 1.

    @_dispatch(B.Numeric, B.Numeric)
    @uprank
    def __call__(self, x, y):
        return Dense(self._compute(kendall_pointwise_distance(x, y)))

    @_dispatch(B.Numeric, B.Numeric)
    @uprank
    def elwise(self, x, y):
        res = self._compute(kendall_elwise_distance(x, y))
        return res

    def _compute(self, dists2):
        return B.exp(-0.5 * dists2 / self.ls)

    @property
    def _stationary(self):
        return True

    @_dispatch(B.Numeric)
    def stretch(self, stretch):
        self.ls = stretch
        return self

    @_dispatch(Self)
    def __eq__(self, other):
        return True


class KendallTauMatern52(Kernel):
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, precision=1., **kwargs):
        super().__init__(**kwargs)

        self.precision = precision

    @_dispatch(B.Numeric, B.Numeric)
    @uprank
    def __call__(self, x, y):
        return Dense(self._compute(tf.math.sqrt(kendall_pointwise_distance(x, y))))

    @_dispatch(B.Numeric, B.Numeric)
    @uprank
    def elwise(self, x, y):
        return self._compute(tf.math.sqrt(kendall_elwise_distance(x, y)))

    def _compute(self, dists):
        r1 = 5 ** .5 * dists
        r2 = 5 * dists ** 2 / 3

        return (1 + r1 + r2) * B.exp(-r1)

    @property
    def _stationary(self):
        return True

    @_dispatch(Self)
    def __eq__(self, other):
        return True
