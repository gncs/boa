from plum import Dispatcher, Self
from stheno.util import uprank
from stheno.kernel import Kernel
from stheno.matrix import Dense
from lab import B
from tensorflow import stop_gradient, round
import tensorflow_probability as tfp
import tensorflow as tf

log_add_exp = tfp.math.log_add_exp


def _round_with_straight_through(x, precision=1.):
    return x + stop_gradient(round(x * precision) / precision - x)


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

    def _compute(self, dists, rounding_eps=1e-6):
        r1 = 5 ** .5 * dists
        r2 = 5 * dists ** 2 / 3

        return (1 + r1 + r2) * B.exp(-r1)

    @property
    def _stationary(self):
        return True

    @_dispatch(Self)
    def __eq__(self, other):
        return True
