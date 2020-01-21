import numpy as np

from .abstract_model_v2 import AbstractModel


class RandomModel(AbstractModel):
    def __init__(self, seed, num_samples, name="random_model", **kwargs):

        super(RandomModel, self).__init__(name=name, **kwargs)

        self.random_state = np.random.RandomState(seed=seed)
        self.num_samples = num_samples

        self.output_dim = 0

        self.ys_mean = 0
        self.ys_std = 1

        self.ys = np.array([[]])

    def condition_on(self, xs, ys):

        # Performs validation of the inputs
        super(RandomModel, self).condition_on(xs, ys)

        self.output_dim = ys.shape[1]
        self.ys = ys

        self._update_mean_std()

    def _update_mean_std(self) -> None:
        min_std = 1e-10

        self.ys_mean = np.mean(self.ys, axis=0)
        self.ys_std = np.maximum(np.std(self.ys, axis=0), min_std)

    def add_true_point(self, x, y):
        self._append_data_point(x, y)

    def add_pseudo_point(self, x):
        pass

    def _append_data_point(self, x: np.ndarray, y: np.ndarray) -> None:
        self.ys = np.vstack((self.ys, y))

    def remove_pseudo_points(self):
        pass

    def fit(self):
        self._update_mean_std()

    def predict(self, xs):
        return (self.random_state.normal(loc=self.ys_mean, scale=self.ys_std, size=(xs.shape[0], self.output_dim)),
                np.var(
                    self.random_state.normal(loc=self.ys_mean,
                                             scale=self.ys_std,
                                             size=(self.num_samples, xs.shape[0], self.output_dim)),
                    axis=0,
                ))
