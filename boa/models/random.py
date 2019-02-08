import numpy as np

from .abstract import AbstractModel


class RandomModel(AbstractModel):
    def __init__(self, seed, num_samples):
        self.random_state = np.random.RandomState(seed=seed)
        self.num_samples = num_samples

        self.dim = 0

    def set_data(self, xs, ys):
        self.dim = ys.shape[1]

    def add_true_point(self, x, y):
        pass

    def add_pseudo_point(self, x):
        pass

    def remove_pseudo_points(self):
        pass

    def train(self):
        pass

    def predict_batch(self, x):
        return self.random_state.rand(x.shape[0], self.dim), np.var(
            np.random.rand(self.num_samples, x.shape[0], self.dim), axis=0)
