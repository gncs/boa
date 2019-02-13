from typing import Tuple, List

import GPy
import numpy as np

from .abstract import AbstractModel


class GPModel(AbstractModel):
    NOISE = 1e-10

    def __init__(self, kernel: str, num_optimizer_restarts: int):
        """
        Constructor of GP model.

        :param kernel: name of kernel
        :param num_optimizer_restarts: number of times the optimization of the hyperparameters is restarted
        """

        super().__init__()

        self.models: List[GPy.models.GPRegression] = []

        self.kernel_name = kernel
        self.num_optimizer_restarts = num_optimizer_restarts

        self.input_dim = 0
        self.output_dim = 0

        self.xs_mean = 0
        self.xs_std = 1

        self.ys_mean = 0
        self.ys_std = 1

        self.xs = np.array([[]])
        self.ys = np.array([[]])

        self.num_pseudo_points = 0
        self.num_true_points = 0

    def set_data(self, xs: np.ndarray, ys: np.ndarray):
        """
        Set data for GP.

        :param xs: dimensions N x D_input
        :param ys: dimensions N x D_output
        """
        self.input_dim = xs.shape[1]
        self.output_dim = ys.shape[1]

        self.xs = xs
        self.ys = ys

        self._update_mean_std()

        # Pseudo and true points
        self.num_true_points = xs.shape[0]

        self._update_models()

    @staticmethod
    def normalize(a: np.ndarray, mean: float, std: float) -> np.ndarray:
        return (a - mean) / std

    def _update_mean_std(self) -> None:
        self.xs_mean = np.mean(self.xs, axis=0)
        self.xs_std = np.std(self.xs, axis=0) + self.NOISE

        self.ys_mean = np.mean(self.ys, axis=0)
        self.ys_std = np.std(self.ys, axis=0) + self.NOISE

    def get_kernel(self):
        if self.kernel_name == 'matern':
            return GPy.kern.Matern52(input_dim=self.input_dim, ARD=True)
        elif self.kernel_name == 'rbf':
            return GPy.kern.RBF(input_dim=self.input_dim, ARD=True)

        raise Exception("Unknown kernel '" + str(self.kernel_name) + "'")

    def train(self):
        self._update_mean_std()
        x_normalized = self.normalize(self.xs, mean=self.xs_mean, std=self.xs_std)
        y_normalized = self.normalize(self.ys, mean=self.ys_mean, std=self.ys_std)

        self.models.clear()

        # Build model for each output
        for i in range(self.output_dim):
            kernel = self.get_kernel()
            model = GPy.models.GPRegression(
                X=x_normalized,
                Y=y_normalized[:, i:i + 1],
                kernel=kernel,
                normalizer=True,
            )

            model.optimize_restarts(num_restarts=self.num_optimizer_restarts, robust=True)
            model.optimize(messages=False)

            self.models.append(model)

    def predict_batch(self, test_xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        means = np.zeros((test_xs.shape[0], self.output_dim))
        var = np.zeros((test_xs.shape[0], self.output_dim))

        for i, model in enumerate(self.models):
            means[:, i:i + 1], var[:, i:i + 1] = model.predict(
                Xnew=self.normalize(test_xs, mean=self.xs_mean, std=self.xs_std), full_cov=False)

        return (means * self.ys_std + self.ys_mean), (var * self.ys_std**2)

    def add_pseudo_point(self, x: np.ndarray) -> None:
        x = x.reshape(1, -1)
        y = np.zeros((1, self.output_dim))

        # Make predictions for pseudo point
        for i, model in enumerate(self.models):
            y[:, i], _ = model.predict(Xnew=self.normalize(x, mean=self.xs_mean, std=self.xs_std), full_cov=False)

        # Renormalize y and add data point to models
        self._append_data_point(x, y * self.ys_std + self.ys_mean)

        self._update_models()

        self.num_pseudo_points += 1

    def add_true_point(self, x: np.ndarray, y: np.ndarray) -> None:
        assert self.num_pseudo_points == 0

        self._append_data_point(x, y)

        self._update_models()

        self.num_true_points += 1

    def remove_pseudo_points(self) -> None:
        self.xs = self.xs[:-self.num_pseudo_points, :]
        self.ys = self.ys[:-self.num_pseudo_points, :]

        self._update_models()

        self.num_pseudo_points = 0

    def _append_data_point(self, x: np.ndarray, y: np.ndarray) -> None:
        self.xs = np.vstack((self.xs, x))
        self.ys = np.vstack((self.ys, y))

    def _update_models(self) -> None:
        x_normalized = self.normalize(self.xs, mean=self.xs_mean, std=self.xs_std)
        y_normalized = self.normalize(self.ys, mean=self.ys_mean, std=self.ys_std)

        for i, model in enumerate(self.models):
            model.set_XY(X=x_normalized, Y=y_normalized[:, i:i + 1])
