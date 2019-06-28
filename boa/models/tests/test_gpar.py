import os
import tempfile
from unittest import TestCase

import numpy as np

from boa.models.gp import GPModel
from boa.models.gpar import GPARModel


class TestGPAR(TestCase):
    @staticmethod
    def f(x: np.ndarray) -> np.ndarray:
        x = x[:, 0]
        return np.sinc(3 * x).reshape(-1, 1)

    def setUp(self) -> None:
        np.random.seed(42)
        self.X_train = np.random.rand(7, 2) * 2 - 1
        self.Y_train = self.f(self.X_train)
        self.pseudo_point = np.array([0.8, 0.3]).reshape(1, -1)

        x_cont = np.arange(-1.5, 1.5, 0.02).reshape(-1, 1)
        self.x_cont = np.hstack([x_cont, x_cont])

    def test_compare_gps(self):
        model = GPModel(kernel='rbf', num_optimizer_restarts=10)
        model.set_data(self.X_train, self.Y_train)
        model.train()

        y_predict, var_predict = model.predict_batch(self.x_cont)

        with tempfile.TemporaryDirectory() as tmp_dir:
            cwd = os.getcwd()
            try:
                os.chdir(tmp_dir)

                model2 = GPARModel(kernel='rbf', num_optimizer_restarts=10)
                model2.set_data(self.X_train, self.Y_train)
                model2.train()
                y_predict_2, var_predict_2 = model2.predict_batch(self.x_cont)

            finally:
                os.chdir(cwd)

        np.array_equal(y_predict, y_predict_2)
