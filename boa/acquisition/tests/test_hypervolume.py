from unittest import TestCase

import numpy as np
import pygmo

from boa.acquisition.util import calculate_hypervolume


class TestHyperVolume(TestCase):

    def test_compare_hvs(self):
        np.random.seed(42)

        # Generate sets of random points in 3D
        datasets = []
        for _ in range(10):
            points = np.random.rand(10, 3)
            ref = np.minimum(np.max(points, axis=0) + np.array([0.1, 0.1, 0.1]), np.array([1, 1, 1]))
            datasets.append((points, ref))

        # Compare results
        for i, (points, reference) in enumerate(datasets):
            method_a = calculate_hypervolume(points, reference)

            hv = pygmo.hypervolume(points)
            method_b = hv.compute(reference)

            self.assertAlmostEqual(method_a, method_b)
