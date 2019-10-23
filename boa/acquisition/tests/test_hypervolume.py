from unittest import TestCase

import numpy as np

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

        # Reference results from pygmo
        reference_results = [
            0.47702243158170915,
            0.6161150115782258,
            0.38987148653281684,
            0.4140610977579395,
            0.5087925750638401,
            0.32921655859579135,
            0.3500480479881244,
            0.40333940645403715,
            0.24307450141182813,
            0.4682523247401579,
        ]

        # Compare results
        for i, (points, reference) in enumerate(datasets):
            method_a = calculate_hypervolume(points, reference)

            self.assertAlmostEqual(method_a, reference_results[i])
