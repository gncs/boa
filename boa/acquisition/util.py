import numpy as np


def get_frontier(points: np.ndarray) -> np.ndarray:
    """
    Return Pareto-optimal points.

    :param points: candidate points
    :return: list of Pareto-optimal points
    """
    indices = []
    for i, value_i in enumerate(points):
        dominated = False
        for j, value_j in enumerate(points):
            if np.all(value_i >= value_j) and (not np.all(value_i == value_j) or i < j):
                dominated = True
                break

        if not dominated:
            indices.append(i)

    return points[indices]


def normalize(data: np.ndarray, mean, std) -> np.ndarray:
    return data
    # copy = data.copy()
    # copy -= mean[None, :]
    # copy /= std[None, :]
    # return copy


def calculate_hypervolume(points: np.ndarray, reference: np.ndarray) -> float:
    """
    Calculate hypervolume (wrt to reference) spanned by <points>.

    :param points: data points
    :param reference: reference point
    :return: hypervolume with respect to reference point
    """
    assert reference.shape[0] == points.shape[1]

    return _calculate_hypervolume(points, reference, 0)


def _calculate_hypervolume(points: np.ndarray, reference: np.ndarray, dim: int) -> float:
    """
    Calculate hypervolume (wrt to reference) spanned by <points>.

    :param points: data points
    :param reference: reference point
    :param dim: dimension index
    :return: hypervolume with respect to reference point
    """
    if dim == points.shape[1] - 1:
        # Last dimension: return distance from reference to lowest point (result cannot be negative)
        return max(reference[dim] - min(points[:, dim]), 0.0)

    # Sort to sweep from "right to left"
    sorted_points = np.array(sorted(points, key=lambda entry: - entry[dim]))

    total = 0
    previous = reference[dim]

    while sorted_points.size != 0:
        current = sorted_points[0, dim]

        # Calculate volume recursively
        total += max(previous - current, 0.0) * _calculate_hypervolume(sorted_points, reference, dim + 1)

        # Update previous state
        previous = min(current, previous)

        # Digest current
        sorted_points = sorted_points[1:, :]

    return total
