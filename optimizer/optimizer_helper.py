import numpy as np


class InvalidParameters(Exception):
    pass


def shift(frontier, mean, std):
    frontier_copy = frontier.copy()
    frontier_copy -= mean[None, :]
    frontier_copy /= std[None, :]
    return frontier_copy


def find_frontier(values):
    frontier_ind = []
    for i in range(0, values.shape[0]):
        dominated = False
        for j in range(0, values.shape[0]):
            if (np.all(np.all(values[i, :] >= values[j, :])) and
                    (not np.all(np.all(values[i, :] == values[j, :])) or i < j)):
                dominated = True

        if not dominated:
            frontier_ind.append(i)

    return values[np.array(frontier_ind), :]


def get_hypervolume(frontier, reference_point):
    return get_volume_recursive(frontier, reference_point, 0)

# This function recursively calculates the hypervolume over the dimensions.
# It is exponential in the number of dimensions.
def get_volume_recursive(frontier, reference_point, dim):
    if dim == frontier.shape[1] - 1:
        return max(reference_point[dim] - min(frontier[:, dim]), 0.0)

    sorted_frontier = np.array(sorted(frontier, key=lambda entry: - entry[dim]))

    accumulator = 0.0
    sweep = reference_point[dim]
    while sorted_frontier.shape[0] > 0:
        accumulator += max(sweep - sorted_frontier[0, dim], 0.0) \
                       * get_volume_recursive(sorted_frontier, reference_point, dim + 1)
        sweep = min(sorted_frontier[0, dim], reference_point[dim])
        sorted_frontier = sorted_frontier[1:, :]

    return accumulator
