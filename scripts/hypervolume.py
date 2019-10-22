# Test Calculation of Hypervolume

import matplotlib.pyplot as plt
import numpy as np

from boa.acquisition.util import get_frontier


# Plot frontier, volume, and reference
def build_polygon(pareto_points, reference):
    points = sorted([tuple(x) for x in pareto_points])

    top_left = (points[0][0], reference[1])
    bottom_right = (reference[0], points[-1][1])

    polygon = [top_left]
    for left, right in zip(points[:-1], points[1:]):
        polygon += [left, (right[0], left[1])]

    polygon += [points[-1], bottom_right, tuple(reference)]

    return np.array(polygon)


def plot_pareto(points, reference, ax):
    frontier = get_frontier(points)

    polygon = build_polygon(frontier, reference)
    ax.fill(polygon[:, 0], polygon[:, 1], c='blue', alpha=0.3, zorder=-1, label='Hypervolume')

    ax.scatter(x=points[:, 0], y=points[:, 1], s=30, c='white', edgecolors='black', label='Suboptimal')
    ax.scatter(x=frontier[:, 0], y=frontier[:, 1], s=30, c='black', edgecolors='black', label='Optimal')
    ax.scatter(*reference, s=30, c='red', edgecolors='red', label='Reference')

    ax.legend(loc='upper left', bbox_to_anchor=(1.03, 1.0))

    ax.set_xlabel('$Y_' + str(0) + '$')
    ax.set_ylabel('$Y_' + str(1) + '$')

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)


# Generate sets of random points in 3D
np.random.seed(42)
points = np.random.rand(10, 3)
ref = np.minimum(np.max(points, axis=0) + np.array([0.1, 0.1, 0.1]), np.array([1, 1, 1]))

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
plot_pareto(points=points[:, :2], reference=ref[:2], ax=ax)
fig.show()
