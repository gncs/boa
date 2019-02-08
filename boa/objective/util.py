import numpy as np


def get_random_choice(a: np.ndarray, size: int, seed: int):
    """
    Return random selection row of size <size> from array <a>.

    :param a: list of items
    :param size: number of items to be selected at random
    :param seed: random seed
    :return: list of selected items
    """

    np.random.seed(seed)
    return a[np.random.choice(len(a), size=size, replace=False, p=None)]
