# FFT Data Set
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .labels.fft import input_labels, output_labels


def generate_histograms(data: pd.DataFrame):
    num_plots = len(data.columns)
    num_cols = 4
    num_rows = num_plots // num_cols + bool(num_plots % num_cols)

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, num_rows * 4))

    for column, ax in zip(data, axes.flatten()):
        ax.hist(data[column], bins=7, density=False, color='white', edgecolor='black', hatch='//')
        ax.set_xlabel(column)

    fig.tight_layout()
    fig.show()


def main(dataset_path: str):
    with open(dataset_path) as f:
        data_set = pd.read_csv(f, sep=' ', dtype=np.float64)

    # Check for duplicates in input
    if data_set.duplicated(subset=input_labels).any():
        raise ValueError('Duplicates present')

    # Generate histograms summarizing data
    generate_histograms(data_set[input_labels])
    generate_histograms(data_set[output_labels])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='file containing stencil 3D dataset', required=True)
    args = parser.parse_args()

    main(dataset_path=args.dataset)
