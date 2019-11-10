# GEMM Dataset
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from labels.gemm import input_labels, output_labels


# Generate histograms summarizing data
def generate_histograms(data: pd.DataFrame, name: str):
    num_plots = len(data.columns)
    num_cols = 4
    num_rows = num_plots // num_cols + bool(num_plots % num_cols)

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols * 4, num_rows * 4))

    for column, ax in zip(data, axes.flatten()):
        ax.hist(data[column], bins=7, density=False, edgecolor='black')
        ax.set_xlabel(column)

    fig.tight_layout()
    fig.savefig(name + '.pdf')
    fig.show()


def main(args):
    with open(args.dataset) as f:
        df = pd.read_csv(f, sep='\t', dtype=np.float64)

    print('Number of rows:', len(df))

    duplicated_rows = df.duplicated(keep=False)
    print('Duplicated rows:', sum(duplicated_rows))

    all_the_same = df.apply(func=lambda c: all(c[0] == c), axis=0)
    print('Only one value in columns:', all_the_same[all_the_same].index.tolist())

    df = df[input_labels + output_labels]

    print('Final inputs:', input_labels)
    print('Final outputs:', output_labels)

    generate_histograms(df[input_labels], name='gemm_inputs')
    generate_histograms(df[output_labels], name='gemm_outputs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='file containing GEMM dataset', required=True)

    main(parser.parse_args())
