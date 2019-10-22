import argparse
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .dataset_loader import load_labels


def average_log_ls(group, input_labels: Sequence[str]):
    log_ls = []

    for i, lengthscales in group['lengthscales'].items():
        log_ls.append(np.log(lengthscales))

    means = np.mean(np.array(log_ls), axis=0)
    return pd.Series(means, index=input_labels)


def plot_ls(log_ls):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    axes = [ax for sub in axes for ax in sub]

    sizes = [25, 50, 100, 150]
    for size, group in log_ls.groupby('size'):
        try:
            ax = axes[sizes.index(size)]
        except ValueError:
            continue

        w = 0.3
        x_pos = np.arange(len(group))
        ax.bar(x_pos - w / 2, group['cycle_time'], width=w, align='center', label='cycle_time')
        ax.bar(x_pos + w / 2, group['cache_size'], width=w, align='center', label='cache_size')

        ax.set_title(f'n = {size}')
        ax.set_xticks([])
        ax.set_xlabel('Outputs', fontsize=12)
        ax.set_ylabel('Log Lengthscale', fontsize=12)

        ax.legend(loc='lower left')

    fig.show()
    fig.savefig('log_ls.pdf')
    plt.close(fig)


def plot_pca(log_ls):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

    w = 0.15
    size_pos_dict = {
        25: -2 * w,
        50: -1 * w,
        100: 0 * w,
        150: 1 * w,
        200: 2 * w,
    }

    sv_dict = {}
    for size, group in log_ls.groupby('size'):
        raw = group.drop(['output', 'size'], axis='columns').values

        pca = PCA()
        pca.fit(raw)
        sv = pca.explained_variance_ratio_

        sv_dict[size] = sv

    for size in sorted(size_pos_dict.keys()):
        x_pos = np.arange(len(sv_dict[size]))
        ax.bar(x_pos + size_pos_dict[size], sv_dict[size], width=w, align='center', label=size)

    ax.set_xticks([])
    ax.legend(loc='upper right')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_ylabel('Explained Variance Ratio', fontsize=12)

    fig.show()
    fig.savefig('pca.pdf')
    plt.close(fig)


def main(args):
    with open(args.file) as f:
        df = pd.read_json(f)

    input_labels, output_labels = load_labels(kind=args.type)

    log_ls = df.groupby(['output', 'size']).apply(lambda g: average_log_ls(g, input_labels=input_labels)).reset_index()

    plot_ls(log_ls)
    plot_pca(log_ls)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='file containing GP runs', required=True)
    parser.add_argument('--type', help='select dataset type', required=True)

    main(parser.parse_args())
