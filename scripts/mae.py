# GP and GPAR Predictions on FFT Data
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from boa.datasets.labels import label_names_dict


def load_gp_data(path, name):
    with open(path) as f:
        gp_df = pd.read_json(f)
        gp_df['model'] = name
        return gp_df


def load_gpar_data(path, name):
    with open(path) as f:
        gpar_data = pd.read_json(f)

        dfs = []
        for i, row in gpar_data.iterrows():
            df = pd.DataFrame({
                'model': name,
                'index': row['index'],
                'output': row['outputs'],
                'size': row['size'],
                'train_time': row['train_time'],
                'predict_time': row['predict_time'],
                'rmse': row['rmse'],
                'mean_abs_err': row['mean_abs_err'],
                'mean_squ_err': row['mean_squ_err'],
            })
            dfs.append(df)

        return pd.concat(dfs, axis=0, ignore_index=True)


def load_mf_gpar_data(path, name):
    with open(path) as f:
        mf_data = pd.read_json(f)

        dfs = []
        for i, row in mf_data.iterrows():
            df = pd.DataFrame({
                'model': name + '_' + str(row['latent_size']),
                'index': row['index'],
                'output': row['outputs'],
                'size': row['size'],
                'train_time': row['train_time'],
                'predict_time': row['predict_time'],
                'rmse': row['rmse'],
                'mean_abs_err': row['mean_abs_err'],
                'mean_squ_err': row['mean_squ_err'],
            })
            dfs.append(df)

        return pd.concat(dfs, axis=0, ignore_index=True)


def plot_stats(aggregate, stat: str):
    line_style_dict = {
        'gp': 'solid',
        'gpar': 'dashed',
        'gp_aux': 'dotted',
        'mf_5': 'dashdot',
    }

    model_to_label = {
        'gp': 'GP',
        'gpar': 'GPAR',
        'gp_aux': 'GPAR-aux',
        'mf_5': 'MF-GPAR (5)',
    }
    """
    line_style_dict = {
        'mf_2': 'solid',
        'mf_5': 'dashed',
        'mf_10': 'dotted',
    }

    model_to_label = {
        'mf_2': 'MF-GPAR (2)',
        'mf_5': 'MF-GPAR (5)',
        'mf_10': 'MF-GPAR (10)',
    }
    """

    outputs = ['avg_power', 'cycle', 'total_area']

    fig, axes = plt.subplots(nrows=1, ncols=len(outputs), figsize=(4 * len(outputs), 4))

    for output, group in aggregate.groupby('output'):
        if output not in outputs:
            continue
        ax = axes[outputs.index(output)]

        for model, subgroup in group.groupby('model'):
            if model not in line_style_dict.keys():
                continue

            d = subgroup.sort_values('size')
            ax.plot(d['size'],
                    d[(stat, 'mean')],
                    label=model_to_label[model],
                    c='black',
                    linestyle=line_style_dict[model],
                    zorder=-1)
            ax.scatter(d['size'], d[(stat, 'mean')], s=30, c='white', edgecolors='black', zorder=1, label=None)
            ax.errorbar(d['size'],
                        d[(stat, 'mean')],
                        yerr=d[(stat, 'std')],
                        ecolor='black',
                        fmt='None',
                        capsize=3,
                        zorder=-1,
                        label=None)

            ax.set_ylabel('MAE ' + label_names_dict[output], fontsize=12)
            ax.set_xlabel('Dataset Size', fontsize=12)

        ax.legend()

    fig.tight_layout()
    return fig


def main(args):
    gp_file = 'gp_runs.json'
    gp_aux_file = 'gp_aux_runs.json'
    gpar_file = 'gpar_runs.json'
    mf_file = 'mf_gpar_runs.json'

    gp_df = load_gp_data(os.path.join(args.dir, gp_file), name='gp')
    gp_aux_df = load_gp_data(os.path.join(args.dir, gp_aux_file), name='gp_aux')
    gpar_df = load_gpar_data(os.path.join(args.dir, gpar_file), name='gpar')
    mf_df = load_mf_gpar_data(os.path.join(args.dir, mf_file), name='mf')

    agg = pd.concat([gp_df, gp_aux_df, gpar_df, mf_df], sort=False, axis=0, ignore_index=True)

    stats = agg.groupby(['model', 'output', 'size']).aggregate({'mean_abs_err': [np.mean, np.std]}).reset_index()

    fig = plot_stats(stats, stat='mean_abs_err')
    fig.subplots_adjust(wspace=0.5)
    fig.show()

    timings = agg.groupby(['model', 'output', 'size']).aggregate({
        'train_time': [np.mean, np.std],
        'predict_time': [np.mean, np.std],
    }).reset_index()

    print(timings[timings['output'] == 'avg_power'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='directory containing experiments', required=True)

    main(parser.parse_args())
