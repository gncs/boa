import glob
import os
from typing import List, Sequence, Dict

import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import boa.acquisition.util as util
from boa.optimization.data import FileHandler, Data

from dataset_config import dataset_ingredient, load_dataset
from dataset_config import prepare_gpar_data, prepare_ff_gp_data, prepare_ff_gp_aux_data


objective_labels = ['cycle', 'avg_power', 'total_area']
save_folder = "../plots/notebook_plots/bayesopt_analysis/"


def load_dataset(dataset_path, separator, input_labels, output_labels) -> DataTuple:
    with open(dataset_path) as f:
        df = pd.read_csv(f, sep=separator, dtype=np.float64)

    return DataTuple(df=df[input_labels + output_labels],
                     input_labels=input_labels,
                     output_labels=output_labels)

def load_runs(directory: str) -> List[Data]:
    data_points = []
    for path in glob.glob(os.path.join(directory, '*.json')):
        handler = FileHandler(path=path)
        data_points.append(handler.load())
    return data_points


def calculate_volumes(dfs: Sequence[pd.DataFrame], min_max: pd.DataFrame) -> List[pd.DataFrame]:
    normed_reference = np.array([1, 1, 1])

    volumes_list = []
    for df in dfs:
        points = df[objective_labels]
        normed_points = (points - min_max['min']) / (min_max['max'] - min_max['min'])
        volumes = [(size, util.calculate_hypervolume(points=normed_points.values[:size], reference=normed_reference))
                   for size in range(10, normed_points.shape[0])]
        volumes_list.append(pd.DataFrame(volumes, columns=['size', 'volume']))

    return volumes_list


def summarize(volumes_list):
    frames = pd.concat([d.set_index('size') for d in volumes_list], axis=1)
    return frames.apply(lambda row: pd.Series({'mean': np.mean(row), 'std': np.std(row)}), axis=1).reset_index()


def calculate_statistics(experiments_log_dir, dataset_path, kind):
    experiment_dict = {}

    # Discover all experiment directories in the folder where we are saving them all
    experiment_directories = [os.path.join(experiments_log_dir, item)
                              for item in os.listdir(experiments_log_dir)
                              if os.path.isdir(os.path.join(experiments_log_dir, item))]

    # Load experiment JSON files
    for directory in experiment_directories:
        experiments = load_runs(directory)

        dfs = [pd.DataFrame(data=np.hstack((experiment.input, experiment.output)),
                            columns=experiment.input_labels + experiment.output_labels)

               for experiment in experiments]

        base_name = os.path.basename(directory)
        experiment_dict[base_name] = dfs

    # Load the dataset
    dataset = load_dataset(path=dataset_path, kind=kind)
    min_max = dataset.df[objective_labels].apply(lambda column: pd.Series({
        'min': column.min(),
        'max': column.max()
    })).transpose()

    # Calculate summary statistics for the dataset
    volumes_dict = {key: calculate_volumes(experiment, min_max) for key, experiment in experiment_dict.items()}
    summary_dict = {key: summarize(volume) for key, volume in volumes_dict.items()}

    return experiment_dict, volumes_dict, summary_dict



