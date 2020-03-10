from collections import namedtuple
from typing import Tuple, Sequence

import numpy as np
import pandas as pd

from sacred import Ingredient
from boa import ROOT_DIR

label_names_dict = {
    'avg_power': 'Avg Power [mW]',
    'total_area': r'Total Area [$\mu$M$^2$]',
    'cycle': 'Cycle',
    "total_energy": "Total Energy",
    "total_time": "Total Time"
}

dataset_ingredient = Ingredient("dataset")


@dataset_ingredient.config
def data_config():
    name = "fft"

    dataset_base_path = f"{ROOT_DIR}/../resources/"

    if name == "fft":
        targets = ('avg_power', 'cycle', 'total_area')
        dataset_path = f"{dataset_base_path}/fft_dataset.csv"

        separator = " "

        input_labels = [
            # NB: cache_bandwith is missing!
            'cycle_time',
            'pipelining',
            'cache_size',
            'cache_assoc',
            'cache_hit_latency',
            'cache_line_sz',
            'cache_queue_size',
            'tlb_hit_latency',
            'tlb_miss_latency',
            'tlb_page_size',
            'tlb_entries',
            'tlb_max_outstanding_walks',
            'tlb_assoc',
            'tlb_bandwidth',
            'l2cache_size',
            'enable_l2',
            'pipelined_dma',
            'ignore_cache_flush',
        ]

        output_labels = [
            'cycle',
            'avg_power',
            'idle_fu_cycles',
            'avg_fu_power',
            'avg_fu_dynamic_power',
            'avg_fu_leakage_power',
            'avg_mem_power',
            'avg_mem_dynamic_power',
            'avg_mem_leakage_power',
            'total_area',
            'fu_area',
            'mem_area',
            'num_double_precision_fp_multipliers',
            'num_double_precision_fp_adders',
            # 'num_trigonometric_units',
            # 'num_bit-wise_operators_32',
            # 'num_shifters_32',
            'num_registers_32',
        ]

        input_transforms = {
                           'cache_size': np.log,
                           'cache_assoc': np.log,
                           'cache_line_sz': np.log,
                           'cache_queue_size': np.log,
                           'tlb_page_size': np.log,
                           'tlb_max_outstanding_walks': np.log,
                           'tlb_assoc': np.log,
                           'l2cache_size': np.log,
                           },
        output_transforms = {
            'avg_mem_power': np.log,
            'avg_mem_dynamic_power': np.log,
            'avg_fu_dynamic_power': np.log,
            'avg_fu_power': np.log,
            'avg_fu_leakage_power': np.log,
            'total_area': np.log,
            'fu_area': np.log,
            'mem_area': np.log,
            'avg_power': np.log
        }

    elif name == "stencil3d":
        targets = ('avg_power', 'cycle', 'total_area')
        dataset_path = f"{dataset_base_path}/stencil3d_dataset.csv"

        separator = "\t"

        input_labels = [
            'cycle_time',
            'cache_size',
            'cache_assoc',
            'cache_hit_latency',
            'tlb_hit_latency',
            'tlb_entries',
            'l2cache_size',
        ]

        output_labels = [
            'cycle',
            'avg_power',
            'fu_power',
            'avg_fu_dynamic_power',
            'avg_fu_leakage_power',
            'avg_mem_power',
            'avg_mem_dynamic_power',
            'avg_mem_leakage_power',
            'total_area',
            'fu_area',
            'mem_area',
            # 'num_sp_multiplier',
            # 'num_sp_adder',
            # 'num_dp_multiplier',
            # 'num_dp_adder',
            # 'num_trig_unit',
            # 'num_multiplier',
            # 'num_adder',
            # 'num_bit_wise',
            # 'num_shifter',
            'num_register',
        ]
        input_transforms = {
                               'cycle_time': np.log,
                               'cache_size': np.log,
                               'cache_assoc': np.log,
                               'cache_hit_latency': np.log,
                               'l2cache_size': np.log,
                               'tlb_hit_latency': np.log,
                               'tlb_entries': np.log
                           },
        output_transforms = {
            'cycle': np.log,
            'avg_power': np.log,
            'avg_mem_power': np.log,
            'avg_mem_dynamic_power': np.log,

        }

    elif name == "gemm":
        targets = ('avg_power', 'cycle', 'total_area')
        dataset_path = f"{dataset_base_path}/gemm_dataset.csv"

        separator = "\t"

        input_labels = [
            'cycle_time',
            'cache_size',
            'cache_assoc',
            'cache_hit_latency',
            'tlb_hit_latency',
            'tlb_entries',
            'l2cache_size',
        ]

        output_labels = [
            'cycle',
            'avg_power',
            'fu_power',
            'avg_fu_dynamic_power',
            'avg_fu_leakage_power',
            'avg_mem_power',
            'avg_mem_dynamic_power',
            'avg_mem_leakage_power',
            'total_area',
            'fu_area',
            'mem_area',
            # 'num_sp_multiplier',
            # 'num_sp_adder',
            # 'num_dp_multiplier',
            # 'num_dp_adder',
            # 'num_trig_unit',
            # 'num_multiplier',
            # 'num_adder',
            # 'num_bit_wise',
            # 'num_shifter',
            'num_register',
        ]

        input_transforms = {
                               'cycle_time': np.log,
                               'cache_size': np.log,
                               'cache_assoc': np.log,
                               'cache_hit_latency': np.log,
                               'l2cache_size': np.log,
                               'tlb_hit_latency': np.log,
                               'tlb_entries': np.log
                           },
        output_transforms = {
            'cycle': np.log,
            'avg_power': np.log,
            'fu_power': np.log,
            'avg_fu_dynamic_power': np.log,
            'avg_mem_power': np.log,
            'avg_mem_dynamic_power': np.log,
            'total_area': np.log,
            'mem_area': np.log,
            'fu_area': np.log
        }

    elif name == "smaug":
        targets = ('total_energy', 'total_time', 'total_area')
        dataset_path = f"{dataset_base_path}/smaug_dataset.csv"

        separator = ","

        input_labels = [
            "num_threads",
            "l2_assoc",
            "accel_clock_time",
            "dma",
            "acp",
            "num_accels",
            "l2_size",
        ]

        output_labels = [
            "total_time",
            "total_accel_time",
            "total_energy",
            "fu_energy",
            "spad_energy",
            "llc_leakage_energy",
            "llc_dynamic_energy",
            "total_area",
            "fu_area",
            "mem_area",
            # "num_sp_multiplier",
            # "num_sp_adder",
            # "num_dp_multiplier",
            # "num_dp_adder",
            # "num_trig_unit",
            "num_multiplier",
            "num_adder",
            "num_bit_wise",
            "num_shifter",
            "num_register"
        ]

        input_transforms = {
                               'num_threads': np.log,
                               'l2_assoc': np.log,
                               'accel_clock_time': np.log,
                               'num_accels': np.log,
                               'l2_size': np.log,
                           },
        output_transforms = {
            'total_time': np.log,
            'total_energy': np.log,
            'fu_energy': np.log,
            'spad_energy': np.log,
            'llc_leakage_energy': np.log,
            'llc_dynamic_energy': np.log,
            'total_area': np.log,
            'fu_area': np.log,
            'mem_area': np.log,
            'num_multiplier': np.log,
            'num_adder': np.log,
            'num_bit_wise': np.log,
            'num_shifter': np.log,
            'num_register': np.log
        }


DataTuple = namedtuple('DataTuple', field_names=['df', 'input_labels', 'output_labels'])


@dataset_ingredient.capture
def load_labels(input_labels, output_labels) -> Tuple[Sequence[str], Sequence[str]]:
    return input_labels, output_labels


@dataset_ingredient.capture
def load_dataset(dataset_path, separator, input_labels, output_labels) -> DataTuple:
    with open(dataset_path) as f:
        df = pd.read_csv(f, sep=separator, dtype=np.float64)

    return DataTuple(df=df[input_labels + output_labels],
                     input_labels=input_labels,
                     output_labels=output_labels)


@dataset_ingredient.capture
def prepare_ff_gp_data(data, input_labels, output_labels):
    return data.df, input_labels, output_labels


@dataset_ingredient.capture
def prepare_ff_gp_aux_data(data, targets, input_labels, output_labels):
    inputs = input_labels + output_labels

    for x in targets:
        inputs.remove(x)

    return data.df, inputs, targets


@dataset_ingredient.capture
def prepare_gpar_data(data, targets, input_labels, output_labels):
    output_labels = output_labels.copy()

    for target in targets:
        output_labels.remove(target)
        output_labels.append(target)

    return data.df, input_labels.copy(), output_labels
