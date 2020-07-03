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
            'avg_power',
            'idle_fu_cycles',
            'avg_fu_power',
            'avg_fu_dynamic_power',
            'avg_fu_leakage_power',
            'avg_mem_power',
            'avg_mem_dynamic_power',
            'avg_mem_leakage_power',
            'fu_area',
            'mem_area',
            'num_double_precision_fp_multipliers',
            'num_double_precision_fp_adders',
            # 'num_trigonometric_units',
            # 'num_bit-wise_operators_32',
            # 'num_shifters_32',
            'num_registers_32',
            'cycle',
            'total_area',
        ]
        targets = ('avg_power', 'cycle', 'total_area')

        input_transforms = {
            'cache_size': "log",
            'cache_assoc': "log",
            'cache_line_sz': "log",
            'cache_queue_size': "log",
            'tlb_page_size': "log",
            'tlb_max_outstanding_walks': "log",
            'tlb_assoc': "log",
            'l2cache_size': "log",
        }
        output_transforms = {
            'avg_mem_power': "log",
            'avg_mem_dynamic_power': "log",
            'avg_fu_dynamic_power': "log",
            'avg_fu_power': "log",
            'avg_fu_leakage_power': "log",
            'total_area': "log",
            'fu_area': "log",
            'mem_area': "log",
            'avg_power': "log"
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
            'cycle_time': "log",
            'cache_size': "log",
            'cache_assoc': "log",
            'cache_hit_latency': "log",
            'l2cache_size': "log",
            'tlb_hit_latency': "log",
            'tlb_entries': "log"
        }
        output_transforms = {
            'cycle': "log",
            'avg_power': "log",
            'avg_mem_power': "log",
            'avg_mem_dynamic_power': "log",
        }

    elif name == "stencil3d_v2":
        targets = ('avg_power', 'cycle', 'total_area')
        dataset_path = f"{dataset_base_path}/stencil3d_dataset_v2.csv"

        separator = ","

        input_labels = [
            "cycle_time",
            "pipelining",
            "cache_size",
            "cache_assoc",
            "cache_hit_latency",
            "cache_line_sz",
            "cache_queue_size",
            "tlb_hit_latency",
            "tlb_miss_latency",
            "tlb_entries",
            "tlb_max_outstanding_walks",
            "tlb_assoc",
            "tlb_bandwidth",
            "l2cache_size",
            "pipelined_dma",
            "ignore_cache_flush",
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
            #             'num_sp_multiplier',
            #             'num_sp_adder',
            #             'num_dp_multiplier',
            #             'num_dp_adder',
            #             'num_trig_unit',
            #             'num_multiplier',
            #             'num_adder',
            #             'num_bit_wise',
            #             'num_shifter',
            'num_register'
        ]

        input_transforms = {
            'cycle_time': "log",
            'cache_size': "log",
            'cache_assoc': "log",
            'cache_hit_latency': "log",
            'l2cache_size': "log",
            'tlb_hit_latency': "log",
            'tlb_entries': "log"
        }
        output_transforms = {
            'cycle': "log",
            'avg_power': "log",
            'avg_mem_power': "log",
            'avg_mem_dynamic_power': "log",
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
            'cycle_time': "log",
            'cache_size': "log",
            'cache_assoc': "log",
            'cache_hit_latency': "log",
            'l2cache_size': "log",
            'tlb_hit_latency': "log",
            'tlb_entries': "log"
        }
        output_transforms = {
            'cycle': "log",
            'avg_power': "log",
            'fu_power': "log",
            'avg_fu_dynamic_power': "log",
            'avg_mem_power': "log",
            'avg_mem_dynamic_power': "log",
            'total_area': "log",
            'mem_area': "log",
            'fu_area': "log"
        }

    elif name == "gemm_v2":
        targets = ('avg_power', 'cycle', 'total_area')
        dataset_path = f"{dataset_base_path}/gemm_dataset_v2.csv"

        separator = ","

        input_labels = [
            "cycle_time",
            "pipelining",
            "cache_size",
            "cache_assoc",
            "cache_hit_latency",
            "cache_line_sz",
            "cache_queue_size",
            "tlb_hit_latency",
            "tlb_miss_latency",
            "tlb_entries",
            "tlb_max_outstanding_walks",
            "tlb_assoc",
            "tlb_bandwidth",
            "l2cache_size",
            "pipelined_dma",
            "ignore_cache_flush",
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
            #             'num_sp_multiplier',
            #             'num_sp_adder',
            #             'num_dp_multiplier',
            #             'num_dp_adder',
            #             'num_trig_unit',
            #             'num_multiplier',
            #             'num_adder',
            #             'num_bit_wise',
            #             'num_shifter',
            'num_register'
        ]

        input_transforms = {
            'cycle_time': "log",
            'cache_size': "log",
            'cache_assoc': "log",
            'cache_hit_latency': "log",
            'l2cache_size': "log",
            'tlb_hit_latency': "log",
            'tlb_entries': "log"
        }
        output_transforms = {
            'cycle': "log",
            'avg_power': "log",
            'fu_power': "log",
            'avg_fu_dynamic_power': "log",
            'avg_mem_power': "log",
            'avg_mem_dynamic_power': "log",
            'total_area': "log",
            'mem_area': "log",
            'fu_area': "log"
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
            'num_threads': "log",
            'l2_assoc': "log",
            'accel_clock_time': "log",
            'num_accels': "log",
            'l2_size': "log",
        }
        output_transforms = {
            'total_time': "log",
            'total_energy': "log",
            'fu_energy': "log",
            'spad_energy': "log",
            'llc_leakage_energy': "log",
            'llc_dynamic_energy': "log",
            'total_area': "log",
            'fu_area': "log",
            'mem_area': "log",
            'num_multiplier': "log",
            'num_adder': "log",
            'num_bit_wise': "log",
            'num_shifter': "log",
            'num_register': "log"
        }

    elif name == "intel_dataset_1":
        targets = ('objective_0', 'objective_1',)
        dataset_path = f"{dataset_base_path}/smaug_dataset.csv"

        separator = ","

        input_labels = [f"knob_{i}" for i in range(12)]

        output_labels = [f"metric_{i}" for i in range(24)]
        output_labels = output_labels + list(targets)

        # These ones are non-informative dimensions
        output_labels.remove("metric_14")
        output_labels.remove("metric_18")

        input_transforms = {}
        output_transforms = {}

    # Rearrange output labels so that targets are last
    for target in targets:
        output_labels.remove(target)
        output_labels.append(target)


DataTuple = namedtuple('DataTuple', field_names=['df', 'input_labels', 'output_labels'])


@dataset_ingredient.capture
def load_labels(input_labels, output_labels) -> Tuple[Sequence[str], Sequence[str]]:
    return input_labels, output_labels


@dataset_ingredient.capture
def load_dataset(dataset_path, separator, input_labels, output_labels) -> DataTuple:
    with open(dataset_path) as f:
        df = pd.read_csv(f, sep=separator, dtype=np.float64)

    return DataTuple(df=df[input_labels + output_labels], input_labels=input_labels, output_labels=output_labels)
