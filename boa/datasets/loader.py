from collections import namedtuple
from typing import Tuple, Sequence

import numpy as np
import pandas as pd

import boa.datasets.labels.fft as fft
import boa.datasets.labels.stencil3d as stencil3d
import boa.datasets.labels.gemm as gemm
import boa.datasets.labels.smaug as smaug

DataTuple = namedtuple('DataTuple', field_names=['df', 'input_labels', 'output_labels'])


def load_labels(kind: str) -> Tuple[Sequence[str], Sequence[str]]:
    if kind == 'fft':
        return fft.input_labels, fft.output_labels

    elif kind == 'stencil3d':
        return stencil3d.input_labels, stencil3d.output_labels

    elif kind == 'gemm':
        return gemm.input_labels, gemm.output_labels

    elif kind == 'smaug':
        return smaug.input_labels, smaug.output_labels

    raise RuntimeError('Unknown label', kind)


def load_dataset(path: str, kind: str) -> DataTuple:
    if kind == 'fft':
        with open(path) as f:
            df = pd.read_csv(f, sep=' ', dtype=np.float64)

        return DataTuple(df=df[fft.input_labels + fft.output_labels],
                         input_labels=fft.input_labels,
                         output_labels=fft.output_labels)

    elif kind == 'stencil3d':
        with open(path) as f:
            df = pd.read_csv(f, sep='\t', dtype=np.float64)

        return DataTuple(df=df[stencil3d.input_labels + stencil3d.output_labels],
                         input_labels=stencil3d.input_labels,
                         output_labels=stencil3d.output_labels)

    elif kind == 'gemm':
        with open(path) as f:
            df = pd.read_csv(f, sep='\t', dtype=np.float64)

        return DataTuple(df=df[gemm.input_labels + gemm.output_labels],
                         input_labels=gemm.input_labels,
                         output_labels=gemm.output_labels)

    elif kind == 'smaug':
        with open(path) as f:
            df = pd.read_csv(f, sep=',', dtype=np.float64)

        return DataTuple(df=df[smaug.input_labels + smaug.output_labels],
                         input_labels=smaug.input_labels,
                         output_labels=smaug.output_labels)

    raise RuntimeError('Unknown label', kind)
