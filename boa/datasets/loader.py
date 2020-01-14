from collections import namedtuple
from typing import Tuple, Sequence

import numpy as np
import pandas as pd

DataTuple = namedtuple('DataTuple', field_names=['df', 'input_labels', 'output_labels'])


def load_labels(kind: str) -> Tuple[Sequence[str], Sequence[str]]:
    if kind == 'fft':
        from boa.datasets.labels.fft import input_labels, output_labels
        return input_labels, output_labels

    elif kind == 'stencil3d':
        from boa.datasets.labels.stencil3d import input_labels, output_labels
        return input_labels, output_labels

    raise RuntimeError('Unknown label', kind)


def load_dataset(path: str, kind: str) -> DataTuple:
    if kind == 'fft':
        from boa.datasets.labels.fft import input_labels, output_labels

        with open(path) as f:
            df = pd.read_csv(f, sep=' ', dtype=np.float64)

        return DataTuple(df=df[input_labels + output_labels], input_labels=input_labels, output_labels=output_labels)

    elif kind == 'stencil3d':
        from boa.datasets.labels import input_labels, output_labels

        with open(path) as f:
            df = pd.read_csv(f, sep='\t', dtype=np.float64)

        return DataTuple(df=df[input_labels + output_labels], input_labels=input_labels, output_labels=output_labels)

    raise RuntimeError('Unknown label', kind)
