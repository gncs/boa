import json
from typing import List

import numpy as np

from .objective.abstract import AbstractObjective
from .objective.util import get_random_choice


class DataError(Exception):
    """Exception whenever an error occurs while saving/loading data"""


class Data(dict):
    SAMPLES_KEY = 'samples'
    LABELS_KEY = 'labels'

    INPUT_KEY = 'input'
    OUTPUT_KEY = 'output'

    def __init__(self, xs: np.ndarray, ys: np.ndarray, x_labels: List[str], y_labels: List[str]):
        super().__init__()

        self.validate_dimensions(xs, x_labels)
        self.validate_dimensions(ys, y_labels)

        self[self.SAMPLES_KEY] = {
            self.INPUT_KEY: xs,
            self.OUTPUT_KEY: ys,
        }

        self[self.LABELS_KEY] = {
            self.INPUT_KEY: x_labels,
            self.OUTPUT_KEY: y_labels,
        }

    @property
    def input(self):
        return self[self.SAMPLES_KEY][self.INPUT_KEY]

    @property
    def output(self):
        return self[self.SAMPLES_KEY][self.OUTPUT_KEY]

    @property
    def input_labels(self):
        return self[self.LABELS_KEY][self.INPUT_KEY]

    @property
    def output_labels(self):
        return self[self.LABELS_KEY][self.OUTPUT_KEY]

    @staticmethod
    def validate_dimensions(values: np.ndarray, labels: List[str]) -> None:
        dim_values = values.shape[1]
        dim_labels = len(labels)
        if dim_values != dim_labels:
            raise DataError('Dimension of labels ({dim_labels}) does not match that of data ({dim_values})'.format(
                dim_labels=dim_labels,
                dim_values=dim_values))

    @classmethod
    def from_json(cls, d: dict):
        return cls(
            xs=np.array(d[cls.SAMPLES_KEY][cls.INPUT_KEY]).T,
            ys=np.array(d[cls.SAMPLES_KEY][cls.OUTPUT_KEY]).T,
            x_labels=d[cls.LABELS_KEY][cls.INPUT_KEY],
            y_labels=d[cls.LABELS_KEY][cls.OUTPUT_KEY],
        )

    def to_json(self):
        json_dict = dict()

        json_dict[self.SAMPLES_KEY] = {key: value.T.tolist() for key, value in self[self.SAMPLES_KEY].items()}
        json_dict[self.LABELS_KEY] = self[self.LABELS_KEY]

        return json_dict


class FileHandler:

    def __init__(self, path: str) -> None:
        self.path = path

    def load(self) -> Data:
        with open(self.path) as f:
            return Data.from_json(json.load(f))

    def save(self, data: Data) -> None:
        with open(self.path, mode='w') as f:
            json.dump(data.to_json(), f, indent=4)


def get_init_data(config, objective: AbstractObjective, handler: FileHandler) -> Data:
    try:
        mode = config['init']
        if mode == 'random':
            candidates = objective.get_candidates()
            subset = get_random_choice(candidates, size=int(config['size']), seed=int(config['seed']))
            inputs, outputs = objective.evaluate_batch(subset)

            input_labels = objective.get_input_labels()
            output_labels = objective.get_output_labels()

            data = Data(xs=inputs, ys=outputs, x_labels=input_labels, y_labels=output_labels)
            handler.save(data)

            return data

        elif mode == 'load':
            return handler.load()
        else:
            raise DataError("Unknown mode '" + str(mode) + "'")

    except ValueError as e:
        raise DataError(e)

    except KeyError as e:
        raise DataError("Cannot initialize data: " + str(e))
