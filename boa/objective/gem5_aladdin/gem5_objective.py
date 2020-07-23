from typing import List, Dict

import numpy as np

from boa.objective import AbstractObjective

class Gem5Objective(AbstractObjective):

    def __init__(self,
                 input_config: Dict[str, List],
                 outputs: List[str],
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)

        self.input_config = input_config
        self.outputs = outputs

    def get_candidates(self) -> np.ndarray:
        pass

    def get_input_labels(self) -> List[str]:
        return list(self.input_config.keys())

    def get_output_labels(self) -> List[str]:
        return self.outputs

    def __call__(self, candidate: np.ndarray) -> np.ndarray:
        # This calls gem5
        pass
