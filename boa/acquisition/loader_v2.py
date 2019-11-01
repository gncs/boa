from .abstract_v2 import AbstractAcquisition
from .smsego_v2 import SMSEGO

_acquisitions = {
    'sms-ego': SMSEGO,
}


class AcquisitionLoaderError(Exception):
    def __init__(self, message: str):
        super().__init__("Cannot load acquisition function: " + message)


def load_acquisition(config) -> AbstractAcquisition:
    try:
        return _acquisitions[config['name']](**config.get('parameters', {}))
    except KeyError as e:
        raise AcquisitionLoaderError("Key " + str(e) + " not found")
    except Exception as e:
        raise AcquisitionLoaderError(str(e))