from .utils import DatasetSpec
from .utils import inspect_dataset
from .utils import register_juniper_dataset
from .test import normalize_dataset
from .test import validate_dataset

__all__ = [
    "DatasetSpec",
    "inspect_dataset",
    "normalize_dataset",
    "register_juniper_dataset",
    "validate_dataset",
]

__version__ = "1.0.0"
__author__ = "Cristian Iranzo"
__email__ = "c.iranzo@unizar.es"
