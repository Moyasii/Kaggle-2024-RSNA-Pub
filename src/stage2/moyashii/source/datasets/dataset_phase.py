from enum import Enum


class DatasetPhase(Enum):
    TRAIN = 1
    VALIDATION = 2
    TEST = 3
    UNKNOWN = 4
