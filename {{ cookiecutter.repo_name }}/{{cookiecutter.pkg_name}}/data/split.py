from enum import Enum


class Split(Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2

    def __str__(self) -> str:
        return str(self).lower().split(".")[1]


DEV_SPLITS = [Split.TRAIN, Split.VAL]
ALL_SPLITS = [Split.TRAIN, Split.VAL, Split.TEST]
