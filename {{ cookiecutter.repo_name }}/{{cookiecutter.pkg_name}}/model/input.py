from enum import Enum


class Input(Enum):
    AUDIO = 0
    MFCC = 1
    XLSR = 2

    def __str__(self) -> str:
        return str(self).lower().split(".")[1]


ALL_INPUTS = [Input.AUDIO, Input.MFCC, Input.XLSR]
