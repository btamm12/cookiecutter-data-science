from enum import Enum
from typing import List

from {{cookiecutter.pkg_name}}.model.input import Input
from {{cookiecutter.pkg_name}}.model.label import Label, NORM_MOS

class Extractor(Enum):
    NONE = 0
    XLSR = 1

class Transformer(Enum):
    NONE = 0
    BLSTM = 1
    TRANSFORMER = 2

class Head(Enum):
    POOLATTFF = 0


class TrainConfig():

    max_epochs: int = None
    batch_size: int = None
    base_lr: float = None
    max_lr: float = None

    def __init__(
        self,
        max_epochs: int,
        batch_size: int,
        base_lr: float,
        max_lr: float,
    ) -> None:
        assert max_epochs > 0
        assert batch_size > 0
        assert base_lr > 0
        assert max_lr > 0

        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.base_lr = base_lr
        self.max_lr = max_lr


class Config():

    name: str = None
    input: Input = None
    extractor: Extractor = None
    transformer: Transformer = None
    head: Head = None
    train_config: TrainConfig = None
    label: Label = None


    # Dimensions.
    feat_seq_len: int = None
    dim_input: int = None
    dim_extractor: int = None
    dim_transformer: int = None
    dim_head_in: int = None
    dim_head_out: int = None
    dropout: float = None

    def __init__(
        self,
        name: str,
        input: Input,
        extractor: Extractor,
        transformer: Transformer,
        head: Head,
        feat_seq_len: int,
        dim_transformer: int = None,
        dropout: float = 0.1,
        train_config: TrainConfig = None,
        label: Label = None,
    ):

        # Check valid parameters.
        if extractor == Extractor.NONE:
            msg = "Extractor NONE is not supported for Input AUDIO."
            assert input != Input.AUDIO, msg
        if extractor == Extractor.XLSR:
            msg = "Extractor XLSR must be used with Input AUDIO."
            assert input == Input.AUDIO, msg
        if transformer == Transformer.BLSTM:
            msg = "Must specify dim_transformer."
            assert dim_transformer is not None, msg
            msg = "dim_transformer must be positive."
            assert dim_transformer > 0, msg
        msg = "feat_seq_len must be positive."
        assert feat_seq_len > 0, msg
        msg = "dropout must be in [0,1)."
        assert dropout >= 0 and dropout < 1, msg

        # Save parameters.
        self.name = name
        self.input = input
        self.extractor = extractor
        self.transformer = transformer
        self.head = head
        self.feat_seq_len = feat_seq_len
        self.dim_transformer = dim_transformer
        self.dropout = dropout
        self.train_config = train_config
        self.label = label

        # INPUT   EXTRACTOR       TRANSFORMER             HEAD         OUTPUT
        #
        # audio -> {XLS-R} -> {BiLSTM/Transformer} -> {PoolAttFF} -> prediction
        # mfcc  ------------> {BiLSTM/Transformer} -> {PoolAttFF} -> prediction
        # xlsr  ------------> {BiLSTM/Transformer} -> {PoolAttFF} -> prediction
        #
        #    ^             ^                       ^
        # dim_input   dim_extractor         dim_transformer


        # Set model parameters.
        if input == Input.AUDIO:
            self.dim_input = 1
        elif input == Input.MFCC:
            self.dim_input = 40
        elif input == Input.XLSR:
            self.dim_input = 1024
        else:
            raise Exception("Unknown input.")

        if extractor == Extractor.NONE:
            self.dim_extractor = self.dim_input
        elif extractor == Extractor.XLSR:
            self.dim_extractor = 1024
        else:
            raise Exception("Unknown extractor.")

        if transformer == Transformer.NONE:
            self.dim_transformer = self.dim_extractor
        elif transformer == Transformer.BLSTM:
            self.dim_transformer = dim_transformer
        elif transformer == Transformer.TRANSFORMER:
            self.dim_transformer = self.dim_extractor
        else:
            raise Exception("Unknown transformer.")

        if head == Head.POOLATTFF:
            self.dim_head_in = self.dim_transformer # * self.feat_seq_len
            self.dim_head_out = 1


train_args = TrainConfig(
    max_epochs=50,
    batch_size=64,
    base_lr=1e-3,
    max_lr=1e-2,
)


MFCC_CONFIG = Config(
    "MFCC_CONFIG",
    Input.MFCC,
    Extractor.NONE, # No extractor needed for the input features "MFCC".
    Transformer.NONE, # No Bi-LSTM or Transformer module
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    train_config=train_args,
    label=NORM_MOS,
)

XLSR_CONFIG = Config(
    "XLSR_CONFIG",
    Input.XLSR,
    Extractor.NONE, # No extractor needed for the input features "XLSR".
    Transformer.NONE, # No Bi-LSTM or Transformer module
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    train_config=train_args,
    label=NORM_MOS,
)

MFCC_BLSTM_CONFIG = Config(
    "MFCC_BLSTM_CONFIG",
    Input.MFCC,
    Extractor.NONE, # No extractor needed for the input features "MFCC".
    Transformer.BLSTM, # Use Bi-LSTM module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=64, # 32 in each direction
    train_config=train_args,
    label=NORM_MOS,
)

XLSR_BLSTM_CONFIG = Config(
    "XLSR_BLSTM_CONFIG",
    Input.XLSR,
    Extractor.NONE, # No extractor needed for the input features "XLSR".
    Transformer.BLSTM, # Use Bi-LSTM module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=64, # 32 in each direction
    train_config=train_args,
    label=NORM_MOS,
)


MFCC_TRANS_CONFIG = Config(
    "MFCC_TRANS_CONFIG",
    Input.MFCC,
    Extractor.NONE, # No extractor needed for the input features "MFCC".
    Transformer.TRANSFORMER, # Use Transformer module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    train_config=train_args,
    label=NORM_MOS,
)

XLSR_TRANS_CONFIG = Config(
    "XLSR_TRANS_CONFIG",
    Input.XLSR,
    Extractor.NONE, # No extractor needed for the input features "XLSR".
    Transformer.TRANSFORMER, # Use Transformer module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    train_config=train_args,
    label=NORM_MOS,
)


# 4 models used by paper.
ALL_CONFIGS: List[Config] = [
    MFCC_CONFIG,
    XLSR_CONFIG,
    MFCC_BLSTM_CONFIG,
    XLSR_BLSTM_CONFIG,
]
