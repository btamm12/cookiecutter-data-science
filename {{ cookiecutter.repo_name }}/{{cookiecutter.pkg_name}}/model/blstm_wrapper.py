from torch import Tensor, nn

from {{cookiecutter.pkg_name}}.model.config import Config


class BlstmWrapper(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        self.norm = nn.BatchNorm1d(config.dim_extractor)
        BIDIRECTIONAL = True
        if BIDIRECTIONAL:
            assert config.dim_transformer % 2 == 0
            hidden_size = config.dim_transformer // 2
        else:
            hidden_size = config.dim_extractor
        self.blstm = nn.LSTM(
            input_size=config.dim_extractor,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=config.dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        # Input: (N, L, C)
        #  - N = batch size
        #  - L = sequence length
        #  - C = feature dim

        # Transform from (N, L, C) to (N, C, L) and back.
        x = self.norm(x.permute((0, 2, 1))).permute((0, 2, 1))
        x, _ = self.blstm(x)
        x = self.dropout(x)
        return x
