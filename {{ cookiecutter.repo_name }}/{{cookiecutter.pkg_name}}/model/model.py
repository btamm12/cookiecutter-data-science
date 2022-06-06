import pytorch_lightning as pl
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from {{cookiecutter.pkg_name}}.model.config import Config, Extractor, Transformer, Head
from {{cookiecutter.pkg_name}}.model.blstm_wrapper import BlstmWrapper
from {{cookiecutter.pkg_name}}.model.head_wrapper import HeadWrapper
from {{cookiecutter.pkg_name}}.model.transformer_wrapper import TransformerWrapper
from {{cookiecutter.pkg_name}}.model.wav2vec2_wrapper import Wav2Vec2Wrapper


class Model(pl.LightningModule):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Needed to configure learning rate.
        # See: training_step()
        self.automatic_optimization = False

        # Feature extractor.
        self.extractor = self._construct_extractor(config.extractor)

        # Intermediate transformer.
        self.transformer = self._construct_transformer(config.transformer)

        # Regression head.
        self.head = self._construct_head(config.head)

        self.save_hyperparameters()

    def forward(self, features: Tensor):

        # Extractor.
        # Note: this is always None in the paper.
        if self.extractor is not None:
            features = self.extractor(features)

        # Transformer.
        # Note: this is either None or Bi-LSTM in the paper.
        if self.transformer is not None:
            features = self.transformer(features)

        # Head.
        # Note: this is always PoolAttFF in the paper.
        features = self.head(features)

        # Squeeze.
        features = features.squeeze()
        return features

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.config.train_config.base_lr,
        )
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer=optimizer,
            base_lr=self.config.train_config.base_lr,
            max_lr=self.config.train_config.max_lr,
            cycle_momentum=False,
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, }

    def training_step(self, train_batch, batch_idx):
        features, labels = train_batch

        # Source: https://pytorch-lightning.readthedocs.io/en/latest/common/optimization.html

        # Zero training gradients
        opt = self.optimizers()
        opt.zero_grad()

        # forward + backward + optimize
        out = self.forward(features)
        loss = F.mse_loss(out, labels)
        self.manual_backward(loss)
        opt.step()

        # update lr
        sch = self.lr_schedulers()
        sch.step()

        # self.log('train_loss', loss, on_step=False, on_epoch=True) # maybe this slows down code
        return loss

    def validation_step(self, val_batch, batch_idx):
        features, labels = val_batch
        out = self.forward(features)
        loss = F.mse_loss(out, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

    def _construct_extractor(self, extractor: Extractor) -> nn.Module:
        msg = "Constructing extractor...\n"
        if extractor == Extractor.NONE:
            msg += "> No extractor constructed: not using extractor."
            result = None
        if extractor == Extractor.XLSR:
            msg += "> Finetuning wav2vec2 model."
            result = Wav2Vec2Wrapper()
        print(msg)
        return result

    def _construct_transformer(self, transformer: Transformer) -> nn.Module:
        msg = "Constructing transformer...\n"
        if transformer == Transformer.NONE:
            msg += "> No transformer constructed: not using transformer."
            result = None
        if transformer == Transformer.BLSTM:
            msg += "> Using BLSTM."
            result = BlstmWrapper(self.config)
        if transformer == Transformer.TRANSFORMER:
            msg += "> Using transformer."
            result = TransformerWrapper(self.config)
        print(msg)
        return result

    def _construct_head(self, head: str) -> nn.Module:
        msg = "Constructing head...\n"
        if head == Head.POOLATTFF:
            msg += "> Using PoolAttFF head."
            result = HeadWrapper(self.config)
        print(msg)
        return result
