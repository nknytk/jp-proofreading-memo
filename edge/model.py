import torch
from transformers import AlbertConfig, AlbertForMaskedLM
import pytorch_lightning as pl


class AlbertPL(pl.LightningModule):
    def __init__(self, model_path: str, lr=1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = AlbertForMaskedLM.from_pretrained(model_path)

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('val_loss', loss)

    def _step(self, batch):
        output = self.model(**batch)
        return output.loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs).logits.argmax(-1)
