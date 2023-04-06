import torch
from transformers import AlbertConfig, AlbertForMaskedLM, BertConfig, BertForMaskedLM
import pytorch_lightning as pl


class AlbertPL(pl.LightningModule):
    def __init__(self, lr=1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = AlbertForMaskedLM(AlbertConfig(vocab_size=11596, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072))

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
