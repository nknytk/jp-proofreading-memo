import torch
from transformers import BertForMaskedLM
import pytorch_lightning as pl


class BertForMaskedLMPL(pl.LightningModule):
    def __init__(self, model_name, lr=1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.bert_mlm = BertForMaskedLM.from_pretrained(model_name)

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('val_loss', loss)

    def _step(self, batch):
        output = self.bert_mlm(**batch)
        return output.loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
