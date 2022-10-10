import json
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import BatchEncoding
from model import BertForMaskedLMPL

BERT_MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'
BATCH_SIZE = 32
MAX_EPOCH = 10


def train():
    train_data = load_data('data/direct/train.jsonl')
    val_data = load_data('data/direct/val.jsonl')
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)

    save_model_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_weights_only=True,
        dirpath='models/direct/'
    )
    early_stopping_checkpoint = pl.callbacks.early_stopping.EarlyStopping(monitor='val_loss', mode='min', patience=2)

    trainer = pl.Trainer(gpus=1, max_epochs=MAX_EPOCH, callbacks=[save_model_checkpoint, early_stopping_checkpoint])
    model = BertForMaskedLMPL(BERT_MODEL_NAME)
    trainer.fit(model, train_dataloader, val_dataloader)
    print(save_model_checkpoint.best_model_path)


def load_data(file_path: str):
    data = []
    with open(file_path) as fp:
        for row in fp:
            sample = BatchEncoding(data=json.loads(row), tensor_type='pt')
            data.append(sample)
    return data


if __name__ == '__main__':
    train()
