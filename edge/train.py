import json
import sys
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import BatchEncoding
from model import AlbertPL

BATCH_SIZE = 32
MAX_EPOCH = 10
MODEL_SIZE = 'medium'
accumulate_grad_batches = 1


def train():
    #train_data = load_data('data/pretrain/train.jsonl')
    #val_data = load_data('data/pretrain/val.jsonl')
    train_data = load_data('data/split/train_replacement.jsonl')
    val_data = load_data('data/split/val_replacement.jsonl')
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)

    save_model_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_weights_only=True,
        dirpath=f'models/{MODEL_SIZE}_trained'
    )
    early_stopping_checkpoint = pl.callbacks.early_stopping.EarlyStopping(monitor='val_loss', mode='min', patience=1)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=MAX_EPOCH,
        callbacks=[save_model_checkpoint, early_stopping_checkpoint],
        accumulate_grad_batches=accumulate_grad_batches
    )
    model = AlbertPL.load_from_checkpoint(sys.argv[1]) if len(sys.argv) > 1 else AlbertPL(f'models/{MODEL_SIZE}')
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
