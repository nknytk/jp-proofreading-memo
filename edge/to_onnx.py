import json
import sys
import torch
from model import AlbertPL
from preprocess import MAX_LENGTH


model = AlbertPL.load_from_checkpoint(sys.argv[1])
sample_data = {
    'input_ids': [0] * MAX_LENGTH,
    'attention_mask': [1] * MAX_LENGTH,
    'token_type_ids': [0] * MAX_LENGTH
}
sample_input = (
    torch.tensor([sample_data['input_ids']]),
    torch.tensor([sample_data['attention_mask']]),
    torch.tensor([sample_data['token_type_ids']])
)
model.to_onnx('./onnx/sample2.onnx', sample_input, export_params=True)
