import json
import sys
import torch
from model import AlbertPL

with open('preprocess_config.json') as fp:
    max_length = json.load(fp)['max_length']

model = AlbertPL.load_from_checkpoint(sys.argv[1])
sample_data = {
    'input_ids': [0] * max_length,
    'attention_mask': [1] * max_length,
    'token_type_ids': [0] * max_length
}
sample_input = (
    torch.tensor([sample_data['input_ids']]),
    torch.tensor([sample_data['attention_mask']]),
    torch.tensor([sample_data['token_type_ids']])
)
model.to_onnx('./onnx/sample.onnx', sample_input, export_params=True)
