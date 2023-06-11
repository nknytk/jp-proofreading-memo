import json
import torch
from transformers import BatchEncoding
from model import AlbertPL
import predict

jp_corrector = AlbertPL.load_from_checkpoint('models/medium_trained/medium.ckpt')
jp_corrector.model.cpu()


def trim_ids(ids: list):
    """
    for i in range(len(ids) - 1, -1, -1):
        if ids[i] != 0:
            return ids[1:i + 1]
    """
    return [_id for _id in ids if _id != 0]


with open('./data/split/test_replacement.jsonl') as fp:
    row_count = 0
    token_count = 0
    correctly_predicted_rows = 0
    correctly_predicted_tokens = 0
    false_positive = 0
    false_negative = 0
    incorrect_correction = 0

    for row in fp:
        data = json.loads(row)
        trimed_input_ids = trim_ids(data['input_ids'])
        trimed_labels = trim_ids(data['labels'])
        bert_input = BatchEncoding({k: [v] for k, v in data.items()}, tensor_type='pt')
        with torch.no_grad():
            scores = jp_corrector.model(**bert_input).logits
            predicted_ids = scores[0].argmax(-1).tolist()
        trimed_predicted_ids = trim_ids(predicted_ids)

        row_count += 1
        token_count += len(trimed_input_ids)
        if trimed_labels == trimed_predicted_ids:
            correctly_predicted_rows += 1

        for input_id, label_id, predicted_id in zip(trimed_input_ids, trimed_labels, trimed_predicted_ids):
            if label_id == predicted_id:
                correctly_predicted_tokens += 1
            elif input_id == label_id:
                false_positive += 1
            elif input_id == predicted_id:
                false_negative += 1
            else:
                incorrect_correction += 1

    print(f'correct rows: {correctly_predicted_rows} / {row_count} ({100 * correctly_predicted_rows / row_count:.02f}%)')
    print(f'correct tokens: {correctly_predicted_tokens} / {token_count} ({100 * correctly_predicted_tokens / token_count:.02f}%)')
    print(f'false positive tokens: {false_positive} ({100 * false_positive / token_count:.02f}%)')
    print(f'false negative tokens: {false_negative} ({100 * false_negative / token_count:.02f}%)')
    print(f'incorrectly corrected tokens: {incorrect_correction} ({100 * incorrect_correction / token_count:.02f}%)')
