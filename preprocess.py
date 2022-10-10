from copy import deepcopy
import json
import random
import os
import re
import unicodedata
from transformers import BertJapaneseTokenizer

MAX_LENGTH = 128
BERT_MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = BertJapaneseTokenizer.from_pretrained(BERT_MODEL_NAME)


def prepare_data():
    train_val_data = load_data('data/jwtd_v2.0/train.jsonl')
    add_correct_samples(train_val_data)
    random.shuffle(train_val_data)
    train_data = train_val_data[:int(len(train_val_data) * 0.9)]
    val_data = train_val_data[int(len(train_val_data) * 0.9):]
    test_data = load_data('data/jwtd_v2.0/test.jsonl')
    add_correct_samples(test_data)

    save_jsonl(train_data, 'data/direct/train.jsonl')
    save_jsonl(val_data, 'data/direct/val.jsonl')
    save_jsonl(test_data, 'data/direct/test.jsonl')


def encode_input(text: str):
    return tokenizer(text, max_length=MAX_LENGTH, padding='max_length', truncation=True)


def normalize(text: str) -> str:
    text = unicodedata.normalize('NFKC', text).strip()
    return re.sub('\s+', ' ', text)


def load_data(file_path: str):
    encodings = []
    with open(file_path) as fp:
        for row in fp:
            data = json.loads(row)
            encoding = encode_input(normalize(data['pre_text']))
            encoding['labels'] = encode_input(normalize(data['post_text']))['input_ids']
            encodings.append(encoding.data)
    return encodings


def add_correct_samples(encodings, r=0.1):
    """ 正しい入力に対しては校正が不要であることを示す教師データを作る """
    random.shuffle(encodings)
    correct_examples = []
    for encoding in encodings[:int(len(encodings) * r)]:
        correct_example = deepcopy(encoding)
        correct_example['input_ids'] = correct_example['labels']
        correct_examples.append(correct_example)
    encodings.extend(correct_examples)


def save_jsonl(data: list, file_path: str):
    dir_name = file_path.rsplit('/', 1)[0]
    os.makedirs(dir_name, exist_ok=True)
    with open(file_path, mode='w') as fp:
        for row in data:
            fp.write(json.dumps(row, ensure_ascii=False))
            fp.write('\n')


if __name__ == '__main__':
    prepare_data()
