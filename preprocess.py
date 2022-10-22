from copy import deepcopy
from difflib import SequenceMatcher
import json
import random
import os
import re
import unicodedata
from transformers import BertJapaneseTokenizer

MAX_LENGTH = 128
R = 0.2
N_VAL = 10000
BERT_MODEL_NAME = 'cl-tohoku/bert-base-japanese-v2'
tokenizer = BertJapaneseTokenizer.from_pretrained(BERT_MODEL_NAME, mecab_kwargs={'mecab_dic': 'unidic_lite'})


def prepare_data():
    test_insertion, test_replacement = load_data('data/jwtd_v2.0/test.jsonl')

    save_jsonl(test_replacement, 'data/split/test_replacement.jsonl')
    save_jsonl(test_insertion, 'data/split/test_insertion.jsonl')

    train_val_insertion, train_val_replacement = load_data('data/jwtd_v2.0/train.jsonl')
    random.shuffle(train_val_insertion)
    train_insertion = train_val_insertion[N_VAL:]
    val_insertion = train_val_insertion[:N_VAL]
    random.shuffle(train_val_replacement)
    train_replacement = train_val_replacement[N_VAL:]
    val_replacement = train_val_replacement[:N_VAL]

    save_jsonl(train_insertion, 'data/split/train_insertion.jsonl')
    save_jsonl(val_insertion, 'data/split/val_insertion.jsonl')
    save_jsonl(train_replacement, 'data/split/train_replacement.jsonl')
    save_jsonl(val_replacement, 'data/split/val_replacement.jsonl')


def encode_input(text: str):
    return tokenizer(text, max_length=MAX_LENGTH, padding='max_length', truncation=True)


def normalize(text: str) -> str:
    text = unicodedata.normalize('NFKC', text).strip()
    return re.sub('\s+', ' ', text)


def load_data(file_path: str):
    insertion_data = []
    replacement_data = []

    with open(file_path) as fp:
        for row in fp:
            data = json.loads(row)
            bert_input = encode_input(normalize(data['pre_text'])).data
            labels = encode_input(normalize(data['post_text'])).data['input_ids']
            diff_info = diff_opcode(bert_input['input_ids'], labels)

            if diff_info['trainable']:
                insertions = deepcopy(bert_input)
                insertions['labels'] = diff_info['insertions']
                insertion_data.append(insertions)
                replaced_ids = deepcopy(bert_input)
                replaced_ids['labels'] = diff_info['replaced_ids']
                replacement_data.append(replaced_ids)
                if replaced_ids['input_ids'] != replaced_ids['labels'] and random.random() < R:
                        replaced_ids2 = deepcopy(replaced_ids)
                        replaced_ids2['input_ids'] = replaced_ids['labels']
                        replacement_data.append(replaced_ids2)

            else:
                # 差分が大きすぎて学習が難しいデータは、訂正後の文だけを使い「修正不要」の教師データとして扱う
                bert_input['input_ids'] = labels
                bert_input['labels'] = labels
                replacement_data.append(bert_input)

    return insertion_data, replacement_data


def save_jsonl(data: list, file_path: str):
    dir_name = file_path.rsplit('/', 1)[0]
    os.makedirs(dir_name, exist_ok=True)
    with open(file_path, mode='w') as fp:
        for row in data:
            fp.write(json.dumps(row, ensure_ascii=False))
            fp.write('\n')


def diff_opcode(input_ids, labels):
    insertions = [0] * len(input_ids)  # 挿入の教師データ
    replaced_ids = deepcopy(input_ids)  # 置換と削除の教師データ
    trainable = True

    for op, i0, i1, j0, j1 in SequenceMatcher(a=input_ids, b=labels).get_opcodes():
        if op == 'equal':
            pass

        elif op == 'replace':
            if i1 - i0 < j1 - j0:
                # 置換後の単語数のほうが多い場合、token位置をずらす必要があり学習が難しいので今回は諦める
                is_trainable = False
            else:
                for k in range(i1 - i0):
                    if j0 + k < j1:
                        replaced_ids[i0 + k] = labels[j0 + k]
                    else:
                        replaced_ids[i0 + k] = 4  # [MASK]

        elif op == 'delete':
            for i in range(i0, i1):
                replaced_ids[i] = 4 # [MASK]

        elif op == 'insert':
            if labels[j0] == 0:  # 末尾の[PAD]挿入は無視
                pass
            elif i0 >= MAX_LENGTH - j1 + j0:  # 最大長を超える挿入は扱えないので諦める
                trainable = False
            elif j1 - j0 == 1:
                insertions[i0] = 1
            elif j1 - j0 == 2:
                insertions[i0] = 2
            else:  # 挿入すべきtoken数が3以上は難しすぎるので諦める
                trainable = False

    # 挿入かtoken数の違う置換があると末尾に非必要な削除フラグが立つので、補正する
    for i in range(len(replaced_ids) - 1, -1, -1):
        if replaced_ids[i] != 4:
            break
        replaced_ids[i] = input_ids[i]

    return {'trainable': trainable, 'insertions': insertions, 'replaced_ids': replaced_ids}


if __name__ == '__main__':
    prepare_data()
