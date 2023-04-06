from difflib import SequenceMatcher
import json
import random
import os
import re
import unicodedata

with open('preprocess_config.json') as fp:
    config = json.load(fp)

R = 0.2
VAL_R = 0.014363915545922156


def prepare_data():
    test_file = prepare_jsonl('data/sjis/test.jsonl')
    for row in load_data('data/jwtd_v2.0/test.jsonl'):
        test_file.write(json.dumps(row) + '\n')
    test_file.close()

    train_file = prepare_jsonl('data/sjis/train.jsonl')
    val_file = prepare_jsonl('data/sjis/val.jsonl')
    for row in load_data('data/jwtd_v2.0/train.jsonl', R):
        _row = json.dumps(row) + '\n'
        if random.random() < VAL_R:
            val_file.write(_row)
        else:
            train_file.write(_row)
    train_file.close()
    val_file.close()


def prepare_pretrain_data():
    train_file = prepare_jsonl('data/sjis/pre_train.jsonl')
    val_file = prepare_jsonl('data/sjis/pre_val.jsonl')
    for row in load_data('../gen_data/generated2.jsonl', R):
        _row = json.dumps(row) + '\n'
        if random.random() < VAL_R:
            val_file.write(_row)
        else:
            train_file.write(_row)
    train_file.close()
    val_file.close()


def encode(text: str) -> list:
    encoding = [config['special_tokens']['CLS']]
    attention_mask = [1]

    for c in text:
        if len(encoding) >= config['max_length']:
            break

        try:
            b = bytes(c, encoding='sjis')
        except:
            encoding.append(config['special_tokens']['UNK'])
            attention_mask.append(1)
            continue

        # 1バイト文字はそのまま使用
        if len(b) == 1:
            encoding.append(b[0])
            attention_mask.append(1)
        # 2バイト文字の場合、未使用領域である
        #   第一バイトの00-7F,A0-Df,FD-FF
        #   第二バイトの00-3F,FD-FF
        # を削除した場合のコードポイントに変換することで、クラス数を削減する
        elif len(b) == 2:
            if 128 < b[0] <= 160:
                b0 = 256 + 189 * (b[0] - 129)
                b1 = b[1] - 64
            elif 224 <= b[0] < 254:
                b0 = 256 + 189 * (b[0] - 193)
                b1 = b[1] - 64
            else:
                raise RuntimeError('unexpected byte pattern {} {}'.format(b[0], b[1]))
            encoding.append(b0 + b1)
            attention_mask.append(1)

    return encoding, attention_mask


def decode(output: list) -> str:
    byte_chars = []

    for c in output:
        if c in (config['special_tokens']['PAD'], config['special_tokens']['MASK'], config['special_tokens']['CLS']):
            continue
        elif c < 256:
            byte_chars.append((c).to_bytes(1, 'big'))
        elif 256 <= c < 256 + 189 * 31:
            b0 = int((c - 256) / 189) + 129
            b1 = (c - 256) % 189 + 64
            byte_chars.append((b0 * 256 + b1).to_bytes(2, 'big'))
        elif 256 + 189 * 31 <= c < 256 + 189 * 61:
            b0 = int((c - 256) / 189) + 193
            b1 = (c - 256) % 189 + 64
            byte_chars.append((b0 * 256 + b1).to_bytes(2, 'big'))
        else:
            raise RuntimeError(f'unexpected byte pattern {c}')

    return b''.join(byte_chars).decode('sjis', 'ignore')


def pad(values: list, pad_value: int, max_length: int):
    if len(values) > max_length:
        del(values[max_length])
    else:
        pads = [pad_value] * (max_length - len(values))
        values.extend(pads)


def normalize(text: str) -> str:
    text = unicodedata.normalize('NFKC', text).strip()
    return re.sub('\s+', ' ', text)


def load_data(file_path: str, r: float=0.0):
    token_type_ids = [0] * config['max_length']

    with open(file_path) as fp:
        for row in fp:
            data = json.loads(row)
            bert_input, attention_mask = encode(normalize(data['pre_text']))
            labels, attention_mask_label = encode(normalize(data['post_text']))
            labels2 = format_labels(bert_input, labels)
 
            pad(bert_input, config['special_tokens']['PAD'], config['max_length'])
            pad(attention_mask, 0, config['max_length'])

            if labels2 is None:
                pad(labels, config['special_tokens']['PAD'], config['max_length'])
                pad(attention_mask_label, 0, config['max_length'])
                yield {
                    'input_ids': labels,
                    'labels': labels,
                    'attention_mask': attention_mask_label,
                    'token_type_ids': token_type_ids
                }

            else:
                pad(labels2, config['special_tokens']['PAD'], config['max_length'])
                yield {
                    'input_ids': bert_input,
                    'labels': labels2,
                    'attention_mask': attention_mask,
                    'token_type_ids': token_type_ids
                }
                if random.random() < r:
                    pad(labels, config['special_tokens']['PAD'], config['max_length'])
                    pad(attention_mask_label, 0, config['max_length'])
                    yield {
                        'input_ids': labels,
                        'labels': labels,
                        'attention_mask': attention_mask_label,
                        'token_type_ids': token_type_ids
                    }


def prepare_jsonl(file_path: str):
    dir_name = file_path.rsplit('/', 1)[0]
    os.makedirs(dir_name, exist_ok=True)
    return open(file_path, mode='w')


def format_labels(input_ids, labels):
    _labels = []

    for op, i0, i1, j0, j1 in SequenceMatcher(a=input_ids, b=labels).get_opcodes():
        if op == 'equal':
            _labels.extend(labels[j0:j1])

        elif op == 'replace':
            if i1 - i0 < j1 - j0:
                # 置換後の単語数のほうが多い場合、token位置をずらす必要があり学習が難しいので今回は諦める
                return None
            else:
                for k in range(i1 - i0):
                    if j0 + k < j1:
                        _labels.append(labels[j0 + k])
                    else:
                        _labels.append(config['special_tokens']['MASK'])

        elif op == 'delete':
            for i in range(i0, i1):
                _labels.append(config['special_tokens']['MASK'])

        elif op == 'insert':
            #挿入はtoken位置をずらす必要があり難しいので今回は諦める
            return None

    return _labels


if __name__ == '__main__':
    #prepare_data()
    prepare_pretrain_data()
