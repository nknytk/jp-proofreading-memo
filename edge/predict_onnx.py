from difflib import SequenceMatcher
import json
import os
import sys
from time import time
import onnxruntime
import preprocess

ort_session = onnxruntime.InferenceSession(sys.argv[1])
input_names = [(i.name, i.type, i.shape) for i in ort_session.get_inputs()]
print(input_names)
output_names = [(i.name, i.type, i.shape) for i in ort_session.get_outputs()]
print(output_names)

with open('preprocess_config.json') as fp:
    config = json.load(fp)

token_type_ids = [[0] * config['max_length']]


def predict(text: str):
    text = preprocess.normalize(text)
    input_ids, attention_mask = preprocess.encode(text)
    preprocess.pad(input_ids, config['special_tokens']['PAD'], config['max_length'])
    preprocess.pad(attention_mask, 0, config['max_length'])


    model_input = {
        input_names[0][0]: [input_ids],
        input_names[1][0]: [attention_mask],
        input_names[2][0]: token_type_ids
    }
    ort_outs = ort_session.run(None, model_input)
    corrected_ids = ort_outs[0][0].argmax(-1).tolist()

    output_tokens = []
    for _orig, _from, _to in zip(list(text), input_ids[1:], corrected_ids[1:]):
        if _from == _to or _to == config['special_tokens']['UNK']:
            output_tokens.append({'from': _orig, 'to': _orig, 'op': None})
        elif _to == config['special_tokens']['MASK']:
            output_tokens.append({'from': _orig, 'to': '', 'op': 'delete'})
        elif _to == config['special_tokens']['PAD']:
            continue
        else:
            output_tokens.append({'from': _orig, 'to': preprocess.decode([_to]), 'op': 'replace'})

    output_text = ''.join(t['to'] for t in output_tokens)
    return {'input': text, 'output': output_text, 'tokens': output_tokens}


if __name__ == '__main__':
    for txt in [
        '最近の家電は以外と壊れやすいい。',
        '取るべき手順が明確でで、誤解サれないことを確認する。',
        '細菌サッカーが流行してているらしい。',
        'この番組では細菌話題のチーズケーキを特集します。',
        'ユーザーの思考に合わせた楽曲を配信する',
        'メールに明日の会議の飼料を添付した。',
        '乳酸菌で牛乳を発行するとヨーグルトにになる。',
        '乳酸菌でで牛乳を発発酵するとヨーグルトになる。',
        '突然、子供が帰省を発した。',
        'これが明日のの懐疑の試料です。',
        '空文字の　位置が　　ずれていないことを　確認する'
    ]:
        t0 = time()
        res = predict(txt)['output']
        t1 = time()
        print(f'from: {txt}')
        print(f'to  : {res}')
        print('-----')
