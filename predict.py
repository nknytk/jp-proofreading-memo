from difflib import SequenceMatcher
import os
from time import time
import numpy
import torch
import unidic_lite
import preprocess
from model import BertForMaskedLMPL
from fugashi import GenericTagger, Tagger

jp_corrector = BertForMaskedLMPL.load_from_checkpoint('models/v0.3.0.ckpt')

mecabrc = os.path.join(unidic_lite.DICDIR, 'mecabrc')
tagger = Tagger(f'-d "{unidic_lite.DICDIR}" -r "{mecabrc}"')


def predict(text: str):
    text = preprocess.normalize(text)
    original_tokens = [token.replace('##', '', 1) for token in preprocess.tokenizer.tokenize(text)]
    i = 0
    spaces = []
    for s, token in enumerate(original_tokens):
        if token == text[i: i + len(token)]:
            i += len(token)
        else:
            spaces.append(s)
            i += len(token) + 1

    bert_input = preprocess.encode_input(text)
    input_ids = _strip_ids(bert_input['input_ids'])

    bert_input_torch = {k: torch.tensor([v]) for k, v in bert_input.items()}
    with torch.no_grad():
        scores = jp_corrector.bert_mlm(**bert_input_torch).logits
        corrected_ids = scores[0].argmax(-1).tolist()[1:len(input_ids)+1]

    output_tokens = []
    for i, (_from, _to) in enumerate(zip(input_ids, corrected_ids)):
        if i in spaces:
            output_tokens.append({'from': ' ', 'to': ' ', 'op': None})
        if _from == _to:
            output_tokens.append({'from': original_tokens[i], 'to': original_tokens[i], 'op': None})
        elif _to == 4:
            output_tokens.append({'from': original_tokens[i], 'to': '', 'op': 'delete'})
            pass
        else:
            new_token = preprocess.tokenizer.convert_ids_to_tokens(_to).replace('##', '', 1)
            if new_token == original_tokens[i]:
                output_tokens.append({'from': original_tokens[i], 'to': original_tokens[i], 'op': None})
            else:
                output_tokens.append({'from': original_tokens[i], 'to': new_token, 'op': 'replace'})


    output_text = ''.join(t['to'] for t in output_tokens)
    return {'input': text, 'output': output_text, 'tokens': output_tokens}


def _strip_ids(ids: list) -> list:
    """ 先頭、末尾のCLS,SEP,PADを除去した単語ID列を返す """
    _min = 0
    _max = len(ids) - 1
    for i in range(_max):
        if ids[i] not in (0, 2, 3):
            _min = i
            break
    for i in range(_max, -1, -1):
        if ids[i] not in (0, 2, 3):
            _max = i
            break
    return ids[_min:_max + 1]


def _get_kana(text: str) -> str:
    kana = []
    for t in tagger.parseToNodeList(text):
        if t.feature.kana is not None:
            kana.append(t.feature.kana)
    return ''.join(kana)


if __name__ == '__main__':
    for txt in [
        'これが明日のの会議の試料です。',
        '最近の家電は以外と壊れやすいい。',
    ]:
        t0 = time()
        res = predict(txt)['output']
        t1 = time()
        print(t1 - t0, res)
