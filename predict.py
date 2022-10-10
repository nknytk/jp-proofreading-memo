from difflib import SequenceMatcher
import os
from time import time
import numpy
import torch
import unidic_lite
import ipadic
import preprocess_bak as preprocess
from model import BertForMaskedLMPL
from fugashi import GenericTagger, Tagger

predictor = BertForMaskedLMPL.load_from_checkpoint('models/v0.1.0.ckpt')

mecabrc = os.path.join(ipadic.DICDIR, 'mecabrc')
tagger = GenericTagger(f'-d "{ipadic.DICDIR}" -r "{mecabrc}" -Oyomi')


def predict(text: str, rescore_window=0, rescore_weight=3.0):
    text = preprocess.normalize(text)
    original_tokens = [token.replace('##', '') for token in preprocess.tokenizer.tokenize(text)]
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
        scores = predictor.bert_mlm(**bert_input_torch).logits
        output_ids = _strip_ids(scores[0].argmax(-1).tolist())
        scores = scores[0].numpy()

    #print(preprocess.tokenizer.convert_ids_to_tokens(input_ids))
    #print(preprocess.tokenizer.convert_ids_to_tokens(output_ids))

    # 入出力の単語IDの差分と、元の文字列を組み合わせて校正後の文字列を作成する。
    # 出力単語IDから文字列を生成すると未知語がすべてUNKNOWNになってしまうため、
    # 元の文字列をベースに差分箇所だけ変更することで未知語も出力にそのまま含められるようにする。
    output_tokens = []
    for op, i0, i1, j0, j1 in SequenceMatcher(a=input_ids, b=output_ids).get_opcodes():
        if op == 'equal':
            # 入出力の単語IDが等しい場合、入力tokenをそのまま出力tokenに採用
            for i in range(i0, i1):
                if i in spaces:
                    output_tokens.append({'from': ' ', 'to': ' ', 'op': None})
                output_tokens.append({'from': original_tokens[i], 'to': original_tokens[i], 'op': None})
        elif op == 'delete':
            # 入力から単語IDが削除されていた場合、出力tokenからもその単語を削除
            for token in original_tokens[i0: i1]:
                output_tokens.append({'from': token, 'to': '', 'op': 'delete'})
        elif op == 'insert':
            # 挿入の場合は出力単語IDからtokenを取得
            for token in preprocess.tokenizer.convert_ids_to_tokens(output_ids[j0: j1]):
                output_tokens.append({'from': '', 'to': token.replace('##', ''), 'op': 'insert'})
        elif op == 'replace':
            # 置換の場合は出力単語IDからtokenを取得、置換前のtoken情報は入力tokenから取得
            new_tokens = preprocess.tokenizer.convert_ids_to_tokens(output_ids[j0: j1])
            orig_tokens = original_tokens[i0: i1]
            if i1 - i0 < j1 - j0:
                for _i in range((j1 - j0) - (i1 - i0)):
                    orig_tokens.append('')
            elif i1 - i0 > j1 - j0:
                for _i in range((i1 - i0) - (j1 - j0)):
                    new_tokens.append('')
            for i, (_from, _to) in enumerate(zip(orig_tokens, new_tokens)):
                if (i + i0) in spaces:
                    output_tokens.append({'from': ' ', 'to': ' ', 'op': None})
                if _to.replace('##', '') == _from:
                    output_tokens.append({'from': _from, 'to': _from, 'op': None})
                    continue
                if _to == '':
                    output_tokens.append({'from': _from, 'to': '', 'op': 'replace'})
                    continue
                elif _from == '':
                    output_tokens.append({'from': _from, 'to': _to.replace('##', ''), 'op': 'replace'})
                    continue

                if rescore_window > 1:
                    # 読みがなが元のtokenと一致する場合、スコアをrescore_weight倍にして単語を選び直す
                    top_ids = numpy.argsort(-scores[j0 + i + 1])[:rescore_window]
                    top_terms = preprocess.tokenizer.convert_ids_to_tokens(top_ids)
                    top_scores = scores[j0 + i + 1][top_ids]
                    rescored_terms = []
                    yomi = _get_kana(_from)
                    for term_id, term, term_score in zip(top_ids, top_terms, top_scores):
                        if term in ('[CLS]', '[SEP]', '[PAD]', '[UNK]', _from):
                            continue
                        weight = 3.0 if _get_kana(term) == yomi else 1.0
                        rescored_terms.append((term, term_score * weight))
                    rescored_terms.sort(key=lambda x: x[1], reverse=True)
                    _to = rescored_terms[0][0]
                output_tokens.append({'from': _from, 'to': _to.replace('##', ''), 'op': 'replace'})

    output_text = ''.join(t['to'] for t in output_tokens if t['to'] not in ('[CLS]', '[SEP]', '[PAD]'))
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
    """
    kana = []
    for t in tagger.parseToNodeList(text):
        if t.feature.kana is not None:
            kana.append(t.feature.kana)
    return ''.join(kana)
    """
    return tagger.parse(text)


if __name__ == '__main__':
    for txt in [
        #'ユーザーの思考に合わせた楽曲を配信する',
        #'メールに明日の会議の飼料を添付した。',
        #'乳酸菌で牛乳を発行するとヨーグルトができる。',
        #'突然、子供が帰省を発した。',
        #'これが明日のの会議の資料です。',
        #'取るべき手順が明確で、誤解サれないことを確認する。',
        '空文字の　位置が　　ずれていないことを　確認する'
    ]:
        t0 = time()
        print(predict(txt)['output'])
        t1 = time()
        print(predict(txt, 30)['output'])
        t2 = time()
        print(t1 - t0, t2 - t1)
