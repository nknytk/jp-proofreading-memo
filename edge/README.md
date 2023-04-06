エッジ動作用モデル作成・学習コード(未整理)


```
$ python3 -m venv .venv
$ . .venv/bin/activate
$ pip install --upgrade pip wheel
$ pip install transformers[onnx]
$ pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
$ pip install pytorch-lightning
$ pip install onnxruntime
```
