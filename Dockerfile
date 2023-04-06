FROM python:3.10.10-slim-bullseye

COPY *.py *.html requirements.txt ./
COPY models/v0.3.0.ckpt models/v0.3.0.ckpt
RUN pip3 install --no-cache-dir torch
RUN pip3 install --no-cache-dir -r requirements.txt

# 一度推論処理をすることで必要なリソースのダウンロードを済ませておく
RUN python3 predict.py

EXPOSE 9310/tcp
CMD ["python3", "web.py"]
