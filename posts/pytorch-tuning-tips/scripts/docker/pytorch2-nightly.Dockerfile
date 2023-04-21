FROM ghcr.io/pytorch/pytorch-nightly:latest

RUN apt update && apt install --no-install-recommends -y \
    g++

RUN pip install -U pip && \
    pip install -U \
    bitsandbytes \
    fvcore \
    transformers \
    sacremoses \
    tqdm \
    boto3 \
    requests \
    regex \
    sentencepiece

ENV TOKENIZERS_PARALLELISM=false

