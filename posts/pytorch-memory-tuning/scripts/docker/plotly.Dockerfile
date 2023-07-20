FROM nvcr.io/nvidia/pytorch:23.02-py3

RUN pip install -U pip && \
    pip install -U \
    plotly \
    pandas==1.5.3 \
    kaleido
