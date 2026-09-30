# Install the CUDA 13.0 PyTorch wheels into a venv rather than starting from
# NVIDIA's large development image. Triton needs a C compiler at inference
# time for attention kernels; only the full build toolchain stays out of runtime.
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir Cython packaging \
    && pip install --no-cache-dir 'torch==2.11.0+cu130' 'torchaudio==2.11.0+cu130' \
       --index-url https://download.pytorch.org/whl/cu130 \
    && pip install --no-cache-dir -r /tmp/requirements.txt

FROM python:3.12-slim
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/models/huggingface \
    NEMO_CACHE_DIR=/models/nemo

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 libgomp1 curl gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
# Flex-attention compiles Triton kernels on the first request, not at startup.
RUN printf 'int main(void) { return 0; }\n' | cc -x c - -o /tmp/cc-check && rm /tmp/cc-check
# Fail at build time if any dependency changed the matched CUDA 13.0 pair.
RUN python -c "import torch, torchaudio; assert torch.version.cuda == '13.0', torch.version.cuda; print('PyTorch', torch.__version__, 'TorchAudio', torchaudio.__version__)"

WORKDIR /app
COPY server.py /app/server.py
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
