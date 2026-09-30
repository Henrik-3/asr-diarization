# Match the published CUDA 13.0 TorchAudio wheels. The NGC 26.08 image ships
# a CUDA 13.4 PyTorch build, which crashes when pip installs CUDA 13.0 TorchAudio.
FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir Cython packaging \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -c "import torch, torchaudio; assert torch.version.cuda == '13.0', torch.version.cuda; print('PyTorch', torch.__version__, 'TorchAudio', torchaudio.__version__)"

WORKDIR /app
COPY server.py /app/server.py

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
