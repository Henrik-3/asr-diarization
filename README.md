# Speech API

Single-model, OpenAI-compatible ASR server: choose a NeMo ASR checkpoint or a faster-whisper (Whisper/CTranslate2) model at startup. Optional NeMo Sortformer diarization is available in the NeMo image. Like vLLM, run one instance per ASR model and point OpenAI-compatible clients at its `/v1` base URL; this is not a general vLLM inference engine.

## OpenAI-compatible transcription endpoint

`POST /v1/audio/transcriptions` uses multipart/form-data and mirrors the OpenAI transcription endpoint shape.

Default `response_format=json` returns:

```json
{"text":"..."}
```

That is intentional: OpenAI's transcription API returns a JSON transcription object by default. `response_format=text` returns plain text.

Supported response formats:

- `json`
- `text`
- `srt`
- `vtt`
- `verbose_json` (segment timestamps; no word-level timestamps)
- `diarized_json` (speaker-attributed segments)

Common OpenAI form fields accepted: `file`, `model`, `language`, `languages[]`, `prompt`, `response_format`, `temperature`, `chunking_strategy`, `timestamp_granularities[]`, `stream`, `include[]`, `known_speaker_names[]`, `known_speaker_references[]`.

Backend limitations are returned as OpenAI-shaped `error` objects instead of silently changing semantics. Streaming, logprobs, known-speaker reference matching, and word-level timestamps are not implemented in this first version.

## Models

`ASR_BACKEND=nemo` loads a NeMo `ASRModel.from_pretrained(ASR_MODEL)` checkpoint (default: `nvidia/nemotron-3.5-asr-streaming-0.6b`). Other compatible NeMo ASR checkpoints can be selected with `ASR_MODEL`. `ASR_BACKEND=faster-whisper` loads a faster-whisper size (e.g. `small`, `large-v3`) or a compatible CTranslate2 model from Hugging Face. **Arbitrary Hugging Face/transformers ASR checkpoints are not supported.** Model weights download on first start; subsequent starts use the mounted Hugging Face cache.

`GET /v1/models` lists the served ASR name (`SERVED_ASR_MODEL`, defaulting to `ASR_MODEL`). With NeMo diarization enabled it also lists `<served-asr>-diarize` (transcription with speaker segments) and the diarization model (raw diarization extension). The real ASR checkpoint ID is also accepted for transcriptions when using an alias. Only one ASR checkpoint is loaded per process; run multiple instances on different ports for multiple models.

Set `DIARIZATION_MODEL=''` to disable diarization, avoid loading its GPU memory, and hide its model IDs. The faster-whisper image requires this setting and sets it by default. Requesting `diarized_json` when disabled returns a 400 error.

## Start

### Pull and run

GitHub Actions publishes `ghcr.io/henrik-3/asr-diarization:latest` (NeMo/CUDA) and `ghcr.io/henrik-3/asr-diarization:latest-whisper` (CPU) on pushes to `main` or manual dispatch. Git tags `v*` also publish matching version tags (e.g. `v1.0.0-whisper`). Make the GHCR package public in GitHub package settings for anonymous pulls; private packages require `docker login ghcr.io`. The workflow uses GitHub-hosted runners by default; to use Blacksmith, enable it on the repository and set the repository variable `BUILD_RUNNER=blacksmith-4vcpu-ubuntu-2404`.

```bash
# NeMo (GPU; docker-compose.yml uses the published image)
docker compose pull && docker compose up -d

# Whisper (CPU, without NeMo or diarization)
docker run --rm -p 8000:8000 -e ASR_MODEL=small \
  -v speech-models:/models/huggingface \
  ghcr.io/henrik-3/asr-diarization:latest-whisper
```

### Persistent model cache

The images contain **code and dependencies, not model weights**. At first startup the selected model(s) download from Hugging Face (or NeMo's model source). `docker-compose.yml` bind-mounts `./model-cache` as `HF_HOME` and `./nemo-cache` as `NEMO_CACHE_DIR`, so downloaded checkpoints survive restarts, image upgrades, and container recreation. Both default NeMo models are Hugging Face `.nemo` files and are cached under `./model-cache`; `./nemo-cache` covers NeMo's separate cache used by other checkpoints. Keep these directories when rebuilding. NeMo still **loads** the cached model into GPU memory on every startup; this is not a new download. A model change or cache deletion can trigger another download, and Hugging Face may check for updates online.

For `docker run`, mount persistent storage yourself: `-v speech-models:/models/huggingface -v speech-nemo:/models/nemo` for NeMo; Whisper needs only the first mount. Without volumes, the cache is lost when the container is removed. Downloaded weights are not baked into the published image, keeping the image reusable for other models and avoiding image growth.

For a different NeMo checkpoint set `ASR_MODEL` in `docker-compose.yml`, or override it at runtime. To build locally: `docker compose build` is replaced by `docker build -t local/speech-api .` (or `docker build -f Dockerfile.whisper -t local/speech-api-whisper .`). The NeMo compose setup requires an NVIDIA GPU and NVIDIA Container Toolkit. Its image uses a slim Python base and CUDA 13.0 PyTorch/TorchAudio wheels, but NeMo and CUDA dependencies still make it multi-gigabyte. The runtime also includes a minimal C compiler: PyTorch/Triton compiles attention kernels on the first inference request, even though model startup succeeds without one. The Whisper image defaults to CPU and is considerably smaller; GPU CTranslate2 deployments require compatible CUDA/cuDNN libraries not included in the CPU image.

### Local Python

For NeMo CPU operation (including systems with an AMD iGPU), run the service in a local virtual environment with `DEVICE=cpu` instead:

```bash
# Install ffmpeg and libsndfile1 using your system package manager first.
# Python 3.12 or newer is recommended by the diarization model publisher.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install Cython packaging
python -m pip install -r requirements.txt
DEVICE=cpu uvicorn server:app --host 0.0.0.0 --port 8000
```

`requirements.txt` pins NVIDIA NeMo Speech to a source revision with RoPE
attention support for `nvidia/Nemotron-3-Diarization`. The NeMo 3.0.0 package
does not support this model's encoder configuration and fails at startup with
`self_attention_model='rope' is not supported`. Install the pinned requirements
to fix this; keep the checkpoint's attention configuration intact.

For a local faster-whisper installation instead, install `requirements-whisper.txt` (plus ffmpeg), then run `ASR_BACKEND=faster-whisper ASR_MODEL=small DIARIZATION_MODEL='' uvicorn server:app --host 0.0.0.0 --port 8000`. Set `DEFAULT_LANGUAGE=auto` for Whisper language detection (already set in the Whisper image).

## Standard transcription

For OpenAI SDKs, use `base_url="http://localhost:8000/v1"`, `api_key="sk-local"` (or the value of `API_KEY` if configured), and a model ID returned by `GET /v1/models`.

Quick test with [Whisper's JFK speech sample](https://github.com/openai/whisper/blob/main/tests/jfk.flac) (~1.1 MB, English, one speaker):

```bash
curl -L -o jfk.flac https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac
curl http://localhost:8000/v1/audio/transcriptions \
  -F 'file=@jfk.flac' \
  -F 'model=nvidia/nemotron-3.5-asr-streaming-0.6b' \
  -F 'language=en'
# With the Whisper image instead, use -F 'model=small'.
```

It should contain JFK's “ask not what your country can do for you” line. This tests transcription, not multiple-speaker diarization.

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F 'file=@meeting.m4a' \
  -F 'model=nvidia/nemotron-3.5-asr-streaming-0.6b' \
  -F 'language=de'
```

Response:

```json
{"text":"..."}
```

Plain text:

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F 'file=@meeting.m4a' \
  -F 'model=nvidia/nemotron-3.5-asr-streaming-0.6b' \
  -F 'response_format=text'
```

## HTTP options that affect behavior

Send these as multipart form fields to `/v1/audio/transcriptions`:

| Field | Behavior |
|---|---|
| `model` | Required served model name; the `-diarize` alias enables speaker attribution. It does not load a new checkpoint per request. |
| `language` | Target language. For the default NeMo model, `de-DE`, `de`, and `auto` are supported and forwarded as `target_lang`. Defaults to `DEFAULT_LANGUAGE`. |
| `languages[]` | Only the first entry is used, and only when `language` is absent. |
| `response_format` | `json`, `text`, `verbose_json`, `srt`, `vtt`, or `diarized_json`. The last enables diarization. |
| `strip_lang_tags` | Extension, default `true`: use NeMo's native decoder option to remove language tags before merging chunks. Set `false` to retain tags emitted by the model. Applies to prompt-conditioned NeMo models supporting this option. |
| `asr_right_context` | Extension, optional: encoder right-context frames, validated against the loaded NeMo model's supported contexts while keeping its left context. Unsupported backends/settings return 400. Omit to retain the checkpoint's default. |
| `timestamp_granularities[]` | `segment` with `verbose_json`; without diarization timestamps cover the whole file. Word timestamps are unsupported. |

For Nemotron 3.5, supported right contexts are `0`, `1`, `3`, `6`, and `13`,
corresponding to native streaming windows of 80, 160, 320, 560, and 1120 ms.
Larger lookahead can improve recognition accuracy. This server applies the encoder
context within its bounded file-transcription windows; it does not implement the
cache-aware streaming CLI or promise those HTTP response times. These settings
are distinct from `ASR_CHUNK_SECONDS`, which bounds memory. Increasing a request
timeout does not improve recognition. The context and tag settings apply only to
the current request and are restored even if transcription fails.

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F 'file=@meeting.m4a' \
  -F 'model=nvidia/nemotron-3.5-asr-streaming-0.6b' \
  -F 'language=de-DE' \
  -F 'strip_lang_tags=true' \
  -F 'asr_right_context=13'
```

`prompt`, `temperature`, and `chunking_strategy` are accepted but **do not affect
inference** (`temperature` is only range-validated). `stream=true`, nonempty
`include[]`, known-speaker references/names, and word timestamps are rejected.
NeMo CLI arguments such as `target_lang`, `att_context_size`, and `batch_size`
are not HTTP fields; use the documented equivalents above. Unknown form fields
are ignored by FastAPI. Chunk duration/overlap, CUDA graph decoding, model
selection, and diarization cache settings are server environment configuration,
not per-request fields. `/docs` provides the generated API schema.

## Long recordings

NeMo ASR automatically processes recordings in sequential windows of at most 30
seconds, with 2 seconds of overlap. No client changes or chunking parameter are
required. This bounds each ASR attention allocation instead of letting it grow
with the square of the entire recording's duration. Only one audio window is read
into a host-side sample array at a time. Short recordings use one inference call.
Long diarized speaker segments and the no-speakers fallback use the same limits;
diarization retains its existing streaming configuration and speaker identities.
Whisper continues to use its backend's own segmentation.

Matching overlap text is removed using conservative word matching. Recognition
can differ between windows, so boundary words may still repeat or be missed;
this is not timestamp-aligned stitching. Overall memory still depends on the
selected models, diarization backend, and hardware. Advanced deployments can tune
`ASR_CHUNK_SECONDS` and `ASR_CHUNK_OVERLAP_SECONDS`; duration must be positive and
finite, and overlap must be nonnegative, finite, and smaller than duration.

### CUDA decoder compatibility

Greedy RNN-T models use eager decoding by default, disabling NeMo's CUDA graph
decoder optimization. This is a compatibility measure for serving variable-length
ASR chunks and speaker segments alongside diarization; it may reduce throughput.
Set `ASR_USE_CUDA_GRAPHS=true` to preserve the checkpoint's decoder setting after
validating it on your GPU. Other decoder types are left unchanged.

A `CUDA error: an illegal memory access was encountered` is distinct from an
out-of-memory error. Restart the container after this error before retrying; the
process can still answer `/health` even though its CUDA context is unusable. The
eager decoder is a mitigation, not a verified fix for every illegal-access error.
If it persists, a diagnostic run with `CUDA_LAUNCH_BLOCKING=1` can help locate the
failing kernel (at a performance cost).

## Diarized transcription, OpenAI-style

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F 'file=@meeting.m4a' \
  -F 'model=nvidia/nemotron-3.5-asr-streaming-0.6b-diarize' \
  -F 'language=de' \
  -F 'response_format=diarized_json' \
  -F 'chunking_strategy=auto'
```

Example response:

```json
{
  "task": "transcribe",
  "duration": 10.2,
  "text": "Hallo. Guten Morgen.",
  "segments": [
    {
      "type": "transcript.text.segment",
      "id": "seg_0",
      "start": 0.2,
      "end": 2.1,
      "text": "Hallo.",
      "speaker": "speaker_0"
    }
  ]
}
```

## Raw diarization extension

This is deliberately outside the OpenAI API surface:

```bash
curl http://localhost:8000/v1/audio/diarizations \
  -F 'file=@meeting.m4a' \
  -F 'model=nvidia/Nemotron-3-Diarization'
```

## Configuration

| Variable | Default |
|---|---|
| `ASR_BACKEND` | `nemo` or `faster-whisper` (image sets its own default) |
| `ASR_MODEL` | NeMo checkpoint ID; Whisper image defaults to `small` |
| `WHISPER_COMPUTE_TYPE` | `auto` (faster-whisper only) |
| `DIARIZATION_MODEL` | `nvidia/Nemotron-3-Diarization` |
| `SERVED_ASR_MODEL` | same as ASR model |
| `SERVED_DIARIZED_MODEL` | `<served-asr>-diarize` |
| `SERVED_DIARIZATION_MODEL` | same as diarization model |
| `DEVICE` | `auto` (NeMo selects CUDA if available; faster-whisper selects its own device) |
| `DEFAULT_LANGUAGE` | `de` (Whisper image uses `auto` for language detection) |
| `MAX_UPLOAD_MB` | `25` |
| `ASR_USE_CUDA_GRAPHS` | `false` (NeMo greedy RNN-T decoder; `true` preserves checkpoint behavior) |
| `ASR_CHUNK_SECONDS` | `30` (NeMo ASR) |
| `ASR_CHUNK_OVERLAP_SECONDS` | `2` (NeMo ASR) |
| `API_KEY` | empty = disabled |
| `DIAR_CHUNK_LEN` | `340` |
| `DIAR_RIGHT_CONTEXT` | `40` |
| `DIAR_FIFO_LEN` | `40` |
| `DIAR_SPKCACHE_UPDATE_PERIOD` | `300` |

Uploads are normalized with ffmpeg to mono 16 kHz PCM WAV before inference. Whisper uses the requested `language` (or `DEFAULT_LANGUAGE`); NeMo models without prompt dictionaries use their own language behavior. `prompt`, `temperature`, and `chunking_strategy` are currently accepted for wire compatibility but do not control inference.

## Compatibility note

The HTTP contract is designed to work with OpenAI-style transcription clients. Exact model capabilities are not identical to OpenAI-hosted transcription models: unsupported capabilities return a 400 error instead of fabricated data. `verbose_json` and subtitles use whole-file timestamps without diarization, not word-accurate timings. The raw `/v1/audio/diarizations` route is an intentional local extension (NeMo diarization only).
