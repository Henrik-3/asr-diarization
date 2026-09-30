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

For a different NeMo checkpoint set `ASR_MODEL` in `docker-compose.yml`, or override it at runtime. To build locally: `docker compose build` is replaced by `docker build -t local/speech-api .` (or `docker build -f Dockerfile.whisper -t local/speech-api-whisper .`). The NeMo compose setup requires an NVIDIA GPU and NVIDIA Container Toolkit. Its image uses a slim Python base and CUDA 13.0 PyTorch/TorchAudio wheels, but NeMo and CUDA dependencies still make it multi-gigabyte. The Whisper image defaults to CPU and is considerably smaller; GPU CTranslate2 deployments require compatible CUDA/cuDNN libraries not included in the CPU image.

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
| `API_KEY` | empty = disabled |
| `DIAR_CHUNK_LEN` | `340` |
| `DIAR_RIGHT_CONTEXT` | `40` |
| `DIAR_FIFO_LEN` | `40` |
| `DIAR_SPKCACHE_UPDATE_PERIOD` | `300` |

Uploads are normalized with ffmpeg to mono 16 kHz PCM WAV before inference. Whisper uses the requested `language` (or `DEFAULT_LANGUAGE`); NeMo models without prompt dictionaries use their own language behavior. `prompt`, `temperature`, and `chunking_strategy` are currently accepted for wire compatibility but do not control inference.

## Compatibility note

The HTTP contract is designed to work with OpenAI-style transcription clients. Exact model capabilities are not identical to OpenAI-hosted transcription models: unsupported capabilities return a 400 error instead of fabricated data. `verbose_json` and subtitles use whole-file timestamps without diarization, not word-accurate timings. The raw `/v1/audio/diarizations` route is an intentional local extension (NeMo diarization only).
