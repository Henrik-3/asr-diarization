# NeMo Speech API

Small vLLM-like HTTP service for NVIDIA Nemotron ASR + speaker diarization.

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

- `nvidia/nemotron-3.5-asr-streaming-0.6b`: normal transcription
- `nvidia/nemotron-3.5-asr-streaming-0.6b-diarize`: ASR + speaker diarization through the same OpenAI endpoint
- `nvidia/Nemotron-3-Diarization`: raw diarization extension endpoint

All served names are configurable via environment variables.

## Start

The Docker configuration requires an NVIDIA GPU. For CPU operation (including
systems with an AMD iGPU), run the service in a local virtual environment with
`DEVICE=cpu` instead:

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

For the NVIDIA Docker setup:

```bash
docker compose build
docker compose up
```

## Standard transcription

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
| `ASR_MODEL` | `nvidia/nemotron-3.5-asr-streaming-0.6b` |
| `DIARIZATION_MODEL` | `nvidia/Nemotron-3-Diarization` |
| `SERVED_ASR_MODEL` | same as ASR model |
| `SERVED_DIARIZED_MODEL` | `<served-asr>-diarize` |
| `SERVED_DIARIZATION_MODEL` | same as diarization model |
| `DEVICE` | `cuda` if available |
| `DEFAULT_LANGUAGE` | `de` |
| `MAX_UPLOAD_MB` | `25` |
| `API_KEY` | empty = disabled |
| `DIAR_CHUNK_LEN` | `340` |
| `DIAR_RIGHT_CONTEXT` | `40` |
| `DIAR_FIFO_LEN` | `40` |
| `DIAR_SPKCACHE_UPDATE_PERIOD` | `300` |

Uploads are normalized with ffmpeg to mono 16 kHz PCM WAV before inference.

## Compatibility note

The HTTP contract is designed to work with OpenAI-style transcription clients. Exact model capabilities are not identical to OpenAI-hosted transcription models: unsupported capabilities return a 400 error instead of fabricated data. The raw `/v1/audio/diarizations` route is an intentional local extension.
