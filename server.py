import asyncio
import json
import logging
import math
import string
import os
import subprocess
import tempfile
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Header, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response

log = logging.getLogger("speech-api")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())

ASR_BACKEND = os.getenv("ASR_BACKEND", "nemo").lower()
ASR_MODEL = os.getenv("ASR_MODEL", "nvidia/nemotron-3.5-asr-streaming-0.6b")
# Set to an empty string to run ASR without loading NeMo diarization.
DIARIZATION_MODEL = os.getenv("DIARIZATION_MODEL", "nvidia/Nemotron-3-Diarization")
SERVED_ASR_MODEL = os.getenv("SERVED_ASR_MODEL", ASR_MODEL)
SERVED_DIARIZATION_MODEL = os.getenv("SERVED_DIARIZATION_MODEL", DIARIZATION_MODEL)
SERVED_DIARIZED_MODEL = os.getenv("SERVED_DIARIZED_MODEL", f"{SERVED_ASR_MODEL}-diarize")
DEVICE = os.getenv("DEVICE", "auto")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "auto")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "de")
API_KEY = os.getenv("API_KEY", "")
# OpenAI file-transcription API documents a 25 MB file limit. Override if desired.
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))

# Bound attention memory independently of upload size or recording duration.
ASR_USE_CUDA_GRAPHS = os.getenv("ASR_USE_CUDA_GRAPHS", "false").lower() in {"1", "true", "yes"}
ASR_CHUNK_SECONDS = float(os.getenv("ASR_CHUNK_SECONDS", "30"))
ASR_CHUNK_OVERLAP_SECONDS = float(os.getenv("ASR_CHUNK_OVERLAP_SECONDS", "2"))

DIAR_CHUNK_LEN = int(os.getenv("DIAR_CHUNK_LEN", "340"))
DIAR_RIGHT_CONTEXT = int(os.getenv("DIAR_RIGHT_CONTEXT", "40"))
DIAR_FIFO_LEN = int(os.getenv("DIAR_FIFO_LEN", "40"))
DIAR_SPKCACHE_UPDATE_PERIOD = int(os.getenv("DIAR_SPKCACHE_UPDATE_PERIOD", "300"))

models: dict[str, Any] = {}
model_lock = asyncio.Lock()


class APIError(Exception):
    def __init__(self, status: int, message: str, *, param: str | None = None,
                 error_type: str = "invalid_request_error", code: str | None = None):
        self.status = status
        self.message = message
        self.param = param
        self.error_type = error_type
        self.code = code
        super().__init__(message)


def require_auth(authorization: str | None) -> None:
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise APIError(401, "Incorrect API key provided.", error_type="invalid_request_error", code="invalid_api_key")


def normalize_audio(input_path: str, output_path: str) -> None:
    proc = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", input_path, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            output_path,
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise APIError(400, f"Invalid audio file: {proc.stderr.strip()}", param="file")


def wav_duration(path: str) -> float:
    with wave.open(path, "rb") as f:
        return f.getnframes() / float(f.getframerate())


def parse_diar_segment(segment: Any) -> dict[str, Any]:
    if isinstance(segment, str):
        parts = segment.strip().replace(",", " ").split()
        if len(parts) >= 3:
            return {"start": float(parts[0]), "end": float(parts[1]), "speaker": str(parts[2])}
    if isinstance(segment, (list, tuple)) and len(segment) >= 3:
        return {"start": float(segment[0]), "end": float(segment[1]), "speaker": str(segment[2])}
    if isinstance(segment, dict):
        return {
            "start": float(segment.get("start", segment.get("begin", 0))),
            "end": float(segment.get("end", 0)),
            "speaker": str(segment.get("speaker", segment.get("speaker_id", "speaker_0"))),
        }
    raise ValueError(f"Unsupported diarization segment: {segment!r}")


def extract_text(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if hasattr(result, "text"):
        return str(result.text)
    if isinstance(result, dict) and "text" in result:
        return str(result["text"])
    return str(result)


def merge_chunk_text(previous: str, current: str) -> str:
    """Remove an exact word overlap, tolerating casing and edge punctuation.

    Without word timestamps this is deliberately conservative: differing
    recognition at a boundary can still leave repeated words.
    """
    left, right = previous.split(), current.split()
    normalize = lambda word: word.strip(string.punctuation + "„“”‘’…").casefold()
    for count in range(min(32, len(left), len(right)), 1, -1):
        suffix = [normalize(word) for word in left[-count:]]
        prefix = [normalize(word) for word in right[:count]]
        if all(suffix) and suffix == prefix:
            return " ".join(left + right[count:])
    return " ".join(left + right)


def transcribe_file(path: str, language: str | None = None) -> str:
    if ASR_BACKEND == "faster-whisper":
        return transcribe_chunk(path, language)
    if (not math.isfinite(ASR_CHUNK_SECONDS) or ASR_CHUNK_SECONDS <= 0
            or not math.isfinite(ASR_CHUNK_OVERLAP_SECONDS)
            or not 0 <= ASR_CHUNK_OVERLAP_SECONDS < ASR_CHUNK_SECONDS):
        raise ValueError("ASR chunk duration must be positive and overlap must be smaller than duration")

    # Read only one chunk into host memory; never decode the whole recording
    # into a float array. Temporary WAVs support both NeMo input interfaces.
    with wave.open(path, "rb") as source:
        rate = source.getframerate()
        chunk_frames = int(ASR_CHUNK_SECONDS * rate)
        overlap_frames = int(ASR_CHUNK_OVERLAP_SECONDS * rate)
        if chunk_frames < 1 or chunk_frames <= overlap_frames:
            raise ValueError("ASR chunk settings must allow progress by at least one audio frame")
        total = source.getnframes()
        if total <= chunk_frames:
            return transcribe_chunk(path, language)
        log.info("Chunking %.1fs audio into at most %.1fs ASR windows", total / rate, ASR_CHUNK_SECONDS)
        text = ""
        with tempfile.TemporaryDirectory(prefix="nemo-asr-chunks-") as tmpdir:
            chunk_path = str(Path(tmpdir) / "chunk.wav")
            start = 0
            while start < total:
                source.setpos(start)
                frames = source.readframes(min(chunk_frames, total - start))
                with wave.open(chunk_path, "wb") as output:
                    output.setparams(source.getparams())
                    output.writeframes(frames)
                chunk_text = transcribe_chunk(chunk_path, language).strip()
                text = (merge_chunk_text(text, chunk_text) if overlap_frames
                        else " ".join(part for part in (text, chunk_text) if part))
                if start + chunk_frames >= total:
                    break
                start += chunk_frames - overlap_frames
        return text


def transcribe_chunk(path: str, language: str | None = None) -> str:
    model = models["asr"]
    if ASR_BACKEND == "faster-whisper":
        # faster-whisper returns (segment iterator, info); consume the iterator
        # inside the inference lock before the temporary input file is deleted.
        segments, _ = model.transcribe(path, language=None if language in (None, "auto") else language)
        return " ".join(s.text.strip() for s in segments if s.text.strip())

    import soundfile as sf
    model_defaults = getattr(model, "cfg", {}).get("model_defaults", {})
    prompt_dictionary = model_defaults.get("prompt_dictionary")
    if prompt_dictionary:
        target_language = language or DEFAULT_LANGUAGE
        if target_language not in prompt_dictionary:
            raise APIError(400, f"Unsupported language: {target_language}", param="language")
        # Normalized WAV samples bypass NeMo's file dataloader, which expects
        # language metadata. The prompt-aware model uses target_lang directly
        # for array inputs, preserving the caller's selected language.
        samples, sample_rate = sf.read(path, dtype="float32")
        if sample_rate != 16000 or samples.ndim != 1:
            raise ValueError("Transcription requires normalized mono 16 kHz audio")
        result = model.transcribe(audio=[samples], batch_size=1, target_lang=target_language)
    else:
        # Other configurable NeMo models use the ordinary file interface.
        try:
            result = model.transcribe(audio=[path], batch_size=1)
        except TypeError:
            result = model.transcribe([path], batch_size=1)
    if isinstance(result, list) and result:
        return extract_text(result[0])
    return extract_text(result)


def run_transcription(path: str, language: str, diarized: bool,
                      strip_lang_tags: bool = True,
                      asr_right_context: int | None = None) -> tuple[str, list[dict[str, Any]]]:
    """Apply request options while the caller holds model_lock, then restore them."""
    model = models.get("asr")
    encoder = getattr(model, "encoder", None)
    previous_context = None
    if asr_right_context is not None:
        if ASR_BACKEND != "nemo" or not callable(getattr(encoder, "set_default_att_context_size", None)):
            raise APIError(400, "ASR right context is not supported by this backend/model.", param="asr_right_context")
        current = getattr(encoder, "att_context_size", None)
        supported = getattr(encoder, "att_context_size_all", [])
        if current is None or len(current) != 2:
            raise APIError(400, "ASR right context is not supported by this encoder.", param="asr_right_context")
        requested = [current[0], asr_right_context]
        if requested not in [list(context) for context in supported]:
            raise APIError(400, f"Unsupported attention context {requested}; model supports {supported}.",
                           param="asr_right_context")
        previous_context = list(current)

    decoder = getattr(model, "decoding", None)
    prompt_dictionary = getattr(model, "cfg", {}).get("model_defaults", {}).get("prompt_dictionary")
    tag_setter = getattr(decoder, "set_strip_lang_tags", None)
    configure_tags = ASR_BACKEND == "nemo" and bool(prompt_dictionary) and callable(tag_setter)
    previous_strip = getattr(decoder, "strip_lang_tags", False)
    previous_pattern = getattr(getattr(decoder, "lang_tag_pattern", None), "pattern", None)
    try:
        if previous_context is not None:
            encoder.set_default_att_context_size(requested)
        if configure_tags:
            tag_setter(strip_lang_tags)
        if diarized:
            return transcribe_diarized(path, language)
        return transcribe_file(path, language), []
    finally:
        if configure_tags:
            tag_setter(previous_strip, lang_tag_pattern=previous_pattern)
        if previous_context is not None:
            encoder.set_default_att_context_size(previous_context)


def diarize_file(path: str) -> list[dict[str, Any]]:
    result = models["diar"].diarize(audio=[path], batch_size=1)
    if not result:
        return []
    return [parse_diar_segment(s) for s in result[0]]


def transcribe_diarized(path: str, language: str | None = None) -> tuple[str, list[dict[str, Any]]]:
    segments = diarize_file(path)
    if not segments:
        return transcribe_file(path, language), []

    enriched: list[dict[str, Any]] = []
    full_text: list[str] = []
    with tempfile.TemporaryDirectory(prefix="nemo-segments-") as tmpdir:
        for idx, seg in enumerate(segments):
            duration = max(0.0, seg["end"] - seg["start"])
            if duration < 0.05:
                continue
            seg_path = str(Path(tmpdir) / f"segment-{idx:05d}.wav")
            proc = subprocess.run(
                [
                    "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", f'{seg["start"]:.3f}', "-t", f"{duration:.3f}",
                    "-i", path, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", seg_path,
                ],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                log.warning("Failed to crop segment %s: %s", idx, proc.stderr.strip())
                continue
            text = transcribe_file(seg_path, language).strip()
            enriched.append({**seg, "text": text})
            if text:
                full_text.append(text)
    return " ".join(full_text).strip(), enriched


def srt_time(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def vtt_time(seconds: float) -> str:
    return srt_time(seconds).replace(",", ".")


def subtitle_response(text: str, segments: list[dict[str, Any]], duration: float, fmt: str) -> Response:
    usable = segments or [{"start": 0.0, "end": duration, "text": text}]
    if fmt == "srt":
        chunks = []
        for i, seg in enumerate(usable, 1):
            chunks.append(f"{i}\n{srt_time(seg['start'])} --> {srt_time(seg['end'])}\n{seg.get('text', '')}\n")
        return Response("\n".join(chunks), media_type="application/x-subrip; charset=utf-8")

    chunks = ["WEBVTT\n"]
    for seg in usable:
        chunks.append(f"{vtt_time(seg['start'])} --> {vtt_time(seg['end'])}\n{seg.get('text', '')}\n")
    return Response("\n".join(chunks), media_type="text/vtt; charset=utf-8")


def diarized_payload(text: str, segments: list[dict[str, Any]], duration: float) -> dict[str, Any]:
    return {
        "task": "transcribe",
        "duration": duration,
        "text": text,
        "segments": [
            {
                "type": "transcript.text.segment",
                "id": f"seg_{i}",
                "start": seg["start"],
                "end": seg["end"],
                "text": seg.get("text", ""),
                "speaker": seg["speaker"],
            }
            for i, seg in enumerate(segments)
        ],
    }


def verbose_payload(text: str, language: str, duration: float,
                    segments: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    src = segments or [{"start": 0.0, "end": duration, "text": text}]
    out_segments = []
    for i, seg in enumerate(src):
        out_segments.append({
            "id": i,
            "seek": 0,
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": seg.get("text", ""),
            "tokens": [],
            "temperature": 0.0,
            "avg_logprob": 0.0,
            "compression_ratio": 0.0,
            "no_speech_prob": 0.0,
        })
    return {
        "task": "transcribe",
        "language": language,
        "duration": duration,
        "text": text,
        "segments": out_segments,
    }


def configure_nemo_decoding(model: Any) -> None:
    """Prefer eager RNN-T decoding for variable-length, multi-model serving."""
    if ASR_USE_CUDA_GRAPHS or not hasattr(model, "joint"):
        return
    change_strategy = getattr(model, "change_decoding_strategy", None)
    config = getattr(model, "cfg", {}).get("decoding")
    if not callable(change_strategy) or config is None:
        return
    if config.get("strategy", "greedy_batch") not in {"greedy", "greedy_batch"}:
        return

    from omegaconf import OmegaConf

    # Reconstruct the decoder: changing only the config after construction
    # leaves the existing graph-enabled decoding computer in place.
    decoding = OmegaConf.merge(config, {"greedy": {"use_cuda_graph_decoder": False}})
    change_strategy(decoding)
    log.info("NeMo RNN-T CUDA graph decoding disabled (compatibility default)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if ASR_BACKEND not in {"nemo", "faster-whisper"}:
        raise ValueError(f"Unknown ASR_BACKEND: {ASR_BACKEND}")
    if ASR_BACKEND == "faster-whisper" and DIARIZATION_MODEL:
        raise ValueError("Set DIARIZATION_MODEL='' when using faster-whisper (NeMo diarization is not installed)")
    served_names = [SERVED_ASR_MODEL]
    if DIARIZATION_MODEL:
        served_names.extend([SERVED_DIARIZED_MODEL, SERVED_DIARIZATION_MODEL])
    if len(set(served_names)) != len(served_names):
        raise ValueError("Served model names must be distinct")

    log.info("Loading %s ASR model %s", ASR_BACKEND, ASR_MODEL)
    if ASR_BACKEND == "nemo":
        import torch
        import nemo.collections.asr as nemo_asr

        device = DEVICE if DEVICE != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        models["asr"] = nemo_asr.models.ASRModel.from_pretrained(ASR_MODEL).to(device).eval()
        configure_nemo_decoding(models["asr"])
        if DIARIZATION_MODEL:
            from nemo.collections.asr.models import SortformerEncLabelModel

            log.info("Loading diarization model %s", DIARIZATION_MODEL)
            diar = SortformerEncLabelModel.from_pretrained(DIARIZATION_MODEL).to(device).eval()
            diar.sortformer_modules.chunk_len = DIAR_CHUNK_LEN
            diar.sortformer_modules.chunk_right_context = DIAR_RIGHT_CONTEXT
            diar.sortformer_modules.fifo_len = DIAR_FIFO_LEN
            diar.sortformer_modules.spkcache_update_period = DIAR_SPKCACHE_UPDATE_PERIOD
            diar._check_streaming_parameters()
            models["diar"] = diar
    else:
        from faster_whisper import WhisperModel

        models["asr"] = WhisperModel(ASR_MODEL, device=DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
    log.info("Models loaded on %s", DEVICE)
    try:
        yield
    finally:
        models.clear()


app = FastAPI(title="Speech OpenAI-Compatible API", version="0.3.0", lifespan=lifespan)


@app.exception_handler(APIError)
async def api_error_handler(_: Request, exc: APIError):
    return JSONResponse(
        status_code=exc.status,
        content={"error": {"message": exc.message, "type": exc.error_type, "param": exc.param, "code": exc.code}},
    )


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "models_loaded": sorted(models.keys())}


@app.get("/v1/models")
def list_models(authorization: str | None = Header(default=None)):
    require_auth(authorization)
    names = [SERVED_ASR_MODEL]
    if DIARIZATION_MODEL:
        names.extend([SERVED_DIARIZED_MODEL, SERVED_DIARIZATION_MODEL])
    return {"object": "list", "data": [
        {"id": name, "object": "model", "owned_by": "local"} for name in names
    ]}


async def save_upload(file: UploadFile, directory: str) -> str:
    suffix = Path(file.filename or "audio.bin").suffix or ".bin"
    raw_path = str(Path(directory) / f"upload{suffix}")
    total = 0
    with open(raw_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_MB * 1024 * 1024:
                raise APIError(413, f"Maximum content size limit ({MAX_UPLOAD_MB} MB) exceeded.", param="file")
            f.write(chunk)
    return raw_path


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
    strip_lang_tags: bool = Form(default=True),
    asr_right_context: int | None = Form(default=None),
    stream: bool = Form(default=False),
    chunking_strategy: str | None = Form(default=None),
    timestamp_granularities: list[str] | None = Form(default=None, alias="timestamp_granularities[]"),
    include: list[str] | None = Form(default=None, alias="include[]"),
    languages: list[str] | None = Form(default=None, alias="languages[]"),
    known_speaker_names: list[str] | None = Form(default=None, alias="known_speaker_names[]"),
    known_speaker_references: list[str] | None = Form(default=None, alias="known_speaker_references[]"),
    authorization: str | None = Header(default=None),
):
    require_auth(authorization)

    valid_models = {SERVED_ASR_MODEL, ASR_MODEL}
    if DIARIZATION_MODEL:
        valid_models.add(SERVED_DIARIZED_MODEL)
    if model not in valid_models:
        raise APIError(404, f"The model '{model}' does not exist", param="model", code="model_not_found")

    valid_formats = {"json", "text", "srt", "verbose_json", "vtt", "diarized_json"}
    if response_format not in valid_formats:
        raise APIError(400, f"Invalid response_format '{response_format}'.", param="response_format")
    if not 0 <= temperature <= 1:
        raise APIError(400, "temperature must be between 0 and 1.", param="temperature")
    if stream:
        raise APIError(400, "Streaming transcription is not supported by this backend.", param="stream")
    if include:
        raise APIError(400, "include/logprobs is not supported by this backend.", param="include")
    if known_speaker_names or known_speaker_references:
        raise APIError(400, "Known-speaker references are not supported by this backend.", param="known_speaker_names")
    if timestamp_granularities and response_format != "verbose_json":
        raise APIError(400, "timestamp_granularities requires response_format='verbose_json'.", param="timestamp_granularities")
    if timestamp_granularities and "word" in timestamp_granularities:
        raise APIError(400, "Word-level timestamps are not supported by this backend.", param="timestamp_granularities")

    # Accepted for wire compatibility, but not applied to inference.
    if prompt:
        log.debug("prompt accepted but not consumed by the ASR backend")
    if chunking_strategy:
        log.debug("chunking_strategy=%s accepted but not consumed by the ASR backend", chunking_strategy)

    selected_language = language or (languages[0] if languages else None) or DEFAULT_LANGUAGE
    want_diarization = model == SERVED_DIARIZED_MODEL or response_format == "diarized_json"
    if want_diarization and not DIARIZATION_MODEL:
        raise APIError(400, "Diarization is not enabled on this server.", param="response_format")

    with tempfile.TemporaryDirectory(prefix="nemo-api-") as tmpdir:
        raw_path = await save_upload(file, tmpdir)
        wav_path = str(Path(tmpdir) / "audio.wav")
        normalize_audio(raw_path, wav_path)
        duration = wav_duration(wav_path)

        async with model_lock:
            text, segments = await asyncio.to_thread(
                run_transcription, wav_path, selected_language, want_diarization,
                strip_lang_tags, asr_right_context,
            )

        if response_format == "text":
            return PlainTextResponse(text, media_type="text/plain; charset=utf-8")
        if response_format in {"srt", "vtt"}:
            return subtitle_response(text, segments, duration, response_format)
        if response_format == "verbose_json":
            return verbose_payload(text, selected_language, duration, segments or None)
        if response_format == "diarized_json":
            return diarized_payload(text, segments, duration)

        # OpenAI's default response_format=json is a JSON transcription object.
        return {"text": text}


# Non-OpenAI extension: raw diarization without ASR.
@app.post("/v1/audio/diarizations")
async def diarizations(
    file: UploadFile = File(...),
    model: str = Form(default=SERVED_DIARIZATION_MODEL),
    authorization: str | None = Header(default=None),
):
    require_auth(authorization)
    if not DIARIZATION_MODEL or model not in {SERVED_DIARIZATION_MODEL, DIARIZATION_MODEL}:
        raise APIError(404, f"The model '{model}' does not exist", param="model", code="model_not_found")

    with tempfile.TemporaryDirectory(prefix="nemo-api-") as tmpdir:
        raw_path = await save_upload(file, tmpdir)
        wav_path = str(Path(tmpdir) / "audio.wav")
        normalize_audio(raw_path, wav_path)
        async with model_lock:
            segments = await asyncio.to_thread(diarize_file, wav_path)

        speakers = sorted({s["speaker"] for s in segments})
        return {"segments": segments, "speakers": speakers, "speaker_count": len(speakers)}
