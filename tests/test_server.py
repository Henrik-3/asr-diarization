import unittest
import tempfile
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

import server


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(server.app)

    def tearDown(self):
        server.models.clear()

    def test_whisper_transcription_and_language_detection(self):
        calls = []

        class FakeModel:
            def transcribe(self, path, language):
                calls.append((path, language))
                return iter([SimpleNamespace(text=" Hello "), SimpleNamespace(text="world")]), None

        server.models["asr"] = FakeModel()
        with patch.object(server, "ASR_BACKEND", "faster-whisper"):
            self.assertEqual(server.transcribe_file("audio.wav", "auto"), "Hello world")
            self.assertEqual(server.transcribe_file("audio.wav", "en"), "Hello world")
        self.assertEqual(calls, [("audio.wav", None), ("audio.wav", "en")])

    def test_long_recording_is_bounded_and_covers_every_frame(self):
        calls = []
        # A low sample rate keeps this 24-minute WAV fixture small. Chunk
        # boundaries are calculated from the WAV rate, not a fixed byte size.
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "long.wav")
            with wave.open(path, "wb") as output:
                output.setparams((1, 2, 100, 0, "NONE", "not compressed"))
                output.writeframes(b"\x01\x00" * (24 * 60 * 100))

            def transcribe(chunk_path, language):
                with wave.open(chunk_path, "rb") as chunk:
                    count = chunk.getnframes()
                    self.assertLessEqual(count, 3000)
                    self.assertEqual(chunk.readframes(count), b"\x01\x00" * count)
                calls.append((chunk_path, count, language))
                return "next words"

            with patch.object(server, "ASR_BACKEND", "nemo"), \
                 patch.object(server, "ASR_CHUNK_SECONDS", 30), \
                 patch.object(server, "ASR_CHUNK_OVERLAP_SECONDS", 2), \
                 patch.object(server, "transcribe_chunk", side_effect=transcribe):
                self.assertEqual(server.transcribe_file(path, "de"), "next words")
            self.assertEqual(len(calls), 52)
            self.assertEqual(sum(count for _, count, _ in calls) - 200 * (len(calls) - 1), 144000)
            self.assertTrue(all(language == "de" for _, _, language in calls))
            self.assertFalse(Path(calls[0][0]).exists())

    def test_short_recording_uses_original_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "short.wav")
            with wave.open(path, "wb") as output:
                output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                output.writeframes(b"\x00\x00" * 16000)
            with patch.object(server, "ASR_BACKEND", "nemo"), \
                 patch.object(server, "transcribe_chunk", return_value="hello") as transcribe:
                self.assertEqual(server.transcribe_file(path, "en"), "hello")
                transcribe.assert_called_once_with(path, "en")

    def test_overlap_merging(self):
        self.assertEqual(server.merge_chunk_text("Hello wonderful world.", "Wonderful world continues."),
                         "Hello wonderful world. continues.")
        self.assertEqual(server.merge_chunk_text("a different ending", "new words here"),
                         "a different ending new words here")
        self.assertEqual(server.merge_chunk_text("go", "go home"), "go go home")
        self.assertEqual(server.merge_chunk_text("", "hello"), "hello")
        self.assertEqual(server.merge_chunk_text("hello", ""), "hello")

    def test_invalid_chunk_configuration(self):
        for duration, overlap in [(0, 0), (30, 30), (30, -1), (float("inf"), 2), (30, float("nan"))]:
            with self.subTest(duration=duration, overlap=overlap), \
                 patch.object(server, "ASR_BACKEND", "nemo"), \
                 patch.object(server, "ASR_CHUNK_SECONDS", duration), \
                 patch.object(server, "ASR_CHUNK_OVERLAP_SECONDS", overlap):
                with self.assertRaises(ValueError):
                    server.transcribe_file("unused.wav")

    def test_asr_only_models_and_diarization_error(self):
        with patch.object(server, "DIARIZATION_MODEL", ""):
            result = self.client.get("/v1/models")
            self.assertEqual([m["id"] for m in result.json()["data"]], [server.SERVED_ASR_MODEL])
            result = self.client.post("/v1/audio/transcriptions", data={
                "model": server.SERVED_ASR_MODEL, "response_format": "diarized_json"
            }, files={"file": ("test.wav", b"audio")})
            self.assertEqual(result.status_code, 400)
            self.assertIn("not enabled", result.json()["error"]["message"])
            result = self.client.post("/v1/audio/diarizations", files={"file": ("test.wav", b"audio")})
            self.assertEqual(result.status_code, 404)

    def test_asr_only_transcription(self):
        with patch.object(server, "DIARIZATION_MODEL", ""), \
             patch.object(server, "normalize_audio"), \
             patch.object(server, "wav_duration", return_value=1.0), \
             patch.object(server, "transcribe_file", return_value="hello"):
            result = self.client.post("/v1/audio/transcriptions", data={
                "model": server.SERVED_ASR_MODEL
            }, files={"file": ("test.wav", b"audio")})
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json(), {"text": "hello"})


if __name__ == "__main__":
    unittest.main()
