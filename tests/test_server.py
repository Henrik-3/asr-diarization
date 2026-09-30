import unittest
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
