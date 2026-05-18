from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

from faster_whisper import WhisperModel


@dataclass
class Transcription:
    text: str
    language: str
    duration_seconds: float
    segments: list

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "language": self.language,
            "duration_seconds": round(self.duration_seconds, 2),
            "segments": self.segments,
        }


class Transcriber:
    def __init__(self, model_name: str = "base") -> None:
        if model_name != "base":
            raise ValueError("Only the 'base' Whisper model is allowed for this MVP.")
        self._model = WhisperModel(model_name, device="cpu", compute_type="int8")

    def transcribe_file(self, path: str) -> Transcription:
        segments_gen, info = self._model.transcribe(path)
        segments = list(segments_gen)
        duration = segments[-1].end if segments else info.duration
        return Transcription(
            text=" ".join(s.text.strip() for s in segments),
            language=info.language,
            duration_seconds=float(duration),
            segments=[
                {"start": s.start, "end": s.end, "text": s.text.strip()}
                for s in segments
            ],
        )

    def transcribe_bytes(
        self, audio_bytes: bytes, suffix: str = ".webm"
    ) -> Transcription:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            return self.transcribe_file(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python -m backend.audio.transcriber <audio_path>")
        raise SystemExit(2)

    t = Transcriber()
    out = t.transcribe_file(sys.argv[1])
    print(
        {
            "language": out.language,
            "duration_seconds": out.duration_seconds,
            "text": out.text[:300],
        }
    )
