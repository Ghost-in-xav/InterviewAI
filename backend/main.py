"""FastAPI app exposing a /ws/session WebSocket for InterviewIQ.

Wire protocol (JSON envelopes from the client):
- {"type": "frame_meta"} immediately followed by a binary JPEG frame
- {"type": "audio_meta"} immediately followed by a binary audio chunk
- {"type": "stop"}                                — finalize, transcribe, score

Server pushes:
- {"type": "metrics", "eye_contact": {...}, "posture": {...}, "frame_index": N}
- {"type": "status", "message": "..."}
- {"type": "report", "metrics": {...}, "report": {...}}
- {"type": "error", "message": "..."}
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
from dataclasses import dataclass, field

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.audio.analyzer import analyze as analyze_audio
from backend.audio.transcriber import Transcriber
from backend.feedback import FeedbackGenerator, _fallback_report
from backend.vision.eye_contact import EyeContactDetector
from backend.vision.face_baseline import FaceBaselineTracker, is_off_center
from backend.vision.face_lines import FaceLinesDetector
from backend.vision.face_occlusion import FaceOcclusionDetector
from backend.vision.motion_detector import MotionDetector
from backend.vision.hand_detector import HandDetector, hand_overlaps_face
from backend.vision.object_detector import ObjectDetector
from backend.vision.posture import PostureDetector

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
)
log = logging.getLogger("interviewiq")

# Heavy singletons — instantiated once at startup.
eye_detector = EyeContactDetector()
posture_detector = PostureDetector()
face_lines_detector = FaceLinesDetector()
hand_detector = HandDetector()
transcriber: Transcriber | None = None  # lazy: Whisper load is slow, defer until needed
feedback_generator = FeedbackGenerator()


def get_transcriber() -> Transcriber:
    global transcriber
    if transcriber is None:
        log.info("Loading Whisper base model (first use)...")
        transcriber = Transcriber()
        log.info("Whisper loaded.")
    return transcriber


MIN_FRAME_INTERVAL_S = 0.9  # server-side guard: never analyze more than ~1 FPS


@dataclass
class SessionState:
    start_time: float = field(default_factory=time.time)
    last_frame_time: float = 0.0
    frame_count: int = 0
    eye_looking_count: int = 0
    eye_face_detected_count: int = 0
    posture_counts: dict[str, int] = field(
        default_factory=lambda: {
            "good": 0,
            "slouched": 0,
            "tilted": 0,
            "no_person": 0,
            "off_center": 0,
            "too_much_angle": 0,
        }
    )
    audio_chunks: list[bytes] = field(default_factory=list)
    face_baseline: FaceBaselineTracker = field(default_factory=FaceBaselineTracker)
    face_occlusion: FaceOcclusionDetector = field(default_factory=FaceOcclusionDetector)
    motion: MotionDetector = field(default_factory=MotionDetector)
    object_detector: ObjectDetector = field(default_factory=ObjectDetector)

    def add_frame_metrics(self, eye: dict, posture: dict) -> None:
        self.frame_count += 1
        if eye["face_detected"]:
            self.eye_face_detected_count += 1
            if eye["looking"]:
                self.eye_looking_count += 1
        self.posture_counts[posture["status"]] = (
            self.posture_counts.get(posture["status"], 0) + 1
        )

    def eye_contact_pct(self) -> float:
        if self.eye_face_detected_count == 0:
            return 0.0
        return round(100 * self.eye_looking_count / self.eye_face_detected_count, 1)


app = FastAPI(title="InterviewIQ", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


def _decode_jpeg(buffer: bytes) -> np.ndarray | None:
    arr = np.frombuffer(buffer, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


async def _send_json(ws: WebSocket, payload: dict) -> None:
    await ws.send_text(json.dumps(payload, ensure_ascii=False))


@app.websocket("/ws/session")
async def session_socket(ws: WebSocket) -> None:
    await ws.accept()
    state = SessionState()
    pending_meta: dict | None = None
    log.info("WebSocket accepted")

    try:
        while True:
            message = await ws.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if "text" in message and message["text"] is not None:
                try:
                    envelope = json.loads(message["text"])
                except json.JSONDecodeError:
                    await _send_json(ws, {"type": "error", "message": "invalid json"})
                    continue

                msg_type = envelope.get("type")
                if msg_type == "frame_meta":
                    pending_meta = envelope
                elif msg_type == "audio_meta":
                    pending_meta = envelope
                elif msg_type == "stop":
                    await _finalize(ws, state)
                    break
                else:
                    await _send_json(
                        ws, {"type": "error", "message": f"unknown type: {msg_type}"}
                    )

            elif "bytes" in message and message["bytes"] is not None:
                if pending_meta is None:
                    log.warning("Received binary frame without meta envelope; dropping")
                    continue

                if pending_meta["type"] == "frame_meta":
                    now = time.time()
                    if now - state.last_frame_time < MIN_FRAME_INTERVAL_S:
                        # Throttle: drop this frame silently.
                        pending_meta = None
                        continue
                    state.last_frame_time = now

                    frame = _decode_jpeg(message["bytes"])
                    if frame is None:
                        pending_meta = None
                        continue

                    eye = eye_detector.analyze(frame).to_dict()
                    posture = posture_detector.analyze(frame).to_dict()
                    face_lines = face_lines_detector.analyze(frame).to_dict()
                    hands = hand_detector.analyze(frame).to_dict()

                    hand_over_face = False
                    face_bbox = eye.get("face_bbox")
                    if face_bbox and hands.get("bboxes"):
                        for hand_bbox in hands["bboxes"]:
                            if hand_overlaps_face(face_bbox, hand_bbox):
                                hand_over_face = True
                                break

                    motion = state.motion.analyze(
                        frame,
                        face_bbox,
                        face_landmarks=eye.get("face_landmarks"),
                    ).to_dict()

                    objects = state.object_detector.analyze(
                        frame,
                        face_bbox=face_bbox,
                        hand_bboxes=hands.get("bboxes") or [],
                    ).to_dict()

                    face_mask = state.face_occlusion.analyze(
                        frame,
                        face_bbox,
                        hand_over_face=hand_over_face,
                        landmark_confidence=face_lines.get("landmark_confidence"),
                        face_landmarks=eye.get("face_landmarks"),
                        low_conf_count=face_lines.get("low_conf_count"),
                        motion_over_face=motion.get("motion_over_face", False),
                    ).to_dict()

                    if eye.get("face_detected") and eye.get("face_bbox"):
                        state.face_baseline.update(eye["face_bbox"])

                    avg_face = state.face_baseline.average()
                    off_center = False
                    if eye.get("face_detected") and eye.get("face_bbox") and avg_face:
                        off_center = is_off_center(
                            eye["face_bbox"], avg_face, threshold=0.10
                        )
                        if off_center:
                            posture["status"] = "off_center"

                    posture["face_bbox"] = eye.get("face_bbox")
                    posture["face_bbox_avg"] = avg_face
                    posture["off_center"] = off_center
                    posture["too_much_angle"] = bool(face_lines.get("too_much_angle"))

                    if (
                        posture.get("status") != "no_person"
                        and posture["too_much_angle"]
                    ):
                        posture["status"] = "too_much_angle"
                    state.add_frame_metrics(eye, posture)

                    await _send_json(
                        ws,
                        {
                            "type": "metrics",
                            "eye_contact": eye,
                            "posture": posture,
                            "face_lines": face_lines,
                            "face_mask": face_mask,
                            "hands": hands,
                            "motion": motion,
                            "objects": objects,
                            "frame_index": state.frame_count,
                            "eye_contact_pct": state.eye_contact_pct(),
                        },
                    )

                elif pending_meta["type"] == "audio_meta":
                    state.audio_chunks.append(message["bytes"])

                pending_meta = None

    except WebSocketDisconnect:
        log.info("WebSocket disconnected by client")
    except Exception:  # noqa: BLE001
        log.error("Unhandled error in session loop:\n%s", traceback.format_exc())
        try:
            await _send_json(ws, {"type": "error", "message": "internal error"})
        except Exception:  # noqa: BLE001
            pass


async def _finalize(ws: WebSocket, state: SessionState) -> None:
    await _send_json(ws, {"type": "status", "message": "Transcribing audio..."})

    audio_blob = b"".join(state.audio_chunks)
    duration = max(time.time() - state.start_time, 1.0)
    transcript_text = ""
    transcript_lang = "unknown"

    if audio_blob:
        try:
            t = await asyncio.to_thread(
                get_transcriber().transcribe_bytes, audio_blob, ".webm"
            )
            transcript_text = t.text
            transcript_lang = t.language
            if t.duration_seconds > 0:
                duration = t.duration_seconds
        except Exception as e:  # noqa: BLE001
            log.error("Transcription failed: %s", e)
            await _send_json(
                ws, {"type": "status", "message": f"Transcription failed: {e}"}
            )

    audio_metrics = analyze_audio(transcript_text, duration)

    metrics = {
        "eye_contact_pct": state.eye_contact_pct(),
        "posture_breakdown": state.posture_counts,
        "frames_analyzed": state.frame_count,
        "duration_seconds": round(duration, 1),
        "language": transcript_lang,
        "transcript_excerpt": transcript_text[:500],
        **audio_metrics,
    }

    await _send_json(ws, {"type": "status", "message": "Generating Claude feedback..."})

    try:
        report = await asyncio.to_thread(feedback_generator.generate, metrics)
    except Exception as e:  # noqa: BLE001
        log.error("Claude feedback failed: %s", e)
        report = _fallback_report(metrics)

    await _send_json(ws, {"type": "report", "metrics": metrics, "report": report})
