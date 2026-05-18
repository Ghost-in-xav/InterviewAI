"""Heuristic face-occlusion detection using an upper-face baseline crop."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DEFAULT_PATCH_SIZE = (64, 64)
DEFAULT_MAX_SAMPLES = 30
DEFAULT_OCCLUSION_THRESHOLD = 0.08
DEFAULT_OCCLUSION_THRESHOLD_HIGH = 0.18  # high diff alone → definitely occluded
DEFAULT_LANDMARK_CONFIDENCE_MIN = 0.55
UPPER_FACE_TOP = (10,)
UPPER_FACE_BOTTOM = (145, 374)
UPPER_FACE_LEFT = (33, 159, 145)
UPPER_FACE_RIGHT = (263, 386, 374)


@dataclass
class FaceOcclusionResult:
    face_detected: bool
    masked: bool
    score: float
    reason: str
    hand_over_face: bool
    motion_over_face: bool
    landmark_confidence: float

    def to_dict(self) -> dict:
        return {
            "face_detected": self.face_detected,
            "masked": self.masked,
            "score": round(self.score, 3),
            "reason": self.reason,
            "hand_over_face": self.hand_over_face,
            "motion_over_face": self.motion_over_face,
            "landmark_confidence": round(self.landmark_confidence, 3),
        }


class FaceOcclusionDetector:
    def __init__(
        self,
        max_samples: int = DEFAULT_MAX_SAMPLES,
        threshold: float = DEFAULT_OCCLUSION_THRESHOLD,
        min_landmark_confidence: float = DEFAULT_LANDMARK_CONFIDENCE_MIN,
        patch_size: tuple[int, int] = DEFAULT_PATCH_SIZE,
    ) -> None:
        self._max_samples = max_samples
        self._threshold = threshold
        self._min_landmark_confidence = min_landmark_confidence
        self._patch_size = patch_size
        self._count = 0
        self._baseline: np.ndarray | None = None

    def reset(self) -> None:
        self._count = 0
        self._baseline = None

    def analyze(
        self,
        frame_bgr: np.ndarray,
        face_bbox: dict | None,
        *,
        hand_over_face: bool = False,
        motion_over_face: bool = False,
        landmark_confidence: float | None = None,
        face_landmarks: list[list[float]] | None = None,
        low_conf_count: int | None = None,
    ) -> FaceOcclusionResult:
        patch = self._crop_face_patch(frame_bgr, face_bbox, face_landmarks)
        if patch is None:
            return FaceOcclusionResult(
                face_detected=False,
                masked=False,
                score=0.0,
                reason="no_face",
                hand_over_face=False,
                motion_over_face=False,
                landmark_confidence=0.0,
            )

        conf = 1.0 if landmark_confidence is None else float(landmark_confidence)
        masked_by_hand = bool(hand_over_face)
        masked_by_motion = bool(motion_over_face)
        low_count = int(low_conf_count or 0)

        if self._baseline is None and self._count < self._max_samples:
            masked = masked_by_hand
            reason = "hand" if masked_by_hand else "ok"
            if not masked:
                self._baseline = patch.copy()
                self._count += 1
            return FaceOcclusionResult(
                face_detected=True,
                masked=masked,
                score=0.0,
                reason=reason,
                hand_over_face=masked_by_hand,
                motion_over_face=masked_by_motion,
                landmark_confidence=conf,
            )

        if self._baseline is None:
            return FaceOcclusionResult(
                face_detected=True,
                masked=masked_by_hand,
                score=0.0,
                reason="hand" if masked_by_hand else "ok",
                hand_over_face=masked_by_hand,
                motion_over_face=masked_by_motion,
                landmark_confidence=conf,
            )

        diff = float(np.mean(np.abs(patch - self._baseline)))
        # Two-tier: very high diff alone means something is clearly covering the face;
        # moderate diff + any landmark degradation also flags occlusion.
        masked_by_score = diff > DEFAULT_OCCLUSION_THRESHOLD_HIGH or (
            diff > self._threshold and low_count >= 1
        )
        masked = masked_by_hand or masked_by_score
        reason = "hand" if masked_by_hand else "occlusion" if masked_by_score else "ok"
        if self._count < self._max_samples and not masked:
            alpha = 1.0 / (self._count + 1)
            self._baseline = (1.0 - alpha) * self._baseline + alpha * patch
            self._count += 1

        return FaceOcclusionResult(
            face_detected=True,
            masked=masked,
            score=diff,
            reason=reason,
            hand_over_face=masked_by_hand,
            motion_over_face=masked_by_motion,
            landmark_confidence=conf,
        )

    def _crop_face_patch(
        self,
        frame_bgr: np.ndarray,
        face_bbox: dict | None,
        face_landmarks: list[list[float]] | None,
    ) -> np.ndarray | None:
        upper_bbox = self._upper_face_bbox(face_landmarks, face_bbox)
        if upper_bbox is None:
            return None

        h, w = frame_bgr.shape[:2]
        x = max(int(upper_bbox.get("x", 0.0) * w), 0)
        y = max(int(upper_bbox.get("y", 0.0) * h), 0)
        bw = int(upper_bbox.get("w", 0.0) * w)
        bh = int(upper_bbox.get("h", 0.0) * h)

        if bw <= 1 or bh <= 1:
            return None

        x2 = min(x + bw, w)
        y2 = min(y + bh, h)
        if x2 <= x or y2 <= y:
            return None

        roi = frame_bgr[y:y2, x:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self._patch_size, interpolation=cv2.INTER_AREA)
        return (resized.astype(np.float32) / 255.0)

    def _upper_face_bbox(
        self,
        face_landmarks: list[list[float]] | None,
        face_bbox: dict | None,
    ) -> dict | None:
        if not face_landmarks:
            return face_bbox

        def pick(indices: tuple[int, ...]) -> list[list[float]]:
            pts: list[list[float]] = []
            for idx in indices:
                if idx < len(face_landmarks):
                    pts.append(face_landmarks[idx])
            return pts

        left_pts = pick(UPPER_FACE_LEFT)
        right_pts = pick(UPPER_FACE_RIGHT)
        top_pts = pick(UPPER_FACE_TOP)
        bottom_pts = pick(UPPER_FACE_BOTTOM)

        if not left_pts or not right_pts or not top_pts or not bottom_pts:
            return face_bbox

        left_x = min(p[0] for p in left_pts)
        right_x = max(p[0] for p in right_pts)
        top_y = min(p[1] for p in top_pts)
        bottom_y = max(p[1] for p in bottom_pts)

        pad_x = (right_x - left_x) * 0.08
        pad_y = (bottom_y - top_y) * 0.12

        x1 = min(1.0, max(0.0, left_x - pad_x))
        x2 = min(1.0, max(0.0, right_x + pad_x))
        y1 = min(1.0, max(0.0, top_y - pad_y))
        y2 = min(1.0, max(0.0, bottom_y + pad_y))

        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        if w <= 0.0 or h <= 0.0:
            return face_bbox

        return {
            "x": round(x1, 4),
            "y": round(y1, 4),
            "w": round(w, 4),
            "h": round(h, 4),
        }
