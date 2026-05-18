"""Simple motion detection inside the face region."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DEFAULT_DIFF_THRESHOLD = 20
DEFAULT_MOTION_RATIO_THRESHOLD = 0.06


@dataclass
class MotionDetectionResult:
    motion_ratio: float
    motion_over_face: bool

    def to_dict(self) -> dict:
        return {
            "motion_ratio": round(self.motion_ratio, 3),
            "motion_over_face": self.motion_over_face,
        }


class MotionDetector:
    def __init__(self) -> None:
        self._prev_gray: np.ndarray | None = None

    def reset(self) -> None:
        self._prev_gray = None

    def analyze(
        self,
        frame_bgr: np.ndarray,
        face_bbox: dict | None,
        *,
        face_landmarks: list[list[float]] | None = None,
        diff_threshold: int = DEFAULT_DIFF_THRESHOLD,
        motion_ratio_threshold: float = DEFAULT_MOTION_RATIO_THRESHOLD,
    ) -> MotionDetectionResult:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return MotionDetectionResult(motion_ratio=0.0, motion_over_face=False)

        diff = cv2.absdiff(gray, self._prev_gray)
        _, thresh = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
        self._prev_gray = gray

        if face_bbox is None:
            return MotionDetectionResult(motion_ratio=0.0, motion_over_face=False)

        h, w = gray.shape[:2]
        x = max(int(face_bbox.get("x", 0.0) * w), 0)
        y = max(int(face_bbox.get("y", 0.0) * h), 0)
        bw = int(face_bbox.get("w", 0.0) * w)
        bh = int(face_bbox.get("h", 0.0) * h)

        x2 = min(x + bw, w)
        y2 = min(y + bh, h)
        if x2 <= x or y2 <= y:
            return MotionDetectionResult(motion_ratio=0.0, motion_over_face=False)

        roi = thresh[y:y2, x:x2]
        motion_pixels = float(np.count_nonzero(roi))
        area = float(roi.size) if roi.size else 1.0
        ratio = motion_pixels / area
        return MotionDetectionResult(
            motion_ratio=ratio,
            motion_over_face=ratio >= motion_ratio_threshold,
        )
