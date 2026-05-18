"""Detect foreign objects near the face using a rolling-average background model.

Strategy
--------
* Instead of a fixed baseline (which drifts and causes false positives), we
  keep a slow exponential running average of the scene.  New objects — moving
  or freshly placed — differ from that average and are detected as contours.
* We restrict the search zone to the face bounding box + a margin so background
  noise elsewhere in the frame cannot trigger false positives.
* Hands already tracked by HandDetector are excluded from the results.

Tuning notes
------------
ALPHA   – lower → background absorbs objects more slowly → longer detection
          window but more ghosting.  At 0.07 a stationary object is absorbed
          in ~40 seconds (1/0.07 ≈ 14 frames ≈ 14 s half-life at 1 FPS).
THRESHOLD – pixel intensity delta to consider "changed".  40 works well for
            typical webcam noise; raise it if you get too many false positives.
FACE_MARGIN – how much to expand the face bbox for the search zone.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

ALPHA = 0.07            # background learning rate
DIFF_THRESHOLD = 40     # intensity units (0-255)
MIN_AREA_RATIO = 0.008  # contour must be ≥ 0.8 % of frame area
FACE_MARGIN = 0.5       # expand face bbox by 50 % on every side
WARMUP_FRAMES = 5       # don't detect until background is stable


def _boxes_overlap(a: dict, b: dict) -> bool:
    ax2 = a["x"] + a["w"]
    ay2 = a["y"] + a["h"]
    bx2 = b["x"] + b["w"]
    by2 = b["y"] + b["h"]
    return a["x"] < bx2 and ax2 > b["x"] and a["y"] < by2 and ay2 > b["y"]


@dataclass
class ObjectDetectionResult:
    bboxes: list[dict] = field(default_factory=list)
    face_overlap: bool = False

    def to_dict(self) -> dict:
        return {"bboxes": self.bboxes, "face_overlap": self.face_overlap}


class ObjectDetector:
    def __init__(
        self,
        alpha: float = ALPHA,
        diff_threshold: int = DIFF_THRESHOLD,
        min_area_ratio: float = MIN_AREA_RATIO,
        face_margin: float = FACE_MARGIN,
        warmup_frames: int = WARMUP_FRAMES,
    ) -> None:
        self._alpha = alpha
        self._diff_threshold = diff_threshold
        self._min_area_ratio = min_area_ratio
        self._face_margin = face_margin
        self._warmup = warmup_frames
        self._background: np.ndarray | None = None
        self._count = 0

    def reset(self) -> None:
        self._background = None
        self._count = 0

    def analyze(
        self,
        frame_bgr: np.ndarray,
        face_bbox: dict | None = None,
        hand_bboxes: list[dict] | None = None,
    ) -> ObjectDetectionResult:
        gray = cv2.GaussianBlur(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), (7, 7), 0
        )
        h, w = gray.shape[:2]

        # — update rolling background every frame —
        if self._background is None:
            self._background = gray.astype(np.float32)
        else:
            self._background = (
                (1.0 - self._alpha) * self._background
                + self._alpha * gray.astype(np.float32)
            )
        self._count += 1

        # Not enough frames yet for a stable background
        if self._count <= self._warmup:
            return ObjectDetectionResult()

        # — diff against recent background —
        diff = cv2.absdiff(gray, self._background.astype(np.uint8))
        _, thresh = cv2.threshold(diff, self._diff_threshold, 255, cv2.THRESH_BINARY)

        # Restrict to face zone + margin (ignore the rest of the frame)
        if face_bbox:
            m = self._face_margin
            x1 = max(0, int((face_bbox["x"] - face_bbox["w"] * m) * w))
            y1 = max(0, int((face_bbox["y"] - face_bbox["h"] * m) * h))
            x2 = min(w, int((face_bbox["x"] + face_bbox["w"] * (1.0 + m)) * w))
            y2 = min(h, int((face_bbox["y"] + face_bbox["h"] * (1.0 + m)) * h))
            mask = np.zeros_like(thresh)
            mask[y1:y2, x1:x2] = 255
            thresh = cv2.bitwise_and(thresh, mask)

        # Morphological cleanup: close small holes, merge nearby blobs
        kernel = np.ones((11, 11), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = h * w * self._min_area_ratio

        bboxes: list[dict] = []
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            bbox = {
                "x": round(cx / w, 4),
                "y": round(cy / h, 4),
                "w": round(cw / w, 4),
                "h": round(ch / h, 4),
            }
            # Skip blobs that are just a tracked hand
            if hand_bboxes and any(_boxes_overlap(bbox, hb) for hb in hand_bboxes):
                continue
            bboxes.append(bbox)

        face_overlap = bool(
            face_bbox and any(_boxes_overlap(b, face_bbox) for b in bboxes)
        )
        return ObjectDetectionResult(bboxes=bboxes, face_overlap=face_overlap)
