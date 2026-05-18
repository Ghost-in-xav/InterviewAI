"""Posture detection using MediaPipe Pose.

Heuristics:
- "tilted" if the shoulder line angle exceeds ~8 degrees from horizontal.
- "slouched" if the head (nose) sits significantly forward of the shoulder
  midpoint in normalized coordinates.
- "good" otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
NOSE = 0


@dataclass
class PostureResult:
    status: str  # "good" | "slouched" | "tilted" | "no_person"
    shoulder_tilt_deg: float
    head_offset: float  # normalized horizontal offset of nose from shoulder midpoint
    person_detected: bool

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "shoulder_tilt_deg": round(self.shoulder_tilt_deg, 2),
            "head_offset": round(self.head_offset, 3),
            "person_detected": self.person_detected,
        }


class PostureDetector:
    def __init__(self) -> None:
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def analyze(self, frame_bgr: np.ndarray) -> PostureResult:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)
        if not result.pose_landmarks:
            return PostureResult(
                status="no_person",
                shoulder_tilt_deg=0.0,
                head_offset=0.0,
                person_detected=False,
            )

        lm = result.pose_landmarks.landmark
        ls = np.array([lm[LEFT_SHOULDER].x, lm[LEFT_SHOULDER].y])
        rs = np.array([lm[RIGHT_SHOULDER].x, lm[RIGHT_SHOULDER].y])
        nose = np.array([lm[NOSE].x, lm[NOSE].y])

        # Shoulder line tilt: angle vs horizontal.
        dx = ls[0] - rs[0]
        dy = ls[1] - rs[1]
        tilt_deg = float(abs(np.degrees(np.arctan2(dy, dx))))
        # arctan2 gives an angle near 180 (or -180) for a horizontal line going
        # left-to-right in image coords, so fold into [0, 90].
        if tilt_deg > 90:
            tilt_deg = abs(180 - tilt_deg)

        shoulder_mid_x = (ls[0] + rs[0]) / 2
        shoulder_width = max(abs(ls[0] - rs[0]), 1e-3)
        head_offset = float(abs(nose[0] - shoulder_mid_x) / shoulder_width)

        if tilt_deg > 8.0:
            status = "tilted"
        elif head_offset > 0.35:
            status = "slouched"
        else:
            status = "good"

        return PostureResult(
            status=status,
            shoulder_tilt_deg=tilt_deg,
            head_offset=head_offset,
            person_detected=True,
        )

    def close(self) -> None:
        self._pose.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python -m backend.vision.posture <image_path>")
        raise SystemExit(2)

    image = cv2.imread(sys.argv[1])
    if image is None:
        print(f"could not read {sys.argv[1]}")
        raise SystemExit(1)

    detector = PostureDetector()
    print(detector.analyze(image).to_dict())
    detector.close()
