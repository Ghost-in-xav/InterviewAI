"""Eye-contact / gaze estimation using MediaPipe FaceMesh iris landmarks.

A frame is considered "looking at camera" when both irises sit close enough
to the center of their respective eye sockets on the horizontal axis AND the
face is roughly frontal (small yaw).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

# FaceMesh landmark indices
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE_CORNERS = (33, 133)  # outer, inner
RIGHT_EYE_CORNERS = (362, 263)  # inner, outer
NOSE_TIP = 1
LEFT_CHEEK = 234
RIGHT_CHEEK = 454


@dataclass
class EyeContactResult:
    looking: bool
    score: float  # 0..1, how centered the gaze is
    face_detected: bool
    face_bbox: dict | None = None
    eye_points: dict | None = None
    face_landmarks: list[list[float]] | None = None

    def to_dict(self) -> dict:
        return {
            "looking": self.looking,
            "score": round(self.score, 3),
            "face_detected": self.face_detected,
            "face_bbox": self.face_bbox,
            "eye_points": self.eye_points,
            "face_landmarks": self.face_landmarks,
        }


class EyeContactDetector:
    def __init__(self) -> None:
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,  # required for iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def analyze(self, frame_bgr: np.ndarray) -> EyeContactResult:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            return EyeContactResult(
                looking=False,
                score=0.0,
                face_detected=False,
                face_bbox=None,
                eye_points=None,
                face_landmarks=None,
            )

        lm = result.multi_face_landmarks[0].landmark

        def clamp01(value: float) -> float:
            return min(1.0, max(0.0, float(value)))

        def px(idx: int) -> np.ndarray:
            return np.array([lm[idx].x * w, lm[idx].y * h])

        # Horizontal gaze: iris center vs eye-corner midpoint, normalized by eye width.
        left_iris = np.mean([px(i) for i in LEFT_IRIS], axis=0)
        right_iris = np.mean([px(i) for i in RIGHT_IRIS], axis=0)
        l_outer, l_inner = px(LEFT_EYE_CORNERS[0]), px(LEFT_EYE_CORNERS[1])
        r_inner, r_outer = px(RIGHT_EYE_CORNERS[0]), px(RIGHT_EYE_CORNERS[1])

        left_center = (l_outer + l_inner) / 2
        right_center = (r_inner + r_outer) / 2
        left_width = max(np.linalg.norm(l_inner - l_outer), 1e-3)
        right_width = max(np.linalg.norm(r_outer - r_inner), 1e-3)

        left_offset = abs((left_iris[0] - left_center[0]) / left_width)
        right_offset = abs((right_iris[0] - right_center[0]) / right_width)
        gaze_offset = (left_offset + right_offset) / 2  # 0 = centered

        # Head yaw proxy: nose horizontal position relative to cheek midpoint.
        nose = px(NOSE_TIP)
        cheek_mid = (px(LEFT_CHEEK) + px(RIGHT_CHEEK)) / 2
        face_width = max(np.linalg.norm(px(RIGHT_CHEEK) - px(LEFT_CHEEK)), 1e-3)
        yaw_offset = abs(nose[0] - cheek_mid[0]) / face_width

        # Score: 1 when both offsets are 0, decays linearly.
        gaze_score = max(0.0, 1.0 - gaze_offset / 0.25)
        yaw_score = max(0.0, 1.0 - yaw_offset / 0.20)
        score = float(min(gaze_score, yaw_score))
        looking = score >= 0.55

        xs = [p.x for p in lm]
        ys = [p.y for p in lm]
        min_x = clamp01(min(xs))
        max_x = clamp01(max(xs))
        min_y = clamp01(min(ys))
        max_y = clamp01(max(ys))

        face_bbox = {
            "x": round(min_x, 4),
            "y": round(min_y, 4),
            "w": round(max_x - min_x, 4),
            "h": round(max_y - min_y, 4),
        }

        eye_points = {
            "left_iris": [
                round(clamp01(left_iris[0] / w), 4),
                round(clamp01(left_iris[1] / h), 4),
            ],
            "right_iris": [
                round(clamp01(right_iris[0] / w), 4),
                round(clamp01(right_iris[1] / h), 4),
            ],
        }

        face_landmarks = [[round(clamp01(p.x), 4), round(clamp01(p.y), 4)] for p in lm]

        return EyeContactResult(
            looking=looking,
            score=score,
            face_detected=True,
            face_bbox=face_bbox,
            eye_points=eye_points,
            face_landmarks=face_landmarks,
        )

    def close(self) -> None:
        self._mesh.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python -m backend.vision.eye_contact <image_path>")
        raise SystemExit(2)

    image = cv2.imread(sys.argv[1])
    if image is None:
        print(f"could not read {sys.argv[1]}")
        raise SystemExit(1)

    detector = EyeContactDetector()
    print(detector.analyze(image).to_dict())
    detector.close()
