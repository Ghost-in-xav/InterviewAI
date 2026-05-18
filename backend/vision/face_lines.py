"""Face line extraction and inclination estimation using MediaPipe FaceMesh."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

# FaceMesh landmark indices
UPPER_EYE_LINE = (159, 386)  # upper eyelids (left, right)
LOWER_EYE_LINE = (145, 374)  # lower eyelids (left, right)
MIDLINE = (10, 152)  # forehead to chin
MAX_ROLL_DEG = 12.0
EYE_GAP_MIN = 0.018
EYE_GAP_MAX = 0.050
CONF_LANDMARKS = (10, 33, 263, 159, 145, 386, 374, 1)
CONF_LOW_THRESHOLD = 0.4


@dataclass
class FaceLinesResult:
    face_detected: bool
    status: str
    roll_deg: float
    too_much_angle: bool
    eye_gap: float
    eye_gap_status: str
    landmark_confidence: float
    low_conf_count: int
    upper_eye_line: dict | None
    lower_eye_line: dict | None
    midline: dict | None

    def to_dict(self) -> dict:
        return {
            "face_detected": self.face_detected,
            "status": self.status,
            "roll_deg": round(self.roll_deg, 2),
            "too_much_angle": self.too_much_angle,
            "eye_gap": round(self.eye_gap, 4),
            "eye_gap_status": self.eye_gap_status,
            "landmark_confidence": round(self.landmark_confidence, 3),
            "low_conf_count": self.low_conf_count,
            "upper_eye_line": self.upper_eye_line,
            "lower_eye_line": self.lower_eye_line,
            "midline": self.midline,
        }


class FaceLinesDetector:
    def __init__(self) -> None:
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def analyze(self, frame_bgr: np.ndarray) -> FaceLinesResult:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            return FaceLinesResult(
                face_detected=False,
                status="no_face",
                roll_deg=0.0,
                too_much_angle=False,
                eye_gap=0.0,
                eye_gap_status="no_face",
                landmark_confidence=0.0,
                low_conf_count=0,
                upper_eye_line=None,
                lower_eye_line=None,
                midline=None,
            )

        lm = result.multi_face_landmarks[0].landmark

        def clamp01(value: float) -> float:
            return min(1.0, max(0.0, float(value)))

        def pt(idx: int) -> tuple[float, float]:
            return clamp01(lm[idx].x), clamp01(lm[idx].y)

        def line(a: tuple[float, float], b: tuple[float, float]) -> dict:
            ax, ay = a
            bx, by = b
            dx = bx - ax
            dy = by - ay
            angle = float(np.degrees(np.arctan2(dy, dx)))
            return {
                "start": [round(ax, 4), round(ay, 4)],
                "end": [round(bx, 4), round(by, 4)],
                "angle_deg": round(angle, 2),
            }

        def normalize_roll(angle: float) -> float:
            while angle > 90:
                angle -= 180
            while angle < -90:
                angle += 180
            return angle

        def landmark_confidence(indices: tuple[int, ...]) -> tuple[float, int]:
            values: list[float] = []
            for idx in indices:
                lm_i = lm[idx]
                vis = getattr(lm_i, "visibility", None)
                pres = getattr(lm_i, "presence", None)
                if vis is not None:
                    values.append(float(vis))
                elif pres is not None:
                    values.append(float(pres))
            if values:
                avg = float(sum(values) / len(values))
                low_count = sum(1 for v in values if v < CONF_LOW_THRESHOLD)
                return avg, low_count
            return 1.0, 0

        upper_a = pt(UPPER_EYE_LINE[0])
        upper_b = pt(UPPER_EYE_LINE[1])
        lower_a = pt(LOWER_EYE_LINE[0])
        lower_b = pt(LOWER_EYE_LINE[1])

        upper_line = line(upper_a, upper_b)
        lower_line = line(lower_a, lower_b)
        mid_line = line(pt(MIDLINE[0]), pt(MIDLINE[1]))

        roll = normalize_roll((upper_line["angle_deg"] + lower_line["angle_deg"]) / 2)
        too_much_angle = abs(roll) > MAX_ROLL_DEG
        status = "too_much_angle" if too_much_angle else "ok"

        upper_mid_y = (upper_a[1] + upper_b[1]) / 2
        lower_mid_y = (lower_a[1] + lower_b[1]) / 2
        eye_gap = abs(lower_mid_y - upper_mid_y)
        if eye_gap < EYE_GAP_MIN:
            eye_gap_status = "head_back"
        elif eye_gap > EYE_GAP_MAX:
            eye_gap_status = "face_down"
        else:
            eye_gap_status = "ok"

        conf, low_conf_count = landmark_confidence(CONF_LANDMARKS)

        return FaceLinesResult(
            face_detected=True,
            status=status,
            roll_deg=roll,
            too_much_angle=too_much_angle,
            eye_gap=eye_gap,
            eye_gap_status=eye_gap_status,
            landmark_confidence=conf,
            low_conf_count=low_conf_count,
            upper_eye_line=upper_line,
            lower_eye_line=lower_line,
            midline=mid_line,
        )

    def close(self) -> None:
        self._mesh.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python -m backend.vision.face_lines <image_path>")
        raise SystemExit(2)

    image = cv2.imread(sys.argv[1])
    if image is None:
        print(f"could not read {sys.argv[1]}")
        raise SystemExit(1)

    detector = FaceLinesDetector()
    print(detector.analyze(image).to_dict())
    detector.close()
