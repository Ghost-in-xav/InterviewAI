"""Hand detection using MediaPipe Hands to find hand bounding boxes."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp

HAND_OVERLAP_MIN = 0.05


@dataclass
class HandDetectionResult:
    bboxes: list[dict]
    hands: list[dict]

    def to_dict(self) -> dict:
        return {"bboxes": self.bboxes, "hands": self.hands}


class HandDetector:
    def __init__(self) -> None:
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def analyze(self, frame_bgr) -> HandDetectionResult:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        bboxes: list[dict] = []
        hands: list[dict] = []
        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                xs = [lm.x for lm in hand.landmark]
                ys = [lm.y for lm in hand.landmark]
                min_x = max(min(xs), 0.0)
                max_x = min(max(xs), 1.0)
                min_y = max(min(ys), 0.0)
                max_y = min(max(ys), 1.0)
                w = max(0.0, max_x - min_x)
                h = max(0.0, max_y - min_y)
                if w > 0.0 and h > 0.0:
                    bbox = {
                        "x": round(min_x, 4),
                        "y": round(min_y, 4),
                        "w": round(w, 4),
                        "h": round(h, 4),
                    }
                    landmarks = [
                        [round(float(lm.x), 4), round(float(lm.y), 4)]
                        for lm in hand.landmark
                    ]
                    bboxes.append(bbox)
                    hands.append({"bbox": bbox, "landmarks": landmarks})
        return HandDetectionResult(bboxes=bboxes, hands=hands)

    def close(self) -> None:
        self._hands.close()


def hand_overlaps_face(
    face_bbox: dict | None, hand_bbox: dict, min_ratio: float = HAND_OVERLAP_MIN
) -> bool:
    if not face_bbox:
        return False

    fx1 = float(face_bbox.get("x", 0.0))
    fy1 = float(face_bbox.get("y", 0.0))
    fx2 = fx1 + float(face_bbox.get("w", 0.0))
    fy2 = fy1 + float(face_bbox.get("h", 0.0))

    hx1 = float(hand_bbox.get("x", 0.0))
    hy1 = float(hand_bbox.get("y", 0.0))
    hx2 = hx1 + float(hand_bbox.get("w", 0.0))
    hy2 = hy1 + float(hand_bbox.get("h", 0.0))

    inter_w = max(0.0, min(fx2, hx2) - max(fx1, hx1))
    inter_h = max(0.0, min(fy2, hy2) - max(fy1, hy1))
    inter_area = inter_w * inter_h
    face_area = max((fx2 - fx1) * (fy2 - fy1), 1e-6)
    return (inter_area / face_area) >= min_ratio
