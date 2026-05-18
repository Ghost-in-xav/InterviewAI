"""Track the average face box for the first N frames."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FaceBaselineTracker:
    max_samples: int = 30
    count: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_w: float = 0.0
    sum_h: float = 0.0

    def reset(self) -> None:
        self.count = 0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_w = 0.0
        self.sum_h = 0.0

    def update(self, face_bbox: dict | None) -> None:
        if not face_bbox or self.count >= self.max_samples:
            return
        self.sum_x += float(face_bbox.get("x", 0.0))
        self.sum_y += float(face_bbox.get("y", 0.0))
        self.sum_w += float(face_bbox.get("w", 0.0))
        self.sum_h += float(face_bbox.get("h", 0.0))
        self.count += 1

    def average(self) -> dict | None:
        if self.count == 0:
            return None
        return {
            "x": round(self.sum_x / self.count, 4),
            "y": round(self.sum_y / self.count, 4),
            "w": round(self.sum_w / self.count, 4),
            "h": round(self.sum_h / self.count, 4),
        }


def is_off_center(current: dict | None, avg: dict | None, threshold: float = 0.30) -> bool:
    if not current or not avg:
        return False

    cur_cx = float(current.get("x", 0.0)) + float(current.get("w", 0.0)) / 2
    cur_cy = float(current.get("y", 0.0)) + float(current.get("h", 0.0)) / 2
    avg_cx = float(avg.get("x", 0.0)) + float(avg.get("w", 0.0)) / 2
    avg_cy = float(avg.get("y", 0.0)) + float(avg.get("h", 0.0)) / 2

    avg_w = max(float(avg.get("w", 0.0)), 1e-6)
    avg_h = max(float(avg.get("h", 0.0)), 1e-6)

    norm_dx = abs(cur_cx - avg_cx) / avg_w
    norm_dy = abs(cur_cy - avg_cy) / avg_h
    return norm_dx > threshold or norm_dy > threshold
