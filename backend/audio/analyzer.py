"""Speech analysis: words-per-minute and filler-word detection.

Filler patterns cover French and English. Multi-word fillers ("du coup", "you
know") are matched as phrases; single tokens are matched against word
boundaries to avoid false positives like "like" inside "alike".
"""

from __future__ import annotations

import re

FILLERS = [
    # French
    "euh",
    "heu",
    "ben",
    "du coup",
    "genre",
    "en fait",
    "voilà",
    "tu vois",
    "quoi",
    # English
    "um",
    "uh",
    "like",
    "you know",
    "actually",
    "basically",
    "literally",
    "i mean",
    "sort of",
    "kind of",
]


def _normalize(text: str) -> str:
    return text.lower().strip()


def count_filler_words(text: str) -> dict:
    norm = _normalize(text)
    breakdown: dict[str, int] = {}
    for filler in FILLERS:
        if " " in filler:
            count = len(re.findall(rf"\b{re.escape(filler)}\b", norm))
        else:
            count = len(re.findall(rf"\b{re.escape(filler)}\b", norm))
        if count:
            breakdown[filler] = count
    total = sum(breakdown.values())
    return {"total": total, "breakdown": breakdown}


def compute_wpm(text: str, duration_seconds: float) -> float:
    if duration_seconds <= 0:
        return 0.0
    words = [w for w in re.findall(r"\b\w+\b", text) if w]
    return round(len(words) / (duration_seconds / 60.0), 1)


def analyze(text: str, duration_seconds: float) -> dict:
    fillers = count_filler_words(text)
    wpm = compute_wpm(text, duration_seconds)
    return {
        "wpm": wpm,
        "filler_total": fillers["total"],
        "filler_breakdown": fillers["breakdown"],
        "word_count": len(re.findall(r"\b\w+\b", text)),
        "duration_seconds": round(duration_seconds, 2),
    }


if __name__ == "__main__":
    sample = (
        "Euh donc voilà, du coup je travaille beaucoup avec genre des équipes "
        "produit, en fait c'est super intéressant tu vois, like a lot, you know."
    )
    print(analyze(sample, duration_seconds=15.0))
