"""Generate a structured interview feedback report via the Claude API.

One call per session, at the end. Uses claude-haiku-4-5 with prompt caching
on the system prompt and a JSON schema to constrain the output.
"""
from __future__ import annotations

import json
import os

import anthropic

MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = """You are InterviewIQ, an expert interview coach.

You receive objective metrics from a mock interview session:
- eye_contact_pct: percentage of analyzed frames where the candidate looked at the camera
- posture_breakdown: counts of frames classified as good / slouched / tilted / no_person
- wpm: words per minute over the full session (ideal: 130 WPM, acceptable: 110-160)
- filler_total and filler_breakdown: counted filler words (FR + EN)
- transcript_excerpt: the first ~500 characters of what was said
- duration_seconds: total session length

Produce honest, actionable, encouraging feedback. Be specific. Reference the
metrics directly. Tailor advice to whether the candidate spoke French or
English (use the transcript_excerpt as a signal). Keep the tone professional
and warm — never condescending.

Output strictly in the requested JSON schema."""

REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "global_score": {
            "type": "integer",
            "description": "Overall score from 0 to 100.",
        },
        "headline": {
            "type": "string",
            "description": "One-sentence summary of the candidate's performance.",
        },
        "strengths": {
            "type": "array",
            "description": "Three concrete strengths grounded in the metrics.",
            "items": {"type": "string"},
        },
        "improvements": {
            "type": "array",
            "description": "Three concrete areas to improve, grounded in the metrics.",
            "items": {"type": "string"},
        },
        "actionable_tip": {
            "type": "string",
            "description": "A single, very specific exercise the candidate can do before their next interview.",
        },
    },
    "required": ["global_score", "headline", "strengths", "improvements", "actionable_tip"],
    "additionalProperties": False,
}


class FeedbackGenerator:
    def __init__(self, api_key: str | None = None) -> None:
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def generate(self, metrics: dict) -> dict:
        user_payload = json.dumps(metrics, ensure_ascii=False, indent=2)
        response = self._client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Here are the metrics from the interview session. "
                        "Generate the feedback report as a JSON object "
                        "matching the schema exactly — no markdown fences.\n\n"
                        f"{user_payload}"
                    ),
                }
            ],
        )
        text = next((b.text for b in response.content if b.type == "text"), "{}")
        return json.loads(_strip_fences(text))


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _fallback_report(metrics: dict) -> dict:
    """Used when the Claude call fails — keeps the UI working."""
    wpm = metrics.get("wpm", 0)
    eye = metrics.get("eye_contact_pct", 0)
    return {
        "global_score": 50,
        "headline": "Report generation failed — showing fallback metrics.",
        "strengths": [
            f"Session completed ({metrics.get('duration_seconds', 0)}s of recording).",
            f"Eye contact: {eye}% of frames.",
            f"Speech pace: {wpm} WPM.",
        ],
        "improvements": [
            "Re-run the session to regenerate a full AI report.",
            "Check that ANTHROPIC_API_KEY is set on the backend.",
            "Review the backend logs for the underlying error.",
        ],
        "actionable_tip": "Try the session again once the backend is configured.",
    }


if __name__ == "__main__":
    sample = {
        "eye_contact_pct": 72,
        "posture_breakdown": {"good": 85, "slouched": 10, "tilted": 5, "no_person": 0},
        "wpm": 168,
        "filler_total": 7,
        "filler_breakdown": {"euh": 4, "du coup": 3},
        "transcript_excerpt": "Euh donc voilà, du coup je travaille beaucoup avec...",
        "duration_seconds": 95,
    }
    try:
        print(json.dumps(FeedbackGenerator().generate(sample), indent=2, ensure_ascii=False))
    except Exception as e:  # noqa: BLE001 — CLI smoke test
        print(f"Live call failed ({e}); fallback:")
        print(json.dumps(_fallback_report(sample), indent=2, ensure_ascii=False))
