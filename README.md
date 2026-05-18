# InterviewIQ

AI Interview Coach MVP — analyzes a mock interview from your webcam + mic and
returns live metrics (eye contact, posture) plus a post-session report (WPM,
filler words, Claude-generated feedback).

## Stack

Python 3.11 + FastAPI + WebSockets, MediaPipe, Whisper (base), React + Vite +
Tailwind, Anthropic Claude API (`claude-haiku-4-5`).

## Quick start (Docker)

```bash
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY
docker-compose up --build
```

Open http://localhost:5173.

## Quick start (local dev, no Docker)

Backend:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# ffmpeg is required for audio decoding (Whisper)
brew install ffmpeg            # macOS
# apt-get install -y ffmpeg    # Debian/Ubuntu
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn backend.main:app --reload --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Then open http://localhost:5173.

## How to use

1. Click **Start Interview** — grant webcam + mic permission.
2. Speak as if you were in a real interview.
3. Watch the live dashboard (eye contact %, posture, frame count).
4. Click **Stop** — wait a few seconds while Whisper transcribes and Claude
   writes your report.
5. Read the score and the actionable feedback.

## Testing modules independently

```bash
# Eye contact on a single image
python -m backend.vision.eye_contact path/to/face.jpg

# Posture on a single image
python -m backend.vision.posture path/to/person.jpg

# Whisper transcription on a wav file
python -m backend.audio.transcriber path/to/audio.wav

# Filler-word / WPM analyzer on a sample text
python -m backend.audio.analyzer
```

## Project layout

See [CLAUDE.md](CLAUDE.md).
