# InterviewIQ — Technical Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Running the Project](#running-the-project)
4. [Backend](#backend)
   - [WebSocket Protocol](#websocket-protocol)
   - [Session State](#session-state)
   - [Vision Pipeline](#vision-pipeline)
   - [Audio Pipeline](#audio-pipeline)
   - [Feedback Generation](#feedback-generation)
5. [Vision Modules](#vision-modules)
   - [Eye Contact](#eye-contact)
   - [Posture](#posture)
   - [Face Lines](#face-lines)
   - [Face Baseline](#face-baseline)
   - [Face Occlusion](#face-occlusion)
   - [Motion Detector](#motion-detector)
   - [Hand Detector](#hand-detector)
   - [Object Detector](#object-detector)
6. [Frontend](#frontend)
7. [Dependencies & Pinning Notes](#dependencies--pinning-notes)
8. [Tuning Parameters](#tuning-parameters)

---

## Overview

InterviewIQ is a local AI Interview Coach. During a mock interview, it:

- Streams your webcam (1 FPS) and microphone to a FastAPI backend over WebSocket
- Runs a full computer-vision pipeline per frame (eye contact, posture, hand detection, face occlusion, object detection)
- Pushes live metrics back to the browser for a real-time dashboard
- After you click Stop, transcribes the full audio with Whisper and calls the Claude API once to generate a structured feedback report

Everything runs locally. No data is stored. The only external call is to the Anthropic Claude API at the end of each session.

---

## Architecture

```
Browser
  │
  │  WebSocket (ws://localhost:8000/ws/session)
  │  ┌── JSON envelope {"type":"frame_meta"} → binary JPEG (quality 60, ~1 FPS)
  │  ├── JSON envelope {"type":"audio_meta"} → binary webm/opus chunk
  │  └── JSON {"type":"stop"}
  │
  ▼
FastAPI Backend (port 8000)
  │
  ├── Vision pipeline (per frame, synchronous):
  │     EyeContactDetector → PostureDetector → FaceLinesDetector
  │     → HandDetector → MotionDetector → ObjectDetector → FaceOcclusionDetector
  │     → FaceBaselineTracker → off-center / too_much_angle override
  │
  ├── Push metrics JSON → browser (live dashboard update)
  │
  └── On stop:
        faster-whisper (base, int8, CPU) → transcript
        analyze() → WPM + filler counts
        Claude API (claude-haiku-4-5, one call) → structured report JSON
        Push report JSON → browser

Browser renders:
  - Canvas overlay: face bbox, eye points, hand skeleton, object bboxes (orange)
  - Warning state: red face bbox + label when face is masked or object overlaps face
  - Live metrics panel: eye contact %, posture status, face mask status, hands, motion, objects
  - Final report card: global score, headline, strengths, improvements, actionable tip
```

---

## Running the Project

### Prerequisites

- Docker + Docker Compose
- An `ANTHROPIC_API_KEY` in a `.env` file at the project root

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
```

### Start

```bash
docker-compose up --build
```

- Frontend: http://localhost:5173
- Backend health check: http://localhost:8000/health

### Force rebuild after code changes

```bash
docker-compose up --build
```

Docker caches layers — always use `--build` after editing Python files or `requirements.txt`.

### Standalone module testing

Each vision module has a `__main__` entry point:

```bash
python -m backend.vision.eye_contact path/to/image.jpg
python -m backend.vision.posture path/to/image.jpg
python -m backend.vision.face_lines path/to/image.jpg
python -m backend.audio.transcriber path/to/audio.wav
python -m backend.audio.analyzer       # runs with a built-in sample
python -m backend.feedback             # calls Claude with a sample payload
```

---

## Backend

### WebSocket Protocol

All communication goes through a single endpoint: `ws://localhost:8000/ws/session`

**Client → Server:**

Every binary payload is preceded by a JSON text frame (the "envelope"):

| Envelope | Binary payload | Description |
|---|---|---|
| `{"type": "frame_meta"}` | JPEG bytes | A video frame (quality 60) |
| `{"type": "audio_meta"}` | webm/opus bytes | An audio chunk from MediaRecorder |
| `{"type": "stop"}` | _(none)_ | Ends the session; triggers transcription + report |

**Server → Client:**

| Message | Fields | Description |
|---|---|---|
| `metrics` | eye_contact, posture, face_lines, face_mask, hands, motion, objects, frame_index, eye_contact_pct | Sent after each frame |
| `status` | message | Progress updates during finalization |
| `report` | metrics, report | Final report at end of session |
| `error` | message | Any error |

**Rate limiting:** The backend silently drops frames that arrive faster than `MIN_FRAME_INTERVAL_S = 0.9` seconds. The frontend also throttles to 1 FPS client-side.

### Session State

`SessionState` (dataclass in `main.py`) holds everything for one WebSocket connection:

| Field | Type | Description |
|---|---|---|
| `start_time` | float | Unix timestamp of session start |
| `last_frame_time` | float | For rate-limiting |
| `frame_count` | int | Total frames analyzed |
| `eye_looking_count` | int | Frames where gaze was detected |
| `eye_face_detected_count` | int | Frames with a face detected |
| `posture_counts` | dict | Counts per posture status |
| `audio_chunks` | list[bytes] | Accumulated webm/opus chunks |
| `face_baseline` | FaceBaselineTracker | Rolling average face bbox |
| `face_occlusion` | FaceOcclusionDetector | Stateful occlusion detector |
| `motion` | MotionDetector | Stateful motion detector (prev frame) |
| `object_detector` | ObjectDetector | Stateful rolling-background detector |

### Vision Pipeline

Per-frame order (synchronous, on the CPU):

```
frame_bgr (numpy array)
  │
  ├─ EyeContactDetector.analyze(frame)
  │    → eye: {looking, score, face_detected, face_bbox, eye_points, face_landmarks}
  │
  ├─ PostureDetector.analyze(frame)
  │    → posture: {status, shoulder_tilt_deg, head_offset, person_detected}
  │
  ├─ FaceLinesDetector.analyze(frame)
  │    → face_lines: {face_detected, status, roll_deg, too_much_angle, eye_gap,
  │                   eye_gap_status, landmark_confidence, low_conf_count,
  │                   upper_eye_line, lower_eye_line, midline}
  │
  ├─ HandDetector.analyze(frame)
  │    → hands: {bboxes: [...], hands: [{bbox, landmarks}, ...]}
  │    → hand_over_face: bool (IoU check with face_bbox)
  │
  ├─ MotionDetector.analyze(frame, face_bbox, face_landmarks)
  │    → motion: {motion_ratio, motion_over_face}
  │
  ├─ ObjectDetector.analyze(frame, face_bbox, hand_bboxes)
  │    → objects: {bboxes: [...], face_overlap: bool}
  │
  ├─ FaceOcclusionDetector.analyze(frame, face_bbox, hand_over_face,
  │                                landmark_confidence, face_landmarks,
  │                                low_conf_count, motion_over_face)
  │    → face_mask: {face_detected, masked, score, reason,
  │                  hand_over_face, motion_over_face, landmark_confidence}
  │
  └─ FaceBaselineTracker.update(face_bbox)  [first 30 frames only]
       → off_center: bool (compared against rolling average)
       → posture status overridden to "off_center" or "too_much_angle" if needed
```

### Audio Pipeline

1. The frontend sends webm/opus chunks continuously via `MediaRecorder` while recording.
2. Each chunk is appended to `state.audio_chunks`.
3. On stop, all chunks are concatenated into a single blob.
4. `Transcriber.transcribe_bytes(blob, ".webm")` writes it to a temp file and runs faster-whisper.
5. `analyze(text, duration)` computes WPM and filler word counts.

### Feedback Generation

`FeedbackGenerator.generate(metrics)` calls `claude-haiku-4-5` with:
- A constant system prompt (sent with `cache_control: ephemeral` for prompt caching)
- A user message containing the metrics JSON
- `max_tokens=1024`

The response is expected to be a JSON object matching `REPORT_SCHEMA`. `_strip_fences()` handles any accidental markdown code fences in the response. If the call fails, `_fallback_report()` returns a minimal report so the UI stays functional.

---

## Vision Modules

### Eye Contact

**File:** [backend/vision/eye_contact.py](backend/vision/eye_contact.py)

Uses MediaPipe FaceMesh with `refine_landmarks=True` (required for iris landmarks 468–477).

**Algorithm:**
1. Compute horizontal offset of each iris center from its eye-corner midpoint, normalized by eye width → `gaze_offset`
2. Compute horizontal offset of nose from cheek midpoint, normalized by face width → `yaw_offset` (head turn proxy)
3. Score = `min(gaze_score, yaw_score)` where each score decays linearly from 1.0
4. `looking = score >= 0.55`

**Output fields:**
- `looking`: bool
- `score`: 0–1 (1 = perfectly centered)
- `face_detected`: bool
- `face_bbox`: `{x, y, w, h}` normalized
- `eye_points`: `{left_iris, right_iris}` normalized coordinates
- `face_landmarks`: list of 478 `[x, y]` normalized points

### Posture

**File:** [backend/vision/posture.py](backend/vision/posture.py)

Uses MediaPipe Pose (model_complexity=1).

**Algorithm:**
1. Shoulder tilt: angle of the line between left/right shoulder landmarks vs horizontal
2. Head offset: horizontal distance of nose from shoulder midpoint, normalized by shoulder width
3. `status = "tilted"` if tilt > 8°, `"slouched"` if head_offset > 0.35, else `"good"`

Posture status can be overridden downstream to `"off_center"` or `"too_much_angle"`.

**Output fields:**
- `status`: `"good" | "slouched" | "tilted" | "no_person" | "off_center" | "too_much_angle"`
- `shoulder_tilt_deg`: float
- `head_offset`: float
- `person_detected`: bool
- `face_bbox`, `face_bbox_avg`, `off_center`, `too_much_angle` (added by main.py)

### Face Lines

**File:** [backend/vision/face_lines.py](backend/vision/face_lines.py)

Uses MediaPipe FaceMesh. Extracts geometric lines from the face.

**What it detects:**
- **Roll angle:** average of upper/lower eye-line angles → `too_much_angle` if > 12°
- **Eye gap:** vertical distance between upper and lower eyelids → `head_back` (too small) or `face_down` (too large)
- **Landmark confidence:** mean visibility/presence of key landmarks; counts how many are below 0.4

**Output fields:**
- `face_detected`, `status`, `roll_deg`, `too_much_angle`
- `eye_gap`, `eye_gap_status` (`"ok" | "head_back" | "face_down"`)
- `landmark_confidence` (0–1), `low_conf_count`
- `upper_eye_line`, `lower_eye_line`, `midline` (each: `{start, end, angle_deg}`)

### Face Baseline

**File:** [backend/vision/face_baseline.py](backend/vision/face_baseline.py)

A simple rolling-average tracker for the face bounding box. Accumulates the first `max_samples=30` face bboxes and computes their average. Used for off-center detection.

`is_off_center(current, avg, threshold=0.10)` returns `True` if the face center has moved more than 10% of the average face width or height from the baseline center.

### Face Occlusion

**File:** [backend/vision/face_occlusion.py](backend/vision/face_occlusion.py)

Detects whether the face is covered (hand, mask, paper, etc.).

**Algorithm:**
1. Crop the upper-face region using FaceMesh eye/forehead landmarks (falls back to full face_bbox)
2. Resize to a 64×64 grayscale patch normalized to [0, 1]
3. Build a running baseline from the first `max_samples=30` unoccluded frames
4. Each frame: compute mean absolute diff between current patch and baseline
5. Two-tier decision:
   - `diff > 0.18` → definitely occluded
   - `diff > 0.08` AND `low_conf_count >= 1` → likely occluded

Hand-over-face (detected via IoU with `hand_overlaps_face()`) is always flagged as `reason: "hand"` regardless of score. `motion_over_face` is passed through for display only — it is NOT used as a masking trigger (causes false positives on head movement).

**Output fields:**
- `face_detected`, `masked` (bool), `score` (diff value), `reason` (`"ok" | "hand" | "occlusion"`)
- `hand_over_face`, `motion_over_face`, `landmark_confidence`

### Motion Detector

**File:** [backend/vision/motion_detector.py](backend/vision/motion_detector.py)

Simple frame-to-frame pixel diff restricted to the face bounding box.

**Algorithm:**
1. Convert to grayscale + Gaussian blur (5×5)
2. `absdiff` with previous frame → threshold at 20 → binary map
3. Count changed pixels inside the face bbox ROI
4. `motion_ratio = changed_pixels / roi_area`
5. `motion_over_face = ratio >= 0.06`

**Output fields:**
- `motion_ratio`: float (0–1)
- `motion_over_face`: bool

### Hand Detector

**File:** [backend/vision/hand_detector.py](backend/vision/hand_detector.py)

Uses MediaPipe Hands (max_num_hands=2, model_complexity=0).

For each detected hand, computes the bounding box from landmark extents and stores all 21 landmark positions.

`hand_overlaps_face(face_bbox, hand_bbox, min_ratio=0.05)`: returns True if the intersection area is at least 5% of the face area (IoU-like check on the face side).

**Output fields:**
- `bboxes`: list of `{x, y, w, h}` (one per hand)
- `hands`: list of `{bbox, landmarks}` where landmarks is a list of 21 `[x, y]` points

**Frontend rendering:** Only the skeleton (connections between landmarks) is drawn. The bounding box is not rendered to avoid visual duplication with the object detector.

### Object Detector

**File:** [backend/vision/object_detector.py](backend/vision/object_detector.py)

Detects foreign objects appearing near the face during an interview (phone, notebook, notes, etc.).

**Algorithm:**
1. Convert frame to grayscale + Gaussian blur (7×7)
2. Update a rolling exponential background average: `bg = (1 - α) * bg + α * frame`
3. Compute absolute diff between current frame and background
4. Threshold at `DIFF_THRESHOLD=40` → binary change map
5. Restrict to the face bbox expanded by 50% on every side (eliminates background noise)
6. Morphological close (11×11 kernel) + dilate to merge nearby blobs
7. Find external contours; filter by minimum area (`MIN_AREA_RATIO=0.008` of frame)
8. Exclude blobs that overlap with any tracked hand bbox
9. Check whether any remaining blob overlaps the face bbox → `face_overlap`

**Key insight:** The rolling background absorbs stationary objects over time (`alpha=0.07` → ~40s half-life). A freshly placed or moving object will differ from the background and be detected. Once an object has been still for ~40 seconds, it gets absorbed into the background.

**Output fields:**
- `bboxes`: list of `{x, y, w, h}` for each detected object
- `face_overlap`: bool (true if any object bbox overlaps the face bbox)

**Tuning constants:**

| Constant | Default | Effect |
|---|---|---|
| `ALPHA` | 0.07 | Background learning rate. Lower = longer detection window, more ghosting |
| `DIFF_THRESHOLD` | 40 | Pixel intensity delta. Raise if too many false positives |
| `MIN_AREA_RATIO` | 0.008 | Min contour size (0.8% of frame). Raise to ignore small blobs |
| `FACE_MARGIN` | 0.5 | Search zone expansion. 0.5 = 50% extra on every side of face bbox |
| `WARMUP_FRAMES` | 5 | Frames before detection starts |

---

## Frontend

**File:** [frontend/src/app.jsx](frontend/src/app.jsx)

Single React component. Key responsibilities:

### WebSocket management
- Opens `ws://[VITE_WS_URL]/ws/session` on mount
- Sends `frame_meta` + JPEG and `audio_meta` + webm chunks during recording
- Sends `stop` and waits for `report` message

### Video capture
- `getUserMedia({video: true, audio: true})`
- Canvas-based JPEG encoding at quality 0.6: `canvas.toBlob(cb, "image/jpeg", 0.6)`
- 1 FPS via `setInterval`

### Canvas overlay (drawn on every metrics update)
- **Face bbox:** green normally, red if `face_mask.masked` or `objects.face_overlap`
- **Label:** "OBJECT DETECTED" or "HAND OVER FACE" on the face bbox when in warning state
- **Eye points:** cyan dots for left/right iris
- **Hand skeleton:** white lines connecting the 21 MediaPipe hand landmarks (no bounding box)
- **Object bboxes:** orange rectangles labeled "OBJECT" around each detected foreign object

### Metrics display
- Live panel: eye contact %, posture status (color-coded), face mask status, hand count, motion, objects
- Eye contact % is a running average sent by the backend (`eye_contact_pct`)
- Post-session: report card with global score, headline, strengths, improvements, actionable tip

---

## Dependencies & Pinning Notes

| Package | Version | Why pinned |
|---|---|---|
| `anthropic` | `==0.39.0` | `output_config` not supported; use `_strip_fences()` instead |
| `httpx` | `>=0.23.0,<0.28.0` | anthropic 0.39.0 passes `proxies` kwarg, removed in httpx 0.28 |
| `faster-whisper` | `==1.1.1` | Replaces `openai-whisper`; pre-built wheels, no build-time `pkg_resources` issues |
| `mediapipe` | `==0.10.14` | Tested against this version; newer may break landmark indices |
| `numpy` | `==1.26.4` | Required by mediapipe 0.10.14 (doesn't support numpy 2.x) |
| `opencv-python-headless` | `==4.10.0.84` | Headless variant for Docker (no GUI libs needed) |

**System packages required in Docker:**
- `ffmpeg` — required by faster-whisper for audio decoding
- `libgl1` — required by OpenCV
- `libglib2.0-0` — required by MediaPipe

---

## Tuning Parameters

### Eye contact
- `score >= 0.55` to count as "looking" (`eye_contact.py:102`)
- Gaze score decays to 0 at 25% iris offset from center (`/ 0.25`, line 99)
- Yaw score decays to 0 at 20% nose offset from face center (`/ 0.20`, line 100)

### Posture
- `tilt_deg > 8.0` → "tilted" (`posture.py:77`)
- `head_offset > 0.35` → "slouched" (`posture.py:79`)
- `is_off_center threshold = 0.10` (10% of face size, `main.py:213`)

### Face occlusion
- `DIFF_THRESHOLD = 0.08` moderate, `= 0.18` high (`face_occlusion.py:12–13`)
- `LANDMARK_CONFIDENCE_MIN = 0.55`, `CONF_LOW_THRESHOLD = 0.4`
- Baseline built from first `max_samples=30` clean frames

### Object detection
- See [Object Detector tuning table](#object-detector) above

### Motion
- `diff_threshold = 20` pixels, `motion_ratio_threshold = 0.06` (6% of face area)
