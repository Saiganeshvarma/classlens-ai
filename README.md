# ClassLens AI

**AI-Powered Online Class Quality Evaluator**

ClassLens AI analyses a video or audio recording of an online class and produces a structured quality assessment — rating, dimension scores, evidence-backed observations, and improvement recommendations.

Built for the Humaniti AI recruitment challenge.

---

## Features

- Upload a class recording (video or audio) or analyse a file directly
- Full processing pipeline: validation → audio extraction → transcription → audio analysis → frame sampling → AI evaluation → scoring
- Five-dimension rubric with configurable weights (Teaching Clarity 30%, Engagement 25%, Lesson Structure 20%, Audio Intelligibility 15%, Visual Support 10%)
- Ratings: Poor / Average / Good / Excellent (score 0–100)
- Every observation is tied to real transcript evidence, audio measurements, or video frames — no fabricated claims
- Python scoring engine enforces weights and thresholds independently of the AI model
- Downloadable JSON report
- Clean Streamlit UI with dimension score chart and collapsible evidence cards

---

## Architecture

```
app.py                  Streamlit UI + pipeline orchestration
config.py               Settings (pydantic-settings, .env)
schemas.py              Pydantic models for all data structures
video_processor.py      Validation, audio extraction, frame sampling (FFmpeg + OpenCV)
audio_analyzer.py       Whisper transcription + local audio feature analysis
evaluator.py            LLM evaluation prompt, API calls, response parsing
scoring.py              Python weighted scoring engine (no LLM involvement)
report_generator.py     Report assembly, JSON serialisation, UI helpers
tests/                  128 pytest unit tests (no API keys required)
docs/                   Architecture, methodology, submission summary
```

Data flows in one direction:

```
Upload → validate_recording()
       → extract_audio()        [FFmpeg]
       → transcribe_audio()     [OpenAI Whisper]
       → analyse_audio_features() [local NumPy]
       → extract_frames()       [OpenCV]
       → EvidencePackage        [all evidence bundled]
       → evaluate()             [GPT-4o / Claude]
       → finalise_report_scores() [Python only]
       → build_report()         [Pydantic validation]
       → Streamlit dashboard
```

---

## Technology Stack

| Component | Library / Tool |
|-----------|---------------|
| UI | Streamlit 1.41 |
| Transcription | OpenAI Whisper (`whisper-1`) |
| AI evaluation | OpenAI GPT-4o or Anthropic Claude |
| Video processing | FFmpeg 9.x, OpenCV 4.10 |
| Data validation | Pydantic v2, pydantic-settings |
| Charts | Plotly |
| Tests | pytest 8.3, pytest-mock |
| Environment | python-dotenv |
| Language | Python 3.12 |

---

## Installation

### Prerequisites

- Python 3.12
- FFmpeg (must be on `PATH`)
- An OpenAI API key (required for transcription and evaluation)

### Install FFmpeg (macOS)

```bash
brew install ffmpeg
```

### Install FFmpeg (Ubuntu/Debian)

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

### Install FFmpeg (Windows)

Download from https://ffmpeg.org/download.html and add `bin/` to your PATH.

### Set up the project

```bash
# Clone or copy the project
cd classlens-ai

# Create and activate a virtual environment
python3.12 -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Environment Variables

Copy the example file and fill in your values:

```bash
cp .env.example .env
```

**Required:**

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (required for Whisper + GPT-4o) |

**Optional / defaults:**

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_PROVIDER` | `openai` | `openai` or `anthropic` |
| `EVALUATION_MODEL` | `gpt-4o` | LLM used for evaluation |
| `TRANSCRIPTION_MODEL` | `whisper-1` | Whisper model |
| `MAX_FILE_SIZE_MB` | `500` | Upload size cap |
| `MAX_DURATION_SECONDS` | `3600` | Recording length cap |
| `MAX_FRAMES` | `12` | Max video frames extracted |
| `FRAME_INTERVAL_SECONDS` | `30` | Seconds between frame samples |
| `THRESHOLD_POOR_MAX` | `39` | Upper bound for Poor rating |
| `THRESHOLD_AVERAGE_MAX` | `59` | Upper bound for Average rating |
| `THRESHOLD_GOOD_MAX` | `79` | Upper bound for Good rating |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

**If using Anthropic:**

Set `AI_PROVIDER=anthropic` and `ANTHROPIC_API_KEY=sk-ant-...`.
You still need `OPENAI_API_KEY` for Whisper transcription (Anthropic has no hosted ASR).

**Never commit `.env` to Git** — it is listed in `.gitignore`.

---

## Running the Application

```bash
streamlit run app.py
```

The app opens at http://localhost:8501 by default.

To use a different port:

```bash
streamlit run app.py --server.port 8502
```

---

## Running Tests

```bash
pytest tests/ -v
```

Expected output: **128 passed** (no API keys required — all tests use synthetic fixtures).

```bash
# Run a specific test file
pytest tests/test_scoring.py -v

# Run with coverage
pip install pytest-cov
pytest tests/ --cov=. --cov-report=term-missing
```

---

## Configuring the AI Provider

### OpenAI (default)

```bash
# .env
AI_PROVIDER=openai
OPENAI_API_KEY=sk-...
EVALUATION_MODEL=gpt-4o
TRANSCRIPTION_MODEL=whisper-1
```

### Anthropic (evaluation only — still needs OpenAI for transcription)

```bash
# .env
AI_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
EVALUATION_MODEL=claude-3-5-sonnet-20241022
OPENAI_API_KEY=sk-...   # still required for Whisper
```

If the API key is missing, the app shows a setup warning on the home page and refuses to start a pipeline — no fabricated results are shown.

---

## How to Upload and Assess a Recording

1. Start the app with `streamlit run app.py`.
2. The home page shows a setup warning if your API key is missing.
3. Click **Choose a recording file** and select a video or audio file.
4. Supported formats: `.mp4 .webm .mov .avi .mkv .m4v .flv .wmv .mp3 .wav .m4a .aac .ogg .flac`
5. Click **Analyse Class**.
6. Watch the 7-step progress indicator.
7. Review the assessment dashboard:
   - Overall rating and score
   - Dimension score chart
   - Evidence-backed observations per dimension
   - Strengths and improvement recommendations
8. Download the JSON report with the **Download JSON Report** button.

---

## Scoring Rubric

Five dimensions are assessed; the Python engine always calculates the final score:

| Dimension | Weight | What it measures |
|-----------|--------|-----------------|
| Teaching Clarity | 30% | Explanation quality, logical progression, examples |
| Engagement & Checks | 25% | Questions, student interaction, understanding checks |
| Lesson Structure | 20% | Intro, coherent flow, transitions, recap |
| Audio Intelligibility | 15% | Speech clarity, noise, distortion |
| Visual Support | 10% | On-screen content relevance and readability |

**Score formula:**

```
overall_score = Σ(dimension_score × weight) / Σ(weights of assessed dimensions)
```

Unavailable dimensions (e.g. no video stream) are excluded from both the numerator and denominator.

**Rating thresholds:**

| Score range | Rating |
|-------------|--------|
| 0 – 39 | Poor |
| 40 – 59 | Average |
| 60 – 79 | Good |
| 80 – 100 | Excellent |

Thresholds are configurable via `.env`. The LLM never determines the final rating.

---

## Missing Evidence Handling

When a dimension cannot be assessed (e.g. no video stream, transcription failed):

- Its status is set to `unavailable` in the report.
- Its weight is excluded from the denominator — it does not count as zero.
- The UI labels it clearly as "Not assessed".
- A limitation note is added to the report.

---

## Data Handling and Privacy

- Recordings are written to a unique temporary directory (`/tmp/classlens_<id>_/`) and deleted in a `finally` block after processing, regardless of success or failure.
- Audio is extracted as a temporary WAV file in the same directory and deleted with it.
- Extracted frames are stored in the same temporary directory and deleted with it.
- The recording and transcript are sent to OpenAI (Whisper for transcription, GPT-4o for evaluation). By using this application you accept OpenAI's data processing terms.
- No recording, transcript, or report is stored permanently by the application.
- API keys are read from environment variables and never logged or echoed to the UI.
- The JSON report may contain transcript excerpts. Keep downloaded reports confidential if the recording contains identifiable content.
- Do not upload recordings of children to any provider unless you have confirmed that the provider's terms permit processing such content and the organiser has approved that provider.

---

## Deployment

### Streamlit Community Cloud (recommended for prototyping)

1. Push the repository to GitHub (without `.env`, recordings, or API keys).
2. Go to https://share.streamlit.io and connect your repository.
3. Set secrets via the Streamlit Cloud dashboard (Settings → Secrets):
   ```toml
   OPENAI_API_KEY = "sk-..."
   ```
4. Streamlit Cloud provides Python 3.12 and `pip install` from `requirements.txt` automatically.
5. FFmpeg is available on Streamlit Cloud's Linux runners. If it is missing, add a `packages.txt` file:
   ```
   ffmpeg
   ```

### Local / self-hosted

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

For HTTPS in production, place a reverse proxy (nginx, Caddy) in front of the Streamlit server.

### Environment-variable checklist for deployment

- `OPENAI_API_KEY` — required
- `AI_PROVIDER` — set to `openai` (or `anthropic` with both keys)
- `MAX_FILE_SIZE_MB` — adjust for your hosting storage limits
- `LOG_LEVEL` — set to `WARNING` in production to avoid logging transcript fragments

---

## Known Limitations

1. **Transcription accuracy**: Whisper is generally accurate but can mis-hear domain-specific terms, names, or non-English speech. Errors propagate to the transcript-based dimension scores.

2. **Frame coverage**: Only `MAX_FRAMES` (default 12) frames are sampled. Visual content between frames is not analysed. Long recordings with rapid slide changes may be underrepresented.

3. **Audio-only recordings**: Visual Instructional Support is unavailable. The score renormalises across the remaining four dimensions.

4. **Student audio**: In most online class recordings, only the teacher's audio is clear. Student responses, if audible at all, are treated as part of the transcript but may be misattributed.

5. **No ground-truth calibration**: The rubric weights and thresholds are reasoned defaults, not empirically calibrated against the 8x8 dataset. Treat ratings as indicative, not definitive.

6. **Whisper 25 MB limit**: Audio files larger than 25 MB cannot be sent to Whisper. For long recordings, reduce the source bitrate or split the file.

7. **Context window**: Transcripts longer than ~6 000 words are truncated before being sent to the evaluation model. The later portion of very long classes is not evaluated.

8. **Confidence levels**: The `confidence` field is determined by the LLM, not by a calibrated probability model. Treat it as a qualitative signal.

9. **No multi-language transcript alignment**: If the teacher switches languages mid-lesson, Whisper may not segment boundaries correctly.

---

## Troubleshooting

| Problem | Resolution |
|---------|------------|
| `ffmpeg not found` | Install FFmpeg and ensure it is on PATH (`which ffmpeg`) |
| `OPENAI_API_KEY not configured` | Copy `.env.example` to `.env` and set your key |
| `Audio extraction failed` | File may have no audio stream; try a different recording |
| `Transcription failed: 25 MB limit` | Reduce audio bitrate with `ffmpeg -i input.mp4 -b:a 32k output.mp4` |
| `Streamlit ImportError` | Activate the virtual environment: `source .venv/bin/activate` |
| `Rate limit error` | Wait 60 seconds and retry; or upgrade your OpenAI plan |
| `Pydantic ValidationError` on report | LLM returned malformed JSON; the app retries automatically up to 2 times |
| Tests fail | Ensure you are in the project root and the venv is activated |

---

## Tested Features (no API key required)

- File upload validation (extension, size, corruption, duration)
- Audio extraction from synthetic MP4
- Frame extraction from synthetic MP4
- Weighted scoring with all five / partial / unavailable dimensions
- Score-to-rating mapping at all boundary values
- Pydantic schema validation for all models
- JSON report serialisation
- Temporary directory lifecycle (create, cleanup, error resilience)

## Features Requiring API Key Configuration

- Whisper speech transcription (`OPENAI_API_KEY`)
- GPT-4o / Claude teaching quality evaluation (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`)
- Vision-based frame description (`OPENAI_API_KEY`, model must support images)
