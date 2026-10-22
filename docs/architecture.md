# ClassLens AI — Architecture

## Overview

ClassLens AI is a single-process Python application.  The Streamlit server handles both the UI and the processing pipeline in the same Python runtime.  No separate backend, message queue, or database is involved.

---

## Module Responsibilities

```
app.py
  │  Streamlit UI, user input, progress display, results rendering
  │  Orchestrates the pipeline by calling the modules below in order
  │
  ├── config.py
  │     pydantic-settings Settings class
  │     Reads .env; validates all settings at startup
  │     Provides dimension_weights(), provider_ready(), setup_errors()
  │
  ├── schemas.py
  │     All Pydantic v2 data models
  │     EvidencePackage (pipeline internal)
  │     AssessmentReport (final output)
  │     Enforces constraints: score bounds, rating consistency, evidence presence
  │
  ├── video_processor.py
  │     validate_recording()   — ffprobe, extension, size, duration checks
  │     extract_audio()        — ffmpeg WAV extraction (16kHz mono)
  │     extract_frames()       — OpenCV frame sampling at configurable intervals
  │     Temp dir management    — create_temp_dir(), cleanup_temp_dir()
  │
  ├── audio_analyzer.py
  │     transcribe_audio()     — OpenAI Whisper, returns TranscriptResult
  │     analyse_audio_features() — local NumPy: silence, SNR, clipping
  │     No LLM required for feature analysis
  │
  ├── evaluator.py
  │     _format_evidence_for_prompt() — serialises EvidencePackage to text
  │     SYSTEM_PROMPT               — rubric, rules, JSON schema spec
  │     _evaluate_openai()           — GPT-4o call, json_object mode
  │     _evaluate_anthropic()        — Claude call, text mode + strip fences
  │     _parse_llm_response()        — validates and reconstructs dim list
  │     evaluate()                   — public entry point with retry loop
  │
  ├── scoring.py
  │     score_to_rating()            — threshold lookup (no LLM)
  │     calculate_weighted_score()   — Σ(score×weight)/Σ(weights), excludes UNAVAILABLE
  │     enforce_config_weights()     — overwrites any LLM-supplied weight tampering
  │     finalise_report_scores()     — end-to-end: enforce → calculate → map
  │
  └── report_generator.py
        build_report()          — assembles AssessmentReport, calls scoring
        report_to_json()        — JSON envelope with version + timestamp
        report_to_json_bytes()  — bytes for Streamlit download_button
        report_to_summary()     — plain-text digest for UI header
        RATING_COLOURS/EMOJI    — UI colour map
```

---

## Data Flow

```
User uploads file
        │
        ▼
save_upload_to_temp()          bytes → /tmp/classlens_<id>/upload.<ext>
        │
        ▼
validate_recording()           ffprobe → RecordingMetadata
        │
        ▼
extract_audio()                ffmpeg → /tmp/classlens_<id>/audio.wav
        │
        ▼
transcribe_audio()             WAV → Whisper API → TranscriptResult
        │
        ▼
analyse_audio_features()       WAV → NumPy → AudioFeatures
        │
        ▼
extract_frames()               OpenCV → List[VideoFrame] (paths in /tmp/)
        │
        ▼
_analyse_frames_with_vision()  frames → GPT-4o vision → descriptions (optional)
        │
        ▼
EvidencePackage                all evidence bundled into one validated object
        │
        ▼
evaluate()                     EvidencePackage → LLM → raw JSON
        │
        ▼
_parse_llm_response()          raw JSON → List[DimensionAssessment] (Pydantic)
        │
        ▼
finalise_report_scores()       enforce weights → weighted sum → rating (Python)
        │
        ▼
build_report()                 → AssessmentReport (fully validated)
        │
        ▼
cleanup_temp_dir()             delete /tmp/classlens_<id>/ (always, finally block)
        │
        ▼
Streamlit dashboard            render scores, evidence, download JSON
```

---

## Key Design Decisions

### Python scoring engine
The LLM supplies dimension scores and evidence only.  The overall score, rating, and weights are always computed and enforced by `scoring.py`.  This prevents the model from gaming the final classification.

### EvidencePackage
All pipeline outputs are bundled into a single typed object before being passed to the evaluator.  This makes the evidence boundary explicit — the evaluator cannot access anything that is not in the package.

### Pydantic at every boundary
Every pipeline output is a Pydantic model.  Malformed LLM responses are caught before they can reach the UI.

### Temp dir lifecycle
Each pipeline run creates an isolated `uuid`-prefixed temp directory.  `cleanup_temp_dir()` is called in a `finally` block so recordings and audio are deleted even when the pipeline raises an exception.

### Provider isolation
All provider-specific code lives in private functions (`_evaluate_openai`, `_evaluate_anthropic`, `_transcribe_openai`).  Adding a new provider requires implementing one private function and routing to it in the public entry point.

### Retry with feedback
If the LLM returns malformed JSON, the error is appended to the next attempt's prompt so the model has context on what went wrong.  Maximum retries are configurable (`LLM_MAX_RETRIES`, default 2).

---

## Security Considerations

- API keys are read from environment variables only; never echoed to the UI or logged.
- Uploaded files are written to isolated temp directories with uuid-based names.
- Filenames are sanitised before use to prevent path traversal.
- URL input is disabled by default; enabling it requires an explicit `ALLOWED_URL_HOSTS` allowlist.
- File size and duration are validated before any processing begins.
- Transcript content is not logged (only word counts).

---

## Dependency Graph

```
app.py
  ← config, schemas, video_processor, audio_analyzer, evaluator, scoring, report_generator

report_generator
  ← schemas, scoring

evaluator
  ← config, schemas

scoring
  ← config, schemas

audio_analyzer
  ← config, schemas

video_processor
  ← config, schemas
```

No circular imports.  `schemas.py` and `config.py` have no internal dependencies.
