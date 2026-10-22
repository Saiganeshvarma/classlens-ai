"""
app.py — ClassLens AI Streamlit application.

Entry point: streamlit run app.py

Coordinates the full processing pipeline:
  1. Validate recording
  2. Extract audio
  3. Transcribe speech (Whisper)
  4. Analyse audio features (local)
  5. Extract video frames
  6. AI evaluation (GPT-4o / Claude)
  7. Score calculation (Python engine)
  8. Report generation + display
"""

from __future__ import annotations

import logging
import os
import traceback
from typing import Optional

import plotly.graph_objects as go
import streamlit as st

from audio_analyzer import TranscriptionError, analyse_audio_features, transcribe_audio
from config import (
    SUPPORTED_EXTENSIONS,
    configure_logging,
    get_settings,
)
from evaluator import EvaluatorError, evaluate
from report_generator import (
    RATING_COLOURS,
    RATING_EMOJI,
    build_report,
    report_to_json_bytes,
    report_to_summary,
)
from schemas import (
    ConfidenceLevel,
    DimensionStatus,
    EvidencePackage,
    ProcessingSummary,
    Rating,
)
from scoring import score_breakdown, normalised_weights
from video_processor import (
    VideoProcessorError,
    cleanup_temp_dir,
    create_temp_dir,
    extract_audio,
    extract_frames,
    save_upload_to_temp,
    validate_recording,
)

# ── Logging ───────────────────────────────────────────────────────────────────
configure_logging()
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ClassLens AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-title { font-size: 2.4rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0; }
    .subtitle   { font-size: 1.1rem; color: #555; margin-top: 0.2rem; }
    .rating-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 2rem;
        font-size: 1.4rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
    }
    .score-number { font-size: 3rem; font-weight: 800; }
    .dim-card {
        background: #f8f9fa;
        border-radius: 0.6rem;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
        border-left: 4px solid #ccc;
    }
    .evidence-box {
        background: #f0f4ff;
        border-radius: 0.4rem;
        padding: 0.7rem 1rem;
        margin: 0.3rem 0;
        font-size: 0.92rem;
    }
    .warn-box {
        background: #fff8e1;
        border-left: 4px solid #ffc107;
        padding: 0.6rem 1rem;
        border-radius: 0.3rem;
        font-size: 0.9rem;
    }
    .privacy-box {
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 0.6rem 1rem;
        border-radius: 0.3rem;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helper: setup warning banner ──────────────────────────────────────────────

def _show_setup_warnings() -> bool:
    """
    Show setup errors and return True if the app is NOT ready to process.
    Returns False (= ready) when fully configured.
    """
    settings = get_settings()
    errors = settings.setup_errors()
    if errors:
        st.warning("⚙️  **Setup required before you can analyse a recording:**")
        for err in errors:
            st.markdown(f"- {err}")
        st.markdown(
            "Copy `.env.example` to `.env`, fill in your API key, "
            "then restart the app.",
            help="See README.md for full setup instructions.",
        )
        return True
    return False


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(file_bytes: bytes, filename: str) -> None:
    """
    Execute the full processing pipeline for an uploaded recording.

    Uses Streamlit status containers to show real progress.
    All temporary files are deleted in a finally block.
    Results are stored in st.session_state so the page can be re-rendered
    without re-running the pipeline.
    """
    settings = get_settings()
    tmp_dir: Optional[str] = None

    # --- clear any previous results ---
    for key in ("report", "report_bytes", "pipeline_error"):
        st.session_state.pop(key, None)

    status_container = st.empty()

    def _step(msg: str) -> None:
        status_container.info(f"⏳ {msg}")

    def _done(msg: str) -> None:
        status_container.success(f"✅ {msg}")

    def _fail(msg: str) -> None:
        status_container.error(f"❌ {msg}")

    try:
        tmp_dir = create_temp_dir()

        # ── Step 1: Validate ─────────────────────────────────────────────────
        _step("Step 1/7 — Validating recording…")
        video_path = save_upload_to_temp(file_bytes, filename, tmp_dir)
        metadata = validate_recording(video_path, len(file_bytes))
        ps = ProcessingSummary(validation_completed=True)

        processing_notes = []
        if not metadata.audio_available:
            processing_notes.append("No audio stream detected — transcription unavailable.")
        if not metadata.video_available:
            processing_notes.append("No video stream detected — visual analysis unavailable.")

        # ── Step 2: Extract audio ────────────────────────────────────────────
        audio_path: Optional[str] = None
        if metadata.audio_available:
            _step("Step 2/7 — Extracting audio…")
            try:
                audio_path = extract_audio(video_path, tmp_dir)
                ps.audio_extraction_completed = True
            except VideoProcessorError as exc:
                processing_notes.append(f"Audio extraction failed: {exc}")
                logger.warning("Audio extraction failed: %s", exc)
        else:
            processing_notes.append("Audio extraction skipped — no audio stream.")

        # ── Step 3: Transcribe ───────────────────────────────────────────────
        transcript = None
        if audio_path and settings.transcription_ready():
            _step("Step 3/7 — Transcribing speech (Whisper)…")
            try:
                transcript = transcribe_audio(audio_path)
                metadata = metadata.model_copy(
                    update={
                        "has_speech": transcript.word_count > 0,
                        "detected_language": transcript.language,
                    }
                )
                ps.transcription_completed = True
                ps.transcript_word_count = transcript.word_count
            except TranscriptionError as exc:
                processing_notes.append(f"Transcription failed: {exc}")
                logger.warning("Transcription failed: %s", exc)
        elif audio_path:
            processing_notes.append(
                "Transcription skipped — OPENAI_API_KEY not configured."
            )
        else:
            processing_notes.append("Transcription skipped — no audio available.")

        # ── Step 4: Audio features ───────────────────────────────────────────
        audio_features = None
        if audio_path:
            _step("Step 4/7 — Analysing audio characteristics…")
            audio_features = analyse_audio_features(audio_path)
            ps.audio_feature_analysis_completed = True
            for note in audio_features.notes:
                processing_notes.append(note)

        # ── Step 5: Extract frames ───────────────────────────────────────────
        frames = []
        if metadata.video_available:
            _step("Step 5/7 — Extracting video frames…")
            try:
                frames = extract_frames(
                    video_path, tmp_dir, metadata.duration_seconds
                )
                ps.frame_extraction_completed = True
                ps.frames_extracted = len(frames)
                if not frames:
                    processing_notes.append(
                        "Frame extraction returned no frames — visual analysis unavailable."
                    )
            except Exception as exc:  # pylint: disable=broad-except
                processing_notes.append(f"Frame extraction failed: {exc}")
                logger.warning("Frame extraction failed: %s", exc)

        # ── Build evidence package ───────────────────────────────────────────
        pkg = EvidencePackage(
            recording_metadata=metadata,
            transcript=transcript,
            audio_features=audio_features,
            frames=frames,
            processing_summary=ps,
            processing_notes=processing_notes,
        )

        # ── Step 6: AI evaluation ────────────────────────────────────────────
        _step("Step 6/7 — Evaluating teaching quality (AI)…")
        if not settings.provider_ready():
            _fail(
                "AI provider not configured. "
                "Set your API key in .env and restart the app."
            )
            st.session_state["pipeline_error"] = (
                "AI provider not configured. "
                "Set your API key in .env and restart the app."
            )
            return

        (
            dimensions,
            strengths,
            improvements,
            overall_rationale,
            confidence,
            limitations,
        ) = evaluate(pkg)
        ps.evaluation_completed = True
        ps.visual_analysis_completed = any(
            f.description for f in frames
        )

        # ── Step 7: Score + report ───────────────────────────────────────────
        _step("Step 7/7 — Calculating score and generating report…")
        all_limitations = limitations + processing_notes
        report = build_report(
            pkg=pkg,
            dimensions=dimensions,
            strengths=strengths,
            improvements=improvements,
            overall_rationale=overall_rationale,
            confidence=confidence,
            limitations=[l for l in all_limitations if l],
        )
        report_bytes = report_to_json_bytes(report)

        st.session_state["report"] = report
        st.session_state["report_bytes"] = report_bytes
        _done("Analysis complete!")

    except VideoProcessorError as exc:
        _fail(str(exc))
        st.session_state["pipeline_error"] = str(exc)
        logger.error("Pipeline VideoProcessorError: %s", exc)

    except EvaluatorError as exc:
        _fail(f"AI evaluation failed: {exc}")
        st.session_state["pipeline_error"] = f"AI evaluation failed: {exc}"
        logger.error("Pipeline EvaluatorError: %s", exc)

    except Exception as exc:  # pylint: disable=broad-except
        tb = traceback.format_exc()
        _fail(f"Unexpected error: {exc}")
        st.session_state["pipeline_error"] = f"Unexpected error: {exc}"
        logger.error("Pipeline unexpected error:\n%s", tb)

    finally:
        if tmp_dir:
            cleanup_temp_dir(tmp_dir)


# ── Dashboard helpers ─────────────────────────────────────────────────────────

def _rating_colour(rating: Rating) -> str:
    return RATING_COLOURS.get(rating, "#999")


def _render_score_chart(report) -> None:
    """Render a horizontal bar chart of dimension scores."""
    breakdown = score_breakdown(report.dimensions)
    norm_w = normalised_weights(report.dimensions)

    names = []
    scores = []
    colours = []
    annotations = []

    base_colour = _rating_colour(report.rating)

    for b in breakdown:
        names.append(b["dimension"])
        s = b["score"]
        scores.append(s if s is not None else 0)
        if s is None:
            colours.append("#ccc")
            annotations.append("Not assessed")
        else:
            colours.append(base_colour)
            eff_w = norm_w.get(b["dimension"], b["weight"])
            annotations.append(f"{s:.0f}/100  (eff. weight {eff_w:.0f}%)")

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=names,
            orientation="h",
            marker_color=colours,
            text=annotations,
            textposition="outside",
            hovertemplate="%{y}: %{x:.0f}/100<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis=dict(range=[0, 115], showgrid=True, gridcolor="#eee", title="Score (0–100)"),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
        height=260,
        font=dict(family="sans-serif", size=13),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_dimension_card(dim, colour: str) -> None:
    """Render a collapsible card for one dimension."""
    score_str = f"{dim.score:.0f}/100" if dim.score is not None else "Not assessed"
    status_icon = {"assessed": "✅", "partial": "⚠️", "unavailable": "🚫"}.get(
        dim.status.value, "❓"
    )

    with st.expander(
        f"{status_icon} **{dim.name}** — {score_str}  (weight {dim.weight:.0f}%)",
        expanded=(dim.status != DimensionStatus.UNAVAILABLE),
    ):
        st.markdown(f"**Rationale:** {dim.rationale}")

        if dim.evidence:
            st.markdown("**Evidence:**")
            for ev in dim.evidence:
                ts_str = (
                    f" @ {ev.timestamp_seconds:.0f}s" if ev.timestamp_seconds is not None else ""
                )
                obs_type = "🔍 Observed" if ev.is_direct_observation else "💭 Interpreted"
                st.markdown(
                    f"""
                    <div class="evidence-box">
                    <strong>{obs_type}{ts_str}</strong>: {ev.observation}<br>
                    <em>Why it matters:</em> {ev.impact}
                    {f'<br><small><code>{ev.source_excerpt}</code></small>' if ev.source_excerpt else ''}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        elif dim.status == DimensionStatus.UNAVAILABLE:
            st.markdown("*No evidence available for this dimension.*")


def _render_report(report) -> None:
    """Render the full assessment dashboard."""
    settings = get_settings()
    colour = _rating_colour(report.rating)
    emoji = RATING_EMOJI.get(report.rating, "")

    # ── Top banner ─────────────────────────────────────────────────────────
    col_rating, col_score, col_conf = st.columns([2, 1, 1])

    with col_rating:
        st.markdown(
            f'<div class="rating-badge" style="background:{colour}">'
            f"{emoji} {report.rating.value}"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"File: {report.recording_metadata.filename}")
        dur = report.recording_metadata.duration_seconds
        st.caption(f"Duration: {dur/60:.1f} min  ({dur:.0f}s)")

    with col_score:
        st.markdown(
            f'<div class="score-number" style="color:{colour}">'
            f"{report.overall_score:.1f}"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.caption("out of 100")

    with col_conf:
        conf_icons = {"low": "🟠", "medium": "🟡", "high": "🟢"}
        st.metric(
            "Confidence",
            f"{conf_icons.get(report.confidence.value,'❓')} {report.confidence.value.capitalize()}",
        )
        ps = report.processing_summary
        st.caption(
            f"Transcript: {'✅' if ps.transcription_completed else '❌'}  "
            f"Frames: {ps.frames_extracted}"
        )

    st.divider()

    # ── Score chart ─────────────────────────────────────────────────────────
    st.subheader("📊 Dimension Scores")
    _render_score_chart(report)

    # ── Overall rationale ───────────────────────────────────────────────────
    st.subheader("📝 Overall Assessment")
    st.write(report.overall_rationale)

    # ── Strengths / improvements ────────────────────────────────────────────
    col_s, col_i = st.columns(2)
    with col_s:
        st.subheader("✅ Strengths")
        for s in report.strengths:
            st.markdown(f"- {s}")

    with col_i:
        st.subheader("🔧 Areas for Improvement")
        for imp in report.improvements:
            st.markdown(f"- {imp}")

    st.divider()

    # ── Dimension cards ─────────────────────────────────────────────────────
    st.subheader("🔬 Detailed Dimension Analysis")
    for dim in report.dimensions:
        _render_dimension_card(dim, colour)

    st.divider()

    # ── Limitations ─────────────────────────────────────────────────────────
    if report.limitations:
        st.subheader("⚠️ Limitations & Caveats")
        for lim in report.limitations:
            st.markdown(
                f'<div class="warn-box">⚠ {lim}</div>', unsafe_allow_html=True
            )

    st.divider()

    # ── Download ────────────────────────────────────────────────────────────
    st.subheader("⬇️ Download Report")
    report_bytes = st.session_state.get("report_bytes", b"")
    safe_name = (
        report.recording_metadata.filename.rsplit(".", 1)[0].replace(" ", "_")
        + "_classlens_report.json"
    )
    st.download_button(
        label="Download JSON Report",
        data=report_bytes,
        file_name=safe_name,
        mime="application/json",
        help="Download the full structured assessment as JSON.",
    )


# ── Main app ──────────────────────────────────────────────────────────────────

def main() -> None:
    settings = get_settings()

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="main-title">🎓 ClassLens AI</div>'
        '<div class="subtitle">AI-Powered Online Class Quality Evaluator</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Setup warning (non-blocking) ─────────────────────────────────────────
    not_ready = _show_setup_warnings()

    # ── Sidebar: how it works ────────────────────────────────────────────────
    with st.sidebar:
        st.header("How ClassLens Works")
        st.markdown(
            """
**ClassLens AI** analyses an online class recording across five dimensions:

| Dimension | Weight |
|-----------|--------|
| Teaching Clarity | 30% |
| Engagement & Checks | 25% |
| Lesson Structure | 20% |
| Audio Intelligibility | 15% |
| Visual Support | 10% |

**Rating thresholds:**
- 🔴 Poor: 0–39
- 🟡 Average: 40–59
- 🟢 Good: 60–79
- ⭐ Excellent: 80–100

**Pipeline stages:**
1. Validate recording
2. Extract audio (FFmpeg)
3. Transcribe speech (Whisper)
4. Analyse audio features
5. Extract video frames
6. AI evaluation (GPT-4o)
7. Score calculation (Python)
            """
        )
        st.divider()
        st.caption(f"Provider: {settings.ai_provider.upper()}")
        st.caption(f"Model: {settings.evaluation_model}")
        ready_icon = "✅" if not not_ready else "❌"
        st.caption(f"Status: {ready_icon} {'Ready' if not not_ready else 'Not configured'}")

    # ── Upload section ───────────────────────────────────────────────────────
    st.subheader("📁 Upload a Class Recording")

    ext_str = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    st.caption(
        f"Supported formats: {ext_str}  •  "
        f"Max size: {settings.max_file_size_mb} MB  •  "
        f"Max duration: {settings.max_duration_seconds // 60} minutes"
    )

    uploaded_file = st.file_uploader(
        "Choose a recording file",
        type=[e.lstrip(".") for e in sorted(SUPPORTED_EXTENSIONS)],
        help="Upload a video or audio recording of an online class.",
        label_visibility="collapsed",
    )

    # ── Privacy notice ───────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="privacy-box">
        🔒 <strong>Privacy notice:</strong>
        Your recording is processed temporarily in memory and a secure local
        directory.  It is sent to OpenAI only for speech transcription (Whisper)
        and quality evaluation (GPT-4o).  No recording is stored permanently or
        shared beyond these API calls.  Temporary files are deleted immediately
        after processing.  Do not upload recordings containing identifiable
        personal information beyond what is needed for the assessment.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    # ── Analyse button ───────────────────────────────────────────────────────
    analyse_clicked = st.button(
        "🔍 Analyse Class",
        type="primary",
        disabled=(uploaded_file is None),
        use_container_width=True,
    )

    if analyse_clicked and uploaded_file is not None:
        # Clear previous results before running
        for key in ("report", "report_bytes", "pipeline_error"):
            st.session_state.pop(key, None)

        file_bytes = uploaded_file.read()
        run_pipeline(file_bytes, uploaded_file.name)

    # ── Show previous error ──────────────────────────────────────────────────
    if "pipeline_error" in st.session_state and "report" not in st.session_state:
        st.error(f"Processing failed: {st.session_state['pipeline_error']}")

    # ── Render report if available ───────────────────────────────────────────
    if "report" in st.session_state:
        st.markdown("---")
        st.subheader("📋 Assessment Report")
        _render_report(st.session_state["report"])


if __name__ == "__main__":
    main()
