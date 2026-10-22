"""
app.py — ClassLens AI Streamlit application (Premium UI).
"""

from __future__ import annotations

import logging
import os
import traceback
from typing import Optional

import plotly.graph_objects as go
import streamlit as st

from audio_analyzer import TranscriptionError, analyse_audio_features, transcribe_audio
from config import SUPPORTED_EXTENSIONS, configure_logging, get_settings
from evaluator import EvaluatorError, evaluate
from report_generator import (
    RATING_COLOURS, RATING_EMOJI,
    build_report, report_to_json_bytes, report_to_summary,
)
from schemas import (
    ConfidenceLevel, DimensionStatus,
    EvidencePackage, ProcessingSummary, Rating,
)
from scoring import score_breakdown, normalised_weights
from video_processor import (
    VideoProcessorError, cleanup_temp_dir, create_temp_dir,
    extract_audio, extract_frames, save_upload_to_temp, validate_recording,
)

configure_logging()
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ClassLens AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Premium CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global reset ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #080b14 !important;
    font-family: 'Inter', sans-serif !important;
    color: #e2e8f0 !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background: #0d1117 !important; border-right: 1px solid #1e2532 !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Main container ── */
.main .block-container {
    padding: 2rem 3rem 4rem !important;
    max-width: 1200px !important;
}

/* ── Hero header ── */
.hero-container {
    background: linear-gradient(135deg, #0f1724 0%, #141d2e 50%, #0a1628 100%);
    border: 1px solid #1e3a5f;
    border-radius: 20px;
    padding: 3rem 3.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -20%;
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(56, 139, 253, 0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-container::after {
    content: '';
    position: absolute;
    bottom: -40%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(139, 92, 246, 0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(56, 139, 253, 0.12);
    border: 1px solid rgba(56, 139, 253, 0.3);
    border-radius: 50px;
    padding: 4px 14px;
    font-size: 0.75rem;
    font-weight: 600;
    color: #58a6ff;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 0%, #94b4d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem 0;
    line-height: 1.1;
    letter-spacing: -0.02em;
}
.hero-subtitle {
    font-size: 1.1rem;
    color: #6b8299;
    font-weight: 400;
    margin: 0;
}

/* ── Glass cards ── */
.glass-card {
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s ease;
}
.glass-card:hover { border-color: rgba(56,139,253,0.2); }

/* ── Upload zone ── */
.upload-zone {
    background: linear-gradient(135deg, rgba(56,139,253,0.04) 0%, rgba(139,92,246,0.04) 100%);
    border: 2px dashed rgba(56,139,253,0.25);
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    transition: all 0.3s ease;
    margin-bottom: 1.5rem;
}
.upload-zone:hover {
    border-color: rgba(56,139,253,0.5);
    background: linear-gradient(135deg, rgba(56,139,253,0.07) 0%, rgba(139,92,246,0.07) 100%);
}
.upload-icon { font-size: 3rem; margin-bottom: 0.75rem; }
.upload-title { font-size: 1.2rem; font-weight: 600; color: #e2e8f0; margin-bottom: 0.3rem; }
.upload-subtitle { font-size: 0.85rem; color: #4a5568; }

/* ── Rating badge ── */
.rating-hero {
    border-radius: 16px;
    padding: 2rem 2.5rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    margin-bottom: 1.5rem;
}
.rating-emoji-big { font-size: 3.5rem; }
.rating-label { font-size: 0.75rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; opacity: 0.7; margin-bottom: 0.2rem; }
.rating-value { font-size: 2.5rem; font-weight: 800; letter-spacing: -0.02em; color: white; }
.rating-score { font-size: 1rem; opacity: 0.8; margin-top: 0.2rem; }

/* ── Score ring ── */
.score-ring-container { text-align: center; padding: 1.5rem; }
.score-big { font-size: 4rem; font-weight: 800; letter-spacing: -0.03em; }
.score-label { font-size: 0.8rem; color: #4a5568; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.2rem; }

/* ── Metric pill ── */
.metric-pill {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-pill-label { font-size: 0.72rem; color: #4a5568; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.3rem; }
.metric-pill-value { font-size: 1.1rem; font-weight: 700; color: #e2e8f0; }

/* ── Section headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 1rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2532;
}

/* ── Dimension cards ── */
.dim-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid #1e2532;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.6rem;
    transition: all 0.2s ease;
}
.dim-card:hover { border-color: #2d3a4a; background: rgba(255,255,255,0.03); }
.dim-card-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem; }
.dim-card-name { font-weight: 600; font-size: 0.95rem; color: #cbd5e1; }
.dim-score-badge {
    font-size: 0.85rem;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 20px;
    background: rgba(56,139,253,0.12);
    color: #58a6ff;
    border: 1px solid rgba(56,139,253,0.2);
}
.dim-unavailable { opacity: 0.4; }
.dim-progress-bar {
    height: 4px;
    background: #1e2532;
    border-radius: 4px;
    margin: 0.5rem 0;
    overflow: hidden;
}
.dim-progress-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.8s ease;
}
.dim-rationale { font-size: 0.85rem; color: #5a6a7e; margin-top: 0.5rem; line-height: 1.5; }

/* ── Evidence items ── */
.evidence-item {
    background: rgba(56,139,253,0.04);
    border: 1px solid rgba(56,139,253,0.12);
    border-left: 3px solid rgba(56,139,253,0.4);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.85rem;
}
.evidence-obs { color: #cbd5e1; font-weight: 500; margin-bottom: 0.3rem; }
.evidence-meta { color: #4a5568; font-size: 0.78rem; }
.evidence-ts {
    display: inline-block;
    background: rgba(56,139,253,0.15);
    color: #58a6ff;
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 0.75rem;
    margin-right: 6px;
    font-family: monospace;
}

/* ── Strength / improvement pills ── */
.strength-item {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    padding: 0.6rem 0.8rem;
    background: rgba(46,204,113,0.05);
    border: 1px solid rgba(46,204,113,0.15);
    border-radius: 10px;
    margin-bottom: 0.4rem;
    font-size: 0.88rem;
    color: #9be0b5;
    line-height: 1.4;
}
.improvement-item {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    padding: 0.6rem 0.8rem;
    background: rgba(241,196,15,0.05);
    border: 1px solid rgba(241,196,15,0.15);
    border-radius: 10px;
    margin-bottom: 0.4rem;
    font-size: 0.88rem;
    color: #f0d070;
    line-height: 1.4;
}

/* ── Limitation box ── */
.limitation-item {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    padding: 0.6rem 0.8rem;
    background: rgba(231,76,60,0.05);
    border: 1px solid rgba(231,76,60,0.12);
    border-radius: 10px;
    margin-bottom: 0.4rem;
    font-size: 0.85rem;
    color: #e88;
    line-height: 1.4;
}

/* ── Privacy notice ── */
.privacy-notice {
    background: rgba(56,139,253,0.05);
    border: 1px solid rgba(56,139,253,0.15);
    border-radius: 12px;
    padding: 0.9rem 1.2rem;
    font-size: 0.82rem;
    color: #4a6a8a;
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    margin-top: 1rem;
}

/* ── Progress steps ── */
.step-container { margin: 0.3rem 0; }
.step-item {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.5rem 0.8rem;
    border-radius: 8px;
    font-size: 0.88rem;
    color: #4a5568;
    margin-bottom: 0.2rem;
}
.step-active {
    background: rgba(56,139,253,0.08);
    color: #58a6ff;
    border: 1px solid rgba(56,139,253,0.15);
}
.step-done { color: #2ecc71; }
.step-error { color: #e74c3c; }

/* ── Setup warning ── */
.setup-warning {
    background: rgba(241,196,15,0.07);
    border: 1px solid rgba(241,196,15,0.2);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.5rem;
    font-size: 0.9rem;
    color: #c9a227;
}

/* ── Streamlit widget overrides ── */
.stButton > button {
    background: linear-gradient(135deg, #1a6bd6 0%, #7c3aed 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 24px rgba(26,107,214,0.3) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 32px rgba(26,107,214,0.45) !important;
}
.stButton > button:disabled {
    background: #1e2532 !important;
    color: #3a4a5a !important;
    box-shadow: none !important;
    transform: none !important;
}
.stDownloadButton > button {
    background: rgba(46,204,113,0.1) !important;
    color: #2ecc71 !important;
    border: 1px solid rgba(46,204,113,0.3) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    box-shadow: none !important;
}
.stFileUploader {
    background: transparent !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: rgba(56,139,253,0.03) !important;
    border: 2px dashed rgba(56,139,253,0.2) !important;
    border-radius: 14px !important;
    color: #4a6a8a !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(56,139,253,0.45) !important;
    background: rgba(56,139,253,0.06) !important;
}
.stExpander {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid #1e2532 !important;
    border-radius: 12px !important;
}
.stExpander:hover { border-color: #2d3a4a !important; }
[data-testid="stExpanderDetails"] { background: transparent !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #94a3b8 !important; font-size: 0.8rem !important; text-transform: uppercase !important; letter-spacing: 0.1em !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li { font-size: 0.82rem !important; color: #4a5568 !important; }

/* ── Divider ── */
hr { border-color: #1e2532 !important; margin: 1.5rem 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #080b14; }
::-webkit-scrollbar-thumb { background: #1e2532; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2d3a4a; }
</style>
""", unsafe_allow_html=True)


# ── Rating theme map ──────────────────────────────────────────────────────────

RATING_GRADIENTS = {
    Rating.POOR:      ("linear-gradient(135deg, #1a0a0a 0%, #2d1010 100%)", "#e74c3c", "rgba(231,76,60,0.15)"),
    Rating.AVERAGE:   ("linear-gradient(135deg, #1a1500 0%, #2d2200 100%)", "#f39c12", "rgba(243,156,18,0.15)"),
    Rating.GOOD:      ("linear-gradient(135deg, #0a1a0f 0%, #0f2d18 100%)", "#2ecc71", "rgba(46,204,113,0.15)"),
    Rating.EXCELLENT: ("linear-gradient(135deg, #0a0f1a 0%, #101828 100%)", "#3498db", "rgba(52,152,219,0.15)"),
}


# ── Setup warning ─────────────────────────────────────────────────────────────

def _show_setup_warnings() -> bool:
    settings = get_settings()
    errors = settings.setup_errors()
    if errors:
        msg = "".join(f"<li>{e}</li>" for e in errors)
        st.markdown(f"""
        <div class="setup-warning">
            ⚙️ <strong>Setup required</strong><br>
            <ul style="margin:0.5rem 0 0 0; padding-left:1.2rem">{msg}</ul>
            <div style="margin-top:0.5rem; font-size:0.8rem; opacity:0.7">
            Copy <code>.env.example</code> to <code>.env</code>, fill in your API key, then restart.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return True
    return False


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(file_bytes: bytes, filename: str) -> None:
    settings = get_settings()
    tmp_dir: Optional[str] = None

    for key in ("report", "report_bytes", "pipeline_error"):
        st.session_state.pop(key, None)

    steps = [
        "Validating recording",
        "Extracting audio",
        "Transcribing speech",
        "Analysing audio features",
        "Extracting video frames",
        "AI evaluation",
        "Generating report",
    ]
    step_state = {s: "pending" for s in steps}

    progress_placeholder = st.empty()

    def _render_steps():
        html = '<div class="step-container">'
        for s, state in step_state.items():
            if state == "active":
                icon = "⏳"
                cls = "step-item step-active"
            elif state == "done":
                icon = "✅"
                cls = "step-item step-done"
            elif state == "error":
                icon = "❌"
                cls = "step-item step-error"
            else:
                icon = "○"
                cls = "step-item"
            html += f'<div class="{cls}">{icon} {s}</div>'
        html += "</div>"
        progress_placeholder.markdown(html, unsafe_allow_html=True)

    def _activate(step):
        step_state[step] = "active"
        _render_steps()

    def _complete(step):
        step_state[step] = "done"
        _render_steps()

    def _fail_step(step):
        step_state[step] = "error"
        _render_steps()

    try:
        tmp_dir = create_temp_dir()

        _activate("Validating recording")
        video_path = save_upload_to_temp(file_bytes, filename, tmp_dir)
        metadata = validate_recording(video_path, len(file_bytes))
        ps = ProcessingSummary(validation_completed=True)
        processing_notes = []
        if not metadata.audio_available:
            processing_notes.append("No audio stream detected.")
        if not metadata.video_available:
            processing_notes.append("No video stream detected.")
        _complete("Validating recording")

        _activate("Extracting audio")
        audio_path: Optional[str] = None
        if metadata.audio_available:
            try:
                audio_path = extract_audio(video_path, tmp_dir)
                ps.audio_extraction_completed = True
                _complete("Extracting audio")
            except VideoProcessorError as exc:
                processing_notes.append(f"Audio extraction failed: {exc}")
                _fail_step("Extracting audio")
        else:
            step_state["Extracting audio"] = "done"
            processing_notes.append("Audio extraction skipped.")
            _render_steps()

        _activate("Transcribing speech")
        transcript = None
        if audio_path and settings.transcription_ready():
            try:
                transcript = transcribe_audio(audio_path)
                metadata = metadata.model_copy(update={
                    "has_speech": transcript.word_count > 0,
                    "detected_language": transcript.language,
                })
                ps.transcription_completed = True
                ps.transcript_word_count = transcript.word_count
                _complete("Transcribing speech")
            except TranscriptionError as exc:
                processing_notes.append(f"Transcription failed: {exc}")
                _fail_step("Transcribing speech")
        else:
            step_state["Transcribing speech"] = "done"
            processing_notes.append("Transcription skipped — provider not configured.")
            _render_steps()

        _activate("Analysing audio features")
        audio_features = None
        if audio_path:
            audio_features = analyse_audio_features(audio_path)
            ps.audio_feature_analysis_completed = True
            for note in audio_features.notes:
                processing_notes.append(note)
        _complete("Analysing audio features")

        _activate("Extracting video frames")
        frames = []
        if metadata.video_available:
            try:
                frames = extract_frames(video_path, tmp_dir, metadata.duration_seconds)
                ps.frame_extraction_completed = True
                ps.frames_extracted = len(frames)
            except Exception as exc:
                processing_notes.append(f"Frame extraction failed: {exc}")
        _complete("Extracting video frames")

        pkg = EvidencePackage(
            recording_metadata=metadata,
            transcript=transcript,
            audio_features=audio_features,
            frames=frames,
            processing_summary=ps,
            processing_notes=processing_notes,
        )

        _activate("AI evaluation")
        if not settings.provider_ready():
            _fail_step("AI evaluation")
            st.session_state["pipeline_error"] = "AI provider not configured."
            return

        (dimensions, strengths, improvements,
         overall_rationale, confidence, limitations) = evaluate(pkg)
        ps.evaluation_completed = True
        ps.visual_analysis_completed = any(f.description for f in frames)
        _complete("AI evaluation")

        _activate("Generating report")
        all_limitations = limitations + processing_notes
        report = build_report(
            pkg=pkg, dimensions=dimensions, strengths=strengths,
            improvements=improvements, overall_rationale=overall_rationale,
            confidence=confidence, limitations=[l for l in all_limitations if l],
        )
        report_bytes = report_to_json_bytes(report)
        st.session_state["report"] = report
        st.session_state["report_bytes"] = report_bytes
        _complete("Generating report")

    except VideoProcessorError as exc:
        for s in step_state:
            if step_state[s] == "active":
                _fail_step(s)
        st.session_state["pipeline_error"] = str(exc)
        logger.error("Pipeline error: %s", exc)

    except EvaluatorError as exc:
        for s in step_state:
            if step_state[s] == "active":
                _fail_step(s)
        st.session_state["pipeline_error"] = f"AI evaluation failed: {exc}"
        logger.error("Evaluator error: %s", exc)

    except Exception as exc:
        for s in step_state:
            if step_state[s] == "active":
                _fail_step(s)
        st.session_state["pipeline_error"] = f"Unexpected error: {exc}"
        logger.error("Unexpected: %s", traceback.format_exc())

    finally:
        if tmp_dir:
            cleanup_temp_dir(tmp_dir)


# ── Chart ─────────────────────────────────────────────────────────────────────

def _render_score_chart(report) -> None:
    breakdown = score_breakdown(report.dimensions)
    norm_w = normalised_weights(report.dimensions)
    _, accent, _ = RATING_GRADIENTS[report.rating]

    names, scores, colours, texts = [], [], [], []
    for b in breakdown:
        names.append(b["dimension"].replace(" and ", " & "))
        s = b["score"]
        scores.append(s if s is not None else 0)
        colours.append(accent if s is not None else "#1e2532")
        ew = norm_w.get(b["dimension"], b["weight"])
        texts.append(f"{s:.0f}" if s is not None else "N/A")

    fig = go.Figure(go.Bar(
        x=scores, y=names, orientation="h",
        marker=dict(
            color=colours,
            opacity=0.85,
            line=dict(color="rgba(0,0,0,0)", width=0),
        ),
        text=texts,
        textposition="outside",
        textfont=dict(color="#94a3b8", size=13, family="Inter"),
        hovertemplate="%{y}: %{x:.0f}/100<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(
            range=[0, 120],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.04)",
            gridwidth=1,
            tickfont=dict(color="#4a5568", size=11),
            zeroline=False,
            title=None,
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(color="#94a3b8", size=12, family="Inter"),
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=60, t=10, b=10),
        height=240,
        font=dict(family="Inter"),
        bargap=0.35,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Dashboard ─────────────────────────────────────────────────────────────────

def _render_report(report) -> None:
    gradient, accent, accent_bg = RATING_GRADIENTS[report.rating]
    emoji = RATING_EMOJI.get(report.rating, "")

    # ── Rating hero banner ────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="rating-hero" style="background:{gradient}; border:1px solid {accent}33;">
        <div class="rating-emoji-big">{emoji}</div>
        <div style="flex:1">
            <div class="rating-label">Overall Rating</div>
            <div class="rating-value">{report.rating.value}</div>
            <div class="rating-score">{report.overall_score:.1f} out of 100</div>
        </div>
        <div style="text-align:right">
            <div class="rating-label">Confidence</div>
            <div style="font-size:1.1rem; font-weight:700; color:{accent}">
                {report.confidence.value.capitalize()}
            </div>
            <div style="font-size:0.78rem; color:#4a5568; margin-top:0.3rem">
                {report.recording_metadata.filename[:30]}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics row ───────────────────────────────────────────────────────────
    dur = report.recording_metadata.duration_seconds
    ps  = report.processing_summary
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class="metric-pill">
            <div class="metric-pill-label">Duration</div>
            <div class="metric-pill-value">{dur/60:.1f} min</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-pill">
            <div class="metric-pill-label">Transcript</div>
            <div class="metric-pill-value">{'✓ ' + str(ps.transcript_word_count) + ' words' if ps.transcription_completed else '—'}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-pill">
            <div class="metric-pill-label">Frames</div>
            <div class="metric-pill-value">{ps.frames_extracted} sampled</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-pill">
            <div class="metric-pill-label">Visual AI</div>
            <div class="metric-pill-value">{'✓ Active' if ps.visual_analysis_completed else '— N/A'}</div>
        </div>""", unsafe_allow_html=True)

    # ── Score chart ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Dimension Scores</div>', unsafe_allow_html=True)
    _render_score_chart(report)

    # ── Overall rationale ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📝 Assessment Summary</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="glass-card" style="color:#94a3b8; font-size:0.92rem; line-height:1.7">{report.overall_rationale}</div>', unsafe_allow_html=True)

    # ── Strengths + Improvements ──────────────────────────────────────────────
    col_s, col_i = st.columns(2)
    with col_s:
        st.markdown('<div class="section-header">✅ Strengths</div>', unsafe_allow_html=True)
        for s in report.strengths:
            st.markdown(f'<div class="strength-item"><span>↗</span><span>{s}</span></div>', unsafe_allow_html=True)
    with col_i:
        st.markdown('<div class="section-header">🔧 Improvements</div>', unsafe_allow_html=True)
        for imp in report.improvements:
            st.markdown(f'<div class="improvement-item"><span>→</span><span>{imp}</span></div>', unsafe_allow_html=True)

    # ── Dimension detail cards ────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔬 Detailed Analysis</div>', unsafe_allow_html=True)
    for dim in report.dimensions:
        unavailable = dim.status == DimensionStatus.UNAVAILABLE
        score_str = f"{dim.score:.0f}/100" if dim.score is not None else "N/A"
        fill_pct = f"{dim.score:.0f}%" if dim.score is not None else "0%"
        fill_colour = accent if not unavailable else "#1e2532"
        opacity_cls = "dim-unavailable" if unavailable else ""

        with st.expander(f"{'🚫' if unavailable else '📌'} {dim.name}  —  {score_str}  (weight {dim.weight:.0f}%)", expanded=not unavailable):
            # Progress bar
            st.markdown(f"""
            <div class="{opacity_cls}">
            <div class="dim-progress-bar">
              <div class="dim-progress-fill" style="width:{fill_pct}; background:{fill_colour}; opacity:0.8"></div>
            </div>
            <div class="dim-rationale">{dim.rationale}</div>
            </div>
            """, unsafe_allow_html=True)

            if dim.evidence:
                st.markdown("<div style='margin-top:0.8rem; font-size:0.78rem; color:#4a5568; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.3rem'>Evidence</div>", unsafe_allow_html=True)
                for ev in dim.evidence:
                    ts_html = f'<span class="evidence-ts">@{ev.timestamp_seconds:.0f}s</span>' if ev.timestamp_seconds is not None else ""
                    tag = "🔍" if ev.is_direct_observation else "💭"
                    excerpt = f"<br><span style='font-family:monospace; font-size:0.78rem; color:#3a5a7a; background:rgba(56,139,253,0.06); padding:2px 6px; border-radius:4px; display:inline-block; margin-top:4px'>{ev.source_excerpt}</span>" if ev.source_excerpt else ""
                    st.markdown(f"""
                    <div class="evidence-item">
                        <div class="evidence-obs">{tag} {ts_html}{ev.observation}</div>
                        <div class="evidence-meta">↳ {ev.impact}{excerpt}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── Limitations ───────────────────────────────────────────────────────────
    if report.limitations:
        st.markdown('<div class="section-header">⚠️ Limitations</div>', unsafe_allow_html=True)
        for lim in report.limitations[:5]:
            st.markdown(f'<div class="limitation-item"><span>⚠</span><span>{lim}</span></div>', unsafe_allow_html=True)

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">⬇️ Export</div>', unsafe_allow_html=True)
    safe_name = report.recording_metadata.filename.rsplit(".", 1)[0].replace(" ", "_") + "_classlens.json"
    col_dl, col_empty = st.columns([1, 3])
    with col_dl:
        st.download_button(
            label="⬇  Download JSON Report",
            data=st.session_state.get("report_bytes", b""),
            file_name=safe_name,
            mime="application/json",
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    settings = get_settings()

    # ── Hero ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">✦ AI Assessment Platform</div>
        <h1 class="hero-title">ClassLens AI</h1>
        <p class="hero-subtitle">Intelligent quality evaluation for online class recordings</p>
    </div>
    """, unsafe_allow_html=True)

    _show_setup_warnings()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### How It Works")
        st.markdown("""
**Five evaluation dimensions:**

| | Dimension | Weight |
|--|-----------|--------|
| 📖 | Teaching Clarity | 30% |
| 💬 | Engagement | 25% |
| 🗂 | Lesson Structure | 20% |
| 🔊 | Audio Quality | 15% |
| 🖥 | Visual Support | 10% |

**Ratings:**
- ⭐ Excellent: 80–100
- 🟢 Good: 60–79
- 🟡 Average: 40–59
- 🔴 Poor: 0–39
        """)
        st.divider()
        provider_icon = "✅" if settings.provider_ready() else "❌"
        st.caption(f"{provider_icon} Provider: `{settings.ai_provider.upper()}`")
        st.caption(f"🤖 Model: `{settings.evaluation_model}`")

    # ── Upload section ────────────────────────────────────────────────────────
    col_upload, col_info = st.columns([3, 2])

    with col_upload:
        ext_list = ", ".join(sorted(e.lstrip(".").upper() for e in SUPPORTED_EXTENSIONS))
        st.markdown(f"""
        <div style="font-size:0.78rem; color:#4a5568; text-transform:uppercase;
             letter-spacing:0.08em; margin-bottom:0.6rem; font-weight:600">
        Upload Recording
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Drop your recording here",
            type=[e.lstrip(".") for e in sorted(SUPPORTED_EXTENSIONS)],
            label_visibility="collapsed",
        )

        st.markdown(f"""
        <div style="font-size:0.75rem; color:#2a3a4a; margin-top:0.4rem">
        Supported: {ext_list[:80]}… &nbsp;·&nbsp;
        Max {settings.max_file_size_mb} MB &nbsp;·&nbsp;
        Max {settings.max_duration_seconds//60} min
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        analyse_clicked = st.button(
            "🔍  Analyse Class",
            type="primary",
            disabled=(uploaded_file is None),
            use_container_width=True,
        )

        st.markdown("""
        <div class="privacy-notice">
            🔒
            <span>Recordings are processed temporarily and sent only to the configured AI provider
            for transcription and evaluation. No data is stored permanently.</span>
        </div>
        """, unsafe_allow_html=True)

    with col_info:
        st.markdown("""
        <div class="glass-card" style="height:100%; min-height:220px">
            <div style="font-size:0.75rem; color:#4a5568; text-transform:uppercase;
                 letter-spacing:0.08em; font-weight:600; margin-bottom:1rem">
            What gets analysed
            </div>
            <div style="font-size:0.85rem; color:#5a6a7e; line-height:1.8">
            🎙 Speech transcription with timestamps<br>
            📊 Audio quality metrics (SNR, silence, clipping)<br>
            🖼 Video frame sampling every 30 seconds<br>
            🤖 AI rubric evaluation across 5 dimensions<br>
            📈 Python-calculated weighted score<br>
            📄 Downloadable structured JSON report
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Run pipeline ──────────────────────────────────────────────────────────
    if analyse_clicked and uploaded_file is not None:
        for key in ("report", "report_bytes", "pipeline_error"):
            st.session_state.pop(key, None)
        file_bytes = uploaded_file.read()
        st.markdown("---")
        st.markdown('<div class="section-header">⚙️ Processing</div>', unsafe_allow_html=True)
        run_pipeline(file_bytes, uploaded_file.name)

    # ── Error ─────────────────────────────────────────────────────────────────
    if "pipeline_error" in st.session_state and "report" not in st.session_state:
        err = st.session_state["pipeline_error"]
        st.markdown(f"""
        <div class="limitation-item" style="padding:1rem 1.2rem; font-size:0.9rem; margin-top:1rem">
            ❌ <strong>Processing failed:</strong> {err}
        </div>
        """, unsafe_allow_html=True)

    # ── Report ────────────────────────────────────────────────────────────────
    if "report" in st.session_state:
        st.markdown("---")
        _render_report(st.session_state["report"])


if __name__ == "__main__":
    main()
