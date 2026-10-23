"""
config.py — Application configuration for ClassLens AI.

Reads settings from environment variables (and a .env file when present).
When running in Streamlit Cloud, also reads from st.secrets.
Never exposes secret values; callers receive only boolean "is configured"
flags when they need to check provider readiness.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings

# Load .env from the project root (silently ignored if the file is absent)
load_dotenv(override=False)

# Try to load Streamlit secrets if available (for Streamlit Cloud deployment)
try:
    import streamlit as st
    # Copy Streamlit secrets to environment variables
    if hasattr(st, 'secrets'):
        for key, value in st.secrets.items():
            if key not in os.environ:
                os.environ[key] = str(value)
except (ImportError, FileNotFoundError, AttributeError):
    # Not running in Streamlit or secrets not configured - continue normally
    pass


class Settings(BaseSettings):
    """
    All application settings with defaults and validation.

    Fields with secrets are never echoed back to callers; use the
    `provider_ready()` and `transcription_ready()` helpers instead.
    """

    # ── AI provider ───────────────────────────────────────────────
    ai_provider: str = Field(default="openai", description="'openai' | 'anthropic' | 'gemini'")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")

    transcription_model: str = Field(
        default="whisper-1",
        description="Whisper model name for OpenAI transcription.",
    )
    evaluation_model: str = Field(
        default="gpt-4o",
        description="LLM model for teaching quality evaluation.",
    )

    # ── Processing limits ─────────────────────────────────────────
    max_file_size_mb: int = Field(default=500, ge=1, le=2000)
    max_duration_seconds: int = Field(default=3600, ge=30, le=7200)
    max_frames: int = Field(default=12, ge=1, le=60)
    frame_interval_seconds: int = Field(default=30, ge=5, le=300)

    # ── Security / URL allowlist ──────────────────────────────────
    allowed_url_hosts: str = Field(
        default="",
        description=(
            "Comma-separated list of trusted hostname:port pairs for URL input. "
            "Empty string disables URL input entirely."
        ),
    )

    # ── Scoring thresholds ────────────────────────────────────────
    threshold_poor_max: int = Field(default=39, ge=0, le=99)
    threshold_average_max: int = Field(default=59, ge=1, le=99)
    threshold_good_max: int = Field(default=79, ge=1, le=99)

    # ── Logging ───────────────────────────────────────────────────
    log_level: str = Field(default="INFO")

    # ── Rubric weights (fixed; not overridable by LLM) ────────────
    weight_teaching_clarity: float = Field(default=30.0, frozen=True)
    weight_engagement: float = Field(default=25.0, frozen=True)
    weight_lesson_structure: float = Field(default=20.0, frozen=True)
    weight_audio_intelligibility: float = Field(default=15.0, frozen=True)
    weight_visual_support: float = Field(default=10.0, frozen=True)

    # ── Retry limits ──────────────────────────────────────────────
    llm_max_retries: int = Field(default=2, ge=1, le=5)
    llm_timeout_seconds: int = Field(default=120, ge=10, le=600)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        populate_by_name = True  # accept both alias and field name

    # ── Validators ────────────────────────────────────────────────

    @field_validator("ai_provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        allowed = {"openai", "anthropic", "gemini", "ollama"}
        v = v.lower().strip()
        if v not in allowed:
            raise ValueError(f"AI_PROVIDER must be one of {allowed}, got '{v}'.")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v = v.upper().strip()
        if v not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}.")
        return v

    @model_validator(mode="after")
    def thresholds_are_ordered(self) -> "Settings":
        if not (
            self.threshold_poor_max
            < self.threshold_average_max
            < self.threshold_good_max
            < 100
        ):
            raise ValueError(
                "Thresholds must satisfy: POOR_MAX < AVERAGE_MAX < GOOD_MAX < 100. "
                f"Got: {self.threshold_poor_max}, {self.threshold_average_max}, "
                f"{self.threshold_good_max}."
            )
        return self

    @model_validator(mode="after")
    def weights_sum_to_100(self) -> "Settings":
        total = (
            self.weight_teaching_clarity
            + self.weight_engagement
            + self.weight_lesson_structure
            + self.weight_audio_intelligibility
            + self.weight_visual_support
        )
        if abs(total - 100.0) > 0.001:
            raise ValueError(f"Dimension weights must sum to 100; got {total}.")
        return self

    # ── Convenience helpers ───────────────────────────────────────

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def allowed_url_host_list(self) -> List[str]:
        """Returns a list of allowed hosts; empty list means URL input disabled."""
        if not self.allowed_url_hosts.strip():
            return []
        return [h.strip() for h in self.allowed_url_hosts.split(",") if h.strip()]

    @property
    def url_input_enabled(self) -> bool:
        return len(self.allowed_url_host_list) > 0

    def provider_ready(self) -> bool:
        """True if the configured AI provider has a non-empty API key."""
        if self.ai_provider == "openai":
            return bool(self.openai_api_key and self.openai_api_key.strip())
        if self.ai_provider == "anthropic":
            return bool(self.anthropic_api_key and self.anthropic_api_key.strip())
        if self.ai_provider == "gemini":
            return bool(self.gemini_api_key and self.gemini_api_key.strip())
        if self.ai_provider == "ollama":
            return True  # Ollama runs locally, no API key needed
        return False

    def transcription_ready(self) -> bool:
        """True if transcription can be attempted."""
        # For Ollama, we use local Whisper (always ready, no API key needed)
        if self.ai_provider == "ollama":
            return True
        # For Gemini provider, check Gemini key
        if self.ai_provider == "gemini":
            return bool(self.gemini_api_key and self.gemini_api_key.strip()
                        and self.gemini_api_key != "your-gemini-api-key-here")
        # For others (OpenAI, Anthropic), check OpenAI key
        return bool(self.openai_api_key and self.openai_api_key.strip()
                    and self.openai_api_key != "your-openai-api-key-here")

    def setup_errors(self) -> List[str]:
        errors: List[str] = []
        if not self.provider_ready():
            errors.append(
                f"AI provider '{self.ai_provider}' is not configured. "
                "Set the corresponding API key environment variable."
            )
        if self.ai_provider == "anthropic" and not self.transcription_ready():
            errors.append(
                "Anthropic is selected for evaluation but transcription still "
                "requires OPENAI_API_KEY (Whisper). "
                "Set OPENAI_API_KEY or switch AI_PROVIDER to 'openai'."
            )
        if self.ai_provider == "gemini" and not self.transcription_ready():
            errors.append(
                "Gemini is selected for evaluation but transcription still "
                "requires GEMINI_API_KEY. "
                "Set GEMINI_API_KEY for transcription."
            )
        # Ollama uses local Whisper, no API key needed for transcription
        return errors

    def dimension_weights(self) -> dict[str, float]:
        """Ordered mapping of canonical dimension name → weight."""
        return {
            "Teaching Clarity": self.weight_teaching_clarity,
            "Engagement and Checks for Understanding": self.weight_engagement,
            "Lesson Structure": self.weight_lesson_structure,
            "Audio Intelligibility": self.weight_audio_intelligibility,
            "Visual Instructional Support": self.weight_visual_support,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the application-wide Settings singleton.

    Using lru_cache means .env is read once.  In tests, call
    `get_settings.cache_clear()` after patching environment variables.
    """
    return Settings()


def configure_logging() -> None:
    """Apply the configured log level to the root logger."""
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ── Supported video formats ───────────────────────────────────────────────────

SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v", ".flv", ".wmv",
}

SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac",
}

SUPPORTED_EXTENSIONS = SUPPORTED_VIDEO_EXTENSIONS | SUPPORTED_AUDIO_EXTENSIONS

# MIME types accepted for upload validation (supplement extension check)
SUPPORTED_MIME_PREFIXES = ("video/", "audio/")
