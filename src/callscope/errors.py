"""Exception hierarchy for callscope.

Every failure a caller can reasonably be expected to handle gets its own type.
Everything else is allowed to propagate as a normal Python exception.
"""

from __future__ import annotations


class CallscopeError(Exception):
    """Base class for every error raised deliberately by callscope."""


class FfmpegNotFoundError(CallscopeError):
    """The ``ffmpeg`` executable is not on PATH."""


class FfmpegError(CallscopeError):
    """ffmpeg ran but exited non-zero, or timed out."""


class AudioFormatError(CallscopeError):
    """A WAV file could not be decoded, or is not in the expected canonical form."""


class RubricError(CallscopeError):
    """A rubric file is missing, unparseable, or structurally invalid."""


class TranscriptionError(CallscopeError):
    """No usable transcription backend, or a backend failed at runtime."""


class DiarizationError(CallscopeError):
    """Diarization could not produce a usable speaker assignment."""


class JudgeNotConfiguredError(CallscopeError):
    """A judge backend was selected but its runtime dependency is unavailable."""
