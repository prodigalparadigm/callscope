"""Exception hierarchy for callscope.

Every failure a caller can reasonably be expected to handle gets its own type.
Everything else is allowed to propagate as a normal Python exception.
"""

from __future__ import annotations


class CallscopeError(Exception):
    """Base class for every error raised deliberately by callscope."""


class UsageError(CallscopeError):
    """The command line was well-formed but asks for something impossible.

    Exits 2, matching argparse's own convention for a usage problem, so a
    scripted caller can tell "you invoked it wrong" from "the run failed".
    """


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
    """A judge backend was selected but its runtime dependency is unavailable.

    Raised before any call is attempted: a missing SDK, a missing credential, an
    unregistered backend name. Distinct from :class:`JudgeError`, which means the
    judge ran and the attempt failed.
    """


class JudgeError(CallscopeError):
    """A judge was invoked and failed on one criterion.

    Refusal, truncation, an unparseable response, a nonsense score. Caught per
    criterion by :func:`~callscope.scoring.base.score_transcript`, so one bad
    criterion costs its own score and not the other thirty-nine.
    """
