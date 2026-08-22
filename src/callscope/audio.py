"""Stage 1: normalize arbitrary call recordings to canonical 16 kHz mono PCM.

Every downstream stage assumes canonical audio. Doing the conversion once, at
the boundary, keeps the VAD, embedding, and paralinguistic code free of
sample-rate and channel-layout branching -- which is where this kind of pipeline
usually rots.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from callscope.errors import AudioFormatError, FfmpegError, FfmpegNotFoundError

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2  # bytes, i.e. signed 16-bit PCM

#: ffmpeg is given a generous but finite budget. An unbounded subprocess in a
#: batch job is how a nightly QA run turns into a stuck queue.
DEFAULT_FFMPEG_TIMEOUT = 900.0


@dataclass(frozen=True, slots=True)
class AudioBuffer:
    """Canonical mono float32 audio in ``[-1, 1]`` with its sample rate."""

    samples: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        if self.samples.ndim != 1:
            raise AudioFormatError(f"expected mono 1-D samples, got shape {self.samples.shape}")
        if self.sample_rate <= 0:
            raise AudioFormatError(f"invalid sample rate {self.sample_rate}")

    @property
    def duration(self) -> float:
        return len(self.samples) / float(self.sample_rate)

    def slice_seconds(self, start: float, end: float) -> np.ndarray:
        """Samples in ``[start, end)`` seconds, clamped to the buffer bounds."""
        i0 = max(0, int(round(start * self.sample_rate)))
        i1 = min(len(self.samples), int(round(end * self.sample_rate)))
        if i1 <= i0:
            return np.zeros(0, dtype=np.float32)
        return self.samples[i0:i1]


def ffmpeg_path() -> str:
    """Absolute path to ffmpeg.

    Raises:
        FfmpegNotFoundError: if ffmpeg is not on PATH.
    """
    found = shutil.which("ffmpeg")
    if found is None:
        raise FfmpegNotFoundError(
            "ffmpeg was not found on PATH. Install it (macOS: `brew install ffmpeg`, "
            "Debian/Ubuntu: `apt install ffmpeg`) and retry."
        )
    return found


def ffmpeg_available() -> bool:
    """True when ffmpeg can be invoked. Never raises."""
    return shutil.which("ffmpeg") is not None


def probe_duration(path: str | Path, *, timeout: float = 60.0) -> float | None:
    """Best-effort container duration in seconds via ffprobe.

    Returns ``None`` rather than raising when ffprobe is unavailable or the
    container has no duration metadata -- callers can always fall back to the
    decoded sample count, which is authoritative anyway.
    """
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    cmd = [
        ffprobe, "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
        if proc.returncode != 0:
            return None
        value = json.loads(proc.stdout or b"{}").get("format", {}).get("duration")
        return float(value) if value is not None else None
    except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError, TypeError):
        return None


def normalize_audio(
    source: str | Path,
    destination: str | Path,
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
    downmix: bool = True,
    timeout: float = DEFAULT_FFMPEG_TIMEOUT,
    overwrite: bool = True,
) -> Path:
    """Transcode ``source`` to 16-bit PCM WAV at ``sample_rate``, mono.

    Args:
        source: Any container/codec ffmpeg can decode (wav, mp3, m4a, opus, ...).
        destination: Output ``.wav`` path. Parent directories are created.
        sample_rate: Output rate in Hz. Defaults to 16 kHz, what Whisper wants.
        downmix: Downmix to a single channel. Set ``False`` only if you have
            genuinely separate per-speaker channels and intend to handle them
            yourself -- callscope's own pipeline requires mono.
        timeout: Hard wall-clock limit for the ffmpeg subprocess.
        overwrite: Overwrite ``destination`` if it exists.

    Returns:
        The destination path.

    Raises:
        FileNotFoundError: ``source`` does not exist.
        FfmpegNotFoundError: ffmpeg is not installed.
        FfmpegError: ffmpeg exited non-zero, timed out, or produced no output.
    """
    src = Path(source)
    dst = Path(destination)
    if not src.exists():
        raise FileNotFoundError(f"input audio not found: {src}")
    if dst.exists() and not overwrite:
        raise FfmpegError(f"destination already exists: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_path(),
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-y" if overwrite else "-n",
        "-i", str(src),
    ]
    if downmix:
        cmd += ["-ac", str(TARGET_CHANNELS)]
    cmd += [
        "-ar", str(sample_rate),
        "-acodec", "pcm_s16le",
        "-map_metadata", "-1",  # strip tags; call recordings carry PII in metadata
        "-f", "wav",
        str(dst),
    ]

    logger.debug("running ffmpeg: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise FfmpegError(f"ffmpeg timed out after {timeout:.0f}s converting {src}") from exc
    except OSError as exc:  # e.g. ffmpeg deleted between which() and exec
        raise FfmpegError(f"failed to execute ffmpeg: {exc}") from exc

    if proc.returncode != 0:
        stderr = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        tail = "\n".join(stderr.splitlines()[-8:]) or "(no stderr)"
        raise FfmpegError(f"ffmpeg exited {proc.returncode} converting {src}:\n{tail}")

    if not dst.exists() or dst.stat().st_size <= 44:  # 44 == bare WAV header
        raise FfmpegError(f"ffmpeg produced no audio data for {src}")
    return dst


def read_wav(path: str | Path, *, require_canonical: bool = True) -> AudioBuffer:
    """Read a PCM WAV file into a float32 mono buffer.

    Uses the standard library ``wave`` module so that reading a normalized file
    costs no third-party dependency. Non-canonical files are rejected up front
    when ``require_canonical`` is set, which is the default -- silently
    resampling here would hide a broken normalization stage.

    Raises:
        AudioFormatError: unreadable, compressed, or non-canonical WAV.
    """
    p = Path(path)
    try:
        with wave.open(str(p), "rb") as wf:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            rate = wf.getframerate()
            frames = wf.getnframes()
            raw = wf.readframes(frames)
    except wave.Error as exc:
        raise AudioFormatError(f"{p} is not a readable PCM WAV file: {exc}") from exc
    except FileNotFoundError:
        raise
    except EOFError as exc:
        raise AudioFormatError(f"{p} is truncated") from exc

    if require_canonical:
        problems = []
        if channels != TARGET_CHANNELS:
            problems.append(f"{channels} channels (expected {TARGET_CHANNELS})")
        if width != TARGET_SAMPLE_WIDTH:
            problems.append(f"{width * 8}-bit samples (expected 16-bit)")
        if rate != TARGET_SAMPLE_RATE:
            problems.append(f"{rate} Hz (expected {TARGET_SAMPLE_RATE})")
        if problems:
            raise AudioFormatError(
                f"{p} is not canonical: {', '.join(problems)}. "
                "Run it through callscope.audio.normalize_audio first."
            )

    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise AudioFormatError(f"{p}: unsupported sample width {width} bytes")

    if channels > 1:
        usable = (len(data) // channels) * channels
        data = data[:usable].reshape(-1, channels).mean(axis=1)

    return AudioBuffer(samples=np.ascontiguousarray(data, dtype=np.float32), sample_rate=rate)


def write_wav(path: str | Path, buffer: AudioBuffer) -> Path:
    """Write a float32 buffer as 16-bit PCM WAV, clipping out-of-range samples."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(buffer.samples, -1.0, 1.0)
    pcm = np.round(clipped * 32767.0).astype("<i2")
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(TARGET_CHANNELS)
        wf.setsampwidth(TARGET_SAMPLE_WIDTH)
        wf.setframerate(buffer.sample_rate)
        wf.writeframes(pcm.tobytes())
    return p


def load_call_audio(
    source: str | Path,
    workdir: str | Path,
    *,
    timeout: float = DEFAULT_FFMPEG_TIMEOUT,
) -> tuple[AudioBuffer, Path]:
    """Normalize ``source`` into ``workdir`` and load the result.

    Returns the decoded buffer and the path to the normalized WAV, which is kept
    on disk because transcription backends take a file path, not an array.
    """
    dst = Path(workdir) / "normalized.wav"
    normalize_audio(source, dst, timeout=timeout)
    return read_wav(dst), dst
