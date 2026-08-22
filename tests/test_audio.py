"""ffmpeg normalization and canonical WAV I/O."""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from callscope.audio import (
    TARGET_SAMPLE_RATE,
    AudioBuffer,
    ffmpeg_available,
    normalize_audio,
    probe_duration,
    read_wav,
    write_wav,
)
from callscope.errors import AudioFormatError, FfmpegError


def _write_noncanonical(path: Path, *, rate: int, channels: int, seconds: float) -> Path:
    """Write a stereo 44.1 kHz WAV -- i.e. exactly what a real recording looks like."""
    n = int(rate * seconds)
    t = np.arange(n) / rate
    left = 0.3 * np.sin(2 * np.pi * 220.0 * t)
    right = 0.3 * np.sin(2 * np.pi * 330.0 * t)
    interleaved = np.empty(n * channels, dtype=np.float64)
    interleaved[0::channels] = left
    if channels > 1:
        interleaved[1::channels] = right
    pcm = np.round(np.clip(interleaved, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())
    return path


@pytest.mark.requires_ffmpeg
def test_normalize_downmixes_and_resamples(tmp_path: Path):
    src = _write_noncanonical(tmp_path / "src.wav", rate=44100, channels=2, seconds=1.5)
    dst = normalize_audio(src, tmp_path / "out.wav")

    with wave.open(str(dst), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == TARGET_SAMPLE_RATE

    buf = read_wav(dst)
    assert buf.sample_rate == TARGET_SAMPLE_RATE
    # Resampling and codec latency shift the length slightly; 30 ms is generous
    # for a 1.5 s file and still tight enough to catch a real duration bug.
    assert buf.duration == pytest.approx(1.5, abs=0.03)


@pytest.mark.requires_ffmpeg
def test_normalize_creates_parent_directories(tmp_path: Path):
    src = _write_noncanonical(tmp_path / "src.wav", rate=8000, channels=1, seconds=0.5)
    dst = normalize_audio(src, tmp_path / "nested" / "deeper" / "out.wav")
    assert dst.exists()


@pytest.mark.requires_ffmpeg
def test_normalize_rejects_garbage_input(tmp_path: Path):
    bad = tmp_path / "not_audio.wav"
    bad.write_bytes(b"this is not a media file at all, not even close")
    with pytest.raises(FfmpegError) as exc:
        normalize_audio(bad, tmp_path / "out.wav")
    # The ffmpeg stderr tail must reach the caller; a bare "conversion failed"
    # is useless when triaging a batch of four hundred recordings.
    assert "ffmpeg exited" in str(exc.value)


def test_normalize_missing_input_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        normalize_audio(tmp_path / "nope.m4a", tmp_path / "out.wav")


@pytest.mark.requires_ffmpeg
def test_normalize_honours_timeout(tmp_path: Path, monkeypatch):
    src = _write_noncanonical(tmp_path / "src.wav", rate=16000, channels=1, seconds=0.2)

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=0.001)

    monkeypatch.setattr("callscope.audio.subprocess.run", fake_run)
    with pytest.raises(FfmpegError, match="timed out"):
        normalize_audio(src, tmp_path / "out.wav", timeout=0.001)


def test_read_wav_rejects_noncanonical(tmp_path: Path):
    src = _write_noncanonical(tmp_path / "stereo.wav", rate=44100, channels=2, seconds=0.2)
    with pytest.raises(AudioFormatError) as exc:
        read_wav(src)
    message = str(exc.value)
    assert "2 channels" in message and "44100 Hz" in message


def test_read_wav_permits_noncanonical_when_asked(tmp_path: Path):
    src = _write_noncanonical(tmp_path / "stereo.wav", rate=44100, channels=2, seconds=0.2)
    buf = read_wav(src, require_canonical=False)
    assert buf.samples.ndim == 1  # downmixed
    assert buf.sample_rate == 44100


def test_read_wav_rejects_non_wav(tmp_path: Path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"RIFFnope")
    with pytest.raises(AudioFormatError):
        read_wav(bad)


def test_write_then_read_roundtrip(tmp_path: Path):
    t = np.arange(TARGET_SAMPLE_RATE) / TARGET_SAMPLE_RATE
    original = AudioBuffer((0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32),
                           TARGET_SAMPLE_RATE)
    path = write_wav(tmp_path / "rt.wav", original)
    restored = read_wav(path)
    assert restored.sample_rate == original.sample_rate
    assert len(restored.samples) == len(original.samples)
    # 16-bit quantization error is bounded by 1/32768.
    assert np.max(np.abs(restored.samples - original.samples)) < 1e-4


def test_write_clips_out_of_range_samples(tmp_path: Path):
    loud = AudioBuffer(np.array([3.0, -3.0, 0.0], dtype=np.float32), TARGET_SAMPLE_RATE)
    restored = read_wav(write_wav(tmp_path / "clip.wav", loud))
    assert restored.samples.max() <= 1.0
    assert restored.samples.min() >= -1.0


def test_slice_seconds_clamps_to_bounds(call_audio):
    assert len(call_audio.slice_seconds(-5.0, 0.5)) == int(0.5 * call_audio.sample_rate)
    assert len(call_audio.slice_seconds(1e6, 1e6 + 1)) == 0


def test_audio_buffer_rejects_multichannel():
    with pytest.raises(AudioFormatError):
        AudioBuffer(np.zeros((10, 2), dtype=np.float32), 16000)


@pytest.mark.requires_ffmpeg
def test_probe_duration_matches(tmp_path: Path):
    src = _write_noncanonical(tmp_path / "src.wav", rate=16000, channels=1, seconds=2.0)
    duration = probe_duration(src)
    if duration is not None:  # ffprobe is usually but not always installed with ffmpeg
        assert duration == pytest.approx(2.0, abs=0.05)


def test_ffmpeg_available_never_raises():
    assert isinstance(ffmpeg_available(), bool)
