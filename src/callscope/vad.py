"""Energy-based voice activity detection with an adaptive noise floor.

WebRTC VAD and Silero are both better at this. Neither is a hard dependency
here: WebRTC's Python bindings are unmaintained wheels that fail to build on
current toolchains, and Silero pulls in torch. An adaptive-threshold energy VAD
with hysteresis is the honest dependency-light choice, and on the clean-ish
audio typical of recorded business calls it is adequate. See README limitations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from callscope.features import (
    DEFAULT_HOP_SECONDS,
    frame_energy_db,
    frame_signal,
    zero_crossing_rate,
)
from callscope.schema import Interval


@dataclass(frozen=True, slots=True)
class VadConfig:
    """Tunables for :func:`detect_speech`."""

    #: Onset threshold, in dB above the estimated noise floor.
    onset_db: float = 9.0
    #: Offset threshold. Lower than onset, giving Schmitt-trigger hysteresis so
    #: that a brief dip mid-word does not split one utterance into two.
    offset_db: float = 5.0
    #: Percentile of frame energy taken as the noise floor.
    noise_percentile: float = 15.0
    #: Absolute floor: frames quieter than this are never speech, which stops a
    #: digitally silent recording from producing a full-file "speech" segment.
    absolute_floor_db: float = -70.0
    min_speech_seconds: float = 0.20
    min_silence_seconds: float = 0.15
    #: Symmetric padding applied to each detected segment.
    pad_seconds: float = 0.05
    #: Frames above the energy threshold but with a very high zero-crossing rate
    #: are usually fricative-like noise or line hiss, not voiced speech.
    max_zcr: float = 0.75
    hop_seconds: float = DEFAULT_HOP_SECONDS
    frame_seconds: float = 0.025


def detect_speech(
    samples: np.ndarray,
    sample_rate: int,
    config: VadConfig | None = None,
) -> list[Interval]:
    """Return speech intervals, in seconds, sorted and non-overlapping.

    Args:
        samples: Mono float32 audio in ``[-1, 1]``.
        sample_rate: Sample rate in Hz.
        config: Tunables; the defaults suit 16 kHz recorded telephony.

    Returns:
        Speech intervals. An empty list means no speech was detected, which is a
        legitimate outcome (hold music, dead line) and not an error.
    """
    cfg = config or VadConfig()
    x = np.asarray(samples, dtype=np.float32)
    if x.size == 0:
        return []

    frames, grid = frame_signal(
        x, sample_rate, frame_seconds=cfg.frame_seconds, hop_seconds=cfg.hop_seconds
    )
    energy = frame_energy_db(frames)
    zcr = zero_crossing_rate(frames)

    noise_floor = float(np.percentile(energy, cfg.noise_percentile))
    onset = max(noise_floor + cfg.onset_db, cfg.absolute_floor_db)
    offset = max(noise_floor + cfg.offset_db, cfg.absolute_floor_db - 3.0)

    active = _hysteresis(energy, onset=onset, offset=offset)
    active &= zcr <= cfg.max_zcr

    hop = cfg.hop_seconds
    spans = _runs_to_intervals(active, grid.frame_to_time, hop)
    spans = _bridge_gaps(spans, cfg.min_silence_seconds)
    spans = [s for s in spans if s.duration >= cfg.min_speech_seconds]

    total = len(x) / float(sample_rate)
    padded: list[Interval] = []
    for span in spans:
        start = max(0.0, span.start - cfg.pad_seconds)
        end = min(total, span.end + cfg.pad_seconds)
        if end > start:
            padded.append(Interval(start, end))
    return _bridge_gaps(padded, 0.0)


def _hysteresis(energy: np.ndarray, *, onset: float, offset: float) -> np.ndarray:
    """Schmitt trigger over the energy contour."""
    active = np.zeros(len(energy), dtype=bool)
    state = False
    for i, value in enumerate(energy):
        if state:
            state = value >= offset
        else:
            state = value >= onset
        active[i] = state
    return active


def _runs_to_intervals(mask: np.ndarray, frame_to_time, hop: float) -> list[Interval]:
    """Convert a boolean frame mask into time intervals."""
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    out: list[Interval] = []
    for start_f, end_f in zip(edges[0::2], edges[1::2], strict=True):
        start = max(0.0, frame_to_time(start_f) - hop / 2.0)
        end = frame_to_time(end_f - 1) + hop / 2.0
        if end > start:
            out.append(Interval(start, end))
    return out


def _bridge_gaps(intervals: list[Interval], min_gap: float) -> list[Interval]:
    """Merge intervals separated by less than ``min_gap`` seconds."""
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda i: i.start)
    merged = [ordered[0]]
    for nxt in ordered[1:]:
        last = merged[-1]
        if nxt.start - last.end < min_gap or nxt.start <= last.end:
            merged[-1] = Interval(last.start, max(last.end, nxt.end))
        else:
            merged.append(nxt)
    return merged


def split_long_segments(
    intervals: list[Interval],
    *,
    max_seconds: float = 4.0,
    target_seconds: float = 2.0,
) -> list[Interval]:
    """Chop overlong VAD segments into clustering-sized pieces.

    A single VAD segment can span a speaker change when neither party pauses.
    Diarization clusters at the segment level, so a segment that straddles a turn
    boundary is unrecoverable. Splitting long segments trades a little embedding
    stability for the ability to place a boundary at all.
    """
    out: list[Interval] = []
    for iv in intervals:
        if iv.duration <= max_seconds:
            out.append(iv)
            continue
        n = max(2, int(round(iv.duration / target_seconds)))
        step = iv.duration / n
        for k in range(n):
            start = iv.start + k * step
            end = iv.start + (k + 1) * step if k < n - 1 else iv.end
            out.append(Interval(start, end))
    return out
