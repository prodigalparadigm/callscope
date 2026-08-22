"""The paralinguistic track: how the call sounded, independent of what was said.

Every metric here is derived from the waveform and the diarization timeline. The
transcript is used for exactly one optional metric (words per minute) and is
never required -- that separation is what makes this a genuinely parallel track
rather than a second view of the transcript.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from callscope.audio import AudioBuffer
from callscope.features import amplitude_envelope, estimate_f0
from callscope.schema import (
    SPEAKER_LABELS,
    Interval,
    ParalinguisticProfile,
    SpeakerParalinguistics,
    SpeakerTurn,
    Transcript,
)


@dataclass(frozen=True, slots=True)
class ParalinguisticConfig:
    """Thresholds for the paralinguistic metrics."""

    #: A silence at least this long counts as dead air, the metric supervisors
    #: actually care about. Shorter gaps are normal conversational rhythm.
    dead_air_seconds: float = 3.0
    #: Two speakers must overlap by at least this long to count as an overlap
    #: event; anything shorter is usually a diarization boundary artifact.
    min_overlap_seconds: float = 0.20
    #: An overlap counts as an interruption when the intruding speaker starts
    #: at least this far before the incumbent's turn ends -- i.e. it is not
    #: mere latch-on at a turn boundary.
    interruption_margin_seconds: float = 0.40
    #: A syllable nucleus must exceed this fraction of the segment's peak
    #: envelope value to be counted.
    syllable_peak_ratio: float = 0.30
    #: Minimum spacing between syllable nuclei. 120 ms caps the rate at a
    #: physically plausible ~8 syllables/second.
    min_syllable_spacing_seconds: float = 0.12


def analyze_paralinguistics(
    audio: AudioBuffer,
    turns: list[SpeakerTurn],
    *,
    transcript: Transcript | None = None,
    config: ParalinguisticConfig | None = None,
) -> ParalinguisticProfile:
    """Compute the full paralinguistic profile for a call.

    Args:
        audio: Canonical mono buffer for the whole call.
        turns: Diarized speaker turns. May be empty.
        transcript: Optional, used only for words-per-minute.
        config: Thresholds; defaults suit business calls.

    Returns:
        A :class:`~callscope.schema.ParalinguisticProfile`. Metrics that cannot
        be computed (no speech, no voiced frames) come back as ``None`` or zero
        rather than raising -- a silent recording is data, not a failure.
    """
    cfg = config or ParalinguisticConfig()
    duration = audio.duration

    ordered = sorted(turns, key=lambda t: t.start)
    speech_intervals = _merge_intervals([Interval(t.start, t.end) for t in ordered])
    #: Union of all turns: the right basis for silence, since two people talking
    #: at once is still one second of non-silence.
    speech_seconds = sum(i.duration for i in speech_intervals)
    #: Sum of per-speaker turn durations, which double-counts overlap. This is
    #: the right basis for talk-time share, so that the two ratios sum to 1.0
    #: regardless of how much cross-talk the call contained.
    total_talk_seconds = sum(t.duration for t in ordered)
    silences = _gaps(speech_intervals, duration)
    silence_seconds = sum(s.duration for s in silences)

    silence_durations = np.array([s.duration for s in silences], dtype=np.float64)
    dead_air = [s for s in silences if s.duration >= cfg.dead_air_seconds]

    overlaps = _overlap_events(ordered, cfg.min_overlap_seconds)
    interruptions = _count_interruptions(ordered, cfg)
    latency = _mean_response_latency(ordered)

    f0, f0_grid = estimate_f0(audio.samples, audio.sample_rate)
    envelope, env_grid = amplitude_envelope(audio.samples, audio.sample_rate)

    speakers: list[SpeakerParalinguistics] = []
    for label in SPEAKER_LABELS:
        # A row is emitted for both labels even when one never speaks: a 100/0
        # talk-time split is a finding, and a consumer keying on SPEAKER_01
        # should not KeyError on a monologue.
        own = [t for t in ordered if t.speaker == label]
        speakers.append(
            _speaker_metrics(
                label=label,
                turns=own,
                total_talk=total_talk_seconds,
                f0=f0,
                f0_grid=f0_grid,
                envelope=envelope,
                env_grid=env_grid,
                transcript=transcript,
                cfg=cfg,
            )
        )

    return ParalinguisticProfile(
        duration_seconds=round(duration, 3),
        speech_seconds=round(speech_seconds, 3),
        silence_seconds=round(silence_seconds, 3),
        silence_ratio=round(silence_seconds / duration, 4) if duration > 0 else 0.0,
        longest_silence_seconds=round(float(silence_durations.max()), 3)
        if silence_durations.size
        else 0.0,
        silence_p50_seconds=round(float(np.percentile(silence_durations, 50)), 3)
        if silence_durations.size
        else 0.0,
        silence_p90_seconds=round(float(np.percentile(silence_durations, 90)), 3)
        if silence_durations.size
        else 0.0,
        dead_air_events=dead_air,
        overlap_seconds=round(sum(o.duration for o in overlaps), 3),
        overlap_events=overlaps,
        interruptions=interruptions,
        mean_response_latency_seconds=latency,
        speakers=speakers,
    )


def _speaker_metrics(
    *,
    label: str,
    turns: list[SpeakerTurn],
    total_talk: float,
    f0: np.ndarray,
    f0_grid,
    envelope: np.ndarray,
    env_grid,
    transcript: Transcript | None,
    cfg: ParalinguisticConfig,
) -> SpeakerParalinguistics:
    talk = sum(t.duration for t in turns)
    durations = [t.duration for t in turns]

    pitch_values: list[float] = []
    voiced_frames = 0
    total_frames = 0
    syllables = 0

    for t in turns:
        a, b = f0_grid.time_to_frame(t.start), f0_grid.time_to_frame(t.end)
        block = f0[a : max(b, a + 1)]
        total_frames += len(block)
        good = block[~np.isnan(block)]
        voiced_frames += len(good)
        pitch_values.extend(good.tolist())

        ea, eb = env_grid.time_to_frame(t.start), env_grid.time_to_frame(t.end)
        syllables += _count_syllable_nuclei(
            envelope[ea : max(eb, ea + 1)], env_grid.frame_rate, cfg
        )

    pitch = np.array(pitch_values, dtype=np.float64)
    if pitch.size >= 2:
        f0_mean = float(pitch.mean())
        f0_std = float(pitch.std())
        # Semitone spread is scale-free: a 20 Hz swing means something very
        # different at a 100 Hz baseline than at a 250 Hz baseline.
        semitones = 12.0 * np.log2(np.maximum(pitch, 1e-6) / max(f0_mean, 1e-6))
        f0_std_semitones = float(semitones.std())
    elif pitch.size == 1:
        f0_mean, f0_std, f0_std_semitones = float(pitch[0]), 0.0, 0.0
    else:
        f0_mean = f0_std = f0_std_semitones = None

    wpm: float | None = None
    if transcript is not None and talk > 0:
        words = transcript.word_count(label)
        if words:
            wpm = round(words / (talk / 60.0), 1)

    return SpeakerParalinguistics(
        speaker=label,
        talk_time_seconds=round(talk, 3),
        talk_time_ratio=round(talk / total_talk, 4) if total_talk > 0 else 0.0,
        turn_count=len(turns),
        mean_turn_seconds=round(float(np.mean(durations)), 3) if durations else 0.0,
        longest_turn_seconds=round(max(durations), 3) if durations else 0.0,
        syllable_rate_hz=round(syllables / talk, 3) if talk > 0 else 0.0,
        words_per_minute=wpm,
        f0_mean_hz=round(f0_mean, 2) if f0_mean is not None else None,
        f0_std_hz=round(f0_std, 2) if f0_std is not None else None,
        f0_std_semitones=round(f0_std_semitones, 3) if f0_std_semitones is not None else None,
        voiced_fraction=round(voiced_frames / total_frames, 4) if total_frames else 0.0,
    )


def _count_syllable_nuclei(
    envelope: np.ndarray, frame_rate: float, cfg: ParalinguisticConfig
) -> int:
    """Count local maxima of the amplitude envelope above a relative threshold.

    A syllable-nucleus proxy for speech rate. It is a proxy, not a syllable
    count: it undercounts fast connected speech and overcounts amplitude-modulated
    noise. Its value is that it needs no transcript, so speech rate stays
    available even when transcription is unavailable or wrong.
    """
    if envelope.size < 3:
        return 0
    peak = float(envelope.max())
    if peak <= 0:
        return 0
    threshold = cfg.syllable_peak_ratio * peak
    min_spacing = max(1, int(round(cfg.min_syllable_spacing_seconds * frame_rate)))

    count = 0
    last_index = -min_spacing
    for i in range(1, len(envelope) - 1):
        v = envelope[i]
        if v < threshold:
            continue
        if v >= envelope[i - 1] and v > envelope[i + 1] and (i - last_index) >= min_spacing:
            count += 1
            last_index = i
    return count


def _merge_intervals(intervals: list[Interval]) -> list[Interval]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda i: i.start)
    out = [ordered[0]]
    for nxt in ordered[1:]:
        last = out[-1]
        if nxt.start <= last.end:
            out[-1] = Interval(last.start, max(last.end, nxt.end))
        else:
            out.append(nxt)
    return out


def _gaps(speech: list[Interval], total: float) -> list[Interval]:
    """Silent stretches, including leading and trailing silence."""
    gaps: list[Interval] = []
    cursor = 0.0
    for iv in speech:
        if iv.start > cursor:
            gaps.append(Interval(cursor, iv.start))
        cursor = max(cursor, iv.end)
    if total > cursor:
        gaps.append(Interval(cursor, total))
    return gaps


def _overlap_events(turns: list[SpeakerTurn], min_seconds: float) -> list[Interval]:
    """Intervals where two different speakers are talking simultaneously."""
    events: list[Interval] = []
    for i, a in enumerate(turns):
        for b in turns[i + 1 :]:
            if b.start >= a.end:
                break  # turns are sorted; nothing later can overlap a
            if b.speaker == a.speaker:
                continue
            start, end = max(a.start, b.start), min(a.end, b.end)
            if end - start >= min_seconds:
                events.append(Interval(start, end))
    return _merge_intervals(events)


def _count_interruptions(
    turns: list[SpeakerTurn], cfg: ParalinguisticConfig
) -> dict[str, int]:
    """Per-speaker count of turns that begin well inside another's turn."""
    counts: dict[str, int] = dict.fromkeys(SPEAKER_LABELS, 0)
    for i, intruder in enumerate(turns):
        for incumbent in turns[:i]:
            if incumbent.speaker == intruder.speaker:
                continue
            if incumbent.start < intruder.start < incumbent.end:
                overlap = min(incumbent.end, intruder.end) - intruder.start
                remaining = incumbent.end - intruder.start
                if (
                    overlap >= cfg.min_overlap_seconds
                    and remaining >= cfg.interruption_margin_seconds
                ):
                    counts[intruder.speaker] = counts.get(intruder.speaker, 0) + 1
                    break
    return counts


def _mean_response_latency(turns: list[SpeakerTurn]) -> float | None:
    """Mean gap between one speaker finishing and the other starting.

    Negative gaps (overlaps) are excluded: they are counted as interruptions, and
    averaging them into latency would let an interrupting agent post a flatteringly
    low response time.
    """
    gaps: list[float] = []
    for prev, nxt in zip(turns, turns[1:], strict=False):
        if prev.speaker == nxt.speaker:
            continue
        gap = nxt.start - prev.end
        if gap >= 0:
            gaps.append(gap)
    if not gaps:
        return None
    return round(float(np.mean(gaps)), 3)
