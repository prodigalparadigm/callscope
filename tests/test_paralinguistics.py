"""Paralinguistic metrics, asserted against constructed signals with known truth.

Every test here builds a signal whose properties are known by construction --
exact talk times, exact silence placement, exact fundamental frequency, exact
syllabic modulation rate -- and checks the metric recovers them. Turns are
supplied directly rather than taken from the diarizer, so a diarization
regression cannot masquerade as a metric regression.
"""

from __future__ import annotations

import numpy as np
import pytest

from callscope.audio import AudioBuffer
from callscope.fixtures import CallScript, Utterance, synthesize_call
from callscope.paralinguistics import ParalinguisticConfig, analyze_paralinguistics
from callscope.schema import SPEAKER_LABELS, SpeakerTurn, Transcript, TranscriptSegment


def _build(script: CallScript):
    """Render a script and return (audio, turns) with turns == ground truth."""
    audio = synthesize_call(script)
    turns = [SpeakerTurn(u.start, u.end, u.speaker) for u in script.utterances]
    return audio, turns


#: 20 s call. SPEAKER_00 talks 0-6 and 14-20 (12 s); SPEAKER_01 talks 6-9 (3 s);
#: silence 9-14 (5 s). No overlap.
SIMPLE = CallScript(
    duration=20.0,
    utterances=(
        Utterance(0.0, 6.0, "SPEAKER_00", f0_hz=110.0, syllable_rate_hz=4.0),
        Utterance(6.0, 9.0, "SPEAKER_01", f0_hz=220.0, syllable_rate_hz=6.0),
        Utterance(14.0, 20.0, "SPEAKER_00", f0_hz=110.0, syllable_rate_hz=4.0),
    ),
)

#: SPEAKER_01 cuts in at 4.0 s while SPEAKER_00 runs to 6.0 s: exactly 2.0 s of
#: overlap, and exactly one interruption, attributed to SPEAKER_01.
INTERRUPTED = CallScript(
    duration=12.0,
    utterances=(
        Utterance(0.0, 6.0, "SPEAKER_00", f0_hz=110.0),
        Utterance(4.0, 8.0, "SPEAKER_01", f0_hz=220.0),
    ),
)


# --- talk time -------------------------------------------------------------

def test_talk_time_matches_construction():
    audio, turns = _build(SIMPLE)
    p = analyze_paralinguistics(audio, turns)
    assert p.speaker("SPEAKER_00").talk_time_seconds == pytest.approx(12.0)
    assert p.speaker("SPEAKER_01").talk_time_seconds == pytest.approx(3.0)


def test_talk_time_ratios_sum_to_one():
    audio, turns = _build(SIMPLE)
    p = analyze_paralinguistics(audio, turns)
    assert p.speaker("SPEAKER_00").talk_time_ratio == pytest.approx(0.8, abs=1e-3)
    assert p.speaker("SPEAKER_01").talk_time_ratio == pytest.approx(0.2, abs=1e-3)
    assert sum(s.talk_time_ratio for s in p.speakers) == pytest.approx(1.0, abs=1e-3)


def test_talk_ratios_still_sum_to_one_under_overlap():
    """Overlap double-counts in the numerator, so it must double-count in the base."""
    audio, turns = _build(INTERRUPTED)
    p = analyze_paralinguistics(audio, turns)
    assert sum(s.talk_time_ratio for s in p.speakers) == pytest.approx(1.0, abs=1e-3)


def test_turn_counts_and_lengths():
    audio, turns = _build(SIMPLE)
    p = analyze_paralinguistics(audio, turns)
    sp0 = p.speaker("SPEAKER_00")
    assert sp0.turn_count == 2
    assert sp0.mean_turn_seconds == pytest.approx(6.0)
    assert sp0.longest_turn_seconds == pytest.approx(6.0)


def test_a_speaker_who_never_talks_still_gets_a_row():
    """A 100/0 split is a finding; consumers must not KeyError on a monologue."""
    script = CallScript(
        duration=6.0, utterances=(Utterance(0.5, 5.5, "SPEAKER_00", f0_hz=120.0),)
    )
    audio, turns = _build(script)
    p = analyze_paralinguistics(audio, turns)
    assert {s.speaker for s in p.speakers} == set(SPEAKER_LABELS)
    silent = p.speaker("SPEAKER_01")
    assert silent.talk_time_seconds == 0.0
    assert silent.turn_count == 0
    assert silent.f0_mean_hz is None


# --- silence and dead air --------------------------------------------------

def test_silence_totals_are_exact():
    audio, turns = _build(SIMPLE)
    p = analyze_paralinguistics(audio, turns)
    assert p.speech_seconds == pytest.approx(15.0)
    assert p.silence_seconds == pytest.approx(5.0)
    assert p.silence_ratio == pytest.approx(0.25, abs=1e-3)
    assert p.duration_seconds == pytest.approx(20.0)


def test_dead_air_event_is_located():
    audio, turns = _build(SIMPLE)
    p = analyze_paralinguistics(audio, turns)
    assert len(p.dead_air_events) == 1
    event = p.dead_air_events[0]
    assert (event.start, event.end) == pytest.approx((9.0, 14.0))
    assert p.longest_silence_seconds == pytest.approx(5.0)


def test_dead_air_threshold_is_respected():
    """A 2 s gap is conversational; a 2 s threshold makes it dead air."""
    script = CallScript(
        duration=10.0,
        utterances=(
            Utterance(0.0, 4.0, "SPEAKER_00", f0_hz=110.0),
            Utterance(6.0, 10.0, "SPEAKER_01", f0_hz=220.0),
        ),
    )
    audio, turns = _build(script)
    assert analyze_paralinguistics(
        audio, turns, config=ParalinguisticConfig(dead_air_seconds=3.0)
    ).dead_air_events == []
    strict = analyze_paralinguistics(
        audio, turns, config=ParalinguisticConfig(dead_air_seconds=2.0)
    )
    assert len(strict.dead_air_events) == 1


def test_leading_and_trailing_silence_are_counted():
    script = CallScript(
        duration=10.0, utterances=(Utterance(2.0, 7.0, "SPEAKER_00", f0_hz=110.0),)
    )
    audio, turns = _build(script)
    p = analyze_paralinguistics(audio, turns)
    assert p.silence_seconds == pytest.approx(5.0)  # 2 s before + 3 s after


# --- overlap and interruption ---------------------------------------------

def test_overlap_duration_is_exact():
    audio, turns = _build(INTERRUPTED)
    p = analyze_paralinguistics(audio, turns)
    assert p.overlap_seconds == pytest.approx(2.0)
    assert len(p.overlap_events) == 1
    assert (p.overlap_events[0].start, p.overlap_events[0].end) == pytest.approx((4.0, 6.0))


def test_interruption_is_attributed_to_the_intruder():
    audio, turns = _build(INTERRUPTED)
    p = analyze_paralinguistics(audio, turns)
    assert p.interruptions["SPEAKER_01"] == 1
    assert p.interruptions["SPEAKER_00"] == 0


def test_clean_turn_taking_registers_no_interruption():
    audio, turns = _build(SIMPLE)
    p = analyze_paralinguistics(audio, turns)
    assert p.overlap_seconds == 0.0
    assert sum(p.interruptions.values()) == 0


def test_brief_latch_on_is_not_an_interruption():
    """A 100 ms overlap at a turn boundary is turn-taking, not interruption."""
    script = CallScript(
        duration=10.0,
        utterances=(
            Utterance(0.0, 5.0, "SPEAKER_00", f0_hz=110.0),
            Utterance(4.9, 9.0, "SPEAKER_01", f0_hz=220.0),
        ),
    )
    audio, turns = _build(script)
    p = analyze_paralinguistics(audio, turns)
    assert sum(p.interruptions.values()) == 0
    assert p.overlap_seconds == 0.0  # below min_overlap_seconds


# --- response latency ------------------------------------------------------

def test_mean_response_latency():
    """Gaps are 6.0->6.0 (0 s) and 9.0->14.0 (5 s); mean 2.5 s."""
    audio, turns = _build(SIMPLE)
    p = analyze_paralinguistics(audio, turns)
    assert p.mean_response_latency_seconds == pytest.approx(2.5, abs=1e-3)


def test_latency_ignores_overlaps():
    """Interrupting must not buy a flatteringly low response time."""
    audio, turns = _build(INTERRUPTED)
    p = analyze_paralinguistics(audio, turns)
    assert p.mean_response_latency_seconds is None


# --- pitch -----------------------------------------------------------------

def test_pitch_recovers_the_synthesized_f0():
    audio, turns = _build(SIMPLE)
    p = analyze_paralinguistics(audio, turns)
    assert p.speaker("SPEAKER_00").f0_mean_hz == pytest.approx(110.0, rel=0.02)
    assert p.speaker("SPEAKER_01").f0_mean_hz == pytest.approx(220.0, rel=0.02)


def test_pitch_variance_rises_with_pitch_variation():
    """A speaker who varies pitch across turns must show a larger semitone spread."""
    flat = CallScript(
        duration=12.0,
        utterances=(
            Utterance(0.0, 3.0, "SPEAKER_00", f0_hz=150.0),
            Utterance(3.5, 6.5, "SPEAKER_00", f0_hz=150.0),
            Utterance(7.0, 10.0, "SPEAKER_00", f0_hz=150.0),
        ),
    )
    varied = CallScript(
        duration=12.0,
        utterances=(
            Utterance(0.0, 3.0, "SPEAKER_00", f0_hz=120.0),
            Utterance(3.5, 6.5, "SPEAKER_00", f0_hz=150.0),
            Utterance(7.0, 10.0, "SPEAKER_00", f0_hz=190.0),
        ),
    )
    flat_sd = analyze_paralinguistics(*_build(flat)).speaker("SPEAKER_00").f0_std_semitones
    varied_sd = analyze_paralinguistics(*_build(varied)).speaker("SPEAKER_00").f0_std_semitones
    assert flat_sd < 1.0
    assert varied_sd > 3.0


def test_voiced_fraction_is_high_for_voiced_synthesis():
    audio, turns = _build(SIMPLE)
    p = analyze_paralinguistics(audio, turns)
    assert p.speaker("SPEAKER_00").voiced_fraction > 0.9


# --- speech rate -----------------------------------------------------------

def test_syllable_rate_recovers_the_modulation_frequency():
    """The envelope is modulated at exactly 4 Hz and 6 Hz by construction."""
    audio, turns = _build(SIMPLE)
    p = analyze_paralinguistics(audio, turns)
    assert p.speaker("SPEAKER_00").syllable_rate_hz == pytest.approx(4.0, abs=0.35)
    assert p.speaker("SPEAKER_01").syllable_rate_hz == pytest.approx(6.0, abs=0.5)


def test_syllable_rate_tracks_a_faster_talker():
    slow = CallScript(
        duration=8.0, utterances=(Utterance(0.0, 8.0, "SPEAKER_00", f0_hz=130.0,
                                            syllable_rate_hz=3.0),))
    fast = CallScript(
        duration=8.0, utterances=(Utterance(0.0, 8.0, "SPEAKER_00", f0_hz=130.0,
                                            syllable_rate_hz=7.0),))
    slow_rate = analyze_paralinguistics(*_build(slow)).speaker("SPEAKER_00").syllable_rate_hz
    fast_rate = analyze_paralinguistics(*_build(fast)).speaker("SPEAKER_00").syllable_rate_hz
    assert slow_rate == pytest.approx(3.0, abs=0.35)
    assert fast_rate == pytest.approx(7.0, abs=0.6)
    assert fast_rate > slow_rate


def test_words_per_minute_uses_the_transcript_when_present():
    audio, turns = _build(SIMPLE)
    # 30 words over SPEAKER_00's 12 s of talk time == 150 wpm.
    transcript = Transcript(
        segments=[
            TranscriptSegment(0.0, 6.0, " ".join(["word"] * 30), speaker="SPEAKER_00")
        ]
    )
    p = analyze_paralinguistics(audio, turns, transcript=transcript)
    assert p.speaker("SPEAKER_00").words_per_minute == pytest.approx(150.0, abs=0.1)
    assert p.speaker("SPEAKER_01").words_per_minute is None


def test_metrics_computed_without_a_transcript():
    """The paralinguistic track must be a genuinely independent track."""
    audio, turns = _build(SIMPLE)
    p = analyze_paralinguistics(audio, turns, transcript=None)
    assert p.speaker("SPEAKER_00").syllable_rate_hz > 0
    assert p.speaker("SPEAKER_00").f0_mean_hz is not None
    assert all(s.words_per_minute is None for s in p.speakers)


# --- degenerate input ------------------------------------------------------

def test_no_turns_produces_a_full_silence_profile():
    audio = AudioBuffer(np.zeros(16_000 * 5, dtype=np.float32), 16_000)
    p = analyze_paralinguistics(audio, [])
    assert p.speech_seconds == 0.0
    assert p.silence_seconds == pytest.approx(5.0)
    assert p.silence_ratio == pytest.approx(1.0)
    assert p.mean_response_latency_seconds is None
    assert len(p.speakers) == 2


def test_unsorted_turns_are_handled():
    audio, turns = _build(SIMPLE)
    shuffled = list(reversed(turns))
    assert (
        analyze_paralinguistics(audio, shuffled).silence_seconds
        == pytest.approx(analyze_paralinguistics(audio, turns).silence_seconds)
    )
