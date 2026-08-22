"""Programmatic two-speaker call generation.

The test suite needs audio with *known* properties -- known talk-time split,
known pitch per speaker, known silence placement -- so that a metric can be
asserted against ground truth rather than against its own previous output. It
also means the repository ships no binary fixtures.

The synthesizer is a source-filter model: a band-limited glottal pulse train at
a chosen F0, shaped by fixed formant resonances, amplitude-modulated at a
syllable rate. It is not speech and Whisper will not transcribe it. It is a
speaker-like signal, which is all the DSP stages need.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import lfilter

from callscope.audio import AudioBuffer
from callscope.schema import Interval

#: Two formant sets that read as clearly different vocal tracts.
Formants = tuple[tuple[float, float], ...]

VOICE_A_FORMANTS: Formants = ((730.0, 90.0), (1090.0, 110.0), (2440.0, 140.0))
VOICE_B_FORMANTS: Formants = ((270.0, 60.0), (2290.0, 100.0), (3010.0, 170.0))


@dataclass(frozen=True, slots=True)
class Utterance:
    """One scripted stretch of synthetic speech."""

    start: float
    end: float
    speaker: str
    f0_hz: float
    syllable_rate_hz: float = 4.0
    amplitude: float = 0.35
    text: str = ""


@dataclass(frozen=True, slots=True)
class CallScript:
    """A complete synthetic call: total length plus the utterances in it."""

    duration: float
    utterances: tuple[Utterance, ...]
    sample_rate: int = 16_000
    noise_db: float = -55.0
    seed: int = 20260822
    formants: dict[str, Formants] = field(
        default_factory=lambda: {
            "SPEAKER_00": VOICE_A_FORMANTS,
            "SPEAKER_01": VOICE_B_FORMANTS,
        }
    )

    def intervals_for(self, speaker: str) -> list[Interval]:
        return [Interval(u.start, u.end) for u in self.utterances if u.speaker == speaker]

    def talk_time(self, speaker: str) -> float:
        return sum(u.end - u.start for u in self.utterances if u.speaker == speaker)


def synthesize_call(script: CallScript) -> AudioBuffer:
    """Render a :class:`CallScript` to a mono float32 buffer.

    Overlapping utterances are summed, which is what makes the overlap and
    interruption metrics testable against a known ground truth.
    """
    n = int(round(script.duration * script.sample_rate))
    rng = np.random.default_rng(script.seed)

    # A low noise floor everywhere: an absolutely silent background makes the
    # VAD's adaptive threshold degenerate, which is not a realistic test.
    noise_amp = 10.0 ** (script.noise_db / 20.0)
    mix = rng.normal(0.0, noise_amp, size=n).astype(np.float64)

    for utt in script.utterances:
        i0 = max(0, int(round(utt.start * script.sample_rate)))
        i1 = min(n, int(round(utt.end * script.sample_rate)))
        if i1 <= i0:
            continue
        formants = script.formants.get(utt.speaker, VOICE_A_FORMANTS)
        mix[i0:i1] += _render_voice(
            n_samples=i1 - i0,
            sample_rate=script.sample_rate,
            f0_hz=utt.f0_hz,
            syllable_rate_hz=utt.syllable_rate_hz,
            amplitude=utt.amplitude,
            formants=formants,
            rng=rng,
        )

    peak = float(np.max(np.abs(mix))) if n else 0.0
    if peak > 0.95:
        mix *= 0.95 / peak
    return AudioBuffer(
        samples=np.clip(mix, -1.0, 1.0).astype(np.float32), sample_rate=script.sample_rate
    )


def _render_voice(
    *,
    n_samples: int,
    sample_rate: int,
    f0_hz: float,
    syllable_rate_hz: float,
    amplitude: float,
    formants: Formants,
    rng: np.random.Generator,
) -> np.ndarray:
    """Source-filter synthesis of one voiced stretch."""
    t = np.arange(n_samples, dtype=np.float64) / sample_rate

    # Source: a sawtooth-like harmonic stack. A pure sine has no harmonics for
    # the autocorrelation pitch tracker to lock onto the way real speech does.
    source = np.zeros(n_samples, dtype=np.float64)
    n_harmonics = max(1, int((sample_rate / 2.0) / max(f0_hz, 1.0)) - 1)
    for h in range(1, min(n_harmonics, 30) + 1):
        source += np.sin(2.0 * np.pi * f0_hz * h * t) / h

    # A little jitter so f0 standard deviation is non-zero, as in real voices.
    source *= 1.0 + 0.01 * rng.normal(size=n_samples)

    # Filter: cascaded two-pole resonators at the formant frequencies.
    filtered = source
    for freq, bandwidth in formants:
        filtered = _resonator(filtered, sample_rate, freq, bandwidth)

    # Syllabic amplitude modulation, raised-cosine so each cycle has one clear
    # nucleus for the envelope peak counter to find.
    envelope = 0.5 * (1.0 - np.cos(2.0 * np.pi * syllable_rate_hz * t))
    envelope = 0.15 + 0.85 * envelope

    # 20 ms raised-cosine fades stop the utterance edges from clicking, which
    # would otherwise show up as spurious energy in the VAD.
    filtered = filtered * envelope * _edge_fade(n_samples, sample_rate, 0.020)

    peak = float(np.max(np.abs(filtered)))
    if peak > 0:
        filtered *= amplitude / peak
    return filtered


def _resonator(x: np.ndarray, sample_rate: int, freq: float, bandwidth: float) -> np.ndarray:
    """Single two-pole resonator, unity gain at the resonant frequency."""
    r = np.exp(-np.pi * bandwidth / sample_rate)
    theta = 2.0 * np.pi * freq / sample_rate
    a = [1.0, -2.0 * r * np.cos(theta), r * r]
    b = [(1.0 - r) * np.sqrt(1.0 - 2.0 * r * np.cos(2.0 * theta) + r * r)]
    return lfilter(b, a, x)


def _edge_fade(n: int, sample_rate: int, fade_seconds: float) -> np.ndarray:
    fade = min(int(fade_seconds * sample_rate), n // 2)
    window = np.ones(n, dtype=np.float64)
    if fade > 0:
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, fade)))
        window[:fade] = ramp
        window[-fade:] = ramp[::-1]
    return window


def default_two_speaker_script() -> CallScript:
    """A short scripted call with deliberately known properties.

    Ground truth baked in:

    * total length 20.0 s
    * SPEAKER_00 (low voice, 110 Hz) talks 8.0 s across 3 turns
    * SPEAKER_01 (high voice, 215 Hz) talks 6.5 s across 3 turns
    * one 3.5 s dead-air gap from 13.0 s to 16.5 s
    * one deliberate 0.6 s overlap at 9.4-10.0 s (SPEAKER_01 cutting in)
    """
    utterances = (
        Utterance(0.5, 3.0, "SPEAKER_00", f0_hz=110.0, syllable_rate_hz=4.0,
                  text="thanks for calling, this is alex, may i take your name"),
        Utterance(3.4, 5.4, "SPEAKER_01", f0_hz=215.0, syllable_rate_hz=5.0,
                  text="hi alex, my order has not arrived yet"),
        Utterance(5.8, 9.8, "SPEAKER_00", f0_hz=110.0, syllable_rate_hz=4.0,
                  text="let me pull that up and check the shipping status for you"),
        Utterance(9.4, 10.0, "SPEAKER_01", f0_hz=215.0, syllable_rate_hz=5.5,
                  text="it was supposed to be here monday"),
        Utterance(10.4, 13.0, "SPEAKER_00", f0_hz=110.0, syllable_rate_hz=4.0,
                  text="i can see the delay, i am reshipping it today at no charge"),
        Utterance(16.5, 20.0, "SPEAKER_01", f0_hz=215.0, syllable_rate_hz=5.0,
                  text="that works, thank you for your help, goodbye"),
    )
    return CallScript(duration=20.0, utterances=utterances)


def script_to_transcript_dict(script: CallScript) -> dict:
    """Whisper-shaped transcript for a script, for the ``fixture`` backend.

    Lets the semantic track be tested end to end without a speech model, since
    the synthetic audio is voice-*like* but carries no words.
    """
    return {
        "language": "en",
        "segments": [
            {
                "start": u.start,
                "end": u.end,
                "text": u.text,
                "speaker": u.speaker,
            }
            for u in script.utterances
            if u.text
        ],
    }
