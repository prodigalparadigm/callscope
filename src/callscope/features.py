"""Signal-processing primitives: framing, MFCCs, and pitch tracking.

Implemented directly on numpy/scipy rather than pulling in librosa. The feature
set here is deliberately small -- it exists to support two-speaker clustering and
the paralinguistic metrics, not to be a general audio library.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import dct
from scipy.signal import get_window

EPS = 1e-10

DEFAULT_FRAME_SECONDS = 0.025
DEFAULT_HOP_SECONDS = 0.010

#: Human speech fundamental frequency bounds. Wide enough to cover low male and
#: high female voices without admitting the halving/doubling errors that a wider
#: search invites.
F0_MIN_HZ = 60.0
F0_MAX_HZ = 400.0


@dataclass(frozen=True, slots=True)
class FrameGrid:
    """The frame/hop geometry used to index every frame-rate feature."""

    sample_rate: int
    frame_length: int
    hop_length: int
    n_frames: int

    @property
    def frame_rate(self) -> float:
        """Frames per second."""
        return self.sample_rate / float(self.hop_length)

    def frame_to_time(self, index: float) -> float:
        """Centre time in seconds of frame ``index``."""
        return (index * self.hop_length + self.frame_length / 2.0) / self.sample_rate

    def time_to_frame(self, seconds: float) -> int:
        """Nearest frame index whose centre is at ``seconds``, clamped in range."""
        raw = (seconds * self.sample_rate - self.frame_length / 2.0) / self.hop_length
        return int(np.clip(round(raw), 0, max(self.n_frames - 1, 0)))


def frame_signal(
    samples: np.ndarray,
    sample_rate: int,
    *,
    frame_seconds: float = DEFAULT_FRAME_SECONDS,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
) -> tuple[np.ndarray, FrameGrid]:
    """Split ``samples`` into overlapping frames.

    Returns a ``(n_frames, frame_length)`` array (a view where possible) and the
    grid describing it. Signals shorter than one frame are zero-padded to exactly
    one frame so that downstream code never has to special-case empty input.
    """
    frame_length = max(1, int(round(frame_seconds * sample_rate)))
    hop_length = max(1, int(round(hop_seconds * sample_rate)))
    x = np.asarray(samples, dtype=np.float32)

    if len(x) < frame_length:
        x = np.pad(x, (0, frame_length - len(x)))

    n_frames = 1 + (len(x) - frame_length) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, frame_length),
        strides=(x.strides[0] * hop_length, x.strides[0]),
        writeable=False,
    )
    grid = FrameGrid(
        sample_rate=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
        n_frames=n_frames,
    )
    return frames, grid


def frame_energy_db(frames: np.ndarray) -> np.ndarray:
    """Per-frame RMS energy in dBFS."""
    rms = np.sqrt(np.mean(np.square(frames.astype(np.float64)), axis=1) + EPS)
    return 20.0 * np.log10(rms + EPS)


def zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    """Per-frame zero-crossing rate in ``[0, 1]``."""
    signs = np.signbit(frames)
    return np.mean(signs[:, 1:] != signs[:, :-1], axis=1)


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int = 40,
    fmin: float = 50.0,
    fmax: float | None = None,
) -> np.ndarray:
    """Slaney-style triangular mel filterbank, shape ``(n_mels, n_fft // 2 + 1)``."""
    if fmax is None:
        fmax = sample_rate / 2.0
    fmax = min(fmax, sample_rate / 2.0)
    n_bins = n_fft // 2 + 1

    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = np.asarray(mel_to_hz(mel_points), dtype=np.float64)
    bin_freqs = np.linspace(0.0, sample_rate / 2.0, n_bins)

    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    for m in range(n_mels):
        left, centre, right = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        if right <= left:
            continue
        rising = (bin_freqs - left) / max(centre - left, EPS)
        falling = (right - bin_freqs) / max(right - centre, EPS)
        fb[m] = np.clip(np.minimum(rising, falling), 0.0, None)
        # Area normalization keeps filter gain independent of bandwidth, so a
        # wide high-frequency filter does not dominate the cepstrum.
        fb[m] *= 2.0 / max(right - left, EPS)
    return fb


def mfcc(
    samples: np.ndarray,
    sample_rate: int,
    *,
    n_mfcc: int = 20,
    n_mels: int = 40,
    frame_seconds: float = DEFAULT_FRAME_SECONDS,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
    preemphasis: float = 0.97,
) -> tuple[np.ndarray, FrameGrid]:
    """Compute MFCCs, shape ``(n_frames, n_mfcc)``.

    Standard recipe: pre-emphasis, Hann window, power spectrum, mel filterbank,
    log, DCT-II with orthonormal scaling.
    """
    x = np.asarray(samples, dtype=np.float32)
    if preemphasis:
        x = np.concatenate(([x[0]] if len(x) else [0.0], x[1:] - preemphasis * x[:-1]))
        x = x.astype(np.float32)

    frames, grid = frame_signal(
        x, sample_rate, frame_seconds=frame_seconds, hop_seconds=hop_seconds
    )
    n_fft = int(2 ** np.ceil(np.log2(max(grid.frame_length, 2))))
    window = get_window("hann", grid.frame_length, fftbins=True).astype(np.float64)

    windowed = frames.astype(np.float64) * window
    spectrum = np.fft.rfft(windowed, n=n_fft, axis=1)
    power = (np.abs(spectrum) ** 2) / float(n_fft)

    fb = mel_filterbank(sample_rate, n_fft, n_mels=n_mels)
    mel_energy = power @ fb.T
    log_mel = np.log(mel_energy + EPS)
    coeffs = dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]
    return coeffs.astype(np.float64), grid


def cepstral_mean_variance_normalize(coeffs: np.ndarray) -> np.ndarray:
    """CMVN over the whole utterance.

    Removes the fixed channel response -- microphone, codec, line -- which
    otherwise dominates MFCC distance and makes two speakers on the same line
    look identical while the same speaker on two lines looks different.
    """
    if coeffs.size == 0:
        return coeffs
    mean = coeffs.mean(axis=0, keepdims=True)
    std = coeffs.std(axis=0, keepdims=True)
    return (coeffs - mean) / np.maximum(std, 1e-6)


def estimate_f0(
    samples: np.ndarray,
    sample_rate: int,
    *,
    frame_seconds: float = 0.040,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
    fmin: float = F0_MIN_HZ,
    fmax: float = F0_MAX_HZ,
    voicing_threshold: float = 0.35,
) -> tuple[np.ndarray, FrameGrid]:
    """Frame-wise fundamental frequency via normalized autocorrelation.

    Returns an array of Hz with ``np.nan`` for unvoiced frames. This is a
    classic autocorrelation tracker, not YIN or CREPE: it is accurate enough for
    per-speaker pitch *statistics* on clean speech and costs no extra dependency.
    Octave errors are suppressed by preferring the earliest lag whose peak is
    within 85% of the global maximum.
    """
    frames, grid = frame_signal(
        samples, sample_rate, frame_seconds=frame_seconds, hop_seconds=hop_seconds
    )
    min_lag = max(2, int(sample_rate / fmax))
    max_lag = min(grid.frame_length - 1, int(sample_rate / fmin))
    f0 = np.full(grid.n_frames, np.nan, dtype=np.float64)
    if max_lag <= min_lag:
        return f0, grid

    n_fft = int(2 ** np.ceil(np.log2(2 * grid.frame_length)))
    x = frames.astype(np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    window = get_window("hann", grid.frame_length, fftbins=True).astype(np.float64)
    x = x * window

    spec = np.fft.rfft(x, n=n_fft, axis=1)
    acf = np.fft.irfft(np.abs(spec) ** 2, n=n_fft, axis=1)[:, : max_lag + 1]

    zero_lag = acf[:, 0]
    silent = zero_lag <= EPS
    norm = np.divide(acf, np.maximum(zero_lag[:, None], EPS))

    search = norm[:, min_lag : max_lag + 1]
    if search.shape[1] == 0:
        return f0, grid

    peak_val = search.max(axis=1)
    voiced = (peak_val >= voicing_threshold) & (~silent)

    for i in np.flatnonzero(voiced):
        row = search[i]
        # Earliest lag within 85% of the max: guards against picking 2*T0.
        candidates = np.flatnonzero(row >= 0.85 * row.max())
        lag_index = int(candidates[0])
        lag = min_lag + lag_index
        # Parabolic interpolation around the chosen peak for sub-sample accuracy.
        if 0 < lag_index < len(row) - 1:
            y0, y1, y2 = row[lag_index - 1], row[lag_index], row[lag_index + 1]
            denom = y0 - 2.0 * y1 + y2
            if abs(denom) > EPS:
                lag = lag + 0.5 * (y0 - y2) / denom
        if lag > 0:
            f0[i] = sample_rate / lag

    out_of_range = ~np.isnan(f0) & ((f0 < fmin) | (f0 > fmax))
    f0[out_of_range] = np.nan
    return f0, grid


def amplitude_envelope(
    samples: np.ndarray,
    sample_rate: int,
    *,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
    frame_seconds: float = 0.025,
) -> tuple[np.ndarray, FrameGrid]:
    """Smoothed RMS amplitude envelope, used for syllable-nucleus counting."""
    frames, grid = frame_signal(
        samples, sample_rate, frame_seconds=frame_seconds, hop_seconds=hop_seconds
    )
    rms = np.sqrt(np.mean(np.square(frames.astype(np.float64)), axis=1) + EPS)
    if len(rms) >= 5:
        kernel = np.hanning(5)
        kernel /= kernel.sum()
        rms = np.convolve(rms, kernel, mode="same")
    return rms, grid
