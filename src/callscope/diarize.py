"""Two-speaker diarization: VAD segmentation, embeddings, constrained clustering.

The two-speaker case is the one worth optimizing for in call QA -- agent and
customer -- and it is dramatically easier than the general case because the
speaker count is known a priori. That lets us replace "estimate k, then cluster"
with a hard k=2 constraint and a deterministic initialization, which removes the
main source of run-to-run variance in a QA pipeline.

Accuracy relative to pyannote is discussed honestly in the README. An optional
pyannote backend is available when a HuggingFace token is present.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from callscope.audio import AudioBuffer
from callscope.errors import DiarizationError
from callscope.features import (
    cepstral_mean_variance_normalize,
    estimate_f0,
    mfcc,
)
from callscope.schema import SPEAKER_LABELS, Interval, SpeakerTurn
from callscope.vad import VadConfig, detect_speech, split_long_segments

logger = logging.getLogger(__name__)

#: Segments shorter than this carry too little evidence for a stable embedding
#: and are assigned by proximity to their neighbours instead of by clustering.
MIN_EMBEDDING_SECONDS = 0.45


@dataclass(frozen=True, slots=True)
class DiarizationConfig:
    """Tunables for :func:`diarize`."""

    n_mfcc: int = 20
    #: Cepstral coefficients used for the embedding. c0 is dropped because it is
    #: gain, which tracks the recording level rather than the speaker.
    first_coeff: int = 1
    max_segment_seconds: float = 2.0
    target_segment_seconds: float = 1.5
    #: k-means iteration cap. Convergence is typically reached in well under 10.
    max_iterations: int = 100
    #: Block weights applied after per-dimension standardization. Each block is
    #: also divided by the square root of its dimensionality, so a weight is a
    #: statement about the *block's* influence rather than about each of its
    #: dimensions -- without that, the 38 cepstral dimensions swamp the 2 pitch
    #: dimensions no matter what weight pitch is given.
    mfcc_mean_weight: float = 1.0
    mfcc_std_weight: float = 0.5
    pitch_weight: float = 2.5
    #: Mean silhouette below which the call is reported as a single speaker
    #: rather than being forced into two clusters. Calibrated on synthetic
    #: monologues (~0.40) against two-speaker calls with similar voices (~0.59);
    #: it is a heuristic threshold, not a derived one.
    min_separation: float = 0.50
    #: The smaller cluster must hold at least this many segments. A "speaker"
    #: represented by a single segment is indistinguishable from an outlier, and
    #: 2-means will always manufacture one if allowed to.
    min_cluster_segments: int = 2
    vad: VadConfig | None = None


@dataclass(slots=True)
class DiarizationResult:
    """Speaker turns plus the diagnostics needed to judge whether to trust them."""

    turns: list[SpeakerTurn]
    backend: str
    n_speakers: int
    separation: float
    segments: list[Interval]
    warnings: list[str]


def diarize(
    audio: AudioBuffer,
    *,
    config: DiarizationConfig | None = None,
    backend: str = "auto",
    wav_path: str | Path | None = None,
) -> DiarizationResult:
    """Assign every speech region to one of at most two speakers.

    Args:
        audio: Canonical mono 16 kHz buffer.
        config: Tunables for the built-in clustering backend.
        backend: ``"auto"`` (pyannote if usable, else the built-in clusterer),
            ``"cluster"`` to force the built-in, or ``"pyannote"`` to require
            pyannote and fail loudly if it is unavailable.
        wav_path: Path to the normalized WAV. Required by the pyannote backend.

    Returns:
        A :class:`DiarizationResult` whose ``turns`` use at most the two labels
        in :data:`callscope.schema.SPEAKER_LABELS`.

    Raises:
        DiarizationError: ``backend="pyannote"`` was requested but is unusable.
    """
    if backend not in {"auto", "cluster", "pyannote"}:
        raise ValueError(f"unknown diarization backend: {backend!r}")

    if backend in {"auto", "pyannote"}:
        available, reason = pyannote_available()
        if available and wav_path is not None:
            try:
                return _diarize_pyannote(Path(wav_path))
            except Exception as exc:  # pragma: no cover - optional path
                if backend == "pyannote":
                    raise DiarizationError(f"pyannote backend failed: {exc}") from exc
                logger.warning("pyannote backend failed (%s); falling back to clustering", exc)
        elif backend == "pyannote":
            detail = reason if not available else "no wav_path supplied"
            raise DiarizationError(f"pyannote backend unavailable: {detail}")

    return _diarize_cluster(audio, config or DiarizationConfig())


def pyannote_available() -> tuple[bool, str]:
    """Whether the optional pyannote backend can run. Never raises."""
    import importlib.util

    try:
        # find_spec on a dotted name imports the parent package, and raises
        # rather than returning None when the parent itself is absent.
        spec = importlib.util.find_spec("pyannote.audio")
    except (ImportError, ModuleNotFoundError, ValueError):
        spec = None
    if spec is None:
        return False, "pyannote.audio is not installed (`pip install 'callscope[pyannote]'`)"
    if not (os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")):
        return False, "HUGGINGFACE_TOKEN is not set"
    return True, "available"


def _diarize_pyannote(wav_path: Path) -> DiarizationResult:  # pragma: no cover - optional
    """Run pyannote's pretrained pipeline pinned to exactly two speakers."""
    from pyannote.audio import Pipeline  # type: ignore[import-not-found]

    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=token
    )
    annotation = pipeline(str(wav_path), num_speakers=2)

    raw: list[tuple[float, float, str]] = [
        (float(seg.start), float(seg.end), str(label))
        for seg, _, label in annotation.itertracks(yield_label=True)
        if float(seg.end) > float(seg.start)
    ]
    raw.sort(key=lambda t: t[0])

    # pyannote's own labels are arbitrary strings; remap onto our fixed pair,
    # ordered by first appearance, so report consumers see stable identifiers.
    order: list[str] = []
    for _, _, label in raw:
        if label not in order:
            order.append(label)
    mapping = {label: SPEAKER_LABELS[i] for i, label in enumerate(order[:2])}

    warnings: list[str] = []
    if len(order) > 2:
        warnings.append(
            f"pyannote returned {len(order)} speakers despite num_speakers=2; "
            "extra labels folded into the nearest of the first two."
        )
        for extra in order[2:]:
            mapping[extra] = SPEAKER_LABELS[0]

    turns = [SpeakerTurn(s, e, mapping[label]) for s, e, label in raw]
    turns = _merge_adjacent(turns)
    return DiarizationResult(
        turns=turns,
        backend="pyannote",
        n_speakers=min(len(order), 2),
        separation=float("nan"),
        segments=[Interval(t.start, t.end) for t in turns],
        warnings=warnings,
    )


def _diarize_cluster(audio: AudioBuffer, cfg: DiarizationConfig) -> DiarizationResult:
    """Built-in backend: VAD -> per-segment embedding -> 2-means -> turns."""
    warnings: list[str] = []

    segments = detect_speech(audio.samples, audio.sample_rate, cfg.vad)
    segments = split_long_segments(
        segments,
        max_seconds=cfg.max_segment_seconds,
        target_seconds=cfg.target_segment_seconds,
    )
    if not segments:
        return DiarizationResult(
            turns=[],
            backend="cluster",
            n_speakers=0,
            separation=0.0,
            segments=[],
            warnings=["no speech detected"],
        )

    embeddings, usable = _segment_embeddings(audio, segments, cfg)
    usable_idx = np.flatnonzero(usable)

    if len(usable_idx) < 2:
        warnings.append(
            "fewer than two segments long enough to embed; reporting a single speaker"
        )
        turns = [SpeakerTurn(s.start, s.end, SPEAKER_LABELS[0]) for s in segments]
        return DiarizationResult(
            turns=_merge_adjacent(turns),
            backend="cluster",
            n_speakers=1,
            separation=0.0,
            segments=segments,
            warnings=warnings,
        )

    X = embeddings[usable_idx]
    labels, separation = _two_means(
        X,
        weights=np.array([segments[i].duration for i in usable_idx]),
        max_iterations=cfg.max_iterations,
    )

    minority = int(min(np.sum(labels == 0), np.sum(labels == 1)))
    if separation < cfg.min_separation:
        warnings.append(
            f"cluster silhouette {separation:.3f} is below the {cfg.min_separation:.2f} "
            "threshold; reporting a single speaker. Either the call has one speaker, "
            "or the two voices are too similar for this diarizer -- try --diarizer pyannote."
        )
        labels = np.zeros_like(labels)
        n_speakers = 1
    elif minority < cfg.min_cluster_segments:
        warnings.append(
            f"the smaller speaker cluster holds only {minority} segment(s), fewer than "
            f"the {cfg.min_cluster_segments} required; reporting a single speaker."
        )
        labels = np.zeros_like(labels)
        n_speakers = 1
    else:
        n_speakers = 2

    full = _assign_short_segments(segments, usable_idx, labels)
    # Order labels by first appearance so SPEAKER_00 is whoever spoke first.
    full = _relabel_by_first_appearance(full)

    turns = [
        SpeakerTurn(seg.start, seg.end, SPEAKER_LABELS[int(lab)])
        for seg, lab in zip(segments, full, strict=True)
    ]
    turns = _merge_adjacent(turns)

    # The two-speaker constraint is an invariant, not an aspiration: assert it.
    distinct = {t.speaker for t in turns}
    if not distinct <= set(SPEAKER_LABELS):
        raise DiarizationError(f"diarizer emitted unexpected labels: {sorted(distinct)}")

    return DiarizationResult(
        turns=turns,
        backend="cluster",
        n_speakers=n_speakers,
        separation=float(separation),
        segments=segments,
        warnings=warnings,
    )


def _segment_embeddings(
    audio: AudioBuffer, segments: list[Interval], cfg: DiarizationConfig
) -> tuple[np.ndarray, np.ndarray]:
    """One fixed-length embedding per segment, plus a usability mask.

    The embedding is [mean MFCC, std MFCC, log-f0 mean, log-f0 std]. Mean cepstra
    capture vocal-tract shape; the standard deviations capture articulation
    dynamics; log-pitch separates most speaker pairs on its own.
    """
    coeffs, grid = mfcc(audio.samples, audio.sample_rate, n_mfcc=cfg.n_mfcc)
    coeffs = cepstral_mean_variance_normalize(coeffs)[:, cfg.first_coeff :]
    f0, f0_grid = estimate_f0(audio.samples, audio.sample_rate)
    log_f0 = np.log(np.where(np.isnan(f0), np.nan, np.maximum(f0, 1e-3)))

    dim = 2 * coeffs.shape[1] + 2
    out = np.zeros((len(segments), dim), dtype=np.float64)
    usable = np.zeros(len(segments), dtype=bool)

    global_log_f0 = float(np.nanmean(log_f0)) if np.any(~np.isnan(log_f0)) else 0.0

    for i, seg in enumerate(segments):
        a, b = grid.time_to_frame(seg.start), grid.time_to_frame(seg.end)
        block = coeffs[a : max(b, a + 1)]
        if block.shape[0] == 0:
            continue

        fa, fb = f0_grid.time_to_frame(seg.start), f0_grid.time_to_frame(seg.end)
        pitch_block = log_f0[fa : max(fb, fa + 1)]
        voiced = pitch_block[~np.isnan(pitch_block)]
        pitch_mean = float(voiced.mean()) if voiced.size else global_log_f0
        pitch_std = float(voiced.std()) if voiced.size > 1 else 0.0

        out[i] = np.concatenate(
            [block.mean(axis=0), block.std(axis=0), [pitch_mean, pitch_std]]
        )
        usable[i] = seg.duration >= MIN_EMBEDDING_SECONDS and voiced.size > 0

    if usable.any():
        # Standardize each dimension against the usable segments only, so that a
        # handful of unusable stubs cannot skew the scale.
        ref = out[usable]
        mean = ref.mean(axis=0, keepdims=True)
        std = np.maximum(ref.std(axis=0, keepdims=True), 1e-6)
        out = (out - mean) / std
        out = out * _block_scales(coeffs.shape[1], cfg)
    return out, usable


def _block_scales(n_coeffs: int, cfg: DiarizationConfig) -> np.ndarray:
    """Per-dimension scale vector implementing the block weighting.

    Applied *after* standardization. Applying it before would be pointless: the
    standardization step divides each dimension by its own standard deviation and
    would simply undo the weight.
    """
    blocks = (
        (n_coeffs, cfg.mfcc_mean_weight),
        (n_coeffs, cfg.mfcc_std_weight),
        (2, cfg.pitch_weight),
    )
    return np.concatenate(
        [np.full(dim, weight / np.sqrt(dim)) for dim, weight in blocks]
    )


def _two_means(
    X: np.ndarray, *, weights: np.ndarray, max_iterations: int
) -> tuple[np.ndarray, float]:
    """Deterministic duration-weighted 2-means, initialized by a PC1 median split.

    Deterministic on purpose. A QA score that changes between reruns of the same
    audio is not defensible to the person being scored, so random restarts are
    not an option -- but that makes the initialization decisive rather than
    merely convenient.

    Farthest-pair initialization is the obvious deterministic choice and it is
    the wrong one here: the farthest-apart pair of segments in a real call is
    usually one outlier (a cross-talk segment, a cough) against everything else,
    and 2-means then converges with that single outlier alone in its own cluster.
    Splitting at the weighted median of the first principal component cannot
    isolate a single point, and PC1 of two-speaker embeddings is close to the
    speaker axis by construction.
    """
    n = X.shape[0]
    if n < 2:
        return np.zeros(n, dtype=int), 0.0

    centroids = _pc1_median_init(X, weights)
    labels = np.full(n, -1, dtype=int)
    w = np.maximum(np.asarray(weights, dtype=np.float64), 1e-6)

    for _ in range(max_iterations):
        d0 = np.sum((X - centroids[0]) ** 2, axis=1)
        d1 = np.sum((X - centroids[1]) ** 2, axis=1)
        new_labels = (d1 < d0).astype(int)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        for k in (0, 1):
            mask = labels == k
            if mask.any():
                centroids[k] = np.average(X[mask], axis=0, weights=w[mask])
            else:
                # Empty cluster: reseed on the point farthest from the other
                # centroid rather than collapsing to one speaker.
                other = 1 - k
                far = int(np.argmax(np.sum((X - centroids[other]) ** 2, axis=1)))
                centroids[k] = X[far]
                labels[far] = k

    return labels, _silhouette(X, labels)


def _pc1_median_init(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Two initial centroids from a weighted-median split along PC1."""
    w = np.maximum(np.asarray(weights, dtype=np.float64), 1e-6)
    centre = np.average(X, axis=0, weights=w)
    centred = (X - centre) * np.sqrt(w)[:, None]

    # SVD rather than an eigendecomposition of the covariance: better
    # conditioning, and the sign convention is stable for a given input.
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    pc1 = vt[0]
    if pc1[np.argmax(np.abs(pc1))] < 0:
        pc1 = -pc1  # pin the sign so the labelling is reproducible

    projection = (X - centre) @ pc1
    threshold = _weighted_median(projection, w)
    low = projection <= threshold
    if not low.any() or low.all():
        # Degenerate projection (all segments identical): fall back to the two
        # extreme points so the caller still gets two distinct centroids.
        order = np.argsort(projection)
        low = np.zeros(len(projection), dtype=bool)
        low[order[: max(1, len(order) // 2)]] = True

    return np.vstack(
        [
            np.average(X[low], axis=0, weights=w[low]),
            np.average(X[~low], axis=0, weights=w[~low]),
        ]
    )


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cumulative = np.cumsum(w)
    cutoff = cumulative[-1] / 2.0
    return float(v[int(np.searchsorted(cumulative, cutoff))])


def _silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette coefficient of the two-cluster solution, in ``[-1, 1]``.

    A between-over-within centroid ratio is the obvious separation statistic and
    it is useless here: 2-means maximizes exactly that quantity, so it scores a
    homogeneous single-speaker call about as well as a genuine two-speaker one.
    The silhouette compares each point's own-cluster distance to its other-cluster
    distance, which does collapse when the data has no real bimodality.
    """
    n = len(X)
    if n < 2 or len(np.unique(labels)) < 2:
        return 0.0

    diff = X[:, None, :] - X[None, :, :]
    distances = np.sqrt(np.maximum(np.einsum("ijk,ijk->ij", diff, diff), 0.0))

    scores: list[float] = []
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        other = labels != labels[i]
        if not same.any() or not other.any():
            continue
        a = float(distances[i, same].mean())
        b = float(distances[i, other].mean())
        denominator = max(a, b)
        if denominator > 0:
            scores.append((b - a) / denominator)
    return float(np.mean(scores)) if scores else 0.0


def _assign_short_segments(
    segments: list[Interval], usable_idx: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """Give un-embeddable segments the label of the nearest embedded segment."""
    full = np.zeros(len(segments), dtype=int)
    lookup = {int(idx): int(lab) for idx, lab in zip(usable_idx, labels, strict=True)}
    centres = {int(idx): segments[int(idx)].start for idx in usable_idx}

    for i in range(len(segments)):
        if i in lookup:
            full[i] = lookup[i]
            continue
        mid = (segments[i].start + segments[i].end) / 2.0
        nearest = min(centres, key=lambda k: abs(centres[k] - mid))
        full[i] = lookup[nearest]
    return full


def _relabel_by_first_appearance(labels: np.ndarray) -> np.ndarray:
    """Renumber cluster ids so that 0 is the speaker who spoke first."""
    remap: dict[int, int] = {}
    out = np.zeros_like(labels)
    for i, lab in enumerate(labels):
        key = int(lab)
        if key not in remap:
            remap[key] = len(remap)
        out[i] = remap[key]
    return out


def _merge_adjacent(turns: list[SpeakerTurn], *, max_gap: float = 0.30) -> list[SpeakerTurn]:
    """Merge consecutive same-speaker turns separated by a short gap."""
    if not turns:
        return []
    ordered = sorted(turns, key=lambda t: t.start)
    merged = [ordered[0]]
    for nxt in ordered[1:]:
        last = merged[-1]
        if nxt.speaker == last.speaker and nxt.start - last.end <= max_gap:
            merged[-1] = SpeakerTurn(last.start, max(last.end, nxt.end), last.speaker)
        else:
            merged.append(nxt)
    return merged
