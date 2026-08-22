"""Command-line entry point: ``callscope``.

Subcommands:

* ``analyze``  -- score one or more recordings against a rubric
* ``doctor``   -- report which optional backends this machine can actually use
* ``demo``     -- synthesize a two-speaker call and score it, no input needed
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from importlib import resources
from pathlib import Path

from callscope import __version__
from callscope.audio import ffmpeg_available, write_wav
from callscope.diarize import DiarizationConfig, pyannote_available
from callscope.errors import CallscopeError, UsageError
from callscope.fixtures import (
    default_two_speaker_script,
    script_to_transcript_dict,
    synthesize_call,
)
from callscope.pipeline import PipelineConfig, analyze_call
from callscope.report import REPORT_FORMATS, render_text, write_reports
from callscope.rubric import load_rubric
from callscope.transcribe import (
    BACKEND_PREFERENCE,
    TranscriptionConfig,
    available_backends,
    select_backend,
)

#: Bundled rubrics ship as package data so the default works from an installed
#: wheel, not just from a source checkout.
BUNDLED_RUBRICS = resources.files("callscope") / "rubrics"
DEFAULT_RUBRIC = Path(str(BUNDLED_RUBRICS / "support_call.yaml"))


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        match args.command:
            case "analyze":
                return _cmd_analyze(args)
            case "doctor":
                return _cmd_doctor(args)
            case "demo":
                return _cmd_demo(args)
        parser.print_help()
        return 2
    except UsageError as exc:
        print(f"callscope: {exc}", file=sys.stderr)
        return 2
    except CallscopeError as exc:
        print(f"callscope: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"callscope: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        # Unreadable rubric, unwritable report directory: a real condition with
        # a real message, not something a user should meet as a traceback.
        print(f"callscope: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="callscope",
        description="Local two-speaker call analysis and QA scoring. No audio leaves this machine.",
    )
    parser.add_argument("--version", action="version", version=f"callscope {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="score recordings against a rubric")
    analyze.add_argument("inputs", nargs="+", type=Path, help="audio files")
    analyze.add_argument("-r", "--rubric", type=Path, default=DEFAULT_RUBRIC,
                         help=f"rubric YAML/JSON (default: {DEFAULT_RUBRIC.name})")
    analyze.add_argument("-o", "--out", type=Path, default=Path("reports"),
                         help="output directory (default: ./reports)")
    analyze.add_argument("--formats", default="json,txt,html",
                         help=f"comma-separated subset of: {', '.join(REPORT_FORMATS)}")
    analyze.add_argument("--whisper", default="auto",
                         choices=("auto", *BACKEND_PREFERENCE),
                         help="transcription backend (default: auto)")
    analyze.add_argument("--model", default=None, help="whisper model id or path")
    analyze.add_argument("--language", default=None, help="force a language code, e.g. en")
    analyze.add_argument("--transcript", type=Path, default=None,
                         help="use this JSON transcript instead of running a model")
    analyze.add_argument("--diarizer", default="auto",
                         choices=("auto", "cluster", "pyannote"),
                         help="diarization backend (default: auto)")
    analyze.add_argument("--skip-normalization", action="store_true",
                         help="input is already 16 kHz mono 16-bit PCM WAV")
    analyze.add_argument("--quiet", action="store_true",
                         help="do not print the text summary to stdout")

    doctor = sub.add_parser("doctor", help="report available backends")
    doctor.add_argument("--json", action="store_true", help="machine-readable output")

    demo = sub.add_parser("demo", help="synthesize a call and score it end to end")
    demo.add_argument("-o", "--out", type=Path, default=Path("reports"),
                      help="output directory (default: ./reports)")
    demo.add_argument("-r", "--rubric", type=Path, default=DEFAULT_RUBRIC)
    demo.add_argument("--keep-audio", type=Path, default=None,
                      help="also write the synthesized WAV here")
    return parser


def _cmd_analyze(args: argparse.Namespace) -> int:
    formats = _parse_formats(args.formats)
    if args.transcript and args.whisper != "auto":
        raise UsageError(
            f"--transcript and --whisper {args.whisper} conflict; a supplied "
            "transcript is used instead of running a model. Drop one of the two."
        )

    rubric = load_rubric(args.rubric)
    config = PipelineConfig(
        rubric=rubric,
        transcription=TranscriptionConfig(
            backend="fixture" if args.transcript else args.whisper,
            model=args.model,
            language=args.language,
            fixture_path=args.transcript,
        ),
        diarization=DiarizationConfig(),
        diarizer_backend=args.diarizer,
        skip_normalization=args.skip_normalization,
    )

    exit_code = 0
    for path in args.inputs:
        try:
            report = analyze_call(path, config)
        except (CallscopeError, FileNotFoundError) as exc:
            print(f"callscope: {path}: {exc}", file=sys.stderr)
            exit_code = 1
            continue

        try:
            written = write_reports(report, args.out, formats=formats)
        except OSError as exc:
            # Read-only output directory, full disk, a path that is really a
            # file. Report it against the input and keep going: the other
            # recordings in the batch may write fine.
            print(f"callscope: {path}: could not write reports: {exc}", file=sys.stderr)
            exit_code = 1
            continue
        if not args.quiet:
            print(render_text(report))
        for fmt, dest in written.items():
            print(f"wrote {fmt}: {dest}", file=sys.stderr)
    return exit_code


def _parse_formats(raw: str) -> tuple[str, ...]:
    """Validate ``--formats`` before any audio is decoded.

    Raises:
        UsageError: an unknown or empty format list. Catching this at parse time
            means a typo costs a one-line message, not a stack trace after a
            forty-minute transcription.
    """
    formats = tuple(f.strip().lower() for f in raw.split(",") if f.strip())
    if not formats:
        raise UsageError(f"--formats is empty; choose from {', '.join(REPORT_FORMATS)}")
    unknown = [f for f in formats if f not in REPORT_FORMATS]
    if unknown:
        raise UsageError(
            f"unknown report format(s) {', '.join(unknown)}; "
            f"choose from {', '.join(REPORT_FORMATS)}"
        )
    return formats


def _cmd_doctor(args: argparse.Namespace) -> int:
    pyannote_ok, pyannote_reason = pyannote_available()
    info = {
        "version": __version__,
        "ffmpeg": ffmpeg_available(),
        "transcription_backends_available": available_backends(),
        "transcription_backend_selected": select_backend("auto"),
        "transcription_preference_order": list(BACKEND_PREFERENCE),
        "diarization_default": "cluster",
        "pyannote_available": pyannote_ok,
        "pyannote_detail": pyannote_reason,
    }
    if args.json:
        print(json.dumps(info, indent=2))
        return 0

    print(f"callscope {info['version']}")
    print(f"  ffmpeg                 {'found' if info['ffmpeg'] else 'MISSING (required)'}")
    print(f"  whisper preference     {' > '.join(info['transcription_preference_order'])}")
    print(f"  whisper available      {', '.join(info['transcription_backends_available'])}")
    print(f"  whisper selected       {info['transcription_backend_selected']}")
    print(f"  diarization default    {info['diarization_default']}")
    print(f"  pyannote               {info['pyannote_detail']}")
    if not info["ffmpeg"]:
        print("\nffmpeg is required. macOS: `brew install ffmpeg`.", file=sys.stderr)
        return 1
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    """Synthesize a scripted two-speaker call and run the whole pipeline on it.

    Uses the scripted transcript rather than a speech model, because the
    synthetic audio is voice-like but wordless. Everything else -- ffmpeg
    normalization, VAD, diarization, both scoring tracks -- runs for real.
    """
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    script = default_two_speaker_script()
    audio = synthesize_call(script)
    wav_path = Path(args.keep_audio) if args.keep_audio else out_dir / "demo_call.wav"
    write_wav(wav_path, audio)

    transcript_path = out_dir / "demo_call.transcript.json"
    transcript_path.write_text(
        json.dumps(script_to_transcript_dict(script), indent=2), encoding="utf-8"
    )

    config = PipelineConfig(
        rubric=load_rubric(args.rubric),
        transcription=TranscriptionConfig(backend="fixture", fixture_path=transcript_path),
        skip_normalization=False,
        call_id="demo_call",
    )
    report = analyze_call(wav_path, config)
    written = write_reports(report, out_dir)
    print(render_text(report))
    for fmt, dest in written.items():
        print(f"wrote {fmt}: {dest}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
