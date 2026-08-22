"""Shared fixtures. Everything here is synthesized: no network, no binaries."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from callscope.audio import write_wav
from callscope.fixtures import (
    CallScript,
    Utterance,
    default_two_speaker_script,
    script_to_transcript_dict,
    synthesize_call,
)
from callscope.rubric import load_rubric

RUBRIC_DIR = Path(__file__).resolve().parents[1] / "src" / "callscope" / "rubrics"


def pytest_collection_modifyitems(config, items):
    """Skip ffmpeg-dependent tests when ffmpeg is not installed."""
    if shutil.which("ffmpeg"):
        return
    skip = pytest.mark.skip(reason="ffmpeg not on PATH")
    for item in items:
        if "requires_ffmpeg" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def script() -> CallScript:
    return default_two_speaker_script()


@pytest.fixture(scope="session")
def call_audio(script: CallScript):
    """The synthetic two-speaker call, rendered once per session."""
    return synthesize_call(script)


@pytest.fixture(scope="session")
def call_wav(tmp_path_factory, call_audio) -> Path:
    path = tmp_path_factory.mktemp("audio") / "call.wav"
    write_wav(path, call_audio)
    return path


@pytest.fixture(scope="session")
def transcript_json(tmp_path_factory, script: CallScript) -> Path:
    path = tmp_path_factory.mktemp("transcript") / "call.json"
    path.write_text(json.dumps(script_to_transcript_dict(script)), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def support_rubric():
    return load_rubric(RUBRIC_DIR / "support_call.yaml")


@pytest.fixture
def single_speaker_audio():
    """A monologue: used to check the diarizer does not invent a second speaker."""
    script = CallScript(
        duration=8.0,
        utterances=(
            Utterance(0.5, 3.0, "SPEAKER_00", f0_hz=120.0),
            Utterance(3.8, 7.5, "SPEAKER_00", f0_hz=120.0),
        ),
    )
    return synthesize_call(script), script
