"""callscope — fully local two-speaker call analysis and QA scoring.

No audio, transcript, or derived metric leaves the machine unless the operator
explicitly configures a remote LLM judge for the semantic track.
"""

from callscope.schema import (
    CallReport,
    CriterionResult,
    Interval,
    ParalinguisticProfile,
    SemanticResult,
    SpeakerTurn,
    Transcript,
    TranscriptSegment,
)

__version__ = "0.1.0"

__all__ = [
    "CallReport",
    "CriterionResult",
    "Interval",
    "ParalinguisticProfile",
    "SemanticResult",
    "SpeakerTurn",
    "Transcript",
    "TranscriptSegment",
    "__version__",
]
