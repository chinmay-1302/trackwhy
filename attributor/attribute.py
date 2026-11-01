"""Rule-based condition assignment.

This is a thin wrapper around ``mot_error_attribution.attribution.attribute_event``.
The decision is a single priority function; the named rules in ``rules.py``
document what each condition tests.
"""

from __future__ import annotations

from mot_error_attribution.attribution import attribute_event, attribute_events
from mot_error_attribution.config import PipelineConfig
from mot_error_attribution.matching import MatchingResult
from mot_error_attribution.models import ErrorEvent, SequenceData

__all__ = ["attribute_event", "attribute_events"]


def attribute(
    event: ErrorEvent,
    sequence: SequenceData,
    matching: MatchingResult,
    config: PipelineConfig | None = None,
) -> ErrorEvent:
    """Assign a scene-condition label to one TrackEval-counted failure."""
    return attribute_event(event, sequence, matching, config or PipelineConfig())
