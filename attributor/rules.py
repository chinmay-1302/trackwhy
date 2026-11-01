"""Named rules for each released condition.

Each function documents what the rule tests and what it ignores. Assignment
itself is performed by ``attribute_event``; these wrappers exist so a reader
can find one entry point per cause without reimplementing the priority order.
"""

from __future__ import annotations

from mot_error_attribution.attribution import attribute_event
from mot_error_attribution.config import PipelineConfig
from mot_error_attribution.matching import MatchingResult
from mot_error_attribution.models import ErrorEvent, SequenceData

CAUSES = (
    "full_occl_static",
    "full_occl_pedestrian",
    "full_occl_crowd_aggregate",
    "partial_occl_pedestrian",
    "partial_occl_static",
    "track_drift_to_static",
    "reentry",
    "gt_annotation_gap",
    "visible_miss",
    "visible_miss_unconfirmed",
    "visible_miss_truncated",
    "excluded_micro_gap",
    "unattributed_fn",
    "unattributed_idsw",
)


def _assigned(
    event: ErrorEvent,
    sequence: SequenceData,
    matching: MatchingResult,
    config: PipelineConfig | None = None,
) -> str | None:
    return attribute_event(
        event, sequence, matching, config or PipelineConfig()
    ).cause


def full_occl_static(event, sequence, matching, config=None) -> bool:
    """FULL visibility band and a static occluder wins the IoA vote.

    Reads ground-truth geometry, the visibility field, and class labels.
    Does not read detector confidence or appearance.
    """
    return _assigned(event, sequence, matching, config) == "full_occl_static"


def full_occl_pedestrian(event, sequence, matching, config=None) -> bool:
    """FULL visibility band and a pedestrian occluder wins the IoA vote."""
    return _assigned(event, sequence, matching, config) == "full_occl_pedestrian"


def full_occl_crowd_aggregate(event, sequence, matching, config=None) -> bool:
    """FULL visibility band explained by aggregated crowd coverage, not one person."""
    return _assigned(event, sequence, matching, config) == "full_occl_crowd_aggregate"


def partial_occl_pedestrian(event, sequence, matching, config=None) -> bool:
    """PARTIAL visibility band; the highest-IoA interactor is a pedestrian."""
    return _assigned(event, sequence, matching, config) == "partial_occl_pedestrian"


def partial_occl_static(event, sequence, matching, config=None) -> bool:
    """PARTIAL visibility band; the highest-IoA interactor is static structure."""
    return _assigned(event, sequence, matching, config) == "partial_occl_static"


def track_drift_to_static(event, sequence, matching, config=None) -> bool:
    """The one exception to pure ground-truth geometry.

    Fires on ``FP_ID_HIJACK`` events where the tracker box sits on a
    static-person or static-object identity. This uses the tracker box, not
    only the ground-truth box.
    """
    return _assigned(event, sequence, matching, config) == "track_drift_to_static"


def reentry(event, sequence, matching, config=None) -> bool:
    """Identity switch after a long absence that is consistent with leaving and returning."""
    return _assigned(event, sequence, matching, config) == "reentry"


def gt_annotation_gap(event, sequence, matching, config=None) -> bool:
    """Reentry-shaped gap that is better explained as a hole in MOT20 annotation."""
    return _assigned(event, sequence, matching, config) == "gt_annotation_gap"


def visible_miss(event, sequence, matching, config=None) -> bool:
    """CLEAR visibility, no interacting occluder, and a counted false negative."""
    return _assigned(event, sequence, matching, config) == "visible_miss"


def visible_miss_unconfirmed(event, sequence, matching, config=None) -> bool:
    """A later occluder-mask postpass: the miss sits on unannotated structure.

    Assigned by ``occluder_masks.apply_visible_miss_postpass``, not by the
    core decision function.
    """
    return _assigned(event, sequence, matching, config) == "visible_miss_unconfirmed"


def visible_miss_truncated(event, sequence, matching, config=None) -> bool:
    """A later postpass: the miss is at the frame boundary."""
    return _assigned(event, sequence, matching, config) == "visible_miss_truncated"


def excluded_micro_gap(event, sequence, matching, config=None) -> bool:
    """One- or two-frame FN/FRAG runs moved out of the scored pool."""
    return False


def unattributed_fn(event, sequence, matching, config=None) -> bool:
    """Residual false-negative event; no rule fired."""
    return _assigned(event, sequence, matching, config) == "unattributed_fn"


def unattributed_idsw(event, sequence, matching, config=None) -> bool:
    """Residual identity-switch event; no rule fired."""
    return _assigned(event, sequence, matching, config) == "unattributed_idsw"


RULES = {
    "full_occl_static": full_occl_static,
    "full_occl_pedestrian": full_occl_pedestrian,
    "full_occl_crowd_aggregate": full_occl_crowd_aggregate,
    "partial_occl_pedestrian": partial_occl_pedestrian,
    "partial_occl_static": partial_occl_static,
    "track_drift_to_static": track_drift_to_static,
    "reentry": reentry,
    "gt_annotation_gap": gt_annotation_gap,
    "visible_miss": visible_miss,
    "visible_miss_unconfirmed": visible_miss_unconfirmed,
    "visible_miss_truncated": visible_miss_truncated,
    "excluded_micro_gap": excluded_micro_gap,
    "unattributed_fn": unattributed_fn,
    "unattributed_idsw": unattributed_idsw,
}
