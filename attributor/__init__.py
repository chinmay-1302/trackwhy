"""TrackWhy attributor: TrackEval-anchored event mining and rule assignment."""

from mot_error_attribution.attribution import attribute_event, attribute_events
from mot_error_attribution.config import PipelineConfig
from mot_error_attribution.pipeline import run_pipeline

__all__ = [
    "PipelineConfig",
    "attribute_event",
    "attribute_events",
    "run_pipeline",
]
