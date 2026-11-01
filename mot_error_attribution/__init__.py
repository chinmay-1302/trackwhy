"""Rule-based, model-free MOT20 tracker error attribution."""

from .config import PipelineConfig
from .pipeline import run_pipeline

__all__ = ["PipelineConfig", "run_pipeline"]

