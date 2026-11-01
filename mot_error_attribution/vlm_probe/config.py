from __future__ import annotations

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
SECOND_MODEL = "gemini-2.5-flash"
FRAMES = 16
CENTRED_FRAMES = 24
MIN_FRAMES_IN = 8
MAX_PIXELS = 768 * 28 * 28
TEMPERATURE = 0.0
MAX_TOKENS = 512
SEED = 2026
CLIP_FPS = 10.0

V2_N = 10
V3_N_EVENTS = 50
V3_BALANCED = True
V4_N = 20
STAYS_ON_PERSON_DROP = 0.10
MAX_PIXELS_HIGH = 1280 * 28 * 28
MIN_HIGHLIGHT_PX = 20

DEFAULT_CLIP_MAP = "out/labelstudio_probe_clips/clip_map.csv"
DEFAULT_CLIPS_DIR = "out/probe_clips/clips"
DEFAULT_CLIPS_META_DIR = "out/probe_clips/clips_meta"
DEFAULT_HASHED_DIR = "out/labelstudio_probe_clips/videos"
DEFAULT_OUTPUT_ROOT = "out/vlm_probe"

SAMPLING_UNIFORM = "uniform"
SAMPLING_CENTRED = "centred"
SAMPLING_FROZEN = "frozen"

SMOKE_CAUSES = (
    "full_occl_pedestrian",
    "visible_miss",
    "visible_miss_truncated",
    "unattributed_idsw",
    "track_drift_to_static",
)
SMOKE_N = 10

HIDDEN_BY_CAUSES = frozenset(
    {
        "hidden_by_object",
        "hidden_by_one_person",
        "hidden_by_standing_person",
        "hidden_by_crowd",
    }
)
PARTLY_BLOCKED_CAUSES = frozenset(
    {
        "partly_blocked_by_person",
        "partly_blocked_by_static",
    }
)
VISIBLE_MISS_TRUTHS = frozenset({"visible_miss", "visible_miss_truncated"})

HUMAN_COULD_NOT_IDENTIFY = "could_not_identify_target"

RULE_CAUSE_TO_VLM = {
    "full_occl_static": "hidden_by_object",
    "full_occl_pedestrian": "hidden_by_one_person",
    "full_occl_crowd_aggregate": "hidden_by_crowd",
    "partial_occl_pedestrian": "partly_blocked_by_person",
    "partial_occl_static": "partly_blocked_by_static",
    "visible_miss": "clearly_visible_but_missed",
    "visible_miss_truncated": "partly_outside_the_frame",
    "visible_miss_unconfirmed": "clearly_visible_but_missed",
    "track_drift_to_static": "drifted_to_standing_person",
    "unattributed_fn": "cannot_determine",
    "unattributed_idsw": "cannot_determine",
}
