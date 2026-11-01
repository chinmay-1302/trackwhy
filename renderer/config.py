"""Shared rendering parameters.

Clips keep every source frame but play at 10 fps against a 25 fps MOT20
source. Windows are 40 frames before and after the event, capped at 200
frames, which produces the released 81–201 frame (median 93) clips.
"""

SOURCE_FPS = 25
PLAYBACK_FPS = 10
PRE_FRAMES = 40
POST_FRAMES = 40
MAX_CLIP_FRAMES = 200
HIGHLIGHT_IOU = 0.30
HIGHLIGHT_COLOR = (0, 255, 255)
HIGHLIGHT_THICKNESS = 3

REGIMES = {
    "supervision_free": (
        "Highlight is pinned to one raw tracker identity at its first match "
        "and never re-anchors."
    ),
    "gt_informed": (
        "Highlight follows the correct physical person, re-anchoring to "
        "whichever tracker box the ground truth matches when the tracker "
        "loses or switches the target."
    ),
}
