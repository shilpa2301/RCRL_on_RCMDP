import numpy as np

PITCH_THRESHOLD = np.pi / 2.0
PITCH_SCALE = np.pi / 2.0
JOINT_NAMES = [
    "back_thigh",
    "back_shin",
    "back_foot",
    "front_thigh",
    "front_shin",
    "front_foot",
]
JOINT_SPEED_THRESHOLDS = np.asarray(
    [
        23.952442696579617,
        21.273920780514413,
        20.869860859139493,
        27.172519265909983,
        21.158679170546527,
        16.234366386891228,
    ],
    dtype=np.float64,
)
JOINT_SPEED_SCALES = np.asarray(
    [
        23.952442696579617,
        21.273920780514413,
        20.869860859139493,
        27.172519265909983,
        21.158679170546527,
        16.234366386891228,
    ],
    dtype=np.float64,
)
PERSISTENT_EPS = 0.1
COMBINED_COST_VERSION = "combined_max_wrapped90_jointp99_v1"