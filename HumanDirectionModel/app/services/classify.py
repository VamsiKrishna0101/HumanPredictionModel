import numpy as np
from app.core.config import settings

LABELS = ["Front", "Left", "Right", "Full Body", "N/A"]

IDX = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16
}

def _visible(kpt):
    return kpt[2] >= settings.KEYPOINT_CONF_THR

def _bbox_coverage(bbox, h):
    y1, y2 = bbox[1], bbox[3]
    return max(0.0, min(1.0, (y2 - y1) / max(1.0, h)))

def _count_visible(kpts):
    return int(np.sum(kpts[:, 2] >= settings.KEYPOINT_CONF_THR))

def _is_full_body(kpts, bbox, img_shape):
    h = img_shape[0]
    visible = _count_visible(kpts)
    feet_ok = _visible(kpts[IDX["left_ankle"]]) and _visible(kpts[IDX["right_ankle"]])
    knees_ok = _visible(kpts[IDX["left_knee"]]) and _visible(kpts[IDX["right_knee"]])
    hips_ok = _visible(kpts[IDX["left_hip"]]) and _visible(kpts[IDX["right_hip"]])

    coverage_ok = _bbox_coverage(bbox, h) >= settings.FULLBODY_MIN_BBOX_COVERAGE
    visible_ok = visible >= settings.FULLBODY_MIN_VISIBLE_KPTS

    if settings.FULLBODY_REQUIRE_FEET:
        limbs_ok = feet_ok and knees_ok and hips_ok
    else:
        limbs_ok = knees_ok and hips_ok

    return coverage_ok and visible_ok and limbs_ok

def _yaw_left_right_front(kpts):
    def v(idx): return kpts[idx, 2] if idx is not None else 0.0
    def x(idx): return kpts[idx, 0] if idx is not None else None

    leye, reye = IDX["left_eye"], IDX["right_eye"]
    lear, rear = IDX["left_ear"], IDX["right_ear"]
    lsh, rsh = IDX["left_shoulder"], IDX["right_shoulder"]

    left_vis = v(leye) + v(lear)
    right_vis = v(reye) + v(rear)

    if left_vis + right_vis >= 0.5:
        if right_vis - left_vis > 0.25:
            return "Right"
        if left_vis - right_vis > 0.25:
            return "Left"

    lx, rx = x(lsh), x(rsh)
    if lx is not None and rx is not None:
        width = abs(rx - lx) + 1e-6
        nose_x = x(IDX["nose"])
        mid = (lx + rx) / 2 if nose_x is not None else None
        if mid is not None:
            delta = (nose_x - mid) / width
            if delta > 0.15:
                return "Right"
            if delta < -0.15:
                return "Left"

    return "Front"

def classify_orientation(kpts, bbox, img_shape):
    if kpts is None or bbox is None:
        return "N/A"

    if _is_full_body(kpts, bbox, img_shape):
        return "Full Body"

    return _yaw_left_right_front(kpts)
