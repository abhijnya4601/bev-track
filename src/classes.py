"""The 5-class taxonomy used throughout BEV-Track, and how nuScenes labels map onto it.

nuScenes has three label vocabularies we have to deal with:
  * fine-grained categories in the raw annotations (``vehicle.car``, ``human.pedestrian.adult``, ...)
  * the 10 "detection names" used by the official detection benchmark and by every public
    BEVFormer checkpoint (``car``, ``truck``, ``construction_vehicle``, ...)
  * our 5 working classes.
"""
from typing import Optional

CLASSES = ("car", "pedestrian", "cyclist", "truck", "barrier")

# Raw nuScenes category -> working class. Prefix match, so all human.pedestrian.* subtypes map.
# bicycle/motorcycle are merged into "cyclist"; truck/construction into "truck".
# Buses, trailers, cones, animals, etc. are deliberately dropped.
_CATEGORY_PREFIXES = (
    ("vehicle.car", "car"),
    ("human.pedestrian.", "pedestrian"),
    ("vehicle.bicycle", "cyclist"),
    ("vehicle.motorcycle", "cyclist"),
    ("vehicle.truck", "truck"),
    ("vehicle.construction", "truck"),
    ("movable_object.barrier", "barrier"),
)

# The 10 nuScenes detection names (what pretrained BEVFormer predicts) -> working class.
DETECTION_NAME_TO_CLASS = {
    "car": "car",
    "pedestrian": "pedestrian",
    "bicycle": "cyclist",
    "motorcycle": "cyclist",
    "truck": "truck",
    "construction_vehicle": "truck",
    "barrier": "barrier",
    # bus, trailer, traffic_cone -> dropped
}

# Evaluation range per class in metres from ego (xy), taken from the devkit's
# detection_cvpr_2019 config for the source classes. Boxes at or beyond this distance are
# ignored in both GT and predictions, exactly as the devkit does.
CLASS_RANGE = {
    "car": 50.0,
    "pedestrian": 40.0,
    "cyclist": 40.0,
    "truck": 50.0,
    "barrier": 30.0,
}

# Barriers are symmetric, so orientation is only defined up to 180 degrees.
ORIENTATION_PERIOD = {c: (3.141592653589793 if c == "barrier" else 2 * 3.141592653589793) for c in CLASSES}

# TP error metrics that are undefined for a class (devkit: barriers have no velocity).
UNDEFINED_TP_METRICS = {"barrier": {"vel_err"}}


def category_to_class(category_name: str) -> Optional[str]:
    """Map a raw nuScenes category (e.g. ``human.pedestrian.adult``) to a working class, or None."""
    for prefix, cls in _CATEGORY_PREFIXES:
        if category_name == prefix or category_name.startswith(prefix):
            return cls
    return None


def detection_name_to_class(detection_name: str) -> Optional[str]:
    """Map a nuScenes detection name (the 10-class benchmark vocabulary) to a working class, or None."""
    if detection_name in CLASSES:
        return detection_name
    return DETECTION_NAME_TO_CLASS.get(detection_name)
