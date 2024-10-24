NUMERICAL_FEATURES = [
    "length",
    "diameter",
    "height",
    "whole_weight",
    "shucked_weight",
    "viscera_weight",
    "shell_weight",
]

SEX_MAPPING = {"M": 1, "I": 0, "F": -1}

OUTLIER_CONDITIONS = [
    ("height", ">", 0.4),
    ("viscera weight", ">", 0.6),
    ("rings", ">", 25),
]

OUTLIER_CONDITIONS_WITHOUT_TARGET = [
    ("height", ">", 0.4),
    ("viscera_weight", ">", 0.6),
]
