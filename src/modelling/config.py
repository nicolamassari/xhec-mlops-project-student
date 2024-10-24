TARGET = "Rings"

NUMERICAL_FEATURES = [
    "Length",
    "Diameter",
    "Height",
    "Whole weight",
    "Shucked weight",
    "Viscera weight",
    "Shell weight",
]

SEX_MAPPING = {"M": 1, "I": 0, "F": -1}

OUTLIER_CONDITIONS = [
    ("Height", ">", 0.4),
    ("Viscera weight", ">", 0.6),
    ("Rings", ">", 25),
]
