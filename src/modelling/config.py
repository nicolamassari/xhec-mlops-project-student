from pathlib import Path

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRPATH = str(PROJECT_ROOT / "data")
MODELS_DIRPATH = str(PROJECT_ROOT / "src/web_service/local_objects/models")
