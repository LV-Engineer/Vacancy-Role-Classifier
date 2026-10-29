"""Constants shared by the data package."""
import re
from pathlib import Path

ROLES = ["Data Analyst", "Data Scientist", "ML / AI Engineer", "Python Backend", "Other"]
LEVELS = ["junior", "not_junior"]
COLUMNS = ["id", "title", "description", "language", "role", "level", "source"]
MIN_DESCRIPTION_CHARS = 200

ROLE_PATTERNS = [
    ("ML / AI Engineer", r"machine learning|\bml\b|ai engineer|mlops|deep learning|\bllm\b"),
    ("Data Scientist", r"data scientist"),
    ("Data Analyst", r"data analyst|business analyst|bi analyst|bi developer"),
    ("Python Backend", r"python|backend|back-end|django|fastapi|flask"),
]
CYRILLIC = re.compile(r"[а-яіїєґ]", re.IGNORECASE)
JUNIOR_LEVELS = {"Entry level", "Internship"}

SEED = 42
TEST_FOLDS = 4

DJINNI_LABELED = Path("data/processed/djinni.csv")
LINKEDIN_PROCESSED = Path("data/processed/linkedin.csv")
SPLITS_DIR = Path("data/splits")
