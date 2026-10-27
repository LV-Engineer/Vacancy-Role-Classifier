"""Common schema and helpers shared by all data sources."""
import re

import pandas as pd

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

SEED = 42

def weak_role(title: str) -> str:
    """Weak label from title keywords. Used for train and as a pre-label for manual work."""
    lowered = title.lower()
    for role, pattern in ROLE_PATTERNS:
        if re.search(pattern, lowered):
            return role
    return "Other"

def level_from_years(years: float | None) -> str | None:
    """junior = up to 1 year of required experience; unknown stays empty."""
    if years is None or pd.isna(years):
        return None
    return "junior" if years <= 1 else "not_junior"

def detect_language(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "en"
    return "uk" if len(CYRILLIC.findall(text)) / len(letters) > 0.3 else "en"