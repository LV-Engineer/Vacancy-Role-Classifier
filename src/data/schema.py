"""Common schema and helpers shared by all data sources."""
import re

import pandas as pd

from data.constants import CYRILLIC, ROLE_PATTERNS

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