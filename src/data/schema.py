"""Common schema and helpers shared by all data sources."""
import pandas as pd

from data.constants import CYRILLIC

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