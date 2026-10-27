"""LinkedIn postings from Kaggle (CC BY-SA 4.0): train data with weak role labels."""
from pathlib import Path

import pandas as pd

from data.base import DataSource
from data.schema import MIN_DESCRIPTION_CHARS, detect_language, weak_role

JUNIOR_LEVELS = {"Entry level", "Internship"}

class LinkedInSource(DataSource):
    name = "linkedin"
    n_other = 1200
    out = Path("data/processed/linkedin.csv")

    def __init__(self, path: Path = Path("data/raw/linkedin-postings-kaggle.csv.gz")):
        self.path = path

    def _read_raw(self) -> pd.DataFrame:
        return pd.read_csv(
            self.path,
            usecols=["job_id", "title", "description", "formatted_experience_level"],
        )

    def _to_schema(self, raw: pd.DataFrame) -> pd.DataFrame:
        raw = raw.dropna(subset=["title", "description"]).copy()
        raw["title"] = raw["title"].str.strip()
        raw["description"] = raw["description"].str.strip()
        raw = raw[raw["description"].str.len() >= MIN_DESCRIPTION_CHARS]
        return pd.DataFrame(
            {
                "id": "linkedin_" + raw["job_id"].astype(str),
                "title": raw["title"],
                "description": raw["description"],
                "language": raw["description"].map(detect_language),
                "role": raw["title"].map(weak_role),
                "level": raw["formatted_experience_level"].map(self._level),
                "source": self.name,
            }
        )

    @staticmethod
    def _level(label: object) -> str | None:
        if not isinstance(label, str) or not label:
            return None
        return "junior" if label in JUNIOR_LEVELS else "not_junior"