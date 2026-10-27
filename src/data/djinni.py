from pathlib import Path

import pandas as pd

from data.base import DataSource
from data.schema import level_from_years, weak_role

class DjinniSource(DataSource):
    name = "djinni"
    n_other = 150
    out = Path("data/labeling/djinni_to_label.csv")

    def __init__(self, path: Path = Path("data/raw/djinni_jobs.jsonl")):
        self.path = path

    def _read_raw(self) -> pd.DataFrame:
        return pd.read_json(self.path, lines=True)

    def _to_schema(self, raw: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "id": raw["id"],
                "title": raw["title"],
                "description": raw["description"],
                "language": raw["language"],
                "role": raw["title"].map(weak_role),
                "level": raw["experience_years"].map(level_from_years),
                "source": self.name,
            }
        )

    def _finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.rename(columns={"role": "weak_role", "level": "level_auto"})
            .assign(role="", level="")
            [["id", "title", "description", "weak_role", "level_auto", "role", "level"]]
        )