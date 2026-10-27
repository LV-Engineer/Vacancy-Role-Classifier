"""Base class for data sources: read, map to the common schema, validate."""
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from data.schema import COLUMNS, LEVELS, MIN_DESCRIPTION_CHARS, ROLES, SEED


class DataSource(ABC):
    name: str
    n_other: int
    out: Path

    @abstractmethod
    def _read_raw(self) -> pd.DataFrame:
        """Read the source file as it is."""

    @abstractmethod
    def _to_schema(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Map raw columns to the common schema."""

    def load(self) -> pd.DataFrame:
        df = self._to_schema(self._read_raw())[COLUMNS].reset_index(drop=True)
        self._validate(df)
        return df

    def _validate(self, df: pd.DataFrame) -> None:
        problems = []
        if df["id"].duplicated().any():
            problems.append("duplicate ids")
        if not df["role"].isin(ROLES).all():
            problems.append("unknown role")
        if not df["level"].dropna().isin(LEVELS).all():
            problems.append("unknown level")
        if (df["description"].str.len() < MIN_DESCRIPTION_CHARS).any():
            problems.append("short description")
        if (df["source"] != self.name).any():
            problems.append("wrong source name")
        if problems:
            raise ValueError(f"{self.name}: {', '.join(problems)}")

    def prepare(self) -> pd.DataFrame:
        df = self.load().drop_duplicates(subset=["title", "description"])
        other = df[df["role"] == "Other"]
        rest = df[df["role"] != "Other"]
        other = other.sample(min(self.n_other, len(other)), random_state=SEED)
        df = pd.concat([rest, other]).sample(frac=1, random_state=SEED)
        return self._finalize(df)

    def _finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hook: source-specific last step. By default nothing changes."""
        return df