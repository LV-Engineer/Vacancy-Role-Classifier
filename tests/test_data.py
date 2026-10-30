from pathlib import Path

import pandas as pd
import pytest

from data.base import DataSource
from data.constants import MIN_DESCRIPTION_CHARS
from data.linkedin import LinkedInSource
from data.split import company_key
from data import split
from data.constants import COLUMNS, ROLES
from data.schema import detect_language

class FakeSource(DataSource):
    name = "fake"
    n_other = 1
    out = Path("fake.csv")

    def _read_raw(self) -> pd.DataFrame:
        return pd.DataFrame()

    def _to_schema(self, raw: pd.DataFrame) -> pd.DataFrame:
        return raw

def posting(**overrides) -> dict:
    row = {
        "id": "fake_1", "title": "Data Analyst", "description": "x" * MIN_DESCRIPTION_CHARS,
        "language": "en", "role": "Data Analyst", "level": "junior", "source": "fake",
    }
    return row | overrides

def test_validate_accepts_a_valid_frame():
    FakeSource()._validate(pd.DataFrame([posting()]))

@pytest.mark.parametrize("bad, message", [
    ({"role": "Chef"}, "unknown role"),
    ({"level": "senior"}, "unknown level"),
    ({"description": "short"}, "short description"),
    ({"source": "other"}, "wrong source name"),
])
def test_validate_rejects_bad_rows(bad, message):
    with pytest.raises(ValueError, match=message):
        FakeSource()._validate(pd.DataFrame([posting(**bad)]))

def test_validate_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="duplicate ids"):
        FakeSource()._validate(pd.DataFrame([posting(), posting()]))

def test_prepare_drops_duplicates_and_limits_other():
    class Source(FakeSource):
        def load(self) -> pd.DataFrame:
            return pd.DataFrame([
                posting(id="a"), posting(id="b"),
                posting(id="c", title="Cook", role="Other"),
                posting(id="d", title="Baker", role="Other"),
                posting(id="e", title="Driver", role="Other"),
            ])

    result = Source().prepare()
    assert len(result) == 2
    assert result["role"].value_counts().to_dict() == {"Data Analyst": 1, "Other": 1}

@pytest.mark.parametrize("label, level", [
    ("Entry level", "junior"), ("Internship", "junior"), ("Mid-Senior level", "not_junior"),
    (None, None), (float("nan"), None), ("", None),
])
def test_linkedin_level(label, level):
    assert LinkedInSource._level(label) == level

def test_company_key_ignores_case_and_spaces():
    assert company_key("  Our   Company builds") == company_key("our company builds")

def test_company_key_is_cut_at_80_characters():
    assert len(company_key("x " * 100)) == 80

def test_detect_language_defaults_to_english_without_letters():
    assert detect_language("123 !!!") == "en"

def test_load_returns_the_common_columns():
    class Source(FakeSource):
        def _read_raw(self) -> pd.DataFrame:
            return pd.DataFrame([posting()])

    assert list(Source().load().columns) == COLUMNS

def test_load_rejects_an_invalid_frame():
    class Source(FakeSource):
        def _read_raw(self) -> pd.DataFrame:
            return pd.DataFrame([posting(role="Chef")])

    with pytest.raises(ValueError, match="unknown role"):
        Source().load()

def test_linkedin_reads_and_maps_to_the_common_schema(tmp_path):
    path = tmp_path / "linkedin.csv"
    pd.DataFrame({
        "job_id": [1, 2, 3, 4],
        "title": ["Data Analyst ", "Senior Python Developer", None, "Intern"],
        "description": ["a" * 300, "b" * 300, "c" * 300, "short"],
        "formatted_experience_level": ["Entry level", "Mid-Senior level", "Entry level", None],
        "company_name": "ignored",
    }).to_csv(path, index=False)

    frame = LinkedInSource(path).load()

    assert list(frame.columns) == COLUMNS
    assert frame["id"].tolist() == ["linkedin_1", "linkedin_2"]
    assert frame["role"].tolist() == ["Data Analyst", "Python Backend"]
    assert frame["level"].tolist() == ["junior", "not_junior"]

def make_labeled(per_role: int = 8) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "id": f"djinni_{role}_{i}", "title": f"{role} {i}",
            "description": f"{role} company {i} " + "x" * 200,
            "role": role, "level": "junior" if i % 2 else "not_junior",
        }
        for role in ROLES for i in range(per_role)
    ])

def make_linkedin(rows: int = 10) -> pd.DataFrame:
    return pd.DataFrame([
        {"id": f"linkedin_{i}", "title": "Cook", "description": f"other text {i}", "language": "en",
         "role": "Other", "level": "junior", "source": "linkedin"}
        for i in range(rows)
    ])

def test_load_labeled_rejects_an_empty_role(tmp_path, monkeypatch):
    path = tmp_path / "djinni.csv"
    labeled = make_labeled()
    labeled.loc[0, "role"] = ""
    labeled.to_csv(path, index=False)
    monkeypatch.setattr(split, "DJINNI_LABELED", path)

    with pytest.raises(ValueError, match="empty or unknown role"):
        split.load_labeled()


@pytest.mark.parametrize("column, value, message", [
    ("level", "senior", "empty or unknown level"),
    ("id", "djinni_Other_0", "duplicate ids"),
])
def test_load_labeled_rejects_bad_levels_and_duplicate_ids(tmp_path, monkeypatch, column, value, message):
    path = tmp_path / "djinni.csv"
    labeled = make_labeled()
    labeled.loc[1, column] = value
    labeled.to_csv(path, index=False)
    monkeypatch.setattr(split, "DJINNI_LABELED", path)

    with pytest.raises(ValueError, match=message):
        split.load_labeled()

def test_split_builds_train_dev_and_test(tmp_path, monkeypatch):
    labeled, linkedin, out = tmp_path / "djinni.csv", tmp_path / "linkedin.csv", tmp_path / "splits"
    make_labeled().to_csv(labeled, index=False)
    make_linkedin().to_csv(linkedin, index=False)
    monkeypatch.setattr(split, "DJINNI_LABELED", labeled)
    monkeypatch.setattr(split, "LINKEDIN_PROCESSED", linkedin)
    monkeypatch.setattr(split, "SPLITS_DIR", out)

    split.main()

    train, dev, test = (pd.read_csv(out / f"{name}.csv", encoding="utf-8-sig") for name in ("train", "djinni_dev", "djinni_test"))
    assert len(dev) + len(test) == 40
    assert set(dev["id"]).isdisjoint(test["id"])
    assert 8 <= len(test) <= 12
    assert len(train) == 10 + len(dev)
    assert set(dev["id"]) == set(train.loc[train["source"] == "djinni", "id"])