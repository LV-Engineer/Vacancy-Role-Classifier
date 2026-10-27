"""Split labeled Djinni into dev and a held-out test, then build the train set."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from data.schema import COLUMNS, LEVELS, ROLES, detect_language, SEED

LABELED = Path("data/processed/djinni.csv")
LINKEDIN = Path("data/processed/linkedin.csv")
OUT_DIR = Path("data/splits")
TEST_FOLDS = 4

def load_labeled() -> pd.DataFrame:
    df = pd.read_csv(LABELED)
    problems = []
    if not df["role"].isin(ROLES).all():
        problems.append("empty or unknown role")
    if not df["level"].isin(LEVELS).all():
        problems.append("empty or unknown level")
    if df["id"].duplicated().any():
        problems.append("duplicate ids")
    if problems:
        raise ValueError(f"{LABELED}: {', '.join(problems)}")
    return df

def company_key(description: str) -> str:
    """Rough group key: vacancies from one company share the opening text."""
    return " ".join(description.lower().split())[:80]

def main() -> None:
    djinni = load_labeled()
    linkedin = pd.read_csv(LINKEDIN)

    splitter = StratifiedGroupKFold(n_splits=TEST_FOLDS, shuffle=True, random_state=SEED)
    groups = djinni["description"].map(company_key)
    dev_idx, test_idx = next(splitter.split(djinni, djinni['role'], groups))
    dev, test = djinni.iloc[dev_idx], djinni.iloc[test_idx]

    leaked = test["description"].isin(linkedin["description"]).sum()
    assert leaked == 0, f"{leaked} test descriptions are also in {LINKEDIN}"

    dev_train = dev.assign(language=dev["description"].map(detect_language), source="djinni")
    train = pd.concat([linkedin[COLUMNS], dev_train[COLUMNS]], ignore_index=True)

    assert list(train.columns) == COLUMNS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT_DIR / "train.csv", index=False, encoding="utf-8-sig")
    dev.to_csv(OUT_DIR / "djinni_dev.csv", index=False, encoding="utf-8-sig")
    test.to_csv(OUT_DIR / "djinni_test.csv", index=False, encoding="utf-8-sig")

    for name, part in (("train", train), ("djinni_dev", dev), ("djinni_test", test)):
        print(f"--- {name}: {len(part)} rows")
        print(part["role"].value_counts().to_string())
        print(part["level"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()