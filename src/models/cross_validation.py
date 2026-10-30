"""Cross-validation shared by all role models: LinkedIn always trains, Djinni is validated by folds."""
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from data.constants import SEED
from data.split import company_key
from models.constants import N_FOLDS, WEIGHT_PARAM

def rows(X: pd.Series | pd.DataFrame | np.ndarray, pos: np.ndarray) -> pd.Series | pd.DataFrame | np.ndarray:
    return X.iloc[pos] if isinstance(X, (pd.Series, pd.DataFrame)) else X[pos]

def cross_validate(train: pd.DataFrame, X: pd.Series | pd.DataFrame | np.ndarray, build_model: Callable[[], Any], djinni_weight: float = 1.0, target: str = "role") -> tuple[pd.DataFrame, np.ndarray]:
    """Predict the Djinni rows fold by fold: each fold is predicted by a model trained without it."""
    is_djinni = train["source"].eq("djinni").to_numpy()
    known = train[target].notna().to_numpy()
    linkedin_pos = np.flatnonzero(~is_djinni & known)
    djinni_pos = np.flatnonzero(is_djinni)
    djinni = train.iloc[djinni_pos]
    folds = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    groups = djinni["description"].map(company_key)

    y_pred = np.empty(len(djinni), dtype=object)
    for fit_idx, val_idx in folds.split(djinni, djinni[target], groups):
        fit_pos = np.concatenate([linkedin_pos, djinni_pos[fit_idx]])
        weights = np.where(is_djinni[fit_pos], djinni_weight, 1.0)
        model = build_model().fit(rows(X, fit_pos), train[target].iloc[fit_pos], **{WEIGHT_PARAM: weights})
        y_pred[val_idx] = model.predict(rows(X, djinni_pos[val_idx]))
    return djinni, y_pred