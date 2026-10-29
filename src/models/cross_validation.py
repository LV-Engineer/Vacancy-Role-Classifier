"""Cross-validation shared by all role models: LinkedIn always trains, Djinni is validated by folds."""
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from data.constants import SEED
from data.split import company_key
from models.constants import N_FOLDS

def rows(X, pos: np.ndarray):
    return X.iloc[pos] if isinstance(X, pd.Series) else X[pos]

def cross_validate(train: pd.DataFrame, X, build_model: Callable) -> tuple[pd.DataFrame, np.ndarray]:
    """Predict the Djinni rows fold by fold: each fold is predicted by a model trained without it."""
    is_djinni = train["source"].eq("djinni").to_numpy()
    linkedin_pos = np.flatnonzero(~is_djinni)
    djinni_pos = np.flatnonzero(is_djinni)
    djinni = train.iloc[djinni_pos]
    folds = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    groups = djinni["description"].map(company_key)

    y_pred = np.empty(len(djinni), dtype=object)
    for fit_idx, val_idx in folds.split(djinni, djinni["role"], groups):
        fit_pos = np.concatenate([linkedin_pos, djinni_pos[fit_idx]])
        model = build_model().fit(rows(X, fit_pos), train["role"].iloc[fit_pos])
        y_pred[val_idx] = model.predict(rows(X, djinni_pos[val_idx]))
    return djinni, y_pred