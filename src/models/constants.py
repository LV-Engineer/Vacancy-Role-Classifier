from pathlib import Path

SPLITS = Path("data/splits")
FIELDS = ("title", "description")
TFIDF_PARAMS = {"ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
LOGREG_PARAMS = {"max_iter": 1000, "class_weight": "balanced"}
N_FOLDS = 5
TRAIN = Path("data/splits/train.csv")