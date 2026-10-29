from pathlib import Path

from data.constants import LEVELS, ROLES

SPLITS = Path("data/splits")
TRAIN = Path("data/splits/train.csv")

FIELDS = ("title", "description")
TFIDF_FIELDS = FIELDS + ("both",)

TFIDF_PARAMS = {"ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
LOGREG_PARAMS = {"max_iter": 1000, "class_weight": "balanced"}
N_FOLDS = 5
WEIGHT_PARAM = "logisticregression__sample_weight"

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_MAX_TOKENS = 256

N_SHOWN = 30

TARGETS = ("role", "level")
LABELS = {"role": ROLES, "level": LEVELS}
MODEL_NAMES = {"role": "vacancy-role-classifier", "level": "vacancy-level-classifier"}
PROMOTION_METRIC = "test_macro_f1"
PROMOTION_TOLERANCE = 0.04