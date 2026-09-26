from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier


RANDOM_STATE = 42
N_SPLITS = 2
N_REPEATS = 20
TOP_K_VALUES = [10, 20, 30, 50]


# Keep feature-selection randomness identical across Robust and OOF.
def mutual_info_scores(X, y):
    return mutual_info_classif(
        X,
        y,
        random_state=RANDOM_STATE,
    )


def build_xgb_model(n_classes: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=0,
    )
