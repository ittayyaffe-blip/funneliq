"""
Shared training/serving logic for the Package 2 lifetime models (XGBoost,
LightGBM, CatBoost). Trained once at API startup from the CSV already in
the repo - no separate model-artifact files to keep in sync.
"""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

DATA_PATH = "funnel_marketing_data.csv"
TARGET = "ltv_months"

# ponytail: leakage columns excluded per design discussion - cumulative_profit
# is derived from tenure itself, upsell/referred happen during the
# relationship, purchased becomes constant after filtering to customers.
FEATURES = [
    "ad_budget", "num_leads", "leads_answered", "leads_not_answered",
    "followup_1", "followup_2", "followup_3", "followup_4", "followup_5",
    "not_closed", "closed", "calls_to_closed", "calls_to_not_closed",
    "customer_acquisition_cost",
]


def make_models():
    return {
        "XGBoost": XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
            importance_type="gain",
        ),
        "CatBoost": CatBoostRegressor(
            iterations=300, learning_rate=0.05, depth=4,
            random_state=42, verbose=False, allow_writing_files=False,
        ),
    }


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df[df["purchased"] == 1]
    df = df.dropna(subset=[TARGET])
    return df[FEATURES], df[TARGET]


def cross_validate(model, X, y, folds=5):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    rmses, r2s = [], []
    for train_idx, val_idx in kf.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[val_idx])
        rmses.append(mean_squared_error(y.iloc[val_idx], pred) ** 0.5)
        r2s.append(r2_score(y.iloc[val_idx], pred))
    return np.array(rmses), np.array(r2s)


class LifetimeModels:
    def __init__(self):
        self.models = {}
        self.comparison = {}
        self.feature_defaults = {}

    def train(self):
        X, y = load_data()
        self.feature_defaults = {f: round(float(X[f].median()), 1) for f in FEATURES}
        self.models = make_models()
        for name, model in self.models.items():
            rmses, r2s = cross_validate(model, X, y)
            model.fit(X, y)  # refit on full data for serving + importances
            importances = pd.Series(model.feature_importances_, index=FEATURES)
            importances = importances / importances.sum()
            top5 = importances.sort_values(ascending=False).head(5)
            self.comparison[name] = {
                "rmse_mean": round(float(rmses.mean()), 3),
                "rmse_std": round(float(rmses.std()), 3),
                "r2_mean": round(float(r2s.mean()), 3),
                "r2_std": round(float(r2s.std()), 3),
                "top_features": [
                    {"feature": f, "importance": round(float(v), 4)}
                    for f, v in top5.items()
                ],
            }

    def predict(self, features: dict):
        row = pd.DataFrame([{f: features.get(f, self.feature_defaults[f]) for f in FEATURES}])
        return {name: round(float(model.predict(row)[0]), 1) for name, model in self.models.items()}


lifetime_models = LifetimeModels()
