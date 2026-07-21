"""
Package 2 - Predict customer lifetime (ltv_months) with gradient boosting.

Trains XGBoost, LightGBM, and CatBoost regressors, compares them with 5-fold
cross-validation on RMSE and R^2, and reports feature importances.

Usage: python train_lifetime_model.py
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

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

MODELS = {
    "XGBoost": XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
        importance_type="gain",  # match XGBoost/CatBoost's gain-based importance
    ),
    "CatBoost": CatBoostRegressor(
        iterations=300, learning_rate=0.05, depth=4,
        random_state=42, verbose=False,
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


def main():
    X, y = load_data()
    print(f"Training rows: {len(X)} (customers with purchased=1)\n")

    results = {}
    for name, model in MODELS.items():
        rmses, r2s = cross_validate(model, X, y)
        results[name] = (rmses.mean(), rmses.std(), r2s.mean(), r2s.std())
        model.fit(X, y)  # refit on full data for feature importance + saving

    print(f"{'Model':<10} {'RMSE (mean+/-std)':<22} {'R2 (mean+/-std)':<20}")
    for name, (rm, rs, r2m, r2s) in results.items():
        print(f"{name:<10} {rm:6.3f} +/- {rs:<12.3f} {r2m:6.3f} +/- {r2s:<12.3f}")

    print("\nTop feature importances (normalized to % of total):")
    for name, model in MODELS.items():
        importances = pd.Series(model.feature_importances_, index=FEATURES)
        importances = importances / importances.sum()
        top = importances.sort_values(ascending=False).head(5)
        print(f"\n{name}:")
        for feat, val in top.items():
            print(f"  {feat:<28} {val:6.1%}")

    MODELS["XGBoost"].save_model("model_lifetime_xgboost.json")
    MODELS["LightGBM"].booster_.save_model("model_lifetime_lightgbm.txt")
    MODELS["CatBoost"].save_model("model_lifetime_catboost.cbm")
    print("\nSaved model files: model_lifetime_{xgboost.json,lightgbm.txt,catboost.cbm}")


if __name__ == "__main__":
    main()
