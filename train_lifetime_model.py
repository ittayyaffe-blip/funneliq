"""
Package 2 - Predict customer lifetime (ltv_months) with gradient boosting.

Trains XGBoost, LightGBM, and CatBoost regressors, compares them with 5-fold
cross-validation on RMSE and R^2, and reports feature importances. Same
training logic the API uses (see models.py) - this script is for exploring
results on the command line.

Usage: python train_lifetime_model.py
"""
from models import LifetimeModels


def main():
    lm = LifetimeModels()
    lm.train()

    print(f"{'Model':<10} {'RMSE (mean+/-std)':<22} {'R2 (mean+/-std)':<20}")
    for name, c in lm.comparison.items():
        print(f"{name:<10} {c['rmse_mean']:6.3f} +/- {c['rmse_std']:<12.3f} {c['r2_mean']:6.3f} +/- {c['r2_std']:<12.3f}")

    print("\nTop feature importances (normalized to % of total):")
    for name, c in lm.comparison.items():
        print(f"\n{name}:")
        for f in c["top_features"]:
            print(f"  {f['feature']:<28} {f['importance']:6.1%}")


if __name__ == "__main__":
    main()
