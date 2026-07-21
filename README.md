# FunnelIQ

Marketing-intelligence tool for Northbound Media: predicts customer lifetime and upsell
probability from funnel data, using gradient boosting (XGBoost, LightGBM, CatBoost).

**Status:** in progress. Data/modeling work is underway; API, database, auth, and
deployment are not wired up yet. This README will grow with the project.

## Live URL

_Not deployed yet._

## Setup

```bash
pip install -r requirements.txt
python train_lifetime_model.py
```

`funnel_marketing_data.csv` (in this repo) is the source dataset; the script trains
all three models with 5-fold CV and saves them locally (gitignored, regenerate by
rerunning the script).
