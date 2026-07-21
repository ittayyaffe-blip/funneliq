"""
Loads funnel_marketing_data.csv into Supabase. Run schema.sql in the Supabase
SQL Editor first to create the table.

Usage: python load_data.py
"""
import math
import os

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

DATA_PATH = "funnel_marketing_data.csv"
TABLE = "funnel_marketing_data"
BATCH_SIZE = 500


def clean(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def main():
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    client = create_client(url, key)

    df = pd.read_csv(DATA_PATH)
    df["purchased"] = df["purchased"].astype(bool)
    df["upsell"] = df["upsell"].astype(bool)
    df["referred"] = df["referred"].map({"Yes": True, "No": False})

    records = [
        {k: clean(v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]

    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        client.table(TABLE).insert(batch).execute()
        print(f"Inserted rows {i} - {i + len(batch)}")

    print(f"\nDone. Loaded {len(records)} rows into {TABLE}.")


if __name__ == "__main__":
    main()
