"""
Downloads the two source datasets used by rebalance_pipeline.py into ./data/.

COMPAS: ProPublica's original two-year recidivism extract (public GitHub repo).
Adult Income: UCI Adult dataset, unheadered CSV mirror.

Run this once before running rebalance_pipeline.py:
    python download_data.py
"""

import os
import urllib.request

COMPAS_URL = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
ADULT_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv"

OUT_DIR = "data"


def fetch(url, out_path):
    if os.path.exists(out_path):
        print(f"already have {out_path}, skipping")
        return
    print(f"downloading {url} -> {out_path}")
    urllib.request.urlretrieve(url, out_path)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    fetch(COMPAS_URL, os.path.join(OUT_DIR, "compas.csv"))
    fetch(ADULT_URL, os.path.join(OUT_DIR, "adult.csv"))
    print("done")
