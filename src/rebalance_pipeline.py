"""
REBALANCE pipeline (MSc-scoped)
Quantifying the Racial Bias Accuracy-Fairness Trade-off in Machine Learning
Author: Joshua Oghosasehe Ekogiawe (B20102926), Ulster University

Scope (trimmed to a realistic MSc dissertation budget):
  - Datasets: COMPAS (criminal justice), Adult Income (employment).
  - Techniques (5, covering all three AIF360 categories):
      pre-processing  : Reweighing, Disparate Impact Remover
      in-processing    : Prejudice Remover (fixed logistic architecture)
      post-processing  : Calibrated Equalised Odds, Reject Option
  - Classifiers (4): Logistic Regression, Random Forest, XGBoost, MLP
  - Seeds: 3 (train/test split + model init), reported as mean +/- std
  - Hyperparameters: fixed, sensible defaults (documented in Section III-E
    of the methodology) rather than a full Bayesian sweep per condition,
    which is disproportionate to an MSc compute/time budget.

Per-dataset condition count: 4 (baseline) + 4 (Reweighing) + 4 (DIR)
  + 1 (Prejudice Remover, fixed architecture) + 4 (CalEqOdds) + 4 (RejectOpt)
  = 21 conditions x 2 datasets = 42 conditions x 3 seeds = 126 runs.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             matthews_corrcoef)
from xgboost import XGBClassifier

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
from aif360.algorithms.inprocessing import PrejudiceRemover
from aif360.algorithms.postprocessing import (CalibratedEqOddsPostprocessing,
                                              RejectOptionClassification)

SEEDS = [42, 7, 123]

# ---------------------------------------------------------------- datasets
def load_compas(path="data/compas.csv"):
    df = pd.read_csv(path)
    df = df[(df.days_b_screening_arrest <= 30) &
            (df.days_b_screening_arrest >= -30) &
            (df.is_recid != -1) &
            (df.c_charge_degree != "O") &
            (df.score_text != "N/A")]
    df = df[df.race.isin(["African-American", "Caucasian"])].copy()
    df["race_bin"] = (df.race == "Caucasian").astype(int)
    df["sex_bin"] = (df.sex == "Male").astype(int)
    df["charge_felony"] = (df.c_charge_degree == "F").astype(int)
    df["label"] = 1 - df.two_year_recid
    feats = ["age", "priors_count", "juv_fel_count", "juv_misd_count",
             "juv_other_count", "sex_bin", "charge_felony", "race_bin"]
    out = df[feats + ["label"]].dropna().reset_index(drop=True)
    return out, "race_bin", "COMPAS"

def load_adult(path="data/adult.csv"):
    cols = ["age", "workclass", "fnlwgt", "education", "education_num",
            "marital_status", "occupation", "relationship", "race", "sex",
            "capital_gain", "capital_loss", "hours_per_week",
            "native_country", "income"]
    df = pd.read_csv(path, names=cols, skipinitialspace=True, na_values="?")
    df = df[df.race.isin(["White", "Black"])].copy()
    df["race_bin"] = (df.race == "White").astype(int)
    df["sex_bin"] = (df.sex == "Male").astype(int)
    df["label"] = df.income.str.contains(">50K").astype(int)
    cat = ["workclass", "marital_status", "occupation", "relationship"]
    df = df.dropna(subset=cat)
    dummies = pd.get_dummies(df[cat], drop_first=True).astype(int)
    num = df[["age", "education_num", "capital_gain", "capital_loss",
              "hours_per_week", "sex_bin", "race_bin"]].reset_index(drop=True)
    out = pd.concat([num, dummies.reset_index(drop=True),
                     df["label"].reset_index(drop=True)], axis=1)
    return out, "race_bin", "Adult"

# ---------------------------------------------------------- AIF360 helpers
def to_bld(df, prot):
    return BinaryLabelDataset(df=df, label_names=["label"],
                              protected_attribute_names=[prot],
                              favorable_label=1, unfavorable_label=0)

def scale_pair(train, test):
    sc = StandardScaler()
    tr, te = train.copy(deepcopy=True), test.copy(deepcopy=True)
    tr.features = sc.fit_transform(tr.features)
    te.features = sc.transform(te.features)
    return tr, te

def classifiers(seed):
    return {
        "LogReg": LogisticRegression(max_iter=1000, random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=100,
                                               random_state=seed, n_jobs=2),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=5,
                                 learning_rate=0.1, random_state=seed,
                                 eval_metric="logloss", n_jobs=2),
        "MLP": MLPClassifier(hidden_layer_sizes=(32,), max_iter=200,
                             early_stopping=True, random_state=seed),
    }

def evaluate(test_bld, pred_bld, prot):
    priv, unpriv = [{prot: 1}], [{prot: 0}]
    cm = ClassificationMetric(test_bld, pred_bld,
                              unprivileged_groups=unpriv,
                              privileged_groups=priv)
    y, yp = test_bld.labels.ravel(), pred_bld.labels.ravel()
    try:
        auc = roc_auc_score(y, pred_bld.scores.ravel())
    except Exception:
        auc = np.nan
    return {
        "accuracy": accuracy_score(y, yp),
        "f1": f1_score(y, yp),
        "auc": auc,
        "mcc": matthews_corrcoef(y, yp),
        "stat_parity_diff": cm.statistical_parity_difference(),
        "eq_odds_diff": cm.average_odds_difference(),
        "eq_opp_diff": cm.equal_opportunity_difference(),
        "disparate_impact": cm.disparate_impact(),
    }

def predict_bld(model, test_scaled, test_raw, threshold=0.5):
    pred = test_raw.copy(deepcopy=True)
    proba = model.predict_proba(test_scaled.features)[:, 1]
    pred.scores = proba.reshape(-1, 1)
    pred.labels = (proba >= threshold).astype(int).reshape(-1, 1)
    return pred

# ------------------------------------------------------------- experiments
def run_seed(loader, seed):
    df, prot, name = loader()
    bld = to_bld(df, prot)
    train, test = bld.split([0.7], shuffle=True, seed=seed)
    tr_s, te_s = scale_pair(train, test)
    rows = []
    priv, unpriv = [{prot: 1}], [{prot: 0}]

    for clf_name, clf in classifiers(seed).items():
        clf.fit(tr_s.features, tr_s.labels.ravel())
        pred_te = predict_bld(clf, te_s, test)
        rows.append({"dataset": name, "seed": seed, "classifier": clf_name,
                     "technique": "Baseline", "category": "none",
                     **evaluate(test, pred_te, prot)})

        pred_tr = predict_bld(clf, tr_s, train)
        for tech_name, tech in [
            ("CalibratedEqOdds", CalibratedEqOddsPostprocessing(
                unprivileged_groups=unpriv, privileged_groups=priv,
                cost_constraint="fnr", seed=seed)),
            ("RejectOption", RejectOptionClassification(
                unprivileged_groups=unpriv, privileged_groups=priv,
                low_class_thresh=0.01, high_class_thresh=0.99,
                num_class_thresh=20, num_ROC_margin=10,
                metric_name="Statistical parity difference",
                metric_ub=0.05, metric_lb=-0.05)),
        ]:
            try:
                tech.fit(train, pred_tr)
                pred_pp = tech.predict(pred_te)
                rows.append({"dataset": name, "seed": seed,
                             "classifier": clf_name, "technique": tech_name,
                             "category": "post",
                             **evaluate(test, pred_pp, prot)})
            except Exception as e:
                print(f"  [skip] {clf_name}+{tech_name}: {e}")

    # ---- pre-processing 1: Reweighing
    rw = Reweighing(unprivileged_groups=unpriv, privileged_groups=priv)
    tr_rw = rw.fit_transform(train)
    tr_rw_s, te_rw_s = scale_pair(tr_rw, test)
    for clf_name, clf in classifiers(seed).items():
        try:
            clf.fit(tr_rw_s.features, tr_rw_s.labels.ravel(),
                    sample_weight=tr_rw.instance_weights)
        except TypeError:
            clf.fit(tr_rw_s.features, tr_rw_s.labels.ravel())
        pred_te = predict_bld(clf, te_rw_s, test)
        rows.append({"dataset": name, "seed": seed, "classifier": clf_name,
                     "technique": "Reweighing", "category": "pre",
                     **evaluate(test, pred_te, prot)})

    # ---- pre-processing 2: Disparate Impact Remover
    dir_ = DisparateImpactRemover(repair_level=1.0, sensitive_attribute=prot)
    tr_dir = dir_.fit_transform(train)
    te_dir = dir_.fit_transform(test)
    tr_dir_s, te_dir_s = scale_pair(tr_dir, te_dir)
    for clf_name, clf in classifiers(seed).items():
        clf.fit(tr_dir_s.features, tr_dir_s.labels.ravel())
        pred_te = predict_bld(clf, te_dir_s, test)
        rows.append({"dataset": name, "seed": seed, "classifier": clf_name,
                     "technique": "DisparateImpactRemover", "category": "pre",
                     **evaluate(test, pred_te, prot)})

    # ---- in-processing: Prejudice Remover (fixed logistic architecture)
    try:
        pr = PrejudiceRemover(eta=25.0, sensitive_attr=prot)
        pr.fit(tr_s)
        pr_out = pr.predict(te_s)
        pred_pr = test.copy(deepcopy=True)
        pred_pr.scores = pr_out.scores
        pred_pr.labels = (pr_out.scores >= 0.5).astype(int)
        rows.append({"dataset": name, "seed": seed, "classifier": "PR-LogReg",
                     "technique": "PrejudiceRemover", "category": "in",
                     **evaluate(test, pred_pr, prot)})
    except Exception as e:
        print(f"  [skip] PrejudiceRemover: {e}")

    return rows

RAW_CSV = "results_full_raw.csv"
LOADERS = {"compas": load_compas, "adult": load_adult}

def checkpoint_run(dataset_key, seed):
    """Run one (dataset, seed) combination and append to the raw results CSV.
    Safe to call repeatedly across separate invocations; skips work already
    recorded for that dataset+seed so re-running is idempotent."""
    import os
    if os.path.exists(RAW_CSV):
        existing = pd.read_csv(RAW_CSV)
        if ((existing.dataset.str.lower() == dataset_key.lower()) &
            (existing.seed == seed)).any():
            print(f"already have {dataset_key} seed={seed}, skipping")
            return
    rows = run_seed(LOADERS[dataset_key], seed)
    df_new = pd.DataFrame(rows).round(4)
    if os.path.exists(RAW_CSV):
        df_new = pd.concat([pd.read_csv(RAW_CSV), df_new], ignore_index=True)
    df_new.to_csv(RAW_CSV, index=False)
    print(f"done {dataset_key} seed={seed}: {len(rows)} rows -> {RAW_CSV} "
          f"(total {len(df_new)} rows)")

def aggregate():
    raw = pd.read_csv(RAW_CSV)
    metrics = ["accuracy", "f1", "auc", "mcc", "stat_parity_diff",
               "eq_odds_diff", "eq_opp_diff", "disparate_impact"]
    agg = (raw.groupby(["dataset", "classifier", "technique", "category"])[metrics]
              .agg(["mean", "std"]).round(4))
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    agg.to_csv("results_full_aggregated.csv", index=False)
    print(f"{len(raw)} total runs; aggregated -> results_full_aggregated.csv")
    print(agg.to_string(index=False))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(LOADERS.keys()))
    p.add_argument("--seed", type=int)
    p.add_argument("--aggregate", action="store_true")
    args = p.parse_args()

    if args.aggregate:
        aggregate()
    elif args.dataset and args.seed is not None:
        checkpoint_run(args.dataset, args.seed)
    else:
        p.error("pass --dataset/--seed for one checkpoint, or --aggregate")
