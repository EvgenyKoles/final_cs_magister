# -*- coding: utf-8 -*-
"""
RQ3: Are model performance and key predictors consistent across demographic subgroups?

Single-output report (JSON) + flat table (CSV):
  - artifacts/interpretability/rq3_subgroups_report.json
  - artifacts/interpretability/rq3_subgroups_table.csv

Содержимое JSON:
  - По каждой подгруппе (gender/year/employment x level) и по каждой модели:
      n, prevalence, PR-AUC, ROC-AUC, F1, Recall@thr, threshold
      top5 features by permutation importance (neg_log_loss) with their scores
  - Сводка (по моделям): вариативность метрик, средний парный Jaccard Top-5

Плоская таблица CSV:
  Columns = [Group, Level, n, Prevalence, Model, PR_AUC, ROC_AUC, F1, Recall@Thr, Thr, Top5]
"""

import argparse, json, pathlib, warnings, sys
import numpy as np
import pandas as pd

from joblib import load
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_fscore_support
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")


# ---------- IO ----------
def load_xy(processed_dir: str):
    p = pathlib.Path(processed_dir)
    X_test = np.loadtxt(p / "X_test.csv",  delimiter=",")
    y_test = np.loadtxt(p / "y_test.csv",  delimiter=",", skiprows=1).astype(int)
    return X_test, y_test

def try_load_feature_names(processed_dir: str, n_features: int):
    path = pathlib.Path(processed_dir) / "feature_names.csv"
    if path.exists():
        try:
            df = pd.read_csv(path, header=None)
            names = df.iloc[:, 0].astype(str).tolist()
            if len(names) == n_features:
                return names
        except Exception:
            pass
    return [f"f{i}" for i in range(n_features)]

def load_models(models_dir: str):
    md = pathlib.Path(models_dir)
    candidates = ["rf.joblib", "xgb.joblib", "xgb_tuned.joblib", "mlp.joblib", "logreg.joblib"]
    models = {}
    for fname in candidates:
        f = md / fname
        if f.exists():
            try:
                models[fname.replace(".joblib","")] = load(f)
            except Exception as e:
                print(f"[WARN] cannot load {f.name}: {type(e).__name__}: {e}")
    return models


# ---------- Metrics ----------
def compute_metrics(y_true, proba, thr):
    uniq = np.unique(y_true)
    pr_auc = float("nan")
    roc_auc = float("nan")
    f1 = float("nan")
    rec_at_thr = float("nan")

    if len(uniq) > 1:
        try:
            pr_auc = float(average_precision_score(y_true, proba))
        except Exception:
            pr_auc = float("nan")
        try:
            roc_auc = float(roc_auc_score(y_true, proba))
        except Exception:
            roc_auc = float("nan")

        y_hat = (proba >= thr).astype(int)
        try:
            f1 = float(f1_score(y_true, y_hat, zero_division=0))
        except Exception:
            f1 = float("nan")
        try:
            prec, rec, _, _ = precision_recall_fscore_support(y_true, y_hat, zero_division=0, average=None)
            rec_at_thr = float(rec[1]) if len(rec) > 1 else float("nan")
        except Exception:
            rec_at_thr = float("nan")

    return {"PR_AUC": pr_auc, "ROC_AUC": roc_auc, "F1": f1, "Recall@Thr": rec_at_thr}

def prevalence(y):
    y = np.asarray(y).astype(int)
    return float(y.mean()) if y.size else float("nan")


# ---------- Importance ----------
def permutation_importance_safe(model, X, y, feature_names, n_repeats=15, seed=42):
    """
    Returns (top5_list, full_sorted_features)
    Each top5 item: {"feature": name, "importance": float}
    """
    if X.shape[0] < 3 or X.shape[1] == 0:
        return [], []

    try:
        res = permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=seed,
            scoring="neg_log_loss",
            n_jobs=-1
        )
        imp = np.maximum(res.importances_mean, 0.0)
        order = np.argsort(-imp)
        feats_sorted = [(feature_names[j], float(imp[j])) for j in order if np.isfinite(imp[j])]
        top5 = [{"feature": feature_names[j], "importance": float(imp[j])} for j in order[:5]]
        return top5, feats_sorted
    except Exception as e:
        print(f"[WARN] permutation importance failed: {type(e).__name__}: {e}")
        return [], []


# ---------- Consistency helpers ----------
def jaccard(a, b):
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def summary_variation(vals):
    arr = np.array([v for v in vals if v is not None and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))])
    if arr.size == 0:
        return {"min": None, "max": None, "mean": None, "std": None, "n": 0}
    return {"min": float(np.min(arr)), "max": float(np.max(arr)),
            "mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=0)), "n": int(arr.size)}


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--models_dir", default="artifacts/models")
    ap.add_argument("--metadata_file", default="artifacts/metadata/subgroups_test.csv")
    ap.add_argument("--artifacts_dir", default="artifacts")
    ap.add_argument("--thr", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_repeats", type=int, default=15)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load data and models
    X_test, y_test = load_xy(args.processed_dir)
    feature_names = try_load_feature_names(args.processed_dir, X_test.shape[1])
    models = load_models(args.models_dir)
    if not models:
        print("[ERROR] No models loaded from artifacts/models")
        sys.exit(1)

    # Load subgroup metadata
    meta_path = pathlib.Path(args.metadata_file)
    if not meta_path.exists():
        print(f"[ERROR] metadata file not found: {meta_path}")
        sys.exit(1)
    df_meta = pd.read_csv(meta_path)
    expected_cols = {"gender","year","employment"}
    if not expected_cols.issubset(set(df_meta.columns)):
        print(f"[ERROR] metadata missing required columns: {expected_cols}")
        sys.exit(1)
    if len(df_meta) != X_test.shape[0]:
        print(f"[ERROR] metadata length {len(df_meta)} != X_test length {X_test.shape[0]}")
        sys.exit(1)

    # Prepare subgroups
    subgroup_vars = ["gender", "year", "employment"]
    subgroups_index = {}
    for var in subgroup_vars:
        subgroups_index[var] = {}
        levels = pd.unique(df_meta[var])
        for lv in levels:
            idx = np.where(df_meta[var].values == lv)[0]
            subgroups_index[var][str(lv)] = idx

    report = {
        "meta": {
            "threshold": args.thr,
            "processed_dir": args.processed_dir,
            "models_dir": args.models_dir,
            "metadata_file": str(meta_path),
            "n_test": int(X_test.shape[0]),
            "n_features": int(X_test.shape[1]),
            "seed": args.seed,
            "n_repeats_permutation": args.n_repeats
        },
        "subgroups": {},
        "consistency": {}
    }

    # будем одновременно собирать строки для плоской таблицы
    flat_rows = []

    # Per-subgroup evaluation
    for var in subgroup_vars:
        report["subgroups"][var] = {}
        for lv, idx in subgroups_index[var].items():
            Xg = X_test[idx, :]
            yg = y_test[idx]
            entry = {
                "n": int(Xg.shape[0]),
                "prevalence": prevalence(yg),
                "models": {}
            }
            for mname, model in models.items():
                # probabilities
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(Xg)[:, 1]
                elif hasattr(model, "decision_function"):
                    s = model.decision_function(Xg)
                    proba = 1.0 / (1.0 + np.exp(-s))
                else:
                    proba = np.zeros(Xg.shape[0], dtype=float)

                mets = compute_metrics(yg, proba, args.thr)
                # importance (Top-5)
                top5, full_sorted = permutation_importance_safe(model, Xg, yg, feature_names,
                                                                n_repeats=args.n_repeats, seed=args.seed)

                entry["models"][mname] = {"metrics": mets, "top5": top5}

                # ---- строка для плоской таблицы ----
                def _r(x):
                    try:
                        return round(float(x), 4)
                    except Exception:
                        return None
                top5_txt = ", ".join([t["feature"] for t in top5]) if top5 else ""
                flat_rows.append({
                    "Group": var,
                    "Level": lv,
                    "n": int(Xg.shape[0]),
                    "Prevalence": _r(entry["prevalence"]),
                    "Model": mname,
                    "PR_AUC": _r(mets.get("PR_AUC")),
                    "ROC_AUC": _r(mets.get("ROC_AUC")),
                    "F1": _r(mets.get("F1")),
                    "Recall@Thr": _r(mets.get("Recall@Thr")),
                    "Thr": args.thr,
                    "Top5": top5_txt
                })

            report["subgroups"][var][lv] = entry

    # Consistency summaries per model
    metrics_names = ["PR_AUC", "ROC_AUC", "F1", "Recall@Thr"]
    for mname in models.keys():
        cons = {"by_variable": {}, "pooled": {}}

        pooled_vals = {mn: [] for mn in metrics_names}
        for var in subgroup_vars:
            vstats = {}
            for mn in metrics_names:
                vals = []
                for lv, entry in report["subgroups"][var].items():
                    val = entry["models"][mname]["metrics"].get(mn, None)
                    vals.append(val)
                    pooled_vals[mn].append(val)
                vstats[mn] = summary_variation(vals)
            cons["by_variable"][var] = vstats
        for mn in metrics_names:
            cons["pooled"][mn] = summary_variation(pooled_vals[mn])

        jacc = {}
        for var in subgroup_vars:
            top_sets = []
            for lv, entry in report["subgroups"][var].items():
                feats = [it["feature"] for it in entry["models"][mname]["top5"]]
                top_sets.append(feats)
            if len(top_sets) >= 2:
                pairs = []
                for i in range(len(top_sets)):
                    for j in range(i+1, len(top_sets)):
                        pairs.append(jaccard(top_sets[i], top_sets[j]))
                jacc[var] = float(np.mean(pairs)) if pairs else None
            else:
                jacc[var] = None
        cons["mean_pairwise_jaccard_top5"] = jacc

        report["consistency"][mname] = cons

    # Save JSON
    out_dir = pathlib.Path(args.artifacts_dir) / "interpretability"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "rq3_subgroups_report.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[OK] RQ3 report saved: {out_json}")

    # Save flat table CSV
    df = pd.DataFrame(flat_rows)
    # человекочитаемые имена моделей
    name_map = {
        "logreg": "LogisticRegression",
        "rf": "RandomForest",
        "xgb": "XGBoost",
        "xgb_tuned": "XGBoost_tuned",
        "mlp": "ANN"
    }
    df["Model"] = df["Model"].map(name_map).fillna(df["Model"])
    df.sort_values(["Group", "Level", "Model"], inplace=True)
    out_csv = out_dir / "rq3_subgroups_table.csv"
    df.to_csv(out_csv, index=False)
    print(f"[OK] Flat table saved: {out_csv}")

    # Pretty preview in console
    with pd.option_context('display.max_rows', 80, 'display.width', 140):
        print("\n=== RQ3 Subgroups (preview) ===")
        print(df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
