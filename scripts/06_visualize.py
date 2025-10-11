# -*- coding: utf-8 -*-
"""
06_visualize_compact.py

Read:
  - RQ1:  artifacts/metrics/rq1_summary.csv
  - RQ2:  artifacts/interpretability/rq2_top5_all_models.csv
          artifacts/interpretability/rq2_top5_pivot.csv
  - RQ3:  artifacts/interpretability/rq3_subgroups_table.csv

Build
1) RQ1 - grouped PR-AUC columns by models (VAL and TEST side by side).
2) RQ2 - the consensus-top of features (Top-10 on the average rank between models).
3) RQ3 is the average PR-AUC across all models for each subgroup (gender/year/employment level).

Optionally saves PNG (--save) to artifacts/figures.
"""

import argparse, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- helpers ----------
def _fallback(path: pathlib.Path) -> pathlib.Path:
    s = str(path)
    if path.exists():
        return path
    if "interpretability" in s:
        alt = pathlib.Path(s.replace("interpretability", "intepretability"))
        if alt.exists(): return alt
    if "intepretability" in s:
        alt = pathlib.Path(s.replace("intepretability", "interpretability"))
        if alt.exists(): return alt
    if "artifacts/metrics/" in s:
        alt = pathlib.Path(s.replace("artifacts/metrics/", "metrics/"))
        if alt.exists(): return alt
    if "metrics/" in s and not path.exists():
        alt = pathlib.Path(s.replace("metrics/", "artifacts/metrics/"))
        if alt.exists(): return alt
    return path

def _savefig(do_save: bool, outdir: pathlib.Path, fname: str):
    if do_save:
        outdir.mkdir(parents=True, exist_ok=True)
        plt.savefig(outdir / fname, dpi=160, bbox_inches="tight")

def _model_name(m):
    mapping = {
        "logreg": "Logistic Regression",
        "Logistic Regression": "Logistic Regression",
        "rf": "Random Forest",
        "Random Forest": "Random Forest",
        "xgb": "XGBoost",
        "XGBoost": "XGBoost",
        "xgb_tuned": "XGBoost (tuned)",
        "XGBoost (tuned)": "XGBoost (tuned)",
        "mlp": "Artificial Neural Network",
        "ANN": "Artificial Neural Network",
        "Artificial Neural Network": "Artificial Neural Network",
    }
    return mapping.get(str(m), str(m))


# ---------- RQ1----------
def figure_rq1(rq1_csv: pathlib.Path, save: bool, outdir: pathlib.Path):
    rq1_csv = _fallback(rq1_csv)
    df = pd.read_csv(rq1_csv)
    df["Model"] = df["Model"].map(_model_name)

    # order of models by TEST PR-AUC (if any), otherwise by VAL
    order = (df[df["Split"]=="TEST"]
             .sort_values("PR_AUC", ascending=False)["Model"].tolist())
    if not order:
        order = (df[df["Split"]=="VAL"]
                 .sort_values("PR_AUC", ascending=False)["Model"].tolist())
    order = list(dict.fromkeys(order))  

  # form matrix [x 2 splice models]
    models = order or sorted(df["Model"].unique())
    splits = ["VAL", "TEST"]
    mat = np.full((len(models), len(splits)), np.nan)
    prevalence = {s: np.nan for s in splits}
    for i, m in enumerate(models):
        for j, s in enumerate(splits):
            sub = df[(df["Model"] == m) & (df["Split"] == s)]
            if len(sub):
                mat[i, j] = float(sub["PR_AUC"].iloc[0])
                prevalence[s] = float(sub["Prevalence"].iloc[0])

    fig = plt.figure(figsize=(9, 5))
    x = np.arange(len(models))
    width = 0.38
    for j, s in enumerate(splits):
        xj = x + (j-0.5)*width
        plt.bar(xj, mat[:, j], width=width, label=s)
    # the baseline of prevalence (we take a test, if any)
    prev = prevalence.get("TEST") if np.isfinite(prevalence.get("TEST", np.nan)) else prevalence.get("VAL")
    if np.isfinite(prev):
        plt.axhline(prev, linestyle="--", linewidth=1.2, label=f"Prevalence={prev:.3f}")
    plt.xticks(x, models, rotation=20, ha="right")
    plt.ylabel("PR-AUC")
    plt.title("RQ1 — PR-AUC by model (VAL vs TEST)")
    plt.legend()
    plt.tight_layout()
    _savefig(save, outdir, "rq1_one_figure_pr_auc.png")
    return fig


# ---------- RQ2 ----------
def figure_rq2(pivot_csv: pathlib.Path, save: bool, outdir: pathlib.Path):
    pivot_csv = _fallback(pivot_csv)
    pv = pd.read_csv(pivot_csv)
    #take all rank columns
    rank_cols = [c for c in pv.columns if c.startswith("rank_")]
    if not rank_cols:
        raise SystemExit("[RQ2] in pivot no columns rank_*")
    pv["rank_mean"] = pv[rank_cols].mean(axis=1, skipna=True)
    pv = pv.sort_values("rank_mean").head(10)

    fig = plt.figure(figsize=(9, 5))
    x = np.arange(len(pv))
    plt.bar(x, pv["rank_mean"].values)
    plt.xticks(x, pv["feature"].astype(str).tolist(), rotation=25, ha="right")
    plt.ylabel("Mean rank across models (lower is better)")
    plt.title("RQ2 — Consensus Top-10 features (by mean rank)")
    plt.tight_layout()
    _savefig(save, outdir, "rq2_one_figure_consensus_top10.png")
    return fig


# ---------- RQ3 ----------
def figure_rq3(tbl_csv: pathlib.Path, save: bool, outdir: pathlib.Path):
    tbl_csv = _fallback(tbl_csv)
    df = pd.read_csv(tbl_csv)
    needed = {"Group", "Level", "Model", "PR_AUC", "Prevalence", "n"}
    if not needed.issubset(df.columns):
        raise SystemExit(f"[RQ3] Not enough columns in {tbl_csv}")

    # Average PR-AUC by model for each subgroup (Group Level)
    grp = (df.groupby(["Group", "Level"], as_index=False)
             .agg(PR_AUC_mean=("PR_AUC", "mean"),
                  n=("n", "first"),
                  Prevalence=("Prevalence", "first")))
    # we form the signatures of "gender:1", "year:2", ...
    grp["label"] = grp["Group"].astype(str) + ":" + grp["Level"].astype(str)

    fig = plt.figure(figsize=(11, 5))
    x = np.arange(len(grp))
    plt.bar(x, grp["PR_AUC_mean"].values)
    # horizontal line on average
    if "Prevalence" in grp.columns and np.any(np.isfinite(grp["Prevalence"].values)):
        plt.axhline(float(np.nanmean(grp["Prevalence"].values)), linestyle="--", linewidth=1.2, label="Prevalence (avg)")
        plt.legend()
    plt.xticks(x, grp["label"].tolist(), rotation=45, ha="right")
    plt.ylabel("PR-AUC (mean across models)")
    plt.title("RQ3 — PR-AUC by subgroup (mean across models)")
    plt.tight_layout()
    _savefig(save, outdir, "rq3_one_figure_subgroups_mean_pr_auc.png")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rq1_csv", default="artifacts/metrics/rq1_summary.csv")
    ap.add_argument("--rq2_pivot_csv", default="artifacts/interpretability/rq2_top5_pivot.csv")
    ap.add_argument("--rq3_tbl_csv", default="artifacts/interpretability/rq3_subgroups_table.csv")
    ap.add_argument("--save", action="store_true", help="save PNG in artifacts/figures")
    args = ap.parse_args()

    outdir = pathlib.Path("artifacts/figures")

    figure_rq1(pathlib.Path(args.rq1_csv), args.save, outdir)
    figure_rq2(pathlib.Path(args.rq2_pivot_csv), args.save, outdir)
    figure_rq3(pathlib.Path(args.rq3_tbl_csv), args.save, outdir)

    plt.show()


if __name__ == "__main__":
    main()
