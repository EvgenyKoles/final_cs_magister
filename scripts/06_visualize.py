# -*- coding: utf-8 -*-
"""
06_visualize_compact.py

Read:
  - RQ1:  artifacts/metrics/rq1_summary.csv
  - RQ2:  artifacts/interpretability/rq2_top5_all_models.csv (optional, not used here)
          artifacts/interpretability/rq2_top5_pivot.csv  (or .cs fallback)
  - RQ3:  artifacts/interpretability/rq3_subgroups_table.csv

Build
1) RQ1 - grouped PR-AUC columns by models (VAL and TEST side by side).
2) RQ2 - HEATMAP of feature ranks across models (lower rank = more important).
3) RQ3 - the average PR-AUC across all models for each subgroup (gender/year/employment level).

Optionally saves PNG (--save) to artifacts/figures.
"""

import argparse, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import MaxNLocator

# ---------- helpers ----------
def _fallback(path: pathlib.Path) -> pathlib.Path:
    """Small path fixer: supports minor typos and .cs -> .csv fallback."""
    s = str(path)
    # handle .cs accidentally saved instead of .csv
    if s.endswith(".cs") and not path.exists():
        alt = pathlib.Path(s + "v")  # .csv
        if alt.exists():
            return alt
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

def _prettify_feature(name: str) -> str:
    """Simple label cleanup for features."""
    if name is None:
        return "?"
    return str(name).replace("_", " ").strip()


# ---------- RQ1 ----------
def figure_rq1(rq1_csv: pathlib.Path, save: bool, outdir: pathlib.Path):
    rq1_csv = _fallback(rq1_csv)
    df = pd.read_csv(rq1_csv)
    df["Model"] = df["Model"].map(_model_name)

    # order models by TEST PR-AUC (if present), otherwise by VAL
    order = (df[df["Split"]=="TEST"]
             .sort_values("PR_AUC", ascending=False)["Model"].tolist())
    if not order:
        order = (df[df["Split"]=="VAL"]
                 .sort_values("PR_AUC", ascending=False)["Model"].tolist())
    order = list(dict.fromkeys(order))

    # form matrix [models x splits]
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

    # draw prevalence baseline (prefer TEST if available)
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


# ---------- RQ2 (heatmap only) ----------
def figure_rq2(pivot_csv: pathlib.Path, save: bool, outdir: pathlib.Path):
    """
    Heatmap of feature ranks across models 
    """
    pivot_csv = _fallback(pivot_csv)
    pv = pd.read_csv(pivot_csv)

    # detect feature column
    feat_col = None
    for c in ["feature", "feature_resolved", "predictor", "name"]:
        if c in pv.columns:
            feat_col = c; break
    if feat_col is None:
        for c in pv.columns:
            if pv[c].dtype == "object":
                feat_col = c; break
    if feat_col is None:
        feat_col = pv.columns[0]

    # detect rank columns
    rank_cols = [c for c in pv.columns if c.startswith("rank_")]
    if not rank_cols:
        bad = {feat_col, "mean_rank", "rank_mean", "std_rank", "n_models"}
        rank_cols = [c for c in pv.columns
                     if c not in bad and np.issubdtype(pv[c].dtype, np.number)]
    if not rank_cols:
        raise SystemExit("[RQ2] No rank columns found (expect 'rank_*' or numeric rank columns).")

    # sort features by mean rank (lower is better)
    pv["__mean_rank__"] = pv[rank_cols].mean(axis=1, skipna=True)
    pv = pv.sort_values("__mean_rank__").drop(columns=["__mean_rank__"])

    # pretty model names for x-axis
    pretty_cols = {c: _model_name(c.replace("rank_", "")) for c in rank_cols}

    # matrix for heatmap
    heat = pv[[feat_col] + rank_cols].copy()
    heat.columns = [feat_col] + [pretty_cols[c] for c in rank_cols]
    heat = heat.set_index(feat_col)

    # discrete colormap for ranks 1..5
    colors = ["#1a9850", "#66bd63", "#fdae61", "#f46d43", "#d73027"]  # ranks 1..5
    cmap = ListedColormap(colors)
    cmap.set_bad(color="#eeeeee", alpha=1.0)  # NaN
    cmap.set_over("#440154")                  # > 5 (colorbar)

    bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]   # bins for 1..5
    norm = BoundaryNorm(bounds, cmap.N)

    arr = heat.values.astype(float)

    fig = plt.figure(figsize=(1.8 + 1.2*heat.shape[1], 0.8 + 0.45*heat.shape[0]))
    # IMPORTANT: do NOT pass vmin/vmax together with norm
    im = plt.imshow(arr, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)

    # colorbar with discrete ticks 1..5
    cbar = plt.colorbar(im, extend="max")
    cbar.set_label("Rank (1 = best)")
    cbar.set_ticks([1, 2, 3, 4, 5])
    cbar.set_ticklabels(["1", "2", "3", "4", "5"])

    # axes ticks/labels
    plt.xticks(np.arange(heat.shape[1]), list(heat.columns), rotation=25, ha="right")
    plt.yticks(np.arange(heat.shape[0]), [str(i).replace("_", " ") for i in heat.index])

    # annotate ranks in cells
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = arr[i, j]
            if np.isfinite(v):
                txt = str(int(v)) if 0.5 < v < 99 else f"{v:.0f}"
                kw = {"fontweight": "bold"} if v <= 3 else {}
                plt.text(j, i, txt, ha="center", va="center", fontsize=9, **kw)

    # light grid
    ax = plt.gca()
    ax.set_facecolor("white")
    ax.set_xticks(np.arange(-0.5, heat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, heat.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.title("RQ2 — Feature rank heatmap across models (lower = better)")
    plt.tight_layout()
    _savefig(save, outdir, "rq2_rank_heatmap.png")
    return fig

# ---------- RQ3 ----------
def figure_rq3(tbl_csv: pathlib.Path, save: bool, outdir: pathlib.Path):
    """
    Panel view for RQ3:
      - One subplot per subgroup (gender / year / employment).
      - Bars: mean PR-AUC across models with SD error bars.
      - Diamonds: prevalence per level; dashed line = global prevalence (n-weighted).
      - Colored markers: per-model PR-AUC at each level (slight x-jitter).
      - 'n=...' above bars; best model label above each bar (only when bar height is finite).
    """
    tbl_csv = _fallback(tbl_csv)
    df = pd.read_csv(tbl_csv)
    needed = {"Group", "Level", "Model", "PR_AUC", "Prevalence", "n"}
    if not needed.issubset(df.columns):
        raise SystemExit(f"[RQ3] Not enough columns in {tbl_csv}")

    # Pretty model names for legend
    df["ModelPretty"] = df["Model"].map(_model_name)

    # Aggregate across models
    agg = (df.groupby(["Group", "Level"], as_index=False)
             .agg(PR_AUC_mean=("PR_AUC", "mean"),
                  PR_AUC_std=("PR_AUC", "std"),
                  n=("n", "first"),
                  Prevalence=("Prevalence", "first")))
    agg["PR_AUC_std"] = agg["PR_AUC_std"].fillna(0.0)

    # Global prevalence (n-weighted; fallback to nanmean)
    try:
        global_prev = float(np.average(agg["Prevalence"].values, weights=agg["n"].values))
    except Exception:
        global_prev = float(np.nanmean(agg["Prevalence"].values))

    # Panel order
    desired = ["gender", "year", "employment"]
    groups_present = [g for g in desired if g in agg["Group"].unique()]
    for g in agg["Group"].unique():
        if g not in groups_present:
            groups_present.append(g)
    G = len(groups_present)
    if G == 0:
        raise SystemExit("[RQ3] No subgroup groups found.")

    # Models & jitter for per-model points
    models = sorted(df["ModelPretty"].dropna().unique().tolist())
    jitters = np.linspace(-0.18, 0.18, num=max(1, len(models)))

    fig, axes = plt.subplots(1, G, figsize=(4.6*G + 2, 5.0), sharey=True)
    if G == 1:
        axes = [axes]

    # Safe ymax with NaN-tolerant maxima
    try:
        mm = float(np.nanmax(agg["PR_AUC_mean"].values))
    except Exception:
        mm = 0.2
    try:
        ss = float(np.nanmax(agg["PR_AUC_std"].values))
    except Exception:
        ss = 0.0
    ymax = float(min(1.0, max(0.05, mm + ss + 0.1)))

    for ax, g in zip(axes, groups_present):
        sub = agg[agg["Group"] == g].copy()

        # Natural level ordering
        sub["_lvl_num"] = pd.to_numeric(sub["Level"], errors="coerce")
        if sub["_lvl_num"].notna().all():
            sub = sub.sort_values("_lvl_num")
            xlabels = [str(int(v)) if float(v).is_integer() else str(v) for v in sub["_lvl_num"]]
        else:
            sub = sub.sort_values("Level")
            xlabels = sub["Level"].astype(str).tolist()
        sub = sub.drop(columns=["_lvl_num"])

        x = np.arange(len(sub))

        # Mean ± SD bars
        ax.bar(x, sub["PR_AUC_mean"].values,
               yerr=sub["PR_AUC_std"].values, capsize=3,
               label=("PR-AUC mean ± SD" if ax is axes[0] else None))

        # Per-level prevalence
        ax.scatter(x, sub["Prevalence"].values, marker="D", s=40, color="black", alpha=0.75,
                   label=("Prevalence (per level)" if ax is axes[0] else None))

        # Global prevalence
        if np.isfinite(global_prev):
            ax.axhline(global_prev, ls="--", lw=1.2, color="tab:blue",
                       label=("Prevalence (global)" if ax is axes[0] else None), alpha=0.7)

        # Per-model points (align by level; small x-jitter)
        sub_models = df[df["Group"] == g].copy()
        sub_models["_order"] = pd.Categorical(sub_models["Level"], categories=sub["Level"], ordered=True)
        sub_models.sort_values(["_order", "ModelPretty"], inplace=True)

        for j, m in enumerate(models):
            sm = sub_models[sub_models["ModelPretty"] == m]
            yvals = []
            for lvl in sub["Level"]:
                row = sm[sm["Level"] == lvl]
                yvals.append(float(row["PR_AUC"].mean()) if len(row) else np.nan)
            ax.plot(x + jitters[j], yvals, marker="o", linestyle="None", ms=5,
                    label=(m if ax is axes[0] else None), alpha=0.9)

        # --- Annotations (only when bar height is finite) ---
        for xi, yi, ni, lvl in zip(x, sub["PR_AUC_mean"].values, sub["n"].values, sub["Level"].values):
            if np.isfinite(yi):
                ax.text(xi, yi + 0.015, f"n={int(ni)}", ha="center", va="bottom", fontsize=9)
                rows = sub_models[sub_models["Level"] == lvl]
                vals = rows["PR_AUC"].astype(float)
                if not rows.empty and vals.notna().any():
                    idx = vals.fillna(-np.inf).idxmax()   # safe with NaNs
                    best = str(rows.loc[idx, "ModelPretty"])
                    ax.text(xi, yi + 0.06, best, ha="center", va="bottom", fontsize=9, color="dimgray")
            else:
                # Skip annotations to avoid "posx/posy should be finite values"
                continue

        ax.set_title(g)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=0)
        ax.set_ylim(0.0, ymax)
        ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.6)

    axes[0].set_ylabel("PR-AUC (mean across models)")
    fig.suptitle("RQ3 — PR-AUC across demographic subgroups (mean ± SD across models)", y=0.98)
    axes[0].legend(loc="upper left", frameon=True)  # single legend

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    _savefig(save, outdir, "rq3_subgroups_pr_auc_panels.png")
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
    figure_rq2(pathlib.Path(args.rq2_pivot_csv), args.save, outdir)  # heatmap only
    figure_rq3(pathlib.Path(args.rq3_tbl_csv), args.save, outdir)

    plt.show()


if __name__ == "__main__":
    main()
