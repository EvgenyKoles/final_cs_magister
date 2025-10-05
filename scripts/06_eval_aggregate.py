# -*- coding: utf-8 -*-
"""
06_eval_aggregate.py
Единый агрегатор результатов под RQ1–RQ3 и H1–H3.

Собирает из артефактов:
- RQ1: таблицу PR-AUC (test) по моделям
- RQ2/Hypotheses: свод по ключевым предикторам (LR β, p-value, знак, ранги важностей XGB/RF)
- RQ3: PR-AUC (test) по подгруппам (sex, year, job), если доступны метаданные подгрупп

Ожидаемые артефакты (по умолчанию):
- metrics:
    artifacts/metrics/logreg_eval.json
    artifacts/metrics/rf_eval.json              [если RF запускался]
    artifacts/metrics/mlp_eval.json             [если MLP запускался]
    artifacts/metrics/xgb_tuned_eval.json       [после 05_tune_xgb.py]
- preds:
    artifacts/preds/logreg_test.csv             (или logreg_calibrated_test.csv)
    artifacts/preds/rf_test.csv                 [если RF сохранён с --save_probas]
    artifacts/preds/mlp_test.csv                [если MLP сохранён с --save_probas]
    artifacts/preds/xgb_tuned_test.csv          [после 05 с --save_probas]
- interpretability:
    artifacts/interpretability/logreg_coefs.csv (из 03, statsmodels или fallback)
    artifacts/interpretability/rf_importances.csv      [из 04, если был RF]
    artifacts/interpretability/xgb_tuned_importances.csv [из 05]
- subgroups (опционально для RQ3):
    artifacts/metadata/subgroups_test.csv  или  data/processed/subgroups_test.csv

Запуск:
python scripts/06_eval_aggregate.py --processed_dir data/processed --artifacts_dir artifacts --alpha 0.05 --topk 10
"""

import argparse
import json
import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


# ====== EXPLICIT FEATURE MAPPING FOR RQ2/H ======
# Колонки исходного датасета: cesd, stai_t, stud_h, amsp, mbi_ea,
# а также категориальные группы: health_* (self-rated health), job_* (employment)
FEATURE_MAP: Dict[str, Dict[str, str]] = {
    # type: 'single' -> точное имя; 'group' -> префикс
    "CES-D":                  {"type": "single", "name": "cesd"},
    "STAI-T":                 {"type": "single", "name": "stai_t"},
    "Self-rated health":      {"type": "group",  "prefix": "health"},
    "Workload (study hours)": {"type": "single", "name": "stud_h"},
    "Employment status":      {"type": "group",  "prefix": "job"},
    "AMSP (motivation)":      {"type": "single", "name": "amsp"},
    "MBI-Efficacy":           {"type": "single", "name": "mbi_ea"},
}


# ====== UTILS: IO ======
def read_json(path: pathlib.Path) -> Optional[dict]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
    return None


def read_preds(path: pathlib.Path) -> Optional[pd.DataFrame]:
    if path.exists():
        try:
            df = pd.read_csv(path)
            if {"y", "proba"}.issubset(df.columns):
                return df[["y", "proba"]].copy()
        except Exception:
            return None
    return None


def find_first_existing(paths: List[pathlib.Path]) -> Optional[pathlib.Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def load_subgroups(processed_dir: pathlib.Path, artifacts_dir: pathlib.Path) -> Optional[pd.DataFrame]:
    candidates = [
        artifacts_dir / "metadata" / "subgroups_test.csv",
        artifacts_dir / "subgroups_test.csv",
        processed_dir / "subgroups_test.csv",
    ]
    p = find_first_existing(candidates)
    if p is None:
        return None
    try:
        df = pd.read_csv(p)
        return df
    except Exception:
        return None


# ====== RQ1: Model PR-AUC on TEST ======
def rq1_table(artifacts_dir: pathlib.Path) -> pd.DataFrame:
    """
    Собрать PR-AUC(test) по доступным моделям.
    Источник приоритета: preds/*.csv (пересчёт), иначе metrics/*.json.
    """
    models = ["xgb_tuned", "mlp", "logreg", "rf"]  # порядок вывода не влияет; итог сортируется по метрике
    out_rows = []

    for m in models:
        # 1) попробуем preds
        if m == "logreg":
            pred_paths = [
                artifacts_dir / "preds" / "logreg_test.csv",
                artifacts_dir / "preds" / "logreg_calibrated_test.csv",
            ]
        else:
            pred_paths = [artifacts_dir / "preds" / f"{m}_test.csv"]

        preds_file = find_first_existing(pred_paths)
        pr_auc = None

        if preds_file is not None:
            df = read_preds(preds_file)
            if df is not None and df["y"].nunique() > 1:
                pr_auc = float(average_precision_score(df["y"].values, df["proba"].values))

        # 2) если не удалось — возьмём из metrics
        if pr_auc is None:
            mfile = artifacts_dir / "metrics" / f"{m}_eval.json"
            js = read_json(mfile)
            if js and "test" in js and "PR_AUC" in js["test"]:
                pr_auc = float(js["test"]["PR_AUC"])

        if pr_auc is not None:
            out_rows.append({"model": m, "PR_AUC_test": pr_auc})

    df_out = pd.DataFrame(out_rows)
    if not df_out.empty:
        df_out = df_out.sort_values("PR_AUC_test", ascending=False).reset_index(drop=True)
    return df_out


# ====== RQ2/H: Load model interpretability artifacts ======
def load_lr_coefs(artifacts_dir: pathlib.Path) -> Optional[pd.DataFrame]:
    p = artifacts_dir / "interpretability" / "logreg_coefs.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    # ожидаемые колонки: feature, beta, SE, z, p_value, CI_low, CI_high
    if "feature" not in df.columns or "beta" not in df.columns:
        return None
    df["feature"] = df["feature"].astype(str)
    return df


def load_importances(artifacts_dir: pathlib.Path, tag: str) -> Optional[pd.DataFrame]:
    """
    tag in {'xgb_tuned','rf'} -> читаем соответствующие csv с важностями.
    Нормализуем к колонкам: __fname__, __imp__
    """
    if tag == "xgb_tuned":
        p = artifacts_dir / "interpretability" / "xgb_tuned_importances.csv"
    elif tag == "rf":
        p = artifacts_dir / "interpretability" / "rf_importances.csv"
    else:
        return None

    if not p.exists():
        return None

    df = pd.read_csv(p)
    # имя фичи
    if "feature_name" in df.columns:
        df["__fname__"] = df["feature_name"].astype(str)
    elif "feature" in df.columns:
        df["__fname__"] = df["feature"].astype(str)
    else:
        return None
    # величина важности
    if "importance_gain" in df.columns:
        df["__imp__"] = df["importance_gain"].astype(float)
    elif "importance" in df.columns:
        df["__imp__"] = df["importance"].astype(float)
    else:
        return None

    df = df[["__fname__", "__imp__"]].copy()
    df.sort_values("__imp__", ascending=False, inplace=True, ignore_index=True)
    return df


def best_group_member_by_lr(lr_df: pd.DataFrame, prefix: str) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """
    Вернуть (feature_name, beta, p_value) для лучшего члена группы (prefix или prefix_*)
    Критерий: min p_value; при NaN p -> max |beta|.
    """
    cand = lr_df[lr_df["feature"].str.startswith(prefix)].copy()
    if cand.empty:
        cand = lr_df[lr_df["feature"].eq(prefix)].copy()
    if cand.empty:
        return None, None, None

    if "p_value" in cand.columns and cand["p_value"].notna().any():
        cand_p = cand.dropna(subset=["p_value"]).copy()
        if not cand_p.empty:
            row = cand_p.sort_values("p_value", ascending=True).iloc[0]
            return row["feature"], float(row["beta"]), float(row["p_value"])

    cand["__absb__"] = cand["beta"].abs()
    row = cand.sort_values("__absb__", ascending=False).iloc[0]
    return row["feature"], float(row["beta"]), float(row.get("p_value", np.nan))


def rank_from_importances_single(df_imp: Optional[pd.DataFrame], name: str) -> Optional[int]:
    """Ранг (1-based) одной фичи по важности."""
    if df_imp is None or df_imp.empty:
        return None
    ranks = {df_imp.loc[i, "__fname__"]: i + 1 for i in range(len(df_imp))}
    r = ranks.get(name, None)
    return int(r) if r is not None else None


def rank_from_importances_group(df_imp: Optional[pd.DataFrame], prefix: str) -> Optional[int]:
    """Лучший (минимальный) ранг среди всех имён, начинающихся на prefix."""
    if df_imp is None or df_imp.empty:
        return None
    best = None
    for i, row in df_imp.iterrows():
        fname = str(row["__fname__"])
        if fname.startswith(prefix):
            r = i + 1  # 1-based
            if best is None or r < best:
                best = r
    return best


def rq2_and_hypotheses_tables_explicit(
    artifacts_dir: pathlib.Path, alpha: float = 0.05, topk: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Возвращает:
      - features_summary: по ключевым предикторам — β (LR), p-value, знак, ранги в XGB/RF
      - hypotheses_summary: статус H1/H2/H3
    """
    lr_df = load_lr_coefs(artifacts_dir)
    xgb_imp = load_importances(artifacts_dir, "xgb_tuned")
    rf_imp = load_importances(artifacts_dir, "rf")

    rows = []
    for predictor, spec in FEATURE_MAP.items():
        if spec["type"] == "single":
            name = spec["name"]
            beta = pval = sign = None
            if lr_df is not None:
                m = lr_df["feature"].astype(str) == name
                if m.any():
                    row = lr_df[m].iloc[0]
                    beta = float(row["beta"])
                    pval = float(row["p_value"]) if "p_value" in row else None
                    sign = "+" if beta > 0 else ("−" if beta < 0 else "0")
            xgb_rank = rank_from_importances_single(xgb_imp, name)
            rf_rank = rank_from_importances_single(rf_imp, name)

            rows.append(
                {
                    "predictor": predictor,
                    "feature_resolved": (name if (beta is not None or xgb_rank or rf_rank) else "NOT_PRESENT"),
                    "LR_beta": beta,
                    "LR_p": pval,
                    "LR_sign": sign,
                    "XGB_rank": xgb_rank,
                    "RF_rank": rf_rank,
                }
            )

        elif spec["type"] == "group":
            pref = spec["prefix"]
            name = beta = pval = sign = None
            if lr_df is not None:
                name, beta, pval = best_group_member_by_lr(lr_df, pref)
                if beta is not None:
                    sign = "+" if beta > 0 else ("−" if beta < 0 else "0")

            xgb_rank = rank_from_importances_group(xgb_imp, pref)
            rf_rank = rank_from_importances_group(rf_imp, pref)

            rows.append(
                {
                    "predictor": predictor,
                    "feature_resolved": (name if name is not None else pref + "_*"),
                    "LR_beta": beta,
                    "LR_p": pval,
                    "LR_sign": sign,
                    "XGB_rank": xgb_rank,
                    "RF_rank": rf_rank,
                }
            )

        else:
            # на всякий случай
            rows.append(
                {
                    "predictor": predictor,
                    "feature_resolved": "INVALID_SPEC",
                    "LR_beta": None,
                    "LR_p": None,
                    "LR_sign": None,
                    "XGB_rank": None,
                    "RF_rank": None,
                }
            )

    features_summary = pd.DataFrame(rows)

    # ===== Hypotheses status =====
    def sig_pos(key: str) -> bool:
        r = features_summary.loc[features_summary["predictor"] == key]
        if r.empty:
            return False
        b, p = r["LR_beta"].values[0], r["LR_p"].values[0]
        return (p is not None and not pd.isna(p) and p < alpha) and (b is not None and not pd.isna(b) and b > 0)

    def sig_neg(key: str) -> bool:
        r = features_summary.loc[features_summary["predictor"] == key]
        if r.empty:
            return False
        b, p = r["LR_beta"].values[0], r["LR_p"].values[0]
        return (p is not None and not pd.isna(p) and p < alpha) and (b is not None and not pd.isna(b) and b < 0)

    def in_topk(key: str, col: str) -> bool:
        r = features_summary.loc[features_summary["predictor"] == key]
        if r.empty:
            return False
        v = r[col].values[0]
        return (v is not None) and (not pd.isna(v)) and (int(v) <= int(topk))

    # H1: CES-D(+) и STAI-T(+) значимы в LR (p<alpha); доп. поддержка важностями XGB/RF (top-k)
    h1_ces, h1_stai = sig_pos("CES-D"), sig_pos("STAI-T")
    h1_ces_sup = in_topk("CES-D", "XGB_rank") or in_topk("CES-D", "RF_rank")
    h1_stai_sup = in_topk("STAI-T", "XGB_rank") or in_topk("STAI-T", "RF_rank")
    if h1_ces and h1_stai and (h1_ces_sup or h1_stai_sup):
        h1_status = "Supported"
    elif (h1_ces or h1_stai):
        h1_status = "Partially"
    else:
        h1_status = "Not"

    # H2: Workload(+) и Employment(+) — как выше
    h2_work, h2_emp = sig_pos("Workload (study hours)"), sig_pos("Employment status")
    h2_work_sup = in_topk("Workload (study hours)", "XGB_rank") or in_topk("Workload (study hours)", "RF_rank")
    h2_emp_sup = in_topk("Employment status", "XGB_rank") or in_topk("Employment status", "RF_rank")
    if h2_work and h2_emp and (h2_work_sup or h2_emp_sup):
        h2_status = "Supported"
    elif (h2_work or h2_emp):
        h2_status = "Partially"
    else:
        h2_status = "Not"

    # H3: AMSP(−) и MBI-Efficacy(−) — значимо в LR и подтверждено важностями (>= 2 моделей всего)
    h3_amsp, h3_effic = sig_neg("AMSP (motivation)"), sig_neg("MBI-Efficacy")
    h3_amsp_sup = in_topk("AMSP (motivation)", "XGB_rank") or in_topk("AMSP (motivation)", "RF_rank")
    h3_effic_sup = in_topk("MBI-Efficacy", "XGB_rank") or in_topk("MBI-Efficacy", "RF_rank")
    if (h3_amsp and h3_amsp_sup) and (h3_effic and h3_effic_sup):
        h3_status = "Supported"
    elif (h3_amsp and h3_amsp_sup) or (h3_effic and h3_effic_sup):
        h3_status = "Partially"
    else:
        h3_status = "Not"

    hypotheses_summary = pd.DataFrame(
        [
            {
                "Hypothesis": "H1 (CES-D+, STAI-T+)",
                "alpha": alpha,
                "topk_for_importance": topk,
                "Status": h1_status,
                "Evidence": f"CES-D sig+={h1_ces}, topk={h1_ces_sup}; STAI-T sig+={h1_stai}, topk={h1_stai_sup}",
            },
            {
                "Hypothesis": "H2 (Workload+, Employment+)",
                "alpha": alpha,
                "topk_for_importance": topk,
                "Status": h2_status,
                "Evidence": f"Workload sig+={h2_work}, topk={h2_work_sup}; Employment sig+={h2_emp}, topk={h2_emp_sup}",
            },
            {
                "Hypothesis": "H3 (AMSP−, MBI-Efficacy−)",
                "alpha": alpha,
                "topk_for_importance": topk,
                "Status": h3_status,
                "Evidence": f"AMSP sig−={h3_amsp}, topk={h3_amsp_sup}; Efficacy sig−={h3_effic}, topk={h3_effic_sup}",
            },
        ]
    )

    return features_summary, hypotheses_summary


# ====== RQ3: Subgroup PR-AUC on TEST ======
def rq3_subgroups_table(artifacts_dir: pathlib.Path, processed_dir: pathlib.Path) -> Optional[pd.DataFrame]:
    """
    Строит PR-AUC по подгруппам (sex, year, job) на тесте, если есть subgroups_test.csv.
    Подразумевается, что порядок строк совпадает с preds_*_test.csv.
    """
    sub = load_subgroups(processed_dir, artifacts_dir)
    if sub is None:
        return None

    # Нормируем имена ожидаемых колонок
    rename_map = {}
    if "sex" in sub.columns:
        rename_map["sex"] = "sex"
    elif "gender" in sub.columns:
        rename_map["gender"] = "sex"
    if "year" in sub.columns:
        rename_map["year"] = "year"
    if "job" in sub.columns:
        rename_map["job"] = "job"
    elif "employment" in sub.columns:
        rename_map["employment"] = "job"

    sub = sub.rename(columns=rename_map)
    keep = [c for c in ["sex", "year", "job"] if c in sub.columns]
    if not keep:
        return None
    sub = sub[keep].copy()

    # Подгрузим предсказания
    models = ["xgb_tuned", "mlp", "logreg", "rf"]
    model_preds = {}

    for m in models:
        if m == "logreg":
            pred_paths = [
                artifacts_dir / "preds" / "logreg_test.csv",
                artifacts_dir / "preds" / "logreg_calibrated_test.csv",
            ]
        else:
            pred_paths = [artifacts_dir / "preds" / f"{m}_test.csv"]

        pf = find_first_existing(pred_paths)
        if pf is None:
            continue
        dfp = read_preds(pf)
        if dfp is None:
            continue
        if len(dfp) != len(sub):
            # длины не совпали => надёжно сопоставить нельзя
            continue

        model_preds[m] = dfp

    rows = []
    for m, dfp in model_preds.items():
        for col in sub.columns:
            for val, g in sub.groupby(col):
                idx = g.index.values
                y = dfp.loc[idx, "y"].values
                p = dfp.loc[idx, "proba"].values
                if len(np.unique(y)) < 2:
                    pr = np.nan
                else:
                    pr = float(average_precision_score(y, p))
                rows.append({"model": m, "subgroup": col, "value": val, "n": int(len(g)), "PR_AUC_test": pr})

    if not rows:
        return None
    return pd.DataFrame(rows).sort_values(["model", "subgroup", "value"]).reset_index(drop=True)


# ====== MAIN ======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--artifacts_dir", default="artifacts")
    ap.add_argument("--alpha", type=float, default=0.05, help="significance for LR p-values")
    ap.add_argument("--topk", type=int, default=10, help="top-K threshold for feature importance support (XGB/RF)")
    args = ap.parse_args()

    processed_dir = pathlib.Path(args.processed_dir)
    artifacts_dir = pathlib.Path(args.artifacts_dir)

    (artifacts_dir / "metrics").mkdir(parents=True, exist_ok=True)

    # ---- RQ1
    rq1 = rq1_table(artifacts_dir)
    rq1_path = artifacts_dir / "metrics" / "aggregate_rq1_model_pr_auc_test.csv"
    rq1.to_csv(rq1_path, index=False)
    print(f"[OK] RQ1 table saved: {rq1_path}")

    # ---- RQ2 + Hypotheses
    feat_sum, hyp_sum = rq2_and_hypotheses_tables_explicit(
        artifacts_dir, alpha=args.alpha, topk=args.topk
    )
    rq2_path = artifacts_dir / "metrics" / "aggregate_rq2_features_summary.csv"
    hyp_path = artifacts_dir / "metrics" / "aggregate_hypotheses_summary.csv"
    feat_sum.to_csv(rq2_path, index=False)
    hyp_sum.to_csv(hyp_path, index=False)
    print(f"[OK] RQ2 features summary saved: {rq2_path}")
    print(f"[OK] Hypotheses summary saved:  {hyp_path}")

    # ---- RQ3 (optional)
    rq3 = rq3_subgroups_table(artifacts_dir, processed_dir)
    if rq3 is not None:
        rq3_path = artifacts_dir / "metrics" / "aggregate_rq3_subgroup_pr_auc_test.csv"
        rq3.to_csv(rq3_path, index=False)
        print(f"[OK] RQ3 subgroup table saved: {rq3_path}")
    else:
        print("[WARN] RQ3 skipped: no usable subgroups_test.csv or length mismatch with preds.")

    print("\nDone.")

if __name__ == "__main__":
    main()
