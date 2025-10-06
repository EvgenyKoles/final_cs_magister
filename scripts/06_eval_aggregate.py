# -*- coding: utf-8 -*-
"""
06_eval_aggregate.py  (RQ1 расширен; RQ2 исправлена инференция LR — Шаг A, без statsmodels)

Новое (Шаг A, без внешних зависимостей):
- Встроенная инференция логит-модели (IRLS) + робастная ковариация (sandwich, HC3-подобная).
- Drop-first для категорий (sex, job, health, glang) выбирая референт по наибольшей частоте.
- Артефакты:
    artifacts/interpretability/logreg_coefs_corrected.csv
    artifacts/interpretability/logreg_reference_levels.csv
    artifacts/interpretability/logreg_factor_tests.csv

RQ1:
- Выбор клинического порога на валидации (f1, f2, target_recall)
- Оценка на тесте: Recall/Precision/Fβ @thr + PR-AUC(test)
- Таблицы: aggregate_rq1_model_pr_auc_test.csv, aggregate_rq1_threshold_eval.csv

RQ3:
- PR-AUC(test) по подгруппам при наличии subgroups_test.csv.
"""

import argparse, json, pathlib, warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import average_precision_score, precision_recall_curve


# ========= УТИЛИТЫ =========
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
            if {"y","proba"}.issubset(df.columns):
                return df[["y","proba"]].copy()
        except Exception:
            return None
    return None

def find_first_existing(paths: List[pathlib.Path]) -> Optional[pathlib.Path]:
    for p in paths:
        if p.exists():
            return p
    return None


# ========= RQ1: базовая PR-AUC (как раньше) =========
def rq1_table_pr_auc_only(artifacts_dir: pathlib.Path) -> pd.DataFrame:
    models = ["xgb_tuned", "mlp", "logreg", "rf"]
    out_rows = []
    for m in models:
        pred_paths = (
            [artifacts_dir/"preds"/"logreg_test.csv", artifacts_dir/"preds"/"logreg_calibrated_test.csv"]
            if m == "logreg" else
            [artifacts_dir/"preds"/f"{m}_test.csv"]
        )
        preds_file = find_first_existing(pred_paths)
        pr_auc = None
        if preds_file is not None:
            df = read_preds(preds_file)
            if df is not None and df["y"].nunique() > 1:
                pr_auc = float(average_precision_score(df["y"].values, df["proba"].values))
        if pr_auc is None:
            mfile = artifacts_dir/"metrics"/f"{m}_eval.json"
            js = read_json(mfile)
            if js and "test" in js and "PR_AUC" in js["test"]:
                pr_auc = float(js["test"]["PR_AUC"])
        if pr_auc is not None:
            out_rows.append({"model": m, "PR_AUC_test": pr_auc})
    return pd.DataFrame(out_rows).sort_values("PR_AUC_test", ascending=False).reset_index(drop=True)


# ========= RQ1: клинический порог =========
def choose_threshold_on_val(y_val: np.ndarray,
                            p_val: np.ndarray,
                            method: str = "f1",
                            beta: float = 1.0,
                            target_recall: float = 0.80) -> Tuple[float, Dict[str, float]]:
    precision, recall, thresholds = precision_recall_curve(y_val, p_val)
    if thresholds.size == 0:
        thr = 0.5
        yhat = (p_val >= thr).astype(int)
        tp = float(((yhat == 1) & (y_val == 1)).sum())
        fp = float(((yhat == 1) & (y_val == 0)).sum())
        fn = float(((yhat == 0) & (y_val == 1)).sum())
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        fbeta = (1+beta**2)*prec*rec / (beta**2*prec + rec + 1e-12)
        return thr, {"val_precision": float(prec), "val_recall": float(rec), "val_Fbeta": float(fbeta)}

    prec_arr  = precision[:-1]
    rec_arr   = recall[:-1]
    thr_arr   = thresholds

    def fbeta_arr(prec, rec, b):
        return (1.0 + b**2) * (prec * rec) / (b**2 * prec + rec + 1e-12)

    if method.lower() == "f2":
        beta = 2.0
        f = fbeta_arr(prec_arr, rec_arr, beta)
        i = int(np.nanargmax(f))
        thr = float(thr_arr[i])
        return thr, {"val_precision": float(prec_arr[i]), "val_recall": float(rec_arr[i]), "val_Fbeta": float(f[i])}

    if method.lower() == "f1":
        f = fbeta_arr(prec_arr, rec_arr, beta)
        i = int(np.nanargmax(f))
        thr = float(thr_arr[i])
        return thr, {"val_precision": float(prec_arr[i]), "val_recall": float(rec_arr[i]), "val_Fbeta": float(f[i])}

    if method.lower() == "target_recall":
        mask = rec_arr >= float(target_recall)
        if mask.any():
            f1_masked = fbeta_arr(prec_arr[mask], rec_arr[mask], 1.0)
            j = int(np.nanargmax(f1_masked))
            i = int(np.where(mask)[0][j])
        else:
            i = int(np.nanargmin(thr_arr))
        thr = float(thr_arr[i])
        f1_i = fbeta_arr(prec_arr[i:i+1], rec_arr[i:i+1], 1.0)[0]
        return thr, {"val_precision": float(prec_arr[i]), "val_recall": float(rec_arr[i]), "val_Fbeta": float(f1_i)}

    # fallback -> F1
    f = fbeta_arr(prec_arr, rec_arr, 1.0)
    i = int(np.nanargmax(f))
    thr = float(thr_arr[i])
    return thr, {"val_precision": float(prec_arr[i]), "val_recall": float(rec_arr[i]), "val_Fbeta": float(f[i])}


def eval_on_test_at_thr(y_test: np.ndarray,
                        p_test: np.ndarray,
                        thr: float,
                        beta: float = 1.0) -> Dict[str, float]:
    yhat = (p_test >= thr).astype(int)
    tp = float(((yhat == 1) & (y_test == 1)).sum())
    fp = float(((yhat == 1) & (y_test == 0)).sum())
    fn = float(((yhat == 0) & (y_test == 1)).sum())
    prec = tp / (tp + fp + 1e-12)
    rec  = tp / (tp + fn + 1e-12)
    fbeta = (1.0 + beta**2) * prec * rec / (beta**2 * prec + rec + 1e-12)
    pr_auc = float(average_precision_score(y_test, p_test)) if np.unique(y_test).size > 1 else float("nan")
    return {"test_precision": float(prec), "test_recall": float(rec), "test_Fbeta": float(fbeta), "PR_AUC_test": pr_auc}


def rq1_table_full(artifacts_dir: pathlib.Path,
                   method: str = "f2",
                   beta: float = 2.0,
                   target_recall: float = 0.80) -> pd.DataFrame:
    models = ["xgb_tuned", "mlp", "logreg", "rf"]
    rows = []
    for m in models:
        if m == "logreg":
            val_candidates  = [artifacts_dir/"preds"/"logreg_val.csv", artifacts_dir/"preds"/"logreg_calibrated_val.csv"]
            test_candidates = [artifacts_dir/"preds"/"logreg_test.csv", artifacts_dir/"preds"/"logreg_calibrated_test.csv"]
        else:
            val_candidates  = [artifacts_dir/"preds"/f"{m}_val.csv"]
            test_candidates = [artifacts_dir/"preds"/f"{m}_test.csv"]

        p_val_path  = find_first_existing(val_candidates)
        p_test_path = find_first_existing(test_candidates)
        if p_val_path is None or p_test_path is None:
            continue

        df_val  = read_preds(p_val_path)
        df_test = read_preds(p_test_path)
        if df_val is None or df_test is None:
            continue

        yv, pv = df_val["y"].values.astype(int), df_val["proba"].values.astype(float)
        yt, pt = df_test["y"].values.astype(int), df_test["proba"].values.astype(float)

        thr, val_info = choose_threshold_on_val(yv, pv, method=method, beta=beta, target_recall=target_recall)
        test_info = eval_on_test_at_thr(yt, pt, thr=thr, beta=(2.0 if method=="f2" else (1.0 if method=="f1" else 1.0)))

        rows.append({
            "model": m, "method": method, "beta": float(beta), "target_recall": float(target_recall),
            "thr": float(thr),
            "val_precision": val_info["val_precision"], "val_recall": val_info["val_recall"], "val_Fbeta": val_info["val_Fbeta"],
            "test_precision": test_info["test_precision"], "test_recall": test_info["test_recall"], "test_Fbeta": test_info["test_Fbeta"],
            "PR_AUC_test": test_info["PR_AUC_test"],
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["test_recall","PR_AUC_test","test_Fbeta"], ascending=[False, False, False]).reset_index(drop=True)
    return df


# ========= ШАГ A: ИНФЕРЕНЦИЯ LOGIT (IRLS + ROBUST) =========
# Нормальное CDF и квантиль z≈1.96 (без scipy)
from math import erf, sqrt

def _phi(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))

Z975 = 1.959963984540054  # quantile for 97.5%

def _load_train_matrix(processed_dir: pathlib.Path) -> Tuple[pd.DataFrame, pd.Series]:
    X_path = processed_dir/"X_train.csv"
    y_path = processed_dir/"y_train.csv"
    fn_path = processed_dir/"feature_names.csv"
    if not X_path.exists() or not y_path.exists() or not fn_path.exists():
        raise FileNotFoundError("Не найдены X_train.csv / y_train.csv / feature_names.csv")
    X = np.loadtxt(X_path, delimiter=",")
    feat_names = pd.read_csv(fn_path, header=None).iloc[:,0].astype(str).tolist()
    if X.shape[1] != len(feat_names):
        raise ValueError(f"Размерность X_train ({X.shape[1]}) != числу имён признаков ({len(feat_names)})")
    Xdf = pd.DataFrame(X, columns=feat_names)
    y = pd.read_csv(y_path).iloc[:,0].astype(int)
    if len(y) != len(Xdf):
        raise ValueError("Длины X_train и y_train не совпадают")
    return Xdf, y

def _group_categorical_columns(columns: List[str], cat_prefixes: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {p: [] for p in cat_prefixes}
    for c in columns:
        for p in cat_prefixes:
            if c.startswith(p + "_"):
                groups[p].append(c)
                break
    return {k: v for k, v in groups.items() if v}

def _pick_reference_by_frequency(Xdf: pd.DataFrame, cols: List[str]) -> str:
    if len(cols) == 1:
        return cols[0]
    freq = {c: float(np.nanmean(Xdf[c].values)) for c in cols}
    return sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[0][0]

def _sigmoid(z):
    # численно устойчивая логистическая функция
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))

def _irls_logit(X: np.ndarray, y: np.ndarray, max_iter=200, tol=1e-7, ridge=1e-8):
    """
    Итерационно-взвешенные наименьшие квадраты (IRLS) для логистической регрессии.
    Возвращает beta, mu, H (информация Фишера), hat (диагональ hat-матрицы).
    """
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(max_iter):
        eta = X @ beta
        mu = _sigmoid(eta)
        W = mu * (1.0 - mu)  # веса
        # защита от нулевых весов
        W = np.clip(W, 1e-8, None)
        # рабочая зависимая переменная
        z = eta + (y - mu) / W
        # матрицы для WLS
        WX = X * W[:, None]
        H = X.T @ WX
        # регуляризация (tiny ridge) для устойчивости инверсии
        H = H + ridge * np.eye(p)
        b = X.T @ (W * z)
        beta_new = np.linalg.solve(H, b)
        # критерий сходимости
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    # Hat-матрица: H = W^{1/2} X (X^T W X)^(-1) X^T W^{1/2}
    W_sqrt = np.sqrt(W)
    Xw = X * W_sqrt[:, None]
    Hinv = np.linalg.pinv(H)
    # diag(Hat) без явной матрицы: h_i = row_i(Xw) @ Hinv @ row_i(Xw)^T
    hat = np.einsum('ij,jk,ik->i', Xw, Hinv, Xw)

    return beta, mu, H, hat

def _robust_covariance_HC3(X: np.ndarray, y: np.ndarray, mu: np.ndarray, H: np.ndarray, hat: np.ndarray):
    """
    Робастная ковариация типа sandwich для логита (HC3-подобная):
    Var(beta) ≈ H^{-1} [ X^T diag( ((y-mu)^2) / (1 - h_i)^2 ) X ] H^{-1}
    где H = X^T W X — информация Фишера (W=mu*(1-mu)).
    """
    Hinv = np.linalg.pinv(H)
    resid = (y - mu)
    adj = resid * (1.0 / (np.clip(1.0 - hat, 1e-6, None)))  # деление на (1 - h_i)
    meat = X.T @ (adj[:, None] * adj[:, None] * X)
    cov = Hinv @ meat @ Hinv
    return cov

def recompute_lr_inference_corrected(processed_dir: pathlib.Path,
                                     artifacts_dir: pathlib.Path,
                                     cat_prefixes: List[str] = ["sex","job","health","glang"],
                                     robust: bool = True) -> Optional[pd.DataFrame]:
    """
    Пересчитывает логит для интерпретации (drop-first по группам) и сохраняет:
      - logreg_coefs_corrected.csv
      - logreg_reference_levels.csv
      - logreg_factor_tests.csv
    """
    try:
        Xdf, y = _load_train_matrix(processed_dir)
    except Exception as e:
        print(f"[WARN] Не удалось загрузить X/y для пересчёта LR-инференции: {e}")
        return None

    groups = _group_categorical_columns(list(Xdf.columns), cat_prefixes=cat_prefixes)
    ref_map = {}
    X_use = Xdf.copy()

    # выберем референт по максимальной частоте и удалим его колонку
    for g, cols in groups.items():
        ref_col = _pick_reference_by_frequency(Xdf, cols)
        ref_map[g] = ref_col
        if ref_col in X_use.columns:
            X_use = X_use.drop(columns=[ref_col])

    # добавим интерсепт
    X_use = X_use.copy()
    X_use.insert(0, "intercept", 1.0)

    # IRLS
    X_mat = X_use.values.astype(float)
    y_vec = y.values.astype(float)
    beta, mu, H, hat = _irls_logit(X_mat, y_vec)

    # ковариация
    if robust:
        cov = _robust_covariance_HC3(X_mat, y_vec, mu, H, hat)
    else:
        cov = np.linalg.pinv(H)

    se = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    z = beta / np.clip(se, 1e-12, None)
    # двусторонний p-value по нормальному приближению
    pvals = 2.0 * (1.0 - _phi(np.abs(z)))
    ci_lo = beta - Z975 * se
    ci_hi = beta + Z975 * se

    out = pd.DataFrame({
        "feature": X_use.columns.tolist(),
        "beta": beta.astype(float),
        "SE": se.astype(float),
        "z": z.astype(float),
        "p_value": pvals.astype(float),
        "CI_low": ci_lo.astype(float),
        "CI_high": ci_hi.astype(float),
    })

    interp_dir = artifacts_dir/"interpretability"
    interp_dir.mkdir(parents=True, exist_ok=True)
    coefs_path = interp_dir/"logreg_coefs_corrected.csv"
    out.to_csv(coefs_path, index=False)

    # референтные уровни
    pd.DataFrame([{"factor": k, "reference_dummy": v} for k, v in ref_map.items()]) \
      .to_csv(interp_dir/"logreg_reference_levels.csv", index=False)

    # Групповые Wald-тесты: H0: все дамми фактора = 0 (кроме опорной)
    try:
        param_index = {name: i for i, name in enumerate(X_use.columns.tolist())}
        rows = []
        for g, cols in groups.items():
            cols_use = [c for c in cols if c != ref_map[g]]
            idxs = [param_index[c] for c in cols_use if c in param_index]
            if not idxs:
                continue
            beta_sub = beta[idxs]
            cov_sub = cov[np.ix_(idxs, idxs)]
            # chi2 = b^T Cov^{-1} b
            try:
                inv_cov_sub = np.linalg.pinv(cov_sub)
                chi2 = float(beta_sub.T @ inv_cov_sub @ beta_sub)
                df = len(idxs)
                # p-value ~ 1 - CDF_chi2(chi2; df), без scipy оставим NA
                p_value = np.nan
                rows.append({"factor": g, "df": df, "chi2": chi2, "p_value": p_value})
            except Exception:
                pass
        if rows:
            pd.DataFrame(rows).to_csv(interp_dir/"logreg_factor_tests.csv", index=False)
    except Exception:
        pass

    print(f"[OK] LR inference corrected saved: {coefs_path}")
    return out


# ========= RQ2 + Гипотезы =========
def load_lr_coefs(artifacts_dir: pathlib.Path) -> Optional[pd.DataFrame]:
    preferred = artifacts_dir/"interpretability"/"logreg_coefs_corrected.csv"
    legacy    = artifacts_dir/"interpretability"/"logreg_coefs.csv"
    for p in [preferred, legacy]:
        if p.exists():
            try:
                return pd.read_csv(p)
            except Exception:
                continue
    return None

def load_importances(artifacts_dir: pathlib.Path, tag: str) -> Optional[pd.DataFrame]:
    if tag == "xgb_tuned":
        p = artifacts_dir/"interpretability"/"xgb_tuned_importances.csv"
    elif tag == "rf":
        p = artifacts_dir/"interpretability"/"rf_importances.csv"
    else:
        return None
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None

def rank_from_importances(df_imp: pd.DataFrame, feature_name: str) -> Optional[int]:
    cols = list(df_imp.columns)
    fcol = "feature_name" if "feature_name" in cols else ("feature" if "feature" in cols else None)
    if fcol is None:
        return None
    icol = "importance_gain" if "importance_gain" in cols else ("importance" if "importance" in cols else None)
    if icol is None:
        return None
    tmp = df_imp[[fcol, icol]].copy().sort_values(icol, ascending=False).reset_index(drop=True)
    mask = (tmp[fcol].astype(str).str.lower() == str(feature_name).lower())
    if not mask.any():
        mask = tmp[fcol].astype(str).str.lower().str.contains(str(feature_name).lower(), na=False)
    if not mask.any():
        return None
    idx = int(np.where(mask.values)[0][0])
    return idx + 1

def find_feature_by_patterns(names: List[str], patterns: List[str]) -> Optional[str]:
    ln = [n.lower() for n in names]
    for pat in patterns:
        pat = pat.lower()
        for n, nl in zip(names, ln):
            if nl == pat:
                return n
        for n, nl in zip(names, ln):
            if pat in nl:
                return n
    return None

def get_feature_name_pool(artifacts_dir: pathlib.Path, processed_dir: pathlib.Path) -> List[str]:
    pool = []
    lr = load_lr_coefs(artifacts_dir)
    if lr is not None and "feature" in lr.columns:
        pool = [f for f in lr["feature"].tolist() if str(f).lower() != "intercept"]
    if not pool:
        fpath = processed_dir/"feature_names.csv"
        if fpath.exists():
            try:
                pool = pd.read_csv(fpath, header=None).iloc[:,0].astype(str).tolist()
            except Exception:
                pass
    return pool

def rq2_and_hypotheses_tables(artifacts_dir: pathlib.Path,
                              processed_dir: pathlib.Path,
                              alpha: float = 0.05,
                              topk: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    lr_df = load_lr_coefs(artifacts_dir)
    xgb_imp = load_importances(artifacts_dir, "xgb_tuned")
    rf_imp  = load_importances(artifacts_dir, "rf")
    pool = get_feature_name_pool(artifacts_dir, processed_dir)

    spec = {
        "CES-D":        ["ces-d", "cesd", "ces", "depression"],
        "STAI-T":       ["stai-t", "stai_t", "stai", "trait_anxiety"],
        "Self-rated health": ["self-rated health", "health", "srh", "health_status", "health_1"],
        "Workload (study hours)": ["study hours", "study_hours", "workload", "hours_study", "stud_h"],
        "Employment status": ["employment", "employed", "job_status", "job", "job_1"],
        "AMSP (motivation)": ["amsp", "academic motivation", "motivation"],
        "MBI-Efficacy": ["mbi-ea", "mbi_ea", "efficacy", "mbi-pe", "professional efficacy"],
    }

    def lr_lookup_beta_p(feat_name: str) -> Tuple[Optional[float], Optional[float]]:
        if lr_df is None or "feature" not in lr_df.columns or feat_name is None:
            return None, None
        m = lr_df["feature"].astype(str).str.lower() == str(feat_name).lower()
        if not m.any():
            m = lr_df["feature"].astype(str).str.lower().str.contains(str(feat_name).lower(), na=False)
        if not m.any():
            return None, None
        row = lr_df[m].iloc[0]
        beta = float(row["beta"]) if "beta" in row else None
        pval = float(row["p_value"]) if "p_value" in row else None
        return beta, pval

    def rank(df_imp: Optional[pd.DataFrame], feat: Optional[str]) -> Optional[int]:
        return rank_from_importances(df_imp, feat) if (df_imp is not None and feat is not None) else None

    rows = []
    names = {}
    for key, pats in spec.items():
        feat = find_feature_by_patterns(pool, pats) if pool else None
        names[key] = feat if feat is not None else "NOT_FOUND"
        beta, pval = lr_lookup_beta_p(feat)
        sign = None if beta is None else ("+" if beta > 0 else ("−" if beta < 0 else "0"))
        rows.append({
            "predictor": key,
            "feature_resolved": names[key],
            "LR_beta": beta, "LR_p": pval, "LR_sign": sign,
            "XGB_rank": rank(xgb_imp, feat), "RF_rank": rank(rf_imp, feat)
        })
    features_summary = pd.DataFrame(rows)

    def is_sig(name_key: str, sign_expected: Optional[str]) -> bool:
        row = features_summary[features_summary["predictor"] == name_key]
        if row.empty: return False
        b, p = row["LR_beta"].values[0], row["LR_p"].values[0]
        if p is None or b is None: return False
        if p >= alpha: return False
        if sign_expected is None: return True
        return (b > 0 and sign_expected == "+") or (b < 0 and sign_expected == "−")

    def in_topk(name_key: str, col: str) -> bool:
        row = features_summary[features_summary["predictor"] == name_key]
        if row.empty: return False
        r = row[col].values[0]
        return (r is not None) and (not pd.isna(r)) and (int(r) <= int(topk))

    # H1: CES-D (+), STAI-T (+)
    h1 = ("Supported" if (is_sig("CES-D","+") and (in_topk("CES-D","XGB_rank") or in_topk("CES-D","RF_rank")))
          and (is_sig("STAI-T","+") or in_topk("STAI-T","XGB_rank") or in_topk("STAI-T","RF_rank"))
          else "Partially" if (is_sig("CES-D","+") or is_sig("STAI-T","+")
                                or in_topk("CES-D","XGB_rank") or in_topk("CES-D","RF_rank")
                                or in_topk("STAI-T","XGB_rank") or in_topk("STAI-T","RF_rank"))
          else "Not")

    # H2: Workload (+), Employment (+)
    h2 = ("Supported" if (is_sig("Workload (study hours)","+") and (in_topk("Workload (study hours)","XGB_rank") or in_topk("Workload (study hours)","RF_rank")))
          and (is_sig("Employment status","+") or in_topk("Employment status","XGB_rank") or in_topk("Employment status","RF_rank"))
          else "Partially" if (is_sig("Workload (study hours)","+") or is_sig("Employment status","+")
                                or in_topk("Workload (study hours)","XGB_rank") or in_topk("Workload (study hours)","RF_rank")
                                or in_topk("Employment status","XGB_rank") or in_topk("Employment status","RF_rank"))
          else "Not")

    # H3: AMSP (−), MBI-Efficacy (−)
    h3 = ("Supported" if (is_sig("AMSP (motivation)","−") and (in_topk("AMSP (motivation)","XGB_rank") or in_topk("AMSP (motivation)","RF_rank")))
          and (is_sig("MBI-Efficacy","−") or in_topk("MBI-Efficacy","XGB_rank") or in_topk("MBI-Efficacy","RF_rank"))
          else "Partially" if (is_sig("AMSP (motivation)","−") or is_sig("MBI-Efficacy","−")
                                or in_topk("AMSP (motivation)","XGB_rank") or in_topk("AMSP (motivation)","RF_rank")
                                or in_topk("MBI-Efficacy","XGB_rank") or in_topk("MBI-Efficacy","RF_rank"))
          else "Not")

    hyp = pd.DataFrame([
        {"Hypothesis": "H1 (CES-D+, STAI-T+)", "alpha": alpha, "topk_for_importance": topk, "Status": h1},
        {"Hypothesis": "H2 (Workload+, Employment+)", "alpha": alpha, "topk_for_importance": topk, "Status": h2},
        {"Hypothesis": "H3 (AMSP−, MBI-Efficacy−)", "alpha": alpha, "topk_for_importance": topk, "Status": h3},
    ])
    return features_summary, hyp


# ========= RQ3: подгруппы =========
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
        return pd.read_csv(p)
    except Exception:
        return None

def rq3_subgroups_table(artifacts_dir: pathlib.Path, processed_dir: pathlib.Path) -> Optional[pd.DataFrame]:
    sub = load_subgroups(processed_dir, artifacts_dir)
    if sub is None:
        return None
    candidate_cols = {
        "gender": ["gender", "sex"],
        "year": ["year", "study_year", "year_of_study"],
        "employment": ["employment", "job", "job_status"]
    }
    present = {}
    for key, pats in candidate_cols.items():
        col = None
        for p in pats:
            if p in sub.columns:
                col = p; break
        present[key] = col

    models = ["xgb_tuned", "mlp", "logreg", "rf"]
    rows = []
    model_preds = {}
    for m in models:
        pred_paths = (
            [artifacts_dir/"preds"/"logreg_test.csv", artifacts_dir/"preds"/"logreg_calibrated_test.csv"]
            if m == "logreg" else
            [artifacts_dir/"preds"/f"{m}_test.csv"]
        )
        pf = find_first_existing(pred_paths)
        if pf is None:
            continue
        df = read_preds(pf)
        if df is None or len(df) != len(sub):
            continue
        model_preds[m] = df

    for m, dfp in model_preds.items():
        for key, col in present.items():
            if col is None:
                continue
            for val, g in sub.groupby(col):
                y = dfp["y"].values[g.index]
                p = dfp["proba"].values[g.index]
                if np.unique(y).size < 2:
                    pr = np.nan
                else:
                    pr = float(average_precision_score(y, p))
                rows.append({"model": m, "subgroup": key, "value": val, "n": int(len(g)), "PR_AUC_test": pr})
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values(["model","subgroup","value"]).reset_index(drop=True)


# ========= MAIN =========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--artifacts_dir", default="artifacts")
    # RQ1 — порог
    ap.add_argument("--clinical_threshold_method", choices=["f1","f2","target_recall"], default="f2",
                    help="метод выбора клинического порога на валидации")
    ap.add_argument("--beta", type=float, default=2.0, help="β для Fβ при методе f2 (игнорируется для f1/target_recall)")
    ap.add_argument("--target_recall", type=float, default=0.80, help="целевой recall для метода target_recall")
    # Шаг A — пересчёт LR-инференции (встроенный)
    ap.add_argument("--recompute_lr_inference", action="store_true", default=True,
                    help="пересчитать LR-инференцию (drop-first + robust) перед сборкой RQ2/H")
    # RQ2/H
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    processed_dir = pathlib.Path(args.processed_dir)
    artifacts_dir = pathlib.Path(args.artifacts_dir)
    (artifacts_dir/"metrics").mkdir(parents=True, exist_ok=True)

    # ---- RQ1: базовая таблица PR-AUC(test)
    rq1_basic = rq1_table_pr_auc_only(artifacts_dir)
    path_basic = artifacts_dir/"metrics"/"aggregate_rq1_model_pr_auc_test.csv"
    rq1_basic.to_csv(path_basic, index=False)
    print(f"[OK] RQ1 basic PR-AUC table: {path_basic}")

    # ---- RQ1: полная клиническая таблица
    rq1_full = rq1_table_full(
        artifacts_dir,
        method=args.clinical_threshold_method,
        beta=args.beta,
        target_recall=args.target_recall
    )
    path_full = artifacts_dir/"metrics"/"aggregate_rq1_threshold_eval.csv"
    rq1_full.to_csv(path_full, index=False)
    print(f"[OK] RQ1 threshold-based table: {path_full}")

    # ---- Шаг A: корректная LR-инференция (перед RQ2/H)
    if args.recompute_lr_inference:
        recompute_lr_inference_corrected(processed_dir, artifacts_dir)

    # ---- RQ2 + Hypotheses (используют corrected, если есть)
    feat_sum, hyp_sum = rq2_and_hypotheses_tables(artifacts_dir, processed_dir, alpha=args.alpha, topk=args.topk)
    rq2_path = artifacts_dir/"metrics"/"aggregate_rq2_features_summary.csv"
    hyp_path = artifacts_dir/"metrics"/"aggregate_hypotheses_summary.csv"
    feat_sum.to_csv(rq2_path, index=False)
    hyp_sum.to_csv(hyp_path, index=False)
    print(f"[OK] RQ2 features summary: {rq2_path}")
    print(f"[OK] Hypotheses summary:  {hyp_path}")

    # ---- RQ3 (опционально)
    rq3 = rq3_subgroups_table(artifacts_dir, processed_dir)
    if rq3 is not None:
        rq3_path = artifacts_dir/"metrics"/"aggregate_rq3_subgroup_pr_auc_test.csv"
        rq3.to_csv(rq3_path, index=False)
        print(f"[OK] RQ3 subgroup PR-AUC: {rq3_path}")
    else:
        print("[WARN] RQ3 skipped (no subgroups_test.csv or length mismatch).")

    print("Done.")


if __name__ == "__main__":
    main()
