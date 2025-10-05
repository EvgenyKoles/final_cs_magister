# -*- coding: utf-8 -*-
"""
Logistic Regression baseline with:
- CV over C/penalty optimizing Average Precision (PR-AUC) for prediction quality (RQ1).
- Optional isotonic calibration for probabilities.
- Inferential statistics (coef, SE, z, p-value, 95% CI) via statsmodels.Logit for H1/H3.
Artifacts saved only for RQ/H.
"""

import argparse, json, pathlib, warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_recall_fscore_support
)
from sklearn.exceptions import ConvergenceWarning

# optional model persistence
from joblib import dump

# inferential stats for hypotheses
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError

# --- suppress convergence spam
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ---------- IO ----------
def load_xy(processed_dir: str):
    p = pathlib.Path(processed_dir)
    X_train = np.loadtxt(p / "X_train.csv", delimiter=",")
    X_val   = np.loadtxt(p / "X_val.csv",   delimiter=",")
    X_test  = np.loadtxt(p / "X_test.csv",  delimiter=",")
    y_train = np.loadtxt(p / "y_train.csv", delimiter=",", skiprows=1).astype(int)
    y_val   = np.loadtxt(p / "y_val.csv",   delimiter=",", skiprows=1).astype(int)
    y_test  = np.loadtxt(p / "y_test.csv",  delimiter=",", skiprows=1).astype(int)

    assert X_train.shape[0] == y_train.shape[0], f"Mismatch train: {X_train.shape[0]} vs {y_train.shape[0]}"
    assert X_val.shape[0]   == y_val.shape[0],   f"Mismatch val: {X_val.shape[0]} vs {y_val.shape[0]}"
    assert X_test.shape[0]  == y_test.shape[0],  f"Mismatch test: {X_test.shape[0]} vs {y_test.shape[0]}"
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def try_load_feature_names(processed_dir: str, n_features: int):
    """Try to read processed/feature_names.csv (single column). If absent, fallback to f0..f{n-1}."""
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


# ---------- Metrics ----------
def compute_metrics(y_true, y_hat, proba, thr):
    if len(np.unique(y_true)) > 1:
        pr_auc = average_precision_score(y_true, proba)
        roc    = roc_auc_score(y_true, proba)
    else:
        pr_auc, roc = float("nan"), float("nan")

    f1 = f1_score(y_true, y_hat, zero_division=0)
    prec, rec, _, _ = precision_recall_fscore_support(
        y_true, y_hat, zero_division=0, average=None
    )
    recall_pos = float(rec[1]) if len(rec) > 1 else 0.0
    return {
        "PR_AUC": float(pr_auc),
        "ROC_AUC": float(roc),
        "F1": float(f1),
        "Recall@Thr": recall_pos,
        "Thr": float(thr),
    }


def evaluate_from_proba(y_true, proba, thr):
    y_hat = (proba >= thr).astype(int)
    return compute_metrics(y_true, y_hat, proba, thr)


def prevalence(y):
    y = np.asarray(y).astype(int)
    return float(y.mean()) if y.size else float("nan")


def save_probas(split, y_true, proba, outdir: pathlib.Path, prefix="logreg"):
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y": y_true, "proba": proba}).to_csv(outdir / f"{prefix}_{split}.csv", index=False)


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed", help="folder with X_*/y_*")
    ap.add_argument("--artifacts_dir", default="artifacts", help="where to save metrics/models/probas/coef")

    # Threshold for Recall@Thr report (PR-AUC always on probabilities)
    ap.add_argument("--thr", type=float, default=0.35, help="Clinical probability threshold for Recall@Thr report")

    # Class weights
    ap.add_argument("--class_weight", default="balanced", choices=["none", "balanced"], help="class weight for LR")

    # Solver/penalty
    ap.add_argument("--solver", default="liblinear", choices=["liblinear", "saga"], help="solver")
    ap.add_argument("--penalty", default="l1", choices=["l2", "l1", "elasticnet"], help="regularization penalty")

    # Hypergrid
    ap.add_argument("--C_grid", default="0.05,0.1,0.2,0.5,1.0,2.0,5.0",
                    help="List of C (comma-separated); higher C = weaker regularization")
    ap.add_argument("--l1_grid", default="0.1,0.3,0.5", help="l1_ratio (elasticnet + saga only)")
    ap.add_argument("--cv_folds", type=int, default=5, help="CV folds for GridSearchCV")

    # Convergence
    ap.add_argument("--max_iter", type=int, default=10000, help="maximum iterations for LR")
    ap.add_argument("--tol", type=float, default=1e-3, help="stopping tolerance")

    # Probability calibration
    ap.add_argument("--calibrate", action="store_true", help="enable isotonic calibration (CalibratedClassifierCV)")
    ap.add_argument("--cal_cv_folds", type=int, default=3, help="internal CV for calibration")

    # Saving options
    ap.add_argument("--save_probas", action="store_true", help="save probabilities for train/val/test")
    ap.add_argument("--save_model", action="store_true", help="save trained sklearn model (optional)")

    args = ap.parse_args()

    # --- load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_xy(args.processed_dir)
    feature_names = try_load_feature_names(args.processed_dir, X_train.shape[1])

    # --- prevalence (reference AP line)
    prev_train, prev_val, prev_test = prevalence(y_train), prevalence(y_val), prevalence(y_test)
    print(f"[INFO] Prevalence  train={prev_train:.4f}  val={prev_val:.4f}  test={prev_test:.4f}")

    # --- build param grid
    C_list = [float(x) for x in args.C_grid.split(",") if x.strip()]
    param_grid = {"C": C_list}

    solver = args.solver
    penalty = args.penalty

    if solver == "liblinear":
        if penalty == "elasticnet":
            penalty = "l2"
        param_grid["penalty"] = [penalty] if penalty in ("l1", "l2") else ["l2"]
    else:
        param_grid["penalty"] = [penalty]
        if penalty == "elasticnet":
            l1_list = [float(x) for x in args.l1_grid.split(",") if x.strip()]
            param_grid["l1_ratio"] = l1_list

    cw = None if args.class_weight == "none" else "balanced"

    base_lr = LogisticRegression(
        solver=solver,
        penalty=("l2" if penalty == "elasticnet" else penalty),
        class_weight=cw,
        max_iter=args.max_iter,
        tol=args.tol,
        random_state=42,
        n_jobs=None
    )

    # --- grid search by Average Precision
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    gs = GridSearchCV(
        estimator=base_lr,
        param_grid=param_grid,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
        error_score="raise"
    )
    gs.fit(X_train, y_train)
    clf = gs.best_estimator_
    print(f"[INFO] Best params: {gs.best_params_} | CV(AP)={gs.best_score_:.4f}")

    # --- optional calibration
    if args.calibrate:
        cal = CalibratedClassifierCV(clf, method="isotonic", cv=args.cal_cv_folds)
        cal.fit(X_train, y_train)
        clf_final = cal
        model_name = "logreg_calibrated"
    else:
        clf_final = clf
        clf_final.fit(X_train, y_train)
        model_name = "logreg"

    # --- probabilities & metrics
    if hasattr(clf_final, "predict_proba"):
        proba_train = clf_final.predict_proba(X_train)[:, 1]
        proba_val   = clf_final.predict_proba(X_val)[:, 1]
        proba_test  = clf_final.predict_proba(X_test)[:, 1]
    else:
        # fallback: rank-transform decision_function to [0,1]
        def _scores2proba(scores):
            r = scores.argsort().argsort().astype(float)
            return r / (len(r) - 1 + 1e-9)
        proba_train = _scores2proba(clf_final.decision_function(X_train))
        proba_val   = _scores2proba(clf_final.decision_function(X_val))
        proba_test  = _scores2proba(clf_final.decision_function(X_test))

    m_train = evaluate_from_proba(y_train, proba_train, thr=args.thr)
    m_val   = evaluate_from_proba(y_val,   proba_val,   thr=args.thr)
    m_test  = evaluate_from_proba(y_test,  proba_test,  thr=args.thr)

    print(f"\n=== Logistic Regression ({model_name}) class_weight={cw} solver={solver} ===")
    print(f"[TRAIN] PR-AUC={m_train['PR_AUC']:.4f} | ROC-AUC={m_train['ROC_AUC']:.4f} | F1={m_train['F1']:.4f} | Recall@{args.thr:.2f}={m_train['Recall@Thr']:.4f}")
    print(f"[VAL  ] PR-AUC={m_val['PR_AUC']:.4f} | ROC-AUC={m_val['ROC_AUC']:.4f} | F1={m_val['F1']:.4f} | Recall@{args.thr:.2f}={m_val['Recall@Thr']:.4f}")
    print(f"[TEST ] PR-AUC={m_test['PR_AUC']:.4f} | ROC-AUC={m_test['ROC_AUC']:.4f} | F1={m_test['F1']:.4f} | Recall@{args.thr:.2f}={m_test['Recall@Thr']:.4f}")

    # --- artifacts dirs
    artifacts = pathlib.Path(args.artifacts_dir)
    (artifacts / "metrics").mkdir(parents=True, exist_ok=True)
    (artifacts / "preds").mkdir(parents=True, exist_ok=True)
    (artifacts / "interpretability").mkdir(parents=True, exist_ok=True)
    (artifacts / "models").mkdir(parents=True, exist_ok=True)

    # --- optional: save model (not required for RQ/H)
    model_path = None
    if args.save_model:
        model_path = artifacts / "models" / f"{model_name}.joblib"
        dump(clf_final, model_path)

    # --- save probabilities (for 06-aggregator)
    if args.save_probas:
        save_probas("train", y_train, proba_train, artifacts / "preds", prefix=model_name)
        save_probas("val",   y_val,   proba_val,   artifacts / "preds", prefix=model_name)
        save_probas("test",  y_test,  proba_test,  artifacts / "preds", prefix=model_name)

    # --- inferential stats for hypotheses (statsmodels Logit without regularization)
    coefs_path = artifacts / "interpretability" / "logreg_coefs.csv"
    try:
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_train_sm = sm.add_constant(X_train_df, has_constant='add')
        logit = sm.Logit(y_train, X_train_sm)
        res = logit.fit(disp=0)

        ci = res.conf_int(alpha=0.05)
        df_coef = pd.DataFrame({
            "feature": ["intercept"] + feature_names,
            "beta":    res.params.values,
            "SE":      res.bse.values,
            "z":       res.tvalues.values,       # for Logit, tvalues are z-scores
            "p_value": res.pvalues.values,
            "CI_low":  ci[0].values,
            "CI_high": ci[1].values
        })
        df_coef.to_csv(coefs_path, index=False)
        coef_info = {"path": str(coefs_path), "n_features": len(feature_names), "method": "statsmodels.Logit (no penalty)"}
        print(f"[OK] coefficients with p-values saved: {coefs_path}")
    except (PerfectSeparationError, np.linalg.LinAlgError, Exception) as e:
        # fallback: save sklearn coefficients without p-values
        print(f"[WARN] statsmodels.Logit failed ({type(e).__name__}: {e}). Saving sklearn coefficients only (no p-values).")
        beta = (clf.coef_.ravel().tolist() if hasattr(clf, "coef_") else [np.nan] * len(feature_names))
        intercept = (float(clf.intercept_[0]) if hasattr(clf, "intercept_") else np.nan)
        df_coef = pd.DataFrame({
            "feature": ["intercept"] + feature_names,
            "beta":    [intercept] + beta,
            "SE":      [np.nan] * (len(feature_names) + 1),
            "z":       [np.nan] * (len(feature_names) + 1),
            "p_value": [np.nan] * (len(feature_names) + 1),
            "CI_low":  [np.nan] * (len(feature_names) + 1),
            "CI_high": [np.nan] * (len(feature_names) + 1),
        })
        df_coef.to_csv(coefs_path, index=False)
        coef_info = {"path": str(coefs_path), "n_features": len(feature_names), "method": "sklearn coefficients (no p-values)"}

    # --- save metrics json (only RQ-relevant metrics)
    out = {
        "model": "logistic_regression",
        "variant": model_name,
        "params_best": getattr(gs, "best_params_", {}),
        "cv_folds": args.cv_folds,
        "cv_best_AP": float(getattr(gs, "best_score_", np.nan)),
        "class_weight": (None if cw is None else "balanced"),
        "threshold_for_report": args.thr,
        "prevalence": {"train": prev_train, "val": prev_val, "test": prev_test},
        "train": m_train,
        "val": m_val,
        "test": m_test,
        "artifacts": {
            "coefs_path": coef_info,
            "preds_saved": bool(args.save_probas),
            "model_path": (str(model_path) if model_path else None)
        }
    }
    metrics_path = artifacts / "metrics" / "logreg_eval.json"
    metrics_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[OK] metrics saved: {metrics_path}")

if __name__ == "__main__":
    main()
