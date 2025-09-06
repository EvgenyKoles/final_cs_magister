# -*- coding: utf-8 -*-
"""
Логистическая регрессия с тюнингом по AP (PR-AUC), опциональной калибровкой и сохранением вероятностей.
Версия устойчиво сходится (по умолчанию liblinear), подавляет ConvergenceWarning и не «спамит» предупреждениями.
"""

import argparse, json, pathlib, warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_recall_fscore_support, brier_score_loss
)
from sklearn.exceptions import ConvergenceWarning
from joblib import dump


# --- подавим «спам» про несходимость (мы поднимем max_iter и ослабим tol; предупреждения нам не нужны)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ---------- IO ----------
def load_xy(processed_dir: str):
    p = pathlib.Path(processed_dir)
    X_train = np.loadtxt(p/"X_train.csv", delimiter=",")
    X_val   = np.loadtxt(p/"X_val.csv",   delimiter=",")
    X_test  = np.loadtxt(p/"X_test.csv",  delimiter=",")
    y_train = np.loadtxt(p/"y_train.csv", delimiter=",", skiprows=1).astype(int)
    y_val   = np.loadtxt(p/"y_val.csv",   delimiter=",", skiprows=1).astype(int)
    y_test  = np.loadtxt(p/"y_test.csv",  delimiter=",", skiprows=1).astype(int)

    assert X_train.shape[0] == y_train.shape[0], f"Mismatch train: {X_train.shape[0]} vs {y_train.shape[0]}"
    assert X_val.shape[0]   == y_val.shape[0],   f"Mismatch val: {X_val.shape[0]} vs {y_val.shape[0]}"
    assert X_test.shape[0]  == y_test.shape[0],  f"Mismatch test: {X_test.shape[0]} vs {y_test.shape[0]}"
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


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
    brier = brier_score_loss(y_true, proba)
    return {
        "PR_AUC": float(pr_auc),
        "ROC_AUC": float(roc),
        "F1": float(f1),
        "Recall@Thr": recall_pos,
        "Brier": float(brier),
        "Thr": float(thr),
    }


def evaluate_from_proba(y_true, proba, thr):
    y_hat = (proba >= thr).astype(int)
    return compute_metrics(y_true, y_hat, proba, thr)


def majority_metrics(y_true, thr):
    y_hat = np.zeros_like(y_true)
    proba = np.zeros_like(y_true, dtype=float)
    return compute_metrics(y_true, y_hat, proba, thr)


# ---------- Utils ----------
def prevalence(y):
    y = np.asarray(y).astype(int)
    return float(y.mean()) if y.size else float("nan")


def save_probas(split, y_true, proba, outdir: pathlib.Path, prefix="logreg"):
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y": y_true, "proba": proba}).to_csv(outdir / f"{prefix}_{split}.csv", index=False)


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed", help="папка с X_*/y_*")
    ap.add_argument("--artifacts_dir", default="artifacts", help="куда сохранять метрики/модели/прогнозы")

    # Порог для отчёта Recall@Thr (PR-AUC всегда по вероятностям)
    ap.add_argument("--thr", type=float, default=0.35, help="клинический порог вероятности для отчёта Recall@Thr")

    # Классовые веса
    ap.add_argument("--class_weight", default="balanced", choices=["none","balanced"],
                    help="вес классов для LR")

    # --- ВАЖНО: по умолчанию ставим liblinear (быстрее и стабильнее на OHE), L1/L2
    ap.add_argument("--solver", default="liblinear", choices=["liblinear", "saga"],
                    help="решатель. saga даёт elasticnet, но может сходиться дольше")
    ap.add_argument("--penalty", default="l1", choices=["l2","l1","elasticnet"],
                    help="штраф. elasticnet доступен только при saga")

    # Сетки гиперпараметров
    ap.add_argument("--C_grid", default="0.05,0.1,0.2,0.5,1.0,2.0,5.0",
                    help="список C через запятую (чем выше C, тем слабее регуляризация)")
    ap.add_argument("--l1_grid", default="0.1,0.3,0.5",
                    help="список l1_ratio через запятую (только для elasticnet+saga)")
    ap.add_argument("--cv_folds", type=int, default=5, help="число фолдов для GridSearchCV")

    # Устойчивость сходимости
    ap.add_argument("--max_iter", type=int, default=10000, help="максимум итераций для LR")
    ap.add_argument("--tol", type=float, default=1e-3, help="критерий остановки (чем больше, тем раньше стоп)")

    # Калибровка вероятностей
    ap.add_argument("--calibrate", action="store_true", help="включить isotonic-калибровку (CalibratedClassifierCV)")
    ap.add_argument("--cal_cv_folds", type=int, default=3, help="фолды для внутренней CV в калибровке")

    # Сохранение вероятностей
    ap.add_argument("--save_probas", action="store_true", help="сохранять proba для train/val/test")

    args = ap.parse_args()

    # --- load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_xy(args.processed_dir)

    # --- prevalence (AP бейзлайн)
    prev_train, prev_val, prev_test = prevalence(y_train), prevalence(y_val), prevalence(y_test)
    print(f"[INFO] Prevalence  train={prev_train:.4f}  val={prev_val:.4f}  test={prev_test:.4f}  (AP бейзлайн ~= prevalence)")

    # --- build param grid
    C_list = [float(x) for x in args.C_grid.split(",") if x.strip()]
    param_grid = {"C": C_list}

    solver = args.solver
    penalty = args.penalty

    if solver == "liblinear":
        # liblinear поддерживает l1/l2; elasticnet не поддерживается — переключим на l2, если вдруг выбрали
        if penalty == "elasticnet":
            penalty = "l2"
        param_grid["penalty"] = [penalty] if penalty in ("l1", "l2") else ["l2"]
    else:
        # saga: поддерживает l1/l2/elasticnet
        param_grid["penalty"] = [penalty]
        if penalty == "elasticnet":
            l1_list = [float(x) for x in args.l1_grid.split(",") if x.strip()]
            param_grid["l1_ratio"] = l1_list

    cw = None if args.class_weight == "none" else "balanced"

    base_lr = LogisticRegression(
        solver=solver,
        penalty=("l2" if penalty == "elasticnet" else penalty),  # реальный penalty уточнит GridSearch
        class_weight=cw,
        max_iter=args.max_iter,
        tol=args.tol,
        random_state=42,
        n_jobs=None  # держим совместимость разных версий sklearn
    )

    # --- grid search by AP
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    gs = GridSearchCV(
        estimator=base_lr,
        param_grid=param_grid,
        scoring="average_precision",  # оптимизируем PR-AUC
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
        error_score="raise"  # если что-то реально сломалось — выкинем исключение, но ConvergenceWarning уже подавлен
    )
    gs.fit(X_train, y_train)
    clf = gs.best_estimator_
    print(f"[INFO] Best params: {gs.best_params_} | CV(AP)={gs.best_score_:.4f}")

    # --- optional calibration (fit на train без утечки)
    if args.calibrate:
        cal = CalibratedClassifierCV(clf, method="isotonic", cv=args.cal_cv_folds)
        cal.fit(X_train, y_train)
        clf_final = cal
        model_name = "logreg_calibrated"
    else:
        clf_final = clf
        clf_final.fit(X_train, y_train)
        model_name = "logreg"

    # --- probabilities
    if hasattr(clf_final, "predict_proba"):
        proba_train = clf_final.predict_proba(X_train)[:, 1]
        proba_val   = clf_final.predict_proba(X_val)[:, 1]
        proba_test  = clf_final.predict_proba(X_test)[:, 1]
    else:
        # fallback (на всякий случай)
        def _scores2proba(scores):
            r = scores.argsort().argsort().astype(float)
            return r / (len(r) - 1 + 1e-9)
        proba_train = _scores2proba(clf_final.decision_function(X_train))
        proba_val   = _scores2proba(clf_final.decision_function(X_val))
        proba_test  = _scores2proba(clf_final.decision_function(X_test))

    # --- evaluate
    m_train = evaluate_from_proba(y_train, proba_train, thr=args.thr)
    m_val   = evaluate_from_proba(y_val,   proba_val,   thr=args.thr)
    m_test  = evaluate_from_proba(y_test,  proba_test,  thr=args.thr)

    base_val  = majority_metrics(y_val,  thr=args.thr)
    base_test = majority_metrics(y_test, thr=args.thr)

    # --- print
    print(f"\n=== Logistic Regression ({model_name}) class_weight={cw} solver={solver} ===")
    print(f"[TRAIN] PR-AUC={m_train['PR_AUC']:.4f} | ROC-AUC={m_train['ROC_AUC']:.4f} | F1={m_train['F1']:.4f} | Recall@{args.thr:.2f}={m_train['Recall@Thr']:.4f} | Brier={m_train['Brier']:.4f}")
    print(f"[VAL  ] PR-AUC={m_val['PR_AUC']:.4f} | ROC-AUC={m_val['ROC_AUC']:.4f} | F1={m_val['F1']:.4f} | Recall@{args.thr:.2f}={m_val['Recall@Thr']:.4f} | Brier={m_val['Brier']:.4f}")
    print(f"[TEST ] PR-AUC={m_test['PR_AUC']:.4f} | ROC-AUC={m_test['ROC_AUC']:.4f} | F1={m_test['F1']:.4f} | Recall@{args.thr:.2f}={m_test['Recall@Thr']:.4f} | Brier={m_test['Brier']:.4f}")

    print("\n--- Majority baseline (always 0) ---")
    print(f"[VAL  ] PR-AUC={base_val['PR_AUC']:.4f} | ROC-AUC={base_val['ROC_AUC']:.4f} | F1={base_val['F1']:.4f} | Recall@{args.thr:.2f}={base_val['Recall@Thr']:.4f} | Brier={base_val['Brier']:.4f}")
    print(f"[TEST ] PR-AUC={base_test['PR_AUC']:.4f} | ROC-AUC={base_test['ROC_AUC']:.4f} | F1={base_test['F1']:.4f} | Recall@{args.thr:.2f}={base_test['Recall@Thr']:.4f} | Brier={base_test['Brier']:.4f}")

    # --- save artifacts
    artifacts = pathlib.Path(args.artifacts_dir)
    (artifacts/"metrics").mkdir(parents=True, exist_ok=True)
    (artifacts/"models").mkdir(parents=True, exist_ok=True)

    # save model
    model_path = artifacts/"models"/f"{model_name}.joblib"
    dump(clf_final, model_path)

    # save probas
    if args.save_probas:
        probas_dir = artifacts/"probas"
        save_probas("train", y_train, proba_train, probas_dir, prefix=model_name)
        save_probas("val",   y_val,   proba_val,   probas_dir, prefix=model_name)
        save_probas("test",  y_test,  proba_test,  probas_dir, prefix=model_name)

    # save metrics json
    out = {
        "model": "logistic_regression",
        "variant": model_name,
        "params_best": getattr(gs, "best_params_", {}),
        "cv_folds": args.cv_folds,
        "cv_best_AP": float(getattr(gs, "best_score_", np.nan)),
        "class_weight": (None if cw is None else "balanced"),
        "threshold_for_report": args.thr,
        "prevalence": {
            "train": prevalence(y_train),
            "val": prevalence(y_val),
            "test": prevalence(y_test)
        },
        "train": m_train,
        "val": m_val,
        "test": m_test,
        "baseline_val": base_val,
        "baseline_test": base_test,
        "artifacts": {
            "model_path": str(model_path),
            "probas_saved": bool(args.save_probas)
        }
    }
    (artifacts/"metrics"/"logreg_eval.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[OK] Метрики сохранены: {artifacts/'metrics'/'logreg_eval.json'}")
    print(f"[OK] Модель сохранена:  {model_path}")

if __name__ == "__main__":
    main()
