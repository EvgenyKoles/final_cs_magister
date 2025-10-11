# -*- coding: utf-8 -*-
"""
train_models
"""

import argparse, json, pathlib, warnings
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_recall_fscore_support
)
from sklearn.exceptions import ConvergenceWarning
from joblib import dump

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
    assert X_train.shape[0]==y_train.shape[0] and X_val.shape[0]==y_val.shape[0] and X_test.shape[0]==y_test.shape[0]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

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

def save_probas(split, y_true, proba, outdir: pathlib.Path, prefix: str):
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y": y_true, "proba": proba}).to_csv(outdir / f"{prefix}_{split}.csv", index=False)

# ---------- Metrics ----------
def compute_metrics(y_true, y_hat, proba, thr):
    if len(np.unique(y_true)) > 1:
        pr_auc = average_precision_score(y_true, proba)
        roc    = roc_auc_score(y_true, proba)
    else:
        pr_auc, roc = float("nan"), float("nan")
    f1 = f1_score(y_true, y_hat, zero_division=0)
    prec, rec, _, _ = precision_recall_fscore_support(y_true, y_hat, zero_division=0, average=None)
    recall_pos = float(rec[1]) if len(rec) > 1 else 0.0
    return {"PR_AUC": float(pr_auc), "ROC_AUC": float(roc), "F1": float(f1), "Recall@Thr": recall_pos, "Thr": float(thr)}

def evaluate_from_proba(y_true, proba, thr):
    y_hat = (proba >= thr).astype(int)
    return compute_metrics(y_true, y_hat, proba, thr)

def prevalence(y):
    y = np.asarray(y).astype(int)
    return float(y.mean()) if y.size else float("nan")

# ---------- Models ----------
def build_model(name: str, y_train):
    name = name.lower()
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        # balanced_subsample — good imbalance default
        return "rf", RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=1,
            n_jobs=-1, random_state=42, class_weight="balanced_subsample"
        )
    elif name == "mlp":
        from sklearn.neural_network import MLPClassifier
        return "mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu",
            alpha=1e-4, learning_rate_init=1e-3, max_iter=500,
            early_stopping=True, n_iter_no_change=20, random_state=42
        )
    elif name == "xgb":
        try:
            import xgboost as xgb
        except Exception as e:
            raise RuntimeError("xgboost is not installed. Use RF/MLP here and tune XGB via 05_tune_xgb.py") from e
        # простой бейзлайн XGB; полноценный тюнинг — в 05
        scale_pos_weight = (len(y_train) - int(sum(y_train))) / max(int(sum(y_train)), 1)
        return "xgb", xgb.XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=4,
            subsample=0.9, colsample_bytree=0.8,
            reg_lambda=1.0, reg_alpha=0.0,
            objective="binary:logistic", eval_metric="logloss",
            n_jobs=-1, random_state=42, tree_method="hist",
            scale_pos_weight=scale_pos_weight
        )
    else:
        raise ValueError(f"Unknown model: {name}. Use one of: rf, mlp, xgb")

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--artifacts_dir", default="artifacts")
    ap.add_argument("--model", default="rf", choices=["rf", "mlp", "xgb"])
    ap.add_argument("--thr", type=float, default=0.35, help="probability threshold for Recall/F1 print (final thresholds chosen in 06)")
    ap.add_argument("--save_probas", action="store_true")
    ap.add_argument("--save_model", action="store_true")
    args = ap.parse_args()

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_xy(args.processed_dir)
    feature_names = try_load_feature_names(args.processed_dir, X_train.shape[1])

    model_tag, clf = build_model(args.model, y_train)

    clf.fit(X_train, y_train)

    # predict probabilities
    if hasattr(clf, "predict_proba"):
        proba_train = clf.predict_proba(X_train)[:, 1]
        proba_val   = clf.predict_proba(X_val)[:, 1]
        proba_test  = clf.predict_proba(X_test)[:, 1]
    else:
        scores = clf.decision_function(X_train); r = scores.argsort().argsort().astype(float); proba_train = r/(len(r)-1+1e-9)
        scores = clf.decision_function(X_val);   r = scores.argsort().argsort().astype(float); proba_val   = r/(len(r)-1+1e-9)
        scores = clf.decision_function(X_test);  r = scores.argsort().argsort().astype(float); proba_test  = r/(len(r)-1+1e-9)

    # metrics
    m_train = evaluate_from_proba(y_train, proba_train, args.thr)
    m_val   = evaluate_from_proba(y_val,   proba_val,   args.thr)
    m_test  = evaluate_from_proba(y_test,  proba_test,  args.thr)

    print(f"\n=== {model_tag.upper()} ===")
    print(f"[TRAIN] PR-AUC={m_train['PR_AUC']:.4f} | ROC-AUC={m_train['ROC_AUC']:.4f} | F1={m_train['F1']:.4f} | Recall@{args.thr:.2f}={m_train['Recall@Thr']:.4f}")
    print(f"[VAL  ] PR-AUC={m_val['PR_AUC']:.4f} | ROC-AUC={m_val['ROC_AUC']:.4f} | F1={m_val['F1']:.4f} | Recall@{args.thr:.2f}={m_val['Recall@Thr']:.4f}")
    print(f"[TEST ] PR-AUC={m_test['PR_AUC']:.4f} | ROC-AUC={m_test['ROC_AUC']:.4f} | F1={m_test['F1']:.4f} | Recall@{args.thr:.2f}={m_test['Recall@Thr']:.4f}")

    # dirs
    artifacts = pathlib.Path(args.artifacts_dir)
    (artifacts/"metrics").mkdir(parents=True, exist_ok=True)
    (artifacts/"preds").mkdir(parents=True, exist_ok=True)
    (artifacts/"interpretability").mkdir(parents=True, exist_ok=True)
    (artifacts/"models").mkdir(parents=True, exist_ok=True)

    # save probas
    if args.save_probas:
        save_probas("train", y_train, proba_train, artifacts/"preds", prefix=model_tag)
        save_probas("val",   y_val,   proba_val,   artifacts/"preds", prefix=model_tag)
        save_probas("test",  y_test,  proba_test,  artifacts/"preds", prefix=model_tag)

    # feature importances where applicable
    importances_path = None
    if model_tag == "rf":
        imp = getattr(clf, "feature_importances_", None)
        if imp is not None:
            df_imp = pd.DataFrame({"feature": feature_names, "importance": imp})
            importances_path = artifacts/"interpretability"/f"{model_tag}_importances.csv"
            df_imp.to_csv(importances_path, index=False)
    elif model_tag == "xgb":
        try:
            # prefer gain importances
            gain = getattr(clf, "feature_importances_", None)  # importance_type="gain" for sklearn API by default
            if gain is not None and len(gain)==len(feature_names):
                df_imp = pd.DataFrame({"feature": feature_names, "importance": gain})
                importances_path = artifacts/"interpretability"/f"{model_tag}_importances.csv"
                df_imp.to_csv(importances_path, index=False)
        except Exception:
            pass

    # save model (optional)
    model_path = None
    if args.save_model:
        model_path = artifacts/"models"/f"{model_tag}.joblib"
        dump(clf, model_path)

    # save metrics json
    prev = {"train": prevalence(y_train), "val": prevalence(y_val), "test": prevalence(y_test)}
    out = {
        "model": model_tag,
        "threshold_for_report": args.thr,
        "prevalence": prev,
        "train": m_train, "val": m_val, "test": m_test,
        "artifacts": {
            "preds_saved": bool(args.save_probas),
            "model_path": (str(model_path) if model_path else None),
            "importances_path": (str(importances_path) if importances_path else None),
        }
    }
    (artifacts/"metrics"/f"{model_tag}_eval.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[OK] metrics saved: {artifacts/'metrics'/f'{model_tag}_eval.json'}")
    if importances_path:
        print(f"[OK] importances saved: {importances_path}")
    if model_path:
        print(f"[OK] model saved: {model_path}")

if __name__ == "__main__":
    main()
