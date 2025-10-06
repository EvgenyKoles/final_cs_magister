# -*- coding: utf-8 -*-
"""
RQ2: Which predictors contribute most? — feature importance with a single Top-5 table across models.

Models: Logistic Regression (LR), Random Forest (RF), XGBoost (XGB; optional), MLP.
Outputs (artifacts/interpretability):
  - rq2_shap_summary_<model>.csv, rq2_shap_rank_<model>.csv        (per-model)
  - rq2_consensus_rank.csv                                         (mean rank across models)
  - rq2_top5_all_models.csv                                        (stacked Top-5 across models)
  - rq2_top5_pivot.csv                                             (pivoted Top-5 table)
  - rq2_method_report.json                                         (what ran, any fallbacks)

Robustness for small/imbalanced datasets:
  * If SHAP unavailable or fails, fallback to permutation importance with scoring="neg_log_loss".
  * Stratified subsampling for background/eval subsets.
  * Direction of effect:
      - LR: sign of standardized coefficients.
      - Others: sign of rank-correlation (pseudo-Spearman) between feature and predicted risk.
"""
\
import argparse, json, pathlib, warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# XGBoost (optional)
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# SHAP (optional)
try:
    import shap
    shap.set_log_level("ERROR")
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

warnings.filterwarnings("ignore")


# ---------------- IO ----------------
def load_xy(processed_dir: str):
    p = pathlib.Path(processed_dir)
    X_train = np.loadtxt(p / "X_train.csv", delimiter=",")
    X_val   = np.loadtxt(p / "X_val.csv",   delimiter=",")
    X_test  = np.loadtxt(p / "X_test.csv",  delimiter=",")
    y_train = np.loadtxt(p / "y_train.csv", delimiter=",", skiprows=1).astype(int)
    y_val   = np.loadtxt(p / "y_val.csv",   delimiter=",", skiprows=1).astype(int)
    y_test  = np.loadtxt(p / "y_test.csv",  delimiter=",", skiprows=1).astype(int)

    assert X_train.shape[0] == y_train.shape[0]
    assert X_val.shape[0]   == y_val.shape[0]
    assert X_test.shape[0]  == y_test.shape[0]
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


# ------------- Utils --------------
def mean_abs_shap_with_ci(shap_values: np.ndarray, B: int = 300, seed: int = 42):
    """Return mean|SHAP| and bootstrap CI per feature."""
    rng = np.random.default_rng(seed)
    abs_vals = np.abs(shap_values)
    n, d = abs_vals.shape
    mean_abs = abs_vals.mean(axis=0)

    lows, highs = np.zeros(d), np.zeros(d)
    if n >= 2:
        for j in range(d):
            idx = rng.integers(0, n, size=n)
            samp = abs_vals[idx, j].mean(axis=0)
            # быстрый перцентиль бутстрапа по одной выборке
            boots = []
            for _ in range(B):
                idx = rng.integers(0, n, size=n)
                boots.append(abs_vals[idx, j].mean())
            qs = np.quantile(boots, [0.025, 0.975])
            lows[j], highs[j] = qs[0], qs[1]
    else:
        lows[:] = np.nan; highs[:] = np.nan
    return mean_abs, lows, highs


def permutation_fallback(estimator, X, y, feature_names, n_repeats=20, seed=42):
    """Permutation importance with scoring='neg_log_loss' (robust to single-class eval)."""
    res = permutation_importance(
        estimator, X, y,
        n_repeats=n_repeats,
        random_state=seed,
        scoring="neg_log_loss",
        n_jobs=-1
    )
    imp = np.maximum(res.importances_mean, 0.0)
    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": imp,   # unify column name
        "ci_low": np.nan,
        "ci_high": np.nan
    }).sort_values("mean_abs_shap", ascending=False)
    return df


def rank_df_from_importance(df_imp: pd.DataFrame, col="mean_abs_shap"):
    df = df_imp.copy()
    df["rank"] = df[col].rank(ascending=False, method="average")
    return df[["feature", col, "rank"]]


def subsample_stratified(X, y, k, seed=42):
    """Stratified subsample (tries to keep both classes; falls back to random if single-class)."""
    n = X.shape[0]
    if k >= n:
        return X, y
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    if len(idx_pos) == 0 or len(idx_neg) == 0:
        idx = rng.choice(n, size=k, replace=False)
        return X[idx], y[idx]
    n_pos = max(1, int(round(k * len(idx_pos) / n)))
    n_pos = min(n_pos, len(idx_pos))
    n_neg = k - n_pos
    n_neg = min(n_neg, len(idx_neg))
    sel_pos = rng.choice(idx_pos, size=n_pos, replace=False)
    sel_neg = rng.choice(idx_neg, size=n_neg, replace=False)
    idx = np.concatenate([sel_pos, sel_neg])
    rem = k - idx.size
    if rem > 0:
        pool = idx_neg if (len(idx_neg) - n_neg) >= (len(idx_pos) - n_pos) else idx_pos
        add = rng.choice(pool, size=rem, replace=False)
        idx = np.concatenate([idx, add])
    rng.shuffle(idx)
    return X[idx], y[idx]


def rankcorr_signs_vs_proba(estimator, X_eval, feature_names):
    """
    Pseudo-Spearman sign between each feature and predicted risk.
    Returns array of {-1,0,+1}.
    """
    # predicted risk (use proba if available)
    if hasattr(estimator, "predict_proba"):
        p = estimator.predict_proba(X_eval)[:, 1]
    else:
        s = estimator.decision_function(X_eval)
        p = 1.0 / (1.0 + np.exp(-s))

    # ranks (Spearman via Pearson on ranks)
    rp = p.argsort().argsort().astype(float)
    rp = (rp - rp.mean()) / (rp.std() + 1e-12)

    n, d = X_eval.shape
    signs = np.zeros(d, dtype=int)
    for j in range(d):
        xj = X_eval[:, j]
        if np.std(xj) == 0:
            signs[j] = 0
            continue
        rx = xj.argsort().argsort().astype(float)
        rx = (rx - rx.mean()) / (rx.std() + 1e-12)
        r = (rx * rp).mean()  # корреляция Пирсона на рангах
        signs[j] = 1 if r > 0 else (-1 if r < 0 else 0)
    return signs


# ------------- SHAP wrappers -------------
def shap_for_lr(pipeline_lr: Pipeline, X_bg, X_eval):
    lr = pipeline_lr.named_steps.get("lr", None)
    scaler = pipeline_lr.named_steps.get("scaler", None)
    X_bg_tr = pipeline_lr.named_steps["scaler"].transform(X_bg) if scaler else X_bg
    X_eval_tr = pipeline_lr.named_steps["scaler"].transform(X_eval) if scaler else X_eval
    expl = shap.LinearExplainer(lr, X_bg_tr, feature_perturbation="interventional")
    sv = expl.shap_values(X_eval_tr)
    return sv if isinstance(sv, np.ndarray) else np.asarray(sv)


def shap_for_tree(estimator, X_bg, X_eval):
    expl = shap.TreeExplainer(estimator, data=X_bg, feature_perturbation="interventional", model_output="probability")
    sv = expl.shap_values(X_eval)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) == 2 else sv[0]
    return np.asarray(sv)


def shap_for_mlp(pipeline_mlp: Pipeline, X_bg, X_eval, nsamples="auto"):
    scaler = pipeline_mlp.named_steps.get("scaler", None)
    clf = pipeline_mlp.named_steps.get("mlp", None)
    X_bg_tr = pipeline_mlp.named_steps["scaler"].transform(X_bg) if scaler else X_bg
    X_eval_tr = pipeline_mlp.named_steps["scaler"].transform(X_eval) if scaler else X_eval
    f = lambda X: clf.predict_proba(X)[:, 1]
    expl = shap.KernelExplainer(f, X_bg_tr, link="logit")
    sv = expl.shap_values(X_eval_tr, nsamples=nsamples)
    return sv if isinstance(sv, np.ndarray) else np.asarray(sv)


# ------------- Main -------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--artifacts_dir", default="artifacts")
    ap.add_argument("--eval_split", default="test", choices=["val","test"], help="split for SHAP/permutation evaluation")
    ap.add_argument("--bg_size", type=int, default=200, help="background size for SHAP Linear/Kernel explainer")
    ap.add_argument("--max_eval", type=int, default=2000, help="cap on eval rows for importance")
    ap.add_argument("--bootstrap", type=int, default=300, help="bootstrap iterations for CI (SHAP only)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # IO
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_xy(args.processed_dir)
    feature_names = try_load_feature_names(args.processed_dir, X_train.shape[1])

    if args.eval_split == "test":
        X_eval, y_eval = X_test, y_test
    else:
        X_eval, y_eval = X_val, y_val

    # Stratified subsampling for background/eval
    X_bg, y_bg = subsample_stratified(X_train, y_train, args.bg_size, seed=args.seed)
    X_eval_sub, y_eval_sub = subsample_stratified(X_eval, y_eval, min(args.max_eval, X_eval.shape[0]), seed=args.seed)

    artifacts = pathlib.Path(args.artifacts_dir)
    out_dir = artifacts / "interpretability"
    out_dir.mkdir(parents=True, exist_ok=True)

    method_report = {
        "shap_available": _HAS_SHAP,
        "xgboost_available": _HAS_XGB,
        "split": args.eval_split,
        "bg_size": int(X_bg.shape[0]),
        "eval_size": int(X_eval_sub.shape[0]),
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "models": {}
    }

    # ----------------- Define and fit models -----------------
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(solver="liblinear", penalty="l1", C=0.1,
                                  class_weight="balanced", max_iter=10000, tol=1e-3, random_state=args.seed))
    ])
    lr_pipe.fit(X_train, y_train)

    rf = RandomForestClassifier(
        n_estimators=600, max_depth=None, min_samples_leaf=2, class_weight="balanced_subsample",
        random_state=args.seed, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    if _HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=600, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss",
            n_jobs=-1, random_state=args.seed, tree_method="hist"
        )
        xgb.fit(X_train, y_train)
    else:
        xgb = None

    mlp_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-4,
                              learning_rate_init=1e-3, max_iter=600, random_state=args.seed))
    ])
    mlp_pipe.fit(X_train, y_train)

    models = {"lr": lr_pipe, "rf": rf, "xgb": xgb, "mlp": mlp_pipe}

    # ----------------- Importance computation -----------------
    all_rankings = []
    top5_rows = []  # rows for the final Top-5 table

    feat_idx = {f: i for i, f in enumerate(feature_names)}

    for name, est in models.items():
        if est is None:
            continue
        print(f"[INFO] Computing importance for {name.upper()} ...")
        used = "shap"
        df_out = None

        # importance
        if _HAS_SHAP:
            try:
                if name == "lr":
                    sv = shap_for_lr(est, X_bg, X_eval_sub)
                elif name == "rf":
                    sv = shap_for_tree(est, X_bg, X_eval_sub)
                elif name == "xgb":
                    sv = shap_for_tree(est, X_bg, X_eval_sub)
                elif name == "mlp":
                    sv = shap_for_mlp(est, X_bg, X_eval_sub, nsamples="auto")
                else:
                    raise RuntimeError("Unknown model")
                mean_abs, lo, hi = mean_abs_shap_with_ci(sv, B=args.bootstrap, seed=args.seed)
                df_out = pd.DataFrame({
                    "feature": feature_names,
                    "mean_abs_shap": mean_abs,
                    "ci_low": lo,
                    "ci_high": hi
                }).sort_values("mean_abs_shap", ascending=False)
            except Exception as e:
                print(f"[WARN] SHAP failed for {name.upper()} ({type(e).__name__}: {e}). Falling back to permutation importance.")
                used = f"permutation_fallback:{type(e).__name__}"
                df_out = permutation_fallback(est, X_eval_sub, y_eval_sub, feature_names, n_repeats=20, seed=args.seed)
        else:
            print(f"[WARN] SHAP not available. Using permutation importance for {name.upper()}.")
            used = "permutation_only"
            df_out = permutation_fallback(est, X_eval_sub, y_eval_sub, feature_names, n_repeats=20, seed=args.seed)

        # per-model files
        out_csv = out_dir / f"rq2_shap_summary_{name}.csv"
        df_out.to_csv(out_csv, index=False)
        df_rank = df_out[["feature", "mean_abs_shap"]].copy()
        df_rank["rank"] = df_rank["mean_abs_shap"].rank(ascending=False, method="average")
        df_rank.sort_values("rank", inplace=True)
        out_rank = out_dir / f"rq2_shap_rank_{name}.csv"
        df_rank.to_csv(out_rank, index=False)

        # consensus prep
        all_rankings.append(df_rank[["feature", "rank"]].rename(columns={"rank": f"rank_{name}"}))
        method_report["models"][name] = {"importance": used,
                                         "summary_csv": str(out_csv),
                                         "rank_csv": str(out_rank)}

        # --------- Direction of effect ----------
        if name == "lr":
            # sign from standardized coefficients
            coef = est.named_steps["lr"].coef_.ravel()
            dir_vec = np.sign(coef).astype(int)
        else:
            dir_vec = rankcorr_signs_vs_proba(est, X_eval_sub, feature_names)

        # --------- Top-5 rows for final table ----------
        top5 = df_rank.head(5).merge(df_out[["feature", "ci_low", "ci_high"]], on="feature", how="left")
        for _, r in top5.iterrows():
            f = r["feature"]
            j = feat_idx[f]
            s = dir_vec[j]
            dir_str = "↑" if s > 0 else ("↓" if s < 0 else "–")
            top5_rows.append({
                "model": name.upper(),
                "feature": f,
                "importance": float(r["mean_abs_shap"]),
                "rank": float(r["rank"]),
                "direction": dir_str,
                "method": used,
                "ci_low": (None if pd.isna(r.get("ci_low")) else float(r.get("ci_low"))),
                "ci_high": (None if pd.isna(r.get("ci_high")) else float(r.get("ci_high")))
            })

    # ----------------- Consensus ranking -----------------
    if all_rankings:
        dfc = all_rankings[0]
        for i in range(1, len(all_rankings)):
            dfc = dfc.merge(all_rankings[i], on="feature", how="outer")
        rank_cols = [c for c in dfc.columns if c.startswith("rank_")]
        dfc["rank_mean"] = dfc[rank_cols].mean(axis=1, skipna=True)
        dfc.sort_values("rank_mean", inplace=True)
        dfc.to_csv(out_dir / "rq2_consensus_rank.csv", index=False)

    # ----------------- Final Top-5 tables -----------------
    if top5_rows:
        df_top = pd.DataFrame(top5_rows)
        # stacked table
        df_top.sort_values(["model", "rank", "feature"], inplace=True)
        top_path = out_dir / "rq2_top5_all_models.csv"
        df_top.to_csv(top_path, index=False)

        # pivot: features x models (importance, rank, direction)
        pivot_blocks = []
        for col in ["importance", "rank", "direction"]:
            p = df_top.pivot_table(index="feature", columns="model", values=col, aggfunc="first")
            p.columns = [f"{col}_{c}" for c in p.columns]
            pivot_blocks.append(p)
        df_pivot = pd.concat(pivot_blocks, axis=1)
        df_pivot = df_pivot.sort_values([c for c in df_pivot.columns if c.startswith("rank_")], axis=0)
        pivot_path = out_dir / "rq2_top5_pivot.csv"
        df_pivot.to_csv(pivot_path)

        method_report["top5_table"] = {"stacked": str(top_path), "pivot": str(pivot_path)}

    # ----------------- Method report -----------------
    method_json = out_dir / "rq2_method_report.json"
    method_json.write_text(json.dumps(method_report, indent=2, ensure_ascii=False))
    print(f"[OK] Saved RQ2 artifacts to: {out_dir}")

if __name__ == "__main__":
    main()
