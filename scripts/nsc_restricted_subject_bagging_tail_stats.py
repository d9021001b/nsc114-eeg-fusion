#!/usr/bin/env python3
"""Patient-aware EEG subject-level feature fusion with restricted parameter search, tail statistics, and multi-seed bagging.

This script restricts the hyperparameter space to prevent inner CV optimization bias,
focusing on ensembling the robust Logistic Regression (L1 or L2) EEG branch with the baseline scores.
It incorporates tail statistics (p05, p10, p90, p95) during subject-level feature aggregation to capture transient pathological events.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Ensure scripts directory is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nsc_eeg_csv_fusion_ablation import (
    EEG_FOLDER_TO_BINARY,
    EEG_FOLDERS,
    EEG_BINARY_CLASSES,
    collect_eeg_file_index,
    extract_one_eeg_csv,
    dump_json,
    load_base_scores,
    load_labels,
    metrics,
    natural_key,
    robust_unit_fit,
    write_csv,
)


def aggregate_trial_features_tail_stats(paths: List[Path], cache_dir: Path, prefix: str, agg_funcs_base: dict) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    if not paths:
        return feats
    trial_feats = [extract_one_eeg_csv(path, cache_dir) for path in paths]
    numeric_names = sorted({k for item in trial_feats for k, v in item.items() if isinstance(v, (int, float))})
    for name in numeric_names:
        vals = np.asarray([float(item.get(name, math.nan)) for item in trial_feats], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        
        # 1. Base aggregations for ALL features
        for agg_name, func in agg_funcs_base.items():
            feats[f"{prefix}/{name}/{agg_name}"] = float(func(vals))
            
        # 2. Domain-targeted tail stats for Delta, Theta, and Alpha fractions
        if name == "band_delta_frac" or name == "band_theta_frac":
            feats[f"{prefix}/{name}/p90"] = float(np.nanpercentile(vals, 90))
            feats[f"{prefix}/{name}/p95"] = float(np.nanpercentile(vals, 95))
        elif name == "band_alpha_frac":
            feats[f"{prefix}/{name}/p05"] = float(np.nanpercentile(vals, 5))
            feats[f"{prefix}/{name}/p10"] = float(np.nanpercentile(vals, 10))
            
    return feats


def build_subject_eeg_features_tail_stats(
    subjects: List[str],
    eeg_root: Path,
    cache_dir: Path,
    include_counts: bool,
    class_aware_features: bool,
) -> Tuple[np.ndarray, List[str], dict]:
    index = collect_eeg_file_index(eeg_root)
    rows: Dict[str, Dict[str, float]] = {}
    audit = {
        "subjects_with_eeg": 0,
        "subjects_without_eeg": 0,
        "files_by_class": Counter(),
        "files_by_subject": {},
    }
    
    # Standard 7 aggregation functions for all features
    agg_funcs_base = {
        "mean": np.nanmean,
        "std": np.nanstd,
        "min": np.nanmin,
        "max": np.nanmax,
        "p25": lambda x: np.nanpercentile(x, 25),
        "p50": lambda x: np.nanpercentile(x, 50),
        "p75": lambda x: np.nanpercentile(x, 75),
    }
    
    for subject in subjects:
        feats: Dict[str, float] = {}
        total_files = 0
        binary_paths: Dict[str, List[Path]] = {cls: index.get(subject, {}).get(cls, []) for cls in EEG_BINARY_CLASSES}
        all_paths = [p for cls in EEG_BINARY_CLASSES for p in binary_paths[cls]]
        for cls, paths in binary_paths.items():
            total_files += len(paths)
            audit["files_by_class"][cls] += len(paths)
            if class_aware_features:
                if include_counts:
                    feats[f"eeg_class{cls}/trial_count"] = float(len(paths))
                    feats[f"eeg_class{cls}/has_trials"] = float(len(paths) > 0)
                feats.update(aggregate_trial_features_tail_stats(paths, cache_dir, f"eeg_class{cls}", agg_funcs_base))
        if not class_aware_features:
            feats.update(aggregate_trial_features_tail_stats(all_paths, cache_dir, "eeg_all", agg_funcs_base))
        if include_counts:
            feats["eeg_all/total_trial_count"] = float(total_files)
            feats["eeg_all/has_any_eeg"] = float(total_files > 0)
            if class_aware_features:
                for folder in EEG_FOLDERS:
                    folder_paths = index.get(subject, {}).get(f"folder{folder}", [])
                    feats[f"folder{folder}/trial_count"] = float(len(folder_paths))
                    feats[f"folder{folder}/trial_fraction"] = float(len(folder_paths) / max(total_files, 1))
                for cls in EEG_BINARY_CLASSES:
                    denom = max(total_files, 1)
                    feats[f"eeg_class{cls}/trial_fraction"] = float(len(index.get(subject, {}).get(cls, [])) / denom)
        if total_files:
            audit["subjects_with_eeg"] += 1
        else:
            audit["subjects_without_eeg"] += 1
        audit["files_by_subject"][subject] = int(total_files)
        rows[subject] = feats

    feature_names = sorted({name for row in rows.values() for name in row})
    X = np.full((len(subjects), len(feature_names)), np.nan, dtype=float)
    idx = {name: i for i, name in enumerate(feature_names)}
    for i, subject in enumerate(subjects):
        for name, value in rows[subject].items():
            X[i, idx[name]] = value
    audit["files_by_class"] = dict(audit["files_by_class"])
    return X, feature_names, audit


def select_topk(X_train: np.ndarray, y_train: np.ndarray, k: int) -> Tuple[SimpleImputer, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_train)
    if X_imp.shape[1] == 0:
        return imputer, np.array([], dtype=int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores, _ = f_classif(X_imp, y_train)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    selected = np.argsort(scores)[::-1][: min(k, X_imp.shape[1])]
    return imputer, selected


def make_model(penalty: str, C: float, seed: int):
    if penalty == "l1":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(penalty="l1", C=C, solver="liblinear", class_weight=None, max_iter=5000, random_state=seed)
        )
    else:  # l2
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(penalty="l2", C=C, solver="liblinear", class_weight=None, max_iter=5000, random_state=seed)
        )


def run_restricted_subject_fusion_oof(
    X_eeg: np.ndarray,
    y: np.ndarray,
    base_scores: Dict[str, np.ndarray],
    base_col: str,
    penalty: str,
    C: float,
    top_k: int,
    w_eegs: np.ndarray,
    objective_type: str,
    seed: int,
    n_splits: int,
    inner_splits: int,
) -> Tuple[np.ndarray, List[dict], List[dict]]:
    """Runs 10-fold patient-aware CV with nested 5-fold inner CV to search ONLY the fusion weight w_eeg."""
    outer_preds = np.zeros(len(y), dtype=float)
    fold_details: List[dict] = []
    inner_logs: List[dict] = []
    
    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    for fold, (train_idx, test_idx) in enumerate(outer.split(X_eeg, y), start=1):
        # 1. Inner CV on train_idx to select best w_eeg
        inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed + fold * 100)
        y_train = y[train_idx]
        X_train = X_eeg[train_idx]
        
        inner_oof_eeg = np.zeros(len(train_idx), dtype=float)
        
        for inner_fold, (tr_local, va_local) in enumerate(inner_cv.split(X_train, y_train), start=1):
            imputer, selected = select_topk(X_train[tr_local], y_train[tr_local], top_k)
            if len(selected) == 0:
                inner_oof_eeg[va_local] = 0.5
                continue
            X_tr_sel = imputer.transform(X_train[tr_local])[:, selected]
            X_va_sel = imputer.transform(X_train[va_local])[:, selected]
            
            clf = make_model(penalty, C, seed + fold * 10 + inner_fold)
            clf.fit(X_tr_sel, y_train[tr_local])
            inner_oof_eeg[va_local] = clf.predict_proba(X_va_sel)[:, 1]
            
        # Scale base and inner EEG predictions independently on train subjects
        base_train_score = base_scores[base_col][train_idx]
        base_scaled, _ = robust_unit_fit(base_train_score, base_train_score)
        eeg_scaled, _ = robust_unit_fit(inner_oof_eeg, inner_oof_eeg)
        
        best_metric = -1.0
        best_w = 0.0
        
        for w_eeg in w_eegs:
            fused = (1.0 - w_eeg) * base_scaled + w_eeg * eeg_scaled
            met = metrics(y_train, fused)
            
            if objective_type == "min_metric":
                obj = min(met["AUROC"], met["AUPRC"])
            elif objective_type == "auprc":
                obj = met["AUPRC"]
            else: # weighted
                obj = met["AUPRC"] + 0.05 * met["AUROC"]
                
            inner_logs.append({
                "fold": fold,
                "w_eeg": w_eeg,
                "AUROC": met["AUROC"],
                "AUPRC": met["AUPRC"],
                "objective": obj
            })
            
            if obj > best_metric:
                best_metric = obj
                best_w = w_eeg
                
        # 2. Train final EEG model on full outer training fold
        imputer, selected = select_topk(X_train, y_train, top_k)
        X_tr_full = imputer.transform(X_train)[:, selected]
        X_te_full = imputer.transform(X_eeg[test_idx])[:, selected]
        
        clf = make_model(penalty, C, seed + fold * 50)
        clf.fit(X_tr_full, y_train)
        
        # Predictions
        eeg_tr_pred = clf.predict_proba(X_tr_full)[:, 1]
        eeg_te_pred = clf.predict_proba(X_te_full)[:, 1]
        
        # Scale and fuse
        base_test_raw = base_scores[base_col][test_idx]
        base_train_unit, base_test_unit = robust_unit_fit(base_train_score, base_test_raw)
        eeg_train_unit, eeg_test_unit = robust_unit_fit(eeg_tr_pred, eeg_te_pred)
        
        fused_test = (1.0 - best_w) * base_test_unit + best_w * eeg_test_unit
        outer_preds[test_idx] = fused_test
        
        fold_details.append({
            "fold": fold,
            "train_cases": len(train_idx),
            "test_cases": len(test_idx),
            "selected_w": best_w,
            "inner_val_objective": best_metric
        })
        
    return outer_preds, fold_details, inner_logs


def make_figures(out_dir: Path, y: np.ndarray, bagged_score: np.ndarray, baseline_score: np.ndarray, label_str: str) -> Tuple[str, str]:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. ROC Curve
    fpr_bag, tpr_bag, _ = roc_curve(y, bagged_score)
    fpr_base, tpr_base, _ = roc_curve(y, baseline_score)
    
    auc_bag = roc_auc_score(y, bagged_score)
    auc_base = roc_auc_score(y, baseline_score)
    
    plt.figure(figsize=(5.4, 4.3))
    plt.plot(fpr_bag, tpr_bag, label=f"Bagged {label_str} (AUROC={auc_bag:.3f})", color="crimson", linewidth=2)
    plt.plot(fpr_base, tpr_base, label=f"Baseline mi_max (AUROC={auc_base:.3f})", color="royalblue", linestyle="--")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Bagged {label_str} ROC")
    plt.grid(alpha=0.25)
    plt.legend(loc="lower right")
    roc_path = fig_dir / "bagged_fusion_roc.png"
    plt.tight_layout()
    plt.savefig(roc_path, dpi=180)
    plt.close()
    
    # 2. PRC Curve
    prec_bag, rec_bag, _ = precision_recall_curve(y, bagged_score)
    prec_base, rec_base, _ = precision_recall_curve(y, baseline_score)
    
    prc_bag = average_precision_score(y, bagged_score)
    prc_base = average_precision_score(y, baseline_score)
    
    plt.figure(figsize=(5.4, 4.3))
    plt.plot(rec_bag, prec_bag, label=f"Bagged {label_str} (AUPRC={prc_bag:.3f})", color="crimson", linewidth=2)
    plt.plot(rec_base, prec_base, label=f"Baseline mi_max (AUPRC={prc_base:.3f})", color="royalblue", linestyle="--")
    plt.axhline(float(np.mean(y)), linestyle=":", color="gray", label=f"prevalence={np.mean(y):.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Bagged {label_str} PRC")
    plt.grid(alpha=0.25)
    plt.legend(loc="lower left")
    prc_path = fig_dir / "bagged_fusion_prc.png"
    plt.tight_layout()
    plt.savefig(prc_path, dpi=180)
    plt.close()
    
    return str(roc_path), str(prc_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="nsc_dataset_images/manifest.csv")
    parser.add_argument("--eeg-root", default="eeg-csv-data-by-class")
    parser.add_argument("--base-predictions", default="analysis/nsc_uncertain_band_patch_refinement_20260520/uncertain_band_predictions.csv")
    parser.add_argument("--cache-dir", default="analysis/nsc_eeg_csv_fusion_ablation_20260520/eeg_feature_cache")
    parser.add_argument("--out-dir", default="analysis/nsc_restricted_subject_bagging_tail_stats_20260521")
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260520, 20260521, 20260522, 20260523, 20260524, 20260525, 20260526, 20260527, 20260528, 20260529])
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--base-col", default="mi_max")
    parser.add_argument("--penalty", default="l1", choices=["l1", "l2"])
    parser.add_argument("--C", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=320)
    parser.add_argument("--objective", default="min_metric", choices=["min_metric", "auprc", "weighted"])
    parser.add_argument("--w-eeg-max", type=float, default=0.25)
    parser.add_argument("--w-eeg-step", type=float, default=0.05)
    args = parser.parse_args()

    labels = load_labels(Path(args.manifest).resolve())
    subjects = list(labels)
    y = np.asarray([labels[s] for s in subjects], dtype=int)
    
    print("Building subject-level aggregated EEG features with tail statistics...")
    X_eeg, feature_names, eeg_audit = build_subject_eeg_features_tail_stats(
        subjects,
        Path(args.eeg_root).resolve(),
        Path(args.cache_dir).resolve(),
        include_counts=False,
        class_aware_features=False
    )
    print(f"Aggregated feature matrix shape: {X_eeg.shape} (feature names size: {len(feature_names)})")
    
    base_scores = load_base_scores(Path(args.base_predictions).resolve(), subjects)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    w_eegs = np.arange(0.0, args.w_eeg_max + 1e-9, args.w_eeg_step)
    
    seed_predictions: Dict[int, np.ndarray] = {}
    all_fold_details: List[dict] = []
    all_inner_logs: List[dict] = []
    
    for seed in args.seeds:
        print(f"\n================ Running Seed {seed} ================")
        preds, fold_details, inner_logs = run_restricted_subject_fusion_oof(
            X_eeg, y, base_scores, args.base_col, args.penalty, args.C, args.top_k,
            w_eegs, args.objective, seed, args.n_splits, args.inner_splits
        )
        seed_predictions[seed] = preds
        
        # Tag summaries
        for fd in fold_details:
            all_fold_details.append({"seed": seed, **fd})
        for il in inner_logs:
            all_inner_logs.append({"seed": seed, **il})
            
        seed_metrics = metrics(y, preds)
        print(f"Seed {seed} Completed: AUROC={seed_metrics['AUROC']:.4f}, AUPRC={seed_metrics['AUPRC']:.4f}")
        
    # Multi-seed Bagging
    bagged_score = np.mean(list(seed_predictions.values()), axis=0)
    bagged_metrics = metrics(y, bagged_score)
    baseline_metrics = metrics(y, base_scores[args.base_col])
    
    label_str = f"LR_{args.penalty}_k{args.top_k}_C{args.C}_wMax{args.w_eeg_max}"
    print(f"\n================ FINAL BAGGED RESULTS ({label_str}) ================")
    print(f"Baseline {args.base_col}:   AUROC={baseline_metrics['AUROC']:.4f}, AUPRC={baseline_metrics['AUPRC']:.4f}")
    print(f"Bagged Fusion:     AUROC={bagged_metrics['AUROC']:.4f}, AUPRC={bagged_metrics['AUPRC']:.4f}")
    print(f"Accuracy:          {bagged_metrics['accuracy']:.4f}")
    print(f"Sensitivity:       {bagged_metrics['sensitivity']:.4f}")
    print(f"Specificity:       {bagged_metrics['specificity']:.4f}")
    print(f"PPV/NPV:           {bagged_metrics['PPV']:.4f} / {bagged_metrics['NPV']:.4f}")
    print(f"Confusion Matrix:  TN={bagged_metrics['TN']}, FP={bagged_metrics['FP']}, FN={bagged_metrics['FN']}, TP={bagged_metrics['TP']}")
    
    # Save predictions and summaries
    write_csv(out_dir / "bagged_predictions.csv", [
        {"subject_id": s, "true_label": int(yy), "bagged_score": float(bs), f"{args.base_col}_score": float(ms)}
        for s, yy, bs, ms in zip(subjects, y, bagged_score, base_scores[args.base_col])
    ], ["subject_id", "true_label", "bagged_score", f"{args.base_col}_score"])
    
    write_csv(out_dir / "seed_fold_details.csv", all_fold_details, sorted({k for r in all_fold_details for k in r}))
    write_csv(out_dir / "inner_search_logs.csv", all_inner_logs, sorted({k for r in all_inner_logs for k in r}))
    
    roc_fig, prc_fig = make_figures(out_dir, y, bagged_score, base_scores[args.base_col], label_str)
    
    summary_row = {
        "method": f"Bagged Restricted Subject Fusion with Tail Stats ({label_str})",
        "AUROC": bagged_metrics["AUROC"],
        "AUPRC": bagged_metrics["AUPRC"],
        "ACC": bagged_metrics["accuracy"],
        "Sens": bagged_metrics["sensitivity"],
        "Spec": bagged_metrics["specificity"],
        "PPV": bagged_metrics["PPV"],
        "NPV": bagged_metrics["NPV"],
        "TN": bagged_metrics["TN"],
        "FP": bagged_metrics["FP"],
        "FN": bagged_metrics["FN"],
        "TP": bagged_metrics["TP"]
    }
    write_csv(out_dir / "bagged_metrics_summary.csv", [summary_row], list(summary_row.keys()))
    
    manifest = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "subjects": len(subjects),
        "seeds_run": args.seeds,
        "parameters": {
            "base_col": args.base_col,
            "penalty": args.penalty,
            "C": args.C,
            "top_k": args.top_k,
            "objective": args.objective,
            "w_eeg_max": args.w_eeg_max,
            "w_eeg_step": args.w_eeg_step
        },
        "bagged_results": bagged_metrics,
        "outputs": {
            "predictions": str(out_dir / "bagged_predictions.csv"),
            "fold_details": str(out_dir / "seed_fold_details.csv"),
            "summary": str(out_dir / "bagged_metrics_summary.csv"),
            "roc_curve": roc_fig,
            "prc_curve": prc_fig
        }
    }
    dump_json(out_dir / "manifest.json", manifest)
    print("\nExecution and saving completed successfully!")


if __name__ == "__main__":
    main()
