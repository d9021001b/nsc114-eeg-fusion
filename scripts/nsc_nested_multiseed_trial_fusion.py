#!/usr/bin/env python3
"""Patient-aware EEG trial-level classifier fusion with nested CV and multi-seed bagging.

This script implements:
1. Strict nested cross-validation: for each outer patient-aware fold, we run an
   inner 5-fold cross-validation on the training subjects to grid search the best:
   - Trial classifier model (ExtraTrees, LR)
   - Feature selection size k (16, 32, 48, 64)
   - Trial aggregation mode (mean, max, top3mean, mean_max)
   - Baseline score column (mi_max, raw_only)
   - Fusion weight w_eeg (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
2. Multi-seed bagging: ensembling out-of-fold patient probability scores across
   multiple random CV seeds (S=5) to reduce variance and boost generalization.
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
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
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
    dump_json,
    load_base_scores,
    load_labels,
    natural_key,
    robust_unit_fit,
    write_csv,
)
from nsc_eeg_trial_level_fusion import (
    aggregate_scores,
    collect_trial_table,
    metrics,
    trial_matrix,
)


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


def make_model(name: str, seed: int):
    if name == "LR":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, C=0.25, class_weight="balanced", solver="liblinear", random_state=seed)
        )
    if name == "ExtraTrees":
        return ExtraTreesClassifier(
            n_estimators=250,
            random_state=seed,
            class_weight="balanced",
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=-1
        )
    if name == "RF":
        return RandomForestClassifier(
            n_estimators=250,
            random_state=seed,
            class_weight="balanced",
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=-1
        )
    if name == "HistGB":
        return HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.035,
            l2_regularization=0.1,
            random_state=seed
        )
    raise ValueError(f"Unknown model: {name}")


def nested_trial_fusion_oof(
    subjects: List[str],
    y_patient: np.ndarray,
    trial_rows: List[dict],
    feature_names: List[str],
    base_scores: Dict[str, np.ndarray],
    seed: int,
    n_splits: int,
    inner_splits: int,
    grid: dict,
) -> Tuple[np.ndarray, List[dict], List[dict]]:
    """Runs 10-fold patient-aware CV with nested 5-fold hyperparameter search."""
    X_trial, y_trial, trial_subjects = trial_matrix(trial_rows, feature_names)
    subject_to_idx = {s: i for i, s in enumerate(subjects)}
    
    outer_preds = np.zeros(len(subjects), dtype=float)
    fold_details: List[dict] = []
    inner_logs: List[dict] = []
    
    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    subjects_arr = np.asarray(subjects, dtype=object)
    
    for fold, (train_idx, test_idx) in enumerate(outer.split(np.zeros(len(subjects)), y_patient), start=1):
        print(f"  --- Outer Fold {fold}/{n_splits} (Seed {seed}) ---")
        train_subs = subjects_arr[train_idx]
        test_subs = subjects_arr[test_idx]
        
        # 1. Inner Cross-Validation on train_subs to select the best hyper-parameters
        inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed + fold * 100)
        y_train_subs = y_patient[train_idx]
        
        # Precompute inner OOF trial predictions for each model and top_k
        inner_oof_trial_scores = {}
        for model_name in grid["models"]:
            for top_k in grid["top_ks"]:
                inner_oof_trial_scores[(model_name, top_k)] = np.zeros(len(trial_subjects), dtype=float)
                
        # Run inner CV splits
        for inner_fold, (tr_local, va_local) in enumerate(inner_cv.split(np.zeros(len(train_subs)), y_train_subs), start=1):
            inner_tr_subs = set(train_subs[tr_local])
            inner_va_subs = set(train_subs[va_local])
            
            # Map trial indices
            tr_mask = np.asarray([s in inner_tr_subs for s in trial_subjects], dtype=bool)
            va_mask = np.asarray([s in inner_va_subs for s in trial_subjects], dtype=bool)
            
            if np.sum(tr_mask) == 0 or np.sum(va_mask) == 0 or len(np.unique(y_trial[tr_mask])) < 2:
                continue
                
            for model_name in grid["models"]:
                for top_k in grid["top_ks"]:
                    imputer, selected = select_topk(X_trial[tr_mask], y_trial[tr_mask], top_k)
                    if len(selected) == 0:
                        continue
                    X_tr_sel = imputer.transform(X_trial[tr_mask])[:, selected]
                    X_va_sel = imputer.transform(X_trial[va_mask])[:, selected]
                    
                    clf = make_model(model_name, seed + fold * 10 + inner_fold)
                    clf.fit(X_tr_sel, y_trial[tr_mask])
                    preds = clf.predict_proba(X_va_sel)[:, 1]
                    inner_oof_trial_scores[(model_name, top_k)][va_mask] = preds

        # Evaluate all grid search combinations on the inner OOF scores
        best_metric = -1.0
        best_params = {}
        
        for model_name in grid["models"]:
            for top_k in grid["top_ks"]:
                # Trial level OOF predictions for train_subs
                tr_subs_mask = np.asarray([s in train_subs for s in trial_subjects], dtype=bool)
                trial_preds = inner_oof_trial_scores[(model_name, top_k)][tr_subs_mask]
                trial_subs_subset = trial_subjects[tr_subs_mask]
                
                # Group trial predictions by subject
                sub_to_trial_preds = defaultdict(list)
                for s, p in zip(trial_subs_subset, trial_preds):
                    sub_to_trial_preds[str(s)].append(p)
                
                for agg in grid["agg_modes"]:
                    # Aggregate trial scores into patient scores for train_subs
                    eeg_train_score = np.zeros(len(train_subs), dtype=float)
                    for i, s in enumerate(train_subs):
                        eeg_train_score[i] = aggregate_scores(np.asarray(sub_to_trial_preds.get(s, [0.5])), agg)
                    
                    for base_col in grid["base_cols"]:
                        base_train_score = base_scores[base_col][train_idx]
                        
                        # Scale EEG and base scores INDEPENDENTLY within the train subjects
                        base_scaled, _ = robust_unit_fit(base_train_score, base_train_score)
                        eeg_scaled, _ = robust_unit_fit(eeg_train_score, eeg_train_score)
                        
                        for w_eeg in grid["w_eegs"]:
                            fused = (1.0 - w_eeg) * base_scaled + w_eeg * eeg_scaled
                            
                            # Evaluate on training subjects
                            met = metrics(y_train_subs, fused)
                            obj = met["AUPRC"] + 0.05 * met["AUROC"]
                            
                            inner_logs.append({
                                "fold": fold,
                                "model": model_name,
                                "top_k": top_k,
                                "agg": agg,
                                "base_col": base_col,
                                "w_eeg": w_eeg,
                                "AUROC": met["AUROC"],
                                "AUPRC": met["AUPRC"],
                                "objective": obj
                            })
                            
                            if obj > best_metric:
                                best_metric = obj
                                best_params = {
                                    "model": model_name,
                                    "top_k": top_k,
                                    "agg": agg,
                                    "base_col": base_col,
                                    "w_eeg": w_eeg
                                }
        
        print(f"    Best inner params: {best_params} (val objective = {best_metric:.4f})")
        
        # 2. Train selected model on the entire training fold and evaluate on outer test fold
        best_model = best_params["model"]
        best_k = best_params["top_k"]
        best_agg = best_params["agg"]
        best_base_col = best_params["base_col"]
        best_w = best_params["w_eeg"]
        
        tr_mask = np.asarray([s in train_subs for s in trial_subjects], dtype=bool)
        te_mask = np.asarray([s in test_subs for s in trial_subjects], dtype=bool)
        
        # Select features on full train_subs trials
        imputer, selected = select_topk(X_trial[tr_mask], y_trial[tr_mask], best_k)
        X_tr_full = imputer.transform(X_trial[tr_mask])[:, selected]
        X_te_full = imputer.transform(X_trial[te_mask])[:, selected]
        
        # Fit final model
        clf = make_model(best_model, seed + fold * 50)
        clf.fit(X_tr_full, y_trial[tr_mask])
        
        # Predict on outer train and test trials
        tr_trial_scores = clf.predict_proba(X_tr_full)[:, 1]
        te_trial_scores = clf.predict_proba(X_te_full)[:, 1]
        
        # Aggregate trial predictions
        sub_to_tr_scores = defaultdict(list)
        for s, p in zip(trial_subjects[tr_mask], tr_trial_scores):
            sub_to_tr_scores[str(s)].append(p)
            
        sub_to_te_scores = defaultdict(list)
        for s, p in zip(trial_subjects[te_mask], te_trial_scores):
            sub_to_te_scores[str(s)].append(p)
            
        eeg_train_agg = np.asarray([aggregate_scores(np.asarray(sub_to_tr_scores.get(s, [0.5])), best_agg) for s in train_subs])
        eeg_test_agg = np.asarray([aggregate_scores(np.asarray(sub_to_te_scores.get(s, [0.5])), best_agg) for s in test_subs])
        
        # Scale and fuse base & EEG scores independently
        base_train_raw = base_scores[best_base_col][train_idx]
        base_test_raw = base_scores[best_base_col][test_idx]
        
        base_train_unit, base_test_unit = robust_unit_fit(base_train_raw, base_test_raw)
        eeg_train_unit, eeg_test_unit = robust_unit_fit(eeg_train_agg, eeg_test_agg)
        
        fused_test = (1.0 - best_w) * base_test_unit + best_w * eeg_test_unit
        outer_preds[test_idx] = fused_test
        
        # Record fold summary
        fold_details.append({
            "fold": fold,
            "train_cases": len(train_idx),
            "test_cases": len(test_idx),
            "selected_model": best_model,
            "selected_k": best_k,
            "selected_agg": best_agg,
            "selected_base": best_base_col,
            "selected_w": best_w,
            "inner_val_objective": best_metric
        })
        
    return outer_preds, fold_details, inner_logs


def make_figures(out_dir: Path, y: np.ndarray, bagged_score: np.ndarray, baseline_score: np.ndarray) -> Tuple[str, str]:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. ROC Curve
    fpr_bag, tpr_bag, _ = roc_curve(y, bagged_score)
    fpr_base, tpr_base, _ = roc_curve(y, baseline_score)
    
    auc_bag = roc_auc_score(y, bagged_score)
    auc_base = roc_auc_score(y, baseline_score)
    
    plt.figure(figsize=(5.4, 4.3))
    plt.plot(fpr_bag, tpr_bag, label=f"Bagged Fusion (AUROC={auc_bag:.3f})", color="crimson", linewidth=2)
    plt.plot(fpr_base, tpr_base, label=f"Baseline mi_max (AUROC={auc_base:.3f})", color="royalblue", linestyle="--")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Bagged EEG-Trial Fusion ROC")
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
    plt.plot(rec_bag, prec_bag, label=f"Bagged Fusion (AUPRC={prc_bag:.3f})", color="crimson", linewidth=2)
    plt.plot(rec_base, prec_base, label=f"Baseline mi_max (AUPRC={prc_base:.3f})", color="royalblue", linestyle="--")
    plt.axhline(float(np.mean(y)), linestyle=":", color="gray", label=f"prevalence={np.mean(y):.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Bagged EEG-Trial Fusion PRC")
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
    parser.add_argument("--out-dir", default="analysis/nsc_nested_multiseed_trial_fusion_20260521")
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260520, 20260521, 20260522, 20260523, 20260524])
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--inner-splits", type=int, default=5)
    args = parser.parse_args()

    labels = load_labels(Path(args.manifest).resolve())
    subjects = list(labels)
    y = np.asarray([labels[s] for s in subjects], dtype=int)
    
    print("Collecting trial features...")
    trial_rows, feature_names = collect_trial_table(
        Path(args.eeg_root).resolve(),
        Path(args.cache_dir).resolve(),
        set(subjects)
    )
    print(f"Loaded {len(trial_rows)} trials across {len({r['subject_id'] for r in trial_rows})} subjects.")
    
    base_scores = load_base_scores(Path(args.base_predictions).resolve(), subjects)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Validation grid
    grid = {
        "models": ["LR", "ExtraTrees"],
        "top_ks": [16, 32, 48, 64],
        "agg_modes": ["mean", "max", "top3mean", "mean_max"],
        "base_cols": ["mi_max", "raw_only"],
        "w_eegs": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    }
    
    seed_predictions: Dict[int, np.ndarray] = {}
    all_fold_details: List[dict] = []
    all_inner_logs: List[dict] = []
    
    for seed in args.seeds:
        print(f"\n================ Running Seed {seed} ================")
        preds, fold_details, inner_logs = nested_trial_fusion_oof(
            subjects, y, trial_rows, feature_names, base_scores,
            seed, args.n_splits, args.inner_splits, grid
        )
        seed_predictions[seed] = preds
        
        # Tag summaries
        for fd in fold_details:
            all_fold_details.append({"seed": seed, **fd})
        for il in inner_logs:
            all_inner_logs.append({"seed": seed, **il})
            
        seed_metrics = metrics(y, preds)
        print(f"Seed {seed} Completed: AUROC={seed_metrics['AUROC']:.4f}, AUPRC={seed_metrics['AUPRC']:.4f}")
        
    # Multi-seed Bagging (averaging predicted probabilities across seeds)
    bagged_score = np.mean(list(seed_predictions.values()), axis=0)
    bagged_metrics = metrics(y, bagged_score)
    baseline_metrics = metrics(y, base_scores["mi_max"])
    
    print("\n================ FINAL BAGGED RESULTS ================")
    print(f"Baseline mi_max:   AUROC={baseline_metrics['AUROC']:.4f}, AUPRC={baseline_metrics['AUPRC']:.4f}")
    print(f"Bagged Fusion:     AUROC={bagged_metrics['AUROC']:.4f}, AUPRC={bagged_metrics['AUPRC']:.4f}")
    print(f"Accuracy:          {bagged_metrics['accuracy']:.4f}")
    print(f"Sensitivity:       {bagged_metrics['sensitivity']:.4f}")
    print(f"Specificity:       {bagged_metrics['specificity']:.4f}")
    print(f"PPV/NPV:           {bagged_metrics['PPV']:.4f} / {bagged_metrics['NPV']:.4f}")
    print(f"Confusion Matrix:  TN={bagged_metrics['TN']}, FP={bagged_metrics['FP']}, FN={bagged_metrics['FN']}, TP={bagged_metrics['TP']}")
    
    # Save predictions and summaries
    write_csv(out_dir / "bagged_predictions.csv", [
        {"subject_id": s, "true_label": int(yy), "bagged_score": float(bs), "mi_max_score": float(ms)}
        for s, yy, bs, ms in zip(subjects, y, bagged_score, base_scores["mi_max"])
    ], ["subject_id", "true_label", "bagged_score", "mi_max_score"])
    
    write_csv(out_dir / "seed_fold_details.csv", all_fold_details, sorted({k for r in all_fold_details for k in r}))
    write_csv(out_dir / "inner_search_logs.csv", all_inner_logs, sorted({k for r in all_inner_logs for k in r}))
    
    roc_fig, prc_fig = make_figures(out_dir, y, bagged_score, base_scores["mi_max"])
    
    summary_row = {
        "method": "Bagged Nested EEG-Trial Fusion",
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
        "validation_strategy": "10-fold patient-aware stratified CV with nested 5-fold inner CV",
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
