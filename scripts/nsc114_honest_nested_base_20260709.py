#!/usr/bin/env python3
"""Nested EEG/base-weight audit for NSC114 Paper 1.

The published base branch (analysis/nsc_restricted_subject_bagging_tail_stats_corrected_20260521)
runs 10-fold patient-aware outer CV with an inner 5-fold loop that searches ONLY the fusion
weight w_eeg. Its (penalty, C, top_k) were fixed from the command line, and the published
choice (l1, C=0.25, top_k=320) is the arg-max of a 9-config grid that was scored on the FULL
out-of-fold set (all 114 subjects). That is hyperparameter-selection optimism leaking into
every "held-out" OOF prediction.

This script does two things:
  PHASE A (fidelity check): reproduce the published base at the fixed (C=0.25, top_k=320)
                            and assert AUROC == 0.7971464019851116.
  PHASE B:                  select (C, top_k, w_eeg) JOINTLY inside the inner 5-fold loop.

The fixed ``mi_max`` input is separately cross-fitted and is not regenerated on these
outer folds; therefore this script does not establish a fully nested image pipeline.

Nothing here overwrites existing artifacts; everything is written to a new out-dir.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import nsc_restricted_subject_bagging_tail_stats as bm  # noqa: E402
from nsc_eeg_csv_fusion_ablation import (  # noqa: E402
    load_labels,
    load_base_scores,
    metrics,
    robust_unit_fit,
)

PUBLISHED_AUROC = 0.7971464019851116
PUBLISHED_AUPRC = 0.7988638952075635
SEEDS = [20260520, 20260521, 20260522, 20260523, 20260524,
         20260525, 20260526, 20260527, 20260528, 20260529]
CS = [0.10, 0.25, 0.50]
TOPKS = [160, 320, 480]


def nested_fusion_oof(X, y, base_vec, penalty, Cs, topks, w_eegs, objective, seed,
                      n_splits, inner_splits):
    """Identical to the published routine EXCEPT (C, top_k) are chosen in the inner loop."""
    outer_preds = np.zeros(len(y), dtype=float)
    chosen = []
    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(outer.split(X, y), start=1):
        Xtr, ytr = X[tr], y[tr]
        base_tr = base_vec[tr]
        base_scaled, _ = robust_unit_fit(base_tr, base_tr)

        inner = StratifiedKFold(n_splits=inner_splits, shuffle=True,
                                random_state=seed + fold * 100)
        splits = list(inner.split(Xtr, ytr))

        best_obj, best_cfg = -1.0, None
        for C in Cs:
            for k in topks:
                oof = np.zeros(len(tr), dtype=float)
                for j, (trl, val) in enumerate(splits, start=1):
                    imp, sel = bm.select_topk(Xtr[trl], ytr[trl], k)
                    if len(sel) == 0:
                        oof[val] = 0.5
                        continue
                    Xa = imp.transform(Xtr[trl])[:, sel]
                    Xb = imp.transform(Xtr[val])[:, sel]
                    clf = bm.make_model(penalty, C, seed + fold * 10 + j)
                    clf.fit(Xa, ytr[trl])
                    oof[val] = clf.predict_proba(Xb)[:, 1]
                eeg_scaled, _ = robust_unit_fit(oof, oof)
                for w in w_eegs:
                    fused = (1.0 - w) * base_scaled + w * eeg_scaled
                    met = metrics(ytr, fused)
                    obj = (min(met["AUROC"], met["AUPRC"]) if objective == "min_metric"
                           else met["AUPRC"])
                    if obj > best_obj:
                        best_obj, best_cfg = obj, (C, k, w)

        C, k, w = best_cfg
        imp, sel = bm.select_topk(Xtr, ytr, k)
        Xa = imp.transform(Xtr)[:, sel]
        Xb = imp.transform(X[te])[:, sel]
        clf = bm.make_model(penalty, C, seed + fold * 50)
        clf.fit(Xa, ytr)
        eeg_tr = clf.predict_proba(Xa)[:, 1]
        eeg_te = clf.predict_proba(Xb)[:, 1]

        _, base_te_u = robust_unit_fit(base_tr, base_vec[te])
        _, eeg_te_u = robust_unit_fit(eeg_tr, eeg_te)
        outer_preds[te] = (1.0 - w) * base_te_u + w * eeg_te_u

        chosen.append({"seed": seed, "fold": fold, "C": C, "top_k": k,
                       "w_eeg": float(w), "inner_obj": float(best_obj)})
    return outer_preds, chosen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/nsc_dataset_images/manifest.csv")
    ap.add_argument("--eeg-root", type=Path, default=ROOT / "data/eeg-csv-data-by-class")
    ap.add_argument("--cache-dir", type=Path,
                    default=ROOT / "analysis/nsc_eeg_csv_fusion_ablation_20260520/eeg_feature_cache")
    ap.add_argument("--base-predictions", type=Path,
                    default=ROOT / "analysis/nsc_uncertain_band_patch_refinement_20260520/uncertain_band_predictions.csv")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "analysis/nsc114_honest_nested_base_20260709")
    ap.add_argument("--base-col", default="mi_max")
    ap.add_argument("--penalty", default="l1")
    ap.add_argument("--objective", default="min_metric")
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--inner-splits", type=int, default=5)
    ap.add_argument("--w-eeg-max", type=float, default=0.25)
    ap.add_argument("--w-eeg-step", type=float, default=0.05)
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    labels = load_labels(args.manifest.resolve())
    subjects = list(labels)
    y = np.asarray([labels[s] for s in subjects], dtype=int)
    print(f"[data] subjects={len(subjects)} class0={(y==0).sum()} class1={(y==1).sum()}", flush=True)

    t0 = time.time()
    X, feat_names, eeg_audit = bm.build_subject_eeg_features_tail_stats(
        subjects, args.eeg_root.resolve(), args.cache_dir.resolve(),
        include_counts=False, class_aware_features=False)
    print(f"[data] X_eeg={X.shape}  feats={len(feat_names)}  eeg_audit={eeg_audit}  ({time.time()-t0:.1f}s)", flush=True)

    base_scores = load_base_scores(args.base_predictions.resolve(), subjects)
    base_vec = base_scores[args.base_col]
    w_eegs = np.arange(0.0, args.w_eeg_max + 1e-9, args.w_eeg_step)
    print(f"[data] base_col={args.base_col} w_eegs={list(np.round(w_eegs,2))}", flush=True)

    # ---------------- PHASE A: fidelity check ----------------
    print("\n=== PHASE A: reproduce published base at fixed C=0.25, top_k=320 ===", flush=True)
    preds_a = {}
    for s in SEEDS:
        p, _, _ = bm.run_restricted_subject_fusion_oof(
            X, y, base_scores, args.base_col, args.penalty, 0.25, 320,
            w_eegs, args.objective, s, args.n_splits, args.inner_splits)
        preds_a[s] = p
        print(f"  seed {s}: AUROC={metrics(y,p)['AUROC']:.4f}", flush=True)
    bag_a = np.mean(list(preds_a.values()), axis=0)
    ma = metrics(y, bag_a)
    ok = abs(ma["AUROC"] - PUBLISHED_AUROC) < 1e-9 and abs(ma["AUPRC"] - PUBLISHED_AUPRC) < 1e-9
    print(f"  bagged AUROC={ma['AUROC']:.10f} (published {PUBLISHED_AUROC:.10f})", flush=True)
    print(f"  bagged AUPRC={ma['AUPRC']:.10f} (published {PUBLISHED_AUPRC:.10f})", flush=True)
    print(f"  FIDELITY: {'PASS ✅ 逐位重現' if ok else 'MISMATCH ❌'}", flush=True)

    # ---------------- PHASE B: honest fully-nested base ----------------
    print("\n=== PHASE B: nest (C, top_k, w_eeg) inside the inner loop ===", flush=True)
    t0 = time.time()
    preds_b, chosen = {}, []
    for s in SEEDS:
        p, ch = nested_fusion_oof(X, y, base_vec, args.penalty, CS, TOPKS,
                                  w_eegs, args.objective, s, args.n_splits, args.inner_splits)
        preds_b[s] = p
        chosen.extend(ch)
        print(f"  seed {s}: AUROC={metrics(y,p)['AUROC']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    bag_b = np.mean(list(preds_b.values()), axis=0)
    mb = metrics(y, bag_b)

    mi = metrics(y, base_vec)
    print("\n=== RESULT ===", flush=True)
    print(f"  image branch only (mi_max) : AUROC={mi['AUROC']:.4f} AUPRC={mi['AUPRC']:.4f}", flush=True)
    print(f"  published base (C,k fixed) : AUROC={ma['AUROC']:.4f} AUPRC={ma['AUPRC']:.4f}  CM={ma['TN']}/{ma['FP']}/{ma['FN']}/{ma['TP']}", flush=True)
    print(f"  HONEST fully-nested base   : AUROC={mb['AUROC']:.4f} AUPRC={mb['AUPRC']:.4f}  CM={mb['TN']}/{mb['FP']}/{mb['FN']}/{mb['TP']}", flush=True)
    print(f"  optimism removed           : AUROC {ma['AUROC']-mb['AUROC']:+.4f}   AUPRC {ma['AUPRC']-mb['AUPRC']:+.4f}", flush=True)

    import csv as _csv
    with (out / "honest_nested_base_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        wtr = _csv.writer(f)
        wtr.writerow(["subject_id", "true_label", "honest_nested_base_score",
                      "published_base_score", "mi_max_score"])
        for s_, yy, hb, pb, mm in zip(subjects, y, bag_b, bag_a, base_vec):
            wtr.writerow([s_, int(yy), float(hb), float(pb), float(mm)])
    with (out / "selected_configs_per_fold.csv").open("w", newline="", encoding="utf-8") as f:
        wtr = _csv.DictWriter(f, fieldnames=["seed", "fold", "C", "top_k", "w_eeg", "inner_obj"])
        wtr.writeheader()
        wtr.writerows(chosen)
    json.dump({
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "honest fully-nested base: (C, top_k, w_eeg) all selected inside inner 5-fold",
        "fidelity_check": {"passed": bool(ok), "reproduced_AUROC": ma["AUROC"],
                           "published_AUROC": PUBLISHED_AUROC},
        "grid": {"C": CS, "top_k": TOPKS, "w_eeg": [float(v) for v in w_eegs]},
        "seeds": SEEDS,
        "image_branch_mi_max": mi,
        "published_base_fixed_config": ma,
        "honest_nested_base": mb,
        "eeg_audit": eeg_audit,
    }, (out / "manifest.json").open("w"), indent=2, default=float)
    print(f"\n[out] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
