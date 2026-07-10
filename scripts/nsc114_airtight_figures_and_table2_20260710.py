#!/usr/bin/env python3
"""Regenerate aggregate figures and Table 2 from the current nested-fusion run."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "analysis/nsc114_airtight_fully_nested_20260710"
THR = 0.50


def auroc(y, s):
    p, n = s[y == 1], s[y == 0]
    if len(p) == 0 or len(n) == 0:
        return np.nan
    a = np.concatenate([p, n]); o = a.argsort()
    r = np.empty(len(o), float); r[o] = np.arange(1, len(o) + 1)
    u, inv, c = np.unique(a, return_inverse=True, return_counts=True)
    for i, cc in enumerate(c):
        if cc > 1:
            m = inv == i; r[m] = r[m].mean()
    return (r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))


def auprc(y, s):
    o = np.argsort(-s); yy = y[o]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
    pr = tp / (tp + fp); rc = tp / max(yy.sum(), 1)
    ap = 0.0; prev = 0.0
    for a_, b_ in zip(pr, rc):
        ap += a_ * (b_ - prev); prev = b_
    return ap


def all_metrics(y, s):
    p = (s >= THR).astype(int)
    TN = int(((y == 0) & (p == 0)).sum()); FP = int(((y == 0) & (p == 1)).sum())
    FN = int(((y == 1) & (p == 0)).sum()); TP = int(((y == 1) & (p == 1)).sum())
    return {
        "AUROC": auroc(y, s), "AUPRC": auprc(y, s),
        "Accuracy": (TN + TP) / len(y),
        "Sensitivity": TP / (TP + FN) if TP + FN else np.nan,
        "Specificity": TN / (TN + FP) if TN + FP else np.nan,
        "PPV": TP / (TP + FP) if TP + FP else np.nan,
        "NPV": TN / (TN + FN) if TN + FN else np.nan,
        "Brier": float(np.mean((s - y) ** 2)),
        "CM": [TN, FP, FN, TP],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()

    out = args.run_dir.resolve()
    fig = out / "figures"
    rows = list(csv.DictReader((out / "airtight_predictions.csv").open(encoding="utf-8-sig")))
    y = np.array([int(float(r["true_label"])) for r in rows])
    f = np.array([float(r["fused_score"]) for r in rows])
    b = np.array([float(r["base_score"]) for r in rows])

    pt = all_metrics(y, f)
    base = all_metrics(y, b)

    # stratified case-level bootstrap CIs
    rng = np.random.default_rng(20260710)
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    keys = ["AUROC", "AUPRC", "Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "Brier"]
    acc = {k: [] for k in keys}
    dR, dP = [], []
    for _ in range(args.bootstrap):
        idx = np.concatenate([rng.choice(i0, len(i0), True), rng.choice(i1, len(i1), True)])
        yy = y[idx]
        if yy.sum() in (0, len(yy)):
            continue
        m = all_metrics(yy, f[idx])
        for k in keys:
            acc[k].append(m[k])
        dR.append(auroc(yy, f[idx]) - auroc(yy, b[idx]))
        dP.append(auprc(yy, f[idx]) - auprc(yy, b[idx]))
    ci = {k: (float(np.nanpercentile(acc[k], 2.5)), float(np.nanpercentile(acc[k], 97.5))) for k in keys}
    dR, dP = np.array(dR), np.array(dP)
    delta = {
        "dAUROC": pt["AUROC"] - base["AUROC"],
        "dAUROC_CI": [float(np.percentile(dR, 2.5)), float(np.percentile(dR, 97.5))],
        "p_AUROC": float(2 * min((dR <= 0).mean(), (dR >= 0).mean())),
        "dAUPRC": pt["AUPRC"] - base["AUPRC"],
        "dAUPRC_CI": [float(np.percentile(dP, 2.5)), float(np.percentile(dP, 97.5))],
        "p_AUPRC": float(2 * min((dP <= 0).mean(), (dP >= 0).mean())),
    }

    print("=== TABLE 2 (current nested fusion) ===")
    for k in keys:
        lo, hi = ci[k]
        print(f"  {k:12s} {pt[k]:.4f}  95% CI {lo:.3f}-{hi:.3f}")
    print(f"  CM (thr .50) {pt['CM']}")
    print(f"  base AUROC {base['AUROC']:.4f}  base AUPRC {base['AUPRC']:.4f}")
    print(f"  dAUROC {delta['dAUROC']:+.4f} CI [{delta['dAUROC_CI'][0]:+.4f},{delta['dAUROC_CI'][1]:+.4f}] p={delta['p_AUROC']:.3f}")
    print(f"  dAUPRC {delta['dAUPRC']:+.4f} CI [{delta['dAUPRC_CI'][0]:+.4f},{delta['dAUPRC_CI'][1]:+.4f}] p={delta['p_AUPRC']:.3f}")

    json.dump({"fused": pt, "base": base, "fused_CI": ci, "delta": delta},
              (out / "table2_airtight.json").open("w"), indent=2, default=float)

    # ---------------- figures ----------------
    fig.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y, f)
    plt.figure(figsize=(5.5, 4.2), dpi=180)
    plt.plot(fpr, tpr, label=f"Nested fusion AUROC = {pt['AUROC']:.3f}", linewidth=2)
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("Nested base + signal-interaction fusion ROC")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(fig / "nested_fusion_roc.png"); plt.close()

    precision, recall, _ = precision_recall_curve(y, f)
    prev = y.mean()
    plt.figure(figsize=(5.5, 4.2), dpi=180)
    plt.plot(recall, precision, label=f"Nested fusion AUPRC = {pt['AUPRC']:.3f}", linewidth=2)
    plt.axhline(prev, linestyle="--", color="gray", linewidth=1, label=f"Prevalence = {prev:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Nested base + signal-interaction fusion PRC")
    plt.legend(loc="lower left"); plt.tight_layout()
    plt.savefig(fig / "nested_fusion_prc.png"); plt.close()

    cm = confusion_matrix(y, (f >= THR).astype(int))
    plt.figure(figsize=(4.6, 4.0), dpi=180)
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion matrix at threshold 0.50")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"]); plt.yticks([0, 1], ["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    plt.colorbar(fraction=0.046, pad=0.04); plt.tight_layout()
    plt.savefig(fig / "nested_fusion_cm.png"); plt.close()

    sm = list(csv.DictReader((out / "seed_metrics.csv").open(encoding="utf-8-sig")))
    seeds = [r["seed"] for r in sm]
    fa = [float(r["fused_AUROC"]) for r in sm]
    fp_ = [float(r["fused_AUPRC"]) for r in sm]
    x = np.arange(len(sm))
    plt.figure(figsize=(6.6, 4.2), dpi=180)
    plt.plot(x, fa, marker="o", label="Seed AUROC")
    plt.plot(x, fp_, marker="o", label="Seed AUPRC")
    plt.axhline(base["AUROC"], linestyle="--", color="#3366aa", linewidth=1,
                label=f"Base AUROC = {base['AUROC']:.3f}")
    plt.axhline(base["AUPRC"], linestyle="--", color="#aa6633", linewidth=1,
                label=f"Base AUPRC = {base['AUPRC']:.3f}")
    plt.xticks(x, seeds, rotation=45, ha="right", fontsize=8)
    plt.ylabel("Metric")
    lo = min(min(fa), min(fp_), base["AUROC"], base["AUPRC"]) - 0.02
    hi = max(max(fa), max(fp_)) + 0.02
    plt.ylim(lo, hi)
    plt.title("Seed-level nested-fusion OOF stability")
    plt.legend(fontsize=8, ncol=2); plt.tight_layout()
    plt.savefig(fig / "nested_fusion_seed_stability.png"); plt.close()

    print(f"\n[figures] {fig}")
    for p in sorted(fig.glob('*.png')):
        print(f"  {p.name}  {p.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
