#!/usr/bin/env python3
"""Table 2 statistics for the truly nested run (pure, fold-aligned image branch).

Same estimator and the same stratified case-level bootstrap as the previous run
(B = 10000, rng seed 20260710) so the two are directly comparable.
Writes analysis/nsc114_true_nested_20260711/table2_true_nested.json
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, roc_auc_score)

OUT = Path(__file__).resolve().parents[1] / "analysis/nsc114_true_nested_top3_grid8_20260710"
B, THR, SEED = 10000, 0.50, 20260710


def all_metrics(y, s):
    tn, fp, fn, tp = confusion_matrix(y, (s >= THR).astype(int), labels=[0, 1]).ravel()
    return {
        "AUROC": roc_auc_score(y, s), "AUPRC": average_precision_score(y, s),
        "Accuracy": (tn + tp) / len(y),
        "Sensitivity": tp / max(tp + fn, 1), "Specificity": tn / max(tn + fp, 1),
        "PPV": tp / max(tp + fp, 1), "NPV": tn / max(tn + fn, 1),
        "Brier": brier_score_loss(y, np.clip(s, 0, 1)),
        "CM": [int(tn), int(fp), int(fn), int(tp)],
    }


def main() -> int:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    OUT = args.out_dir.resolve()
    rows = list((OUT / "true_nested_predictions.csv").open(encoding="utf-8-sig"))
    r = list(csv.DictReader(rows))
    y = np.array([int(float(x["true_label"])) for x in r])
    f = np.array([float(x["fused_score"]) for x in r])
    b = np.array([float(x["base_score"]) for x in r])
    m = np.array([float(x["image_score"]) for x in r])

    pt, base, img = all_metrics(y, f), all_metrics(y, b), all_metrics(y, m)

    rng = np.random.default_rng(SEED)
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    keys = ["AUROC", "AUPRC", "Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "Brier"]
    acc = {k: [] for k in keys}
    bacc = {k: [] for k in ("AUROC", "AUPRC")}
    dR, dP = [], []
    for _ in range(B):
        idx = np.concatenate([rng.choice(i0, len(i0), True), rng.choice(i1, len(i1), True)])
        yy = y[idx]
        if yy.sum() in (0, len(yy)):
            continue
        mf = all_metrics(yy, f[idx])
        for k in keys:
            acc[k].append(mf[k])
        bacc["AUROC"].append(roc_auc_score(yy, b[idx]))
        bacc["AUPRC"].append(average_precision_score(yy, b[idx]))
        dR.append(mf["AUROC"] - bacc["AUROC"][-1])
        dP.append(mf["AUPRC"] - bacc["AUPRC"][-1])

    q = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
    ci = {k: q(acc[k]) for k in keys}
    bci = {k: q(bacc[k]) for k in bacc}
    dR, dP = np.asarray(dR), np.asarray(dP)
    delta = {
        "dAUROC": pt["AUROC"] - base["AUROC"], "dAUROC_CI": q(dR),
        "p_AUROC": float(2 * min((dR <= 0).mean(), (dR >= 0).mean())),
        "dAUPRC": pt["AUPRC"] - base["AUPRC"], "dAUPRC_CI": q(dP),
        "p_AUPRC": float(2 * min((dP <= 0).mean(), (dP >= 0).mean())),
    }

    res = {"fused": pt, "base": base, "image": img,
           "fused_CI": ci, "base_CI": bci, "delta": delta}
    json.dump(res, (OUT / "table2_true_nested.json").open("w"), indent=2, default=float)

    print(f"  image  AUROC {img['AUROC']:.4f}  AUPRC {img['AUPRC']:.4f}")
    print(f"  base   AUROC {base['AUROC']:.4f} [{bci['AUROC'][0]:.3f},{bci['AUROC'][1]:.3f}]  "
          f"AUPRC {base['AUPRC']:.4f} [{bci['AUPRC'][0]:.3f},{bci['AUPRC'][1]:.3f}]  CM={base['CM']}")
    print(f"  fused  AUROC {pt['AUROC']:.4f} [{ci['AUROC'][0]:.3f},{ci['AUROC'][1]:.3f}]  "
          f"AUPRC {pt['AUPRC']:.4f} [{ci['AUPRC'][0]:.3f},{ci['AUPRC'][1]:.3f}]  CM={pt['CM']}")
    print(f"  dAUROC {delta['dAUROC']:+.4f} [{delta['dAUROC_CI'][0]:+.4f},{delta['dAUROC_CI'][1]:+.4f}] "
          f"p={delta['p_AUROC']:.3f}")
    print(f"  dAUPRC {delta['dAUPRC']:+.4f} [{delta['dAUPRC_CI'][0]:+.4f},{delta['dAUPRC_CI'][1]:+.4f}] "
          f"p={delta['p_AUPRC']:.3f}")
    print(f"  Acc {pt['Accuracy']:.4f}  Sens {pt['Sensitivity']:.4f}  Spec {pt['Specificity']:.4f}  "
          f"PPV {pt['PPV']:.4f}  NPV {pt['NPV']:.4f}  Brier {pt['Brier']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
