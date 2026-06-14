"""Validation du pipeline ICL TabICL — protocole holdout stratifié multi-graines.

1. Holdout stratifié : 2000 exemples par cellule (6 cellules = bucket × genre)
   = 12000 images de validation, tirés avec la graine de holdout.
2. Contextes ICL (3 tirages, seeds 1000/1001/1002) tirés EXCLUSIVEMENT dans les
   ~88k restants — zéro fuite vers la validation.
3. Répété sur 5 graines de holdout (7, 13, 21, 35, 49) → 5 scores appariés.
   Une amélioration n'est retenue que si elle gagne sur la majorité des graines
   (comparaison appariée, même graine).

Repondération vers le test : le holdout est uniforme en buckets (stratifié) ;
chaque score est recalculé sous 3 mélanges de buckets via des poids ω par image :
  - uniform   : brut (1/3 par bucket — le mélange du holdout stratifié)
  - train-mix : proportions du train (B0 .502 / B1 .384 / B2 .114)
  - brief     : proportions officielles du test (task_brief : B0 .14 / B1 .45 / B2 .41)
(le proxy par propagation d'identités peut être branché via TARGET_MIXES,
 diagnostic seulement)

Scores par graine + moyenne/écart-type, sauvés en JSON pour comparaison appariée
de configs futures.

Usage : python p7_validation.py [--config-name tabicl_pca100_3draws]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from p6_tabicl_icl import (ROOT, SEEDS as CTX_SEEDS, N_PER_CELL, N_PCA, W0, EPS,
                           build_matrices, weighted_decile_edges, bayes_decode)

HOLDOUT_SEEDS = [7, 13, 21, 35, 49]
N_VAL_PER_CELL = 2000          # 6 cellules × 2000 = 12000 images de validation

TARGET_MIXES = {
    "uniform": None,                                   # mélange brut du holdout
    "train-mix": {0: 0.5023, 1: 0.3835, 2: 0.1142},    # proportions du train
    "brief": {0: 0.14, 1: 0.45, 2: 0.41},              # officiel task_brief
}


def cells_of(tr_meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bucket = tr_meta["bucket"].map({"B0": 0, "B1": 1, "B2": 2}).to_numpy()
    gender = tr_meta["gender"].to_numpy().astype(int)
    return bucket, gender, bucket + 3 * gender


def stratified_draw(cell: np.ndarray, n_per_cell: int, rng,
                    allowed: np.ndarray | None = None) -> np.ndarray:
    out = []
    for c in range(6):
        idx = np.flatnonzero((cell == c) if allowed is None
                             else (cell == c) & allowed)
        out.append(rng.choice(idx, n_per_cell, replace=False))
    return np.concatenate(out)


def weighted_metric(p, gt, gender, omega) -> float:
    """Métrique de l'énoncé sous repondération ω par image."""
    w = W0 + gt
    errs = []
    for g in (0, 1):
        m = gender == g
        ww = omega[m] * w[m]
        errs.append((ww * (p[m] - gt[m]) ** 2).sum() / ww.sum())
    eF, eM = errs
    return (eF + eM) / 2 + abs(eF - eM)


def scores_all_mixes(p, gt, gender, bucket) -> dict[str, float]:
    out = {}
    for name, mix in TARGET_MIXES.items():
        if mix is None:
            omega = np.ones_like(gt)
        else:
            # part observée de chaque bucket DANS le holdout (par construction 1/3)
            obs = {b: (bucket == b).mean() for b in (0, 1, 2)}
            omega = np.array([mix[b] / obs[b] for b in bucket])
        out[name] = weighted_metric(p, gt, gender, omega)
    return out


def run_pipeline(Xtr, ed, y_all, cell, ctx_allowed, val_idx, ctx_seeds):
    """Pipeline ICL identique à p6, contextes restreints à ctx_allowed,
    prédiction sur val_idx. Retourne la moyenne des tirages."""
    from sklearn.decomposition import PCA
    from tabicl import TabICLClassifier

    preds = []
    for seed in ctx_seeds:
        rng = np.random.default_rng(seed)
        ci = stratified_draw(cell, N_PER_CELL, rng, allowed=ctx_allowed)
        y_ctx = y_all[ci]
        Xc, Xv = Xtr[ci].copy(), Xtr[val_idx].copy()

        mu = Xc[:, ed:].mean(axis=0)
        sd = np.maximum(Xc[:, ed:].std(axis=0), EPS)
        for M in (Xc, Xv):
            M[:, ed:] = (M[:, ed:] - mu) / sd

        pca = PCA(n_components=N_PCA, random_state=seed).fit(Xc[:, :ed])
        Zc = np.hstack([pca.transform(Xc[:, :ed]), Xc[:, ed:]])
        Zv = np.hstack([pca.transform(Xv[:, :ed]), Xv[:, ed:]])

        edges = weighted_decile_edges(y_ctx)
        b = np.digitize(y_ctx, edges)
        clf = TabICLClassifier(n_estimators=8, random_state=seed)
        clf.fit(Zc, b)
        preds.append(bayes_decode(clf.predict_proba(Zv), clf.classes_, b, y_ctx))
    return np.mean(preds, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="tabicl_pca100_3draws")
    args = ap.parse_args()

    tr_meta = pd.read_parquet(ROOT / "data/labels_trainfull.parquet")
    y_all = tr_meta["FaceOcclusion"].to_numpy(dtype=np.float64)
    bucket, gender, cell = cells_of(tr_meta)
    Xtr, _, ed = build_matrices()
    print(f"Xtr {Xtr.shape}, ed={ed}", flush=True)

    results = {}
    for h in HOLDOUT_SEEDS:
        t0 = time.perf_counter()
        val_idx = stratified_draw(cell, N_VAL_PER_CELL, np.random.default_rng(h))
        in_val = np.zeros(len(y_all), bool)
        in_val[val_idx] = True
        p = run_pipeline(Xtr, ed, y_all, cell, ~in_val, val_idx, CTX_SEEDS)
        sc = scores_all_mixes(p, y_all[val_idx], gender[val_idx], bucket[val_idx])
        results[h] = sc
        print(f"[holdout {h:2d}] " + "  ".join(f"{k}={v:.5f}" for k, v in sc.items())
              + f"  ({time.perf_counter() - t0:.0f}s)", flush=True)

    print("\n── synthèse (5 graines) " + "─" * 40)
    summary = {}
    for mix in TARGET_MIXES:
        vals = np.array([results[h][mix] for h in HOLDOUT_SEEDS])
        summary[mix] = {"mean": vals.mean(), "std": vals.std(),
                        "per_seed": dict(zip(map(str, HOLDOUT_SEEDS), vals))}
        print(f"{mix:10s}: {vals.mean():.5f} ± {vals.std():.5f}   "
              + " ".join(f"{v:.5f}" for v in vals))

    out = ROOT / "validation"
    out.mkdir(exist_ok=True)
    path = out / f"scores_{args.config_name}.json"
    json.dump({"config": args.config_name, "protocol":
               {"n_val_per_cell": N_VAL_PER_CELL, "holdout_seeds": HOLDOUT_SEEDS,
                "ctx_seeds": CTX_SEEDS, "mixes": {k: v for k, v in TARGET_MIXES.items()}},
               "summary": summary}, open(path, "w"), indent=1)
    print(f"\nécrit {path}", flush=True)


if __name__ == "__main__":
    main()
