"""Prédiction FaceOcclusion sur le test 29980 par IN-CONTEXT LEARNING (TabICL figé).

Aucun entraînement, aucune statistique dataset-entier : z-score, PCA, frontières
de buckets et décodage Bayes sont tous recalculés PAR TIRAGE sur les 1002
exemples du contexte uniquement.

Pipeline par tirage (seeds 1000/1001/1002) :
  contexte stratifié 167×6 cellules (bucket + 3×gender)
  → z-score parsing+clib (mu/sd du contexte seul)
  → PCA-100 sur les encodeurs seuls (ajustée sur le contexte)
  → buckets = déciles pondérés (w = 1/30 + y) des étiquettes du contexte
  → TabICLClassifier(n_estimators=8) sur (PCA|parsing|clib, bucket)
  → décodage Bayes-optimal sous la métrique, clip [0,1]
Prédiction finale = moyenne des 3 tirages → CSV (filename, FaceOcclusion, gender='x').
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("ICL_ROOT", Path(__file__).resolve().parent.parent))
FEAT = ROOT / "features"
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))

SEEDS = [1000, 1001, 1002]
N_PER_CELL = 167          # 6 cellules × 167 = 1002 exemples de contexte
N_PCA = 100
W0 = 1.0 / 30.0           # poids métrique w = 1/30 + y
EPS = 1e-12


def l2n(a: np.ndarray) -> np.ndarray:
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), EPS)


def build_matrices():
    """[faceptor_L2(512) | effnet_L2(1280) | parsing(47) | clib(2)] → ed = 1792."""
    blocks = {}
    for split, tag in [("trainfull", "trainfull"), ("test29980", "test29980")]:
        fac = l2n(np.load(FEAT / f"faceptor512_image_{split}.npy").astype(np.float64))
        eff = l2n(np.load(FEAT / f"effnetv2m_image_{split}.npy").astype(np.float64))
        par = np.load(FEAT / ("parsing2_feats_trainfull.npy" if split == "trainfull"
                              else "parsing2_feats_test.npy")).astype(np.float64)
        cli = np.load(FEAT / f"clibfiqa_{split}.npy").astype(np.float64)
        blocks[tag] = np.hstack([fac, eff, par, cli])
    ed = 512 + 1280
    return blocks["trainfull"], blocks["test29980"], ed


def context_indices(tr_meta: pd.DataFrame, seed: int) -> np.ndarray:
    """167 indices par cellule (cell = bucket(0/1/2) + 3×gender), sans remise."""
    bucket = tr_meta["bucket"].map({"B0": 0, "B1": 1, "B2": 2}).to_numpy()
    gender = tr_meta["gender"].to_numpy().astype(int)
    cell = bucket + 3 * gender
    rng = np.random.default_rng(seed)
    ci = np.concatenate([
        rng.choice(np.flatnonzero(cell == c), N_PER_CELL, replace=False)
        for c in range(6)])
    return ci


def weighted_decile_edges(y_ctx: np.ndarray) -> np.ndarray:
    """Frontières = valeurs de y aux déciles de la masse pondérée w = 1/30 + y."""
    order = np.argsort(y_ctx, kind="stable")
    ys = y_ctx[order]
    cw = np.cumsum(W0 + ys)
    pos = np.searchsorted(cw, np.arange(1, 10) / 10.0 * cw[-1])
    return np.unique(ys[np.clip(pos, 0, len(ys) - 1)])


def bayes_decode(P: np.ndarray, classes: np.ndarray,
                 b_ctx: np.ndarray, y_ctx: np.ndarray) -> np.ndarray:
    """p_i = Σ_k P[i,k]·E[w·y|k] / Σ_k P[i,k]·E[w|k], stats calculées sur le contexte."""
    w = W0 + y_ctx
    num = np.array([(w[b_ctx == k] * y_ctx[b_ctx == k]).mean() for k in classes])
    den = np.array([w[b_ctx == k].mean() for k in classes])
    return np.clip((P @ num) / np.maximum(P @ den, EPS), 0.0, 1.0)


def metric(p: np.ndarray, gt: np.ndarray, gender: np.ndarray) -> float:
    w = W0 + gt
    errs = []
    for g in (0, 1):
        m = gender == g
        errs.append((w[m] * (p[m] - gt[m]) ** 2).sum() / w[m].sum())
    eF, eM = errs
    return (eF + eM) / 2 + abs(eF - eM)


def main():
    from sklearn.decomposition import PCA
    from tabicl import TabICLClassifier

    # Checkpoint TabICL FIGÉ : version explicite (insensible aux futures versions de
    # tabicl), et usage du checkpoint bundlé en local si présent (reproductible hors-ligne).
    TABICL_VERSION = "tabicl-classifier-v2-20260212.ckpt"
    TABICL_LOCAL = os.environ.get("TABICL_CKPT", str(ROOT / "features" / TABICL_VERSION))
    tabicl_kwargs = dict(n_estimators=8, checkpoint_version=TABICL_VERSION)
    if Path(TABICL_LOCAL).exists():
        tabicl_kwargs["model_path"] = TABICL_LOCAL
        tabicl_kwargs["allow_auto_download"] = False
        print(f"TabICL : checkpoint local figé {TABICL_LOCAL}", flush=True)
    else:
        print("TabICL : checkpoint téléchargé depuis HF jingang/TabICL (version figée)", flush=True)

    tr_meta = pd.read_parquet(ROOT / "data/labels_trainfull.parquet")
    te_files = pd.read_csv(ROOT / "data/test_students.csv")["filename"]
    y_all = tr_meta["FaceOcclusion"].to_numpy(dtype=np.float64)
    Xtr, Xte, ed = build_matrices()
    assert Xtr.shape == (len(tr_meta), 1841) and Xte.shape == (len(te_files), 1841)
    print(f"Xtr {Xtr.shape}  Xte {Xte.shape}  ed={ed}", flush=True)

    # éval locale (pour info) : 4000 images train hors contexte, tirage fixe
    eval_idx = np.random.default_rng(42).choice(len(y_all), 4000, replace=False)

    preds, preds_eval = [], []
    for seed in SEEDS:
        ci = context_indices(tr_meta, seed)
        y_ctx = y_all[ci]
        Xc, Xt = Xtr[ci].copy(), Xte.copy()
        Xe = Xtr[eval_idx].copy()

        # z-score parsing+clib avec mu/sd du CONTEXTE SEUL
        mu = Xc[:, ed:].mean(axis=0)
        sd = np.maximum(Xc[:, ed:].std(axis=0), EPS)
        for M in (Xc, Xt, Xe):
            M[:, ed:] = (M[:, ed:] - mu) / sd

        # PCA-100 sur les encodeurs seuls, ajustée sur le contexte
        pca = PCA(n_components=N_PCA, random_state=seed).fit(Xc[:, :ed])
        Zc = np.hstack([pca.transform(Xc[:, :ed]), Xc[:, ed:]])
        Zt = np.hstack([pca.transform(Xt[:, :ed]), Xt[:, ed:]])
        Ze = np.hstack([pca.transform(Xe[:, :ed]), Xe[:, ed:]])
        assert Zc.shape[1] == N_PCA + 49

        edges = weighted_decile_edges(y_ctx)
        b = np.digitize(y_ctx, edges)
        print(f"[seed {seed}] contexte {len(ci)}, {len(edges)} frontières, "
              f"{len(np.unique(b))} classes", flush=True)

        clf = TabICLClassifier(random_state=seed, **tabicl_kwargs)
        clf.fit(Zc, b)
        P = clf.predict_proba(Zt)
        p = bayes_decode(P, clf.classes_, b, y_ctx)
        preds.append(p)

        pe = bayes_decode(clf.predict_proba(Ze), clf.classes_, b, y_ctx)
        preds_eval.append(pe)
        s = metric(pe, y_all[eval_idx], tr_meta["gender"].to_numpy()[eval_idx].astype(int))
        print(f"[seed {seed}] éval locale (4000 train hors contexte) : score={s:.5f}",
              flush=True)

    final = np.mean(preds, axis=0)
    s_ens = metric(np.mean(preds_eval, axis=0), y_all[eval_idx],
                   tr_meta["gender"].to_numpy()[eval_idx].astype(int))
    print(f"[ensemble 3 tirages] éval locale : score={s_ens:.5f}", flush=True)

    out = ROOT / "predictions"
    out.mkdir(exist_ok=True)
    sub = pd.DataFrame({"filename": te_files, "FaceOcclusion": final, "gender": "x"})
    sub.to_csv(out / "test_predictions_tabicl.csv", index=False)
    print(f"écrit {out / 'test_predictions_tabicl.csv'}  ({len(sub)} lignes)  "
          f"pred: min={final.min():.4f} max={final.max():.4f} mean={final.mean():.4f}",
          flush=True)


if __name__ == "__main__":
    main()
