"""Extraction de features FIGÉES — 4 modèles publics, inférence pure, train/test séparés.

Modèles (conventions de préprocessing publiées, respectées à l'identique) :
  effnet   : timm tf_efficientnetv2_m.in21k_ft_in1k, resize 224, norm ImageNet
             (mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225] — valeurs de l'énoncé),
             embedding global pooled pré-tête → (N, 1280)
  faceptor : Faceptor officiel (repo lxq1000/Faceptor, stage_1 6 tâches,
             checkpoint_rank0_iter_50000.pth.tar) — encodeur visuel CLIP ViT-B/16
             extrait du checkpoint multi-tâches, embedding positionnel 32×32
             interpolé en bicubique vers 14×14, inférence 224 norm CLIP,
             CLS visuel projeté → (N, 512)
  clibfiqa : CLIB-FIQA RN50 (repo oufuzhao/CLIB-FIQA), convention du inference.py
             officiel : Resize([224,224]) + norm CLIP (le RN50 CLIP n'accepte pas 112),
             softmax joint sur la grille de prompts blur×occ×pose×exp×ill×quality,
             marginalisé → [P(obstructed), quality] → (N, 2)
  parsing  : SegFormer jonathandinu/face-parsing (CelebAMask-HQ, 19 classes),
             norm ImageNet, inféré à 512×512, argmax par pixel → 47 features → (N, 47)

Chaque image est traitée indépendamment (aucune statistique dataset). Ordre des
lignes = ordre EXACT de la colonne 'filename' du split. Train et test ne sont
JAMAIS mélangés : listes, runs et fichiers de sortie distincts.

Usage :
  python extract.py --model effnet --split trainfull --device cuda:0 [--limit N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

warnings.filterwarnings("ignore")

ROOT = Path(os.environ.get("ICL_ROOT", Path(__file__).resolve().parent.parent))
CROPS = Path(os.environ.get("CROPS_DIR", "/root/crops/Crop_224_5fp_100K"))
OUT = ROOT / "features"
CLIB = Path(os.environ.get("CLIB_DIR", ROOT / "external/CLIB-FIQA"))
FACEPTOR = Path(os.environ.get("FACEPTOR_CKPT", ROOT / "external/faceptor_checkpoint_rank0_iter_50000.pth.tar"))
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Nom de fichier de sortie par (modèle, split) — noms exigés par l'énoncé.
OUT_NAME = {
    ("effnet", "trainfull"): "effnetv2m_image_trainfull.npy",
    ("effnet", "test29980"): "effnetv2m_image_test29980.npy",
    ("faceptor", "trainfull"): "faceptor512_image_trainfull.npy",
    ("faceptor", "test29980"): "faceptor512_image_test29980.npy",
    ("clibfiqa", "trainfull"): "clibfiqa_trainfull.npy",
    ("clibfiqa", "test29980"): "clibfiqa_test29980.npy",
    ("parsing", "trainfull"): "parsing2_feats_trainfull.npy",
    ("parsing", "test29980"): "parsing2_feats_test.npy",
}


def load_filenames(split: str) -> list[str]:
    if split == "trainfull":
        return pd.read_parquet(ROOT / "data/labels_trainfull.parquet")["filename"].tolist()
    if split == "test29980":
        return pd.read_csv(ROOT / "data/test_students.csv")["filename"].tolist()
    raise ValueError(split)


class ImgDataset(torch.utils.data.Dataset):
    """Charge un crop 224×224, retourne un tenseur normalisé."""

    def __init__(self, files, mean, std, size=224, interp=Image.BICUBIC):
        self.files = files
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(3, 1, 1)
        self.size = size
        self.interp = interp

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(CROPS / self.files[i]).convert("RGB")
        if img.size != (self.size, self.size):
            img = img.resize((self.size, self.size), self.interp)
        x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy((x - self.mean) / self.std)


def make_loader(files, mean, std, bs, workers=16):
    return torch.utils.data.DataLoader(
        ImgDataset(files, mean, std), batch_size=bs, shuffle=False,
        num_workers=workers, pin_memory=True, drop_last=False)


# ───────────────────────────── modèle 1 : EffNetV2-M ─────────────────────────
@torch.inference_mode()
def run_effnet(files, dev, bs=384):
    import timm
    model = timm.create_model("tf_efficientnetv2_m.in21k_ft_in1k",
                              pretrained=True, num_classes=0)
    model.eval().to(dev, memory_format=torch.channels_last)
    print(f"[effnet] resize 224, norm ImageNet {IMAGENET_MEAN}/{IMAGENET_STD}", flush=True)
    out = np.zeros((len(files), 1280), dtype=np.float32)
    i0 = 0
    for x in make_loader(files, IMAGENET_MEAN, IMAGENET_STD, bs):
        x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
        f = model(x)  # pooled pré-tête (num_classes=0) → (B, 1280)
        out[i0:i0 + x.size(0)] = f.float().cpu().numpy()
        i0 += x.size(0)
        if (i0 // bs) % 50 == 0:
            print(f"[effnet] {i0}/{len(files)}", flush=True)
    return out


# ──────────── modèle 2 : Faceptor (encodeur visuel CLIP ViT-B/16) ────────────
@torch.inference_mode()
def run_faceptor(files, dev, bs=384):
    sys.path.insert(0, str(CLIB))
    from model.models import VisionTransformer  # implémentation CLIP officielle (copie repo)
    ckpt = torch.load(FACEPTOR, map_location="cpu", weights_only=False)
    pfx = "module.backbone_module.visual."
    sd = {k.removeprefix(pfx): v for k, v in ckpt["state_dict"].items()
          if k.startswith(pfx)}
    # Faceptor est entraîné en 512×512 (pos emb 1025 = CLS + 32×32) ; on infère en
    # 224 (CLIP-native) : interpolation BICUBIQUE de la grille 32×32 → 14×14.
    pe = sd["positional_embedding"]                       # (1025, 768)
    cls_tok, grid = pe[:1], pe[1:]
    g = int(grid.shape[0] ** 0.5)
    grid = grid.reshape(g, g, -1).permute(2, 0, 1).unsqueeze(0)
    grid = F.interpolate(grid.float(), size=(14, 14), mode="bicubic",
                         align_corners=False)
    grid = grid.squeeze(0).permute(1, 2, 0).reshape(14 * 14, -1).to(pe.dtype)
    sd["positional_embedding"] = torch.cat([cls_tok, grid], dim=0)  # (197, 768)
    model = VisionTransformer(input_resolution=224, patch_size=16, width=768,
                              layers=12, heads=12, output_dim=512)
    model.load_state_dict(sd, strict=True)
    model = model.eval().to(dev)
    print("[faceptor] Faceptor stage_1 (6 tâches) chargé, pos emb 32×32→14×14 "
          "bicubique, CLS projeté 512", flush=True)
    out = np.zeros((len(files), 512), dtype=np.float32)
    i0 = 0
    for x in make_loader(files, CLIP_MEAN, CLIP_STD, bs):
        x = x.to(dev, non_blocking=True)
        f = model(x)  # CLS projeté → (B, 512)
        out[i0:i0 + x.size(0)] = f.float().cpu().numpy()
        i0 += x.size(0)
        if (i0 // bs) % 50 == 0:
            print(f"[faceptor] {i0}/{len(files)}", flush=True)
    return out


# ───────────────────────────── modèle 3 : CLIB-FIQA ──────────────────────────
@torch.inference_mode()
def run_clibfiqa(files, dev, bs=256):
    sys.path.insert(0, str(CLIB))
    from model import clip
    from utilities import dist_to_score, load_net_param
    QL = ["bad", "poor", "fair", "good", "perfect"]
    BL = ["hazy", "blurry", "clear"]
    OL = ["obstructed", "unobstructed"]
    PL = ["profile", "slight angle", "frontal"]
    EL = ["exaggerated expression", "typical expression"]
    IL = ["extreme lighting", "normal lighting"]
    joint = torch.cat([
        clip.tokenize(f"a photo of a {b}, {o}, and {p} face with {e} under "
                      f"{ll}, which is of {q} quality")
        for b, o, p, e, ll, q in product(BL, OL, PL, EL, IL, QL)]).to(dev)
    net, _ = clip.load(str(CLIB / "weights/RN50.pt"), device=dev, jit=False)
    net = load_net_param(net, str(CLIB / "weights/CLIB-FIQA_R50.pth")).eval()
    out = np.zeros((len(files), 2), dtype=np.float32)
    i0 = 0
    for x in make_loader(files, CLIP_MEAN, CLIP_STD, bs):
        x = x.to(dev, non_blocking=True)
        lpi, _ = net.forward(x, joint)
        lpi = F.softmax(lpi.view(x.size(0), -1), dim=1).view(
            -1, len(BL), len(OL), len(PL), len(EL), len(IL), len(QL))
        pocc = lpi.sum(6).sum(5).sum(4).sum(3).sum(1)[:, 0]  # P(obstructed)
        qual = dist_to_score(lpi.sum(1).sum(1).sum(1).sum(1).sum(1))
        out[i0:i0 + x.size(0), 0] = pocc.float().cpu().numpy()
        out[i0:i0 + x.size(0), 1] = qual.float().cpu().numpy()
        i0 += x.size(0)
        if (i0 // bs) % 50 == 0:
            print(f"[clibfiqa] {i0}/{len(files)}", flush=True)
    return out


# ─────────────────── modèle 4 : face parsing → 47 features ───────────────────
OCCLUDERS = [3, 13, 14, 18]
OCC_NONHAIR = [3, 14, 18]
VISIBLE = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]
EPS = 1e-9


def parsing_features(m: torch.Tensor) -> torch.Tensor:
    """m : (B, H, W) int64 ∈ [0,18] → (B, 47) float32. Tout est par image."""
    B, H, W = m.shape
    oh = F.one_hot(m, 19).permute(0, 3, 1, 2).float()        # (B,19,H,W)
    fr = oh.mean(dim=(2, 3))                                  # [1-19]
    occ = fr[:, OCCLUDERS].sum(1)
    vis = fr[:, VISIBLE].sum(1)
    den = occ + vis
    occ_in_face = torch.where(den > EPS, occ / den.clamp_min(EPS),
                              torch.zeros_like(den))          # [20]
    nonface = 1.0 - vis                                       # [21]
    ent = -(fr * fr.clamp_min(1e-12).log()).where(fr > 0, torch.zeros_like(fr)).sum(1)  # [22]
    frL = oh[:, :, :, : W // 2].mean(dim=(2, 3))
    frR = oh[:, :, :, W // 2:].mean(dim=(2, 3))
    asym = (frL - frR).abs().sum(1)                           # [23]
    occmask = oh[:, OCCLUDERS].sum(1)                         # (B,H,W) ∈ {0,1}
    rows = torch.arange(H, device=m.device, dtype=torch.float32).view(1, H, 1)
    cnt = occmask.sum(dim=(1, 2))
    s1 = (occmask * rows).sum(dim=(1, 2))
    mean_r = s1 / cnt.clamp_min(EPS)
    vcent = torch.where(cnt > 0, mean_r / H, torch.zeros_like(cnt))          # [24]
    s2 = (occmask * rows * rows).sum(dim=(1, 2))
    var = (s2 / cnt.clamp_min(EPS) - mean_r * mean_r).clamp_min(0)
    spread = torch.where(cnt > 0, var.sqrt() / H, torch.zeros_like(cnt))     # [25]
    hb = [0, H // 3, 2 * H // 3, H]
    wb = [0, W // 3, 2 * W // 3, W]
    hair = oh[:, 13]
    occnh = oh[:, OCC_NONHAIR].sum(1)
    hair_cells, occnh_cells = [], []
    for r in range(3):
        for c in range(3):
            sl = (slice(None), slice(hb[r], hb[r + 1]), slice(wb[c], wb[c + 1]))
            hair_cells.append(hair[sl].mean(dim=(1, 2)))      # [26-34]
            occnh_cells.append(occnh[sl].mean(dim=(1, 2)))    # [35-43]
    hair_tot = hair.sum(dim=(1, 2))
    f44 = torch.where(hair_tot > 0,
                      hair[:, :, wb[1]:wb[2]].sum(dim=(1, 2)) / hair_tot.clamp_min(EPS),
                      torch.zeros_like(hair_tot))
    f45 = torch.where(hair_tot > 0,
                      hair[:, hb[1]:hb[2], :].sum(dim=(1, 2)) / hair_tot.clamp_min(EPS),
                      torch.zeros_like(hair_tot))
    f46 = torch.where(cnt > 0,
                      occmask[:, hb[1]:hb[2], wb[1]:wb[2]].sum(dim=(1, 2)) / cnt.clamp_min(EPS),
                      torch.zeros_like(cnt))
    vismask = oh[:, VISIBLE].sum(1)
    f47 = vismask[:, hb[1]:hb[2], :].mean(dim=(1, 2))
    feats = torch.cat(
        [fr, occ_in_face[:, None], nonface[:, None], ent[:, None], asym[:, None],
         vcent[:, None], spread[:, None]]
        + [v[:, None] for v in hair_cells] + [v[:, None] for v in occnh_cells]
        + [f44[:, None], f45[:, None], f46[:, None], f47[:, None]], dim=1)
    assert feats.shape[1] == 47
    return feats


@torch.inference_mode()
def run_parsing(files, dev, bs=64):
    from transformers import SegformerForSemanticSegmentation
    model = SegformerForSemanticSegmentation.from_pretrained(
        "jonathandinu/face-parsing").eval().to(dev)
    out = np.zeros((len(files), 47), dtype=np.float32)
    i0 = 0
    for x in make_loader(files, IMAGENET_MEAN, IMAGENET_STD, bs):
        x = x.to(dev, non_blocking=True)
        x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(pixel_values=x).logits             # (B,19,128,128)
        # comme l'original : argmax à la résolution native, PUIS upsample nearest
        m_lowres = logits.float().argmax(1)
        m = F.interpolate(m_lowres.unsqueeze(1).float(), size=(512, 512),
                          mode="nearest").squeeze(1).long()   # carte de labels 512×512
        out[i0:i0 + x.size(0)] = parsing_features(m).cpu().numpy()
        i0 += x.size(0)
        if (i0 // bs) % 100 == 0:
            print(f"[parsing] {i0}/{len(files)}", flush=True)
    return out


RUNNERS = {"effnet": run_effnet, "faceptor": run_faceptor,
           "clibfiqa": run_clibfiqa, "parsing": run_parsing}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(RUNNERS))
    ap.add_argument("--split", required=True, choices=["trainfull", "test29980"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="smoke test : N premières images")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    files = load_filenames(args.split)
    if args.limit:
        files = files[: args.limit]
    name = OUT_NAME[(args.model, args.split)]
    out_path = OUT / (name if not args.limit else f"smoke_{name}")
    if out_path.exists():
        print(f"{out_path.name} existe déjà — skip", flush=True)
        return

    t0 = time.perf_counter()
    torch.cuda.set_device(torch.device(args.device))
    arr = RUNNERS[args.model](files, args.device)
    assert arr.shape[0] == len(files) and arr.dtype == np.float32
    np.save(out_path, arr)
    dt = time.perf_counter() - t0
    order_sha = hashlib.sha1("\n".join(files).encode()).hexdigest()[:16]
    meta = {"file": name, "model": args.model, "split": args.split,
            "n": len(files), "dims": int(arr.shape[1]),
            "order_sha1_16": order_sha, "seconds": round(dt, 1)}
    with open(OUT / f"{out_path.stem}.meta.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"DONE {name}: {arr.shape} en {dt:.0f}s  "
          f"mean={arr.mean():.4f}  order_sha={order_sha}", flush=True)


if __name__ == "__main__":
    main()
