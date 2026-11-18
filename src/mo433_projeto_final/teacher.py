"""Etapa do professor (Seção 3.3): linha D, checagem de sanidade e cache das views.

1. Carrega o ViT-S/16 DINO do ``timm`` (congelado, ``eval()``, fp16, sem gradiente).
2. Extrai as features de avaliação (treino rotulado + teste) com o pré-processamento
   determinístico do professor (upsample bicúbico 224 px + normalização ImageNet).
   Elas são as features da **linha D**.
3. **Checagem de sanidade** (k-NN, std dos embeddings, NMI do k-means). Se o
   professor for reprovado, a etapa falha: nenhum número posterior faria sentido.
4. Calcula, UMA vez, a feature do professor para cada uma das K views de cada
   imagem do pool (``teacher_view_feats.npy``, fp16). A destilação (C) lê esse
   cache em todas as sementes, tamanhos e na ablação. O tempo gasto aqui é o
   custo de GPU do professor durante a destilação e é cobrado de C no relatório.
"""

from __future__ import annotations

import numpy as np
import torch

from .config import ExperimentConfig
from .data import load_split, to_teacher
from .diagnostics import sanity_check
from .features import extract, save_features, teacher_preprocess
from .models import build_teacher
from .utils import count_parameters, get_device, get_logger, save_json, timer

log = get_logger("teacher")


def run_teacher_stage(cfg: ExperimentConfig, pretrained: bool = True) -> None:
    device = get_device()
    teacher = build_teacher(cfg.teacher_name, pretrained=pretrained).to(device)
    feats_path = cfg.results_dir / "features" / "D_teacher.npz"

    # ---------------------------------------------- linha D + sanidade
    X_tr, y_tr = load_split(cfg, "train")
    X_te, y_te = load_split(cfg, "test")
    pre = teacher_preprocess(cfg.teacher_res)
    with timer(log, "features de avaliação do professor (linha D)") as t_eval:
        F_tr = extract(teacher, X_tr, pre, device, batch_size=128, amp_dtype=torch.float16)
        F_te = extract(teacher, X_te, pre, device, batch_size=128, amp_dtype=torch.float16)
    save_features(feats_path, F_tr, F_te)
    log.info("Features D: treino %s, teste %s (pooling = token CLS, dim=%d)", F_tr.shape, F_te.shape, F_tr.shape[1])

    sanity = sanity_check(F_tr, y_tr, F_te, y_te, "teacher")
    sanity.update(checkpoint=cfg.teacher_name, params=count_parameters(teacher), feat_dim=int(F_tr.shape[1]),
                  pooling="CLS token (timm global_pool='token')", input_res=cfg.teacher_res,
                  eval_seconds=t_eval["seconds"])
    save_json(sanity, cfg.results_dir / "teacher_sanity.json")
    if not sanity["passed"] and cfg.dataset != "smoke":
        raise RuntimeError(f"Professor reprovado na checagem de sanidade: {sanity['problems']}")

    # ------------------------------------------- cache das K views do pool
    out = cfg.cache_dir / "teacher_view_feats.npy"
    if out.exists():
        log.info("teacher_view_feats.npy já existe — pulando.")
        return
    views = np.load(cfg.cache_dir / "views.npy", mmap_mode="r")
    n, k = views.shape[:2]
    feats = np.lib.format.open_memmap(out.with_suffix(".tmp.npy"), mode="w+", dtype=np.float16,
                                      shape=(n, k, teacher.num_features))
    bs = 128
    with timer(log, f"cache do professor: {n} imagens x {k} views a {cfg.teacher_res}px (fp16)") as t_cache, \
            torch.no_grad():
        for start in range(0, n, bs):
            chunk = torch.from_numpy(np.ascontiguousarray(views[start:start + bs]))  # (b, K, H, W, 3)
            b = chunk.shape[0]
            x = to_teacher(chunk.reshape(b * k, *chunk.shape[2:]).to(device), cfg.teacher_res)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                f = teacher(x)
            feats[start:start + b] = f.float().reshape(b, k, -1).cpu().numpy().astype(np.float16)
            if (start // bs) % 20 == 0:
                log.info("  %d/%d imagens", start + b, n)
    feats.flush()
    del feats
    out.with_suffix(".tmp.npy").replace(out)

    cost = {"teacher_cache_seconds": t_cache["seconds"], "teacher_cache_gpu_hours": t_cache["seconds"] / 3600,
            "teacher_eval_seconds": t_eval["seconds"], "n_images": n, "n_views": k,
            "teacher_imgs_per_s": n * k / t_cache["seconds"],
            "teacher_pretrain_gpu_hours_published": cfg.teacher_pretrain_gpu_hours}
    save_json(cost, cfg.results_dir / "teacher_cost.json")
    log.info("Custo do professor: %s", cost)
