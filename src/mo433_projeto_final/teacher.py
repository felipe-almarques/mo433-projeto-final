"""Professor (linha D): features de avaliação, checagem de sanidade e cache das K views.

As features do professor para cada view do pool são calculadas UMA vez (fp16) e
reaproveitadas por todas as execuções de C; esse tempo é o custo de GPU do professor
durante a destilação e é cobrado de C no relatório.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch

from .config import TEACHER_PRETRAIN_GPU_HOURS, Config, count_params, get_device, save_json, sync
from .data import extract, load_split, save_features, to_teacher
from .evaluate import sanity_check
from .models import build_teacher

log = logging.getLogger("mo433.teacher")
BATCH = 128


def run_teacher(cfg: Config) -> None:
    device = get_device()
    teacher = build_teacher(cfg.teacher_name).to(device)

    def pre(x: torch.Tensor) -> torch.Tensor:
        return to_teacher(x, cfg.teacher_res)

    # linha D + checagem de sanidade (se reprovar, nada depois faria sentido)
    X_tr, y_tr = load_split(cfg, "train")
    X_te, y_te = load_split(cfg, "test")
    sync(device)
    t0 = time.perf_counter()
    F_tr = extract(teacher, X_tr, pre, device, batch_size=BATCH, amp_dtype=torch.float16)
    F_te = extract(teacher, X_te, pre, device, batch_size=BATCH, amp_dtype=torch.float16)
    sync(device)
    eval_seconds = time.perf_counter() - t0
    save_features(cfg.results_dir / "features" / "D_teacher.npz", F_tr, F_te)
    sanity = sanity_check(F_tr, y_tr, F_te, y_te, "teacher")
    sanity.update(checkpoint=cfg.teacher_name, params=count_params(teacher), feat_dim=int(F_tr.shape[1]),
                  pooling="CLS token (timm global_pool='token')", input_res=cfg.teacher_res,
                  eval_seconds=eval_seconds)
    save_json(sanity, cfg.results_dir / "teacher_sanity.json")
    if not sanity["passed"]:
        raise RuntimeError(f"Professor reprovado na checagem de sanidade: {sanity['problems']}")

    # cache das features do professor para as K views de cada imagem do pool
    out = cfg.cache_dir / "teacher_view_feats.npy"
    if out.exists():
        log.info("teacher_view_feats.npy já existe — pulando.")
        return
    views = np.load(cfg.cache_dir / "views.npy", mmap_mode="r")
    n, k = views.shape[:2]
    tmp = out.with_suffix(".tmp.npy")
    feats = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(n, k, teacher.num_features))
    sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, n, BATCH):
            chunk = torch.from_numpy(np.ascontiguousarray(views[start:start + BATCH]))  # (b, K, H, W, 3)
            b = chunk.shape[0]
            x = pre(chunk.reshape(b * k, *chunk.shape[2:]).to(device))
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                feats[start:start + b] = teacher(x).float().reshape(b, k, -1).cpu().numpy().astype(np.float16)
    sync(device)
    cache_seconds = time.perf_counter() - t0
    feats.flush()
    del feats
    tmp.replace(out)

    cost = {"teacher_cache_seconds": cache_seconds, "teacher_cache_gpu_hours": cache_seconds / 3600,
            "teacher_eval_seconds": eval_seconds, "n_images": n, "n_views": k,
            "teacher_imgs_per_s": n * k / cache_seconds,
            "teacher_pretrain_gpu_hours_published": TEACHER_PRETRAIN_GPU_HOURS}
    save_json(cost, cfg.results_dir / "teacher_cost.json")
    log.info("Custo do professor: %s", cost)
