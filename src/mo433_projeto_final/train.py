"""Treino dos alunos: linha B (VICReg direto) e linha C (destilação do ViT DINO).

As duas linhas usam o MESMO laço de treino (``train_student``); só muda a
função de perda e o projetor. Isso garante, por construção, o "mesmo orçamento"
da Seção 3.4:

* mesma arquitetura e mesma inicialização (``build_student(size, seed)``);
* mesmo pool de imagens, mesmas K views pré-geradas (mesmo aumento de dados);
* mesmo nº de épocas, mesmo batch, mesmo nº de passos do otimizador;
* mesmo otimizador (AdamW), mesmo agendamento (warm-up linear + cosseno),
  mesma precisão mista (bf16) e ``channels_last``.

Diferença intrínseca a cada método (documentada no README): a cada passo, B
processa **duas** views de cada imagem (o VICReg compara duas views), enquanto C
processa **uma** view (a mesma que foi dada ao professor — Seção 3.5). As
features do professor de cada view foram pré-computadas uma única vez
(``teacher.py``), então o custo do professor fica fora do laço de C e é
contabilizado à parte como "GPU-horas do professor".
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ExperimentConfig
from .data import load_split, to_student
from .diagnostics import sanity_check
from .features import extract, save_features, student_preprocess
from .models import build_student, distill_projector, vicreg_projector
from .utils import count_parameters, get_device, get_logger, save_json, set_seed

log = get_logger("train")


# -----------------------------------------------------------------------------
# Nomes e caminhos
# -----------------------------------------------------------------------------
def run_name(row: str, size: str, seed: int, data_frac: float = 1.0, tag: str = "") -> str:
    """Nome canônico de uma execução, ex.: ``B_small_seed0`` ou ``C_large_seed1_data50``."""
    name = f"{row}_{size}_seed{seed}"
    if data_frac < 1.0:
        name += f"_data{int(round(data_frac * 100))}"
    return name + (f"_{tag}" if tag else "")


def pool_indices(cfg: ExperimentConfig, n_pool: int, data_frac: float) -> np.ndarray:
    """Subconjunto do pool não rotulado usado na ablação (os mesmos índices p/ B, C e sementes).

    Os primeiros ``round(frac * N)`` elementos de uma permutação fixa: 50% ⊂ 100%.
    """
    perm = np.random.default_rng(cfg.split_seed + 123).permutation(n_pool)
    return np.sort(perm[: int(round(data_frac * n_pool))])


# -----------------------------------------------------------------------------
# Otimizador e agendamento
# -----------------------------------------------------------------------------
def make_optimizer(params: list[nn.Parameter], named: list[tuple[str, nn.Parameter]], lr: float,
                   wd: float) -> torch.optim.Optimizer:
    """AdamW sem weight decay em vieses e parâmetros de normalização (prática padrão)."""
    decay = [p for n, p in named if p.ndim > 1]
    no_decay = [p for n, p in named if p.ndim <= 1]
    assert len(decay) + len(no_decay) == len(params)
    return torch.optim.AdamW([{"params": decay, "weight_decay": wd},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.999))


def lr_at(step: int, total: int, warmup: int, base_lr: float) -> float:
    """Warm-up linear por ``warmup`` passos e depois decaimento cosseno até 0."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# -----------------------------------------------------------------------------
# Perdas
# -----------------------------------------------------------------------------
class VICRegObjective(nn.Module):
    """Perda VICReg do ``lightly`` (λ=25, μ=25, ν=1) + componentes para o log."""

    def __init__(self) -> None:
        super().__init__()
        from lightly.loss import VICRegLoss

        self.loss = VICRegLoss()

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        from lightly.loss.vicreg_loss import covariance_loss, invariance_loss, variance_loss

        z1, z2 = z1.float(), z2.float()  # perda sempre em fp32 (estabilidade numérica)
        loss = self.loss(z1, z2)
        with torch.no_grad():  # componentes (não ponderados) só para o log
            parts = {
                "invariance": invariance_loss(z1, z2).item(),
                "variance": 0.5 * (variance_loss(z1) + variance_loss(z2)).item(),
                "covariance": (covariance_loss(z1) + covariance_loss(z2)).item(),
            }
        return loss, parts


def cosine_distill_loss(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Perda de destilação padrão da Seção 3.5: 1 - cos(projetor(aluno), professor)."""
    return 1.0 - F.cosine_similarity(s.float(), t.float(), dim=-1).mean()


# -----------------------------------------------------------------------------
# Laço de treino comum a B e C
# -----------------------------------------------------------------------------
def train_student(
    cfg: ExperimentConfig,
    row: str,
    size: str,
    seed: int,
    data_frac: float = 1.0,
    epochs: int | None = None,
    lr: float | None = None,
    tag: str = "",
    out_root: Path | None = None,
    run_collapse_check: bool = True,
) -> dict[str, Any]:
    """Treina um aluno (linha ``row`` ∈ {"B", "C"}) e devolve o resumo da execução.

    Produz em ``<out_root>/<run_name>/``: ``checkpoint.pt`` (backbone + projetor),
    ``history.csv`` (perda por época) e ``summary.json`` (tempo, passos, colapso).
    As features de avaliação do backbone final vão para ``results/<ds>/features``.
    """
    assert row in ("B", "C"), row
    device = get_device()
    epochs = epochs or cfg.epochs
    lr = lr if lr is not None else cfg.lr_for(row)
    name = run_name(row, size, seed, data_frac, tag)
    out_dir = (out_root or cfg.results_dir / "runs") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    # ------------------------------------------------------------- dados
    # As views ficam em memory-map (não são copiadas para a RAM do processo): cada
    # batch lê só as imagens de que precisa e o SO faz o cache. Economiza ~2 GB de RAM.
    all_views = np.load(cfg.cache_dir / "views.npy", mmap_mode="r")   # (N_total, K, H, W, 3) uint8
    sel = pool_indices(cfg, all_views.shape[0], data_frac)
    log.info("[%s] usando %d imagens x %d views do pool (memory-map)", name, len(sel), all_views.shape[1])

    class _PoolView:
        """Indexa o subconjunto ``sel`` do pool sem copiá-lo: pool[i, k] -> all_views[sel[i], k]."""
        shape = (len(sel),) + all_views.shape[1:]

        def __getitem__(self, key):
            i, k = key
            order = np.argsort(sel[i])  # leitura ordenada no arquivo é mais rápida
            out = np.empty((len(i),) + all_views.shape[2:], dtype=np.uint8)
            out[order] = all_views[sel[i][order], np.asarray(k)[order]]
            return out

    views = _PoolView()
    t_feats = None
    if row == "C":
        t_feats = np.load(cfg.cache_dir / "teacher_view_feats.npy", mmap_mode="r")
        t_feats = torch.from_numpy(np.ascontiguousarray(t_feats[sel]))  # (N, K, D_t) fp16
    n, k_views = views.shape[:2]
    steps_per_epoch = n // cfg.batch_size                          # drop_last=True
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * cfg.warmup_epochs

    # ------------------------------------------------------------ modelo
    student = build_student(size, cfg.student_sizes, seed).to(device).to(memory_format=torch.channels_last)
    if row == "B":
        head = vicreg_projector(student.num_features, cfg.vicreg_hidden, cfg.vicreg_out).to(device)
        objective = VICRegObjective().to(device)
    else:
        head = distill_projector(student.num_features, cfg.distill_hidden, t_feats.shape[-1]).to(device)
    named = list(student.named_parameters()) + [(f"head.{n_}", p) for n_, p in head.named_parameters()]
    params = [p for _, p in named]
    opt = make_optimizer(params, named, lr, cfg.weight_decay)

    log.info("[%s] linha=%s | aluno=%s (m=%.2f, %d params, dim=%d) | cabeça=%d params", name, row, size,
             cfg.student_sizes[size], count_parameters(student), student.num_features, count_parameters(head))
    log.info("[%s] pool=%d imgs (fração %.0f%%) | K=%d | batch=%d | passos/época=%d | épocas=%d | "
             "passos totais=%d | lr=%.2e | wd=%.3f | warm-up=%d passos", name, n, 100 * data_frac, k_views,
             cfg.batch_size, steps_per_epoch, epochs, total_steps, lr, cfg.weight_decay, warmup_steps)

    # ------------------------------------------------------------- treino
    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    step = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for epoch in range(epochs):
        student.train(); head.train()
        perm = rng.permutation(n)
        ep_loss, ep_extra, ep_std, t_ep = 0.0, {}, 0.0, time.perf_counter()
        for it in range(steps_per_epoch):
            idx = np.sort(perm[it * cfg.batch_size:(it + 1) * cfg.batch_size])
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, warmup_steps, lr)

            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                if row == "B":
                    # Duas views DIFERENTES da mesma imagem, sorteadas entre as K.
                    k1 = rng.integers(0, k_views, size=len(idx))
                    k2 = (k1 + rng.integers(1, k_views, size=len(idx))) % k_views
                    x1 = to_student(torch.from_numpy(views[idx, k1]).to(device, non_blocking=True), cfg.img_size)
                    x2 = to_student(torch.from_numpy(views[idx, k2]).to(device, non_blocking=True), cfg.img_size)
                    f1 = student(x1.contiguous(memory_format=torch.channels_last))
                    f2 = student(x2.contiguous(memory_format=torch.channels_last))
                    loss, extra = objective(head(f1), head(f2))
                    feats_for_std = f1
                else:
                    # Uma view; o professor viu exatamente esta view (feature em cache).
                    kv = rng.integers(0, k_views, size=len(idx))
                    x = to_student(torch.from_numpy(views[idx, kv]).to(device, non_blocking=True), cfg.img_size)
                    target = t_feats[torch.from_numpy(idx), torch.from_numpy(kv)].to(device, non_blocking=True)
                    f = student(x.contiguous(memory_format=torch.channels_last))
                    loss = cosine_distill_loss(head(f), target)
                    extra = {}
                    feats_for_std = f

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1

            if not torch.isfinite(loss):
                log.error("[%s] perda não finita na época %d passo %d — abortando.", name, epoch, it)
                raise FloatingPointError("loss NaN/Inf")
            ep_loss += loss.item()
            for kk, vv in extra.items():
                ep_extra[kk] = ep_extra.get(kk, 0.0) + vv
            # Monitor de colapso barato: std por dimensão das features L2-normalizadas do backbone.
            with torch.no_grad():
                ep_std += F.normalize(feats_for_std.float(), dim=1).std(dim=0).mean().item()

        rec = {"epoch": epoch + 1, "loss": ep_loss / steps_per_epoch,
               "lr_end": opt.param_groups[0]["lr"], "feat_std": ep_std / steps_per_epoch,
               "seconds": time.perf_counter() - t_ep}
        rec.update({k_: v_ / steps_per_epoch for k_, v_ in ep_extra.items()})
        history.append(rec)
        log.info("[%s] época %3d/%d | perda=%.4f%s | feat_std=%.4f (ref %.4f) | lr=%.2e | %.1fs", name,
                 epoch + 1, epochs, rec["loss"],
                 "".join(f" | {k_}={v_:.4f}" for k_, v_ in rec.items() if k_ in ("invariance", "variance", "covariance")),
                 rec["feat_std"], 1 / math.sqrt(student.num_features), rec["lr_end"], rec["seconds"])

    if device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
    log.info("[%s] treino concluído em %.1f min (%.4f GPU-h), pico de memória %.2f GB",
             name, train_seconds / 60, train_seconds / 3600, peak_mem)

    # --------------------------------------------------- checkpoint + log
    torch.save({"backbone": student.state_dict(), "head": head.state_dict(), "row": row, "size": size,
                "seed": seed, "data_frac": data_frac, "epochs": epochs, "lr": lr}, out_dir / "checkpoint.pt")
    with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader(); w.writerows(history)

    summary: dict[str, Any] = {
        "name": name, "row": row, "size": size, "seed": seed, "data_frac": data_frac, "tag": tag,
        "epochs": epochs, "steps": total_steps, "batch_size": cfg.batch_size, "lr": lr,
        "n_pool": n, "n_views": k_views, "params": count_parameters(student),
        "feat_dim": student.num_features, "train_seconds": train_seconds,
        "train_gpu_hours": train_seconds / 3600, "peak_mem_gb": peak_mem, "final_loss": history[-1]["loss"],
    }

    # ------------------------------------- features finais + checagem de colapso
    if run_collapse_check:
        X_tr, y_tr = load_split(cfg, "train")
        X_te, y_te = load_split(cfg, "test")
        pre = student_preprocess(cfg.img_size)
        F_tr, F_te = extract(student, X_tr, pre, device), extract(student, X_te, pre, device)
        if not tag:  # execuções do estudo: guarda as features para a etapa de avaliação
            save_features(cfg.results_dir / "features" / f"{name}.npz", F_tr, F_te)
        summary["collapse_check"] = sanity_check(F_tr, y_tr, F_te, y_te, name)
    save_json(summary, out_dir / "summary.json")
    return summary
