"""Treino dos alunos — B (VICReg direto) e C (destilação do ViT DINO) — e ajuste simétrico de lr.

B e C usam o MESMO laço (``train_student``); só mudam a perda e o projetor. Logo, por
construção, têm a mesma arquitetura e inicialização, o mesmo pool e as mesmas K views,
o mesmo nº de épocas/passos/batch, o mesmo otimizador (AdamW, warm-up + cosseno) e a mesma
precisão (bf16, ``channels_last``). Diferença intrínseca: B processa 2 views por imagem
(o VICReg compara duas views); C processa 1 (a mesma vista pelo professor, cuja feature
foi pré-calculada em ``teacher.py``).
"""

from __future__ import annotations

import csv
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightly.loss import VICRegLoss
from lightly.loss.vicreg_loss import covariance_loss, invariance_loss, variance_loss
from sklearn.model_selection import train_test_split

from .config import Config, count_params, extra_log_file, get_device, run_name, save_json, set_seed, sync
from .data import extract, load_split, save_features, to_student
from .evaluate import knn_acc, preprocess, sanity_check
from .models import build_student, distill_projector, vicreg_projector

log = logging.getLogger("mo433.train")

LR_GRID = [5e-4, 1e-3, 2e-3]
_VICREG = VICRegLoss()


def pool_indices(cfg: Config, n_pool: int, data_frac: float) -> np.ndarray:
    """Subconjunto do pool para a ablação: prefixo de uma permutação fixa (50% ⊂ 100%)."""
    perm = np.random.default_rng(cfg.split_seed + 123).permutation(n_pool)
    return np.sort(perm[: int(round(data_frac * n_pool))])


def make_optimizer(modules: list[nn.Module], lr: float, wd: float) -> torch.optim.Optimizer:
    """AdamW sem weight decay em vieses e parâmetros de normalização."""
    params = [p for m in modules for p in m.parameters()]
    return torch.optim.AdamW([{"params": [p for p in params if p.ndim > 1], "weight_decay": wd},
                              {"params": [p for p in params if p.ndim <= 1], "weight_decay": 0.0}], lr=lr)


def lr_at(step: int, total: int, warmup: int, base_lr: float) -> float:
    """Warm-up linear e depois decaimento cosseno até 0."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def vicreg_loss(z1: torch.Tensor, z2: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """Perda VICReg do lightly (λ=25, μ=25, ν=1), em fp32, + componentes não ponderados para o log."""
    z1, z2 = z1.float(), z2.float()
    loss = _VICREG(z1, z2)
    with torch.no_grad():
        parts = {"invariance": invariance_loss(z1, z2).item(),
                 "variance": 0.5 * (variance_loss(z1) + variance_loss(z2)).item(),
                 "covariance": (covariance_loss(z1) + covariance_loss(z2)).item()}
    return loss, parts


def distill_loss(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """1 - cos(projetor(aluno), professor)."""
    return 1.0 - F.cosine_similarity(s.float(), t.float(), dim=-1).mean()


def train_student(cfg: Config, row: str, size: str, seed: int, data_frac: float = 1.0,
                  epochs: int | None = None, lr: float | None = None, tag: str = "",
                  out_root: Path | None = None, collapse_check: bool = True) -> dict:
    """Treina um aluno (``row`` ∈ {"B", "C"}) e grava ``checkpoint.pt``, ``history.csv`` e ``summary.json``."""
    assert row in ("B", "C"), row
    device = get_device()
    epochs = epochs or cfg.epochs
    lr = lr if lr is not None else cfg.lr_for(row)
    name = run_name(row, size, seed, data_frac, tag)
    out_dir = (out_root or cfg.results_dir / "runs") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    # dados: views em memory-map (N, K, H, W, 3); C também lê as features do professor de cada view
    views = np.load(cfg.cache_dir / "views.npy", mmap_mode="r")
    sel = pool_indices(cfg, views.shape[0], data_frac)
    n, k_views = len(sel), views.shape[1]
    if row == "C":
        t_feats = torch.from_numpy(np.ascontiguousarray(
            np.load(cfg.cache_dir / "teacher_view_feats.npy", mmap_mode="r")[sel]))  # (n, K, 384) fp16

    def batch(idx: np.ndarray, k: np.ndarray) -> torch.Tensor:
        x = to_student(torch.from_numpy(views[sel[idx], k]).to(device, non_blocking=True))
        return x.contiguous(memory_format=torch.channels_last)

    steps_per_epoch = n // cfg.batch_size  # drop_last
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * cfg.warmup_epochs

    # modelo
    student = build_student(size, cfg.student_sizes, seed).to(device).to(memory_format=torch.channels_last)
    if row == "B":
        head = vicreg_projector(student.num_features, cfg.vicreg_hidden, cfg.vicreg_out).to(device)
    else:
        head = distill_projector(student.num_features, cfg.distill_hidden, t_feats.shape[-1]).to(device)
    opt = make_optimizer([student, head], lr, cfg.weight_decay)
    log.info("[%s] aluno %s (%d params, dim %d) | pool=%d x %d views | %d épocas x %d passos | batch=%d | lr=%.2e",
             name, size, count_params(student), student.num_features, n, k_views, epochs, steps_per_epoch,
             cfg.batch_size, lr)

    # treino
    rng = np.random.default_rng(seed)
    history, step = [], 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for epoch in range(epochs):
        student.train()
        head.train()
        perm = rng.permutation(n)
        sums, t_ep = {"loss": 0.0, "feat_std": 0.0}, time.perf_counter()
        for it in range(steps_per_epoch):
            idx = np.sort(perm[it * cfg.batch_size:(it + 1) * cfg.batch_size])
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, warmup_steps, lr)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                if row == "B":  # duas views DIFERENTES da mesma imagem
                    k1 = rng.integers(0, k_views, size=len(idx))
                    k2 = (k1 + rng.integers(1, k_views, size=len(idx))) % k_views
                    f, f2 = student(batch(idx, k1)), student(batch(idx, k2))
                    loss, parts = vicreg_loss(head(f), head(f2))
                else:  # uma view, a mesma que o professor viu
                    kv = rng.integers(0, k_views, size=len(idx))
                    target = t_feats[torch.from_numpy(idx), torch.from_numpy(kv)].to(device, non_blocking=True)
                    f = student(batch(idx, kv))
                    loss, parts = distill_loss(head(f), target), {}
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if not torch.isfinite(loss):
                raise FloatingPointError(f"[{name}] perda não finita na época {epoch}, passo {it}")
            # monitor de colapso: std por dimensão das features L2-normalizadas do backbone
            with torch.no_grad():
                feat_std = F.normalize(f.float(), dim=1).std(dim=0).mean().item()
            for key, v in {"loss": loss.item(), "feat_std": feat_std, **parts}.items():
                sums[key] = sums.get(key, 0.0) + v

        rec = {"epoch": epoch + 1, "loss": sums.pop("loss") / steps_per_epoch, "lr_end": opt.param_groups[0]["lr"],
               "feat_std": sums.pop("feat_std") / steps_per_epoch, "seconds": time.perf_counter() - t_ep}
        rec.update({key: v / steps_per_epoch for key, v in sums.items()})
        history.append(rec)
        log.info("[%s] época %3d/%d | perda=%.4f%s | feat_std=%.4f (ref %.4f) | lr=%.2e | %.1fs", name, epoch + 1,
                 epochs, rec["loss"], "".join(f" | {key}={sums[key] / steps_per_epoch:.4f}" for key in sums),
                 rec["feat_std"], 1 / math.sqrt(student.num_features), rec["lr_end"], rec["seconds"])

    sync(device)
    train_seconds = time.perf_counter() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
    log.info("[%s] treino concluído em %.1f min (%.4f GPU-h), pico de memória %.2f GB",
             name, train_seconds / 60, train_seconds / 3600, peak_mem)

    torch.save({"backbone": student.state_dict(), "head": head.state_dict(), "row": row, "size": size,
                "seed": seed, "data_frac": data_frac, "epochs": epochs, "lr": lr}, out_dir / "checkpoint.pt")
    with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    summary = {"name": name, "row": row, "size": size, "seed": seed, "data_frac": data_frac, "tag": tag,
               "epochs": epochs, "steps": total_steps, "batch_size": cfg.batch_size, "lr": lr, "n_pool": n,
               "n_views": k_views, "params": count_params(student), "feat_dim": student.num_features,
               "train_seconds": train_seconds, "train_gpu_hours": train_seconds / 3600, "peak_mem_gb": peak_mem,
               "final_loss": history[-1]["loss"]}

    # features finais (usadas na avaliação) + checagem de colapso
    if collapse_check:
        X_tr, y_tr = load_split(cfg, "train")
        X_te, y_te = load_split(cfg, "test")
        F_tr, F_te = extract(student, X_tr, to_student, device), extract(student, X_te, to_student, device)
        save_features(cfg.results_dir / "features" / f"{name}.npz", F_tr, F_te)
        summary["collapse_check"] = sanity_check(F_tr, y_tr, F_te, y_te, name)
    save_json(summary, out_dir / "summary.json")
    return summary


def train_all(cfg: Config) -> None:
    """Estudo principal (100% do pool) e ablação. Execuções concluídas são puladas (retomável);
    B e C alternam por semente para que uma interrupção não deixe uma linha à frente da outra."""
    plan = [(frac, size, seed, row) for frac in [1.0, *cfg.ablation_data_fractions]
            for size, seeds in cfg.seeds.items() for seed in seeds for row in ("B", "C")]
    for i, (frac, size, seed, row) in enumerate(plan, 1):
        name = run_name(row, size, seed, frac)
        if (cfg.results_dir / "runs" / name / "summary.json").exists():
            log.info("(%d/%d) %s já concluída — pulando.", i, len(plan), name)
            continue
        log.info("(%d/%d) iniciando %s", i, len(plan), name)
        with extra_log_file(cfg.log_dir / "runs" / f"{name}.log"):
            train_student(cfg, row, size, seed, data_frac=frac)


def run_tuning(cfg: Config) -> None:
    """Mesma grade de lr para B e C (aluno small, semente 0, epochs // 4 épocas), escolhida por
    k-NN numa validação de 20% tirada do treino rotulado. O teste nunca é usado."""
    if (cfg.results_dir / "tuning.json").exists():
        log.info("tuning.json já existe — lr escolhidos: B=%s C=%s", cfg.lr_B, cfg.lr_C)
        return
    epochs = max(5, cfg.epochs // 4)
    out_root = cfg.results_dir / "tuning"
    device = get_device()
    X, y = load_split(cfg, "train")
    i_tr, i_va = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=0)
    results: dict[str, dict[str, float]] = {"B": {}, "C": {}}
    for row in ("B", "C"):
        for lr in LR_GRID:
            s = train_student(cfg, row, "small", 0, epochs=epochs, lr=lr, tag=f"tune_lr{lr:g}",
                              out_root=out_root, collapse_check=False)
            model = build_student("small", cfg.student_sizes, 0)
            model.load_state_dict(torch.load(out_root / s["name"] / "checkpoint.pt", map_location="cpu")["backbone"])
            feats = extract(model, X, to_student, device)
            Ztr, Zva = preprocess(feats[i_tr], feats[i_va])
            results[row][f"{lr:g}"] = acc = knn_acc(Ztr, y[i_tr], Zva, y[i_va])
            log.info("[tune] %s lr=%g: k-NN validação = %.4f", row, lr, acc)
    best = {row: float(max(r, key=r.get)) for row, r in results.items()}
    cfg.lr_B, cfg.lr_C = best["B"], best["C"]
    cfg.save()
    save_json({"grid": LR_GRID, "epochs": epochs, "val_knn": results, "chosen": best},
              cfg.results_dir / "tuning.json")
    log.info("[tune] lr escolhidos (gravados em config.json): B=%g C=%g", best["B"], best["C"])
