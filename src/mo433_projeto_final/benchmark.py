"""Custos: vazão do DataLoader e dos passos de treino de B/C, e parâmetros, FLOPs e latência
de cada encoder (após aquecimento, média de muitos batches). Gera ``benchmark.json``."""

from __future__ import annotations

import logging
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode

from .config import Config, count_params, describe_hardware, get_device, save_json, sync
from .data import ViewDataset
from .models import build_student, build_teacher, distill_projector, vicreg_projector
from .train import distill_loss, vicreg_loss

log = logging.getLogger("mo433.benchmark")


def count_flops(model: torch.nn.Module, res: int, device: torch.device) -> int:
    """FLOPs de um forward de UMA imagem (1 MAC = 2 FLOPs)."""
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model.eval()(torch.randn(1, 3, res, res, device=device))
    return int(fc.get_total_flops())


@torch.no_grad()
def measure_latency(model, res: int, device: torch.device, batch: int, amp: bool, iters: int) -> dict[str, float]:
    model.eval().to(device)
    x = torch.randn(batch, 3, res, res, device=device)
    with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
        for _ in range(10):  # aquecimento
            model(x)
        sync(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        sync(device)
    dt = (time.perf_counter() - t0) / iters
    return {"ms_per_batch": dt * 1000, "ms_per_image": dt * 1000 / batch, "imgs_per_s": batch / dt}


def loader_throughput(cfg: Config, n_batches: int = 10) -> float:
    """Vazão (img/s) do DataLoader com o aumento do lightly."""
    ds = ViewDataset(cfg.cache_dir / "X_unlabeled.npy", cfg.img_size, cfg.min_scale)
    it = iter(DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers))
    next(it)  # descarta o 1º batch (inicialização dos workers)
    t0 = time.perf_counter()
    n = sum(len(next(it)) for _ in range(n_batches))
    return n / (time.perf_counter() - t0)


def step_throughput(cfg: Config, row: str, size: str, device: torch.device, iters: int = 20) -> float:
    """Vazão (imagens do pool/s) de um passo completo de treino de B ou C."""
    student = build_student(size, cfg.student_sizes, 0).to(device).to(memory_format=torch.channels_last)
    head = (vicreg_projector(student.num_features, cfg.vicreg_hidden, cfg.vicreg_out) if row == "B"
            else distill_projector(student.num_features, cfg.distill_hidden, 384)).to(device)
    opt = torch.optim.AdamW(list(student.parameters()) + list(head.parameters()), lr=1e-3)
    x1 = torch.randn(cfg.batch_size, 3, cfg.img_size, cfg.img_size, device=device).contiguous(
        memory_format=torch.channels_last)
    x2, target = torch.randn_like(x1), torch.randn(cfg.batch_size, 384, device=device)

    def step() -> None:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = (vicreg_loss(head(student(x1)), head(student(x2)))[0] if row == "B"
                    else distill_loss(head(student(x1)), target))
        opt.zero_grad()
        loss.backward()
        opt.step()

    for _ in range(5):
        step()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    sync(device)
    return cfg.batch_size * iters / (time.perf_counter() - t0)


def run_benchmark(cfg: Config) -> None:
    device = get_device()
    res = {"hardware": describe_hardware(), "batch_size": cfg.batch_size,
           "precision": "bf16 (alunos), fp16 (professor)", "loader_imgs_per_s": loader_throughput(cfg)}
    log.info("DataLoader + aumento lightly (%d workers): %.0f img/s", cfg.num_workers, res["loader_imgs_per_s"])
    res["step_imgs_per_s"] = {f"{row}_{size}": step_throughput(cfg, row, size, device)
                              for size in cfg.student_sizes for row in ("B", "C")}
    log.info("Passo de treino (img/s): %s", {k: round(v) for k, v in res["step_imgs_per_s"].items()})

    # parâmetros, FLOPs e latência (GPU: batch 256 fp16; CPU: batch 1 fp32)
    encoders = {f"student_{s}": (build_student(s, cfg.student_sizes, 0), cfg.img_size) for s in cfg.student_sizes}
    encoders["teacher"] = (build_teacher(cfg.teacher_name), cfg.teacher_res)
    res["encoders"] = {}
    for name, (model, r) in encoders.items():
        model = model.to(device)
        rec = {"params": count_params(model), "input_res": r, "feat_dim": int(model.num_features),
               "gflops": count_flops(model, r, device) / 1e9}
        if hasattr(model, "widths"):
            rec["widths"] = model.widths
        rec["latency_gpu"] = measure_latency(model, r, device, batch=256, amp=True, iters=50)
        rec["latency_cpu"] = measure_latency(model.to("cpu"), r, torch.device("cpu"), batch=1, amp=False, iters=20)
        res["encoders"][name] = rec
        log.info("%-14s params=%.2fM | %.3f GFLOPs/img @%dpx | dim=%d | GPU b256 fp16: %.0f img/s | CPU b1: %.2f ms",
                 name, rec["params"] / 1e6, rec["gflops"], r, rec["feat_dim"], rec["latency_gpu"]["imgs_per_s"],
                 rec["latency_cpu"]["ms_per_batch"])
    res["teacher_imgs_per_s"] = res["encoders"]["teacher"]["latency_gpu"]["imgs_per_s"]

    # projeção do tempo do estudo inteiro (minutos)
    n_pool = np.load(cfg.cache_dir / "X_unlabeled.npy", mmap_mode="r").shape[0]
    proj = {"view_generation_min": n_pool * cfg.n_views / res["loader_imgs_per_s"] / 60,
            "teacher_cache_min": n_pool * cfg.n_views / res["teacher_imgs_per_s"] / 60}
    total = proj["view_generation_min"] + proj["teacher_cache_min"]
    for size, seeds in cfg.seeds.items():
        for row in ("B", "C"):
            one = n_pool * cfg.epochs / res["step_imgs_per_s"][f"{row}_{size}"] / 60
            proj[f"one_run_{row}_{size}_min"] = one
            total += one * len(seeds) * (1 + sum(cfg.ablation_data_fractions))
    proj["total_training_hours_estimate"] = total / 60
    res["projection"] = proj
    log.info("Projeção (minutos, sem I/O/avaliação): %s", {k: round(v, 1) for k, v in proj.items()})
    save_json(res, cfg.results_dir / "benchmark.json")
