"""Medidas de custo (Seções 2.1, 3.8 e 3.9).

* Vazão (imagens/s) de: (i) DataLoader com o aumento do lightly, (ii) um passo
  de treino de B e de C para cada tamanho de aluno, (iii) forward do professor a
  224 px — todos com o batch e a precisão mista do estudo. O mais lento é o gargalo.
* Parâmetros, FLOPs (``torch.utils.flop_counter``) e latência de inferência de
  cada encoder, num dispositivo e batch fixos, após aquecimento e com média
  sobre muitos batches.
* Projeção do tempo total do estudo a partir dessas medidas.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode

from .config import ExperimentConfig
from .data import _ViewDataset
from .models import build_student, build_teacher, distill_projector, vicreg_projector
from .train import VICRegObjective, cosine_distill_loss
from .utils import count_parameters, describe_hardware, get_device, get_logger, save_json

log = get_logger("benchmark")

LATENCY_BATCH_GPU = 256
LATENCY_BATCH_CPU = 1
WARMUP_ITERS = 10
TIMED_ITERS = 50


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def count_flops(model: torch.nn.Module, res: int, device: torch.device) -> int:
    """FLOPs de UM forward de uma imagem (1 MAC = 2 FLOPs)."""
    model.eval()
    x = torch.randn(1, 3, res, res, device=device)
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(x)
    return int(fc.get_total_flops())


@torch.no_grad()
def measure_latency(model: torch.nn.Module, res: int, device: torch.device, batch: int,
                    amp: bool, iters: int = TIMED_ITERS) -> dict[str, float]:
    """Latência média por batch (ms) e vazão (img/s), após ``WARMUP_ITERS`` de aquecimento."""
    model.eval().to(device)
    x = torch.randn(batch, 3, res, res, device=device)
    ctx = torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda")
    with ctx:
        for _ in range(WARMUP_ITERS):
            model(x)
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        _sync(device)
    dt = (time.perf_counter() - t0) / iters
    return {"ms_per_batch": dt * 1000, "ms_per_image": dt * 1000 / batch, "imgs_per_s": batch / dt}


def encoder_costs(cfg: ExperimentConfig, device: torch.device, teacher: torch.nn.Module | None = None
                  ) -> dict[str, dict[str, Any]]:
    """Parâmetros, FLOPs e latência (GPU batch 256 fp16; CPU batch 1 fp32) de cada encoder."""
    out: dict[str, dict[str, Any]] = {}
    encoders = {f"student_{s}": (build_student(s, cfg.student_sizes, 0), cfg.img_size) for s in cfg.student_sizes}
    encoders["teacher"] = (teacher or build_teacher(cfg.teacher_name), cfg.teacher_res)
    for name, (model, res) in encoders.items():
        model = model.to(device)
        rec: dict[str, Any] = {"params": count_parameters(model), "input_res": res,
                               "feat_dim": int(model.num_features), "gflops": count_flops(model, res, device) / 1e9}
        if hasattr(model, "widths"):
            rec["widths"] = model.widths
        rec["latency_gpu"] = measure_latency(model, res, device, LATENCY_BATCH_GPU, amp=True)
        cpu_model = model.to("cpu")
        rec["latency_cpu"] = measure_latency(cpu_model, res, torch.device("cpu"), LATENCY_BATCH_CPU, amp=False,
                                             iters=20)
        model.to(device)
        log.info("%-14s params=%.2fM | %.3f GFLOPs/img @%dpx | dim=%d | GPU b%d fp16: %.2f ms/batch (%.0f img/s)"
                 " | CPU b1: %.2f ms", name, rec["params"] / 1e6, rec["gflops"], res, rec["feat_dim"],
                 LATENCY_BATCH_GPU, rec["latency_gpu"]["ms_per_batch"], rec["latency_gpu"]["imgs_per_s"],
                 rec["latency_cpu"]["ms_per_batch"])
        out[name] = rec
    return out


def loader_throughput(cfg: ExperimentConfig, n_batches: int = 10) -> float:
    """(i) Vazão do DataLoader com o aumento do lightly (img/s, CPU, ``num_workers`` processos)."""
    ds = _ViewDataset(cfg.cache_dir / "X_unlabeled.npy", cfg.img_size, cfg.min_scale)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    it = iter(loader)
    next(it)  # descarta o 1º batch (inicialização dos workers)
    t0, n = time.perf_counter(), 0
    for _ in range(n_batches):
        n += len(next(it))
    return n / (time.perf_counter() - t0)


def step_throughput(cfg: ExperimentConfig, row: str, size: str, device: torch.device, iters: int = 20) -> float:
    """(ii) Vazão (imagens do pool/s) de um passo completo de treino de B ou C (fwd+bwd+opt)."""
    student = build_student(size, cfg.student_sizes, 0).to(device).to(memory_format=torch.channels_last)
    bs, r = cfg.batch_size, cfg.img_size
    if row == "B":
        head = vicreg_projector(student.num_features, cfg.vicreg_hidden, cfg.vicreg_out).to(device)
        obj = VICRegObjective()
    else:
        head = distill_projector(student.num_features, cfg.distill_hidden, 384).to(device)
    opt = torch.optim.AdamW(list(student.parameters()) + list(head.parameters()), lr=1e-3)
    x1 = torch.randn(bs, 3, r, r, device=device).contiguous(memory_format=torch.channels_last)
    x2 = torch.randn_like(x1)
    t = torch.randn(bs, 384, device=device)

    def step() -> None:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if row == "B":
                loss, _ = obj(head(student(x1)), head(student(x2)))
            else:
                loss = cosine_distill_loss(head(student(x1)), t)
        opt.zero_grad(); loss.backward(); opt.step()

    for _ in range(5):
        step()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    _sync(device)
    return bs * iters / (time.perf_counter() - t0)


def run_benchmark(cfg: ExperimentConfig) -> dict[str, Any]:
    device = get_device()
    hw = describe_hardware()
    log.info("Hardware: %s", hw)
    res: dict[str, Any] = {"hardware": hw, "batch_size": cfg.batch_size, "precision": "bf16 (alunos), fp16 (professor)"}

    res["loader_imgs_per_s"] = loader_throughput(cfg)
    log.info("(i) DataLoader + aumento lightly (%d workers): %.0f img/s", cfg.num_workers, res["loader_imgs_per_s"])

    res["step_imgs_per_s"] = {}
    for size in cfg.student_sizes:
        for row in ("B", "C"):
            v = step_throughput(cfg, row, size, device)
            res["step_imgs_per_s"][f"{row}_{size}"] = v
            log.info("(ii) passo de treino %s/%s: %.0f img/s", row, size, v)

    res["encoders"] = encoder_costs(cfg, device)
    res["teacher_imgs_per_s"] = res["encoders"]["teacher"]["latency_gpu"]["imgs_per_s"]
    log.info("(iii) forward do professor @%dpx fp16: %.0f img/s", cfg.teacher_res, res["teacher_imgs_per_s"])

    # ---------------- projeção do custo do estudo inteiro (com as views pré-geradas)
    n_pool = np.load(cfg.cache_dir / "X_unlabeled.npy", mmap_mode="r").shape[0]
    proj: dict[str, float] = {
        "view_generation_min": n_pool * cfg.n_views / res["loader_imgs_per_s"] / 60,
        "teacher_cache_min": n_pool * cfg.n_views / res["teacher_imgs_per_s"] / 60,
    }
    total = proj["view_generation_min"] + proj["teacher_cache_min"]
    for size, seeds in cfg.seeds.items():
        for row in ("B", "C"):
            one = n_pool * cfg.epochs / res["step_imgs_per_s"][f"{row}_{size}"] / 60
            proj[f"one_run_{row}_{size}_min"] = one
            # estudo principal + ablação (mesmas sementes, fração de dados menor)
            total += one * len(seeds) * (1 + sum(cfg.ablation_data_fractions))
    proj["total_training_hours_estimate"] = total / 60
    res["projection"] = proj
    log.info("Projeção (minutos, sem I/O/avaliação): %s", {k: round(v, 1) for k, v in proj.items()})
    save_json(res, cfg.results_dir / "benchmark.json")
    return res
