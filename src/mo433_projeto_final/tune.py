"""Ajuste pequeno e SIMÉTRICO do learning rate para B e C (Seção 3.6, último parágrafo).

"A mesma pequena grade de hiperparâmetros e o mesmo esforço de ajuste devem ser
dados ao treino de B e de C." Aqui:

* grade: lr ∈ {5e-4, 1e-3, 2e-3} (em torno do 1e-3 sugerido), igual para B e C;
* mesmo protocolo para ambos: aluno small, semente 0, ``epochs // 4`` épocas;
* critério: acurácia k-NN (regra fixa do projeto) numa **validação** tirada do
  split de treino rotulado (80/20 estratificado). O teste nunca é usado.

O lr vencedor de cada linha é gravado em ``config.json`` (``lr_B``/``lr_C``)
e usado em todos os treinos do estudo. Se ambos escolherem o mesmo valor, B e C
têm literalmente as mesmas configurações de otimizador.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from .config import ExperimentConfig
from .data import load_split
from .features import extract, student_preprocess
from .models import build_student
from .probes import knn_k, preprocess
from .train import train_student
from .utils import get_device, get_logger, save_json

log = get_logger("tune")

LR_GRID = [5e-4, 1e-3, 2e-3]


def _val_knn(cfg: ExperimentConfig, ckpt_path, size: str) -> float:
    """k-NN treinado em 80% do treino rotulado e avaliado nos outros 20% (validação)."""
    device = get_device()
    model = build_student(size, cfg.student_sizes, 0)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["backbone"])
    X, y = load_split(cfg, "train")
    F = extract(model, X, student_preprocess(cfg.img_size), device)
    i_tr, i_va = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=0)
    Ztr, Zva = preprocess(F[i_tr], F[i_va])
    return float(KNeighborsClassifier(knn_k(y[i_tr]), metric="cosine").fit(Ztr, y[i_tr]).score(Zva, y[i_va]))


def run_tuning(cfg: ExperimentConfig, epochs: int | None = None) -> dict:
    epochs = epochs or max(5, cfg.epochs // 4)
    out_root = cfg.results_dir / "tuning"
    results: dict[str, dict[str, float]] = {"B": {}, "C": {}}
    for row in ("B", "C"):
        for lr in LR_GRID:
            s = train_student(cfg, row, "small", 0, epochs=epochs, lr=lr, tag=f"tune_lr{lr:g}",
                              out_root=out_root, run_collapse_check=False)
            acc = _val_knn(cfg, out_root / s["name"] / "checkpoint.pt", "small")
            results[row][f"{lr:g}"] = acc
            log.info("[tune] linha %s lr=%g (%d épocas): k-NN validação = %.4f", row, lr, epochs, acc)
    best = {row: float(max(r, key=r.get)) for row, r in results.items()}
    cfg.lr_B, cfg.lr_C = best["B"], best["C"]
    cfg.save()
    summary = {"grid": LR_GRID, "epochs": epochs, "val_knn": results, "chosen": best}
    save_json(summary, cfg.results_dir / "tuning.json")
    log.info("[tune] lr escolhidos (gravados em config.json): B=%g C=%g", best["B"], best["C"])
    return summary
