"""Extração de features — UMA função, chamada de forma idêntica para A, B, C e D.

A única diferença entre encoders é a função de pré-processamento da entrada
(``to_student`` em 64/96 px ou ``to_teacher`` em 224 px), que é determinística
(Seção 3.1, "Evaluation transform"). O modelo é sempre colocado em ``eval()``:
para a linha A isso significa que o BatchNorm usa as estatísticas *default*
(média 0, variância 1), e as features não dependem da composição do batch
(Seção 2, nota sobre A).
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ArrayDataset, to_student, to_teacher
from .utils import get_logger

log = get_logger("features")


@torch.no_grad()
def extract(
    model: torch.nn.Module,
    X: np.ndarray,
    preprocess: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    batch_size: int = 256,
    amp_dtype: torch.dtype | None = None,
) -> np.ndarray:
    """Extrai features (N, D) float32 de um array de imagens uint8 (N, H, W, 3).

    Args:
        model: encoder sem cabeça (aluno ou professor). Fica congelado/eval.
        X: imagens originais (sem aumento).
        preprocess: transformação determinística aplicada na GPU.
        amp_dtype: ``torch.float16`` para o professor (Seção 3.9), ``None`` = fp32.
    """
    model.eval().to(device)
    loader = DataLoader(ArrayDataset(X), batch_size=batch_size, shuffle=False, num_workers=0)
    feats = []
    for x, _ in loader:
        x = preprocess(x.to(device, non_blocking=True))
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            f = model(x)
        feats.append(f.float().cpu().numpy())
    out = np.concatenate(feats)
    if not np.isfinite(out).all():
        log.error("Features contêm NaN/Inf! (%d valores)", int((~np.isfinite(out)).sum()))
    return out


def student_preprocess(img_size: int) -> Callable[[torch.Tensor], torch.Tensor]:
    return partial(to_student, size=img_size)


def teacher_preprocess(res: int) -> Callable[[torch.Tensor], torch.Tensor]:
    return partial(to_teacher, res=res)


def save_features(path: Path, feats_train: np.ndarray, feats_test: np.ndarray) -> None:
    """Cache em disco das features de um encoder (treino rotulado e teste)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, train=feats_train.astype(np.float32), test=feats_test.astype(np.float32))


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    return z["train"], z["test"]
