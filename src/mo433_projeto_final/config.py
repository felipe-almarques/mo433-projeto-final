"""Configuração do estudo e utilitários comuns (caminhos, JSON, logging, sementes).

B e C leem exatamente os mesmos campos (épocas, batch, otimizador, views...), então
é impossível, por construção, treiná-los com orçamentos diferentes. A configuração
efetiva fica salva em ``results/<dataset>/config.json`` e é relida por todas as etapas.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_ROOT = REPO_ROOT / "results"

# Custo de pré-treino publicado do professor (README oficial do DINO): ViT-S/16 com
# 100 épocas = 1,75 dia em 8 V100 = 336 GPU-h. O checkpoint usado tem 800 épocas
# => ~2688 GPU-h (estimativa por extrapolação linear; provavelmente um limite inferior).
TEACHER_PRETRAIN_GPU_HOURS = 2688.0


@dataclass
class Config:
    # dados (EuroSAT: 27k imagens 64x64, divisão estratificada 80/20)
    dataset: str = "eurosat"
    img_size: int = 64
    test_fraction: float = 0.2
    split_seed: int = 0
    # K views aumentadas pré-geradas por imagem (as mesmas para B e C)
    n_views: int = 8
    min_scale: float = 0.2
    # treino (idêntico para B e C)
    epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-3                 # usado se o ajuste de lr não foi feito
    weight_decay: float = 0.05
    warmup_epochs: int = 3
    vicreg_hidden: int = 2048
    vicreg_out: int = 2048
    distill_hidden: int = 2048
    lr_B: float | None = None        # escolhidos pela etapa `tune`
    lr_C: float | None = None
    # alunos: regra de escala única = multiplicador de largura
    student_sizes: dict[str, float] = field(default_factory=lambda: {"small": 0.5, "large": 1.5})
    seeds: dict[str, list[int]] = field(default_factory=lambda: {"small": [0, 1, 2], "large": [0, 1]})
    # professor
    teacher_name: str = "vit_small_patch16_224.dino"
    teacher_res: int = 224
    # sondas
    label_fractions: list[float] = field(default_factory=lambda: [0.01, 0.10, 0.50, 1.00])
    n_draws: int = 3
    rbf_max_samples: int = 10_000
    # ablação: fração do pool não rotulado
    ablation_data_fractions: list[float] = field(default_factory=lambda: [0.5])
    num_workers: int = 6

    @property
    def results_dir(self) -> Path:
        return RESULTS_ROOT / self.dataset

    @property
    def cache_dir(self) -> Path:
        """Arquivos grandes regeneráveis (arrays, views, features do professor)."""
        return self.results_dir / "cache"

    @property
    def log_dir(self) -> Path:
        return self.results_dir / "logs"

    def lr_for(self, row: str) -> float:
        chosen = self.lr_B if row == "B" else self.lr_C
        return float(chosen if chosen is not None else self.lr)

    def save(self) -> None:
        save_json(asdict(self), self.results_dir / "config.json")

    @classmethod
    def load(cls, dataset: str = "eurosat") -> "Config":
        """Lê ``config.json`` (ou devolve a configuração padrão se ainda não existir)."""
        path = RESULTS_ROOT / dataset / "config.json"
        if not path.exists():
            return cls(dataset=dataset)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in load_json(path).items() if k in known})


def run_name(row: str, size: str, seed: int, data_frac: float = 1.0, tag: str = "") -> str:
    """Nome canônico de uma execução, ex.: ``B_small_seed0`` ou ``C_large_seed1_data50``."""
    name = f"{row}_{size}_seed{seed}"
    if data_frac < 1.0:
        name += f"_data{int(round(data_frac * 100))}"
    return name + (f"_{tag}" if tag else "")


# ----------------------------------------------------------------------------- JSON
def save_json(obj, path: Path) -> None:
    """JSON indentado; escalares/arrays numpy viram tipos nativos (``.tolist()``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=lambda o: o.tolist()), encoding="utf-8")


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# -------------------------------------------------------------------------- logging
_FORMATTER = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s", "%Y-%m-%d %H:%M:%S")


def setup_logging(log_dir: Path, name: str) -> None:
    """Log raiz no console e em ``<log_dir>/<name>.log`` (modo append)."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    log_dir.mkdir(parents=True, exist_ok=True)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")):
        h.setFormatter(_FORMATTER)
        root.addHandler(h)
    for noisy in ("PIL", "matplotlib", "urllib3", "fsspec", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    log = logging.getLogger(f"mo433.{name}")
    log.info("=" * 90)
    log.info("Início da etapa '%s' (pid=%d) | comando: %s", name, os.getpid(), " ".join(sys.argv))


@contextmanager
def extra_log_file(path: Path):
    """Duplica o log raiz em ``path`` enquanto o bloco durar (um log por treino)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(_FORMATTER)
    logging.getLogger().addHandler(fh)
    try:
        yield
    finally:
        logging.getLogger().removeHandler(fh)
        fh.close()


# --------------------------------------------------------------------- torch/seeds
def set_seed(seed: int) -> None:
    """Fixa random/numpy/torch. cuDNN fica em modo *benchmark* (não determinístico):
    a variação residual entra no desvio-padrão entre sementes."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    logging.getLogger("mo433").warning("CUDA indisponível: usando CPU (será MUITO mais lento).")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def describe_hardware() -> dict:
    info = {"python": sys.version.split()[0], "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(), "cpu_count": os.cpu_count(), "platform": sys.platform}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(gpu=props.name, gpu_memory_gb=round(props.total_memory / 1024**3, 2),
                    cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    return info
