"""Configuração central do experimento.

Todas as escolhas que afetam a comparação justa entre as linhas B e C ficam
aqui, em um único lugar, para que seja impossível (por construção) treinar B e C
com orçamentos diferentes: os dois treinos leem os mesmos campos (épocas,
batch, otimizador, *pool* de views...).

A configuração efetivamente usada é salva em ``results/<dataset>/config.json``
na etapa ``prepare`` e relida por todas as etapas seguintes. Assim, mudar um
valor no meio do estudo exige reexecutar o pipeline (não há mistura silenciosa).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .utils import load_json, save_json

# Raiz do repositório (…/mo433_projeto_final), independente do diretório atual.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
RESULTS_ROOT = REPO_ROOT / "results"

# ---------------------------------------------------------------------------
# Custo de pré-treino do professor (Seção 6, pergunta 3).
# README oficial do DINO (github.com/facebookresearch/dino): ViT-S/16, 100 épocas,
# 1 nó com 8 GPUs (V100) -> 1,75 dia  => 8 * 1,75 * 24 = 336 GPU-h / 100 épocas.
# O checkpoint público ``dino_deitsmall16`` foi treinado por 800 épocas, então a
# extrapolação linear é ~ 8 * 336 = 2688 GPU-h (V100). É uma ESTIMATIVA (limite
# inferior: o multi-crop completo usado no checkpoint final é mais caro).
# ---------------------------------------------------------------------------
TEACHER_PRETRAIN_GPU_HOURS = 2688.0


@dataclass
class ExperimentConfig:
    """Hiperparâmetros e protocolo de um estudo (um por dataset)."""

    # ----------------------------- dados -----------------------------------
    dataset: str = "eurosat"
    img_size: int = 64                 # resolução de entrada do aluno (64 EuroSAT, 96 STL-10)
    n_unlabeled: int | None = None     # tamanho do pool não rotulado (None = split inteiro)
    test_fraction: float = 0.2         # só p/ datasets sem split de teste oficial (EuroSAT)
    split_seed: int = 0                # semente da divisão treino/teste e do subconjunto não rotulado

    # ---------------- views pré-geradas (Seção 3.9, "Caching") --------------
    n_views: int = 8                   # K views aumentadas por imagem, iguais para B e C
    min_scale: float = 0.2             # escala mínima do RandomResizedCrop (crops < 20% ficam sem conteúdo a 64 px)

    # --------------------------- treino (B e C) -----------------------------
    epochs: int = 60
    batch_size: int = 256
    lr: float = 1e-3                   # AdamW, sugestão do enunciado (Seção 3.4)
    weight_decay: float = 0.05
    warmup_epochs: int = 3
    vicreg_hidden: int = 2048          # projetor do VICReg (3 camadas)
    vicreg_out: int = 2048
    distill_hidden: int = 2048         # projetor da destilação (padrão do enunciado)
    lr_B: float | None = None          # lr escolhido pelo ajuste (None = usa ``lr``)
    lr_C: float | None = None

    # ---------------------------- famílias ----------------------------------
    # Regra de escala única: multiplicador de largura. Ver ``models.py``.
    student_sizes: dict[str, float] = field(default_factory=lambda: {"small": 0.5, "large": 1.5})
    seeds: dict[str, list[int]] = field(default_factory=lambda: {"small": [0, 1, 2], "large": [0, 1]})

    # ----------------------------- professor --------------------------------
    teacher_name: str = "vit_small_patch16_224.dino"
    teacher_res: int = 224

    # ------------------------------ sondas ----------------------------------
    label_fractions: list[float] = field(default_factory=lambda: [0.01, 0.10, 0.50, 1.00])
    n_draws: int = 3                   # subconjuntos rotulados independentes por fração
    rbf_max_samples: int = 10_000      # sub-amostragem do SVM RBF (Seção 3.6)

    # ------------------------------ ablação ---------------------------------
    # Fator escolhido: quantidade de dados não rotulados (100% vs 50%).
    ablation_data_fractions: list[float] = field(default_factory=lambda: [0.5])

    # ---------------------------- utilidades --------------------------------
    num_workers: int = 6
    teacher_pretrain_gpu_hours: float = TEACHER_PRETRAIN_GPU_HOURS

    # ---------------------------------------------------------------- paths
    @property
    def results_dir(self) -> Path:
        return RESULTS_ROOT / self.dataset

    @property
    def data_dir(self) -> Path:
        return DATA_ROOT

    @property
    def cache_dir(self) -> Path:
        """Arquivos grandes intermediários (views, features do professor)."""
        return self.results_dir / "cache"

    @property
    def log_dir(self) -> Path:
        return self.results_dir / "logs"

    def lr_for(self, row: str) -> float:
        """Learning rate da linha ``row`` ("B" ou "C") após o ajuste (se houve)."""
        chosen = self.lr_B if row == "B" else self.lr_C
        return float(chosen) if chosen is not None else float(self.lr)

    # ---------------------------------------------------------- persistence
    def save(self) -> None:
        save_json(asdict(self), self.results_dir / "config.json")

    @classmethod
    def load(cls, dataset: str) -> "ExperimentConfig":
        path = RESULTS_ROOT / dataset / "config.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} não existe: rode primeiro `python -m mo433_projeto_final prepare --dataset {dataset}`."
            )
        raw: dict[str, Any] = load_json(path)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


def default_config(dataset: str) -> ExperimentConfig:
    """Configuração padrão por dataset (resolução casada com o dataset, Seção 4.3)."""
    if dataset == "eurosat":
        # 27k imagens 64x64, fora do ImageNet (sem viés a favor de C/D). Pool = split
        # de treino inteiro (~21,6k imagens), dentro da faixa 10k-30k sugerida.
        return ExperimentConfig(dataset="eurosat", img_size=64, n_unlabeled=None)
    if dataset == "stl10":
        # 96x96; usamos 20k das 100k imagens não rotuladas (faixa 10k-30k sugerida).
        # K=6 views para caber na RAM: 20k*6*96*96*3 B = 3,3 GB.
        return ExperimentConfig(dataset="stl10", img_size=96, n_unlabeled=20_000, n_views=6)
    if dataset == "smoke":
        # Dados sintéticos minúsculos para testar o pipeline inteiro em minutos.
        return ExperimentConfig(
            dataset="smoke", img_size=32, n_views=2, epochs=2, batch_size=64,
            warmup_epochs=1, seeds={"small": [0, 1], "large": [0]}, n_draws=2,
            vicreg_hidden=256, vicreg_out=256, distill_hidden=256, num_workers=0,
        )
    raise ValueError(f"dataset desconhecido: {dataset}")
