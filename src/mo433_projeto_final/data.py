"""Dados: splits do EuroSAT, subconjuntos rotulados, K views pré-geradas e extração de features.

* ``unlabeled`` — pool para B e C, salvo SEM rótulos (o treino não tem como vê-los);
* ``train`` (+ rótulos) — usado só para ajustar as sondas nas frações 1/10/50/100%;
* ``test`` (+ rótulos) — só para medir as sondas e os diagnósticos.

Dois pipelines de transformação, nunca misturados: o de *treino* (aumento do VICReg
do lightly, aplicado K vezes por imagem uma única vez e salvo em ``views.npy``) e o de
*avaliação* (determinístico, ``to_student`` / ``to_teacher``, executado na GPU).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import DATA_DIR, Config

log = logging.getLogger("mo433.data")

# Normalização do ImageNet (a que o professor espera), usada também no aluno.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ------------------------------------------------------------------ splits
def prepare_arrays(cfg: Config) -> None:
    """Baixa o EuroSAT e salva os splits como ``.npy`` uint8 (N, H, W, 3) em ``cache/``."""
    out = cfg.cache_dir
    if (out / "X_test.npy").exists():
        log.info("Arrays do dataset já existem em %s — pulando.", out)
        return
    from torchvision.datasets import EuroSAT

    ds = EuroSAT(root=str(DATA_DIR), download=True)
    log.info("EuroSAT carregado: %d imagens, classes=%s", len(ds), ds.classes)
    X = np.stack([np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
                  for p, _ in tqdm(ds.samples, desc="lendo EuroSAT")])
    y = np.asarray(ds.targets, dtype=np.int64)
    i_tr, i_te = train_test_split(np.arange(len(y)), test_size=cfg.test_fraction, stratify=y,
                                  random_state=cfg.split_seed)
    arrays = {"X_unlabeled": X[i_tr],  # mesmas imagens do treino, rótulos descartados
              "X_train": X[i_tr], "y_train": y[i_tr], "X_test": X[i_te], "y_test": y[i_te]}
    out.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        np.save(out / f"{name}.npy", arr)
        log.info("  salvo %-12s shape=%s", name, arr.shape)
    (out / "classes.txt").write_text("\n".join(ds.classes), encoding="utf-8")
    log.info("Distribuição de classes: treino=%s | teste=%s",
             np.bincount(y[i_tr]).tolist(), np.bincount(y[i_te]).tolist())


def load_split(cfg: Config, split: str) -> tuple[np.ndarray, np.ndarray]:
    """``split`` ∈ {"train", "test"}: devolve (imagens em memory-map, rótulos)."""
    d = cfg.cache_dir
    return np.load(d / f"X_{split}.npy", mmap_mode="r"), np.load(d / f"y_{split}.npy")


def class_names(cfg: Config) -> list[str]:
    return (cfg.cache_dir / "classes.txt").read_text(encoding="utf-8").splitlines()


def make_label_subsets(cfg: Config) -> None:
    """Subconjuntos rotulados estratificados, gerados UMA vez (3 sorteios por fração)."""
    path = cfg.results_dir / "label_subsets.pkl"
    if path.exists():
        log.info("label_subsets.pkl já existe — NÃO será regenerado.")
        return
    _, y = load_split(cfg, "train")
    subsets = {}
    for frac in cfg.label_fractions:
        for draw in range(cfg.n_draws):
            if frac == 1.0:
                subsets[(frac, draw)] = np.arange(len(y))
            else:
                subsets[(frac, draw)] = train_test_split(np.arange(len(y)), train_size=frac, stratify=y,
                                                         random_state=draw)[0]
            per_class = np.bincount(y[subsets[(frac, draw)]])
            log.info("  subset frac=%.2f draw=%d: n=%d (por classe: min=%d max=%d)",
                     frac, draw, len(subsets[(frac, draw)]), per_class.min(), per_class.max())
    with open(path, "wb") as f:
        pickle.dump(subsets, f)


def load_label_subsets(cfg: Config) -> dict[tuple[float, int], np.ndarray]:
    with open(cfg.results_dir / "label_subsets.pkl", "rb") as f:
        return pickle.load(f)


# ------------------------------------------------- transformação de TREINO (views)
class ViewDataset(Dataset):
    """Aplica o aumento do VICReg (lightly) a cada imagem do pool; devolve uint8 (H, W, 3)."""

    def __init__(self, npy_path: Path, img_size: int, min_scale: float):
        from lightly.transforms.vicreg_transform import VICRegViewTransform

        self.npy_path, self.arr = npy_path, None  # aberto em cada worker (memory-map)
        self.n = np.load(npy_path, mmap_mode="r").shape[0]
        # normalize=None -> float em [0,1]; a normalização é feita na GPU (to_student)
        self.transform = VICRegViewTransform(input_size=img_size, min_scale=min_scale, normalize=None)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> torch.Tensor:
        if self.arr is None:
            self.arr = np.load(self.npy_path, mmap_mode="r")
        view = self.transform(Image.fromarray(np.asarray(self.arr[i])))
        return (view.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0)


def generate_views(cfg: Config) -> None:
    """Gera ``views.npy`` (N, K, H, W, 3) uint8; a passada k usa a semente ``1000 + k``."""
    path = cfg.cache_dir / "views.npy"
    if path.exists():
        log.info("views.npy já existe — pulando.")
        return
    ds = ViewDataset(cfg.cache_dir / "X_unlabeled.npy", cfg.img_size, cfg.min_scale)
    log.info("Transformação de treino (lightly):\n%s", ds.transform.transform)
    tmp = path.with_suffix(".tmp.npy")
    views = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8,
                                      shape=(len(ds), cfg.n_views, cfg.img_size, cfg.img_size, 3))
    log.info("Gerando %d views x %d imagens (%.2f GB em disco)...", cfg.n_views, len(ds), views.nbytes / 1024**3)
    for k in range(cfg.n_views):
        loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=cfg.num_workers,
                            generator=torch.Generator().manual_seed(1000 + k))
        start = 0
        for batch in tqdm(loader, desc=f"view {k + 1}/{cfg.n_views}"):
            views[start:start + len(batch), k] = batch.numpy()
            start += len(batch)
    views.flush()
    del views
    tmp.replace(path)  # só renomeia no fim: um arquivo parcial nunca é usado


# ---------------------------------------------- transformação de AVALIAÇÃO (GPU)
def _to_float(x_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 (B, H, W, 3) -> float (B, 3, H, W) em [0, 1]."""
    return x_uint8.permute(0, 3, 1, 2).float().div_(255.0)


def _normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def to_student(x_uint8: torch.Tensor) -> torch.Tensor:
    """Entrada do aluno: imagem na resolução nativa (64 px), só normalizada."""
    return _normalize(_to_float(x_uint8))


def to_teacher(x_uint8: torch.Tensor, res: int = 224) -> torch.Tensor:
    """Entrada do professor: upsample bicúbico para 224 px (ViT com patch 16) + normalização."""
    x = F.interpolate(_to_float(x_uint8), size=(res, res), mode="bicubic", align_corners=False).clamp_(0, 1)
    return _normalize(x)


# ------------------------------------------------------------ extração de features
@torch.no_grad()
def extract(model: torch.nn.Module, X: np.ndarray, preprocess, device: torch.device,
            batch_size: int = 256, amp_dtype: torch.dtype | None = None) -> np.ndarray:
    """A ÚNICA função de extração (A, B, C e D): modelo em ``eval()``, entrada determinística.

    Para a linha A, ``eval()`` faz o BatchNorm usar as estatísticas default (média 0,
    variância 1): as features não dependem do batch.
    """
    model.eval().to(device)
    feats = []
    for i in range(0, len(X), batch_size):
        x = preprocess(torch.from_numpy(np.ascontiguousarray(X[i:i + batch_size])).to(device))
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            feats.append(model(x).float().cpu().numpy())
    out = np.concatenate(feats)
    if not np.isfinite(out).all():
        log.error("Features contêm NaN/Inf!")
    return out


def save_features(path: Path, f_train: np.ndarray, f_test: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, train=f_train.astype(np.float32), test=f_test.astype(np.float32))


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    return z["train"], z["test"]
