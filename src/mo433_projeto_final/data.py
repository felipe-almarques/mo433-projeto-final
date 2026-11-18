"""Pipeline de dados (Seção 3.1 e Seção 4 do enunciado).

Responsabilidades:

1. **Carregar o dataset** e materializá-lo como arrays ``uint8`` (N, H, W, 3)
   em disco, com três papéis bem separados:

   * ``unlabeled`` — pool usado para pré-treino (B) e destilação (C). Os rótulos
     **nunca** são salvos junto com ele: o carregador de treino não tem como
     devolvê-los nem por engano.
   * ``train`` (+ ``y_train``) — split rotulado usado **apenas** para ajustar as
     sondas (k-NN / SVMs) nas frações 1%, 10%, 50% e 100%.
   * ``test`` (+ ``y_test``) — split de teste, usado apenas para medir as sondas
     e para os diagnósticos (k-means, t-SNE).

2. **Gerar os subconjuntos rotulados uma única vez** (estratificados, 3 sorteios
   por fração) e salvá-los em ``label_subsets.pkl`` — exatamente o código da
   Seção 3.1. Todas as sondas de todos os encoders leem esse mesmo arquivo.

3. **Dois pipelines de transformação, nunca misturados**:

   * *treino*: ``lightly.transforms.VICRegViewTransform`` (aumento forte do
     VICReg). Ele é aplicado K vezes por imagem **uma única vez** e o resultado é
     guardado em ``views.npy`` (Seção 3.9, "Caching the teacher"). B e C leem o
     mesmo pool de views, logo têm exatamente o mesmo aumento de dados.
   * *avaliação*: determinístico (resize, center crop, normalização) — funções
     ``eval_to_student`` / ``to_teacher`` abaixo, executadas na GPU.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import ExperimentConfig
from .utils import get_logger

log = get_logger("data")

# Normalização do ImageNet: é a que o professor DINO espera. Usamos a MESMA no
# aluno para não introduzir mais uma diferença entre as linhas.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# =============================================================================
# 1. Carregamento dos datasets brutos
# =============================================================================
def _pil_list_to_uint8(imgs) -> np.ndarray:
    """Converte uma sequência de imagens PIL em um array (N, H, W, 3) uint8."""
    return np.stack([np.asarray(im.convert("RGB"), dtype=np.uint8) for im in imgs])


def _load_eurosat(cfg: ExperimentConfig):
    """EuroSAT RGB (27k imagens 64x64, 10 classes, fora do ImageNet).

    O EuroSAT não tem split oficial de teste, então fazemos uma divisão
    estratificada fixa 80/20 (semente ``cfg.split_seed``). Como o EuroSAT não tem
    split não rotulado, simulamos um (Seção 4.2): o pool não rotulado é o split
    de treino com os rótulos descartados.
    """
    from torchvision.datasets import EuroSAT

    ds = EuroSAT(root=str(cfg.data_dir), download=True)
    log.info("EuroSAT carregado: %d imagens, classes=%s", len(ds), ds.classes)
    X = _pil_list_to_uint8(Image.open(p) for p, _ in tqdm(ds.samples, desc="lendo EuroSAT"))
    y = np.asarray(ds.targets, dtype=np.int64)

    idx_train, idx_test = train_test_split(
        np.arange(len(y)), test_size=cfg.test_fraction, stratify=y, random_state=cfg.split_seed
    )
    X_train, y_train = X[idx_train], y[idx_train]
    X_test, y_test = X[idx_test], y[idx_test]
    X_unl = X_train  # mesmas imagens, rótulos descartados
    return X_unl, X_train, y_train, X_test, y_test, list(ds.classes)


def _load_stl10(cfg: ExperimentConfig):
    """STL-10: 100k não rotuladas + 5k treino + 8k teste, 96x96 (imagens do ImageNet).

    Usamos ``cfg.n_unlabeled`` imagens sorteadas (semente fixa) do split
    ``unlabeled``. ATENÇÃO: as imagens vêm do ImageNet, que o professor DINO viu
    no pré-treino — isso favorece C e D (discutido no README).
    """
    from torchvision.datasets import STL10

    def as_nhwc(ds) -> np.ndarray:  # STL10.data é (N, 3, 96, 96)
        return np.ascontiguousarray(ds.data.transpose(0, 2, 3, 1))

    unl = STL10(root=str(cfg.data_dir), split="unlabeled", download=True)
    tr = STL10(root=str(cfg.data_dir), split="train", download=True)
    te = STL10(root=str(cfg.data_dir), split="test", download=True)
    rng = np.random.default_rng(cfg.split_seed)
    sel = np.sort(rng.choice(len(unl.data), size=cfg.n_unlabeled or len(unl.data), replace=False))
    X_unl = as_nhwc(unl)[sel]
    return (X_unl, as_nhwc(tr), tr.labels.astype(np.int64), as_nhwc(te), te.labels.astype(np.int64),
            list(tr.classes))


def _load_smoke(cfg: ExperimentConfig):
    """Dataset sintético (só para o teste de fumaça): 10 classes de cor/padrão.

    Cada classe tem uma cor média e uma frequência de listras própria, mais ruído,
    para que um encoder razoável consiga separá-las (e o teste de sanidade do
    professor não seja trivialmente reprovado).
    """
    rng = np.random.default_rng(0)
    n_cls, s = 10, cfg.img_size
    base = rng.uniform(0, 255, size=(n_cls, 3))  # cor de cada classe: a MESMA no treino e no teste

    def make(n):
        y = rng.integers(0, n_cls, size=n)
        xx = np.linspace(0, 2 * np.pi, s)[None, :, None]
        X = np.empty((n, s, s, 3), dtype=np.float32)
        for i, c in enumerate(y):
            stripes = 60 * np.sin((c + 1) * xx + rng.uniform(0, 6.28))
            X[i] = base[c] + stripes + rng.normal(0, 25, size=(s, s, 3))
        return np.clip(X, 0, 255).astype(np.uint8), y.astype(np.int64)

    X_train, y_train = make(2000)  # 1% = 20 >= 10 classes
    X_test, y_test = make(500)
    return X_train, X_train, y_train, X_test, y_test, [f"c{i}" for i in range(n_cls)]


def prepare_arrays(cfg: ExperimentConfig) -> None:
    """Carrega o dataset escolhido e salva os arrays em ``cfg.cache_dir``.

    Arquivos produzidos (todos ``.npy``, podem ser abertos com ``mmap_mode``):
    ``X_unlabeled``, ``X_train``, ``y_train``, ``X_test``, ``y_test`` e
    ``classes.txt``. Se já existirem, nada é refeito (etapa idempotente).
    """
    out = cfg.cache_dir
    if (out / "X_test.npy").exists():
        log.info("Arrays do dataset já existem em %s — pulando.", out)
        return
    out.mkdir(parents=True, exist_ok=True)
    loader = {"eurosat": _load_eurosat, "stl10": _load_stl10, "smoke": _load_smoke}[cfg.dataset]
    X_unl, X_tr, y_tr, X_te, y_te, classes = loader(cfg)

    # Opcionalmente reduz o pool não rotulado (para EuroSAT o padrão é usar tudo).
    if cfg.n_unlabeled is not None and len(X_unl) > cfg.n_unlabeled and cfg.dataset != "stl10":
        rng = np.random.default_rng(cfg.split_seed)
        X_unl = X_unl[np.sort(rng.choice(len(X_unl), cfg.n_unlabeled, replace=False))]

    for name, arr in [("X_unlabeled", X_unl), ("X_train", X_tr), ("y_train", y_tr),
                      ("X_test", X_te), ("y_test", y_te)]:
        np.save(out / f"{name}.npy", arr)
        log.info("  salvo %-12s shape=%s dtype=%s", name, arr.shape, arr.dtype)
    (out / "classes.txt").write_text("\n".join(classes), encoding="utf-8")
    log.info("Splits: não rotulado=%d | treino rotulado=%d | teste=%d | classes=%d",
             len(X_unl), len(X_tr), len(X_te), len(classes))
    log.info("Distribuição de classes (treino): %s", np.bincount(y_tr).tolist())
    log.info("Distribuição de classes (teste):  %s", np.bincount(y_te).tolist())


def load_split(cfg: ExperimentConfig, split: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Lê um split salvo. ``split`` ∈ {"unlabeled", "train", "test"}.

    Para ``unlabeled`` devolve ``(X, None)``: não existem rótulos para ele.
    """
    d = cfg.cache_dir
    if split == "unlabeled":
        return np.load(d / "X_unlabeled.npy", mmap_mode="r"), None
    return np.load(d / f"X_{split}.npy", mmap_mode="r"), np.load(d / f"y_{split}.npy")


def class_names(cfg: ExperimentConfig) -> list[str]:
    return (cfg.cache_dir / "classes.txt").read_text(encoding="utf-8").splitlines()


# =============================================================================
# 2. Subconjuntos rotulados (gerados UMA vez — Seção 3.1)
# =============================================================================
def make_label_subsets(cfg: ExperimentConfig) -> Path:
    """Gera e salva os subconjuntos rotulados estratificados (código da Seção 3.1).

    Para cada fração em ``cfg.label_fractions`` e cada sorteio ``draw``, guarda os
    índices (sobre o split de treino rotulado). Em 100% os sorteios são idênticos
    (todos os índices), como no código do enunciado.
    """
    path = cfg.results_dir / "label_subsets.pkl"
    if path.exists():
        log.info("label_subsets.pkl já existe — NÃO será regenerado (mesmos subconjuntos p/ todos).")
        return path
    _, y_train = load_split(cfg, "train")
    subsets: dict[tuple[float, int], np.ndarray] = {}
    for frac in cfg.label_fractions:
        for draw in range(cfg.n_draws):
            if frac == 1.0:
                subsets[(frac, draw)] = np.arange(len(y_train))
            else:
                idx, _ = train_test_split(
                    np.arange(len(y_train)), train_size=frac, stratify=y_train, random_state=draw
                )
                subsets[(frac, draw)] = idx
            per_class = np.bincount(y_train[subsets[(frac, draw)]])
            log.info("  subset frac=%.2f draw=%d: n=%d (por classe: min=%d max=%d)",
                     frac, draw, len(subsets[(frac, draw)]), per_class.min(), per_class.max())
    with open(path, "wb") as f:
        pickle.dump(subsets, f)
    log.info("Subconjuntos rotulados salvos em %s", path)
    return path


def load_label_subsets(cfg: ExperimentConfig) -> dict[tuple[float, int], np.ndarray]:
    with open(cfg.results_dir / "label_subsets.pkl", "rb") as f:
        return pickle.load(f)


# =============================================================================
# 3a. Transformação de TREINO: K views pré-geradas com o lightly
# =============================================================================
class _ViewDataset(Dataset):
    """Aplica a transformação de treino do VICReg (lightly) a cada imagem do pool.

    O array é aberto com ``mmap_mode`` dentro de cada worker (não é copiado ao
    serializar o dataset para os processos do DataLoader no Windows).
    """

    def __init__(self, npy_path: Path, img_size: int, min_scale: float):
        from lightly.transforms.vicreg_transform import VICRegViewTransform

        self.npy_path = npy_path
        self.arr: np.ndarray | None = None
        # mmap só lê o cabeçalho: obtemos N sem carregar o array na memória.
        self.n = np.load(npy_path, mmap_mode="r").shape[0]
        # normalize=None -> tensor float em [0,1]; a normalização é feita na GPU
        # no momento do treino (mesma função usada na avaliação).
        self.transform = VICRegViewTransform(input_size=img_size, min_scale=min_scale, normalize=None)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> torch.Tensor:
        if self.arr is None:
            self.arr = np.load(self.npy_path, mmap_mode="r")
        view = self.transform(Image.fromarray(np.asarray(self.arr[i])))
        # [0,1] float (C,H,W) -> uint8 (H,W,C) para economizar 4x de memória.
        return (view.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0)


def describe_train_transform(cfg: ExperimentConfig) -> str:
    """Texto com os parâmetros da transformação de treino (vai para o log/README)."""
    from lightly.transforms.vicreg_transform import VICRegViewTransform

    t = VICRegViewTransform(input_size=cfg.img_size, min_scale=cfg.min_scale, normalize=None)
    return repr(t.transform)


def generate_views(cfg: ExperimentConfig) -> Path:
    """Gera ``views.npy`` com shape (N, K, H, W, 3) uint8 (Seção 3.9).

    Cada uma das K passadas sobre o pool usa uma semente diferente (``1000 + k``),
    então a geração é reprodutível. B sorteia 2 das K views por imagem a cada
    época; C sorteia 1 (e usa a feature do professor daquela MESMA view).
    """
    path = cfg.cache_dir / "views.npy"
    if path.exists():
        log.info("views.npy já existe — pulando geração de views.")
        return path
    src = cfg.cache_dir / "X_unlabeled.npy"
    ds = _ViewDataset(src, cfg.img_size, cfg.min_scale)
    log.info("Transformação de treino (lightly VICRegViewTransform):\n%s", describe_train_transform(cfg))
    tmp = path.with_suffix(".tmp.npy")
    views = np.lib.format.open_memmap(
        tmp, mode="w+", dtype=np.uint8, shape=(len(ds), cfg.n_views, cfg.img_size, cfg.img_size, 3)
    )
    log.info("Gerando %d views x %d imagens (%.2f GB em disco)...",
             cfg.n_views, len(ds), views.nbytes / 1024**3)
    for k in range(cfg.n_views):
        g = torch.Generator().manual_seed(1000 + k)
        loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=cfg.num_workers,
                            generator=g, persistent_workers=False)
        start = 0
        for batch in tqdm(loader, desc=f"view {k + 1}/{cfg.n_views}"):
            views[start:start + len(batch), k] = batch.numpy()
            start += len(batch)
        log.info("  view %d/%d pronta", k + 1, cfg.n_views)
    views.flush()
    del views
    tmp.replace(path)  # renomeia só no fim: um arquivo parcial nunca é usado
    return path


# =============================================================================
# 3b. Transformação de AVALIAÇÃO (determinística) e entrada do professor
# =============================================================================
def _uint8_nhwc_to_float(x: torch.Tensor) -> torch.Tensor:
    """uint8 (B,H,W,3) -> float (B,3,H,W) em [0,1]."""
    return x.permute(0, 3, 1, 2).float().div_(255.0)


def _normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def _resize_center_crop(x: torch.Tensor, size: int, mode: str = "bilinear") -> torch.Tensor:
    """Redimensiona o menor lado para ``size`` e recorta o centro (determinístico)."""
    h, w = x.shape[-2:]
    if (h, w) != (size, size):
        scale = size / min(h, w)
        nh, nw = max(size, round(h * scale)), max(size, round(w * scale))
        x = F.interpolate(x, size=(nh, nw), mode=mode, align_corners=False, antialias=(scale < 1))
        top, left = (nh - size) // 2, (nw - size) // 2
        x = x[..., top:top + size, left:left + size]
    return x


def to_student(x_uint8: torch.Tensor, size: int) -> torch.Tensor:
    """Entrada do aluno: resize + center crop (identidade na resolução nativa) + normalização.

    Usada tanto na avaliação (imagens originais) quanto no treino (views já
    aumentadas, que já estão em ``size``): a normalização é sempre a mesma.
    """
    return _normalize(_resize_center_crop(_uint8_nhwc_to_float(x_uint8), size))


def to_teacher(x_uint8: torch.Tensor, res: int = 224) -> torch.Tensor:
    """Entrada do professor: upsample bicúbico para ``res`` (224) + normalização ImageNet.

    Um ViT com patch 16 precisa de ~224 px para funcionar como foi treinado
    (Seção 3.3); imagens de 64/96 px são, portanto, ampliadas.
    """
    x = _uint8_nhwc_to_float(x_uint8)
    x = F.interpolate(x, size=(res, res), mode="bicubic", align_corners=False).clamp_(0, 1)
    return _normalize(x)


class ArrayDataset(Dataset):
    """Dataset trivial sobre um array uint8 (N,H,W,3) [+ rótulos], para extração de features."""

    def __init__(self, X: np.ndarray, y: np.ndarray | None = None):
        self.X, self.y = X, y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        x = torch.from_numpy(np.ascontiguousarray(self.X[i]))
        return x, (int(self.y[i]) if self.y is not None else -1)
