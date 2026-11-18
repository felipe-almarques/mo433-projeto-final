"""Utilitários compartilhados por todas as etapas do pipeline.

Este módulo concentra tudo o que é "infraestrutura" e não depende do método:

* configuração de *logging* (console + arquivo), para que cada etapa deixe um
  rastro completo e com horário em ``results/<dataset>/logs/``;
* controle de sementes (reprodutibilidade);
* leitura/escrita de JSON;
* um cronômetro simples usado para contabilizar GPU-horas (Seção 3.4 do enunciado).
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

# Formato único de log para todo o projeto: horário, nível, módulo e mensagem.
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_dir: Path | None, name: str, level: int = logging.INFO) -> logging.Logger:
    """Configura o logger raiz para escrever no console e em ``<log_dir>/<name>.log``.

    Cada execução de uma etapa (ex.: ``train_B_small_seed0``) ganha o seu próprio
    arquivo de log, em modo *append*, de modo que reexecuções ficam registradas
    em sequência no mesmo arquivo (útil para auditar o que foi feito).

    Args:
        log_dir: diretório onde o arquivo de log será criado (``None`` = só console).
        name: nome do arquivo de log (sem extensão) e do logger devolvido.
        level: nível mínimo de log.

    Returns:
        Logger com o nome ``mo433.<name>``.
    """
    root = logging.getLogger()
    root.setLevel(level)
    # Remove handlers antigos: o mesmo processo pode chamar várias etapas.
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()

    fmt = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{name}.log", mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

    # Bibliotecas muito verbosas ficam em WARNING para não poluir os logs.
    for noisy in ("PIL", "matplotlib", "urllib3", "numba", "h5py", "fsspec", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger = logging.getLogger(f"mo433.{name}")
    logger.info("=" * 90)
    logger.info("Início da etapa '%s' (pid=%d)", name, os.getpid())
    logger.info("Linha de comando: %s", " ".join(sys.argv))
    return logger


def get_logger(name: str) -> logging.Logger:
    """Atalho para obter um logger filho do namespace do projeto."""
    return logging.getLogger(f"mo433.{name}")


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Fixa as sementes de ``random``, ``numpy`` e ``torch`` (CPU e GPU).

    ``deterministic=True`` também força kernels determinísticos do cuDNN, ao
    custo de velocidade. Nos treinos usamos ``False`` (cuDNN benchmark ligado):
    a variação residual entra no desvio-padrão entre sementes, que é o que o
    enunciado pede para medir.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def get_device() -> torch.device:
    """Devolve a GPU se disponível, senão a CPU (com aviso no log)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    get_logger("utils").warning("CUDA indisponível: usando CPU (será MUITO mais lento).")
    return torch.device("cpu")


def describe_hardware() -> dict[str, Any]:
    """Coleta uma descrição do hardware/software para o relatório (Seção 3.9)."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": os.cpu_count(),
        "platform": sys.platform,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            gpu=props.name,
            gpu_memory_gb=round(props.total_memory / 1024**3, 2),
            cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),
        )
    return info


def save_json(obj: Any, path: Path) -> None:
    """Salva ``obj`` em JSON legível (indentado), criando o diretório se preciso."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def load_json(path: Path) -> Any:
    """Carrega um arquivo JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    """Converte tipos numpy/Path para tipos serializáveis em JSON."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Tipo não serializável: {type(o)}")


@contextmanager
def timer(logger: logging.Logger, what: str) -> Iterator[dict[str, float]]:
    """Context manager que mede o tempo de parede de um bloco e o registra no log.

    Sincroniza a GPU antes e depois, para que o tempo medido inclua os kernels
    assíncronos (senão o tempo de GPU seria subestimado).

    Uso::

        with timer(log, "cache do professor") as t:
            ...
        print(t["seconds"])
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    out: dict[str, float] = {}
    start = time.perf_counter()
    logger.info("[início] %s", what)
    try:
        yield out
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        out["seconds"] = time.perf_counter() - start
        logger.info("[fim]    %s -> %.1f s (%.4f h)", what, out["seconds"], out["seconds"] / 3600)


def count_parameters(model: torch.nn.Module) -> int:
    """Número total de parâmetros de um módulo."""
    return sum(p.numel() for p in model.parameters())
