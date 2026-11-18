"""Avaliação (Seções 3.6 e 3.7): a MESMA função de avaliação para A, B, C e D.

Passos:

1. Garante as features cacheadas de todos os encoders (A é extraído aqui, sem
   treino nenhum, com BatchNorm em ``eval()`` e estatísticas *default*; B/C já
   foram salvos no fim de cada treino; D na etapa do professor).
2. Escolhe UMA vez os hiperparâmetros das SVMs (CV no treino, encoder
   B/small/seed 0/100%) e os congela em ``probe_hparams.json``.
3. Roda as sondas em todas as frações x sorteios (``probe_results.csv``) e os
   diagnósticos não supervisionados (``diagnostics.csv``) — em paralelo entre
   encoders, com cache por encoder (a etapa pode ser retomada).
4. Calcula as projeções t-SNE (mesmos parâmetros e mesmo subconjunto do teste
   para todos os encoders) usadas no painel de quatro.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

from .config import ExperimentConfig
from .data import load_label_subsets, load_split
from .diagnostics import embedding_std, kmeans_scores, tsne_2d
from .features import extract, load_features, save_features, student_preprocess
from .models import build_student
from .probes import run_probes, tune_svm_hparams
from .train import run_name
from .utils import get_device, get_logger, load_json, save_json

log = get_logger("evaluate")

TSNE_MAX_POINTS = 3000


@dataclass(frozen=True)
class Encoder:
    """Identifica um encoder avaliado (uma linha da tabela antes de agregar sementes)."""

    row: str          # "A", "B", "C" ou "D"
    size: str         # "small", "large" ou "teacher"
    seed: int         # -1 para o professor
    data_frac: float  # fração do pool não rotulado (ablação); 1.0 no estudo principal

    @property
    def name(self) -> str:
        if self.row == "D":
            return "D_teacher"
        if self.row == "A":
            return f"A_{self.size}_seed{self.seed}"
        return run_name(self.row, self.size, self.seed, self.data_frac)


def study_encoders(cfg: ExperimentConfig, include_ablation: bool = True) -> list[Encoder]:
    """Lista todos os encoders do estudo: A/B/C por tamanho e semente, ablação e D."""
    encs: list[Encoder] = []
    for size, seeds in cfg.seeds.items():
        for seed in seeds:
            encs.append(Encoder("A", size, seed, 1.0))
            for row in ("B", "C"):
                encs.append(Encoder(row, size, seed, 1.0))
                if include_ablation:
                    encs.extend(Encoder(row, size, seed, f) for f in cfg.ablation_data_fractions)
    encs.append(Encoder("D", "teacher", -1, 1.0))
    return encs


def feature_path(cfg: ExperimentConfig, enc: Encoder) -> Path:
    return cfg.results_dir / "features" / f"{enc.name}.npz"


def ensure_features(cfg: ExperimentConfig, encs: list[Encoder]) -> list[Encoder]:
    """Garante as features cacheadas; devolve só os encoders disponíveis."""
    device = get_device()
    X_tr, _ = load_split(cfg, "train")
    X_te, _ = load_split(cfg, "test")
    pre = student_preprocess(cfg.img_size)
    ok = []
    for enc in encs:
        path = feature_path(cfg, enc)
        if path.exists():
            ok.append(enc)
            continue
        if enc.row == "A":
            # Linha A: rede aleatória, SEM treino; eval() => BN com estatísticas default.
            model = build_student(enc.size, cfg.student_sizes, enc.seed)
        elif enc.row in ("B", "C"):
            ckpt = cfg.results_dir / "runs" / enc.name / "checkpoint.pt"
            if not ckpt.exists():
                log.warning("Checkpoint ausente para %s — encoder ignorado (treine-o primeiro).", enc.name)
                continue
            model = build_student(enc.size, cfg.student_sizes, enc.seed)
            model.load_state_dict(torch.load(ckpt, map_location="cpu")["backbone"])
        else:
            log.warning("Features do professor ausentes — rode a etapa `teacher`.")
            continue
        save_features(path, extract(model, X_tr, pre, device), extract(model, X_te, pre, device))
        log.info("Features extraídas: %s", enc.name)
        ok.append(enc)
    return ok


def _evaluate_one(cfg: ExperimentConfig, enc: Encoder, hp: dict[str, Any], out_json: Path) -> dict[str, Any]:
    """Sondas + diagnósticos de um encoder (executado num processo worker)."""
    F_tr, F_te = load_features(feature_path(cfg, enc))
    _, y_tr = load_split(cfg, "train")
    _, y_te = load_split(cfg, "test")
    subsets = load_label_subsets(cfg)
    rows = run_probes(F_tr, y_tr, F_te, y_te, subsets, hp, cfg.rbf_max_samples)
    diag = {**kmeans_scores(F_tr, F_te, y_te, int(y_te.max() + 1)), **embedding_std(F_te)}
    res = {"encoder": enc.__dict__, "name": enc.name, "feat_dim": int(F_tr.shape[1]), "probes": rows, "diag": diag}
    save_json(res, out_json)
    return res


def run_evaluation(cfg: ExperimentConfig, n_jobs: int = 4, include_ablation: bool = True) -> None:
    encs = ensure_features(cfg, study_encoders(cfg, include_ablation))
    log.info("%d encoders com features disponíveis.", len(encs))

    # -------------------- hiperparâmetros das SVMs: escolhidos UMA vez e congelados
    hp_path = cfg.results_dir / "probe_hparams.json"
    if hp_path.exists():
        hp = load_json(hp_path)
        log.info("Hiperparâmetros das sondas (congelados): %s", hp)
    else:
        ref = Encoder("B", "small", cfg.seeds["small"][0], 1.0)
        F_tr, _ = load_features(feature_path(cfg, ref))
        _, y_tr = load_split(cfg, "train")
        log.info("Escolhendo C/gamma por CV (3 folds) no treino, encoder de referência %s, 100%% rótulos", ref.name)
        hp = tune_svm_hparams(F_tr, y_tr, cfg.rbf_max_samples)
        hp["reference_encoder"] = ref.name
        save_json(hp, hp_path)

    # -------------------- sondas + diagnósticos (cache por encoder; paralelo)
    pdir = cfg.results_dir / "probes"
    pdir.mkdir(parents=True, exist_ok=True)
    todo = [e for e in encs if not (pdir / f"{e.name}.json").exists()]
    log.info("Sondas: %d encoders a avaliar (%d já em cache), n_jobs=%d", len(todo), len(encs) - len(todo), n_jobs)
    jobs = (delayed(_evaluate_one)(cfg, e, hp, pdir / f"{e.name}.json") for e in todo)
    for res in Parallel(n_jobs=n_jobs, return_as="generator_unordered")(jobs):
        lin = {r["fraction"]: r["acc"] for r in res["probes"] if r["probe"] == "linear_svm" and r["draw"] == 0}
        log.info("  %-24s dim=%3d | SVM linear (draw 0): %s | NMI=%.3f ARI=%.3f", res["name"], res["feat_dim"],
                 " ".join(f"{f:g}->{a:.3f}" for f, a in sorted(lin.items())),
                 res["diag"]["nmi"], res["diag"]["ari"])

    # -------------------- consolida em CSVs
    probe_rows, diag_rows = [], []
    for e in encs:
        res = load_json(pdir / f"{e.name}.json")
        base = {"encoder": e.name, "row": e.row, "size": e.size, "seed": e.seed, "data_frac": e.data_frac,
                "feat_dim": res["feat_dim"]}
        probe_rows += [{**base, **r} for r in res["probes"]]
        diag_rows.append({**base, **res["diag"]})
    pd.DataFrame(probe_rows).to_csv(cfg.results_dir / "probe_results.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(cfg.results_dir / "diagnostics.csv", index=False)
    log.info("probe_results.csv (%d linhas) e diagnostics.csv (%d linhas) salvos.", len(probe_rows), len(diag_rows))

    run_tsne(cfg)


def run_tsne(cfg: ExperimentConfig) -> None:
    """t-SNE de A, B, C, D (semente 0, estudo principal) para cada tamanho de aluno."""
    out = cfg.results_dir / "tsne.npz"
    if out.exists():
        log.info("tsne.npz já existe — pulando.")
        return
    _, y_te = load_split(cfg, "test")
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(y_te), size=min(TSNE_MAX_POINTS, len(y_te)), replace=False))  # mesmo p/ todos
    arrays: dict[str, np.ndarray] = {"idx": idx, "labels": y_te[idx]}
    for size, seeds in cfg.seeds.items():
        for row in ("A", "B", "C"):
            enc = Encoder(row, size, seeds[0], 1.0)
            if feature_path(cfg, enc).exists():
                arrays[enc.name] = tsne_2d(*load_features(feature_path(cfg, enc)), idx)
                log.info("t-SNE calculado: %s", enc.name)
    d = Encoder("D", "teacher", -1, 1.0)
    arrays[d.name] = tsne_2d(*load_features(feature_path(cfg, d)), idx)
    np.savez(out, **arrays)
    log.info("Projeções t-SNE salvas em %s", out)
