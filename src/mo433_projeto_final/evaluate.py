"""Avaliação idêntica para A, B, C e D: sondas congeladas e diagnósticos não supervisionados.

Sondas (k-NN, SVM linear, SVM RBF):

* pré-processamento idêntico: L2-normalização + ``StandardScaler`` ajustado só no subconjunto de treino;
* mesmos subconjuntos rotulados (``label_subsets.pkl``, gerado uma vez);
* k-NN: cosseno, k = min(20, tamanho da menor classe) — regra fixa;
* C (linear) e (C, gamma) (RBF) escolhidos UMA vez por CV no treino (encoder B/small/seed 0/100%)
  e congelados; gamma é congelado como multiplicador de 1/d, pois as dimensões diferem (128 vs 384);
* SVM RBF subamostrado (estratificado) para no máximo ``rbf_max_samples`` exemplos.

Diagnósticos: std dos embeddings L2-normalizados (colapso -> 0; saudável ~ 1/sqrt(d)),
k-means no teste (NMI/ARI) e t-SNE com os mesmos parâmetros para todos.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVC, LinearSVC

from .config import Config, get_device, load_json, run_name, save_json
from .data import extract, load_features, load_label_subsets, load_split, save_features, to_student
from .models import build_student

log = logging.getLogger("mo433.evaluate")

LINEAR_C_GRID = [1e-3, 1e-2, 1e-1, 1.0]
RBF_C_GRID = [0.1, 1.0, 10.0, 100.0]
RBF_GAMMA_MULT_GRID = [0.1, 1.0, 10.0]   # gamma = mult / d
LINEAR_MAX_ITER = 5000
TSNE_MAX_POINTS = 3000


# ============================================================================ sondas
def preprocess(F_train: np.ndarray, F_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """L2-normaliza e padroniza (scaler ajustado só no treino)."""
    scaler = StandardScaler().fit(normalize(F_train))
    return scaler.transform(normalize(F_train)), scaler.transform(normalize(F_test))


def knn_k(y: np.ndarray) -> int:
    return int(max(1, min(20, np.bincount(y).min())))


def knn_acc(Xtr, ytr, Xte, yte) -> float:
    return float(KNeighborsClassifier(n_neighbors=knn_k(ytr), metric="cosine").fit(Xtr, ytr).score(Xte, yte))


def subsample(X: np.ndarray, y: np.ndarray, n_max: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Subamostragem estratificada para no máximo ``n_max`` exemplos."""
    if len(y) <= n_max:
        return X, y
    idx = train_test_split(np.arange(len(y)), train_size=n_max, stratify=y, random_state=seed)[0]
    return X[idx], y[idx]


def tune_svm_hparams(F_train: np.ndarray, y_train: np.ndarray, rbf_max: int) -> dict:
    """CV de 3 folds (só no treino) para C linear e (C, gamma) RBF."""
    X = StandardScaler().fit_transform(normalize(F_train))
    d = X.shape[1]
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    lin = GridSearchCV(LinearSVC(max_iter=LINEAR_MAX_ITER, random_state=0), {"C": LINEAR_C_GRID}, cv=cv, n_jobs=-1)
    lin.fit(X, y_train)
    Xs, ys = subsample(X, y_train, rbf_max, seed=0)
    rbf = GridSearchCV(SVC(kernel="rbf"), {"C": RBF_C_GRID, "gamma": [m / d for m in RBF_GAMMA_MULT_GRID]},
                       cv=cv, n_jobs=-1)
    rbf.fit(Xs, ys)
    for search in (lin, rbf):
        for params, score in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"]):
            log.info("CV %s: %s -> %.4f", type(search.estimator).__name__, params, score)
    hp = {"linear_C": float(lin.best_params_["C"]), "rbf_C": float(rbf.best_params_["C"]),
          "rbf_gamma_mult": float(rbf.best_params_["gamma"] * d),
          "cv_linear_acc": float(lin.best_score_), "cv_rbf_acc": float(rbf.best_score_)}
    for key, grid in (("linear_C", LINEAR_C_GRID), ("rbf_C", RBF_C_GRID)):
        if hp[key] in (grid[0], grid[-1]):
            log.warning("%s ótimo (%s) está na borda da grade %s", key, hp[key], grid)
    log.info("Hiperparâmetros das sondas CONGELADOS: %s", hp)
    return hp


def run_probes(F_train, y_train, F_test, y_test, subsets, hp: dict, rbf_max: int) -> list[dict]:
    """Ajusta as 3 sondas em cada (fração, sorteio) e mede a acurácia no teste."""
    rows = []
    for (frac, draw), idx in sorted(subsets.items()):
        Xtr, Xte = preprocess(F_train[idx], F_test)
        ytr = y_train[idx]
        Xr, yr = subsample(Xtr, ytr, rbf_max, seed=draw)
        # A 100% todos os sorteios têm os mesmos índices: reaproveita o sorteio anterior
        # (k-NN e SVM linear são determinísticos; o RBF só muda se houver subamostragem).
        same_as_previous = frac == 1.0 and draw > 0
        if not same_as_previous:
            acc_knn = knn_acc(Xtr, ytr, Xte, y_test)
            acc_lin = float(LinearSVC(C=hp["linear_C"], max_iter=LINEAR_MAX_ITER, random_state=0)
                            .fit(Xtr, ytr).score(Xte, y_test))
        if not same_as_previous or len(yr) < len(ytr):
            gamma = hp["rbf_gamma_mult"] / Xtr.shape[1]
            acc_rbf = float(SVC(kernel="rbf", C=hp["rbf_C"], gamma=gamma).fit(Xr, yr).score(Xte, y_test))
        rows += [dict(fraction=frac, draw=draw, probe="knn", acc=acc_knn, n_train=len(ytr), k=knn_k(ytr)),
                 dict(fraction=frac, draw=draw, probe="linear_svm", acc=acc_lin, n_train=len(ytr)),
                 dict(fraction=frac, draw=draw, probe="rbf_svm", acc=acc_rbf, n_train=len(yr))]
    return rows


# ====================================================================== diagnósticos
def embedding_std(F: np.ndarray) -> dict[str, float]:
    std, ref = float(normalize(F).std(axis=0).mean()), 1.0 / np.sqrt(F.shape[1])
    return {"emb_std": std, "emb_std_ref": float(ref), "emb_std_ratio": float(std / ref)}


def kmeans_scores(F_train: np.ndarray, F_test: np.ndarray, y_test: np.ndarray, n_clusters: int) -> dict[str, float]:
    """k-means nas features de teste (pré-processadas como nas sondas) -> NMI/ARI."""
    Z = preprocess(F_train, F_test)[1]
    pred = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(Z)
    return {"nmi": float(normalized_mutual_info_score(y_test, pred)), "ari": float(adjusted_rand_score(y_test, pred))}


def sanity_check(F_train, y_train, F_test, y_test, name: str) -> dict:
    """Checagem de sanidade/colapso. Reprova se k-NN < 1,5x acaso, std/ref < 0,1 ou NMI < 0,05."""
    n_cls = int(max(y_train.max(), y_test.max()) + 1)
    Ztr, Zte = preprocess(F_train, F_test)
    out = {"knn_acc": knn_acc(Ztr, y_train, Zte, y_test), "chance": 1.0 / n_cls}
    out.update(embedding_std(F_test))
    out.update(kmeans_scores(F_train, F_test, y_test, n_cls))
    problems = [msg for bad, msg in [(out["knn_acc"] < 1.5 * out["chance"], "k-NN perto do acaso"),
                                     (out["emb_std_ratio"] < 0.1, "std dos embeddings ~ 0 (colapso)"),
                                     (out["nmi"] < 0.05, "NMI do k-means ~ 0")] if bad]
    out["passed"], out["problems"] = not problems, problems
    (log.info if out["passed"] else log.error)(
        "[sanity:%s] k-NN=%.4f | std/ref=%.2f | NMI=%.4f ARI=%.4f -> %s", name, out["knn_acc"],
        out["emb_std_ratio"], out["nmi"], out["ari"], "OK" if out["passed"] else "REPROVADO: " + "; ".join(problems))
    return out


def tsne_2d(F_train: np.ndarray, F_test: np.ndarray, idx: np.ndarray) -> np.ndarray:
    Z = preprocess(F_train, F_test[idx])[1]
    return TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(Z)


# ======================================================================= encoders
@dataclass(frozen=True)
class Encoder:
    row: str          # "A", "B", "C" ou "D"
    size: str         # "small", "large" ou "teacher"
    seed: int         # -1 para o professor
    data_frac: float  # fração do pool não rotulado (ablação)

    @property
    def name(self) -> str:
        if self.row == "D":
            return "D_teacher"
        if self.row == "A":
            return f"A_{self.size}_seed{self.seed}"
        return run_name(self.row, self.size, self.seed, self.data_frac)


def study_encoders(cfg: Config) -> list[Encoder]:
    encs = []
    for size, seeds in cfg.seeds.items():
        for seed in seeds:
            encs.append(Encoder("A", size, seed, 1.0))
            for row in ("B", "C"):
                encs += [Encoder(row, size, seed, f) for f in [1.0, *cfg.ablation_data_fractions]]
    return encs + [Encoder("D", "teacher", -1, 1.0)]


def feature_path(cfg: Config, enc: Encoder) -> Path:
    return cfg.results_dir / "features" / f"{enc.name}.npz"


def ensure_features(cfg: Config, encs: list[Encoder]) -> list[Encoder]:
    """Extrai as features que faltam (A sem treino; B/C do checkpoint); devolve os disponíveis."""
    device = get_device()
    X_tr, _ = load_split(cfg, "train")
    X_te, _ = load_split(cfg, "test")
    ok = []
    for enc in encs:
        if not feature_path(cfg, enc).exists():
            if enc.row == "D":
                log.warning("Features do professor ausentes — rode a etapa `teacher`.")
                continue
            model = build_student(enc.size, cfg.student_sizes, enc.seed)
            if enc.row != "A":
                ckpt = cfg.results_dir / "runs" / enc.name / "checkpoint.pt"
                if not ckpt.exists():
                    log.warning("Checkpoint ausente para %s — ignorado.", enc.name)
                    continue
                model.load_state_dict(torch.load(ckpt, map_location="cpu")["backbone"])
            save_features(feature_path(cfg, enc), extract(model, X_tr, to_student, device),
                          extract(model, X_te, to_student, device))
            log.info("Features extraídas: %s", enc.name)
        ok.append(enc)
    return ok


def _evaluate_one(cfg: Config, enc: Encoder, hp: dict, out_json: Path) -> dict:
    """Sondas + diagnósticos de um encoder (roda num processo worker)."""
    F_tr, F_te = load_features(feature_path(cfg, enc))
    _, y_tr = load_split(cfg, "train")
    _, y_te = load_split(cfg, "test")
    rows = run_probes(F_tr, y_tr, F_te, y_te, load_label_subsets(cfg), hp, cfg.rbf_max_samples)
    diag = {**kmeans_scores(F_tr, F_te, y_te, int(y_te.max() + 1)), **embedding_std(F_te)}
    res = {"encoder": enc.__dict__, "name": enc.name, "feat_dim": int(F_tr.shape[1]), "probes": rows, "diag": diag}
    save_json(res, out_json)
    return res


def run_evaluation(cfg: Config, n_jobs: int = 4) -> None:
    encs = ensure_features(cfg, study_encoders(cfg))
    rd = cfg.results_dir

    # hiperparâmetros das SVMs: escolhidos UMA vez e congelados
    hp_path = rd / "probe_hparams.json"
    if hp_path.exists():
        hp = load_json(hp_path)
    else:
        ref = Encoder("B", "small", cfg.seeds["small"][0], 1.0)
        hp = tune_svm_hparams(load_features(feature_path(cfg, ref))[0], load_split(cfg, "train")[1],
                              cfg.rbf_max_samples)
        hp["reference_encoder"] = ref.name
        save_json(hp, hp_path)

    # sondas + diagnósticos (um JSON por encoder; retomável; paralelo)
    pdir = rd / "probes"
    todo = [e for e in encs if not (pdir / f"{e.name}.json").exists()]
    log.info("Sondas: %d encoders a avaliar (%d já prontos)", len(todo), len(encs) - len(todo))
    jobs = (delayed(_evaluate_one)(cfg, e, hp, pdir / f"{e.name}.json") for e in todo)
    for res in Parallel(n_jobs=n_jobs, return_as="generator_unordered")(jobs):
        lin = {r["fraction"]: r["acc"] for r in res["probes"] if r["probe"] == "linear_svm" and r["draw"] == 0}
        log.info("  %-24s SVM linear (draw 0): %s | NMI=%.3f ARI=%.3f", res["name"],
                 " ".join(f"{f:g}->{a:.3f}" for f, a in sorted(lin.items())), res["diag"]["nmi"], res["diag"]["ari"])

    # consolida em CSVs
    probe_rows, diag_rows = [], []
    for e in encs:
        res = load_json(pdir / f"{e.name}.json")
        base = {"encoder": e.name, "row": e.row, "size": e.size, "seed": e.seed, "data_frac": e.data_frac,
                "feat_dim": res["feat_dim"]}
        probe_rows += [{**base, **r} for r in res["probes"]]
        diag_rows.append({**base, **res["diag"]})
    pd.DataFrame(probe_rows).to_csv(rd / "probe_results.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(rd / "diagnostics.csv", index=False)
    run_tsne(cfg)


def run_tsne(cfg: Config) -> None:
    """t-SNE de A, B, C (semente 0) e D, no mesmo subconjunto fixo do teste."""
    out = cfg.results_dir / "tsne.npz"
    if out.exists():
        return
    _, y_te = load_split(cfg, "test")
    idx = np.sort(np.random.default_rng(0).choice(len(y_te), size=min(TSNE_MAX_POINTS, len(y_te)), replace=False))
    arrays = {"idx": idx, "labels": y_te[idx]}
    encs = [Encoder(row, size, seeds[0], 1.0) for size, seeds in cfg.seeds.items() for row in "ABC"]
    for enc in encs + [Encoder("D", "teacher", -1, 1.0)]:
        arrays[enc.name] = tsne_2d(*load_features(feature_path(cfg, enc)), idx)
    np.savez(out, **arrays)
