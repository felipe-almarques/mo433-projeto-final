"""Sondas congeladas (Seção 3.6): k-NN, SVM linear e SVM RBF.

Regras que tornam a comparação justa (todas implementadas aqui, em um só lugar):

* **Pré-processamento idêntico**: L2-normalização de cada vetor e depois
  ``StandardScaler`` ajustado APENAS no subconjunto rotulado de treino em uso.
* **Mesmos subconjuntos**: os índices vêm de ``label_subsets.pkl`` (gerado uma vez).
* **Mesmos hiperparâmetros**: C (linear) e (C, gamma) (RBF) são escolhidos UMA
  vez por validação cruzada no split de treino (nunca no teste), sobre o encoder
  B / small / seed 0 / 100% dos rótulos, e congelados para todos os encoders.
  O gamma é congelado como **multiplicador** de ``1/n_features`` (o "scale" do
  scikit-learn após padronização), porque os encoders têm dimensões diferentes
  (128, 384); um gamma absoluto teria significados diferentes em cada um.
* **k-NN**: métrica cosseno e k = min(20, n_por_classe), regra fixa, não ajustada.
* **SVM RBF**: subamostragem estratificada para no máximo ``rbf_max`` exemplos
  (10 000) quando o subconjunto é maior — custo quadrático/cúbico.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVC, LinearSVC

from .utils import get_logger

log = get_logger("probes")

LINEAR_C_GRID = [1e-3, 1e-2, 1e-1, 1.0]
RBF_C_GRID = [0.1, 1.0, 10.0, 100.0]
RBF_GAMMA_MULT_GRID = [0.1, 1.0, 10.0]   # gamma = mult / n_features
LINEAR_MAX_ITER = 5000


def preprocess(F_train: np.ndarray, F_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """L2-normaliza e padroniza (scaler ajustado só no treino)."""
    F_train, F_test = normalize(F_train), normalize(F_test)
    scaler = StandardScaler().fit(F_train)
    return scaler.transform(F_train), scaler.transform(F_test)


def knn_k(y: np.ndarray) -> int:
    """k = min(20, n_por_classe) — usa a menor classe do subconjunto."""
    return int(max(1, min(20, np.bincount(y).min())))


def _subsample(X: np.ndarray, y: np.ndarray, n_max: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Subamostragem estratificada para no máximo ``n_max`` exemplos."""
    if len(y) <= n_max:
        return X, y
    idx, _ = train_test_split(np.arange(len(y)), train_size=n_max, stratify=y, random_state=seed)
    return X[idx], y[idx]


# -----------------------------------------------------------------------------
# Escolha (única) dos hiperparâmetros das SVMs
# -----------------------------------------------------------------------------
def tune_svm_hparams(F_train: np.ndarray, y_train: np.ndarray, rbf_max: int) -> dict[str, Any]:
    """Valida cruzadamente (3 folds, só no treino) C linear e (C, gamma) RBF."""
    # Mesmo pré-processamento das sondas (L2 + StandardScaler). O scaler é ajustado
    # no split de treino inteiro: usa só estatísticas não supervisionadas do treino,
    # nunca rótulos de validação nem o teste.
    X = StandardScaler().fit_transform(normalize(F_train))
    d = X.shape[1]
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

    lin = GridSearchCV(LinearSVC(max_iter=LINEAR_MAX_ITER, random_state=0),
                       {"C": LINEAR_C_GRID}, cv=cv, n_jobs=-1)
    lin.fit(X, y_train)
    log.info("CV SVM linear: %s", {f"C={p['C']}": round(s, 4) for p, s in
                                   zip(lin.cv_results_["params"], lin.cv_results_["mean_test_score"])})

    Xs, ys = _subsample(X, y_train, rbf_max, seed=0)
    rbf = GridSearchCV(SVC(kernel="rbf"),
                       {"C": RBF_C_GRID, "gamma": [m / d for m in RBF_GAMMA_MULT_GRID]}, cv=cv, n_jobs=-1)
    rbf.fit(Xs, ys)
    for p, s in zip(rbf.cv_results_["params"], rbf.cv_results_["mean_test_score"]):
        log.info("CV SVM RBF: C=%-6s gamma_mult=%-5s -> %.4f", p["C"], round(p["gamma"] * d, 3), s)

    hp = {
        "linear_C": float(lin.best_params_["C"]),
        "rbf_C": float(rbf.best_params_["C"]),
        "rbf_gamma_mult": float(rbf.best_params_["gamma"] * d),
        "cv_linear_acc": float(lin.best_score_),
        "cv_rbf_acc": float(rbf.best_score_),
    }
    # Aviso se o ótimo está na borda da grade (talvez a grade devesse ser maior).
    if hp["linear_C"] in (LINEAR_C_GRID[0], LINEAR_C_GRID[-1]):
        log.warning("C linear ótimo (%s) está na borda da grade %s", hp["linear_C"], LINEAR_C_GRID)
    if hp["rbf_C"] in (RBF_C_GRID[0], RBF_C_GRID[-1]):
        log.warning("C RBF ótimo (%s) está na borda da grade %s", hp["rbf_C"], RBF_C_GRID)
    log.info("Hiperparâmetros das sondas CONGELADOS: %s", hp)
    return hp


# -----------------------------------------------------------------------------
# Avaliação de um encoder em todas as frações x sorteios x sondas
# -----------------------------------------------------------------------------
def run_probes(
    F_train: np.ndarray,
    y_train: np.ndarray,
    F_test: np.ndarray,
    y_test: np.ndarray,
    subsets: dict[tuple[float, int], np.ndarray],
    hp: dict[str, Any],
    rbf_max: int,
) -> list[dict[str, Any]]:
    """Ajusta as 3 sondas em cada (fração, sorteio) e mede a acurácia no teste.

    Em 100% os sorteios têm os mesmos índices; k-NN e SVM linear são
    determinísticos, então são calculados uma vez e replicados. O SVM RBF é
    recalculado por sorteio quando há subamostragem (a amostra depende do sorteio).
    """
    rows: list[dict[str, Any]] = []
    cache_full: dict[str, float] = {}
    for (frac, draw), idx in sorted(subsets.items()):
        Xtr, Xte = preprocess(F_train[idx], F_test)
        ytr = y_train[idx]
        full = frac == 1.0

        # --- k-NN (cosseno, k fixo pela regra) ---------------------------------
        k = knn_k(ytr)
        if full and "knn" in cache_full:
            acc_knn = cache_full["knn"]
        else:
            acc_knn = KNeighborsClassifier(n_neighbors=k, metric="cosine").fit(Xtr, ytr).score(Xte, y_test)
            if full:
                cache_full["knn"] = acc_knn
        rows.append(dict(fraction=frac, draw=draw, probe="knn", acc=float(acc_knn), n_train=len(ytr), k=k))

        # --- SVM linear --------------------------------------------------------
        if full and "linear" in cache_full:
            acc_lin = cache_full["linear"]
        else:
            clf = LinearSVC(C=hp["linear_C"], max_iter=LINEAR_MAX_ITER, random_state=0)
            acc_lin = clf.fit(Xtr, ytr).score(Xte, y_test)
            if full:
                cache_full["linear"] = acc_lin
        rows.append(dict(fraction=frac, draw=draw, probe="linear_svm", acc=float(acc_lin), n_train=len(ytr)))

        # --- SVM RBF (subamostrado se necessário) ------------------------------
        Xr, yr = _subsample(Xtr, ytr, rbf_max, seed=draw)
        if full and len(ytr) <= rbf_max and "rbf" in cache_full:
            acc_rbf = cache_full["rbf"]
        else:
            gamma = hp["rbf_gamma_mult"] / Xtr.shape[1]
            acc_rbf = SVC(kernel="rbf", C=hp["rbf_C"], gamma=gamma).fit(Xr, yr).score(Xte, y_test)
            if full:
                cache_full["rbf"] = acc_rbf
        rows.append(dict(fraction=frac, draw=draw, probe="rbf_svm", acc=float(acc_rbf), n_train=len(yr)))
    return rows
