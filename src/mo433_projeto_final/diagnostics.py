"""Diagnósticos não supervisionados (Seções 3.3, 3.4 e 3.7).

* ``embedding_std``: desvio-padrão por dimensão dos embeddings L2-normalizados.
  Para um embedding saudável e isotrópico em d dimensões ele fica perto de
  1/sqrt(d); perto de 0 indica **colapso** (todas as imagens no mesmo ponto).
* ``kmeans_scores``: k-means com k = nº de classes, comparado aos rótulos
  verdadeiros por NMI e ARI (a estrutura de classes existe sem supervisão?).
* ``sanity_check``: combina k-NN, std e NMI em um veredito (usado para o
  professor antes de tudo e como *collapse check* ao fim de cada treino de B/C).
* ``tsne_2d``: projeção 2-D com os MESMOS parâmetros para todos os encoders.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, normalize

from .probes import knn_k
from .utils import get_logger

log = get_logger("diagnostics")


def embedding_std(F: np.ndarray) -> dict[str, float]:
    """Média do desvio-padrão por dimensão dos vetores L2-normalizados."""
    Z = normalize(F)
    std = float(Z.std(axis=0).mean())
    ref = 1.0 / np.sqrt(F.shape[1])
    return {"emb_std": std, "emb_std_ref": float(ref), "emb_std_ratio": float(std / ref)}


def kmeans_scores(F_train: np.ndarray, F_test: np.ndarray, y_test: np.ndarray, n_clusters: int,
                  seed: int = 0) -> dict[str, float]:
    """k-means sobre as features de TESTE (pré-processadas como nas sondas) -> NMI/ARI."""
    scaler = StandardScaler().fit(normalize(F_train))  # scaler do treino, como nas sondas
    Z = scaler.transform(normalize(F_test))
    pred = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(Z)
    return {"nmi": float(normalized_mutual_info_score(y_test, pred)),
            "ari": float(adjusted_rand_score(y_test, pred))}


def sanity_check(F_train: np.ndarray, y_train: np.ndarray, F_test: np.ndarray, y_test: np.ndarray,
                 name: str) -> dict[str, Any]:
    """Checagem de sanidade/colapso: k-NN (100% rótulos), std dos embeddings e NMI do k-means.

    Critérios de reprovação (Seção 3.3): acurácia k-NN < 1,5x o acaso, razão
    std/(1/sqrt(d)) < 0,1, ou NMI < 0,05. O veredito vai para o log em destaque.
    """
    n_cls = int(max(y_train.max(), y_test.max()) + 1)
    scaler = StandardScaler().fit(normalize(F_train))
    Ztr, Zte = scaler.transform(normalize(F_train)), scaler.transform(normalize(F_test))
    knn = KNeighborsClassifier(n_neighbors=knn_k(y_train), metric="cosine").fit(Ztr, y_train)
    out: dict[str, Any] = {"knn_acc": float(knn.score(Zte, y_test)), "chance": 1.0 / n_cls}
    out.update(embedding_std(F_test))
    out.update(kmeans_scores(F_train, F_test, y_test, n_cls))
    problems = []
    if out["knn_acc"] < 1.5 * out["chance"]:
        problems.append("k-NN perto do acaso")
    if out["emb_std_ratio"] < 0.1:
        problems.append("std dos embeddings ~ 0 (colapso)")
    if out["nmi"] < 0.05:
        problems.append("NMI do k-means ~ 0")
    out["passed"] = not problems
    out["problems"] = problems
    level = log.info if out["passed"] else log.error
    level("[sanity:%s] k-NN=%.4f (acaso=%.3f) | emb_std=%.4f (ref 1/sqrt(d)=%.4f, razão=%.2f) | "
          "NMI=%.4f ARI=%.4f -> %s", name, out["knn_acc"], out["chance"], out["emb_std"],
          out["emb_std_ref"], out["emb_std_ratio"], out["nmi"], out["ari"],
          "OK" if out["passed"] else "REPROVADO: " + "; ".join(problems))
    return out


def tsne_2d(F_train: np.ndarray, F_test: np.ndarray, idx: np.ndarray, seed: int = 0) -> np.ndarray:
    """t-SNE 2-D (perplexidade 30, init PCA, semente fixa) de um subconjunto fixo do teste."""
    scaler = StandardScaler().fit(normalize(F_train))
    Z = scaler.transform(normalize(F_test[idx]))
    return TSNE(n_components=2, perplexity=30, init="pca", random_state=seed).fit_transform(Z)
