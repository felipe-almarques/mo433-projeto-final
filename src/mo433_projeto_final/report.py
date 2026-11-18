"""Relatório (Seções 3.8, 5 e 6): tabelas, figuras e ``summary.md``.

Estatística usada em todo o relatório
-------------------------------------
Para cada (encoder, tamanho, fração de dados, fração de rótulos, sonda):

* ``per-seed`` = média sobre os 3 sorteios de subconjunto rotulado;
* ``mean``     = média sobre sementes (= média sobre todas as execuções);
* ``std_seed`` = desvio-padrão (ddof=1) entre sementes das médias por semente;
* ``std_all``  = desvio-padrão sobre TODAS as execuções (semente x sorteio),
  que inclui a variância de *quais* imagens foram rotuladas (Seção 2).

Lacuna C − B: ``gap = mean_C − mean_B``, ``spread = sqrt(std_all_B² + std_all_C²)``;
a diferença só é chamada de *conclusiva* se ``|gap| > spread`` (Seção 2, "Treat
differences smaller than the seed-to-seed spread as inconclusive").
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import ExperimentConfig  # noqa: E402
from .data import class_names  # noqa: E402
from .utils import get_logger, load_json  # noqa: E402

log = get_logger("report")

# ------------------------------------------------------------------ estilo
# Paleta categórica validada (slots 1-3 passam em todos os pares, inclusive
# daltonismo); A é neutro (cinza) por ser a linha de base.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ROW_COLOR = {"A": "#8f8d87", "B": "#2a78d6", "C": "#eb6834", "D": "#1baf7a"}
ROW_LABEL = {"A": "A: aleatório", "B": "B: VICReg direto", "C": "C: destilado", "D": "D: professor ViT-S/16"}
SIZE_STYLE = {"small": ("-", "o"), "large": ("--", "s")}
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
                "#52514e", "#b5b3ad"]
CLASS_MARKERS = ["o"] * 8 + ["^", "v"]  # classes 9-10 (neutras) ganham marcador próprio
PROBE_LABEL = {"knn": "k-NN", "linear_svm": "SVM linear", "rbf_svm": "SVM RBF"}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "lines.linewidth": 2,
        "legend.frameon": False,
    })


# =============================================================================
# Agregação
# =============================================================================
def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega execuções (semente x sorteio) em média ± desvios, conforme o docstring do módulo."""
    keys = ["row", "size", "data_frac", "fraction", "probe"]
    per_seed = df.groupby(keys + ["seed"], as_index=False)["acc"].mean()
    a = per_seed.groupby(keys).agg(mean=("acc", "mean"), std_seed=("acc", "std"), n_seeds=("acc", "size"))
    b = df.groupby(keys).agg(std_all=("acc", "std"), n_runs=("acc", "size"))
    out = a.join(b).reset_index()
    return out.fillna({"std_seed": 0.0, "std_all": 0.0})


def gap_table(agg: pd.DataFrame) -> pd.DataFrame:
    """Lacuna C − B por (tamanho, fração de dados, fração de rótulos, sonda)."""
    keys = ["size", "data_frac", "fraction", "probe"]
    b = agg[agg.row == "B"].set_index(keys)
    c = agg[agg.row == "C"].set_index(keys)
    j = b[["mean", "std_all", "std_seed"]].join(c[["mean", "std_all", "std_seed"]], lsuffix="_B", rsuffix="_C",
                                                how="inner")
    j["gap_C_minus_B"] = j.mean_C - j.mean_B
    j["spread"] = np.sqrt(j.std_all_B ** 2 + j.std_all_C ** 2)
    j["spread_seed"] = np.sqrt(j.std_seed_B ** 2 + j.std_seed_C ** 2)
    j["conclusive"] = j.gap_C_minus_B.abs() > j.spread
    j["verdict"] = np.where(~j.conclusive, "inconclusivo", np.where(j.gap_C_minus_B > 0, "C > B", "B > C"))
    return j.reset_index()


def _pm(m: float, s: float) -> str:
    return f"{100 * m:.1f} ± {100 * s:.1f}"


def _md(df: pd.DataFrame) -> str:
    """DataFrame -> tabela Markdown (sem depender do pacote ``tabulate``)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


# =============================================================================
# Figuras
# =============================================================================
def fig_label_efficiency(agg: pd.DataFrame, path: Path, probe: str = "linear_svm", ax=None) -> None:
    """Curva de eficiência de rótulos (figura principal): x = fração (log), y = acurácia."""
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 4.6))
    sub = agg[(agg.probe == probe) & (agg.data_frac == 1.0)]
    for row in ("A", "B", "C"):
        for size, (ls, mk) in SIZE_STYLE.items():
            s = sub[(sub.row == row) & (sub["size"] == size)].sort_values("fraction")
            if s.empty:
                continue
            ax.errorbar(s.fraction, 100 * s["mean"], yerr=100 * s.std_all, color=ROW_COLOR[row], ls=ls,
                        marker=mk, ms=6, capsize=3, label=f"{ROW_LABEL[row]} ({size})")
    d = sub[sub.row == "D"].sort_values("fraction")
    if not d.empty:
        ax.errorbar(d.fraction, 100 * d["mean"], yerr=100 * d.std_all, color=ROW_COLOR["D"], marker="D", ms=6,
                    capsize=3, label=ROW_LABEL["D"])
    ax.set_xscale("log")
    ax.set_xticks([0.01, 0.1, 0.5, 1.0], ["1%", "10%", "50%", "100%"])
    ax.set_xlabel("fração do treino rotulado (escala log)")
    ax.set_ylabel(f"acurácia no teste, {PROBE_LABEL[probe]} (%)")
    ax.set_title(f"Eficiência de rótulos — {PROBE_LABEL[probe]}", loc="left")
    if own:
        ax.legend(fontsize=8, loc="lower right", ncol=2)
        fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def fig_all_probes(agg: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=False)
    for ax, probe in zip(axes, ("knn", "linear_svm", "rbf_svm")):
        fig_label_efficiency(agg, path, probe, ax=ax)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(path, dpi=150); plt.close(fig)


def fig_gap_vs_params(gaps: pd.DataFrame, params: dict[str, int], path: Path) -> None:
    """Lacuna C − B (SVM linear) vs nº de parâmetros do aluno, uma linha por fração de rótulos."""
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    g = gaps[(gaps.probe == "linear_svm") & (gaps.data_frac == 1.0)]
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    for color, (frac, s) in zip(palette, g.groupby("fraction")):
        s = s.assign(p=s["size"].map(params)).sort_values("p")
        ax.errorbar(s.p, 100 * s.gap_C_minus_B, yerr=100 * s.spread, color=color, marker="o", ms=7, capsize=4,
                    label=f"{int(round(frac * 100))}% rótulos")
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xscale("log")
    ps = sorted(params.values())
    ax.set_xticks(ps, [f"{p / 1e6:.2f} M" for p in ps])
    ax.minorticks_off()
    ax.set_xlim(ps[0] / 1.6, ps[-1] * 1.6)
    ax.set_xlabel("parâmetros do aluno (escala log)")
    ax.set_ylabel("C − B, SVM linear (pontos %)")
    ax.set_title("Lacuna destilação − treino direto vs tamanho do aluno", loc="left")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def fig_ablation(gaps: pd.DataFrame, path: Path) -> None:
    """Grade da ablação: lacuna C − B por (fração de dados não rotulados x tamanho), por fração de rótulos."""
    g = gaps[gaps.probe == "linear_svm"]
    fracs = sorted(g.fraction.unique())
    fig, axes = plt.subplots(1, len(fracs), figsize=(3.6 * len(fracs), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    dfracs = sorted(g.data_frac.unique())
    colors = {df_: c for df_, c in zip(dfracs, ["#9fc3ee", "#2a78d6"])}
    w = 0.38
    for ax, frac in zip(axes, fracs):
        s = g[g.fraction == frac]
        for i, size in enumerate(["small", "large"]):
            for j, df_ in enumerate(dfracs):
                r = s[(s["size"] == size) & (s.data_frac == df_)]
                if r.empty:
                    continue
                x = i + (j - (len(dfracs) - 1) / 2) * w
                ax.bar(x, 100 * r.gap_C_minus_B.iloc[0], w * 0.92, yerr=100 * r.spread.iloc[0], capsize=3, ecolor=INK2,
                       color=colors[df_], label=f"{int(df_ * 100)}% do pool" if (i == 0 and frac == fracs[0]) else None)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks([0, 1], ["small", "large"])
        ax.set_title(f"{int(round(frac * 100))}% rótulos", loc="left")
    axes[0].set_ylabel("C − B, SVM linear (pontos %)")
    fig.legend(loc="lower center", ncol=2, fontsize=8)
    fig.suptitle("Ablação: quantidade de dados não rotulados x tamanho do aluno", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(path, dpi=160); plt.close(fig)


def fig_tsne(cfg: ExperimentConfig, path_tpl: str) -> list[Path]:
    """Painel de quatro projeções t-SNE (A, B, C, D) para cada tamanho de aluno."""
    p = cfg.results_dir / "tsne.npz"
    if not p.exists():
        return []
    z = np.load(p)
    labels, names = z["labels"], class_names(cfg)
    out = []
    for size, seeds in cfg.seeds.items():
        keys = [f"A_{size}_seed{seeds[0]}", f"B_{size}_seed{seeds[0]}", f"C_{size}_seed{seeds[0]}", "D_teacher"]
        if not all(k in z for k in keys):
            continue
        fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
        for ax, key, row in zip(axes, keys, "ABCD"):
            emb = z[key]
            for c in range(int(labels.max()) + 1):
                m = labels == c
                ax.scatter(emb[m, 0], emb[m, 1], s=5, color=CLASS_COLORS[c % 10], marker=CLASS_MARKERS[c % 10],
                           label=names[c] if c < len(names) else str(c), linewidths=0)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            ax.set_title(ROW_LABEL[row] + ("" if row == "D" else f" ({size})"), loc="left", fontsize=10)
        h, lab = axes[0].get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=min(10, len(lab)), fontsize=8, markerscale=3)
        fig.suptitle(f"t-SNE das features de teste (mesmos parâmetros p/ todos) — aluno {size}, semente {seeds[0]}",
                     x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        path = Path(path_tpl.format(size=size))
        fig.savefig(path, dpi=150); plt.close(fig)
        out.append(path)
    return out


def fig_training_curves(cfg: ExperimentConfig, path: Path) -> None:
    """Perda e std das features do backbone por época, para B e C (estudo principal)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for j, row in enumerate(("B", "C")):
        for d in sorted(glob.glob(str(cfg.results_dir / "runs" / f"{row}_*"))):
            name = Path(d).name
            if "_data" in name or not (Path(d) / "history.csv").exists():
                continue
            h = pd.read_csv(Path(d) / "history.csv")
            size = name.split("_")[1]
            ls = SIZE_STYLE.get(size, ("-", "o"))[0]
            axes[0, j].plot(h.epoch, h.loss, ls=ls, color=ROW_COLOR[row], alpha=0.85, lw=1.5)
            axes[1, j].plot(h.epoch, h.feat_std, ls=ls, color=ROW_COLOR[row], alpha=0.85, lw=1.5)
        axes[0, j].set_title(f"{ROW_LABEL[row]} — perda por época (— small, -- large)", loc="left", fontsize=10)
        axes[1, j].set_title("std por dimensão das features L2-normalizadas (colapso -> 0)", loc="left", fontsize=10)
        axes[1, j].set_xlabel("época")
        axes[1, j].set_ylim(bottom=0)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# =============================================================================
# Relatório completo
# =============================================================================
def run_report(cfg: ExperimentConfig) -> None:
    _style()
    rd = cfg.results_dir
    df = pd.read_csv(rd / "probe_results.csv")
    diag = pd.read_csv(rd / "diagnostics.csv")
    bench = load_json(rd / "benchmark.json") if (rd / "benchmark.json").exists() else {}
    tcost = load_json(rd / "teacher_cost.json") if (rd / "teacher_cost.json").exists() else {}
    sanity = load_json(rd / "teacher_sanity.json") if (rd / "teacher_sanity.json").exists() else {}
    runs = [load_json(Path(p)) for p in glob.glob(str(rd / "runs" / "*" / "summary.json"))]
    runs_df = pd.DataFrame(runs)

    # ---------------- custo dos encoders (parâmetros / FLOPs / latência)
    enc_cost = bench.get("encoders", {})
    def cost_of(row: str, size: str) -> dict[str, Any]:
        key = "teacher" if row == "D" else f"student_{size}"
        e = enc_cost.get(key, {})
        return {"params": e.get("params"), "feat_dim": e.get("feat_dim"), "gflops": e.get("gflops"),
                "latency_gpu_ms_per_img": e.get("latency_gpu", {}).get("ms_per_image"),
                "latency_cpu_ms_b1": e.get("latency_cpu", {}).get("ms_per_batch")}

    # ---------------- tabela de resultados (encoder x fração x sonda)
    agg = aggregate(df)
    extra = pd.DataFrame([{**{"row": r, "size": s}, **cost_of(r, s)} for r, s in
                          agg[["row", "size"]].drop_duplicates().itertuples(index=False)])
    table = agg.merge(extra, on=["row", "size"], how="left")
    table.to_csv(rd / "results_table.csv", index=False)

    gaps = gap_table(agg)
    gaps.to_csv(rd / "gap_table.csv", index=False)
    abl = gaps[gaps.probe == "linear_svm"].pivot_table(index=["fraction", "size"], columns="data_frac",
                                                        values="gap_C_minus_B")
    abl.to_csv(rd / "ablation_grid.csv")

    diag_agg = diag.groupby(["row", "size", "data_frac"]).agg(
        nmi=("nmi", "mean"), nmi_std=("nmi", "std"), ari=("ari", "mean"), ari_std=("ari", "std"),
        emb_std_ratio=("emb_std_ratio", "mean"), n=("nmi", "size")).reset_index().fillna(0.0)
    diag_agg.to_csv(rd / "diagnostics_table.csv", index=False)

    # ---------------- custo em GPU-horas (Seção 6, pergunta 3)
    cost_rows = []
    t_cache_h = tcost.get("teacher_cache_gpu_hours", np.nan)
    t_pre_h = cfg.teacher_pretrain_gpu_hours
    main = runs_df[(runs_df.data_frac == 1.0) & (runs_df.tag.fillna("") == "")] if not runs_df.empty else runs_df
    for size in cfg.student_sizes:
        for frac in cfg.label_fractions:
            g = gaps[(gaps["size"] == size) & (gaps.data_frac == 1.0) & (gaps.fraction == frac)
                     & (gaps.probe == "linear_svm")]
            if g.empty:
                continue
            hb = main[(main.row == "B") & (main["size"] == size)].train_gpu_hours.mean()
            hc = main[(main.row == "C") & (main["size"] == size)].train_gpu_hours.mean()
            gap = float(g.gap_C_minus_B.iloc[0])
            extra_local = hc + t_cache_h - hb          # GPU-h a mais de C (sem o pré-treino do professor)
            extra_total = extra_local + t_pre_h        # incluindo o pré-treino publicado
            cost_rows.append({
                "size": size, "label_fraction": frac, "gap_C_minus_B_pts": 100 * gap,
                "gpu_h_B": hb, "gpu_h_C_student": hc, "gpu_h_teacher_cache_one_off": t_cache_h,
                "gpu_h_teacher_pretrain_published": t_pre_h,
                # custo líquido local de C em relação a B (negativo = C foi MAIS BARATO que B)
                "extra_gpu_h_C_local": extra_local,
                "extra_gpu_h_C_with_pretrain": extra_total,
                "pts_per_extra_gpu_h_with_pretrain": 100 * gap / extra_total,
            })
    cost_df = pd.DataFrame(cost_rows)
    cost_df.to_csv(rd / "cost_table.csv", index=False)

    # ---------------- figuras
    fig_label_efficiency(agg, rd / "fig_label_efficiency.png")
    fig_all_probes(agg, rd / "fig_label_efficiency_all_probes.png")
    params = {s: cost_of("B", s)["params"] for s in cfg.student_sizes}
    if all(params.values()):
        fig_gap_vs_params(gaps, params, rd / "fig_gap_vs_params.png")
    if gaps.data_frac.nunique() > 1:
        fig_ablation(gaps, rd / "fig_ablation.png")
    tsne_paths = fig_tsne(cfg, str(rd / "fig_tsne_{size}.png"))
    fig_training_curves(cfg, rd / "fig_training_curves.png")

    # ---------------- summary.md
    write_summary(cfg, agg, gaps, diag_agg, cost_df, runs_df, df, bench, tcost, sanity, enc_cost, tsne_paths)
    log.info("Relatório gerado em %s", rd)


def write_summary(cfg, agg, gaps, diag_agg, cost_df, runs_df, df, bench, tcost, sanity, enc_cost, tsne_paths):
    rd = cfg.results_dir
    L: list[str] = [f"# Resultados — {cfg.dataset}", "",
                    "Gerado automaticamente por `python -m mo433_projeto_final report`. Valores em % de acurácia "
                    "no teste, média ± desvio sobre todas as execuções (semente x sorteio do subconjunto rotulado).",
                    "", "## Configuração", "",
                    f"- Dataset `{cfg.dataset}`, entrada do aluno {cfg.img_size}px, professor `{cfg.teacher_name}` "
                    f"@ {cfg.teacher_res}px; K={cfg.n_views} views/imagem; {cfg.epochs} épocas, batch "
                    f"{cfg.batch_size}, AdamW (lr B={cfg.lr_for('B'):g}, lr C={cfg.lr_for('C'):g}, "
                    f"wd={cfg.weight_decay}), warm-up {cfg.warmup_epochs} épocas + cosseno, bf16.",
                    f"- Sementes: {cfg.seeds}; sorteios de subconjunto rotulado por fração: {cfg.n_draws}.",
                    f"- Hardware: {bench.get('hardware', {})}", ""]
    if sanity:
        L += ["## Checagem de sanidade do professor", "",
              f"k-NN = {100 * sanity['knn_acc']:.1f}% (acaso {100 * sanity['chance']:.0f}%), std dos embeddings "
              f"= {sanity['emb_std']:.4f} (ref. 1/sqrt(d) = {sanity['emb_std_ref']:.4f}), NMI = {sanity['nmi']:.3f}, "
              f"ARI = {sanity['ari']:.3f} -> **{'aprovado' if sanity['passed'] else 'REPROVADO'}**.", ""]

    # custo dos encoders
    if enc_cost:
        rows = []
        for k, e in enc_cost.items():
            rows.append({"encoder": k, "params (M)": f"{e['params'] / 1e6:.2f}", "dim": e["feat_dim"],
                         "GFLOPs/img": f"{e['gflops']:.3f}", "entrada": f"{e['input_res']}px",
                         "GPU ms/img (b256 fp16)": f"{e['latency_gpu']['ms_per_image']:.4f}",
                         "CPU ms (b1 fp32)": f"{e['latency_cpu']['ms_per_batch']:.2f}"})
        L += ["## Encoders: parâmetros, FLOPs e latência", "", _md(pd.DataFrame(rows)), ""]

    # tabela principal (estudo principal, data_frac=1)
    m = agg[agg.data_frac == 1.0].copy()
    m["enc"] = m.apply(lambda r: ROW_LABEL[r.row] + ("" if r.row == "D" else f" ({r['size']})"), axis=1)
    for probe in ("linear_svm", "knn", "rbf_svm"):
        s = m[m.probe == probe]
        piv = s.pivot_table(index="enc", columns="fraction", values="mean", aggfunc="first")
        sd = s.pivot_table(index="enc", columns="fraction", values="std_all", aggfunc="first")
        out = pd.DataFrame({"encoder": piv.index})
        for f in piv.columns:
            out[f"{int(round(f * 100))}% rótulos"] = [_pm(piv.loc[i, f], sd.loc[i, f]) for i in piv.index]
        L += [f"## Tabela de resultados — {PROBE_LABEL[probe]}", "", _md(out), ""]

    # lacunas
    g = gaps[(gaps.data_frac == 1.0)].copy()
    g = g.assign(gap=lambda x: (100 * x.gap_C_minus_B).round(2), spread_=lambda x: (100 * x.spread).round(2))
    L += ["## Lacuna C − B (estudo principal)", "",
          "`spread` = sqrt(std_B² + std_C²) sobre semente x sorteio; *conclusivo* só se |gap| > spread.", "",
          _md(g[["size", "fraction", "probe", "gap", "spread_", "verdict"]].rename(columns={"spread_": "spread"})), ""]

    # execuções individuais do aluno grande (só 2 sementes)
    lg = df[(df["size"] == "large") & (df.data_frac == 1.0) & (df.probe == "linear_svm") & df.row.isin(["A", "B", "C"])]
    if not lg.empty:
        ind = lg.groupby(["row", "seed", "fraction"]).acc.mean().unstack("fraction").mul(100).round(2).reset_index()
        ind.columns = [str(c) for c in ind.columns]
        L += ["## Execuções individuais — aluno large (SVM linear, média dos sorteios)", "",
              "Com só 2 sementes o desvio-padrão é uma estimativa pobre (Seção 2); por isso as execuções aparecem "
              "individualmente.", "", _md(ind), ""]

    # ablação
    ab = gaps[(gaps.probe == "linear_svm")]
    if ab.data_frac.nunique() > 1:
        ab = ab.assign(cell=lambda x: [f"{100 * a:+.2f} ± {100 * b:.2f}" for a, b in zip(x.gap_C_minus_B, x.spread)])
        piv = ab.pivot_table(index=["fraction", "size"], columns="data_frac", values="cell", aggfunc="first")
        piv.columns = [f"pool {int(c * 100)}%" for c in piv.columns]
        L += ["## Ablação — lacuna C − B (SVM linear) por quantidade de dados não rotulados x tamanho", "",
              _md(piv.reset_index()), ""]

    # diagnósticos
    d = diag_agg.copy()
    d["NMI"] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(d.nmi, d.nmi_std)]
    d["ARI"] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(d.ari, d.ari_std)]
    d["std/ref"] = d.emb_std_ratio.round(2)
    L += ["## Diagnósticos não supervisionados (k-means no teste, k = nº de classes)", "",
          _md(d[["row", "size", "data_frac", "NMI", "ARI", "std/ref", "n"]]), ""]

    # custo
    if not cost_df.empty:
        c = cost_df.round(4)
        L += ["## Custo em GPU-horas e ganho por GPU-hora", "",
              f"Cache do professor (one-off, compartilhado por todas as execuções de C): "
              f"{tcost.get('teacher_cache_gpu_hours', float('nan')):.4f} GPU-h nesta máquina. Pré-treino publicado do "
              f"professor (estimativa, V100): {cfg.teacher_pretrain_gpu_hours:.0f} GPU-h.", "", _md(c), ""]
    if not runs_df.empty:
        tot = runs_df.train_gpu_hours.sum() + tcost.get("teacher_cache_gpu_hours", 0)
        L += [f"GPU-horas totais de treino deste estudo (todas as execuções + cache do professor): {tot:.2f} h.", ""]
        cc = runs_df[["name", "collapse_check"]].dropna()
        bad = [r["name"] for _, r in cc.iterrows() if not r["collapse_check"].get("passed", True)]
        L += [f"Checagem de colapso ao fim de cada treino: {len(cc) - len(bad)}/{len(cc)} aprovados"
              + (f"; reprovados: {bad}" if bad else "") + ".", ""]

    L += ["## Figuras", "", "- `fig_label_efficiency.png` — curva de eficiência de rótulos (figura principal)",
          "- `fig_label_efficiency_all_probes.png` — as três sondas", "- `fig_gap_vs_params.png` — C − B vs parâmetros",
          "- `fig_ablation.png` — grade da ablação", "- `fig_training_curves.png` — perdas e std por época"]
    L += [f"- `{p.name}` — painel t-SNE" for p in tsne_paths]
    (rd / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
