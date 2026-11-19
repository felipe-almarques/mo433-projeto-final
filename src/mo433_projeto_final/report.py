"""Relatório: tabelas CSV, figuras e ``summary.md``.

Estatística, para cada (linha, tamanho, fração do pool, fração de rótulos, sonda):
``mean`` = média sobre todas as execuções (semente x sorteio); ``std_seed`` = desvio entre
as médias por semente; ``std_all`` = desvio sobre todas as execuções (inclui a variância de
*quais* imagens são rotuladas). Lacuna C − B: ``gap = mean_C − mean_B``,
``spread = sqrt(std_all_B² + std_all_C²)``, *conclusiva* só se ``|gap| > spread``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import TEACHER_PRETRAIN_GPU_HOURS, Config, load_json  # noqa: E402
from .data import class_names  # noqa: E402

log = logging.getLogger("mo433.report")

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ROW_COLOR = {"A": "#8f8d87", "B": "#2a78d6", "C": "#eb6834", "D": "#1baf7a"}
ROW_LABEL = {"A": "A: aleatório", "B": "B: VICReg direto", "C": "C: destilado", "D": "D: professor ViT-S/16"}
SIZE_STYLE = {"small": ("-", "o"), "large": ("--", "s")}
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
                "#52514e", "#b5b3ad"]
CLASS_MARKERS = ["o"] * 8 + ["^", "v"]
PROBE_LABEL = {"knn": "k-NN", "linear_svm": "SVM linear", "rbf_svm": "SVM RBF"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "lines.linewidth": 2,
    "legend.frameon": False,
})


# ================================================================== agregação
def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["row", "size", "data_frac", "fraction", "probe"]
    per_seed = df.groupby(keys + ["seed"], as_index=False)["acc"].mean()
    a = per_seed.groupby(keys).agg(mean=("acc", "mean"), std_seed=("acc", "std"), n_seeds=("acc", "size"))
    b = df.groupby(keys).agg(std_all=("acc", "std"), n_runs=("acc", "size"))
    return a.join(b).reset_index().fillna({"std_seed": 0.0, "std_all": 0.0})


def gap_table(agg: pd.DataFrame) -> pd.DataFrame:
    keys = ["size", "data_frac", "fraction", "probe"]
    cols = ["mean", "std_all", "std_seed"]
    j = agg[agg.row == "B"].set_index(keys)[cols].join(agg[agg.row == "C"].set_index(keys)[cols],
                                                       lsuffix="_B", rsuffix="_C", how="inner")
    j["gap_C_minus_B"] = j.mean_C - j.mean_B
    j["spread"] = np.sqrt(j.std_all_B ** 2 + j.std_all_C ** 2)
    j["spread_seed"] = np.sqrt(j.std_seed_B ** 2 + j.std_seed_C ** 2)
    j["conclusive"] = j.gap_C_minus_B.abs() > j.spread
    j["verdict"] = np.where(~j.conclusive, "inconclusivo", np.where(j.gap_C_minus_B > 0, "C > B", "B > C"))
    return j.reset_index()


def _md(df: pd.DataFrame) -> str:
    """DataFrame -> tabela Markdown."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join(lines)


def _save(fig, path: Path, dpi: int = 160, **layout) -> None:
    fig.tight_layout(**layout)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ===================================================================== figuras
def fig_label_efficiency(agg: pd.DataFrame, ax, probe: str = "linear_svm") -> None:
    """Curva de eficiência de rótulos: x = fração rotulada (log), y = acurácia."""
    sub = agg[(agg.probe == probe) & (agg.data_frac == 1.0)]
    for row in ("A", "B", "C"):
        for size, (ls, mk) in SIZE_STYLE.items():
            s = sub[(sub.row == row) & (sub["size"] == size)].sort_values("fraction")
            ax.errorbar(s.fraction, 100 * s["mean"], yerr=100 * s.std_all, color=ROW_COLOR[row], ls=ls,
                        marker=mk, ms=6, capsize=3, label=f"{ROW_LABEL[row]} ({size})")
    d = sub[sub.row == "D"].sort_values("fraction")
    ax.errorbar(d.fraction, 100 * d["mean"], yerr=100 * d.std_all, color=ROW_COLOR["D"], marker="D", ms=6,
                capsize=3, label=ROW_LABEL["D"])
    ax.set_xscale("log")
    ax.set_xticks([0.01, 0.1, 0.5, 1.0], ["1%", "10%", "50%", "100%"])
    ax.set_xlabel("fração do treino rotulado (escala log)")
    ax.set_ylabel(f"acurácia no teste, {PROBE_LABEL[probe]} (%)")
    ax.set_title(f"Eficiência de rótulos — {PROBE_LABEL[probe]}", loc="left")


def fig_gap_vs_params(gaps: pd.DataFrame, params: dict[str, int], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    g = gaps[(gaps.probe == "linear_svm") & (gaps.data_frac == 1.0)]
    for color, (frac, s) in zip(["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], g.groupby("fraction")):
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
    _save(fig, path)


def fig_ablation(gaps: pd.DataFrame, path: Path) -> None:
    """Lacuna C − B por (fração do pool x tamanho), um painel por fração de rótulos."""
    g = gaps[gaps.probe == "linear_svm"]
    fracs, dfracs = sorted(g.fraction.unique()), sorted(g.data_frac.unique())
    colors = dict(zip(dfracs, ["#9fc3ee", "#2a78d6"]))
    fig, axes = plt.subplots(1, len(fracs), figsize=(3.6 * len(fracs), 3.8), sharey=True)
    w = 0.38
    for ax, frac in zip(axes, fracs):
        for i, size in enumerate(["small", "large"]):
            for j, df_ in enumerate(dfracs):
                r = g[(g.fraction == frac) & (g["size"] == size) & (g.data_frac == df_)]
                ax.bar(i + (j - (len(dfracs) - 1) / 2) * w, 100 * r.gap_C_minus_B.iloc[0], w * 0.92,
                       yerr=100 * r.spread.iloc[0], capsize=3, ecolor=INK2, color=colors[df_],
                       label=f"{int(df_ * 100)}% do pool" if (i == 0 and frac == fracs[0]) else None)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks([0, 1], ["small", "large"])
        ax.set_title(f"{int(round(frac * 100))}% rótulos", loc="left")
    axes[0].set_ylabel("C − B, SVM linear (pontos %)")
    fig.legend(loc="lower center", ncol=2, fontsize=8)
    fig.suptitle("Ablação: quantidade de dados não rotulados x tamanho do aluno", x=0.01, ha="left")
    _save(fig, path, rect=(0, 0.08, 1, 1))


def fig_tsne(cfg: Config) -> list[Path]:
    """Painel de quatro t-SNE (A, B, C, D) para cada tamanho de aluno."""
    z = np.load(cfg.results_dir / "tsne.npz")
    labels, names = z["labels"], class_names(cfg)
    paths = []
    for size, seeds in cfg.seeds.items():
        keys = [f"A_{size}_seed{seeds[0]}", f"B_{size}_seed{seeds[0]}", f"C_{size}_seed{seeds[0]}", "D_teacher"]
        fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
        for ax, key, row in zip(axes, keys, "ABCD"):
            for c in range(int(labels.max()) + 1):
                m = labels == c
                ax.scatter(z[key][m, 0], z[key][m, 1], s=5, color=CLASS_COLORS[c % 10], marker=CLASS_MARKERS[c % 10],
                           label=names[c], linewidths=0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_title(ROW_LABEL[row] + ("" if row == "D" else f" ({size})"), loc="left", fontsize=10)
        h, lab = axes[0].get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=min(10, len(lab)), fontsize=8, markerscale=3)
        fig.suptitle(f"t-SNE das features de teste (mesmos parâmetros p/ todos) — aluno {size}, semente {seeds[0]}",
                     x=0.01, ha="left")
        paths.append(cfg.results_dir / f"fig_tsne_{size}.png")
        _save(fig, paths[-1], dpi=150, rect=(0, 0.07, 1, 1))
    return paths


def fig_training_curves(cfg: Config, path: Path) -> None:
    """Perda e std das features por época para B e C (estudo principal)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for j, row in enumerate(("B", "C")):
        for d in sorted((cfg.results_dir / "runs").glob(f"{row}_*")):
            if "_data" in d.name or not (d / "history.csv").exists():
                continue
            h = pd.read_csv(d / "history.csv")
            ls = SIZE_STYLE[d.name.split("_")[1]][0]
            axes[0, j].plot(h.epoch, h.loss, ls=ls, color=ROW_COLOR[row], alpha=0.85, lw=1.5)
            axes[1, j].plot(h.epoch, h.feat_std, ls=ls, color=ROW_COLOR[row], alpha=0.85, lw=1.5)
        axes[0, j].set_title(f"{ROW_LABEL[row]} — perda por época (— small, -- large)", loc="left", fontsize=10)
        axes[1, j].set_title("std por dimensão das features L2-normalizadas (colapso -> 0)", loc="left", fontsize=10)
        axes[1, j].set_xlabel("época")
        axes[1, j].set_ylim(bottom=0)
    _save(fig, path, dpi=150)


# ===================================================================== relatório
def run_report(cfg: Config) -> None:
    rd = cfg.results_dir
    df = pd.read_csv(rd / "probe_results.csv")
    diag = pd.read_csv(rd / "diagnostics.csv")
    bench = load_json(rd / "benchmark.json")
    tcost = load_json(rd / "teacher_cost.json")
    sanity = load_json(rd / "teacher_sanity.json")
    runs = pd.DataFrame([load_json(p) for p in (rd / "runs").glob("*/summary.json")])

    # tabela de resultados + parâmetros/FLOPs/latência de cada encoder
    agg = aggregate(df)
    enc_cost = bench["encoders"]

    def cost_of(row: str, size: str) -> dict:
        e = enc_cost["teacher" if row == "D" else f"student_{size}"]
        return {"params": e["params"], "feat_dim": e["feat_dim"], "gflops": e["gflops"],
                "latency_gpu_ms_per_img": e["latency_gpu"]["ms_per_image"],
                "latency_cpu_ms_b1": e["latency_cpu"]["ms_per_batch"]}

    extra = pd.DataFrame([{"row": r, "size": s, **cost_of(r, s)}
                          for r, s in agg[["row", "size"]].drop_duplicates().itertuples(index=False)])
    agg.merge(extra, on=["row", "size"], how="left").to_csv(rd / "results_table.csv", index=False)

    gaps = gap_table(agg)
    gaps.to_csv(rd / "gap_table.csv", index=False)
    gaps[gaps.probe == "linear_svm"].pivot_table(index=["fraction", "size"], columns="data_frac",
                                                 values="gap_C_minus_B").to_csv(rd / "ablation_grid.csv")

    diag_agg = diag.groupby(["row", "size", "data_frac"]).agg(
        nmi=("nmi", "mean"), nmi_std=("nmi", "std"), ari=("ari", "mean"), ari_std=("ari", "std"),
        emb_std_ratio=("emb_std_ratio", "mean"), n=("nmi", "size")).reset_index().fillna(0.0)
    diag_agg.to_csv(rd / "diagnostics_table.csv", index=False)

    # custo em GPU-horas: C paga o seu treino + o cache do professor (+ o pré-treino publicado)
    t_cache_h = tcost["teacher_cache_gpu_hours"]
    main = runs[(runs.data_frac == 1.0) & (runs.tag.fillna("") == "")]
    cost_rows = []
    for size in cfg.student_sizes:
        hb = main[(main.row == "B") & (main["size"] == size)].train_gpu_hours.mean()
        hc = main[(main.row == "C") & (main["size"] == size)].train_gpu_hours.mean()
        for frac in cfg.label_fractions:
            g = gaps[(gaps["size"] == size) & (gaps.data_frac == 1.0) & (gaps.fraction == frac)
                     & (gaps.probe == "linear_svm")]
            gap = float(g.gap_C_minus_B.iloc[0])
            extra_local = hc + t_cache_h - hb  # negativo = C foi mais barato que B
            extra_total = extra_local + TEACHER_PRETRAIN_GPU_HOURS
            cost_rows.append({"size": size, "label_fraction": frac, "gap_C_minus_B_pts": 100 * gap,
                              "gpu_h_B": hb, "gpu_h_C_student": hc, "gpu_h_teacher_cache_one_off": t_cache_h,
                              "gpu_h_teacher_pretrain_published": TEACHER_PRETRAIN_GPU_HOURS,
                              "extra_gpu_h_C_local": extra_local, "extra_gpu_h_C_with_pretrain": extra_total,
                              "pts_per_extra_gpu_h_with_pretrain": 100 * gap / extra_total})
    cost_df = pd.DataFrame(cost_rows)
    cost_df.to_csv(rd / "cost_table.csv", index=False)

    # figuras
    fig, ax = plt.subplots(figsize=(7, 4.6))
    fig_label_efficiency(agg, ax)
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    _save(fig, rd / "fig_label_efficiency.png")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, probe in zip(axes, PROBE_LABEL):
        fig_label_efficiency(agg, ax, probe)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=8)
    _save(fig, rd / "fig_label_efficiency_all_probes.png", dpi=150, rect=(0, 0.1, 1, 1))
    fig_gap_vs_params(gaps, {s: cost_of("B", s)["params"] for s in cfg.student_sizes}, rd / "fig_gap_vs_params.png")
    fig_ablation(gaps, rd / "fig_ablation.png")
    tsne_paths = fig_tsne(cfg)
    fig_training_curves(cfg, rd / "fig_training_curves.png")

    write_summary(cfg, agg, gaps, diag_agg, cost_df, runs, df, bench, tcost, sanity, tsne_paths)
    log.info("Relatório gerado em %s", rd)


def write_summary(cfg, agg, gaps, diag_agg, cost_df, runs, df, bench, tcost, sanity, tsne_paths) -> None:
    pm = lambda m, s: f"{100 * m:.1f} ± {100 * s:.1f}"  # noqa: E731
    L = [f"# Resultados — {cfg.dataset}", "",
         "Gerado automaticamente por `python -m mo433_projeto_final report`. Valores em % de acurácia "
         "no teste, média ± desvio sobre todas as execuções (semente x sorteio do subconjunto rotulado).",
         "", "## Configuração", "",
         f"- Dataset `{cfg.dataset}`, entrada do aluno {cfg.img_size}px, professor `{cfg.teacher_name}` "
         f"@ {cfg.teacher_res}px; K={cfg.n_views} views/imagem; {cfg.epochs} épocas, batch "
         f"{cfg.batch_size}, AdamW (lr B={cfg.lr_for('B'):g}, lr C={cfg.lr_for('C'):g}, "
         f"wd={cfg.weight_decay}), warm-up {cfg.warmup_epochs} épocas + cosseno, bf16.",
         f"- Sementes: {cfg.seeds}; sorteios de subconjunto rotulado por fração: {cfg.n_draws}.",
         f"- Hardware: {bench['hardware']}", "",
         "## Checagem de sanidade do professor", "",
         f"k-NN = {100 * sanity['knn_acc']:.1f}% (acaso {100 * sanity['chance']:.0f}%), std dos embeddings "
         f"= {sanity['emb_std']:.4f} (ref. 1/sqrt(d) = {sanity['emb_std_ref']:.4f}), NMI = {sanity['nmi']:.3f}, "
         f"ARI = {sanity['ari']:.3f} -> **{'aprovado' if sanity['passed'] else 'REPROVADO'}**.", ""]

    enc = pd.DataFrame([{"encoder": k, "params (M)": f"{e['params'] / 1e6:.2f}", "dim": e["feat_dim"],
                         "GFLOPs/img": f"{e['gflops']:.3f}", "entrada": f"{e['input_res']}px",
                         "GPU ms/img (b256 fp16)": f"{e['latency_gpu']['ms_per_image']:.4f}",
                         "CPU ms (b1 fp32)": f"{e['latency_cpu']['ms_per_batch']:.2f}"}
                        for k, e in bench["encoders"].items()])
    L += ["## Encoders: parâmetros, FLOPs e latência", "", _md(enc), ""]

    # tabelas principais (pool 100%)
    m = agg[agg.data_frac == 1.0].copy()
    m["enc"] = m.apply(lambda r: ROW_LABEL[r.row] + ("" if r.row == "D" else f" ({r['size']})"), axis=1)
    for probe in ("linear_svm", "knn", "rbf_svm"):
        s = m[m.probe == probe]
        mean = s.pivot_table(index="enc", columns="fraction", values="mean", aggfunc="first")
        sd = s.pivot_table(index="enc", columns="fraction", values="std_all", aggfunc="first")
        out = pd.DataFrame({"encoder": mean.index})
        for f in mean.columns:
            out[f"{int(round(f * 100))}% rótulos"] = [pm(mean.loc[i, f], sd.loc[i, f]) for i in mean.index]
        L += [f"## Tabela de resultados — {PROBE_LABEL[probe]}", "", _md(out), ""]

    g = gaps[gaps.data_frac == 1.0].assign(gap=lambda x: (100 * x.gap_C_minus_B).round(2),
                                           spread=lambda x: (100 * x.spread).round(2))
    L += ["## Lacuna C − B (estudo principal)", "",
          "`spread` = sqrt(std_B² + std_C²) sobre semente x sorteio; *conclusivo* só se |gap| > spread.", "",
          _md(g[["size", "fraction", "probe", "gap", "spread", "verdict"]]), ""]

    lg = df[(df["size"] == "large") & (df.data_frac == 1.0) & (df.probe == "linear_svm") & df.row.isin(["A", "B", "C"])]
    ind = lg.groupby(["row", "seed", "fraction"]).acc.mean().unstack("fraction").mul(100).round(2).reset_index()
    ind.columns = [str(c) for c in ind.columns]
    L += ["## Execuções individuais — aluno large (SVM linear, média dos sorteios)", "",
          "Com só 2 sementes o desvio-padrão é uma estimativa pobre (Seção 2); por isso as execuções aparecem "
          "individualmente.", "", _md(ind), ""]

    ab = gaps[gaps.probe == "linear_svm"]
    ab = ab.assign(cell=[f"{100 * a:+.2f} ± {100 * b:.2f}" for a, b in zip(ab.gap_C_minus_B, ab.spread)])
    piv = ab.pivot_table(index=["fraction", "size"], columns="data_frac", values="cell", aggfunc="first")
    piv.columns = [f"pool {int(c * 100)}%" for c in piv.columns]
    L += ["## Ablação — lacuna C − B (SVM linear) por quantidade de dados não rotulados x tamanho", "",
          _md(piv.reset_index()), ""]

    d = diag_agg.copy()
    d["NMI"] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(d.nmi, d.nmi_std)]
    d["ARI"] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(d.ari, d.ari_std)]
    d["std/ref"] = d.emb_std_ratio.round(2)
    L += ["## Diagnósticos não supervisionados (k-means no teste, k = nº de classes)", "",
          _md(d[["row", "size", "data_frac", "NMI", "ARI", "std/ref", "n"]]), ""]

    L += ["## Custo em GPU-horas e ganho por GPU-hora", "",
          f"Cache do professor (one-off, compartilhado por todas as execuções de C): "
          f"{tcost['teacher_cache_gpu_hours']:.4f} GPU-h nesta máquina. Pré-treino publicado do "
          f"professor (estimativa, V100): {TEACHER_PRETRAIN_GPU_HOURS:.0f} GPU-h.", "", _md(cost_df.round(4)), ""]
    total = runs.train_gpu_hours.sum() + tcost["teacher_cache_gpu_hours"]
    checks = runs[["name", "collapse_check"]].dropna()
    bad = [r["name"] for _, r in checks.iterrows() if not r["collapse_check"]["passed"]]
    L += [f"GPU-horas totais de treino deste estudo (todas as execuções + cache do professor): {total:.2f} h.", "",
          f"Checagem de colapso ao fim de cada treino: {len(checks) - len(bad)}/{len(checks)} aprovados"
          + (f"; reprovados: {bad}" if bad else "") + ".", ""]

    L += ["## Figuras", "", "- `fig_label_efficiency.png` — curva de eficiência de rótulos (figura principal)",
          "- `fig_label_efficiency_all_probes.png` — as três sondas", "- `fig_gap_vs_params.png` — C − B vs parâmetros",
          "- `fig_ablation.png` — grade da ablação", "- `fig_training_curves.png` — perdas e std por época"]
    L += [f"- `{p.name}` — painel t-SNE" for p in tsne_paths]
    (cfg.results_dir / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
