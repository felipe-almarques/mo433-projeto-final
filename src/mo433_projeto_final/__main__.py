"""Interface de linha de comando: ``python -m mo433_projeto_final <etapa> --dataset <ds>``.

Etapas (na ordem do pipeline; todas são idempotentes/retomáveis):

==============  ===============================================================
``prepare``     baixa o dataset, cria splits, subconjuntos rotulados e K views
``teacher``     professor: linha D, checagem de sanidade, cache das views
``benchmark``   vazão, parâmetros, FLOPs, latência e projeção de custo
``tune``        grade pequena e simétrica de lr para B e C (validação no treino)
``train``       uma execução (``--row B|C --size small|large --seed N``)
``train-all``   todas as execuções do estudo principal + ablação
``evaluate``    linha A, sondas e diagnósticos para todos os encoders
``report``      tabelas, figuras e ``summary.md``
``all``         tudo acima, em sequência
==============  ===============================================================

Cada etapa escreve um log próprio em ``results/<dataset>/logs/<etapa>.log`` e
cada treino também em ``logs/runs/<nome_da_execução>.log``.
"""

from __future__ import annotations

import argparse
import logging
from contextlib import contextmanager
from dataclasses import asdict
from typing import Iterator

from .config import ExperimentConfig, default_config
from .utils import describe_hardware, setup_logging

log = logging.getLogger("mo433.cli")


@contextmanager
def run_log(cfg: ExperimentConfig, name: str) -> Iterator[None]:
    """Duplica o log raiz num arquivo próprio da execução enquanto ela durar."""
    path = cfg.log_dir / "runs" / f"{name}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(path, mode="a", encoding="utf-8")
    fh.setFormatter(logging.getLogger().handlers[0].formatter)
    logging.getLogger().addHandler(fh)
    try:
        yield
    finally:
        logging.getLogger().removeHandler(fh)
        fh.close()


def _load_or_create_config(args: argparse.Namespace) -> ExperimentConfig:
    """Na etapa ``prepare`` cria/atualiza a config; nas demais, apenas a lê."""
    overrides = {k: v for k, v in {
        "epochs": args.epochs, "batch_size": args.batch_size, "n_views": args.n_views,
        "n_unlabeled": args.n_unlabeled, "num_workers": args.num_workers}.items() if v is not None}
    path = default_config(args.dataset).results_dir / "config.json"
    if args.stage in ("prepare", "all") and not path.exists():
        cfg = default_config(args.dataset)
    else:
        cfg = ExperimentConfig.load(args.dataset)
    for k, v in overrides.items():
        if getattr(cfg, k) != v:
            logging.getLogger("mo433.cli").warning("Sobrescrevendo config: %s = %s (antes %s)", k, v, getattr(cfg, k))
            setattr(cfg, k, v)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    cfg.save()
    return cfg


def stage_prepare(cfg: ExperimentConfig) -> None:
    from .data import describe_train_transform, generate_views, make_label_subsets, prepare_arrays

    prepare_arrays(cfg)
    make_label_subsets(cfg)
    generate_views(cfg)
    log.info("Transformação de treino:\n%s", describe_train_transform(cfg))


def stage_train_all(cfg: ExperimentConfig, ablation: bool = True) -> None:
    """Estudo principal (100% do pool) e depois a ablação (frações menores do pool).

    Execuções já concluídas (``summary.json`` presente) são puladas: a etapa pode
    ser interrompida e retomada. A ordem alterna B e C por semente, de modo que
    uma interrupção nunca deixa uma linha muito à frente da outra.
    """
    from .train import run_name, train_student

    plan = [(1.0, size, seed, row) for size, seeds in cfg.seeds.items() for seed in seeds for row in ("B", "C")]
    if ablation:
        plan += [(f, size, seed, row) for f in cfg.ablation_data_fractions
                 for size, seeds in cfg.seeds.items() for seed in seeds for row in ("B", "C")]
    log.info("Plano de treino: %d execuções", len(plan))
    for i, (frac, size, seed, row) in enumerate(plan, 1):
        name = run_name(row, size, seed, frac)
        if (cfg.results_dir / "runs" / name / "summary.json").exists():
            log.info("(%d/%d) %s já concluída — pulando.", i, len(plan), name)
            continue
        log.info("(%d/%d) iniciando %s", i, len(plan), name)
        with run_log(cfg, name):
            train_student(cfg, row, size, seed, data_frac=frac)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="mo433_projeto_final", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["prepare", "teacher", "benchmark", "tune", "train", "train-all",
                                     "evaluate", "report", "all"])
    p.add_argument("--dataset", default="eurosat", choices=["eurosat", "stl10", "smoke"])
    # sobrescritas da configuração (persistidas em config.json)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--n-views", type=int)
    p.add_argument("--n-unlabeled", type=int)
    p.add_argument("--num-workers", type=int)
    # etapa train
    p.add_argument("--row", choices=["B", "C"])
    p.add_argument("--size", choices=["small", "large"])
    p.add_argument("--seed", type=int)
    p.add_argument("--data-frac", type=float, default=1.0)
    # outras
    p.add_argument("--no-ablation", action="store_true", help="pula a ablação em train-all/evaluate")
    p.add_argument("--skip-tune", action="store_true", help="em `all`: não roda o ajuste de lr")
    p.add_argument("--n-jobs", type=int, default=4, help="processos paralelos nas sondas")
    p.add_argument("--random-teacher", action="store_true", help="(só smoke test) professor sem pesos pré-treinados")
    args = p.parse_args(argv)

    cfg = _load_or_create_config(args)
    setup_logging(cfg.log_dir, args.stage if args.stage != "train" else "train_single")
    log.info("Dataset=%s | hardware=%s", cfg.dataset, describe_hardware())
    log.info("Configuração: %s", asdict(cfg))

    stages = ["prepare", "teacher", "benchmark", "tune", "train-all", "evaluate", "report"] \
        if args.stage == "all" else [args.stage]
    if args.stage == "all" and args.skip_tune:
        stages.remove("tune")
    for st in stages:
        if args.stage == "all":
            setup_logging(cfg.log_dir, st)
        if st == "prepare":
            stage_prepare(cfg)
        elif st == "teacher":
            from .teacher import run_teacher_stage
            run_teacher_stage(cfg, pretrained=not args.random_teacher)
        elif st == "benchmark":
            from .benchmark import run_benchmark
            run_benchmark(cfg)
        elif st == "tune":
            from .tune import run_tuning
            if (cfg.results_dir / "tuning.json").exists():
                log.info("tuning.json já existe — lr já escolhidos: B=%s C=%s", cfg.lr_B, cfg.lr_C)
            else:
                run_tuning(cfg)
                cfg = ExperimentConfig.load(cfg.dataset)
        elif st == "train":
            from .train import run_name, train_student
            if None in (args.row, args.size, args.seed):
                p.error("train exige --row, --size e --seed")
            with run_log(cfg, run_name(args.row, args.size, args.seed, args.data_frac)):
                train_student(cfg, args.row, args.size, args.seed, data_frac=args.data_frac)
        elif st == "train-all":
            stage_train_all(cfg, ablation=not args.no_ablation)
        elif st == "evaluate":
            from .evaluate import run_evaluation
            run_evaluation(cfg, n_jobs=args.n_jobs, include_ablation=not args.no_ablation)
        elif st == "report":
            from .report import run_report
            run_report(cfg)
        log.info("Etapa '%s' concluída.", st)


if __name__ == "__main__":
    main()
