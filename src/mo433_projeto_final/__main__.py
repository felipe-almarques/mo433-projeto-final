"""CLI: ``python -m mo433_projeto_final <etapa>``.

Etapas (na ordem; todas retomáveis — o que já foi calculado é pulado):

  prepare     baixa o EuroSAT, cria splits, subconjuntos rotulados e as K views
  teacher     professor: linha D, checagem de sanidade, cache das features das views
  benchmark   vazão, parâmetros, FLOPs e latência
  tune        grade simétrica de lr para B e C (validação tirada do treino)
  train       uma execução (--row B|C --size small|large --seed N [--data-frac 0.5])
  train-all   todas as execuções do estudo principal + ablação
  evaluate    linha A, sondas e diagnósticos para todos os encoders
  report      tabelas, figuras e summary.md
  all         todas as etapas acima (exceto `train`), em sequência

Cada etapa grava ``results/eurosat/logs/<etapa>.log``; cada treino também ``logs/runs/<execução>.log``.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict

from .config import Config, describe_hardware, extra_log_file, run_name, setup_logging

PIPELINE = ["prepare", "teacher", "benchmark", "tune", "train-all", "evaluate", "report"]


def run_stage(stage: str, cfg: Config, args: argparse.Namespace) -> None:
    if stage == "prepare":
        from .data import generate_views, make_label_subsets, prepare_arrays
        prepare_arrays(cfg)
        make_label_subsets(cfg)
        generate_views(cfg)
    elif stage == "teacher":
        from .teacher import run_teacher
        run_teacher(cfg)
    elif stage == "benchmark":
        from .benchmark import run_benchmark
        run_benchmark(cfg)
    elif stage == "tune":
        from .train import run_tuning
        run_tuning(cfg)
    elif stage == "train":
        from .train import train_student
        with extra_log_file(cfg.log_dir / "runs" / f"{run_name(args.row, args.size, args.seed, args.data_frac)}.log"):
            train_student(cfg, args.row, args.size, args.seed, data_frac=args.data_frac)
    elif stage == "train-all":
        from .train import train_all
        train_all(cfg)
    elif stage == "evaluate":
        from .evaluate import run_evaluation
        run_evaluation(cfg, n_jobs=args.n_jobs)
    elif stage == "report":
        from .report import run_report
        run_report(cfg)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="mo433_projeto_final", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=PIPELINE + ["train", "all"])
    p.add_argument("--epochs", type=int, help="sobrescreve a config (fica salvo em config.json)")
    p.add_argument("--num-workers", type=int, help="sobrescreve a config (fica salvo em config.json)")
    p.add_argument("--row", choices=["B", "C"])
    p.add_argument("--size", choices=["small", "large"])
    p.add_argument("--seed", type=int)
    p.add_argument("--data-frac", type=float, default=1.0)
    p.add_argument("--n-jobs", type=int, default=4, help="processos paralelos nas sondas")
    args = p.parse_args(argv)
    if args.stage == "train" and None in (args.row, args.size, args.seed):
        p.error("train exige --row, --size e --seed")

    # A config fica salva em config.json e é relida por todas as etapas; só é regravada
    # na primeira execução ou quando há sobrescritas na linha de comando.
    cfg = Config.load()
    overrides = {k: getattr(args, k) for k in ("epochs", "num_workers") if getattr(args, k) is not None}
    for key, value in overrides.items():
        setattr(cfg, key, value)
    if overrides or not (cfg.results_dir / "config.json").exists():
        cfg.save()

    for stage in PIPELINE if args.stage == "all" else [args.stage]:
        setup_logging(cfg.log_dir, "train_single" if stage == "train" else stage)
        log = logging.getLogger("mo433.cli")
        log.info("Hardware: %s", describe_hardware())
        log.info("Configuração: %s", asdict(cfg))
        run_stage(stage, cfg, args)
        if stage == "tune":
            cfg = Config.load()  # relê os lr escolhidos
        log.info("Etapa '%s' concluída.", stage)


if __name__ == "__main__":
    main()
