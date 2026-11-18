#!/usr/bin/env bash
# Reproduz o estudo completo de um dataset do zero (etapas retomáveis).
# Uso: bash scripts/run_all.sh [eurosat|stl10]   (padrão: eurosat)
# Variáveis opcionais: EPOCHS, N_JOBS
set -euo pipefail
DS="${1:-eurosat}"
PY="${PYTHON:-python}"
EXTRA=()
[[ -n "${EPOCHS:-}" ]] && EXTRA+=(--epochs "$EPOCHS")
for stage in prepare teacher benchmark tune train-all evaluate report; do
  echo ">>> etapa: $stage ($DS)"
  "$PY" -m mo433_projeto_final "$stage" --dataset "$DS" --n-jobs "${N_JOBS:-4}" "${EXTRA[@]}"
  EXTRA=()   # sobrescritas de config só na primeira etapa (ficam salvas em config.json)
done
echo "Resultados em results/$DS/ (summary.md, tabelas CSV e figuras PNG)."
