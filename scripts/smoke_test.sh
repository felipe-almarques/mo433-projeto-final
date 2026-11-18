#!/usr/bin/env bash
# Teste de fumaça: pipeline inteiro em dados sintéticos (~5 min). Use antes de gastar GPU.
set -euo pipefail
rm -rf results/smoke
"${PYTHON:-python}" -m mo433_projeto_final all --dataset smoke --n-jobs 2
echo "OK: veja results/smoke/summary.md"
