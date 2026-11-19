# Resultados — stl10

Gerado automaticamente por `python -m mo433_projeto_final report`. Valores em % de acurácia no teste, média ± desvio sobre todas as execuções (semente x sorteio do subconjunto rotulado).

## Configuração

- Dataset `stl10`, entrada do aluno 96px, professor `vit_small_patch16_224.dino` @ 224px; K=6 views/imagem; 100 épocas, batch 256, AdamW (lr B=0.002, lr C=0.002, wd=0.05), warm-up 3 épocas + cosseno, bf16.
- Sementes: {'small': [0, 1, 2], 'large': [0, 1]}; sorteios de subconjunto rotulado por fração: 3.
- Hardware: {'python': '3.13.7', 'torch': '2.11.0+cu128', 'cuda_available': True, 'cpu_count': 12, 'platform': 'win32', 'gpu': 'NVIDIA GeForce RTX 4060', 'gpu_memory_gb': 8.0, 'cuda': '12.8', 'cudnn': 91900}

## Checagem de sanidade do professor

k-NN = 98.6% (acaso 10%), std dos embeddings = 0.0458 (ref. 1/sqrt(d) = 0.0510), NMI = 0.900, ARI = 0.877 -> **aprovado**.

## Encoders: parâmetros, FLOPs e latência

| encoder | params (M) | dim | GFLOPs/img | entrada | GPU ms/img (b256 fp16) | CPU ms (b1 fp32) |
|---|---|---|---|---|---|---|
| student_small | 0.70 | 128 | 0.158 | 96px | 0.0297 | 3.43 |
| student_large | 6.28 | 384 | 1.407 | 96px | 0.1596 | 11.00 |
| teacher | 21.67 | 384 | 9.197 | 224px | 0.5933 | 44.93 |

## Tabela de resultados — SVM linear

| encoder | 1% rótulos | 10% rótulos | 50% rótulos | 100% rótulos |
|---|---|---|---|---|
| A: aleatório (large) | 21.0 ± 1.0 | 27.3 ± 0.9 | 34.2 ± 0.6 | 37.8 ± 0.4 |
| A: aleatório (small) | 18.6 ± 1.3 | 26.0 ± 1.3 | 31.2 ± 0.8 | 33.1 ± 0.8 |
| B: VICReg direto (large) | 52.8 ± 0.9 | 62.8 ± 0.6 | 69.1 ± 0.2 | 71.1 ± 0.2 |
| B: VICReg direto (small) | 45.5 ± 1.4 | 58.6 ± 0.8 | 65.1 ± 0.2 | 66.3 ± 0.3 |
| C: destilado (large) | 75.9 ± 1.6 | 84.2 ± 0.3 | 87.6 ± 0.3 | 88.8 ± 0.1 |
| C: destilado (small) | 64.8 ± 0.8 | 76.3 ± 0.4 | 80.7 ± 0.5 | 81.8 ± 0.4 |
| D: professor ViT-S/16 | 95.7 ± 0.4 | 97.9 ± 0.2 | 98.4 ± 0.1 | 98.8 ± 0.0 |

## Tabela de resultados — k-NN

| encoder | 1% rótulos | 10% rótulos | 50% rótulos | 100% rótulos |
|---|---|---|---|---|
| A: aleatório (large) | 19.9 ± 1.5 | 26.4 ± 0.5 | 31.4 ± 0.5 | 32.8 ± 0.3 |
| A: aleatório (small) | 17.6 ± 1.4 | 23.2 ± 0.9 | 27.9 ± 0.5 | 29.3 ± 0.3 |
| B: VICReg direto (large) | 48.6 ± 1.2 | 62.4 ± 0.6 | 67.5 ± 0.3 | 69.1 ± 0.3 |
| B: VICReg direto (small) | 42.7 ± 1.1 | 57.2 ± 0.7 | 62.9 ± 0.2 | 64.7 ± 0.3 |
| C: destilado (large) | 69.8 ± 1.4 | 83.4 ± 0.5 | 87.0 ± 0.4 | 88.3 ± 0.1 |
| C: destilado (small) | 59.0 ± 1.0 | 72.8 ± 0.4 | 77.7 ± 0.4 | 79.2 ± 0.6 |
| D: professor ViT-S/16 | 91.7 ± 1.5 | 97.1 ± 0.3 | 98.3 ± 0.0 | 98.6 ± 0.0 |

## Tabela de resultados — SVM RBF

| encoder | 1% rótulos | 10% rótulos | 50% rótulos | 100% rótulos |
|---|---|---|---|---|
| A: aleatório (large) | 21.5 ± 0.6 | 32.2 ± 0.2 | 39.0 ± 0.2 | 42.0 ± 0.0 |
| A: aleatório (small) | 19.0 ± 1.7 | 27.7 ± 0.6 | 33.9 ± 0.6 | 36.1 ± 0.2 |
| B: VICReg direto (large) | 52.5 ± 1.6 | 65.4 ± 0.8 | 71.3 ± 0.3 | 73.1 ± 0.1 |
| B: VICReg direto (small) | 46.3 ± 0.9 | 60.8 ± 0.7 | 67.5 ± 0.2 | 69.4 ± 0.1 |
| C: destilado (large) | 74.0 ± 1.2 | 85.7 ± 0.2 | 89.1 ± 0.1 | 90.3 ± 0.0 |
| C: destilado (small) | 64.2 ± 1.0 | 76.8 ± 0.4 | 81.6 ± 0.3 | 83.1 ± 0.3 |
| D: professor ViT-S/16 | 92.8 ± 1.4 | 97.4 ± 0.4 | 98.6 ± 0.1 | 98.8 ± 0.0 |

## Lacuna C − B (estudo principal)

`spread` = sqrt(std_B² + std_C²) sobre semente x sorteio; *conclusivo* só se |gap| > spread.

| size | fraction | probe | gap | spread | verdict |
|---|---|---|---|---|---|
| large | 0.01 | knn | 21.18 | 1.87 | C > B |
| large | 0.01 | linear_svm | 23.03 | 1.84 | C > B |
| large | 0.01 | rbf_svm | 21.48 | 2.04 | C > B |
| large | 0.1 | knn | 21.04 | 0.79 | C > B |
| large | 0.1 | linear_svm | 21.4 | 0.71 | C > B |
| large | 0.1 | rbf_svm | 20.33 | 0.82 | C > B |
| large | 0.5 | knn | 19.54 | 0.47 | C > B |
| large | 0.5 | linear_svm | 18.54 | 0.36 | C > B |
| large | 0.5 | rbf_svm | 17.86 | 0.36 | C > B |
| large | 1.0 | knn | 19.19 | 0.29 | C > B |
| large | 1.0 | linear_svm | 17.66 | 0.21 | C > B |
| large | 1.0 | rbf_svm | 17.14 | 0.11 | C > B |
| small | 0.01 | knn | 16.34 | 1.51 | C > B |
| small | 0.01 | linear_svm | 19.32 | 1.58 | C > B |
| small | 0.01 | rbf_svm | 17.91 | 1.32 | C > B |
| small | 0.1 | knn | 15.59 | 0.82 | C > B |
| small | 0.1 | linear_svm | 17.66 | 0.89 | C > B |
| small | 0.1 | rbf_svm | 15.98 | 0.78 | C > B |
| small | 0.5 | knn | 14.79 | 0.48 | C > B |
| small | 0.5 | linear_svm | 15.57 | 0.51 | C > B |
| small | 0.5 | rbf_svm | 14.19 | 0.41 | C > B |
| small | 1.0 | knn | 14.47 | 0.66 | C > B |
| small | 1.0 | linear_svm | 15.48 | 0.52 | C > B |
| small | 1.0 | rbf_svm | 13.68 | 0.28 | C > B |

## Execuções individuais — aluno large (SVM linear, média dos sorteios)

Com só 2 sementes o desvio-padrão é uma estimativa pobre (Seção 2); por isso as execuções aparecem individualmente.

| row | seed | 0.01 | 0.1 | 0.5 | 1.0 |
|---|---|---|---|---|---|
| A | 0 | 21.37 | 28.14 | 34.63 | 38.14 |
| A | 1 | 20.72 | 26.52 | 33.85 | 37.41 |
| B | 0 | 52.6 | 62.48 | 69.1 | 70.94 |
| B | 1 | 53.08 | 63.2 | 69.01 | 71.29 |
| C | 0 | 76.35 | 84.08 | 87.71 | 88.7 |
| C | 1 | 75.39 | 84.4 | 87.48 | 88.84 |

## Ablação — lacuna C − B (SVM linear) por quantidade de dados não rotulados x tamanho

| fraction | size | pool 50% | pool 100% |
|---|---|---|---|
| 0.01 | large | +21.76 ± 1.73 | +23.03 ± 1.84 |
| 0.01 | small | +19.41 ± 1.48 | +19.32 ± 1.58 |
| 0.1 | large | +19.29 ± 0.74 | +21.40 ± 0.71 |
| 0.1 | small | +17.18 ± 0.83 | +17.66 ± 0.89 |
| 0.5 | large | +16.82 ± 0.50 | +18.54 ± 0.36 |
| 0.5 | small | +15.69 ± 0.59 | +15.57 ± 0.51 |
| 1.0 | large | +16.31 ± 0.69 | +17.66 ± 0.21 |
| 1.0 | small | +15.07 ± 0.30 | +15.48 ± 0.52 |

## Diagnósticos não supervisionados (k-means no teste, k = nº de classes)

| row | size | data_frac | NMI | ARI | std/ref | n |
|---|---|---|---|---|---|---|
| A | large | 1.0 | 0.099 ± 0.007 | 0.045 ± 0.003 | 0.07 | 2 |
| A | small | 1.0 | 0.071 ± 0.013 | 0.035 ± 0.006 | 0.09 | 3 |
| B | large | 0.5 | 0.382 ± 0.002 | 0.238 ± 0.002 | 0.63 | 2 |
| B | large | 1.0 | 0.422 ± 0.003 | 0.270 ± 0.004 | 0.68 | 2 |
| B | small | 0.5 | 0.300 ± 0.005 | 0.178 ± 0.012 | 0.51 | 3 |
| B | small | 1.0 | 0.372 ± 0.010 | 0.242 ± 0.005 | 0.54 | 3 |
| C | large | 0.5 | 0.585 ± 0.000 | 0.455 ± 0.012 | 0.48 | 2 |
| C | large | 1.0 | 0.665 ± 0.000 | 0.570 ± 0.017 | 0.54 | 2 |
| C | small | 0.5 | 0.483 ± 0.008 | 0.352 ± 0.009 | 0.46 | 3 |
| C | small | 1.0 | 0.523 ± 0.006 | 0.392 ± 0.009 | 0.47 | 3 |
| D | teacher | 1.0 | 0.900 ± 0.000 | 0.877 ± 0.000 | 0.9 | 1 |

## Custo em GPU-horas e ganho por GPU-hora

Cache do professor (one-off, compartilhado por todas as execuções de C): 0.0207 GPU-h nesta máquina. Pré-treino publicado do professor (estimativa, V100): 2688 GPU-h.

| size | label_fraction | gap_C_minus_B_pts | gpu_h_B | gpu_h_C_student | gpu_h_teacher_cache_one_off | gpu_h_teacher_pretrain_published | extra_gpu_h_C_local | extra_gpu_h_C_with_pretrain | pts_per_extra_gpu_h_with_pretrain |
|---|---|---|---|---|---|---|---|---|---|
| small | 0.01 | 19.3222 | 0.2216 | 0.0757 | 0.0207 | 2688.0 | -0.1251 | 2687.8749 | 0.0072 |
| small | 0.1 | 17.6556 | 0.2216 | 0.0757 | 0.0207 | 2688.0 | -0.1251 | 2687.8749 | 0.0066 |
| small | 0.5 | 15.5681 | 0.2216 | 0.0757 | 0.0207 | 2688.0 | -0.1251 | 2687.8749 | 0.0058 |
| small | 1.0 | 15.4792 | 0.2216 | 0.0757 | 0.0207 | 2688.0 | -0.1251 | 2687.8749 | 0.0058 |
| large | 0.01 | 23.0333 | 0.5062 | 0.2201 | 0.0207 | 2688.0 | -0.2655 | 2687.7345 | 0.0086 |
| large | 0.1 | 21.4042 | 0.5062 | 0.2201 | 0.0207 | 2688.0 | -0.2655 | 2687.7345 | 0.008 |
| large | 0.5 | 18.5354 | 0.5062 | 0.2201 | 0.0207 | 2688.0 | -0.2655 | 2687.7345 | 0.0069 |
| large | 1.0 | 17.6562 | 0.5062 | 0.2201 | 0.0207 | 2688.0 | -0.2655 | 2687.7345 | 0.0066 |

GPU-horas totais de treino deste estudo (todas as execuções + cache do professor): 3.52 h.

Checagem de colapso ao fim de cada treino: 20/20 aprovados.

## Figuras

- `fig_label_efficiency.png` — curva de eficiência de rótulos (figura principal)
- `fig_label_efficiency_all_probes.png` — as três sondas
- `fig_gap_vs_params.png` — C − B vs parâmetros
- `fig_ablation.png` — grade da ablação
- `fig_training_curves.png` — perdas e std por época
- `fig_tsne_small.png` — painel t-SNE
- `fig_tsne_large.png` — painel t-SNE
