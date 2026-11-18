# Resultados — eurosat

Gerado automaticamente por `python -m mo433_projeto_final report`. Valores em % de acurácia no teste, média ± desvio sobre todas as execuções (semente x sorteio do subconjunto rotulado).

## Configuração

- Dataset `eurosat`, entrada do aluno 64px, professor `vit_small_patch16_224.dino` @ 224px; K=8 views/imagem; 100 épocas, batch 256, AdamW (lr B=0.002, lr C=0.002, wd=0.05), warm-up 3 épocas + cosseno, bf16.
- Sementes: {'small': [0, 1, 2], 'large': [0, 1]}; sorteios de subconjunto rotulado por fração: 3.
- Hardware: {'python': '3.13.7', 'torch': '2.11.0+cu128', 'cuda_available': True, 'cpu_count': 12, 'platform': 'win32', 'gpu': 'NVIDIA GeForce RTX 4060', 'gpu_memory_gb': 8.0, 'cuda': '12.8', 'cudnn': 91900}

## Checagem de sanidade do professor

k-NN = 95.5% (acaso 10%), std dos embeddings = 0.0382 (ref. 1/sqrt(d) = 0.0510), NMI = 0.760, ARI = 0.712 -> **aprovado**.

## Encoders: parâmetros, FLOPs e latência

| encoder | params (M) | dim | GFLOPs/img | entrada | GPU ms/img (b256 fp16) | CPU ms (b1 fp32) |
|---|---|---|---|---|---|---|
| student_small | 0.70 | 128 | 0.070 | 64px | 0.0139 | 2.45 |
| student_large | 6.28 | 384 | 0.626 | 64px | 0.0565 | 7.49 |
| teacher | 21.67 | 384 | 9.197 | 224px | 0.5801 | 47.29 |

## Tabela de resultados — SVM linear

| encoder | 1% rótulos | 10% rótulos | 50% rótulos | 100% rótulos |
|---|---|---|---|---|
| A: aleatório (large) | 31.0 ± 1.7 | 45.8 ± 0.7 | 59.1 ± 0.7 | 61.3 ± 0.8 |
| A: aleatório (small) | 29.5 ± 1.5 | 44.6 ± 1.8 | 49.3 ± 2.0 | 50.0 ± 2.2 |
| B: VICReg direto (large) | 87.1 ± 0.6 | 88.8 ± 0.5 | 90.1 ± 0.4 | 92.2 ± 0.1 |
| B: VICReg direto (small) | 78.7 ± 1.1 | 84.3 ± 0.3 | 88.8 ± 0.2 | 89.2 ± 0.1 |
| C: destilado (large) | 89.3 ± 0.8 | 93.1 ± 0.3 | 94.1 ± 0.3 | 95.8 ± 0.1 |
| C: destilado (small) | 86.2 ± 1.3 | 90.9 ± 0.8 | 94.2 ± 0.3 | 94.7 ± 0.2 |
| D: professor ViT-S/16 | 88.8 ± 0.7 | 94.0 ± 0.1 | 95.3 ± 0.2 | 96.7 ± 0.0 |

## Tabela de resultados — k-NN

| encoder | 1% rótulos | 10% rótulos | 50% rótulos | 100% rótulos |
|---|---|---|---|---|
| A: aleatório (large) | 33.8 ± 1.7 | 43.7 ± 1.3 | 46.2 ± 0.7 | 47.2 ± 0.7 |
| A: aleatório (small) | 30.8 ± 1.2 | 38.7 ± 0.7 | 41.6 ± 0.3 | 43.0 ± 0.4 |
| B: VICReg direto (large) | 86.2 ± 0.3 | 90.2 ± 0.4 | 91.0 ± 0.4 | 91.4 ± 0.4 |
| B: VICReg direto (small) | 80.5 ± 0.6 | 85.6 ± 0.4 | 87.5 ± 0.2 | 88.1 ± 0.1 |
| C: destilado (large) | 86.6 ± 1.1 | 92.8 ± 0.5 | 94.7 ± 0.3 | 95.2 ± 0.0 |
| C: destilado (small) | 84.9 ± 0.6 | 91.2 ± 0.4 | 93.0 ± 0.1 | 93.7 ± 0.2 |
| D: professor ViT-S/16 | 85.0 ± 1.3 | 92.5 ± 0.3 | 95.0 ± 0.1 | 95.5 ± 0.0 |

## Tabela de resultados — SVM RBF

| encoder | 1% rótulos | 10% rótulos | 50% rótulos | 100% rótulos |
|---|---|---|---|---|
| A: aleatório (large) | 41.8 ± 1.2 | 58.5 ± 0.5 | 64.1 ± 0.3 | 64.0 ± 0.4 |
| A: aleatório (small) | 36.4 ± 2.0 | 49.8 ± 2.1 | 54.8 ± 1.7 | 54.7 ± 1.6 |
| B: VICReg direto (large) | 87.9 ± 0.6 | 91.4 ± 0.3 | 92.5 ± 0.5 | 92.5 ± 0.4 |
| B: VICReg direto (small) | 83.1 ± 0.9 | 87.7 ± 0.4 | 89.6 ± 0.2 | 89.6 ± 0.2 |
| C: destilado (large) | 91.4 ± 0.2 | 95.5 ± 0.2 | 96.7 ± 0.1 | 96.6 ± 0.2 |
| C: destilado (small) | 89.3 ± 0.2 | 93.6 ± 0.3 | 94.9 ± 0.2 | 95.0 ± 0.2 |
| D: professor ViT-S/16 | 91.4 ± 0.3 | 96.3 ± 0.3 | 97.7 ± 0.1 | 97.7 ± 0.1 |

## Lacuna C − B (estudo principal)

`spread` = sqrt(std_B² + std_C²) sobre semente x sorteio; *conclusivo* só se |gap| > spread.

| size | fraction | probe | gap | spread | verdict |
|---|---|---|---|---|---|
| large | 0.01 | knn | 0.4 | 1.1 | inconclusivo |
| large | 0.01 | linear_svm | 2.14 | 1.03 | C > B |
| large | 0.01 | rbf_svm | 3.55 | 0.63 | C > B |
| large | 0.1 | knn | 2.6 | 0.63 | C > B |
| large | 0.1 | linear_svm | 4.37 | 0.61 | C > B |
| large | 0.1 | rbf_svm | 4.17 | 0.33 | C > B |
| large | 0.5 | knn | 3.69 | 0.51 | C > B |
| large | 0.5 | linear_svm | 3.96 | 0.5 | C > B |
| large | 0.5 | rbf_svm | 4.2 | 0.48 | C > B |
| large | 1.0 | knn | 3.79 | 0.44 | C > B |
| large | 1.0 | linear_svm | 3.58 | 0.19 | C > B |
| large | 1.0 | rbf_svm | 4.1 | 0.45 | C > B |
| small | 0.01 | knn | 4.4 | 0.79 | C > B |
| small | 0.01 | linear_svm | 7.43 | 1.68 | C > B |
| small | 0.01 | rbf_svm | 6.22 | 0.9 | C > B |
| small | 0.1 | knn | 5.65 | 0.56 | C > B |
| small | 0.1 | linear_svm | 6.56 | 0.83 | C > B |
| small | 0.1 | rbf_svm | 5.98 | 0.45 | C > B |
| small | 0.5 | knn | 5.56 | 0.27 | C > B |
| small | 0.5 | linear_svm | 5.39 | 0.33 | C > B |
| small | 0.5 | rbf_svm | 5.33 | 0.27 | C > B |
| small | 1.0 | knn | 5.63 | 0.2 | C > B |
| small | 1.0 | linear_svm | 5.49 | 0.19 | C > B |
| small | 1.0 | rbf_svm | 5.4 | 0.26 | C > B |

## Execuções individuais — aluno large (SVM linear, média dos sorteios)

Com só 2 sementes o desvio-padrão é uma estimativa pobre (Seção 2); por isso as execuções aparecem individualmente.

| row | seed | 0.01 | 0.1 | 0.5 | 1.0 |
|---|---|---|---|---|---|
| A | 0 | 32.45 | 46.14 | 59.67 | 62.04 |
| A | 1 | 29.59 | 45.52 | 58.45 | 60.65 |
| B | 0 | 87.29 | 88.9 | 90.39 | 92.33 |
| B | 1 | 86.95 | 88.62 | 89.81 | 92.07 |
| C | 0 | 89.3 | 93.02 | 94.22 | 95.91 |
| C | 1 | 89.21 | 93.23 | 93.9 | 95.67 |

## Ablação — lacuna C − B (SVM linear) por quantidade de dados não rotulados x tamanho

| fraction | size | pool 50% | pool 100% |
|---|---|---|---|
| 0.01 | large | +6.74 ± 1.27 | +2.14 ± 1.03 |
| 0.01 | small | +13.58 ± 1.73 | +7.43 ± 1.68 |
| 0.1 | large | +8.88 ± 0.56 | +4.37 ± 0.61 |
| 0.1 | small | +10.85 ± 0.65 | +6.56 ± 0.83 |
| 0.5 | large | +6.05 ± 0.28 | +3.96 ± 0.50 |
| 0.5 | small | +9.38 ± 0.33 | +5.39 ± 0.33 |
| 1.0 | large | +5.83 ± 0.10 | +3.58 ± 0.19 |
| 1.0 | small | +9.44 ± 0.25 | +5.49 ± 0.19 |

## Diagnósticos não supervisionados (k-means no teste, k = nº de classes)

| row | size | data_frac | NMI | ARI | std/ref | n |
|---|---|---|---|---|---|---|
| A | large | 1.0 | 0.248 ± 0.009 | 0.155 ± 0.010 | 0.09 | 2 |
| A | small | 1.0 | 0.227 ± 0.023 | 0.144 ± 0.008 | 0.11 | 3 |
| B | large | 0.5 | 0.652 ± 0.023 | 0.557 ± 0.043 | 0.7 | 2 |
| B | large | 1.0 | 0.704 ± 0.034 | 0.636 ± 0.071 | 0.72 | 2 |
| B | small | 0.5 | 0.552 ± 0.008 | 0.432 ± 0.018 | 0.61 | 3 |
| B | small | 1.0 | 0.638 ± 0.006 | 0.548 ± 0.015 | 0.64 | 3 |
| C | large | 0.5 | 0.764 ± 0.002 | 0.695 ± 0.001 | 0.57 | 2 |
| C | large | 1.0 | 0.782 ± 0.001 | 0.703 ± 0.002 | 0.59 | 2 |
| C | small | 0.5 | 0.709 ± 0.005 | 0.635 ± 0.006 | 0.57 | 3 |
| C | small | 1.0 | 0.744 ± 0.006 | 0.680 ± 0.008 | 0.54 | 3 |
| D | teacher | 1.0 | 0.760 ± 0.000 | 0.712 ± 0.000 | 0.75 | 1 |

## Custo em GPU-horas e ganho por GPU-hora

Cache do professor (one-off, compartilhado por todas as execuções de C): 0.0291 GPU-h nesta máquina. Pré-treino publicado do professor (estimativa, V100): 2688 GPU-h.

| size | label_fraction | gap_C_minus_B_pts | gpu_h_B | gpu_h_C_student | gpu_h_teacher_cache_one_off | gpu_h_teacher_pretrain_published | extra_gpu_h_C_local | extra_gpu_h_C_with_pretrain | pts_per_extra_gpu_h_with_pretrain |
|---|---|---|---|---|---|---|---|---|---|
| small | 0.01 | 7.43 | 0.152 | 0.051 | 0.0291 | 2688.0 | -0.0719 | 2687.9281 | 0.0028 |
| small | 0.1 | 6.5597 | 0.152 | 0.051 | 0.0291 | 2688.0 | -0.0719 | 2687.9281 | 0.0024 |
| small | 0.5 | 5.3889 | 0.152 | 0.051 | 0.0291 | 2688.0 | -0.0719 | 2687.9281 | 0.002 |
| small | 1.0 | 5.4877 | 0.152 | 0.051 | 0.0291 | 2688.0 | -0.0719 | 2687.9281 | 0.002 |
| large | 0.01 | 2.1358 | 0.2711 | 0.1014 | 0.0291 | 2688.0 | -0.1406 | 2687.8594 | 0.0008 |
| large | 0.1 | 4.3673 | 0.2711 | 0.1014 | 0.0291 | 2688.0 | -0.1406 | 2687.8594 | 0.0016 |
| large | 0.5 | 3.9568 | 0.2711 | 0.1014 | 0.0291 | 2688.0 | -0.1406 | 2687.8594 | 0.0015 |
| large | 1.0 | 3.5833 | 0.2711 | 0.1014 | 0.0291 | 2688.0 | -0.1406 | 2687.8594 | 0.0013 |

GPU-horas totais de treino deste estudo (todas as execuções + cache do professor): 2.06 h.

Checagem de colapso ao fim de cada treino: 20/20 aprovados.

## Figuras

- `fig_label_efficiency.png` — curva de eficiência de rótulos (figura principal)
- `fig_label_efficiency_all_probes.png` — as três sondas
- `fig_gap_vs_params.png` — C − B vs parâmetros
- `fig_ablation.png` — grade da ablação
- `fig_training_curves.png` — perdas e std por época
- `fig_tsne_small.png` — painel t-SNE
- `fig_tsne_large.png` — painel t-SNE
