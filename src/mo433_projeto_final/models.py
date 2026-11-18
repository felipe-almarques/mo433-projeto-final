"""Arquiteturas: família de alunos (projeto próprio), professor congelado e projetores.

Aluno (Seção 2.1) — ``StudentNet``
---------------------------------
Rede convolucional residual pequena, sem cabeça de classificação: o ``forward``
devolve o vetor de features após *global average pooling*.

Decisões de projeto (justificativas detalhadas no README):

* **Blocos**: residuais "básicos" (conv3x3-BN-ReLU-conv3x3-BN + atalho). Os
  atalhos residuais tornam o treino estável a partir de inicialização aleatória
  sob dois objetivos muito diferentes (VICReg e destilação por cosseno). Convs
  densas 3x3 foram preferidas a depthwise-separáveis porque, em GPU, estas são
  limitadas por memória e não seriam mais rápidas no relógio para larguras tão
  pequenas.
* **Profundidade**: 4 estágios x 2 blocos (8 blocos, 17 camadas conv + atalhos),
  o mesmo "esqueleto" de uma ResNet-18, para que a profundidade não seja o fator
  limitante e só a largura varie entre os tamanhos.
* **Stem**: uma única conv 3x3 com stride 2, **sem max-pool**. O stem do ImageNet
  (7x7/stride 2 + max-pool, redução 4x) destruiria detalhes em imagens de
  64–96 px (Seção 4.3); a conv 3x3/2 reduz só 2x.
* **Downsampling**: conv 3x3 com stride 2 no primeiro bloco dos estágios 2–4
  (atalho 1x1/2 + BN). Mapa final = entrada/16: 4x4 a 64 px (EuroSAT) e 6x6 a 96 px
  (STL-10) — ainda há resolução espacial antes do pooling.
* **Normalização / ativação**: BatchNorm + ReLU, a combinação mais simples e
  testada para convnets treinadas com VICReg e com destilação.
* **Regra de escala única**: multiplicador de largura ``m`` aplicado às larguras
  base (32, 64, 128, 256), arredondadas a múltiplos de 8. Profundidade, stem,
  normalização e ativação NÃO mudam entre tamanhos, logo a comparação entre
  tamanhos não é confundida por outras mudanças.
  - small (m=0,5): larguras 16-32-64-128, dim. de feature 128
  - large (m=1,5): larguras 48-96-192-384, dim. de feature 384
* **Inicialização**: Kaiming (fan_out, ReLU) nas convs, BN com peso 1 e viés 0,
  sem zero-init nos ramos residuais. A mesma função inicializa A, B e C (para
  uma mesma semente, B e C partem exatamente dos mesmos pesos que A).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .utils import count_parameters, get_logger

log = get_logger("models")

BASE_WIDTHS = (32, 64, 128, 256)
DEPTHS = (2, 2, 2, 2)


def _round8(x: float) -> int:
    """Arredonda uma largura para o múltiplo de 8 mais próximo (eficiência em GPU)."""
    return max(8, int(round(x / 8)) * 8)


class BasicBlock(nn.Module):
    """Bloco residual: conv3x3(stride)-BN-ReLU-conv3x3-BN + atalho, seguido de ReLU."""

    def __init__(self, c_in: int, c_out: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)
        # Atalho de projeção apenas quando a forma muda (downsampling ou nova largura).
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.shortcut(x))


class StudentNet(nn.Module):
    """Família de alunos convolucionais parametrizada pelo multiplicador de largura."""

    def __init__(self, width_mult: float):
        super().__init__()
        widths = [_round8(w * width_mult) for w in BASE_WIDTHS]
        self.widths = widths
        self.width_mult = width_mult
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )
        stages, c_in = [], widths[0]
        for i, (w, d) in enumerate(zip(widths, DEPTHS)):
            blocks = []
            for j in range(d):
                stride = 2 if (i > 0 and j == 0) else 1
                blocks.append(BasicBlock(c_in, w, stride))
                c_in = w
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.num_features = widths[-1]
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stages(self.stem(x))
        return torch.flatten(self.pool(x), 1)  # (B, num_features) — sem cabeça


def build_student(size: str, sizes: dict[str, float], seed: int) -> StudentNet:
    """Constrói o aluno ``size`` com pesos iniciais determinados pela ``seed``.

    A inicialização depende só de (size, seed): B, C e A com a mesma semente
    partem dos mesmos pesos, como exige a Seção 2.1.
    """
    g_state = torch.random.get_rng_state()
    torch.manual_seed(10_000 + seed)
    model = StudentNet(sizes[size])
    torch.random.set_rng_state(g_state)
    log.debug("Aluno %s (m=%.2f) seed=%d: larguras=%s, params=%d",
              size, sizes[size], seed, model.widths, count_parameters(model))
    return model


# =============================================================================
# Professor (linha D) — ViT-S/16 DINO, congelado
# =============================================================================
def build_teacher(name: str, pretrained: bool = True) -> nn.Module:
    """Carrega o professor do ``timm``, congelado e em ``eval()``.

    ``vit_small_patch16_224.dino`` = checkpoint oficial ``dino_deitsmall16`` (Caron
    et al., 2021), pré-treinado sem rótulos no ImageNet-1k. Com ``num_classes=0`` o
    timm devolve a feature agrupada pelo **token CLS** (``global_pool="token"``),
    dimensão 384 — é a feature que o DINO usa nas suas avaliações k-NN/linear.
    """
    import timm

    teacher = timm.create_model(name, pretrained=pretrained, num_classes=0)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    log.info("Professor %s carregado (pretrained=%s): %d params, dim=%d, pool=%s",
             name, pretrained, count_parameters(teacher), teacher.num_features,
             getattr(teacher, "global_pool", "?"))
    return teacher


# =============================================================================
# Projetores (descartados após o treino; só o backbone é avaliado)
# =============================================================================
def vicreg_projector(in_dim: int, hidden: int, out: int) -> nn.Module:
    """Expansor do VICReg (3 camadas lineares, BN + ReLU), via ``lightly``."""
    from lightly.models.modules.heads import VICRegProjectionHead

    return VICRegProjectionHead(input_dim=in_dim, hidden_dim=hidden, output_dim=out, num_layers=3)


def distill_projector(in_dim: int, hidden: int, teacher_dim: int) -> nn.Module:
    """Projetor da destilação, exatamente o da Seção 3.5 (Linear-BN-GELU-Linear)."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Linear(hidden, teacher_dim)
    )
