"""Arquiteturas: família de alunos (projeto próprio), professor congelado e projetores.

``StudentNet``: convnet residual sem cabeça (saída = *global average pooling*).

* blocos residuais básicos (conv3x3-BN-ReLU-conv3x3-BN + atalho), 4 estágios x 2 blocos
  (esqueleto da ResNet-18), de modo que só a largura varia entre tamanhos;
* stem = uma conv 3x3/2 **sem max-pool** (o stem do ImageNet destruiria detalhes a 64 px);
* downsampling por conv 3x3/2 no 1º bloco dos estágios 2-4 -> mapa final 4x4 a 64 px;
* regra de escala única: multiplicador de largura ``m`` sobre (32, 64, 128, 256),
  arredondado a múltiplos de 8 — small (m=0,5): 16-32-64-128; large (m=1,5): 48-96-192-384;
* init Kaiming (fan_out), BN peso 1 / viés 0: A, B e C com a mesma semente partem dos
  mesmos pesos.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from .config import count_params

log = logging.getLogger("mo433.models")

BASE_WIDTHS = (32, 64, 128, 256)
BLOCKS_PER_STAGE = 2


class BasicBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn2(self.conv2(self.act(self.bn1(self.conv1(x)))))
        return self.act(out + self.shortcut(x))


class StudentNet(nn.Module):
    def __init__(self, width_mult: float):
        super().__init__()
        self.widths = [max(8, int(round(w * width_mult / 8)) * 8) for w in BASE_WIDTHS]
        self.num_features = self.widths[-1]
        self.stem = nn.Sequential(nn.Conv2d(3, self.widths[0], 3, stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(self.widths[0]), nn.ReLU(inplace=True))
        stages, c_in = [], self.widths[0]
        for i, w in enumerate(self.widths):
            blocks = []
            for j in range(BLOCKS_PER_STAGE):
                blocks.append(BasicBlock(c_in, w, stride=2 if (i > 0 and j == 0) else 1))
                c_in = w
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.pool(self.stages(self.stem(x))), 1)


def build_student(size: str, sizes: dict[str, float], seed: int) -> StudentNet:
    """Aluno ``size`` com pesos iniciais que dependem só de (size, seed); não altera o RNG global."""
    state = torch.random.get_rng_state()
    torch.manual_seed(10_000 + seed)
    model = StudentNet(sizes[size])
    torch.random.set_rng_state(state)
    return model


def build_teacher(name: str) -> nn.Module:
    """Professor DINO ViT-S/16 do timm, congelado. ``num_classes=0`` -> feature do token CLS (dim 384)."""
    import timm

    teacher = timm.create_model(name, pretrained=True, num_classes=0).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    log.info("Professor %s: %d params, dim=%d", name, count_params(teacher), teacher.num_features)
    return teacher


# Projetores: usados só no treino e descartados (apenas o backbone é avaliado).
def vicreg_projector(in_dim: int, hidden: int, out: int) -> nn.Module:
    from lightly.models.modules.heads import VICRegProjectionHead

    return VICRegProjectionHead(input_dim=in_dim, hidden_dim=hidden, output_dim=out, num_layers=3)


def distill_projector(in_dim: int, hidden: int, teacher_dim: int) -> nn.Module:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Linear(hidden, teacher_dim))
