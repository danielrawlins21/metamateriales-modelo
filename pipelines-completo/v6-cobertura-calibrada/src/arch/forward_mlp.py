"""Cabeza forward: vector latente geométrico -> vector objetivo espectral."""

from __future__ import annotations

import torch
import torch.nn as nn


class ForwardMLP(nn.Module):
    """MLP deliberadamente simple; devuelve logits de ``n_bins`` componentes."""

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dims: tuple[int, ...] = (256, 512),
        n_bins: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        dims = (latent_dim, *hidden_dims)
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(dims[-1], n_bins))
        self.network = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


class ForwardMLPDividido(nn.Module):
    """Forward con cabeza dividida — la «opción B» que v0 dejó pendiente.

    v0 usaba una salida plana de 45 valores con MSE, tratando por igual tres
    magnitudes de naturaleza distinta:

        slot i = [presencia (0 o 1), frecuencia (continua), anchura (continua)]

    Eso tenía dos problemas medidos: `presencia` se entrenaba como si fuera
    continua, y en los slots vacíos la frecuencia y la anchura valen 0 **por
    relleno**, no porque se haya medido nada — con una media de 6 gaps sobre 15
    slots, más de la mitad del target era relleno que el modelo gastaba
    capacidad en reproducir.

    Aquí el tronco es compartido y se bifurca:

        z ──▶ tronco ──┬──▶ K logits          presencia  (BCE, sobre las K)
                       └──▶ K×2 valores       frecuencia y log(anchura)
                                              (MSE enmascarado por presencia real)

    La máscara es lo esencial: en los slots sin gap no hay nada que acertar, y
    promediarlos daría un error artificialmente bajo.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dims: tuple[int, ...] = (128, 128),
        k_max: int = 15,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.k_max = k_max

        dims = (latent_dim, *hidden_dims)
        capas: list[nn.Module] = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            capas.extend([
                nn.Linear(d_in, d_out),
                nn.LayerNorm(d_out),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        self.tronco = nn.Sequential(*capas)

        self.cabeza_presencia = nn.Linear(dims[-1], k_max)
        self.cabeza_regresion = nn.Linear(dims[-1], k_max * 2)

    def forward(self, z: torch.Tensor):
        """z (B, latent_dim) -> (logits_presencia (B,K), regresion (B,K,2))

        La regresión devuelve [frecuencia, log(anchura)]; el sigmoide de la
        presencia se aplica fuera, en la pérdida o al evaluar.
        """
        h = self.tronco(z)
        logits = self.cabeza_presencia(h)
        reg = self.cabeza_regresion(h).view(-1, self.k_max, 2)
        return logits, reg


def perdida_dividida(logits, reg, target, pesos=(1.0, 1.0, 1.0)):
    """Pérdida de la cabeza dividida.

    Args:
        logits: (B, K)      logits de presencia
        reg:    (B, K, 2)   [frecuencia, log(anchura)] predichos
        target: (B, K, 3)   [presencia, frecuencia, log(anchura)] reales
        pesos:  (bce, mse_frecuencia, mse_anchura)

    La BCE se calcula sobre las K ranuras completas — importa acertar tanto
    cuándo hay gap como cuándo no. La regresión solo sobre las ranuras con
    `presencia_real = 1`.
    """
    w_bce, w_f, w_a = pesos
    pres_real = target[:, :, 0]

    bce = nn.functional.binary_cross_entropy_with_logits(logits, pres_real)

    n = pres_real.sum().clamp(min=1.0)
    err_f = ((reg[:, :, 0] - target[:, :, 1]) ** 2 * pres_real).sum() / n
    err_a = ((reg[:, :, 1] - target[:, :, 2]) ** 2 * pres_real).sum() / n

    total = w_bce * bce + w_f * err_f + w_a * err_a
    return total, {'bce': bce.item(), 'mse_frecuencia': err_f.item(),
                   'mse_anchura': err_a.item()}
