"""Forward de conjuntos con incertidumbre de anchura.

El checkpoint entregado contiene dos ``state_dict``: el autoencoder usado como
extractor de características y una cabeza que devuelve 15 candidatos sin orden.
Cada candidato tiene este formato fijo::

    [presencia_logit, frecuencia, q10_log, q50_log, q90_log]

``q50_log`` es la estimación central de la anchura. ``q10_log`` y ``q90_log``
describen incertidumbre sobre esa anchura; no son los bordes físicos del gap.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import torch
import torch.nn as nn

from arch.cnn_autoencoder import Autoencoder


def sha256_archivo(path: str | Path) -> str:
    """SHA-256 hexadecimal de un fichero, leído por bloques."""
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(bloque)
    return digest.hexdigest()


class CabezaConjuntoCuantiles(nn.Module):
    """Latente ``(B,128)`` -> conjunto ``(B,15,5)``."""

    def __init__(self, d_latente: int = 128, k_max: int = 15,
                 d_oculta: int = 512):
        super().__init__()
        self.k_max = int(k_max)
        self.n_salidas = 5
        self.red = nn.Sequential(
            nn.Linear(d_latente, d_oculta),
            nn.SiLU(),
            nn.Linear(d_oculta, d_oculta),
            nn.SiLU(),
            nn.Linear(d_oculta, self.k_max * self.n_salidas),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.red(z).view(-1, self.k_max, self.n_salidas)


@dataclass(frozen=True)
class SalidaCuantiles:
    """Vista con nombres de la salida cruda del modelo."""

    presencia_logit: torch.Tensor
    frecuencia: torch.Tensor
    q10_log: torch.Tensor
    q50_log: torch.Tensor
    q90_log: torch.Tensor

    @property
    def regresion_compatible(self) -> torch.Tensor:
        """Formato histórico ``[frecuencia, anchura_log]`` de v4."""
        return torch.stack((self.frecuencia, self.q50_log), dim=-1)

    def anchuras(self, minimo: float, maximo: float) -> torch.Tensor:
        """Devuelve q10, q50 y q90 en unidades físicas de anchura."""
        limites = (math.log(minimo), math.log(maximo))
        q_log = torch.stack((self.q10_log, self.q50_log, self.q90_log), dim=-1)
        return torch.exp(q_log.clamp(min=limites[0], max=limites[1]))


def separar_salida(pred: torch.Tensor) -> SalidaCuantiles:
    """Valida y separa un tensor ``(B,K,5)``."""
    if pred.ndim != 3 or pred.shape[-1] != 5:
        raise ValueError(f'Salida de cuantiles inválida: se esperaba (B,K,5), '
                         f'se recibió {tuple(pred.shape)}')
    return SalidaCuantiles(
        presencia_logit=pred[:, :, 0],
        frecuencia=pred[:, :, 1],
        q10_log=pred[:, :, 2],
        q50_log=pred[:, :, 3],
        q90_log=pred[:, :, 4],
    )


class ForwardCuantiles(nn.Module):
    """Autoencoder de JC como encoder y cabeza de conjuntos con cuantiles."""

    def __init__(self, latent_dim: int = 128, channels=(32, 64, 128, 256),
                 dropout_ae: float = 0.2, input_size: int = 128,
                 padding_mode: str = 'zeros', k_max: int = 15,
                 d_oculta: int = 512):
        super().__init__()
        self.input_size = int(input_size)
        self.k_max = int(k_max)
        self.ae = Autoencoder(
            latent_dim=latent_dim,
            channels=list(channels),
            dropout=dropout_ae,
            input_size=input_size,
            padding_mode=padding_mode,
        )
        self.cabeza = CabezaConjuntoCuantiles(latent_dim, k_max, d_oculta)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        if imgs.ndim != 4 or imgs.shape[1:] != (1, self.input_size, self.input_size):
            raise ValueError(
                f'Entrada inválida: se esperaba (B,1,{self.input_size},'
                f'{self.input_size}), se recibió {tuple(imgs.shape)}')
        return self.cabeza(self.ae.encode(imgs))

    def salida(self, imgs: torch.Tensor) -> SalidaCuantiles:
        return separar_salida(self(imgs))

    def compatible_v4(self, imgs: torch.Tensor):
        """Interfaz histórica: ``logits, [frecuencia, q50_log]``."""
        salida = self.salida(imgs)
        return salida.presencia_logit, salida.regresion_compatible


def cargar_forward_cuantiles(path: str | Path, cfg: dict,
                             device: torch.device) -> ForwardCuantiles:
    """Reconstruye el modelo, verifica el fichero y carga los pesos."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'No existe el checkpoint del Forward: {path}')

    esperado = str(cfg.get('sha256', '')).strip().lower()
    obtenido = sha256_archivo(path)
    if esperado and obtenido != esperado:
        raise ValueError(f'SHA-256 incorrecto para {path}: esperado {esperado}, '
                         f'obtenido {obtenido}')

    if int(cfg['k_max']) != 15:
        raise ValueError('El checkpoint entregado está fijado a 15 candidatos')
    if list(cfg['cuantiles']) != [0.1, 0.5, 0.9]:
        raise ValueError('El orden del checkpoint debe ser q10, q50, q90')

    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if set(checkpoint) != {'ae', 'cabeza'}:
        raise ValueError('Checkpoint incompatible: se esperaban las claves '
                         "'ae' y 'cabeza'")

    modelo = ForwardCuantiles(
        latent_dim=int(cfg['latent_dim']),
        channels=tuple(cfg['channels']),
        dropout_ae=float(cfg['dropout_ae']),
        input_size=int(cfg['input_size']),
        padding_mode=str(cfg['padding_mode']),
        k_max=int(cfg['k_max']),
        d_oculta=int(cfg['d_oculta']),
    )
    modelo.ae.load_state_dict(checkpoint['ae'], strict=True)
    modelo.cabeza.load_state_dict(checkpoint['cabeza'], strict=True)
    return modelo.to(device)
