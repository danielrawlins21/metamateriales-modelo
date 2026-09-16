"""
cVAE condicional para diseño inverso de geometrías — TFM metamateriales.

CONVOLUCIONES CIRCULARES (v4).
    El decoder genera una CELDA UNIDAD, que es periódica: lo que hay a la
    derecha del borde derecho es el borde izquierdo. Con relleno de ceros el
    decoder aprende lo contrario, y una candidata no periódica no se puede
    mallar — el mallador no encuentra qué nodo de un borde es el mismo que otro
    del opuesto, y simularla daría contorno libre. Es exactamente el fallo que
    dejó el 23,5 % del dataset sin física (docs/periodicidad_bloch.md).
    Ver `ConditionalDecoder` para el detalle de por qué desaparece
    `ConvTranspose2d`.

    target c (vector objetivo paramétrico, R^45 = 15 gaps x 3)
              +
        geometría x  ->  encoder (pluggable, ver abajo)  ->  features
              v
      condition encoder + recognition head  ->  (mu, logvar)  ->  z (reparam.)
              v
        decoder condicional (z, c)
              v
        geometría candidata

Objetivo estándar de cVAE:
    L = reconstruction_loss(x, x_hat) + beta * KL(q(z|x,c) || p(z))

EL ENCODER GEOMÉTRICO NO ESTÁ EN ESTE ARCHIVO A PROPÓSITO.
`ConditionalVAE` recibe el encoder como argumento de su constructor, no lo
define él mismo. Cualquier módulo sirve mientras cumpla el contrato:

    encoder.latent_dim: int
    encoder(x) -> tensor (batch, latent_dim), donde x es (batch, 1, 128, 128)

Ese es justo el contrato que ya cumple el `Autoencoder.encode()` del CNN
autoencoder del proyecto; `FrozenCNNEncoder` (al final de este archivo) es el
adaptador que lo conecta.

CAMBIOS EN v0 respecto al paquete original de ENGANCHE:
  - `proxy_dim` (1024, perfil FFT sin significado físico) pasa a llamarse
    `target_dim` y vale 45: el vector objetivo paramétrico de JC,
    15 gaps x [presencia, frecuencia_media, anchura].
  - Eliminados `_InnerCNNEncoder` y `PretrainedCNNEncoderAdapter`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

TARGET_DIM = 45  # 15 gaps x [presencia, frecuencia_media, anchura]


class ConditionEncoder(nn.Module):
    """Proyecta el vector objetivo a un embedding mas grande antes de condicionar.

    EXISTIA "para que no quede eclipsado por un z de 128 dimensiones". En v2 el
    z del VAE bajo a 16, asi que **ese motivo ya no existe**: con la condicion
    en crudo (67) sigue siendo el 81 % de lo que recibe el decoder. Y no aporta
    expresividad, solo profundidad — el decoder ya empieza con
    `Linear(z + cond, ...)`.

    Con `cond_dim=None` (o 0, o 'crudo') se salta: la condicion pasa tal cual.
    Es P8 de `pendientes_tras_mallador.md`, a confirmar con barrido.
    """

    def __init__(self, target_dim: int = TARGET_DIM, cond_dim: int | None = 128):
        super().__init__()
        self.crudo = cond_dim in (None, 0, 'crudo')
        self.out_dim = target_dim if self.crudo else int(cond_dim)
        self.net = nn.Identity() if self.crudo else nn.Sequential(
            nn.Linear(target_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, int(cond_dim)),
            nn.ReLU(inplace=True),
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.net(c)


class RecognitionHead(nn.Module):
    """features_encoder (+ condición) -> (mu, logvar) para la reparametrización.

    Necesaria porque un encoder determinista (como un autoencoder normal) no
    tiene un q(z|x) incorporado; esta cabeza estocástica se lo añade encima.
    """

    def __init__(self, feat_dim: int, cond_dim: int, latent_dim: int):
        super().__init__()
        self.mu = nn.Linear(feat_dim + cond_dim, latent_dim)
        self.logvar = nn.Linear(feat_dim + cond_dim, latent_dim)

    def forward(self, feats: torch.Tensor, cond: torch.Tensor):
        h = torch.cat([feats, cond], dim=-1)
        return self.mu(h), self.logvar(h)


class ConditionalDecoder(nn.Module):
    """(z, cond_embed) -> logits de la geometría reconstruida (B, 1, img_size, img_size)."""

    def __init__(self, latent_dim: int, cond_dim: int, img_size: int = 128,
                 base_channels: int = 256, dropout: float = 0.1,
                 padding_mode: str = 'circular'):
        super().__init__()
        self.img_size = img_size
        self.start_res = img_size // 16  # 4 bloques de upsampling: /16 -> *16
        self.base_channels = base_channels
        self.padding_mode = padding_mode

        self.proj = nn.Linear(latent_dim + cond_dim, base_channels * self.start_res * self.start_res)

        # v4: Upsample + Conv2d en vez de ConvTranspose2d. NO es cosmético.
        # `ConvTranspose2d` solo admite padding_mode='zeros' —PyTorch lo prohíbe
        # explícitamente—, y con ceros el decoder aprende que fuera del borde no
        # hay material. En una celda unidad eso es falso: fuera del borde
        # derecho está el borde izquierdo. Y la candidata tiene que salir
        # periódica o el mallador no puede aparejar sus nodos.
        # Efecto secundario bienvenido: Upsample+Conv no produce el tablero de
        # ajedrez típico de la convolución transpuesta.
        channels = [base_channels, 128, 64, 32, 16]
        blocks = []
        for c_in, c_out in zip(channels[:-1], channels[1:]):
            blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(c_in, c_out, kernel_size=3, padding=1,
                              padding_mode=padding_mode),
                    nn.BatchNorm2d(c_out),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.to_logits = nn.Conv2d(channels[-1], 1, kernel_size=3, padding=1,
                                   padding_mode=padding_mode)

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = torch.cat([z, cond], dim=-1)
        h = self.proj(h)
        h = h.view(-1, self.base_channels, self.start_res, self.start_res)
        for block in self.blocks:
            h = block(h)
        return self.to_logits(h)


class ConditionalVAE(nn.Module):
    """El cVAE completo. `encoder` es pluggable: pasa aquí cualquier módulo
    con `.latent_dim` y `forward(x) -> (batch, latent_dim)`.

    DOS LATENTES DISTINTOS, NO CONFUNDIRLOS
    ---------------------------------------
    - `encoder.latent_dim` (128): las **features** que el CNN extrae de la
      imagen. Es una descripción determinista de la geometría, y es también lo
      que consume el ForwardMLP.
    - `latent_dim` (este argumento): la dimensión de la **variable estocástica
      del VAE**, de la que se muestrea al generar. Solo la usa el decoder.

    Hasta v1 estaban igualados (`self.latent_dim = encoder.latent_dim`), pero
    no tienen por qué: `RecognitionHead` ya recibe ambas dimensiones por
    separado. Desacoplarlos **no obliga a reentrenar el autoencoder**: el
    encoder sigue dando sus 128 features y la recognition head las comprime a
    la dimensión que se pida.

    Por qué importa: con `z` de 128 y `cond` de 128, la condición es la mitad
    de la entrada del decoder — y `z`, que viene de la imagen real, basta por
    sí solo para reconstruir, así que `c` sobra. Medido en v0 y v1: el decoder
    la ignoraba (+0,6 % y +1,6 % entre `c` real y `c` barajado). Bajando `z` a
    16, la condición pasa a ser el 89 % de lo que el decoder recibe.
    """

    def __init__(self, encoder: nn.Module, img_size: int = 128,
                 target_dim: int = TARGET_DIM, cond_dim: int = 128,
                 decoder_dropout: float = 0.1, latent_dim: int | None = None):
        super().__init__()
        self.encoder = encoder
        self.feat_dim = encoder.latent_dim
        # Por defecto se iguala al del encoder, para no cambiar el
        # comportamiento de versiones anteriores sin querer.
        self.latent_dim = latent_dim if latent_dim is not None else encoder.latent_dim
        self.condition_encoder = ConditionEncoder(target_dim, cond_dim)
        # La dimension efectiva la manda el ConditionEncoder, no el config: con
        # `cond_dim=crudo` es `target_dim`, y leerlo del config habria montado
        # un decoder con la entrada equivocada sin avisar.
        cond_dim = self.condition_encoder.out_dim
        # Normaliza las features del encoder antes de RecognitionHead.
        # Necesario si el encoder que conectes no tiene ninguna restricción de
        # escala en su salida (p.ej. entrenado como autoencoder simple, sin
        # KL): sin esto, un encoder con features de escala grande puede hacer
        # que logvar explote (KL desbordada en la primera época).
        self.feature_norm = nn.BatchNorm1d(self.feat_dim)
        self.recognition_head = RecognitionHead(self.feat_dim, cond_dim, self.latent_dim)
        self.decoder = ConditionalDecoder(self.latent_dim, cond_dim, img_size, dropout=decoder_dropout)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        cond = self.condition_encoder(c)
        feats = self.feature_norm(self.encoder(x))
        mu, logvar = self.recognition_head(feats, cond)
        z = self.reparameterize(mu, logvar)
        recon_logits = self.decoder(z, cond)
        return recon_logits, mu, logvar

    @torch.no_grad()
    def sample(self, c: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Genera `n_samples` geometrías candidatas para cada fila de `c`.

        c: (B, target_dim) -> devuelve (B, n_samples, 1, img_size, img_size), valores en {0, 1}.
        No usa self.encoder en absoluto: z se muestrea directamente de N(0,I),
        no se codifica ninguna geometría para generar.
        """
        device = c.device
        cond = self.condition_encoder(c)  # (B, cond_dim)
        b = c.shape[0]
        cond_rep = cond.unsqueeze(1).expand(b, n_samples, -1).reshape(b * n_samples, -1)
        z = torch.randn(b * n_samples, self.latent_dim, device=device)
        logits = self.decoder(z, cond_rep)
        masks = (torch.sigmoid(logits) > 0.5).float()
        _, ch, h, w = masks.shape
        return masks.view(b, n_samples, ch, h, w)


class FrozenCNNEncoder(nn.Module):
    """Adaptador que conecta el CNN Autoencoder del proyecto como encoder del cVAE.

    Cumple el contrato que pide ConditionalVAE (`.latent_dim` y
    `forward(x) -> (batch, latent_dim)`) delegando en `Autoencoder.encode()`.
    Los pesos van congelados y en eval() permanentemente: el cVAE no debe
    modificar el espacio latente que ya validamos con t-SNE/PCA.

    NOTA: el paquete original de ENGANCHE traía aquí otro encoder
    (`PretrainedCNNEncoderAdapter`) contra el que se entrenó `cvae_pesos.pt`.
    Se ha eliminado: en v0 entrenamos el cVAE desde cero contra este encoder,
    y esos pesos no son cargables encima (espacios latentes distintos).
    """

    def __init__(self, autoencoder: nn.Module):
        super().__init__()
        self.inner = autoencoder
        self.latent_dim = autoencoder.latent_dim
        self.inner.eval()
        for p in self.inner.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        # Ignora el modo: el encoder nunca sale de eval() (BN y dropout fijos).
        return super().train(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.inner.encode(x)


def cvae_loss(recon_logits: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0):
    """BCE sumada sobre píxeles + KL sumada sobre dimensiones latentes, ambas
    promediadas sobre el batch (convención estándar del ELBO del VAE).
    """
    batch_size = x.shape[0]
    recon_loss = nn.functional.binary_cross_entropy_with_logits(recon_logits, x, reduction="sum") / batch_size
    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
    total = recon_loss + beta * kl_loss
    return {"total": total, "recon": recon_loss, "kl": kl_loss}
