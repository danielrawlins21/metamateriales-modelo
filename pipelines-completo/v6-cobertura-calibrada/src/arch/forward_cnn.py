"""Forward CNN: imagen -> comportamiento (c). El experimento de v4 sobre P3.

QUÉ SE ESTÁ PROBANDO, EXACTAMENTE
----------------------------------
El forward de v0-v4 (`forward_mlp.ForwardMLPDividido`) no ve la geometría: ve el
latente que el **autoencoder** produjo. Y ese autoencoder se entrenó para
reconstruir píxeles, no para predecir bandas. Nada garantiza que las 128
componentes que guardó sean las que hacen falta: puede haber tirado justo lo que
determina dónde caen los gaps.

Esta arquitectura es la MISMA cabeza sobre el MISMO tamaño de representación —
solo que la representación se aprende contra el target en vez de heredarse:

    forward actual   imagen ─[encoder CONGELADO]→ z(128) ─┬─ +forma(7) → cabeza
                                                          └─ entrenado para
                                                             reconstruir píxeles

    forward CNN      imagen ─[tronco conv ENTRENABLE]→ h(128) ─┬─ +forma(7) → cabeza
                                                                └─ entrenado para
                                                                   predecir gaps

Todo lo demás es idéntico a propósito: mismos 128, misma concatenación de la
forma, misma `ForwardMLPDividido` de cabeza, misma pérdida, mismo split, mismas
métricas. Si el error baja, la culpa era del latente congelado. Si no se mueve,
el límite es físico y P3 queda cerrado.

EL TRONCO ES EL DEL ENCODER, LITERALMENTE
------------------------------------------
Se reutiliza `EncoderBlock` de `cnn_autoencoder` — convoluciones circulares
incluidas, que es lo que hace la periodicidad estructural en vez de aprendida
(verificado en `test_circular.py`: error 0,00e+00 al desplazar, frente a 2,6e-02
con relleno de ceros). Así `--init-encoder` puede arrancar desde los pesos ya
entrenados, y la comparación no mezcla "otra arquitectura" con "otro objetivo".

EL 1x1 ANTES DE APLANAR NO ES DECORACIÓN
-----------------------------------------
Tras los 4 bloques quedan 256 canales a 8x8 = 16 384 valores. Un `Linear` de ahí
a 128 son 2,1 M de parámetros para 6 986 muestras de entrenamiento: 300 por
muestra, y la lección de v0 fue que con 265 por muestra lo batía un k-NN sin
parámetros. La convolución 1x1 baja a 64 canales (16 k parámetros) y deja el
aplanado en 4 096, así que el `Linear` son 524 k. No toca los 4 bloques, de modo
que `--init-encoder` sigue valiendo.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from arch.cnn_autoencoder import EncoderBlock
from arch.forward_mlp import ForwardMLPDividido


class ForwardCNN(nn.Module):
    """imagen (B,1,128,128) [+ forma (B,7)] -> (logits (B,K), reg (B,K,2)).

    Args:
        n_extra:    dimensiones concatenadas tras el tronco conv (7 = forma de
                    celda). 0 para no usarla.
        embed_dim:  tamaño de la representación aprendida. 128 por defecto para
                    igualar al latente del encoder — es el control del experimento.
        channels:   canales por bloque conv. Los del encoder, para poder heredarlos.
        canales_1x1: canales tras la 1x1 que precede al aplanado.
        hidden_dims / dropout_cabeza / k_max: van tal cual a ForwardMLPDividido.
    """

    def __init__(self, n_extra: int = 7, embed_dim: int = 128,
                 channels: tuple[int, ...] = (32, 64, 128, 256),
                 canales_1x1: int = 64, input_size: int = 128,
                 hidden_dims: tuple[int, ...] = (128, 128), k_max: int = 15,
                 dropout_conv: float = 0.1, dropout_cabeza: float = 0.3,
                 padding_mode: str = 'circular'):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_extra = n_extra
        channels = tuple(channels)
        self.channels = channels

        enc_in = (1,) + channels[:-1]
        # Dropout en todos menos el último, igual que en el autoencoder: el
        # último bloque alimenta directamente el cuello de botella.
        self.bloques = nn.ModuleList([
            EncoderBlock(enc_in[i], channels[i],
                         dropout=dropout_conv if i < len(channels) - 1 else 0.0,
                         padding_mode=padding_mode)
            for i in range(len(channels))
        ])

        lado = input_size // (2 ** len(channels))          # 128 -> 8
        self.reduce = nn.Sequential(
            nn.Conv2d(channels[-1], canales_1x1, kernel_size=1, bias=False),
            nn.BatchNorm2d(canales_1x1),
            nn.ReLU(inplace=True),
        )
        self.flat_dim = canales_1x1 * lado * lado          # 64*8*8 = 4096
        self.fc_embed = nn.Linear(self.flat_dim, embed_dim)

        # La cabeza es LA MISMA CLASE que usa el forward de latentes. No es
        # reutilización por comodidad: es lo que hace comparable el experimento.
        self.cabeza = ForwardMLPDividido(latent_dim=embed_dim + n_extra,
                                         hidden_dims=tuple(hidden_dims),
                                         k_max=k_max, dropout=dropout_cabeza)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """La representación aprendida, (B, embed_dim). El análogo de z."""
        for b in self.bloques:
            x = b(x)
        x = self.reduce(x).flatten(start_dim=1)
        return self.fc_embed(x)

    def forward(self, x: torch.Tensor, extra: torch.Tensor | None = None):
        h = self.embed(x)
        if self.n_extra:
            if extra is None:
                raise ValueError(f'ForwardCNN espera `extra` de {self.n_extra} '
                                 f'dimensiones (la forma de celda)')
            h = torch.cat([h, extra], dim=1)
        return self.cabeza(h)

    # -----------------------------------------------------------------
    def cargar_tronco_del_encoder(self, state_dict: dict) -> int:
        """Copia los pesos de los 4 bloques conv del autoencoder ya entrenado.

        Control del experimento: separa "la arquitectura conv ayuda" de
        "entrenar contra el target ayuda". Arrancando desde el encoder, lo único
        que cambia respecto al forward actual es que el tronco puede moverse.

        Devuelve cuántos tensores se copiaron; 0 significa que no encajó nada y
        hay que mirar por qué en vez de entrenar a ciegas.
        """
        propio = self.state_dict()
        copiados = 0
        for k, v in state_dict.items():
            if not k.startswith('encoder_blocks.'):
                continue
            destino = k.replace('encoder_blocks.', 'bloques.')
            if destino in propio and propio[destino].shape == v.shape:
                propio[destino] = v.clone()
                copiados += 1
        self.load_state_dict(propio)
        return copiados


if __name__ == '__main__':
    m = ForwardCNN()
    n = sum(p.numel() for p in m.parameters())
    print(f'Parámetros: {n:,}')
    for nom, mod in [('tronco conv', m.bloques), ('1x1', m.reduce),
                     ('fc_embed', m.fc_embed), ('cabeza', m.cabeza)]:
        print(f'  {nom:<12} {sum(p.numel() for p in mod.parameters()):>10,}')

    x = torch.randn(4, 1, 128, 128)
    extra = torch.eye(7)[[0, 1, 2, 3]]
    logits, reg = m(x, extra)
    print(f'\nimagen {tuple(x.shape)} + forma {tuple(extra.shape)}'
          f' -> logits {tuple(logits.shape)}  reg {tuple(reg.shape)}')
    print(f'embed  : {tuple(m.embed(x).shape)}   (el análogo de z)')
