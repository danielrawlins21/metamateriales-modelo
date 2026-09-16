"""
Fase C — Arquitectura del autoencoder CNN.

Encoder: 4 bloques Conv2d(stride=2) + BN + ReLU + Dropout → Flatten → Linear → z
Decoder: Linear → Reshape → 4 bloques Upsample + Conv2d + BN + ReLU → logits

TODAS las convoluciones usan `padding_mode='circular'` (v4).
    Una celda unidad es periódica: lo que hay a la derecha del borde derecho es
    el borde izquierdo, no vacío. El relleno de ceros le enseña a la red que
    fuera de la imagen no hay nada, que es falso.

    Con las imágenes de `data/images/` daba igual, porque estaban cizalladas y
    el 0 % tenía material en el borde. Con `data/images_celda/` el 99,9 % lo
    tiene, y ahí la periodicidad deja de ser gratis: hay que aprenderla. El
    relleno circular la hace estructural en vez de aprendida.

    Ojo con la costura: la imagen tiene la columna 0 repetida en la 127 (ambas
    son el borde de la celda, que es el mismo sitio), así que su periodo real
    es 127 y `circular` envuelve con 128. Es un desfase de 1 píxel en la
    costura — despreciable frente al relleno de ceros, que es incorrecto en los
    cuatro bordes a la vez, pero está aquí anotado.

Dropout:
    Se aplica después de cada bloque del encoder (excepto el último) para
    reducir el overfitting. Durante inferencia (model.eval()) PyTorch lo
    desactiva automáticamente.

La salida del decoder son logits (sin Sigmoid).
Usar BCEWithLogitsLoss para entrenar y torch.sigmoid() para visualizar.
"""

import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0, padding_mode='circular'):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False,
                      padding_mode=padding_mode),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, last=False, padding_mode='circular'):
        super().__init__()
        layers = [
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False,
                      padding_mode=padding_mode),
        ]
        if not last:
            layers += [nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Autoencoder(nn.Module):
    """Autoencoder CNN para imágenes binarias 128×128.

    Args:
        latent_dim: tamaño del vector latente z (default: 64)
        channels:   canales por bloque del encoder (default: [32, 64, 128, 256])
        dropout:    dropout tras cada bloque del encoder excepto el último (default: 0.0)
        padding_mode: 'circular' (por defecto, v4) o 'zeros' (v0-v3). Ver el
                    encabezado del módulo: la celda es periódica.
    """

    def __init__(self, latent_dim=64, channels=None, dropout=0.0, input_size=128,
                 padding_mode='circular'):
        super().__init__()
        self.padding_mode = padding_mode
        if channels is None:
            channels = [32, 64, 128, 256]

        self.latent_dim = latent_dim
        self.channels = channels
        # Cada bloque encoder reduce la resolución espacial a la mitad (stride=2).
        # Con 4 bloques: 128→64→32→16→8  o  256→128→64→32→16
        self.bottleneck_spatial = input_size // (2 ** len(channels))
        self.bottleneck_ch = channels[-1]
        self.bottleneck_flat = self.bottleneck_ch * self.bottleneck_spatial ** 2

        # --- Encoder ---
        # Dropout en todos los bloques menos el último, igual que en el GNN:
        # el último bloque alimenta directamente el cuello de botella.
        enc_in = [1] + channels[:-1]
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(enc_in[i], channels[i],
                         dropout=dropout if i < len(channels) - 1 else 0.0,
                         padding_mode=padding_mode)
            for i in range(len(channels))
        ])
        self.fc_enc = nn.Linear(self.bottleneck_flat, latent_dim)

        # --- Decoder ---
        self.fc_dec = nn.Linear(latent_dim, self.bottleneck_flat)
        dec_ch = list(reversed(channels))           # [256, 128, 64, 32]
        dec_in  = dec_ch                            # entrada de cada bloque
        dec_out = dec_ch[1:] + [1]                 # salida de cada bloque
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(dec_in[i], dec_out[i], last=(i == len(dec_ch) - 1),
                         padding_mode=padding_mode)
            for i in range(len(dec_ch))
        ])

    def encode(self, x):
        for block in self.encoder_blocks:
            x = block(x)
        x = x.flatten(start_dim=1)
        return self.fc_enc(x)

    def decode(self, z):
        x = self.fc_dec(z)
        x = x.view(-1, self.bottleneck_ch,
                   self.bottleneck_spatial, self.bottleneck_spatial)
        for block in self.decoder_blocks:
            x = block(x)
        return x  # logits — sin Sigmoid

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


if __name__ == '__main__':
    model = Autoencoder(latent_dim=64)

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parámetros totales:     {total_params:,}')
    print(f'Parámetros entrenables: {trainable:,}')

    # Verificar forward pass con batch de 4 imágenes
    x = torch.randn(4, 1, 128, 128)
    z    = model.encode(x)
    logits = model.decode(z)
    recon  = torch.sigmoid(logits)

    print(f'\nForward pass:')
    print(f'  input  : {x.shape}')
    print(f'  z      : {z.shape}     ← espacio latente')
    print(f'  logits : {logits.shape}')
    print(f'  recon  : {recon.shape}  valores [{recon.min():.3f}, {recon.max():.3f}]')

    # Verificar que la pérdida funciona
    loss_fn = nn.BCEWithLogitsLoss()
    target  = (x > 0).float()
    loss    = loss_fn(logits, target)
    print(f'\nBCEWithLogitsLoss: {loss.item():.4f}  (valor inicial esperado ~0.69)')
