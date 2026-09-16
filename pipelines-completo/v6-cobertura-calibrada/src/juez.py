"""El juez: geometría -> comportamiento predicho. Una interfaz, tres modelos.

POR QUÉ EXISTE ESTE MÓDULO
---------------------------
Cinco sitios cargan el forward para juzgar candidatas —`04_train_cvae`,
`05_design`, `07_evaluate`, `08_panel` y `prototipo/disenador.py`— y cada uno lo
montaba a mano. Con dos forwards posibles eso serían diez montajes distintos,
y ya había uno mal: `08_panel` y `07_evaluate` construían el MLP con
`encoder.latent_dim` (128) cuando desde v4 come 135 (128 + forma de celda).

Aquí se monta una vez y se elige por config:

    forward:
      activo: cuantiles    # o `cnn` / `mlp`

LA INTERFAZ ES LA IMAGEN, NO EL LATENTE
----------------------------------------
    logits, reg = juez(imgs, forma)

`imgs` es (B, 1, 128, 128) y `forma` el one-hot (B, 7). La forma se conserva en
la interfaz aunque el checkpoint de cuantiles no la consuma. El consumidor no
necesita conocer la arquitectura interna del juez.

Esto además refleja lo que de verdad cambió: con el CNN la cadena se acorta.

    mlp:  decoder -> imagen -> [encoder congelado] -> z(128) -> +forma -> c
    cnn:  decoder -> imagen ------------------------------------> +forma -> c

Un modelo congelado menos entre la candidata y su juicio.

CUÁL USAR, Y POR QUÉ
--------------------
Medido sobre el test de v4 (1495 geometrías), el 2026-09-02:

                          CNN      MLP     k-NN
    frecuencia MAE       0,227    0,264    0,526
    error en anchuras     10,8     12,5      —
    pendiente frecuencia 0,676    0,595      —
    F1 del nº de gaps    0,802    0,778    0,763

El CNN gana en las cinco, +14,1 % en el criterio que decide. El MLP se conserva
seleccionable para poder rehacer la comparación, no por compatibilidad.

LO QUE NO ARREGLA NINGUNO DE LOS DOS
-------------------------------------
La ANCHURA del gap. El CNN le gana a "predecir siempre la mediana" por un
2,3 % (MAE 0,0392 vs 0,0401), pendiente 0,028. No se predice.

Pero NO es el cuello de botella, y está medido — IoU 1D del intervalo
[f - w/2, f + w/2] sobre el test:

    modelo tal cual                     0,023
    ANCHURA perfecta, f del modelo      0,061    <- arreglarla no sirve de nada
    FRECUENCIA perfecta, w del modelo   0,407    <- aqui esta todo

Arreglar la frecuencia vale 10x mas. La razon es aritmetica: el error de
frecuencia (0,227) mide 10,8 anchuras medianas (0,021), asi que el intervalo
predicho cae en otra calle y da igual lo ancho que sea.

**Consecuencia de diseño, sin cambios:** el juez preselecciona, el mallador
decide.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from common import version_path
from arch.cnn_autoencoder import Autoencoder
from arch.cvae import FrozenCNNEncoder
from arch.forward_mlp import ForwardMLPDividido
from arch.forward_cnn import ForwardCNN
from arch.forward_cuantiles import (SalidaCuantiles,
                                    cargar_forward_cuantiles)


class Juez(nn.Module):
    """Envoltorio uniforme. `juez(imgs, forma) -> (logits (B,K), reg (B,K,2))`.

    Siempre en `eval()` y sin gradientes: es un juez, no se entrena. `train()`
    se ignora a propósito, igual que hace `FrozenCNNEncoder`, para que meterlo
    dentro de un bucle de entrenamiento no lo despierte por accidente.
    """

    def __init__(self, tipo: str, modelo: nn.Module, encoder: nn.Module | None,
                 n_extra: int, etiqueta: str = ''):
        super().__init__()
        self.tipo = tipo
        self.modelo = modelo
        self.encoder = encoder
        self.n_extra = n_extra
        self.etiqueta = etiqueta
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def train(self, mode: bool = True):
        return super().train(False)

    def _paso(self, imgs, forma):
        if self.n_extra and forma is None:
            raise ValueError(f'El juez «{self.tipo}» necesita la forma de celda '
                             f'({self.n_extra} dimensiones)')
        if self.tipo == 'cuantiles':
            return self.modelo.compatible_v4(imgs)
        if self.tipo == 'cnn':
            return self.modelo(imgs, forma if self.n_extra else None)
        z = self.encoder(imgs)
        if self.n_extra:
            z = torch.cat([z, forma], dim=1)
        return self.modelo(z)

    def forward(self, imgs: torch.Tensor, forma: torch.Tensor | None = None):
        """Juicio normal, sin gradiente. Es lo que quieren los cuatro
        consumidores que solo evaluan."""
        with torch.no_grad():
            return self._paso(imgs, forma)

    def con_gradiente(self, imgs: torch.Tensor, forma: torch.Tensor | None = None):
        """Igual, pero SIN cortar el grafo: el gradiente llega hasta `imgs`.

        Lo necesita P9 —acoplar el juez a la perdida del cVAE— y solo eso. Los
        pesos del juez siguen con `requires_grad=False`, asi que no se entrena:
        lo unico que se deriva es la imagen que le entra. Y sigue en `eval()`,
        asi que sus BatchNorm usan las estadisticas guardadas y el juicio no
        depende de con quien le toque compartir batch.

        OJO — ESTO ES OPTIMIZAR CONTRA UN JUEZ FIJO. Es territorio de ejemplos
        adversarios: el decoder puede encontrar imagenes que el juez puntua bien
        y que fisicamente no lo son. Dos cosas lo contienen, y las dos hay que
        vigilarlas: el termino de reconstruccion, que ancla la salida a que
        parezca una geometria real, y la puerta de mallado. Ademas el juez tiene
        su propio error (0,227 en frecuencia), asi que la pendiente de la cadena
        no puede pasar de su techo por mucho que se apriete.
        """
        return self._paso(imgs, forma)

    def salida_cuantiles(self, imgs: torch.Tensor,
                         con_gradiente: bool = False) -> SalidaCuantiles:
        """Salida completa del nuevo Forward, incluidos q10 y q90.

        La interfaz histórica de :meth:`forward` usa q50 como anchura y permite
        que los consumidores de v4 sigan funcionando. Este método se reserva
        para métricas y visualizaciones de incertidumbre.
        """
        if self.tipo != 'cuantiles':
            raise TypeError('El juez activo no produce cuantiles')
        if con_gradiente:
            return self.modelo.salida(imgs)
        with torch.no_grad():
            return self.modelo.salida(imgs)


def cargar_juez(cfg: dict, device, cual: str | None = None,
                etiqueta: str = '') -> Juez:
    """Monta el juez que diga el config (o `cual`, que lo sobrescribe).

    Args:
        cual: 'cuantiles' | 'cnn' | 'mlp'. Por defecto
            `cfg['forward']['activo']`.
        etiqueta: sufijo del checkpoint, p. ej. '_sin_aug'.
    """
    cual = (cual or cfg['forward'].get('activo', 'mlp')).lower()
    if cual not in ('cuantiles', 'cnn', 'mlp'):
        raise SystemExit("forward.activo debe ser 'cuantiles', 'cnn' o 'mlp', "
                         f'no {cual!r}')

    if cual == 'cuantiles':
        fc = cfg['forward_cuantiles']
        ruta = version_path(cfg['artifacts']['forward_cuantiles'], mkdir=False) / fc['weights']
        try:
            modelo = cargar_forward_cuantiles(ruta, fc, device)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            raise SystemExit(f'No se pudo cargar el Forward de cuantiles: {exc}') from exc
        return Juez('cuantiles', modelo, None, 0, etiqueta)

    if cual == 'cnn':
        fc = cfg['forward_cnn']
        ruta = version_path(cfg['artifacts']['forward_cnn'], mkdir=False) / f'best{etiqueta}.pt'
        if not ruta.exists():
            raise SystemExit(f'Falta {ruta} — ejecuta antes 03b_train_forward_cnn.py')
        ck = torch.load(ruta, map_location=device, weights_only=False)
        # n_extra se lee del CHECKPOINT, no del config: si se entreno sin forma
        # de celda, montarlo con 7 fallaria al cargar los pesos sin decir por que.
        n_extra = ck['model_state']['cabeza.tronco.0.weight'].shape[1] - fc['embed_dim']
        modelo = ForwardCNN(n_extra=n_extra, embed_dim=fc['embed_dim'],
                            channels=tuple(cfg['encoder']['channels']),
                            canales_1x1=fc['canales_1x1'],
                            input_size=cfg['encoder']['img_size'],
                            hidden_dims=tuple(fc['hidden_dims']),
                            k_max=cfg['target']['k_max'],
                            dropout_conv=fc['dropout_conv'],
                            dropout_cabeza=fc['dropout_cabeza']).to(device)
        modelo.load_state_dict(ck['model_state'])
        return Juez('cnn', modelo, None, n_extra, etiqueta)

    # --- MLP sobre el latente congelado ---
    ec = cfg['encoder']
    ruta = version_path(cfg['artifacts']['forward'], mkdir=False) / f'best{etiqueta}.pt'
    if not ruta.exists():
        raise SystemExit(f'Falta {ruta} — ejecuta antes 03_train_forward.py')
    ck = torch.load(ruta, map_location=device, weights_only=False)
    # Igual que arriba: la entrada sale del checkpoint. Montarlo con
    # `encoder.latent_dim` daba 128 cuando desde v4 son 135, y ese era el bug
    # que arrastraban 07_evaluate y 08_panel.
    d_in = ck['model_state']['tronco.0.weight'].shape[1]
    modelo = ForwardMLPDividido(latent_dim=d_in,
                                hidden_dims=tuple(cfg['forward']['hidden_dims']),
                                k_max=cfg['target']['k_max'],
                                dropout=cfg['forward']['dropout']).to(device)
    modelo.load_state_dict(ck['model_state'])

    ae = Autoencoder(latent_dim=ec['latent_dim'], channels=ec['channels'],
                     dropout=ec['dropout'], input_size=ec['img_size'])
    p_enc = version_path(cfg['artifacts']['encoder'], mkdir=False) / 'best.pt'
    if not p_enc.exists():
        raise SystemExit(f'Falta {p_enc} — ejecuta antes 01_train_encoder.py')
    ae.load_state_dict(torch.load(p_enc, map_location='cpu',
                                  weights_only=False)['model_state'])
    encoder = FrozenCNNEncoder(ae).to(device)
    return Juez('mlp', modelo, encoder, d_in - ec['latent_dim'], etiqueta)


def describir(juez: Juez) -> str:
    n = sum(p.numel() for p in juez.parameters())
    if juez.tipo == 'cuantiles':
        cadena = 'imagen -> [encoder JC congelado] -> conjunto de 15 gaps'
    elif juez.tipo == 'cnn':
        cadena = 'imagen -> conv -> +forma -> c'
    else:
        cadena = 'imagen -> [encoder congelado] -> z -> +forma -> c'
    return f'Juez: {juez.tipo.upper()}  ({n:,} parámetros)   {cadena}'


def techo_pendiente(cfg: dict, cual: str | None = None) -> float:
    """Pendiente del juez sobre geometrías REALES: el techo de la cadena.

    El cVAE no puede seguir el encargo mejor de lo que el juez sabe leerlo, así
    que la pendiente de la cadena hay que dividirla por esta. Se lee del
    metrics.json del juez ACTIVO — estaba fijada a mano a 0,595 (el MLP) y con
    el CNN el techo es 0,676, así que el porcentaje salía inflado un 14 %.
    """
    from common import load_json
    cual = (cual or cfg['forward'].get('activo', 'mlp')).lower()
    clave = ('forward_cuantiles' if cual == 'cuantiles'
             else 'forward_cnn' if cual == 'cnn' else 'forward')
    p = version_path(cfg['artifacts'][clave], mkdir=False) / 'metrics.json'
    if not p.exists():
        raise SystemExit(f'Falta {p}: no se puede leer el techo de la pendiente')
    return float(load_json(p)['pendientes']['frecuencia'])
