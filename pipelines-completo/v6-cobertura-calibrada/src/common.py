"""Utilidades compartidas por todos los scripts del pipeline v0.

Centraliza tres cosas que si no acabarían copiadas en cada script:
  - resolución de rutas (raíz del proyecto vs carpeta de esta versión)
  - carga de config.yaml
  - semilla y dispositivo

Uso desde cualquier script del pipeline:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from common import load_config, project_path, version_path, set_seed, get_device
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import yaml

# .../pipelines-completo/v0-original/src/common.py
_SRC_DIR      = Path(__file__).resolve().parent
VERSION_DIR   = _SRC_DIR.parent                  # .../v0-original
PIPELINES_DIR = VERSION_DIR.parent               # .../pipelines-completo
PROJECT_ROOT  = PIPELINES_DIR.parent             # .../TFM_autoencoder


def load_config(path: str | Path | None = None) -> dict:
    """Carga config.yaml de esta versión."""
    cfg_path = Path(path) if path else VERSION_DIR / 'config.yaml'
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def project_path(rel: str | Path) -> Path:
    """Resuelve una ruta del bloque `data:` / `encoder:` del config.

    Son relativas a la raíz del proyecto porque los datos pesados se
    comparten entre versiones y viven fuera de pipelines-completo/.
    """
    return PROJECT_ROOT / rel


def version_path(rel: str | Path, mkdir: bool = True) -> Path:
    """Resuelve una ruta del bloque `artifacts:` del config.

    Son relativas a la carpeta de esta versión: cada versión guarda sus
    propios latentes, checkpoints y métricas.
    """
    p = VERSION_DIR / rel
    if mkdir:
        p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    """Fija la semilla en random, numpy y torch (si está disponible)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(preferred: str = 'cuda'):
    """Devuelve el device pedido, o cpu si no hay GPU disponible."""
    import torch
    if preferred == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def save_json(obj, path: str | Path) -> None:
    """Guarda un dict a JSON con indentación, creando el directorio si falta."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# --- Grupos wallpaper -------------------------------------------------
# Ordenados por longitud descendente: al parsear un stem hay que probar
# primero 'p31m' antes que 'p3', o 'p3' se lo comería.
WALLPAPER_GROUPS = [
    'p31m', 'p3m1', 'p4m', 'p4g', 'pmm', 'pmg', 'pgg', 'cmm',
    'p6m', 'p1', 'p2', 'p3', 'p4', 'p6', 'pm', 'pg', 'cm',
]


def parse_group(stem: str) -> str:
    """Extrae el grupo wallpaper del nombre de fichero.

    'cm_hexagonal1_2024-05-22_14-23-52.891252' -> 'cm'
    """
    for g in sorted(WALLPAPER_GROUPS, key=len, reverse=True):
        if stem.startswith(g + '_'):
            return g
    return ''


# --- Forma de celda ---------------------------------------------------
def forma_one_hot(df, cfg):
    """(N, 7) one-hot de la forma de celda, en el orden de `target.formas`.

    Por qué hace falta. La imagen corregida ES la celda unidad, así que todas
    las geometrías salen dibujadas en el mismo cuadrado y la silueta ya no
    delata si la red es cuadrada, hexagonal o rómbica. Y Ω depende de la red:
    misma forma dibujada en dos celdas distintas da espectros distintos.

    Medido con ridge sobre la imagen 32x32 al target recortado:
        imagen corregida                 +5,8 %
        imagen corregida + forma        +21,9 %   <- devolver la forma recupera
        imagen distorsionada (v0-v3)    +22,4 %      casi todo el hueco

    Es propiedad de la GEOMETRÍA ENTERA, no de cada gap: se añade una sola vez
    al final del vector, no por ranura. Por eso la condición del cVAE pasa de
    60 a 67 y no de 60 a 75.

    El orden de `target.formas` es fijo y está avisado en config.yaml:
    cambiarlo invalida cualquier modelo ya entrenado.
    """
    import numpy as np
    formas = list(cfg['target']['formas'])
    idx = {f: i for i, f in enumerate(formas)}
    desconocidas = set(df['shape']) - set(formas)
    if desconocidas:
        raise SystemExit(f'Formas fuera de config.target.formas: {sorted(desconocidas)}')
    oh = np.zeros((len(df), len(formas)), dtype='float32')
    oh[np.arange(len(df)), [idx[s] for s in df['shape']]] = 1.0
    return oh
