"""Diagnóstico de conectividad del material en una máscara periódica."""

from __future__ import annotations

from collections import Counter

import numpy as np


def diagnosticar(mask: np.ndarray) -> dict:
    """Cuenta componentes sobre un toro usando vecindad de cuatro píxeles.

    Además de exigir una sola componente, se comprueba que el material cruce
    los dos pares de bordes. Una única inclusión aislada no forma una red
    continua con sus copias y, por tanto, tampoco se acepta.
    """
    material = np.asarray(mask) > 0.5
    if material.ndim != 2:
        raise ValueError('La máscara debe ser bidimensional')
    alto, ancho = material.shape
    activos = np.flatnonzero(material.ravel())
    if not len(activos):
        return {'n_componentes_periodicas': 0, 'conecta_a1': False,
                'conecta_a2': False, 'fraccion_componente_principal': 0.0,
                'valida': False, 'motivo': 'sin_material'}

    padre = np.arange(alto * ancho, dtype=np.int64)
    tamano = np.ones(alto * ancho, dtype=np.int64)

    def raiz(i: int) -> int:
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = int(padre[i])
        return i

    def unir(a: int, b: int) -> None:
        ra, rb = raiz(a), raiz(b)
        if ra == rb:
            return
        if tamano[ra] < tamano[rb]:
            ra, rb = rb, ra
        padre[rb] = ra
        tamano[ra] += tamano[rb]

    # Derecha y abajo con wrap periódico; cada arista se visita una vez.
    for y, x in zip(*np.nonzero(material)):
        i = int(y * ancho + x)
        xr, ya = (x + 1) % ancho, (y + 1) % alto
        if material[y, xr]:
            unir(i, int(y * ancho + xr))
        if material[ya, x]:
            unir(i, int(ya * ancho + x))

    conteos = Counter(raiz(int(i)) for i in activos)
    conecta_a1 = bool(np.any(material[:, 0] & material[:, -1]))
    conecta_a2 = bool(np.any(material[0, :] & material[-1, :]))
    n_componentes = len(conteos)
    fraccion = max(conteos.values()) / len(activos)
    valida = n_componentes == 1 and conecta_a1 and conecta_a2
    if n_componentes != 1:
        motivo = f'{n_componentes}_componentes_periodicas'
    elif not conecta_a1 or not conecta_a2:
        ejes = '/'.join(nombre for nombre, ok in
                        (('a1', conecta_a1), ('a2', conecta_a2)) if not ok)
        motivo = f'no_conecta_{ejes}'
    else:
        motivo = 'ok'
    return {
        'n_componentes_periodicas': int(n_componentes),
        'conecta_a1': conecta_a1,
        'conecta_a2': conecta_a2,
        'fraccion_componente_principal': float(fraccion),
        'valida': bool(valida),
        'motivo': motivo,
    }
