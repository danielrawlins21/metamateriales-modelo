"""Métricas de entrega de v6 para encargos de un único bandgap.

Estas métricas separan dos preguntas distintas:

* ``toca_gap`` / ``cumple``: ¿existe solapamiento positivo con lo pedido?
* ``cobertura``: ¿qué fracción del intervalo pedido queda cubierta?

El exceso fuera del intervalo pedido no participa en ninguna de las dos.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

import metricas_cvae as mc


TOLERANCIA_SOLAPAMIENTO = 1e-6


def evaluar_candidata(c_objetivo: np.ndarray, c_respuesta: np.ndarray,
                      tolerancia: float = TOLERANCIA_SOLAPAMIENTO) -> dict:
    """Evalúa una respuesta frente a un encargo de exactamente un gap.

    ``cumple`` no significa cobertura completa: significa que FEM ha detectado
    al menos un solapamiento de longitud superior a ``tolerancia``.
    """
    pedidos = mc.intervalos(c_objetivo)
    if len(pedidos) != 1:
        raise ValueError('La evaluación admite exactamente un gap por encargo')

    pedido = pedidos[0]
    anchura_pedida = float(pedido[1] - pedido[0])
    if anchura_pedida <= 0:
        raise ValueError('El gap pedido debe tener anchura positiva')

    respuesta = mc.intervalos(c_respuesta)
    intersecciones = mc._interseccion(pedidos, respuesta)
    solapamiento = mc._union_longitud(intersecciones)
    # Los vectores suelen llegar en float32: dos extremos matemáticamente
    # iguales pueden dejar un residuo del orden de 1e-8.
    if solapamiento <= tolerancia:
        solapamiento = 0.0
    cobertura = float(np.clip(solapamiento / anchura_pedida, 0.0, 1.0))
    toca = bool(solapamiento > tolerancia)

    if len(respuesta):
        solapes_individuales = np.maximum(
            np.minimum(pedido[1], respuesta[:, 1])
            - np.maximum(pedido[0], respuesta[:, 0]),
            0.0,
        )
        n_gaps_que_tocan = int(np.sum(solapes_individuales > tolerancia))
    else:
        n_gaps_que_tocan = 0

    return {
        'cumple': toca,
        'toca_gap': toca,
        'cobertura': cobertura,
        'solapamiento': float(solapamiento),
        'anchura_pedida': anchura_pedida,
        'n_gaps_que_tocan': n_gaps_que_tocan,
        'tolerancia_solapamiento': float(tolerancia),
    }


def resumir_encargo(candidatas: Sequence[dict]) -> dict:
    """Resume candidatas ya ordenadas por el ranking del Forward.

    Cada elemento debe contener ``cumple`` (o ``toca_gap``) y ``cobertura``.
    ``rank_forward`` es opcional; si falta se utiliza la posición, empezando en
    uno. El máximo de cobertura solo se busca entre candidatas que tocan.
    """
    normalizadas = []
    for posicion, candidata in enumerate(candidatas, start=1):
        toca = bool(candidata.get('toca_gap', candidata.get('cumple', False)))
        cobertura = float(candidata.get('cobertura', 0.0))
        normalizadas.append({
            'toca': toca,
            'cobertura': cobertura,
            'rank_forward': int(candidata.get('rank_forward', posicion)),
        })

    aciertos = [c for c in normalizadas if c['toca']]
    if aciertos:
        mejor = min(aciertos, key=lambda c: (-c['cobertura'], c['rank_forward']))
        cobertura_maxima = mejor['cobertura']
        rank_mejor = mejor['rank_forward']
    else:
        cobertura_maxima = 0.0
        rank_mejor = None

    return {
        'n_candidatas': len(normalizadas),
        'n_candidatas_tocan': len(aciertos),
        'alguna_toca': bool(aciertos),
        'cobertura_maxima_entre_aciertos': float(cobertura_maxima),
        'rank_forward_mejor_cobertura': rank_mejor,
    }


def resumir_barrido(encargos: Sequence[Sequence[dict]]) -> dict:
    """Agrega las métricas de contacto y cobertura de todo un barrido."""
    por_encargo = [resumir_encargo(candidatas) for candidatas in encargos]
    n_encargos = len(por_encargo)
    n_candidatas = sum(x['n_candidatas'] for x in por_encargo)
    n_candidatas_tocan = sum(x['n_candidatas_tocan'] for x in por_encargo)
    maximos = [x['cobertura_maxima_entre_aciertos'] for x in por_encargo]
    n_encargos_tocan = sum(x['alguna_toca'] for x in por_encargo)

    return {
        'n_encargos': n_encargos,
        'n_encargos_tocan': int(n_encargos_tocan),
        'tasa_encargos_tocan': (
            float(n_encargos_tocan / n_encargos) if n_encargos else 0.0),
        'n_candidatas': int(n_candidatas),
        'n_candidatas_tocan': int(n_candidatas_tocan),
        'tasa_candidatas_tocan': (
            float(n_candidatas_tocan / n_candidatas) if n_candidatas else 0.0),
        'cobertura_maxima_media_por_encargo': (
            float(np.mean(maximos)) if maximos else 0.0),
        'cobertura_maxima_mediana_por_encargo': (
            float(np.median(maximos)) if maximos else 0.0),
        'coberturas_maximas_por_encargo': maximos,
        'resultados_por_encargo': por_encargo,
    }
