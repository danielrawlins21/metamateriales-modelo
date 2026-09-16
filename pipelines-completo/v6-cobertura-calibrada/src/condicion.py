"""Construcción de la condición del cVAE, con y sin enmascarado.

EL PROBLEMA QUE RESUELVE
------------------------
Hasta v2 el cVAE solo veía vectores objetivo COMPLETOS: los 15 slots con todos
los gaps reales de la geometría. Un encargo de verdad no es así — es "quiero un
gap en Omega ~ 1,2, del resto no opino" — y ante ese vector, con un gap y
catorce ceros, el modelo entiende "una geometría con exactamente un gap", que es
una petición mucho más restrictiva de la que se le hizo.

El enmascarado le enseña a rellenar lo que no se le dice. Pero para eso hay que
deshacer antes una ambigüedad: en el vector de 45 números, un slot a cero puede
significar dos cosas opuestas.

    "aquí NO hay gap"   (la geometría tiene menos de 15)
    "aquí no te digo"   (el encargo no se pronuncia)

Se resuelve con un CUARTO CANAL por slot:

    45 = 15 x [presencia, frecuencia, anchura]
    60 = 15 x [presencia, frecuencia, anchura, ESPECIFICADO]

        especificado = 1  ->  forma parte del encargo. Si además presencia = 0,
                              se está pidiendo explícitamente que NO haya gap.
        especificado = 0  ->  el encargo calla; el modelo decide.

Solo cambia lo que consume el cVAE. El ForwardMLP sigue prediciendo sus 45
números y el vector objetivo del proyecto no se toca.
"""

from __future__ import annotations

import numpy as np

K_MAX = 15
DIM_SIN_MASCARA = 45      # 15 x 3
DIM_CON_MASCARA = 60      # 15 x 4


def _partir(c: np.ndarray):
    """(N, 45+E) -> los 45 del vector objetivo y las E columnas de cola.

    Las de cola son propiedades de la GEOMETRIA ENTERA, no de cada gap — hoy la
    forma de celda en one-hot (7). Se dejan pasar tal cual: no se enmascaran ni
    se les anade canal de "especificado", porque el encargo siempre las lleva.
    Por eso la condicion pasa de 60 a 67 y no de 60 a 75.
    """
    c = np.asarray(c, dtype=np.float32)
    return c[:, :DIM_SIN_MASCARA], c[:, DIM_SIN_MASCARA:]


def completa(c: np.ndarray) -> np.ndarray:
    """(N, 45) -> (N, 60) con TODOS los slots marcados como especificados.

    Es el encargo total: "quiero exactamente estos gaps y ningún otro". Los
    slots vacíos quedan con especificado=1 y presencia=0, que es la forma de
    pedir explícitamente que ahí no haya nada.
    """
    c, extra = _partir(c)
    n = len(c)
    out = np.zeros((n, K_MAX, 4), dtype=np.float32)
    out[:, :, :3] = c.reshape(n, K_MAX, 3)
    out[:, :, 3] = 1.0
    return np.hstack([out.reshape(n, DIM_CON_MASCARA), extra])


def enmascarar(c: np.ndarray, rng: np.random.Generator, prob: float = 0.5,
               min_gaps: int = 1, max_gaps: int = 4) -> np.ndarray:
    """(N, 45) -> (N, 60), ocultando parte del encargo en algunas muestras.

    Con probabilidad `prob` se conserva un subconjunto aleatorio de entre
    `min_gaps` y `max_gaps` de los gaps REALES y se oculta todo lo demás
    (valores a cero Y especificado a cero). El resto de las muestras pasan
    completas.

    Por qué no se enmascara siempre: el modelo tiene que seguir sabiendo
    responder a un encargo total. Mezclando ambos casos aprende las dos
    lecturas del cero, que es justo lo que distingue el cuarto canal.

    Se enmascaran solo slots con gap real. Ocultar un slot vacío no aportaría
    nada — pasaría de "no quiero gap aquí" a "me da igual", y no hay ninguna
    geometría que enseñe la diferencia.
    """
    c, extra = _partir(c)
    n = len(c)
    g = c.reshape(n, K_MAX, 3)
    out = np.zeros((n, K_MAX, 4), dtype=np.float32)
    out[:, :, :3] = g
    out[:, :, 3] = 1.0

    for i in np.where(rng.random(n) < prob)[0]:
        presentes = np.flatnonzero(g[i, :, 0] > 0.5)
        if len(presentes) <= min_gaps:
            continue                       # nada que ocultar sin vaciar el encargo
        k = int(rng.integers(min_gaps, min(max_gaps, len(presentes)) + 1))
        visibles = rng.choice(presentes, k, replace=False)
        oculto = np.ones(K_MAX, dtype=bool)
        oculto[visibles] = False
        out[i, oculto, :] = 0.0
    return np.hstack([out.reshape(n, DIM_CON_MASCARA), extra])


def solo_k_gaps(c: np.ndarray, k: int = 1, rng: np.random.Generator | None = None,
                por_anchura: bool = True) -> np.ndarray:
    """(N, 45) -> (N, 60) conservando exactamente `k` gaps de cada objetivo.

    Determinista y sin aleatoriedad por defecto: se queda con los `k` gaps más
    ANCHOS, que son los que un encargo real mencionaría. Sirve para evaluar el
    modelo en el régimen para el que existe el enmascarado — encargos parciales
    — sobre objetivos que sabemos alcanzables porque salen de geometrías reales.
    """
    c, extra = _partir(c)
    n = len(c)
    g = c.reshape(n, K_MAX, 3)
    out = np.zeros((n, K_MAX, 4), dtype=np.float32)
    for i in range(n):
        presentes = np.flatnonzero(g[i, :, 0] > 0.5)
        if len(presentes) == 0:
            continue
        if por_anchura:
            elegidos = presentes[np.argsort(-g[i, presentes, 2])[:k]]
        else:
            rng = rng or np.random.default_rng(0)
            elegidos = rng.choice(presentes, min(k, len(presentes)), replace=False)
        out[i, elegidos, :3] = g[i, elegidos]
        out[i, elegidos, 3] = 1.0
    return np.hstack([out.reshape(n, DIM_CON_MASCARA), extra])


def solo_lo_pedido(c: np.ndarray) -> np.ndarray:
    """(N, 45) -> (N, 60) marcando como especificados SOLO los slots con gap.

    Es la lectura correcta de un encargo escrito por una persona: "quiero un
    gap en Omega=1,2 de anchura 0,3" no dice nada de los otros catorce slots.
    Tratarlo con `completa` lo convertiría en "y ningún gap más", que es una
    petición mucho más restrictiva y que casi ninguna geometría real cumple.
    """
    c, extra = _partir(c)
    n = len(c)
    g = c.reshape(n, K_MAX, 3)
    out = np.zeros((n, K_MAX, 4), dtype=np.float32)
    hay = g[:, :, 0] > 0.5
    out[:, :, :3] = np.where(hay[:, :, None], g, 0.0)
    out[:, :, 3] = hay.astype(np.float32)
    return np.hstack([out.reshape(n, DIM_CON_MASCARA), extra])


def colocaciones(c45: np.ndarray, k_max: int = K_MAX) -> np.ndarray:
    """(1, 45) con k gaps pedidos -> (K_MAX-k+1, 60): el encargo en cada posición.

    POR QUÉ HACE FALTA
    ------------------
    Las ranuras están ordenadas por frecuencia ascendente, así que la posición
    del gap NO es arbitraria: dice cuántos gaps de menor frecuencia tiene la
    geometría. Poner el encargo en la ranura 0 es pedir "y que no haya ninguno
    por debajo", que en el dataset real solo ocurre el 11,5 % de las veces —
    la mediana está en la ranura 4.

    Medido, con el mismo presupuesto de 30 candidatas por encargo:

        siempre en la ranura 0     cobertura  1,2 %
        repartidas por las 15      cobertura 15,0 %

    Doce veces mejor. El usuario no sabe (ni tiene por qué) cuántos gaps de
    menor frecuencia va a tener la geometría, así que esa incógnita se explora
    en vez de fijarse en cero.

    El bloque de gaps pedidos se desplaza entero, conservando su orden relativo:
    con 2 gaps salen las colocaciones (0,1), (1,2) … (13,14).
    """
    c, extra = _partir(np.asarray(c45, dtype=np.float32).reshape(1, -1))
    g = c.reshape(-1, 3)
    pedidos = g[g[:, 0] > 0.5]
    n = len(pedidos)
    if n == 0:
        vacio = np.zeros((1, k_max * 3 + extra.shape[1]), dtype=np.float32)
        vacio[:, k_max * 3:] = extra
        return completa(vacio)
    out = np.zeros((k_max - n + 1, k_max, 4), dtype=np.float32)
    for off in range(k_max - n + 1):
        out[off, off:off + n, :3] = pedidos
        out[off, off:off + n, 3] = 1.0
    # la forma de celda es la misma en las 15 colocaciones: es propiedad de la
    # geometria, no del sitio donde caiga el gap
    return np.hstack([out.reshape(-1, DIM_CON_MASCARA),
                      np.repeat(extra, len(out), axis=0)])


def a_45(c60: np.ndarray) -> np.ndarray:
    """(N, 60+E) -> (N, 45), tirando el canal de máscara y la cola.

    Para comparar contra lo que predice el forward, que devuelve 45. Los slots
    no especificados ya vienen a cero, así que no hace falta limpiarlos.
    """
    c = np.asarray(c60)[:, :DIM_CON_MASCARA]
    return c.reshape(len(c), K_MAX, 4)[:, :, :3].reshape(len(c), DIM_SIN_MASCARA)


def dim_condicion(usa_mascara: bool, n_extra: int = 0) -> int:
    return (DIM_CON_MASCARA if usa_mascara else DIM_SIN_MASCARA) + n_extra


def nombre_checkpoint(cfg: dict) -> str:
    """Checkpoint del cVAE según la variante que pida el config.

    Las dos variantes no son intercambiables (45 vs 60 de condición) y hay que
    poder compararlas, así que no comparten fichero: entrenar una borraría la
    otra.
    """
    return 'best_masking.pt' if cfg['cvae'].get('condition_masking') else 'best.pt'


def preparar(c45: np.ndarray, target_dim: int, k: int = 0) -> np.ndarray:
    """Deja la condición lista para el decoder de un modelo de `target_dim`.

    `k > 0` recorta el encargo a los k gaps más anchos — el uso real. Se aplica
    también a los modelos de 45, aunque no sepan interpretarlo: es lo que un
    usuario les daría, y medirlos con el encargo completo mientras al otro se
    le recorta no compararía lo mismo.
    """
    c = solo_k_gaps(c45, k) if k else completa(c45)
    return a_45(c) if target_dim == DIM_SIN_MASCARA else c
