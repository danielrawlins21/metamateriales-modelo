"""Métricas del cVAE — las que responden a la pregunta del proyecto.

    "Si pido un comportamiento acústico, ¿me devuelve una geometría que lo tiene?"

Este módulo sustituye a la instrumentación de v0/v1, que medía tres cosas y
ninguna era esa:

    iou_reconstruccion   si el modelo copia la imagen que le enseñas
    delta_condicion      IoU(gen con c real) vs IoU(gen con c ajeno)
    periodicidad         nada útil (94 % candidatas vs 78,6 % reales)

El `delta` se retira por un motivo que se ve en los números del barrido de v2:
`z16_b20` reconstruye con IoU 0,766, pero al generar desde el prior el parecido
con la imagen original cae a 0,372. **Y está bien que caiga**: el diseño inverso
es uno-a-muchos, hay muchas geometrías distintas con el mismo comportamiento.
Medir el parecido con *una* imagen concreta castiga al generador por hacer
justo lo que le pedimos.

LAS TRES MÉTRICAS
-----------------
  1. COBERTURA      ¿tiene la candidata el comportamiento pedido?  <- decide
  2. PLAUSIBILIDAD  ¿parece una geometría?                          <- prerrequisito
  3. DIVERSIDAD     ¿son distintas entre sí?                        <- guardarraíl

Ninguna significa nada sin sus referencias. Ver `08_panel.py`, que las calcula.

CONVENIO DEL VECTOR OBJETIVO
----------------------------
`c` tiene 45 componentes = 15 gaps x [presencia, frecuencia_media, anchura],
aplanado en ese orden (pres_0, freq_0, anch_0, pres_1, ...). Aquí siempre se
usa **crudo**: sin normalizar y sin log, igual que en el resto del pipeline.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label

# Un gap con anchura 0 daría un intervalo degenerado y una unión nula.
_EPS_UNION = 1e-9


# =====================================================================
#  1 · COBERTURA
# =====================================================================
#
#  Un band gap es un INTERVALO de frecuencias prohibidas: [f - w/2, f + w/2].
#  Así que "¿cumple el objetivo?" tiene respuesta geométrica: ¿solapan el
#  intervalo pedido y el obtenido?
#
#      pedido:    ────[ 0.70 ──────── 0.90 ]────
#      obtenido:  ──────────[ 0.75 ──────── 0.95 ]──
#                           └─ solape ─┘
#                 IoU = 0.15 / 0.25 = 0.60
#
#  Se prefiere esto a un MSE sobre las 45 componentes por dos razones: es
#  interpretable físicamente, y no premia al que no se moja (la lección que
#  dejó el k-NN en v1).


def c_desde_forward(logits: np.ndarray, reg: np.ndarray,
                    log_anchura: bool = True, log_epsilon: float = 1e-4) -> np.ndarray:
    """Salida cruda del ForwardMLPDividido -> vector objetivo (K, 45).

    Aplica la MÁSCARA de presencia: si el modelo dice que ese slot no tiene
    gap, sus valores de frecuencia y anchura salen a cero. La cabeza dividida
    solo entrena la regresión sobre los slots con gap real, así que en los
    vacíos devuelve valores arbitrarios; dejarlos pasar rompe cualquier
    comparación con el objetivo (en v1 costó 0,634 de MSE frente a 0,152).

    Vive aquí, y no en cada script, para que el entrenamiento y el panel
    conviertan exactamente igual — si divergen, sus métricas dejan de ser
    comparables sin que nada avise.
    """
    pres = 1.0 / (1.0 + np.exp(-logits))
    anch = (np.clip(np.exp(reg[:, :, 1]) - log_epsilon, 0, None)
            if log_anchura else reg[:, :, 1])
    m = pres > 0.5
    return np.stack([pres,
                     np.where(m, reg[:, :, 0], 0.0),
                     np.where(m, anch, 0.0)], axis=2).reshape(len(pres), -1)


def intervalos(c: np.ndarray) -> np.ndarray:
    """`c` (45,) o (15,3) -> array (k, 2) con [inicio, fin] de cada gap presente.

    Los slots de relleno (presencia = 0) se descartan: son padding, no gaps.
    """
    g = np.asarray(c, dtype=np.float64).reshape(-1, 3)
    hay = g[:, 0] > 0.5
    f = g[hay, 1]
    w = np.maximum(g[hay, 2], 0.0)
    return np.stack([f - w / 2.0, f + w / 2.0], axis=1)


def iou_intervalos(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """(m,2) x (n,2) -> matriz (m,n) con el IoU de cada par de intervalos."""
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)))
    lo = np.maximum(A[:, None, 0], B[None, :, 0])
    hi = np.minimum(A[:, None, 1], B[None, :, 1])
    inter = np.maximum(hi - lo, 0.0)
    union = (A[:, 1] - A[:, 0])[:, None] + (B[:, 1] - B[:, 0])[None, :] - inter
    return inter / np.maximum(union, _EPS_UNION)


def emparejar(M: np.ndarray) -> list[tuple[int, int, float]]:
    """Emparejamiento voraz: toma el par de mayor IoU y descarta su fila y columna.

    Voraz y no óptimo (Hungarian daría el máximo global), pero con gaps
    ordenados por frecuencia el solape es prácticamente diagonal y la
    diferencia es despreciable. A cambio no añade dependencias ni coste.
    """
    M = M.copy()
    pares = []
    while M.size and M.max() > 0:
        i, j = np.unravel_index(M.argmax(), M.shape)
        pares.append((int(i), int(j), float(M[i, j])))
        M[i, :] = -1.0
        M[:, j] = -1.0
    return pares


# ---------------------------------------------------------------------
#  La métrica de decisión de v4 — CUBRIMIENTO
# ---------------------------------------------------------------------
#
#  POR QUÉ SE JUBILA EL IoU. Castiga producir un gap MÁS ANCHO de lo pedido,
#  que es lo contrario de lo que quiere un cliente. Medido: pedido
#  [1,17 · 1,23], una candidata [1,10 · 1,30] que lo bloquea ENTERO puntúa
#  0,300, y otra que solo bloquea la mitad puntúa 0,333 — más alto. La métrica
#  prefería la peor.
#
#  El IoU se conserva abajo, jubilado, porque hace falta para releer los
#  números de v0–v2.


def _union_longitud(I: np.ndarray) -> float:
    """Longitud total de la unión de intervalos (m, 2). Tolera solapes."""
    I = np.asarray(I, dtype=np.float64).reshape(-1, 2)
    if len(I) == 0:
        return 0.0
    I = I[np.argsort(I[:, 0])]
    total, lo, hi = 0.0, I[0, 0], I[0, 1]
    for a, b in I[1:]:
        if a > hi:
            total += hi - lo
            lo, hi = a, b
        else:
            hi = max(hi, b)
    return float(total + hi - lo)


def _interseccion(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Trozos comunes a A y B. Su unión es |unión(A) ∩ unión(B)|."""
    A = np.asarray(A).reshape(-1, 2)
    B = np.asarray(B).reshape(-1, 2)
    if len(A) == 0 or len(B) == 0:
        return np.zeros((0, 2))
    lo = np.maximum(A[:, None, 0], B[None, :, 0])
    hi = np.minimum(A[:, None, 1], B[None, :, 1])
    m = hi > lo
    return np.stack([lo[m], hi[m]], axis=1)


def cubrimiento(c_obj: np.ndarray, c_cand: np.ndarray,
                min_servido: float = 0.5) -> dict:
    """Cuánto de la banda PEDIDA queda de verdad bloqueada.

    Tres números, y los tres hacen falta:

        cubierto        banda pedida bloqueada / banda pedida total.  DECIDE.
                        Un gap más ancho de lo pedido cubre el 100 %: no se
                        castiga pasarse, que es el defecto del IoU.

        gaps_servidos   fracción de los gaps PEDIDOS cubiertos al menos a
                        `min_servido`. Sin él, `cubierto` premia atender bien
                        un pedido ancho y abandonar uno estrecho, porque pesa
                        por área.

        exceso          banda bloqueada FUERA de lo pedido, en unidades de la
                        banda pedida. NO es criterio, es control: sin él, una
                        candidata que bloquease todo el espectro sacaría
                        cubierto = 1,00 y parecería perfecta.

    Asimétrica a propósito, igual que `comparar_c`: se recorre lo pedido.
    """
    A = intervalos(c_obj)
    B = intervalos(c_cand)
    la = _union_longitud(A)
    if la <= _EPS_UNION:
        # Sin nada pedido no hay cobertura que medir.
        return {'cubierto': np.nan, 'gaps_servidos': np.nan, 'exceso': np.nan,
                'n_gaps_error': len(B)}

    comun = _union_longitud(_interseccion(A, B))
    servido = np.array([
        _union_longitud(_interseccion(a[None, :], B)) / max(a[1] - a[0], _EPS_UNION)
        for a in A])

    return {'cubierto': float(comun / la),
            'gaps_servidos': float((servido >= min_servido).mean()),
            'exceso': float((_union_longitud(B) - comun) / la),
            'n_gaps_error': len(B) - len(A)}


# ---------------------------------------------------------------------
#  JUBILADO — el IoU de v0–v2. Se conserva para releer sus números.
# ---------------------------------------------------------------------

def comparar_c(c_obj: np.ndarray, c_cand: np.ndarray) -> dict:
    """Cuánto del comportamiento PEDIDO aparece en el OBTENIDO.

    Devuelve:
        iou_principal  solape del gap más ancho pedido con su mejor pareja.
                       Es lo que pediría un cliente real: "quiero un gap ahí".
        iou_medio      solape promediado sobre TODOS los gaps pedidos. Los que
                       no encuentran pareja cuentan 0, así que penaliza tanto
                       fallar un gap como no producirlo.
        n_gaps_error   diferencia en número de gaps (con signo: + = sobran).

    Asimétrica a propósito: se recorre lo pedido, no lo obtenido. Producir
    gaps de más no se castiga aquí — eso lo recoge `n_gaps_error`.
    """
    A = intervalos(c_obj)
    B = intervalos(c_cand)
    if len(A) == 0:
        return {'iou_principal': np.nan, 'iou_medio': np.nan,
                'n_gaps_error': len(B)}

    ious = np.zeros(len(A))
    for i, _, v in emparejar(iou_intervalos(A, B)):
        ious[i] = v

    principal = int(np.argmax(A[:, 1] - A[:, 0]))
    return {'iou_principal': float(ious[principal]),
            'iou_medio': float(ious.mean()),
            'n_gaps_error': len(B) - len(A)}


def rankear(c_obj: np.ndarray, c_pred: np.ndarray) -> np.ndarray:
    """(45,) x (N,45) -> indices que ordenan las candidatas, de mejor a peor.

    P7 — SE RANKEA CON LA MISMA METRICA CON LA QUE SE EVALUA. Hasta v3 el
    ranking era `norm(c_pred - c_obj)` sobre los 45 numeros crudos, que suma
    presencia (0/1), frecuencia (~1,2) y anchura (~0,05) como si fueran
    comparables. Lo que se evaluaba y lo que se entregaba no eran la misma
    metrica, y la que veia el cliente era la peor de las dos.

    Ordena únicamente por `cubierto` descendente. En empates conserva el orden
    de generación. El exceso se calcula para informar, pero v6 no lo utiliza
    para tomar decisiones.
    """
    q = [cubrimiento(c_obj, c) for c in np.asarray(c_pred)]
    cub = np.array([0.0 if np.isnan(x['cubierto']) else x['cubierto'] for x in q])
    return np.argsort(-cub, kind='stable')


def cobertura(c_obj: np.ndarray, c_cand: np.ndarray, umbral: float = 0.5,
              orden: np.ndarray | None = None) -> dict:
    """La métrica de decisión de v2.

    c_obj  : (M, 45)      objetivos pedidos
    c_cand : (M, N, 45)   c predicho de las N candidatas de cada objetivo
    orden  : (M, N)       índices que ordenan las candidatas de mejor a peor
                          según el ranking del pipeline. Opcional.

    Se reportan dos niveles, y la diferencia entre ellos importa:

        cobertura@N   ¿alguna de las N candidatas cumple?  -> ¿PUEDE el generador?
        cobertura@1   ¿cumple la que el ranking pone primera? -> ¿la ENCUENTRA?

    Si @N es alta y @1 baja, el generador sirve y el problema está en el
    ranking (es decir, en el forward). Son dos arreglos distintos.
    """
    M, N = c_cand.shape[0], c_cand.shape[1]
    princ = np.full((M, N), np.nan)
    medio = np.full((M, N), np.nan)
    cubre = np.full((M, N), np.nan)
    servi = np.full((M, N), np.nan)
    exces = np.full((M, N), np.nan)
    n_err = np.zeros((M, N))

    for i in range(M):
        for j in range(N):
            r = comparar_c(c_obj[i], c_cand[i, j])
            princ[i, j] = r['iou_principal']
            medio[i, j] = r['iou_medio']
            n_err[i, j] = r['n_gaps_error']
            q = cubrimiento(c_obj[i], c_cand[i, j])
            cubre[i, j] = q['cubierto']
            servi[i, j] = q['gaps_servidos']
            exces[i, j] = q['exceso']

    # Objetivos sin ningún gap: no se puede medir cobertura sobre ellos.
    validos = ~np.isnan(princ[:, 0])
    princ, medio, n_err = princ[validos], medio[validos], n_err[validos]
    cubre, servi, exces = cubre[validos], servi[validos], exces[validos]
    mejor = cubre.argmax(axis=1)            # la mejor candidata SEGUN CUBIERTO
    fila_m = np.arange(len(mejor))

    out = {
        'n_objetivos': int(validos.sum()),
        'n_candidatas': int(N),
        'umbral': float(umbral),
        'iou_principal_mejor': float(np.mean(princ.max(axis=1))),
        'iou_medio_mejor': float(np.mean(medio.max(axis=1))),
        'cobertura_principal': float(np.mean(princ.max(axis=1) > umbral)),
        'cobertura_completa': float(np.mean(medio.max(axis=1) > umbral)),
        'iou_principal_mediana_cand': float(np.mean(np.median(princ, axis=1))),
        'n_gaps_error_abs': float(np.mean(np.abs(n_err))),
        # --- la metrica de decision de v4 ---
        'cubierto_mejor': float(np.mean(cubre.max(axis=1))),
        'cobertura_cubierto': float(np.mean(cubre.max(axis=1) > umbral)),
        'cubierto_mediana_cand': float(np.mean(np.median(cubre, axis=1))),
        # los dos acompanantes, EVALUADOS EN LA MISMA CANDIDATA que maximiza
        # `cubierto` — mezclar la mejor de cada metrica describiria a una
        # candidata que no existe
        'gaps_servidos_mejor': float(np.mean(servi[fila_m, mejor])),
        'exceso_mejor': float(np.mean(exces[fila_m, mejor])),
    }

    if orden is not None:
        primera = orden[validos][:, 0]
        fila = np.arange(len(primera))
        out['cobertura_principal_top1'] = float(
            np.mean(princ[fila, primera] > umbral))
        out['iou_principal_top1'] = float(np.mean(princ[fila, primera]))
        out['cubierto_top1'] = float(np.mean(cubre[fila, primera]))
        out['cobertura_cubierto_top1'] = float(np.mean(cubre[fila, primera] > umbral))
        out['gaps_servidos_top1'] = float(np.mean(servi[fila, primera]))
        out['exceso_top1'] = float(np.mean(exces[fila, primera]))

    # Valores por objetivo, para poder contrastar contra la baraja de forma
    # PAREADA (mismo objetivo en ambas condiciones). Sin esto solo se pueden
    # comparar medias, y una diferencia de 2 puntos no se distingue del ruido.
    out['_raw'] = {'iou_max': princ.max(axis=1),
                   'cumple': princ.max(axis=1) > umbral,
                   'cubierto_max': cubre.max(axis=1),
                   'cumple_cubierto': cubre.max(axis=1) > umbral}
    return out


# =====================================================================
#  2 · PLAUSIBILIDAD
# =====================================================================
#
#  NO es validez. La validez de verdad es "¿se puede mallar y simular?", y esa
#  pregunta la responde el mallador que todavía no existe. Esto mide parecido
#  estadístico con las geometrías reales, y conviene llamarlo por su nombre.
#
#  Los umbrales salen del percentil 2,5-97,5 de las 10.999 reales, no de un
#  criterio inventado. En particular NO se exige "una sola componente conexa":
#  solo el 83 % de las geometrías reales la tienen, y el 17 % restante se
#  simula perfectamente.


def _iou_lote(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    inter = np.logical_and(a, b).sum(axis=(1, 2))
    union = np.logical_or(a, b).sum(axis=(1, 2))
    return inter / np.maximum(union, 1)


CLAVES_PLAUSIBILIDAD = ('componentes', 'vol_fraction', 'simetria_max')


def propiedades(masks: np.ndarray) -> dict:
    """Propiedades estructurales de un lote de máscaras (K, 128, 128).

    Medidas geométricas exactas, ninguna necesita simular nada. Se retiró la
    periodicidad de bordes: no discrimina (candidatas 94 % vs reales 78,6 %).
    """
    m = np.asarray(masks) > 0.5

    n_comp = np.empty(len(m), dtype=np.float32)
    for i in range(len(m)):
        n_comp[i] = label(m[i])[1]

    sim_max = np.stack([
        _iou_lote(m, m[:, :, ::-1]),                  # reflexión vertical
        _iou_lote(m, m[:, ::-1, :]),                  # reflexión horizontal
        _iou_lote(m, np.rot90(m, k=2, axes=(1, 2))),  # rotación 180
    ]).max(axis=0)

    return {'componentes': n_comp,
            'vol_fraction': m.mean(axis=(1, 2)).astype(np.float32),
            'simetria_max': sim_max.astype(np.float32)}


def rangos_referencia(props: dict, q: float = 2.5) -> dict:
    """Rango del (100 - 2q) % central de cada propiedad. Los umbrales, sacados
    de los datos.
    """
    return {k: [float(np.percentile(props[k], q)),
                float(np.percentile(props[k], 100 - q))]
            for k in CLAVES_PLAUSIBILIDAD}


def plausibilidad(props: dict, rangos: dict) -> dict:
    """% de muestras dentro de TODOS los rangos, más el desglose por criterio.

    Aplicado a las propias geometrías reales debe rondar el 86-90 % (0,95 por
    criterio, correlacionados entre sí). Ese es el control de que la métrica
    está bien construida.
    """
    dentro = {}
    for k in CLAVES_PLAUSIBILIDAD:
        lo, hi = rangos[k]
        dentro[k] = (props[k] >= lo) & (props[k] <= hi)
    todos = np.logical_and.reduce(list(dentro.values()))
    out = {'plausibles': float(todos.mean()), 'n': int(len(todos))}
    out.update({f'ok_{k}': float(v.mean()) for k, v in dentro.items()})
    out.update({f'mediana_{k}': float(np.median(props[k]))
                for k in CLAVES_PLAUSIBILIDAD})
    return out


# =====================================================================
#  3 · DIVERSIDAD
# =====================================================================


def diversidad(masks: np.ndarray, max_pares: int = 28,
               seed: int = 0) -> float:
    """1 - IoU medio entre pares de candidatas del MISMO objetivo.

    masks: (M, N, H, W). Si N es grande se muestrean `max_pares` pares por
    objetivo en vez de los N(N-1)/2, que crecen cuadráticamente.

    Guardarraíl contra el colapso: al subir beta el decoder puede acabar
    generando siempre lo mismo, y entonces la cobertura sube por el motivo
    equivocado (una única geometría que casualmente encaja en muchos objetivos).
    """
    M, N = masks.shape[0], masks.shape[1]
    if N < 2:
        return float('nan')
    rng = np.random.default_rng(seed)
    todos = [(i, j) for i in range(N) for j in range(i + 1, N)]
    if len(todos) > max_pares:
        todos = [todos[k] for k in rng.choice(len(todos), max_pares, replace=False)]

    ious = []
    for i, j in todos:
        ious.append(_iou_lote(masks[:, i] > 0.5, masks[:, j] > 0.5))
    return float(1.0 - np.mean(ious))


# =====================================================================
#  Baselines
# =====================================================================


def vecino_mas_cercano(c_obj: np.ndarray, c_pool: np.ndarray,
                       bloque: int = 256) -> np.ndarray:
    """Índice en `c_pool` del vector más cercano a cada `c_obj` (euclídea).

    Es el baseline duro: buscar en el catálogo en vez de generar. Devuelve una
    geometría REAL cuyo comportamiento se conoce exactamente, sin pasar por el
    forward — así que es un baseline optimista a propósito. Si el cVAE no lo
    bate, el generador no aporta nada sobre consultar el dataset.
    """
    idx = np.empty(len(c_obj), dtype=np.int64)
    for i in range(0, len(c_obj), bloque):
        d = np.linalg.norm(c_obj[i:i + bloque, None, :] - c_pool[None, :, :], axis=2)
        idx[i:i + bloque] = d.argmin(axis=1)
    return idx
