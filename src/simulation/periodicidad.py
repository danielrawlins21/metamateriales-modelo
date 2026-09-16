"""
Emparejado de nodos periódicos — la pieza que faltaba hacer bien.

Por qué existe este módulo
--------------------------
El solver de Juan Carlos NO calcula qué nodo del borde de la celda es pareja de
cuál: lo lee de la malla (`periodicSourceNodes` / `periodicImageNodes`). Al
portarlo a Python, los `.pkl` de Hendriks no traían esa lista a nivel de nodo, y
se escribió `bloch_sh.find_periodic_pairs` para deducirla mirando la geometría.

Esa función supone que la celda es un **paralelogramo anclado en el origen**:
busca nodos con coordenada fraccionaria ~0 y ~1. Cuatro grupos wallpaper no lo
cumplen — `cmm` y `p6m` tienen la celda desplazada, y `p3` y `p3m1` la tienen
**hexagonal**, con 6 lados en 3 parejas, porque así la declara
`helper_funcs.wallpaper_groups[g]['unit cell shape']`.

En esos casos no encuentra ninguna pareja, devuelve listas vacías, y la
reducción de Bloch no impone NINGUNA restricción: se simula un trozo de material
suelto en el vacío en vez de un medio periódico. Sin error, sin aviso. Afectó a
2348 de 10 159 geometrías (23,1 %), y se reconocen porque el 78 % de ellas sale
con 10 gaps frente al 2,7 % de las sanas.

Qué hace en su lugar
--------------------
Leer las parejas del propio `.pkl`. `uc['bounds']` da los lados de la celda y
`uc['linked_bounds']` cuál va con cuál. Verificado sobre los 6 grupos probados,
hexágonos incluidos: los nodos FEM caen exactamente sobre esos lados y emparejan
al 100 % con la traslación declarada (p3: 13/13 en las tres parejas).

La representación es un único formato para todos los casos —una lista de
`(a, b, T)` con `p[b] = p[a] + T`— en vez de `(pairs_a1, pairs_a2, corners)`.
Eso es lo que permite tratar el hexágono sin casos especiales: da igual que haya
2 pares de lados o 3, y las esquinas salen solas porque un nodo de esquina
aparece en varios enlaces y las órbitas se funden.
"""

from __future__ import annotations

import numpy as np


# ── construir los enlaces ────────────────────────────────────────────────────

def _nodos_en_segmento(P, p0, d, tol=1e-7):
    """Índices de los nodos que caen sobre el segmento p0 -> p0+d."""
    L = float(np.linalg.norm(d))
    if L < 1e-12:
        return np.array([], dtype=int)
    u = d / L
    n = np.array([-u[1], u[0]])
    v = P - p0
    s = v @ u
    h = np.abs(v @ n)
    return np.where((h < tol) & (s > -tol) & (s < L + tol))[0]


def enlaces_desde_uc(p_active, uc, tol=1e-7):
    """Enlaces `(a, b, T)` a partir de los lados que declara Hendriks.

    Devuelve None si el `.pkl` no trae `bounds`/`linked_bounds` (los de Hendriks
    no los traen; los generados sí).
    """
    if 'bounds' not in uc or 'linked_bounds' not in uc:
        return None

    P = np.asarray(p_active, dtype=np.float64)
    bounds = uc['bounds']
    enlaces = []
    for (i, j, signo) in uc['linked_bounds']:
        pi = np.asarray(bounds[i][0], dtype=np.float64)
        di = np.asarray(bounds[i][1], dtype=np.float64)
        pj = np.asarray(bounds[j][0], dtype=np.float64)
        dj = np.asarray(bounds[j][1], dtype=np.float64)
        # el lado pareja se recorre al revés cuando el signo es -1
        T = (pj + dj) - pi if signo < 0 else pj - pi

        na = _nodos_en_segmento(P, pi, di, tol)
        nb = _nodos_en_segmento(P, pj, dj, tol)
        if not len(na) or not len(nb):
            continue
        objetivo = P[nb]
        for a in na:
            d = np.linalg.norm(objetivo - (P[a] + T), axis=1)
            k = int(d.argmin())
            if d[k] < tol:
                enlaces.append((int(a), int(nb[k]), T))
    return enlaces


def enlaces_desde_geometria(p_active, lattice_vectors, tol=1e-6):
    """Enlaces por el método antiguo, para las mallas sin `uc['bounds']`.

    Sigue suponiendo paralelogramo anclado en el origen. Solo se usa en las 180
    de Hendriks que son celda primitiva, donde se verificó que acierta al 100 %.
    """
    # find_periodic_pairs viene de bloch_sh.py, copiada abajo.

    lv = np.asarray(lattice_vectors, dtype=np.float64)
    a1, a2, esquinas = find_periodic_pairs(p_active, lv, tol=tol)
    enlaces = [(int(i), int(j), lv[0]) for i, j in a1]
    enlaces += [(int(i), int(j), lv[1]) for i, j in a2]

    # las esquinas: todas equivalentes a la primera, con su traslación
    despl = [None, lv[0], lv[1], lv[0] + lv[1]]
    maestro = esquinas[0][0] if esquinas and esquinas[0] else None
    if maestro is not None:
        for grupo, T in zip(esquinas[1:], despl[1:]):
            for idx in grupo:
                enlaces.append((int(maestro), int(idx), T))
    return enlaces


# ── órbitas y verificación ───────────────────────────────────────────────────

def orbitas(enlaces, tol=1e-7):
    """Agrupa los nodos enlazados y calcula el desplazamiento desde su maestro.

    Devuelve `{nodo: (maestro, desplazamiento)}`. Un nodo de esquina se alcanza
    por varios caminos; que todos den el MISMO desplazamiento es justamente la
    condición que hay que verificar, y se comprueba aquí.
    """
    ady = {}
    for a, b, T in enlaces:
        T = np.asarray(T, dtype=np.float64)
        ady.setdefault(a, []).append((b, T))
        ady.setdefault(b, []).append((a, -T))

    asignado, inconsistencias = {}, []
    for semilla in sorted(ady):
        if semilla in asignado:
            continue
        asignado[semilla] = (semilla, np.zeros(2))
        pila = [semilla]
        while pila:
            n = pila.pop()
            maestro, dn = asignado[n]
            for v, T in ady[n]:
                nuevo = dn + T
                if v not in asignado:
                    asignado[v] = (maestro, nuevo)
                    pila.append(v)
                elif np.abs(asignado[v][1] - nuevo).max() > tol:
                    inconsistencias.append((n, v))
    return asignado, inconsistencias


def verificar(p_active, enlaces, lattice_vectors, tol=1e-6):
    """Comprueba que los enlaces son geométricamente correctos.

    Esto es lo que nunca se llamó en producción. `verify_periodic_pairs` existía
    y hacía casi esto, pero solo se invocaba desde una función de demo con
    gráficas, así que los 2348 casos rotos pasaron sin que nadie se enterara.
    """
    P = np.asarray(p_active, dtype=np.float64)
    lv = np.asarray(lattice_vectors, dtype=np.float64)
    A = np.column_stack([lv[0], lv[1]])
    problemas = []

    if not enlaces:
        problemas.append('sin_enlaces')
        return {'ok': False, 'problemas': problemas, 'n_enlaces': 0,
                'n_orbitas': 0, 'error_max': np.inf}

    err = 0.0
    for a, b, T in enlaces:
        err = max(err, float(np.abs(P[b] - P[a] - T).max()))
        c = np.linalg.solve(A, np.asarray(T, dtype=np.float64))
        if np.abs(c - np.round(c)).max() > 1e-6:
            problemas.append(f'traslacion_no_es_de_red:{np.round(c, 3)}')
    if err > tol:
        problemas.append(f'error_geometrico:{err:.2e}')

    asignado, inconsistencias = orbitas(enlaces)
    if inconsistencias:
        problemas.append(f'orbitas_inconsistentes:{len(inconsistencias)}')

    maestros = {m for m, _ in asignado.values()}
    return {
        'ok': not problemas,
        'problemas': sorted(set(problemas)),
        'n_enlaces': len(enlaces),
        'n_nodos_ligados': len(asignado),
        'n_orbitas': len(maestros),
        'error_max': err,
    }



# ── emparejado por coordenadas fraccionarias (copiado de bloch_sh.py) ──────

def find_periodic_pairs(p_active, lattice_vectors, tol=1e-6):
    """Identifica los pares de nodos periodicos en los bordes opuestos.

    Estrategia: coordenadas fraccionarias
    --------------------------------------
    Cualquier punto de la celda se puede escribir como:
        p = s*a1 + t*a2   con  s, t en [0, 1]

    En estas coordenadas, identificar bordes es inmediato:
        t ~ 0  ->  borde inferior (bottom)
        t ~ 1  ->  borde superior (top)     <- pareja de bottom, offset = a2
        s ~ 0  ->  borde izquierdo (left)
        s ~ 1  ->  borde derecho (right)    <- pareja de left, offset = a1

    Las 4 esquinas (combinaciones s~0/1, t~0/1) son todas equivalentes
    entre si y se tratan por separado.

    Parametros
    ----------
    p_active        : (N_active, 2)  coordenadas de los nodos activos
    lattice_vectors : (2, 2)         fila 0 = a1, fila 1 = a2
    tol             : tolerancia para considerar un nodo "en el borde"

    Devuelve
    --------
    pairs_a1 : lista de (i, j)  ->  p[j] = p[i] + a1  (left-right)
    pairs_a2 : lista de (i, j)  ->  p[j] = p[i] + a2  (bottom-top)
    corner_groups : lista de listas de indices, cada grupo = esquina equivalente
    """
    from scipy.spatial import KDTree

    a1 = lattice_vectors[0]
    a2 = lattice_vectors[1]

    # Matriz de cambio de base: p = A @ [s, t]^T  =>  [s,t] = A^-1 @ p
    A    = np.column_stack([a1, a2])   # columnas = a1, a2
    Ainv = np.linalg.inv(A)
    frac = (Ainv @ p_active.T).T      # (N_active, 2) coordenadas (s, t)

    # Clasificar nodos por borde
    on_bottom = frac[:, 1] < tol
    on_top    = frac[:, 1] > 1 - tol
    on_left   = frac[:, 0] < tol
    on_right  = frac[:, 0] > 1 - tol

    # Esquinas: en dos bordes a la vez
    is_corner = (on_bottom | on_top) & (on_left | on_right)

    # Nodos de cada borde excluyendo esquinas
    bot_idx = np.where(on_bottom & ~is_corner)[0]
    top_idx = np.where(on_top    & ~is_corner)[0]
    lft_idx = np.where(on_left   & ~is_corner)[0]
    rgt_idx = np.where(on_right  & ~is_corner)[0]

    # --- Emparejar bottom-top por coordenada s (misma posicion en a1) ---
    pairs_a2 = []
    if len(bot_idx) > 0 and len(top_idx) > 0:
        # KDTree sobre la coordenada s del borde top
        tree = KDTree(frac[top_idx, 0:1])
        dists, hits = tree.query(frac[bot_idx, 0:1], k=1)
        for k, (dist, hit) in enumerate(zip(dists, hits)):
            if dist < tol:
                pairs_a2.append((bot_idx[k], top_idx[hit]))

    # --- Emparejar left-right por coordenada t (misma posicion en a2) ---
    pairs_a1 = []
    if len(lft_idx) > 0 and len(rgt_idx) > 0:
        tree = KDTree(frac[rgt_idx, 1:2])
        dists, hits = tree.query(frac[lft_idx, 1:2], k=1)
        for k, (dist, hit) in enumerate(zip(dists, hits)):
            if dist < tol:
                pairs_a1.append((lft_idx[k], rgt_idx[hit]))

    # --- Grupos de esquinas ---
    # Las 4 esquinas son equivalentes: (0,0), (1,0), (0,1), (1,1)
    corner_groups = []
    corner_defs = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for (s0, t0) in corner_defs:
        mask = (np.abs(frac[:, 0] - s0) < tol) & (np.abs(frac[:, 1] - t0) < tol)
        idxs = np.where(mask)[0].tolist()
        corner_groups.append(idxs)

    return pairs_a1, pairs_a2, corner_groups


