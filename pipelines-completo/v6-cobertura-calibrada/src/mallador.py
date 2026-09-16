"""
Fase 2 — contorno → malla FEM periódica.

Convierte los polígonos de `contornos.extraer_periodico` en una malla de
triángulos cuadráticos (T6) con nodos apareados exactamente en bordes opuestos,
que es lo que exigen las condiciones de Bloch y lo que espera
`build_fem_matrices_elastic`.

Se malla en FRACCIONARIAS, no en físicas
----------------------------------------
La celda es el cuadrado unidad [0,1]², y al final se aplica `p = frac @ lv`.
Dos razones:

  - la periodicidad queda alineada con los ejes (izquierda↔derecha es una
    traslación (1,0)), en vez de ser una traslación oblicua por a1;
  - el mapa es LINEAL, así que rectas siguen siendo rectas y el punto medio de
    un T6 sigue siendo el punto medio. No hay que remallar ni recolocar nada.

El precio es que la malla se cizalla al pasar a físicas. Medido sobre el
dataset, el peor caso es la celda hexagonal de 60°, con número de condición
1,732: un triángulo equilátero se convierte en uno de relación de aspecto 1,73.
Aceptable para FEM.

El recorte a la celda
---------------------
Los polígonos vienen del teselado 3×3 y abarcan [-1,2]². El recorte a [0,1]² lo
hace una intersección booleana de OCC, no un recorte a mano: cortar un polígono
con agujeros contra un cuadrado puede partirlo en varios trozos y reasignar qué
agujero pertenece a cuál, y ese es exactamente el paso donde se pierde la
topología si se hace a mano.

Uso
---
    from mallador import mallar
    r = mallar(mask, lattice_vectors, h=0.0344)
    r['p_all'], r['t_all'], r['lattice_vectors']
"""

from __future__ import annotations

import numpy as np

import contornos as C

H_POR_DEFECTO = 0.0344      # arista típica del dataset, en fraccionarias
TIPO_T6 = 9                 # gmsh: triángulo de 2.º orden, 6 nodos


class ErrorMallado(RuntimeError):
    pass


# ── construcción de la geometría ─────────────────────────────────────────────

def _sin_repetidos(p, tol=1e-12):
    """Quita puntos consecutivos coincidentes, incluido el cierre.

    Aparecen porque `_insertar_cruces` puede insertar un cruce que ya era un
    vértice. OCC no acepta una línea de longitud cero.
    """
    p = np.asarray(p, dtype=np.float64)
    d = np.abs(np.diff(np.vstack([p, p[:1]]), axis=0)).max(axis=1)
    return p[d > tol]


def _bucle(occ, puntos, h):
    """Puntos -> curve loop de OCC. Devuelve el tag del bucle."""
    puntos = _sin_repetidos(puntos)
    if len(puntos) < 3:
        raise ErrorMallado('bucle degenerado tras quitar repetidos')
    tags = [occ.addPoint(float(x), float(y), 0.0, h) for x, y in puntos]
    lineas = [occ.addLine(tags[i], tags[(i + 1) % len(tags)]) for i in range(len(tags))]
    return occ.addCurveLoop(lineas)


CUADRADO = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])


def _profundidad(polis):
    """Cuántos contornos contienen a cada polígono. Par = material, impar = hueco."""
    out = []
    for q in polis:
        d, padre, visto = 0, q.padre, set()
        while padre is not None and padre not in visto:
            visto.add(padre)
            d += 1
            padre = polis[padre].padre
        out.append(d)
    return out


def _material_recortado(occ, polis, h):
    """Polígonos -> material dentro de la celda, como dimTags de OCC.

    Se construye con booleanos explícitos nivel a nivel, no con
    `addPlaneSurface([exterior, agujero, ...])`. Ese atajo produce caras
    inválidas: en un caso medido devolvía área 8,51 cuando su contorno exterior
    medía 7,16 — no restaba los agujeros —, y esa cara rota hacía que la
    intersección con la celda saliera vacía sin dar ningún error.

    Recorriendo por profundidad de anidamiento (par añade material, impar lo
    quita) sale bien y además soporta cualquier nivel: material dentro de un
    hueco dentro de material.
    """
    prof = _profundidad(polis)
    niveles = {}
    omitidos = 0
    for k, q in enumerate(polis):
        try:
            bucle = _bucle(occ, q.puntos, h)
        except ErrorMallado:
            # Un contorno que colapsa a menos de 3 puntos al quitar repetidos es
            # una astilla. Tirarlo cambia el área menos que la tolerancia de
            # simplificación, y abortar por él costaba el 4 % de las geometrías.
            omitidos += 1
            continue
        niveles.setdefault(prof[k], []).append(occ.addPlaneSurface([bucle]))

    actual = []
    for d in sorted(niveles):
        caps = [(2, t) for t in niveles[d]]
        if d % 2 == 0:
            actual = caps if not actual else occ.fuse(actual, caps)[0]
        elif actual:
            actual = occ.cut(actual, caps)[0]
    if not actual:
        return [], omitidos

    celda = occ.addPlaneSurface([_bucle(occ, CUADRADO, h)])
    return occ.intersect(actual, [(2, celda)])[0], omitidos


def _relevantes(polis, margen=1e-9):
    """Descarta los polígonos que no tocan [0,1]². Solo es velocidad."""
    fuera = []
    for q in polis:
        p = q.puntos
        if (p[:, 0].max() < -margen or p[:, 0].min() > 1 + margen or
                p[:, 1].max() < -margen or p[:, 1].min() > 1 + margen):
            continue
        fuera.append(q)
    # los índices de `padre` apuntaban a la lista vieja
    mapa = {id(q): i for i, q in enumerate(fuera)}
    viejos = {id(q): q for q in polis}
    for q in fuera:
        if q.padre is not None:
            padre_obj = polis[q.padre]
            q.padre = mapa.get(id(padre_obj))
    return fuera


# ── periodicidad ─────────────────────────────────────────────────────────────

def _curvas_en(gmsh, eje, valor, eps=1e-7):
    """Curvas contenidas en la recta eje=valor del borde de la celda."""
    caja = [-eps, -eps, -eps, 1 + eps, 1 + eps, eps]
    caja[eje] = valor - eps
    caja[eje + 3] = valor + eps
    return [t for d, t in gmsh.model.getEntitiesInBoundingBox(*caja, 1)]


def _emparejar(gmsh, origen, destino, eje, eps=1e-7):
    """Empareja cada curva de `origen` con la de `destino` desplazada una celda.

    Se empareja por caja envolvente en la coordenada libre. Es exacto porque los
    contornos vienen ya trasladados de forma exacta: `extraer_periodico` clava
    los cruces con el borde, y se midió desajuste 0,00e+00 en 150 geometrías.
    """
    def clave(t):
        x0, y0, _, x1, y1, _ = gmsh.model.getBoundingBox(1, t)
        libre = (y0, y1) if eje == 0 else (x0, x1)
        return (round(libre[0], 9), round(libre[1], 9))

    tabla = {}
    for t in destino:
        tabla.setdefault(clave(t), []).append(t)

    pares = []
    for t in origen:
        cands = tabla.get(clave(t))
        if not cands:
            return None
        pares.append((t, cands.pop()))
    if any(v for v in tabla.values()):
        return None
    return pares


def _imponer_periodicidad(gmsh):
    """setPeriodic en los dos pares de bordes. Devuelve el diagnóstico."""
    info = {}
    for eje, nombre, desp in ((0, 'izq_der', (1.0, 0.0)), (1, 'aba_arr', (0.0, 1.0))):
        origen = _curvas_en(gmsh, eje, 0.0)
        destino = _curvas_en(gmsh, eje, 1.0)
        info[f'n_{nombre}'] = (len(origen), len(destino))
        if not origen and not destino:
            info[nombre] = 'sin_curvas'
            continue
        pares = _emparejar(gmsh, origen, destino, eje)
        if pares is None:
            info[nombre] = 'sin_emparejar'
            continue
        T = np.eye(4)
        T[0, 3], T[1, 3] = desp
        gmsh.model.mesh.setPeriodic(1, [d for _, d in pares], [o for o, _ in pares],
                                    T.ravel().tolist())
        info[nombre] = 'ok'
    return info


# ── entrada principal ────────────────────────────────────────────────────────

def mallar(mask, lattice_vectors, h=H_POR_DEFECTO, tol_contorno=C.TOL_SIMPLIFICAR,
           orden=2, verbose=False):
    """Máscara periódica + vectores de red -> malla T6 periódica.

    Devuelve un dict con `p_all` (N,2), `t_all` (M,6), `lattice_vectors` (2,2)
    y `diagnostico`.
    """
    import gmsh

    polis = _relevantes(C.extraer_periodico(mask, tol=tol_contorno))
    if not polis:
        raise ErrorMallado('sin polígonos que toquen la celda')

    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 1 if verbose else 0)
        gmsh.model.add('celda')
        occ = gmsh.model.occ

        trozos, omitidos = _material_recortado(occ, polis, h)
        if not trozos:
            raise ErrorMallado('la intersección con la celda quedó vacía')
        occ.synchronize()

        diag = _imponer_periodicidad(gmsh)

        gmsh.option.setNumber('Mesh.MeshSizeMin', h * 0.5)
        gmsh.option.setNumber('Mesh.MeshSizeMax', h)
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)
        gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
        gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)
        gmsh.model.mesh.generate(2)
        if orden == 2:
            gmsh.model.mesh.setOrder(2)

        tags, coords, _ = gmsh.model.mesh.getNodes()
        xy = np.asarray(coords).reshape(-1, 3)[:, :2]
        indice = {int(t): i for i, t in enumerate(tags)}

        et, en = gmsh.model.mesh.getElementsByType(TIPO_T6 if orden == 2 else 2)
        k = 6 if orden == 2 else 3
        elems = np.array([indice[int(n)] for n in en], dtype=np.int64).reshape(-1, k)
    finally:
        gmsh.finalize()

    if len(elems) == 0:
        raise ErrorMallado('malla vacía')

    lv = np.asarray(lattice_vectors, dtype=np.float64)
    return {
        'p_all': xy @ lv,                 # fraccionarias -> físicas. Lineal.
        'p_frac': xy,
        't_all': elems,
        'lattice_vectors': lv,
        'diagnostico': dict(diag, n_nodos=len(xy), n_elementos=len(elems),
                            n_poligonos=len(polis), n_bucles_omitidos=omitidos),
    }
