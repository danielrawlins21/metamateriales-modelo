"""Fase 1 del mallador — imagen binaria 128x128 -> polígonos cerrados.

El primer tramo del puente que falta:

    máscara 128x128  ──►  contornos  ──►  polígonos válidos para gmsh

TODO OCURRE EN COORDENADAS FRACCIONARIAS
----------------------------------------
La imagen cubre exactamente una celda, así que en coordenadas (s, t) la celda
**es el cuadrado [0,1]²** — da igual que la celda física sea un paralelogramo
o un hexágono. La transformación a coordenadas físicas (`a_fisicas`) se aplica
al final, y convierte el cuadrado en el paralelogramo correspondiente.

Eso simplifica el problema: recortar contra el borde de la celda es recortar
contra un cuadrado, no contra un paralelogramo.

QUÉ ES UN POLÍGONO VÁLIDO
-------------------------
Para que gmsh pueda mallar el dominio, cada polígono tiene que estar cerrado,
no cortarse a sí mismo y encerrar área. Además hace falta saber **quién es
agujero de quién**: un contorno dentro de otro define un hueco, y gmsh lo
necesita como bucle interior con orientación contraria.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import contourpy

# Un contorno de menos área que esto es ruido de rasterización, no geometría.
# 8 píxeles de 16384 = 0,05 % de la celda.
AREA_MIN_PIXELES = 8.0

# Tolerancia de Douglas-Peucker, en píxeles. 1,0 baja de ~965 puntos a ~16 por
# geometría sin mover el área. Es un PARÁMETRO A BARRER en la Fase 3: los band
# gaps dependen de detalles finos, y aquí no se puede saber cuánta
# simplificación aguantan.
TOL_SIMPLIFICAR = 1.0


@dataclass
class Poligono:
    """Un contorno cerrado en coordenadas fraccionarias [0,1]²."""
    puntos: np.ndarray          # (N, 2), sin repetir el primero al final
    es_agujero: bool
    padre: int | None           # índice del polígono que lo contiene

    @property
    def area(self) -> float:
        """Área con signo por la fórmula del zapato. Positiva = antihorario."""
        x, y = self.puntos[:, 0], self.puntos[:, 1]
        return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# ---------------------------------------------------------------------
#  Geometría auxiliar
# ---------------------------------------------------------------------
def _area_con_signo(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def simplificar(p: np.ndarray, tol: float,
                fijos: np.ndarray | None = None) -> np.ndarray:
    """Douglas-Peucker iterativo sobre un contorno CERRADO.

    Iterativo y no recursivo a propósito: con 965 puntos la versión recursiva
    puede agotar la pila, y aquí entran contornos de ese tamaño.

    `fijos` marca puntos que NO se pueden tirar y que además parten el recorrido
    en tramos independientes. Lo usa `extraer_periodico` para clavar los cruces
    con el borde de la celda: sin eso, DP simplifica cada copia del teselado por
    su cuenta y los bordes opuestos dejan de ser trasladados exactos, que es
    justo lo que necesitan las condiciones de Bloch.
    """
    n = len(p)
    if n <= 4:
        return p
    conservar = np.zeros(n, dtype=bool)
    conservar[0] = conservar[-1] = True
    if fijos is not None:
        conservar |= np.asarray(fijos, dtype=bool)
    anclas = np.flatnonzero(conservar)
    pila = [(int(i), int(j)) for i, j in zip(anclas[:-1], anclas[1:])]
    while pila:
        i, j = pila.pop()
        if j <= i + 1:
            continue
        seg = p[j] - p[i]
        norma = np.hypot(*seg)
        if norma < 1e-12:
            d = np.hypot(*(p[i + 1:j] - p[i]).T)
        else:
            d = np.abs(np.cross(seg, p[i + 1:j] - p[i])) / norma
        k = int(d.argmax())
        if d[k] > tol:
            k += i + 1
            conservar[k] = True
            pila += [(i, k), (k, j)]
    return p[conservar]


def simplificar_seguro(p: np.ndarray, tol: float,
                       fijos: np.ndarray | None = None) -> np.ndarray:
    """Douglas-Peucker que no se corta a sí mismo.

    DP sobre un contorno CERRADO puede generar auto-intersecciones: al saltarse
    puntos, dos tramos alejados en el recorrido pueden acabar cruzándose. Es un
    efecto conocido y la solución práctica es bajar la tolerancia hasta que
    desaparezca — en el peor caso se devuelve el contorno sin simplificar, que
    nunca se corta porque viene de marching squares.
    """
    if tol <= 0:
        # Sin simplificar no hay nada que pueda cruzarse: marching squares no
        # devuelve contornos que se corten. Y el chequeo es O(n²) sobre miles
        # de puntos, así que saltárselo no es una optimización menor.
        return simplificar(p, 0.0, fijos)

    for factor in (1.0, 0.5, 0.25, 0.1):
        q = simplificar(p, tol * factor, fijos)
        if len(q) >= 3 and not _se_corta_a_si_mismo(q):
            return q
    return p


def _se_corta_a_si_mismo(p: np.ndarray) -> bool:
    """¿Hay dos aristas no consecutivas que se cruzan?

    O(n²), pero tras simplificar quedan decenas de puntos, no cientos.
    """
    n = len(p)
    if n < 4:
        return False
    a = p
    b = np.roll(p, -1, axis=0)

    def cruzan(p1, p2, p3, p4):
        d1 = np.cross(p4 - p3, p1 - p3)
        d2 = np.cross(p4 - p3, p2 - p3)
        d3 = np.cross(p2 - p1, p3 - p1)
        d4 = np.cross(p2 - p1, p4 - p1)
        return ((d1 * d2 < 0) and (d3 * d4 < 0))

    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:      # aristas adyacentes por el cierre
                continue
            if cruzan(a[i], b[i], a[j], b[j]):
                return True
    return False


def _dentro(punto: np.ndarray, poligono: np.ndarray) -> bool:
    """Ray casting horizontal."""
    x, y = punto
    px, py = poligono[:, 0], poligono[:, 1]
    qx, qy = np.roll(px, -1), np.roll(py, -1)
    cruza = ((py > y) != (qy > y))
    with np.errstate(divide='ignore', invalid='ignore'):
        xint = px + (y - py) * (qx - px) / (qy - py)
    return bool(np.sum(cruza & (x < xint)) % 2 == 1)


# ---------------------------------------------------------------------
#  El paso principal
# ---------------------------------------------------------------------
def _lado_material(p_pix: np.ndarray, m: np.ndarray) -> bool:
    """¿El material queda DENTRO del contorno? Se decide muestreando la máscara.

    Criterio local, no topológico: se toman varias aristas, se sale medio píxel
    hacia cada lado por la normal y se mira qué hay. No necesita saber nada de
    la jerarquía, y por eso no se equivoca cuando la fase exterior es el hueco
    —que en una celda de verdad pasa la mitad de las veces, porque el material
    conecta con las celdas vecinas y lo aislado son los agujeros.

    (En la Fase 1 se había elegido decidirlo por anidamiento. Aquella medida se
    hizo sobre imágenes que no eran la celda, donde el material siempre quedaba
    inscrito con margen y la fase exterior era siempre el hueco. Deja de valer
    en cuanto la imagen es la celda.)
    """
    H, W = m.shape
    n = len(p_pix)
    dentro = fuera = 0
    for k in range(0, n, max(1, n // 16)):
        a, b = p_pix[k], p_pix[(k + 1) % n]
        medio = (a + b) / 2.0
        d = b - a
        L = np.hypot(*d)
        if L < 1e-9:
            continue
        nor = np.array([-d[1], d[0]]) / L          # normal izquierda
        for signo, contador in ((+1, 'i'), (-1, 'd')):
            q = medio + signo * 0.75 * nor
            j, i = int(round(q[0])), int(round(q[1]))
            if not (0 <= i < H and 0 <= j < W):
                continue
            hay = m[i, j] > 0.5
            # la normal izquierda apunta al interior si el contorno es antihorario
            if signo > 0:
                dentro += hay
            else:
                fuera += hay
    if _area_con_signo(p_pix) < 0:               # horario: la normal se invierte
        dentro, fuera = fuera, dentro
    return dentro >= fuera


def extraer(mask: np.ndarray, tol: float = TOL_SIMPLIFICAR,
            area_min: float = AREA_MIN_PIXELES) -> list[Poligono]:
    """Máscara binaria (H, W) -> lista de polígonos en fraccionarias [0,1]².

    El material es 1. Los contornos salen del nivel 0,5 (marching squares), se
    filtran por área, se simplifican y se ordenan en jerarquía padre/agujero.

    Los contornos que tocan el borde de la imagen se cierran siguiéndolo, que
    es lo correcto: ahí el material continúa en la celda vecina y el borde de
    la imagen ES el borde de la celda.
    """
    m = (np.asarray(mask) > 0.5).astype(np.float64)
    H, W = m.shape

    gen = contourpy.contour_generator(z=m, name='serial', corner_mask=False,
                                      line_type=contourpy.LineType.SeparateCode)
    lineas, _ = gen.lines(0.5)

    brutos = []
    for linea in lineas:
        p = np.asarray(linea, dtype=np.float64)
        if len(p) > 1 and np.allclose(p[0], p[-1]):
            p = p[:-1]                       # el cierre es implícito
        if len(p) < 3 or abs(_area_con_signo(p)) < area_min:
            continue
        p = simplificar_seguro(p, tol)
        if len(p) < 3 or abs(_area_con_signo(p)) < area_min:
            continue
        brutos.append(p)

    # Jerarquía por ANIDAMIENTO: contenido en un número impar de contornos =
    # agujero. Vale porque quien llama garantiza que la fase de fuera es hueco
    # (`extraer_periodico` rodea la imagen con un anillo vacío). Sin esa
    # garantía la regla se invierte en cuanto el material es la fase conexa.
    areas = [abs(_area_con_signo(p)) for p in brutos]
    orden = np.argsort(areas)[::-1]           # de mayor a menor
    polis: list[Poligono] = []
    for i in orden:
        p = brutos[i]
        contenedores = [j for j in orden
                        if j != i and areas[j] > areas[i] and _dentro(p[0], brutos[j])]
        padre_local = min(contenedores, key=lambda j: areas[j]) if contenedores else None
        es_agujero = len(contenedores) % 2 == 1

        # Orientación: exteriores antihorarios, agujeros horarios. gmsh usa el
        # sentido del bucle para saber qué es material y qué es hueco.
        a = _area_con_signo(p)
        if (a < 0) != es_agujero:
            p = p[::-1]

        # A fraccionarias: la imagen cubre la celda, así que (0,0)-(W-1,H-1)
        # mapea a [0,1]². contourpy da (columna, fila) = (x, y).
        frac = np.column_stack([p[:, 0] / (W - 1), p[:, 1] / (H - 1)])
        polis.append(Poligono(puntos=frac, es_agujero=es_agujero, padre=padre_local))

    # Los índices del padre apuntaban a `brutos`; reindexar a `polis`.
    mapa = {int(j): k for k, j in enumerate(orden)}
    for q in polis:
        q.padre = mapa.get(q.padre) if q.padre is not None else None
    return polis


def teselar(mask: np.ndarray, n: int = 3, relleno: int = 1) -> np.ndarray:
    """Replica la máscara n×n y la rodea de hueco.

    Dos detalles, los dos necesarios:

    **El periodo son W-1 columnas, no W.** La máscara tiene la columna 0
    repetida en la W-1 porque ambas son el borde de la celda, que es el mismo
    sitio. Teselar sin quitarla duplicaría una columna por copia.

    **El anillo de hueco.** Marching squares devuelve contornos ABIERTOS donde
    el material toca el borde de la imagen, y cerrarlos traza una cuerda recta
    que puede cruzar media geometría. Con el anillo ningún contorno toca el
    borde, todos salen cerrados, y además la fase de fuera es hueco por
    construcción — que es lo que hace válida la regla de anidamiento.
    """
    m = np.asarray(mask)
    nucleo = m[:-1, :-1]
    return np.pad(np.tile(nucleo, (n, n)), relleno, constant_values=0)


def extraer_periodico(mask: np.ndarray, tol: float = TOL_SIMPLIFICAR,
                      area_min: float = AREA_MIN_PIXELES) -> list[Poligono]:
    """Como `extraer`, pero teselando 3×3 antes. Devuelve fraccionarias en [-1,2]².

    Existe porque `extraer` no sabe tratar el material que llega al borde de la
    celda, y en una imagen que de verdad ES la celda eso pasa siempre: medido
    sobre las 10 159, el 100 % tiene material en el borde. Teselando, ese
    material continúa en la copia vecina y el contorno se cierra solo.

    El recorte a [0,1]² NO se hace aquí: lo hace el mallador con una
    intersección booleana, que sabe partir un polígono en varios trozos y
    reasignar los agujeros. Recortarlo a mano es justo el paso donde se pierde
    la topología.

    La celda original es [0,1]²; alrededor vienen sus ocho vecinas.
    """
    H, W = np.asarray(mask).shape
    lado = W - 1                       # un periodo, en píxeles
    relleno = 1

    # Sin simplificar todavía: hay que insertar los cruces con el borde ANTES,
    # o DP los tira y cada copia del teselado acaba cortando por otro sitio.
    polis = extraer(teselar(mask, 3, relleno), tol=0.0, area_min=area_min)

    # `extraer` normalizó dividiendo por (ancho_teselado - 1). Deshacerlo y
    # recentrar: el origen de la celda del medio cae en el píxel `relleno+lado`.
    ancho = 3 * lado + 2 * relleno
    off = relleno + lado

    salida = []
    for q in polis:
        pts = (q.puntos * (ancho - 1) - off) / lado
        pts, fijos = _insertar_cruces(pts, (0.0, 1.0))
        pts = simplificar_seguro(pts, tol / lado, fijos)   # tol va en píxeles
        if len(pts) < 3:
            continue

        # Filtrar por área OTRA VEZ, ahora sobre el contorno simplificado.
        # Las astillas de medio píxel —el nivel 0,5 entre un píxel de material y
        # uno de hueco— son más estrechas que la tolerancia, así que DP las
        # aplana, y no siempre igual a los dos lados de la celda: se midió una
        # que quedaba con 0,004 de ancho junto al borde izquierdo y con ancho
        # CERO junto al derecho. Eso rompe la periodicidad, que es lo único que
        # esta fase tiene que garantizar.
        if abs(_area_con_signo(pts)) * lado ** 2 < area_min:
            continue

        q.puntos = pts
        salida.append(q)
    return salida


def _insertar_cruces(p: np.ndarray, valores=(0.0, 1.0)) -> tuple:
    """Inserta los puntos donde el contorno cruza s=c o t=c, y los marca fijos.

    Son los únicos puntos que el mallador necesita que coincidan exactamente
    entre bordes opuestos: definen dónde corta el recorte a la celda. El
    contorno de partida SÍ es exactamente periódico (sale de marching squares
    sobre una máscara exactamente periódica), así que clavarlos basta.
    """
    n = len(p)
    puntos, fijos = [], []
    for i in range(n):
        a, b = p[i], p[(i + 1) % n]
        puntos.append(a)
        fijos.append(any(abs(a[e] - c) < 1e-12 for e in (0, 1) for c in valores))
        cortes = []
        for eje in (0, 1):
            for c in valores:
                da, db = a[eje] - c, b[eje] - c
                if da * db < 0:
                    u = da / (da - db)
                    cortes.append((u, a + u * (b - a)))
        for _, q in sorted(cortes, key=lambda z: z[0]):
            puntos.append(q)
            fijos.append(True)
    return np.asarray(puntos), np.asarray(fijos, dtype=bool)


def validar(polis: list[Poligono], mask: np.ndarray | None = None,
            tol_area: float = 0.05) -> dict:
    """Comprueba que el conjunto es mallable por gmsh. Devuelve el diagnóstico.

    Si se pasa `mask`, se exige además que el área neta de los polígonos
    reproduzca la densidad de la imagen dentro de `tol_area`. Es la comprobación
    que de verdad importa: un conjunto puede ser geométricamente impecable y
    describir OTRA geometría — por ejemplo si la jerarquía marca como exterior
    algo que era un hueco. Sin esto, esos casos pasan silenciosamente y se
    descubren al simular, mucho más tarde y mucho más caro.
    """
    problemas = []
    if not polis:
        problemas.append('sin_contornos')
    if not any(not q.es_agujero for q in polis):
        problemas.append('sin_exterior')
    for k, q in enumerate(polis):
        if len(q.puntos) < 3:
            problemas.append(f'{k}:degenerado')
        if abs(q.area) < 1e-9:
            problemas.append(f'{k}:area_nula')
        if _se_corta_a_si_mismo(q.puntos):
            problemas.append(f'{k}:se_corta')

    area_neta = sum(abs(q.area) * (-1 if q.es_agujero else 1) for q in polis)
    if mask is not None:
        densidad = float((np.asarray(mask) > 0.5).mean())
        if abs(area_neta - densidad) > tol_area:
            problemas.append(f'area_no_cuadra:{area_neta:.3f}vs{densidad:.3f}')

    return {
        'valido': not problemas,
        'problemas': problemas,
        'n_poligonos': len(polis),
        'n_agujeros': sum(q.es_agujero for q in polis),
        'n_puntos': sum(len(q.puntos) for q in polis),
        'area_neta': area_neta,
    }


def a_fisicas(polis: list[Poligono], lattice_vectors: np.ndarray) -> list[np.ndarray]:
    """Fraccionarias -> coordenadas físicas. `(s,t) -> s·a1 + t·a2`.

    Es donde el cuadrado [0,1]² se convierte en el paralelogramo real de la
    celda. Hasta aquí nada dependía de la forma de la celda.
    """
    lv = np.asarray(lattice_vectors, dtype=np.float64)
    return [q.puntos @ lv for q in polis]
