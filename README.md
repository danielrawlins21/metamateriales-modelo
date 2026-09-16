# Diseño inverso de metamateriales con cVAE y Forward de cuantiles

Modelo con pesos abiertos para el diseño inverso de celdas unidad 2D de
metamateriales mecánicos con un *band gap* dado. Pides una banda prohibida
(frecuencia central y anchura) y el modelo propone geometrías que deberían
abrirla, ordenadas por la cobertura que predice.

Es la versión `v6-cobertura-calibrada` del modelo del prototipo. Este
repositorio contiene **la parte de aprendizaje automático**: generación,
ranking y filtro geométrico. La verificación por elementos finitos (FEM), la
plataforma web y el despliegue no se publican.

## Cómo funciona

```
encargo (f, Δf) ──► cVAE ──► 210 candidatas ──► Forward de cuantiles ──► ranking ──► filtro ──► 5 geometrías
                    7 redes × 15 colocaciones × 2      predice 15 gaps        cobertura     conexa y
                                                     con incertidumbre       descendente    mallable
```

1. **Encargo.** Un único gap `frecuencia:anchura`, en frecuencia normalizada, dentro
   de `Ω = [0.0443, 2.544]` y con anchura mayor que `0.01`.
2. **Generación (cVAE).** Un autoencoder variacional condicional, que parte de un
   encoder CNN congelado, genera máscaras binarias de celda. Se muestrea
   `7 redes de Bravais × 15 colocaciones del gap × 2 muestras = 210`
   candidatas. Checkpoint: época 60, elegida por cobertura.
3. **Ranking (Forward de cuantiles).** Un modelo directo predice, para cada
   imagen, un conjunto de hasta 15 gaps con sus cuantiles. Las candidatas se
   ordenan por la **cobertura predicha** del gap pedido.
4. **Filtro.** Se recorre el ranking y se descartan las celdas que no son
   conexas bajo periodicidad o que no se pueden mallar con condiciones
   periódicas coherentes (Gmsh). Se paran las 5 primeras válidas.

El Forward es muy bueno **filtrando y ordenando**: casi todas las candidatas
del top pasan su criterio. No sustituye a la simulación FEM para confirmar el
gap: esa comprobación se hacía fuera de este repositorio.

## Contenido

| Ruta | Qué es |
|---|---|
| `pipelines-completo/v6-cobertura-calibrada/artifacts/cvae/best_masking.pt` | Pesos del cVAE (27 MB) |
| `pipelines-completo/v6-cobertura-calibrada/artifacts/forward_cuantiles/modelo_cuantiles.pt` | Pesos del Forward de cuantiles (22 MB) |
| `pipelines-completo/v6-cobertura-calibrada/config.yaml` | Configuración efectiva con la que se entrenó y se infiere |
| `pipelines-completo/v6-cobertura-calibrada/src/05_design.py` | Script de inferencia: generar, rankear y filtrar |
| `pipelines-completo/v6-cobertura-calibrada/src/arch/` | Arquitecturas (cVAE, autoencoder, forwards) |
| `pipelines-completo/v6-cobertura-calibrada/src/` (resto) | Condición, métricas, conectividad, contornos y mallador |
| `pipelines-completo/v6-cobertura-calibrada/recursos/redes_canonicas.json` | Vectores de red de las 7 formas de celda |
| `src/simulation/periodicidad.py` | Solo la comprobación geométrica de nodos periódicos, sin el solver |
| `ejemplo/referencia_1p2_0p06/` | Salida de referencia para comprobar la reproducción |

No se incluyen el dataset, los scripts de entrenamiento ni el solver FEM. Las
secciones de entrenamiento de `config.yaml` y sus rutas a `../v4-limpio` se
conservan como registro de cómo se obtuvieron los pesos; la inferencia no las
usa.

## Reproducir

Probado en Ubuntu 24.04 (WSL) con Python 3.12 en CPU. Una ejecución completa
tarda unos 30 segundos.

### 1. Dependencias del sistema (Gmsh)

Sin estas librerías el mallado falla con
`OSError: libGLU.so.1` y todas las candidatas salen `no_mallable`.

```bash
sudo apt-get install -y libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1
```

### 2. Entorno de Python

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # o cu128 con GPU
pip install -r requirements.txt
```

### 3. Ejecutar el encargo de referencia

Desde la raíz del repositorio:

```bash
python pipelines-completo/v6-cobertura-calibrada/src/05_design.py \
  --gaps 1.2:0.06 --nombre referencia --seed 42
```

Salida esperada:

```
Combinaciones     : 105
Candidatas brutas : 210
Candidatas únicas : 210
Revisadas         : 37
No conexas        : 32
Malladas          : 5
Mallables elegidas: 5/5
Ranks elegidos    : 3, 17, 18, 29, 37
Estado            : completado
```

Los resultados quedan en
`pipelines-completo/v6-cobertura-calibrada/artifacts/design/referencia/`:

| Fichero | Contenido |
|---|---|
| `ranking.csv` | Las 210 candidatas ordenadas, con cobertura, exceso, incertidumbre y estado de mallado |
| `generacion.npz` | Máscaras, cuantiles predichos y metadatos de todas las candidatas |
| `candidatas_mallables.npy` | Las 5 máscaras seleccionadas |
| `candidatas.png` | Figura de las candidatas elegidas |
| `resultados.json` | Resumen de la ejecución |
| `objetivo.json` | Encargo normalizado |

### 4. Comprobar contra la referencia

```bash
diff <(cut -d, -f1-4,14 pipelines-completo/v6-cobertura-calibrada/artifacts/design/referencia/ranking.csv) \
     <(cut -d, -f1-4,14 ejemplo/referencia_1p2_0p06/ranking.csv)
```

La referencia se generó con GPU. En CPU la cobertura predicha puede cambiar
a partir del sexto decimal, lo que puede reordenar candidatas empatadas en
la cola del ranking, pero las 5 geometrías seleccionadas coinciden.

### Opciones

| Argumento | Por defecto | Significado |
|---|---|---|
| `--gaps` | obligatorio | Gap pedido como `frecuencia:anchura` |
| `--nombre` | automático | Carpeta de salida |
| `--seed` | `config.yaml` | Semilla |
| `--muestras-por-combinacion` | `config.yaml` | Muestras por red y colocación |
| `--top-k` | `config.yaml` | Geometrías válidas a seleccionar |
| `--batch-size` | `config.yaml` | Tamaño de lote en la generación |

El dispositivo (`cpu` / `cuda`) se elige en `device` dentro de `config.yaml`.

## Cargar los pesos desde tu código

Los checkpoints se guardaron con objetos de Python además del `state_dict`,
así que hay que cargarlos con `weights_only=False`. Hazlo solo con ficheros
de este repositorio.

```python
import torch
ck = torch.load("pipelines-completo/v6-cobertura-calibrada/artifacts/cvae/best_masking.pt",
                map_location="cpu", weights_only=False)
print(ck["epoch"], ck.keys())
```

La forma completa de montar los modelos está en `cargar_modelos()`
(`src/05_design.py`) y en `cargar_juez()` (`src/juez.py`).

## Autoría y licencia

Modelo desarrollado en el Trabajo Fin de Máster por Marcos (cVAE y pipeline de
diseño), Juan Carlos (Forward de cuantiles) y Daniel Rawlins (integración y
publicación).

Código y pesos bajo licencia [Apache 2.0](LICENSE).
