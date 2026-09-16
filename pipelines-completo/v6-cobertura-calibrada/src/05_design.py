#!/usr/bin/env python3
"""Fase 5 — generación, ranking rápido y filtro de mallabilidad.

Para un encargo parcial se exploran las siete formas de celda, todas las
colocaciones compatibles en los 15 slots y varias muestras de ``z``. El
Forward de cuantiles ordena las máscaras; después se intenta mallarlas en ese
orden hasta reunir ``top_k`` candidatas válidas. No se ejecuta FEM aquí.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from arch.cnn_autoencoder import Autoencoder
from arch.cvae import ConditionalVAE, FrozenCNNEncoder
from common import (PROJECT_ROOT, load_config, load_json, save_json, set_seed,
                    get_device, version_path)
import condicion
from conectividad import diagnosticar as diagnosticar_conectividad
from juez import cargar_juez, describir
import metricas_cvae as mc


def objetivo_desde_gaps(spec: str, cfg: dict) -> tuple[np.ndarray, dict]:
    """Valida un único ``frecuencia:anchura`` y construye el vector de 45."""
    k_max = int(cfg['target']['k_max'])
    omega_min = float(cfg['target']['omega_min'])
    omega_max = float(cfg['target']['omega_max'])
    gaps = []
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError('Se requiere al menos un gap frecuencia:anchura')
    for i, fragmento in enumerate(spec.split(',')):
        partes = fragmento.strip().split(':')
        if len(partes) != 2:
            raise ValueError(f'No entiendo {fragmento!r}; usa frecuencia:anchura')
        try:
            frecuencia, anchura = map(float, partes)
        except ValueError as exc:
            raise ValueError(f'gap[{i}] no contiene dos números') from exc
        if not math.isfinite(frecuencia) or not math.isfinite(anchura):
            raise ValueError(f'gap[{i}] contiene un valor no finito')
        if anchura <= 0:
            raise ValueError(f'gap[{i}].anchura debe ser positiva')
        inicio, fin = frecuencia - anchura / 2, frecuencia + anchura / 2
        if inicio < omega_min or fin > omega_max:
            raise ValueError(
                f'gap[{i}] ocupa [{inicio:.4f}, {fin:.4f}], fuera de '
                f'Omega=[{omega_min:.4f}, {omega_max:.4f}]')
        gaps.append((frecuencia, anchura, inicio, fin))
    if len(gaps) != 1:
        raise ValueError('Esta versión admite exactamente un gap por encargo')
    gaps.sort(key=lambda g: g[0])
    for anterior, actual in zip(gaps, gaps[1:]):
        if actual[2] < anterior[3]:
            raise ValueError(
                f'Los gaps centrados en {anterior[0]:g} y {actual[0]:g} se solapan')

    vector = np.zeros((k_max, 3), dtype=np.float32)
    for i, (frecuencia, anchura, _, _) in enumerate(gaps):
        vector[i] = (1.0, frecuencia, anchura)
    meta = {
        'tipo': 'parcial',
        'significado': 'solo obliga los gaps listados; del resto no opina',
        'spec': spec,
        'gaps': [
            {'frecuencia_media': f, 'anchura': a, 'inicio': lo, 'fin': hi}
            for f, a, lo, hi in gaps
        ],
    }
    return vector.reshape(-1), meta


def cargar_modelos(cfg: dict, device):
    """Carga el cVAE definitivo y el nuevo Forward congelado."""
    ruta = (version_path(cfg['artifacts']['cvae'], mkdir=False)
            / condicion.nombre_checkpoint(cfg))
    if not ruta.exists():
        raise FileNotFoundError(f'Falta el cVAE entrenado: {ruta}')
    checkpoint = torch.load(ruta, map_location=device, weights_only=False)
    ec = cfg['encoder']
    ae = Autoencoder(latent_dim=ec['latent_dim'], channels=ec['channels'],
                     dropout=ec['dropout'], input_size=ec['img_size'])
    estado = checkpoint['model_state']
    # Con ``cond_dim: crudo`` el codificador es una identidad y no tiene pesos
    # de los que deducir la entrada. La dimensión sale de la configuración
    # efectiva guardada en el propio checkpoint.
    cfg_ck = checkpoint['config']
    cvae_ck = cfg_ck['cvae']
    n_extra = (len(cfg_ck['target']['formas'])
               if cvae_ck.get('usar_forma_celda') else 0)
    target_dim = condicion.dim_condicion(
        bool(cvae_ck.get('condition_masking')), n_extra)
    cvae = ConditionalVAE(
        FrozenCNNEncoder(ae), img_size=ec['img_size'], target_dim=target_dim,
        cond_dim=cvae_ck['cond_dim'],
        decoder_dropout=cvae_ck['decoder_dropout'],
        latent_dim=int(estado['recognition_head.mu.weight'].shape[0]),
    ).to(device)
    cvae.load_state_dict(estado, strict=True)
    cvae.eval()
    juez = cargar_juez(cfg, device, cual='cuantiles')
    return cvae, juez, checkpoint, target_dim


def construir_condiciones(c_obj: np.ndarray, cfg: dict, target_dim: int):
    """Combina siete formas con las ``16-k`` colocaciones del encargo."""
    formas = list(cfg['target']['formas'])
    condiciones, nombres, posiciones = [], [], []
    for indice, forma in enumerate(formas):
        one_hot = np.zeros((1, len(formas)), dtype=np.float32)
        one_hot[0, indice] = 1.0
        variantes = condicion.colocaciones(
            np.hstack([c_obj.reshape(1, -1), one_hot]))
        if variantes.shape[1] != target_dim:
            raise RuntimeError(
                f'La condición tiene {variantes.shape[1]} valores y el cVAE '
                f'espera {target_dim}')
        condiciones.append(variantes)
        nombres.extend([forma] * len(variantes))
        posiciones.extend(range(len(variantes)))
    return (np.vstack(condiciones).astype(np.float32),
            np.asarray(nombres, dtype='U20'),
            np.asarray(posiciones, dtype=np.int16))


@torch.no_grad()
def generar_y_juzgar(c_obj, cfg, cvae, juez, target_dim, device,
                     muestras_por_combinacion: int, batch_size: int):
    condiciones, formas_c, posiciones_c = construir_condiciones(
        c_obj, cfg, target_dim)
    masks_l, formas_l, posiciones_l, muestras_l = [], [], [], []
    img_size = int(cfg['encoder']['img_size'])
    for inicio in range(0, len(condiciones), batch_size):
        fin = min(inicio + batch_size, len(condiciones))
        cb = torch.from_numpy(condiciones[inicio:fin]).to(device)
        generado = cvae.sample(cb, n_samples=muestras_por_combinacion)
        masks_l.append(generado[:, :, 0].cpu().numpy().astype(np.uint8)
                       .reshape(-1, img_size, img_size))
        formas_l.append(np.repeat(formas_c[inicio:fin],
                                  muestras_por_combinacion))
        posiciones_l.append(np.repeat(posiciones_c[inicio:fin],
                                      muestras_por_combinacion))
        muestras_l.append(np.tile(np.arange(muestras_por_combinacion),
                                  fin - inicio))
    masks = np.concatenate(masks_l)
    formas = np.concatenate(formas_l)
    posiciones = np.concatenate(posiciones_l)
    muestras = np.concatenate(muestras_l).astype(np.int16)
    n_brutas = len(masks)

    # Una misma máscara con dos redes distintas representa dos diseños físicos.
    vistos, conservar = set(), []
    for i, (mask, forma) in enumerate(zip(masks, formas)):
        clave = (str(forma), np.packbits(mask).tobytes())
        if clave not in vistos:
            vistos.add(clave)
            conservar.append(i)
    masks, formas = masks[conservar], formas[conservar]
    posiciones, muestras = posiciones[conservar], muestras[conservar]

    prob_l, frecuencia_l, cuantiles_l = [], [], []
    fc = cfg['forward_cuantiles']
    for inicio in range(0, len(masks), batch_size):
        x = torch.from_numpy(masks[inicio:inicio + batch_size].astype(np.float32))
        salida = juez.salida_cuantiles(x[:, None].to(device))
        prob_l.append(torch.sigmoid(salida.presencia_logit).cpu().numpy())
        frecuencia_l.append(salida.frecuencia.cpu().numpy())
        cuantiles_l.append(salida.anchuras(
            float(fc['anchura_min_real']), float(fc['anchura_max_real']))
                           .cpu().numpy())
    probabilidades = np.vstack(prob_l)
    frecuencias = np.vstack(frecuencia_l)
    cuantiles = np.vstack(cuantiles_l)  # N,K,[q10,q50,q90]
    presentes = probabilidades > float(fc['umbral_presencia'])
    c_pred = np.stack([
        probabilidades,
        np.where(presentes, frecuencias, 0.0),
        np.where(presentes, cuantiles[:, :, 1], 0.0),
    ], axis=2).reshape(len(masks), -1)

    puntuaciones = [mc.cubrimiento(c_obj, pred) for pred in c_pred]
    cubierto = np.asarray([p['cubierto'] for p in puntuaciones], float)
    servido = np.asarray([p['gaps_servidos'] for p in puntuaciones], float)
    exceso = np.asarray([p['exceso'] for p in puntuaciones], float)
    n_error = np.asarray([p['n_gaps_error'] for p in puntuaciones], int)
    # El ranking de v6 depende exclusivamente de la cobertura. `stable` hace
    # reproducibles los empates sin introducir el exceso como criterio oculto.
    orden = mc.rankear(c_obj, c_pred)

    def ordenar(x):
        return x[orden]
    return {
        'n_brutas': n_brutas,
        'n_combinaciones': len(condiciones),
        'masks': ordenar(masks), 'formas': ordenar(formas),
        'posiciones': ordenar(posiciones), 'muestras': ordenar(muestras),
        'probabilidades': ordenar(probabilidades),
        'frecuencias': ordenar(frecuencias), 'cuantiles': ordenar(cuantiles),
        'c_pred': ordenar(c_pred), 'cubierto': ordenar(cubierto),
        'gaps_servidos': ordenar(servido), 'exceso': ordenar(exceso),
        'n_gaps_error': ordenar(n_error),
    }


def cargar_redes() -> dict:
    ruta = Path(__file__).resolve().parents[1] / 'recursos' / 'redes_canonicas.json'
    return load_json(ruta)['formas']


def comprobar_mallabilidad(mask: np.ndarray, lattice_vectors: np.ndarray):
    """Malla una máscara y verifica sus dos emparejamientos periódicos."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from mallador import mallar
    from src.simulation.periodicidad import enlaces_desde_geometria, verificar
    try:
        salida = mallar(mask > 0.5, lattice_vectors)
        diagnostico = salida['diagnostico']
        if diagnostico['izq_der'] != 'ok' or diagnostico['aba_arr'] != 'ok':
            return False, f"{diagnostico['izq_der']}/{diagnostico['aba_arr']}"
        enlaces = enlaces_desde_geometria(salida['p_all'], lattice_vectors)
        verificacion = verificar(salida['p_all'], enlaces, lattice_vectors)
        return bool(verificacion['ok']), ('ok' if verificacion['ok']
                                          else 'periodicidad_invalida')
    except Exception as exc:  # el fallo se registra y se prueba la siguiente
        return False, f'{type(exc).__name__}: {exc}'


def seleccionar_mallables(resultado: dict, redes: dict, top_k: int):
    estados = np.full(len(resultado['masks']), 'no_evaluada', dtype=object)
    motivos = np.full(len(resultado['masks']), '', dtype=object)
    diagnosticos = []
    seleccion = []
    for i, (mask, forma) in enumerate(zip(resultado['masks'], resultado['formas'])):
        conexion = diagnosticar_conectividad(mask)
        diagnosticos.append(conexion)
        if not conexion['valida']:
            estados[i] = 'no_conexa'
            motivos[i] = conexion['motivo']
            continue
        lv = np.asarray(redes[str(forma)]['lattice_vectors'], dtype=float)
        ok, motivo = comprobar_mallabilidad(mask, lv)
        estados[i] = 'mallable' if ok else 'no_mallable'
        motivos[i] = motivo
        if ok:
            seleccion.append(i)
            if len(seleccion) >= top_k:
                break
    diagnosticos.extend([None] * (len(resultado['masks']) - len(diagnosticos)))
    return np.asarray(seleccion, dtype=int), estados, motivos, diagnosticos


def guardar_ranking(resultado, estados, motivos, diagnosticos, ruta: Path):
    campos = ['rank', 'forma', 'colocacion', 'muestra', 'cubierto',
              'gaps_servidos', 'exceso', 'n_gaps_error', 'densidad',
              'incertidumbre_anchura_media', 'n_componentes_periodicas',
              'conecta_a1', 'conecta_a2', 'mallabilidad', 'motivo_mallado']
    with ruta.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        for i, mask in enumerate(resultado['masks']):
            conexion = diagnosticos[i]
            presente = resultado['probabilidades'][i] > 0.5
            intervalo = (resultado['cuantiles'][i, :, 2]
                         - resultado['cuantiles'][i, :, 0])
            incertidumbre = float(intervalo[presente].mean()) if presente.any() else 0.0
            writer.writerow({
                'rank': i + 1, 'forma': resultado['formas'][i],
                'colocacion': int(resultado['posiciones'][i]),
                'muestra': int(resultado['muestras'][i]),
                'cubierto': float(resultado['cubierto'][i]),
                'gaps_servidos': float(resultado['gaps_servidos'][i]),
                'exceso': float(resultado['exceso'][i]),
                'n_gaps_error': int(resultado['n_gaps_error'][i]),
                'densidad': float(mask.mean()),
                'incertidumbre_anchura_media': incertidumbre,
                'n_componentes_periodicas': (
                    conexion['n_componentes_periodicas'] if conexion else ''),
                'conecta_a1': conexion['conecta_a1'] if conexion else '',
                'conecta_a2': conexion['conecta_a2'] if conexion else '',
                'mallabilidad': estados[i], 'motivo_mallado': motivos[i],
            })


def plot_seleccion(resultado, seleccion, ruta: Path):
    import os
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-tfm-fase5')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not len(seleccion):
        return
    fig, axes = plt.subplots(1, len(seleccion),
                             figsize=(3.2 * len(seleccion), 4.0), squeeze=False)
    for puesto, indice in enumerate(seleccion):
        ax = axes[0, puesto]
        ax.imshow(resultado['masks'][indice], cmap='gray_r')
        ax.set_title(f"#{indice + 1} · {resultado['formas'][indice]}\n"
                     f"cub={resultado['cubierto'][indice]:.3f} · "
                     f"exc={resultado['exceso'][indice]:.2f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Fase 5 · primeras candidatas mallables del ranking', y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Fase 5 — generar, rankear y filtrar candidatas mallables')
    parser.add_argument('--gaps', required=True,
                        help='un único gap como frecuencia:anchura')
    parser.add_argument('--nombre', default=None)
    parser.add_argument('--muestras-por-combinacion', type=int, default=None)
    parser.add_argument('--top-k', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    seed = int(cfg['seed'] if args.seed is None else args.seed)
    n_muestras = int(args.muestras_por_combinacion or
                     cfg['design']['muestras_por_combinacion'])
    top_k = int(args.top_k or cfg['design']['top_k'])
    batch_size = int(args.batch_size or cfg['design']['batch_size'])
    if n_muestras < 1 or top_k < 1 or batch_size < 1:
        raise SystemExit('muestras, top-k y batch-size deben ser positivos')
    set_seed(seed)
    device = get_device(cfg['device'])
    try:
        c_obj, meta = objetivo_desde_gaps(args.gaps, cfg)
    except ValueError as exc:
        raise SystemExit(f'Encargo inválido: {exc}') from exc

    print(f"=== Fase 5 · generación y ranking · {cfg['version']} ===\n")
    print(f'Device            : {device}')
    print(f'Encargo           : {args.gaps} (parcial)')
    cvae, juez, checkpoint, target_dim = cargar_modelos(cfg, device)
    print(f'cVAE              : época {checkpoint["epoch"]}')
    print(describir(juez))
    t0 = time.time()
    resultado = generar_y_juzgar(
        c_obj, cfg, cvae, juez, target_dim, device, n_muestras, batch_size)
    print(f'Combinaciones     : {resultado["n_combinaciones"]}')
    print(f'Candidatas brutas : {resultado["n_brutas"]}')
    print(f'Candidatas únicas : {len(resultado["masks"])}')

    redes = cargar_redes()
    seleccion, estados, motivos, diagnosticos = seleccionar_mallables(
        resultado, redes, top_k)
    revisadas = int(np.sum(estados != 'no_evaluada'))
    malladas = int(np.sum(np.isin(estados, ['mallable', 'no_mallable'])))
    no_conexas = int(np.sum(estados == 'no_conexa'))
    print(f'Revisadas         : {revisadas}')
    print(f'No conexas        : {no_conexas}')
    print(f'Malladas          : {malladas}')
    print(f'Mallables elegidas: {len(seleccion)}/{top_k}')
    if len(seleccion):
        print('Ranks elegidos    : ' + ', '.join(str(i + 1) for i in seleccion))

    nombre = args.nombre or ('gaps_' + args.gaps.replace(':', '-').replace(',', '_'))
    out = version_path(cfg['artifacts']['design']) / nombre
    out.mkdir(parents=True, exist_ok=True)
    save_json({'encargo': meta, 'vector_objetivo': c_obj.tolist(),
               'seed': seed}, out / 'objetivo.json')
    guardar_ranking(resultado, estados, motivos, diagnosticos,
                    out / 'ranking.csv')
    n_componentes = np.asarray([
        d['n_componentes_periodicas'] if d else -1 for d in diagnosticos],
        dtype=np.int16)
    conecta_a1 = np.asarray([d['conecta_a1'] if d else False
                             for d in diagnosticos], dtype=bool)
    conecta_a2 = np.asarray([d['conecta_a2'] if d else False
                             for d in diagnosticos], dtype=bool)
    np.savez_compressed(
        out / 'generacion.npz', masks=resultado['masks'],
        formas=resultado['formas'], colocaciones=resultado['posiciones'],
        muestras=resultado['muestras'], c_pred=resultado['c_pred'],
        probabilidades=resultado['probabilidades'],
        frecuencias=resultado['frecuencias'], cuantiles=resultado['cuantiles'],
        cubierto=resultado['cubierto'], gaps_servidos=resultado['gaps_servidos'],
        exceso=resultado['exceso'], n_gaps_error=resultado['n_gaps_error'],
        n_componentes_periodicas=n_componentes, conecta_a1=conecta_a1,
        conecta_a2=conecta_a2,
        mallabilidad=estados.astype('U30'), seleccion=seleccion)
    np.save(out / 'candidatas_mallables.npy', resultado['masks'][seleccion])
    resumen = {
        'estado': 'completado' if len(seleccion) == top_k else 'incompleto',
        'n_combinaciones': resultado['n_combinaciones'],
        'n_candidatas_brutas': resultado['n_brutas'],
        'n_candidatas_unicas': len(resultado['masks']),
        'n_revisadas_hasta_seleccion': revisadas,
        'n_descartadas_por_conectividad': no_conexas,
        'n_malladas_hasta_seleccion': malladas,
        'n_mallables_seleccionadas': len(seleccion),
        'top_k_solicitado': top_k,
        'ranks_seleccionados': (seleccion + 1).tolist(),
        'fallos_mallado': dict(Counter(
            motivos[i] for i in range(revisadas) if estados[i] == 'no_mallable')),
        'fallos_conectividad': dict(Counter(
            motivos[i] for i in range(revisadas) if estados[i] == 'no_conexa')),
        'duracion_segundos': time.time() - t0,
        'checkpoint_cvae_epoca': int(checkpoint['epoch']),
        'forward': 'cuantiles',
        'ranking': 'cobertura descendente; empates en orden de generación',
    }
    save_json(resumen, out / 'resultados.json')
    plot_seleccion(resultado, seleccion, out / 'candidatas.png')
    print(f'\nResultado         : {out}')
    print(f'Estado            : {resumen["estado"]}')


if __name__ == '__main__':
    main()
