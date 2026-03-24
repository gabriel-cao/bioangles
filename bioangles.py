#!/usr/bin/env python3
"""
BioAngles v1.1 — Extractor de ángulos articulares desde video
Gabriel Cao Di Marco & Daniela | 2026
Usa MediaPipe Tasks PoseLandmarker para detectar landmarks y calcular ángulos articulares.
Output: CSV con ángulos por frame, comparables contra datos CDC.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

import numpy as np
import pandas as pd
import argparse
import os
import sys
import json
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# RANGOS CDC - Mujeres 20-44 años (Range of Motion, grados)
# Fuente: CDC Anthropometric Reference Data
# ═══════════════════════════════════════════════════════════════
CDC_RANGES = {
    "hombro_flexion":     {"min": 0,   "max": 180, "nombre": "Shoulder Flexion"},
    "hombro_abduccion":   {"min": 0,   "max": 180, "nombre": "Shoulder Abduction"},
    "codo_flexion":       {"min": 0,   "max": 150, "nombre": "Elbow Flexion"},
    "cadera_flexion":     {"min": 0,   "max": 120, "nombre": "Hip Flexion"},
    "rodilla_flexion":    {"min": 0,   "max": 135, "nombre": "Knee Flexion"},
    "tobillo_dorsiflexion": {"min": 0, "max": 20,  "nombre": "Ankle Dorsiflexion"},
    "tobillo_plantarflexion": {"min": 0, "max": 50, "nombre": "Ankle Plantarflexion"},
    "cuello_flexion":     {"min": 0,   "max": 45,  "nombre": "Neck Flexion"},
    "tronco_flexion":     {"min": 0,   "max": 80,  "nombre": "Trunk Flexion"},
}

SCRIPT_DIR = Path(__file__).parent

# ═══════════════════════════════════════════════════════════════
# FUNCIONES DE CÁLCULO
# ═══════════════════════════════════════════════════════════════

def calcular_angulo(a, b, c):
    """Calcula el ángulo en el punto b formado por los segmentos ba y bc."""
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosine = np.clip(cosine, -1.0, 1.0)
    return round(np.degrees(np.arccos(cosine)), 2)


def extraer_angulos(landmarks, w, h):
    """Extrae todos los ángulos articulares relevantes de un frame."""
    def get(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h, lm.z * w])
    
    angulos = {}
    
    # HOMBRO (Flexión: cadera-hombro-codo)
    angulos["hombro_L_flexion"] = calcular_angulo(get(23), get(11), get(13))
    angulos["hombro_R_flexion"] = calcular_angulo(get(24), get(12), get(14))
    
    # CODO (Flexión: hombro-codo-muñeca)
    angulos["codo_L_flexion"] = calcular_angulo(get(11), get(13), get(15))
    angulos["codo_R_flexion"] = calcular_angulo(get(12), get(14), get(16))
    
    # CADERA (Flexión: hombro-cadera-rodilla)
    angulos["cadera_L_flexion"] = calcular_angulo(get(11), get(23), get(25))
    angulos["cadera_R_flexion"] = calcular_angulo(get(12), get(24), get(26))
    
    # RODILLA (Flexión: cadera-rodilla-tobillo)
    angulos["rodilla_L_flexion"] = calcular_angulo(get(23), get(25), get(27))
    angulos["rodilla_R_flexion"] = calcular_angulo(get(24), get(26), get(28))
    
    # TOBILLO (rodilla-tobillo-pie)
    angulos["tobillo_L"] = calcular_angulo(get(25), get(27), get(31))
    angulos["tobillo_R"] = calcular_angulo(get(26), get(28), get(32))
    
    # TRONCO (nariz-midhombro-midcadera)
    mid_hombro = (get(11) + get(12)) / 2
    mid_cadera = (get(23) + get(24)) / 2
    angulos["tronco_inclinacion"] = calcular_angulo(get(0), mid_hombro, mid_cadera)
    
    # CUELLO (nariz-midoreja-midhombro)
    mid_ear = (get(7) + get(8)) / 2
    angulos["cuello"] = calcular_angulo(get(0), mid_ear, mid_hombro)
    
    # Confidence (visibility promedio)
    vis_indices = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32]
    angulos["confidence"] = round(np.mean([landmarks[i].visibility for i in vis_indices]), 4)
    
    return angulos


def evaluar_cdc(angulos_df):
    """Evalúa qué porcentaje de frames tienen ángulos dentro de rangos CDC."""
    resumen = {}
    mappings = {
        "hombro_flexion": ["hombro_L_flexion", "hombro_R_flexion"],
        "codo_flexion": ["codo_L_flexion", "codo_R_flexion"],
        "cadera_flexion": ["cadera_L_flexion", "cadera_R_flexion"],
        "rodilla_flexion": ["rodilla_L_flexion", "rodilla_R_flexion"],
    }
    for cdc_key, cols in mappings.items():
        rango = CDC_RANGES[cdc_key]
        for col in cols:
            if col in angulos_df.columns:
                valores = angulos_df[col]
                dentro = ((valores >= rango["min"]) & (valores <= rango["max"])).sum()
                total = len(valores)
                pct = round(100 * dentro / total, 1) if total > 0 else 0
                resumen[col] = {
                    "media": round(valores.mean(), 2),
                    "std": round(valores.std(), 2),
                    "min": round(valores.min(), 2),
                    "max": round(valores.max(), 2),
                    "rango_CDC": f"{rango['min']}°-{rango['max']}°",
                    "dentro_CDC_%": pct
                }
    return resumen


# ═══════════════════════════════════════════════════════════════
# PROCESAMIENTO DE VIDEO
# ═══════════════════════════════════════════════════════════════

MODEL_FILES = {
    0: "pose_landmarker_lite.task",
    1: "pose_landmarker_full.task",
    2: "pose_landmarker_heavy.task",
}

def procesar_video(video_path, output_dir=None, visualizar=False, cada_n_frames=1, model_complexity=1):
    """Procesa un video y extrae ángulos articulares frame por frame."""
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"ERROR: No se encuentra el video: {video_path}")
        sys.exit(1)
    
    if output_dir is None:
        output_dir = video_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    nombre_base = video_path.stem
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: No se puede abrir el video: {video_path}")
        sys.exit(1)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duracion = total_frames / fps if fps > 0 else 0
    
    model_file = MODEL_FILES.get(model_complexity, MODEL_FILES[1])
    model_path = SCRIPT_DIR / model_file
    if not model_path.exists():
        print(f"ERROR: Modelo no encontrado: {model_path}")
        print(f"Descargá los modelos con:")
        print(f"  wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"  BioAngles v1.1 — Gabriel & Daniela")
    print(f"{'='*60}")
    print(f"  Video: {video_path.name}")
    print(f"  Resolución: {w}x{h} @ {fps:.1f} fps")
    print(f"  Frames totales: {total_frames}")
    print(f"  Duración: {duracion:.1f}s")
    print(f"  Cada {cada_n_frames} frame(s) | Modelo: {model_file}")
    print(f"{'='*60}\n")
    
    # Inicializar PoseLandmarker (nueva Tasks API)
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = PoseLandmarker.create_from_options(options)
    
    datos = []
    frame_num = 0
    procesados = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        if frame_num % cada_n_frames != 0:
            continue
        
        # Convertir a MediaPipe Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_num * 1000 / fps) if fps > 0 else frame_num
        
        results = landmarker.detect_for_video(mp_image, timestamp_ms)
        
        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            landmarks = results.pose_landmarks[0]  # Primera persona
            angulos = extraer_angulos(landmarks, w, h)
            angulos["frame"] = frame_num
            angulos["tiempo_s"] = round(frame_num / fps, 3) if fps > 0 else 0
            datos.append(angulos)
            procesados += 1
        
        if frame_num % 100 == 0:
            pct = round(100 * frame_num / total_frames, 1)
            print(f"  Procesando... {pct}% ({frame_num}/{total_frames})")
    
    cap.release()
    landmarker.close()
    
    if not datos:
        print("ERROR: No se detectaron poses en el video.")
        sys.exit(1)
    
    df = pd.DataFrame(datos)
    cols_orden = ["frame", "tiempo_s", "confidence",
                  "hombro_L_flexion", "hombro_R_flexion",
                  "codo_L_flexion", "codo_R_flexion",
                  "cadera_L_flexion", "cadera_R_flexion",
                  "rodilla_L_flexion", "rodilla_R_flexion",
                  "tobillo_L", "tobillo_R",
                  "tronco_inclinacion", "cuello"]
    df = df[[c for c in cols_orden if c in df.columns]]
    
    # Guardar CSV
    csv_path = output_dir / f"{nombre_base}_angulos.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  CSV guardado: {csv_path}")
    
    # Evaluación CDC
    resumen_cdc = evaluar_cdc(df)
    
    # Stats JSON
    stats = {
        "video": video_path.name,
        "resolucion": f"{w}x{h}",
        "fps": fps,
        "total_frames": total_frames,
        "frames_procesados": procesados,
        "frames_con_pose": len(datos),
        "duracion_s": round(duracion, 2),
        "modelo": model_file,
        "confidence_media": round(df["confidence"].mean(), 4),
        "angulos_promedio": {},
        "evaluacion_CDC": resumen_cdc
    }
    for col in df.columns:
        if col not in ["frame", "tiempo_s", "confidence"]:
            stats["angulos_promedio"][col] = {
                "media": round(df[col].mean(), 2),
                "std": round(df[col].std(), 2),
                "min": round(df[col].min(), 2),
                "max": round(df[col].max(), 2)
            }
    
    json_path = output_dir / f"{nombre_base}_stats.json"
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Stats guardado: {json_path}")
    
    # Imprimir resumen
    print(f"\n{'='*60}")
    print(f"  RESUMEN — {video_path.name}")
    print(f"{'='*60}")
    print(f"  Frames procesados: {procesados}")
    print(f"  Confidence media: {stats['confidence_media']}")
    print(f"\n  ÁNGULOS PROMEDIO:")
    print(f"  {'Articulación':<25} {'Media':>8} {'±Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*57}")
    for col, vals in stats["angulos_promedio"].items():
        print(f"  {col:<25} {vals['media']:>7.1f}° {vals['std']:>7.1f}° "
              f"{vals['min']:>7.1f}° {vals['max']:>7.1f}°")
    if resumen_cdc:
        print(f"\n  EVALUACIÓN CDC (Mujeres 20-44):")
        print(f"  {'Articulación':<25} {'Media':>8} {'Rango CDC':>15} {'Dentro':>8}")
        print(f"  {'-'*57}")
        for col, vals in resumen_cdc.items():
            print(f"  {col:<25} {vals['media']:>7.1f}° {vals['rango_CDC']:>14} "
                  f"{vals['dentro_CDC_%']:>6.1f}%")
    print(f"\n{'='*60}")
    print(f"  Archivos: {csv_path}")
    print(f"           {json_path}")
    print(f"{'='*60}\n")
    return df


# ═══════════════════════════════════════════════════════════════
# BATCH + MAIN
# ═══════════════════════════════════════════════════════════════

def procesar_batch(directorio, output_dir=None, cada_n_frames=1, extensiones=None, model_complexity=1):
    """Procesa todos los videos en un directorio."""
    if extensiones is None:
        extensiones = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    directorio = Path(directorio)
    videos = [f for f in directorio.iterdir() if f.suffix.lower() in extensiones]
    if not videos:
        print(f"No se encontraron videos en {directorio}")
        return
    print(f"\nBatch: {len(videos)} videos encontrados\n")
    resultados = []
    for i, video in enumerate(sorted(videos), 1):
        print(f"\n[{i}/{len(videos)}] {video.name}")
        try:
            df = procesar_video(video, output_dir, cada_n_frames=cada_n_frames, model_complexity=model_complexity)
            resultados.append({"video": video.name, "frames": len(df), "status": "OK"})
        except Exception as e:
            print(f"  ERROR: {e}")
            resultados.append({"video": video.name, "frames": 0, "status": str(e)})
    print(f"\n{'='*60}")
    print(f"  BATCH: {len(resultados)} videos")
    print(f"{'='*60}")
    for r in resultados:
        s = "✓" if r["status"] == "OK" else "✗"
        print(f"  {s} {r['video']} — {r['frames']} frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BioAngles v1.1 — Extractor de ángulos articulares (Gabriel & Daniela)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python3 bioangles.py video.mp4                    # Un video
  python3 bioangles.py video.mp4 -o resultados/     # Guardar en carpeta
  python3 bioangles.py video.mp4 -n 3               # Cada 3 frames
  python3 bioangles.py video.mp4 -m 2               # Modelo heavy (más preciso)
  python3 bioangles.py --batch /carpeta/videos/      # Todos los videos
        """
    )
    parser.add_argument("video", nargs="?", help="Ruta al video")
    parser.add_argument("-o", "--output", help="Directorio de salida")
    parser.add_argument("-n", "--cada-n", type=int, default=1,
                        help="Procesar cada N frames (default: 1)")
    parser.add_argument("-v", "--visualizar", action="store_true",
                        help="Mostrar video con skeleton overlay")
    parser.add_argument("-m", "--model-complexity", type=int, default=1, choices=[0, 1, 2],
                        help="0=lite, 1=full, 2=heavy (default: 1)")
    parser.add_argument("--batch", help="Directorio con videos")
    args = parser.parse_args()
    
    if args.batch:
        procesar_batch(args.batch, args.output, args.cada_n, model_complexity=args.model_complexity)
    elif args.video:
        procesar_video(args.video, args.output, args.visualizar, args.cada_n, args.model_complexity)
    else:
        parser.print_help()
