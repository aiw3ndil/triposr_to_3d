#!/usr/bin/env python3
"""
triposr_to_3d.py — Reconstrucción 3D completa (360°) a partir de una sola imagen 2D usando TripoSR.

Genera modelos 3D cerrados con geometría completa y texturas horneadas.
Optimizado para GPUs con 4 GB de VRAM (GTX 1650) o ejecución en CPU.
"""

import argparse
import logging
import os
import sys
import time

# Configurar variables de entorno para cachés antes de cargar PyTorch y Hugging Face
current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TMPDIR", "/media/aiw3ndil/kingston/pip_tmp")
os.environ.setdefault("HF_HOME", "/media/aiw3ndil/kingston/huggingface_cache")
os.environ.setdefault("U2NET_HOME", "/media/aiw3ndil/kingston/u2net_cache")

# Incluir carpeta TripoSR en sys.path
triposr_dir = os.path.join(current_dir, "TripoSR")
if triposr_dir not in sys.path:
    sys.path.insert(0, triposr_dir)

import numpy as np
import rembg
import torch
import xatlas
from PIL import Image

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, save_video
from tsr.bake_texture import bake_texture


class Timer:
    def __init__(self):
        self.items = {}

    def start(self, name: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.items[name] = time.time()
        logging.info(f"⏳ Iniciando: {name}...")

    def end(self, name: str) -> float:
        if name not in self.items:
            return 0.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = self.items.pop(name)
        delta = time.time() - start_time
        logging.info(f"✅ {name} completado en {delta:.2f} s.")
        return delta


def main():
    logging.basicConfig(
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Genera modelos 3D completos (360°) usando TripoSR de Stability AI & Tripo."
    )
    parser.add_argument("image", type=str, nargs="+", help="Ruta de la imagen o imágenes de entrada.")
    parser.add_argument(
        "--output-dir", "-o",
        default="output_triposr",
        type=str,
        help="Directorio donde guardar el modelo 3D resultante (default: output_triposr/)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        type=str,
        help="Dispositivo de cómputo ('cuda:0' o 'cpu'). Default: automático según GPU disponible.",
    )
    parser.add_argument(
        "--chunk-size",
        default=4096,
        type=int,
        help="Tamaño de chunk para renderizado/extracción. 4096 es óptimo para 4GB VRAM (GTX 1650). Default: 4096",
    )
    parser.add_argument(
        "--mc-resolution",
        default=256,
        type=int,
        help="Resolución de la cuadrícula de Marching Cubes (ej. 192 o 256). Default: 256",
    )
    parser.add_argument(
        "--format",
        default="obj",
        choices=["obj", "glb"],
        help="Formato del modelo 3D ('obj' o 'glb'). Default: obj",
    )
    parser.add_argument(
        "--bake-texture",
        action="store_true",
        default=True,
        help="Hornear atlas de texturas UV (.png) en lugar de solo colores por vértice (recomendado). Default: True",
    )
    parser.add_argument(
        "--no-bake-texture",
        action="store_false",
        dest="bake_texture",
        help="Exportar solo con colores por vértice sin mapa de texturas UV.",
    )
    parser.add_argument(
        "--texture-resolution",
        default=1024,
        type=int,
        help="Resolución del mapa de texturas si --bake-texture está activo (1024 o 2048). Default: 1024",
    )
    parser.add_argument(
        "--no-remove-bg",
        action="store_true",
        help="No eliminar el fondo automáticamente (usar si la imagen ya tiene fondo transparente o neutro).",
    )
    parser.add_argument(
        "--foreground-ratio",
        default=0.85,
        type=float,
        help="Proporción del objeto respecto al lienzo. Default: 0.85",
    )
    parser.add_argument(
        "--render-video",
        action="store_true",
        help="Guardar un vídeo en rotación 360° (render.mp4).",
    )

    args = parser.parse_args()

    timer = Timer()
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA no está disponible en PyTorch. Usando CPU como respaldo.")
        device = "cpu"

    logging.info(f"Dispositivo seleccionado: {device}")
    if device.startswith("cuda"):
        logging.info(f"GPU detectada: {torch.cuda.get_device_name(0)}")

    # 1. Cargar el modelo preentrenado
    timer.start("Carga del modelo TripoSR")
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(args.chunk_size)
    model.to(device)
    timer.end("Carga del modelo TripoSR")

    # 2. Preprocesar las imágenes (eliminación de fondo y centrado)
    timer.start("Preprocesamiento y segmentación de imágenes")
    images = []
    rembg_session = None if args.no_remove_bg else rembg.new_session()

    for idx, img_path in enumerate(args.image):
        if not os.path.exists(img_path):
            logging.error(f"Imagen no encontrada: {img_path}")
            continue

        raw_img = Image.open(img_path)
        if args.no_remove_bg:
            processed_img = np.array(raw_img.convert("RGB"))
        else:
            logging.info(f"Segmentando fondo de '{img_path}' con rembg...")
            bg_removed = remove_background(raw_img, rembg_session)
            resized = resize_foreground(bg_removed, args.foreground_ratio)
            arr = np.array(resized).astype(np.float32) / 255.0
            # Componer sobre fondo gris medio neutral requerido por TripoSR
            composite = arr[:, :, :3] * arr[:, :, 3:4] + (1.0 - arr[:, :, 3:4]) * 0.5
            processed_img = Image.fromarray((composite * 255.0).astype(np.uint8))

        subfolder = os.path.join(args.output_dir, os.path.splitext(os.path.basename(img_path))[0])
        os.makedirs(subfolder, exist_ok=True)
        if not args.no_remove_bg:
            processed_img.save(os.path.join(subfolder, "input_processed.png"))
        images.append((img_path, processed_img, subfolder))

    timer.end("Preprocesamiento y segmentación de imágenes")

    if not images:
        logging.error("No se pudo procesar ninguna imagen válida.")
        return

    # 3. Inferencia 3D y extracción de malla
    for original_path, pil_img, subfolder in images:
        logging.info(f"\n==========================================")
        logging.info(f"Procesando: {original_path}")
        logging.info(f"Directorio de salida: {subfolder}")

        timer.start("Inferencia neuronal 3D (LRM Triplane)")
        with torch.no_grad():
            scene_codes = model([pil_img], device=device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        timer.end("Inferencia neuronal 3D (LRM Triplane)")

        if args.render_video:
            timer.start("Renderizando vídeo de rotación")
            render_images = model.render(scene_codes, n_views=30, return_type="pil")
            video_path = os.path.join(subfolder, "render.mp4")
            save_video(render_images[0], video_path, fps=30)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            timer.end("Renderizando vídeo de rotación")

        timer.start("Extracción de malla (Marching Cubes)")
        meshes = model.extract_mesh(
            scene_codes,
            has_vertex_color=(not args.bake_texture),
            resolution=args.mc_resolution,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        timer.end("Extracción de malla (Marching Cubes)")

        out_mesh_path = os.path.join(subfolder, f"modelo_3d.{args.format}")

        if args.bake_texture:
            out_texture_path = os.path.join(subfolder, "texture.png")
            timer.start("Horneado de textura UV (xatlas)")
            bake_output = bake_texture(
                meshes[0],
                model,
                scene_codes[0],
                args.texture_resolution,
            )
            timer.end("Horneado de textura UV (xatlas)")

            timer.start("Exportando modelo 3D y atlas de textura")
            xatlas.export(
                out_mesh_path,
                meshes[0].vertices[bake_output["vmapping"]],
                bake_output["indices"],
                bake_output["uvs"],
                meshes[0].vertex_normals[bake_output["vmapping"]],
            )
            baked_img = Image.fromarray((bake_output["colors"] * 255.0).astype(np.uint8))
            baked_img.transpose(Image.FLIP_TOP_BOTTOM).save(out_texture_path)
            timer.end("Exportando modelo 3D y atlas de textura")

            logging.info(f"🎉 Modelo guardado en: {out_mesh_path}")
            logging.info(f"🎨 Textura guardada en: {out_texture_path}")
        else:
            timer.start("Exportando modelo con colores de vértice")
            meshes[0].export(out_mesh_path)
            timer.end("Exportando modelo con colores de vértice")
            logging.info(f"🎉 Modelo guardado en: {out_mesh_path}")

    logging.info("\n✨ ¡Proceso completado con éxito!")


if __name__ == "__main__":
    main()
