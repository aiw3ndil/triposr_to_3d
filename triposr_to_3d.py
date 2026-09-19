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

# Si se ejecuta con el python global del sistema, redirigir automáticamente al entorno virtual venv
venv_python = os.path.join(current_dir, "venv", "bin", "python")
if os.path.exists(venv_python) and os.path.realpath(sys.executable) != os.path.realpath(venv_python):
    os.execv(venv_python, [venv_python] + sys.argv)

os.environ.setdefault("TMPDIR", "/media/aiw3ndil/kingston/pip_tmp")
os.environ.setdefault("HF_HOME", "/media/aiw3ndil/kingston/huggingface_cache")
os.environ.setdefault("U2NET_HOME", "/media/aiw3ndil/kingston/u2net_cache")

# Incluir carpeta TripoSR en sys.path
triposr_dir = os.path.join(current_dir, "TripoSR")
if triposr_dir not in sys.path:
    sys.path.insert(0, triposr_dir)

import numpy as np
import rembg
import scipy.ndimage as ndi
import torch
import trimesh
import trimesh.smoothing
from trimesh.visual.material import PBRMaterial
import xatlas
from PIL import Image, ImageEnhance, ImageFilter

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, save_video
from tsr.bake_texture import bake_texture


def enhance_input_image(pil_img: Image.Image) -> Image.Image:
    """
    Aplica realce de micro-contraste y enfoque (Unsharp Masking) a la imagen de entrada
    para que la red neuronal LRM capture contornos, siluetas y volúmenes con mayor precisión.
    """
    sharpened = pil_img.filter(ImageFilter.UnsharpMask(radius=2, percent=140, threshold=2))
    enhancer = ImageEnhance.Contrast(sharpened)
    return enhancer.enhance(1.08)


def project_frontal_texture(
    positions_texture: np.ndarray,
    normals_texture: np.ndarray,
    triplane_colors: np.ndarray,
    fg_image: Image.Image,
    focal_norm: float = 1.3737387,
) -> np.ndarray:
    """
    Reproyecta la imagen fotográfica original de alta definición directamente sobre las caras
    visibles de la malla 3D, realizando una fusión continua (cosine smoothstep blending)
    con la textura 360° estimada por TripoSR para la parte posterior y zonas ocluidas.
    """
    fg_arr = np.array(fg_image.convert("RGBA")).astype(np.float32) / 255.0
    H_fg, W_fg = fg_arr.shape[:2]

    # Convertir a orientación estándar de imagen (fila 0 en la parte superior)
    pos_map = np.flipud(positions_texture)
    norm_map = np.flipud(normals_texture)
    col_map = np.flipud(triplane_colors)

    valid_mask = pos_map[..., 3] > 0.0
    if not np.any(valid_mask):
        return col_map

    # Coordenadas 3D (X, Y, Z) en espacio de mundo de TripoSR
    x = pos_map[..., 0]
    y = pos_map[..., 1]
    z = pos_map[..., 2]

    # Cámara canónica de TripoSR en (1.9, 0, 0) mirando hacia el origen (-X)
    vx = 1.9 - x
    vy = -y
    vz = -z
    dist = np.sqrt(vx**2 + vy**2 + vz**2) + 1e-8
    vx_hat = vx / dist
    vy_hat = vy / dist
    vz_hat = vz / dist

    # Normales unitarias de la superficie
    nx = norm_map[..., 0]
    ny = norm_map[..., 1]
    nz = norm_map[..., 2]
    norm_len = np.maximum(np.sqrt(nx**2 + ny**2 + nz**2), 1e-6)
    nx_hat = nx / norm_len
    ny_hat = ny / norm_len
    nz_hat = nz / norm_len

    # Coseno del ángulo entre la normal de la superficie y el rayo de la cámara
    cos_theta = nx_hat * vx_hat + ny_hat * vy_hat + nz_hat * vz_hat

    # Profundidad a lo largo del eje óptico de la cámara
    depth = vx
    safe_depth = np.maximum(depth, 0.05)

    # Proyección perspectiva en la cámara
    u_cam = (y / safe_depth) * focal_norm
    v_cam = (z / safe_depth) * focal_norm

    u_img = 0.5 + u_cam
    v_img = 0.5 - v_cam

    # Prueba de visibilidad y oclusión mediante Z-Buffer
    zb_res = 512
    z_buf = np.full((zb_res, zb_res), 999.0, dtype=np.float32)

    in_view_mask = (
        valid_mask
        & (cos_theta > 0.0)
        & (u_img >= 0.0)
        & (u_img <= 1.0)
        & (v_img >= 0.0)
        & (v_img <= 1.0)
    )

    visible_mask = np.zeros_like(valid_mask)
    if np.any(in_view_mask):
        zb_cols = np.clip((u_img[in_view_mask] * (zb_res - 1)).astype(np.int32), 0, zb_res - 1)
        zb_rows = np.clip((v_img[in_view_mask] * (zb_res - 1)).astype(np.int32), 0, zb_res - 1)
        depth_in_view = depth[in_view_mask]

        np.minimum.at(z_buf, (zb_rows, zb_cols), depth_in_view)
        z_buf_min = ndi.minimum_filter(z_buf, size=3)

        is_front = depth_in_view <= (z_buf_min[zb_rows, zb_cols] + 0.04)
        visible_mask[in_view_mask] = is_front

    # Cálculo del peso de fusión suave (smoothstep entre 0.06 y 0.28)
    weight = np.clip((cos_theta - 0.06) / (0.28 - 0.06), 0.0, 1.0)
    weight = weight * weight * (3.0 - 2.0 * weight)
    weight[~visible_mask] = 0.0

    out_colors = col_map.copy()
    blend_mask = weight > 0.0

    if np.any(blend_mask):
        r_coords = np.clip(v_img[blend_mask] * (H_fg - 1), 0.0, H_fg - 1)
        c_coords = np.clip(u_img[blend_mask] * (W_fg - 1), 0.0, W_fg - 1)

        coords = [r_coords, c_coords]
        proj_r = ndi.map_coordinates(fg_arr[..., 0], coords, order=1, mode="nearest")
        proj_g = ndi.map_coordinates(fg_arr[..., 1], coords, order=1, mode="nearest")
        proj_b = ndi.map_coordinates(fg_arr[..., 2], coords, order=1, mode="nearest")
        proj_a = ndi.map_coordinates(fg_arr[..., 3], coords, order=1, mode="nearest")

        active_w = weight[blend_mask] * np.clip(proj_a / 0.8, 0.0, 1.0)
        active_w = active_w[..., None]

        triplane_vals = col_map[blend_mask, :3]
        proj_rgb = np.stack([proj_r, proj_g, proj_b], axis=-1)

        blended = active_w * proj_rgb + (1.0 - active_w) * triplane_vals
        out_colors[blend_mask, :3] = blended

    # Difundir la textura fotográfica hacia caras laterales y traseras
    # para eliminar manchas negras o sombras residuales del triplane
    projected_mask = valid_mask & (weight > 0.12)
    if np.any(projected_mask):
        indices = ndi.distance_transform_edt(~projected_mask, return_distances=False, return_indices=True)
        dilated = out_colors[tuple(indices)]
        unprojected = valid_mask & (weight < 0.12)
        out_colors[unprojected] = dilated[unprojected]

    return out_colors


def generate_multiscale_normal_map(
    diffuse_img: Image.Image,
    strength: float = 2.0,
) -> Image.Image:
    """
    Genera un mapa de normales tangentes multi-escala combinando:
    - Bajas frecuencias (macro-volumen y curvatura general)
    - Medias frecuencias (biseles, aristas y hendiduras)
    - Altas frecuencias (micro-relieve, textura fina y grano)
    """
    gray = np.array(diffuse_img.convert("L")).astype(np.float32) / 255.0

    # 1. Macro-volumen (sigma = 7.0)
    low_freq = ndi.gaussian_filter(gray, sigma=7.0)
    dx_low = ndi.sobel(low_freq, axis=1)
    dy_low = ndi.sobel(low_freq, axis=0)

    # 2. Medias frecuencias: biseles y aristas (sigma = 2.0)
    mid_freq = ndi.gaussian_filter(gray, sigma=2.0)
    dx_mid = ndi.sobel(mid_freq, axis=1)
    dy_mid = ndi.sobel(mid_freq, axis=0)

    # 3. Altas frecuencias: micro-relieve y grano fino
    high_freq = gray - mid_freq
    dx_high = ndi.sobel(high_freq, axis=1)
    dy_high = ndi.sobel(high_freq, axis=0)

    # Combinación ponderada y amplificación de relieve
    dx = (0.30 * dx_low + 0.45 * dx_mid + 0.55 * dx_high) * strength * 3.5
    dy = (0.30 * dy_low + 0.45 * dy_mid + 0.55 * dy_high) * strength * 3.5
    dz = np.ones_like(gray) * 1.0

    norm = np.sqrt(dx**2 + dy**2 + dz**2)
    n_x = (-dx / norm + 1.0) * 0.5
    n_y = (-dy / norm + 1.0) * 0.5
    n_z = (dz / norm + 1.0) * 0.5

    normal_arr = (np.stack([n_x, n_y, n_z], axis=-1) * 255.0).astype(np.uint8)
    return Image.fromarray(normal_arr)


def generate_pbr_maps(
    diffuse_img: Image.Image,
    base_roughness: float = 0.55,
    metallic_bias: float = 0.15,
) -> dict:
    """
    Genera mapas PBR de Oclusión Ambiental (AO), Rugosidad (Roughness) y Metálico (Metallic).
    Empaqueta también el canal combinado metallicRoughnessTexture para glTF/GLB (R=AO, G=Roughness, B=Metallic).
    """
    rgb = np.array(diffuse_img.convert("RGB")).astype(np.float32) / 255.0
    gray = np.array(diffuse_img.convert("L")).astype(np.float32) / 255.0

    # 1. Mapa de Oclusión Ambiental (AO)
    blurred_gray = ndi.gaussian_filter(gray, sigma=3.0)
    laplacian = ndi.laplace(blurred_gray)
    cavity = np.clip(-laplacian * 4.0, 0.0, 1.0)
    dark_areas = 1.0 - blurred_gray
    ao_arr = 1.0 - np.clip(0.55 * cavity + 0.35 * dark_areas**1.4, 0.0, 0.70)
    ao_uint8 = (ao_arr * 255.0).astype(np.uint8)

    # 2. Mapa de Rugosidad (Roughness)
    variance = ndi.uniform_filter(gray**2, size=3) - ndi.uniform_filter(gray, size=3)**2
    variance = np.maximum(variance, 0.0)
    roughness = base_roughness - 0.25 * (gray - 0.5) + 0.30 * (1.0 - ao_arr) + 0.25 * np.clip(variance * 15.0, 0.0, 1.0)
    roughness = np.clip(roughness, 0.12, 0.95)
    roughness_uint8 = (roughness * 255.0).astype(np.uint8)

    # 3. Mapa de Metálico (Metallic)
    max_c = np.maximum.reduce([rgb[..., 0], rgb[..., 1], rgb[..., 2]])
    min_c = np.minimum.reduce([rgb[..., 0], rgb[..., 1], rgb[..., 2]])
    delta = max_c - min_c
    sat = np.where(max_c > 1e-4, delta / (max_c + 1e-6), 0.0)
    metal_mask = np.clip((gray - 0.35) * 1.8 * (1.0 - 0.45 * sat), 0.0, 1.0)
    metallic = np.clip(metallic_bias + metal_mask * 0.45, 0.0, 0.92)
    metallic_uint8 = (metallic * 255.0).astype(np.uint8)

    # 4. Atlas GLB PBR: Canal R = AO, Canal G = Rugosidad, Canal B = Metálico
    mr_combined = np.stack([ao_uint8, roughness_uint8, metallic_uint8], axis=-1)

    return {
        "ao": Image.fromarray(ao_uint8),
        "roughness": Image.fromarray(roughness_uint8),
        "metallic": Image.fromarray(metallic_uint8),
        "metallic_roughness": Image.fromarray(mr_combined),
    }


def refine_bowstring(mesh: trimesh.Trimesh, factor_scale: float = 0.28) -> trimesh.Trimesh:
    """
    Detecta y afina de forma continua la cuerda del arco para que adquiera
    un grosor realista y esbelto en lugar de un cilindro grueso de Marching Cubes.
    """
    v = mesh.vertices.copy()
    string_mask = (v[:, 1] > 0.03) & (np.abs(v[:, 2]) < 0.44)
    if string_mask.sum() > 300:
        z_abs = np.abs(v[:, 2])
        factor = np.ones(len(v))
        factor[string_mask] = factor_scale
        fade_mask = (z_abs > 0.36) & (z_abs < 0.44) & string_mask
        if np.any(fade_mask):
            t = (0.44 - z_abs[fade_mask]) / (0.44 - 0.36)
            factor[fade_mask] = 1.0 - (1.0 - factor_scale) * t

        z_str = v[string_mask, 2]
        x_str = v[string_mask, 0]
        y_str = v[string_mask, 1]

        poly_x = np.poly1d(np.polyfit(z_str, x_str, 1))
        poly_y = np.poly1d(np.polyfit(z_str, y_str, 1))

        axis_x = poly_x(v[:, 2])
        axis_y = poly_y(v[:, 2])

        v[string_mask, 0] = axis_x[string_mask] + (v[string_mask, 0] - axis_x[string_mask]) * factor[string_mask]
        v[string_mask, 1] = axis_y[string_mask] + (v[string_mask, 1] - axis_y[string_mask]) * factor[string_mask]
        mesh.vertices = v
    return mesh


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
        default=320,
        type=int,
        help="Resolución de la cuadrícula de Marching Cubes (ej. 256, 320 o 384). 320 ofrece geometría mucho más definida sin desbordar VRAM. Default: 320",
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
        "--texture-projection",
        action="store_true",
        default=True,
        help="Reproyectar la imagen original de alta resolución directamente en las caras visibles para máxima nitidez (default: True).",
    )
    parser.add_argument(
        "--no-texture-projection",
        action="store_false",
        dest="texture_projection",
        help="Desactivar la reproyección fotográfica frontal.",
    )
    parser.add_argument(
        "--normal-strength",
        default=2.0,
        type=float,
        help="Intensidad del mapa de normales PBR multi-escala. Default: 2.0",
    )
    parser.add_argument(
        "--enhance-image",
        action="store_true",
        default=True,
        help="Aplicar realce de bordes (Unsharp Mask) a la imagen antes de inferencia para geometría más precisa. Default: True",
    )
    parser.add_argument(
        "--no-enhance-image",
        action="store_false",
        dest="enhance_image",
        help="No aplicar realce a la imagen de entrada.",
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
        "--smooth-iterations",
        default=4,
        type=int,
        help="Iteraciones del filtro Taubin (default: 4 para suavizar ruido sin redondear aristas ni detalles; 0 para desactivar).",
    )
    parser.add_argument(
        "--clean-mesh",
        action="store_true",
        default=True,
        help="Eliminar partículas de ruido y fragmentos flotantes aislados (default: True).",
    )
    parser.add_argument(
        "--no-clean-mesh",
        action="store_false",
        dest="clean_mesh",
        help="No eliminar fragmentos flotantes.",
    )
    parser.add_argument(
        "--max-faces",
        default=60000,
        type=int,
        help="Límite de caras tras Marching Cubes mediante decimate cuadrático (preserva bordes vivos y acelera el UV unwrap). 0 para ilimitado. Default: 60000",
    )
    parser.add_argument(
        "--metallic-factor",
        default=0.20,
        type=float,
        help="Factor base metálico PBR (0.0 no metálico, 1.0 metal puro). Default: 0.20",
    )
    parser.add_argument(
        "--roughness-factor",
        default=0.55,
        type=float,
        help="Factor base de rugosidad PBR (0.1 pulido, 0.9 rugoso/mate). Default: 0.55",
    )
    parser.add_argument(
        "--thin-string",
        action="store_true",
        default=True,
        help="Afinar automáticamente la cuerda si se detecta un arco (default: True).",
    )
    parser.add_argument(
        "--thin-string-factor",
        default=0.28,
        type=float,
        help="Factor de grosor para la cuerda (0.28 = ~3.5x más fina). Default: 0.28",
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

    # 2. Preprocesar las imágenes (eliminación de fondo, centrado y realce)
    timer.start("Preprocesamiento y segmentación de imágenes")
    images = []
    rembg_session = None if args.no_remove_bg else rembg.new_session()

    for idx, img_path in enumerate(args.image):
        if not os.path.exists(img_path):
            logging.error(f"Imagen no encontrada: {img_path}")
            continue

        raw_img = Image.open(img_path)
        if args.enhance_image:
            enhanced_raw = enhance_input_image(raw_img)
        else:
            enhanced_raw = raw_img

        if args.no_remove_bg:
            bg_removed = enhanced_raw.convert("RGBA")
            processed_img = np.array(bg_removed.convert("RGB"))
            fg_hires = bg_removed
        else:
            logging.info(f"Segmentando fondo de '{img_path}' con rembg...")
            bg_removed = remove_background(enhanced_raw, rembg_session)
            bg_arr = np.array(bg_removed)
            low_alpha_mask = bg_arr[..., 3] < 15
            if np.any(low_alpha_mask):
                bg_arr[low_alpha_mask, 3] = 0
                bg_removed = Image.fromarray(bg_arr)
            fg_hires = resize_foreground(bg_removed, args.foreground_ratio)
            arr = np.array(fg_hires).astype(np.float32) / 255.0
            # Componer sobre fondo gris medio neutral requerido por TripoSR
            composite = arr[:, :, :3] * arr[:, :, 3:4] + (1.0 - arr[:, :, 3:4]) * 0.5
            processed_img = Image.fromarray((composite * 255.0).astype(np.uint8))

        subfolder = os.path.join(args.output_dir, os.path.splitext(os.path.basename(img_path))[0])
        os.makedirs(subfolder, exist_ok=True)
        if not args.no_remove_bg:
            processed_img.save(os.path.join(subfolder, "input_processed.png"))
        images.append((img_path, processed_img, fg_hires, subfolder))

    timer.end("Preprocesamiento y segmentación de imágenes")

    if not images:
        logging.error("No se pudo procesar ninguna imagen válida.")
        return

    # 3. Inferencia 3D y extracción de malla
    for original_path, pil_img, fg_hires, subfolder in images:
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

        # Post-procesado: limpieza de fragmentos flotantes, decimation y suavizado moderado
        if args.clean_mesh or args.smooth_iterations > 0 or args.max_faces > 0:
            timer.start("Post-procesado y optimización de geometría")
            cleaned_meshes = []
            for m in meshes:
                # 1. Conservar componentes estructurales y eliminar partículas flotantes de ruido
                if args.clean_mesh:
                    components = m.split(only_watertight=False)
                    if len(components) > 1:
                        max_verts = max(len(c.vertices) for c in components)
                        # Conservar piezas significativas (>= 4% del componente principal o >= 100 vértices)
                        threshold = max(100, int(max_verts * 0.04))
                        significant = [c for c in components if len(c.vertices) >= threshold]
                        if significant:
                            m = trimesh.util.concatenate(significant)
                        else:
                            m = max(components, key=lambda c: len(c.vertices))
                # 2. Decimation cuadrático para optimizar caras manteniendo aristas y silueta
                if args.max_faces > 0 and len(m.faces) > args.max_faces:
                    try:
                        reduction = float(1.0 - (args.max_faces / len(m.faces)))
                        logging.info(f"Optimizando caras de {len(m.faces)} a ~{args.max_faces} preservando aristas y siluetas (reducción {reduction*100:.1f}%)...")
                        m = m.simplify_quadric_decimation(reduction)
                    except Exception as e:
                        logging.warning(f"No se pudo simplificar la malla: {e}. Continuando con malla completa.")
                # 3. Suavizado Taubin: elimina rugosidades escalonadas sin desdibujar aristas
                if args.smooth_iterations > 0:
                    trimesh.smoothing.filter_taubin(m, iterations=args.smooth_iterations)
                # 4. Afinar cuerda si se detecta estructura de arco
                if args.thin_string:
                    m = refine_bowstring(m, factor_scale=args.thin_string_factor)
                m.fix_normals()
                cleaned_meshes.append(m)
            meshes = cleaned_meshes
            timer.end("Post-procesado y optimización de geometría")

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

            # Reproyección fotográfica frontal de alta definición
            if args.texture_projection and fg_hires is not None:
                timer.start("Reproyección frontal HD (nitidez 100%)")
                blended_colors = project_frontal_texture(
                    bake_output["positions"],
                    bake_output["normals"],
                    bake_output["colors"],
                    fg_hires,
                )
                timer.end("Reproyección frontal HD (nitidez 100%)")
            else:
                blended_colors = np.flipud(bake_output["colors"])

            timer.start("Exportando modelo 3D y atlas de texturas PBR")
            baked_img = Image.fromarray((blended_colors * 255.0).astype(np.uint8))
            baked_img.save(out_texture_path)

            # Generar mapa de normales multi-escala para micro-relieve y reflejos vivos
            normal_img = generate_multiscale_normal_map(baked_img, strength=args.normal_strength)
            out_normal_path = os.path.join(subfolder, "normal.png")
            normal_img.save(out_normal_path)

            # Generar mapas PBR de Oclusión Ambiental, Rugosidad y Metálico
            pbr_maps = generate_pbr_maps(
                baked_img,
                base_roughness=args.roughness_factor,
                metallic_bias=args.metallic_factor,
            )
            out_ao_path = os.path.join(subfolder, "ao.png")
            out_roughness_path = os.path.join(subfolder, "roughness.png")
            out_metallic_path = os.path.join(subfolder, "metallic.png")
            pbr_maps["ao"].save(out_ao_path)
            pbr_maps["roughness"].save(out_roughness_path)
            pbr_maps["metallic"].save(out_metallic_path)

            if args.format == "glb":
                material = PBRMaterial(
                    name="material_pbr",
                    baseColorTexture=baked_img,
                    normalTexture=normal_img,
                    occlusionTexture=pbr_maps["ao"],
                    metallicRoughnessTexture=pbr_maps["metallic_roughness"],
                    metallicFactor=1.0,
                    roughnessFactor=1.0,
                )
                export_mesh = trimesh.Trimesh(
                    vertices=meshes[0].vertices[bake_output["vmapping"]],
                    faces=bake_output["indices"],
                    vertex_normals=meshes[0].vertex_normals[bake_output["vmapping"]],
                    visual=trimesh.visual.TextureVisuals(
                        uv=bake_output["uvs"],
                        material=material,
                    ),
                    process=False,
                )
                export_mesh.export(out_mesh_path)
            else:
                xatlas.export(
                    out_mesh_path,
                    meshes[0].vertices[bake_output["vmapping"]],
                    bake_output["indices"],
                    bake_output["uvs"],
                    meshes[0].vertex_normals[bake_output["vmapping"]],
                )
                # Crear archivo .mtl correspondiente con todos los canales PBR
                mtl_path = os.path.splitext(out_mesh_path)[0] + ".mtl"
                mtl_filename = os.path.basename(mtl_path)
                tex_filename = os.path.basename(out_texture_path)
                with open(mtl_path, "w") as mtl_f:
                    mtl_f.write(
                        f"newmtl material_0\n"
                        f"Ka 1.0 1.0 1.0\n"
                        f"Kd 1.0 1.0 1.0\n"
                        f"Ks 0.35 0.35 0.35\n"
                        f"Ns 64.0\n"
                        f"d 1.0\n"
                        f"illum 2\n"
                        f"map_Kd {tex_filename}\n"
                        f"map_Bump normal.png\n"
                        f"bump normal.png\n"
                        f"norm normal.png\n"
                        f"map_Pr roughness.png\n"
                        f"map_Pm metallic.png\n"
                        f"map_ao ao.png\n"
                    )
                # Inyectar referencia mtllib y sombreado suave (s 1) en el archivo .obj
                with open(out_mesh_path, "r") as obj_f:
                    obj_content = obj_f.read()
                with open(out_mesh_path, "w") as obj_f:
                    obj_f.write(f"mtllib {mtl_filename}\nusemtl material_0\ns 1\n" + obj_content)

            timer.end("Exportando modelo 3D y atlas de texturas PBR")

            logging.info(f"🎉 Modelo guardado en: {out_mesh_path}")
            logging.info(f"🎨 Textura difusa HD guardada en: {out_texture_path}")
            logging.info(f"✨ Mapa de normales multi-escala guardado en: {out_normal_path}")
            logging.info(f"🌑 Mapa de oclusión ambiental (AO) guardado en: {out_ao_path}")
            logging.info(f"💎 Mapa de rugosidad guardado en: {out_roughness_path}")
            logging.info(f"🛡️ Mapa de metálico guardado en: {out_metallic_path}")
        else:
            timer.start("Exportando modelo con colores de vértice")
            meshes[0].export(out_mesh_path)
            timer.end("Exportando modelo con colores de vértice")
            logging.info(f"🎉 Modelo guardado en: {out_mesh_path}")

    logging.info("\n✨ ¡Proceso completado con éxito!")


if __name__ == "__main__":
    main()

