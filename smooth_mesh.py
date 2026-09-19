#!/usr/bin/env python3
"""
smooth_mesh.py — Suavizado rápido de mallas 3D (.obj / .glb) usando el filtro Taubin.

Elimina la rugosidad (piel de naranja) y micro-asperezas sin deformar ni encoger el modelo,
conservando las coordenadas UV y mapas de texturas.
"""

import argparse
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
venv_python = os.path.join(current_dir, "venv", "bin", "python")
if os.path.exists(venv_python) and os.path.realpath(sys.executable) != os.path.realpath(venv_python):
    os.execv(venv_python, [venv_python] + sys.argv)

import trimesh
import trimesh.smoothing


def main():
    parser = argparse.ArgumentParser(description="Aplica suavizado Taubin a un archivo 3D (.obj o .glb).")
    parser.add_argument("input", type=str, help="Ruta de la malla 3D de entrada.")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Ruta de salida (por defecto sobreescribe el original o usa _smoothed).",
    )
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=4,
        help="Número de iteraciones del filtro Taubin (default: 4 para preservar aristas y detalles sin redondearlas).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=True,
        help="Conservar solo el componente conexo más grande (eliminar motas flotantes).",
    )

    args = parser.parse_args()

    print(f"📦 Cargando malla: {args.input}...")
    mesh = trimesh.load(args.input, process=False)

    if args.clean and isinstance(mesh, trimesh.Trimesh):
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            print(f"🧹 Eliminando {len(components) - 1} fragmento(s) flotante(s)...")
            mesh = max(components, key=lambda c: len(c.vertices))

    print(f"✨ Aplicando suavizado Taubin ({args.iterations} iteraciones)...")
    trimesh.smoothing.filter_taubin(mesh, iterations=args.iterations)
    mesh.fix_normals()

    out_path = args.output if args.output else args.input
    print(f"💾 Guardando modelo suavizado en: {out_path}...")
    mesh.export(out_path)
    print("🎉 ¡Completado con éxito!")


if __name__ == "__main__":
    main()
