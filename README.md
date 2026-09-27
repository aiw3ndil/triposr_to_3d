# script-to-3d

Tools to convert 2D images into 3D models (OBJ/GLB/PLY).

Features two methods:
1. **TripoSR (360° Generative AI)**: Reconstructs full 360-degree geometry and bakes automatic UV texture maps from a single photograph.
2. **2.5D Relief Estimation (`image_to_3d.py`)**: Generates organic relief and volume using multi-scale frequency decomposition and depth maps.

---

## 1. TripoSR Method (Full 360° 3D Reconstruction)

Powered by the LRM neural network from **Stability AI** and **Tripo AI**.

### Environment Activation

```bash
source venv/bin/activate
```

### Basic Usage

```bash
python triposr_to_3d.py monedas.jpeg
```

This will generate the `output_triposr/monedas/` folder containing:
- `modelo_3d.obj` (or `.glb`): Full 360° 3D mesh.
- `texture.png`: High-definition UV texture atlas with frontal photographic reprojection.
- `normal.png`: Multi-scale tangent normal map (macro-volume, bevels, and micro-relief).
- `ao.png`: Ambient Occlusion map (contact shadows and cavities).
- `roughness.png`: Surface roughness map.
- `metallic.png`: Metallic response map.
- `modelo_3d.mtl`: Full material definition with linked PBR channels.

### Advanced Quality and Realism Options

```bash
# Export in GLB format with integrated full PBR materials
python triposr_to_3d.py monedas.jpeg --format glb

# Maximum geometric definition (Marching Cubes at 384 and 2048 texture)
python triposr_to_3d.py sword.png --mc-resolution 384 --texture-resolution 2048

# Increase normal map relief intensity
python triposr_to_3d.py monedas.jpeg --normal-strength 3.0

# For images that already have a transparent or neutral background
python triposr_to_3d.py objeto_sin_fondo.png --no-remove-bg

# Also generate a 360° orbital rotation video
python triposr_to_3d.py monedas.jpeg --render-video
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `image` | Path to the input image (or images) | (required) |
| `--output-dir`, `-o` | Output directory | `output_triposr` |
| `--format` | Output format (`obj` or `glb`) | `obj` |
| `--mc-resolution` | Marching Cubes resolution (256, 320, or 384) | `320` |
| `--texture-projection` | High-definition frontal photographic reprojection | `True` |
| `--normal-strength` | Multi-scale PBR normal map intensity | `2.0` |
| `--smooth-iterations` | Taubin smoothing iterations (4 preserves sharp edges without rounding) | `4` |
| `--max-faces` | Post-MC face limit via quadric decimation | `60000` |
| `--texture-resolution` | UV texture map resolution (1024 or 2048) | `1024` |
| `--metallic-factor` | Base PBR metallic factor (0.0 to 1.0) | `0.20` |
| `--roughness-factor` | Base PBR roughness factor (0.1 to 0.9) | `0.55` |
| `--enhance-image` | Sharpening enhancement (Unsharp Mask) prior to inference | `True` |
| `--chunk-size` | Chunk size to optimize VRAM (4096 for 4GB) | `4096` |
| `--render-video` | Generates a 360° orbiting MP4 video | `False` |

---

## 2. 2.5D Relief Method (`image_to_3d.py`)

Generates an extruded/inflated 3D mesh from luminance gradients and frequencies:

```bash
python image_to_3d.py monedas.jpeg --output modelo.obj --scale 150.0 --downsample 2
```
