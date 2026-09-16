# script-to-3d

Herramientas para convertir imágenes 2D en modelos 3D (OBJ/GLB/PLY).

Cuenta con dos métodos:
1. **TripoSR (IA Generativa 360°)**: Reconstruye la geometría completa en 360 grados y hornea mapas de texturas UV automáticos a partir de una sola foto.
2. **Estimación de Relieve 2.5D (`image_to_3d.py`)**: Genera relieve y volumen orgánico mediante descomposición multiescala de frecuencias y mapas de profundidad.

---

## 1. Método TripoSR (Reconstrucción 3D Completa 360°)

Utiliza la red neuronal LRM de **Stability AI** y **Tripo AI**.

### Activación del entorno

```bash
source venv/bin/activate
```

### Uso básico

```bash
python triposr_to_3d.py monedas.jpeg
```

Esto generará la carpeta `output_triposr/monedas/` con:
- `modelo_3d.obj`: Malla 3D completa en 360°.
- `texture.png`: Atlas de texturas UV horneado.
- `input_processed.png`: Imagen con fondo segmentado neutral.

### Opciones avanzadas de TripoSR

```bash
# Guardar en formato GLB (ideal para web y visores 3D directos)
python triposr_to_3d.py monedas.jpeg --format glb

# Ajustar resolución de textura (ej. 2048)
python triposr_to_3d.py monedas.jpeg --texture-resolution 2048

# Generar además un vídeo de rotación 360°
python triposr_to_3d.py monedas.jpeg --render-video

# Para imágenes que ya tienen fondo transparente o neutro
python triposr_to_3d.py objeto_sin_fondo.png --no-remove-bg
```

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `image` | Ruta de la imagen (o imágenes) | (obligatorio) |
| `--output-dir`, `-o` | Carpeta de salida | `output_triposr` |
| `--format` | Formato (`obj` o `glb`) | `obj` |
| `--bake-texture` | Hornear textura UV (`True` por defecto) | `True` |
| `--no-bake-texture` | Exportar solo colores por vértice | `False` |
| `--texture-resolution` | Resolución del mapa de textura | `1024` |
| `--chunk-size` | Chunk para optimizar VRAM (4096 para 4GB) | `4096` |
| `--mc-resolution` | Resolución de Marching Cubes | `256` |
| `--render-video` | Genera vídeo MP4 en órbita 360° | `False` |

---

## 2. Método Relieve 2.5D (`image_to_3d.py`)

Genera una malla 3D extrusionada/inflada a partir de los gradientes de luminancia y frecuencias:

```bash
python image_to_3d.py monedas.jpeg --output modelo.obj --scale 150.0 --downsample 2
```
