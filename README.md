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
- `modelo_3d.obj` (o `.glb`): Malla 3D completa en 360°.
- `texture.png`: Atlas de texturas UV en alta definición con reproyección frontal fotográfica.
- `normal.png`: Mapa de normales tangentes multi-escala (macro-volumen, biseles y micro-relieve).
- `ao.png`: Mapa de Oclusión Ambiental (sombras de contacto y cavidades).
- `roughness.png`: Mapa de rugosidad de superficie.
- `metallic.png`: Mapa de respuesta metálica.
- `modelo_3d.mtl`: Definición de materiales completa con canales PBR enlazados.

### Opciones avanzadas de definición y realismo

```bash
# Exportar en formato GLB con materiales PBR completos integrados
python triposr_to_3d.py monedas.jpeg --format glb

# Máxima definición geométrica (Marching Cubes a 384 y textura 2048)
python triposr_to_3d.py sword.png --mc-resolution 384 --texture-resolution 2048

# Incrementar el relieve del mapa de normales
python triposr_to_3d.py monedas.jpeg --normal-strength 3.0

# Para imágenes que ya tienen fondo transparente o neutro
python triposr_to_3d.py objeto_sin_fondo.png --no-remove-bg

# Generar además un vídeo de rotación 360°
python triposr_to_3d.py monedas.jpeg --render-video
```

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `image` | Ruta de la imagen (o imágenes) de entrada | (obligatorio) |
| `--output-dir`, `-o` | Carpeta de salida | `output_triposr` |
| `--format` | Formato de salida (`obj` o `glb`) | `obj` |
| `--mc-resolution` | Resolución Marching Cubes (256, 320 o 384) | `320` |
| `--texture-projection` | Reproyección fotográfica frontal de alta definición | `True` |
| `--normal-strength` | Intensidad del mapa de normales PBR multi-escala | `2.0` |
| `--smooth-iterations` | Iteraciones Taubin (4 conserva aristas sin redondearlas) | `4` |
| `--max-faces` | Límite de caras tras MC mediante decimation cuadrático | `60000` |
| `--texture-resolution` | Resolución del mapa de textura UV (1024 o 2048) | `1024` |
| `--metallic-factor` | Factor base metálico PBR (0.0 a 1.0) | `0.20` |
| `--roughness-factor` | Factor base de rugosidad PBR (0.1 a 0.9) | `0.55` |
| `--enhance-image` | Realce de enfoque (Unsharp Mask) antes de inferencia | `True` |
| `--chunk-size` | Chunk para optimizar VRAM (4096 para 4GB) | `4096` |
| `--render-video` | Genera vídeo MP4 en órbita 360° | `False` |

---

## 2. Método Relieve 2.5D (`image_to_3d.py`)

Genera una malla 3D extrusionada/inflada a partir de los gradientes de luminancia y frecuencias:

```bash
python image_to_3d.py monedas.jpeg --output modelo.obj --scale 150.0 --downsample 2
```
