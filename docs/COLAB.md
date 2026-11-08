# Entrenar en Colab

Todo usa tu cuenta **Gmail con el plan** (la que tiene la copia de datos en `UTEC/DEEP/MarioKart_copia`).
El notebook solo llama a `training/pipeline.py`; la lógica vive en el repo.

## Flujo (3 pasos)

1. **PC:** subir el código a Drive.
   ```bash
   python tools/apex.py sync
   ```
2. **Colab:** abrir `colab/entrenar.ipynb`, runtime **GPU (T4)**, *Ejecutar todo*. Hace solo, en orden:
   preprocesar (si falta) → copiar datos al disco local → prueba de humo → entrenar.
   Todo es reanudable: si se corta, vuelve a ejecutar.
3. **PC:** traer y evaluar el modelo.
   ```bash
   python tools/apex.py status        # avance de las corridas
   python tools/apex.py pull bc_v1    # baja el modelo a agents/bc_cnn/weights/
   python tools/apex.py eval bc_v1    # evalúa en el simulador (corre en tu PC)
   ```

Abrir el notebook: colab.research.google.com → Archivo → Subir cuaderno, o desde VS Code con la extensión de Colab.

## Presets: de menos a más intensidad

| Preset | Datos | Para qué | Tiempo aprox. |
|---|---|---|---|
| `smoke` | 6 episodios (12 min de juego) | Comprobar que el pipeline funciona en Colab | unos minutos |
| `elite` | 59 episodios, 3 h: 4 mejores equipos, 1º-2º, máx. 8 por pista | Primer modelo real y rápido de iterar | ~10 min preprocesar + ~20 min entrenar |
| `v1` | 207 episodios, 10 h: podio de todos los equipos, máx. 20 por pista | Más datos y variedad de pistas | ~30 min preprocesar + ~1 h entrenar |

Después de cada preset: `pull` y `eval` en tu PC antes de subir de nivel. Cada uno guarda en su propia carpeta de `apex_kart_runs`.

## Ruta del profesor (la principal)

| Preset | Datos | Cómo se generan |
|---|---|---|
| `teacher_r0` | 126 carreras del profesor en 21 pistas, con ruido | `python tools/overnight.py --round 0` (PC, desatendido, sube a Drive) |
| `teacher_r1` | `teacher_r0` + carreras donde conduce el estudiante y el profesor corrige (DAgger) | `python tools/overnight.py --round 1 --student agents/bc_cnn/weights/model.pt` |

Ciclo: generar datos (PC) → entrenar (Colab) → `pull` el modelo → generar la siguiente ronda con el estudiante (PC) → entrenar con todo. No necesitan preprocesado en Colab: las carreras ya salen en formato de entrenamiento.
Para dejar un proceso corriendo aunque se cierre la sesión, lanzarlo con `Start-Process` (PowerShell) y seguirlo en `logs/`.

## Ajustes (celda 1 del notebook)

| Variable | Para qué |
|---|---|
| `PRESET` | Qué configuración usar: `smoke`, `elite`, `v1`, `teacher_r0` o `teacher_r1` (`configs/train/<preset>.json`) |
| `STAGES` | `preprocess,fetch,smoke,train`. Con solo `preprocess` y runtime CPU no gastas GPU |
| `EXTRA` | Parámetros extra de entrenamiento, p. ej. `--epochs 20 --batch 128` |

## Dónde vive cada cosa

| Qué | Dónde |
|---|---|
| Código | GitHub `apex_kart` y tu PC; copia para Colab en Drive `UTEC/DEEP/apex_kart_code` |
| Datos crudos (123 GiB) | Drive `UTEC/DEEP/MarioKart_copia` (solo lectura) |
| Datos preprocesados (~22 GB) | Drive `UTEC/DEEP/apex_kart_processed/v1` |
| Corridas (checkpoints, log) | Drive `UTEC/DEEP/apex_kart_runs/<run>` |

## Si algo falla

- *"Falta el codigo en Drive"*: `python tools/apex.py sync` en tu PC.
- *"SIN GPU"*: Entorno de ejecución → Cambiar tipo → GPU.
- *Se cortó Colab*: vuelve a ejecutar todo; salta lo hecho y reanuda el entrenamiento.
- *Poca memoria o disco*: baja `batch` o `samples_per_epoch` en el preset y vuelve a ejecutar `sync`.
- El `.ipynb` se genera con `python tools/make_notebooks.py`; edita ese script, no el notebook.
