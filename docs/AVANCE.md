# Avance y bloqueos

Bitácora de lo hecho (más reciente al final) y de lo que solo puede resolver una persona. El estado por paso está en `PLAN_PROYECTO.md`.

## Registro

| Fecha | Qué se hizo | Resultado / dónde |
|-------|-------------|--------------------|
| 2026-10-30 | F0.2 entorno Python 3.11 (venv fuera del repo: `C:\Users\Yuri Abel\venvs\apex`), `doctor --engine` OK | Se corrigió `roster.py` corrupto. 168 pasos/s a 60 Hz ≈ 2.8× tiempo real (RGB + estado, sin grabar) |
| 2026-10-30 | F1.1 activos | `results/assets.json`: 21 pistas, 18 karts (coincide con lista del profesor) |
| 2026-10-30 | F1.2/F1.3 velocidad y agente trivial | `agents/const_accel`; `play` con agente en el bucle ≈ 2.5× tiempo real, latencia ≈ 1 ms. `STKPixelEnv` 34–44 pasos de política/s (con otros procesos en paralelo) |
| 2026-10-30 | F2.2/F3.1 auditoría y exploración de las 4 muestras | `results/audit_muestras.json`, `results/exploracion_muestras.md`: accel≈1 siempre, drift ~39 %, nitro 2.5 %, rescue 0 % |
| 2026-10-30 | F3.3 preprocesado | `tools/preprocess.py` → `data/processed/` (160×90, stride 2): 2.2 GB/hora (casi igual que el crudo; usar `--down 8` para reducir) |
| 2026-10-30 | F4.3 arnés | `tools/evaluate.py` + `configs/eval_set.json` (provisional) |
| 2026-10-30 | F5A.1 humo del entrenamiento | `training/train_bc.py`, `agents/bc_cnn/`: pérdida 1.22→1.06 en 6 épocas (CPU, 4 episodios ≈ 0.13 h). El agente carga el checkpoint y corre en `play` (23 ms/acción en CPU con carga), pero conduce mal (17 m en 60 s): esperable con tan pocos datos; confirma que el pipeline funciona, no que el modelo sirva |
| 2026-10-30 | F1.4 línea base | `results/baseline_const_accel/`: acelerar recto = 0 % de carreras terminadas, posición 8/8, 223 m de media en 12 carreras (6 pistas × 2 semillas). Resultado idéntico entre semillas: el simulador es determinista con una política determinista |
| 2026-10-30 | F5B.1 entorno + recompensa | `training/rewards.py`, `tools/check_env.py` |
| 2026-11-02 | F2.1 tamaño real del Drive del profesor (`rclone size`) | 24 carpetas de grupos, 6541 archivos, 123.5 GiB (~930 episodios, ~38 h de conducción). La carpeta es pública y la cuenta Gmail de pago la ve con `root_folder_id`; copia a `UTEC/DEEP/MarioKart_copia` en curso |
| 2026-11-02 | F2.2 auditoría completa por metadatos | 928 episodios válidos, 41.2 h, 24 grupos. `results/audit_dataset.md`, `results/episodios.csv`. 7 grupos anidan sus episodios una carpeta más adentro (176 episodios): el primer informe los contó como ~70 «grupos» falsos; corregido en `tools/audit_meta.py` |
| 2026-11-02 | F3.4 subconjunto v1 | `tools/make_subset.py` → `results/subset_v1.txt`: 207 episodios (10.1 h), podio, dificultad ≥2, máx. 20 por pista |
| 2026-11-02 | F2.1 copia completa lanzada | `rclone copy` servidor a servidor hacia `UTEC/DEEP/MarioKart_copia`; log `rclone_copia_full.log` (fuera de git) |
| 2026-11-06 | F2.1 copia completa verificada | `MarioKart_copia`: 6539 objetos, 123.514 GiB. Los 207 episodios del subconjunto existen en la copia |
| 2026-11-06 | F3.3 preprocesado v2 | `tools/preprocess.py` con ffmpeg: ~3× más rápido, diferencia ≤1 nivel de color con el método anterior; reanudable; acepta `--list`. Arregla nombres con `＞` (ancho completo en Windows) y con espacio inicial |
| 2026-11-06 | F5A.2 preparación | `training/train_bc.py` (AMP, aumento de brillo, validación en pistas no vistas, reanuda sola, guarda mejor modelo), `configs/train/v1.json`, `training/pipeline.py` (preprocesar → copiar → humo → entrenar, reanudable, probado de punta a punta en local), `colab/entrenar.ipynb` (1 notebook, 3 celdas de código), `tools/apex.py` (sync/status/pull/eval) |

| 2026-11-06 | Estrategia por etapas | Calidad por equipo (de `results/episodios.csv`): `piedad_con_los_de_elmt` 1.20 de posición media, pero 35 de 49 episodios en `lighthouse`. Presets nuevos `smoke` (6 ep.) y `elite` (59 ep., 3 h); `v1` queda como segunda etapa. `tools/make_subset.py` acepta `--groups` y `--max-pos` |

| 2026-11-07 | F5A.2 `smoke` en Colab OK | GPU T4, ~3700 muestras/s, checkpoints en Drive. `pull smoke` + `eval` en el PC funcionan (el modelo de 2 épocas no avanza: 104 m en lighthouse, igual que la línea base, como se esperaba). Con esa velocidad `elite` sube a 40 épocas × 60k muestras (~20 min) |

| 2026-11-07 | F5A.2 `elite` entrenado | Mejor modelo por pérdida de validación = época 1 (casi sin entrenar). Época 15 termina `lighthouse` (880 m, 64 s) pero falla en pistas no vistas (187 m en `candela_city`). La pérdida de validación predice mal la conducción |
| 2026-11-08 | Profesor privilegiado | `training/teacher.py` (pure pursuit sobre `path_nodes`): termina 21/21 pistas con 2 vueltas tras arreglar un bucle de rescate (botón mantenido). Es correcto pero lento (posición 5-8 de 8). Variantes de derrape/freno/nitro: ±5 %, sin salto (el kart no recoge nitro); se mantiene el profesor base |
| 2026-11-08 | Datos del profesor | `training/dagger.py` (ruido OU para recuperaciones, karts variados, sube a Drive y borra local) + `tools/overnight.py`. Corre `teacher_r0`: 126 carreras (21 pistas × 6 semillas × 2 vueltas) → `apex_kart_processed/teacher_r0` |
| 2026-11-08 | Episodios incompletos | 36 de 928 episodios del profesor no tienen `data.h5`/`rgb.mp4` (subidos incompletos; casi todos `IAn & 6Soto`): `results/episodios_incompletos.txt`. Fuera de los subconjuntos. El preprocesado ya omite episodios rotos |

## Bloqueos que solo puede resolver el humano

1. **F5B.3** Ejecutar `colab/entrenar.ipynb` con `PRESET='teacher_r0'` cuando termine la generación nocturna (pasos en `docs/COLAB.md`).
2. **F0.1** Invitar al compañero al repo; **F0.2** que él repita la instalación (Python 3.11).
3. **F0.4** Preguntar al profesor el criterio de nota y el formato de entrega.
