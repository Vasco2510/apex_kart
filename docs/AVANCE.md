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

## Bloqueos que solo puede resolver el humano

1. **F2.1** Datos completos del Drive del profesor: copiarlos a tu Drive (ver `docs/GUIA_HUMANO.md` §5) y procesarlos en Colab (`docs/COLAB.md`). Hasta entonces F2.2, F2.4, F3.4 y F4.2 quedan con las 4 muestras.
2. **F0.1** Invitar al compañero al repo; **F0.2** que él repita la instalación (en Windows hay que usar Python 3.11: pygame no compila en 3.14).
3. **F5A.2** Primer entrenamiento real: necesita GPU (Colab) y los datos procesados.
