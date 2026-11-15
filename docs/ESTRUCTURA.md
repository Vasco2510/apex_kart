# Estructura y convenciones

Una regla: **código, configuración y resultados pequeños van a git; datos y pesos no** (viven en Drive o en carpetas ignoradas).

| Carpeta | Qué hay | Git |
|---|---|---|
| `src/stk_agent/` | Librería del curso (motor, evaluación). No tocar salvo error | sí |
| `agents/<agente>/` | Agentes entregables (`agent.json`, clase). `weights/<corrida>/<archivo>.pt` = pesos | código sí, pesos no |
| `training/` | Entrenar y generar datos (`teacher`, `dagger`, `train_bc`, `pipeline`) | sí |
| `tools/` | Utilidades: `apex.py` (sync/pull/eval), `overnight.py`, `bench_sim.py`, `evaluate.py` | sí |
| `configs/train/<preset>.json` | Una receta de entrenamiento por archivo | sí |
| `colab/` | Notebook generado por `tools/make_notebooks.py` (no editar a mano) | sí |
| `results/` | Tablas pequeñas: `evals/` (resumen por evaluación), listas de episodios, informes | sí |
| `docs/` | `PLAN_PROYECTO` (estado), `AVANCE` (bitácora), `COLAB`, `GUIA_HUMANO`, `ENUNCIADO` | sí |
| `data/`, `runs/`, `logs/` | Datos locales, corridas y evaluaciones, registros de procesos | no |
| `contexto/` | Muestras del profesor (material de apoyo) | solo metadatos |

## Nombres (una ronda = una familia)

- Datos: Drive `apex_kart_processed/teacher_r<N>` · preset `configs/train/teacher_r<N>.json` · corrida `bc_teacher_r<N>`.
- Pesos locales: `agents/bc_cnn/weights/bc_teacher_r<N>/model.pt` (mejor) y `ep20.pt`, etc. `weights/model.pt` = último descargado.
- Evaluaciones: `runs/eval_<corrida>_<pesos>_<fecha>/` y su resumen versionable en `results/evals/`.
- Cada paso importante: actualizar estado en `PLAN_PROYECTO.md` y una línea en `AVANCE.md`.

## Ciclo

`overnight.py --round N` (PC) → `apex.py sync` → Colab (`PRESET=teacher_r<N>`) → `apex.py pull bc_teacher_r<N>` → `apex.py eval bc_teacher_r<N>` → siguiente ronda con `--student agents/bc_cnn/weights/bc_teacher_r<N>/model.pt`.

## Entrega

Carpeta del agente (`agent.json` + clase + pesos `.pt`) que el profesor ejecuta junto a otros modelos con `race`/`league`. Debe correr sin red y en CPU o GPU. Se prepara en F7.
