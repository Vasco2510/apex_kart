# Plan del proyecto: agente visual para SuperTuxKart

Objetivo: un agente que conduzca bien viendo **solo la imagen con HUD**, listo para que el profesor lo ejecute solo.
Plazo: ~30 días desde 2026-10-30, ~2 h/día.

**Estados:** ✅ hecho · 🔄 en curso · ⬜ pendiente
**GPU:** No = CPU basta · Opc = acelera · **Sí** = necesaria
**Quién:** H = humano · C = Claude

## Dónde estoy

**Semana 2 · día 8.** Listo: datos copiados y auditados, subconjunto v1, preprocesado rápido, entrenamiento reanudable, notebooks de Colab.
**Siguiente (H):** ejecutar `colab/entrenar.ipynb` (un solo notebook, GPU, *Ejecutar todo*). Guía en `docs/COLAB.md`.

## Mapa

```mermaid
graph LR
  F0[F0 Base] --> F1[F1 Motor]
  F0 --> F2[F2 Datos]
  F1 --> F4[F4 Evaluación]
  F2 --> F3[F3 Pipeline]
  F3 --> F5A[F5A Imitación]
  F4 --> F5A
  F1 --> F5B[F5B Refuerzo]
  F4 --> F5B
  F5A --> F6[F6 Mejora]
  F5B --> F6
  F6 --> F7[F7 Entrega]
  F7 --> F8[F8 Informe]
```

**En paralelo desde ya:** motor/evaluación (F1→F4) · datos (F2→F3) · refuerzo (F5B.1).
**Ruta crítica:** F2.1 → F2.2 → F2.4 → F3.3 → F5A.2 → F5A.3 → F6 → F7 → F8.2

---

## F0 · Base (sin GPU)

| | Paso | Entregable | Depende | Quién |
|---|---|---|---|---|
| ✅ F0.2 | Entorno Python 3.11 | `doctor --engine` OK | — | H |
| ✅ F0.3 | Decidir cómputo | Colab con plan Google AI Pro | — | H |
| 🔄 F0.1 | Repo y compañero | Repo `apex_kart` creado; **falta invitar al compañero** | — | H |
| 🔄 F0.4 | Dudas del enunciado | Respuestas en `ENUNCIADO.md`; falta criterio de nota y formato de entrega | — | H |
| 🔄 F0.5 | Convenciones | Carpetas, datos fuera de git, semillas, registro | F0.1 | H+C |

## F1 · Motor y línea base (sin GPU)

| | Paso | Entregable | Depende | Quién |
|---|---|---|---|---|
| ✅ F1.1 | Listar pistas y karts | `results/assets.json` | F0.2 | C |
| ✅ F1.2 | Velocidad del simulador | ~2.5× tiempo real | F1.1 | C |
| ✅ F1.3 | Agente trivial | `agents/const_accel` | F1.1 | C |
| ✅ F1.4 | Línea base | 0 de 12 carreras terminadas | F1.3 | C |

## F2 · Datos (sin GPU)

| | Paso | Entregable | Depende | Quién |
|---|---|---|---|---|
| ✅ F2.1 | Datos de todos los grupos | Copia en tu Drive: 928 episodios, 123.5 GiB | F0.1 | **H** |
| ✅ F2.2 | Auditar | `results/audit_dataset.md` (41 h, 24 grupos) | F2.1 | C |
| ⬜ F2.3 | Grabar datos propios | Episodios que cubran los huecos | F2.2 | H |
| 🔄 F2.4 | Congelar split | Validación en pistas no vistas: `configs/train/bc_v1.json` | F2.2 | H+C |

## F3 · Pipeline de datos (sin GPU)

| | Paso | Entregable | Depende | Quién |
|---|---|---|---|---|
| 🔄 F3.1 | Explorar | `results/exploracion_muestras.md` | F2.2 | C |
| ✅ F3.2 | Alinear imagen y acción | Verificado en las muestras | F2.2 | C |
| ✅ F3.3 | Formato compacto | `tools/preprocess.py` | F3.2 | C |
| ✅ F3.4 | Curar datos | `results/subset_v1.txt`: 207 episodios, podio, ≤20 por pista | F3.1 | H+C |

## F4 · Evaluación (sin GPU)

| | Paso | Entregable | Depende | Quién |
|---|---|---|---|---|
| 🔄 F4.1 | Métricas | Terminadas, tiempo, posición, distancia | F0.4 | H+C |
| 🔄 F4.2 | Pistas de evaluación | `configs/eval_set.json` (provisional) | F2.4 | H+C |
| ✅ F4.3 | Arnés automático | `tools/evaluate.py` | F1.3 | C |
| ⬜ F4.4 | Medir ruido | Varianza entre semillas | F4.3 | C |

## F5A · Imitación

| | Paso | Entregable | Depende | GPU | Quién |
|---|---|---|---|---|---|
| ✅ F5A.1 | Prueba de humo | Entrena y el agente corre | F3.3 | Opc | C |
| 🔄 F5A.2 | Primer modelo real | Notebook listo: `colab/entrenar.ipynb`; **falta ejecutarlo (H)** | F5A.1, F2.4 | **Sí** | H+C |
| ⬜ F5A.3 | Medir conduciendo | Tabla del arnés | F5A.2, F4.3 | Opc | C |

## F5B · Refuerzo

| | Paso | Entregable | Depende | GPU | Quién |
|---|---|---|---|---|---|
| ✅ F5B.1 | Entorno y recompensa | `training/rewards.py` | F1.2 | No | C |
| ⬜ F5B.2 | Profesor con estado interno (opcional) | Política que conduce bien | F5B.1 | Opc | C |
| ⬜ F5B.3 | Estudiante desde imagen | Política de píxeles | F5B.1, F4.3 | **Sí** | H+C |

Con ~1.5× tiempo real, RL puro desde píxeles es caro: se decide el día 14.

## F6 · Mejora (GPU según el caso)

| | Paso | Entregable | Depende | Quién |
|---|---|---|---|---|
| ⬜ F6.1 | Analizar fallos | Dónde falla el mejor agente | F5A.3 | C |
| ⬜ F6.2 | Una mejora por vez | Más datos, aumento, corrección con datos propios del agente | F6.1 | H+C |
| ⬜ F6.3 | Comparar | Tabla antes/después, mismas semillas | F6.2, F4.4 | C |

## F7 · Entrega (sin GPU)

| | Paso | Entregable | Depende | Quién |
|---|---|---|---|---|
| ⬜ F7.1 | Agente autocontenido | Carpeta con `agent.json`, pesos y pesos portables; usa GPU si hay, si no CPU | F6 | C |
| ⬜ F7.2 | Probar en máquina limpia | `play` y `race` desde cero | F7.1 | H |
| ⬜ F7.3 | Latencia y robustez | Tiempo por acción, pistas nuevas | F7.1 | C |
| ⬜ F7.4 | Congelar versión | Etiqueta en git + dependencias | F7.2, F7.3 | H+C |

## F8 · Informe (sin GPU)

| | Paso | Entregable | Depende | Quién |
|---|---|---|---|---|
| ⬜ F8.1 | Bitácora | Experimentos y tablas | F4.3 | H+C |
| ⬜ F8.2 | Informe o presentación | Formato según F0.4 | F6.3, F8.1 | H |

---

## Calendario

| Semana | Foco | Pasos |
|---|---|---|
| 1 (días 1–7) | Base y datos | F0, F1, F2.1–F2.2, F4.1–F4.3 |
| 2 (días 8–14) | Primer modelo | F2.4, F3, F5A. **Corte día 14: un agente que complete vueltas** |
| 3 (días 15–21) | Mejora | F6, F2.3, F4.4; F5B solo si el agente base ya anda |
| 4 (días 22–30) | Cierre | F6.3, F7, F8. Últimos 3 días solo para empaquetar |

**Reglas:** sin agente que complete vueltas el día 14, se descarta RL. Nada nuevo después del día 26.

## GPU de un vistazo

- **Sin GPU (puedes avanzar ya):** F0, F1, F2, F3, F4, F7, F8, F5B.1
- **Opcional:** F5A.1, F5A.3, F5B.2
- **Necesaria:** F5A.2, F5B.3, entrenamiento de F6.2
