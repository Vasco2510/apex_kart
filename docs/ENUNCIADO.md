# Enunciado y material del curso: Deep Learning (UTEC)

Fuente: mensajes del profesor Victor Flores Benites (Discord), ordenados y transcritos.
Los campos [COMPLETAR] faltan; no inventar su contenido.

## 1. Instalación y verificación (Linux/Windows)

```
conda create -n stk python=3.11 -y
conda activate stk
python -m pip install -e .
python -m stk_agent doctor --engine --config configs/test_collect.json
```

Material base: `MarioCar.zip` (librería `stk_agent`), ya descomprimido en la raíz del repo.

## 2. Recolección de datos

```
python -m stk_agent collect --config configs/collect.json --keymap configs/keymap.yaml --group NOMBRE_GRUPO --driver NOMBRE_ALUMNO --out data/local
```

- La carrera se graba hasta que el jugador presiona `Esc` o se alcanza el tiempo máximo de seguridad.
- Los controles se modifican en `configs/keymap.yaml`.
- Cada ejecución genera un episodio nuevo dentro de `data/local/`.
- Varias carreras consecutivas: agregar `--episodes N`.

## 3. Configuración de partida (`configs/collect.json`)

- `kart` (personaje): `random`, `adiumy`, `amanda`, `beastie`, `emule`, `gavroche`, `gnu`, `hexley`, `kiki`, `konqi`, `nolok`, `pidgin`, `puffy`, `sara_the_racer`, `sara_the_wizard`, `suzanne`, `tux`, `wilber`, `xue`
- `track` (mapa): `abyss`, `black_forest`, `candela_city`, `cocoa_temple`, `cornfield_crossing`, `fortmagma`, `gran_paradiso_island`, `hacienda`, `lighthouse`, `mines`, `minigolf`, `olivermath`, `ravenbridge_mansion`, `sandtrack`, `scotland`, `snowmountain`, `snowtuxpeak`, `stk_enterprise`, `volcano_island`, `xr591`, `zengarden`
- `difficulty` (bots): 0 = Novice, 1 = Intermediate, 2 = Expert, 3 = Best (según el profesor)
- `bot_kart`: `"random"` para personajes aleatorios en los rivales

## 4. Proyecto Parcial (cerrado)

Fecha límite: 27 de setiembre, 11:59 pm.

- Cada grupo debía recolectar al menos 1 hora de conducción con la librería.
- Objetivo: un conjunto de demostraciones diverso y útil para entrenar modelos.
- Usar distintos circuitos y, en lo posible, distintos personajes.
- Conducir bien: completar las carreras y seguir la pista.
- No repetir el mismo tipo de carrera.
- Experimentar con la dinámica del juego: adelantar, competir con bots, usar objetos, nitro, drift, frenar, recuperarse de errores.
- Son válidos errores, salidas de pista o choques ocasionales: también aportan ejemplos útiles.
- Entrega: carpeta de Google Drive compartida (ID `1-XlHVsB2ntmou9OYiioy9fwpisnZPbTH`), en una subcarpeta con el nombre del grupo.

Resultado: la carpeta contiene cientos de episodios de todos los grupos. Cada episodio contiene `meta.json`, `data.h5`, `rgb.mp4`, `track.npz`, `checksums.json`, `READY.json` (y `rgb.mp4.ffmpeg.log`).

## 5. Proyecto final

Declaración del profesor: "vale todo, toda tecnica toda estratgia o cheat con tal de cosneguir un mejor performance", sin restricciones de método.

- Enunciado oficial: No tenemos enunciado oficial, pero el objetivo es entrenar un agente que conduzca bien en el juego. Se puede usar cualquier técnica de aprendizaje profundo, aprendizaje por refuerzo, o incluso heurísticas y trucos. Tenemos informacion en el readme sobre de que trata el proyecto. 
- Criterio de evaluación (confirmado por el usuario, 2026-11-15): **primero** puntúa terminar las carreras base (pistas conocidas); **luego** se añaden pistas nuevas y ahí está el puntaje especial para llegar a la nota máxima (20). El profesor ejecuta los modelos de todos los grupos juntos y compara cuáles rinden mejor (competencia entre agentes).
- Formato de entrega (¿`agent.json` más checkpoint? ¿informe? ¿presentación?): No hay formato de entrega pero si deberiamos tener un agente o model olisto para que el profe l oejecute independientemente cuando requeira hacer competir a los agentes entrados por los alumnos. 
- Restricciones de inferencia (CPU o GPU, tiempo por acción): no hay
- Pistas de evaluación: se usarán pistas conocidas y también pistas nuevas que el profesor no ha publicado (podrían estar en internet; eventualmente se podrían usar para entrenar). **Hay que generalizar a pistas no vistas.**
- Recursos/herramientas del agente: sin limitaciones (confirmado por el usuario). Intérprete y librerías: libres, pero el intérprete del manifest (`python`) debe tener `stk-agent` instalado.
- Formato de entrega: no especificado; lo seguro es un modelo o sus pesos que el profesor ejecute junto a los de otros grupos. Supuesto de trabajo = carpeta con `agent.json` + clase + pesos, más exportación portable de los pesos para convertir sin reentrenar.
- Plazo del proyecto final: ~30 días desde 2026-10-30, ~2 h/día del usuario.
- Tamaño del grupo: 2 personas
- ¿Se permite usar los datos de todos los grupos?: totalmmente si. De hecho es lo recomendado

## 6. Notas de consistencia (verificar contra el código)

- El profesor lista `difficulty` hasta 3, pero `RaceSettings` en `config.py` solo acepta 0, 1 o 2. **Decisión: manda el código disponible (0–2).**
- Los identificadores reales de pistas y karts los devuelve `python -m stk_agent assets`; esa salida manda sobre las listas de arriba.
## 7. Hallazgos al ejecutar el código (2026-10-30)

- Los episodios de las muestras se grabaron con `difficulty: 3` (y 2) y librería 0.5.0, aunque `RaceSettings` actual solo acepta 0–2. `Episode`/`audit` los leen sin problema (usan el dict de meta, no `RaceSettings`). Para evaluar/entrenar en el simulador, usar 0–2.
- `src/stk_agent/roster.py` tenía un carácter basura (`ñ`) al inicio que rompía el motor (`SyntaxError`); se restauró desde `MarioCar.zip`. Es el único archivo que difería del zip.
- Los humanos de las 3 muestras con fase de carrera terminaron en posiciones 6–8 de 8 contra bots difficulty 3: los datos humanos no son expertos; conviene curarlos (F3.4).
- Tamaño real de los datos: ≈3.2 GiB por hora de conducción cruda (mp4+h5); los episodios van de 12 s a 182 s (11–179 MiB).
