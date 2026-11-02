# Guía para ti (lo que solo puede hacer un humano)

Orden recomendado. Cada bloque indica el paso de `PLAN_PROYECTO.md` que desbloquea. Tomarse el tiempo aquí ahorra fricción después.

## 1. GitHub (F0.1) — ahora mismo

La sesión de `gh` no estaba iniciada, así que el repo **no está creado todavía**. El código local ya tiene `git init` y un commit inicial.

1. Inicia sesión (abre navegador, tú pones tus credenciales):
   ```bash
   gh auth login
   ```
   Elige GitHub.com → HTTPS → "Login with a web browser".
2. Crea el repo **privado** y súbelo (desde esta carpeta):
   ```bash
   gh repo create apex_kart --private --source=. --remote=origin --push
   ```
   Si el repo ya existe con otro nombre: `gh repo rename apex_kart` (mantiene el contenido y la URL antigua redirige). Déjalo privado: es trabajo del curso y el material base es del profesor.
3. Invita a tu compañero: repo → Settings → Collaborators → Add people.
4. Revisa en `git config user.name` y `user.email` que son los que quieres mostrar en los commits (tienes configurado `Vasco2510` como nombre).

## 2. Entorno (F0.2)

1. Instala Miniconda o Miniforge si no lo tienes.
2. **Usa Python 3.11.** Tu Python por defecto es 3.14, que la librería no recomienda; crea un entorno aparte:
   ```bash
   conda create -n stk python=3.11 -y
   conda activate stk
   python -m pip install --upgrade pip
   python -m pip install -e .
   python -m stk_agent doctor --engine --config configs/test_collect.json
   ```
3. En Windows el primer `doctor --engine` puede preparar el runtime nativo. Si falla, guarda el mensaje completo y pásamelo.
4. Repite el proceso en tu compañero: ambos deben llegar al mismo resultado.

## 3. Decisión de cómputo (F0.3)

No hay GPU NVIDIA detectada en esta máquina (`nvidia-smi` no existe aquí). Necesitas decidir dónde entrenar antes de F5A.2.

1. Averigua qué tienes: tarjeta de video de tu PC y la de tu compañero (Administrador de tareas → Rendimiento → GPU).
2. Averigua opciones: laboratorio/cluster de UTEC, cuotas gratuitas de notebooks en la nube, o alquiler por horas.
3. Anota en un archivo (puedes pedirme que lo cree): máquina elegida, VRAM, horas disponibles por semana, costo si lo hay.
4. Pregunta a tus compañeros de otros grupos qué usan; suele haber información útil gratis.

Elige con una regla: que se pueda **reanudar** un entrenamiento interrumpido y que los datos sean accesibles desde ahí.

## 4. Aclarar el enunciado con el profesor (F0.4) — mayormente resuelto, ver ENUNCIADO.md §5

Pregunta (por Discord) y copia las respuestas a `docs/ENUNCIADO.md`:

1. ¿Cómo se califica? ¿Ranking de `league`, informe, presentación, todo?
2. ¿Qué pistas/semillas/dificultad usará para evaluar? ¿Incluye pistas que no están en los datos?
3. ¿Cómo ejecutará los agentes: CPU o GPU, y hay algún límite de tiempo por acción? (el código usa 10 s de timeout por defecto)
4. ¿Qué formato exacto de entrega: carpeta con `agent.json` + pesos, ¿dónde y hasta cuándo?
5. ¿La dificultad 3 existe en la versión instalada? (el código solo acepta 0–2)
6. ¿Puede el agente usar librerías extra (torch, etc.) instaladas en su entorno? ¿Cuál es el intérprete que usará?

Sin la respuesta 3 y 6 no conviene fijar arquitectura.

## 5. Datos (F2.1, F2.3)

1. **No descargues todo.** Cada episodio pesa ~85 MB (rgb.mp4 ~65 MB + data.h5 ~19 MB) y hay cientos: decenas de GB. Tu disco libre ahora: C: ~12 GB, Z: ~7 GB. Baja solo 2–3 episodios de muestra para desarrollar en local.
2. Averigua el tamaño real: en Drive, clic derecho en la carpeta `MarioKart` → "Información de la carpeta" / Detalles. Anota el total y el número de episodios.
3. Estrategia: **datos crudos en la nube, trabajo local con muestras**. Los archivos "compartidos conmigo" no cuentan en tu cuota hasta que los copias.
   - **Acceso directo (lo que ya tienes en `Mi unidad/UTEC/DEEP/MarioKart`):** sirve para explorar y para probar en Colab, pero no es una copia: si el dueño quita el acceso o mueve carpetas, desaparece, y Drive limita las lecturas masivas de archivos ajenos ("demasiados usuarios han visto o descargado este archivo"). Verifícalo con la celda de `docs/COLAB.md` §0 antes de fiarte.
   - **Copia real a tu Drive (recomendado), sin pasar por tu PC:** `rclone` copia de servidor a servidor. Primero activa el plan de 5 TB (con el gratuito de 15 GB no cabe). Pasos:
     1. `winget install Rclone.Rclone` y reabre la terminal.
     2. `rclone config` → `n` (nuevo) → nombre `gdrive` → tipo `drive` → deja client_id/secret vacíos → scope `1` (acceso total) → el resto por defecto → "Use web browser" = `y` e inicia sesión con tu cuenta UTEC.
     3. Comprueba que ve la carpeta compartida: `rclone lsd "gdrive,shared_with_me=true:"` (debe aparecer `MarioKart`).
     4. Prueba con un grupo: `rclone copy "gdrive,shared_with_me=true:MarioKart/piedad_con_los_de_elmt" "gdrive:UTEC/DEEP/MarioKart_copia/piedad_con_los_de_elmt" -P --transfers 8`
     5. Si funciona, copia todo: `rclone copy "gdrive,shared_with_me=true:MarioKart" "gdrive:UTEC/DEEP/MarioKart_copia" -P --transfers 8` (Drive limita ~750 GB/día de copia).
     6. Comparte `MarioKart_copia` con tu compañero (permiso de lector) y que él cree un acceso directo; así no se duplica el almacenamiento.
   - Si prefieres no usar `rclone`, la alternativa es copiar dentro de Colab con `cp -r` desde el montaje de Drive (más lento y se corta si se desconecta la sesión).
4. Decide **dónde viven los datos**: nunca dentro de git. Local: `data/` (ya ignorado), solo muestras.
3. Cuando yo te entregue el informe de auditoría (F2.2), graba episodios nuevos que cubran los huecos. Reglas del profesor: completar carreras, variar pistas y karts, incluir adelantamientos, nitro, drift, recuperación de errores.

## 6. Decisiones que conviene tomar con tu compañero

1. Reparto de roles: quién cubre datos, quién evaluación, quién entrenamiento.
2. Cómo se registran los experimentos (hoja de cálculo, archivo en el repo, o una herramienta). Cualquiera sirve si es la misma para ambos.
3. Ramas: sugerencia `main` estable + una rama por experimento.
4. Cuándo se revisa el avance: una cita fija semanal.

## 7. Qué hago yo y qué haces tú

| Tú | Claude |
|---|---|
| Iniciar sesión en GitHub, invitar al compañero | Preparar el repo local, `.gitignore`, commits |
| Preguntas al profesor | Redactar y actualizar `ENUNCIADO.md` con las respuestas |
| Elegir/contratar GPU, instalar drivers | Escribir el código de entrenamiento y evaluación |
| Descargar datos y grabar conducción | Auditar, procesar y analizar datos |
| Probar el agente final en una máquina limpia | Empaquetar y medir |

Cuando termines un bloque, dime qué hiciste y actualizo el estado en `PLAN_PROYECTO.md`.
