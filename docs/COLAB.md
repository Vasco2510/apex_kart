# Entrenar en Colab

Todo el flujo usa tu cuenta **Gmail con el plan** (la que tiene la copia de datos en `UTEC/DEEP/MarioKart_copia`).

## Dónde vive cada cosa

| Qué | Dónde |
|---|---|
| Código (fuente de verdad) | GitHub `apex_kart` + tu PC |
| Código para Colab | Drive `UTEC/DEEP/apex_kart_code` (lo sube `tools/apex.py sync`) |
| Datos crudos del profesor | Drive `UTEC/DEEP/MarioKart_copia` (123 GiB, solo lectura) |
| Datos preprocesados | Drive `UTEC/DEEP/apex_kart_processed/v1` (~22 GB) |
| Corridas (checkpoints, logs) | Drive `UTEC/DEEP/apex_kart_runs/<nombre>` |

## Flujo (3 pasos)

1. **En tu PC:** subir el código.
   ```bash
   python tools/apex.py sync
   ```
2. **Colab, runtime CPU:** abrir `colab/01_preprocesar.ipynb` y ejecutar todas las celdas. Se hace una vez por subconjunto (~20–40 min, sin gastar GPU).
3. **Colab, runtime GPU (T4):** abrir `colab/02_entrenar.ipynb` y ejecutar todas las celdas. Los checkpoints se guardan en Drive cada época; si se corta, vuelve a ejecutar y reanuda sola.

Luego, en tu PC:
```bash
python tools/apex.py status        # ver avance de las corridas
python tools/apex.py pull bc_v1    # bajar el modelo a agents/bc_cnn/weights/
python tools/apex.py eval bc_v1    # evaluarlo en el simulador (el simulador corre en tu PC, no en Colab)
```

## Cómo abrir los notebooks

- **Navegador:** colab.research.google.com → Archivo → Subir cuaderno → elegir el `.ipynb` de la carpeta `colab/`.
- **Desde VS Code:** si tienes la extensión de Colab, abre el `.ipynb` y elige el kernel de Colab. Si no, usa el navegador.

## Si algo falla

- *"Falta el codigo en Drive"*: ejecuta `python tools/apex.py sync` en tu PC.
- *Se desconectó Colab*: vuelve a ejecutar las celdas de preparación y la de entrenar.
- *Sin memoria o disco*: baja `batch` o `samples_per_epoch` en `configs/train/bc_v1.json` y sincroniza de nuevo.
- Los notebooks se generan con `python tools/make_notebooks.py`; edita ese script, no el `.ipynb`.
