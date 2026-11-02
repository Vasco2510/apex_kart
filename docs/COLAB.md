# Entrenar en Colab (F5A.2 en adelante)

Solo lo necesitas cuando tengas los datos del Drive procesados. Todo lo anterior corre en tu PC.

## 0. Una vez
1. Colab → Runtime → Change runtime type → GPU (T4 o la que tenga tu plan). Mira el costo en compute units por hora.
2. Código: el repo es privado. Opciones: (a) crear un *fine-grained token* de GitHub con permiso de lectura solo a `apex_kart` y clonar con `git clone https://<usuario>:<token>@github.com/Vasco2510/apex_kart.git` (no pegues el token en celdas que compartas); (b) subir un zip del repo a Drive y descomprimirlo.

## 0.5 Verificar que Colab ve los datos (acceso directo o copia)
Con Drive montado, esta celda cuenta episodios y tamaño; cámbiala por tu ruta (`/content/drive/MyDrive/UTEC/DEEP/MarioKart`). Si da 0 episodios o errores de permiso, el acceso directo no sirve y hay que hacer la copia real (`docs/GUIA_HUMANO.md` §5).
```python
import os, glob
root = '/content/drive/MyDrive/UTEC/DEEP/MarioKart'
ready = glob.glob(root + '/*/*/READY.json')
print('episodios con READY.json:', len(ready))
tot = sum(os.path.getsize(os.path.join(os.path.dirname(r), f)) for r in ready[:20]
          for f in os.listdir(os.path.dirname(r)) if os.path.isfile(os.path.join(os.path.dirname(r), f)))
print('tamano medio por episodio (MB):', tot / max(1, min(20, len(ready))) / 1e6)
```

## 1. Celdas
```python
from google.colab import drive
drive.mount('/content/drive')
```
```bash
%cd /content
!git clone https://github.com/Vasco2510/apex_kart.git   # ver paso 0.2
%cd apex_kart
!pip install -q -e .                                    # trae el simulador (necesario solo para evaluar/RL)
```
Datos procesados (se crean una vez con `tools/preprocess.py`, ver abajo) copiados al disco local de la VM, que es mucho mas rapido que leer Drive archivo por archivo:
```bash
!mkdir -p data && cp -r /content/drive/MyDrive/apex_kart/processed data/processed
```
Entrenar (reanudable; guarda en Drive para sobrevivir a desconexiones):
```bash
!python training/train_bc.py --data data/processed --out /content/drive/MyDrive/apex_kart/runs/bc_v1 \
    --epochs 20 --batch 256 --workers 2 --val-tracks lighthouse
# si se corto:  ... --resume /content/drive/MyDrive/apex_kart/runs/bc_v1/last.pt
```
`model.pt` resultante → copiarlo a `agents/bc_cnn/weights/model.pt` en tu PC y evaluar con `python tools/evaluate.py agents/bc_cnn/agent.json --out runs/eval_bc_v1`.

## 2. Preprocesar los datos del profesor (una vez, en la nube)
Leer `rgb.mp4` desde Drive es lo lento. Hazlo en Colab (CPU, sin GPU) para no bajar nada a tu PC:
```bash
!python tools/preprocess.py --root /content/drive/MyDrive/apex_kart/raw --out /content/drive/MyDrive/apex_kart/processed --workers 2
```
`raw` es la copia en TU Drive de la carpeta compartida (ver `docs/GUIA_HUMANO.md` §5). Tamaño esperado: ~2.2 GB por hora de conduccion con `--down 4 --stride 2`; para subconjuntos usa `--down 8` (≈4× menos).
