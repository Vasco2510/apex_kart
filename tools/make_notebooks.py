"""Genera colab/01_preprocesar.ipynb y colab/02_entrenar.ipynb. Editar aqui y volver a ejecutar:
  python tools/make_notebooks.py
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "colab"

HEAD = '''from google.colab import drive
drive.mount('/content/drive')
DRIVE = '/content/drive/MyDrive/UTEC/DEEP'
import os, subprocess
assert os.path.isdir(DRIVE + '/apex_kart_code'), "Falta el codigo en Drive: en tu PC ejecuta  python tools/apex.py sync"
!rm -rf /content/apex && cp -r {DRIVE}/apex_kart_code /content/apex
%cd /content/apex'''


def nb(cells):
    out = []
    for kind, src in cells:
        c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(True)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        out.append(c)
    return {"cells": out, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"}}, "nbformat": 4, "nbformat_minor": 5}


PRE = [
    ("markdown", """# 01 · Preprocesar datos (runtime **CPU**, sin gastar GPU)
Cuenta: la de Gmail con el plan. Menu: *Entorno de ejecucion → Cambiar tipo → CPU*.
Convierte los episodios del subconjunto v1 (`results/subset_v1.txt`) a formato compacto en `apex_kart_processed/v1` (Drive).
Es **reanudable**: si se corta, vuelve a ejecutar la celda 3. Tarda ~20-40 min."""),
    ("code", HEAD),
    ("code", """!pip -q install --no-deps -e /content/apex
!pip -q install imageio-ffmpeg h5py
import os
print('CPUs:', os.cpu_count())"""),
    ("code", """!python tools/preprocess.py --root {DRIVE}/MarioKart_copia --list results/subset_v1.txt \\
    --out {DRIVE}/apex_kart_processed/v1 --down 4 --stride 2 --workers {os.cpu_count()}"""),
    ("code", """import glob, json, collections
infos = [json.load(open(f)) for f in glob.glob(DRIVE + '/apex_kart_processed/v1/*/info.json')]
print(len(infos), 'episodios procesados de', len(open('results/subset_v1.txt').read().split()))
print('frames:', sum(i['num_frames'] for i in infos))
print(collections.Counter(i['track'] for i in infos))
drive.flush_and_unmount(); print('Drive sincronizado')"""),
]

TRAIN = [
    ("markdown", """# 02 · Entrenar (runtime **GPU**)
Menu: *Entorno de ejecucion → Cambiar tipo → GPU (T4)*. Cuenta: Gmail con el plan.
Antes: en tu PC `python tools/apex.py sync` (codigo) y haber corrido el notebook 01 (datos).
- Cambia `RUN` y `CONFIG` para otra corrida. Los checkpoints se guardan en Drive cada epoca.
- Si Colab se desconecta: vuelve a ejecutar las celdas 2, 3 y 4 y **reanuda sola**."""),
    ("code", HEAD),
    ("code", """!nvidia-smi --query-gpu=name,memory.total --format=csv
!mkdir -p /content/processed_v1
!rsync -a --info=progress2 {DRIVE}/apex_kart_processed/v1/ /content/processed_v1/
!du -sh /content/processed_v1 && ls /content/processed_v1 | wc -l"""),
    ("code", """RUN = 'bc_v1'
CONFIG = 'configs/train/bc_v1.json'
!python training/train_bc.py --config {CONFIG} --data /content/processed_v1 \\
    --out {DRIVE}/apex_kart_runs/{RUN} --max-minutes 150"""),
    ("code", """import json
for r in json.load(open(f'{DRIVE}/apex_kart_runs/{RUN}/log.json')):
    v = r.get('val', {})
    print(r['epoch'], 'train', round(r['train_loss'], 4), 'val', round(v.get('loss', float('nan')), 4),
          'steer', round(v.get('steer_huber', float('nan')), 4), r['samples_per_s'], 'muestras/s')
drive.flush_and_unmount(); print('Listo. En tu PC: python tools/apex.py pull', RUN)"""),
]

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, cells in (("01_preprocesar", PRE), ("02_entrenar", TRAIN)):
        (OUT / f"{name}.ipynb").write_text(json.dumps(nb(cells), indent=1, ensure_ascii=False), encoding="utf-8")
        print("escrito", OUT / f"{name}.ipynb")
