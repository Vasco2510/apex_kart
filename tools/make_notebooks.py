"""Genera colab/entrenar.ipynb (1 notebook, 3 celdas de codigo; la logica vive en training/pipeline.py).
Editar aqui y volver a ejecutar:  python tools/make_notebooks.py
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "colab"

CELLS = [
    ("markdown", """# apex_kart · Entrenar (todo automático)
**Ejecutar todo** (Ctrl+F9). Cuenta: la de Gmail con el plan. Runtime: **GPU (T4)**.
Hace solo: preprocesar (si falta) → copiar datos al disco local → prueba de humo → entrenar. Todo es **reanudable**: si Colab se corta, vuelve a ejecutar.
Antes, una vez en tu PC: `python tools/apex.py sync` (sube el código a Drive). Al terminar: `python tools/apex.py pull <run>`.
Para preprocesar sin gastar GPU: runtime CPU y `STAGES = 'preprocess'`."""),
    ("code", """PERSIST = '/content/drive/MyDrive/UTEC/DEEP'
PRESET  = 'teacher_r0'                      # smoke | elite | v1 | teacher_r0 | teacher_r1  (configs/train/<preset>.json)
STAGES  = 'preprocess,fetch,smoke,train'    # 'preprocess' solo = en runtime CPU
EXTRA   = ''                                # p. ej. '--epochs 20 --batch 128'
import os, subprocess, sys
from google.colab import drive
drive.mount('/content/drive')
print(subprocess.run('nvidia-smi -L', shell=True, capture_output=True, text=True).stdout.strip()
      or '!! SIN GPU: cambia el runtime a GPU (T4) antes de seguir')"""),
    ("code", """assert os.path.isdir(f'{PERSIST}/apex_kart_code'), 'Falta el codigo en Drive: en tu PC ejecuta  python tools/apex.py sync'
os.chdir('/content')  # idempotente: re-ejecutar no anida carpetas
subprocess.check_call(f'rm -rf /content/repo && cp -r {PERSIST}/apex_kart_code /content/repo', shell=True)
os.chdir('/content/repo')
!pip -q install --no-deps -e .
!pip -q install imageio-ffmpeg h5py"""),
    ("code", """try: PRESET, STAGES, EXTRA, PERSIST
except NameError: raise SystemExit('Faltan las variables: ejecuta primero la celda 1 (la de configuracion).')
cmd = [sys.executable, '-u', 'training/pipeline.py', '--preset', PRESET, '--persist', PERSIST, '--stages', STAGES] + EXTRA.split()
print('$', ' '.join(cmd))
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd='/content/repo')
for line in p.stdout:
    print(line, end='')
p.wait()
if p.returncode: raise SystemExit(f'el pipeline termino con error (codigo {p.returncode})')
drive.flush_and_unmount(); print('Drive sincronizado')"""),
]


def build():
    cells = []
    for kind, src in CELLS:
        c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(True)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        cells.append(c)
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"}}, "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    path = OUT / "entrenar.ipynb"
    path.write_text(json.dumps(build(), indent=1, ensure_ascii=False), encoding="utf-8")
    print("escrito", path)
