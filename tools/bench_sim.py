"""Mide cuanto simulador rinde una maquina (para decidir donde generar datos o entrenar con refuerzo).

  python tools/bench_sim.py                       # prueba 1, 2, 4 y 8 carreras en paralelo
  python tools/bench_sim.py --workers 1 4 8 16 --track lighthouse

Corre 1 vuelta de `--track` con el profesor, N carreras a la vez, y mide cuantos segundos de juego
se simulan por cada segundo real (en total y por proceso). Borra lo que genera. Dura unos minutos.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(n, track):
    tmp = Path(tempfile.mkdtemp(prefix="bench_"))
    t0 = time.time()
    r = subprocess.run([sys.executable, "-u", str(ROOT / "training" / "dagger.py"), "--out", str(tmp), "--tracks", track,
                        "--seeds", *map(str, range(1, n + 1)), "--laps", "1", "--noise", "0", "--workers", str(n),
                        "--max-seconds", "120"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    wall = time.time() - t0
    shutil.rmtree(tmp, ignore_errors=True)
    sim = [float(m) for m in re.findall(r" t=(\d+(?:\.\d+)?)s", r.stdout)]
    ok = len(sim)
    return wall, ok, sum(sim)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--track", default="lighthouse")
    a = p.parse_args()
    print(f"CPU: {os.cpu_count()} hilos logicos | pista {a.track}, 1 vuelta, profesor sin ruido")
    print("procesos | carreras ok | s reales | s de juego por s real (total) | por proceso | carreras/hora")
    for n in a.workers:
        wall, ok, sim = run(n, a.track)
        if not ok:
            print(f"{n:8d} | 0 carreras terminadas: el motor fallo (revisa `doctor --engine`)")
            continue
        total = sim / wall
        print(f"{n:8d} | {ok:11d} | {wall:8.0f} | {total:30.1f} | {total / n:11.1f} | {ok / wall * 3600:13.0f}", flush=True)


if __name__ == "__main__":
    main()
