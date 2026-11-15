"""Prueba si el motor carga y el profesor termina una pista (1 vuelta). Cada pista en su proceso: si el motor cae, solo falla esa.

  python tools/probe_tracks.py cave oasis tutorial            # pistas no listadas del motor
  python tools/probe_tracks.py lighthouse abyss --reverse     # las mismas, al reves
Escribe results/probe_pistas.csv (pista, reverse, terminada, distancia, error).
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tracks", nargs="+")
    p.add_argument("--reverse", action="store_true")
    p.add_argument("--max-seconds", type=float, default=150)
    a = p.parse_args()
    env = {**os.environ, "STK_EXTRA_TRACKS": ",".join(a.tracks)}
    out = ROOT / "results" / "probe_pistas.csv"
    rows = list(csv.reader(out.open(encoding="utf-8"))) if out.exists() else [["pista", "reverse", "terminada", "distancia", "error"]]
    for t in a.tracks:
        cmd = [sys.executable, "-u", str(ROOT / "training" / "teacher.py"), "--tracks", t, "--laps", "1", "--max-seconds", str(a.max_seconds)]
        if a.reverse:
            cmd.append("--reverse")
        try:
            r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=a.max_seconds * 3 + 120)
            m = re.search(r"\{'track'.*\}", r.stdout)
            if m:
                d = eval(m.group(0))  # salida propia de teacher.run
                row = [t, a.reverse, d["finished"], round(d["distance"]), ""]
            else:
                err = (r.stderr.strip().splitlines() or ["sin salida"])[-1][:120]
                row = [t, a.reverse, False, 0, err]
        except subprocess.TimeoutExpired:
            row = [t, a.reverse, False, 0, "timeout"]
        print(row, flush=True)
        rows = [x for x in rows if not (x[0] == t and x[1] == str(a.reverse))] + [row]
    csv.writer(out.open("w", newline="", encoding="utf-8")).writerows(rows)


if __name__ == "__main__":
    main()
