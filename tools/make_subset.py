"""Subconjunto de entrenamiento desde results/episodios.csv: podio, dificultad>=2, 8 karts, 3 vueltas, sin reversa ni pista 'random'.
Uso: python tools/make_subset.py results/episodios.csv --out results/subset_v1.txt --max-per-track 20
Cada linea es la ruta relativa del episodio (grupo/[subcarpeta/]episodio)."""
import argparse, csv, collections
from pathlib import Path

a = argparse.ArgumentParser()
a.add_argument("csv"); a.add_argument("--out", required=True); a.add_argument("--max-per-track", type=int, default=20)
a = a.parse_args()
rows = [r for r in csv.DictReader(open(a.csv, encoding="utf-8")) if r["ready"] == "True"]
ok = [r for r in rows if r["finished"] == "True" and r["position"] and int(float(r["position"])) <= 3
      and int(r["difficulty"]) >= 2 and r["num_karts"] == "8" and r["laps"] == "3"
      and r["reverse"] != "True" and r["track"] != "random"]
ok.sort(key=lambda r: (float(r["position"]), float(r["finish_time"] or 1e9)))
cnt, sel = collections.Counter(), []
for r in ok:
    if cnt[r["track"]] < a.max_per_track:
        cnt[r["track"]] += 1; sel.append(r)
Path(a.out).write_text("\n".join(f'{r["group"]}/{r["episode"]}' if "/" not in r["episode"] else r["episode"] for r in sel) + "\n", encoding="utf-8")
print(len(sel), "episodios,", round(sum(float(r["seconds"]) for r in sel) / 3600, 1), "h;", dict(sorted(cnt.items())))
