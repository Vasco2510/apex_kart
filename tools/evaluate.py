"""Arnes de evaluacion (F4.3): corre un agente en pistas x semillas y resume resultados.

Uso:
  python tools/evaluate.py agents/const_accel/agent.json --out runs/eval_const
  python tools/evaluate.py AGENT --set configs/eval_set.json --tracks lighthouse --seeds 1 2 --out runs/x
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

from stk_agent.config import RaceSettings
from stk_agent.evaluation import run_race


def load_set(path):
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    base = json.loads(Path(spec["base_config"]).read_text(encoding="utf-8"))
    base.update(spec.get("overrides", {}))
    return spec, base


def summarize(rows):
    n = len(rows)
    fin = [r for r in rows if r["finished"]]
    def mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None
    return {
        "races": n,
        "finish_rate": len(fin) / n if n else 0,
        "mean_finish_time": mean([r["finish_time"] for r in fin]),
        "mean_position": mean([r["position"] for r in rows]),
        "mean_distance": mean([r["distance"] for r in rows]),
        "mean_latency_ms": mean([r["latency_ms"] for r in rows]),
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("--set", default="configs/eval_set.json")
    p.add_argument("--tracks", nargs="+")
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--video", action="store_true", help="Guarda view_0.mp4 por carrera")
    p.add_argument("--out", required=True, help="Carpeta NUEVA de resultados")
    a = p.parse_args(argv)
    spec, base = load_set(a.set)
    tracks = a.tracks or spec["tracks"]
    seeds = a.seeds or spec["seeds"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for track in tracks:
        for seed in seeds:
            cfg = RaceSettings.from_dict({**base, "track": track, "seed": seed}) if hasattr(RaceSettings, "from_dict") \
                else RaceSettings(**{**base, "track": track, "seed": seed})
            try:
                rep = run_race([a.manifest], cfg, out / f"{track}_s{seed}", video=a.video)
                r = rep["results"][0]
                rows.append({"track": track, "seed": seed, "finished": bool(r["finished"]),
                             "finish_time": r["finish_time"], "position": r["native_position"],
                             "distance": r["overall_distance"],
                             "latency_ms": r["timing"]["host_latency"].get("mean_ms"),
                             "wall_s": rep.get("wall_seconds"), "error": ""})
            except Exception as e:  # una carrera invalida no detiene el resto
                rows.append({"track": track, "seed": seed, "finished": False, "finish_time": None,
                             "position": None, "distance": None, "latency_ms": None,
                             "wall_s": None, "error": f"{type(e).__name__}: {e}"[:200]})
            print(rows[-1], file=sys.stderr, flush=True)
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    summary = summarize(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
