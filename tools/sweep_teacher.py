"""Barrido de parametros del profesor: mide posicion/tiempo en pocas pistas. Uso: python tools/sweep_teacher.py --out results/sweep_teacher.json"""
import argparse, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

CONFIGS = {
    "base": {},
    "agresivo": dict(lookahead=(5.0, 0.4, 22.0), steer_gain=1.3),
    "agresivo_drift": dict(lookahead=(5.0, 0.4, 22.0), steer_gain=1.3, drift=True),
    "muy_agresivo": dict(lookahead=(4.0, 0.3, 18.0), steer_gain=1.5, brake_curve=1.1, brake_speed=18.0, drift=True),
    "items": dict(use_items=True),
    "items_fire": dict(use_items=True, fire=True),
    "items_nitro": dict(use_items=True, fire=True, nitro_err=0.15, nitro_curve=0.4),
    "nitro": dict(nitro_err=0.15, nitro_curve=0.4),
}

def job(a):
    name, track, laps, seed = a
    from teacher import run
    try:
        r = run(track, laps, 1.0, seed, 200.0, verbose=False, **CONFIGS[name])
    except Exception as e:
        r = {"error": str(e)[:150]}
    return {"cfg": name, "track": track, "seed": seed, **r}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tracks", nargs="+", default=["lighthouse", "scotland", "snowtuxpeak"])
    p.add_argument("--configs", nargs="+", default=list(CONFIGS))
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--laps", type=int, default=1)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    jobs = [(c, t, a.laps, a.seed) for c in a.configs for t in a.tracks]
    res = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(job, jobs):
            print(r, flush=True); res.append(r)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
