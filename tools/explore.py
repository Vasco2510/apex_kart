"""F3.1: estadisticas de los episodios procesados -> markdown.
Uso: python tools/explore.py data/processed --out results/exploracion_muestras.md
"""
import argparse, glob, json
from pathlib import Path
import numpy as np

A = ["steer", "acceleration", "brake", "nitro", "drift", "fire", "rescue"]

def main():
    p = argparse.ArgumentParser(); p.add_argument("root"); p.add_argument("--out", required=True)
    a = p.parse_args()
    lines = ["# Exploracion de episodios procesados", "",
             "Solo fase RACE_PHASE. `off-road` = fraccion de frames con is_on_road=0.", "",
             "| episodio | pista | dif | seg | pos | fin | v.media | accel>0.5 | |steer|>0.3 | brake | nitro | drift | fire | off-road |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tot = {k: [] for k in A}
    for d in sorted(glob.glob(a.root + "/*")):
        info = json.load(open(d + "/info.json")); lab = np.load(d + "/labels.npz")
        race = np.array(info["phases"])[lab["phase_id"]] == "RACE_PHASE"
        if race.sum() == 0:
            lines.append(f"| {info['episode_id'][:8]} | {info['track']} | {info['difficulty']} | {info['seconds']:.0f} | - | - | sin fase de carrera | | | | | | | |"); continue
        act = lab["actions"][race]; r = info.get("result") or {}
        for i, k in enumerate(A): tot[k].append(act[:, i])
        lines.append("| {} | {} | {} | {:.0f} | {} | {} | {:.1f} | {:.2f} | {:.2f} | {:.3f} | {:.3f} | {:.2f} | {:.3f} | {:.3f} |".format(
            info["episode_id"][:8], info["track"], info["difficulty"], info["seconds"], r.get("native_position", "-"),
            r.get("finished", "-"), lab["aux_speed"][race].mean(), (act[:, 1] > .5).mean(), (np.abs(act[:, 0]) > .3).mean(),
            act[:, 2].mean(), act[:, 3].mean(), act[:, 4].mean(), act[:, 5].mean(), 1 - lab["aux_is_on_road"][race].mean()))
    lines += ["", "## Global (RACE_PHASE)", ""]
    for i, k in enumerate(A):
        v = np.concatenate(tot[k]) if tot[k] else np.zeros(1)
        if i == 0: lines.append(f"- steer: media {v.mean():+.3f}, std {v.std():.3f}, cuantiles 5/50/95 = {np.percentile(v,[5,50,95]).round(2).tolist()}, valores unicos ~{len(np.unique(v.round(2)))}")
        elif i == 1: lines.append(f"- acceleration: media {v.mean():.3f}, fraccion=1: {(v>0.99).mean():.3f}")
        else: lines.append(f"- {k}: activo {v.mean()*100:.2f}% de los frames")
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8"); print("\n".join(lines))

if __name__ == "__main__":
    main()
