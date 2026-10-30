"""F3.3: convierte episodios crudos (mp4+h5) a un formato compacto y rapido de leer.

Por episodio crea <out>/<episode_id>/ con:
  frames.npy   uint8 (N, H/down, W/down, 3)  RGB con HUD, reducido por promedio de bloques
  labels.npz   actions (N,7) = accion del tick del frame; actions_mean (N,7) = promedio de los ticks
               hasta el siguiente frame RGB; aux_* = estado privilegiado del kart humano (solo para
               filtrar/etiquetar, NO es entrada de la politica); rgb_index (N,) indice original
  info.json    pista, kart, slot, semilla, dificultad, fases, parametros de reduccion

Uso: python tools/preprocess.py --root contexto/muestras --out data/processed --down 4 --stride 2
"""
from __future__ import annotations
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

from stk_agent.dataset import Episode

AUX_KEYS = ("speed", "distance_down_track", "overall_distance", "position", "finished_laps",
            "energy", "is_on_road", "jumping", "has_finished_race")


def _shrink(frame, down):
    h, w, c = frame.shape
    x = frame.reshape(h // down, down, w // down, down, c).mean(axis=(1, 3))
    return x.astype(np.uint8)


def process_episode(args):
    path, out_root, down, stride = args
    ep = Episode(path)
    out = Path(out_root) / ep.meta["episode_id"]
    if (out / "info.json").exists():
        return str(out), "skip"
    out.mkdir(parents=True, exist_ok=True)
    slot = int(ep.meta["human_slot"])
    n_total = len(ep) + 1  # frames RGB almacenados
    keep = np.arange(0, n_total, stride)
    frames, rgb_idx = [], []
    video = ep.frames()
    try:
        for i, fr in enumerate(video):
            if i % stride == 0:
                frames.append(_shrink(fr, down))
                rgb_idx.append(i)
    finally:
        video.close()
    frames = np.stack(frames)
    ids = ep.rgb_tick_ids
    act_first, act_mean = [], []
    aux = {k: [] for k in AUX_KEYS}
    phases = []
    with h5py.File(ep.path / "data.h5", "r") as f:
        values = f["actions/value"][:]
        for i in rgb_idx:
            a = int(ids[i])
            b = int(ids[i + 1]) if i + 1 < len(ids) else a + 1
            b = min(max(b, a + 1), len(values))
            a = min(a, len(values) - 1)
            seq = values[a:b]
            act_first.append(seq[0])
            act_mean.append(seq.mean(axis=0))
            st = ep._state(f, int(ids[i]))
            k = st["karts"][slot]
            for key in AUX_KEYS:
                aux[key].append(float(k.get(key, 0.0)))
            phases.append(st.get("phase", "UNKNOWN"))
    names = sorted(set(phases))
    pid = np.array([names.index(p) for p in phases], dtype=np.int8)
    n = min(len(frames), len(act_first))
    np.save(out / "frames.npy", frames[:n])
    np.savez(out / "labels.npz", actions=np.asarray(act_first[:n], np.float32),
             actions_mean=np.asarray(act_mean[:n], np.float32), phase_id=pid[:n],
             rgb_index=np.asarray(rgb_idx[:n], np.int64),
             **{f"aux_{k}": np.asarray(v[:n], np.float32) for k, v in aux.items()})
    s = ep.settings
    info = {"episode_id": ep.meta["episode_id"], "track": s["track"], "kart": s["kart"],
            "difficulty": s["difficulty"], "laps": s["laps"], "num_karts": s["num_karts"],
            "reverse": s["reverse"], "seed": ep.meta["effective_seed"], "human_slot": slot,
            "group_id": ep.meta.get("group_id"), "driver_id": ep.meta.get("driver_id"),
            "seconds": ep.meta["simulated_seconds"], "num_frames": int(n), "phases": names,
            "down": down, "stride": stride, "frame_shape": list(frames.shape[1:]),
            "result": ep.meta.get("result")}
    (out / "info.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
    return str(out), f"ok {n} frames"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--down", type=int, default=4)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    eps = sorted(p.parent for p in Path(a.root).rglob("READY.json"))
    jobs = [(e, a.out, a.down, a.stride) for e in eps]
    print(f"{len(jobs)} episodios", file=sys.stderr)
    if a.workers <= 1:
        results = map(process_episode, jobs)
    else:
        results = ProcessPoolExecutor(a.workers).map(process_episode, jobs)
    for path, status in results:
        print(path, status, flush=True)


if __name__ == "__main__":
    main()
