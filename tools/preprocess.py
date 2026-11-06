"""F3.3: convierte episodios crudos (mp4+h5) a un formato compacto y rapido de leer.

Por episodio crea <out>/<episode_id>/ con:
  frames.npy   uint8 (N, H/down, W/down, 3)  RGB con HUD, reducido por promedio de bloques
  labels.npz   actions (N,7) = accion del tick del frame; actions_mean (N,7) = promedio de los ticks
               hasta el siguiente frame RGB; aux_* = estado privilegiado del kart humano (solo para
               filtrar/etiquetar, NO es entrada de la politica); rgb_index (N,) indice original
  info.json    pista, kart, slot, semilla, dificultad, fases, parametros (se escribe al final: marca "completo")

Es reanudable: salta episodios con info.json. Usa ffmpeg para submuestrear y reducir (rapido).

Uso:
  python tools/preprocess.py --root contexto/muestras --out data/processed
  python tools/preprocess.py --root /ruta/MarioKart_copia --list results/subset_v1.txt --out data/processed_v1 --workers 2
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

from stk_agent.dataset import Episode
from stk_agent.video import ffmpeg_executable

AUX_KEYS = ("speed", "distance_down_track", "overall_distance", "position", "finished_laps",
            "energy", "is_on_road", "jumping", "has_finished_race")


def read_frames(video, width, height, down, stride):
    """Frames RGB reducidos (promedio de bloques down x down) cada `stride` frames, via ffmpeg."""
    ow, oh = width // down, height // down
    vf = f"select='not(mod(n,{stride}))',format=rgb24,scale={ow}:{oh}:flags=area"
    cmd = [ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-threads", "1", "-i", str(video),
           "-vf", vf, "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    size = ow * oh * 3
    n = len(raw) // size
    return np.frombuffer(raw, np.uint8, n * size).reshape(n, oh, ow, 3)


def process_episode(args):
    path, out_root, down, stride = args
    ep = Episode(path)
    out = Path(out_root) / ep.meta["episode_id"]
    if (out / "info.json").exists():
        return str(out), "skip"
    out.mkdir(parents=True, exist_ok=True)
    s = ep.settings
    slot = int(ep.meta["human_slot"])
    frames = read_frames(ep.path / ep.meta["rgb_file"], s["width"], s["height"], down, stride)
    rgb_idx = list(range(0, len(ep) + 1, stride))
    ids = ep.rgb_tick_ids
    act_first, act_mean = [], []
    aux = {k: [] for k in AUX_KEYS}
    phases = []
    n = min(len(frames), len(rgb_idx))
    with h5py.File(ep.path / "data.h5", "r") as f:
        values = f["actions/value"][:]
        for i in rgb_idx[:n]:
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
    np.save(out / "frames.npy", frames[:n])
    np.savez(out / "labels.npz", actions=np.asarray(act_first, np.float32),
             actions_mean=np.asarray(act_mean, np.float32), phase_id=pid,
             rgb_index=np.asarray(rgb_idx[:n], np.int64),
             **{f"aux_{k}": np.asarray(v, np.float32) for k, v in aux.items()})
    info = {"episode_id": ep.meta["episode_id"], "track": s["track"], "kart": s["kart"],
            "difficulty": s["difficulty"], "laps": s["laps"], "num_karts": s["num_karts"],
            "reverse": s["reverse"], "seed": ep.meta["effective_seed"], "human_slot": slot,
            "group_id": ep.meta.get("group_id"), "driver_id": ep.meta.get("driver_id"),
            "seconds": ep.meta["simulated_seconds"], "num_frames": int(n), "phases": names,
            "down": down, "stride": stride, "frame_shape": list(frames.shape[1:]),
            "result": ep.meta.get("result")}
    (out / "info.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
    return str(out), f"ok {n} frames"


# rclone en Windows escribe estos caracteres de nombre en ancho completo; en Drive/Colab son ASCII
FULLWIDTH = str.maketrans({"＜": "<", "＞": ">", "：": ":", "＂": '"', "｜": "|", "？": "?", "＊": "*"})


def resolve(root, rel):
    cand = root / rel
    return cand if cand.exists() else root / rel.translate(FULLWIDTH)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--list", help="archivo con rutas relativas a --root (una por linea); si falta, busca READY.json")
    p.add_argument("--down", type=int, default=4)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    root = Path(a.root)
    if a.list:
        eps = [resolve(root, l) for l in Path(a.list).read_text(encoding="utf-8").splitlines() if l.strip()]
        missing = [e for e in eps if not (e / "READY.json").exists()]
        if missing:
            sys.exit(f"{len(missing)} episodios de la lista no existen bajo {root}; ejemplo: {missing[0]}")
    else:
        eps = sorted(x.parent for x in root.rglob("READY.json"))
    jobs = [(e, a.out, a.down, a.stride) for e in eps]
    print(f"{len(jobs)} episodios", file=sys.stderr, flush=True)
    t0 = time.time()
    results = map(process_episode, jobs) if a.workers <= 1 else ProcessPoolExecutor(a.workers).map(process_episode, jobs)
    for i, (path, status) in enumerate(results, 1):
        print(f"[{i}/{len(jobs)}] {time.time() - t0:.0f}s {Path(path).name} {status}", flush=True)


if __name__ == "__main__":
    main()
