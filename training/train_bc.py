"""F5A: clonacion de comportamiento desde episodios procesados (tools/preprocess.py).

Pensado para Colab (GPU) pero corre en CPU para pruebas. Reanuda solo si existe <out>/last.pt.

  python training/train_bc.py --config configs/train/bc_v1.json
  python training/train_bc.py --data data/processed --out runs/bc_smoke --epochs 2 --samples-per-epoch 2000 --batch 32

Los argumentos de linea de comandos pisan los del --config.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents" / "bc_cnn"))
from model import Net  # noqa: E402

TRAIN_PHASES = ("GO_PHASE", "MUSIC_PHASE", "RACE_PHASE")
DEFAULTS = dict(data=None, out=None, frames=3, epochs=10, batch=128, lr=1e-3, wd=1e-4, workers=2,
                val_tracks=[], samples_per_epoch=0, max_minutes=0.0, seed=0, device="auto", amp=True,
                augment=True, flip=False, save_every=5, select="steer", dropout=0.2, val_samples=40000, only_finished=False, w_steer=1.0, w_accel=0.3, w_brake=1.0, w_nitro=1.0, w_drift=1.0, w_fire=1.0)


class Frames(Dataset):
    """Cada muestra apila `frames` frames guardados consecutivos (ya separados por el stride del preprocesado)."""

    def __init__(self, episodes, frames, only_finished=False):
        self.eps, self.index, self.k = [], [], frames
        for d in episodes:
            info = json.loads((Path(d) / "info.json").read_text())
            if only_finished and not (info.get("result") or {}).get("finished", False):
                continue  # carrera atascada o abandonada: sus etiquetas no sirven
            lab = np.load(Path(d) / "labels.npz")
            ok = np.isin(np.array(info["phases"])[lab["phase_id"]], TRAIN_PHASES)
            e = len(self.eps)
            self.eps.append((np.load(Path(d) / "frames.npy", mmap_mode="r"), lab["actions"], lab["actions_mean"]))
            self.index += [(e, int(i)) for i in np.where(ok)[0]]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, n):
        e, i = self.index[n]
        fr, act, actm = self.eps[e]
        ids = [max(0, i - j) for j in range(self.k - 1, -1, -1)]
        x = np.concatenate([fr[j] for j in ids], axis=2)
        y = np.empty(6, np.float32)
        y[0] = actm[i, 0]
        y[1:] = act[i, 1:6]
        return torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1), torch.from_numpy(y)


def augment(x):
    """Brillo/contraste por muestra (x en [0,1]); NO toca la geometria porque el giro depende de ella."""
    b = x.shape[0]
    gain = 1 + (torch.rand(b, 1, 1, 1, device=x.device) - 0.5) * 0.4
    bias = (torch.rand(b, 1, 1, 1, device=x.device) - 0.5) * 0.2
    return (x * gain + bias).clamp(0, 1)


def make_loss(a, dev):
    w = torch.tensor([a.w_accel, a.w_brake, a.w_nitro, a.w_drift, a.w_fire], device=dev)

    def loss_fn(out, y):
        out = out.float()
        steer = nn.functional.smooth_l1_loss(torch.tanh(out[:, 0]), y[:, 0])
        bce = nn.functional.binary_cross_entropy_with_logits(out[:, 1:], y[:, 1:], reduction="none").mean(0)
        return a.w_steer * steer + (bce * w).sum(), steer.detach(), bce.detach()
    return loss_fn


def evaluate(net, loader, dev, loss_fn, amp):
    net.eval()
    tot = sm = n = 0
    acc = np.zeros(5)
    with torch.no_grad(), torch.autocast(dev.type, enabled=amp):
        for x, y in loader:
            x, y = x.to(dev, non_blocking=True).float().div(255), y.to(dev, non_blocking=True)
            out = net(x)
            l, s, _ = loss_fn(out, y)
            tot += l.item() * len(x)
            sm += s.item() * len(x)
            n += len(x)
            acc += ((out[:, 1:].float() > 0).float() == y[:, 1:]).float().sum(0).cpu().numpy()
    return {"loss": tot / n, "steer_huber": sm / n, "bin_acc": (acc / n).round(3).tolist(), "n": n}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    for k, v in DEFAULTS.items():
        t = type(v) if v is not None and not isinstance(v, (list, bool)) else None
        if isinstance(v, bool):
            p.add_argument(f"--{k.replace('_', '-')}", dest=k, default=None, type=lambda s: s.lower() in ("1", "true", "yes"))
        elif isinstance(v, list):
            p.add_argument(f"--{k.replace('_', '-')}", dest=k, nargs="*", default=None)
        else:
            p.add_argument(f"--{k.replace('_', '-')}", dest=k, type=t, default=None)
    a = p.parse_args()
    cfg = dict(DEFAULTS)
    if a.config:
        cfg.update(json.loads(Path(a.config).read_text(encoding="utf-8")))
    cfg.update({k: v for k, v in vars(a).items() if k != "config" and v is not None})
    if not cfg["data"] or not cfg["out"]:
        p.error("faltan data y out (por --config o por linea de comandos)")
    return argparse.Namespace(**cfg)


def main():
    a = parse()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device(a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = bool(a.amp) and dev.type == "cuda"
    dirs = [str(Path(d).parent) for d in sorted(glob.glob(a.data + "/*/info.json"))]
    infos = {d: json.loads((Path(d) / "info.json").read_text()) for d in dirs}
    val = [d for d in dirs if infos[d]["track"] in a.val_tracks]
    train = [d for d in dirs if d not in val]
    shape = infos[train[0]]["frame_shape"]
    spacing = int(infos[train[0]]["stride"])
    tr = Frames(train, a.frames, a.only_finished)
    va = Frames(val, a.frames, a.only_finished) if val else None
    if va and a.val_samples and len(va) > a.val_samples:  # submuestreo fijo: validar cuesta menos y es comparable entre epocas
        va.index = va.index[:: -(-len(va) // a.val_samples)]
    print(f"device={dev} amp={amp} train_eps={len(train)} val_eps={len(val)} train={len(tr)} val={len(va) if va else 0} "
          f"frame={shape} spacing={spacing}", flush=True)
    net = Net(a.frames, shape[0], shape[1], dropout=a.dropout).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    scaler = torch.amp.GradScaler(enabled=amp)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    start, log, best = 0, [], float("inf")
    last = out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        net.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["opt"])
        start, log, best = ck["epoch"], ck.get("log", []), ck.get("best", best)
        print(f"reanudando desde epoca {start}", flush=True)
    loss_fn = make_loss(a, dev)
    sampler = RandomSampler(tr, replacement=False, num_samples=min(a.samples_per_epoch or len(tr), len(tr)))
    kw = dict(num_workers=min(a.workers, os.cpu_count() or 1), pin_memory=dev.type == "cuda", persistent_workers=a.workers > 0)
    tl = DataLoader(tr, a.batch, sampler=sampler, drop_last=True, **kw)
    vl = DataLoader(va, a.batch, **kw) if va else None
    meta = {"frames": a.frames, "spacing": spacing, "height": shape[0], "width": shape[1]}
    t_start = time.time()
    for ep in range(start, a.epochs):
        net.train()
        t0 = time.time()
        run = n = 0
        for x, y in tl:
            x, y = x.to(dev, non_blocking=True).float().div(255), y.to(dev, non_blocking=True)
            if a.augment:
                x = augment(x)
            if a.flip:  # espejo horizontal: el giro cambia de signo (duplica la variedad de curvas)
                m = torch.rand(x.shape[0], device=dev) < 0.5
                x = torch.where(m[:, None, None, None], x.flip(-1), x)
                y = y.clone()
                y[:, 0] = torch.where(m, -y[:, 0], y[:, 0])
            with torch.autocast(dev.type, enabled=amp):
                l, _, _ = loss_fn(net(x.contiguous(memory_format=torch.channels_last)), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(l).backward()
            scaler.step(opt)
            scaler.update()
            run += l.item() * len(x)
            n += len(x)
        dt = time.time() - t0
        rec = {"epoch": ep + 1, "train_loss": run / max(n, 1), "seconds": round(dt, 1), "samples_per_s": round(n / dt)}
        if vl:
            rec["val"] = evaluate(net, vl, dev, loss_fn, amp)
        log.append(rec)
        print(json.dumps(rec), flush=True)
        # criterio de "mejor": error de giro en pistas no vistas (la perdida total la domina el ruido de drift/nitro)
        score = (rec["val"]["steer_huber"] if a.select == "steer" else rec["val"]["loss"]) if vl else rec["train_loss"]
        ck = {"state_dict": net.state_dict(), "opt": opt.state_dict(), "epoch": ep + 1, "net": net.meta,
              "meta": meta, "args": vars(a), "log": log, "best": min(best, score)}
        torch.save(ck, last)
        light = {k: v for k, v in ck.items() if k not in ("opt", "log")}
        if score < best:
            best = score
            torch.save(light, out / "model.pt")  # mejor checkpoint, liviano, listo para entregar
        if a.save_every and (ep + 1) % a.save_every == 0:
            torch.save(light, out / f"ep{ep + 1:02d}.pt")  # instantaneas para comparar en el simulador
        (out / "log.json").write_text(json.dumps(log, indent=1))
        if a.max_minutes and (time.time() - t_start) / 60 > a.max_minutes:
            print("limite de tiempo alcanzado; reanuda volviendo a ejecutar el mismo comando", flush=True)
            break
    if not (out / "model.pt").exists():
        shutil.copy(last, out / "model.pt")


if __name__ == "__main__":
    main()
