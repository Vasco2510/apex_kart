"""F5A: clonacion de comportamiento desde episodios procesados (tools/preprocess.py).

Ejemplos:
  python training/train_bc.py --data data/processed --out runs/bc_smoke --epochs 2 --max-steps 15   # humo en CPU
  python training/train_bc.py --data data/processed --out runs/bc_v1 --epochs 20 --val-tracks lighthouse  # GPU (Colab)
Soporta reanudar: --resume runs/bc_v1/last.pt
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents" / "bc_cnn"))
from model import Net  # noqa: E402

TRAIN_PHASES = ("GO_PHASE", "MUSIC_PHASE", "RACE_PHASE")


class Frames(Dataset):
    """Cada muestra apila `frames` frames guardados consecutivos (ya separados por el stride del preprocesado)."""

    def __init__(self, episodes, frames):
        self.eps, self.index, self.k = [], [], frames
        for d in episodes:
            info = json.loads((Path(d) / "info.json").read_text())
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


def losses(out, y, w):
    steer = nn.functional.smooth_l1_loss(torch.tanh(out[:, 0]), y[:, 0])
    bce = nn.functional.binary_cross_entropy_with_logits(out[:, 1:], y[:, 1:], reduction="none").mean(0)
    return steer + (bce * w).sum(), steer.detach(), bce.detach()


def evaluate(net, loader, dev, w):
    net.eval()
    tot = n = 0
    sm = 0.0
    acc = np.zeros(5)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(dev).float().div(255), y.to(dev)
            out = net(x)
            l, s, _ = losses(out, y, w)
            tot += l.item() * len(x)
            sm += s.item() * len(x)
            n += len(x)
            acc += ((out[:, 1:] > 0).float() == y[:, 1:]).float().sum(0).cpu().numpy()
    return {"loss": tot / n, "steer_huber": sm / n, "bin_acc": (acc / n).round(3).tolist(), "n": n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--frames", type=int, default=3)
    p.add_argument("--spacing", type=int, default=2,
                   help="pasos de politica (30 Hz) entre frames apilados; debe igualar el stride del preprocesado")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--val-tracks", nargs="*", default=[])
    p.add_argument("--max-steps", type=int, default=0, help="limita pasos por epoca (pruebas de humo)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume")
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(dev)
    dirs = [str(Path(d).parent) for d in sorted(glob.glob(a.data + "/*/info.json"))]
    infos = {d: json.loads((Path(d) / "info.json").read_text()) for d in dirs}
    val = [d for d in dirs if infos[d]["track"] in a.val_tracks]
    train = [d for d in dirs if d not in val]
    shape = infos[train[0]]["frame_shape"]
    tr = Frames(train, a.frames)
    va = Frames(val, a.frames) if val else None
    print(f"device={dev} train={len(tr)} val={len(va) if va else 0} frame={shape}", flush=True)
    net = Net(a.frames, shape[0], shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    start = 0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=False)
        net.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["opt"])
        start = ck["epoch"]
    w = torch.tensor([0.5, 1.0, 1.0, 1.0, 1.0], device=dev)  # acelerar pesa menos: casi siempre es 1
    tl = DataLoader(tr, a.batch, shuffle=True, num_workers=a.workers, drop_last=True)
    vl = DataLoader(va, a.batch, num_workers=a.workers) if va else None
    log = []
    ck = None
    for ep in range(start, a.epochs):
        net.train()
        t0 = time.time()
        run = n = 0
        for step, (x, y) in enumerate(tl):
            if a.max_steps and step >= a.max_steps:
                break
            x, y = x.to(dev).float().div(255), y.to(dev)
            l, _, _ = losses(net(x), y, w)
            opt.zero_grad()
            l.backward()
            opt.step()
            run += l.item() * len(x)
            n += len(x)
        rec = {"epoch": ep + 1, "train_loss": run / max(n, 1), "seconds": round(time.time() - t0, 1)}
        if vl:
            rec["val"] = evaluate(net, vl, dev, w)
        log.append(rec)
        print(json.dumps(rec), flush=True)
        ck = {"state_dict": net.state_dict(), "opt": opt.state_dict(), "epoch": ep + 1, "net": net.meta,
              "meta": {"frames": a.frames, "spacing": a.spacing, "height": shape[0], "width": shape[1]},
              "args": vars(a)}
        torch.save(ck, out / "last.pt")
    if ck:
        ck.pop("opt")
        torch.save(ck, out / "model.pt")  # liviano, para entregar
    (out / "log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
