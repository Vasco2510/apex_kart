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
from model import N_BIN, Net, steer_from_bins  # noqa: E402

TRAIN_PHASES = ("GO_PHASE", "MUSIC_PHASE", "RACE_PHASE")
DEFAULTS = dict(data=None, out=None, frames=3, epochs=10, batch=128, lr=1e-3, wd=1e-4, workers=2,
                val_tracks=[], samples_per_epoch=0, max_minutes=0.0, seed=0, device="auto", amp=True,
                augment=True, flip=False, save_every=5, select="steer", dropout=0.2, val_samples=40000, only_finished=False, w_steer=1.0, w_accel=0.3, w_brake=1.0, w_nitro=1.0, w_drift=1.0, w_fire=1.0,
                steer_loss="huber", steer_bins=21, bin_smooth=1.0, pos_weight=False, pos_weight_cap=5.0,
                aux_heads=[], w_aux=0.3, aug_color=False, aug_noise=0.0, aug_blur=0.0)

# cabezas auxiliares: nombre -> (clave en labels.npz, escala, firmada (cambia de signo al espejar), tipo)
AUX = {
    "speed": ("aux_speed", 25.0, False, "reg"), "is_on_road": ("aux_is_on_road", 1.0, False, "bce"),
    "energy": ("aux_energy", 1.0, False, "reg"),
    "lat": ("aux_lat", 1.0, True, "reg"), "hdg": ("aux_hdg", 1.0, True, "reg"),
    "ang5": ("aux_ang5", 1.0, True, "reg"), "ang10": ("aux_ang10", 1.0, True, "reg"),
    "ang20": ("aux_ang20", 1.0, True, "reg"), "ang40": ("aux_ang40", 1.0, True, "reg"),
    "curve": ("aux_curve", 1.0, False, "reg"),
}


class Frames(Dataset):
    """Cada muestra apila `frames` frames guardados consecutivos (ya separados por el stride del preprocesado)."""

    def __init__(self, episodes, frames, only_finished=False, aux=()):
        self.eps, self.index, self.k, self.aux = [], [], frames, list(aux)
        for d in episodes:
            info = json.loads((Path(d) / "info.json").read_text())
            if only_finished and not (info.get("result") or {}).get("finished", False):
                continue  # carrera atascada o abandonada: sus etiquetas no sirven
            lab = np.load(Path(d) / "labels.npz")
            ok = np.isin(np.array(info["phases"])[lab["phase_id"]], TRAIN_PHASES)
            e = len(self.eps)
            n = len(lab["actions"])
            ax = np.full((n, len(self.aux)), np.nan, np.float32)  # NaN = etiqueta ausente (episodios viejos): se ignora
            for j, name in enumerate(self.aux):
                key, scale = AUX[name][:2]
                if key in lab.files:
                    ax[:, j] = lab[key] / scale
            self.eps.append((np.load(Path(d) / "frames.npy", mmap_mode="r"), lab["actions"], lab["actions_mean"], ax))
            self.index += [(e, int(i)) for i in np.where(ok)[0]]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, n):
        e, i = self.index[n]
        fr, act, actm, ax = self.eps[e]
        ids = [max(0, i - j) for j in range(self.k - 1, -1, -1)]
        x = np.concatenate([fr[j] for j in ids], axis=2)
        y = np.empty(6 + len(self.aux), np.float32)
        y[0] = actm[i, 0]
        y[1:6] = act[i, 1:6]
        y[6:] = ax[i]
        return torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1), torch.from_numpy(y)


def augment(x, color=False, noise=0.0, blur=0.0):
    """Fotometrico (x en [0,1]); NO toca la geometria porque el giro depende de ella.
    Base: brillo/contraste por muestra. color: ganancia distinta por canal RGB (igual en los frames apilados).
    noise: ruido gaussiano (sigma); blur: probabilidad de desenfoque leve 3x3."""
    b, c = x.shape[:2]
    gain = 1 + (torch.rand(b, 1, 1, 1, device=x.device) - 0.5) * 0.4
    bias = (torch.rand(b, 1, 1, 1, device=x.device) - 0.5) * 0.2
    if color:
        g = 1 + (torch.rand(b, 1, 3, 1, 1, device=x.device) - 0.5) * 0.3
        x = (x.reshape(b, c // 3, 3, *x.shape[2:]) * g).reshape(x.shape)
    x = x * gain + bias
    if noise > 0:
        x = x + torch.randn_like(x) * noise
    if blur > 0:
        m = (torch.rand(b, device=x.device) < blur)[:, None, None, None]
        xb = nn.functional.avg_pool2d(nn.functional.pad(x, (1, 1, 1, 1), mode="replicate"), 3, 1)
        x = torch.where(m, xb, x)
    return x.clamp(0, 1)


def label_rates(ds):
    """Tasa de positivos de accel/brake/nitro/drift/fire sobre las muestras de entrenamiento."""
    pos, n = np.zeros(N_BIN), 0
    for e, i in ds.index:
        pos += ds.eps[e][1][i, 1:6]
        n += 1
    return (pos / max(n, 1)).tolist()


def steer_targets(y0, k, smooth):
    """Giro [-1,1] -> distribucion sobre k bins (interpolada entre los 2 vecinos; smooth=0: bin mas cercano)."""
    pos = (y0.clamp(-1, 1) + 1) * 0.5 * (k - 1)
    lo = pos.floor().clamp(0, k - 2)
    fr = (pos - lo) if smooth else (pos - lo).round()
    t = torch.zeros(len(y0), k, device=y0.device)
    t.scatter_(1, lo.long()[:, None], (1 - fr)[:, None])
    t.scatter_add_(1, (lo.long() + 1)[:, None], fr[:, None])
    return t


def make_loss(a, dev, rates=None):
    w = torch.tensor([a.w_accel, a.w_brake, a.w_nitro, a.w_drift, a.w_fire], device=dev)
    pw = None
    if a.pos_weight and rates is not None:  # sqrt(negativos/positivos), acotado: brake/nitro son raros
        r = torch.tensor(rates, device=dev, dtype=torch.float32).clamp(1e-3, 1 - 1e-3)
        pw = ((1 - r) / r).sqrt().clamp(1.0, a.pos_weight_cap)
        pw[0] = 1.0  # acelerar es casi siempre 1; no se pondera
    bins = a.steer_loss == "bins"
    aux = list(a.aux_heads or [])
    kinds = [AUX[n][3] for n in aux]
    ns = a.steer_bins if bins else 1

    def split(out):
        out = out.float()
        return out[:, :ns], out[:, ns:ns + N_BIN], out[:, ns + N_BIN:]

    def steer_pred(sl):
        return torch.tanh(steer_from_bins(sl)[:, 0]) if bins else torch.tanh(sl[:, 0])

    def loss_fn(out, y):
        sl, bl, al = split(out)
        if bins:
            t = steer_targets(y[:, 0], ns, a.bin_smooth)
            steer = -(t * torch.log_softmax(sl, 1)).sum(1).mean()
            with torch.no_grad():
                sh = nn.functional.smooth_l1_loss(steer_pred(sl), y[:, 0])
        else:
            steer = sh = nn.functional.smooth_l1_loss(torch.tanh(sl[:, 0]), y[:, 0])
        bce = nn.functional.binary_cross_entropy_with_logits(bl, y[:, 1:6], reduction="none", pos_weight=pw).mean(0)
        tot = a.w_steer * steer + (bce * w).sum()
        if aux:
            ya = y[:, 6:]
            parts = []
            for j, kd in enumerate(kinds):
                m = ~torch.isnan(ya[:, j])
                if not m.any():
                    continue
                p, t = al[m, j], ya[m, j]
                parts.append(nn.functional.binary_cross_entropy_with_logits(p, t.clamp(0, 1)) if kd == "bce"
                             else nn.functional.smooth_l1_loss(p, t))
            if parts:
                tot = tot + a.w_aux * torch.stack(parts).mean()
        return tot, sh.detach(), bce.detach()

    loss_fn.steer_pred = lambda out: steer_pred(split(out)[0])
    loss_fn.split = split
    return loss_fn


def evaluate(net, loader, dev, loss_fn, amp):
    net.eval()
    tot = sm = n = 0
    mae = mae_turn = nturn = sign_ok = 0.0
    acc = np.zeros(5)
    with torch.no_grad(), torch.autocast(dev.type, enabled=amp):
        for x, y in loader:
            x, y = x.to(dev, non_blocking=True).float().div(255), y.to(dev, non_blocking=True)
            out = net(x)
            l, s, _ = loss_fn(out, y)
            tot += l.item() * len(x)
            sm += s.item() * len(x)
            n += len(x)
            sp = loss_fn.steer_pred(out)
            err = (sp - y[:, 0]).abs()
            t = y[:, 0].abs() > 0.3  # frames de curva: donde el estudiante "gira timido"
            mae += err.sum().item()
            mae_turn += err[t].sum().item()
            nturn += int(t.sum())
            sign_ok += (torch.sign(sp[t]) == torch.sign(y[t, 0])).float().sum().item()
            acc += ((loss_fn.split(out)[1] > 0).float() == y[:, 1:6]).float().sum(0).cpu().numpy()
    return {"loss": tot / n, "steer_huber": sm / n, "steer_mae": mae / n, "turn_mae": mae_turn / max(nturn, 1),
            "turn_sign_acc": sign_ok / max(nturn, 1), "bin_acc": (acc / n).round(3).tolist(), "n": n}


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
    aux = list(a.aux_heads or [])
    bad = [n for n in aux if n not in AUX]
    if bad:
        sys.exit(f"aux_heads desconocidas {bad}; validas: {list(AUX)}")
    if a.steer_loss not in ("huber", "bins"):
        sys.exit("steer_loss debe ser 'huber' o 'bins'")
    tr = Frames(train, a.frames, a.only_finished, aux)
    va = Frames(val, a.frames, a.only_finished, aux) if val else None
    if va and a.val_samples and len(va) > a.val_samples:  # submuestreo fijo: validar cuesta menos y es comparable entre epocas
        va.index = va.index[:: -(-len(va) // a.val_samples)]
    print(f"device={dev} amp={amp} train_eps={len(train)} val_eps={len(val)} train={len(tr)} val={len(va) if va else 0} "
          f"frame={shape} spacing={spacing}", flush=True)
    rates = label_rates(tr)
    print("tasas de etiquetas [accel, brake, nitro, drift, fire]:", [round(r, 4) for r in rates], flush=True)
    if aux:
        have = [float(np.mean([not np.isnan(e[3][:, j]).all() for e in tr.eps])) for j in range(len(aux))]
        print("cobertura de cabezas auxiliares (fraccion de episodios con la etiqueta):",
              dict(zip(aux, [round(h, 2) for h in have])), flush=True)
    net = Net(a.frames, shape[0], shape[1], dropout=a.dropout, steer_bins=a.steer_bins if a.steer_loss == "bins" else 0,
              n_aux=len(aux)).to(dev).to(memory_format=torch.channels_last)
    net.full = True  # entrenamiento: todas las salidas (el agente usa full=False)
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
    loss_fn = make_loss(a, dev, rates)
    signed = torch.tensor([AUX[n][2] for n in aux], device=dev)
    sampler = RandomSampler(tr, replacement=False, num_samples=min(a.samples_per_epoch or len(tr), len(tr)))
    kw = dict(num_workers=min(a.workers, os.cpu_count() or 1), pin_memory=dev.type == "cuda", persistent_workers=a.workers > 0)
    tl = DataLoader(tr, a.batch, sampler=sampler, drop_last=True, **kw)
    vl = DataLoader(va, a.batch, **kw) if va else None
    meta = {"frames": a.frames, "spacing": spacing, "height": shape[0], "width": shape[1], "steer_loss": a.steer_loss,
            "aux_heads": aux}
    if start == 0:
        (out / "label_rates.json").write_text(json.dumps(
            {"order": ["accel", "brake", "nitro", "drift", "fire"], "rates": rates}))
    t_start = time.time()
    for ep in range(start, a.epochs):
        net.train()
        t0 = time.time()
        run = n = 0
        for x, y in tl:
            x, y = x.to(dev, non_blocking=True).float().div(255), y.to(dev, non_blocking=True)
            if a.augment:
                x = augment(x, a.aug_color, a.aug_noise, a.aug_blur)
            if a.flip:  # espejo horizontal: el giro cambia de signo (duplica la variedad de curvas)
                m = torch.rand(x.shape[0], device=dev) < 0.5
                x = torch.where(m[:, None, None, None], x.flip(-1), x)
                y = y.clone()
                y[:, 0] = torch.where(m, -y[:, 0], y[:, 0])
                if aux:  # las etiquetas geometricas con signo tambien se invierten
                    y[:, 6:] = y[:, 6:] * ((m[:, None] & signed[None]).float() * -2 + 1)
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
        if vl:
            v = rec["val"]
            score = {"steer": v["steer_huber"], "loss": v["loss"],
                     "turn": 0.5 * v["steer_mae"] + 0.5 * v["turn_mae"]}[a.select]  # 'turn': penaliza giros timidos en curvas
        else:
            score = rec["train_loss"]
        ck = {"state_dict": net.state_dict(), "opt": opt.state_dict(), "epoch": ep + 1, "net": net.meta,
              "meta": meta, "args": vars(a), "log": log, "best": min(best, score)}
        torch.save(ck, last)
        light = {k: v for k, v in ck.items() if k not in ("opt", "log")}
        torch.save(light, out / "model_last.pt")  # ultimo epoch (el "mejor" por val puede ser uno temprano)
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
