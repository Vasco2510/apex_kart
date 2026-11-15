"""Agente por clonacion de comportamiento. Carga el checkpoint indicado en config['checkpoint']."""
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Net, decode  # noqa: E402
from stk_agent import Action  # noqa: E402


class Agent:
    def __init__(self, config):
        device = config.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        ckpt = torch.load(config["checkpoint"], map_location="cpu", weights_only=False)
        self.meta = ckpt["meta"]
        self.net = Net(**ckpt["net"]).to(self.device).eval()
        self.net.load_state_dict(ckpt["state_dict"])
        self.thr = float(config.get("threshold", 0.5))
        self.buf = deque(maxlen=self.meta["spacing"] * (self.meta["frames"] - 1) + 1)
        # Regla anti-atasco (solo mira la imagen): imagen casi quieta = el kart no avanza.
        self.unstick = bool(config.get("unstick", True))
        self.still_thr = float(config.get("still_thr", 1.5))   # dif. media de pixeles (0-255) entre pasos
        self.turn_after = int(config.get("turn_after", 30))     # pasos quieto antes de girar moderado (~1 s a 30 Hz)
        self.rescue_after = int(config.get("rescue_after", 60))  # pasos quieto antes de rescatar (~2 s)
        self.grace = int(config.get("grace", 180))              # pasos iniciales sin regla (cuenta atras de la salida)
        self.reset()

    def reset(self):
        self.buf.clear()
        self.diffs = deque(maxlen=max(self.rescue_after, self.turn_after))
        self.prev = None
        self.t = 0
        self.pulse = 0
        self.cool = 0
        self.last_steer = 0.0

    def _stuck_level(self, small):
        """0 = avanza, 1 = quieto ~1 s (girar moderado), 2 = quieto ~2 s (rescatar)."""
        f = small.astype(np.float32)
        if self.prev is not None:
            self.diffs.append(float(np.abs(f - self.prev).mean()))
        self.prev = f
        self.t += 1
        if self.cool > 0:
            self.cool -= 1
        if not self.unstick or self.t < self.grace or self.cool > 0:
            return 0
        d = list(self.diffs)
        if len(d) >= self.rescue_after and np.mean(d[-self.rescue_after:]) < self.still_thr:
            return 2
        if len(d) >= self.turn_after and np.mean(d[-self.turn_after:]) < self.still_thr:
            return 1
        return 0

    def _shrink(self, rgb):
        h, w, c = rgb.shape
        d = h // self.meta["height"]
        return rgb.reshape(h // d, d, w // d, d, c).mean(axis=(1, 3)).astype(np.uint8)

    @torch.no_grad()
    def act(self, rgb):
        small = self._shrink(np.asarray(rgb))
        self.buf.append(small)
        level = self._stuck_level(small)
        s, k = self.meta["spacing"], self.meta["frames"]
        idx = [max(0, len(self.buf) - 1 - s * j) for j in range(k - 1, -1, -1)]
        x = np.concatenate([self.buf[i] for i in idx], axis=2)  # H, W, 3k
        x = torch.from_numpy(x).permute(2, 0, 1).float().div(255).unsqueeze(0).to(self.device)
        a = decode(self.net(x)[0].cpu(), self.thr)
        if self.pulse > 0:                      # pulso de rescate en curso
            self.pulse -= 1
            a["rescue"] = True
        elif level == 2:                        # ~2 s quieto: rescatar y esperar ~5 s
            self.pulse, self.cool = 2, 150
            self.diffs.clear()
            a["rescue"] = True
        elif level == 1:                        # contra una pared: girar moderado y acelerar
            sign = 1.0 if (a["steer"] > 0 or (a["steer"] == 0 and self.last_steer >= 0)) else -1.0
            a.update(steer=0.5 * sign, acceleration=1.0, brake=False)
        self.last_steer = a["steer"]
        return Action(**a)

    def close(self):
        pass
