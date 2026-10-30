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

    def reset(self):
        self.buf.clear()

    def _shrink(self, rgb):
        h, w, c = rgb.shape
        d = h // self.meta["height"]
        return rgb.reshape(h // d, d, w // d, d, c).mean(axis=(1, 3)).astype(np.uint8)

    @torch.no_grad()
    def act(self, rgb):
        self.buf.append(self._shrink(np.asarray(rgb)))
        s, k = self.meta["spacing"], self.meta["frames"]
        idx = [max(0, len(self.buf) - 1 - s * j) for j in range(k - 1, -1, -1)]
        x = np.concatenate([self.buf[i] for i in idx], axis=2)  # H, W, 3k
        x = torch.from_numpy(x).permute(2, 0, 1).float().div(255).unsqueeze(0).to(self.device)
        return Action(**decode(self.net(x)[0].cpu(), self.thr))

    def close(self):
        pass
