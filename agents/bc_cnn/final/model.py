"""Red CNN para clonacion de comportamiento (F5A). Compartida por entrenamiento y agente.

Salida "completa" (entrenamiento):  [steer (1 logit tanh, o `steer_bins` logits)] + 5 logits binarios + `n_aux` auxiliares.
Salida "de conduccion" (agente):    6 valores [steer_logit, accel, brake, nitro, drift, fire] -> `decode`.
Checkpoints antiguos (steer_bins=0, n_aux=0) se leen igual: `net` en el checkpoint puede no traer esas claves.
"""
import torch
from torch import nn

N_BIN = 5  # acceleration, brake, nitro, drift, fire
N_OUT = 6  # tamano de la salida de conduccion (compatibilidad)


class Net(nn.Module):
    def __init__(self, frames=3, height=90, width=160, width_mult=1.0, dropout=0.2, steer_bins=0, n_aux=0, **_ignored):
        super().__init__()
        c = [int(x * width_mult) for x in (32, 64, 96, 128)]
        self.conv = nn.Sequential(
            nn.Conv2d(3 * frames, c[0], 5, 2, 2), nn.ReLU(inplace=True),
            nn.Conv2d(c[0], c[1], 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c[1], c[2], 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c[2], c[3], 3, 2, 1), nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            n = self.conv(torch.zeros(1, 3 * frames, height, width)).numel()
        self.steer_bins, self.n_aux = int(steer_bins), int(n_aux)
        self.n_steer = self.steer_bins if self.steer_bins > 0 else 1
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(n, 256), nn.ReLU(inplace=True),
                                  nn.Linear(256, self.n_steer + N_BIN + self.n_aux))
        self.meta = dict(frames=frames, height=height, width=width, width_mult=width_mult, dropout=dropout,
                         steer_bins=self.steer_bins, n_aux=self.n_aux)
        self.full = False  # True solo durante el entrenamiento: devuelve todas las salidas

    def forward(self, x):  # x: (B, 3*frames, H, W) float en [0, 1]
        out = self.head(self.conv(x))
        if self.full:
            return out
        steer = steer_from_bins(out[:, :self.n_steer]) if self.steer_bins > 0 else out[:, :1]
        return torch.cat([steer, out[:, self.n_steer:self.n_steer + N_BIN]], dim=1)


def bin_centers(k, device=None):
    return torch.linspace(-1.0, 1.0, k, device=device)


def steer_from_bins(logits, window=2):
    """Valor esperado SOLO alrededor del bin mas probable (no promedia modas opuestas). Devuelve un logit tal que tanh(logit)=steer."""
    p = torch.softmax(logits.float(), dim=1)
    k = p.shape[1]
    idx = p.argmax(dim=1, keepdim=True)
    ar = torch.arange(k, device=p.device)[None]
    w = p * ((ar - idx).abs() <= window).float()
    e = (w * bin_centers(k, p.device)[None]).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp_min(1e-9)
    return torch.atanh(e.clamp(-0.999, 0.999))


def decode(out, thr=0.5):
    """Salida de conduccion (6,) -> dict de accion valido para stk_agent.Action."""
    steer = float(torch.tanh(out[0]))
    p = torch.sigmoid(out[1:6])
    return {"steer": max(-1.0, min(1.0, steer)), "acceleration": float(p[0] > thr), "brake": bool(p[1] > thr),
            "nitro": bool(p[2] > thr), "drift": bool(p[3] > thr), "fire": bool(p[4] > thr), "rescue": False}
