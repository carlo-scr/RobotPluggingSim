"""Frame encoders for the latent world model.

ConvAE        small conv autoencoder trained on nominal sim frames (runs anywhere; local default).
CosmosEncoder the lab's frozen NVIDIA Cosmos tokenizer (as in GaugeFailClassification / Ward et al.),
              for GPU nodes: needs the `cosmos_tokenizer` package and the JIT checkpoints, e.g.
              <ckpt_dir>/Cosmos-0.1-Tokenizer-CI16x16/encoder.jit (render at 256x144 so H, W are
              multiples of 16).
Both expose `encode(frames uint8 [N,H,W,3]) -> float32 [N, D]`.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAE(nn.Module):
    def __init__(self, hw=(72, 128), latent=32):
        super().__init__()
        H, W = hw
        assert H % 8 == 0 and W % 8 == 0, "frame size must be divisible by 8"
        self.hw, self.latent = (H, W), latent
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.SiLU(), nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.SiLU(), nn.Conv2d(128, 64, 3, 1, 1), nn.SiLU(),
        )
        self.h8, self.w8 = H // 8, W // 8
        self.fc_e = nn.Linear(64 * self.h8 * self.w8, latent)
        self.fc_d = nn.Linear(latent, 64 * self.h8 * self.w8)
        self.dec = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1), nn.SiLU(), nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.SiLU(), nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )

    def _x(self, frames):
        return torch.as_tensor(frames).permute(0, 3, 1, 2).float() / 255.0

    def forward_latent(self, x):
        return self.fc_e(self.enc(x * 2 - 1).flatten(1))

    def decode(self, z):
        return torch.sigmoid(self.dec(self.fc_d(z).view(-1, 64, self.h8, self.w8)))

    @torch.no_grad()
    def encode(self, frames, bs=512):
        dev = next(self.parameters()).device
        out = [self.forward_latent(self._x(frames[i : i + bs]).to(dev)).cpu() for i in range(0, len(frames), bs)]
        return torch.cat(out).numpy().astype(np.float32)


def edge_weights(x, strength):
    """Per-pixel loss weights 1 .. 1 + strength from the target's gradient magnitude. Thin, high-contrast
    structures (the cable, the plug, port edges) cover few pixels, so a plain MSE autoencoder blurs them
    away while the table and the gripper fingers dominate the loss."""
    g = x.mean(1, keepdim=True)
    m = F.pad((g[..., :, 1:] - g[..., :, :-1]).abs(), (0, 1)) + F.pad((g[..., 1:, :] - g[..., :-1, :]).abs(), (0, 0, 0, 1))
    m = m / (m.mean(dim=(1, 2, 3), keepdim=True) + 1e-6)
    return 1 + strength * (m / 10).clamp(max=1)


def train_ae(frames: np.ndarray, latent=32, steps=3000, bs=128, lr=1e-3, device="cpu", log=print, log_every=500, edge=0.0):
    """frames uint8 [N,H,W,3] (all cameras pooled). edge > 0 weights the reconstruction loss towards
    edges (see edge_weights); the logged mse is always the plain one."""
    ae = ConvAE(frames.shape[1:3], latent).to(device)
    opt = torch.optim.AdamW(ae.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0, hist = time.time(), []
    for it in range(steps):
        j = np.random.randint(0, len(frames), bs)
        x = ae._x(frames[j]).to(device)
        err = (ae.decode(ae.forward_latent(x)) - x) ** 2
        if edge > 0:
            w = edge_weights(x, edge)
            loss = (w * err).mean() / w.mean()
        else:
            loss = err.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        hist.append(err.mean().item())
        if log and (it % log_every == 0 or it == steps - 1):
            log(f"[ae] step {it:5d}  mse {np.mean(hist[-log_every:]):.5f}  ({time.time() - t0:.0f}s)")
    return ae.eval(), hist


class CosmosEncoder:
    """Frozen Cosmos continuous image tokenizer -> spatially pooled latent vector."""

    def __init__(self, ckpt_dir, name="Cosmos-0.1-Tokenizer-CI16x16", device="cuda", pool=(3, 4)):
        from cosmos_tokenizer.image_lib import ImageTokenizer  # noqa: import only on GPU nodes

        self.tok = ImageTokenizer(checkpoint_enc=f"{ckpt_dir}/{name}/encoder.jit", device=device)
        self.device, self.pool = device, pool

    @torch.no_grad()
    def encode(self, frames, bs=64):
        out = []
        for i in range(0, len(frames), bs):
            x = torch.as_tensor(frames[i : i + bs]).permute(0, 3, 1, 2).to(self.device).to(torch.bfloat16) / 127.5 - 1
            (z,) = self.tok.encode(x)
            out.append(F.adaptive_avg_pool2d(z.float(), self.pool).flatten(1).cpu())
        return torch.cat(out).numpy().astype(np.float32)


def save_ae(ae: ConvAE, path):
    torch.save(dict(hw=ae.hw, latent=ae.latent, state=ae.state_dict()), path)


def load_ae(path, device="cpu") -> ConvAE:
    ck = torch.load(path, map_location=device, weights_only=False)
    ae = ConvAE(tuple(ck["hw"]), ck["latent"])
    ae.load_state_dict(ck["state"])
    return ae.to(device).eval()
