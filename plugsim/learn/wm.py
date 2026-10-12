"""Probabilistic latent world model (Ward et al. 2026 recipe; same structure as dnh.world_model.ProbWM).

State x_t = [image latents of the recorded cameras (concatenated), proprio subset]. An ensemble of
history-conditioned MLPs predicts a Gaussian over the normalised next-step delta x_{t+1} - x_t given
the last H states/actions and the next action. It is trained with Gaussian NLL on *nominal* data
only. Non-conformity scores:
  alea  mean predicted std             (the paper's WM uncertainty sigma)
  epi   ensemble disagreement of the means
  nll   NLL of what actually happened  (available on recorded/executed data, not for imagination)
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

# Proprio subset used by the WM: per arm TCP pos (3) + quat (4) + grip width (1).
PROPRIO_IDX = list(range(14, 22)) + list(range(36, 44))


def wm_features(latents: np.ndarray, state: np.ndarray) -> np.ndarray:
    """latents [T, n_cams*latent] + state [T, 44] -> x [T, D]"""
    return np.concatenate([latents, state[:, PROPRIO_IDX]], 1).astype(np.float32)


class _Member(nn.Module):
    def __init__(self, d_in, d_out, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2 * d_out))

    def forward(self, x):
        mu, log_std = self.net(x).chunk(2, -1)
        return mu, log_std.clamp(-6.0, 2.0).exp()


class LatentWM(nn.Module):
    def __init__(self, x_dim, a_dim=16, H=2, n_members=5, hidden=512):
        super().__init__()
        self.x_dim, self.a_dim, self.H = x_dim, a_dim, H
        self.members = nn.ModuleList([_Member(H * (x_dim + a_dim) + a_dim, x_dim, hidden) for _ in range(n_members)])
        for n, d in (("x_mean", x_dim), ("x_std", x_dim), ("a_mean", a_dim), ("a_std", a_dim), ("d_mean", x_dim), ("d_std", x_dim)):
            self.register_buffer(n, torch.zeros(d) if "mean" in n else torch.ones(d))

    def fit_norm(self, X, A, D):
        for (m, s), V in zip([("x_mean", "x_std"), ("a_mean", "a_std"), ("d_mean", "d_std")], [X, A, D]):
            getattr(self, m).copy_(torch.as_tensor(V.mean(0)))
            getattr(self, s).copy_(torch.as_tensor(V.std(0) + 1e-5))

    def _inp(self, hx, ha, a):
        hx = (hx - self.x_mean) / self.x_std
        ha = (ha - self.a_mean) / self.a_std
        a = (a - self.a_mean) / self.a_std
        return torch.cat([hx.flatten(1), ha.flatten(1), a], -1)

    def forward(self, hx, ha, a):
        x = self._inp(hx, ha, a)
        outs = [m(x) for m in self.members]
        return torch.stack([o[0] for o in outs]), torch.stack([o[1] for o in outs])  # [M,N,D]

    def nll(self, hx, ha, a, delta):
        mu, std = self(hx, ha, a)
        d = (delta - self.d_mean) / self.d_std
        return (0.5 * ((d - mu) / std) ** 2 + std.log()).mean()

    @torch.no_grad()
    def step_scores(self, hx, ha, a, delta=None):
        """-> dict of [N] scores for one predicted step (and NLL if the true delta is given)."""
        mu, std = self(hx, ha, a)
        out = dict(alea=std.mean(dim=(0, 2)), epi=mu.std(0).mean(-1))
        if delta is not None:
            d = (delta - self.d_mean) / self.d_std
            out["nll"] = (0.5 * ((d - mu) / std) ** 2 + std.log()).mean(dim=(0, 2))
        return out

    @torch.no_grad()
    def imagine(self, hx, ha, chunk):
        """Mean-propagate an action chunk. hx [N,H,D], ha [N,H,A], chunk [N,T,A] -> alea, epi [N,T]."""
        alea, epi = [], []
        hx, ha = hx.clone(), ha.clone()
        for t in range(chunk.shape[1]):
            a = chunk[:, t]
            mu, std = self(hx, ha, a)
            alea.append(std.mean(dim=(0, 2)))
            epi.append(mu.std(0).mean(-1))
            x_next = hx[:, -1] + mu.mean(0) * self.d_std + self.d_mean
            hx = torch.cat([hx[:, 1:], x_next[:, None]], 1)
            ha = torch.cat([ha[:, 1:], a[:, None]], 1)
        return torch.stack(alea, 1), torch.stack(epi, 1)


def windows(feats, actions, H):
    """feats [T,D], actions [T,A] (action t leads from x_t to x_{t+1}) -> (hx, ha, a, delta) with
    left padding by repetition; ha holds the actions that led *into* each history state."""
    T = len(feats)
    A_in = np.concatenate([np.zeros((1, actions.shape[1]), np.float32), actions[:-1]], 0)
    HX, HA, AN, D = [], [], [], []
    for t in range(T - 1):
        idx = [max(0, t - H + 1 + i) for i in range(H)]
        HX.append(feats[idx])
        HA.append(A_in[idx])
        AN.append(actions[t])
        D.append(feats[t + 1] - feats[t])
    f = lambda x: np.asarray(x, np.float32)
    return f(HX), f(HA), f(AN), f(D)


def train_wm(episodes, H=2, n_members=5, hidden=512, steps=4000, bs=512, lr=1e-3, device="cpu", log=print, log_every=1000):
    """episodes: list of dict(x [T,D], action [T,A])."""
    parts = [windows(e["x"], e["action"], H) for e in episodes]
    HX, HA, AN, D = (np.concatenate([p[i] for p in parts]) for i in range(4))
    wm = LatentWM(HX.shape[-1], AN.shape[-1], H, n_members, hidden)
    wm.fit_norm(HX.reshape(-1, HX.shape[-1]), AN, D)
    wm = wm.to(device)
    t = lambda x: torch.as_tensor(x, device=device)
    HX, HA, AN, D = t(HX), t(HA), t(AN), t(D)
    opt = torch.optim.AdamW(wm.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    n, t0, hist = len(HX), time.time(), []
    for it in range(steps):
        # each member sees its own bootstrap batch (ensemble diversity)
        loss = 0
        for m_i in range(n_members):
            j = torch.randint(0, n, (bs,), device=device)
            mu, std = wm.members[m_i](wm._inp(HX[j], HA[j], AN[j]))
            d = (D[j] - wm.d_mean) / wm.d_std
            loss = loss + (0.5 * ((d - mu) / std) ** 2 + std.log()).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        hist.append(loss.item() / n_members)
        if log and (it % log_every == 0 or it == steps - 1):
            log(f"[wm] step {it:5d}  nll {np.mean(hist[-log_every:]):.4f}  ({time.time() - t0:.0f}s)")
    return wm.eval(), hist


@torch.no_grad()
def score_episode(wm: LatentWM, x, actions, device="cpu"):
    """One-step scores along a recorded episode -> dict of [T-1] arrays."""
    HX, HA, AN, D = windows(x, actions, wm.H)
    t = lambda v: torch.as_tensor(v, device=device)
    s = wm.step_scores(t(HX), t(HA), t(AN), t(D))
    return {k: v.cpu().numpy() for k, v in s.items()}


def save_wm(wm: LatentWM, path):
    torch.save(dict(x_dim=wm.x_dim, a_dim=wm.a_dim, H=wm.H, n=len(wm.members), hidden=wm.members[0].net[0].out_features, state=wm.state_dict()), path)


def load_wm(path, device="cpu") -> LatentWM:
    ck = torch.load(path, map_location=device, weights_only=False)
    wm = LatentWM(ck["x_dim"], ck["a_dim"], ck["H"], ck["n"], ck["hidden"])
    wm.load_state_dict(ck["state"])
    return wm.to(device).eval()
