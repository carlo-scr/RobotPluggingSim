"""Diffusion Policy (Chi et al. 2023) for the plugging task: 1D-conv U-Net over action chunks with FiLM
conditioning, DDPM training, DDIM sampling.

Two observation modes:
  "state"  proprio (44) + privileged task features (18). Trains in minutes on a laptop, gives a
           strong base policy for steering experiments.
  "image"  proprio (44) + camera frames (small CNN with spatial softmax per camera). The realistic
           setting; meant for GPU nodes (more demos, longer training).
Action representation (default "abs"): absolute TCP *targets* per arm [pos(3), rot6d(6), grip, compliant]
(22-D), converted to the env's delta actions at execution time; the controller's current target is
part of the observation. (Chi et al. found position targets far more robust than velocity/delta
actions; with deltas the expert's hidden target lag makes labels partially unpredictable.)
"delta" predicts the env's 16-D actions directly. Chunks of `horizon` steps, `n_act` executed per replan.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import geometry as G
from ..env import ARMS
from .data import TASK_DIM, Normalizer, task_features_from_env

STATE_DIM, ENV_ACT_DIM, ABS_DIM, TGT_DIM = 44, 16, 22, 20
REL_DIM = 16
BINARY = {"delta": [6, 7, 14, 15], "abs": [9, 10, 20, 21], "rel": [6, 7, 14, 15]}  # grip / compliant per arm


def quat_to_rot6(q):
    """[..., 4] (w,x,y,z) -> [..., 6] first two columns of the rotation matrix."""
    q = np.asarray(q, np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    c0 = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], -1)
    c1 = np.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], -1)
    return np.concatenate([c0, c1], -1).astype(np.float32)


def rot6_to_quat(r6):
    a, b = np.asarray(r6[:3], float), np.asarray(r6[3:6], float)
    x = a / (np.linalg.norm(a) + 1e-9)
    b = b - (x @ b) * x
    y = b / (np.linalg.norm(b) + 1e-9)
    return G.mat_to_quat(np.stack([x, y, np.cross(x, y)], 1))


def targets_after(ee_target, actions):
    """Controller target after each executed action: [T,16] (pos,quat,grip per arm) + [T,16] actions
    -> [T,22] (pos, rot6, grip, compliant per arm). Mirrors PlugEnv.step exactly."""
    out = np.zeros((len(actions), ABS_DIM), np.float32)
    for t in range(len(actions)):
        for i in range(len(ARMS)):
            e, a = ee_target[t, 8 * i : 8 * i + 8], actions[t, 8 * i : 8 * i + 8]
            p = e[:3] + a[:3]
            q = G.quat_mul(G.quat_from_rotvec(a[3:6]), e[3:7])
            out[t, 11 * i : 11 * i + 11] = np.concatenate([p, quat_to_rot6(q), [a[6], a[7]]])
    return out


def intended_targets(ee_target, actions):
    """Controller targets the (expert) actions aim for: pos [T,2,3], quat [T,2,4], grip/compliant [T,2,2]."""
    T = len(actions)
    P, Q, B = np.zeros((T, 2, 3)), np.zeros((T, 2, 4)), np.zeros((T, 2, 2))
    for t in range(T):
        for i in range(len(ARMS)):
            e, a = ee_target[t, 8 * i : 8 * i + 8], actions[t, 8 * i : 8 * i + 8]
            P[t, i] = e[:3] + a[:3]
            Q[t, i] = G.quat_mul(G.quat_from_rotvec(a[3:6]), e[3:7])
            B[t, i] = a[6], a[7]
    return P, Q, B


def rel_chunk_labels(ee_target, actions, horizon):
    """Relative chunk labels: targets t..t+H-1 relative to the controller target at t.
    -> [T, H, 16] (per arm dpos(3), rotvec(3), grip, compliant)."""
    P, Q, B = intended_targets(ee_target, actions)
    T = len(actions)
    out = np.zeros((T, horizon, REL_DIM), np.float32)
    for t in range(T):
        for k in range(horizon):
            j = min(t + k, T - 1)
            for i in range(len(ARMS)):
                ref_p, ref_q = ee_target[t, 8 * i : 8 * i + 3], ee_target[t, 8 * i + 3 : 8 * i + 7]
                out[t, k, 8 * i : 8 * i + 3] = P[j, i] - ref_p
                out[t, k, 8 * i + 3 : 8 * i + 6] = G.quat_error(Q[j, i], ref_q)
                out[t, k, 8 * i + 6 : 8 * i + 8] = B[j, i]
    return out


def rel_to_abs(chunk, targets):
    """[H,16] relative chunk + current controller targets -> [H,22] absolute (pos, rot6, grip, compliant)."""
    out = np.zeros((len(chunk), ABS_DIM), np.float32)
    for i, s in enumerate(ARMS):
        p0, q0 = targets[s]
        for k, x in enumerate(chunk):
            y = x[8 * i : 8 * i + 8]
            q = G.quat_mul(G.quat_from_rotvec(y[3:6]), q0)
            out[k, 11 * i : 11 * i + 11] = np.concatenate([p0 + y[:3], quat_to_rot6(q), y[6:8]])
    return out


def target_obs(ee_target):
    """[T,16] (pos, quat, grip per arm) -> [T,20] (pos, rot6, grip per arm)."""
    parts = []
    for i in range(len(ARMS)):
        e = ee_target[:, 8 * i : 8 * i + 8]
        parts += [e[:, :3], quat_to_rot6(e[:, 3:7]), e[:, 7:8]]
    return np.concatenate(parts, 1).astype(np.float32)


def env_target_obs(env):
    e = np.concatenate([np.concatenate([env.targets[s][0], env.targets[s][1], [env.grip_cmd[s]]]) for s in ARMS])
    return target_obs(e[None])[0]


def abs_to_env_action(abs_a, targets):
    """One absolute target step [22] -> env action [16] relative to the current controller targets."""
    a = np.zeros(ENV_ACT_DIM)
    for i, s in enumerate(ARMS):
        x = abs_a[11 * i : 11 * i + 11]
        p0, q0 = targets[s]
        a[8 * i : 8 * i + 3] = x[:3] - p0
        a[8 * i + 3 : 8 * i + 6] = G.quat_error(rot6_to_quat(x[3:9]), q0)
        a[8 * i + 6], a[8 * i + 7] = x[9], x[10]
    return a


def abs_chunk_to_deltas(chunk, targets):
    """Absolute chunk [H,22] -> env actions [H,16], assuming each target is reached in turn."""
    tg = {s: (np.asarray(targets[s][0]).copy(), np.asarray(targets[s][1]).copy()) for s in ARMS}
    out = []
    for x in chunk:
        out.append(abs_to_env_action(x, tg))
        for i, s in enumerate(ARMS):
            y = x[11 * i : 11 * i + 11]
            tg[s] = (y[:3].copy(), rot6_to_quat(y[3:9]))
    return np.asarray(out)


@dataclass
class PolicyCfg:
    obs_mode: str = "state"  # "state" | "image"
    action_mode: str = "rel"  # "rel" (targets relative to the current target) | "abs" | "delta"
    cams: tuple = ("right_gripper_front", "left_gripper_front")
    img_hw: tuple = (72, 128)  # render size of the policy cameras (must match the demos)
    n_obs: int = 2
    horizon: int = 16
    n_act: int = 8
    down_dims: tuple = (64, 128, 256)
    cond_dim: int = 256
    img_feat: int = 64
    diffusion_steps: int = 100
    prediction: str = "sample"  # "sample" (predict the clean chunk; converges much faster) | "epsilon"
    ddim_steps: int = 10
    ddim_eta: float = 1.0  # 1 = stochastic sampling (diverse candidates for steering), 0 = deterministic


# ---------------------------------------------------------------------- networks
class SinPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        f = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000) / (half - 1)))
        e = t[:, None].float() * f[None]
        return torch.cat([e.sin(), e.cos()], -1)


class Conv1dBlock(nn.Module):
    def __init__(self, i, o, k=5, g=8):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.GroupNorm(g, o), nn.Mish())

    def forward(self, x):
        return self.net(x)


class CondResBlock(nn.Module):
    def __init__(self, i, o, cond, k=5):
        super().__init__()
        self.b1, self.b2 = Conv1dBlock(i, o, k), Conv1dBlock(o, o, k)
        self.film = nn.Sequential(nn.Mish(), nn.Linear(cond, 2 * o))
        self.res = nn.Conv1d(i, o, 1) if i != o else nn.Identity()

    def forward(self, x, c):
        h = self.b1(x)
        s, b = self.film(c).unsqueeze(-1).chunk(2, 1)
        h = self.b2(h * (1 + s) + b)
        return h + self.res(x)


class CondUnet1D(nn.Module):
    def __init__(self, act_dim, cond_dim, dims=(64, 128, 256)):
        super().__init__()
        t_dim = 64
        self.t_emb = nn.Sequential(SinPosEmb(t_dim), nn.Linear(t_dim, 4 * t_dim), nn.Mish(), nn.Linear(4 * t_dim, t_dim))
        c = cond_dim + t_dim
        ch = [act_dim] + list(dims)
        self.down = nn.ModuleList()
        for j, (i, o) in enumerate(zip(ch[:-1], ch[1:])):
            last = j == len(dims) - 1
            self.down.append(nn.ModuleList([CondResBlock(i, o, c), CondResBlock(o, o, c), nn.Conv1d(o, o, 3, 2, 1) if not last else nn.Identity()]))
        self.mid = nn.ModuleList([CondResBlock(dims[-1], dims[-1], c), CondResBlock(dims[-1], dims[-1], c)])
        self.up = nn.ModuleList()
        rev = list(reversed(dims))
        for j, (i, o) in enumerate(zip(rev[:-1], rev[1:])):
            self.up.append(nn.ModuleList([CondResBlock(2 * i, o, c), CondResBlock(o, o, c), nn.ConvTranspose1d(o, o, 4, 2, 1)]))
        self.final = nn.Sequential(Conv1dBlock(dims[0], dims[0]), nn.Conv1d(dims[0], act_dim, 1))

    def forward(self, x, t, cond):
        """x [B,T,A], t [B], cond [B,C] -> eps [B,T,A]"""
        c = torch.cat([cond, self.t_emb(t)], -1)
        h = x.transpose(1, 2)
        skips = []
        for r1, r2, ds in self.down:
            h = r2(r1(h, c), c)
            skips.append(h)
            h = ds(h)
        for m in self.mid:
            h = m(h, c)
        for r1, r2, us in self.up:
            h = torch.cat([h, skips.pop()], 1)
            h = us(r2(r1(h, c), c))
        return self.final(h).transpose(1, 2)


class SpatialSoftmax(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        p = F.softmax(x.flatten(2), -1).view(B, C, H, W)
        ys = torch.linspace(-1, 1, H, device=x.device)
        xs = torch.linspace(-1, 1, W, device=x.device)
        return torch.cat([(p.sum(2) * xs).sum(-1), (p.sum(3) * ys).sum(-1)], -1)


class ImageEncoder(nn.Module):
    """Small CNN + spatial-softmax keypoints, shared across cameras (+ learned camera embedding)."""

    def __init__(self, n_cams, feat=64):
        super().__init__()
        g = lambda c: nn.GroupNorm(8, c)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2), g(32), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), g(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), g(64), nn.ReLU(), nn.Conv2d(64, 64, 3, 1, 1),
        )
        self.ss = SpatialSoftmax()
        self.proj = nn.Linear(128, feat)
        self.cam_emb = nn.Parameter(torch.zeros(n_cams, feat))

    def forward(self, imgs):
        """imgs [B, n_cams, 3, H, W] float in [0,1] -> [B, n_cams*feat]"""
        B, N = imgs.shape[:2]
        f = self.proj(self.ss(self.cnn(imgs.flatten(0, 1) * 2 - 1))).view(B, N, -1) + self.cam_emb
        return f.flatten(1)


class DiffusionPolicy(nn.Module):
    def __init__(self, cfg: PolicyCfg):
        super().__init__()
        self.cfg = cfg
        low = STATE_DIM + (TASK_DIM if cfg.obs_mode == "state" else 0) + (TGT_DIM if cfg.action_mode in ("abs", "rel") else 0)
        self.low_dim = low
        self.act_dim = {"abs": ABS_DIM, "rel": REL_DIM, "delta": ENV_ACT_DIM}[cfg.action_mode]
        img = 0
        if cfg.obs_mode == "image":
            self.img_enc = ImageEncoder(len(cfg.cams), cfg.img_feat)
            img = len(cfg.cams) * cfg.img_feat
        self.obs_mlp = nn.Sequential(nn.Linear(cfg.n_obs * (low + img), cfg.cond_dim), nn.Mish(), nn.Linear(cfg.cond_dim, cfg.cond_dim))
        self.net = CondUnet1D(self.act_dim, cfg.cond_dim, cfg.down_dims)
        T = cfg.diffusion_steps
        s = 0.008  # squared-cosine schedule
        t = torch.linspace(0, T, T + 1) / T
        ac = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        betas = torch.clamp(1 - ac[1:] / ac[:-1], max=0.999)
        self.register_buffer("alphas_cumprod", torch.cumprod(1 - betas, 0))
        self.obs_norm = Normalizer()
        self.act_norm = Normalizer()

    # ---- conditioning
    def cond(self, low, imgs=None):
        """low [B, n_obs, low_dim] (normalised), imgs [B, n_obs, n_cams, 3, H, W] in [0,1]"""
        B = low.shape[0]
        parts = [low]
        if self.cfg.obs_mode == "image":
            f = self.img_enc(imgs.flatten(0, 1)).view(B, self.cfg.n_obs, -1)
            parts.append(f)
        return self.obs_mlp(torch.cat(parts, -1).flatten(1))

    def loss(self, low, act, imgs=None):
        B = act.shape[0]
        c = self.cond(low, imgs)
        t = torch.randint(0, self.cfg.diffusion_steps, (B,), device=act.device)
        eps = torch.randn_like(act)
        a = self.alphas_cumprod[t].view(B, 1, 1)
        x = a.sqrt() * act + (1 - a).sqrt() * eps
        return F.mse_loss(self.net(x, t, c), act if self.cfg.prediction == "sample" else eps)

    @torch.no_grad()
    def sample(self, low, imgs=None, K=1, steps=None, eta=None, generator=None):
        """-> normalised action chunks [B, K, horizon, ACT_DIM]"""
        cfg = self.cfg
        steps, eta = steps or cfg.ddim_steps, cfg.ddim_eta if eta is None else eta
        B = low.shape[0]
        c = self.cond(low, imgs).repeat_interleave(K, 0)
        x = torch.randn((B * K, cfg.horizon, self.act_dim), device=low.device, generator=generator)
        ts = torch.linspace(cfg.diffusion_steps - 1, 0, steps).round().long().tolist()
        for i, t in enumerate(ts):
            a_t = self.alphas_cumprod[t]
            a_prev = self.alphas_cumprod[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=low.device)
            out = self.net(x, torch.full((B * K,), t, device=low.device), c)
            if cfg.prediction == "sample":
                x0 = out.clamp(-1.5, 1.5)
                eps = (x - a_t.sqrt() * x0) / (1 - a_t).sqrt()
            else:
                eps = out
                x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1, 1)
            sig = eta * ((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)).clamp(min=0).sqrt()
            x = a_prev.sqrt() * x0 + (1 - a_prev - sig**2).clamp(min=0).sqrt() * eps
            if i + 1 < len(ts) and eta > 0:
                x = x + sig * torch.randn(x.shape, device=x.device, generator=generator)
        return x.view(B, K, cfg.horizon, self.act_dim)


# ---------------------------------------------------------------------- data
def make_windows(episodes, cfg: PolicyCfg):
    """(obs history, action chunk) training windows. Chunks past the episode end are padded with zero
    motion and the last grip/compliant command."""
    lows, acts, idx = [], [], []
    for e, ep in enumerate(episodes):
        parts = [ep["state"]] + ([ep["task"]] if cfg.obs_mode == "state" else [])
        acts_lbl = ep["action_expert"] if ep.get("action_expert") is not None else ep["action"]
        if cfg.action_mode == "rel":
            parts.append(target_obs(ep["ee_target"]))
            low = np.concatenate(parts, 1)
            R = rel_chunk_labels(ep["ee_target"], acts_lbl, cfg.horizon)
            for t in range(len(R)):
                o = [max(0, t - cfg.n_obs + 1 + i) for i in range(cfg.n_obs)]
                lows.append(low[o])
                acts.append(R[t])
                idx.append((e, o))
            continue
        if cfg.action_mode == "abs":
            parts.append(target_obs(ep["ee_target"]))
            A = targets_after(ep["ee_target"], acts_lbl)
            pad = np.repeat(A[-1:], cfg.horizon, 0)  # hold the last target
        else:
            A = ep["action"]
            pad = np.zeros((cfg.horizon, A.shape[1]), np.float32)
            pad[:, BINARY["delta"]] = A[-1, BINARY["delta"]]
        low = np.concatenate(parts, 1)
        T = len(A)
        Ap = np.concatenate([A, pad], 0)
        for t in range(T):
            o = [max(0, t - cfg.n_obs + 1 + i) for i in range(cfg.n_obs)]
            lows.append(low[o])
            acts.append(Ap[t : t + cfg.horizon])
            idx.append((e, o))
    return np.asarray(lows, np.float32), np.asarray(acts, np.float32), idx


def train_policy(episodes, cfg: PolicyCfg, steps=3000, bs=256, lr=3e-4, device="cpu", log_every=250, log=print, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    lows, acts, idx = make_windows(episodes, cfg)
    pol = DiffusionPolicy(cfg)
    pol.obs_norm.fit(lows)
    pol.act_norm.fit(acts, robust=cfg.action_mode == "rel")
    L = torch.as_tensor(pol.obs_norm.norm(lows), device=device)
    A = torch.as_tensor(pol.act_norm.norm(acts), device=device)
    frames = None
    if cfg.obs_mode == "image":
        frames = [{c: torch.as_tensor(ep["frames"][c]) for c in cfg.cams} for ep in episodes]  # uint8, CPU
    pol = pol.to(device)
    ema = copy.deepcopy(pol).eval()
    opt = torch.optim.AdamW(pol.parameters(), lr=lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    hist, t0 = [], time.time()
    n = len(L)
    for it in range(steps):
        j = np.random.randint(0, n, bs)
        imgs = None
        if frames is not None:
            imgs = torch.stack([torch.stack([torch.stack([frames[idx[k][0]][c][o] for c in cfg.cams]) for o in idx[k][1]]) for k in j])
            imgs = (imgs.permute(0, 1, 2, 5, 3, 4).float() / 255).to(device)
        loss = pol.loss(L[j], A[j], imgs)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        opt.step()
        sched.step()
        d = min(0.999, (1 + it) / (10 + it))
        with torch.no_grad():
            for pe, pp in zip(ema.parameters(), pol.parameters()):
                pe.mul_(d).add_(pp.detach(), alpha=1 - d)
        hist.append(loss.item())
        if log and (it % log_every == 0 or it == steps - 1):
            log(f"[policy] step {it:5d}  loss {np.mean(hist[-log_every:]):.4f}  ({time.time() - t0:.0f}s)")
    ema.obs_norm, ema.act_norm = pol.obs_norm, pol.act_norm
    return ema.eval(), hist


def save_policy(pol: DiffusionPolicy, path):
    torch.save(dict(cfg=asdict(pol.cfg), state=pol.state_dict(), obs_norm=pol.obs_norm.state_dict(), act_norm=pol.act_norm.state_dict()), path)


def load_policy(path, device="cpu") -> DiffusionPolicy:
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = PolicyCfg(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck["cfg"].items()})
    pol = DiffusionPolicy(cfg)
    pol.load_state_dict(ck["state"])
    pol.obs_norm, pol.act_norm = Normalizer.from_state(ck["obs_norm"]), Normalizer.from_state(ck["act_norm"])
    return pol.to(device).eval()


# ---------------------------------------------------------------------- closed loop
class PolicyRunner:
    """Receding-horizon execution of a DiffusionPolicy in a PlugEnv (keeps the obs history)."""

    def __init__(self, pol: DiffusionPolicy, env, device="cpu", seed=0):
        self.pol, self.env, self.device = pol, env, device
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.reset()

    def reset(self):
        self.hist_low, self.hist_img, self.queue = [], [], []

    def _obs(self):
        env, cfg = self.env, self.pol.cfg
        parts = [env.observe()] + ([task_features_from_env(env)] if cfg.obs_mode == "state" else [])
        if cfg.action_mode in ("abs", "rel"):
            parts.append(env_target_obs(env))
        self.hist_low.append(np.concatenate(parts))
        if cfg.obs_mode == "image":
            self.hist_img.append(np.stack([env.render(c, cfg.img_hw[1], cfg.img_hw[0]) for c in cfg.cams]))
        self.hist_low, self.hist_img = self.hist_low[-cfg.n_obs :], self.hist_img[-cfg.n_obs :]

    def _tensors(self):
        cfg = self.pol.cfg
        H = [self.hist_low[0]] * (cfg.n_obs - len(self.hist_low)) + self.hist_low
        low = torch.as_tensor(self.pol.obs_norm.norm(np.stack(H))[None], device=self.device)
        imgs = None
        if cfg.obs_mode == "image":
            I = [self.hist_img[0]] * (cfg.n_obs - len(self.hist_img)) + self.hist_img
            imgs = torch.as_tensor(np.stack(I)[None]).permute(0, 1, 2, 5, 3, 4).float().to(self.device) / 255
        return low, imgs

    def observe(self):
        """Record the current observation (call once per env step, before acting)."""
        self._obs()

    def sample_chunks(self, K):
        """K candidate chunks in the policy's action representation (denormalised, binary dims
        thresholded) from the current history: [K, horizon, 22 (abs) | 16 (delta)]."""
        low, imgs = self._tensors()
        x = self.pol.sample(low, imgs, K=K, generator=self.gen)[0].cpu().numpy()
        mode = self.pol.cfg.action_mode
        a = self.pol.act_norm.denorm(x)
        a[..., BINARY[mode]] = (a[..., BINARY[mode]] > 0.5).astype(np.float32)
        if mode == "rel":  # anchor to the current controller targets -> absolute format
            a = np.stack([rel_to_abs(c, self.env.targets) for c in a])
        return a

    def to_env_actions(self, chunk):
        """Policy chunk -> env actions [H,16] (for world-model imagination of a candidate)."""
        return abs_chunk_to_deltas(chunk, self.env.targets) if self.pol.cfg.action_mode in ("abs", "rel") else np.asarray(chunk)

    def _env_action(self, x):
        return abs_to_env_action(x, self.env.targets) if self.pol.cfg.action_mode in ("abs", "rel") else x

    def act(self):
        """Observe, (re)plan when the queue is empty, return the next env action."""
        self._obs()
        if not self.queue:
            self.queue = list(self.sample_chunks(1)[0][: self.pol.cfg.n_act])
        return self._env_action(self.queue.pop(0))

    def set_plan(self, chunk):
        self.queue = list(chunk[: self.pol.cfg.n_act])


def rollout(env, runner: PolicyRunner, scenario, max_s=16.0, settle_after_latch_s=1.0, cams=(), fps_stride=1):
    """Run the policy until the target is latched (+ settle) or timeout. -> dict(success, frames, ...)"""
    env.reset(scenario)
    runner.reset()
    frames = {c: [] for c in cams}
    t_latch = None
    while env.time < max_s and not env.unstable:
        if cams and env.step_count % fps_stride == 0:
            for c in cams:
                frames[c].append(env.render(c))
        env.step(runner.act())
        if env.latched_port >= 0 and t_latch is None:
            t_latch = env.time
        if t_latch is not None and env.time - t_latch > settle_after_latch_s:
            break
    return dict(success=env.latched_port == scenario.port_idx, latched_port=env.latched_port, latch_time=t_latch, duration=env.time, frames=frames)
