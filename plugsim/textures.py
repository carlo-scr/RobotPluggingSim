"""Procedural textures (PNG) for a more realistic data-center scene.

Generated deterministically on first use into `plugsim/_assets/` (gitignored) and referenced
by absolute path from the MJCF. Textured fronts are applied to thin visual *planes* (box faces mirror
2D textures in MuJoCo), oriented with `FRONT_QUAT` so that, seen from the robot (looking +x),
texture-right = world -y and texture-up = world +z.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ASSET_DIR = Path(__file__).resolve().parent / "_assets"
# Plane local axes: x -> world -y, y -> world +z, z (normal) -> world -x (towards the robot).
FRONT_MAT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)


def _font(size):
    for f in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def _save(img: Image.Image, name: str) -> str:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    p = ASSET_DIR / f"{name}.png"
    img.save(p)
    return str(p)


def _cached(name: str, key: str, make) -> str:
    """Rebuild only if the generator parameters changed."""
    tag = hashlib.md5(key.encode()).hexdigest()[:8]
    p = ASSET_DIR / f"{name}_{tag}.png"
    if not p.exists():
        return _save(make(), f"{name}_{tag}")
    return str(p)


def port_labels(cols: int, start: int, pitch_px: int = 64) -> str:
    """White label strip with port numbers centred over each column."""

    def make():
        w, h = cols * pitch_px, 40
        im = Image.new("RGB", (w, h), (236, 236, 230))
        d = ImageDraw.Draw(im)
        f = _font(26)
        for c in range(cols):
            s = str(start + c)
            tw = d.textlength(s, font=f)
            d.text((c * pitch_px + pitch_px / 2 - tw / 2, 6), s, fill=(20, 20, 20), font=f)
            if c % 6 == 0 and c:
                d.line((c * pitch_px, 4, c * pitch_px, h - 4), fill=(150, 150, 150), width=2)
        return im

    return _cached(f"labels_{cols}_{start}", f"{cols}-{start}-{pitch_px}", make)


def server_bezel(seed: int, units: int = 1) -> str:
    """Server front: dark bezel, hex vents, drive bays, status LEDs."""

    def make():
        rng = np.random.default_rng(seed)
        w, h = 1024, 120 * units
        base = int(rng.integers(30, 60))
        im = Image.new("RGB", (w, h), (base, base, base + 4))
        d = ImageDraw.Draw(im)
        # drive bays
        nb = int(rng.choice([4, 8, 10, 12]))
        bw = 600 // nb
        for i in range(nb):
            x0 = 40 + i * bw
            d.rectangle((x0, 15, x0 + bw - 6, h - 15), fill=(base + 25,) * 3, outline=(base + 45,) * 3, width=2)
            d.rectangle((x0 + 6, h - 35, x0 + 16, h - 27), fill=(40, 200, 60) if rng.random() < 0.8 else (230, 150, 20))
        # vents
        for y in range(20, h - 15, 14):
            for x in range(680 + (y // 14 % 2) * 7, w - 60, 14):
                d.ellipse((x, y, x + 8, y + 8), fill=(10, 10, 12))
        d.rectangle((w - 45, h // 2 - 8, w - 29, h // 2 + 8), fill=(40, 120, 255))
        return im.filter(ImageFilter.GaussianBlur(0.6))

    return _cached(f"server_{units}u_{seed}", f"{seed}-{units}", make)


def switch_front(ports: int = 48, seed: int = 0) -> str:
    """Top-of-rack switch: two rows of RJ45 ports with link/activity LEDs, SFP cages."""

    def make():
        rng = np.random.default_rng(seed)
        w, h = 1024, 120
        im = Image.new("RGB", (w, h), (52, 56, 62))
        d = ImageDraw.Draw(im)
        per = ports // 2
        pw = 760 // per
        for r in range(2):
            for c in range(per):
                x0, y0 = 30 + c * pw + (c // 6) * 4, 18 + r * 50
                d.rectangle((x0, y0, x0 + pw - 5, y0 + 30), fill=(8, 8, 10))
                if rng.random() < 0.7:
                    d.rectangle((x0 + 1, y0 - 6, x0 + 5, y0 - 2), fill=(40, 220, 70))
                if rng.random() < 0.4:
                    d.rectangle((x0 + pw - 10, y0 - 6, x0 + pw - 6, y0 - 2), fill=(240, 160, 20))
        for i in range(4):
            x0 = 830 + i * 45
            d.rectangle((x0, 30, x0 + 36, 85), fill=(20, 20, 24), outline=(120, 120, 120), width=2)
        return im

    return _cached(f"switch_{ports}_{seed}", f"{ports}-{seed}", make)


def rack_rail() -> str:
    """Rack post with square cage-nut holes."""

    def make():
        w, h = 64, 1024
        im = Image.new("RGB", (w, h), (34, 34, 38))
        d = ImageDraw.Draw(im)
        for y in range(8, h, 22):
            if (y // 22) % 3 != 1:
                d.rectangle((24, y, 38, y + 12), fill=(6, 6, 8))
        return im

    return _cached("rack_rail", "v1", make)


def floor_tile() -> str:
    """Raised-floor tile with perforations and seams."""

    def make():
        w = 512
        rng = np.random.default_rng(0)
        a = np.full((w, w, 3), 168, np.float32) + rng.normal(0, 4, (w, w, 1))
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, w - 1, w - 1), outline=(110, 110, 112), width=6)
        for y in range(60, w - 60, 18):
            for x in range(60, w - 60, 18):
                d.ellipse((x, y, x + 7, y + 7), fill=(90, 92, 95))
        return im

    return _cached("floor_tile", "v1", make)


def esd_mat(color=(70, 110, 140)) -> str:
    def make():
        rng = np.random.default_rng(1)
        a = np.ones((256, 256, 3), np.float32) * np.asarray(color, np.float32) + rng.normal(0, 5, (256, 256, 1))
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

    return _cached("esd_mat", str(color), make)


def blanking_panel() -> str:
    def make():
        im = Image.new("RGB", (512, 64), (28, 28, 30))
        d = ImageDraw.Draw(im)
        for x in (10, 490):
            d.ellipse((x, 26, x + 12, 38), fill=(120, 120, 124))
        for x in range(60, 450, 10):
            d.line((x, 18, x, 46), fill=(18, 18, 20), width=3)
        return im

    return _cached("blanking", "v1", make)
