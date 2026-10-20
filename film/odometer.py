import math

import moderngl
import numpy as np
from PIL import Image, ImageDraw

import geo
import machine as mc
from chamber import CH_FS, CH_VS
from gl3d import look, m4, perspective
from timeline import BAR, H, W

FACTS = {
    "kanal_01_tabo": dict(value=906890, unit="TREES", line="PLANTED FROM OPENSTREETMAP WOOD, ORCHARD AND PALM AREAS"),
    "kanal_02_jeepney": dict(value=618, unit="METRES ÷ 100", line="HOME TO STI  ·  6.18 KM ON THE OSM ROAD GRAPH",
                             shown="6.18", unit_shown="KILOMETRES"),
    "kanal_03_tricycle": dict(value=49, unit="METRES", line="DROP FROM HOME (371 M) TO STI (322 M)  ·  TERRARIUM DEM",
                              shown="49", unit_shown="METRES DOWNHILL"),
    "kanal_04_sarisari": dict(value=2424, unit="BUILDINGS", line="OSM FOOTPRINTS RAISED AROUND THE FOUR LANDMARKS"),
    "kanal_05_rooster": dict(value=387392, unit="PEOPLE", line="LIPA CITY  ·  2024 CENSUS"),
    "kanal_06_videoke": dict(value=1887, unit="YEAR", line="VILLA DE LIPA  ·  21 OCTOBER 1887"),
}


def drum_texture(size=256):
    im = Image.new("RGB", (size, size * 10), (0, 0, 0))
    d = ImageDraw.Draw(im)
    for y in range(size * 10):
        k = 0.5 + 0.5 * math.cos((y % size) / size * 2 * math.pi)
        c = tuple(int(v) for v in (np.array([196, 150, 72]) * (0.82 + 0.18 * k)))
        d.line([(0, y), (size, y)], fill=c)
    f = geo.font(int(size * 0.72), 900)
    for i in range(10):
        cy = size * 10 - (i + 0.5) * size
        d.text((size / 2 + 3, cy + 4), str(i), font=f, fill=(70, 46, 16), anchor="mm")
        d.text((size / 2, cy), str(i), font=f, fill=(34, 20, 6), anchor="mm")
        d.line([(0, cy - size / 2), (size, cy - size / 2)], fill=(90, 60, 24), width=3)
    return np.asarray(im)


def drum_mesh(n=40):
    out = []
    for i in range(n):
        a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
        for (x0, x1) in ((-0.5, 0.5),):
            p = [(x0, math.cos(a0), math.sin(a0)), (x1, math.cos(a0), math.sin(a0)),
                 (x1, math.cos(a1), math.sin(a1)), (x0, math.cos(a1), math.sin(a1))]
            nm = (0.0, math.cos((a0 + a1) / 2), math.sin((a0 + a1) / 2))
            uv = [(0, i / n), (1, i / n), (1, (i + 1) / n), (0, (i + 1) / n)]
            mc._quad(out, *p, nm, *uv)
    return np.array(out, "f4")


def label_texture(text, sub, w=1600, h=200):
    im = Image.new("RGB", (w, h), (0, 0, 0))
    d = ImageDraw.Draw(im)
    for y in range(h):
        k = 1 - abs(y - h / 2) / (h / 2)
        d.line([(0, y), (w, y)], fill=tuple(int(v) for v in (np.array([120, 84, 34]) + np.array([70, 56, 30]) * k ** 0.7)))
    d.rectangle([4, 4, w - 5, h - 5], outline=(60, 40, 14), width=4)
    f1, f2 = geo.font(70, 900), geo.font(34, 700)

    def spaced(y, s, f, tr, fill):
        ws = [d.textlength(c, font=f) for c in s]
        x = (w - (sum(ws) + tr * (len(s) - 1))) / 2
        for c, wc in zip(s, ws):
            d.text((x, y), c, font=f, fill=fill, anchor="lm")
            x += wc + tr

    spaced(h * 0.36, text, f1, 12, (40, 24, 8))
    spaced(h * 0.76, sub, f2, 5, (52, 34, 12))
    return np.asarray(im)


class Odometer:
    def __init__(self, ctx, fbo_ms, read_fn, chamber):
        self.ctx = ctx
        self.fbo_ms = fbo_ms
        self.read = read_fn
        self.ch = chamber
        self.prog = chamber.prog
        self.drum = ctx.vertex_array(self.prog, [(ctx.buffer(drum_mesh()), "3f 3f 2f", "in_pos", "in_n", "in_uv")])
        dt = drum_texture()
        self.drum_tex = ctx.texture((dt.shape[1], dt.shape[0]), 3, np.ascontiguousarray(dt).tobytes(), alignment=1)
        self.drum_tex.build_mipmaps()
        self.drum_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.drum_tex.anisotropy = 8.0
        self.labels = {}
        for k, f in FACTS.items():
            lt = label_texture(f.get("unit_shown", f["unit"]), f["line"])
            tex = ctx.texture((lt.shape[1], lt.shape[0]), 3, np.ascontiguousarray(lt).tobytes(), alignment=1)
            tex.build_mipmaps()
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            tex.anisotropy = 8.0
            self.labels[k] = tex

    def digits(self, key):
        f = FACTS[key]
        s = f.get("shown", str(f["value"]))
        if len(s.replace(".", "")) <= 6 and "." not in s and len(s) < 6:
            s = s.rjust(max(4, len(s)), "0") if key == "kanal_03_tricycle" else s
        return s

    def render(self, t, lt, dur, key, seed):
        ctx, p = self.ctx, self.prog
        s = self.digits(key)
        cols = [c for c in s]
        nd = sum(1 for c in cols if c != ".")
        e = min(1.0, max(0.0, lt / dur))
        pitch = 2.3
        width = pitch * len(cols)
        q = 1 - (1 - e) ** 3
        orbit = math.radians(-34 + 44 * q + (seed % 2) * 8)
        dist = (width + 7.0) / 0.70 * (1.55 - 0.55 * q)
        target = np.array([0.0, 0.0, -1.0])
        cam = target + np.array([math.sin(orbit) * dist, -math.cos(orbit) * dist, 2.6 - 1.2 * e])
        f = target - cam
        f /= np.linalg.norm(f)
        rr = np.cross(f, (0, 0, 1.0))
        rr /= np.linalg.norm(rr)
        u = np.cross(rr, f)
        P = perspective(0.5, 400.0)
        jx, jy = getattr(self, "jitter", (0.0, 0.0))
        P[0, 2] += jx
        P[1, 2] += jy
        VP = P @ look(cam, rr, u, f)
        p["VP"].write(m4(VP))
        p["cam"].value = tuple(cam)
        p["lamp"].value = (-4.0, -12.0, 7.0)
        p["lamp_col"].value = (4.2, 3.4, 2.4)
        p["fill_dir"].value = (0.5, -0.4, -0.3)
        p["fog_k"].value = 0.01
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE)
        self.fbo_ms.use()
        self.fbo_ms.clear(0.035, 0.026, 0.018, 1.0, depth=1.0)
        beats = BAR / 3.0
        settle_end = dur * 0.62
        x = -width / 2 + pitch / 2
        di = 0
        drum_axes = []
        for c in cols:
            if c == ".":
                p["mode"].value = 0
                self.ch.draw("sphere", mc.T(x, -1.05, -1.0) @ mc.S(0.22), (0.20, 0.13, 0.05), rough=0.5)
                x += pitch * 0.55
                continue
            target_d = int(c)
            stop = settle_end * (0.35 + 0.65 * (nd - di) / nd)
            stop = round(stop / beats) * beats
            k = min(1.0, max(0.0, lt / max(stop, 1e-3)))
            k = 1 - (1 - k) ** 3
            spins = 2 + (nd - di)
            turns = (spins * 10 + target_d) * k
            if lt >= stop:
                over = math.exp(-(lt - stop) * 9.0) * math.sin((lt - stop) * 30.0) * 0.08
                turns = spins * 10 + target_d + over
            ang = turns / 10.0 * 2 * math.pi
            self.drum_tex.use(3)
            p["tex"].value = 3
            M = mc.T(x, 0, 0) @ mc.rot((1, 0, 0), ang - 0.9 * math.pi) @ mc.S(pitch * 0.86, 1.25, 1.25)
            p["M"].write(m4(M))
            p["base"].value = (1, 1, 1)
            p["metal"].value = 0.6
            p["rough"].value = 0.3
            p["mode"].value = 1
            p["emissive"].value = 1.05
            self.drum.render()
            for sx in (-0.5, 0.5):
                self.ch.gear_at(24, (x + sx * pitch * 0.92, 0, 0), mc.rot((0, 1, 0), math.pi / 2), 1.38, 0.12,
                                -ang * (1 if di % 2 else -1) * 0.5, base=mc.BRASS_DARK)
            drum_axes.append((x, ang))
            x += pitch
            di += 1
        self.ch.draw("box", mc.T(0, 0.9, -1.75) @ mc.S(width + 2.2, 1.4, 0.5), mc.BRASS, rough=0.28)
        self.ch.draw("box", mc.T(0, 0.9, 1.75) @ mc.S(width + 2.2, 1.4, 0.5), mc.BRASS, rough=0.28)
        for sx in (-1, 1):
            self.ch.draw("box", mc.T(sx * (width / 2 + 0.9), 0.9, 0) @ mc.S(0.5, 1.4, 4.0), mc.BRASS, rough=0.28)
        self.ch.draw("cyl", mc.T(-width / 2 - 1.4, 0, 0) @ mc.rot((0, 1, 0), math.pi / 2) @ mc.S(0.18, 0.18, width + 2.8),
                     mc.STEEL, rough=0.2)
        lbl = self.labels[key]
        lbl.use(3)
        self.ch.draw("screen", mc.T(0, -0.3, -3.25) @ mc.S(width + 1.6, 1, (width + 1.6) / 8.0), (1, 1, 1), mode=1,
                     emissive=1.0)
        self.ch.tex.use(3)
        crank = lt * 6.0
        gx = width / 2 + 3.4
        self.ch.gear_at(48, (gx, 1.2, 0.6), mc.rot((1, 0, 0), math.pi / 2), 2.6, 0.4, crank * 0.25)
        th2 = 0.35 + math.pi - math.pi / 16 - (48 / 16) * (crank * 0.25 - 0.35)
        self.ch.gear_at(16, (gx + (2.6 + 0.87) * math.cos(0.35), 1.2, 0.6 + (2.6 + 0.87) * math.sin(0.35)),
                        mc.rot((1, 0, 0), math.pi / 2), 0.87, 0.4, th2, base=mc.COPPER)
        self.ch.gear_at(48, (-gx, 1.2, -0.4), mc.rot((1, 0, 0), math.pi / 2), 2.6, 0.4, -crank * 0.25)
        for k2 in range(2):
            ext = min(1.0, lt / (dur * 0.5))
            base_pt = np.array([(-1 if k2 else 1) * (width / 2 + 1.0), 2.4, -4.6])
            d = np.array([0.0, -1.0, 1.0]) * 3.0
            fm = mc.frame_to(d)
            for off, seg, r2 in mc.telescope(4.0, ext, 3):
                self.ch.draw("cyl", mc.T(*base_pt) @ fm @ mc.T(0, 0, off) @ mc.S(0.3 * r2, 0.3 * r2, seg),
                             mc.BRASS if r2 > 0.9 else mc.STEEL, rough=0.22)
        self.ch.draw("box", mc.T(0, 6.0, 0) @ mc.S(80, 1, 60), (0.08, 0.06, 0.04), metal=0.0, rough=0.9)
        for i in range(14):
            xx = -24 + i * 48 / 13
            self.ch.draw("cyl", mc.T(xx, 5.0, -16) @ mc.S(0.3, 0.3, 32), mc.IRON, metal=0.6, rough=0.6)
        return self.read(4)[..., :3]
