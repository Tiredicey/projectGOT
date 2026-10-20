import argparse
import logging
import math
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import geo
import gl3d
import route
from timeline import BAR, CREDITS, FINAL_HIT, FPS, H, LAND, LEAD, NFRAMES, RINGS, SHOTS, TITLES, W

log = logging.getLogger("render")
ROOT = Path(__file__).resolve().parent
SPP = 3

FOV = math.radians(40)
FPX = 0.5 * W / math.tan(FOV / 2)
PARCH_BG = (205, 182, 136)
BRASS = np.array([178, 138, 72], np.float32)
HOME_YELLOW = np.array([236, 190, 52], np.float32)
CATHEDRAL_ROOF = np.array([199, 103, 103], np.float32)
LIGHT = np.array([-0.55, 0.75, 0.37])
LIGHT = LIGHT / np.linalg.norm(LIGHT)
CREAM = (243, 230, 196)
GOLD = (232, 199, 122)


def ease(p):
    p = min(1.0, max(0.0, p))
    return p * p * (3 - 2 * p)


def ease_io(p):
    p = min(1.0, max(0.0, p))
    return 4 * p ** 3 if p < 0.5 else 1 - (-2 * p + 2) ** 3 / 2


def spring(tau, zeta=0.55, omega=9.0):
    if tau <= 0:
        return 0.0
    wd = omega * math.sqrt(1 - zeta ** 2)
    return 1 - math.exp(-zeta * omega * tau) * (math.cos(wd * tau) + zeta * omega / wd * math.sin(wd * tau))


def first_crossing():
    tau = np.linspace(0, 2, 4001)
    vals = np.array([spring(x) for x in tau])
    return float(tau[np.argmax(vals >= 1.0)])


TAU1 = first_crossing()


def lerp(a, b, p):
    return tuple(x + (y - x) * p for x, y in zip(a, b))


def lerp_angle(a, b, p):
    d = (b - a + 180) % 360 - 180
    return a + d * p


class Camera:
    def __init__(self, tx, ty, dist, pitch, yaw):
        self.t = np.array([tx, ty, 0.0])
        yr, pr = math.radians(yaw), math.radians(pitch)
        fwd_ground = np.array([math.sin(yr), math.cos(yr), 0.0])
        self.c = self.t - fwd_ground * dist * math.sin(pr) + np.array([0, 0, dist * math.cos(pr)])
        f = self.t - self.c
        f /= np.linalg.norm(f)
        up_hint = fwd_ground if pitch < 1 else np.array([0, 0, 1.0])
        r = np.cross(f, up_hint)
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        self.R = np.stack([r, u, f])
        self.dist = dist

    def project(self, pts):
        pc = (pts - self.c) @ self.R.T
        z = pc[:, 2]
        safe = np.where(z > 1.0, z, np.nan)
        u = W / 2 + FPX * pc[:, 0] / safe
        v = H / 2 - FPX * pc[:, 1] / safe
        return np.stack([u, v], -1), z

    def ground_h(self):
        k = np.array([[FPX, 0, W / 2], [0, -FPX, H / 2], [0, 0, 1]])
        rt = np.column_stack([self.R[:, 0], self.R[:, 1], -self.R @ self.c])
        return k @ rt


def leg_p(rt, i, t):
    leg = rt.legs[i]
    return (t - leg["t0"]) / (leg["t1"] - leg["t0"])


def shot_camera(kind, p, centers, rt=None, t=0.0):
    hx, hy = centers["home"]
    sx, sy = centers["sti"]
    cx, cy = centers["cathedral"]
    mx, my = centers["sm"]
    yaw_ride = math.degrees(math.atan2(sx - hx, sy - hy))
    e = ease_io(p)
    if kind == "region":
        q = 1 - (1 - e) ** 2
        dist = math.exp(math.log(40000) + (math.log(18000) - math.log(40000)) * q)
        return Camera(*lerp((300, 2000), (300, 1300), e), dist, 44 - 16 * e, -12 + 20 * e)
    if kind == "descend":
        keys = [(0.0, 300, 1300, 18000, 28, 8), (0.45, -6400, 1800, 5200, 50, -70),
                (0.72, -3500, 650, 2600, 52, -10), (1.0, hx, hy, 1500, 48, 40)]
        for (p0, *k0), (p1, *k1) in zip(keys[:-1], keys[1:]):
            if p <= p1:
                q = ease_io((p - p0) / (p1 - p0))
                tx, ty = lerp(k0[:2], k1[:2], q)
                dist = math.exp(math.log(k0[2]) + (math.log(k1[2]) - math.log(k0[2])) * q)
                return Camera(tx, ty, dist, k0[3] + (k1[3] - k0[3]) * q, lerp_angle(k0[4], k1[4], q))
        return Camera(hx, hy, 1500, 48, 40)
    if kind == "home":
        return Camera(hx, hy, 900 - 520 * ease(p * 1.2), 50 + 5 * e, 40 + 30 * e)
    if kind == "home_orbit":
        return Camera(hx, hy, 300 + 40 * e, 58, 70 + 70 * e)
    if kind == "ride":
        x, y, yaw = rt.rail(0, leg_p(rt, 0, t))
        q = leg_p(rt, 0, t)
        dist = 380 + 2600 * math.sin(math.pi * min(1.0, q)) ** 0.8
        blend = ease(q / 0.12)
        return Camera(x, y, dist, 50 - 4 * math.sin(math.pi * q), lerp_angle(140, yaw, blend))
    if kind == "sti":
        return Camera(sx, sy, 600 - 180 * e, 52 + 4 * e, yaw_ride + 15 + 40 * e)
    if kind == "sti_orbit":
        return Camera(sx, sy + 10, 230, 62, yaw_ride + 80 + 60 * e)
    if kind == "to_cathedral":
        x, y, yaw = rt.rail(1, leg_p(rt, 1, t))
        return Camera(x, y, 520 + 160 * math.sin(math.pi * e), 52, lerp_angle(yaw_ride + 140, 60, e))
    if kind == "cathedral":
        return Camera(cx, cy, 520 - 160 * e, 54, 60 + 45 * e)
    if kind == "cathedral_orbit":
        return Camera(cx, cy, 220, 63, 120 + 70 * e)
    if kind == "to_sm":
        x, y, yaw = rt.rail(2, leg_p(rt, 2, t))
        return Camera(x, y, 520 + 900 * math.sin(math.pi * e), 50, lerp_angle(190, yaw, ease(p * 2.5)))
    if kind == "sm":
        return Camera(*lerp(tuple(rt.legs[2]["rail"][-1]), (mx, my), e), 950 - 250 * e, 52, 0 + 35 * e)
    if kind == "sm_orbit":
        return Camera(mx, my, 520, 60, 35 + 70 * e)
    if kind == "pullback":
        tx, ty = lerp((mx, my), (300, 2000), e)
        q = e ** 1.5
        dist = math.exp(math.log(600) + (math.log(42000) - math.log(600)) * q)
        return Camera(tx, ty, dist, 58 - 14 * e, lerp_angle(105, 0, e))
    raise ValueError(kind)


class MipLayer:
    def __init__(self, meta, name, feather_px=60):
        self.layer, img = geo.load_layer(meta, name)
        self.levels = [np.ascontiguousarray(img)]
        while max(self.levels[-1].shape[:2]) > 400:
            self.levels.append(cv2.pyrDown(self.levels[-1]))
        a = np.zeros(img.shape[:2], np.uint8)
        a[feather_px:-feather_px, feather_px:-feather_px] = 255
        a = cv2.GaussianBlur(a, (0, 0), feather_px / 2.2)
        self.alpha = [a]
        while len(self.alpha) < len(self.levels):
            self.alpha.append(cv2.pyrDown(self.alpha[-1]))

    def warp(self, cam, out, full=False):
        mpp = cam.dist / FPX
        lvl = int(np.clip(math.floor(math.log2(max(1.0, mpp / self.layer.res / 1.2))), 0, len(self.levels) - 1))
        img = self.levels[lvl]
        sc = img.shape[1] / self.levels[0].shape[1]
        t2w = self.layer.tex_to_world() @ np.diag([1 / sc, 1 / sc, 1])
        hmat = cam.ground_h() @ t2w
        corners = np.array([[0, 0, 1], [img.shape[1], 0, 1], [img.shape[1], img.shape[0], 1], [0, img.shape[0], 1]], float)
        pc = corners @ hmat.T
        if full:
            warped = cv2.warpPerspective(img, hmat, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=PARCH_BG)
            out[:] = warped
            return
        if np.all(pc[:, 2] > 0):
            uv = pc[:, :2] / pc[:, 2:3]
            if uv[:, 0].max() < 0 or uv[:, 0].min() > W or uv[:, 1].max() < 0 or uv[:, 1].min() > H:
                return
        warped = cv2.warpPerspective(img, hmat, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        a = cv2.warpPerspective(self.alpha[lvl], hmat, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if a.max() == 0:
            return
        af = (a.astype(np.float32) / 255.0)[..., None]
        out[:] = (out * (1 - af) + warped * af).astype(np.uint8)


WALLS = np.array([[236, 226, 204], [242, 214, 160], [226, 178, 150], [196, 214, 190], [182, 204, 220],
                  [240, 196, 196], [236, 232, 220], [214, 190, 228], [250, 220, 120], [200, 226, 214]], np.float32)
ROOFS = np.array([[176, 64, 48], [196, 92, 52], [60, 110, 150], [64, 128, 92], [168, 172, 176],
                  [140, 50, 44], [210, 120, 64], [92, 100, 108]], np.float32)
STI_BLUE = np.array([30, 72, 150], np.float32)
STI_GOLD = np.array([246, 196, 40], np.float32)
SM_BLUE = np.array([20, 82, 168], np.float32)
SM_WALL = np.array([236, 236, 230], np.float32)
CATHEDRAL_WALL = np.array([226, 214, 190], np.float32)


def hex_rgb(s):
    try:
        s = s.strip().lstrip("#")
        if len(s) == 6:
            return np.array([int(s[i:i + 2], 16) for i in (0, 2, 4)], np.float32)
    except (AttributeError, ValueError):
        pass
    named = {"red": (176, 64, 48), "blue": (60, 110, 150), "green": (64, 128, 92), "grey": (160, 162, 166),
             "gray": (160, 162, 166), "white": (236, 236, 232), "brown": (130, 84, 56), "yellow": (240, 210, 90),
             "orange": (214, 124, 60), "black": (60, 60, 64)}
    if isinstance(s, str) and s.lower() in named:
        return np.array(named[s.lower()], np.float32)
    return None


class City:
    def __init__(self):
        bl, centers = geo.buildings()
        self.centers = {k: np.array(v) for k, v in centers.items()}
        self.items = []
        for b in bl:
            h = b["h"]
            if b["hero"] == "home":
                h = 7.0
            if b["hero"] == "cathedral":
                h = 16.0
            land = LAND[b["cluster"]]
            if b["hero"]:
                start = land - LEAD - TAU1
            else:
                start = land - LEAD + 0.25 + (b["dist"] / 420.0) * 1.6 * BAR
            hs = geo.stable_hash(b["id"], "paint")
            col = WALLS[hs % len(WALLS)] * (0.94 + 0.06 * ((hs >> 8) % 100) / 100)
            roof = ROOFS[(hs >> 4) % len(ROOFS)]
            wt, rt = hex_rgb(b.get("wall_tag")), hex_rgb(b.get("roof_tag"))
            if wt is not None:
                col = wt
            if rt is not None:
                roof = rt
            if b["hero"] == "home":
                col, roof = HOME_YELLOW, np.array([150, 60, 44], np.float32)
            elif b["hero"] == "cathedral":
                col, roof = CATHEDRAL_WALL, CATHEDRAL_ROOF
            elif b["hero"] == "sti":
                col, roof = STI_BLUE, STI_GOLD
            elif b["hero"] == "sm":
                col, roof = SM_WALL, SM_BLUE
            self.items.append(dict(fp=b["fp"], c=b["c"], h=h, start=start, col=col, roof=roof, hero=b["hero"],
                                   cluster=b["cluster"]))

    def draw(self, img, cam, t):
        vis = []
        for it in self.items:
            if t < it["start"]:
                continue
            if np.hypot(*(it["c"] - cam.t[:2])) > cam.dist * 1.6 + 300:
                continue
            k = spring(t - it["start"])
            if k <= 0.01:
                continue
            d = np.linalg.norm(np.append(it["c"], 0) - cam.c)
            vis.append((d, it, k))
        vis.sort(key=lambda x: -x[0])
        for d, it, k in vis:
            fp = it["fp"]
            h = it["h"] * k
            n = len(fp)
            base = np.column_stack([fp, np.zeros(n)])
            top = np.column_stack([fp, np.full(n, h)])
            uvb, zb = cam.project(base)
            uvt, zt = cam.project(top)
            if np.isnan(uvb).any() or np.isnan(uvt).any():
                continue
            if uvt[:, 0].max() < -50 or uvt[:, 0].min() > W + 50 or uvt[:, 1].max() < -50 or uvb[:, 1].min() > H + 50:
                continue
            col = it["col"]
            for i in range(n):
                a, b = fp[i], fp[(i + 1) % n]
                nrm = np.array([b[1] - a[1], -(b[0] - a[0])])
                ln = np.linalg.norm(nrm)
                if ln < 1e-6:
                    continue
                nrm /= ln
                mid = (a + b) / 2
                if np.dot(nrm, cam.c[:2] - mid) <= 0:
                    continue
                shade = 0.5 + 0.45 * max(0.0, nrm[0] * LIGHT[0] + nrm[1] * LIGHT[1])
                quad = np.array([uvb[i], uvb[(i + 1) % n], uvt[(i + 1) % n], uvt[i]])
                c = tuple(int(x) for x in np.clip(col * shade, 0, 255))
                cv2.fillPoly(img, [np.round(quad * 4).astype(np.int32)], c, cv2.LINE_AA, shift=2)
            roof = tuple(int(x) for x in np.clip(col * 1.08, 0, 255))
            p = np.round(uvt * 4).astype(np.int32)
            cv2.fillPoly(img, [p], roof, cv2.LINE_AA, shift=2)
            edge = (60, 42, 24)
            cv2.polylines(img, [p], True, edge, 1, cv2.LINE_AA, shift=2)
            if it["hero"]:
                self.cog(img, cam, it, t)

    def cog(self, img, cam, it, t):
        k = spring(t - it["start"])
        r0 = float(np.max(np.linalg.norm(it["fp"] - it["c"], axis=1))) * 1.35 + 6
        teeth = 24
        ang = np.linspace(0, 2 * np.pi, teeth * 4, endpoint=False) + 0.35 * (t - it["start"])
        rr = r0 * (1 + 0.07 * (np.floor(np.arange(teeth * 4) / 2) % 2))
        pts = np.column_stack([it["c"][0] + rr * np.cos(ang), it["c"][1] + rr * np.sin(ang), np.zeros(len(ang))])
        uv, z = cam.project(pts)
        if np.isnan(uv).any():
            return
        thick = max(2, int(round(FPX * 2.2 / max(1.0, np.linalg.norm(np.append(it["c"], 0) - cam.c)))))
        col = tuple(int(x) for x in (np.array([120, 84, 36]) * min(1.0, k)))
        cv2.polylines(img, [np.round(uv * 4).astype(np.int32)], True, col, thick, cv2.LINE_AA, shift=2)


def ring_texture(text, length, height):
    im = Image.new("RGB", (length, height))
    d = ImageDraw.Draw(im)
    for y in range(height):
        k = 1 - abs(y - height / 2) / (height / 2)
        c = tuple(int(v) for v in (np.array([150, 108, 46]) + np.array([80, 70, 50]) * k ** 0.6))
        d.line([(0, y), (length, y)], fill=c)
    d.line([(0, 3), (length, 3)], fill=(80, 56, 24), width=2)
    d.line([(0, height - 4), (length, height - 4)], fill=(80, 56, 24), width=2)
    f = geo.font(int(height * 0.5), 800)
    unit = d.textlength(text, font=f)
    reps = max(1, round(length / unit))
    track = (length / reps - unit) / max(1, len(text))
    x = 0.0
    for _ in range(reps):
        for ch in text:
            d.text((x, height / 2), ch, font=f, fill=(58, 38, 16), anchor="lm")
            x += d.textlength(ch, font=f) + track
    return np.asarray(im)


class Sphere:
    def gl_draw(self, r, img, t, scale, center, align=0.0, spin=0.0):
        rings = []
        for i, (vao, tex) in enumerate(r.ring_vaos):
            rot, phase = self.ring_rot(i, t, align)
            M = np.eye(4)
            M[:3, :3] = rot
            rings.append((vao, tex, M @ gl3d.rot_z(phase)))
        tilt = math.radians(14 * (1 - align) + 6)
        cx, sx = math.cos(tilt), math.sin(tilt)
        rx = np.array([[1, 0, 0, 0], [0, cx, -sx, 0], [0, sx, cx, 0], [0, 0, 0, 1.0]])
        sun = rx @ gl3d.rot_z(t * 0.12 + spin) @ gl3d.scale(0.95, 0.95, 0.95)
        rgba = r.scene.render_sphere(rings, sun, scale, center)
        a = rgba[..., 3:4].astype(np.float32) / 255.0
        img[:] = (img * (1 - a) + rgba[..., :3] * a).astype(np.uint8)

    def __init__(self):
        self.rings = []
        for i, txt in enumerate(RINGS):
            r = 1.0 - 0.14 * i
            length = int(2 * math.pi * 420 * r)
            self.rings.append(dict(r=r, tex=ring_texture(txt, length, 54), w=0.11))
        self.sun = self.sun_sprite(520)

    def sun_sprite(self, s):
        im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        c = s / 2
        for i in range(8):
            a = i * math.pi / 4
            for off, ln, wd, col in ((0, 0.48, 0.10, (190, 140, 52, 255)), (0, 0.46, 0.05, (236, 196, 104, 255))):
                tip = (c + ln * s * math.cos(a), c + ln * s * math.sin(a))
                l1 = (c + 0.2 * s * math.cos(a + wd * 2.2), c + 0.2 * s * math.sin(a + wd * 2.2))
                l2 = (c + 0.2 * s * math.cos(a - wd * 2.2), c + 0.2 * s * math.sin(a - wd * 2.2))
                d.polygon([tip, l1, l2], fill=col)
        for rr, col in ((0.22, (120, 84, 30, 255)), (0.205, (214, 170, 80, 255)), (0.15, (240, 206, 120, 255)), (0.06, (170, 120, 44, 255))):
            d.ellipse([c - rr * s, c - rr * s, c + rr * s, c + rr * s], fill=col)
        return np.asarray(im).astype(np.float32)

    def ring_rot(self, i, t, align):
        a = t * (0.35 + 0.17 * i) + i * 1.3
        tilt_x = math.radians([62, 20, -40][i] * (1 - align) + 68 * align)
        tilt_y = math.radians([0, 55, 30][i]) * (1 - align)
        cx, sx = math.cos(tilt_x), math.sin(tilt_x)
        cy, sy = math.cos(tilt_y), math.sin(tilt_y)
        rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        return ry @ rx, a * (1 - align) + align * 0.0

    def draw(self, img, t, scale, center, align=0.0, spin=0.0):
        cxy = np.array(center, float)
        R = 420 * scale
        D = 5.0
        segs = []
        for i, ring in enumerate(self.rings):
            rot, phase = self.ring_rot(i, t, align)
            nseg = 120
            tex = ring["tex"]
            L = tex.shape[1]
            th = np.linspace(0, 2 * np.pi, nseg + 1) + phase
            r = ring["r"]
            hw = ring["w"]
            for j in range(nseg):
                t0, t1 = th[j], th[j + 1]
                p = np.array([[r * math.cos(t0), r * math.sin(t0), -hw],
                              [r * math.cos(t1), r * math.sin(t1), -hw],
                              [r * math.cos(t0), r * math.sin(t0), hw],
                              [r * math.cos(t1), r * math.sin(t1), hw]])
                pw = p @ rot.T
                if align > 0:
                    flat = np.array([[r * math.cos(t0) * (1 + hw * 0), r * math.sin(t0), 0]])
                tm = (t0 + t1) / 2
                nrm = np.array([math.cos(tm), math.sin(tm), 0]) @ rot.T
                mid = pw.mean(axis=0)
                to_cam = np.array([0, 0, -D]) - mid
                facing = float(np.dot(nrm, to_cam)) > 0
                segs.append((mid[2], i, j, pw, facing, nseg, L))
        sun_depth = 0.0
        segs.sort(key=lambda s: -s[0])
        sun_drawn = False
        for depth, i, j, pw, facing, nseg, L in segs:
            if not sun_drawn and depth < sun_depth:
                self.draw_sun(img, t, cxy, R, spin)
                sun_drawn = True
            z = D + pw[:, 2]
            uv = cxy + R * D * pw[:, :2] / z[:, None] * np.array([1, -1])
            tex = self.rings[i]["tex"]
            u0 = int(j * L / nseg)
            u1 = min(L, int((j + 1) * L / nseg) + 2)
            src = np.float32([[u0, 0], [u1, 0], [u0, tex.shape[0]]])
            dst = np.float32([uv[0], uv[1], uv[2]])
            bx0, by0 = np.floor(uv.min(axis=0)).astype(int) - 2
            bx1, by1 = np.ceil(uv.max(axis=0)).astype(int) + 2
            bx0, by0 = max(bx0, 0), max(by0, 0)
            bx1, by1 = min(bx1, W), min(by1, H)
            if bx1 <= bx0 or by1 <= by0:
                continue
            m = cv2.getAffineTransform(src, dst - np.float32([bx0, by0]))
            patch_src = tex if facing else (tex * 0.45).astype(np.uint8)
            patch = cv2.warpAffine(patch_src, m, (bx1 - bx0, by1 - by0), flags=cv2.INTER_LINEAR)
            mask = np.zeros((by1 - by0, bx1 - bx0), np.uint8)
            quad = np.array([uv[0], uv[1], uv[3], uv[2]]) - np.array([bx0, by0])
            cv2.fillConvexPoly(mask, np.round(quad * 4).astype(np.int32), 255, cv2.LINE_AA, shift=2)
            a = (mask.astype(np.float32) / 255)[..., None]
            roi = img[by0:by1, bx0:bx1]
            roi[:] = (roi * (1 - a) + patch * a).astype(np.uint8)
        if not sun_drawn:
            self.draw_sun(img, t, cxy, R, spin)

    def draw_sun(self, img, t, cxy, R, spin):
        s = self.sun
        size = int(R * 0.95)
        if size < 8:
            return
        ang = math.degrees(t * 0.12 + spin)
        m = cv2.getRotationMatrix2D((s.shape[1] / 2, s.shape[0] / 2), ang, size / s.shape[0])
        m[:, 2] += np.array([cxy[0] - s.shape[1] / 2, cxy[1] - s.shape[0] / 2]) + (np.array([s.shape[1], s.shape[0]]) / 2) * (1 - size / s.shape[0]) * 0
        m = cv2.getRotationMatrix2D((0, 0), ang, size / s.shape[0])
        m[:, 2] = cxy - m[:, :2] @ np.array([s.shape[1] / 2, s.shape[0] / 2])
        x0, y0 = int(max(0, cxy[0] - size)), int(max(0, cxy[1] - size))
        x1, y1 = int(min(W, cxy[0] + size)), int(min(H, cxy[1] + size))
        if x1 <= x0 or y1 <= y0:
            return
        m[:, 2] -= [x0, y0]
        spr = cv2.warpAffine(s, m, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
        a = spr[..., 3:4] / 255.0
        roi = img[y0:y1, x0:x1]
        roi[:] = (roi * (1 - a) + spr[..., :3] * a).astype(np.uint8)


def dark_bg():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2)
    k = np.clip(1 - r * 1.25, 0, 1)[..., None]
    return (np.array([18, 13, 9]) + np.array([46, 32, 18]) * k).astype(np.uint8)


class Titles:
    def __init__(self):
        self.cache = {}

    def sprite(self, main, sub, dark):
        key = (main, sub, dark)
        if key in self.cache:
            return self.cache[key]
        big = 132 if len(main) <= 10 else 96
        f1, f2 = geo.font(big, 800), geo.font(34, 700)
        im = Image.new("RGBA", (W, 300), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        col = GOLD if dark else CREAM

        def spaced(y, s, f, tracking, fill):
            widths = [d.textlength(c, font=f) for c in s]
            total = sum(widths) + tracking * (len(s) - 1)
            x = (W - total) / 2 if dark else 120
            for c, wch in zip(s, widths):
                d.text((x, y), c, font=f, fill=fill, anchor="ls")
                x += wch + tracking
            return total

        spaced(170, main, f1, big * 0.16, col + (255,))
        spaced(238, sub, f2, 34 * 0.32, col + (255,))
        arr = np.asarray(im).astype(np.float32)
        self.cache[key] = arr
        return arr

    def draw(self, img, t, dark):
        for b0, b1, kind, main, sub in TITLES:
            t0, t1 = b0, b1
            if not (t0 <= t < t1):
                continue
            spr = self.sprite(main, sub, dark)
            enter = ease((t - t0) / 0.4)
            exitp = ease((t - (t1 - 0.45)) / 0.45)
            y_top = H - 360 if not dark else H - 330
            if not dark:
                band = np.linspace(0, 1, 420, dtype=np.float32) ** 1.4
                region = img[H - 420:H].astype(np.float32)
                region *= (1 - 0.62 * band)[:, None, None]
                img[H - 420:H] = region.astype(np.uint8)
            offset = int((1 - enter) * 120 - exitp * 120)
            a = spr[..., 3:4] / 255.0
            mask_top, mask_bot = y_top + 40, y_top + 260
            for row0 in range(0, 300):
                pass
            ys = np.arange(300) + y_top + offset
            valid = (ys >= mask_top) & (ys < mask_bot) & (ys >= 0) & (ys < H)
            if not valid.any():
                continue
            src_rows = np.where(valid)[0]
            dst_rows = ys[valid]
            roi = img[dst_rows].astype(np.float32)
            aa = a[src_rows]
            img[dst_rows] = (roi * (1 - aa) + spr[src_rows, :, :3] * aa).astype(np.uint8)


SPLIT_SHADOW = np.array([0.94, 0.98, 1.06], np.float32)
SPLIT_HIGH = np.array([1.05, 1.0, 0.92], np.float32)
HALATION = np.array([1.0, 0.42, 0.12], np.float32)
GRAIN_KERNEL = np.array([[0, 0.125, 0], [0.125, 0.5, 0.125], [0, 0.125, 0]], np.float32)


def grade(img, frame_idx, vign):
    x = img.astype(np.float32) * (1.0 / 255.0)
    lum = x @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    lo = np.clip(1.0 - lum / 0.45, 0, 1)[..., None] ** 2
    hi = np.clip((lum - 0.5) / 0.5, 0, 1)[..., None] ** 2
    x *= 1.0 + (SPLIT_SHADOW - 1.0) * lo + (SPLIT_HIGH - 1.0) * hi
    bright = np.clip(lum - 0.78, 0, None)
    small = cv2.resize(bright, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    halo = cv2.resize(cv2.GaussianBlur(small, (0, 0), 3.0), (W, H), interpolation=cv2.INTER_LINEAR)
    x += halo[..., None] * HALATION * 0.55
    x *= vign
    rng = np.random.default_rng(frame_idx * 7919 + 13)
    g = rng.standard_normal((H // 2, W // 2)).astype(np.float32)
    g = cv2.filter2D(g, -1, GRAIN_KERNEL) * 2.2
    g = cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)
    mid = 4.0 * lum * (1.0 - lum)
    x += (g * (0.004 + 0.012 * mid))[..., None]
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


def vignette():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    return np.clip(1.08 - 0.32 * r ** 2, 0.55, 1.0)[..., None].astype(np.float32)


def halton(i, b):
    f, r = 1.0, 0.0
    while i > 0:
        f /= b
        r += f * (i % b)
        i //= b
    return r


HALTON = [(halton(i + 1, 2), halton(i + 1, 3)) for i in range(16)]


def shot_at(t):
    for s in SHOTS:
        if s[0] <= t < s[1]:
            return s
    return SHOTS[-1]


class Renderer:
    def __init__(self):
        meta = geo.ensure_cache()
        self.meta = meta
        self.city = City()
        self.centers = self.city.centers
        self.sphere = Sphere()
        self.scene = gl3d.Scene(meta, self.city.items, self.centers)
        self.spp = SPP
        self.route = route.Route(self.centers)
        self.route.dump(str(ROOT / "build" / "route_report.json"))
        self.scene.set_route(self.route)
        self.ring_vaos = []
        for ring in self.sphere.rings:
            vao = self.scene.obj_vao(gl3d.band_mesh(ring["r"], ring["w"], 0.035))
            self.ring_vaos.append((vao, self.scene.add_ring_texture(ring["tex"])))
        self.titles = Titles()
        self.bg = dark_bg()
        self.vign = vignette()
        self.pins = [(k, self.centers[k]) for k in ("home", "sti", "cathedral", "sm")]

    def frame(self, n):
        t = n / FPS
        b0, b1, mode, kind = shot_at(t)
        p = (t - b0) / (b1 - b0)
        img = np.empty((H, W, 3), np.uint8)
        dark = mode != "map"
        if mode == "map":
            cam = shot_camera(kind, p, self.centers, self.route, t)
            self.scene.route_progress = self.route.progress(t)
            img[:] = self.accumulate(kind, p, t, b0, b1)
            if kind in ("region", "pullback"):
                self.draw_pins(img, cam, t)
        elif mode == "sphere":
            img[:] = self.bg
            sd = lambda *a, **k: self.sphere.gl_draw(self, img, t, *a, **k)
            if kind == "open":
                sd(0.95 + 0.25 * ease_io(p), (W / 2, H / 2 - 40))
            elif kind == "band":
                self.sphere.gl_draw(self, img, t * 0.4, 2.9 + 0.3 * p, (W / 2 + 120, H / 2 + 980))
            elif kind == "wide":
                sd(1.05 + 1.8 * ease_io(p) ** 2, (W / 2, H / 2), spin=p * 1.5)
            elif kind == "align":
                sd(1.0, (W / 2, H / 2 - 40), align=ease_io(p))
            else:
                sd(1.0 + 0.06 * p, (W / 2, H / 2 - 80), align=1.0)
        else:
            img[:] = (14, 10, 7)
            self.credits(img, t, b0)
        self.titles.draw(img, t, dark)
        return grade(img, n, self.vign)

    def motion_px(self, kind, t, b0, b1, dt):
        def cam_at(tt):
            tt = min(max(tt, b0), b1 - 1e-4)
            return shot_camera(kind, (tt - b0) / (b1 - b0), self.centers, self.route, tt)
        a, b = cam_at(t - dt / 2), cam_at(t + dt / 2)
        pts = []
        for fx, fy in ((0.5, 0.5), (0.2, 0.25), (0.8, 0.25), (0.2, 0.75), (0.8, 0.75)):
            ray = a.R[0] * (fx - 0.5) * W / FPX - a.R[1] * (fy - 0.5) * H / FPX + a.R[2]
            if ray[2] >= -1e-3:
                continue
            s = -a.c[2] / ray[2]
            pts.append(a.c + ray * s)
        if not pts:
            return 0.0
        pts = np.array(pts)
        ua, _ = a.project(pts)
        ub, _ = b.project(pts)
        d = np.linalg.norm(ua - ub, axis=1)
        d = d[np.isfinite(d)]
        return float(d.max()) if len(d) else 0.0

    def accumulate(self, kind, p, t, b0, b1):
        shutter = 0.5 / FPS
        blur = self.motion_px(kind, t, b0, b1, shutter)
        n = int(np.clip(math.ceil(blur / 6.0), self.spp, 12)) if self.spp > 1 else 1
        self.last_spp, self.last_blur = n, blur
        if n <= 1:
            cam = shot_camera(kind, p, self.centers, self.route, t)
            return self.scene.render_map(cam, t, spring)
        acc = np.zeros((H, W, 3), np.float32)
        order = np.random.default_rng(int(t * FPS)).permutation(n)
        for i in range(n):
            ts = t + ((order[i] + 0.5) / n - 0.5) * shutter
            ts = min(max(ts, b0), b1 - 1e-4)
            ps = (ts - b0) / (b1 - b0)
            cam = shot_camera(kind, ps, self.centers, self.route, ts)
            jx, jy = HALTON[i % len(HALTON)]
            self.scene.jitter = ((jx - 0.5) * 2.0 / W, (jy - 0.5) * 2.0 / H)
            self.scene.route_progress = self.route.progress(ts)
            acc += self.scene.render_map(cam, ts, spring)
        self.scene.jitter = (0.0, 0.0)
        return np.clip(acc / n + 0.5, 0, 255).astype(np.uint8)

    def haze(self, img, cam):
        pitch_k = min(1.0, max(0.0, (math.degrees(math.acos(max(-1, min(1, -cam.R[2, 2])))) - 30) / 40))
        if pitch_k <= 0:
            return
        g = np.clip(1 - np.arange(H, dtype=np.float32) / (H * 0.55), 0, 1) ** 1.6 * 0.55 * pitch_k
        col = np.array(PARCH_BG, np.float32)
        img[:] = (img * (1 - g[:, None, None]) + col * g[:, None, None]).astype(np.uint8)

    def draw_pins(self, img, cam, t):
        labels = {"home": "HOME", "sti": "STI", "cathedral": "CATHEDRAL", "sm": "SM CITY"}
        pil = None
        for k, c in self.pins:
            uv = self.scene.project(self.scene.ground_point(c[0], c[1]))
            if np.isnan(uv).any():
                continue
            u, v = uv[0]
            if not (0 <= u < W and 0 <= v < H):
                continue
            col = (236, 190, 52) if k == "home" else (150, 104, 40)
            cv2.circle(img, (int(u), int(v)), 11, (52, 36, 22), -1, cv2.LINE_AA)
            cv2.circle(img, (int(u), int(v)), 8, col, -1, cv2.LINE_AA)
        return pil

    def credits(self, img, t, b0):
        t0 = b0
        pil = Image.fromarray(img)
        d = ImageDraw.Draw(pil)
        f = geo.font(30, 700)
        y = H / 2 - 90
        for i, line in enumerate(CREDITS):
            k = ease((t - t0 - 0.35 * i) / 0.6)
            if k <= 0:
                continue
            yy = y + i * 60 + (1 - k) * 30
            d.text((W / 2, yy), line, font=f, fill=GOLD, anchor="mm")
        img[:] = np.asarray(pil)
        clip_top = int(H / 2 - 130)
        img[: clip_top] = (14, 10, 7)


def main():
    ap = argparse.ArgumentParser(description="Render the Lipa map-sequence film to H.264.")
    ap.add_argument("--out", default="build/video_only.mp4")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=NFRAMES)
    ap.add_argument("--stills", nargs="*", type=float, help="render PNG stills at these seconds instead")
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--raw", help="write raw rgb24 frames to this file instead of piping to x264")
    ap.add_argument("--spp", type=int, default=3, help="jittered sub-samples per frame (motion blur + AA)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    r = Renderer()
    r.spp = max(1, args.spp)
    if args.stills:
        Path("build/stills").mkdir(parents=True, exist_ok=True)
        for s in args.stills:
            n = int(round(s * FPS))
            t0 = time.time()
            fr = r.frame(n)
            cv2.imwrite(f"build/stills/f{n:05d}.png", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            log.info("still %.2fs frame %d in %.2fs", s, n, time.time() - t0)
        return 0
    if args.raw:
        t0 = time.time()
        with open(args.raw, "wb") as fh:
            for n in range(args.start, args.end):
                fh.write(r.frame(n).tobytes())
                if n % 24 == 0:
                    el = time.time() - t0
                    done = n - args.start + 1
                    log.info("frame %d/%d  %.2f s/frame  eta %.0f s", n, args.end, el / done, el / done * (args.end - n))
        log.info("wrote %s in %.0f s", args.raw, time.time() - t0)
        return 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf), "-x264-params", "rc-lookahead=10:ref=3",
           "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
           "-threads", "1", "-movflags", "+faststart", args.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    try:
        for n in range(args.start, args.end):
            proc.stdin.write(r.frame(n).tobytes())
            if n % 48 == 0:
                el = time.time() - t0
                done = n - args.start + 1
                log.info("frame %d/%d  %.2f s/frame  eta %.0f s", n, args.end, el / done, el / done * (args.end - n))
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        log.error("ffmpeg exited %d", rc)
        return rc
    log.info("wrote %s in %.0f s", args.out, time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
