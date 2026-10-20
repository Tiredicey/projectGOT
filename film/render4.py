import argparse
import logging
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import chamber
import gl3d
import glsl
import machine as mc
import render as r3
import route
import timeline
import timeline4 as tl
from timeline import FPS, H, W

log = logging.getLogger("render4")
ROOT = Path(__file__).resolve().parent
KANAL = ROOT / "build" / "kanal"

timeline.LEGS[:] = tl.LEGS4
route.LEGS[:] = tl.LEGS4
r3.TITLES[:] = tl.TITLES4
r3.CREDITS[:] = tl.CREDITS4
r3.SHOTS[:] = [s for s in tl.SHOTS4]

BOWL_C = (300.0, 1300.0)
BOWL_R = 4600.0


class FreeCam:
    def __init__(self, c, target, near=5.0, far=120000.0, fog=60000.0, up=(0, 0, 1.0)):
        c = np.asarray(c, float)
        target = np.asarray(target, float)
        f = target - c
        f /= np.linalg.norm(f)
        upv = np.asarray(up, float)
        if abs(np.dot(f, upv)) > 0.98:
            upv = np.array([0.0, 1.0, 0.0])
        rr = np.cross(f, upv)
        rr /= np.linalg.norm(rr)
        u = np.cross(rr, f)
        self.c, self.R = c, np.stack([rr, u, f])
        self.t = np.array([target[0], target[1], 0.0])
        self.dist = float(np.linalg.norm(target - c))
        self.free = True
        self.near, self.far, self.fog = near, far, fog


def ease(p):
    p = min(1.0, max(0.0, p))
    return p * p * (3 - 2 * p)


def ease_io(p):
    return r3.ease_io(p)


def bowl_cam(p, R, phase="in"):
    cx, cy = BOWL_C
    centre = np.array([cx, cy, R])
    if phase == "in":
        az = math.radians(205 + 50 * p)
        back = R * (0.80 - 0.18 * p)
        lift = R * (0.30 - 0.12 * p)
        look_el = -0.48 + 0.10 * p
    else:
        az = math.radians(255 + 50 * p)
        back = R * (0.62 + 0.22 * p)
        lift = R * (0.18 + 0.14 * p)
        look_el = -0.40 - 0.10 * p
    fwd = np.array([math.cos(az), math.sin(az), 0.0])
    c = centre - fwd * back + np.array([0.0, 0.0, lift])
    d = fwd + np.array([0.0, 0.0, look_el])
    d /= np.linalg.norm(d)
    target = centre + d * R * 0.9
    return FreeCam(c, target, near=20.0, far=R * 5, fog=R * 4.5)


_shot_camera_v3 = r3.shot_camera


def shot_camera4(kind, p, centers, rt=None, t=0.0):
    hx, hy = centers["home"]
    if kind == "descend4":
        keys = [(0.0, 300, 1300, 16000, 32, 8), (0.55, -6400, 1800, 5200, 52, -70),
                (0.80, -3800, 900, 2600, 54, -20), (1.0, hx, hy, 1500, 50, 40)]
        for (p0, *k0), (p1, *k1) in zip(keys[:-1], keys[1:]):
            if p <= p1:
                q = r3.ease_io((p - p0) / (p1 - p0))
                tx, ty = r3.lerp(k0[:2], k1[:2], q)
                dist = math.exp(math.log(k0[2]) + (math.log(k1[2]) - math.log(k0[2])) * q)
                return r3.Camera(tx, ty, dist, k0[3] + (k1[3] - k0[3]) * q, r3.lerp_angle(k0[4], k1[4], q))
        return r3.Camera(hx, hy, 1500, 50, 40)
    if kind == "ride_a":
        return _shot_camera_v3("ride", p, centers, rt, t)
    if kind == "pullback4":
        return _shot_camera_v3("pullback", p, centers, rt, t)
    return _shot_camera_v3(kind, p, centers, rt, t)


r3.shot_camera = shot_camera4


class Renderer4(r3.Renderer):
    def __init__(self):
        super().__init__()
        self.chamber = chamber.Chamber(self.scene.ctx, self.scene.fbo_ms, self.scene.read)
        self.panels = {k: mc.Panel(str(KANAL / f"{k}.mp4")) for k in tl.PANELS}
        self.astro_on = False
        self.scene.extra = self.draw_astro
        self.astro_t = 0.0

    def draw_astro(self, VP, LVP, c, fog_dist, t):
        if not self.astro_on:
            return
        sc = self.scene
        p = sc.p_obj
        R = glsl.STATE["bend"][3]
        cx, cy = BOWL_C
        centre = np.array([cx, cy, R])
        saved = glsl.STATE["bend"]
        glsl.STATE["bend"] = (0.0, 0.0, 0.0, 0.0)
        sc.set_common(p, VP, np.eye(4), c, fog_dist * 4, 0, gl3d.SUN_DIR)
        glsl.STATE["bend"] = saved
        p["use_fog"].value = 0
        size = R * 0.11
        for i, (vao, tex) in enumerate(self.ring_vaos):
            rot, phase = self.sphere.ring_rot(i, t, 0.0)
            M = np.eye(4)
            M[:3, :3] = rot
            tex.use(0)
            p["tex"].value = 0
            p["M"].write(gl3d.m4(mc.T(*centre) @ mc.S(size) @ M @ gl3d.rot_z(phase)))
            p["use_tex"].value = 1
            p["tex_dim"].value = 1.0
            p["spec"].value = 0.8
            p["emis"].value = 0.12
            vao.render()
        p["M"].write(gl3d.m4(mc.T(*centre) @ mc.S(size * 0.95) @ gl3d.rot_z(t * 0.12)))
        p["use_tex"].value = 0
        p["base"].value = (0.95, 0.72, 0.30)
        p["spec"].value = 1.2
        p["emis"].value = 0.55
        sc.vao_sun.render()
        p["emis"].value = 0.0

    def bowl_frame(self, t, p, kind):
        if kind == "hollow":
            R = BOWL_R
            cam = bowl_cam(ease_io(p), R, "in")
        elif kind == "hollow_out":
            R = BOWL_R
            cam = bowl_cam(ease_io(p), R, "out")
        else:
            raise ValueError(kind)
        cx, cy = BOWL_C
        glsl.STATE["bend"] = (cx, cy, 0.0, R)
        glsl.STATE["sunp"] = (cx, cy, R, 0.55)
        self.astro_on = True
        try:
            img = self.scene.render_map(cam, t, r3.spring)
        finally:
            glsl.STATE["bend"] = (0.0, 0.0, 0.0, 0.0)
            glsl.STATE["sunp"] = (0.0, 0.0, 0.0, 0.0)
            self.astro_on = False
        return img

    def chamber_frame(self, t, b0, b1, key):
        lt = t - b0
        pan = self.panels[key]
        src_len = pan.n / FPS if pan.n else 10.0
        start = max(0.0, (src_len - (b1 - b0)) * 0.5)
        idx = int(round((start + lt) * FPS))
        img, _ = pan.frame(idx)
        seed = list(tl.PANELS).index(key) + 1
        return self.chamber.render(t, lt, img, seed, ease_io(lt / (b1 - b0)))

    def frame(self, n):
        t = n / FPS
        b0, b1, mode, kind = r3.shot_at(t)
        p = (t - b0) / (b1 - b0)
        if mode == "chamber":
            img = self.chamber_frame(t, b0, b1, kind).copy()
            self.titles.draw(img, t, False)
            return r3.grade(img, n, self.vign)
        if mode == "map" and kind in ("hollow", "hollow_out"):
            img = self.accum_bowl(t, b0, b1, kind)
            self.titles.draw(img, t, False)
            return r3.grade(img, n, self.vign)
        return super().frame(n)

    def accum_bowl(self, t, b0, b1, kind):
        nsub = max(1, self.spp)
        shutter = 0.5 / FPS
        acc = np.zeros((H, W, 3), np.float32)
        for i in range(nsub):
            ts = min(max(t + ((i + 0.5) / nsub - 0.5) * shutter, b0), b1 - 1e-4)
            jx, jy = r3.HALTON[i % len(r3.HALTON)]
            self.scene.jitter = ((jx - 0.5) * 2.0 / W, (jy - 0.5) * 2.0 / H)
            acc += self.bowl_frame(ts, (ts - b0) / (b1 - b0), kind)
        self.scene.jitter = (0.0, 0.0)
        return np.clip(acc / nsub + 0.5, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="Render the v4 orbital-machine cut.")
    ap.add_argument("--stills", nargs="*", type=float)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=tl.NFRAMES)
    ap.add_argument("--raw")
    ap.add_argument("--spp", type=int, default=3)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rr = Renderer4()
    rr.spp = max(1, args.spp)
    if args.stills:
        Path("build/stills4").mkdir(parents=True, exist_ok=True)
        for s in args.stills:
            n = int(round(s * FPS))
            t0 = time.time()
            fr = rr.frame(n)
            cv2.imwrite(f"build/stills4/f{n:05d}.png", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            log.info("still %.2fs frame %d in %.2fs", s, n, time.time() - t0)
        return 0
    t0 = time.time()
    with open(args.raw, "wb") as fh:
        for n in range(args.start, args.end):
            fh.write(rr.frame(n).tobytes())
            if n % 24 == 0:
                el = time.time() - t0
                done = n - args.start + 1
                log.info("frame %d/%d  %.2f s/frame  eta %.0f s", n, args.end, el / done, el / done * (args.end - n))
    log.info("wrote %s in %.0f s", args.raw, time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
