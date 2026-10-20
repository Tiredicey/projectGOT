import math

import cv2
import numpy as np

BRASS = (0.70, 0.50, 0.22)
BRASS_DARK = (0.42, 0.29, 0.13)
COPPER = (0.66, 0.36, 0.20)
IRON = (0.17, 0.15, 0.13)
STEEL = (0.42, 0.42, 0.40)


def _v(p, n, uv):
    return (p[0], p[1], p[2], n[0], n[1], n[2], uv[0], uv[1])


def _quad(out, a, b, c, d, n, uva=(0, 0), uvb=(1, 0), uvc=(1, 1), uvd=(0, 1)):
    out += [_v(a, n, uva), _v(b, n, uvb), _v(c, n, uvc), _v(a, n, uva), _v(c, n, uvc), _v(d, n, uvd)]


def _normal(a, b, c):
    n = np.cross(np.subtract(b, a), np.subtract(c, a))
    ln = np.linalg.norm(n)
    return tuple(n / ln) if ln > 1e-12 else (0.0, 0.0, 1.0)


def extrude(poly, z0, z1, hole=None):
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        nrm = _normal((*a, z0), (*b, z0), (*b, z1))
        _quad(out, (*a, z0), (*b, z0), (*b, z1), (*a, z1), nrm, (i / n, 0), ((i + 1) / n, 0), ((i + 1) / n, 1), (i / n, 1))
    if hole is not None:
        for i in range(n):
            a, b = hole[i], hole[(i + 1) % n]
            nrm = _normal((*b, z0), (*a, z0), (*a, z1))
            _quad(out, (*b, z0), (*a, z0), (*a, z1), (*b, z1), nrm)
        for z, up in ((z1, True), (z0, False)):
            nz = (0, 0, 1.0) if up else (0, 0, -1.0)
            for i in range(n):
                a, b = poly[i], poly[(i + 1) % n]
                ia, ib = hole[i], hole[(i + 1) % n]
                if up:
                    _quad(out, (*ia, z), (*a, z), (*b, z), (*ib, z), nz)
                else:
                    _quad(out, (*ib, z), (*b, z), (*a, z), (*ia, z), nz)
    else:
        cx = sum(p[0] for p in poly) / n
        cy = sum(p[1] for p in poly) / n
        for z, up in ((z1, True), (z0, False)):
            nz = (0, 0, 1.0) if up else (0, 0, -1.0)
            for i in range(n):
                a, b = poly[i], poly[(i + 1) % n]
                tri = ((cx, cy, z), (*a, z), (*b, z)) if up else ((cx, cy, z), (*b, z), (*a, z))
                out += [_v(p, nz, (0.5, 0.5)) for p in tri]
    return out


def _circle(r, n, phase=0.0):
    return [(r * math.cos(phase + 2 * math.pi * i / n), r * math.sin(phase + 2 * math.pi * i / n)) for i in range(n)]


def gear(teeth, width=1.0, spokes=5):
    p = 2 * math.pi / teeth
    m = 2.0 / teeth
    rt, rr = 1.0 + m, 1.0 - 1.25 * m
    prof = []
    for k in range(teeth):
        a = k * p
        for da, r in ((-0.5, rr), (-0.27, rr), (-0.13, rt), (0.13, rt), (0.27, rr)):
            prof.append((r * math.cos(a + da * p), r * math.sin(a + da * p)))
    rim_in = rr - 0.14
    rim = [(rim_in * math.cos(math.atan2(y, x)), rim_in * math.sin(math.atan2(y, x))) for x, y in prof]
    out = extrude(prof, -width / 2, width / 2, hole=rim)
    hub_r, bore = 0.30, 0.14
    out += extrude(_circle(hub_r, 32), -width * 0.75, width * 0.75, hole=_circle(bore, 32))
    for s in range(spokes):
        a = 2 * math.pi * s / spokes
        ca, sa = math.cos(a), math.sin(a)
        px, py = -sa * 0.06, ca * 0.06
        sp = [(hub_r * 0.9 * ca - px, hub_r * 0.9 * sa - py), (rim_in * 1.02 * ca - px, rim_in * 1.02 * sa - py),
              (rim_in * 1.02 * ca + px, rim_in * 1.02 * sa + py), (hub_r * 0.9 * ca + px, hub_r * 0.9 * sa + py)]
        out += extrude(sp, -width * 0.32, width * 0.32)
    return np.array(out, "f4")


def ring_gear(teeth, width=1.0, rim=0.16):
    p = 2 * math.pi / teeth
    m = 2.0 / teeth
    rt, rr = 1.0 - m, 1.0 + 1.25 * m
    prof = []
    for k in range(teeth):
        a = k * p + p / 2
        for da, r in ((-0.5, rr), (-0.27, rr), (-0.13, rt), (0.13, rt), (0.27, rr)):
            prof.append((r * math.cos(a + da * p), r * math.sin(a + da * p)))
    ro = rr + rim
    outer = [(ro * math.cos(math.atan2(y, x)), ro * math.sin(math.atan2(y, x))) for x, y in prof]
    return np.array(extrude(outer, -width / 2, width / 2, hole=prof), "f4")


def cylinder(n=32, inward=False, caps=True):
    out = []
    for i in range(n):
        a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
        c0, s0, c1, s1 = math.cos(a0), math.sin(a0), math.cos(a1), math.sin(a1)
        nm = (math.cos((a0 + a1) / 2), math.sin((a0 + a1) / 2), 0.0)
        if inward:
            nm = (-nm[0], -nm[1], 0.0)
        _quad(out, (c0, s0, 0), (c1, s1, 0), (c1, s1, 1), (c0, s0, 1), nm,
              (i / n, 0), ((i + 1) / n, 0), ((i + 1) / n, 1), (i / n, 1))
        if caps:
            for z, nz in ((1.0, 1.0), (0.0, -1.0)):
                out += [_v((0, 0, z), (0, 0, nz), (0.5, 0.5)), _v((c0, s0, z), (0, 0, nz), (0.5, 0.5)),
                        _v((c1, s1, z), (0, 0, nz), (0.5, 0.5))]
    return np.array(out, "f4")


def annulus(r0, r1, n=64, down=False):
    out = []
    nz = (0, 0, -1.0) if down else (0, 0, 1.0)
    for i in range(n):
        a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
        p = [(r0 * math.cos(a0), r0 * math.sin(a0), 0), (r1 * math.cos(a0), r1 * math.sin(a0), 0),
             (r1 * math.cos(a1), r1 * math.sin(a1), 0), (r0 * math.cos(a1), r0 * math.sin(a1), 0)]
        uv = [(r0 * math.cos(a0) / r1 * 0.5 + 0.5, r0 * math.sin(a0) / r1 * 0.5 + 0.5),
              (math.cos(a0) * 0.5 + 0.5, math.sin(a0) * 0.5 + 0.5),
              (math.cos(a1) * 0.5 + 0.5, math.sin(a1) * 0.5 + 0.5),
              (r0 * math.cos(a1) / r1 * 0.5 + 0.5, r0 * math.sin(a1) / r1 * 0.5 + 0.5)]
        _quad(out, *p, nz, *uv)
    return np.array(out, "f4")


def box():
    return np.array(extrude([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)], -0.5, 0.5), "f4")


def screen():
    out = []
    _quad(out, (-0.5, 0, -0.5), (0.5, 0, -0.5), (0.5, 0, 0.5), (-0.5, 0, 0.5), (0, -1.0, 0),
          (0, 1), (1, 1), (1, 0), (0, 0))
    return np.array(out, "f4")


def dome(n=32, rings=10):
    out = []
    for j in range(rings):
        t0, t1 = (math.pi / 2) * j / rings, (math.pi / 2) * (j + 1) / rings
        for i in range(n):
            a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
            pts = [(math.cos(t) * math.cos(a), math.cos(t) * math.sin(a), math.sin(t))
                   for t, a in ((t0, a0), (t0, a1), (t1, a1), (t1, a0))]
            out += [_v(p, p, (i / n, j / rings)) for p in (pts[0], pts[1], pts[2], pts[0], pts[2], pts[3])]
    return np.array(out, "f4")


def sphere(n=24, rings=12):
    d = dome(n, rings // 2)
    flip = d.copy()
    flip[:, 2] *= -1
    flip[:, 5] *= -1
    flip = flip.reshape(-1, 3, 8)[:, ::-1].reshape(-1, 8)
    return np.vstack([d, flip])


def gable(length=1.0, width=1.0, height=1.0):
    out = []
    hw, hl = width / 2, length / 2
    a, b, c, d = (-hl, -hw, 0.0), (hl, -hw, 0.0), (hl, hw, 0.0), (-hl, hw, 0.0)
    r0, r1 = (-hl, 0.0, height), (hl, 0.0, height)
    _quad(out, a, b, r1, r0, _normal(a, b, r1))
    _quad(out, c, d, r0, r1, _normal(c, d, r0))
    for tri in ((a, r0, d), (b, c, r1)):
        n = _normal(*tri)
        out += [_v(p, n, (0, 0)) for p in tri]
    return np.array(out, "f4")


def rot(axis, ang):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = math.cos(ang), math.sin(ang)
    C = 1 - c
    m = np.eye(4)
    m[:3, :3] = [[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                 [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                 [z * x * C - y * s, z * y * C + x * s, c + z * z * C]]
    return m


def T(x, y, z):
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def S(x, y=None, z=None):
    return np.diag([x, x if y is None else y, x if z is None else z, 1.0])


def frame_to(z_axis, x_hint=(0, 0, 1.0)):
    z = np.asarray(z_axis, float)
    z /= np.linalg.norm(z)
    x = np.cross(x_hint, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross((1.0, 0, 0), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2] = x, y, z
    return m


def transform(mesh, M):
    out = mesh.copy()
    p = np.column_stack([mesh[:, :3], np.ones(len(mesh))]) @ M.T
    out[:, :3] = p[:, :3]
    nm = mesh[:, 3:6] @ np.linalg.inv(M[:3, :3])
    nm /= np.linalg.norm(nm, axis=1, keepdims=True) + 1e-12
    out[:, 3:6] = nm
    return out.astype("f4")


def merge(parts):
    return np.vstack([transform(m, M) for m, M in parts]).astype("f4")


def train(radii, ang0, omega, t):
    out = []
    phi = ang0
    theta = omega * t
    for i, r in enumerate(radii):
        teeth = max(8, int(round(r * 4)))
        if i > 0:
            prev_r, prev_n, prev_th = out[-1][0], out[-1][1], out[-1][3]
            theta = phi + math.pi - math.pi / teeth - (prev_n / teeth) * (prev_th - phi)
        out.append((r, teeth, phi, theta))
        phi += [0.35, -0.55, 0.45, -0.3, 0.5][i % 5]
    return out


def telescope(length_max, ext, stages=3):
    seg = length_max / stages * 1.06
    total = seg + (length_max - seg) * float(np.clip(ext, 0, 1))
    return [(i * (total - seg) / max(1, stages - 1), seg, 1.0 - 0.18 * i) for i in range(stages)]


class Panel:
    def __init__(self, path, size=(768, 432)):
        self.path = path
        self.size = size
        self.cap = None
        self.n = 0
        self.last = -2
        self.img = np.zeros((size[1], size[0], 3), np.uint8)

    def frame(self, idx):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.path)
            self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = int(np.clip(idx, 0, max(0, self.n - 1)))
        if idx == self.last:
            return self.img, False
        if idx != self.last + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = self.cap.read()
        if ok:
            fr = cv2.resize(fr, self.size, interpolation=cv2.INTER_AREA)
            self.img = np.ascontiguousarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        self.last = idx
        return self.img, True

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.last = -2
