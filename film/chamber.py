import math

import moderngl
import numpy as np

import machine as mc
from gl3d import FPX, look, m4, perspective
from timeline import H, W

CH_VS = """
#version 330
uniform mat4 VP;
uniform mat4 M;
in vec3 in_pos;
in vec3 in_n;
in vec2 in_uv;
out vec3 w;
out vec3 n;
out vec2 uv;
out vec3 lp;
void main() {
    vec4 p = M * vec4(in_pos, 1.0);
    w = p.xyz;
    n = normalize(mat3(M) * in_n);
    uv = in_uv;
    lp = in_pos;
    gl_Position = VP * p;
}
"""

CH_FS = """
#version 330
in vec3 w;
in vec3 n;
in vec2 uv;
in vec3 lp;
out vec4 frag;
uniform vec3 cam;
uniform vec3 base;
uniform float metal;
uniform float rough;
uniform int mode;
uniform sampler2D tex;
uniform vec3 lamp;
uniform vec3 lamp_col;
uniform vec3 fill_dir;
uniform float emissive;
uniform float fog_k;
float hash(vec3 p) { return fract(sin(dot(p, vec3(12.9898, 78.233, 37.719))) * 43758.5453); }
float vnoise(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = mix(mix(hash(i), hash(i + vec3(1, 0, 0)), f.x), mix(hash(i + vec3(0, 1, 0)), hash(i + vec3(1, 1, 0)), f.x), f.y);
    float b = mix(mix(hash(i + vec3(0, 0, 1)), hash(i + vec3(1, 0, 1)), f.x), mix(hash(i + vec3(0, 1, 1)), hash(i + vec3(1, 1, 1)), f.x), f.y);
    return mix(a, b, f.z);
}
void main() {
    vec3 nn = normalize(n);
    vec3 v = normalize(cam - w);
    if (dot(nn, v) < 0.0) nn = -nn;
    if (mode == 1) {
        vec3 c = texture(tex, uv).rgb;
        float lum = dot(c, vec3(0.299, 0.587, 0.114));
        vec3 sep = mix(c, lum * vec3(1.10, 0.92, 0.70), 0.25);
        vec2 e = abs(uv - 0.5) * 2.0;
        float vig = 1.0 - 0.35 * pow(max(e.x, e.y), 6.0);
        frag = vec4(sep * emissive * vig, 1.0);
        return;
    }
    vec3 alb = base;
    float tarnish = vnoise(lp * 6.0) * 0.6 + vnoise(lp * 23.0) * 0.4;
    alb *= 0.78 + 0.34 * tarnish;
    float r = clamp(rough + 0.25 * (tarnish - 0.5), 0.08, 1.0);
    vec3 L = normalize(lamp - w);
    float dist = length(lamp - w);
    float att = 1.0 / (1.0 + dist * dist * 0.0009);
    float diff = max(dot(nn, L), 0.0);
    vec3 h = normalize(v + L);
    float shin = mix(180.0, 12.0, r);
    vec3 F0 = mix(vec3(0.04), alb, metal);
    float fres = pow(1.0 - max(dot(nn, v), 0.0), 5.0);
    vec3 spec = (F0 + (1.0 - F0) * fres) * pow(max(dot(nn, h), 0.0), shin) * (shin + 2.0) / 8.0;
    float fill = max(dot(nn, normalize(fill_dir)), 0.0);
    vec3 refl = reflect(-v, nn);
    vec3 env = mix(vec3(0.05, 0.035, 0.022), vec3(0.55, 0.40, 0.22), clamp(refl.z * 0.5 + 0.5, 0.0, 1.0));
    spec = min(spec, vec3(2.5));
    vec3 c = alb * (1.0 - metal) * (0.06 + diff * att * lamp_col) + spec * att * lamp_col
           + alb * fill * 0.10 + env * F0 * (0.35 + 0.65 * fres) * (1.0 - r * 0.6);
    float d = length(cam - w);
    c = mix(c, vec3(0.045, 0.032, 0.022), 1.0 - exp(-d * fog_k));
    c = c / (1.0 + max(max(c.r, c.g), c.b) * 0.35);
    frag = vec4(c, 1.0);
}
"""


class Chamber:
    def __init__(self, ctx, fbo_ms, read_fn):
        self.ctx = ctx
        self.fbo_ms = fbo_ms
        self.read = read_fn
        self.prog = ctx.program(vertex_shader=CH_VS, fragment_shader=CH_FS)
        self.meshes = {}
        for name, data in (("g8", mc.gear(8)), ("g12", mc.gear(12)), ("g16", mc.gear(16)), ("g24", mc.gear(24)),
                           ("g32", mc.gear(32)), ("g48", mc.gear(48, spokes=6)), ("ring", mc.ring_gear(60)),
                           ("cyl", mc.cylinder()), ("box", mc.box()), ("screen", mc.screen()),
                           ("annulus", mc.annulus(0.80, 1.0)), ("dome", mc.dome()), ("sphere", mc.sphere())):
            self.meshes[name] = ctx.vertex_array(self.prog, [(ctx.buffer(data), "3f 3f 2f", "in_pos", "in_n", "in_uv")])
        self.tex = ctx.texture((768, 432), 3)
        self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.prog["tex"].value = 3

    def draw(self, name, M, base, metal=1.0, rough=0.35, mode=0, emissive=1.0):
        p = self.prog
        p["M"].write(m4(M))
        p["base"].value = base
        p["metal"].value = metal
        p["rough"].value = rough
        p["mode"].value = mode
        p["emissive"].value = emissive
        self.meshes[name].render()

    def gear_at(self, teeth, pos, axis_m, radius, width, ang, base=mc.BRASS, rough=0.32):
        key = {8: "g8", 12: "g12", 16: "g16", 24: "g24", 32: "g32", 48: "g48"}[teeth]
        M = mc.T(*pos) @ axis_m @ mc.rot((0, 0, 1), ang) @ mc.S(radius, radius, width)
        self.draw(key, M, base, rough=rough)

    def render(self, t, local_t, panel_img, seed, cam_p):
        ctx = self.ctx
        if panel_img is not None:
            self.tex.write(panel_img.tobytes())
        self.tex.use(3)
        rng = np.random.default_rng(seed)
        e = cam_p
        orbit = math.radians(-24 + 48 * e)
        dist = 34.0 - 6.0 * e
        target = np.array([0.0, 0.0, 0.2])
        cam = target + np.array([math.sin(orbit) * dist, -math.cos(orbit) * dist, 2.2 + 3.6 * (1 - e)])
        f = target - cam
        f /= np.linalg.norm(f)
        r = np.cross(f, (0, 0, 1.0))
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        V = look(cam, r, u, f)
        jx, jy = getattr(self, "jitter", (0.0, 0.0))
        P = perspective(0.5, 400.0)
        P[0, 2] += jx
        P[1, 2] += jy
        VP = P @ V
        p = self.prog
        p["VP"].write(m4(VP))
        p["cam"].value = tuple(cam)
        p["lamp"].value = (-4.0, -20.0, 9.0)
        p["lamp_col"].value = (3.4, 2.7, 1.9)
        p["fill_dir"].value = (0.6, -0.3, -0.4)
        p["fog_k"].value = 0.006
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE)
        self.fbo_ms.use()
        self.fbo_ms.clear(0.04, 0.03, 0.02, 1.0, depth=1.0)
        face = mc.rot((1, 0, 0), math.pi / 2)
        sw, sh = 16.0, 9.0
        open_k = min(1.0, max(0.0, local_t / 0.55))
        open_k = 1 - (1 - open_k) ** 3
        self.draw("screen", mc.T(0, -0.02, 0) @ mc.S(sw * (0.25 + 0.75 * open_k), 1, sh * (0.25 + 0.75 * open_k)),
                  (1, 1, 1), mode=1, emissive=0.15 + 0.95 * open_k)
        fr = 0.55
        for x, z, lx, lz in ((0, sh / 2 + fr / 2, sw + 2 * fr, fr), (0, -sh / 2 - fr / 2, sw + 2 * fr, fr),
                             (sw / 2 + fr / 2, 0, fr, sh), (-sw / 2 - fr / 2, 0, fr, sh)):
            self.draw("box", mc.T(x, 0.25, z) @ mc.S(lx, 0.9, lz), mc.BRASS, rough=0.28)
        for x, z, lx, lz in ((0, sh / 2 + fr + 0.18, sw + 2 * fr + 0.6, 0.36), (0, -sh / 2 - fr - 0.18, sw + 2 * fr + 0.6, 0.36)):
            self.draw("box", mc.T(x, 0.45, z) @ mc.S(lx, 1.1, lz), mc.BRASS_DARK, rough=0.4)
        for sx in (-1, 1):
            for sz in (-1, 1):
                self.draw("cyl", mc.T(sx * (sw / 2 + fr / 2), -0.25, sz * (sh / 2 + fr / 2)) @ face @ mc.S(0.42, 0.42, 0.3),
                          mc.COPPER, rough=0.25)
        shutter = 1.0 - open_k
        if shutter > 0.001:
            for i in range(12):
                a = i * math.pi / 6 + shutter * 0.9
                M = (mc.T(0, -0.6, 0) @ face @ mc.rot((0, 0, 1), a) @ mc.T(5.4 * (1 - shutter * 0.75), 0, 0)
                     @ mc.S(6.2, 2.8, 0.08))
                self.draw("box", M, mc.BRASS_DARK, rough=0.45)
        spin = t * 1.2 + seed
        backz = 3.2
        big = [(-sw / 2 - 3.2, backz, sh / 2 + 1.0, 4.6, 48, 1), (sw / 2 + 3.4, backz, -sh / 2 - 0.4, 4.2, 48, -1)]
        for x, y, z, rad, teeth, sgn in big:
            self.gear_at(teeth, (x, y, z), face, rad, 0.7, sgn * spin * 0.25)
        tr = [(sw / 2 + 2.6, 1.6, sh / 2 + 1.6, 1.0, 12), (sw / 2 + 3.6, 1.6, sh / 2 - 0.15, 1.33, 16),
              (sw / 2 + 2.0, 1.6, sh / 2 - 2.6, 2.0, 24)]
        omega = 1.6
        th_prev, n_prev = spin * omega, None
        for i, (x, y, z, rad, teeth) in enumerate(tr):
            if i == 0:
                th = spin * omega
            else:
                px, _, pz, _, _ = tr[i - 1]
                phi = math.atan2(z - pz, x - px)
                th = phi + math.pi - math.pi / teeth - (n_prev / teeth) * (th_prev - phi)
            self.gear_at(teeth, (x, y, z), face, rad, 0.5, th)
            th_prev, n_prev = th, teeth
        sun_c = (-sw / 2 - 2.4, 1.2, -sh / 2 - 1.2)
        ring_r = 2.6
        self.draw("ring", mc.T(*sun_c) @ face @ mc.S(ring_r, ring_r, 0.55), mc.BRASS_DARK, rough=0.38)
        ns, npl = 12, 24
        sun_r = ring_r * ns / (ns + 2 * npl) * 1.0
        pl_r = ring_r * npl / (ns + 2 * npl)
        carrier = spin * 0.5
        sun_th = carrier * (1 + 2 * npl / ns) * 1.0
        self.gear_at(12, sun_c, face, sun_r, 0.5, sun_th)
        for k in range(3):
            a = carrier + k * 2 * math.pi / 3
            c = (sun_c[0] + math.cos(a) * (sun_r + pl_r), sun_c[1], sun_c[2] + math.sin(a) * (sun_r + pl_r))
            self.gear_at(24, c, face, pl_r, 0.45, -carrier * (ns / npl) + a, base=mc.COPPER)
        for k in range(3):
            ext = 0.5 + 0.5 * math.sin(t * 2.0 + k * 2.1 + seed)
            base_pt = np.array([(-1 + k) * 6.0, 2.4, -sh / 2 - 3.6])
            tgt = np.array([(-1 + k) * 5.0, -0.6, -sh / 2 - 0.9])
            d = tgt - base_pt
            L = np.linalg.norm(d)
            fm = mc.frame_to(d)
            for off, seg, rr in mc.telescope(L, ext, 3):
                self.draw("cyl", mc.T(*base_pt) @ fm @ mc.T(0, 0, off) @ mc.S(0.36 * rr, 0.36 * rr, seg),
                          mc.BRASS if rr > 0.9 else mc.STEEL, rough=0.22)
        for i in range(10):
            x = -sw / 2 - 1 + i * (sw + 2) / 9
            self.draw("cyl", mc.T(x, 5.0, -14) @ mc.S(0.32, 0.32, 30), mc.IRON, metal=0.6, rough=0.6)
        self.draw("box", mc.T(0, 7.0, 0) @ mc.S(70, 1, 60), (0.09, 0.065, 0.045), metal=0.0, rough=0.9)
        self.draw("box", mc.T(0, 0, -sh / 2 - 6.5) @ mc.S(70, 40, 1), (0.16, 0.11, 0.07), metal=0.2, rough=0.7)
        return self.read(4)[..., :3]
