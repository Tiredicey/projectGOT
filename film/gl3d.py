import math

import moderngl
import numpy as np
import shapely
from shapely.geometry import Polygon

import geo
import landcover
import nature
from timeline import H, LAND, LEAD, W

FPX = 0.5 * W / math.tan(math.radians(40) / 2)
HAZE = (0.80, 0.71, 0.53)
VOID = (0.075, 0.058, 0.043)
SUN_DIR = np.array([-0.50, 0.62, 0.60])
SUN_DIR = SUN_DIR / np.linalg.norm(SUN_DIR)

SPRING_GLSL = """
float spring(float tau) {
    if (tau <= 0.0) return 0.0;
    float z = 0.55, w = 9.0;
    float wd = w * sqrt(1.0 - z * z);
    return 1.0 - exp(-z * w * tau) * (cos(wd * tau) + z * w / wd * sin(wd * tau));
}
"""

SHADOW_GLSL = """
uniform sampler2DShadow shadow_map;
uniform mat4 LVP;
uniform int use_shadow;
float shadow_at(vec3 w, vec3 n) {
    if (use_shadow == 0) return 1.0;
    vec4 l = LVP * vec4(w + n * 0.6, 1.0);
    vec3 p = l.xyz / l.w * 0.5 + 0.5;
    if (p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0) return 1.0;
    float s = 0.0;
    vec2 texel = 1.0 / vec2(textureSize(shadow_map, 0));
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++)
            s += texture(shadow_map, vec3(p.xy + vec2(i, j) * texel * 1.5, p.z - 0.0008));
    return s / 9.0;
}
"""

FOG_GLSL = """
uniform vec3 cam;
uniform float fog_dist;
uniform vec3 haze;
vec3 fog(vec3 c, vec3 w) {
    float d = length(w - cam) / fog_dist;
    float k = 1.0 - exp(-d * d);
    return mix(c, haze, k * 0.88);
}
"""

TERRAIN_VS = """
#version 330
uniform mat4 VP;
in vec3 in_pos;
in vec3 in_n;
out vec3 w;
out vec3 n;
void main() {
    w = in_pos;
    n = in_n;
    gl_Position = VP * vec4(in_pos, 1.0);
}
"""

TERRAIN_FS = """
#version 330
in vec3 w;
in vec3 n;
out vec4 frag;
uniform sampler2D t0, t1, t2, t3;
uniform vec4 r0, r1, r2, r3;
uniform float f0, f1, f2, f3;
uniform vec3 L;
""" + SHADOW_GLSL + FOG_GLSL + """
vec4 lay(sampler2D t, vec4 r, float f) {
    vec2 uv = vec2((w.x - r.x) / (r.y - r.x), (r.w - w.y) / (r.w - r.z));
    float ex = min(w.x - r.x, r.y - w.x);
    float ey = min(w.y - r.z, r.w - w.y);
    float a = smoothstep(0.0, f, min(ex, ey));
    return vec4(texture(t, uv).rgb, a);
}
void main() {
    vec3 c = haze;
    vec4 l = lay(t0, r0, f0); c = mix(c, l.rgb, l.a);
    l = lay(t1, r1, f1); c = mix(c, l.rgb, l.a);
    l = lay(t2, r2, f2); c = mix(c, l.rgb, l.a);
    l = lay(t3, r3, f3); c = mix(c, l.rgb, l.a);
    vec3 nn = normalize(n);
    float sh = shadow_at(w, nn);
    float diff = max(dot(nn, L), 0.0);
    float light = 0.50 + 0.62 * diff * sh;
    c *= light;
    frag = vec4(fog(c, w), 1.0);
}
"""

CITY_VS = """
#version 330
uniform mat4 VP;
uniform float t;
in vec3 in_pos;
in float in_hf;
in float in_h;
in vec3 in_n;
in vec3 in_col;
in float in_start;
in vec3 in_roof;
in vec2 in_uv;
out vec3 w;
out vec3 n;
out vec3 col;
out vec3 roof;
out float hf;
out vec2 uv;
out float hh;
""" + SPRING_GLSL + """
void main() {
    float k = spring(t - in_start);
    float z = in_pos.z - 3.0 + in_hf * (3.0 + in_h) * k;
    w = vec3(in_pos.xy, z);
    n = in_n;
    col = in_col;
    roof = in_roof;
    hf = in_hf;
    uv = vec2(in_uv.x, in_hf * in_h * k);
    hh = in_h * k;
    gl_Position = VP * vec4(w, 1.0);
}
"""

CITY_FS = """
#version 330
in vec3 w;
in vec3 n;
in vec3 col;
in vec3 roof;
in float hf;
in vec2 uv;
in float hh;
out vec4 frag;
uniform vec3 L;
""" + SHADOW_GLSL + FOG_GLSL + """
float hash2(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
void main() {
    vec3 nn = normalize(n);
    float sh = shadow_at(w, nn);
    float diff = max(dot(nn, L), 0.0);
    vec3 v = normalize(cam - w);
    vec3 h = normalize(v + L);
    vec3 alb;
    float spec_k = 0.25;
    float dist = length(cam - w);
    float detail = 1.0 - smoothstep(600.0, 2200.0, dist);
    if (nn.z > 0.5) {
        alb = roof;
        float corr = 0.5 + 0.5 * sin((w.x * 0.7 + w.y * 0.7) * 3.14159 * 2.4);
        alb *= mix(1.0, 0.86 + 0.14 * corr, detail);
        alb *= 0.92 + 0.16 * hash2(floor(w.xy * 0.25));
        spec_k = 0.55;
    } else {
        alb = col;
        float grime = smoothstep(0.0, 2.5, uv.y);
        alb *= mix(0.78, 1.0, grime);
        float fl = uv.y / 3.2;
        float bay = uv.x / 3.0;
        vec2 cell = vec2(fract(bay), fract(fl));
        float floor_id = floor(fl);
        float win = step(0.22, cell.x) * step(cell.x, 0.78) * step(0.32, cell.y) * step(cell.y, 0.80);
        win *= step(0.6, uv.y) * step(uv.y, hh - 0.6);
        float lit = hash2(vec2(floor(bay), floor_id));
        vec3 glass = mix(vec3(0.18, 0.24, 0.30), vec3(0.42, 0.56, 0.66), lit);
        vec3 frame = alb * 0.72;
        float ledge = smoothstep(0.03, 0.0, abs(cell.y - 0.04)) * step(0.6, uv.y);
        alb = mix(alb, mix(frame, glass, step(0.26, cell.x) * step(cell.x, 0.74) * step(0.36, cell.y) * step(cell.y, 0.76)), win * detail);
        alb = mix(alb, alb * 0.82, ledge * detail);
        if (win * detail > 0.5) spec_k = 0.9;
    }
    float sp = pow(max(dot(nn, h), 0.0), 40.0) * spec_k * sh;
    float ao = mix(0.62, 1.0, clamp(hf, 0.0, 1.0));
    vec3 c = alb * (0.40 + 0.80 * diff * sh) * ao + vec3(1.0, 0.88, 0.66) * sp;
    frag = vec4(fog(c, w), 1.0);
}
"""

ROUTE_VS = """
#version 330
uniform mat4 VP;
uniform vec3 cam;
uniform float width;
in vec3 in_pos;
in vec2 in_side;
in float in_s;
out float s;
out float across;
void main() {
    vec3 p = in_pos + vec3(in_side * width, 0.0);
    vec3 toc = normalize(cam - p);
    p += toc * width * 1.5;
    s = in_s;
    across = sign(in_side.x + in_side.y * 0.0001) * length(in_side);
    gl_Position = VP * vec4(p, 1.0);
}
"""

ROUTE_FS = """
#version 330
in float s;
in float across;
out vec4 frag;
uniform float progress;
uniform float layer;
uniform float t;
void main() {
    if (s > progress) discard;
    float a = abs(across);
    float behind = progress - s;
    float head = exp(-behind / 60.0);
    float shimmer = 0.85 + 0.15 * sin(s * 0.03 - t * 6.0);
    if (layer < 0.5) {
        if (a > 1.0) discard;
        frag = vec4(0.18, 0.11, 0.05, 1.0);
    } else {
        float core = smoothstep(0.55, 0.15, a);
        vec3 gold = vec3(1.0, 0.78, 0.36);
        vec3 c = gold * core * shimmer * 1.15 + vec3(1.0, 0.95, 0.8) * head * core * 1.6;
        frag = vec4(c, 1.0);
    }
}
"""

SLAB_VS = """
#version 330
uniform mat4 VP;
in vec3 in_pos;
in vec3 in_n;
out vec3 w;
out vec3 n;
void main() {
    w = in_pos;
    n = in_n;
    gl_Position = VP * vec4(in_pos, 1.0);
}
"""

SLAB_FS = """
#version 330
in vec3 w;
in vec3 n;
out vec4 frag;
uniform vec3 L;
uniform float top_z;
""" + FOG_GLSL + """
float hash(float x) { return fract(sin(x * 127.1) * 43758.5453); }
void main() {
    float along = w.x + w.y;
    float wav = sin(along * 0.0021) * 22.0 + sin(along * 0.0057 + 1.3) * 9.0;
    float d = (w.z + wav) / 38.0;
    float band = floor(d);
    float k = hash(band);
    vec3 a = vec3(0.47, 0.34, 0.20), b = vec3(0.70, 0.56, 0.36), c = vec3(0.33, 0.23, 0.14);
    vec3 col = k < 0.5 ? mix(a, b, k * 2.0) : mix(b, c, (k - 0.5) * 2.0);
    col *= 0.9 + 0.1 * smoothstep(0.0, 0.08, fract(d));
    if (w.z < top_z - 1150.0) col = vec3(0.16, 0.12, 0.09);
    float diff = max(dot(normalize(n), L), 0.0);
    col *= 0.45 + 0.65 * diff;
    frag = vec4(fog(col, w), 1.0);
}
"""

OBJ_VS = """
#version 330
uniform mat4 VP;
uniform mat4 M;
in vec3 in_pos;
in vec3 in_n;
in vec2 in_uv;
out vec3 w;
out vec3 n;
out vec2 uv;
void main() {
    vec4 p = M * vec4(in_pos, 1.0);
    w = p.xyz;
    n = mat3(M) * in_n;
    uv = in_uv;
    gl_Position = VP * p;
}
"""

OBJ_FS = """
#version 330
in vec3 w;
in vec3 n;
in vec2 uv;
out vec4 frag;
uniform vec3 L;
uniform vec3 base;
uniform int use_tex;
uniform sampler2D tex;
uniform float tex_dim;
uniform float spec;
uniform int use_fog;
""" + SHADOW_GLSL + FOG_GLSL + """
void main() {
    vec3 nn = normalize(n);
    vec3 v = normalize(cam - w);
    if (dot(nn, v) < 0.0) nn = -nn;
    vec3 alb = use_tex == 1 ? texture(tex, uv).rgb * tex_dim : base;
    float sh = shadow_at(w, nn);
    float diff = max(dot(nn, L), 0.0);
    vec3 h = normalize(v + L);
    float sp = pow(max(dot(nn, h), 0.0), 56.0) * spec * sh;
    float rim = pow(1.0 - max(dot(nn, v), 0.0), 3.0);
    float hemi = 0.5 + 0.5 * nn.y;
    vec3 c = alb * (0.22 + 0.18 * hemi + 0.85 * diff * sh) + vec3(1.0, 0.85, 0.55) * sp + alb * rim * 0.35;
    if (use_fog == 1) c = fog(c, w);
    frag = vec4(c, 1.0);
}
"""


def perspective(near, far, cx=W / 2, cy=H / 2):
    p = np.zeros((4, 4))
    p[0, 0] = 2 * FPX / W
    p[1, 1] = 2 * FPX / H
    p[0, 2] = -(cx - W / 2) / (W / 2)
    p[1, 2] = (cy - H / 2) / (H / 2)
    p[2, 2] = (far + near) / (near - far)
    p[2, 3] = 2 * far * near / (near - far)
    p[3, 2] = -1
    return p


def look(c, r, u, f):
    v = np.eye(4)
    v[0, :3], v[1, :3], v[2, :3] = r, u, -f
    v[:3, 3] = -v[:3, :3] @ c
    return v


def ortho(half, near, far):
    o = np.eye(4)
    o[0, 0] = o[1, 1] = 1 / half
    o[2, 2] = -2 / (far - near)
    o[2, 3] = -(far + near) / (far - near)
    return o


def m4(a):
    return np.ascontiguousarray(a.T, dtype="f4").tobytes()


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4)
    m[:2, :2] = [[c, -s], [s, c]]
    return m


def translate(x, y, z):
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def scale(x, y, z):
    return np.diag([x, y, z, 1.0])


def gear_mesh(teeth=24, inner=0.86, h=1.0):
    n = teeth * 8
    th = np.linspace(0, 2 * np.pi, n + 1)
    prof = 1.0 + 0.08 * ((np.floor(np.arange(n + 1) / 4) % 2) == 0)
    verts = []

    def quad(a, b, c, d, nrm):
        for p in (a, b, c, a, c, d):
            verts.append((*p, *nrm, 0.0, 0.0))

    for i in range(n):
        t0, t1 = th[i], th[i + 1]
        r0, r1 = prof[i], prof[i + 1]
        o0 = (r0 * math.cos(t0), r0 * math.sin(t0))
        o1 = (r1 * math.cos(t1), r1 * math.sin(t1))
        i0 = (inner * math.cos(t0), inner * math.sin(t0))
        i1 = (inner * math.cos(t1), inner * math.sin(t1))
        nm = (math.cos((t0 + t1) / 2), math.sin((t0 + t1) / 2), 0.0)
        quad((*o0, 0), (*o1, 0), (*o1, h), (*o0, h), nm)
        quad((*i1, 0), (*i0, 0), (*i0, h), (*i1, h), (-nm[0], -nm[1], 0.0))
        quad((*i0, h), (*o0, h), (*o1, h), (*i1, h), (0.0, 0.0, 1.0))
        if r0 != r1:
            sx = math.cos(t1) * (r1 - r0)
            quad((r0 * math.cos(t1), r0 * math.sin(t1), 0), (*o1, 0), (*o1, h), (r0 * math.cos(t1), r0 * math.sin(t1), h),
                 (-math.sin(t1) * np.sign(r1 - r0), math.cos(t1) * np.sign(r1 - r0), 0.0))
    return np.array(verts, "f4")


def band_mesh(r, hw, thick, nseg=240):
    verts = []
    th = np.linspace(0, 2 * np.pi, nseg + 1)
    ri = r - thick
    for i in range(nseg):
        t0, t1 = th[i], th[i + 1]
        u0, u1 = i / nseg, (i + 1) / nseg
        c0, s0, c1, s1 = math.cos(t0), math.sin(t0), math.cos(t1), math.sin(t1)
        for rad, sgn, flip in ((r, 1.0, False), (ri, -1.0, True)):
            a = (rad * c0, rad * s0, -hw, sgn * c0, sgn * s0, 0, u0, 1.0)
            b = (rad * c1, rad * s1, -hw, sgn * c1, sgn * s1, 0, u1, 1.0)
            c = (rad * c1, rad * s1, hw, sgn * c1, sgn * s1, 0, u1, 0.0)
            d = (rad * c0, rad * s0, hw, sgn * c0, sgn * s0, 0, u0, 0.0)
            verts += [a, b, c, a, c, d]
        for z, nz in ((hw, 1.0), (-hw, -1.0)):
            a = (r * c0, r * s0, z, 0, 0, nz, 0.5, 0.02)
            b = (r * c1, r * s1, z, 0, 0, nz, 0.5, 0.02)
            c = (ri * c1, ri * s1, z, 0, 0, nz, 0.5, 0.02)
            d = (ri * c0, ri * s0, z, 0, 0, nz, 0.5, 0.02)
            verts += [a, b, c, a, c, d]
    return np.array(verts, "f4")


def sun_mesh():
    verts = []

    def tri(a, b, c):
        nrm = np.cross(np.subtract(b, a), np.subtract(c, a))
        nrm = nrm / (np.linalg.norm(nrm) + 1e-9)
        for p in (a, b, c):
            verts.append((*p, *nrm, 0.0, 0.0))

    for i in range(8):
        a = i * math.pi / 4
        ca, sa = math.cos(a), math.sin(a)
        px, py = -sa, ca
        tip = (0.47 * ca, 0.47 * sa, 0.0)
        base_c = (0.19 * ca, 0.19 * sa)
        l = (base_c[0] + px * 0.075, base_c[1] + py * 0.075, 0.0)
        rr = (base_c[0] - px * 0.075, base_c[1] - py * 0.075, 0.0)
        top = (base_c[0], base_c[1], 0.06)
        bot = (base_c[0], base_c[1], -0.06)
        tri(tip, top, l)
        tri(tip, rr, top)
        tri(tip, l, bot)
        tri(tip, bot, rr)
    n = 72
    for i in range(n):
        t0, t1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
        for (ra, za), (rb, zb) in (((0.215, 0.05), (0.215, -0.05)), ((0.215, 0.05), (0.15, 0.085)),
                                    ((0.15, 0.085), (0.0, 0.11))):
            pa0 = (ra * math.cos(t0), ra * math.sin(t0), za)
            pa1 = (ra * math.cos(t1), ra * math.sin(t1), za)
            pb0 = (rb * math.cos(t0), rb * math.sin(t0), zb)
            pb1 = (rb * math.cos(t1), rb * math.sin(t1), zb)
            tri(pa0, pa1, pb1)
            tri(pa0, pb1, pb0)
    return np.array(verts, "f4")


class Scene:
    def __init__(self, meta, city_items, centers):
        self.ctx = moderngl.create_standalone_context(backend="egl")
        ctx = self.ctx
        self.fbo_ms = ctx.framebuffer(color_attachments=[ctx.renderbuffer((W, H), 4, samples=4)],
                                      depth_attachment=ctx.depth_renderbuffer((W, H), samples=4))
        self.fbo = ctx.framebuffer(color_attachments=[ctx.renderbuffer((W, H), 4)])
        self.shadow_tex = ctx.depth_texture((1536, 1536))
        self.shadow_tex.compare_func = "<="
        self.shadow_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.shadow_fbo = ctx.framebuffer(depth_attachment=self.shadow_tex)
        self.dummy_shadow = ctx.depth_texture((1, 1), data=np.ones(1, "f4").tobytes())
        self.dummy_shadow.compare_func = "<="

        self.terrain = geo.Terrain()
        self.lake_z = landcover.lake_level_z(self.terrain)
        self.carve_lake()
        tv, ti = self.terrain.mesh()
        self.p_terrain = ctx.program(vertex_shader=TERRAIN_VS, fragment_shader=TERRAIN_FS)
        self.vao_terrain = ctx.vertex_array(self.p_terrain, [(ctx.buffer(tv), "3f 3f", "in_pos", "in_n")],
                                            ctx.buffer(ti))
        self.layers = []
        for i, (name, feather) in enumerate((("region", 400.0), ("corridor", 700.0), ("city", 300.0), ("home", 200.0))):
            layer, img = geo.load_layer(meta, name)
            tex = ctx.texture((layer.w, layer.h), 3, np.ascontiguousarray(img).tobytes(), alignment=1)
            del img
            tex.build_mipmaps()
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            tex.anisotropy = 8.0
            tex.repeat_x = tex.repeat_y = False
            self.layers.append((tex, (layer.x0, layer.x1, layer.y0, layer.y1), feather))
            self.p_terrain[f"t{i}"].value = i
            self.p_terrain[f"r{i}"].value = (layer.x0, layer.x1, layer.y0, layer.y1)
            self.p_terrain[f"f{i}"].value = feather

        self.p_city = ctx.program(vertex_shader=CITY_VS, fragment_shader=CITY_FS)
        cv = self.city_mesh(city_items)
        self.vao_city = ctx.vertex_array(self.p_city, [(ctx.buffer(cv), "3f 1f 1f 3f 3f 1f 3f 2f", "in_pos", "in_hf", "in_h",
                                                        "in_n", "in_col", "in_start", "in_roof", "in_uv")])
        self.p_obj = ctx.program(vertex_shader=OBJ_VS, fragment_shader=OBJ_FS)
        self.p_route = ctx.program(vertex_shader=ROUTE_VS, fragment_shader=ROUTE_FS)
        self.p_slab = ctx.program(vertex_shader=SLAB_VS, fragment_shader=SLAB_FS)
        self.vao_slab = ctx.vertex_array(self.p_slab, [(ctx.buffer(self.slab_mesh()), "3f 3f", "in_pos", "in_n")])
        self.vao_route = None
        self.route_progress = 0.0
        self.vao_gear = self.obj_vao(gear_mesh())
        self.vao_sun = self.obj_vao(sun_mesh())
        self.heroes = []
        for it in city_items:
            if it["hero"] and it["hero"] != "terminal":
                c = it["c"]
                r0 = min(float(np.max(np.linalg.norm(it["fp"] - c, axis=1))) * 1.35 + 6, 140.0)
                ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
                z = float(np.max(self.terrain.sample(c[0] + r0 * np.cos(ang), c[1] + r0 * np.sin(ang))))
                self.heroes.append(dict(c=c, r=r0, z=z, start=it["start"], hero=it["hero"]))
        for p in (self.p_terrain, self.p_city, self.p_obj):
            if "shadow_map" in p:
                p["shadow_map"].value = 7
        self.rings = []
        self.nature = nature.Nature(ctx, self.terrain, self.lake_z)
        import gc
        landcover.load.cache_clear()
        gc.collect()

    def carve_lake(self):
        import cv2
        T = self.terrain
        polys = landcover.water_polygons()
        if not polys:
            return
        inside = np.zeros(T.z.shape, bool)
        U = shapely.union_all(polys)
        inside = shapely.contains_xy(U, T.gx.ravel(), T.gy.ravel()).reshape(T.z.shape)
        dist = cv2.distanceTransform(inside.astype(np.uint8), cv2.DIST_L2, 5) * T.step
        bed = self.lake_z - 4.0 - np.clip(dist / 1500.0, 0, 1) * 60.0
        T.z = np.where(inside, np.minimum(T.z, bed), T.z).astype(np.float32)
        T.zs = cv2.GaussianBlur(T.z, (0, 0), 5)

    def slab_mesh(self, base=-1300.0, plinth_out=260.0, plinth_h=160.0):
        T = self.terrain
        x0, y0 = T.x0, T.y0
        x1, y1 = x0 + (T.nx - 1) * T.step, y0 + (T.ny - 1) * T.step
        out = []

        def wall(xs, ys, zs, nrm):
            for i in range(len(xs) - 1):
                a = (xs[i], ys[i], zs[i]); b = (xs[i + 1], ys[i + 1], zs[i + 1])
                ab = (xs[i], ys[i], base); bb = (xs[i + 1], ys[i + 1], base)
                for p in (ab, bb, b, ab, b, a):
                    out.append((*p, *nrm))

        gx = x0 + np.arange(T.nx) * T.step
        gy = y0 + np.arange(T.ny) * T.step
        wall(gx, np.full(T.nx, y0), T.z[0, :], (0, -1, 0))
        wall(gx[::-1], np.full(T.nx, y1), T.z[-1, ::-1], (0, 1, 0))
        wall(np.full(T.ny, x0), gy[::-1], T.z[::-1, 0], (-1, 0, 0))
        wall(np.full(T.ny, x1), gy, T.z[:, -1], (1, 0, 0))
        px0, px1, py0, py1 = x0 - plinth_out, x1 + plinth_out, y0 - plinth_out, y1 + plinth_out
        zt, zb = base, base - plinth_h
        for (ax, ay), (bx, by), nrm in (((px0, py0), (px1, py0), (0, -1, 0)), ((px1, py0), (px1, py1), (1, 0, 0)),
                                        ((px1, py1), (px0, py1), (0, 1, 0)), ((px0, py1), (px0, py0), (-1, 0, 0))):
            for p in ((ax, ay, zb), (bx, by, zb), (bx, by, zt), (ax, ay, zb), (bx, by, zt), (ax, ay, zt)):
                out.append((*p, *nrm))
        for p in ((px0, py0, zt), (px1, py0, zt), (px1, py1, zt), (px0, py0, zt), (px1, py1, zt), (px0, py1, zt)):
            out.append((*p, 0, 0, 1))
        self.slab_top = float(np.median(T.z))
        return np.array(out, "f4")

    def set_route(self, rt):
        pts = rt.pts
        z = self.terrain.sample(pts[:, 0], pts[:, 1]) + 2.0
        tan = np.gradient(pts, axis=0)
        tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-9
        nrm = np.column_stack([-tan[:, 1], tan[:, 0]])
        v = []
        for i in range(len(pts) - 1):
            a = (pts[i, 0], pts[i, 1], z[i]); b = (pts[i + 1, 0], pts[i + 1, 1], z[i + 1])
            na, nb = nrm[i], nrm[i + 1]
            sa, sb = rt.s[i], rt.s[i + 1]
            quad = [(a, na, sa), (b, nb, sb), (b, -nb, sb), (a, na, sa), (b, -nb, sb), (a, -na, sa)]
            for p, side, sv in quad:
                v.append((*p, side[0], side[1], sv))
        self.vao_route = self.ctx.vertex_array(self.p_route, [(self.ctx.buffer(np.array(v, "f4")), "3f 2f 1f",
                                                                "in_pos", "in_side", "in_s")])

    def draw_route(self, VP, c, cam_dist, progress, t):
        if self.vao_route is None or progress <= 0:
            return
        ctx = self.ctx
        p = self.p_route
        p["VP"].write(m4(VP))
        p["cam"].value = tuple(c)
        p["progress"].value = progress
        p["t"].value = t
        w_core = float(np.clip(cam_dist / FPX * 2.4, 0.9, 60.0))
        p["width"].value = w_core * 1.9
        p["layer"].value = 0.0
        self.vao_route.render()
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        ctx.depth_mask = False
        p["width"].value = w_core * 2.6
        p["layer"].value = 1.0
        self.vao_route.render()
        ctx.depth_mask = True
        ctx.disable(moderngl.BLEND)

    def obj_vao(self, data):
        return self.ctx.vertex_array(self.p_obj, [(self.ctx.buffer(data), "3f 3f 2f", "in_pos", "in_n", "in_uv")])

    def city_mesh(self, items):
        out = []
        for it in items:
            fp = it["fp"]
            zb = float(np.min(self.terrain.sample(fp[:, 0], fp[:, 1])))
            col = np.asarray(it["col"], float) / 255.0
            roofc = np.asarray(it.get("roof", it["col"]), float) / 255.0
            h, st = it["h"], it["start"]
            n = len(fp)
            u = 0.0
            for i in range(n):
                a, b = fp[i], fp[(i + 1) % n]
                nrm = np.array([b[1] - a[1], -(b[0] - a[0]), 0.0])
                ln = np.linalg.norm(nrm)
                if ln < 1e-6:
                    continue
                nrm /= ln
                ua, ub = u, u + ln
                u = ub
                quad = [(a, 0, ua), (b, 0, ub), (b, 1, ub), (a, 0, ua), (b, 1, ub), (a, 1, ua)]
                for p, hf, uu in quad:
                    out.append((p[0], p[1], zb, hf, h, *nrm, *col, st, *roofc, uu, 0.0))
            try:
                tris = shapely.constrained_delaunay_triangles(Polygon(fp))
                for tr in shapely.get_parts(tris):
                    xy = np.array(tr.exterior.coords)[:3]
                    if np.cross(xy[1] - xy[0], xy[2] - xy[0]) < 0:
                        xy = xy[::-1]
                    for p in xy:
                        out.append((p[0], p[1], zb, 1.0, h, 0.0, 0.0, 1.0, *col, st, *roofc, 0.0, 0.0))
            except Exception:
                continue
        return np.array(out, "f4")

    def bind_textures(self, shadow):
        for i, (tex, _, _) in enumerate(self.layers):
            tex.use(i)
        (self.shadow_tex if shadow else self.dummy_shadow).use(7)

    def set_common(self, prog, VP, LVP, cam, fog_dist, use_shadow, L):
        for k, v in (("VP", m4(VP)), ("LVP", m4(LVP))):
            if k in prog:
                prog[k].write(v)
        for k, v in (("cam", tuple(cam)), ("fog_dist", fog_dist), ("haze", HAZE), ("use_shadow", use_shadow),
                     ("L", tuple(L))):
            if k in prog:
                prog[k].value = v

    def map_matrices(self, cam):
        target_z = float(self.terrain.sample(cam.t[0], cam.t[1], smooth=True))
        c = cam.c + np.array([0, 0, target_z])
        ground = float(self.terrain.sample(c[0], c[1], smooth=True))
        if c[2] < ground + 40:
            c[2] = ground + 40
        V = look(c, cam.R[0], cam.R[1], cam.R[2])
        near = max(2.0, cam.dist * 0.02)
        P = perspective(near, cam.dist * 12 + 90000)
        jx, jy = getattr(self, "jitter", (0.0, 0.0))
        P[0, 2] += jx
        P[1, 2] += jy
        half = float(np.clip(cam.dist * 1.6, 350, 14000))
        tgt = cam.t + np.array([0, 0, target_z])
        lpos = tgt + SUN_DIR * 6000
        f = -SUN_DIR
        r = np.cross(f, [0, 0, 1.0])
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        LVP = ortho(half, 10, 14000) @ look(lpos, r, u, f)
        return P @ V, LVP, c

    def draw_gears(self, t, k_fn):
        for hr in self.heroes:
            k = k_fn(t - hr["start"])
            if k <= 0.01:
                continue
            M = translate(hr["c"][0], hr["c"][1], hr["z"] - 0.5) @ rot_z(0.35 * (t - hr["start"])) @ scale(hr["r"], hr["r"], 2.6 * min(k, 1.3))
            self.p_obj["M"].write(m4(M))
            self.p_obj["base"].value = (0.52, 0.36, 0.15)
            self.p_obj["use_tex"].value = 0
            self.p_obj["spec"].value = 0.9
            self.p_obj["use_fog"].value = 1
            self.vao_gear.render()

    def render_map(self, cam, t, k_fn):
        ctx = self.ctx
        VP, LVP, c = self.map_matrices(cam)
        fog_dist = cam.dist * 3.4 + 2500
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE)
        self.shadow_fbo.use()
        self.shadow_fbo.clear(depth=1.0)
        ctx.polygon_offset = (2.0, 4.0)
        self.bind_textures(False)
        for prog in (self.p_terrain, self.p_city, self.p_obj):
            self.set_common(prog, LVP, LVP, c, fog_dist, 0, SUN_DIR)
        self.p_city["t"].value = t
        self.vao_city.render()
        self.draw_gears(t, k_fn)
        self.nature.draw_trees(LVP, LVP, c, cam.t, cam.dist, fog_dist, 0, SUN_DIR, HAZE, t, shadow_pass=True)
        ctx.polygon_offset = (0.0, 0.0)
        self.fbo_ms.use()
        void = 1.0 - min(1.0, max(0.0, (cam.dist - 9000.0) / 12000.0))
        bg = tuple(HAZE[i] * void + VOID[i] * (1 - void) for i in range(3))
        self.fbo_ms.clear(*bg, 1.0, depth=1.0)
        self.bind_textures(True)
        for prog in (self.p_terrain, self.p_city, self.p_obj):
            self.set_common(prog, VP, LVP, c, fog_dist, 1, SUN_DIR)
        self.vao_terrain.render()
        self.set_common(self.p_slab, VP, LVP, c, fog_dist, 0, SUN_DIR)
        self.p_slab["top_z"].value = self.slab_top
        self.vao_slab.render()
        self.p_city["t"].value = t
        self.vao_city.render()
        self.draw_gears(t, k_fn)
        self.nature.draw_trees(VP, LVP, c, cam.t, cam.dist, fog_dist, 1, SUN_DIR, HAZE, t)
        self.nature.draw_water(VP, LVP, c, fog_dist, 1, SUN_DIR, HAZE, t)
        self.draw_route(VP, c, cam.dist, self.route_progress, t)
        self.VP = VP
        return self.read(4)[..., :3]

    def project(self, pts):
        p = np.column_stack([pts, np.ones(len(pts))]) @ self.VP.T
        w = p[:, 3:4]
        ndc = p[:, :3] / np.where(w > 0, w, np.nan)
        return np.column_stack([(ndc[:, 0] + 1) * W / 2, (1 - ndc[:, 1]) * H / 2])

    def ground_point(self, x, y):
        return np.array([[x, y, float(self.terrain.sample(x, y))]])

    def add_ring_texture(self, tex_rgb):
        tex = self.ctx.texture((tex_rgb.shape[1], tex_rgb.shape[0]), 3, np.ascontiguousarray(tex_rgb).tobytes(),
                               alignment=1)
        tex.build_mipmaps()
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        tex.anisotropy = 8.0
        return tex

    def render_sphere(self, rings, sun_rot, scale_px, center):
        ctx = self.ctx
        dist = FPX / (420.0 * scale_px)
        c = np.array([0, 0, dist])
        V = look(c, np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, -1.0]))
        P = perspective(max(0.02, dist * 0.05), dist + 10, center[0], center[1])
        VP = P @ V
        L = np.array([-0.45, 0.55, 0.70])
        L /= np.linalg.norm(L)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE)
        self.fbo_ms.use()
        self.fbo_ms.clear(0, 0, 0, 0, depth=1.0)
        self.dummy_shadow.use(7)
        self.set_common(self.p_obj, VP, np.eye(4), c, 1e6, 0, L)
        self.p_obj["use_fog"].value = 0
        self.p_obj["tex"].value = 0
        for vao, tex, M in rings:
            tex.use(0)
            self.p_obj["M"].write(m4(M))
            self.p_obj["use_tex"].value = 1
            self.p_obj["tex_dim"].value = 1.0
            self.p_obj["spec"].value = 0.8
            vao.render()
        self.p_obj["M"].write(m4(sun_rot))
        self.p_obj["use_tex"].value = 0
        self.p_obj["base"].value = (0.86, 0.66, 0.30)
        self.p_obj["spec"].value = 1.4
        self.vao_sun.render()
        return self.read(4)

    def read(self, comps):
        self.ctx.copy_framebuffer(self.fbo, self.fbo_ms)
        raw = self.fbo.read(components=comps, alignment=1)
        return np.frombuffer(raw, np.uint8).reshape(H, W, comps)[::-1]
