import math

import moderngl
import numpy as np
import shapely
from shapely.geometry import Polygon

import geo
import landcover
from glsl import BEND, SUNLIGHT, STATE

TILE = 600.0

TREE_VS = """
#version 330
uniform mat4 VP;
uniform float t;
uniform float wind;
in vec3 in_pos;
in vec3 in_n;
in float in_k;
in vec4 i_pos;
in vec2 i_var;
out vec3 w;
out vec3 n;
out float k;
out float var;
""" + BEND + """
void main() {
    float h = i_pos.z;
    float a = i_pos.w;
    float ca = cos(a), sa = sin(a);
    float s = h * (0.85 + 0.3 * i_var.y);
    vec3 p = in_pos * vec3(s, s, h);
    p.xy = vec2(ca * p.x - sa * p.y, sa * p.x + ca * p.y);
    float sway = in_pos.z * in_pos.z * wind * (sin(t * 1.7 + i_pos.x * 0.03 + i_pos.y * 0.021) * 0.6
               + sin(t * 3.1 + i_pos.y * 0.05) * 0.25);
    p.x += sway * h * 0.04;
    p.y += sway * h * 0.025;
    vec3 wu = vec3(i_pos.xy, i_var.x) + p;
    vec3 nn = in_n;
    nn.xy = vec2(ca * nn.x - sa * nn.y, sa * nn.x + ca * nn.y);
    n = bendn(wu, nn);
    w = bendp(wu);
    k = in_k;
    var = i_var.y;
    gl_Position = VP * vec4(w, 1.0);
}
"""

TREE_FS = """
#version 330
in vec3 w;
in vec3 n;
in float k;
in float var;
out vec4 frag;
uniform vec3 Lsun;
""" + SUNLIGHT + """
uniform vec3 leaf_a;
uniform vec3 leaf_b;
uniform vec3 trunk;
uniform sampler2DShadow shadow_map;
uniform mat4 LVP;
uniform int use_shadow;
uniform vec3 cam;
uniform float fog_dist;
uniform vec3 haze;
float shadow_at(vec3 p, vec3 nn) {
    if (use_shadow == 0) return 1.0;
    vec4 l = LVP * vec4(p + nn * 0.6, 1.0);
    vec3 q = l.xyz / l.w * 0.5 + 0.5;
    if (q.x < 0.0 || q.x > 1.0 || q.y < 0.0 || q.y > 1.0) return 1.0;
    float s = 0.0;
    vec2 texel = 1.0 / vec2(textureSize(shadow_map, 0));
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++)
            s += texture(shadow_map, vec3(q.xy + vec2(i, j) * texel * 1.5, q.z - 0.0012));
    return s / 9.0;
}
float hash3(vec3 p) { return fract(sin(dot(p, vec3(12.9898, 78.233, 37.719))) * 43758.5453); }
void main() {
    vec3 L = light_dir(w, Lsun);
    vec3 nn = normalize(n);
    float sh = shadow_at(w, nn);
    float diff = max(dot(nn, L), 0.0);
    float wrap = max(dot(nn, L) * 0.5 + 0.5, 0.0);
    vec3 alb;
    if (k < 0.5) {
        alb = trunk;
    } else {
        float speck = hash3(floor(w * 1.3));
        alb = mix(leaf_a, leaf_b, clamp(var * 0.8 + speck * 0.35, 0.0, 1.0));
        alb *= 0.85 + 0.3 * smoothstep(0.0, 1.0, nn.z * 0.5 + 0.5);
    }
    vec3 v = normalize(cam - w);
    float rim = pow(1.0 - max(dot(nn, v), 0.0), 2.5) * 0.25;
    vec3 c = alb * (0.30 + 0.55 * wrap * sh + 0.35 * diff * sh) + alb * rim * vec3(1.0, 0.95, 0.7);
    float d = length(w - cam) / fog_dist;
    c = mix(c, haze, (1.0 - exp(-d * d)) * 0.88);
    frag = vec4(c, 1.0);
}
"""

WATER_VS = """
#version 330
uniform mat4 VP;
uniform float t;
in vec2 in_xy;
in float in_z;
in float in_shore;
out vec3 w;
out float shore;
out vec3 upv;
""" + BEND + """
void main() {
    float z = in_z + 0.35 * sin(in_xy.x * 0.011 + t * 1.3) + 0.25 * sin(in_xy.y * 0.017 - t * 1.1);
    vec3 wu = vec3(in_xy, z);
    upv = bendn(wu, vec3(0.0, 0.0, 1.0));
    w = bendp(wu);
    shore = in_shore;
    gl_Position = VP * vec4(w, 1.0);
}
"""

WATER_FS = """
#version 330
in vec3 w;
in float shore;
in vec3 upv;
out vec4 frag;
uniform float t;
uniform vec4 bend;
uniform vec3 Lsun;
""" + SUNLIGHT + """
uniform vec3 cam;
uniform float fog_dist;
uniform vec3 haze;
uniform sampler2DShadow shadow_map;
uniform mat4 LVP;
uniform int use_shadow;
float shadow_at(vec3 p) {
    if (use_shadow == 0) return 1.0;
    vec4 l = LVP * vec4(p, 1.0);
    vec3 q = l.xyz / l.w * 0.5 + 0.5;
    if (q.x < 0.0 || q.x > 1.0 || q.y < 0.0 || q.y > 1.0) return 1.0;
    return texture(shadow_map, vec3(q.xy, q.z - 0.0015));
}
vec2 wave_grad(vec2 p, float fp) {
    vec2 g = vec2(0.0);
    vec2 dirs[5] = vec2[5](vec2(1.0, 0.2), vec2(-0.3, 1.0), vec2(0.7, -0.7), vec2(-0.9, -0.4), vec2(0.2, 0.95));
    float freqs[5] = float[5](0.045, 0.071, 0.113, 0.19, 0.31);
    float amps[5] = float[5](0.55, 0.38, 0.24, 0.14, 0.08);
    float speeds[5] = float[5](1.1, 1.5, 2.0, 2.7, 3.4);
    for (int i = 0; i < 5; i++) {
        vec2 d = normalize(dirs[i]);
        float ph = dot(d, p) * freqs[i] + t * speeds[i];
        float aa = 1.0 - smoothstep(0.08, 0.30, freqs[i] * fp);
        g += d * cos(ph) * freqs[i] * amps[i] * 6.0 * aa;
    }
    return g;
}
void main() {
    vec3 L = light_dir(w, Lsun);
    vec3 dx = dFdx(w), dy = dFdy(w);
    vec3 v0 = normalize(cam - w);
    float graze = 1.0 - clamp(v0.z, 0.0, 1.0);
    float fp = max(length(dx.xy), length(dy.xy)) * (1.0 + 6.0 * graze * graze);
    float dist = length(w - cam);
    fp = max(fp, smoothstep(2500.0, 6000.0, dist) * 40.0);
    vec2 g = wave_grad(w.xy, fp);
    vec3 nn = normalize(vec3(-g, 1.0));
    if (bend.w > 0.0) nn = normalize(upv + vec3(-g, 0.0) * 0.6);
    vec3 v = normalize(cam - w);
    float fres = 0.04 + 0.96 * pow(1.0 - max(dot(nn, v), 0.0), 5.0);
    vec3 deep = vec3(0.07, 0.30, 0.42);
    vec3 shallow = vec3(0.24, 0.58, 0.60);
    vec3 body = mix(shallow, deep, smoothstep(0.0, 1.0, shore));
    vec3 sky = mix(vec3(0.86, 0.80, 0.66), vec3(0.55, 0.70, 0.86), clamp(reflect(-v, nn).z, 0.0, 1.0));
    float sh = shadow_at(w);
    vec3 h = normalize(v + L);
    float rough = clamp(fp * 0.12, 0.0, 1.0);
    float expo = mix(220.0, 18.0, rough);
    float glint = pow(max(dot(nn, h), 0.0), expo) * mix(4.0, 0.0, smoothstep(0.3, 1.0, rough)) * sh;
    vec3 c = mix(body * (0.55 + 0.45 * sh), sky, fres * 0.85) + vec3(1.0, 0.92, 0.72) * glint;
    float foam = smoothstep(0.08, 0.0, shore) * (0.55 + 0.45 * sin(w.x * 0.08 + w.y * 0.06 + t * 2.0) * (1.0 - smoothstep(4.0, 12.0, fp)));
    c = mix(c, vec3(0.93, 0.92, 0.84), foam * 0.75);
    float d = length(w - cam) / fog_dist;
    c = mix(c, haze, (1.0 - exp(-d * d)) * 0.88);
    frag = vec4(c, 1.0);
}
"""


def _cone(n_seg, r0, r1, z0, z1, k):
    out = []
    for i in range(n_seg):
        a0, a1 = 2 * math.pi * i / n_seg, 2 * math.pi * (i + 1) / n_seg
        p = [(r0 * math.cos(a0), r0 * math.sin(a0), z0), (r0 * math.cos(a1), r0 * math.sin(a1), z0),
             (r1 * math.cos(a1), r1 * math.sin(a1), z1), (r1 * math.cos(a0), r1 * math.sin(a0), z1)]
        slope = (r0 - r1) / max(1e-6, z1 - z0)
        for (x, y, z) in (p[0], p[1], p[2], p[0], p[2], p[3]):
            ln = math.hypot(x, y) or 1.0
            nrm = np.array([x / ln, y / ln, slope])
            nrm /= np.linalg.norm(nrm)
            out.append((x, y, z, *nrm, k))
    return out


def _blob(cx, cy, cz, rx, ry, rz, k, rings=6, seg=10, seed=0):
    rng = np.random.default_rng(seed)
    bump = rng.uniform(0.82, 1.12, (rings + 1, seg + 1))
    bump[:, -1] = bump[:, 0]
    out = []
    for i in range(rings):
        for j in range(seg):
            quad = []
            for di, dj in ((0, 0), (0, 1), (1, 1), (1, 0)):
                th = math.pi * (i + di) / rings
                ph = 2 * math.pi * (j + dj) / seg
                b = bump[i + di, j + dj]
                nx, ny, nz = math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)
                quad.append((cx + rx * nx * b, cy + ry * ny * b, cz + rz * nz * b, nx, ny, nz))
            for q in (quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]):
                out.append((*q, k))
    return out


def broadleaf_mesh():
    v = _cone(6, 0.045, 0.03, 0.0, 0.45, 0.0)
    v += _blob(0.0, 0.0, 0.66, 0.36, 0.36, 0.30, 1.0, seed=1)
    v += _blob(0.16, 0.08, 0.56, 0.24, 0.24, 0.2, 1.0, rings=5, seg=8, seed=2)
    v += _blob(-0.14, -0.1, 0.58, 0.22, 0.22, 0.2, 1.0, rings=5, seg=8, seed=3)
    v += _blob(-0.02, 0.14, 0.82, 0.2, 0.2, 0.17, 1.0, rings=5, seg=8, seed=4)
    return np.array(v, "f4")


def palm_mesh():
    v = []
    segs = 7
    for s in range(segs):
        z0, z1 = 0.82 * s / segs, 0.82 * (s + 1) / segs
        bend0, bend1 = 0.12 * (s / segs) ** 2, 0.12 * ((s + 1) / segs) ** 2
        for x, y, z, nx, ny, nz, k in _cone(6, 0.028 - 0.008 * s / segs, 0.028 - 0.008 * (s + 1) / segs, z0, z1, 0.0):
            v.append((x + (bend0 if z <= z0 + 1e-6 else bend1), y, z, nx, ny, nz, k))
    top = (0.12, 0.0, 0.84)
    n_fronds = 9
    for f in range(n_fronds):
        a = 2 * math.pi * f / n_fronds + 0.2 * (f % 2)
        ca, sa = math.cos(a), math.sin(a)
        L, droop = 0.42 + 0.06 * (f % 3), 0.28
        pts = []
        for i in range(7):
            s = i / 6
            r = L * s
            pts.append((top[0] + ca * r, top[1] + sa * r, top[2] + 0.07 * s - droop * s * s, 0.075 * math.sin(math.pi * min(1, s * 1.15))))
        for i in range(6):
            (x0, y0, z0, w0), (x1, y1, z1, w1) = pts[i], pts[i + 1]
            px, py = -sa, ca
            a0 = (x0 + px * w0, y0 + py * w0, z0 - 0.015)
            b0 = (x0 - px * w0, y0 - py * w0, z0 - 0.015)
            a1 = (x1 + px * w1, y1 + py * w1, z1 - 0.015)
            b1 = (x1 - px * w1, y1 - py * w1, z1 - 0.015)
            c0, c1 = (x0, y0, z0), (x1, y1, z1)
            for tri in ((a0, c0, c1), (a0, c1, a1), (c0, b0, b1), (c0, b1, c1)):
                e1, e2 = np.subtract(tri[1], tri[0]), np.subtract(tri[2], tri[0])
                nrm = np.cross(e1, e2)
                nrm = nrm / (np.linalg.norm(nrm) + 1e-9)
                if nrm[2] < 0:
                    nrm = -nrm
                for p in tri:
                    v.append((*p, *nrm, 1.0))
    v += _blob(top[0], top[1], top[2] - 0.02, 0.05, 0.05, 0.04, 0.0, rings=4, seg=6, seed=9)
    return np.array(v, "f4")


def banana_mesh():
    v = _cone(6, 0.06, 0.04, 0.0, 0.5, 0.0)
    for f in range(6):
        a = 2 * math.pi * f / 6
        ca, sa = math.cos(a), math.sin(a)
        base = (0.0, 0.0, 0.5)
        tip = (ca * 0.55, sa * 0.55, 0.62)
        mid = (ca * 0.3, sa * 0.3, 0.86)
        px, py = -sa * 0.13, ca * 0.13
        quad = [(base[0] + px * 0.3, base[1] + py * 0.3, base[2]), (mid[0] + px, mid[1] + py, mid[2]), tip,
                (mid[0] - px, mid[1] - py, mid[2]), (base[0] - px * 0.3, base[1] - py * 0.3, base[2])]
        for tri in ((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[4]), (quad[4], quad[2], quad[3])):
            e1, e2 = np.subtract(tri[1], tri[0]), np.subtract(tri[2], tri[0])
            nrm = np.cross(e1, e2)
            nrm = nrm / (np.linalg.norm(nrm) + 1e-9)
            if nrm[2] < 0:
                nrm = -nrm
            for p in tri:
                v.append((*p, *nrm, 1.0))
    return np.array(v, "f4")


def bush_mesh():
    v = _blob(0.0, 0.0, 0.42, 0.5, 0.5, 0.42, 1.0, rings=4, seg=8, seed=11)
    v += _blob(0.28, 0.1, 0.32, 0.3, 0.3, 0.28, 1.0, rings=4, seg=7, seed=12)
    return np.array(v, "f4")


def broadleaf_far():
    v = _cone(4, 0.05, 0.03, 0.0, 0.45, 0.0)
    v += _blob(0.0, 0.0, 0.66, 0.40, 0.40, 0.32, 1.0, rings=3, seg=6, seed=1)
    return np.array(v, "f4")


def palm_far():
    v = _cone(4, 0.03, 0.02, 0.0, 0.82, 0.0)
    v += _blob(0.06, 0.0, 0.80, 0.36, 0.36, 0.08, 1.0, rings=2, seg=8, seed=8)
    return np.array(v, "f4")


def banana_far():
    v = _cone(4, 0.06, 0.04, 0.0, 0.5, 0.0)
    v += _blob(0.0, 0.0, 0.66, 0.42, 0.42, 0.2, 1.0, rings=2, seg=6, seed=7)
    return np.array(v, "f4")


def bush_far():
    return np.array(_blob(0.0, 0.0, 0.42, 0.55, 0.55, 0.42, 1.0, rings=2, seg=6, seed=11), "f4")


SPECIES = {
    "broad": dict(mesh=broadleaf_mesh, far=broadleaf_far, leaf_a=(0.20, 0.42, 0.14), leaf_b=(0.42, 0.62, 0.20), trunk=(0.30, 0.20, 0.12), wind=0.5),
    "palm": dict(mesh=palm_mesh, far=palm_far, leaf_a=(0.30, 0.52, 0.16), leaf_b=(0.62, 0.70, 0.24), trunk=(0.44, 0.34, 0.24), wind=1.0),
    "banana": dict(mesh=banana_mesh, far=banana_far, leaf_a=(0.36, 0.62, 0.18), leaf_b=(0.62, 0.78, 0.26), trunk=(0.40, 0.46, 0.20), wind=0.9),
    "bush": dict(mesh=bush_mesh, far=bush_far, leaf_a=(0.28, 0.44, 0.16), leaf_b=(0.50, 0.58, 0.24), trunk=(0.30, 0.22, 0.12), wind=0.3),
}


class Nature:
    def __init__(self, ctx, terrain, lake_z):
        self.ctx = ctx
        self.T = terrain
        self.p_tree = ctx.program(vertex_shader=TREE_VS, fragment_shader=TREE_FS)
        self.p_water = ctx.program(vertex_shader=WATER_VS, fragment_shader=WATER_FS)
        self.p_tree["shadow_map"].value = 7
        self.p_water["shadow_map"].value = 7
        inst = landcover.tree_instances()
        import gc
        self.species = {}
        self.counts = {}
        for name, spec in SPECIES.items():
            arr = inst[name]
            if len(arr) == 0:
                continue
            z = terrain.sample(arr[:, 0], arr[:, 1]).astype(np.float32)
            ok = z > lake_z + 1.0
            arr, z = arr[ok], z[ok]
            tx = np.floor((arr[:, 0] - terrain.x0) / TILE).astype(np.int32)
            ty = np.floor((arr[:, 1] - terrain.y0) / TILE).astype(np.int32)
            order = np.lexsort((ty, tx))
            arr, z, tx, ty = arr[order], z[order], tx[order], ty[order]
            data = np.column_stack([arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], z - 0.4, arr[:, 4]]).astype("f4")
            key = tx * 100000 + ty
            uniq, start, cnt = np.unique(key, return_index=True, return_counts=True)
            tiles = [((k // 100000 + 0.5) * TILE + terrain.x0, (k % 100000 + 0.5) * TILE + terrain.y0, s, c)
                     for k, s, c in zip(uniq, start, cnt)]
            mesh = spec["mesh"]()
            far = spec["far"]()
            mesh_buf, far_buf = ctx.buffer(mesh), ctx.buffer(far)
            rng = np.random.default_rng(len(name))
            tile_vaos = []
            for cx, cy, st, cnt in tiles:
                chunk = data[st:st + cnt][rng.permutation(cnt)]
                buf = ctx.buffer(np.ascontiguousarray(chunk).tobytes())
                near_vao = ctx.vertex_array(self.p_tree, [(mesh_buf, "3f 3f 1f", "in_pos", "in_n", "in_k"),
                                                          (buf, "4f 2f/i", "i_pos", "i_var")])
                far_vao = ctx.vertex_array(self.p_tree, [(far_buf, "3f 3f 1f", "in_pos", "in_n", "in_k"),
                                                         (buf, "4f 2f/i", "i_pos", "i_var")])
                tz = float(np.median(chunk[:, 4]))
                tile_vaos.append((cx, cy, tz, int(cnt), near_vao, far_vao))
            self.species[name] = dict(tiles=tile_vaos, spec=spec, nv=len(mesh), nvf=len(far))
            self.counts[name] = int(len(data))
            del data, arr, z, tx, ty, chunk
            inst[name] = None
            gc.collect()
        self.vao_water, self.n_water = self.water_mesh(lake_z)

    def water_mesh(self, lake_z):
        T = self.T
        polys = landcover.water_polygons()
        if not polys:
            return None, 0
        tris = []
        for poly in polys:
            if poly.area < 2000:
                continue
            poly = poly.simplify(15.0, preserve_topology=True)
            if poly.is_empty or poly.geom_type != "Polygon":
                continue
            step = 220.0
            x0, y0, x1, y1 = poly.bounds
            gx, gy = np.meshgrid(np.arange(x0, x1 + step, step), np.arange(y0, y1 + step, step))
            pts = np.column_stack([gx.ravel(), gy.ravel()])
            inside = pts[shapely.contains_xy(poly.buffer(-30), pts[:, 0], pts[:, 1])]
            from shapely.geometry import MultiPoint
            ring = np.array(poly.exterior.coords)
            allp = np.vstack([ring, inside]) if len(inside) else ring
            dt = shapely.delaunay_triangles(MultiPoint(allp))
            for tr in shapely.get_parts(dt):
                c = np.array(tr.exterior.coords)[:3]
                if not poly.buffer(1.0).contains(tr.centroid):
                    continue
                tris.append(c)
        if not tris:
            return None, 0
        tris = np.array(tris)
        flat = tris.reshape(-1, 2)
        allpoly = shapely.union_all([p for p in polys])
        bnd = allpoly.boundary
        d = shapely.distance(bnd, shapely.points(flat))
        shore = np.clip(d / 700.0, 0, 1).astype(np.float32)
        z = np.full(len(flat), lake_z, np.float32)
        data = np.column_stack([flat.astype(np.float32), z, shore]).astype("f4")
        vao = self.ctx.vertex_array(self.p_water, [(self.ctx.buffer(data), "2f 1f 1f", "in_xy", "in_z", "in_shore")])
        return vao, len(data)

    def set_common(self, prog, VP, LVP, cam, fog_dist, use_shadow, L, haze, t):
        from gl3d import m4
        prog["VP"].write(m4(VP))
        for k, v in (("LVP", None),):
            if k in prog:
                prog[k].write(m4(LVP))
        for k, v in (("cam", tuple(cam)), ("fog_dist", fog_dist), ("haze", haze), ("use_shadow", use_shadow),
                     ("L", tuple(L)), ("Lsun", tuple(L)), ("t", t), ("bend", STATE["bend"]),
                     ("sunp", STATE["sunp"])):
            if k in prog:
                prog[k].value = v

    def draw_trees(self, VP, LVP, cam_pos, target, cam_dist, fog_dist, use_shadow, L, haze, t, shadow_pass=False):
        from gl3d import FPX
        self.set_common(self.p_tree, VP, LVP, cam_pos, fog_dist, use_shadow, L, haze, t)
        jobs = []
        for name, s in self.species.items():
            href = 3.0 if name == "bush" else 12.0
            for cx, cy, tz, cnt, nvao, fvao in s["tiles"]:
                d = math.sqrt((cx - cam_pos[0]) ** 2 + (cy - cam_pos[1]) ** 2 + (tz - cam_pos[2]) ** 2) - TILE * 0.7
                d = max(d, 1.0)
                px = href * FPX / d
                if px < (4.0 if shadow_pass else 2.5):
                    continue
                if shadow_pass and d > cam_dist * 1.4 + 400:
                    continue
                frac = 1.0 if px >= 10.0 else max(0.2, (px / 10.0) ** 1.5)
                jobs.append((d, name, max(1, int(cnt * frac)), nvao if (px >= 16.0 and not shadow_pass) else fvao,
                             s["nv"] if (px >= 16.0 and not shadow_pass) else s["nvf"]))
        jobs.sort(key=lambda j: j[0])
        jobs = jobs[: (60 if shadow_pass else 220)]
        budget_v = (2.5e6 if shadow_pass else 9e6)
        used_v, drawn, cur = 0.0, 0, None
        for d, name, n, vao, nv in jobs:
            if used_v + n * nv > budget_v:
                n = int((budget_v - used_v) / nv)
                if n <= 0:
                    break
            if name != cur:
                spec = self.species[name]["spec"]
                self.p_tree["leaf_a"].value = spec["leaf_a"]
                self.p_tree["leaf_b"].value = spec["leaf_b"]
                self.p_tree["trunk"].value = spec["trunk"]
                self.p_tree["wind"].value = spec["wind"]
                cur = name
            vao.render(moderngl.TRIANGLES, instances=n)
            used_v += n * nv
            drawn += n
        self.last_drawn = drawn
        return drawn

    def draw_water(self, VP, LVP, cam_pos, fog_dist, use_shadow, L, haze, t):
        if self.vao_water is None:
            return
        self.set_common(self.p_water, VP, LVP, cam_pos, fog_dist, use_shadow, L, haze, t)
        self.vao_water.render()
