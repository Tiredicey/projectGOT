import json
import math
import zlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon

from timeline import HERO_IDS, LAND, LAT0, LON0

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = ROOT / "build" / "cache"
FONT = str(ROOT / "fonts" / "Cinzel.ttf")

KX = 111320.0 * math.cos(math.radians(LAT0))
KY = 110574.0

PARCH_LIGHT = np.array([222, 203, 160], np.float32)
PARCH_DARK = np.array([176, 146, 98], np.float32)
INK = (52, 36, 22)
INK_SOFT = (96, 72, 46)
RIVER = (88, 104, 104)
WATER = (128, 146, 140)


def to_xy(lat, lon):
    return (lon - LON0) * KX, (lat - LAT0) * KY


def stable_hash(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def font(size, weight=600):
    f = ImageFont.truetype(FONT, size)
    try:
        f.set_variation_by_axes([weight])
    except OSError:
        pass
    return f


def value_noise(h, w, scale, seed):
    rng = np.random.default_rng(seed)
    gh, gw = max(2, int(h / scale) + 2), max(2, int(w / scale) + 2)
    g = rng.random((gh, gw)).astype(np.float32)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)


def parchment(h, w, seed, px_m):
    n = np.zeros((h, w), np.float32)
    total = 0.0
    import gc
    for i, metres in enumerate((2400, 900, 300, 90)):
        s = metres / px_m
        if s < 2:
            continue
        amp = 0.6 ** i
        n += amp * value_noise(h, w, s, seed + i)
        total += amp
    n = n / max(total, 1e-6)
    n = (n - n.min()) / (np.ptp(n) + 1e-6)
    rng = np.random.default_rng(seed + 99)
    fib = rng.standard_normal((h, w)).astype(np.float32)
    fib = cv2.GaussianBlur(fib, (0, 0), sigmaX=3.0, sigmaY=0.6) * 0.05
    n *= 0.85
    n += fib
    n += 0.08
    del fib
    np.clip(n, 0, 1, out=n)
    img = np.empty((h, w, 3), np.uint8)
    for c in range(3):
        img[..., c] = (PARCH_DARK[c] + (PARCH_LIGHT[c] - PARCH_DARK[c]) * n).astype(np.uint8)
    del n
    gc.collect()
    return img


class Layer:
    def __init__(self, name, x0, x1, y0, y1, res):
        self.name, self.x0, self.x1, self.y0, self.y1, self.res = name, x0, x1, y0, y1, res
        self.w = int(round((x1 - x0) / res))
        self.h = int(round((y1 - y0) / res))

    def px(self, xy):
        xy = np.asarray(xy, np.float64)
        u = (xy[..., 0] - self.x0) / self.res
        v = (self.y1 - xy[..., 1]) / self.res
        return np.stack([u, v], -1)

    def tex_to_world(self):
        return np.array([[self.res, 0, self.x0], [0, -self.res, self.y1], [0, 0, 1]], np.float64)


def load_elements(name):
    return json.load(open(DATA / name))["elements"]


def way_xy(e):
    return np.array([to_xy(p["lat"], p["lon"]) for p in e.get("geometry", []) if p], np.float64)


ROAD_W = {"motorway": 22, "trunk": 16, "primary": 13, "secondary": 10, "tertiary": 8,
          "unclassified": 6, "residential": 5, "service": 3.5, "living_street": 4, "track": 2.5}


def draw_roads(img, layer, elements, scale_w=1.0, min_px=1):
    order = sorted((e for e in elements if e["type"] == "way" and e["tags"].get("highway") in ROAD_W),
                   key=lambda e: ROAD_W[e["tags"]["highway"]])
    for e in order:
        pts = layer.px(way_xy(e))
        if len(pts) < 2:
            continue
        w = max(min_px, int(round(ROAD_W[e["tags"]["highway"]] * scale_w / layer.res)))
        p = np.round(pts * 4).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [p], False, INK_SOFT, w + 2, cv2.LINE_AA, shift=2)
        cv2.polylines(img, [p], False, (214, 190, 140), max(1, w - 1), cv2.LINE_AA, shift=2)


def text_on(img, xy, s, size, color=INK, weight=700, spacing=0.18, anchor="mm"):
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    f = font(size, weight)
    s = s.upper()
    widths = [d.textlength(c, font=f) for c in s]
    total = sum(widths) + spacing * size * (len(s) - 1)
    x = xy[0] - total / 2
    for c, cw in zip(s, widths):
        d.text((x, xy[1]), c, font=f, fill=color, anchor="lm")
        x += cw + spacing * size
    img[:] = np.asarray(pil)


def mountain_glyph(img, c, size):
    x, y = c
    for k, (dx, s) in enumerate(((-0.55, 0.7), (0.0, 1.0), (0.5, 0.8))):
        cx, sz = x + dx * size, size * s
        tri = np.array([[cx - sz * 0.6, y + sz * 0.35], [cx, y - sz * 0.55], [cx + sz * 0.6, y + sz * 0.35]])
        cv2.fillPoly(img, [np.round(tri * 4).astype(np.int32)], (190, 160, 110), cv2.LINE_AA, shift=2)
        cv2.polylines(img, [np.round(tri * 4).astype(np.int32)], False, INK, max(1, int(size / 14)), cv2.LINE_AA, shift=2)
        hatch = np.array([[cx, y - sz * 0.55], [cx + sz * 0.25, y + sz * 0.35]])
        cv2.line(img, tuple(np.round(hatch[0]).astype(int)), tuple(np.round(hatch[1]).astype(int)), INK_SOFT, 1, cv2.LINE_AA)


def build_region():
    import landcover
    layer = Layer("region", -11000, 12000, -7500, 12500, 8.0)
    img = parchment(layer.h, layer.w, 11, layer.res)
    extra = load_elements("osm_extra.json")
    landcover.paint_terrain_tint(img, layer)
    landcover.paint_landcover(img, layer, stipple=True)
    landcover.draw_roads(img, layer, load_elements("osm_lipa.json"), scale_w=2.2, min_px=2)
    peaks = []
    for e in extra:
        t = e.get("tags", {})
        if e["type"] == "node" and t.get("natural") == "peak":
            u, v = layer.px(to_xy(e["lat"], e["lon"]))
            if 40 < u < layer.w - 40 and 40 < v < layer.h - 40:
                ele = float(t.get("ele", "300") or 300)
                if t.get("name") and t.get("ele"):
                    peaks.append((ele, u, v, t["name"]))
    placed = []
    for ele, u, v, name in sorted(peaks, reverse=True):
        if all(abs(u - pu) > 260 or abs(v - pv) > 90 for pu, pv in placed):
            text_on(img, (u, v), name, 30, INK_SOFT, 600)
            placed.append((u, v))
    for lat, lon, label, size in REGION_LABELS:
        u, v = layer.px(to_xy(lat, lon))
        text_on(img, (u, v), label, size, INK if size > 100 else INK_SOFT, 800 if size > 100 else 700, 0.3)
    return layer, img


REGION_LABELS = [
    (13.9700, 121.1450, "LIPA", 300),
    (13.9850, 121.0700, "TAAL LAKE", 70),
    (14.007505, 121.165597, "LIMA", 56),
]


DEM_Z, DEM_X0, DEM_Y0, DEM_N = 12, 3425, 1886, 3
EXAG = 1.5
GRID = 60.0


def dem_mosaic():
    rows = []
    for y in range(DEM_Y0, DEM_Y0 + DEM_N):
        row = []
        for x in range(DEM_X0, DEM_X0 + DEM_N):
            a = cv2.imread(str(DATA / "dem" / f"{DEM_Z}_{x}_{y}.png"))[..., ::-1].astype(np.float64)
            row.append(a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768)
        rows.append(np.hstack(row))
    return np.clip(np.vstack(rows), 0, None).astype(np.float32)


def elevation(xs, ys, mosaic):
    lon = LON0 + np.asarray(xs) / KX
    lat = LAT0 + np.asarray(ys) / KY
    n = 2 ** DEM_Z
    px = ((lon + 180) / 360 * n - DEM_X0) * 256 - 0.5
    py = ((1 - np.arcsinh(np.tan(np.radians(lat))) / math.pi) / 2 * n - DEM_Y0) * 256 - 0.5
    return cv2.remap(mosaic, px.astype(np.float32), py.astype(np.float32), cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_REPLICATE)


class Terrain:
    def __init__(self, x0=-11000, x1=12000, y0=-7500, y1=12500, step=GRID):
        self.x0, self.y0, self.step = x0, y0, step
        self.nx = int((x1 - x0) / step) + 1
        self.ny = int((y1 - y0) / step) + 1
        xs = x0 + np.arange(self.nx) * step
        ys = y0 + np.arange(self.ny) * step
        gx, gy = np.meshgrid(xs, ys)
        mosaic = dem_mosaic()
        self.elev = elevation(gx, gy, mosaic)
        self.e0 = float(np.median(self.elev))
        self.z = ((self.elev - self.e0) * EXAG).astype(np.float32)
        self.zs = cv2.GaussianBlur(self.z, (0, 0), 5)
        self.gx, self.gy = gx.astype(np.float32), gy.astype(np.float32)

    def sample(self, x, y, smooth=False):
        g = self.zs if smooth else self.z
        fx = np.clip((np.asarray(x, float) - self.x0) / self.step, 0, self.nx - 1.001)
        fy = np.clip((np.asarray(y, float) - self.y0) / self.step, 0, self.ny - 1.001)
        ix, iy = np.floor(fx).astype(int), np.floor(fy).astype(int)
        ax, ay = fx - ix, fy - iy
        return (g[iy, ix] * (1 - ax) * (1 - ay) + g[iy, ix + 1] * ax * (1 - ay)
                + g[iy + 1, ix] * (1 - ax) * ay + g[iy + 1, ix + 1] * ax * ay)

    def mesh(self):
        dzdy, dzdx = np.gradient(self.z, self.step)
        n = np.dstack([-dzdx, -dzdy, np.ones_like(self.z)])
        n /= np.linalg.norm(n, axis=2, keepdims=True)
        verts = np.dstack([self.gx, self.gy, self.z, n]).reshape(-1, 6).astype(np.float32)
        i = np.arange(self.nx * self.ny).reshape(self.ny, self.nx)
        a, b, c, d = i[:-1, :-1], i[:-1, 1:], i[1:, :-1], i[1:, 1:]
        idx = np.stack([a, b, d, a, d, c], -1).reshape(-1).astype(np.int32)
        return verts, idx


def footprint(e):
    pts = way_xy(e)
    if len(pts) < 4:
        return None
    poly = Polygon(pts)
    if not poly.is_valid or poly.area < 12:
        return None
    poly = poly.simplify(0.4, preserve_topology=True)
    if poly.geom_type != "Polygon":
        return None
    xy = np.array(poly.exterior.coords)[:-1]
    if Polygon(xy).exterior.is_ccw is False:
        xy = xy[::-1]
    return xy


def building_height(e):
    t = e["tags"]
    try:
        if "height" in t:
            return float(t["height"]), "osm:height"
        if "building:levels" in t:
            return float(t["building:levels"]) * 3.2, "osm:levels"
    except ValueError:
        pass
    h = 3.2 * (1 + stable_hash(e["id"]) % 2) + (stable_hash(e["id"], "x") % 100) / 100
    return h, "assumed"


def build_local(name, cx, cy, half, res, seed):
    import landcover
    layer = Layer(name, cx - half, cx + half, cy - half, cy + half, res)
    img = parchment(layer.h, layer.w, seed, layer.res)
    landcover.paint_terrain_tint(img, layer)
    landcover.paint_landcover(img, layer, stipple=True)
    local = load_elements("osm_local.json")
    landcover.draw_roads(img, layer, local, scale_w=1.0, min_px=2)
    for e in local:
        if "building" in e.get("tags", {}):
            fp = footprint(e)
            if fp is None:
                continue
            p = np.round(layer.px(fp) * 4).astype(np.int32)
            cv2.fillPoly(img, [p], (214, 196, 168), cv2.LINE_AA, shift=2)
            cv2.polylines(img, [p], True, INK_SOFT, 1, cv2.LINE_AA, shift=2)
    return layer, img


LOCAL_SPECS = {
    "home": dict(center="home", half=900, res=0.9, seed=21),
    "city": dict(center=(2350, 300), half=1500, res=1.1, seed=31),
}


def buildings():
    local = load_elements("osm_local.json")
    centers = {}
    for key, oid in HERO_IDS.items():
        if key == "terminal":
            continue
        e = next(x for x in local if x["id"] == oid)
        fp = footprint(e)
        centers[key] = fp.mean(axis=0)
    out = []
    for e in local:
        if "building" not in e.get("tags", {}):
            continue
        fp = footprint(e)
        if fp is None:
            continue
        c = fp.mean(axis=0)
        cluster = min(centers, key=lambda k: np.hypot(*(centers[k] - c)))
        dist = float(np.hypot(*(centers[cluster] - c)))
        if dist > 420:
            continue
        h, hsrc = building_height(e)
        hero = next((k for k, v in HERO_IDS.items() if v == e["id"]), None)
        out.append(dict(id=e["id"], fp=fp, c=c, h=h, hsrc=hsrc, cluster=cluster, dist=dist, hero=hero,
                        name=e["tags"].get("name"), roof_tag=e["tags"].get("roof:colour"),
                        wall_tag=e["tags"].get("building:colour")))
    return out, centers


def build_corridor():
    import landcover
    layer = Layer("corridor", -3800, 3400, -1700, 1500, 2.0)
    img = parchment(layer.h, layer.w, 41, layer.res)
    landcover.paint_terrain_tint(img, layer)
    landcover.paint_landcover(img, layer, stipple=True)
    landcover.draw_roads(img, layer, load_elements("osm_lipa.json"), scale_w=1.6, min_px=3)
    landcover.draw_roads(img, layer, load_elements("osm_local.json"), scale_w=1.0, min_px=2)
    for lat, lon, label, size in CORRIDOR_LABELS:
        u, v = layer.px(to_xy(lat, lon))
        text_on(img, (u, v), label, size, INK_SOFT, 700, 0.3)
    return layer, img


CORRIDOR_LABELS = [
]


def ensure_cache(force=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    meta_path = CACHE / "meta.json"
    if meta_path.exists() and not force:
        return json.loads(meta_path.read_text())
    meta = {"layers": {}}
    layer, img = build_region()
    np.save(CACHE / "region.npy", img)
    meta["layers"]["region"] = layer.__dict__
    layer, img = build_corridor()
    np.save(CACHE / "corridor.npy", img)
    meta["layers"]["corridor"] = layer.__dict__
    del img
    bl, centers = buildings()
    for name, spec in LOCAL_SPECS.items():
        c = centers[spec["center"]] if isinstance(spec["center"], str) else spec["center"]
        layer, img = build_local(name, float(c[0]), float(c[1]), spec["half"], spec["res"], spec["seed"])
        np.save(CACHE / f"{name}.npy", img)
        meta["layers"][name] = layer.__dict__
    meta["centers"] = {k: [float(v[0]), float(v[1])] for k, v in centers.items()}
    meta["n_buildings"] = len(bl)
    meta["height_sources"] = {s: sum(1 for b in bl if b["hsrc"] == s) for s in ("osm:height", "osm:levels", "assumed")}
    meta_path.write_text(json.dumps(meta, indent=1))
    return meta


def load_layer(meta, name):
    d = meta["layers"][name]
    layer = Layer(d["name"], d["x0"], d["x1"], d["y0"], d["y1"], d["res"])
    img = np.load(CACHE / f"{name}.npy", mmap_mode="r")
    return layer, img
