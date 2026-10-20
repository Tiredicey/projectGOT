import json
from functools import lru_cache

import cv2
import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Polygon, box
from shapely.ops import linemerge, polygonize, unary_union

import geo

SLAB = box(-11000, -7500, 12000, 12500)
PAPER_REF = np.array([206, 186, 140], np.float32)

WASH = {
    "grassland": [(168, 182, 96), (178, 186, 104), (156, 176, 92)],
    "grass": [(150, 184, 98), (138, 178, 92)],
    "meadow": [(176, 188, 98)],
    "greenfield": [(196, 182, 120)],
    "farmland": [(214, 186, 92), (178, 192, 96), (196, 150, 92), (222, 202, 120), (160, 184, 88)],
    "scrub": [(120, 146, 76), (132, 152, 82)],
    "orchard": [(118, 162, 70), (128, 170, 76)],
    "wood": [(70, 118, 60), (78, 126, 64), (64, 110, 58)],
    "forest": [(64, 110, 58)],
}
ORDER = ["greenfield", "grassland", "grass", "meadow", "farmland", "scrub", "orchard", "forest", "wood"]
DEEP = np.array([34, 92, 128], np.float32)
SHALLOW = np.array([96, 168, 170], np.float32)
RIVER_RGB = (58, 120, 150)
ROAD_CASE = (92, 66, 42)
ROAD_FILL = {"motorway": (204, 112, 78), "trunk": (226, 160, 92), "primary": (236, 196, 120),
             "secondary": (240, 220, 160)}
ROAD_DEFAULT = (236, 222, 186)


def _rings(geom):
    return [geo.to_xy(p["lat"], p["lon"]) for p in geom if p]


def _relation_polygon(e):
    outers = [LineString(_rings(m["geometry"])) for m in e.get("members", []) if m.get("role") == "outer" and m.get("geometry")]
    inners = [LineString(_rings(m["geometry"])) for m in e.get("members", []) if m.get("role") == "inner" and m.get("geometry")]
    if not outers:
        return None
    out = unary_union(list(polygonize(linemerge(outers))))
    if inners:
        inn = unary_union(list(polygonize(linemerge(inners))))
        out = out.difference(inn)
    return out.buffer(0)


def _polys(g):
    if g is None or g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    if isinstance(g, MultiPolygon):
        return list(g.geoms)
    return [p for p in getattr(g, "geoms", []) if isinstance(p, Polygon)]


@lru_cache(maxsize=1)
def load():
    els = geo.load_elements("osm_green.json")
    classes = {k: [] for k in ORDER}
    orchard_kind = []
    water = []
    rows = []
    for e in els:
        t = e.get("tags", {})
        k = t.get("landuse") or t.get("natural")
        if e["type"] == "way":
            pts = _rings(e["geometry"])
            if k == "tree_row":
                if len(pts) >= 2:
                    rows.append(LineString(pts))
                continue
            if len(pts) < 4:
                continue
            g = Polygon(pts).buffer(0)
        else:
            g = _relation_polygon(e)
        if g is None:
            continue
        g = g.intersection(SLAB)
        if g.is_empty:
            continue
        if k == "water":
            water.extend(_polys(g))
        elif k in classes:
            for p in _polys(g):
                classes[k].append((e["id"], p))
                if k == "orchard":
                    orchard_kind.append((p, t.get("trees", "")))
    for e in geo.load_elements("osm_extra.json"):
        t = e.get("tags", {})
        if e["type"] == "way" and t.get("natural") == "water" and len(e.get("geometry", [])) > 3:
            g = Polygon(_rings(e["geometry"])).buffer(0).intersection(SLAB)
            water.extend(_polys(g))
    rivers = []
    for e in geo.load_elements("osm_extra.json"):
        t = e.get("tags", {})
        if e["type"] == "way" and t.get("waterway") in ("river", "stream"):
            rivers.append((t["waterway"], LineString(_rings(e["geometry"]))))
    lake = unary_union(water) if water else Polygon()
    return dict(classes=classes, orchard_kind=orchard_kind, water=lake, rows=rows, rivers=rivers)


def _fill(mask, layer, poly, off):
    ext = (layer.px(np.array(poly.exterior.coords)) - off) * 4
    cv2.fillPoly(mask, [np.round(ext).astype(np.int32)], 255, cv2.LINE_AA, shift=2)
    for ring in poly.interiors:
        p = (layer.px(np.array(ring.coords)) - off) * 4
        cv2.fillPoly(mask, [np.round(p).astype(np.int32)], 0, cv2.LINE_AA, shift=2)


def _crop(layer, poly, pad=6):
    x0, y0, x1, y1 = poly.bounds
    (u0, v1), (u1, v0) = layer.px(np.array([[x0, y0], [x1, y1]]))
    a, b = max(0, int(u0) - pad), min(layer.w, int(u1) + pad + 1)
    c, d = max(0, int(v0) - pad), min(layer.h, int(v1) + pad + 1)
    if b <= a or d <= c:
        return None
    return a, b, c, d


def _wash(img, layer, poly, rgb, seed, strength=0.88, stripes=None):
    cr = _crop(layer, poly)
    if cr is None:
        return None
    a, b, c, d = cr
    m = np.zeros((d - c, b - a), np.uint8)
    _fill(m, layer, poly, np.array([a, c]))
    if not m.any():
        return None
    mf = m.astype(np.float32) / 255.0
    h, w = mf.shape
    rng_scale = max(3.0, 60.0 / layer.res)
    n = 0.88 + 0.24 * geo.value_noise(h, w, rng_scale, seed)
    if stripes is not None:
        ang, period = stripes
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ph = (xx * np.cos(ang) + yy * np.sin(ang)) * layer.res / period
        n *= 0.93 + 0.07 * np.sin(ph * 2 * np.pi)
    edge = np.clip(mf - cv2.GaussianBlur(mf, (0, 0), max(1.0, 3.0 / layer.res * 2)), 0, 1)
    roi = img[c:d, a:b].astype(np.float32)
    pig = np.array(rgb, np.float32)
    for ch in range(3):
        out = roi[..., ch] / PAPER_REF[ch] * pig[ch] * n * (1 - 0.45 * edge)
        roi[..., ch] = roi[..., ch] * (1 - strength * mf) + out * (strength * mf)
    img[c:d, a:b] = np.clip(roi, 0, 255).astype(np.uint8)
    return (a, b, c, d, m)


def _stipple(img, layer, crop, spacing_m, radius_m, seed, dark, light, rows_angle=None):
    a, b, c, d, m = crop
    sp = spacing_m / layer.res
    if sp < 2.2:
        sp = 2.2
    r = max(1, int(round(radius_m / layer.res)))
    rng = np.random.default_rng(seed)
    h, w = m.shape
    if rows_angle is None:
        gy, gx = np.mgrid[0:h:sp, 0:w:sp]
        pts = np.column_stack([gx.ravel(), gy.ravel()]) + rng.uniform(-0.45, 0.45, (gx.size, 2)) * sp
    else:
        L = int(np.hypot(h, w) / sp) + 2
        ii, jj = np.mgrid[-L:L, -L:L]
        ca, sa = np.cos(rows_angle), np.sin(rows_angle)
        pts = np.column_stack([(ii * ca - jj * sa).ravel(), (ii * sa + jj * ca).ravel()]) * sp + np.array([w / 2, h / 2])
    pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)]
    pts = pts[m[pts[:, 1].astype(int), pts[:, 0].astype(int)] > 128]
    sub = img[c:d, a:b]
    sh = max(1, r // 2)
    for x, y in pts.astype(int):
        cv2.circle(sub, (x + sh, y + sh), r, dark, -1, cv2.LINE_AA)
    for x, y in pts.astype(int):
        cv2.circle(sub, (x, y), r, light, -1, cv2.LINE_AA)
        if r >= 3:
            cv2.circle(sub, (x - r // 3, y - r // 3), max(1, r // 3), (min(255, light[0] + 40), min(255, light[1] + 40), light[2] + 20), -1, cv2.LINE_AA)


def _orient(poly):
    mrr = poly.minimum_rotated_rectangle
    xy = np.array(mrr.exterior.coords)
    e = xy[1] - xy[0]
    return float(np.arctan2(-e[1], e[0]))


def paint_terrain_tint(img, layer):
    h, w = img.shape[:2]
    step = max(1, int(round(30.0 / layer.res)))
    gh, gw = h // step + 2, w // step + 2
    vs, us = np.mgrid[0:gh, 0:gw].astype(np.float64) * step
    xs = layer.x0 + us * layer.res
    ys = layer.y1 - vs * layer.res
    el = geo.elevation(xs, ys, geo.dem_mosaic()).astype(np.float32)
    dzdy, dzdx = np.gradient(el, step * layer.res)
    lx, ly, lz = -0.5, 0.62, 0.6
    nz = 1.0 / np.sqrt(dzdx ** 2 + dzdy ** 2 + 1)
    shade = np.clip((-dzdx * lx * 1.0 + dzdy * ly * 1.0 + lz) * nz * 2.2, 0.0, 2.0) * 3.0
    shade = cv2.resize(shade, (gw * step, gh * step), interpolation=cv2.INTER_CUBIC)[:h, :w]
    elr = cv2.resize(el, (gw * step, gh * step), interpolation=cv2.INTER_CUBIC)[:h, :w]
    stops = np.array([0, 120, 300, 520, 760, 1000], np.float32)
    cols = np.array([[0.96, 1.04, 0.98], [1.0, 1.03, 0.94], [1.03, 1.0, 0.92], [1.06, 0.96, 0.88],
                     [1.02, 0.94, 0.96], [1.06, 1.04, 1.08]], np.float32)
    for ch in range(3):
        tint = np.interp(elr, stops, cols[:, ch]).astype(np.float32)
        x = img[..., ch].astype(np.float32) * tint * (0.82 + 0.22 * np.clip(shade, 0, 1.6))
        img[..., ch] = np.clip(x, 0, 255).astype(np.uint8)


def paint_water(img, layer, lc, seed):
    lake = lc["water"]
    lbox = box(layer.x0, layer.y0, layer.x1, layer.y1)
    for i, poly in enumerate(_polys(lake.intersection(lbox))):
        cr = _crop(layer, poly, pad=4)
        if cr is None:
            continue
        a, b, c, d = cr
        m = np.zeros((d - c, b - a), np.uint8)
        _fill(m, layer, poly, np.array([a, c]))
        if not m.any():
            continue
        dist = cv2.distanceTransform((m > 127).astype(np.uint8), cv2.DIST_L2, 5) * layer.res
        k = np.clip(dist / 900.0, 0, 1)[..., None] ** 0.6
        col = SHALLOW * (1 - k) + DEEP * k
        n = (0.94 + 0.12 * geo.value_noise(m.shape[0], m.shape[1], max(3, 220 / layer.res), seed + i))[..., None]
        col = col * n
        hatch = np.zeros_like(dist)
        for ring_m in (40, 90, 150, 230):
            hatch = np.maximum(hatch, np.clip(1 - np.abs(dist - ring_m) / max(1.2 * layer.res, 2.0), 0, 1) * (1 - ring_m / 300))
        col = col * (1 - 0.0 * hatch[..., None]) + np.array([200, 226, 222], np.float32) * hatch[..., None] * 0.55 + col * 0
        col = np.clip(col + hatch[..., None] * 40, 0, 255)
        mf = (m.astype(np.float32) / 255.0)[..., None]
        roi = img[c:d, a:b].astype(np.float32)
        paper = roi / PAPER_REF
        out = paper * col
        roi = roi * (1 - 0.95 * mf) + out * 0.95 * mf
        img[c:d, a:b] = np.clip(roi, 0, 255).astype(np.uint8)
        ext = (layer.px(np.array(poly.exterior.coords)) - np.array([0, 0])) * 4
        cv2.polylines(img, [np.round(ext).astype(np.int32)], True, (40, 74, 92), max(1, int(2 / max(layer.res, 1) + 1)), cv2.LINE_AA, shift=2)
    for kind, line in lc["rivers"]:
        if not line.intersects(lbox):
            continue
        p = np.round(layer.px(np.array(line.coords)) * 4).astype(np.int32).reshape(-1, 1, 2)
        wpx = max(1, int(round((8 if kind == "river" else 4) / layer.res)))
        cv2.polylines(img, [p], False, (44, 86, 104), wpx + 1, cv2.LINE_AA, shift=2)
        cv2.polylines(img, [p], False, RIVER_RGB, wpx, cv2.LINE_AA, shift=2)


def paint_landcover(img, layer, stipple=True):
    lc = load()
    lbox = box(layer.x0, layer.y0, layer.x1, layer.y1)
    for k in ORDER:
        pal = WASH[k]
        for oid, poly in lc["classes"][k]:
            if not poly.intersects(lbox):
                continue
            hsh = geo.stable_hash(oid, k)
            rgb = pal[hsh % len(pal)]
            stripes = (_orient(poly), 6.0) if k == "farmland" else None
            crop = _wash(img, layer, poly, rgb, hsh % 9973, 0.9 if k in ("wood", "forest") else 0.8, stripes)
            if crop is None or not stipple:
                continue
            if k in ("wood", "forest"):
                _stipple(img, layer, crop, 11.0, 4.2, hsh, (40, 70, 36), (82, 132, 66))
            elif k == "orchard":
                _stipple(img, layer, crop, 9.0, 3.0, hsh, (52, 86, 40), (110, 160, 72), rows_angle=_orient(poly))
            elif k == "scrub":
                _stipple(img, layer, crop, 9.0, 2.2, hsh, (70, 96, 50), (128, 152, 82))
    for line in lc["rows"]:
        if not line.intersects(lbox):
            continue
        L = line.length
        for s in np.arange(0, L, 8.0):
            x, y = line.interpolate(s).coords[0]
            u, v = layer.px((x, y))
            r = max(1, int(round(3.5 / layer.res)))
            cv2.circle(img, (int(u) + 1, int(v) + 1), r, (40, 70, 36), -1, cv2.LINE_AA)
            cv2.circle(img, (int(u), int(v)), r, (86, 136, 68), -1, cv2.LINE_AA)
    paint_water(img, layer, lc, 7)


def draw_roads(img, layer, elements, scale_w=1.0, min_px=1):
    order = sorted((e for e in elements if e["type"] == "way" and e["tags"].get("highway") in geo.ROAD_W),
                   key=lambda e: geo.ROAD_W[e["tags"]["highway"]])
    for e in order:
        pts = layer.px(geo.way_xy(e))
        if len(pts) < 2:
            continue
        hw = e["tags"]["highway"]
        w = max(min_px, int(round(geo.ROAD_W[hw] * scale_w / layer.res)))
        p = np.round(pts * 4).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [p], False, ROAD_CASE, w + 2, cv2.LINE_AA, shift=2)
        cv2.polylines(img, [p], False, ROAD_FILL.get(hw, ROAD_DEFAULT), max(1, w - 1), cv2.LINE_AA, shift=2)


def lake_level_z(T):
    return float((5.0 + 1.2 - T.e0) * geo.EXAG)


def water_polygons():
    return _polys(load()["water"])


def tree_instances(seed=5):
    lc = load()
    rng = np.random.default_rng(seed)
    out = {"broad": [], "palm": [], "banana": [], "bush": []}

    def scatter(poly, spacing, kind, hmin, hmax, jitter=0.45, angle=None):
        x0, y0, x1, y1 = poly.bounds
        if angle is None:
            gx, gy = np.meshgrid(np.arange(x0, x1, spacing), np.arange(y0, y1, spacing))
            pts = np.column_stack([gx.ravel(), gy.ravel()]) + rng.uniform(-jitter, jitter, (gx.size, 2)) * spacing
        else:
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            R = np.hypot(x1 - x0, y1 - y0) / 2 + spacing
            ii, jj = np.meshgrid(np.arange(-R, R, spacing), np.arange(-R, R, spacing))
            ca, sa = np.cos(angle), np.sin(angle)
            pts = np.column_stack([cx + ii.ravel() * ca - jj.ravel() * sa, cy + ii.ravel() * sa + jj.ravel() * ca])
            pts += rng.uniform(-0.12, 0.12, pts.shape) * spacing
        if len(pts) == 0:
            return
        pts = pts[shapely.contains_xy(poly, pts[:, 0], pts[:, 1])]
        if len(pts):
            hs = rng.uniform(hmin, hmax, len(pts))
            out[kind].append(np.column_stack([pts, hs, rng.uniform(0, 2 * np.pi, len(pts)), rng.uniform(0, 1, len(pts))]))

    for k in ("wood", "forest"):
        for oid, p in lc["classes"][k]:
            scatter(p, 9.0, "broad", 8.0, 15.0)
    for oid, p in lc["classes"]["scrub"]:
        scatter(p, 8.0, "bush", 2.0, 4.0)
    for p, trees in lc["orchard_kind"]:
        ang = -_orient(p)
        if "banana" in trees and "coconut" not in trees:
            scatter(p, 5.0, "banana", 3.5, 5.5, angle=ang)
        elif "mango" in trees and "coconut" not in trees:
            scatter(p, 10.0, "broad", 6.0, 10.0, angle=ang)
        else:
            scatter(p, 9.0, "palm", 11.0, 19.0, angle=ang)
    for line in lc["rows"]:
        s = np.arange(0, line.length, 8.0)
        pts = np.array([line.interpolate(v).coords[0] for v in s])
        if len(pts):
            out["broad"].append(np.column_stack([pts, rng.uniform(7, 12, len(pts)), rng.uniform(0, 2 * np.pi, len(pts)),
                                                 rng.uniform(0, 1, len(pts))]))
    return {k: (np.vstack(v).astype(np.float32) if v else np.zeros((0, 5), np.float32)) for k, v in out.items()}
