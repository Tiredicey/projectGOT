import json
import math

import networkx as nx
import numpy as np
from scipy.ndimage import gaussian_filter1d

import geo
from timeline import LEGS

DRIVABLE = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
            "living_street", "service", "motorway_link", "trunk_link", "primary_link", "secondary_link",
            "tertiary_link"}
STOPS = ["home", "sti", "cathedral", "sm"]
STEP = 4.0


def road_graph():
    g = nx.Graph()
    pos = {}
    for name in ("osm_lipa.json", "osm_local.json"):
        for e in geo.load_elements(name):
            if e["type"] != "way" or e.get("tags", {}).get("highway") not in DRIVABLE:
                continue
            nodes, geom = e.get("nodes"), e.get("geometry")
            if not nodes or not geom or len(nodes) != len(geom):
                continue
            for nid, p in zip(nodes, geom):
                if p:
                    pos[nid] = geo.to_xy(p["lat"], p["lon"])
            for a, b in zip(nodes[:-1], nodes[1:]):
                if a in pos and b in pos:
                    g.add_edge(a, b, w=math.dist(pos[a], pos[b]))
    return g, pos


def nearest(g, pos, xy):
    ids = [n for n in g.nodes if n in pos]
    arr = np.array([pos[n] for n in ids])
    return ids[int(np.argmin(np.hypot(*(arr - xy).T)))]


def resample(pts, step):
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(2, int(s[-1] / step) + 1)
    si = np.linspace(0, s[-1], n)
    return np.column_stack([np.interp(si, s, pts[:, 0]), np.interp(si, s, pts[:, 1])]), si


class Route:
    def __init__(self, centers):
        g, pos = road_graph()
        comp = max(nx.connected_components(g), key=len)
        g = g.subgraph(comp)
        self.legs = []
        report = []
        for a, b, t0, t1 in LEGS:
            na, nb = nearest(g, pos, centers[a]), nearest(g, pos, centers[b])
            path = nx.shortest_path(g, na, nb, weight="w")
            raw = np.array([centers[a]] + [pos[n] for n in path] + [centers[b]])
            pts, s = resample(raw, STEP)
            self.legs.append(dict(a=a, b=b, t0=t0, t1=t1, pts=pts, s=s, L=float(s[-1])))
            report.append(dict(leg=f"{a}->{b}", km=round(s[-1] / 1000, 2), osm_nodes=len(path),
                               snap_a_m=round(float(np.hypot(*(np.array(pos[na]) - centers[a]))), 1),
                               snap_b_m=round(float(np.hypot(*(np.array(pos[nb]) - centers[b]))), 1)))
        self.report = report
        offsets = np.cumsum([0] + [leg["L"] for leg in self.legs])
        self.pts = np.vstack([leg["pts"] for leg in self.legs])
        self.s = np.concatenate([leg["s"] + o for leg, o in zip(self.legs, offsets[:-1])])
        self.total = float(offsets[-1])
        self.leg_offsets = offsets
        for leg, o in zip(self.legs, offsets[:-1]):
            sig = 300.0 / STEP
            leg["rail"] = np.column_stack([gaussian_filter1d(leg["pts"][:, i], sig, mode="nearest") for i in range(2)])
            tan = np.gradient(gaussian_filter1d(leg["pts"], 650.0 / STEP, axis=0, mode="nearest"), axis=0)
            leg["yaw"] = np.degrees(np.unwrap(np.arctan2(tan[:, 0], tan[:, 1])))
            leg["off"] = float(o)

    @staticmethod
    def ease(p):
        p = min(1.0, max(0.0, p))
        return p * p * (3 - 2 * p)

    def progress(self, t):
        s = 0.0
        for leg in self.legs:
            if t >= leg["t1"]:
                s = leg["off"] + leg["L"]
            elif t > leg["t0"]:
                s = leg["off"] + leg["L"] * self.ease((t - leg["t0"]) / (leg["t1"] - leg["t0"]))
                break
            else:
                break
        return s

    def head(self, t):
        s = self.progress(t)
        return np.array([np.interp(s, self.s, self.pts[:, 0]), np.interp(s, self.s, self.pts[:, 1])])

    def rail(self, leg_idx, p):
        leg = self.legs[leg_idx]
        s = leg["L"] * self.ease(p)
        x = np.interp(s, leg["s"], leg["rail"][:, 0])
        y = np.interp(s, leg["s"], leg["rail"][:, 1])
        yaw = float(np.interp(s, leg["s"], leg["yaw"]))
        return x, y, yaw

    def dump(self, path):
        json.dump(self.report, open(path, "w"), indent=1)
