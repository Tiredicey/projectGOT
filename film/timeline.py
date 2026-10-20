import json
from pathlib import Path

import numpy as np

W, H, FPS = 1920, 1080, 24
DUR = 105.0
NFRAMES = int(round(DUR * FPS))
LEAD = 3 / FPS

_GRID = json.loads((Path(__file__).resolve().parent / "data" / "song_grid.json").read_text())
SONG_BARS = np.array(_GRID["bars"])
BAR = float(_GRID["bar_s"])
SONG_FILE = "build/song/theme.mp3"
SONG_SOURCE = "Official Opening Credits Game of Thrones (HBO) (1).mp3"


def snap(t):
    return float(SONG_BARS[np.argmin(np.abs(SONG_BARS - t))])


LAT0, LON0 = 13.945, 121.14

HOME_PIN = (13.948398, 121.115924)

HERO_IDS = {
    "home": 469001694,
    "sti": 470593026,
    "cathedral": 249138738,
    "sm": 31705509,
    "terminal": 409054176,
}

LAND = {
    "home": 22.468,
    "sti": 42.278,
    "cathedral": 60.325,
    "sm": 71.288,
}
FINAL_HIT = 93.576

_CUTS = [0.0, 3.39, 5.515, 8.696, 12.957, snap(18.2), 23.557, 26.738, 38.417, 43.712, snap(49.0), 55.403,
         61.765, 64.946, 68.127, 73.433, 77.682, 84.056, 89.385, 96.792, DUR]
_KINDS = [("sphere", "open"), ("sphere", "band"), ("sphere", "wide"), ("map", "region"), ("map", "descend"),
          ("map", "home"), ("map", "home_orbit"), ("map", "ride"), ("map", "sti"), ("map", "sti_orbit"),
          ("map", "to_cathedral"), ("map", "cathedral"), ("map", "cathedral_orbit"), ("map", "to_sm"),
          ("map", "sm"), ("map", "sm_orbit"), ("map", "pullback"), ("sphere", "align"), ("sphere", "finale"),
          ("card", "credits")]
SHOTS = [(a, b, m, k) for a, b, (m, k) in zip(_CUTS[:-1], _CUTS[1:], _KINDS)]

LEGS = [
    ("home", "sti", 26.738, 38.417),
    ("sti", "cathedral", snap(49.0), 55.403),
    ("cathedral", "sm", 64.946, 68.127),
]

TITLES = [
    (1.06, 3.39, "sphere", "LIPA", "BATANGAS  ·  PHILIPPINES"),
    (snap(9.7), 12.957, "map", "LIPA CITY", "387,392 PEOPLE  ·  2024 CENSUS"),
    (snap(14.0), snap(17.2), "map", "TAAL LAKE", "234.2 KM²  ·  5 M ABOVE SEA LEVEL"),
    (22.468, snap(25.7), "map", "HOME", "THE YELLOW HOUSE"),
    (42.278, snap(45.8), "map", "STI COLLEGE LIPA", "C. M. RECTO AVENUE  ·  EIGHT STOREYS"),
    (60.325, 64.0, "map", "LIPA CATHEDRAL", "SEAT OF THE ARCHDIOCESE OF LIPA"),
    (71.288, snap(74.5), "map", "SM CITY LIPA", "AYALA HIGHWAY  ·  SABANG"),
    (FINAL_HIT, 96.792, "sphere", "LIPA", "VILLA DE LIPA  ·  21 OCTOBER 1887"),
]

CREDITS = [
    "MUSIC  “GAME OF THRONES” MAIN TITLE THEME  ·  RAMIN DJAWADI  ·  HBO",
    "MAP DATA © OPENSTREETMAP CONTRIBUTORS  ·  ODbL 1.0",
    "ELEVATION  MAPZEN TERRARIUM  ·  SRTM / GMTED2010 COURTESY USGS",
    "TYPE  CINZEL  ·  SIL OPEN FONT LICENSE 1.1",
    "AN UNOFFICIAL FAN TRIBUTE  ·  NOT AFFILIATED WITH HBO",
]

RINGS = [
    "VILLA DE LIPA  ·  XXI OCTOBER MDCCCLXXXVII  ·  ",
    "METROPOLITAN CATHEDRAL OF SAN SEBASTIAN  ·  ARCHDIOCESE OF LIPA  ·  ",
    "BATANGAS  ·  ONE OF THE EIGHT RAYS  ·  MDCCCXCVI  ·  ",
]
