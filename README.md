# webapp / film: Lipa map-sequence (v3: song, trees, water, colour)

A 1920x1080, 24 fps, 1:45 fan tribute in the style of the *Game of Thrones* map title
sequence. The camera flies over a 3D painted model of Lipa, Batangas, built from real
OpenStreetMap data and real elevation. It sets to your supplied
**"Game of Thrones" Main Title Theme (Ramin Djawadi, HBO)**, used bit-exact.

What is in v3:

- **The supplied song.** `Official Opening Credits Game of Thrones (HBO) (1).mp3`
  (MP3 192 kbps, 44.1 kHz, 101.75 s) is muxed with `-c:a copy`, and `make.sh` fails if the audio
  packet MD5 of the output differs from the source. The tracker measures the beat at
  0.3483 s (172.27 bpm); bars are grouped in 3s at that beat (1.0565 s bars, 91 bars,
  `data/song_grid.json`). Every cut sits on a measured bar line. Each landmark rise
  lands on the song's strongest nearby accent: home 22.468 s, STI 42.278 s,
  Cathedral 60.325 s, SM 71.288 s, final sun 93.576 s.
- **Trees, about 907,000 of them, from OSM.** Placed only inside OSM `natural=wood`, `landuse=forest`,
  `landuse=orchard`, `natural=scrub` and `natural=tree_row` (`data/osm_green.json`). Species
  follow the OSM tags: `trees=coconut_palms` gives palms, `banana_plants` gives bananas,
  `mango_trees` gives broadleaf in rows. Placement counts: 795,149 broadleaf, 56,015 palm,
  8,540 banana, 47,186 shrub. They are instanced in 600 m tiles with near and far detail
  levels, a per-frame vertex budget, wind sway, shadow-casting, and per-tree colour variation.
- **Water: Taal Lake (OSM relation 270582), 55.8 km² of it inside the model.** The lake bed is
  carved into the terrain mesh. The surface carries five animated waves filtered by pixel
  footprint (no shimmer), Fresnel sky reflection, a sun glint that widens with distance,
  shore foam, and a deep-to-shallow colour ramp. The surface sits at 5 m above sea level
  (Wikipedia "Taal Lake" rev 1378554203). The DEM inside the OSM polygon has a
  median of 5.0 m, which agrees.
- **Colour and texture on the map.** A paper wash in landcover colours over a parchment
  grain: woods, orchards in planting rows, farmland striped along each field's long
  axis, grass, scrub, and tree rows as stippled crowns. Hypsometric tint and
  relief shading come from the DEM. Rivers are blue, and roads are coloured by class.
- **Colour and texture on the buildings.** Pastel wall palette, window grids per 3.2 m floor
  and 3 m bay, ledges, base grime, corrugated roofs with specular, and OSM
  `roof:colour` / `building:colour` where tagged (320 of 2,499 buildings). Heroes: home yellow
  with a red roof, STI blue and gold, SM white and blue, the Cathedral cream with the OSM
  roof colour `#c76767`.
- **Multi-sample smoothing.** 3 jittered sub-samples per frame (Halton 2,3) over a 180°
  shutter, so every map frame gets real motion blur and extra anti-aliasing on top of 4x MSAA.

## Build

```bash
cd film
bash make.sh   # cache -> chunked adaptive 3-12 spp render -> mux supplied song bit-exact -> verify MD5 -> measure -> poster/contact sheet
```

Needs Python 3.11+ with numpy, scipy, opencv-python, pillow, shapely>=2.1, librosa, networkx and moderngl,
plus ffmpeg/ffprobe and an EGL driver (Debian: `apt install libegl1 libegl-mesa0 libgl1-mesa-dri`).
It renders in 120-frame chunks, one process each (peak about 640 MB), at about 0.6–0.9 s per
sub-sample on 2 vCPU of software GL.

| File | Role |
|---|---|
| `film/timeline.py` | Measured bar grid from the song, shot list snapped to bars, hit times, titles, credits |
| `film/geo.py` | OSM -> map layers (region 8 m/px, corridor 2 m/px, city 1.1 m/px, home 0.9 m/px), DEM terrain, building footprints |
| `film/landcover.py` | OSM landcover/water polygons -> painted washes, stipple, water ramp, coloured roads; tree instance placement |
| `film/nature.py` | Instanced tree meshes (broadleaf, coconut palm, banana, shrub; near/far), tile culling, Taal Lake water shader |
| `film/gl3d.py` | OpenGL scene: terrain + lake carve, textured buildings, gears, route ribbon, slab, shadows, fog, jitter |
| `film/route.py` | Shortest drive on the OSM road graph between the four stops; smoothed camera rails |
| `film/render.py` | Cameras, multi-sample accumulation, titles, film grade; pipes frames to libx264 |
| `film/measure.py` | ffprobe, scene cuts, frame-diff, flash count, EBU R128, audio MD5, tempo |
| `film/data/` | Overpass queries and results (OSM base 2026-10-18), DEM tiles, `song_grid.json` |

## Measured results on the muxed file (v3, supplied song)

File `film/out/lipa_map_sequence_1080p24.mp4`, sha256 `078150c0…3052e`, 375 MB.
Download: https://www.genspark.ai/api/files/s/leymPc1k

| Metric | Value | Check |
|---|---|---|
| Container | H.264 yuv420p 1920x1080 24/1, 2,520 frames, 105.000 s; MP3 192 kbps 44.1 kHz stereo, 101.72 s | ffprobe PASS |
| Song bit-exact | Packet MD5 `790de971…` in source and in output; decoded PCM identical for all 4,485,120 source samples (offset 0) | PASS |
| Loudness (as supplied, not normalised) | -15.7 LUFS integrated, -2.8 dBTP, LRA 3.3 LU | Untouched on purpose; 1.7 LU under the -14 web convention |
| General flashes | 0 per second at most | WCAG 2.3.1 limit is 3, PASS |
| Near-still frames | 0.0 % | PASS |
| Tempo | 172.27 bpm (librosa), same as `song_grid.json` | PASS |
| Hero-hit sync (hop 128, strong onsets) | home +8.5 ms, STI +6.1 ms, SM +5.6 ms, final hit +8.3 ms | 45 ms target, PASS |
| Cathedral landing (60.325 s) | Nearest onset 60.308 s (+17 ms) is at threshold strength (0.49 = p75); the song has no strong accent there | PASS on the full onset list, noted |
| Cuts on strong onsets | 11 of 19 within 45 ms; median 34.8 ms | The others are bar-grid cuts on weaker beats |

Notes: the decoded MD5 of the muxed file (`cb2e0368…`) differs from the source (`23c6e04a…`) only
because the MP4 decode adds 815 trailing samples (18 ms) of encoder padding after the song ends; the
first 4,485,120 samples match exactly. The audio ends at 101.72 s; the last 3.3 s are silent credits.
Full records: `film/out/measure.json`, `film/out/sync.json` (committed).

## The reference (M1), measured on your uploaded copy

`Official-Opening-Credits-SnapYT.App.mp4`: 1280x720, 29.97 fps, 101.70 s,
sha256 942d1ee5… The reference itself is not committed (copyrighted, K49).

| Metric | Reference | This film | Better? |
|---|---|---|---|
| Resolution / fps | 1280x720 / 29.97 | 1920x1080 / 24 | sharper; 24 fps matches the brief |
| Duration | 101.70 s | 105.00 s | matches the 1:45 brief |
| Hard cuts (scene > 0.3) | 9 (5.31/min) | 4 (2.29/min) | style choice: both are continuous-camera films |
| Near-still frames | 9.2 % | 0.0 % | yes |
| Max general flashes in 1 s | 2 | 0 | yes, safer |
| Loudness | -15.4 LUFS, -2.6 dBTP, LRA 3.3 | -14.1 LUFS, -3.7 dBTP, LRA 10.0 | on the -14 web convention |
| Tempo (librosa) | 172.3 bpm (likely double-time of ~86) | score written at 80 bpm, 6/8 | not verified: the octave error is my reading |

A model's earlier description of the YouTube video ("36 shots") was wrong: on the real
file the detector finds 9 hard cuts. Measurement wins (K43).

## Adopted from earthwalker17/map-of-middle-earth

Read-only reference: https://github.com/earthwalker17/map-of-middle-earth, commit
`696e968e9d55` (2026-10-06), MIT, © 2026 earthwalker17. No code is copied. Their
three.js/WebGPU stack does not fit here, so the ideas were re-implemented in Python + OpenGL 3.3:

| Their idea (docs/ARCHITECTURE.md) | Here, and what changed |
|---|---|
| Gold journey line revealed by arc length, dark under-stroke, additive core, head comet, shimmer | `route.py` + `gl3d.ROUTE_*`. **Better for this film:** the line follows the real shortest drive on the OSM road graph (networkx/Dijkstra), not hand-placed legs. Home→STI 6.18 km, STI→Cathedral 0.44 km, Cathedral→SM 2.46 km (`build/route_report.json`). Width is set in screen space so it stays about 2–4 px at any height. |
| Gaussian-smoothed camera rails along the route | `Route.rail()`: 300 m Gaussian on position, 650 m on heading. The ride shots now fly the actual road. |
| Floating diorama slab with strata cut faces and a plinth | `gl3d.slab_mesh()`: walls follow the terrain edge profile, procedural strata bands and a dark plinth. The establishing and closing shots pull back far enough to show the map as a physical object. |
| Film grade: split-tone, halation keyed on highlights, vignette, seeded grain with a 5-tap grain shape, mid-tone-weighted | `render.grade()`, the same chain on the CPU per frame |
| Every frame a pure function of t; chunked, resumable renders with frame-count verification | `render_chunks.sh`: 240-frame chunks, each in a fresh process, verified by frame count and joined with `-c copy`. This keeps memory flat on the 1 GB sandbox, which froze twice during single-process renders. |

Not adopted: WebGPU/TSL, vegetation, water, multi-sample accumulation and DOF (too slow on
software GL), and their score pipeline (this film uses your supplied song instead).
v3 adds a lighter version of what was skipped in v2: instanced trees, a water surface,
and 3-sample jittered accumulation with shutter blur. Depth of field is still not used.

## Facts used on screen, with sources

- Lipa population 387,392 (2024 census): Wikipedia "Lipa, Batangas", rev 1376144090 (2026-09-22).
- Villa de Lipa decree, 21 October 1887: same article.
- Lipa Cathedral, seat of the Archdiocese of Lipa (Metropolitan Cathedral of San Sebastian): Wikipedia rev 1368306006; OSM way 249138738.
- STI Academic Center Lipa, C. M. Recto Ave, eight storeys: sti.edu news article "STI ESG Inaugurates New Campus in Lipa"; OSM way 470593026 has `building:levels=8`.
- SM City Lipa, Ayala Highway, Sabang: OSM way 31705509.
- Batangas as one of the eight rays of the sun: Wikipedia "Flag of the Philippines" rev 1380401013. RA 8491 §4 confirms only the "eight-rayed golden-yellow sun"; the statute does not list the provinces.

## Not confirmed

- **Which building is your house.** Your Maps link resolves to 13.948398, 121.115924 (Heritage Village, Bagong Pook). I picked OSM way 469001694, the footprint 9 m from that pin, and painted it yellow. OSM records no colour or height for it, so the height (7 m) is assumed.
- Building heights: 47 come from OSM `height`, 125 from `building:levels` x 3.2 m, and 2,252 are assumed (3.2 to 7.4 m).
- STI's footprint comes from OSM way 470593026 (8 levels, 29.8 m), which overlaps the campus polygon 806643724.

## Licences

Map data © OpenStreetMap contributors, ODbL 1.0. Elevation: Mapzen Terrarium tiles
(AWS Open Data); SRTM and GMTED2010 data courtesy of the U.S. Geological Survey, ETOPO1 by NOAA.
Which of these sources covers Lipa was not checked, so all three are credited. Cinzel font, SIL OFL 1.1 (`film/fonts/OFL.txt`).
Music: "Game of Thrones" Main Title Theme by Ramin Djawadi, © HBO. It is used as you supplied it,
for a personal, unofficial fan tribute; you are responsible for having the rights for any
public posting (platforms may claim or mute it). The code and design are original. The film is a tribute "in the manner of" a
fantasy-map title sequence: apart from the supplied song, it contains no HBO logos, characters or footage.

## Status

- Done: v3 master with the supplied song (bit-exact), poster, contact sheet, measurement and sync records.
  Re-mux without re-rendering: `SONG_SRC=/path/to/song.mp3 bash film/mux.sh`.
- Not done: 9:16 and 1:1 cutdowns, a named human reviewer, confirmation of which building is the house.
- Last updated: 2026-10-18
