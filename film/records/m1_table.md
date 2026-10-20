| File | Res / fps | Dur s | Hard cuts (scene>0.3) | Near-still | Mean frame diff | Max flashes/s | LUFS / dBTP / LRA | Tempo |
|---|---|---|---|---|---|---|---|---|
| Game-of-Thrones-SnapYT.App.mp4 | 1920x1078 / 23.976 | 110.43 | 16 (8.69/min, mean shot 6.5 s) | 4.2 % | 9.048 | 4 | -26.1 / -11.0 / 2.6 | 172.27 |
| Opening-Credits-SnapYT.App.mp4 | 1920x1080 / 23.976 | 117.10 | 4 (2.05/min, mean shot 23.4 s) | 13.7 % | 6.489 | 0 | -14.3 / -2.0 / 2.9 | 172.27 |
| Official-Opening-Credits-SnapYT.App.mp4 | 1280x720 / 29.970 | 101.70 | 9 (5.31/min, mean shot 10.2 s) | 9.2 % | 5.822 | 2 | -15.4 / -2.6 / 3.3 | 172.27 |
| Our v3 draft | 1920x1080 / 24.000 | 105.00 | 6 (3.43/min, mean shot 15.0 s) | 0.0 % | 9.834 | 0 | -15.7 / -2.8 / 3.3 | 172.27 |

Notes
- Measured 2026-10-20 with `film/measure.py` on the uploaded files (scene > 0.3, ebur128, librosa).
- Which file is which YouTube link is NOT confirmed: the uploads carry no video id. Only
  `Official-Opening-Credits-SnapYT.App.mp4` matches by sha256 (942d1ee5…) the copy measured in v2 for s7L2PVdrb_8.
- `Game-of-Thrones-SnapYT.App.mp4` exceeds the WCAG 2.3.1 general-flash limit on this detector:
  up to 4 flashes in one second around 15.6-16.0 s, and 3 around 57.4-57.7 s (luminance-only check; red-flash not tested).
- All three references and our draft share the same tempo reading (172.27 bpm), consistent with one theme recording.
- `kanal_qc.json`: the six uploaded clips, 240 frames each at 24 fps, 0 scene cuts, 0 flashes.
