# HANDOFF (read this first after any interruption)

Repo: https://github.com/Tiredicey/projectGOT (branch main). Genspark git keeps a mirror.

## Restore in one command
    bash film/bootstrap.sh
Needs these uploads in /home/user/uploaded_files: the GoT MP3, the 3 reference MP4s, kanal_01..06.mp4.
Copyrighted media is never committed (see .gitignore).

## State
- GOAL: v4 film, 1:42-1:50, 1920x1080: brass gear-train machines, hollow-sphere map around an astrolabe sun,
  one continuous macro dive into machine chambers, 1:1 Lipa landmarks, kanal-humour automaton panels,
  supplied song bit-exact.
- DONE: v3 shipped (see film/CHUNKS.md); M1 table for 3 refs + v3 (film/records/m1_table.md);
  kanal clips QC'd (film/records/kanal_qc.json); landmark geometry checked against OSM.
- OPEN: v4 renderer (sphere warp, gear trains, chambers, landmark models, panels), render, mux, measure.
- BLOCKED: none.
- NEXT: implement film/machine.py (gear trains + chambers) and the sphere warp in gl3d.py, render stills.
