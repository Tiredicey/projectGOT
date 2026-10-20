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
- DONE: v3 shipped (film/CHUNKS.md); M1 table (film/records/m1_table.md); kanal QC (film/records/kanal_qc.json);
  film/machine.py (mesh kit), film/glsl.py (sphere bend + central sun), film/timeline4.py (bar-locked v4 shots),
  film/chamber.py (automaton chamber: iris shutter, frame, gear train, planetary set, telescopes; 0.37 s/frame).
  Bend wired into every shader; bend off is pixel-identical to v3 (checked frames 372 and 960).
- OPEN: render4.py (shot cameras for hollow/descend4/pullback4/hollow_out using the bend; astrolabe sun at the
  sphere centre; chamber shots; landmark detail models), stills review, chunked render, mux, measure, README.
- BLOCKED: none.
- NEXT: write film/render4.py; probes in film/tools/ (bend_probe.py, chamber_probe.py) show the current look.
