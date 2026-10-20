# HANDOFF (read this first after any interruption)

Repo: https://github.com/Tiredicey/projectGOT (branch main). Genspark git keeps a mirror.

## Restore in one command
    bash film/bootstrap.sh
Needs these uploads in /home/user/uploaded_files: the GoT MP3, the 3 reference MP4s, kanal_01..06.mp4.
Copyrighted media is never committed (see .gitignore).

## State
- GOAL: v4 film, 1:45 at 1920x1080/24: brass gear-train machines, hollow-sphere map around an astrolabe sun,
  automaton chambers with the six kanal clips, Lipa landmarks rising with gear trains, supplied song bit-exact.
- DONE: all v4 code (machine.py, glsl.py, chamber.py, timeline4.py, render4.py); every one of the 22 shots
  renders; v3 frames unchanged with the bend off.
- RUNNING: `bash film/render_chunks4.sh` (21 chunks x 120 frames). Each finished chunk is uploaded and its URL
  is committed to film/records/chunks4.tsv, so a reset loses at most one chunk.
- AFTER A RESET: `bash film/bootstrap.sh` then
  `setsid nohup bash film/render_chunks4.sh > film/build/render4.log 2>&1 < /dev/null &`
  (it downloads finished chunks from the ledger and renders only the missing ones).
- THEN: `bash film/mux4.sh` (bit-exact song, MD5 check, measure, poster, contact sheet), README, commit.
