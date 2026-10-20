# Render checkpoints (survive sandbox resets)

Finished chunks are uploaded as they complete. After a reset: run `bash film/bootstrap.sh`,
download any listed chunk into `film/build/chunks/`, write `120` into `<chunk>.done`, then run
`bash film/render_chunks.sh` (it skips chunks that have a .done file).

## v3 (shipped)
- Master with song: https://www.genspark.ai/api/files/s/g2KSD2BC
- Video only: https://www.genspark.ai/api/files/s/RVYDLR1p

## v4 (in progress)
(none yet)
