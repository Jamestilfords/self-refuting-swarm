# Unblind run order

Run from a clean checkout of published `main`, with the Modal CLI logged in to the account that holds `skeptic-sealed`.

```sh
uv run --frozen python -m skeptic.store unseal-check
uv run --frozen python -m modal_jobs.plain_ece
make unblind
git add runs/unblind.json runs/guard.log && git commit -m "Unblind" && git push
make fetch-sealed
make report
```

`plain_ece` writes `/sealed/plain_ece.json` on the volume and prints `{"status": "written", "cells": 4, "values": 40}`. `make unblind` hashes that volume and writes `runs/unblind.json` locally, with a count of 492 files in each of the four cells. `make fetch-sealed` downloads `scores/` and `plain_ece.json` into `./sealed`. `make report` checks every sha256 in `runs/unblind.json` before it writes `tables/`.
