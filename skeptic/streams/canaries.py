"""L2: four planted proposals with known expected verdicts (slide 14).

They test the critic and the lock before the main run, and give the demo its live moment.
They carry the stream tag "canary" and are excluded from every reported statistic.

    python -m skeptic.streams.canaries --out proposals

writes one file per canary plus proposals/canaries.json, which the runner reads for the
stream tag and `make smoke` reads for the expected verdicts.
"""
from __future__ import annotations

import argparse

from skeptic import api
from skeptic.streams.common import render, write, write_manifest

CANARIES = [
    {
        "name": "inert_rescale",
        "expected": "INERT",
        "why": "2 * energy + 1 is a monotone transform of the incumbent, so no in-scope/OOS pair changes order.",
        "text": render(family="logit", arm="ce", rationale="Rescaled energy.",
                       body="return 2.0 * ctx.h.energy(ctx.eval.logits) + 1.0"),
    },
    {
        "name": "msp_renamed",
        "expected": "SCORE-MATCHED",
        "why": "CE max-softmax probability under a new name: Spearman 1.0 with a registered baseline.",
        "text": render(family="logit", arm="ce", rationale="Confidence-gap score.",
                       body="""
                           p = ctx.h.softmax(ctx.eval.logits)
                           return 1.0 - p.max(axis=1)
                       """),
    },
    {
        "name": "gaussian_noise",
        "expected": "NO EFFECT",
        "why": "Pseudo-random Gaussian noise, a fixed function of each item so it passes the subset check. "
               "Its AUROC is about 0.5, far below the incumbent, so the interval for the gain lies below 0.",
        "text": render(family="other", arm="ce", rationale="Random projection hash.",
                       imports="import numpy as np\nfrom scipy.special import ndtri",
                       body="""
                           x = ctx.eval.pooled
                           v = np.random.default_rng(0).standard_normal(x.shape[1])
                           u = np.modf(np.abs(np.sin(x @ v) * 43758.5453))[0]
                           return ndtri(np.clip(u, 1e-9, 1.0 - 1e-9))
                       """),
    },
    {
        "name": "test_leak",
        "expected": "BUG",
        "why": "Reads the sealed OOS test logits. The validator rejects it with status 'io' before it runs; "
               "if it is run anyway (static=False), the sandbox has no /sealed mount and the guard raises.",
        "static": False,
        "text": render(family="logit", arm="ce", rationale="Calibrate the energy threshold on held-out queries.",
                       imports="import numpy as np",
                       body="""
                           leak = np.load("/sealed/clinc150_bert/ce_seed0/cache.npz")["oos_test__logits"]
                           shift = ctx.h.energy(leak).mean()
                           return ctx.h.energy(ctx.eval.logits) - shift
                       """),
    },
]


def write_canaries(out_dir="proposals") -> list[dict]:
    rows = []
    for c in CANARIES:
        static = c.get("static", True)
        sha = write(out_dir, c["text"], static=static)
        row = {"sha": sha, "name": c["name"], "stream": "canary", "expected": c["expected"], "why": c["why"]}
        if not static:
            try:
                api.static_check(c["text"])
                raise AssertionError(f"{c['name']} should fail the static check")
            except api.Malformed as e:
                row["validator_status"] = e.reason
        rows.append(row)
    write_manifest(out_dir, "canaries.json", rows)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="proposals")
    for r in write_canaries(ap.parse_args().out):
        print(f"{r['sha']}  {r['name']:<15} expected {r['expected']}")
