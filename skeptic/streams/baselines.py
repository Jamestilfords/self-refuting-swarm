"""Registered baselines as proposal files, so the runner scores them like any other proposal.

Gate G2 compares every proposal with these, and CE energy is the starting incumbent (slide 12).

    python -m skeptic.streams.baselines --out proposals

writes the files plus proposals/baselines.json, which maps each name (e.g. "energy@ce") to its hash.
"""
from __future__ import annotations

import argparse

from skeptic.streams.common import write, write_manifest
from skeptic.streams.templated import Config, render_config, render_logit

BASELINES = {
    "energy@ce": render_logit("energy", "ce"),
    "msp@ce": render_logit("msp", "ce"),
    "energy@edl": render_logit("energy", "edl"),
    "msp@edl": render_logit("msp", "edl"),
    "vacuity@edl": render_logit("vacuity", "edl"),
    "d_port@ce": render_config(Config("feats", 128, "gda", "ce")),
    "d_port@edl": render_config(Config("feats", 128, "gda", "edl")),
    "d_full@ce": render_config(Config("pooled", None, "gda", "ce")),
    "d_full@edl": render_config(Config("pooled", None, "gda", "edl")),
}


def write_baselines(out_dir="proposals") -> list[dict]:
    rows = [{"name": name, "sha": write(out_dir, text), "stream": "baseline"} for name, text in BASELINES.items()]
    write_manifest(out_dir, "baselines.json", rows)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="proposals")
    for r in write_baselines(ap.parse_args().out):
        print(f"{r['sha']}  {r['name']}")
