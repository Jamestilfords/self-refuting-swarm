"""Plan or replay R0/R1/R2 using explicit registered settings and open scores."""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import math
import numpy as np

from skeptic.rules.common import CELLS, load_ledger, matched_orderings
from skeptic.rules.critic_bridge import ensure_er1_path, critic_factory
from skeptic.rules.engine import replay_cell, combine_cells, write_decision_logs
from skeptic.score_artifacts import load_scores, auroc
from skeptic.store import _check_registration


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_config(path, rows):
    from skeptic.critic.config import CriticConfig
    def invalid(value):
        raise ValueError(f"Non-finite configuration value: {value}")
    config = json.loads(Path(path).read_text(), parse_constant=invalid)
    required = {"orderings", "ordering_seed", "initial", "r1_threshold", "r1_split_seed", "r1_noise", "critic"}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("Replay configuration must explicitly supply every required field")
    for name in ("orderings", "ordering_seed", "r1_split_seed"):
        if type(config[name]) is not int or config[name] < (1 if name == "orderings" else 0):
            raise ValueError(f"Invalid {name}")
    if type(config["r1_threshold"]) not in (int, float) or not math.isfinite(config["r1_threshold"]) or config["r1_threshold"] < 0:
        raise ValueError("R1 needs a finite nonnegative threshold")
    for name in ("initial", "r1_noise"):
        if not isinstance(config[name], dict) or set(config[name]) != set(CELLS):
            raise ValueError(f"{name} must cover all four cells")
    for cell in CELLS:
        initial = rows.get((config["initial"][cell], cell), {})
        if initial.get("stream") != "baseline" or initial.get("status") != "ok":
            raise ValueError("Initial incumbent must be a scored baseline for each cell")
        n = len(matched_orderings(rows, cell, 1, config["ordering_seed"])[0])
        noise = config["r1_noise"][cell]
        if not isinstance(noise, list) or len(noise) != config["orderings"]:
            raise ValueError("Noise schedule must cover every ordering")
        for ordering in noise:
            if not isinstance(ordering, list) or len(ordering) != n:
                raise ValueError("Noise schedule must cover every proposal position")
            if any(not isinstance(pair, list) or len(pair) != 2 or
                   any(type(x) not in (int, float) or not math.isfinite(x) for x in pair) for pair in ordering):
                raise ValueError("Each noise pair must contain two finite numbers")
    critic = config["critic"]
    if not isinstance(critic, dict) or set(critic) != {f.name for f in fields(CriticConfig)}:
        raise ValueError("Every critic setting must be explicit; proposed defaults are not registered")
    for name in ("bootstrap_B", "g5_bins", "g6_angles"):
        if type(critic[name]) is not int or critic[name] < 1:
            raise ValueError(f"Invalid critic setting: {name}")
    if type(critic["bootstrap_seed"]) is not int or critic["bootstrap_seed"] < 0:
        raise ValueError("Invalid bootstrap seed")
    for name in ("g1_flip_thresh", "g2_rho_thresh", "fdr_alpha", "fdr_w0", "g4_band", "mdd_power", "g5_ece_tol", "g6_tol"):
        if type(critic[name]) not in (int, float) or not math.isfinite(critic[name]) or not 0 <= critic[name] <= 1:
            raise ValueError(f"Invalid critic setting: {name}")
    if not 0 < critic["fdr_w0"] < critic["fdr_alpha"] < 1 or not 0 < critic["mdd_power"] < 1:
        raise ValueError("Invalid LORD++ or power settings")
    if type(critic["g2_weak_incumbent_check"]) is not bool or critic["seed_majority"] != "ceil_half":
        raise ValueError("Invalid critic majority/check settings")
    return config, CriticConfig(**critic)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--open-root", required=True, type=Path,
                        help="local root containing scores/ or scores_isolated/")
    parser.add_argument("--source", required=True, choices=("legacy", "isolated"))
    parser.add_argument("--baselines", type=Path, default=Path("proposals/baselines.json"))
    parser.add_argument("--output", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    rows = load_ledger(args.ledger)
    config, critic_config = read_config(args.config, rows)
    summary = {"rules": ["R0", "R1", "R2"], "orderings": config["orderings"],
               "decision_files": 3 * config["orderings"],
               "proposals_per_cell": {c: len(matched_orderings(rows, c, 1, config["ordering_seed"])[0]) for c in CELLS}}
    if args.command == "plan":
        print(json.dumps(summary, sort_keys=True))
        return 0
    _check_registration(args.output)
    ensure_er1_path()
    from skeptic.critic_inputs import load_cell
    folder = "scores_isolated" if args.source == "isolated" else "scores"
    artifacts = {}
    hashes = {}
    for (sha, cell), row in rows.items():
        if row["status"] != "ok":
            continue
        artifact = load_scores(args.open_root, cell, sha, source=args.source)
        measured = [auroc(s[:artifact.n_in], s[artifact.n_in:]) for s in artifact.scores]
        if not np.allclose(row["val"]["auroc"], measured, rtol=0, atol=1e-6):
            raise ValueError(f"Ledger AUROC and score artifact differ: {cell}/{sha}")
        artifacts[(sha, cell)] = artifact
        relative = f"{folder}/{CELLS[cell]}/{sha}.npz"
        hashes[relative] = _hash(args.open_root / relative)
    cells = {cell: load_cell(cell, args.open_root / folder, args.ledger, args.baselines) for cell in CELLS}
    factory = critic_factory(cells, critic_config)
    replays = [replay_cell(rows, cell, initial_sha=config["initial"][cell],
                          ordering_count=config["orderings"], ordering_seed=config["ordering_seed"],
                          score_loader=lambda c, sha: artifacts[(sha, c)],
                          r1_threshold=config["r1_threshold"], r1_split_seed=config["r1_split_seed"],
                          noise_factory=lambda c, order: lambda position: tuple(config["r1_noise"][c][order][position]),
                          critic_factory=factory) for cell in CELLS]
    metadata = {"config": config, "config_sha256": _hash(args.config),
                "ledger_sha256": _hash(args.ledger), "baselines_sha256": _hash(args.baselines),
                "source": args.source, "score_sha256": hashes}
    paths = write_decision_logs(args.output, combine_cells(*replays),
                                ordering_count=config["orderings"], metadata=metadata)
    print(json.dumps({**summary, "written": paths}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
