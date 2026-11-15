"""Plan or replay R0/R1/R2 using explicit registered settings and open scores."""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import math
import re
import sys
import numpy as np

from skeptic.rules.common import CELLS, IGNORE_PARENTS, load_ledger, matched_orderings
from skeptic.rules.critic_bridge import ensure_er1_path, critic_factory
from skeptic.rules.engine import replay_cell, combine_cells, write_decision_logs
from skeptic.rules.thresholdout import laplace_noise, split_aurocs, stratified_split
from skeptic.score_artifacts import load_scores, auroc
from skeptic.store import _check_registration


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_score_manifest(path, expected_paths):
    """Require one registered hash for every scored artifact and no extras."""
    manifest = Path(path)
    if not manifest.is_file() or manifest.is_symlink():
        raise ValueError("Replay requires runs/score_manifest.sha256 before the first decision file")
    hashes = {}
    for number, line in enumerate(manifest.read_text(encoding="ascii").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (runs/scores(?:_isolated)?/[a-z0-9_]+/[0-9a-f]{12}\.npz)", line)
        if match is None or match[2] in hashes:
            raise ValueError(f"Invalid score manifest row {number}")
        hashes[match[2]] = match[1]
    if set(hashes) != expected_paths:
        raise ValueError("Score manifest does not match the registered ledger and selected score source")
    return hashes


def replay_rows(ledger_path, baselines_path):
    """Exclude named baseline identities even when discovery tagged them templated."""
    rows = load_ledger(ledger_path)
    manifest = json.loads(Path(baselines_path).read_text())
    named = manifest.get("proposals") if isinstance(manifest, dict) else manifest
    if not isinstance(named, list) or not named or any(
        not isinstance(item, dict) or not isinstance(item.get("sha"), str) for item in named
    ):
        raise ValueError("Invalid baseline manifest")
    baseline_shas = {item["sha"] for item in named}
    for sha in baseline_shas:
        for cell in CELLS:
            row = rows.get((sha, cell))
            if row is None or row.get("status") != "ok":
                raise ValueError(f"Missing scored baseline: {cell}/{sha}")
            rows[(sha, cell)] = {**row, "stream": "baseline"}
    return rows


def read_config(path, rows):
    from skeptic.critic.config import CriticConfig
    def invalid(value):
        raise ValueError(f"Non-finite configuration value: {value}")
    config = json.loads(Path(path).read_text(), parse_constant=invalid)
    required = {"orderings", "ordering_seed", "initial", "r1_threshold", "r1_split_seed",
                "r1_noise", "critic", "ignore_parents"}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("Replay configuration must explicitly supply every required field")
    for name in ("orderings", "ordering_seed", "r1_split_seed"):
        if type(config[name]) is not int or config[name] < (1 if name == "orderings" else 0):
            raise ValueError(f"Invalid {name}")
    if type(config["r1_threshold"]) not in (int, float) or not math.isfinite(config["r1_threshold"]) or config["r1_threshold"] < 0:
        raise ValueError("R1 needs a finite nonnegative threshold")
    if config["ignore_parents"] != sorted(IGNORE_PARENTS):
        raise ValueError("ignore_parents must be the three registered coupling shas")
    if not isinstance(config["initial"], dict) or set(config["initial"]) != set(CELLS):
        raise ValueError("initial must cover all four cells")
    for cell in CELLS:
        initial = rows.get((config["initial"][cell], cell), {})
        if initial.get("stream") != "baseline" or initial.get("status") != "ok":
            raise ValueError("Initial incumbent must be a scored baseline for each cell")
        matched_orderings(rows, cell, 1, config["ordering_seed"])
    noise = config["r1_noise"]
    if not isinstance(noise, dict) or set(noise) != {"distribution", "sigma", "seed", "algorithm"} \
            or noise["distribution"] != "laplace" \
            or noise["algorithm"] != "numpy-pcg64-seedsequence-v1" \
            or type(noise["sigma"]) not in (int, float) or not math.isfinite(noise["sigma"]) \
            or noise["sigma"] <= 0 or type(noise["seed"]) is not int or noise["seed"] < 0:
        raise ValueError("R1 needs a registered finite Laplace sigma, seed, and algorithm")
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
    rows = replay_rows(args.ledger, args.baselines)
    config, critic_config = read_config(args.config, rows)
    summary = {"rules": ["R0", "R1", "R2"], "orderings": config["orderings"],
               "decision_files": 3 * config["orderings"],
               "proposals_per_cell": {c: len(matched_orderings(rows, c, 1, config["ordering_seed"])[0]) for c in CELLS}}
    if args.command == "plan":
        print(json.dumps(summary, sort_keys=True))
        return 0
    _check_registration(args.output)
    folder = "scores_isolated" if args.source == "isolated" else "scores"
    expected_scores = {f"runs/{folder}/{CELLS[cell]}/{sha}.npz"
                       for (sha, cell), row in rows.items() if row["status"] == "ok"}
    manifest_path = args.output / "runs" / "score_manifest.sha256"
    registered_scores = read_score_manifest(manifest_path, expected_scores)
    ensure_er1_path()
    from skeptic.critic_inputs import load_cell
    artifacts = {}
    hashes = {}
    measured_aurocs = {cell: {} for cell in CELLS}
    for (sha, cell), row in rows.items():
        if row["status"] != "ok":
            continue
        artifact = load_scores(args.open_root, cell, sha, source=args.source)
        measured = [auroc(s[:artifact.n_in], s[artifact.n_in:]) for s in artifact.scores]
        if not np.allclose(row["val"]["auroc"], measured, rtol=0, atol=1e-6):
            raise ValueError(f"Ledger AUROC and score artifact differ: {cell}/{sha}")
        artifacts[(sha, cell)] = artifact
        measured_aurocs[cell][sha] = measured
        relative = f"runs/{folder}/{CELLS[cell]}/{sha}.npz"
        score_hash = _hash(args.open_root / relative.removeprefix("runs/"))
        if score_hash != registered_scores[relative]:
            raise ValueError(f"Score artifact differs from registered manifest: {relative}")
        hashes[relative] = score_hash
    cells = {cell: load_cell(cell, args.open_root / folder, args.ledger, args.baselines) for cell in CELLS}
    factory = critic_factory(cells, critic_config)
    r1_split_scores = {}
    for cell in CELLS:
        initial = artifacts[(config["initial"][cell], cell)]
        split = stratified_split(initial.n_in, initial.scores.shape[1], config["r1_split_seed"])
        r1_split_scores[cell] = {sha: split_aurocs(artifact, split)
                                 for (sha, artifact_cell), artifact in artifacts.items() if artifact_cell == cell}
    orderings = {cell: matched_orderings(rows, cell, config["orderings"], config["ordering_seed"])
                 for cell in CELLS}

    def decision_groups():
        for ordering in range(config["orderings"]):
            print(f"Replaying ordering {ordering + 1}/{config['orderings']}", file=sys.stderr, flush=True)
            replays = [replay_cell(rows, cell, initial_sha=config["initial"][cell],
                                  ordering_count=config["orderings"], ordering_seed=config["ordering_seed"],
                                  score_loader=lambda c, sha: artifacts[(sha, c)],
                                  r1_threshold=config["r1_threshold"], r1_split_seed=config["r1_split_seed"],
                                  noise_factory=lambda c, order: laplace_noise(c, order,
                                      sigma=config["r1_noise"]["sigma"], seed=config["r1_noise"]["seed"]),
                                  critic_factory=factory, validated_aurocs=measured_aurocs[cell],
                                  r1_split_scores=r1_split_scores[cell],
                                  selected_orderings=[(ordering, orderings[cell][ordering])])
                       for cell in CELLS]
            combined = combine_cells(*replays)
            for rule in ("R0", "R1", "R2"):
                yield (rule, ordering), combined[(rule, ordering)]

    metadata = {"config": config, "config_sha256": _hash(args.config),
                "ledger_sha256": _hash(args.ledger), "baselines_sha256": _hash(args.baselines),
                "score_manifest_sha256": _hash(manifest_path),
                "source": args.source, "score_sha256": hashes}
    paths = write_decision_logs(args.output, decision_groups(),
                                ordering_count=config["orderings"], metadata=metadata)
    print(json.dumps({**summary, "written": paths}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
