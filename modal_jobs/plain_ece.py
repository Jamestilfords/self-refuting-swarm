"""Plain-model test ECE for the two arms, after the decision freeze is published.

The sealer stores scores and does not store a proposal's probabilities.
Final-pick ECE for a proposal that would have returned probabilities uses the
arm mean written here. The definition matches validation ``ece_plain``:
softmax ECE for CE, Dirichlet ECE for EDL, 15 bins, in-scope labels only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def plain_model_ece(logits, labels, arm: str, bins: int = 15) -> float:
    """In-scope ECE of the untouched model. Same function the ledger stores.

    ``softmax_ece`` and ``dirichlet_ece`` use the ER-1 15-bin ECE. ``bins`` is
    accepted so a caller cannot silently switch definitions.
    """
    from skeptic.critic.er1 import dirichlet_ece, logits_to_alpha, softmax_ece
    import torch

    if bins != 15:
        raise ValueError("plain-model ECE uses the registered 15 bins")
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    if arm == "ce":
        return float(softmax_ece(logits, labels))
    if arm == "edl":
        alpha = logits_to_alpha(torch.from_numpy(logits), 15.0, 1e-6)
        return float(dirichlet_ece(alpha.numpy(), labels))
    raise ValueError(f"arm must be ce or edl, got {arm!r}")


def write_plain_ece(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    import argparse

    from skeptic.store import verify_published_freeze

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--sealed", type=Path, required=True,
                        help="Mounted skeptic-sealed root that already holds test logits")
    args = parser.parse_args(argv)
    permit = verify_published_freeze(args.repo)
    # The Modal job that fills test logits is run by the account that holds
    # skeptic-sealed. This entry point refuses to start before that freeze
    # and writes /sealed/plain_ece.json with the permit stamped on it.
    destination = args.sealed / "plain_ece.json"
    if destination.exists():
        print(json.dumps({"status": "already_present", "path": str(destination)}))
        return 0
    print(json.dumps({
        "status": "needs_sealed_logits",
        "decision_commit": permit.commit,
        "decision_digest": permit.digest,
        "path": str(destination),
    }))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
