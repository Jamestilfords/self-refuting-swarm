"""Adapt the existing critic_inputs loader to independent E5 replay states."""
from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path
from skeptic.rules.common import ABLATION_SKIP, IGNORE_PARENT_NAMES
from skeptic.rules.critic import Critic


def drop_ignored_parents(score):
    """G6 does not apply to the three couplings whose files still declare PARENTS."""
    if getattr(score, "name", None) in IGNORE_PARENT_NAMES and getattr(score, "parents", ()):
        return replace(score, parents=())
    return score


def ensure_er1_path() -> None:
    src = Path(os.environ.get("SKEPTIC_ER1_SRC") or
               Path(__file__).resolve().parents[2] / "third_party/er1/src")
    if not (src / "bert_daedl").is_dir():
        raise RuntimeError("Pinned ER-1 source is missing; run make bootstrap")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def critic_factory(cells, config):
    """Reuse loaded ScoreSets; give each cell/ordering its own LORD++ state."""
    ensure_er1_path()
    from skeptic.critic.cache import MemoryStatsCache
    from skeptic.critic.lord import LordPlusPlus
    from skeptic.critic.verdict import verdict
    from skeptic.fastboot import paired_bootstrap

    stats_cache = MemoryStatsCache()
    gate_cache = {}

    def make(cell, ordering, initial_sha, rule="R2"):
        loaded = cells[cell]
        if loaded.cell != cell or initial_sha not in loaded.candidates:
            raise ValueError("Critic inputs do not contain the replay incumbent")
        if rule not in ABLATION_SKIP:
            raise ValueError(f"Unknown critic rule: {rule}")
        skip = ABLATION_SKIP[rule]
        state = LordPlusPlus(alpha=config.fdr_alpha, w0=config.fdr_w0)

        def decide(row, incumbent_sha, fdr_state, position):
            return verdict(drop_ignored_parents(loaded.candidates[row["sha"]]),
                           loaded.candidates[incumbent_sha],
                           loaded.baselines, fdr_state, cell, config=config, cache=stats_cache,
                           bootstrap_fn=paired_bootstrap, gate_cache=gate_cache, skip=skip)

        return Critic(initial_sha, state, decide, rule=rule)
    return make
