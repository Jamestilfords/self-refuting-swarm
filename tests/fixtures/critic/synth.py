"""Deterministic synthetic cell for the critic tests.

Latent in-scope scores are N(0, 1) and out-of-scope scores are N(1.5, 1),
drawn once per seed from ``np.random.default_rng(seed)``. Registered scores are
noisier views of that latent. ``energy_ce`` has the least noise, so it is the
strongest baseline and the incumbent.

Expected verdicts follow from the construction. Effect sizes were checked
against the frozen thresholds; the thresholds were not moved.

- ``monotone``: ``2 * energy_ce + 1``. Every pair credit matches, so the flip
  rate is 0 on every seed, which is below 0.001. Majority INERT.
- ``renamed``: the ``msp_ce`` arrays under a new name. ``msp_ce`` is a much
  noisier view than ``energy_ce``, so the flip rate against the incumbent is
  above 0.001 on every seed, and Spearman with ``msp_ce`` is exactly 1.
  SCORE-MATCHED.
- ``noise``: i.i.d. N(0, 1), independent of the latent. AUROC stays near 0.5,
  below ``energy_ce``, so the seed-mean gap is negative. NO EFFECT, p = 1,
  one LORD++ step, no bootstrap.
- ``strong``: the latent plus N(0, 0.05). That is far cleaner than
  ``energy_ce`` (noise 1.35), so the AUROC gap is well above 0.05, Spearman
  with every baseline is far below 0.999, and the one-sided p is below the
  first LORD++ level. ACCEPT.
- ``small_true``: ``energy_ce`` plus an independent N(0, 0.35) jiggle and a
  +0.08 shift on out-of-scope items. The jiggle keeps Spearman with
  ``energy_ce`` under 0.999 (measured max about 0.985). The shift leaves a
  seed-mean gain of about 0.007, too small for the first LORD++ level. The
  combined interval is not inside the 0.001 band, so G4 returns INCONCLUSIVE
  and the verdict is UNDERPOWERED, with mdd above the gain.
- ``strong_miscal``: ``strong`` with per-seed ECE 0.05 above the plain model.
  MISCALIBRATES.
- ``beats_ceiling``: in-scope scores 0 and out-of-scope scores 1, parents
  ``("energy", "d_full")``. AUROC is 1, above the linear ceiling of those
  noisy parents by more than 0.0001. BUG.
- ``guard``: ``status="guard"``. BUG at the lock.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from skeptic.critic.baselines import G2_NAMES
from skeptic.critic.types import BaselineSet, ScoreSet

_PLAIN = 0.02


@dataclass(frozen=True)
class SynthCell:
    baselines: BaselineSet
    incumbent: ScoreSet
    latent_id: tuple[np.ndarray, ...]
    latent_oos: tuple[np.ndarray, ...]

    @property
    def cell(self) -> str:
        return self.baselines.cell


def make_cell(seed: int = 0, n_id: int = 400, n_oos: int = 80, n_seeds: int = 5) -> SynthCell:
    rng = np.random.default_rng(seed)
    latent_id, latent_oos = [], []
    for _ in range(n_seeds):
        latent_id.append(rng.normal(0.0, 1.0, n_id))
        latent_oos.append(rng.normal(1.5, 1.0, n_oos))
    latent_id_t = tuple(latent_id)
    latent_oos_t = tuple(latent_oos)

    # name, arm, noise sd, stream id. energy_ce is the least noisy baseline.
    specs = (
        ("energy_ce", "ce", 1.35, 11),
        ("msp_ce", "ce", 3.2, 12),
        ("d_port_ce", "ce", 3.6, 13),
        ("d_full_ce", "ce", 3.4, 14),
        ("energy_edl", "edl", 2.4, 15),
        ("msp_edl", "edl", 3.8, 16),
        ("d_port_edl", "edl", 4.0, 17),
        ("d_full_edl", "edl", 3.5, 18),
        ("vacuity_edl", "edl", 2.8, 19),
    )
    scores = {}
    for name, arm, noise, stream in specs:
        view = np.random.default_rng(seed + 1000 + stream)
        ident, oos = [], []
        for index in range(n_seeds):
            ident.append(latent_id_t[index] + view.normal(0.0, noise, n_id))
            oos.append(latent_oos_t[index] + view.normal(0.0, noise, n_oos))
        scores[name] = ScoreSet(name=name, arm=arm, val_id=tuple(ident), val_oos=tuple(oos))
    plain = tuple(_PLAIN for _ in range(n_seeds))
    baselines = BaselineSet(
        cell="synth/bert",
        scores=scores,
        g2_names=G2_NAMES,
        plain_ece={"ce": plain, "edl": plain},
    )
    return SynthCell(baselines, scores["energy_ce"], latent_id_t, latent_oos_t)


def _view(cell: SynthCell, name: str, noise: float, stream: int, shift_oos: float = 0.0) -> ScoreSet:
    view = np.random.default_rng(10_000 + stream)
    ident, oos = [], []
    for lat_id, lat_oos in zip(cell.latent_id, cell.latent_oos):
        ident.append(lat_id + view.normal(0.0, noise, lat_id.shape[0]))
        oos.append(lat_oos + view.normal(0.0, noise, lat_oos.shape[0]) + shift_oos)
    return ScoreSet(name=name, arm="ce", val_id=tuple(ident), val_oos=tuple(oos))


def monotone(cell: SynthCell) -> ScoreSet:
    energy = cell.baselines.scores["energy_ce"]
    return ScoreSet(
        name="canary_monotone",
        arm="ce",
        val_id=tuple(2.0 * array + 1.0 for array in energy.val_id),
        val_oos=tuple(2.0 * array + 1.0 for array in energy.val_oos),
    )


def renamed(cell: SynthCell) -> ScoreSet:
    source = cell.baselines.scores["msp_ce"]
    return ScoreSet(
        name="canary_renamed",
        arm="ce",
        val_id=source.val_id,
        val_oos=source.val_oos,
    )


def noise(cell: SynthCell) -> ScoreSet:
    rng = np.random.default_rng(7)
    ident, oos = [], []
    for lat_id, lat_oos in zip(cell.latent_id, cell.latent_oos):
        ident.append(rng.normal(0.0, 1.0, lat_id.shape[0]))
        oos.append(rng.normal(0.0, 1.0, lat_oos.shape[0]))
    return ScoreSet(name="canary_noise", arm="ce", val_id=tuple(ident), val_oos=tuple(oos))


def strong(cell: SynthCell) -> ScoreSet:
    return _view(cell, "canary_strong", noise=0.05, stream=1)


def small_true(cell: SynthCell) -> ScoreSet:
    energy = cell.baselines.scores["energy_ce"]
    jiggle = np.random.default_rng(21)
    ident, oos = [], []
    for id_scores, oos_scores in zip(energy.val_id, energy.val_oos):
        ident.append(id_scores + jiggle.normal(0.0, 0.35, id_scores.shape[0]))
        oos.append(oos_scores + jiggle.normal(0.0, 0.35, oos_scores.shape[0]) + 0.08)
    return ScoreSet(name="canary_small", arm="ce", val_id=tuple(ident), val_oos=tuple(oos))


def strong_miscal(cell: SynthCell) -> ScoreSet:
    base = strong(cell)
    plain = cell.baselines.plain_ece["ce"]
    return ScoreSet(
        name="canary_miscal",
        arm="ce",
        val_id=base.val_id,
        val_oos=base.val_oos,
        ece=tuple(value + 0.05 for value in plain),
    )


def beats_ceiling(cell: SynthCell) -> ScoreSet:
    ident = tuple(np.zeros(array.shape[0], dtype=np.float64) for array in cell.latent_id)
    oos = tuple(np.ones(array.shape[0], dtype=np.float64) for array in cell.latent_oos)
    return ScoreSet(
        name="canary_ceiling",
        arm="ce",
        val_id=ident,
        val_oos=oos,
        parents=("energy", "d_full"),
    )


def guard(cell: SynthCell) -> ScoreSet:
    del cell
    return ScoreSet(name="canary_guard", arm="ce", val_id=(), val_oos=(), status="guard")
