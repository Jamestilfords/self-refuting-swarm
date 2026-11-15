import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Training empirical-CDF calibration maps energy and encoder-space Mahalanobis distance onto the same tail-probability scale. A max fusion emphasizes whichever signal provides stronger OOS evidence for an individual query, avoiding domination by one raw score scale."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_distance = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_distance = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    def train_ecdf(train_values, values):
        reference = np.sort(np.nan_to_num(
            train_values, nan=0.0, posinf=1e12, neginf=-1e12
        ))
        query = np.nan_to_num(values, nan=0.0, posinf=1e12, neginf=-1e12)
        ranks = np.searchsorted(reference, query, side="right")
        return ranks.astype(float) / float(max(reference.size, 1))

    energy_tail = train_ecdf(train_energy, eval_energy)
    distance_tail = train_ecdf(train_distance, eval_distance)
    return np.maximum(energy_tail, distance_tail)
