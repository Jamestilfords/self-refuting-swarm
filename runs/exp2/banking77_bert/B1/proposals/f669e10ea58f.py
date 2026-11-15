import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Train-calibrated upper-tail probabilities place logit energy and pooled-space Mahalanobis distance on a common support scale. Their product acts as a conjunctive detector, prioritizing queries that are both classifier-uncertain and geometrically unsupported."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0142"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_distance = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_distance = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    def upper_tail(train_values, values):
        train_values = np.nan_to_num(
            train_values, nan=0.0, posinf=1e12, neginf=-1e12
        )
        values = np.nan_to_num(values, nan=0.0, posinf=1e12, neginf=-1e12)
        reference = np.sort(train_values)
        ranks = np.searchsorted(reference, values, side="right")
        return (ranks + 1.0) / (reference.size + 2.0)

    energy_tail = upper_tail(train_energy, eval_energy)
    distance_tail = upper_tail(train_distance, eval_distance)

    scores = energy_tail * distance_tail
    return np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
