import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Train-calibrated empirical percentile scales make logit energy and pooled-space class distance directly comparable without using evaluation-set statistics. Their product emphasizes queries that are simultaneously logit-uncertain and representation-atypical."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    train_energy = np.nan_to_num(
        ctx.h.energy(tr.logits), nan=0.0, posinf=1e12, neginf=-1e12
    )
    eval_energy = np.nan_to_num(
        ctx.h.energy(ev.logits), nan=0.0, posinf=1e12, neginf=-1e12
    )
    train_dist = np.nan_to_num(
        g.min_mahalanobis_sq(tr.pooled), nan=0.0, posinf=1e12, neginf=-1e12
    )
    eval_dist = np.nan_to_num(
        g.min_mahalanobis_sq(ev.pooled), nan=0.0, posinf=1e12, neginf=-1e12
    )

    def percentile_from_train(train_values, values):
        reference = np.sort(train_values)
        n = max(reference.size, 1)
        return np.searchsorted(reference, values, side="right") / float(n)

    energy_pct = percentile_from_train(train_energy, eval_energy)
    distance_pct = percentile_from_train(train_dist, eval_dist)

    return energy_pct * distance_pct
