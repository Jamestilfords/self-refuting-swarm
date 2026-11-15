import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Train-calibrated empirical percentiles place logit energy and encoder-space Mahalanobis distance on the same scale. Their geometric interaction emphasizes queries that are jointly uncertain under the classifier and atypical in feature space."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    def percentile(reference, values):
        reference = np.sort(np.nan_to_num(
            reference, nan=0.0, posinf=1e12, neginf=-1e12
        ))
        values = np.nan_to_num(values, nan=0.0, posinf=1e12, neginf=-1e12)
        ranks = np.searchsorted(reference, values, side="right").astype(float)
        return (ranks + 1.0) / (reference.size + 2.0)

    energy_p = percentile(train_energy, eval_energy)
    dist_p = percentile(train_dist, eval_dist)

    return np.sqrt(energy_p * dist_p)
