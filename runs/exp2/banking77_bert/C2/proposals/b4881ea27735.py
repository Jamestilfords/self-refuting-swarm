import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Empirical training quantiles place energy and bottleneck-space class-conditional distance on the same percentile scale. A nonlinear agreement term emphasizes queries that are simultaneously uncertain to the classifier and atypical in feature space."
SOURCE = ""
PARENTS = ["energy", "d_port"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.feats), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.feats), dtype=float)

    def train_ecdf(train_values, values):
        reference = np.sort(np.nan_to_num(
            train_values, nan=0.0, posinf=1e12, neginf=-1e12
        ))
        query = np.nan_to_num(values, nan=0.0, posinf=1e12, neginf=-1e12)
        return np.searchsorted(reference, query, side="right") / float(len(reference))

    energy_q = train_ecdf(train_energy, eval_energy)
    dist_q = train_ecdf(train_dist, eval_dist)

    scores = 0.35 * energy_q + 0.65 * dist_q + 0.50 * energy_q * dist_q
    return np.nan_to_num(scores, nan=0.0, posinf=2.0, neginf=0.0)
