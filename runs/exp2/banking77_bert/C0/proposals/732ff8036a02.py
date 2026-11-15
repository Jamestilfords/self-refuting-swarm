import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy and pooled-feature Mahalanobis distance capture complementary logit uncertainty and representation-space atypicality. Mapping each through its training empirical CDF puts them on a common, train-calibrated scale before fusion."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    def train_ecdf(train_values, values):
        train_values = np.nan_to_num(
            train_values, nan=0.0, posinf=1e12, neginf=-1e12
        )
        values = np.nan_to_num(values, nan=0.0, posinf=1e12, neginf=-1e12)
        ordered = np.sort(train_values)
        return np.searchsorted(ordered, values, side="right") / float(len(ordered))

    energy_q = train_ecdf(train_energy, eval_energy)
    dist_q = train_ecdf(train_dist, eval_dist)
    return 0.5 * energy_q + 0.5 * dist_q
