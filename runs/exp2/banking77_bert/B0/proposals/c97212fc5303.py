import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Classifier energy and full encoder-space Mahalanobis distance capture complementary uncertainty signals. Mapping each score to its train-fitted empirical percentile puts both on a shared robust scale before combining their upper-tail evidence."
SOURCE = "https://ar5iv.labs.arxiv.org/html/2305.16966"
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0048"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.nan_to_num(
        ctx.h.energy(tr.logits), nan=0.0, posinf=1e6, neginf=-1e6
    )
    eval_energy = np.nan_to_num(
        ctx.h.energy(ev.logits), nan=0.0, posinf=1e6, neginf=-1e6
    )
    train_dist = np.nan_to_num(
        g.min_mahalanobis_sq(tr.pooled), nan=0.0, posinf=1e6, neginf=-1e6
    )
    eval_dist = np.nan_to_num(
        g.min_mahalanobis_sq(ev.pooled), nan=0.0, posinf=1e6, neginf=-1e6
    )

    def upper_tail_logit(train_values, values):
        reference = np.sort(train_values)
        n = reference.size
        ranks = np.searchsorted(reference, values, side="right")
        p = (ranks + 0.5) / (n + 1.0)
        p = np.clip(p, 1e-5, 1.0 - 1e-5)
        return np.log(p) - np.log1p(-p)

    e = upper_tail_logit(train_energy, eval_energy)
    d = upper_tail_logit(train_dist, eval_dist)
    return 0.5 * (e + d)
