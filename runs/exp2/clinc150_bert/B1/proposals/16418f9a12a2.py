import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy and pooled-feature Mahalanobis novelty capture complementary logit and representation-space evidence. Train-fitted empirical percentile scaling puts both signals on a common distribution-free scale before combining their upper-tail surprise."
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

    def upper_tail_surprise(train_values, values):
        ref = np.sort(np.nan_to_num(
            train_values,
            nan=0.0,
            posinf=np.finfo(float).max / 4.0,
            neginf=-np.finfo(float).max / 4.0,
        ))
        x = np.nan_to_num(
            values,
            nan=0.0,
            posinf=np.finfo(float).max / 4.0,
            neginf=-np.finfo(float).max / 4.0,
        )
        cdf = np.searchsorted(ref, x, side="right") / float(ref.size + 1)
        tail = np.clip(1.0 - cdf, 1e-6, 1.0)
        return -np.log(tail)

    return upper_tail_surprise(train_energy, eval_energy) + upper_tail_surprise(
        train_dist, eval_dist
    )
