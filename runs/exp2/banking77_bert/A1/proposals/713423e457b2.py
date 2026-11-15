import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy is most informative for confidently classified queries, while feature-space departure becomes more reliable as classifier uncertainty rises. A train-fitted energy percentile softly routes uncertain items toward full encoder Mahalanobis surprise without using evaluation-set statistics."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0043", "m_0063"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    gda = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(gda.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(gda.min_mahalanobis_sq(ev.pooled), dtype=float)

    train_energy = np.nan_to_num(train_energy, nan=0.0, posinf=1e12, neginf=-1e12)
    eval_energy = np.nan_to_num(eval_energy, nan=0.0, posinf=1e12, neginf=-1e12)
    train_dist = np.nan_to_num(train_dist, nan=0.0, posinf=1e12, neginf=-1e12)
    eval_dist = np.nan_to_num(eval_dist, nan=0.0, posinf=1e12, neginf=-1e12)

    e_ref = np.sort(train_energy)
    d_ref = np.sort(train_dist)
    n = float(len(train_energy))

    e_q = np.searchsorted(e_ref, eval_energy, side="right") / n
    d_q = np.searchsorted(d_ref, eval_dist, side="right") / n

    # Low-energy items remain energy-driven; high-energy items are routed to geometry.
    gate = np.clip(e_q, 0.0, 1.0)
    return (1.0 - gate) * e_q + gate * d_q
