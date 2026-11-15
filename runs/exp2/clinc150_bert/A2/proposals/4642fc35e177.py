import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Confident in-scope predictions retain the energy ordering, while high-energy uncertain predictions are increasingly scored by encoder-space atypicality. This conditional blend uses feature distance only where classifier confidence is insufficient."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0015"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)
    train_dist = g.min_mahalanobis_sq(tr.pooled)
    eval_dist = g.min_mahalanobis_sq(ev.pooled)

    e_ref = np.sort(np.asarray(train_energy, dtype=float))
    d_ref = np.sort(np.asarray(train_dist, dtype=float))
    n = max(len(e_ref), 1)

    def ecdf(x, ref):
        return np.searchsorted(ref, x, side="right").astype(float) / n

    e_score = ecdf(eval_energy, e_ref)
    d_score = ecdf(eval_dist, d_ref)

    # Energy percentile is an uncertainty gate: low-energy items preserve
    # energy's ranking, while uncertain items increasingly rely on geometry.
    gate = e_score * e_score
    return (1.0 - gate) * e_score + gate * d_score
