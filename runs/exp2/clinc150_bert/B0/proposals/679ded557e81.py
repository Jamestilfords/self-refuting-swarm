import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Confident classifier predictions retain the energy ranking, while queries in the high-energy uncertainty region are increasingly ranked by pooled-feature Mahalanobis distance. This conditional handoff targets OOS examples whose logits are ambiguous without diluting reliable logit evidence."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0095"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    e_tr = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    e_ev = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    d_tr = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    d_ev = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    def ecdf(train_values, values):
        ref = np.sort(np.asarray(train_values, dtype=float))
        return np.searchsorted(ref, values, side="right") / float(ref.size + 1)

    e_rank = ecdf(e_tr, e_ev)
    d_rank = ecdf(d_tr, d_ev)

    center = np.quantile(e_tr, 0.70)
    q25, q75 = np.quantile(e_tr, [0.25, 0.75])
    scale = max(float(q75 - q25), 1e-6)
    gate = 1.0 / (1.0 + np.exp(-np.clip((e_ev - center) / scale, -30.0, 30.0)))

    return (1.0 - gate) * e_rank + gate * d_rank
