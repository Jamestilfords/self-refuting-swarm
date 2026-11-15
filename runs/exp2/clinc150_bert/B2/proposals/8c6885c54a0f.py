import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Confident predictions retain the energy ranking, while low-margin predictions progressively defer to pooled-feature Mahalanobis distance. This uses representation geometry specifically where logits provide weak decision evidence."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0058"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    d_train = g.min_mahalanobis_sq(tr.pooled)
    d_eval = g.min_mahalanobis_sq(ev.pooled)

    e_train = ctx.h.energy(tr.logits)
    e_eval = ctx.h.energy(ev.logits)

    def percentile(train_values, values):
        ref = np.sort(np.asarray(train_values, dtype=float))
        ranks = np.searchsorted(ref, values, side="right")
        return ranks / float(ref.size + 1)

    e_score = percentile(e_train, e_eval)
    d_score = percentile(d_train, d_eval)

    def margin(logits):
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        return top2[:, 1] - top2[:, 0]

    m_train = margin(tr.logits)
    m_eval = margin(ev.logits)

    lo, hi = np.quantile(m_train, [0.10, 0.90])
    scale = max(hi - lo, 1e-8)
    confident_weight = np.clip((m_eval - lo) / scale, 0.0, 1.0)

    return confident_weight * e_score + (1.0 - confident_weight) * d_score
