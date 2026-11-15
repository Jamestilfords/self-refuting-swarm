import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy remains the dominant signal for confidently in-scope-looking queries, while a train-calibrated uncertainty gate increasingly incorporates pooled-space Mahalanobis distance when classifier energy is suspicious. This selectively uses representation support where logit confidence is least reliable."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0018"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_d = g.min_mahalanobis_sq(tr.pooled)
    eval_d = g.min_mahalanobis_sq(ev.pooled)

    train_e = ctx.h.energy(tr.logits)
    eval_e = ctx.h.energy(ev.logits)

    def train_ecdf(train_values, values):
        ref = np.sort(np.asarray(train_values, dtype=float))
        return np.searchsorted(ref, values, side="right") / float(ref.size + 1)

    e_rank = train_ecdf(train_e, eval_e)
    d_rank = train_ecdf(train_d, eval_d)

    q50, q75 = np.percentile(train_e, [50.0, 75.0])
    scale = max(q75 - q50, 1e-6)
    gate = 1.0 / (1.0 + np.exp(-np.clip((eval_e - q75) / scale, -40.0, 40.0)))

    return e_rank + gate * d_rank
