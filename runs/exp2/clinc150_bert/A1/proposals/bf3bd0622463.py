import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Classifier energy is retained for training-calibrated confident regions, while smoothly switching to pooled-space Gaussian novelty in the uncertain energy tail. This lets representation distance resolve cases where logits provide weak OOS separation."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0085"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_d = g.min_mahalanobis_sq(tr.pooled)
    eval_d = g.min_mahalanobis_sq(ev.pooled)

    train_e = ctx.h.energy(tr.logits)
    eval_e = ctx.h.energy(ev.logits)

    def ecdf(train_values, values):
        ref = np.sort(np.asarray(train_values, dtype=float))
        return np.searchsorted(ref, values, side="right") / max(len(ref), 1)

    energy_tail = ecdf(train_e, eval_e)
    density_tail = ecdf(train_d, eval_d)

    lo, hi = np.quantile(train_e, [0.70, 0.90])
    width = max(float(hi - lo), 1e-8)
    gate = np.clip((eval_e - lo) / width, 0.0, 1.0)
    gate = gate * gate * (3.0 - 2.0 * gate)

    return (1.0 - gate) * energy_tail + gate * density_tail
