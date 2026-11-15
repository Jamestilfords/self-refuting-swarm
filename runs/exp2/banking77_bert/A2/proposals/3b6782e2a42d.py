import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "A train-calibrated energy gate preserves the energy ranking for confident items, while smoothly replacing it with encoder-space novelty when the classifier's energy is unusually uncertain. This avoids applying representation distance indiscriminately to examples already well supported by the classifier."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0015", "m_0019"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    train_log_density = g.log_density(tr.pooled)
    eval_log_density = g.log_density(ev.pooled)

    energy_cdf = ctx.h.normalizer("ecdf", train_energy)
    density_cdf = ctx.h.normalizer("ecdf", train_log_density)

    e = energy_cdf(eval_energy)
    d = 1.0 - density_cdf(eval_log_density)

    gate = e * e
    return (1.0 - gate) * e + gate * d
