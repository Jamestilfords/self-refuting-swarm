import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy is retained for confidently classified items, while a train-calibrated energy uncertainty gate increasingly emphasizes encoder-space distance when classifier evidence is weak. This targets uncertain OOS queries whose logits alone are ambiguous but whose representations are also atypical."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0043"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_dist = g.min_mahalanobis_sq(tr.pooled)
    eval_dist = g.min_mahalanobis_sq(ev.pooled)

    energy_cdf = ctx.h.normalizer("ecdf", train_energy)
    dist_cdf = ctx.h.normalizer("ecdf", train_dist)

    e = energy_cdf(eval_energy)
    d = dist_cdf(eval_dist)

    # Higher energy means less confident classifier evidence, so defer more
    # strongly to feature-space atypicality in that regime.
    uncertainty = np.clip(e, 0.0, 1.0)
    return (1.0 - uncertainty) * e + uncertainty * d
