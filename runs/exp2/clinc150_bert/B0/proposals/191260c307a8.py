import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Classifier uncertainty and bottleneck-space distance are converted to training empirical percentiles before fusion, so either unusually weak logit confidence or poor geometric support can flag OOS queries on a common scale."
SOURCE = ""
PARENTS = ["msp", "d_port"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    train_msp = ctx.h.msp(tr.logits)
    eval_msp = ctx.h.msp(ev.logits)
    train_dist = g.min_mahalanobis_sq(tr.feats)
    eval_dist = g.min_mahalanobis_sq(ev.feats)

    msp_ref = np.sort(np.asarray(train_msp, dtype=float))
    dist_ref = np.sort(np.asarray(train_dist, dtype=float))

    def percentile(x, ref):
        p = np.searchsorted(ref, x, side="right") / float(ref.size)
        return np.clip(p, 1.0 / (ref.size + 1.0), 1.0 - 1.0 / (ref.size + 1.0))

    u_msp = percentile(eval_msp, msp_ref)
    u_dist = percentile(eval_dist, dist_ref)

    return 1.0 - (1.0 - u_msp) * (1.0 - u_dist)
