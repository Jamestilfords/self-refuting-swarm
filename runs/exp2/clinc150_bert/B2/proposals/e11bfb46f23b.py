import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "MSP uncertainty and bottleneck-space class-conditional distance capture complementary logit and representation failures. Train-only empirical tail calibration maps both signals to a shared surprise scale before combining them."
SOURCE = ""
PARENTS = ["msp", "d_port"]
MEMORY_USED = ["m_0137"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)
    train_logit = ctx.h.msp(tr.logits)
    eval_logit = ctx.h.msp(ev.logits)
    train_dist = g.min_mahalanobis_sq(tr.feats)
    eval_dist = g.min_mahalanobis_sq(ev.feats)

    def tail_surprise(train_values, values):
        reference = np.sort(np.asarray(train_values, dtype=float))
        n = reference.size
        ranks = np.searchsorted(reference, np.asarray(values, dtype=float),
                                side="right")
        u = ranks / (n + 1.0)
        u = np.clip(u, 0.0, 1.0 - 1.0 / (n + 1.0))
        return -np.log1p(-u)

    logit_surprise = tail_surprise(train_logit, eval_logit)
    distance_surprise = tail_surprise(train_dist, eval_dist)
    return 0.5 * (logit_surprise + distance_surprise)
