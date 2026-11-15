import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Maximum-softmax uncertainty and bottleneck-space distance capture complementary failures: uncertain class assignment "
    "and geometric departure from known-intent representations. Train empirical-CDF calibration enables a nonlinear "
    "soft-OR fusion that emphasizes either strong novelty signal without using evaluation-set statistics."
)
SOURCE = ""
PARENTS = ["msp", "d_port"]
MEMORY_USED = ["m_0117"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    train_logit = np.asarray(ctx.h.msp(tr.logits), dtype=float)
    eval_logit = np.asarray(ctx.h.msp(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.feats), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.feats), dtype=float)

    train_logit = np.nan_to_num(train_logit, nan=0.0, posinf=1e12, neginf=-1e12)
    eval_logit = np.nan_to_num(eval_logit, nan=0.0, posinf=1e12, neginf=-1e12)
    train_dist = np.nan_to_num(train_dist, nan=0.0, posinf=1e12, neginf=0.0)
    eval_dist = np.nan_to_num(eval_dist, nan=0.0, posinf=1e12, neginf=0.0)

    logit_ref = np.sort(train_logit)
    dist_ref = np.sort(train_dist)
    n = float(max(len(train_logit), 1))

    q_logit = np.searchsorted(logit_ref, eval_logit, side="right") / n
    q_dist = np.searchsorted(dist_ref, eval_dist, side="right") / n

    return np.clip(q_logit + q_dist - q_logit * q_dist, 0.0, 1.0)
