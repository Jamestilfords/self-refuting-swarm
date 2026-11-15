import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Maximum softmax uncertainty and pooled-space prototype distance detect complementary failures: ambiguous classifier competition and weak representation support. Train-fitted empirical tail probabilities put the two signals on a common scale before their joint surprisal is computed."
SOURCE = ""
PARENTS = ["msp", "d_port"]
MEMORY_USED = ["m_0010", "m_0021", "m_0167"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_logit = ctx.h.msp(tr.logits)
    train_dist = g.min_mahalanobis_sq(tr.pooled)

    eval_logit = ctx.h.msp(ev.logits)
    eval_dist = g.min_mahalanobis_sq(ev.pooled)

    logit_sorted = np.sort(np.asarray(train_logit, dtype=float))
    dist_sorted = np.sort(np.asarray(train_dist, dtype=float))
    n_logit = max(logit_sorted.size, 1)
    n_dist = max(dist_sorted.size, 1)

    u_logit = np.searchsorted(logit_sorted, eval_logit, side="right") / n_logit
    u_dist = np.searchsorted(dist_sorted, eval_dist, side="right") / n_dist

    eps = 1.0 / (max(n_logit, n_dist) + 1.0)
    u_logit = np.clip(u_logit, eps, 1.0 - eps)
    u_dist = np.clip(u_dist, eps, 1.0 - eps)

    return -np.log((1.0 - u_logit) * (1.0 - u_dist))
