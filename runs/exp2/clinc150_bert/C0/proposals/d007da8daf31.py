import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Train-derived empirical CDFs place classifier entropy and encoder-space distance on the same percentile scale. A geometric fusion emphasizes queries that are simultaneously uncertain for the classifier and atypical in feature space."
SOURCE = ""
PARENTS = ["msp", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_logit = ctx.h.entropy(tr.logits)
    eval_logit = ctx.h.entropy(ev.logits)

    train_dist = g.min_mahalanobis_sq(tr.pooled)
    eval_dist = g.min_mahalanobis_sq(ev.pooled)

    logit_sorted = np.sort(np.asarray(train_logit, dtype=float))
    dist_sorted = np.sort(np.asarray(train_dist, dtype=float))
    n = max(len(logit_sorted), 1)

    logit_u = (
        np.searchsorted(logit_sorted, np.asarray(eval_logit, dtype=float), side="right")
        + 0.5
    ) / (n + 1.0)
    dist_u = (
        np.searchsorted(dist_sorted, np.asarray(eval_dist, dtype=float), side="right")
        + 0.5
    ) / (n + 1.0)

    scores = np.sqrt(np.clip(logit_u, 1e-12, 1.0) * np.clip(dist_u, 1e-12, 1.0))
    return np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
