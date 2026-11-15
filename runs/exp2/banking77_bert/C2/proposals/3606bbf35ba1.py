import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Train-ECDF calibration makes maximum-softmax uncertainty and local feature-space novelty comparable without using evaluation-set statistics. Their geometric combination prioritizes queries that are simultaneously poorly supported by the classifier and far from nearby in-distribution representations."
SOURCE = ""
PARENTS = ["msp"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_logit = ctx.h.msp(tr.logits)
    eval_logit = ctx.h.msp(ev.logits)

    train_dist = ctx.h.knn(tr.pooled, tr.pooled, k=11, l2=True)
    eval_dist = ctx.h.knn(tr.pooled, ev.pooled, k=10, l2=True)

    def ecdf(train_values, values):
        ref = np.sort(np.asarray(train_values, dtype=float))
        n = ref.size
        ranks = np.searchsorted(ref, np.asarray(values, dtype=float), side="right")
        return (ranks + 0.5) / (n + 1.0)

    logit_tail = ecdf(train_logit, eval_logit)
    dist_tail = ecdf(train_dist, eval_dist)

    return np.sqrt(logit_tail * dist_tail)
