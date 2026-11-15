import numpy as np

FAMILY = "knn"
ARM = "edl"
RATIONALE = "Out-of-scope inputs can induce Dirichlet concentration patterns whose relative class evidence differs from patterns seen for in-scope training examples. A kNN distance in centered log-ratio concentration space measures this evidential-shape novelty independently of the total evidence scale."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def clr(logits):
        alpha = ctx.h.alpha(logits)
        alpha = np.maximum(alpha, 1e-12)
        log_alpha = np.log(alpha)
        return log_alpha - np.mean(log_alpha, axis=1, keepdims=True)

    train_shape = clr(tr.logits)
    eval_shape = clr(ev.logits)
    return ctx.h.knn(train_shape, eval_shape, k=10, l2=False)
