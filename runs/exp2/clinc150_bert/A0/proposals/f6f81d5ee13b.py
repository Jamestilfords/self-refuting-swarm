import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "The normalized Dirichlet concentration shape describes how evidential mass is allocated "
    "among intents independently of total evidence. OOS examples should have an allocation "
    "pattern poorly supported by every class-conditional in-scope evidential shape model."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0101"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def clr_alpha(logits):
        alpha = np.asarray(ctx.h.alpha(logits), dtype=float)
        alpha = np.maximum(alpha, 1e-8)
        log_alpha = np.log(alpha)
        return log_alpha - log_alpha.mean(axis=1, keepdims=True)

    z_train = clr_alpha(tr.logits)
    z_eval = clr_alpha(ev.logits)
    labels = np.asarray(tr.labels, dtype=int)
    n_classes = tr.logits.shape[1]

    global_var = np.var(z_train, axis=0) + 1e-5
    best = np.full(z_eval.shape[0], np.inf, dtype=float)

    for c in range(n_classes):
        members = z_train[labels == c]
        if members.shape[0] == 0:
            continue
        mean = members.mean(axis=0)
        if members.shape[0] > 1:
            var = np.var(members, axis=0)
            shrink = members.shape[0] / (members.shape[0] + 10.0)
            var = shrink * var + (1.0 - shrink) * global_var
        else:
            var = global_var
        dist = np.sum((z_eval - mean) ** 2 / (var + 1e-5), axis=1)
        best = np.minimum(best, dist)

    return np.maximum(best, 0.0)
