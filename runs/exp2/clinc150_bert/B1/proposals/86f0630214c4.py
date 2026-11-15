import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = "The normalized Dirichlet concentration profile contains evidence-shape information independent of total concentration and therefore is not a monotone transform of vacuity or energy. Class-conditional calibration of entropy and top-concentration contrast flags queries with atypical evidential allocation for their routed intent."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0120"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    k = tr.logits.shape[1]

    def shape_stats(logits):
        alpha = np.maximum(ctx.h.alpha(logits), 1e-12)
        p = alpha / np.maximum(alpha.sum(axis=1, keepdims=True), 1e-12)

        entropy = -(p * np.log(p)).sum(axis=1) / np.log(max(k, 2))
        top2 = np.partition(p, kth=max(k - 2, 0), axis=1)[:, -2:]
        contrast = np.log(np.maximum(top2[:, 1], 1e-12)) - np.log(
            np.maximum(top2[:, 0], 1e-12)
        )
        return np.column_stack((entropy, contrast))

    train_stats = shape_stats(tr.logits)
    eval_stats = shape_stats(ev.logits)

    labels = np.asarray(tr.labels, dtype=int)
    routed = np.argmax(ev.logits, axis=1)
    global_mean = train_stats.mean(axis=0)
    global_var = train_stats.var(axis=0) + 1e-6

    means = np.empty((k, 2), dtype=float)
    vars_ = np.empty((k, 2), dtype=float)
    for c in range(k):
        members = train_stats[labels == c]
        n = len(members)
        if n == 0:
            means[c] = global_mean
            vars_[c] = global_var
        else:
            weight = n / (n + 12.0)
            means[c] = weight * members.mean(axis=0) + (1.0 - weight) * global_mean
            local_var = members.var(axis=0) if n > 1 else global_var
            vars_[c] = weight * local_var + (1.0 - weight) * global_var + 1e-6

    z = np.abs(eval_stats - means[routed]) / np.sqrt(vars_[routed])
    return np.maximum(z[:, 0], z[:, 1])
