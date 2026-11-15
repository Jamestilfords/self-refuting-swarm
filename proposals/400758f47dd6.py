import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "EDL concentration vectors encode both total evidence and how that evidence is distributed "
    "across classes; OOS queries often have an atypical concentration shape even when their energy is confident."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    k = tr.logits.shape[1]

    def signature(logits):
        alpha = np.asarray(ctx.h.alpha(logits), dtype=float)
        alpha = np.clip(alpha, 1e-12, 1e12)
        total = np.sum(alpha, axis=1, keepdims=True)
        p = alpha / total
        top2 = np.partition(p, -2, axis=1)[:, -2:]
        entropy = -np.sum(p * np.log(p), axis=1)
        return np.column_stack((
            np.log(total[:, 0]),
            entropy,
            np.max(p, axis=1),
            np.min(top2, axis=1),
            np.sum(p * p, axis=1),
        ))

    x_train = signature(tr.logits)
    x_eval = signature(ev.logits)
    labels = np.asarray(tr.labels, dtype=int)

    global_mean = np.mean(x_train, axis=0)
    global_var = np.var(x_train, axis=0)
    var_floor = np.maximum(0.05 * global_var, 1e-6)

    means = np.tile(global_mean, (k, 1))
    variances = np.tile(global_var + var_floor, (k, 1))

    for c in range(k):
        xc = x_train[labels == c]
        if len(xc):
            means[c] = np.mean(xc, axis=0)
            variances[c] = np.var(xc, axis=0) + var_floor

    predicted = np.argmax(ev.logits, axis=1)
    centered = x_eval - means[predicted]
    scores = np.mean((centered * centered) / variances[predicted], axis=1)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
