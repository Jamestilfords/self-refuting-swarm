import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Class-specific diagonal covariances retain intent-dependent feature scales that a tied covariance suppresses. "
    "Shrinking each class variance toward the global variance stabilizes small intent clusters while flagging queries atypical for their predicted intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0010"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    xtr = np.asarray(tr.pooled, dtype=float)
    xev = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    pred = np.asarray(ev.logits).argmax(axis=1)

    k = tr.logits.shape[1]
    global_var = np.var(xtr, axis=0) + 1e-6
    scores = np.empty(xev.shape[0], dtype=float)

    for c in range(k):
        train_mask = y == c
        eval_mask = pred == c

        if not np.any(eval_mask):
            continue

        xc = xtr[train_mask]
        n_c = xc.shape[0]
        if n_c == 0:
            center = np.mean(xtr, axis=0)
            var = global_var
            own_dist = np.sum((xtr - center) ** 2 / var, axis=1)
        else:
            center = np.mean(xc, axis=0)
            if n_c > 1:
                class_var = np.var(xc, axis=0)
            else:
                class_var = global_var
            shrink = 20.0 / (n_c + 20.0)
            var = (1.0 - shrink) * class_var + shrink * global_var + 1e-6
            own_dist = np.sum((xc - center) ** 2 / var, axis=1)

        test_dist = np.sum((xev[eval_mask] - center) ** 2 / var, axis=1)
        reference = np.sort(own_dist)
        ranks = np.searchsorted(reference, test_dist, side="right")
        percentile = ranks / (reference.size + 1.0)

        scale = np.percentile(own_dist, 75.0) + 1e-6
        scores[eval_mask] = percentile + 1e-4 * (test_dist / scale)

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
