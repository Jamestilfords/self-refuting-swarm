import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "OOS inputs can produce Dirichlet concentration proportions that are atypical for every "
    "in-scope intent, even when their total evidence (and thus vacuity or energy) appears normal."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    a_tr = np.asarray(ctx.h.alpha(tr.logits), dtype=float)
    a_ev = np.asarray(ctx.h.alpha(ev.logits), dtype=float)

    a_tr = np.nan_to_num(a_tr, nan=1.0, posinf=1e12, neginf=1e-12)
    a_ev = np.nan_to_num(a_ev, nan=1.0, posinf=1e12, neginf=1e-12)
    a_tr = np.maximum(a_tr, 1e-12)
    a_ev = np.maximum(a_ev, 1e-12)

    p_tr = a_tr / np.sum(a_tr, axis=1, keepdims=True)
    p_ev = a_ev / np.sum(a_ev, axis=1, keepdims=True)

    labels = np.asarray(tr.labels, dtype=int)
    n_classes = a_tr.shape[1]
    global_mean = np.mean(p_tr, axis=0)
    global_var = np.var(p_tr, axis=0) + 1e-6

    means = np.tile(global_mean, (n_classes, 1))
    variances = np.tile(global_var, (n_classes, 1))

    for c in range(n_classes):
        mask = labels == c
        count = int(np.sum(mask))
        if count >= 2:
            pc = p_tr[mask]
            means[c] = np.mean(pc, axis=0)
            variances[c] = np.var(pc, axis=0) + 1e-6
        elif count == 1:
            means[c] = p_tr[mask][0]
            variances[c] = global_var

    pred = np.argmax(a_ev, axis=1)
    centered = p_ev - means[pred]
    z2 = centered * centered / variances[pred]

    # Averaging makes the scale stable as the number of intents changes.
    return np.mean(z2, axis=1)
