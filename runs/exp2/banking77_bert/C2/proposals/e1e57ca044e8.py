import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = "OOS queries can induce Dirichlet evidence allocations unlike the class-conditional allocation shapes seen for in-scope training examples. This score measures atypicality of the normalized concentration profile rather than total evidence, so it is not a monotone transform of energy or vacuity."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    a_tr = np.clip(ctx.h.alpha(tr.logits), 1e-12, 1e12)
    a_ev = np.clip(ctx.h.alpha(ev.logits), 1e-12, 1e12)

    q_tr = a_tr / np.sum(a_tr, axis=1, keepdims=True)
    q_ev = a_ev / np.sum(a_ev, axis=1, keepdims=True)

    k = q_tr.shape[1]
    labels = np.asarray(tr.labels, dtype=int)
    counts = np.bincount(labels, minlength=k).astype(float)

    sums = np.zeros((k, k), dtype=float)
    sums_sq = np.zeros((k, k), dtype=float)
    np.add.at(sums, labels, q_tr)
    np.add.at(sums_sq, labels, q_tr * q_tr)

    global_mean = np.mean(q_tr, axis=0)
    global_var = np.var(q_tr, axis=0)
    denom = np.maximum(counts[:, None], 1.0)

    means = sums / denom
    variances = sums_sq / denom - means * means

    missing = counts == 0
    means[missing] = global_mean
    variances[missing] = global_var

    variances = np.maximum(variances, 1e-8)
    predicted = np.argmax(q_ev, axis=1)

    delta = q_ev - means[predicted]
    return np.mean((delta * delta) / variances[predicted], axis=1)
