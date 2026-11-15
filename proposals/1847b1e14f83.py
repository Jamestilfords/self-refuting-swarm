import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A classifier-selected intent should have a better class-specific diagonal Gaussian fit than its strongest logit rival. "
    "OOS inputs are penalized when they are both unlikely under the selected intent and geometrically better explained by a competing intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0013", "m_0236"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    q = np.asarray(ev.feats, dtype=np.float64)

    n, d = x.shape
    k = tr.logits.shape[1]

    counts = np.bincount(y, minlength=k).astype(np.float64)
    sums = np.zeros((k, d), dtype=np.float64)
    np.add.at(sums, y, x)

    global_mean = x.mean(axis=0)
    means = sums / np.maximum(counts[:, None], 1.0)
    means[counts == 0] = global_mean

    sq_sums = np.zeros((k, d), dtype=np.float64)
    np.add.at(sq_sums, y, x * x)
    raw_var = sq_sums / np.maximum(counts[:, None], 1.0) - means * means
    raw_var = np.maximum(raw_var, 0.0)

    global_var = np.mean((x - global_mean) ** 2, axis=0)
    scale = max(float(np.mean(global_var)), 1e-8)
    floor = 1e-6 * scale

    # Empirical-Bayes shrinkage stabilizes intents with few examples while
    # retaining dimensions that are unusually constrained for a given intent.
    weight = counts[:, None] / (counts[:, None] + 20.0)
    variances = weight * raw_var + (1.0 - weight) * global_var[None, :]
    variances = np.maximum(variances, floor)
    logdet = np.log(variances).sum(axis=1)

    logits = np.asarray(ev.logits, dtype=np.float64)
    selected = np.argmax(logits, axis=1)

    if k > 1:
        rival = np.argpartition(logits, -2, axis=1)[:, -2]
    else:
        rival = selected.copy()

    def diagonal_nll(classes):
        delta = q - means[classes]
        return 0.5 * (
            np.sum(delta * delta / variances[classes], axis=1)
            + logdet[classes]
        )

    selected_nll = diagonal_nll(selected)
    rival_nll = diagonal_nll(rival)

    # The second term is a discriminative consistency check: a logit-selected
    # class should not be substantially less plausible than its closest rival.
    mismatch = np.maximum(selected_nll - rival_nll, 0.0)
    out = (selected_nll + 0.75 * mismatch) / float(d)

    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=0.0)
