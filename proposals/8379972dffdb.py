import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Low-variance within-class directions are tightly constrained for known intents, so deviations "
    "there are informative for OOS detection. Conditioning the trailing-subspace distance on the "
    "logit-predicted intent also exposes classifier-density disagreement."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0006", "m_0007"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    z = np.asarray(ev.feats, dtype=np.float64)
    k = tr.logits.shape[1]
    d = x.shape[1]

    global_mean = np.mean(x, axis=0)
    means = np.tile(global_mean, (k, 1))
    counts = np.bincount(y, minlength=k)
    for c in range(k):
        if counts[c] > 0:
            means[c] = np.mean(x[y == c], axis=0)

    centered = x - means[y]
    cov = (centered.T @ centered) / max(x.shape[0] - k, 1)
    cov = 0.5 * (cov + cov.T)

    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)
    positive = vals[vals > 1e-12]
    scale = np.median(positive) if positive.size else 1.0
    floor = max(scale * 1e-4, 1e-8)

    n_tail = min(d, max(16, d // 8))
    tail_vecs = vecs[:, :n_tail]
    tail_vars = np.maximum(vals[:n_tail], floor)

    pred = np.argmax(ev.logits, axis=1)
    diff = z - means[pred]
    projected = diff @ tail_vecs
    scores = np.sum((projected * projected) / tail_vars, axis=1)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
