import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A predicted-class conditional distance in the trailing within-class covariance subspace highlights small, atypical representation deviations. Combining this tail residual with classifier entropy avoids reproducing the rank ordering of an unconditional trailing-distance score."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    z = np.asarray(ev.pooled, dtype=float)
    k = tr.logits.shape[1]
    d = x.shape[1]
    tail_dim = min(128, d)

    means = np.zeros((k, d), dtype=float)
    counts = np.bincount(y, minlength=k).astype(float)
    for c in range(k):
        if counts[c] > 0:
            means[c] = x[y == c].mean(axis=0)

    centered = x - means[y]
    cov = (centered.T @ centered) / max(len(x) - k, 1)
    scale = np.trace(cov) / max(d, 1)
    cov.flat[::d + 1] += max(scale, 1e-8) * 1e-4

    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[:tail_dim]
    basis = vecs[:, idx]
    inv_vals = 1.0 / np.maximum(vals[idx], max(scale, 1e-8) * 1e-5)

    train_proj = (x - means[y]) @ basis
    train_tail = np.sum(train_proj * train_proj * inv_vals, axis=1)

    pred = np.argmax(ev.logits, axis=1)
    eval_proj = (z - means[pred]) @ basis
    eval_tail = np.sum(eval_proj * eval_proj * inv_vals, axis=1)

    tail_norm = ctx.h.normalizer("qsigmoid", train_tail)
    entropy_train = ctx.h.entropy(tr.logits)
    entropy_norm = ctx.h.normalizer("qsigmoid", entropy_train)

    return tail_norm(eval_tail) + 0.35 * entropy_norm(ctx.h.entropy(ev.logits))
