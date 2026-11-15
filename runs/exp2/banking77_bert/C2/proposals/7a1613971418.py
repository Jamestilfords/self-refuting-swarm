import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Low-variance within-class directions can expose subtle deviations that broad high-variance semantic directions hide. Class-conditional empirical calibration makes trailing-subspace residuals comparable across predicted intents."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    classes = np.unique(y)
    k = int(np.max(classes)) + 1
    d = x.shape[1]
    means = np.zeros((k, d), dtype=float)
    counts = np.bincount(y, minlength=k).astype(float)

    for c in classes:
        means[c] = x[y == c].mean(axis=0)

    centered = x - means[y]
    cov = centered.T @ centered / max(float(len(x) - len(classes)), 1.0)
    vals, vecs = np.linalg.eigh(cov)

    q = min(64, d)
    tail_vals = vals[:q]
    positive = vals[vals > 1e-10]
    floor = float(np.median(positive)) * 1e-4 if positive.size else 1e-6
    tail_scale = 1.0 / np.sqrt(np.maximum(tail_vals, floor))
    tail_basis = vecs[:, :q] * tail_scale[None, :]

    train_proj = centered @ tail_basis
    train_tail = np.sum(train_proj * train_proj, axis=1)

    pred = np.argmax(ev.logits, axis=1).astype(int)
    residual = z - means[pred]
    eval_proj = residual @ tail_basis
    raw = np.sum(eval_proj * eval_proj, axis=1)

    scores = np.empty(len(z), dtype=float)
    for c in np.unique(pred):
        reference = np.sort(train_tail[y == c])
        idx = np.searchsorted(reference, raw[pred == c], side="left")
        exceed = reference.size - idx
        scores[pred == c] = -np.log((exceed + 1.0) / (reference.size + 1.0))

    return np.nan_to_num(scores, nan=0.0, posinf=50.0, neginf=0.0)
