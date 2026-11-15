import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "In-scope examples should remain close to their predicted intent centroid in low-variance within-class directions. A predicted-class trailing-subspace residual emphasizes these tightly constrained directions without using a full Gaussian density."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    z = np.asarray(ev.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    n, d = x.shape
    k = tr.logits.shape[1]

    counts = np.bincount(y, minlength=k).astype(float)
    means = np.zeros((k, d), dtype=float)
    np.add.at(means, y, x)
    present = counts > 0
    means[present] /= counts[present, None]

    global_mean = np.mean(x, axis=0)
    means[~present] = global_mean

    centered = x - means[y]
    cov = (centered.T @ centered) / max(n - int(np.sum(present)), 1)
    cov = 0.5 * (cov + cov.T)

    vals, vecs = np.linalg.eigh(cov)
    tail_dim = min(48, d)
    tail_vals = vals[:tail_dim]
    tail_vecs = vecs[:, :tail_dim]

    positive = vals[vals > 1e-10]
    floor = float(np.median(positive)) * 1e-3 if positive.size else 1.0
    floor = max(floor, 1e-8)
    tail_scale = np.maximum(tail_vals, floor)

    pred = np.argmax(np.asarray(ev.logits), axis=1)
    residual = z - means[pred]
    projected = residual @ tail_vecs
    standardized_sq = (projected * projected) / tail_scale[None, :]

    scores = np.mean(np.log1p(standardized_sq), axis=1)

    train_residual = x - means[np.argmax(np.asarray(tr.logits), axis=1)]
    train_projected = train_residual @ tail_vecs
    train_reference = np.mean(np.log1p((train_projected * train_projected) /
                                       tail_scale[None, :]), axis=1)
    scale = float(np.std(train_reference))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0

    scores = scores / scale
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
