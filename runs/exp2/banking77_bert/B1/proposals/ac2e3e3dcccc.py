import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Within-class trailing covariance directions encode tight semantic constraints that are weakly expressed by global variance. Class-conditionally normalizing trailing-direction residual energy makes violations comparable across intents without using evaluation statistics."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    z = np.asarray(ev.pooled, dtype=np.float64)

    n_classes = int(max(np.max(y), tr.logits.shape[1] - 1) + 1)
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    sums = np.zeros((n_classes, x.shape[1]), dtype=np.float64)
    np.add.at(sums, y, x)
    global_mean = np.mean(x, axis=0)
    means = np.tile(global_mean, (n_classes, 1))
    present = counts > 0
    means[present] = sums[present] / counts[present, None]

    centered = x - means[y]
    cov = (centered.T @ centered) / max(len(x) - n_classes, 1)
    cov.flat[:: cov.shape[0] + 1] += 1e-6
    eigvals, eigvecs = np.linalg.eigh(cov)

    tail_dim = min(96, x.shape[1])
    tail_basis = eigvecs[:, :tail_dim]
    train_tail = np.sum((centered @ tail_basis) ** 2, axis=1)

    class_scale = np.full(n_classes, np.median(train_tail) + 1e-8)
    for c in np.flatnonzero(present):
        values = train_tail[y == c]
        class_scale[c] = np.median(values) + 1e-8

    pred = np.argmax(ev.logits, axis=1).astype(np.int64)
    pred = np.clip(pred, 0, n_classes - 1)
    eval_centered = z - means[pred]
    eval_tail = np.sum((eval_centered @ tail_basis) ** 2, axis=1)

    return np.log1p(eval_tail / class_scale[pred])
