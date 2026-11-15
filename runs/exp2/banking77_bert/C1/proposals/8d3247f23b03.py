import numpy as np
from scipy.special import logsumexp

FAMILY = "density"
ARM = "ce"
RATIONALE = "A posterior-weighted mixture of class-specific diagonal Gaussians requires an item to be compatible with the feature variance geometry of the classes supported by its logits. Unlike a predicted-class score, uncertain predictions cannot hide low density under all plausible intent classes."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    n_classes = tr.logits.shape[1]

    pca = ctx.h.pca(x, 96)
    x = np.asarray(pca.transform(x), dtype=float)
    z = np.asarray(pca.transform(z), dtype=float)

    global_mean = np.mean(x, axis=0)
    global_var = np.mean((x - global_mean) ** 2, axis=0)
    global_var = np.maximum(global_var, 1e-6)

    counts = np.bincount(y, minlength=n_classes).astype(float)
    sums = np.zeros((n_classes, x.shape[1]), dtype=float)
    sq_sums = np.zeros((n_classes, x.shape[1]), dtype=float)
    np.add.at(sums, y, x)
    np.add.at(sq_sums, y, x * x)

    means = np.tile(global_mean, (n_classes, 1))
    variances = np.tile(global_var, (n_classes, 1))
    present = counts > 0
    means[present] = sums[present] / counts[present, None]

    empirical = np.zeros_like(variances)
    empirical[present] = (
        sq_sums[present] / counts[present, None] - means[present] ** 2
    )
    empirical = np.maximum(empirical, 0.0)

    shrink = counts / (counts + 32.0)
    variances[present] = (
        shrink[present, None] * empirical[present]
        + (1.0 - shrink[present, None]) * global_var
    )
    variances = np.maximum(variances, global_var[None, :] * 1e-4 + 1e-7)

    log_det = np.sum(np.log(variances), axis=1)
    log_post = np.log(np.maximum(ctx.h.softmax(ev.logits), 1e-12))
    out = np.empty(z.shape[0], dtype=float)
    dim = z.shape[1]
    constant = dim * np.log(2.0 * np.pi)

    for start in range(0, z.shape[0], 256):
        stop = min(start + 256, z.shape[0])
        batch = z[start:stop]
        class_log_density = np.empty((stop - start, n_classes), dtype=float)

        for c in range(n_classes):
            diff = batch - means[c]
            quad = np.sum(diff * diff / variances[c], axis=1)
            class_log_density[:, c] = -0.5 * (quad + log_det[c] + constant)

        out[start:stop] = -logsumexp(
            log_post[start:stop] + class_log_density,
            axis=1,
        )

    return out
