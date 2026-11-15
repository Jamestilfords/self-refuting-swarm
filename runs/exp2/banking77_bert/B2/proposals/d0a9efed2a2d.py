import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A query should be plausible under the feature dispersion of the intent selected by the classifier. "
    "Shrunk diagonal class covariances preserve intent-specific variability while remaining stable for small classes."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0059"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    k = tr.logits.shape[1]
    d = x.shape[1]

    counts = np.bincount(y, minlength=k).astype(np.float64)
    global_mean = np.mean(x, axis=0)
    global_var = np.var(x, axis=0)
    floor = np.maximum(global_var * 1e-6, 1e-10)

    sums = np.zeros((k, d), dtype=np.float64)
    np.add.at(sums, y, x)
    means = np.tile(global_mean, (k, 1))
    present = counts > 0
    means[present] = sums[present] / counts[present, None]

    centered = x - means[y]
    sq_sums = np.zeros((k, d), dtype=np.float64)
    np.add.at(sq_sums, y, centered * centered)

    empirical = np.tile(global_var, (k, 1))
    enough = counts > 1
    empirical[enough] = sq_sums[enough] / (counts[enough, None] - 1.0)

    shrink = d / (counts + d)
    variances = (1.0 - shrink[:, None]) * empirical + shrink[:, None] * global_var
    variances = np.maximum(variances, floor)

    predicted = np.argmax(ev.logits, axis=1)
    delta = z - means[predicted]
    chosen_var = variances[predicted]
    mahal = np.sum((delta * delta) / chosen_var, axis=1)
    logdet = np.sum(np.log(chosen_var), axis=1)
    out = 0.5 * (mahal + logdet)

    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=-1e12)
