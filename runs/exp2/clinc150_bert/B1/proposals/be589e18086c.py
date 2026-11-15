import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A predicted-intent conditional diagonal likelihood uses each intent's own feature "
    "variability instead of a tied covariance. Shrinkage toward global variances keeps "
    "rare intent estimates stable while exposing atypical decision-bottleneck deviations."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0078"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=np.float64)
    z = np.asarray(ev.feats, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    n_classes = tr.logits.shape[1]
    dim = x.shape[1]

    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    sums = np.zeros((n_classes, dim), dtype=np.float64)
    np.add.at(sums, y, x)
    means = sums / np.maximum(counts[:, None], 1.0)

    global_mean = x.mean(axis=0)
    global_var = ((x - global_mean) ** 2).mean(axis=0)
    global_var = np.maximum(global_var, 1e-8)

    residuals = x - means[y]
    sq_sums = np.zeros((n_classes, dim), dtype=np.float64)
    np.add.at(sq_sums, y, residuals * residuals)
    class_var = sq_sums / np.maximum(counts[:, None], 1.0)

    shrink = counts[:, None] / (counts[:, None] + 20.0)
    variances = shrink * class_var + (1.0 - shrink) * global_var[None, :]
    variances = np.maximum(variances, global_var[None, :] * 1e-4 + 1e-8)

    predicted = np.argmax(ev.logits, axis=1)
    delta = z - means[predicted]
    chosen_var = variances[predicted]
    score_values = np.sum(delta * delta / chosen_var + np.log(chosen_var), axis=1)

    return np.nan_to_num(score_values, nan=0.0, posinf=1e12, neginf=-1e12)
