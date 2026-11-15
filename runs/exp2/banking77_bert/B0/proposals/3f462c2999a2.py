import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Each intent can have distinct feature-wise dispersion, so a shrunk diagonal "
    "class model can detect queries that are atypical for their predicted intent. "
    "Per-class train-fitted tail calibration makes heteroscedastic residuals comparable."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0139", "m_0141"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=np.float64)
    z = np.asarray(ev.feats, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)

    k = tr.logits.shape[1]
    d = x.shape[1]
    counts = np.bincount(y, minlength=k).astype(np.float64)

    means = np.zeros((k, d), dtype=np.float64)
    np.add.at(means, y, x)
    means /= np.maximum(counts[:, None], 1.0)

    centered = x - means[y]
    class_ss = np.zeros((k, d), dtype=np.float64)
    np.add.at(class_ss, y, centered * centered)

    global_var = np.var(x, axis=0)
    global_var = np.maximum(global_var, 1e-8)
    raw_var = class_ss / np.maximum(counts[:, None], 1.0)

    # Empirical-Bayes shrinkage prevents unstable estimates for rare intents.
    strength = 24.0
    weight = counts[:, None] / (counts[:, None] + strength)
    variances = weight * raw_var + (1.0 - weight) * global_var[None, :]
    variances = np.maximum(variances, global_var[None, :] * 1e-4 + 1e-8)
    logdet = np.sum(np.log(variances), axis=1)

    train_residual = centered * centered / variances[y]
    train_nll = 0.5 * (
        np.sum(train_residual, axis=1) + logdet[y]
    )

    sorted_tails = []
    for c in range(k):
        vals = np.sort(train_nll[y == c])
        if vals.size == 0:
            vals = np.array([0.0], dtype=np.float64)
        sorted_tails.append(vals)

    pred = np.argmax(ev.logits, axis=1).astype(np.int64)
    delta = z - means[pred]
    eval_nll = 0.5 * (
        np.sum(delta * delta / variances[pred], axis=1) + logdet[pred]
    )

    out = np.empty(z.shape[0], dtype=np.float64)
    for c in range(k):
        mask = pred == c
        if np.any(mask):
            reference = sorted_tails[c]
            ranks = np.searchsorted(reference, eval_nll[mask], side="right")
            out[mask] = (ranks + 0.5) / (reference.size + 1.0)

    return np.clip(out, 0.0, 1.0)
