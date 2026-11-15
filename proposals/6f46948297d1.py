import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Low-variance within-class directions are tightly constrained for supported intents. "
    "A class-selected empirical tail pseudo-likelihood in this trailing subspace detects coordinate-wise residuals that are rare under the predicted intent without assuming Gaussian tails."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0017", "m_0144"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    z = np.asarray(ev.feats, dtype=float)

    k = tr.logits.shape[1]
    d = x.shape[1]
    global_mean = x.mean(axis=0)

    means = np.empty((k, d), dtype=float)
    counts = np.bincount(y, minlength=k)
    for c in range(k):
        if counts[c] > 0:
            means[c] = x[y == c].mean(axis=0)
        else:
            means[c] = global_mean

    residuals = x - means[y]
    cov = (residuals.T @ residuals) / max(len(x) - k, 1)
    cov.flat[::d + 1] += 1e-6 * (np.trace(cov) / max(d, 1) + 1.0)
    _, vectors = np.linalg.eigh(cov)

    n_tail = min(32, d)
    basis = vectors[:, :n_tail]
    train_proj = residuals @ basis

    pred = np.argmax(ev.logits, axis=1)
    eval_proj = (z - means[pred]) @ basis
    out = np.empty(len(z), dtype=float)

    for c in range(k):
        idx = np.flatnonzero(pred == c)
        if idx.size == 0:
            continue

        class_proj = train_proj[y == c]
        if class_proj.shape[0] < 4:
            class_proj = train_proj

        med = np.median(class_proj, axis=0)
        mad = np.median(np.abs(class_proj - med), axis=0)
        scale = np.maximum(1.4826 * mad, 1e-6)

        standardized_train = np.abs((class_proj - med) / scale)
        standardized_eval = np.abs((eval_proj[idx] - med) / scale)
        sorted_train = np.sort(standardized_train, axis=0)

        tails = np.empty_like(standardized_eval)
        n_class = sorted_train.shape[0]
        for j in range(n_tail):
            pos = np.searchsorted(sorted_train[:, j], standardized_eval[:, j], side="left")
            tails[:, j] = (n_class - pos + 1.0) / (n_class + 1.0)

        out[idx] = -np.mean(np.log(np.maximum(tails, 1e-12)), axis=1)

    return out
